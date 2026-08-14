#!/usr/bin/env python3
"""Lower frontend execution-plan phases into a generic NPU executable."""

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


FORMAT = "OPENNPUX_NPU_EXECUTABLE_V2"
HEADER = struct.Struct("<8I3Q2I3Q")
ENTRY = struct.Struct("<4I4Q")
COMMAND = struct.Struct("<8I6Q")
MAGIC = 0x4558504E
INVOCATION_HEADER_SIZE = 144
TENSOR_BINDING_SIZE = 112
INVOCATION_COMMAND_SIZE = 112
RECORD_ALIGNMENT = 64

OPCODE = {
    "EMBED": 1, "MATMUL": 2, "ADD": 3, "MUL": 4, "NORMALIZE": 5,
    "ROPE": 6, "SOFTMAX": 7, "TOPK": 8, "CONVOLUTION": 9,
    "CAUSAL_CONVOLUTION": 10, "RECURRENT_UPDATE": 11, "ROUTER": 12,
    "EXPERT": 13, "DMA": 14, "ATTENTION": 15, "ACTIVATION": 16,
    "COMBINE": 17,
}

PHASE_OPCODE = {
    "attention_norm": "NORMALIZE",
    "qkv_projection": "MATMUL",
    "rope": "ROPE",
    "paged_kv_cache_update": "DMA",
    "scaled_dot_product_attention": "ATTENTION",
    "attention_output_projection": "MATMUL",
    "linear_attention_projection": "MATMUL",
    "causal_depthwise_conv": "CAUSAL_CONVOLUTION",
    "recurrent_state_update": "RECURRENT_UPDATE",
    "linear_attention_gate_norm": "NORMALIZE",
    "linear_attention_output_projection": "MATMUL",
    "residual_add": "ADD",
    "ffn_norm": "NORMALIZE",
    "router_topk": "ROUTER",
    "routed_experts_active_only": "EXPERT",
    "shared_expert": "EXPERT",
    "moe_combine": "COMBINE",
}

STATE_PHASES = {"paged_kv_cache_update", "recurrent_state_update"}


def load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def lower_commands(plan: dict[str, Any]) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    dependency = 0
    for layer in plan.get("layers", []):
        layer_index = int(layer["index"])
        layer_type = str(layer["type"])
        for phase_index, phase_value in enumerate(layer.get("phases", [])):
            phase = str(phase_value)
            opcode = PHASE_OPCODE.get(phase)
            if opcode is None:
                raise ValueError(
                    f"no generic opcode for layer {layer_index} phase {phase}"
                )
            completion = len(commands) + 1
            commands.append(
                {
                    "command_id": len(commands),
                    "opcode": opcode,
                    "capability": f"op.{opcode.lower()}.v1",
                    "dependency_token": dependency,
                    "completion_token": completion,
                    "parameter_symbol": f"layer.{layer_index}.{phase}",
                    "profiling_tag": (layer_index << 16) | phase_index,
                    "attributes": {
                        "layer": layer_index,
                        "layer_type": layer_type,
                        "phase": phase,
                        "persistent_state": phase in STATE_PHASES,
                    },
                }
            )
            dependency = completion
    if not commands:
        raise ValueError("execution plan contains no commands")
    return commands


def build_executable(
    manifest: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    if plan.get("format") != "OPENNPUX_QWEN_EXECUTION_PLAN_V1":
        raise ValueError("unsupported frontend execution plan")
    if int(plan.get("observed_layer_count", 0)) != int(
        manifest.get("layer_count", -1)
    ):
        raise ValueError("execution plan does not cover every decoder layer")
    unclassified_layers = [
        int(layer["index"])
        for layer in plan.get("layers", [])
        if layer.get("type") == "unclassified_moe"
    ]
    if unclassified_layers:
        raise ValueError(
            "unclassified decoder layers: "
            + ",".join(str(layer) for layer in unclassified_layers)
        )
    unknown_decoder = plan.get("unknown_decoder_tensor_patterns", {})
    if unknown_decoder:
        patterns = ", ".join(list(unknown_decoder)[:8])
        raise ValueError(
            f"decoder has {len(unknown_decoder)} unclassified tensor patterns: "
            f"{patterns}"
        )
    commands = lower_commands(plan)
    capabilities = sorted({command["capability"] for command in commands})
    return {
        "format": FORMAT,
        "version": 2,
        "default_active_experts": int(manifest.get("experts_per_token", 1)),
        "target": "opennpux-coral-generic-v1",
        "source": {
            "model_manifest": plan["model_manifest"],
            "architecture": plan["architecture"],
            "frontend_plan_format": plan["format"],
        },
        "memory_requirements": {
            "weights": "paged-external",
            "scratch": "runtime-sized",
            "persistent_state": "runtime-bound",
            "alignment": 64,
        },
        "logical_bindings": [
            {"id": 0, "name": "input", "access": "read", "dynamic": True},
            {"id": 1, "name": "output", "access": "write", "dynamic": True},
            {"id": 2, "name": "weights", "access": "read", "dynamic": True},
            {
                "id": 3,
                "name": "persistent_state",
                "access": "read_write",
                "dynamic": True,
            },
            {"id": 4, "name": "scratch", "access": "read_write", "dynamic": True},
        ],
        "entry_points": [
            {"id": 1, "name": "prefill", "first_command": 0, "command_count": len(commands)},
            {"id": 2, "name": "decode", "first_command": 0, "command_count": len(commands)},
        ],
        "required_capabilities": capabilities,
        "commands": commands,
    }


def hash64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def checksum(data: bytes, checksum_offset: int) -> int:
    mutable = bytearray(data)
    mutable[checksum_offset : checksum_offset + 4] = b"\0\0\0\0"
    value = 2166136261
    for byte in mutable:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def align(value: int, alignment: int = RECORD_ALIGNMENT) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def write_binary(executable: dict[str, Any], path: Path) -> None:
    entries = executable["entry_points"]
    commands = executable["commands"]
    entry_offset = HEADER.size
    command_offset = entry_offset + len(entries) * ENTRY.size
    total_size = command_offset + len(commands) * COMMAND.size
    executable_id = hash64(json.dumps(executable["source"], sort_keys=True)) or 1
    header = HEADER.pack(
        MAGIC, 2, HEADER.size, total_size, len(entries), len(commands),
        ENTRY.size, COMMAND.size, entry_offset, command_offset, executable_id,
        0, int(executable["default_active_experts"]), 0, 0, 0,
    )
    entry_data = b"".join(
        ENTRY.pack(
            int(entry["id"]), int(entry["first_command"]),
            int(entry["command_count"]), 0, 0, 0, 0, 0,
        )
        for entry in entries
    )
    command_data = b"".join(
        COMMAND.pack(
            int(command["command_id"]), OPCODE[command["opcode"]], 0,
            hash64(command["capability"]) & 0xFFFFFFFF, 0, 5,
            int(command["dependency_token"]), int(command["completion_token"]),
            hash64(command["parameter_symbol"]), 0, 0,
            int(command["profiling_tag"]),
            0,
            2 | (3 << 16) | (4 << 32),
        )
        for command in commands
    )
    data = header + entry_data + command_data
    # offsetof(opennpux_npu_executable_header, checksum)
    checksum_offset = 56
    value = checksum(data, checksum_offset)
    data = data[:checksum_offset] + struct.pack("<I", value) + data[checksum_offset + 4 :]
    path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    executable = build_executable(load_object(args.manifest), load_object(args.plan))
    args.output.write_text(json.dumps(executable, indent=2, sort_keys=True) + "\n")
    binary_output = args.output.with_suffix(".npxc")
    write_binary(executable, binary_output)
    print(f"npu_executable={args.output}")
    print(f"npu_command_template={binary_output}")
    print(f"npu_executable_commands={len(executable['commands'])}")
    print(f"npu_command_template_bytes={binary_output.stat().st_size}")
    binding_offset = align(INVOCATION_HEADER_SIZE)
    command_offset = align(binding_offset + 5 * TENSOR_BINDING_SIZE)
    invocation_bytes = align(
        command_offset + len(executable["commands"]) * INVOCATION_COMMAND_SIZE
    )
    print(f"npu_invocation_bytes_upper_bound={invocation_bytes}")
    print(
        "npu_executable_capabilities="
        + ",".join(executable["required_capabilities"])
    )
    print("npu_executable=PASS")


if __name__ == "__main__":
    main()
