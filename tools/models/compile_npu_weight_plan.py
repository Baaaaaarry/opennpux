#!/usr/bin/env python3
"""Bind generic NPU commands to model tensor-index ranges."""

import argparse
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_qwen_execution_plan import (
    EXPERT_RE, LAYER_RE, tensor_component, tensor_role,
)


FORMAT = "OPENNPUX_NPU_WEIGHT_PLAN_V1"
HEADER = struct.Struct("<8I")
RECORD = struct.Struct("<160s4I8Q2Q")
MAGIC = 0x5458504E
RANGE_FORMAT = "OPENNPUX_NPU_WEIGHT_RANGE_INDEX_V1"
RANGE_MAGIC = 0x5258504E
RANGE_HEADER = struct.Struct("<8I4Q")
RANGE_RECORD = struct.Struct("<4I6Q")

PHASE_ROLES = {
    "token_embedding": ("embedding",),
    "attention_norm": ("attention_norm",),
    "qkv_projection": (
        "attention_q_proj", "attention_k_proj", "attention_v_proj",
        "attention_q_norm", "attention_k_norm",
    ),
    "attention_output_projection": ("attention_o_proj",),
    "linear_attention_projection": (
        "linear_attention_qkv", "linear_attention_alpha",
        "linear_attention_beta",
    ),
    "causal_depthwise_conv": ("linear_attention_conv",),
    "linear_attention_gate_norm": (
        "linear_attention_gate", "linear_attention_norm",
        "linear_attention_decay",
    ),
    "linear_attention_output_projection": ("linear_attention_output",),
    "ffn_norm": ("ffn_norm",),
    "router_topk": ("router",),
    "routed_experts_active_only": ("routed_expert",),
    "shared_expert": ("shared_expert",),
    "final_norm": ("final_norm",),
    "lm_head": ("lm_head",),
}

TENSOR_SLOTS = {
    "q_proj": 1,
    "k_proj": 2,
    "v_proj": 3,
    "o_proj": 4,
    "gate_proj": 5,
    "up_proj": 6,
    "down_proj": 7,
    "in_proj_qkv": 8,
}

TENSOR_INDEX_TO_NPU_DTYPE = {
    2: 2,  # U8 storage for packed/int8 tensors
    3: 2,  # I8
    5: 3,  # I32
    7: 4,  # F16
    8: 5,  # BF16
    9: 6,  # F32
}


def load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def hash64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def tensor_slot(name: str) -> int:
    fields = name.split(".")
    for field in reversed(fields[:-1]):
        if field in TENSOR_SLOTS:
            return TENSOR_SLOTS[field]
    return 0


def tensor_records(manifest_path: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    index_path = manifest_path.parent / str(manifest["tensor_index"])
    data = index_path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("truncated tensor index")
    magic, version, header_size, record_size, count = HEADER.unpack_from(data)[:5]
    if (
        magic != MAGIC or version != 1 or header_size != HEADER.size
        or record_size != RECORD.size
        or count != int(manifest["tensor_count"])
        or len(data) != header_size + count * record_size
    ):
        raise ValueError("invalid tensor index ABI")

    shards = manifest.get("shards", [])
    records = []
    for index in range(count):
        values = RECORD.unpack_from(data, header_size + index * record_size)
        name = values[0].split(b"\0", 1)[0].decode("utf-8")
        shard_index = int(values[1])
        if shard_index >= len(shards):
            raise ValueError(f"tensor {name} has invalid shard index {shard_index}")
        layer_match = LAYER_RE.search(name)
        expert_match = EXPERT_RE.search(name)
        records.append(
            {
                "name": name,
                "name_hash": f"0x{hash64(name):016x}",
                "layer": int(layer_match.group(1)) if layer_match else None,
                "expert": int(expert_match.group(1)) if expert_match else None,
                "role": tensor_role(name),
                "component": tensor_component(name),
                "slot": tensor_slot(name),
                "dtype": (
                    1 if tensor_component(name) == "qweight"
                    else TENSOR_INDEX_TO_NPU_DTYPE.get(int(values[2]), 0)
                ),
                "shard": str(shards[shard_index]["path"]),
                "shard_index": shard_index,
                "offset": int(values[-2]),
                "size": int(values[-1]),
            }
        )
    return records


def command_layer_phase(command: dict[str, Any]) -> tuple[int | None, str]:
    attributes = command.get("attributes", {})
    if not isinstance(attributes, dict):
        raise ValueError("command attributes must be an object")
    layer = attributes.get("layer")
    return (int(layer) if layer is not None else None), str(attributes["phase"])


def selection_policy(phase: str, matched: list[dict[str, Any]]) -> str:
    if phase == "routed_experts_active_only":
        return "router-selected-active-experts"
    if matched:
        return "static-layer-ranges"
    return "none"


def build_weight_plan(
    manifest_path: Path, manifest: dict[str, Any], executable: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if executable.get("format") != "OPENNPUX_NPU_EXECUTABLE_V2":
        raise ValueError("unsupported generic NPU executable")
    records = tensor_records(manifest_path, manifest)
    by_layer_role: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["layer"] is not None:
            by_layer_role[(int(record["layer"]), str(record["role"]))].append(record)

    commands = []
    mapped_commands = 0
    exact_ranges = 0
    mapped_bytes = 0
    range_records = []
    for command in executable.get("commands", []):
        layer, phase = command_layer_phase(command)
        roles = PHASE_ROLES.get(phase, ())
        matched = sorted(
            (
                record
                for role in roles
                for record in (
                    by_layer_role.get((layer, role), []) if layer is not None
                    else [item for item in records if item["role"] == role]
                )
            ),
            key=lambda record: (
                record["shard_index"], record["offset"], record["name"]
            ),
        )
        total_bytes = sum(int(record["size"]) for record in matched)
        range_start = len(range_records)
        for record in matched:
            range_records.append(
                {
                    "command_id": int(command["command_id"]),
                    "parameter_symbol": str(command["parameter_symbol"]),
                    **record,
                }
            )
        if matched:
            mapped_commands += 1
            exact_ranges += len(matched)
            mapped_bytes += total_bytes
        primary = None
        if matched:
            first = matched[0]
            primary = {
                key: first[key]
                for key in (
                    "name_hash", "role", "component", "shard",
                    "shard_index", "offset", "size",
                )
            }
        commands.append(
            {
                "command_id": int(command["command_id"]),
                "parameter_symbol": str(command["parameter_symbol"]),
                "parameter_symbol_hash": (
                    f"0x{hash64(str(command['parameter_symbol'])):016x}"
                ),
                "layer": layer,
                "phase": phase,
                "required_roles": list(roles),
                "selection": selection_policy(phase, matched),
                "matched_tensor_count": len(matched),
                "matched_tensor_bytes": total_bytes,
                "range_start": range_start,
                "range_count": len(matched),
                "primary_range": primary,
            }
        )

    plan = {
        "format": FORMAT,
        "version": 1,
        "executable": manifest_path.with_name("model.npxc").name,
        "tensor_index": str(manifest["tensor_index"]),
        "shards": manifest.get("shards", []),
        "command_count": len(commands),
        "mapped_command_count": mapped_commands,
        "weightless_command_count": sum(
            not command["required_roles"] for command in commands
        ),
        "unresolved_weight_command_count": sum(
            bool(command["required_roles"]) and command["primary_range"] is None
            for command in commands
        ),
        "matched_tensor_range_count": exact_ranges,
        "matched_tensor_bytes": mapped_bytes,
        "commands": commands,
    }
    return plan, range_records


def checksum(data: bytes, checksum_offset: int) -> int:
    mutable = bytearray(data)
    mutable[checksum_offset : checksum_offset + 4] = bytes(4)
    value = 2166136261
    for byte in mutable:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def write_range_index(
    executable: dict[str, Any], plan: dict[str, Any],
    ranges: list[dict[str, Any]], output: Path
) -> None:
    record_offset = RANGE_HEADER.size
    total_size = record_offset + len(ranges) * RANGE_RECORD.size
    executable_id = hash64(json.dumps(executable["source"], sort_keys=True)) or 1
    header = RANGE_HEADER.pack(
        RANGE_MAGIC, 1, RANGE_HEADER.size, RANGE_RECORD.size,
        int(plan["command_count"]), len(ranges), len(plan["shards"]), 0,
        record_offset, total_size, executable_id, 0,
    )
    records = b"".join(
        RANGE_RECORD.pack(
            int(record["command_id"]), int(record["shard_index"]),
            hash64(str(record["role"])) & 0xFFFFFFFF,
            hash64(str(record["component"])) & 0xFFFFFFFF,
            int(record["offset"]), int(record["size"]),
            int(str(record["name_hash"]), 16),
            hash64(str(record["parameter_symbol"])),
            UINT64_MAX if record["expert"] is None else int(record["expert"]),
            int(record["slot"]) | (int(record["dtype"]) << 16),
        )
        for record in ranges
    )
    data = header + records
    checksum_offset = 28
    value = checksum(data, checksum_offset)
    output.write_bytes(
        data[:checksum_offset] + struct.pack("<I", value) +
        data[checksum_offset + 4 :]
    )


UINT64_MAX = (1 << 64) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--range-output", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output = args.output or args.executable.with_suffix(".npxw")
    executable = load_object(args.executable)
    plan, ranges = build_weight_plan(
        manifest_path, load_object(manifest_path), executable
    )
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    range_output = args.range_output or output.with_suffix(".npxr")
    write_range_index(executable, plan, ranges, range_output)
    print(f"npu_weight_plan={output}")
    print(f"npu_weight_plan_commands={plan['command_count']}")
    print(f"npu_weight_plan_mapped_commands={plan['mapped_command_count']}")
    print(f"npu_weight_plan_weightless_commands={plan['weightless_command_count']}")
    print(
        "npu_weight_plan_unresolved_weight_commands="
        f"{plan['unresolved_weight_command_count']}"
    )
    print(f"npu_weight_plan_tensor_ranges={plan['matched_tensor_range_count']}")
    print(f"npu_weight_plan_tensor_bytes={plan['matched_tensor_bytes']}")
    print(f"npu_weight_range_index={range_output}")
    print(f"npu_weight_range_records={len(ranges)}")
    print(f"npu_weight_range_bytes={range_output.stat().st_size}")
    if args.require_complete and plan["unresolved_weight_command_count"] != 0:
        raise ValueError(
            "weight plan has unresolved weight-bearing commands: "
            f"{plan['unresolved_weight_command_count']}"
        )
    print("npu_weight_plan=PASS")


if __name__ == "__main__":
    main()
