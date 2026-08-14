#!/usr/bin/env python3
"""Bind generic NPU commands to model tensor-index ranges."""

import argparse
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_qwen_execution_plan import LAYER_RE, tensor_component, tensor_role


FORMAT = "OPENNPUX_NPU_WEIGHT_PLAN_V1"
HEADER = struct.Struct("<8I")
RECORD = struct.Struct("<160s4I8Q2Q")
MAGIC = 0x5458504E

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


def load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def hash64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


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
        records.append(
            {
                "name": name,
                "name_hash": f"0x{hash64(name):016x}",
                "layer": int(layer_match.group(1)) if layer_match else None,
                "role": tensor_role(name),
                "component": tensor_component(name),
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
) -> dict[str, Any]:
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
                "primary_range": primary,
            }
        )

    return {
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output = args.output or args.executable.with_suffix(".npxw")
    plan = build_weight_plan(
        manifest_path, load_object(manifest_path), load_object(args.executable)
    )
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
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
    if args.require_complete and plan["unresolved_weight_command_count"] != 0:
        raise ValueError(
            "weight plan has unresolved weight-bearing commands: "
            f"{plan['unresolved_weight_command_count']}"
        )
    print("npu_weight_plan=PASS")


if __name__ == "__main__":
    main()
