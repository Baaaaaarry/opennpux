#!/usr/bin/env python3
"""Build a command-indexed table of samples from real tensor ranges."""

import argparse
import json
from pathlib import Path
from typing import Any


FORMAT = "OPENNPUX_NPU_WEIGHT_PLAN_V1"
SAMPLE_SIZE = 4


def load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def fnv1a32(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def read_sample(plan_path: Path, tensor: dict[str, Any]) -> bytes:
    shard = plan_path.parent / str(tensor["shard"])
    offset = int(tensor["offset"])
    tensor_size = int(tensor["size"])
    if offset < 0 or tensor_size <= 0 or not shard.is_file():
        raise ValueError(f"invalid tensor range in {shard}")
    if offset + tensor_size > shard.stat().st_size:
        raise ValueError(f"tensor range is outside shard: {shard}")
    with shard.open("rb") as source:
        source.seek(offset)
        return source.read(min(SAMPLE_SIZE, tensor_size)).ljust(SAMPLE_SIZE, b"\0")


def materialize(
    plan_path: Path, output: Path, size: int, allow_unresolved: bool
) -> None:
    plan = load_object(plan_path)
    if plan.get("format") != FORMAT:
        raise ValueError("unsupported NPU weight plan")
    command_count = int(plan.get("command_count", 0))
    if size <= 0 or command_count <= 0 or command_count * SAMPLE_SIZE > size:
        raise ValueError("weight sample table does not fit output buffer")
    table = bytearray(size)
    sampled = 0
    unresolved = 0
    shards: set[str] = set()
    for command in plan.get("commands", []):
        command_id = int(command["command_id"])
        if command_id < 0 or command_id >= command_count:
            raise ValueError(f"invalid command id {command_id}")
        tensor = command.get("primary_range")
        requires_weight = bool(command.get("required_roles"))
        if tensor is None:
            if requires_weight:
                unresolved += 1
            continue
        if not isinstance(tensor, dict):
            raise ValueError(f"invalid primary range for command {command_id}")
        begin = command_id * SAMPLE_SIZE
        table[begin : begin + SAMPLE_SIZE] = read_sample(plan_path, tensor)
        shards.add(str(tensor["shard"]))
        sampled += 1
    if unresolved and not allow_unresolved:
        raise ValueError(f"weight sample table has {unresolved} unresolved commands")
    output.write_bytes(table)
    print(f"weight_sample_commands={command_count}")
    print(f"weight_sample_mapped_commands={sampled}")
    print(f"weight_sample_unresolved_commands={unresolved}")
    print(f"weight_sample_shards={len(shards)}")
    print(f"weight_sample_bytes={len(table)}")
    print(f"weight_sample_checksum=0x{fnv1a32(table):08x}")
    print("weight_sample_materialize=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weight_plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--allow-unresolved", action="store_true")
    args = parser.parse_args()
    materialize(
        args.weight_plan.resolve(), args.output, args.size,
        args.allow_unresolved,
    )


if __name__ == "__main__":
    main()
