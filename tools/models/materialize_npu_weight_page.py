#!/usr/bin/env python3
"""Materialize one real tensor page selected by an NPU weight plan."""

import argparse
import json
from pathlib import Path
from typing import Any


def load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def select_command(plan: dict[str, Any], command_id: int | None) -> dict[str, Any]:
    commands = plan.get("commands", [])
    if not isinstance(commands, list):
        raise ValueError("weight plan commands must be an array")
    for command in commands:
        if not isinstance(command, dict):
            continue
        if command_id is not None and int(command.get("command_id", -1)) != command_id:
            continue
        if command.get("primary_range") is not None:
            return command
    if command_id is None:
        raise ValueError("weight plan contains no mapped tensor range")
    raise ValueError(f"command {command_id} has no mapped tensor range")


def materialize(plan_path: Path, output: Path, size: int, command_id: int | None) -> None:
    plan = load_object(plan_path)
    if plan.get("format") != "OPENNPUX_NPU_WEIGHT_PLAN_V1":
        raise ValueError("unsupported NPU weight plan")
    command = select_command(plan, command_id)
    tensor = command["primary_range"]
    if not isinstance(tensor, dict):
        raise ValueError("invalid primary tensor range")
    shard = plan_path.parent / str(tensor["shard"])
    offset = int(tensor["offset"])
    tensor_size = int(tensor["size"])
    if size <= 0 or offset < 0 or tensor_size <= 0:
        raise ValueError("invalid page or tensor range")
    if not shard.is_file() or offset + tensor_size > shard.stat().st_size:
        raise ValueError(f"tensor range is outside shard: {shard}")

    with shard.open("rb") as source:
        source.seek(offset)
        payload = source.read(min(size, tensor_size))
    output.write_bytes(payload.ljust(size, b"\0"))
    print(f"weight_page_command={command['command_id']}")
    print(f"weight_page_parameter={command['parameter_symbol']}")
    print(f"weight_page_tensor_hash={tensor['name_hash']}")
    print(f"weight_page_role={tensor['role']}")
    print(f"weight_page_component={tensor['component']}")
    print(f"weight_page_shard={shard}")
    print(f"weight_page_file_offset={offset}")
    print(f"weight_page_tensor_bytes={tensor_size}")
    print(f"weight_page_materialized_bytes={size}")
    print("weight_page_materialize=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weight_plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--command-id", type=int)
    args = parser.parse_args()
    materialize(
        args.weight_plan.resolve(), args.output, args.size, args.command_id
    )


if __name__ == "__main__":
    main()
