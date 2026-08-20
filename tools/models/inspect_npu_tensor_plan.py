#!/usr/bin/env python3
"""Validate an OpenNPUX generic executable tensor allocation plan."""

import argparse
import json
from pathlib import Path
from typing import Any


FORMAT = "OPENNPUX_NPU_TENSOR_PLAN_V1"


def load_plan(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        plan = json.load(source)
    if not isinstance(plan, dict) or plan.get("format") != FORMAT:
        raise ValueError("unsupported NPU tensor plan")
    return plan


def validate(plan: dict[str, Any]) -> None:
    tensors = plan.get("tensors")
    command_io = plan.get("command_io")
    slots = plan.get("scratch_slots")
    if not isinstance(tensors, list) or not isinstance(command_io, list) or not isinstance(slots, list):
        raise ValueError("tensor plan tables are missing")
    if len(tensors) != int(plan.get("tensor_count", -1)):
        raise ValueError("tensor count mismatch")
    if len(command_io) != int(plan.get("command_count", -1)):
        raise ValueError("command count mismatch")
    if len(slots) != int(plan.get("scratch_slot_count", -1)):
        raise ValueError("scratch slot count mismatch")

    by_id: dict[int, dict[str, Any]] = {}
    for expected_id, tensor in enumerate(tensors):
        tensor_id = int(tensor.get("id", -1))
        if tensor_id != expected_id or tensor_id in by_id:
            raise ValueError("tensor IDs are not dense and unique")
        by_id[tensor_id] = tensor

    command_ids: set[int] = set()
    for record in command_io:
        command_id = int(record.get("command_id", -1))
        if command_id < 0 or command_id in command_ids:
            raise ValueError("command IDs are not unique")
        command_ids.add(command_id)
        inputs = record.get("input_tensor_ids", [])
        outputs = record.get("output_tensor_ids", [])
        if not inputs or not outputs:
            raise ValueError(f"command {command_id} has incomplete IO")
        for tensor_id in inputs:
            tensor = by_id.get(int(tensor_id))
            if tensor is None:
                raise ValueError(f"command {command_id} reads unknown tensor {tensor_id}")
            producer = tensor.get("producer_command")
            if producer is not None and int(producer) >= command_id:
                raise ValueError(f"command {command_id} reads tensor before production")
        for tensor_id in outputs:
            tensor = by_id.get(int(tensor_id))
            if tensor is None or int(tensor.get("producer_command", -1)) != command_id:
                raise ValueError(f"command {command_id} does not own output tensor {tensor_id}")

    if command_ids != set(range(len(command_io))):
        raise ValueError("command coverage is not dense")
    outputs = [tensor for tensor in tensors if tensor.get("storage") == "output"]
    if not outputs or any(tensor.get("producer_command") is None for tensor in outputs):
        raise ValueError("external output has no producer")

    active_by_slot: dict[int, list[tuple[int, int, int]]] = {}
    for tensor in tensors:
        if tensor.get("storage") != "scratch":
            continue
        slot = int(tensor.get("allocation_slot", -1))
        if slot < 0 or slot >= len(slots):
            raise ValueError("scratch tensor has invalid allocation slot")
        begin = int(tensor["producer_command"])
        end = int(tensor["last_consumer_command"])
        if end < begin:
            raise ValueError("scratch tensor lifetime is inverted")
        for other_begin, other_end, other_id in active_by_slot.setdefault(slot, []):
            if max(begin, other_begin) <= min(end, other_end):
                raise ValueError(
                    f"scratch slot {slot} overlaps tensors {other_id} and {tensor['id']}"
                )
        active_by_slot[slot].append((begin, end, int(tensor["id"])))

    expected_bytes = sum(int(slot["bytes_per_runtime_row"]) for slot in slots)
    if expected_bytes != int(plan.get("scratch_bytes_per_runtime_row", -1)):
        raise ValueError("scratch byte total mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    validate(plan)
    print(f"tensor_plan_commands={plan['command_count']}")
    print(f"tensor_plan_tensors={plan['tensor_count']}")
    print(f"tensor_plan_scratch_slots={plan['scratch_slot_count']}")
    print(f"tensor_plan_scratch_bytes_per_runtime_row={plan['scratch_bytes_per_runtime_row']}")
    print("tensor_plan=PASS")


if __name__ == "__main__":
    main()
