#!/usr/bin/env python3
"""Compare matching vLLM and OpenNPUX decoder-layer boundary traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_records(path: Path, source: str, step: int) -> dict[int, list[float]]:
    records: dict[int, list[float]] = {}
    with path.open(encoding="utf-8") as trace:
        for line_number, line in enumerate(trace, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("source") != source or int(value.get("step", -1)) != step:
                continue
            if source == "vllm" and int(value.get("rank", 0)) != 0:
                continue
            layer = int(value["layer"])
            vector = [float(item) for item in value["values"]]
            if not vector:
                raise ValueError(f"{path}:{line_number}: empty layer vector")
            records[layer] = vector
    if not records:
        raise ValueError(f"{path}: no {source} records for step {step}")
    return records


def metrics(left: list[float], right: list[float]) -> tuple[float, float, float]:
    if len(left) != len(right):
        raise ValueError(f"vector size differs: {len(left)} != {len(right)}")
    square_error = 0.0
    maximum = 0.0
    left_square = 0.0
    right_square = 0.0
    dot = 0.0
    for lhs, rhs in zip(left, right):
        difference = lhs - rhs
        square_error += difference * difference
        maximum = max(maximum, abs(difference))
        left_square += lhs * lhs
        right_square += rhs * rhs
        dot += lhs * rhs
    rmse = math.sqrt(square_error / len(left))
    denominator = math.sqrt(left_square * right_square)
    cosine = dot / denominator if denominator else 0.0
    return rmse, maximum, cosine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_trace", type=Path)
    parser.add_argument("host_trace", type=Path)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--rmse-threshold", type=float, default=1e-2)
    args = parser.parse_args()

    reference = load_records(args.vllm_trace, "vllm", args.step)
    host = load_records(args.host_trace, "host-cpp", args.step)
    common = sorted(reference.keys() & host.keys())
    if not common:
        raise SystemExit("layer_trace_compare=FAIL reason=no-common-layers")
    first_divergent = None
    for layer in common:
        rmse, maximum, cosine = metrics(reference[layer], host[layer])
        print(
            f"layer_trace=step:{args.step},layer:{layer},rmse:{rmse:.9g},"
            f"max_abs:{maximum:.9g},cosine:{cosine:.9g}"
        )
        if first_divergent is None and rmse > args.rmse_threshold:
            first_divergent = layer
    print(
        "layer_trace_first_divergent="
        + ("none" if first_divergent is None else str(first_divergent))
    )
    print(f"layer_trace_compared_layers={len(common)}")
    print("layer_trace_compare=PASS")


if __name__ == "__main__":
    main()
