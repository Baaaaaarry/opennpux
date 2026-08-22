#!/usr/bin/env python3
"""Compare matching vLLM and OpenNPUX decoder-layer boundary traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_records(path: Path, source: str, step: int) -> list[dict]:
    records: list[dict] = []
    available_steps: set[int] = set()
    with path.open(encoding="utf-8") as trace:
        for line_number, line in enumerate(trace, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("source") != source:
                continue
            record_step = int(value.get("step", -1))
            available_steps.add(record_step)
            if record_step != step:
                continue
            if source == "vllm" and int(value.get("rank", 0)) != 0:
                continue
            vector = [float(item) for item in value["values"]]
            if not vector:
                raise ValueError(f"{path}:{line_number}: empty layer vector")
            value["values"] = vector
            records.append(value)
    if not records:
        available = ",".join(str(value) for value in sorted(available_steps))
        raise ValueError(
            f"{path}: no {source} records for step {step}; "
            f"available steps: {available or 'none'}"
        )
    return records


def layer_boundaries(records: list[dict]) -> dict[int, list[float]]:
    return {
        int(record["layer"]): record["values"]
        for record in records
        if record.get("point", "layer_boundary") == "layer_boundary"
    }


def plan_metadata(
    path: Path,
) -> tuple[dict[tuple[int, int], str], dict[int, str]]:
    plan = json.loads(path.read_text())
    names: dict[tuple[int, int], str] = {}
    layer_types: dict[int, str] = {}
    for layer in plan.get("layers", []):
        layer_index = int(layer["index"])
        layer_types[layer_index] = str(layer.get("type", "unknown"))
        for phase_index, phase in enumerate(layer.get("phases", [])):
            names[(layer_index, phase_index)] = str(phase)
    return names, layer_types


def phase_points(records: list[dict], plan_path: Path) -> dict[tuple[int, str], list[float]]:
    names, _ = plan_metadata(plan_path)
    residual_count: dict[int, int] = {}
    points: dict[tuple[int, str], list[float]] = {}
    phase_output_to_point = {
        ("attention_norm", 0): "attention_norm",
        ("qkv_projection", 0): "qkv_query",
        ("qkv_projection", 1): "qkv_key",
        ("rope", 0): "rope_query",
        ("rope", 1): "rope_key",
        ("scaled_dot_product_attention", 0): "attention_core",
        ("attention_output_projection", 0): "attention_output_projection",
        ("recurrent_state_update", 0): "recurrent_state_update",
        ("linear_attention_gate_norm", 0): "linear_attention_gate_norm",
        ("linear_attention_output_projection", 0):
            "linear_attention_output_projection",
        ("ffn_norm", 0): "ffn_norm",
        ("moe_combine", 0): "moe",
    }
    for record in records:
        if record.get("point") != "command":
            continue
        layer = int(record["layer"])
        phase = names.get((layer, int(record["phase_index"])))
        output = int(record.get("output", -1))
        point = phase_output_to_point.get((phase or "", output))
        if phase == "residual_add" and output == 0:
            index = residual_count.get(layer, 0)
            residual_count[layer] = index + 1
            point = "attention_residual" if index == 0 else "layer_boundary"
        if point is not None:
            points[(layer, point)] = record["values"]
        if phase in {
            "attention_output_projection",
            "linear_attention_output_projection",
        } and output == 0:
            points[(layer, "token_mixer")] = record["values"]
    return points


def metrics(
    left: list[float], right: list[float]
) -> tuple[float, float, float, float, float]:
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
    scale = dot / right_square if right_square else 0.0
    aligned_square_error = sum(
        (lhs - scale * rhs) ** 2 for lhs, rhs in zip(left, right)
    )
    aligned_rmse = math.sqrt(aligned_square_error / len(left))
    return rmse, maximum, cosine, scale, aligned_rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_trace", type=Path)
    parser.add_argument("host_trace", type=Path)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--rmse-threshold", type=float, default=1e-2)
    parser.add_argument("--execution-plan", type=Path)
    args = parser.parse_args()

    reference_records = load_records(args.vllm_trace, "vllm", args.step)
    host_records = load_records(args.host_trace, "host-cpp", args.step)
    reference = layer_boundaries(reference_records)
    host = layer_boundaries(host_records)
    common = sorted(reference.keys() & host.keys())
    if not common:
        raise SystemExit("layer_trace_compare=FAIL reason=no-common-layers")
    first_divergent = None
    for layer in common:
        rmse, maximum, cosine, scale, aligned_rmse = metrics(
            reference[layer], host[layer]
        )
        print(
            f"layer_trace=step:{args.step},layer:{layer},rmse:{rmse:.9g},"
            f"max_abs:{maximum:.9g},cosine:{cosine:.9g},"
            f"reference_over_host_scale:{scale:.9g},"
            f"scale_aligned_rmse:{aligned_rmse:.9g}"
        )
        if first_divergent is None and rmse > args.rmse_threshold:
            first_divergent = layer
    print(
        "layer_trace_first_divergent="
        + ("none" if first_divergent is None else str(first_divergent))
    )
    print(f"layer_trace_compared_layers={len(common)}")
    if args.execution_plan is not None:
        _, layer_types = plan_metadata(args.execution_plan)
        reference_embedding = next(
            (
                record["values"]
                for record in reference_records
                if record.get("point") == "embedding"
            ),
            None,
        )
        host_embedding = next(
            (
                record["values"]
                for record in host_records
                if record.get("point") == "embedding"
            ),
            None,
        )
        if reference_embedding is not None and host_embedding is not None:
            rmse, maximum, cosine, scale, aligned_rmse = metrics(
                reference_embedding, host_embedding
            )
            print(
                f"phase_trace=step:{args.step},layer:-1,point:embedding,"
                f"rmse:{rmse:.9g},max_abs:{maximum:.9g},cosine:{cosine:.9g},"
                f"reference_over_host_scale:{scale:.9g},"
                f"scale_aligned_rmse:{aligned_rmse:.9g}"
            )
        reference_phases = {
            (int(record["layer"]), str(record["point"])): record["values"]
            for record in reference_records
            if record.get("point") != "layer_boundary"
        }
        host_phases = phase_points(host_records, args.execution_plan)
        phase_order = (
            "attention_norm",
            "qkv_query",
            "qkv_key",
            "rope_query",
            "rope_key",
            "attention_core",
            "attention_output_projection",
            "recurrent_state_update",
            "linear_attention_gate_norm",
            "linear_attention_output_projection",
            "token_mixer",
            "attention_residual",
            "ffn_norm",
            "moe",
            "layer_boundary",
        )
        first_phase = None
        compared_phases = 0
        for layer in common:
            for point in phase_order:
                key = (layer, point)
                if key not in reference_phases or key not in host_phases:
                    continue
                rmse, maximum, cosine, scale, aligned_rmse = metrics(
                    reference_phases[key], host_phases[key]
                )
                print(
                    f"phase_trace=step:{args.step},layer:{layer},"
                    f"layer_type:{layer_types.get(layer, 'unknown')},"
                    f"point:{point},"
                    f"rmse:{rmse:.9g},max_abs:{maximum:.9g},cosine:{cosine:.9g},"
                    f"reference_over_host_scale:{scale:.9g},"
                    f"scale_aligned_rmse:{aligned_rmse:.9g}"
                )
                compared_phases += 1
                if first_phase is None and rmse > args.rmse_threshold:
                    first_phase = (layer, point)
        print(
            "phase_trace_first_divergent="
            + (
                "none"
                if first_phase is None
                else f"layer:{first_phase[0]},point:{first_phase[1]}"
            )
        )
        print(f"phase_trace_compared_points={compared_phases}")
    print("layer_trace_compare=PASS")


if __name__ == "__main__":
    main()
