#!/usr/bin/env python3

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Iterable, List


FORMAT = "OPENNPUX_QWEN_TINY_V1"
DEFAULT_OUTPUT = "build/models/qwen-tiny.npxm"


def matmul_vector(vector: List[float], matrix: List[List[float]]) -> List[float]:
    return [
        sum(vector[index] * matrix[index][column] for index in range(len(vector)))
        for column in range(len(matrix[0]))
    ]


def add(left: List[float], right: List[float]) -> List[float]:
    return [a + b for a, b in zip(left, right)]


def rms_norm(vector: List[float], weight: List[float], epsilon: float) -> List[float]:
    mean_square = sum(value * value for value in vector) / len(vector)
    scale = 1.0 / math.sqrt(mean_square + epsilon)
    return [value * scale * weight[index] for index, value in enumerate(vector)]


def silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


def softmax(values: List[float]) -> List[float]:
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def transpose(matrix: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*matrix)]


def deterministic_matrix(rows: int, cols: int, seed: int, scale: float) -> List[List[float]]:
    values: List[List[float]] = []
    state = seed & 0x7FFFFFFF
    for _ in range(rows):
        row = []
        for _ in range(cols):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            centered = ((state % 257) - 128) / 128.0
            row.append(centered * scale)
        values.append(row)
    return values


def deterministic_vector(size: int, seed: int, base: float, scale: float) -> List[float]:
    return [base + scale * (((seed + index * 17) % 29) - 14) for index in range(size)]


def float_checksum(values: Iterable[float]) -> int:
    checksum = 2166136261
    for value in values:
        for byte in struct.pack("<f", float(value)):
            checksum ^= byte
            checksum = (checksum * 16777619) & 0xFFFFFFFF
    return checksum


def flatten(rows: Iterable[Iterable[float]]) -> List[float]:
    output: List[float] = []
    for row in rows:
        output.extend(row)
    return output


def build_package() -> dict:
    vocab_size = 16
    hidden_size = 8
    intermediate_size = 12
    head_count = 2
    head_dim = hidden_size // head_count
    epsilon = 1e-5
    input_ids = [1, 5, 7, 2]

    token_embedding = [
        deterministic_vector(hidden_size, 100 + token * 13, 0.0, 0.015)
        for token in range(vocab_size)
    ]
    lm_head = deterministic_matrix(hidden_size, vocab_size, 2001, 0.04)
    rms_attn_weight = deterministic_vector(hidden_size, 301, 1.0, 0.005)
    rms_ffn_weight = deterministic_vector(hidden_size, 401, 1.0, 0.004)
    wq = deterministic_matrix(hidden_size, hidden_size, 501, 0.035)
    wk = deterministic_matrix(hidden_size, hidden_size, 601, 0.035)
    wv = deterministic_matrix(hidden_size, hidden_size, 701, 0.035)
    wo = deterministic_matrix(hidden_size, hidden_size, 801, 0.035)
    w_gate = deterministic_matrix(hidden_size, intermediate_size, 901, 0.025)
    w_up = deterministic_matrix(hidden_size, intermediate_size, 1001, 0.025)
    w_down = deterministic_matrix(intermediate_size, hidden_size, 1101, 0.025)

    hidden = [token_embedding[token][:] for token in input_ids]

    normed = [rms_norm(token, rms_attn_weight, epsilon) for token in hidden]
    q_values = [matmul_vector(token, wq) for token in normed]
    k_values = [matmul_vector(token, wk) for token in normed]
    v_values = [matmul_vector(token, wv) for token in normed]

    attended: List[List[float]] = []
    for position, query in enumerate(q_values):
        context = [0.0] * hidden_size
        for head in range(head_count):
            begin = head * head_dim
            end = begin + head_dim
            scores = []
            for source in range(position + 1):
                dot = sum(
                    query[index] * k_values[source][index]
                    for index in range(begin, end)
                )
                scores.append(dot / math.sqrt(head_dim))
            probs = softmax(scores)
            for source, prob in enumerate(probs):
                for index in range(begin, end):
                    context[index] += prob * v_values[source][index]
        attended.append(context)

    attention_out = [matmul_vector(token, wo) for token in attended]
    hidden = [add(base, delta) for base, delta in zip(hidden, attention_out)]

    ffn_norm = [rms_norm(token, rms_ffn_weight, epsilon) for token in hidden]
    ffn_out: List[List[float]] = []
    for token in ffn_norm:
        gate = matmul_vector(token, w_gate)
        up = matmul_vector(token, w_up)
        gated = [silu(gate[index]) * up[index] for index in range(intermediate_size)]
        ffn_out.append(matmul_vector(gated, w_down))
    hidden = [add(base, delta) for base, delta in zip(hidden, ffn_out)]

    final_state = rms_norm(hidden[-1], rms_ffn_weight, epsilon)
    logits = matmul_vector(final_state, lm_head)
    next_token = max(range(vocab_size), key=lambda index: logits[index])

    operator_trace = [
        {"op": "EMBED", "count": len(input_ids)},
        {"op": "RMS_NORM", "layer": 0, "phase": "attention", "shape": [len(input_ids), hidden_size]},
        {"op": "MATMUL", "layer": 0, "name": "q_proj", "shape": [len(input_ids), hidden_size, hidden_size]},
        {"op": "MATMUL", "layer": 0, "name": "k_proj", "shape": [len(input_ids), hidden_size, hidden_size]},
        {"op": "MATMUL", "layer": 0, "name": "v_proj", "shape": [len(input_ids), hidden_size, hidden_size]},
        {"op": "ROPE", "layer": 0, "heads": head_count, "head_dim": head_dim},
        {"op": "SOFTMAX", "layer": 0, "phase": "attention", "axis": -1},
        {"op": "MATMUL", "layer": 0, "name": "o_proj", "shape": [len(input_ids), hidden_size, hidden_size]},
        {"op": "ADD", "layer": 0, "name": "attention_residual", "shape": [len(input_ids), hidden_size]},
        {"op": "RMS_NORM", "layer": 0, "phase": "ffn", "shape": [len(input_ids), hidden_size]},
        {"op": "MATMUL", "layer": 0, "name": "gate_proj", "shape": [len(input_ids), hidden_size, intermediate_size]},
        {"op": "MATMUL", "layer": 0, "name": "up_proj", "shape": [len(input_ids), hidden_size, intermediate_size]},
        {"op": "SILU", "layer": 0, "shape": [len(input_ids), intermediate_size]},
        {"op": "MUL", "layer": 0, "name": "swiglu_gate", "shape": [len(input_ids), intermediate_size]},
        {"op": "MATMUL", "layer": 0, "name": "down_proj", "shape": [len(input_ids), intermediate_size, hidden_size]},
        {"op": "ADD", "layer": 0, "name": "ffn_residual", "shape": [len(input_ids), hidden_size]},
        {"op": "RMS_NORM", "layer": 0, "phase": "final", "shape": [hidden_size]},
        {"op": "MATMUL", "layer": 0, "name": "lm_head", "shape": [hidden_size, vocab_size]},
        {"op": "TOPK", "k": 1, "vocab_size": vocab_size},
    ]

    weight_checksum = float_checksum(
        flatten(token_embedding)
        + flatten(lm_head)
        + rms_attn_weight
        + rms_ffn_weight
        + flatten(wq)
        + flatten(wk)
        + flatten(wv)
        + flatten(wo)
        + flatten(w_gate)
        + flatten(w_up)
        + flatten(w_down)
    )
    logits_checksum = float_checksum(logits)

    return {
        "format": FORMAT,
        "version": 1,
        "model": {
            "name": "qwen-tiny-synthetic",
            "family": "qwen-compatible",
            "layer_count": 1,
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "head_count": head_count,
            "head_dim": head_dim,
            "dtype": "float32-reference",
        },
        "prompt": {
            "text": "open npux",
            "input_ids": input_ids,
            "input_checksum": float_checksum(float(token) for token in input_ids),
        },
        "golden": {
            "next_token": next_token,
            "logits_checksum": f"0x{logits_checksum:08x}",
            "weight_checksum": f"0x{weight_checksum:08x}",
            "logits": [round(value, 9) for value in logits],
        },
        "operator_trace": operator_trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    package = build_package()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")

    if args.print_summary:
        print(f"qwen_model={package['model']['name']}")
        print(f"qwen_package={output}")
        print(f"qwen_next_token={package['golden']['next_token']}")
        print(f"qwen_logits_checksum={package['golden']['logits_checksum']}")
        print(f"qwen_weight_checksum={package['golden']['weight_checksum']}")
        print(f"qwen_operator_count={len(package['operator_trace'])}")
    else:
        print(f"built: {output}")


if __name__ == "__main__":
    main()
