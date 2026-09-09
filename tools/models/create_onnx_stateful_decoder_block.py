#!/usr/bin/env python3
"""Create a standard ONNX stateful attention and gated-MLP decoder block."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-value))


def decoder_step(
    hidden: np.ndarray, state: np.ndarray, weights: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    query = hidden @ weights["query_weight"]
    key = hidden @ weights["key_weight"]
    value = hidden @ weights["value_weight"]
    scores = query @ np.swapaxes(key, -1, -2)
    scores = scores * weights["attention_scale"] + weights["causal_mask"]
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(shifted) / np.sum(
        np.exp(shifted), axis=-1, keepdims=True
    )
    projected = (probabilities @ value) @ weights["output_weight"]
    updated_state = projected + state
    attention_residual = updated_state + hidden
    gate = attention_residual @ weights["gate_weight"]
    up = attention_residual @ weights["up_weight"]
    down = (silu(gate) * up) @ weights["down_weight"]
    return attention_residual + down, updated_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("requests", type=Path)
    parser.add_argument("expected", type=Path)
    args = parser.parse_args()

    hidden0 = np.array(
        [[-1.0, 0.5, 1.0, 2.0], [0.25, 1.5, -0.5, 1.0]],
        dtype=np.float32,
    )
    hidden1 = hidden0 * np.float32(0.75) + np.float32(0.125)
    state = np.arange(8, dtype=np.float32).reshape(2, 4) / np.float32(32.0)
    identity = np.eye(4, dtype=np.float32)
    weights = {
        "query_weight": identity + np.roll(identity, 1, axis=1) * 0.25,
        "key_weight": identity * 0.5 + np.flip(identity, axis=1) * 0.25,
        "value_weight": identity - np.roll(identity, 2, axis=1) * 0.125,
        "output_weight": identity * 0.75 + np.roll(identity, 1, axis=1) * 0.25,
        "gate_weight": np.arange(24, dtype=np.float32).reshape(4, 6) / 64.0,
        "up_weight": np.flip(
            np.arange(24, dtype=np.float32).reshape(4, 6), axis=1
        ) / 96.0,
        "down_weight": (
            np.arange(24, dtype=np.float32).reshape(6, 4) - 12.0
        ) / 80.0,
        "attention_scale": np.array([1.0 / math.sqrt(4.0)], dtype=np.float32),
        "causal_mask": np.array(
            [[0.0, -10000.0], [0.0, 0.0]], dtype=np.float32
        ),
    }
    _, state1 = decoder_step(hidden0, state, weights)
    expected, _ = decoder_step(hidden1, state1, weights)

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["hidden", "query_weight"], ["query"]),
            helper.make_node("MatMul", ["hidden", "key_weight"], ["key"]),
            helper.make_node("MatMul", ["hidden", "value_weight"], ["value"]),
            helper.make_node("Transpose", ["key"], ["key_t"], perm=[1, 0]),
            helper.make_node("MatMul", ["query", "key_t"], ["scores"]),
            helper.make_node("Mul", ["scores", "attention_scale"], ["scaled"]),
            helper.make_node("Add", ["scaled", "causal_mask"], ["masked"]),
            helper.make_node("Softmax", ["masked"], ["probabilities"], axis=-1),
            helper.make_node("MatMul", ["probabilities", "value"], ["context"]),
            helper.make_node("MatMul", ["context", "output_weight"], ["projected"]),
            helper.make_node("Add", ["projected", "state"], ["state_mixed"]),
            helper.make_node("Add", ["state_mixed", "hidden"], ["attention_out"]),
            helper.make_node("MatMul", ["attention_out", "gate_weight"], ["gate"]),
            helper.make_node("MatMul", ["attention_out", "up_weight"], ["up"]),
            helper.make_node("Sigmoid", ["gate"], ["gate_sigmoid"]),
            helper.make_node("Mul", ["gate", "gate_sigmoid"], ["gate_silu"]),
            helper.make_node("Mul", ["gate_silu", "up"], ["gated"]),
            helper.make_node("MatMul", ["gated", "down_weight"], ["down"]),
            helper.make_node("Add", ["attention_out", "down"], ["output"]),
        ],
        "opennpux_stateful_decoder_block",
        [
            helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("state", TensorProto.FLOAT, [2, 4]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])],
        [numpy_helper.from_array(value, name) for name, value in weights.items()],
    )
    model = helper.make_model(
        graph,
        producer_name="opennpux-system-test",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    requests = {
        "format": "OPENNPUX_TVM_ONNX_REQUESTS_V1",
        "state_parameters": ["state"],
        "state_updates": ["@update=state"],
        "state_appends": [],
        "module_values": {"state": state.reshape(-1).tolist()},
        "invocations": [
            {
                "name": f"decode-{index:03d}",
                "values": {"hidden": hidden.reshape(-1).tolist()},
                "scalars": {},
            }
            for index, hidden in enumerate((hidden0, hidden1))
        ],
    }
    for path in (args.model, args.requests, args.expected):
        path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.model))
    args.requests.write_text(
        json.dumps(requests, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.expected.write_bytes(expected.astype("<f4").tobytes())
    print("onnx_stateful_decoder_block=PASS")


if __name__ == "__main__":
    main()
