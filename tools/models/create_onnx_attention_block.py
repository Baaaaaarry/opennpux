#!/usr/bin/env python3
"""Create a decomposed standard ONNX attention block for system tests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("requests", type=Path)
    parser.add_argument("expected", type=Path)
    args = parser.parse_args()

    hidden_states = np.array(
        [[-1.0, 0.5, 1.0, 2.0], [0.25, 1.5, -0.5, 1.0]],
        dtype=np.float32,
    )
    identity = np.eye(4, dtype=np.float32)
    query_weight = identity + np.roll(identity, 1, axis=1) * 0.25
    key_weight = identity * 0.5 + np.flip(identity, axis=1) * 0.25
    value_weight = identity - np.roll(identity, 2, axis=1) * 0.125
    output_weight = identity * 0.75 + np.roll(identity, 1, axis=1) * 0.25
    query = hidden_states @ query_weight
    key = hidden_states @ key_weight
    value = hidden_states @ value_weight
    scale = np.array([1.0 / math.sqrt(query.shape[-1])], dtype=np.float32)
    causal_mask = np.array(
        [[0.0, -10000.0], [0.0, 0.0]], dtype=np.float32
    )
    scores = (query @ np.swapaxes(key, -1, -2)) * scale + causal_mask
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    expected = (probabilities @ value) @ output_weight + hidden_states

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["hidden_states", "query_weight"], ["query"]),
            helper.make_node("MatMul", ["hidden_states", "key_weight"], ["key"]),
            helper.make_node("MatMul", ["hidden_states", "value_weight"], ["value"]),
            helper.make_node("Transpose", ["key"], ["key_transposed"], perm=[1, 0]),
            helper.make_node("MatMul", ["query", "key_transposed"], ["scores"]),
            helper.make_node("Mul", ["scores", "attention_scale"], ["scaled_scores"]),
            helper.make_node("Add", ["scaled_scores", "causal_mask"], ["masked_scores"]),
            helper.make_node("Softmax", ["masked_scores"], ["probabilities"], axis=-1),
            helper.make_node("MatMul", ["probabilities", "value"], ["context"]),
            helper.make_node("MatMul", ["context", "output_weight"], ["projected"]),
            helper.make_node("Add", ["projected", "hidden_states"], ["output"]),
        ],
        "opennpux_decomposed_attention",
        [
            helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [2, 4]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])],
        [
            numpy_helper.from_array(query_weight, "query_weight"),
            numpy_helper.from_array(key_weight, "key_weight"),
            numpy_helper.from_array(value_weight, "value_weight"),
            numpy_helper.from_array(output_weight, "output_weight"),
            numpy_helper.from_array(scale, "attention_scale"),
            numpy_helper.from_array(causal_mask, "causal_mask"),
        ],
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
        "state_parameters": [],
        "state_updates": [],
        "state_appends": [],
        "module_values": {},
        "invocations": [{
            "name": "prefill-000",
            "values": {
                "hidden_states": hidden_states.reshape(-1).tolist(),
            },
            "scalars": {},
        }],
    }
    for path in (args.model, args.requests, args.expected):
        path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.model))
    args.requests.write_text(
        json.dumps(requests, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.expected.write_bytes(expected.astype("<f4").tobytes())
    print("onnx_attention_block=PASS")


if __name__ == "__main__":
    main()
