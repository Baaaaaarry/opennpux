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

    query = np.array(
        [[[-1.0, 0.5, 1.0, 2.0], [0.25, 1.5, -0.5, 1.0]]],
        dtype=np.float32,
    )
    key = np.array(
        [[0.5, 1.0, -0.5, 0.75], [-1.0, 0.25, 1.5, 0.5]],
        dtype=np.float32,
    )
    value = np.array(
        [[1.0, 2.0, 3.0, 4.0], [-2.0, 1.0, 0.5, 3.0]], dtype=np.float32
    )
    residual = np.arange(8, dtype=np.float32).reshape(1, 2, 4) / 16.0
    scale = np.array([1.0 / math.sqrt(query.shape[-1])], dtype=np.float32)
    causal_mask = np.array(
        [[[0.0, -10000.0], [0.0, 0.0]]], dtype=np.float32
    )
    scores = (query @ np.swapaxes(key, -1, -2)) * scale + causal_mask
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    expected = probabilities @ value + residual

    graph = helper.make_graph(
        [
            helper.make_node("Reshape", ["query_flat", "query_shape"], ["query"]),
            helper.make_node("Transpose", ["key"], ["key_transposed"], perm=[1, 0]),
            helper.make_node("MatMul", ["query", "key_transposed"], ["scores"]),
            helper.make_node("Mul", ["scores", "attention_scale"], ["scaled_scores"]),
            helper.make_node("Add", ["scaled_scores", "causal_mask"], ["masked_scores"]),
            helper.make_node("Softmax", ["masked_scores"], ["probabilities"], axis=-1),
            helper.make_node("MatMul", ["probabilities", "value"], ["context"]),
            helper.make_node("Add", ["context", "residual"], ["output"]),
        ],
        "opennpux_decomposed_attention",
        [
            helper.make_tensor_value_info("query_flat", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("residual", TensorProto.FLOAT, [1, 2, 4]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 4])],
        [
            numpy_helper.from_array(np.array([1, 2, 4], dtype=np.int64), "query_shape"),
            numpy_helper.from_array(key, "key"),
            numpy_helper.from_array(value, "value"),
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
                "query_flat": query.reshape(-1).tolist(),
                "residual": residual.reshape(-1).tolist(),
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
