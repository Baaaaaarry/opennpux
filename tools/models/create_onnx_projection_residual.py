#!/usr/bin/env python3
"""Create a small standard ONNX projection-residual graph for system tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("deployment", type=Path)
    parser.add_argument("expected", type=Path)
    args = parser.parse_args()

    hidden = np.arange(-8, 8, dtype=np.float32).reshape(2, 8) / 4.0
    residual = np.arange(16, dtype=np.float32).reshape(2, 8) / 16.0
    weight = np.eye(8, dtype=np.float32) * 1.5
    weight += np.fliplr(np.eye(8, dtype=np.float32)) * 0.25
    expected = hidden @ weight + residual

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["hidden", "projection_weight"], ["projected"]),
            helper.make_node("Add", ["projected", "residual"], ["output"]),
        ],
        "opennpux_projection_residual",
        [
            helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [2, 8]),
            helper.make_tensor_value_info("residual", TensorProto.FLOAT, [2, 8]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 8])],
        [numpy_helper.from_array(weight, "projection_weight")],
    )
    model = helper.make_model(
        graph,
        producer_name="opennpux-system-test",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)

    deployment = {
        "format": "OPENNPUX_TVM_BYOC_DEPLOYMENT_V1",
        "constant_parameters": ["projection_weight"],
        "state_parameters": [],
        "state_updates": [],
        "state_appends": [],
        "module_values": {"projection_weight": weight.reshape(-1).tolist()},
        "invocations": [{
            "name": "request-000",
            "values": {
                "hidden": hidden.reshape(-1).tolist(),
                "residual": residual.reshape(-1).tolist(),
            },
            "scalars": {},
        }],
    }
    for path in (args.model, args.deployment, args.expected):
        path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.model))
    args.deployment.write_text(
        json.dumps(deployment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.expected.write_bytes(expected.astype("<f4").tobytes())
    print(f"onnx_projection_residual_model={args.model}")
    print("onnx_projection_residual=PASS")


if __name__ == "__main__":
    main()
