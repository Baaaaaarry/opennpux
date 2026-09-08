#!/usr/bin/env python3
"""Create a standard ONNX graph with device-resident recurrent state."""

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
    parser.add_argument("requests", type=Path)
    parser.add_argument("expected", type=Path)
    args = parser.parse_args()

    token = np.arange(-8, 8, dtype=np.float32).reshape(2, 8) / 8.0
    state = np.arange(16, dtype=np.float32).reshape(2, 8) / 32.0
    weight = np.eye(8, dtype=np.float32) * 0.75
    delta = token @ weight
    expected = state + delta + delta
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["token", "projection_weight"], ["projected"]),
            helper.make_node("Add", ["projected", "state"], ["updated"]),
        ],
        "opennpux_stateful_decode",
        [
            helper.make_tensor_value_info("token", TensorProto.FLOAT, [2, 8]),
            helper.make_tensor_value_info("state", TensorProto.FLOAT, [2, 8]),
        ],
        [helper.make_tensor_value_info("updated", TensorProto.FLOAT, [2, 8])],
        [numpy_helper.from_array(weight, "projection_weight")],
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
        "state_updates": ["@output=state"],
        "state_appends": [],
        "module_values": {"state": state.reshape(-1).tolist()},
        "invocations": [
            {
                "name": f"decode-{index:03d}",
                "values": {"token": token.reshape(-1).tolist()},
                "scalars": {},
            }
            for index in range(2)
        ],
    }
    for path in (args.model, args.requests, args.expected):
        path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.model))
    args.requests.write_text(
        json.dumps(requests, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.expected.write_bytes(expected.astype("<f4").tobytes())
    print("onnx_stateful_decode=PASS")


if __name__ == "__main__":
    main()
