#!/usr/bin/env python3
"""Build a public MAX Graph and export the same calls to OpenNPUX XGraph."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from opennpux_backend import compile_frontend
from opennpux_backend.artifacts import write_graph_artifact
from opennpux_backend.compiler import compile_graph
from opennpux_mojo import MaxExportBuilder, MaxGraphAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lowering-library")
    parser.add_argument("--require-max", action="store_true")
    args = parser.parse_args()
    try:
        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType, ops
    except ImportError as error:
        if args.require_max:
            raise SystemExit(
                "max_sdk_export=FAIL: MAX SDK unavailable; run "
                "./tools/models/setup_max_sdk_env.sh and retry with "
                ".venv/max-sdk/bin/python"
            ) from error
        print("max_sdk_export=SKIP reason=max-sdk-unavailable")
        return

    device = DeviceRef.CPU()
    lhs_type = TensorType(DType.float32, (2, 4), device=device)
    rhs_type = TensorType(DType.float32, (4, 3), device=device)
    recorder = MaxExportBuilder("opennpux_projection")
    with Graph(
        "opennpux_projection", input_types=[lhs_type, rhs_type]
    ) as graph:
        lhs = graph.inputs[0].tensor
        rhs = graph.inputs[1].tensor
        recorder.add_tensor("lhs", lhs, storage="input")
        recorder.add_tensor("rhs", rhs, storage="input")
        projected = recorder.call(
            "max.matmul",
            ops.matmul,
            [lhs, rhs],
            ["projected"],
            name="projection",
        )
        graph.output(projected)
        recorder.add_output(projected)

    compilation = compile_frontend(
        recorder, MaxGraphAdapter(), args.lowering_library
    )
    expected = {
        "format": "OPENNPUX_BACKEND_GRAPH_V1",
        "tensors": [
            {"name": "lhs", "shape": [2, 4], "dtype": "float32", "storage": "input"},
            {"name": "rhs", "shape": [4, 3], "dtype": "float32", "storage": "input"},
            {"name": "projected", "shape": [2, 3], "dtype": "float32", "storage": "output"},
        ],
        "nodes": [
            {"op": "matmul", "inputs": ["lhs", "rhs"], "outputs": ["projected"]}
        ],
        "outputs": ["projected"],
    }
    expected_artifact, _ = compile_graph(expected, args.lowering_library)
    if compilation["artifact"] != expected_artifact:
        raise SystemExit("max_sdk_export=FAIL: artifact differs from Backend IR")
    write_graph_artifact(
        args.output, compilation["artifact"], compilation["metadata"]
    )
    print(f"max_sdk_artifact={args.output}")
    print(f"max_sdk_commands={compilation['metadata']['command_count']}")
    print("max_sdk_export=PASS")


if __name__ == "__main__":
    main()
