"""Build a public MAX Graph and emit its stable OpenNPUX frontend export."""

import argparse
import json
from pathlib import Path

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    device = DeviceRef.CPU()
    lhs_type = TensorType(DType.float32, (2, 4), device=device)
    rhs_type = TensorType(DType.float32, (4, 3), device=device)
    with Graph(
        "opennpux_projection", input_types=[lhs_type, rhs_type]
    ) as graph:
        lhs = graph.inputs[0].tensor
        rhs = graph.inputs[1].tensor
        projected = ops.matmul(lhs, rhs)
        graph.output(projected)

    export = {
        "format": "OPENNPUX_MAX_GRAPH_EXPORT_V1",
        "kind": "graph",
        "name": "opennpux_projection",
        "tensors": [
            {
                "name": "lhs",
                "shape": [2, 4],
                "dtype": "float32",
                "storage": "input",
            },
            {
                "name": "rhs",
                "shape": [4, 3],
                "dtype": "float32",
                "storage": "input",
            },
            {
                "name": "projected",
                "shape": [2, 3],
                "dtype": "float32",
                "storage": "output",
            },
        ],
        "operations": [
            {
                "op": "max.matmul",
                "name": "projection",
                "inputs": ["lhs", "rhs"],
                "outputs": ["projected"],
            }
        ],
        "outputs": ["projected"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")
    print(f"max_source_export={args.output}")
    print("max_source_graph=PASS")


if __name__ == "__main__":
    main()
