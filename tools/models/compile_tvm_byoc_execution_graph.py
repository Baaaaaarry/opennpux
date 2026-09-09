#!/usr/bin/env python3
"""Compile a relocatable full-model graph for the TVM OpenNPUX BYOC path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from opennpux_tvm_byoc.execution_graph import (
    build_execution_graph, save_relax_execution_module,
)
from opennpux_tvm_byoc.xgraph_codegen import CodegenError


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CodegenError(f"{path}: root must be an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path, help="model.npxe JSON")
    parser.add_argument("tensor_plan", type=Path, help="model.npxt JSON")
    parser.add_argument("output", type=Path, help="output model.npxtvm JSON")
    parser.add_argument("--require-node-count", type=int)
    parser.add_argument("--relax-output", type=Path)
    parser.add_argument("--require-tvm", action="store_true")
    args = parser.parse_args()
    try:
        graph = build_execution_graph(_load(args.executable), _load(args.tensor_plan))
        if (args.require_node_count is not None and
                graph["node_count"] != args.require_node_count):
            raise CodegenError(
                f"expected {args.require_node_count} nodes, found {graph['node_count']}"
            )
        if args.relax_output is not None:
            graph["relax_ir"] = {
                "path": args.relax_output.name,
                "sha256": save_relax_execution_module(graph, args.relax_output),
            }
        elif args.require_tvm:
            raise CodegenError("--require-tvm requires --relax-output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError, CodegenError) as error:
        print(f"tvm_byoc_execution_graph=FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"tvm_byoc_execution_graph={args.output}")
    print(f"tvm_byoc_execution_nodes={graph['node_count']}")
    print(f"tvm_byoc_execution_tensors={graph['tensor_count']}")
    print("tvm_byoc_execution_operations=" + ",".join(
        f"{name}:{count}" for name, count in graph["operation_counts"].items()
    ))
    if "relax_ir" in graph:
        print(f"tvm_byoc_relax_module={args.relax_output}")
        print(f"tvm_byoc_relax_sha256={graph['relax_ir']['sha256']}")
    print("tvm_byoc_execution_graph=PASS")


if __name__ == "__main__":
    main()
