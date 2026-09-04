#!/usr/bin/env python3
"""Compile a TVM/OpenNPUX BYOC graph into an XGraph command artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from opennpux_tvm_byoc import CodegenError, compile_graph
from opennpux_tvm_byoc.xgraph_codegen import FORMAT


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise CodegenError("input must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="lower a normalized TVM BYOC graph to XGraph v2 commands"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="normalized BYOC graph JSON or a TVM IRModule JSON",
    )
    parser.add_argument("output", type=Path, help="output .npxg command artifact")
    parser.add_argument(
        "--metadata",
        type=Path,
        help="inspectable metadata path (default: <output>.json)",
    )
    parser.add_argument(
        "--partitioned",
        action="store_true",
        help="TVM IR input is already partitioned for Codegen=opennpux",
    )
    parser.add_argument(
        "--dump-byoc-graph",
        type=Path,
        help="write the normalized graph extracted from TVM",
    )
    parser.add_argument(
        "--lowering-library",
        default=os.environ.get("OPENNPUX_XGRAPH_LOWERING_LIB"),
        help="runtime C lowering shared library for tiled/composite operations",
    )
    args = parser.parse_args()
    metadata_path = args.metadata or Path(f"{args.output}.json")
    try:
        source = load_json(args.input)
        if source.get("format") == FORMAT:
            graph = source
        else:
            try:
                import tvm
            except ImportError as error:
                raise CodegenError(
                    "input is not normalized BYOC JSON and Apache TVM is unavailable"
                ) from error
            from opennpux_tvm_byoc.relax_backend import (
                normalized_graph_from_relax,
                partition_for_opennpux,
            )

            module = tvm.ir.load_json(args.input.read_text(encoding="utf-8"))
            if not args.partitioned:
                module = partition_for_opennpux(module)
            graph = normalized_graph_from_relax(module)
        if args.dump_byoc_graph is not None:
            args.dump_byoc_graph.write_text(
                json.dumps(graph, indent=2, sort_keys=True) + "\n"
            )
        binary, metadata = compile_graph(graph, args.lowering_library)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(binary)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    except (OSError, CodegenError, json.JSONDecodeError) as error:
        print(f"xgraph_codegen=FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"xgraph_artifact={args.output}")
    print(f"xgraph_metadata={metadata_path}")
    print(f"xgraph_commands={metadata['command_count']}")
    print(f"xgraph_arena_bytes={metadata['arena_size']}")
    print("xgraph_codegen=PASS")


if __name__ == "__main__":
    main()
