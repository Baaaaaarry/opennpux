#!/usr/bin/env python3
"""Compile a TVM/OpenNPUX BYOC graph into an XGraph command artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from opennpux_backend.artifacts import write_graph_artifact
from opennpux_backend.compiler import CodegenError, compile_graph
from opennpux_backend.frontend import adapt_frontend
from opennpux_backend.ir import is_graph


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
    parser.add_argument("--constant-parameter", action="append", default=[])
    parser.add_argument("--state-parameter", action="append", default=[])
    args = parser.parse_args()
    metadata_path = args.metadata or Path(f"{args.output}.json")
    try:
        source = load_json(args.input)
        if is_graph(source):
            graph = source
        else:
            try:
                import tvm
            except ImportError as error:
                raise CodegenError(
                    "input is not normalized BYOC JSON and Apache TVM is unavailable"
                ) from error
            from opennpux_tvm_byoc.relax_backend import RelaxFrontendAdapter

            module = tvm.ir.load_json(args.input.read_text(encoding="utf-8"))
            graph = adapt_frontend(
                module,
                RelaxFrontendAdapter(module=False, partitioned=args.partitioned),
            )
        from opennpux_backend.storage_policy import apply_parameter_storage

        apply_parameter_storage(
            graph, args.constant_parameter, args.state_parameter
        )
        if args.dump_byoc_graph is not None:
            args.dump_byoc_graph.write_text(
                json.dumps(graph, indent=2, sort_keys=True) + "\n"
            )
        binary, metadata = compile_graph(graph, args.lowering_library)
        write_graph_artifact(args.output, binary, metadata, metadata_path)
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
