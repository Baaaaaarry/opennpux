#!/usr/bin/env python3
"""Compile a MAX/Mojo graph export through the common OpenNPUX backend."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from opennpux_backend import compile_frontend
from opennpux_backend.artifacts import write_graph_artifact, write_module_artifacts
from opennpux_backend.compiler import CodegenError
from opennpux_mojo import MaxGraphAdapter


def _bindings(values: list[str]) -> dict[str, int]:
    result = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name:
            raise ValueError(f"invalid shape binding {value!r}; expected NAME=VALUE")
        result[name] = int(raw, 0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="MAX graph export JSON")
    parser.add_argument("output", type=Path, help="output .npxg or module directory")
    parser.add_argument("--shape", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument(
        "--lowering-library",
        default=os.environ.get("OPENNPUX_XGRAPH_LOWERING_LIB"),
    )
    args = parser.parse_args()
    try:
        source = json.loads(args.input.read_text(encoding="utf-8"))
        compilation = compile_frontend(
            source,
            MaxGraphAdapter(),
            args.lowering_library,
            shape_bindings=_bindings(args.shape),
        )
        if compilation["kind"] == "graph":
            write_graph_artifact(
                args.output, compilation["artifact"], compilation["metadata"]
            )
            print(f"mojo_max_artifact={args.output}")
            print(f"mojo_max_commands={compilation['metadata']['command_count']}")
        else:
            path = write_module_artifacts(
                args.output, compilation["artifacts"], compilation["manifest"]
            )
            print(f"mojo_max_manifest={path}")
            print(f"mojo_max_regions={compilation['manifest']['region_count']}")
        print("mojo_max_codegen=PASS")
    except (OSError, ValueError, json.JSONDecodeError, CodegenError) as error:
        print(f"mojo_max_codegen=FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
