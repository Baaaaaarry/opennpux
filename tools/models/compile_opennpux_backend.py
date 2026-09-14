#!/usr/bin/env python3
"""Compile frontend-neutral OpenNPUX backend IR into XGraph artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from opennpux_backend.artifacts import write_graph_artifact, write_module_artifacts
from opennpux_backend.compiler import CodegenError, compile_graph, compile_module
from opennpux_backend.ir import is_graph, is_module


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CodegenError("backend IR must contain a JSON object")
    return value


def _shape_bindings(values: list[str]) -> dict[str, int]:
    result = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name:
            raise CodegenError(f"invalid shape binding {value!r}; expected NAME=VALUE")
        try:
            result[name] = int(raw, 0)
        except ValueError as error:
            raise CodegenError(f"invalid shape binding value in {value!r}") from error
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="lower frontend-neutral OpenNPUX backend IR"
    )
    parser.add_argument("input", type=Path, help="backend graph/module JSON")
    parser.add_argument("output", type=Path, help="output .npxg file or module directory")
    parser.add_argument(
        "--lowering-library",
        default=os.environ.get("OPENNPUX_XGRAPH_LOWERING_LIB"),
    )
    parser.add_argument(
        "--shape", action="append", default=[], metavar="NAME=VALUE",
        help="bind one symbolic Tensor dimension before command lowering",
    )
    args = parser.parse_args()
    try:
        source = _load(args.input)
        shape_bindings = _shape_bindings(args.shape)
        if is_graph(source):
            binary, metadata = compile_graph(
                source, args.lowering_library, shape_bindings
            )
            write_graph_artifact(args.output, binary, metadata)
            print(f"opennpux_backend_artifact={args.output}")
            print(f"opennpux_backend_commands={metadata['command_count']}")
        elif is_module(source):
            artifacts, manifest = compile_module(
                source, args.lowering_library, shape_bindings
            )
            manifest_path = write_module_artifacts(args.output, artifacts, manifest)
            print(f"opennpux_backend_manifest={manifest_path}")
            print(f"opennpux_backend_regions={manifest['region_count']}")
            print(f"opennpux_backend_commands={manifest['total_commands']}")
        else:
            raise CodegenError("unsupported OpenNPUX backend IR format")
    except (OSError, ValueError, json.JSONDecodeError, CodegenError) as error:
        print(f"opennpux_backend_codegen=FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("opennpux_backend_codegen=PASS")


if __name__ == "__main__":
    main()
