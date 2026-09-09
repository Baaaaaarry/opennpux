#!/usr/bin/env python3
"""Compile frontend-neutral OpenNPUX backend IR into XGraph artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from opennpux_backend.compiler import CodegenError, compile_graph, compile_module
from opennpux_backend.ir import is_graph, is_module


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CodegenError("backend IR must contain a JSON object")
    return value


def _write_module(
    output: Path,
    artifacts: dict[str, tuple[bytes, dict]],
    manifest: dict,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    for region in manifest["regions"]:
        binary, metadata = artifacts[region["name"]]
        artifact_path = output / region["artifact"]
        artifact_path.write_bytes(binary)
        Path(f"{artifact_path}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest_path = output / "module.npxgm.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


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
    args = parser.parse_args()
    try:
        source = _load(args.input)
        if is_graph(source):
            binary, metadata = compile_graph(source, args.lowering_library)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(binary)
            metadata_path = Path(f"{args.output}.json")
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"opennpux_backend_artifact={args.output}")
            print(f"opennpux_backend_commands={metadata['command_count']}")
        elif is_module(source):
            artifacts, manifest = compile_module(source, args.lowering_library)
            manifest_path = _write_module(args.output, artifacts, manifest)
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
