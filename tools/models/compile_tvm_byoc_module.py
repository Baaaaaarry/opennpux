#!/usr/bin/env python3
"""Compile a normalized multi-region BYOC module into XGraph artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from opennpux_tvm_byoc import CodegenError
from opennpux_tvm_byoc.module_codegen import MODULE_FORMAT, compile_module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="normalized module or TVM IRModule JSON")
    parser.add_argument("output", type=Path, help="output artifact directory")
    parser.add_argument(
        "--partitioned",
        action="store_true",
        help="TVM IR input is already partitioned for Codegen=opennpux",
    )
    parser.add_argument("--dump-byoc-module", type=Path)
    parser.add_argument(
        "--lowering-library",
        default=os.environ.get("OPENNPUX_XGRAPH_LOWERING_LIB"),
    )
    args = parser.parse_args()
    try:
        source = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            raise CodegenError("input must contain a JSON object")
        if source.get("format") != MODULE_FORMAT:
            try:
                import tvm
            except ImportError as error:
                raise CodegenError(
                    "input is not a normalized BYOC module and Apache TVM is unavailable"
                ) from error
            from opennpux_tvm_byoc.relax_backend import (
                normalized_module_from_relax,
                partition_for_opennpux,
            )

            tvm_module = tvm.ir.load_json(args.input.read_text(encoding="utf-8"))
            if not args.partitioned:
                tvm_module = partition_for_opennpux(tvm_module)
            source = normalized_module_from_relax(tvm_module)
        if args.dump_byoc_module is not None:
            args.dump_byoc_module.write_text(
                json.dumps(source, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        artifacts, manifest = compile_module(source, args.lowering_library)
        args.output.mkdir(parents=True, exist_ok=True)
        for region in manifest["regions"]:
            binary, metadata = artifacts[region["name"]]
            artifact_path = args.output / region["artifact"]
            artifact_path.write_bytes(binary)
            Path(f"{artifact_path}.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        manifest_path = args.output / "module.npxgm.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError, CodegenError) as error:
        print(f"xgraph_module_codegen=FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"xgraph_module_manifest={manifest_path}")
    print(f"xgraph_module_regions={manifest['region_count']}")
    print(f"xgraph_module_commands={manifest['total_commands']}")
    print("xgraph_module_codegen=PASS")


if __name__ == "__main__":
    main()
