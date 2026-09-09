#!/usr/bin/env python3
"""Compile a normalized multi-region BYOC module into XGraph artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from opennpux_tvm_byoc import CodegenError
from opennpux_backend.ir import is_module
from opennpux_tvm_byoc.module_codegen import compile_module


def parse_parameter_alias(value: str) -> tuple[str, str]:
    internal, separator, source = value.partition("=")
    if not separator or not internal or not source:
        raise argparse.ArgumentTypeError("parameter alias must use INTERNAL=SOURCE")
    return internal, source


def compile_input(
    input_path: Path,
    output: Path,
    *,
    partitioned: bool = False,
    lowering_library: str | None = None,
    constant_parameters: list[str] | None = None,
    state_parameters: list[str] | None = None,
    parameter_aliases: list[tuple[str, str]] | None = None,
    state_updates: list[str] | None = None,
    state_appends: list[str] | None = None,
    dump_module: Path | None = None,
) -> tuple[Path, dict]:
    """Compile TVM IR or normalized backend IR without a subprocess boundary."""
    stage = "load-input"
    try:
        source = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            raise CodegenError("input must contain a JSON object")
        if not is_module(source):
            stage = "import-tvm-relax"
            try:
                import tvm
            except ImportError as error:
                raise CodegenError(
                    "input is not a normalized backend module and Apache TVM is unavailable"
                ) from error
            from opennpux_tvm_byoc.relax_backend import (
                normalized_module_from_relax,
                partition_for_opennpux,
            )

            tvm_module = tvm.ir.load_json(input_path.read_text(encoding="utf-8"))
            if not partitioned:
                stage = "partition-tvm-relax"
                tvm_module = partition_for_opennpux(tvm_module)
            stage = "normalize-backend-module"
            source = normalized_module_from_relax(tvm_module)
        from opennpux_tvm_byoc.storage_policy import (
            apply_parameter_aliases,
            apply_parameter_storage,
            apply_state_updates,
        )

        alias_pairs = parameter_aliases or []
        aliases = dict(alias_pairs)
        if len(aliases) != len(alias_pairs):
            raise CodegenError("duplicate internal parameter alias")
        stage = "apply-parameter-aliases"
        apply_parameter_aliases(source, aliases)
        stage = "apply-storage-policy"
        apply_parameter_storage(
            source, constant_parameters or [], state_parameters or []
        )
        stage = "apply-state-updates"
        apply_state_updates(source, state_updates or [], state_appends or [])
        if dump_module is not None:
            dump_module.parent.mkdir(parents=True, exist_ok=True)
            dump_module.write_text(
                json.dumps(source, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        stage = "lower-backend-module"
        artifacts, manifest = compile_module(source, lowering_library)
        stage = "write-artifacts"
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
        return manifest_path, manifest
    except (OSError, ValueError, json.JSONDecodeError, CodegenError) as error:
        raise CodegenError(f"stage={stage}: {error}") from error


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
    parser.add_argument("--constant-parameter", action="append", default=[])
    parser.add_argument("--state-parameter", action="append", default=[])
    parser.add_argument(
        "--parameter-alias", action="append", default=[], type=parse_parameter_alias
    )
    parser.add_argument(
        "--state-update", action="append", default=[], metavar="OUTPUT=STATE"
    )
    parser.add_argument(
        "--state-append", action="append", default=[], metavar="OUTPUT=STATE"
    )
    args = parser.parse_args()
    try:
        manifest_path, manifest = compile_input(
            args.input,
            args.output,
            partitioned=args.partitioned,
            lowering_library=args.lowering_library,
            constant_parameters=args.constant_parameter,
            state_parameters=args.state_parameter,
            parameter_aliases=args.parameter_alias,
            state_updates=args.state_update,
            state_appends=args.state_append,
            dump_module=args.dump_byoc_module,
        )
    except (OSError, ValueError, json.JSONDecodeError, CodegenError) as error:
        print(f"xgraph_module_codegen=FAIL {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"xgraph_module_manifest={manifest_path}")
    print(f"xgraph_module_regions={manifest['region_count']}")
    print(f"xgraph_module_commands={manifest['total_commands']}")
    print("xgraph_module_codegen=PASS")


if __name__ == "__main__":
    main()
