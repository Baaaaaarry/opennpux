#!/usr/bin/env python3
"""Compile a TVM Relax module and deployment values into NPU artifacts."""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

from build_xgraph_tensor_image import build_image
from opennpux_tvm_byoc import CodegenError


FORMAT = "OPENNPUX_TVM_BYOC_DEPLOYMENT_V1"


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CodegenError(f"{path}: root must be an object")
    return value


def checked_names(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise CodegenError(f"{label} must be an array of strings")
    if len(set(value)) != len(value):
        raise CodegenError(f"{label} contains duplicates")
    return value


def tensor_value(values: dict[str, Any], region: str, tensor: str) -> Any | None:
    qualified = f"{region}.{tensor}"
    if qualified in values:
        return values[qualified]
    return values.get(tensor)


def zero_value(dtype: str) -> dict[str, int | float]:
    if dtype == "float32":
        return {"fill": 0.0}
    if dtype == "int32":
        return {"fill": 0}
    raise CodegenError(f"cannot initialize unsupported dtype {dtype}")


def build_region_arena(
    metadata: dict[str, Any], region: dict[str, Any], values: dict[str, Any],
    *, invocation: bool,
) -> bytes:
    selected: dict[str, Any] = {}
    invocation_bindings = set(region.get("invocation_bindings", []))
    binding_sources = region.get("binding_sources", {})
    if not isinstance(binding_sources, dict):
        raise CodegenError(f"region {region['name']} has invalid binding_sources")
    for tensor in metadata["tensors"]:
        if tensor["storage"] not in {"input", "constant", "state"}:
            continue
        name = tensor["name"]
        source_name = binding_sources.get(name, name)
        supplied = tensor_value(values, region["name"], source_name)
        required = (name in invocation_bindings if invocation else
                    tensor["storage"] in {"constant", "state"})
        if supplied is None and required:
            kind = "invocation" if invocation else "module"
            raise CodegenError(
                f"{kind} values missing {region['name']}.{name} "
                f"(source={source_name}, available={sorted(values)})"
            )
        selected[name] = supplied if supplied is not None else zero_value(tensor["dtype"])
    return build_image(metadata, selected)


def run(command: list[str]) -> None:
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        detail = result.stderr.strip() or "command failed"
        raise CodegenError(f"{Path(command[1]).name}: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="compile a TVM Relax model into reusable OpenNPUX deployment artifacts"
    )
    parser.add_argument("input", type=Path, help="TVM IRModule JSON or normalized module")
    parser.add_argument("deployment", type=Path, help="deployment values and storage policy")
    parser.add_argument("output", type=Path)
    parser.add_argument("--partitioned", action="store_true")
    parser.add_argument(
        "--lowering-library",
        default=os.environ.get("OPENNPUX_XGRAPH_LOWERING_LIB"),
    )
    args = parser.parse_args()
    try:
        spec = load_object(args.deployment)
        if spec.get("format") != FORMAT:
            raise CodegenError(f"deployment format must be {FORMAT}")
        constants = checked_names(spec.get("constant_parameters", []), "constant_parameters")
        states = checked_names(spec.get("state_parameters", []), "state_parameters")
        state_updates = checked_names(spec.get("state_updates", []), "state_updates")
        state_appends = checked_names(spec.get("state_appends", []), "state_appends")
        module_values = spec.get("module_values", {})
        invocations = spec.get("invocations")
        if not isinstance(module_values, dict):
            raise CodegenError("module_values must be an object")
        if not isinstance(invocations, list) or not invocations:
            raise CodegenError("invocations must be a non-empty array")

        args.output.mkdir(parents=True, exist_ok=True)
        compiled = args.output / "compiled"
        compile_command = [
            sys.executable,
            str(Path(__file__).with_name("compile_tvm_byoc_module.py")),
            str(args.input),
            str(compiled),
        ]
        if args.partitioned:
            compile_command.append("--partitioned")
        if args.lowering_library:
            compile_command.extend(["--lowering-library", args.lowering_library])
        for name in constants:
            compile_command.extend(["--constant-parameter", name])
        for name in states:
            compile_command.extend(["--state-parameter", name])
        for value in state_updates:
            compile_command.extend(["--state-update", value])
        for value in state_appends:
            compile_command.extend(["--state-append", value])
        run(compile_command)

        manifest = load_object(compiled / "module.npxgm.json")
        metadata = {
            region["name"]: load_object(compiled / f"{region['artifact']}.json")
            for region in manifest["regions"]
        }
        base_arenas: list[tuple[str, Path]] = []
        arena_dir = args.output / "arenas"
        arena_dir.mkdir(exist_ok=True)
        for region in manifest["regions"]:
            path = arena_dir / f"{region['name']}.module.bin"
            path.write_bytes(build_region_arena(
                metadata[region["name"]], region, module_values, invocation=False
            ))
            base_arenas.append((region["name"], path))

        package = args.output / "model.npxgm"
        package_command = [
            sys.executable,
            str(Path(__file__).with_name("build_tvm_byoc_module_package.py")),
            str(compiled),
            str(package),
            "--clear-external-bindings",
        ]
        for region, path in base_arenas:
            package_command.extend(["--arena", f"{region}={path}"])
        run(package_command)

        invocation_records = []
        seen_names: set[str] = set()
        for index, invocation in enumerate(invocations):
            if not isinstance(invocation, dict):
                raise CodegenError(f"invocation {index} must be an object")
            name = invocation.get("name", f"invocation-{index:03d}")
            values = invocation.get("values", {})
            scalars = invocation.get("scalars", {})
            if (not isinstance(name, str) or not name or name in seen_names or
                    not isinstance(values, dict) or not isinstance(scalars, dict)):
                raise CodegenError(f"invocation {index} is invalid")
            seen_names.add(name)
            invocation_command = [
                sys.executable,
                str(Path(__file__).with_name("build_tvm_byoc_invocation.py")),
                str(compiled),
                str(args.output / f"{name}.npxmi"),
            ]
            for region in manifest["regions"]:
                path = arena_dir / f"{name}.{region['name']}.bin"
                path.write_bytes(build_region_arena(
                    metadata[region["name"]], region, values, invocation=True
                ))
                invocation_command.extend(["--arena", f"{region['name']}={path}"])
            for scalar_name, scalar_value in scalars.items():
                if not isinstance(scalar_name, str) or not isinstance(scalar_value, int):
                    raise CodegenError(f"invocation {name} has an invalid scalar")
                invocation_command.extend([
                    "--scalar", f"{scalar_name}={scalar_value}"
                ])
            run(invocation_command)
            invocation_records.append({
                "name": name,
                "artifact": f"{name}.npxmi",
            })

        deployment_manifest = {
            "format": FORMAT,
            "module": package.name,
            "region_count": manifest["region_count"],
            "command_count": manifest["total_commands"],
            "invocations": invocation_records,
        }
        manifest_path = args.output / "deployment.json"
        manifest_path.write_text(
            json.dumps(deployment_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, KeyError, ValueError, struct.error, CodegenError) as error:
        print(f"tvm_byoc_deployment=FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"tvm_byoc_deployment_manifest={manifest_path}")
    print(f"tvm_byoc_deployment_module={package}")
    print(f"tvm_byoc_deployment_invocations={len(invocation_records)}")
    print("tvm_byoc_deployment=PASS")


if __name__ == "__main__":
    main()
