#!/usr/bin/env python3
"""Compile ONNX initializers and requests into an OpenNPUX deployment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


FORMAT = "OPENNPUX_TVM_ONNX_REQUESTS_V1"


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def run(command: list[str]) -> None:
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or f"{Path(command[1]).name} failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="compile an ONNX model and runtime requests for OpenNPUX"
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("requests", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lowering-library")
    args = parser.parse_args()
    try:
        import numpy as np
        import onnx
        from onnx import numpy_helper

        request_spec = load_object(args.requests)
        if request_spec.get("format") != FORMAT:
            raise ValueError(f"request format must be {FORMAT}")
        invocations = request_spec.get("invocations")
        if not isinstance(invocations, list) or not invocations:
            raise ValueError("invocations must be a non-empty array")

        args.output.mkdir(parents=True, exist_ok=True)
        relax_path = args.output / "model.relax.json"
        signature_path = args.output / "model.signature.json"
        run([
            sys.executable,
            str(Path(__file__).with_name("import_onnx_to_relax.py")),
            str(args.model), str(relax_path), "--signature", str(signature_path),
        ])
        signature = load_object(signature_path)
        parameter_names = {value["name"] for value in signature["parameters"]}

        model = onnx.load(str(args.model), load_external_data=True)
        constant_values = {}
        constant_names = []
        folded_initializers = []
        constants_dir = args.output / "constants"
        constants_dir.mkdir(exist_ok=True)
        for index, initializer in enumerate(model.graph.initializer):
            if initializer.name not in parameter_names:
                folded_initializers.append(initializer.name)
                continue
            array = np.asarray(numpy_helper.to_array(initializer))
            if array.dtype not in (np.dtype("float32"), np.dtype("int32")):
                raise ValueError(
                    f"initializer {initializer.name} has unsupported dtype {array.dtype}"
                )
            path = constants_dir / f"{index:06d}.bin"
            encoded = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<")))
            path.write_bytes(encoded.tobytes())
            constant_names.append(initializer.name)
            constant_values[initializer.name] = {
                "binary": str(path.resolve()), "bytes": encoded.nbytes,
            }

        deployment = {
            "format": "OPENNPUX_TVM_BYOC_DEPLOYMENT_V1",
            "constant_parameters": constant_names,
            "state_parameters": request_spec.get("state_parameters", []),
            "state_updates": request_spec.get("state_updates", []),
            "state_appends": request_spec.get("state_appends", []),
            "module_values": {
                **constant_values, **request_spec.get("module_values", {})
            },
            "invocations": invocations,
        }
        deployment_path = args.output / "generated-deployment.json"
        deployment_path.write_text(
            json.dumps(deployment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        command = [
            sys.executable,
            str(Path(__file__).with_name("compile_tvm_byoc_deployment.py")),
            str(relax_path), str(deployment_path), str(args.output / "deployment"),
        ]
        if args.lowering_library:
            command.extend(["--lowering-library", args.lowering_library])
        run(command)
        module_manifest = load_object(
            args.output / "deployment/compiled/module.npxgm.json"
        )
        host_operations = sum(
            len(binding.get("pipeline", []))
            for binding in module_manifest.get("host_bindings", [])
            if isinstance(binding, dict)
        )
        onnx_operators = {}
        for node in model.graph.node:
            onnx_operators[node.op_type] = onnx_operators.get(node.op_type, 0) + 1
        audit = {
            "format": "OPENNPUX_TVM_ONNX_PARTITION_AUDIT_V1",
            "onnx_nodes": len(model.graph.node),
            "onnx_operators": onnx_operators,
            "npu_regions": module_manifest["region_count"],
            "npu_commands": module_manifest["total_commands"],
            "host_bindings": len(module_manifest.get("host_bindings", [])),
            "host_operations": host_operations,
            "folded_initializers": folded_initializers,
        }
        (args.output / "partition-audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (ImportError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"onnx_byoc_deployment=FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"onnx_byoc_constants={len(constant_names)}")
    print(f"onnx_byoc_folded_initializers={len(folded_initializers)}")
    print(f"onnx_byoc_invocations={len(invocations)}")
    print(f"onnx_byoc_npu_regions={audit['npu_regions']}")
    print(f"onnx_byoc_npu_commands={audit['npu_commands']}")
    print(f"onnx_byoc_host_operations={audit['host_operations']}")
    print(f"onnx_byoc_module={args.output / 'deployment/model.npxgm'}")
    print("onnx_byoc_partition_audit=PASS")
    print("onnx_byoc_deployment=PASS")


if __name__ == "__main__":
    main()
