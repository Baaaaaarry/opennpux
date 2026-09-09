#!/usr/bin/env python3
"""Import a static ONNX model into a TVM Relax IRModule JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# These ONNX inputs describe graph structure rather than device-resident Tensor
# data. Bind only these initializers before BYOC partitioning; weights must stay
# as Relax parameters so the deployment compiler can place them in NPU storage.
COMPILE_TIME_INITIALIZER_INPUTS = {
    "Reshape": frozenset({1}),
}


def compile_time_initializer_names(model) -> list[str]:
    initializer_names = {value.name for value in model.graph.initializer}
    selected = set()
    for node in model.graph.node:
        for index in COMPILE_TIME_INITIALIZER_INPUTS.get(node.op_type, ()):
            if index < len(node.input) and node.input[index] in initializer_names:
                selected.add(node.input[index])
    return sorted(selected)


def parse_shape(value: str) -> tuple[str, list[int]]:
    name, separator, dimensions = value.partition("=")
    if not separator or not name or not dimensions:
        raise argparse.ArgumentTypeError("shape must use NAME=DIM,DIM,...")
    try:
        shape = [int(dimension) for dimension in dimensions.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid shape {value}") from error
    if not shape or any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError(f"shape dimensions must be positive: {value}")
    return name, shape


def parse_dtype(value: str) -> tuple[str, str]:
    name, separator, dtype = value.partition("=")
    if not separator or not name or not dtype:
        raise argparse.ArgumentTypeError("dtype must use NAME=DTYPE")
    return name, dtype


def unique_map(values, label: str):
    result = {}
    for name, value in values:
        if name in result:
            raise ValueError(f"duplicate {label} override for {name}")
        result[name] = value
    return result


def source_parameter_names(model, compile_time_names: list[str]) -> list[str]:
    compile_time = set(compile_time_names)
    names = []
    for value in [*model.graph.input, *model.graph.initializer]:
        if value.name not in compile_time and value.name not in names:
            names.append(value.name)
    return names


def tensor_signature(parameter, source_name: str) -> dict:
    info = parameter.struct_info
    dimensions = []
    for dimension in info.shape.values:
        value = getattr(dimension, "value", dimension)
        if not isinstance(value, int):
            raise ValueError(f"parameter {parameter.name_hint} has a dynamic dimension")
        dimensions.append(value)
    return {
        "name": str(parameter.name_hint),
        "source_name": source_name,
        "shape": dimensions,
        "dtype": str(info.dtype),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="import an ONNX model through TVM's public Relax frontend"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--signature", type=Path)
    parser.add_argument("--shape", action="append", default=[], type=parse_shape)
    parser.add_argument("--dtype", action="append", default=[], type=parse_dtype)
    parser.add_argument("--opset", type=int)
    args = parser.parse_args()
    try:
        import onnx
        import tvm
        from onnx import numpy_helper
        from tvm import relax
        from tvm.relax.frontend.onnx import from_onnx

        shape_dict = unique_map(args.shape, "shape")
        dtype_dict = unique_map(args.dtype, "dtype")
        model = onnx.load(str(args.input))
        onnx.checker.check_model(model)
        kwargs = {
            "shape_dict": shape_dict or None,
            "dtype_dict": dtype_dict or "float32",
            "keep_params_in_input": True,
        }
        if args.opset is not None:
            kwargs["opset"] = args.opset
        module = from_onnx(model, **kwargs)
        compile_time_names = compile_time_initializer_names(model)
        initializer_values = {
            value.name: numpy_helper.to_array(value)
            for value in model.graph.initializer
        }
        if compile_time_names:
            module = relax.transform.BindParams(
                "main",
                {name: initializer_values[name] for name in compile_time_names},
            )(module)
            module = relax.transform.FoldConstant()(module)
        module = relax.transform.CanonicalizeBindings()(module)
        main_function = module["main"]
        source_names = source_parameter_names(model, compile_time_names)
        if len(source_names) != len(main_function.params):
            raise ValueError(
                "ONNX source parameters do not match Relax main parameters: "
                f"source={len(source_names)} relax={len(main_function.params)}"
            )
        signature = {
            "format": "OPENNPUX_TVM_ONNX_SIGNATURE_V1",
            "source": str(args.input),
            "parameters": [
                tensor_signature(value, source_name)
                for value, source_name in zip(main_function.params, source_names)
            ],
            "onnx_initializers": [value.name for value in model.graph.initializer],
            "compile_time_initializers": compile_time_names,
            "outputs": [value.name for value in model.graph.output],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(tvm.ir.save_json(module), encoding="utf-8")
        signature_path = args.signature or Path(f"{args.output}.json")
        signature_path.write_text(
            json.dumps(signature, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (ImportError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"onnx_relax_import=FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"onnx_model={args.input}")
    print(f"onnx_relax_module={args.output}")
    print(f"onnx_relax_parameters={len(signature['parameters'])}")
    print(f"onnx_relax_compile_time_initializers={len(compile_time_names)}")
    print("onnx_relax_import=PASS")


if __name__ == "__main__":
    main()
