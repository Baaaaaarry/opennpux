"""Optional Apache TVM Relax BYOC partitioning and XGraph extraction.

This module deliberately imports TVM lazily. The XGraph encoder and its ABI
tests therefore remain usable on build hosts that do not install TVM.
"""

from __future__ import annotations

from typing import Any

from .module_codegen import MODULE_FORMAT
from .xgraph_codegen import CodegenError, FORMAT


PATTERN_OPS = {
    "opennpux.matmul": "relax.matmul",
    "opennpux.add": "relax.add",
    "opennpux.multiply": "relax.multiply",
    "opennpux.rms_norm": "relax.nn.rms_norm",
    "opennpux.softmax": "relax.nn.softmax",
    "opennpux.silu": "relax.nn.silu",
    "opennpux.take": "relax.take",
}

PATTERN_ARITY = {
    "relax.matmul": 2,
    "relax.add": 2,
    "relax.multiply": 2,
    "relax.nn.rms_norm": 2,
    "relax.nn.softmax": 1,
    "relax.nn.silu": 1,
    "relax.take": 2,
}


def _tvm_modules():
    try:
        import tvm
        from tvm import relax
        from tvm.relax.dpl import is_op, wildcard
    except ImportError as error:
        raise CodegenError(
            "Apache TVM with Relax is required for a TVM IRModule input"
        ) from error
    return tvm, relax, is_op, wildcard


def opennpux_patterns() -> list[tuple[str, Any]]:
    """Return the stage-one OpenNPUX Relax BYOC pattern table."""
    _, _, is_op, wildcard = _tvm_modules()
    patterns = []
    for composite, op_name in PATTERN_OPS.items():
        arity = PATTERN_ARITY[op_name]
        pattern = is_op(op_name)(*(wildcard() for _ in range(arity)))
        patterns.append((composite, pattern))
    return patterns


def partition_for_opennpux(module):
    """Fuse supported Relax ops and annotate merged regions for OpenNPUX."""
    _, relax, _, _ = _tvm_modules()
    module = relax.transform.FuseOpsByPattern(
        opennpux_patterns(), bind_constants=False, annotate_codegen=False
    )(module)
    return relax.transform.MergeCompositeFunctions()(module)


def _attr(attrs: Any, name: str, default: Any) -> Any:
    if attrs is None or not hasattr(attrs, name):
        return default
    value = getattr(attrs, name)
    if hasattr(value, "value"):
        return value.value
    return value


def _static_tensor(expr: Any) -> tuple[list[int], str]:
    info = getattr(expr, "struct_info", None)
    shape = getattr(getattr(info, "shape", None), "values", None)
    dtype = str(getattr(info, "dtype", ""))
    if shape is None or not dtype:
        raise CodegenError("every OpenNPUX BYOC value must have TensorStructInfo")
    dimensions = []
    for dim in shape:
        value = getattr(dim, "value", dim)
        if not isinstance(value, int) or value <= 0:
            raise CodegenError("dynamic Relax dimensions are not supported in stage one")
        dimensions.append(value)
    return dimensions, dtype


def _name(expr: Any) -> str:
    name = getattr(expr, "name_hint", None)
    if not name:
        raise CodegenError(f"unsupported unnamed Relax value {type(expr).__name__}")
    return str(name)


def _codegen_name(function: Any) -> str | None:
    attrs = getattr(function, "attrs", None)
    value = _attr(attrs, "Codegen", None)
    return None if value is None else str(value)


def _composite_name(function: Any) -> str | None:
    value = _attr(getattr(function, "attrs", None), "Composite", None)
    return None if value is None else str(value)


def _find_primitive_call(expression: Any) -> Any | None:
    operator = getattr(expression, "op", None)
    if getattr(operator, "name", None):
        return expression
    for block in getattr(expression, "blocks", ()):
        for binding in getattr(block, "bindings", ()):
            found = _find_primitive_call(getattr(binding, "value", None))
            if found is not None:
                return found
    body = getattr(expression, "body", None)
    if body is not None and body is not expression:
        return _find_primitive_call(body)
    return None


def _call_operator(call: Any, local_functions: dict[Any, Any]) -> tuple[str, Any]:
    operator = call.op
    if operator in local_functions:
        operator = local_functions[operator]
    composite = _composite_name(operator)
    if composite is not None:
        if composite not in PATTERN_OPS:
            raise CodegenError(f"unsupported OpenNPUX composite {composite}")
        primitive = _find_primitive_call(operator.body)
        if primitive is None:
            raise CodegenError(f"composite {composite} has no primitive call")
        return PATTERN_OPS[composite], getattr(primitive, "attrs", None)
    op_name = getattr(operator, "name", None)
    if not op_name:
        raise CodegenError("OpenNPUX region contains a non-composite function call")
    return str(op_name), getattr(call, "attrs", None)


def _call_attrs(op_name: str, attrs: Any) -> dict[str, Any]:
    if op_name == "relax.nn.rms_norm":
        return {"epsilon": float(_attr(attrs, "epsilon", 1.0e-5))}
    if op_name == "relax.nn.softmax":
        return {"axis": int(_attr(attrs, "axis", -1))}
    if op_name == "relax.take":
        return {"axis": int(_attr(attrs, "axis", 0))}
    if op_name == "relax.topk":
        return {
            "axis": int(_attr(attrs, "axis", -1)),
            "k": int(_attr(attrs, "k", 1)),
        }
    return {}


def _normalized_graph_from_function(function: Any, relax: Any) -> dict[str, Any]:
    tensors: dict[str, dict[str, Any]] = {}
    value_names: dict[Any, str] = {}
    used_names: set[str] = set()

    def assign_name(value: Any) -> str:
        if value in value_names:
            return value_names[value]
        base = _name(value)
        candidate = base
        suffix = 1
        while candidate in used_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used_names.add(candidate)
        value_names[value] = candidate
        return candidate

    for parameter in function.params:
        shape, dtype = _static_tensor(parameter)
        parameter_name = assign_name(parameter)
        tensors[parameter_name] = {
            "name": parameter_name,
            "shape": shape,
            "dtype": dtype,
            "storage": "input",
        }

    body = function.body
    bindings = []
    if isinstance(body, relax.SeqExpr):
        for block in body.blocks:
            bindings.extend(block.bindings)
        result = body.body
    else:
        result = body
    if not bindings and isinstance(result, relax.Call):
        raise CodegenError("direct-call Relax regions must be normalized to dataflow form")

    local_functions = {
        binding.var: binding.value
        for binding in bindings
        if isinstance(binding, relax.VarBinding)
        and isinstance(binding.value, relax.Function)
    }
    nodes = []
    for binding in bindings:
        if isinstance(binding, relax.VarBinding) and isinstance(binding.value, relax.Function):
            continue
        if not isinstance(binding, relax.VarBinding) or not isinstance(binding.value, relax.Call):
            raise CodegenError("OpenNPUX regions currently require call-only VarBindings")
        output_name = assign_name(binding.var)
        shape, dtype = _static_tensor(binding.var)
        tensors[output_name] = {
            "name": output_name,
            "shape": shape,
            "dtype": dtype,
            "storage": "scratch",
        }
        op_name, attrs = _call_operator(binding.value, local_functions)
        nodes.append(
            {
                "op": op_name,
                "inputs": [assign_name(argument) for argument in binding.value.args],
                "outputs": [output_name],
                "attrs": _call_attrs(op_name, attrs),
            }
        )
    result_name = assign_name(result)
    if result_name not in tensors:
        raise CodegenError("OpenNPUX region result is not produced by a supported binding")
    tensors[result_name]["storage"] = "output"
    return {
        "format": FORMAT,
        "tensors": list(tensors.values()),
        "nodes": nodes,
        "outputs": [result_name],
    }


def normalized_graph_from_relax(module) -> dict[str, Any]:
    """Extract one fully offloaded Relax region into the stable graph boundary."""
    _, relax, _, _ = _tvm_modules()
    regions = [
        function
        for function in module.functions.values()
        if isinstance(function, relax.Function) and _codegen_name(function) == "opennpux"
    ]
    if len(regions) != 1:
        raise CodegenError(
            f"single-region codegen requires exactly one OpenNPUX region, found {len(regions)}"
        )
    return _normalized_graph_from_function(regions[0], relax)


def normalized_module_from_relax(module) -> dict[str, Any]:
    """Extract all OpenNPUX regions and direct device-to-device Tensor edges."""
    _, relax, _, _ = _tvm_modules()
    regions: dict[Any, tuple[str, Any]] = {}
    used_names: set[str] = set()
    for global_var, function in module.functions.items():
        if not isinstance(function, relax.Function) or _codegen_name(function) != "opennpux":
            continue
        base = str(_attr(function.attrs, "global_symbol", global_var.name_hint))
        name = base
        suffix = 1
        while name in used_names:
            name = f"{base}_{suffix}"
            suffix += 1
        used_names.add(name)
        regions[global_var] = (name, function)
    if not regions:
        raise CodegenError("partitioned module contains no OpenNPUX regions")

    try:
        main = module["main"]
    except (KeyError, TypeError) as error:
        raise CodegenError("partitioned module has no Relax main function") from error
    if not isinstance(main, relax.Function):
        raise CodegenError("partitioned module main is not a Relax function")

    invocation_order: list[Any] = []
    producer: dict[Any, tuple[str, str]] = {}
    edges = []
    body = main.body
    bindings = []
    if isinstance(body, relax.SeqExpr):
        for block in body.blocks:
            bindings.extend(block.bindings)
    for binding in bindings:
        if not isinstance(binding, relax.VarBinding) or not isinstance(binding.value, relax.Call):
            continue
        call = binding.value
        if call.op not in regions:
            continue
        region_name, function = regions[call.op]
        if call.op in invocation_order:
            raise CodegenError(f"OpenNPUX region {region_name} is invoked more than once")
        invocation_order.append(call.op)
        graph = _normalized_graph_from_function(function, relax)
        if len(call.args) != len(function.params):
            raise CodegenError(f"OpenNPUX region {region_name} argument count mismatch")
        parameter_names = [
            tensor["name"] for tensor in graph["tensors"][: len(function.params)]
        ]
        for argument, parameter_name in zip(call.args, parameter_names):
            if argument not in producer:
                continue
            source_region, source_tensor = producer[argument]
            edges.append({
                "from": {"region": source_region, "tensor": source_tensor},
                "to": {"region": region_name, "tensor": parameter_name},
            })
        producer[binding.var] = (region_name, graph["outputs"][0])

    if len(invocation_order) != len(regions):
        missing = [name for key, (name, _) in regions.items() if key not in invocation_order]
        raise CodegenError(
            "OpenNPUX regions are not each invoked once from main: " + ", ".join(missing)
        )
    return {
        "format": MODULE_FORMAT,
        "regions": [
            {
                "name": regions[global_var][0],
                "graph": _normalized_graph_from_function(regions[global_var][1], relax),
            }
            for global_var in invocation_order
        ],
        "edges": edges,
    }
