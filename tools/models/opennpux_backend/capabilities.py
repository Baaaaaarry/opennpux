"""Frontend-neutral OpenNPUX operation capability declarations."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


CAPABILITY_FORMAT = "OPENNPUX_BACKEND_CAPABILITIES_V1"

OPERATION_ALIASES = {
    "add": "add",
    "relax.add": "add",
    "multiply": "multiply",
    "relax.multiply": "multiply",
    "matmul": "matmul",
    "relax.matmul": "matmul",
    "rms_norm": "rms_norm",
    "relax.nn.rms_norm": "rms_norm",
    "softmax": "softmax",
    "relax.nn.softmax": "softmax",
    "silu": "silu",
    "relax.nn.silu": "silu",
    "take": "take",
    "relax.take": "take",
    "reshape": "copy",
    "relax.reshape": "copy",
    "transpose": "transpose",
    "relax.permute_dims": "transpose",
    "topk": "topk",
    "relax.topk": "topk",
    "rope": "rope",
    "opennpux.rope": "rope",
    "copy": "copy",
    "opennpux.copy": "copy",
    "kv_pack": "kv_pack",
    "opennpux.kv_pack": "kv_pack",
    "attention": "attention",
    "opennpux.attention": "attention",
    "relu": "relu",
    "relax.nn.relu": "relu",
}

OPERATION_CAPABILITIES = {
    "add": {
        "commands": ["TADD"], "dtypes": ["float32"], "arity": [2, 1],
        "shape_relation": "same-or-rhs-scalar",
    },
    "multiply": {
        "commands": ["TMUL"], "dtypes": ["float32"], "arity": [2, 1],
        "shape_relation": "same-or-rhs-scalar",
    },
    "matmul": {
        "commands": ["TMMA"], "dtypes": ["float32"], "arity": [2, 1],
        "input_ranks": [">=2", "2"], "shape_relation": "matmul",
        "tiled": True, "instruction_extent_limit": 1023,
    },
    "rms_norm": {
        "commands": ["TRMSNORM"], "dtypes": ["float32"], "arity": [2, 1],
        "shape_relation": "input-output-same;weight-last-dimension",
        "attributes": {"epsilon": "positive-finite-float"},
    },
    "softmax": {"commands": ["TSOFTMAX"], "dtypes": ["float32"], "arity": [1, 1], "shape_relation": "same", "attributes": {"axis": "innermost"}},
    "silu": {"commands": ["TSILU"], "dtypes": ["float32"], "arity": [1, 1], "shape_relation": "same"},
    "take": {"commands": ["TGATHER"], "dtypes": ["float32", "int32"], "arity": [2, 1], "input_ranks": ["2", ">=0"], "shape_relation": "embedding-axis-0", "attributes": {"axis": 0}},
    "topk": {"commands": ["TTOPK"], "dtypes": ["float32", "int32"], "arity": [1, 2], "shape_relation": "replace-last-dimension-with-k", "attributes": {"k": "positive-static-int"}},
    "rope": {"commands": ["TROPE"], "dtypes": ["float32"], "arity": [2, 1], "shape_relation": "input-output-same", "attributes": {"layout": ["adjacent", "half_split"]}},
    "copy": {"commands": ["TDMA"], "dtypes": ["float32", "int32"], "arity": [1, 1], "shape_relation": "same-byte-size-and-dtype"},
    "kv_pack": {"commands": ["TDMA", "TDMA"], "dtypes": ["float32"], "arity": [2, 1], "shape_relation": "single-token-kv-pack"},
    "attention": {"commands": ["TATTENTION"], "dtypes": ["float32"], "arity": [2, 1], "input_ranks": ["3", "4"], "shape_relation": "gqa-attention-state", "attributes": {"kv_length": "positive-static-int"}},
    "transpose": {
        "commands": [],
        "dtypes": ["float32"],
        "compile_time_only": True,
        "arity": [1, 1],
        "input_ranks": ["2"],
        "shape_relation": "matmul-rhs-transpose-only",
    },
}


class ContractViolation(ValueError):
    """A concrete operation invocation is outside backend capabilities."""


def _field(tensor: object, name: str) -> Any:
    if isinstance(tensor, Mapping):
        return tensor.get(name)
    return getattr(tensor, name, None)


def _shape(tensor: object, label: str) -> tuple[int, ...]:
    shape = _field(tensor, "shape")
    if not isinstance(shape, (list, tuple)) or any(
        not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape
    ):
        raise ContractViolation(f"{label} requires a positive static shape")
    return tuple(shape)


def validate_operation_contract(
    name: object,
    inputs: Sequence[object],
    outputs: Sequence[object],
    attrs: Mapping[str, Any] | None = None,
) -> str:
    """Validate the frontend-neutral portion of one backend invocation."""
    operation = normalize_operation(name)
    capability = OPERATION_CAPABILITIES.get(operation or "")
    if capability is None:
        raise ContractViolation(f"unsupported backend operation {name!r}")
    expected_inputs, expected_outputs = capability["arity"]
    if len(inputs) != expected_inputs or len(outputs) != expected_outputs:
        raise ContractViolation(
            f"{operation} expects {expected_inputs} inputs and {expected_outputs} outputs"
        )
    tensors = tuple(inputs) + tuple(outputs)
    shapes = [
        _shape(tensor, f"{operation} tensor {index}")
        for index, tensor in enumerate(tensors)
    ]
    dtypes = [_field(tensor, "dtype") for tensor in tensors]
    if any(dtype not in capability["dtypes"] for dtype in dtypes):
        raise ContractViolation(f"{operation} has an unsupported tensor dtype")
    attributes = dict(attrs or {})
    if operation == "matmul":
        lhs, rhs, output = shapes
        if len(lhs) < 2 or len(rhs) != 2:
            raise ContractViolation("matmul requires rank >= 2 lhs and rank-2 rhs")
        rhs_k, n = (rhs[1], rhs[0]) if attributes.get("transpose_rhs", False) else rhs
        if lhs[-1] != rhs_k or output != lhs[:-1] + (n,):
            raise ContractViolation("matmul input/output shapes are inconsistent")
    elif operation in {"add", "multiply"}:
        lhs, rhs, output = shapes
        if lhs != output or (rhs != output and _element_count(rhs) != 1):
            raise ContractViolation(
                f"{operation} requires equal shapes or one FP32 RHS scalar"
            )
    elif operation in {"softmax", "silu", "rope"} and shapes[0] != shapes[-1]:
        raise ContractViolation(f"{operation} input/output shapes must match")
    if operation == "softmax" and int(attributes.get("axis", -1)) not in {
        -1, len(shapes[0]) - 1,
    }:
        raise ContractViolation("softmax requires the innermost axis")
    if operation == "rope" and attributes.get("layout", "adjacent") not in {
        "adjacent", "half_split",
    }:
        raise ContractViolation("rope layout must be adjacent or half_split")
    return operation


def _element_count(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def supports_operation_contract(
    name: object,
    inputs: Sequence[object],
    outputs: Sequence[object],
    attrs: Mapping[str, Any] | None = None,
) -> bool:
    try:
        validate_operation_contract(name, inputs, outputs, attrs)
    except (ContractViolation, TypeError, ValueError):
        return False
    return True

HOST_OPERATION_CAPABILITIES = {
    "relu": {"dtypes": ["float32"]},
}


def normalize_operation(name: object) -> str | None:
    """Return the backend operation name for a known frontend spelling."""
    return OPERATION_ALIASES.get(name) if isinstance(name, str) else None


def supports_operation(name: object) -> bool:
    return normalize_operation(name) in OPERATION_CAPABILITIES


def supports_host_operation(name: object) -> bool:
    return normalize_operation(name) in HOST_OPERATION_CAPABILITIES


def capability_manifest() -> dict[str, Any]:
    """Return a deterministic machine-readable partitioning contract."""
    return {
        "format": CAPABILITY_FORMAT,
        "operations": {
            name: dict(OPERATION_CAPABILITIES[name])
            for name in sorted(OPERATION_CAPABILITIES)
        },
    }
