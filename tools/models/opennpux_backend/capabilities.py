"""Frontend-neutral OpenNPUX operation capability declarations."""

from __future__ import annotations

from typing import Any


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
    "add": {"commands": ["TADD"], "dtypes": ["float32"]},
    "multiply": {"commands": ["TMUL"], "dtypes": ["float32"]},
    "matmul": {"commands": ["TMMA"], "dtypes": ["float32"], "tiled": True},
    "rms_norm": {"commands": ["TRMSNORM"], "dtypes": ["float32"]},
    "softmax": {"commands": ["TSOFTMAX"], "dtypes": ["float32"]},
    "silu": {"commands": ["TSILU"], "dtypes": ["float32"]},
    "take": {"commands": ["TGATHER"], "dtypes": ["float32", "int32"]},
    "topk": {"commands": ["TTOPK"], "dtypes": ["float32", "int32"]},
    "rope": {"commands": ["TROPE"], "dtypes": ["float32"]},
    "copy": {"commands": ["TDMA"], "dtypes": ["float32", "int32"]},
    "kv_pack": {"commands": ["TDMA", "TDMA"], "dtypes": ["float32"]},
    "attention": {"commands": ["TATTENTION"], "dtypes": ["float32"]},
    "transpose": {
        "commands": [],
        "dtypes": ["float32"],
        "compile_time_only": True,
    },
}

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
