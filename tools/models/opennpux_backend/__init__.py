"""Stable frontend-neutral API for the OpenNPUX compiler backend."""

from .capabilities import (
    CAPABILITY_FORMAT,
    ContractViolation,
    capability_manifest,
    normalize_operation,
    supports_host_operation,
    supports_operation,
    supports_operation_contract,
    validate_operation_contract,
)
from .canonicalize import canonicalize_graph, canonicalize_ir, canonicalize_module
from .artifacts import write_graph_artifact, write_module_artifacts
from .frontend import FrontendAdapter, adapt_frontend, compile_frontend
from .ir import (
    GRAPH_FORMAT,
    GRAPH_FORMATS,
    LEGACY_GRAPH_FORMAT,
    LEGACY_MODULE_FORMAT,
    MODULE_FORMAT,
    MODULE_FORMATS,
    is_graph,
    is_module,
)

__all__ = [
    "CAPABILITY_FORMAT",
    "ContractViolation",
    "GRAPH_FORMAT",
    "GRAPH_FORMATS",
    "LEGACY_GRAPH_FORMAT",
    "LEGACY_MODULE_FORMAT",
    "MODULE_FORMAT",
    "MODULE_FORMATS",
    "FrontendAdapter",
    "adapt_frontend",
    "canonicalize_graph",
    "canonicalize_ir",
    "canonicalize_module",
    "capability_manifest",
    "compile_frontend",
    "is_graph",
    "is_module",
    "normalize_operation",
    "supports_host_operation",
    "supports_operation",
    "supports_operation_contract",
    "validate_operation_contract",
    "write_graph_artifact",
    "write_module_artifacts",
]
