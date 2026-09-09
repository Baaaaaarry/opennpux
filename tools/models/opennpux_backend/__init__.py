"""Stable frontend-neutral API for the OpenNPUX compiler backend."""

from .capabilities import (
    CAPABILITY_FORMAT,
    capability_manifest,
    normalize_operation,
    supports_host_operation,
    supports_operation,
)
from .canonicalize import canonicalize_graph, canonicalize_ir, canonicalize_module
from .frontend import FrontendAdapter, compile_frontend
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
    "GRAPH_FORMAT",
    "GRAPH_FORMATS",
    "LEGACY_GRAPH_FORMAT",
    "LEGACY_MODULE_FORMAT",
    "MODULE_FORMAT",
    "MODULE_FORMATS",
    "FrontendAdapter",
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
]
