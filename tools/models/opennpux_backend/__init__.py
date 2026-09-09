"""Stable frontend-neutral API for the OpenNPUX compiler backend."""

from .capabilities import (
    CAPABILITY_FORMAT,
    capability_manifest,
    normalize_operation,
    supports_operation,
)
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
    "capability_manifest",
    "is_graph",
    "is_module",
    "normalize_operation",
    "supports_operation",
]
