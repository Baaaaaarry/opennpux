"""Stable frontend-neutral API for the OpenNPUX compiler backend."""

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
    "GRAPH_FORMAT",
    "GRAPH_FORMATS",
    "LEGACY_GRAPH_FORMAT",
    "LEGACY_MODULE_FORMAT",
    "MODULE_FORMAT",
    "MODULE_FORMATS",
    "is_graph",
    "is_module",
]
