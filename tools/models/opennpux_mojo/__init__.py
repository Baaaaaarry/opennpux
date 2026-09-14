"""Thin MAX/Mojo frontend adapter for the OpenNPUX compiler backend."""

from .graph_adapter import MAX_GRAPH_EXPORT_FORMAT, MaxGraphAdapter

__all__ = ["MAX_GRAPH_EXPORT_FORMAT", "MaxGraphAdapter"]
