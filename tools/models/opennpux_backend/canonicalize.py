"""Canonicalize frontend spellings at the OpenNPUX backend boundary."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .capabilities import normalize_operation
from .ir import GRAPH_FORMAT, MODULE_FORMAT, is_graph, is_module


def _canonical_operation(record: dict[str, Any]) -> None:
    operation = normalize_operation(record.get("op"))
    if operation is not None:
        record["op"] = operation


def canonicalize_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Return a detached graph using backend identities and operation names."""
    if not is_graph(graph):
        raise ValueError("invalid OpenNPUX backend graph format")
    result = deepcopy(graph)
    result["format"] = GRAPH_FORMAT
    nodes = result.get("nodes")
    if isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict):
                _canonical_operation(node)
    return result


def canonicalize_module(module: dict[str, Any]) -> dict[str, Any]:
    """Return a detached module whose regions and Host pipeline are canonical."""
    if not is_module(module):
        raise ValueError("invalid OpenNPUX backend module format")
    result = deepcopy(module)
    result["format"] = MODULE_FORMAT
    for region in result.get("regions", []):
        if isinstance(region, dict) and isinstance(region.get("graph"), dict):
            region["graph"] = canonicalize_graph(region["graph"])
    for binding in result.get("host_bindings", []):
        if not isinstance(binding, dict):
            continue
        for operation in binding.get("pipeline", []):
            if isinstance(operation, dict):
                _canonical_operation(operation)
    return result


def canonicalize_ir(value: dict[str, Any]) -> dict[str, Any]:
    if is_graph(value):
        return canonicalize_graph(value)
    if is_module(value):
        return canonicalize_module(value)
    raise ValueError("frontend adapter returned unsupported backend IR")


__all__ = ["canonicalize_graph", "canonicalize_ir", "canonicalize_module"]
