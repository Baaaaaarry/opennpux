"""Stable frontend-neutral entry points for OpenNPUX backend compilation."""

from __future__ import annotations

from typing import Any

from .module_codegen import compile_module as _compile_module
from .xgraph_codegen import (
    CodegenError,
    compile_graph as _compile_graph,
)
from .shapes import ShapeSpecializationError, specialize_ir


def _specialize(
    value: dict[str, Any], shape_bindings: dict[str, int] | None
) -> dict[str, Any]:
    try:
        return specialize_ir(value, shape_bindings)
    except ShapeSpecializationError as error:
        raise CodegenError(f"shape specialization failed: {error}") from error


def compile_graph(
    graph: dict[str, Any],
    lowering_library: str | None = None,
    shape_bindings: dict[str, int] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Lower frontend-neutral backend graph IR to one XGraph artifact."""
    return _compile_graph(_specialize(graph, shape_bindings), lowering_library)


def compile_module(
    module: dict[str, Any],
    lowering_library: str | None = None,
    shape_bindings: dict[str, int] | None = None,
) -> tuple[dict[str, tuple[bytes, dict[str, Any]]], dict[str, Any]]:
    """Lower frontend-neutral backend module IR to reusable XGraph regions."""
    return _compile_module(_specialize(module, shape_bindings), lowering_library)


__all__ = ["CodegenError", "compile_graph", "compile_module"]
