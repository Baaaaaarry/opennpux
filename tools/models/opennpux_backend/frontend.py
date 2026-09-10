"""Frontend adapter contract shared by TVM, MAX/Mojo, and future importers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .canonicalize import canonicalize_ir
from .compiler import CodegenError, compile_graph, compile_module
from .ir import is_graph, is_module


@runtime_checkable
class FrontendAdapter(Protocol):
    """Translate one frontend-owned object into frontend-neutral backend IR."""

    name: str

    def to_backend_ir(self, source: Any) -> dict[str, Any]:
        """Return an OPENNPUX_BACKEND_GRAPH_V1 or MODULE_V1 object."""


def adapt_frontend(source: Any, adapter: FrontendAdapter) -> dict[str, Any]:
    """Convert one frontend object into detached canonical backend IR."""
    adapter_name = getattr(adapter, "name", None)
    if not isinstance(adapter_name, str) or not adapter_name:
        raise CodegenError("frontend adapter must provide a non-empty name")
    try:
        return canonicalize_ir(adapter.to_backend_ir(source))
    except (AttributeError, TypeError, ValueError) as error:
        raise CodegenError(f"frontend adapter {adapter_name} failed: {error}") from error


def compile_frontend(
    source: Any,
    adapter: FrontendAdapter,
    lowering_library: str | None = None,
) -> dict[str, Any]:
    """Adapt and compile a frontend object through the common backend path."""
    backend_ir = adapt_frontend(source, adapter)
    adapter_name = adapter.name
    if is_graph(backend_ir):
        artifact, metadata = compile_graph(backend_ir, lowering_library)
        metadata = dict(metadata)
        metadata["frontend_adapter"] = adapter_name
        return {
            "kind": "graph",
            "backend_ir": backend_ir,
            "artifact": artifact,
            "metadata": metadata,
        }
    if is_module(backend_ir):
        artifacts, manifest = compile_module(backend_ir, lowering_library)
        manifest = dict(manifest)
        manifest["frontend_adapter"] = adapter_name
        return {
            "kind": "module",
            "backend_ir": backend_ir,
            "artifacts": artifacts,
            "manifest": manifest,
        }
    raise CodegenError(f"frontend adapter {adapter_name} returned unsupported IR")


__all__ = ["FrontendAdapter", "adapt_frontend", "compile_frontend"]
