"""Apply explicit runtime storage roles to normalized BYOC parameters."""

from __future__ import annotations

from typing import Any

from .module_codegen import MODULE_FORMAT
from .xgraph_codegen import CodegenError, FORMAT


def apply_parameter_storage(
    source: dict[str, Any],
    constant_parameters: list[str],
    state_parameters: list[str],
) -> None:
    """Change named input parameters to constant or mutable state storage."""
    constants = set(constant_parameters)
    states = set(state_parameters)
    overlap = constants & states
    if overlap:
        raise CodegenError(
            "parameters cannot be both constant and state: "
            + ", ".join(sorted(overlap))
        )
    if source.get("format") == FORMAT:
        graphs = [source]
    elif source.get("format") == MODULE_FORMAT:
        graphs = [region.get("graph") for region in source.get("regions", [])]
    else:
        raise CodegenError("storage policy requires a normalized graph or module")
    matched: set[str] = set()
    for graph in graphs:
        if not isinstance(graph, dict):
            raise CodegenError("storage policy encountered an invalid region graph")
        for tensor in graph.get("tensors", []):
            if not isinstance(tensor, dict):
                continue
            name = tensor.get("name")
            target = "constant" if name in constants else "state" if name in states else None
            if target is None:
                continue
            if tensor.get("storage") not in {"input", target}:
                raise CodegenError(
                    f"parameter {name} with storage {tensor.get('storage')} "
                    f"cannot become {target}"
                )
            tensor["storage"] = target
            matched.add(name)
    missing = (constants | states) - matched
    if missing:
        raise CodegenError(
            "storage policy parameters were not found: "
            + ", ".join(sorted(missing))
        )
