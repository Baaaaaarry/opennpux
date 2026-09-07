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


def apply_state_updates(source: dict[str, Any], updates: list[str]) -> None:
    """Attach explicit graph-output to persistent-state feedback edges."""
    if not updates:
        return
    if source.get("format") != MODULE_FORMAT:
        raise CodegenError("state updates require a normalized module")
    regions = source.get("regions", [])

    def resolve(selector: str, expected_storage: str) -> tuple[str, str]:
        if selector == "@output" and expected_storage == "produced":
            consumed = {
                (record.get("from", {}).get("region"),
                 record.get("from", {}).get("tensor"))
                for key in ("edges", "host_bindings")
                for record in source.get(key, [])
                if isinstance(record, dict)
            }
            matches = []
            for region in regions:
                name = region.get("name")
                graph = region.get("graph", {})
                for tensor_name in graph.get("outputs", []):
                    if (name, tensor_name) not in consumed:
                        matches.append((name, tensor_name))
            if len(matches) != 1:
                raise CodegenError(
                    "state update @output must resolve to exactly one graph output"
                )
            return matches[0]
        region_name, separator, tensor_name = selector.partition(".")
        if not separator:
            tensor_name = region_name
            region_name = ""
        matches = []
        for region in regions:
            name = region.get("name")
            if region_name and name != region_name:
                continue
            for tensor in region.get("graph", {}).get("tensors", []):
                storage = tensor.get("storage")
                storage_matches = (
                    storage in {"scratch", "output"}
                    if expected_storage == "produced"
                    else storage == expected_storage
                )
                if tensor.get("name") == tensor_name and storage_matches:
                    matches.append((name, tensor_name))
        if len(matches) != 1:
            raise CodegenError(
                f"state update endpoint {selector} must resolve to exactly one "
                f"{expected_storage} Tensor"
            )
        return matches[0]

    records = source.setdefault("state_updates", [])
    if not isinstance(records, list):
        raise CodegenError("module state_updates must be an array")
    for specification in updates:
        output, separator, state = specification.partition("=")
        if not separator or not output or not state:
            raise CodegenError("state update must use OUTPUT=STATE syntax")
        source_region, source_tensor = resolve(output, "produced")
        target_region, target_tensor = resolve(state, "state")
        records.append({
            "from": {"region": source_region, "tensor": source_tensor},
            "to": {"region": target_region, "tensor": target_tensor},
        })
