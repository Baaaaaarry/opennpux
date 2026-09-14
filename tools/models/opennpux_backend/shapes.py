"""Frontend-neutral symbolic shape validation and specialization."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


SHAPE_CONTRACT_FORMAT = "OPENNPUX_BACKEND_SHAPES_V1"


class ShapeSpecializationError(ValueError):
    """A symbolic backend IR shape cannot be specialized safely."""


def _definitions(value: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    raw = value.get("shape_symbols", {})
    if not isinstance(raw, Mapping):
        raise ShapeSpecializationError("shape_symbols must be an object")
    result: dict[str, dict[str, int]] = {}
    for name, constraint in raw.items():
        if not isinstance(name, str) or not name:
            raise ShapeSpecializationError("shape symbol names must be non-empty strings")
        if not isinstance(constraint, Mapping):
            raise ShapeSpecializationError(f"shape symbol {name} requires constraints")
        minimum = constraint.get("min", 1)
        maximum = constraint.get("max")
        if (
            not isinstance(minimum, int)
            or isinstance(minimum, bool)
            or minimum <= 0
            or not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or maximum < minimum
        ):
            raise ShapeSpecializationError(
                f"shape symbol {name} requires positive min/max bounds"
            )
        result[name] = {"min": minimum, "max": maximum}
    return result


def _graphs(value: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(value.get("tensors"), list):
        return [value]
    result = []
    for region in value.get("regions", []):
        if isinstance(region, dict) and isinstance(region.get("graph"), dict):
            result.append(region["graph"])
    return result


def specialize_ir(
    value: dict[str, Any], bindings: Mapping[str, int] | None = None
) -> dict[str, Any]:
    """Return detached IR with every symbolic Tensor dimension made concrete."""
    result = deepcopy(value)
    definitions = _definitions(result)
    concrete = dict(bindings or {})
    unknown = sorted(set(concrete) - set(definitions))
    if unknown:
        raise ShapeSpecializationError(
            f"unknown shape binding {unknown[0]}"
        )
    used: set[str] = set()
    for graph in _graphs(result):
        for tensor in graph.get("tensors", []):
            if not isinstance(tensor, dict) or not isinstance(tensor.get("shape"), list):
                continue
            shape = []
            for dimension in tensor["shape"]:
                if isinstance(dimension, str):
                    used.add(dimension)
                    if dimension not in definitions:
                        raise ShapeSpecializationError(
                            f"undeclared shape symbol {dimension}"
                        )
                    if dimension not in concrete:
                        raise ShapeSpecializationError(
                            f"missing shape binding {dimension}"
                        )
                    dimension = concrete[dimension]
                if (
                    not isinstance(dimension, int)
                    or isinstance(dimension, bool)
                    or dimension <= 0
                ):
                    raise ShapeSpecializationError(
                        "Tensor dimensions must be positive integers or symbols"
                    )
                shape.append(dimension)
            tensor["shape"] = shape
    for name in used:
        dimension = concrete[name]
        constraint = definitions[name]
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < constraint["min"]
            or dimension > constraint["max"]
        ):
            raise ShapeSpecializationError(
                f"shape binding {name}={dimension!r} is outside "
                f"[{constraint['min']}, {constraint['max']}]"
            )
    result.pop("shape_symbols", None)
    if definitions or used:
        result["shape_specialization"] = {
            "format": SHAPE_CONTRACT_FORMAT,
            "bindings": {name: concrete[name] for name in sorted(used)},
        }
    return result


__all__ = [
    "SHAPE_CONTRACT_FORMAT",
    "ShapeSpecializationError",
    "specialize_ir",
]
