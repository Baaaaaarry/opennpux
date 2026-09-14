"""Convert a stable MAX/Mojo graph export into frontend-neutral backend IR."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from opennpux_backend.ir import GRAPH_FORMAT, MODULE_FORMAT


MAX_GRAPH_EXPORT_FORMAT = "OPENNPUX_MAX_GRAPH_EXPORT_V1"

_OPERATIONS = {
    "max.add": "add",
    "max.attention": "attention",
    "max.copy": "copy",
    "max.kv_pack": "kv_pack",
    "max.matmul": "matmul",
    "max.multiply": "multiply",
    "max.permute_dims": "transpose",
    "max.relu": "relu",
    "max.rms_norm": "rms_norm",
    "max.rope": "rope",
    "max.silu": "silu",
    "max.softmax": "softmax",
    "max.take": "take",
    "max.topk": "topk",
}


class MaxGraphAdapter:
    """Translate MAX-owned graph exports without importing the MAX SDK."""

    name = "mojo-max"

    def to_backend_ir(self, source: Any) -> dict[str, Any]:
        export = self._export(source)
        kind = export.get("kind", "graph")
        if kind == "graph":
            return self._graph(export)
        if kind == "module":
            return self._module(export)
        raise ValueError(f"unsupported MAX export kind {kind!r}")

    @staticmethod
    def _export(source: Any) -> dict[str, Any]:
        if not isinstance(source, Mapping):
            exporter = getattr(source, "to_opennpux_export", None)
            if not callable(exporter):
                raise TypeError(
                    "MAX source must be a mapping or implement to_opennpux_export()"
                )
            source = exporter()
        if not isinstance(source, Mapping):
            raise TypeError("MAX graph exporter must return a mapping")
        result = deepcopy(dict(source))
        if result.get("format") != MAX_GRAPH_EXPORT_FORMAT:
            raise ValueError("invalid MAX graph export format")
        return result

    def _graph(self, export: Mapping[str, Any]) -> dict[str, Any]:
        tensors = export.get("tensors")
        operations = export.get("operations")
        outputs = export.get("outputs")
        if not isinstance(tensors, list) or not isinstance(operations, list):
            raise ValueError("MAX graph export requires Tensor and operation lists")
        if not isinstance(outputs, list):
            raise ValueError("MAX graph export requires an output list")
        names: set[str] = set()
        normalized_tensors = []
        for tensor in tensors:
            if not isinstance(tensor, Mapping):
                raise ValueError("MAX Tensor records must be objects")
            value = deepcopy(dict(tensor))
            name = value.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("MAX Tensor names must be unique non-empty strings")
            names.add(name)
            normalized_tensors.append(value)
        nodes = []
        for operation in operations:
            if not isinstance(operation, Mapping):
                raise ValueError("MAX operation records must be objects")
            value = deepcopy(dict(operation))
            name = value.pop("name", None)
            if name is not None:
                value["frontend_name"] = name
            op = value.get("op")
            if op not in _OPERATIONS:
                raise ValueError(f"unsupported MAX operation {op!r}")
            value["op"] = _OPERATIONS[op]
            for field in ("inputs", "outputs"):
                references = value.get(field)
                if not isinstance(references, list) or any(
                    reference not in names for reference in references
                ):
                    raise ValueError(
                        f"MAX operation {op} has invalid {field} Tensor references"
                    )
            nodes.append(value)
        if any(output not in names for output in outputs):
            raise ValueError("MAX graph output references an unknown Tensor")
        graph = {
            "format": GRAPH_FORMAT,
            "tensors": normalized_tensors,
            "nodes": nodes,
            "outputs": list(outputs),
        }
        for field in ("shape_symbols", "arena", "metadata"):
            if field in export:
                graph[field] = deepcopy(export[field])
        return graph

    def _module(self, export: Mapping[str, Any]) -> dict[str, Any]:
        regions = export.get("regions")
        if not isinstance(regions, list):
            raise ValueError("MAX module export requires a region list")
        result = deepcopy(dict(export))
        result["format"] = MODULE_FORMAT
        result.pop("kind", None)
        result.pop("operations", None)
        for region in result["regions"]:
            if not isinstance(region, dict) or not isinstance(region.get("graph"), dict):
                raise ValueError("MAX module regions require embedded graphs")
            graph_export = dict(region["graph"])
            graph_export.setdefault("format", MAX_GRAPH_EXPORT_FORMAT)
            graph_export.setdefault("kind", "graph")
            region["graph"] = self._graph(graph_export)
        return result


__all__ = ["MAX_GRAPH_EXPORT_FORMAT", "MaxGraphAdapter"]
