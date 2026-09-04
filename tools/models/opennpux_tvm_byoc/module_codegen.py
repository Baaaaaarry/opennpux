"""Compile a DAG of OpenNPUX BYOC regions into reusable XGraph artifacts."""

from __future__ import annotations

import re
from typing import Any

from .xgraph_codegen import CodegenError, FORMAT, compile_graph


MODULE_FORMAT = "OPENNPUX_TVM_BYOC_MODULE_V1"


def _region_name(value: Any, index: int) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise CodegenError(f"region {index} has an invalid name")
    return value


def _endpoint(value: Any, label: str) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise CodegenError(f"{label} must be an object")
    region = value.get("region")
    tensor = value.get("tensor")
    if not isinstance(region, str) or not isinstance(tensor, str):
        raise CodegenError(f"{label} must identify region and tensor")
    return region, tensor


def _tensor_table(graph: dict[str, Any], region: str) -> dict[str, dict[str, Any]]:
    records = graph.get("tensors")
    if not isinstance(records, list):
        raise CodegenError(f"region {region} has no tensor table")
    table: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise CodegenError(f"region {region} has an invalid tensor record")
        name = record["name"]
        if name in table:
            raise CodegenError(f"region {region} has duplicate tensor {name}")
        table[name] = record
    return table


def compile_module(
    module: dict[str, Any], lowering_library: str | None = None,
) -> tuple[dict[str, tuple[bytes, dict[str, Any]]], dict[str, Any]]:
    """Compile each region and return artifacts plus a validated DAG manifest."""
    if not isinstance(module, dict) or module.get("format") != MODULE_FORMAT:
        raise CodegenError(f"module format must be {MODULE_FORMAT}")
    raw_regions = module.get("regions")
    if not isinstance(raw_regions, list) or not raw_regions:
        raise CodegenError("module regions must be a non-empty array")

    graphs: dict[str, dict[str, Any]] = {}
    tensor_tables: dict[str, dict[str, dict[str, Any]]] = {}
    declaration_order: dict[str, int] = {}
    for index, record in enumerate(raw_regions):
        if not isinstance(record, dict):
            raise CodegenError(f"region {index} must be an object")
        name = _region_name(record.get("name"), index)
        if name in graphs:
            raise CodegenError(f"duplicate region name {name}")
        graph = record.get("graph")
        if not isinstance(graph, dict) or graph.get("format") != FORMAT:
            raise CodegenError(f"region {name} must contain a normalized BYOC graph")
        graphs[name] = graph
        tensor_tables[name] = _tensor_table(graph, name)
        declaration_order[name] = index

    edges = module.get("edges", [])
    if not isinstance(edges, list):
        raise CodegenError("module edges must be an array")
    successors = {name: set() for name in graphs}
    indegree = {name: 0 for name in graphs}
    bound_inputs: set[tuple[str, str]] = set()
    normalized_edges = []
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise CodegenError(f"edge {index} must be an object")
        source_region, source_tensor = _endpoint(edge.get("from"), f"edge {index} source")
        target_region, target_tensor = _endpoint(edge.get("to"), f"edge {index} target")
        if source_region not in graphs or target_region not in graphs:
            raise CodegenError(f"edge {index} references an unknown region")
        source = tensor_tables[source_region].get(source_tensor)
        target = tensor_tables[target_region].get(target_tensor)
        if source is None or target is None:
            raise CodegenError(f"edge {index} references an unknown tensor")
        if source.get("storage") != "output" or target.get("storage") != "input":
            raise CodegenError(f"edge {index} must bind output storage to input storage")
        if source.get("shape") != target.get("shape") or source.get("dtype") != target.get("dtype"):
            raise CodegenError(f"edge {index} tensor type mismatch")
        target_key = (target_region, target_tensor)
        if target_key in bound_inputs:
            raise CodegenError(f"input {target_region}.{target_tensor} has multiple producers")
        bound_inputs.add(target_key)
        normalized_edges.append({
            "from_region": source_region,
            "from_tensor": source_tensor,
            "to_region": target_region,
            "to_tensor": target_tensor,
            "bytes": 4 * _product(source["shape"]),
        })
        if target_region not in successors[source_region]:
            successors[source_region].add(target_region)
            indegree[target_region] += 1

    ready = sorted(
        (name for name, degree in indegree.items() if degree == 0),
        key=declaration_order.get,
    )
    execution_order = []
    while ready:
        name = ready.pop(0)
        execution_order.append(name)
        for successor in sorted(successors[name], key=declaration_order.get):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort(key=declaration_order.get)
    if len(execution_order) != len(graphs):
        raise CodegenError("module region graph contains a cycle")

    artifacts = {}
    region_manifest = []
    for sequence, name in enumerate(execution_order):
        binary, metadata = compile_graph(graphs[name], lowering_library)
        artifact_name = f"region-{sequence:03d}-{name}.npxg"
        artifacts[name] = (binary, metadata)
        external_inputs = [
            tensor_name
            for tensor_name, tensor in tensor_tables[name].items()
            if tensor.get("storage") == "input" and (name, tensor_name) not in bound_inputs
        ]
        region_manifest.append({
            "name": name,
            "sequence": sequence,
            "artifact": artifact_name,
            "command_count": metadata["command_count"],
            "arena_size": metadata["arena_size"],
            "external_inputs": external_inputs,
            "outputs": list(graphs[name].get("outputs", [])),
        })

    manifest = {
        "format": MODULE_FORMAT,
        "region_count": len(region_manifest),
        "execution_order": execution_order,
        "regions": region_manifest,
        "edges": normalized_edges,
        "total_commands": sum(region["command_count"] for region in region_manifest),
    }
    return artifacts, manifest


def _product(shape: Any) -> int:
    if not isinstance(shape, list) or not shape or any(
        not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape
    ):
        raise CodegenError("module edge tensor has an invalid shape")
    result = 1
    for dim in shape:
        result *= dim
    return result
