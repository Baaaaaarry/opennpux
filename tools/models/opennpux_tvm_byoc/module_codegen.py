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
    binding_sources: dict[str, dict[str, str]] = {}
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
        raw_sources = record.get("binding_sources", {})
        if (not isinstance(raw_sources, dict) or
                any(not isinstance(key, str) or not isinstance(value, str)
                    for key, value in raw_sources.items())):
            raise CodegenError(f"region {name} has invalid binding_sources")
        unknown_sources = set(raw_sources) - set(tensor_tables[name])
        if unknown_sources:
            raise CodegenError(
                f"region {name} binding_sources reference unknown tensors: "
                + ", ".join(sorted(unknown_sources))
            )
        binding_sources[name] = dict(raw_sources)
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

    host_bindings = module.get("host_bindings", [])
    if not isinstance(host_bindings, list):
        raise CodegenError("module host_bindings must be an array")
    normalized_host_bindings = []
    for index, binding in enumerate(host_bindings):
        if not isinstance(binding, dict):
            raise CodegenError(f"host binding {index} must be an object")
        source_region, source_tensor = _endpoint(
            binding.get("from"), f"host binding {index} source"
        )
        target_region, target_tensor = _endpoint(
            binding.get("to"), f"host binding {index} target"
        )
        if source_region not in graphs or target_region not in graphs:
            raise CodegenError(f"host binding {index} references an unknown region")
        source = tensor_tables[source_region].get(source_tensor)
        target = tensor_tables[target_region].get(target_tensor)
        if source is None or target is None:
            raise CodegenError(f"host binding {index} references an unknown tensor")
        if source.get("storage") != "output" or target.get("storage") != "input":
            raise CodegenError(
                f"host binding {index} must bind output storage to input storage"
            )
        if source.get("shape") != target.get("shape") or source.get("dtype") != target.get("dtype"):
            raise CodegenError(f"host binding {index} tensor type mismatch")
        target_key = (target_region, target_tensor)
        if target_key in bound_inputs:
            raise CodegenError(f"input {target_region}.{target_tensor} has multiple producers")
        pipeline = binding.get("pipeline")
        if not isinstance(pipeline, list) or not pipeline:
            raise CodegenError(f"host binding {index} has an empty pipeline")
        normalized_pipeline = []
        for operation in pipeline:
            if not isinstance(operation, dict) or not isinstance(operation.get("op"), str):
                raise CodegenError(f"host binding {index} has an invalid operation")
            attrs = operation.get("attrs", {})
            if not isinstance(attrs, dict):
                raise CodegenError(f"host binding {index} operation attrs must be an object")
            normalized_pipeline.append({"op": operation["op"], "attrs": attrs})
        bound_inputs.add(target_key)
        normalized_host_bindings.append({
            "from_region": source_region,
            "from_tensor": source_tensor,
            "to_region": target_region,
            "to_tensor": target_tensor,
            "bytes": 4 * _product(source["shape"]),
            "shape": source["shape"],
            "dtype": source["dtype"],
            "pipeline": normalized_pipeline,
        })
        if target_region not in successors[source_region]:
            successors[source_region].add(target_region)
            indegree[target_region] += 1

    state_updates = module.get("state_updates", [])
    if not isinstance(state_updates, list):
        raise CodegenError("module state_updates must be an array")
    normalized_state_updates = []
    state_targets: set[tuple[str, str]] = set()
    for index, update in enumerate(state_updates):
        if not isinstance(update, dict):
            raise CodegenError(f"state update {index} must be an object")
        source_region, source_tensor = _endpoint(
            update.get("from"), f"state update {index} source"
        )
        target_region, target_tensor = _endpoint(
            update.get("to"), f"state update {index} target"
        )
        if source_region not in graphs or target_region not in graphs:
            raise CodegenError(f"state update {index} references an unknown region")
        source = tensor_tables[source_region].get(source_tensor)
        target = tensor_tables[target_region].get(target_tensor)
        if source is None or target is None:
            raise CodegenError(f"state update {index} references an unknown tensor")
        produced_tensors = {
            tensor_name
            for node in graphs[source_region].get("nodes", [])
            for tensor_name in node.get("outputs", [])
        }
        if (source_tensor not in produced_tensors or
                source.get("storage") not in {"scratch", "output"} or
                target.get("storage") != "state"):
            raise CodegenError(
                f"state update {index} must bind a produced Tensor to state storage"
            )
        mode = update.get("mode", "replace")
        if mode not in {"replace", "append", "append_planar2"}:
            raise CodegenError(f"state update {index} has an invalid mode")
        if source.get("dtype") != target.get("dtype"):
            raise CodegenError(f"state update {index} tensor type mismatch")
        source_shape = source.get("shape")
        target_shape = target.get("shape")
        if mode == "replace":
            if source_shape != target_shape:
                raise CodegenError(f"state update {index} tensor type mismatch")
            capacity = 1
        elif mode == "append":
            if (not isinstance(target_shape, list) or not target_shape or
                    target_shape[1:] != source_shape):
                raise CodegenError(
                    f"state append {index} requires state shape [capacity, *source]"
                )
            capacity = target_shape[0]
            stride = 4 * _product(source_shape)
        else:
            if (not isinstance(source_shape, list) or len(source_shape) < 2 or
                    source_shape[0] != 2 or
                    not isinstance(target_shape, list) or
                    len(target_shape) != len(source_shape) + 1 or
                    target_shape[0] != 2 or
                    target_shape[2:] != source_shape[1:]):
                raise CodegenError(
                    f"state planar append {index} requires source [2,...] "
                    "and state [2,capacity,...]"
                )
            capacity = target_shape[1]
            stride = 4 * _product(source_shape[1:])
        if mode == "replace":
            stride = 4 * _product(source_shape)
        target_key = (target_region, target_tensor)
        if target_key in state_targets:
            raise CodegenError(
                f"state {target_region}.{target_tensor} has multiple updates"
            )
        state_targets.add(target_key)
        normalized_state_updates.append({
            "from_region": source_region,
            "from_tensor": source_tensor,
            "to_region": target_region,
            "to_tensor": target_tensor,
            "bytes": 4 * _product(source["shape"]),
            "mode": mode,
            "stride": stride,
            "capacity": capacity,
        })
        if (source_region != target_region and
                target_region not in successors[source_region]):
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
    edge_sources = {
        (edge["from_region"], edge["from_tensor"]) for edge in normalized_edges
    }
    edge_sources.update(
        (binding["from_region"], binding["from_tensor"])
        for binding in normalized_host_bindings
    )
    for sequence, name in enumerate(execution_order):
        binary, metadata = compile_graph(graphs[name], lowering_library)
        artifact_name = f"region-{sequence:03d}-{name}.npxg"
        artifacts[name] = (binary, metadata)
        external_inputs = [
            tensor_name
            for tensor_name, tensor in tensor_tables[name].items()
            if tensor.get("storage") == "input" and (name, tensor_name) not in bound_inputs
        ]
        external_bindings = [
            tensor_name
            for tensor_name, tensor in tensor_tables[name].items()
            if tensor.get("storage") in {"input", "constant", "state"}
            and (name, tensor_name) not in bound_inputs
        ]
        invocation_bindings = [
            tensor_name
            for tensor_name, tensor in tensor_tables[name].items()
            if tensor.get("storage") == "input"
            and (name, tensor_name) not in bound_inputs
        ]
        constant_bindings = [
            tensor_name
            for tensor_name, tensor in tensor_tables[name].items()
            if tensor.get("storage") == "constant"
            and (name, tensor_name) not in bound_inputs
        ]
        state_bindings = [
            tensor_name
            for tensor_name, tensor in tensor_tables[name].items()
            if tensor.get("storage") == "state"
        ]
        region_manifest.append({
            "name": name,
            "sequence": sequence,
            "artifact": artifact_name,
            "command_count": metadata["command_count"],
            "arena_size": metadata["arena_size"],
            "external_inputs": external_inputs,
            "external_bindings": external_bindings,
            "invocation_bindings": invocation_bindings,
            "constant_bindings": constant_bindings,
            "state_bindings": state_bindings,
            "binding_sources": {
                tensor_name: binding_sources[name].get(tensor_name, tensor_name)
                for tensor_name in external_bindings
            },
            "outputs": list(graphs[name].get("outputs", [])),
        })

    scalar_bindings = module.get("scalar_bindings", [])
    if not isinstance(scalar_bindings, list):
        raise CodegenError("module scalar_bindings must be an array")
    allowed_scalar_fields = {
        "flags", "dim0", "dim1", "dim2", "scalar0",
        "reserved0", "reserved1", "reserved2", "reserved3", "reserved4",
    }
    region_records = {region["name"]: region for region in region_manifest}
    normalized_scalar_bindings = []
    scalar_names: set[str] = set()
    for index, binding in enumerate(scalar_bindings):
        if not isinstance(binding, dict):
            raise CodegenError(f"scalar binding {index} must be an object")
        name = binding.get("name")
        region = binding.get("region")
        command = binding.get("command")
        field = binding.get("field")
        minimum = binding.get("minimum", 0)
        maximum = binding.get("maximum", 0xFFFFFFFF)
        if (not isinstance(name, str) or
                not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", name) or
                name in scalar_names):
            raise CodegenError(f"scalar binding {index} has an invalid name")
        if region not in region_records:
            raise CodegenError(f"scalar binding {name} references an unknown region")
        if (not isinstance(command, int) or isinstance(command, bool) or
                command < 0 or command >= region_records[region]["command_count"]):
            raise CodegenError(f"scalar binding {name} has an invalid command")
        if field not in allowed_scalar_fields:
            raise CodegenError(f"scalar binding {name} has an invalid field")
        if (not isinstance(minimum, int) or isinstance(minimum, bool) or
                not isinstance(maximum, int) or isinstance(maximum, bool) or
                minimum < 0 or maximum > 0xFFFFFFFF or minimum > maximum):
            raise CodegenError(f"scalar binding {name} has an invalid range")
        scalar_names.add(name)
        normalized_scalar_bindings.append({
            "name": name,
            "region": region,
            "command": command,
            "field": field,
            "minimum": minimum,
            "maximum": maximum,
        })

    manifest = {
        "format": MODULE_FORMAT,
        "region_count": len(region_manifest),
        "execution_order": execution_order,
        "regions": region_manifest,
        "edges": normalized_edges,
        "host_bindings": normalized_host_bindings,
        "state_updates": normalized_state_updates,
        "scalar_bindings": normalized_scalar_bindings,
        "module_outputs": [
            {"region": name, "tensor": tensor}
            for name in execution_order
            for tensor in graphs[name].get("outputs", [])
            if (name, tensor) not in edge_sources
        ],
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
