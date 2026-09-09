"""Build the relocatable full-model graph consumed by the OpenNPUX BYOC backend."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from opennpux_backend.ir import EXECUTION_GRAPH_FORMAT
from .xgraph_codegen import CodegenError


EXECUTABLE_FORMAT = "OPENNPUX_NPU_EXECUTABLE_V2"
TENSOR_PLAN_FORMAT = "OPENNPUX_NPU_TENSOR_PLAN_V1"


def _digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dense_ids(records: list[dict[str, Any]], field: str, label: str) -> None:
    identifiers = [record.get(field) for record in records]
    if identifiers != list(range(len(records))):
        raise CodegenError(f"{label} {field}s must be dense and ordered")


def build_execution_graph(
    executable: dict[str, Any], tensor_plan: dict[str, Any]
) -> dict[str, Any]:
    """Join compiler commands and SSA tensors without resolving runtime addresses."""
    if executable.get("format") != EXECUTABLE_FORMAT:
        raise CodegenError(f"expected {EXECUTABLE_FORMAT}")
    if tensor_plan.get("format") != TENSOR_PLAN_FORMAT:
        raise CodegenError(f"expected {TENSOR_PLAN_FORMAT}")
    commands = executable.get("commands")
    tensors = tensor_plan.get("tensors")
    command_io = tensor_plan.get("command_io")
    if not isinstance(commands, list) or not commands:
        raise CodegenError("executable commands must be a non-empty array")
    if not isinstance(tensors, list) or not tensors:
        raise CodegenError("tensor plan tensors must be a non-empty array")
    if not isinstance(command_io, list) or len(command_io) != len(commands):
        raise CodegenError("tensor plan must cover every executable command")
    if tensor_plan.get("command_count") != len(commands):
        raise CodegenError("tensor plan command_count does not match executable")
    if tensor_plan.get("tensor_count") != len(tensors):
        raise CodegenError("tensor plan tensor_count does not match tensor table")
    revision = executable.get("functional_graph_revision")
    if revision != tensor_plan.get("functional_graph_revision"):
        raise CodegenError("functional graph revisions do not match")

    _dense_ids(commands, "command_id", "command")
    _dense_ids(tensors, "id", "tensor")
    _dense_ids(command_io, "command_id", "command IO")
    tensor_ids = set(range(len(tensors)))
    consumers: dict[int, list[int]] = {index: [] for index in tensor_ids}
    nodes = []
    for command, io_record in zip(commands, command_io):
        command_id = command["command_id"]
        inputs = io_record.get("input_tensor_ids")
        outputs = io_record.get("output_tensor_ids")
        if not isinstance(inputs, list) or not inputs:
            raise CodegenError(f"command {command_id} has no input tensors")
        if not isinstance(outputs, list) or not outputs:
            raise CodegenError(f"command {command_id} has no output tensors")
        if any(value not in tensor_ids for value in inputs + outputs):
            raise CodegenError(f"command {command_id} references an unknown tensor")
        for tensor_id in inputs:
            consumers[tensor_id].append(command_id)
        nodes.append({
            "id": command_id,
            "opcode": command.get("opcode"),
            "operation": command.get("opcode"),
            "capability": command.get("capability"),
            "dependency_token": command.get("dependency_token"),
            "completion_token": command.get("completion_token"),
            "flags": command.get("flags"),
            "parameter_symbol": command.get("parameter_symbol"),
            "profiling_tag": command.get("profiling_tag"),
            "estimated_operations": command.get("estimated_operations"),
            "estimated_bytes": command.get("estimated_bytes"),
            "parameters": dict(command.get("parameters", {})),
            "attributes": dict(command.get("attributes", {})),
            "inputs": list(inputs),
            "outputs": list(outputs),
        })

    for tensor in tensors:
        tensor_id = tensor["id"]
        if list(tensor.get("consumer_commands", [])) != consumers[tensor_id]:
            raise CodegenError(
                f"tensor {tensor_id} consumer list disagrees with command IO"
            )
        producer = tensor.get("producer_command")
        if producer is not None:
            if producer not in range(len(nodes)) or tensor_id not in nodes[producer]["outputs"]:
                raise CodegenError(f"tensor {tensor_id} has an invalid producer")

    available_tokens = {0}
    for node in nodes:
        dependency = node["dependency_token"]
        completion = node["completion_token"]
        if dependency not in available_tokens:
            raise CodegenError(
                f"command {node['id']} depends on unavailable token {dependency}"
            )
        if not isinstance(completion, int) or completion <= 0:
            raise CodegenError(f"command {node['id']} has an invalid completion token")
        available_tokens.add(completion)

    observed_capabilities = {node["capability"] for node in nodes}
    required_capabilities = set(executable.get("required_capabilities", []))
    if None in observed_capabilities or observed_capabilities != required_capabilities:
        raise CodegenError("required capabilities do not match full graph nodes")

    entry_points = executable.get("entry_points")
    if not isinstance(entry_points, list) or not entry_points:
        raise CodegenError("executable has no entry points")
    for entry in entry_points:
        first = entry.get("first_command")
        count = entry.get("command_count")
        if not isinstance(first, int) or not isinstance(count, int) or first < 0:
            raise CodegenError("entry point has an invalid command range")
        if count <= 0 or first + count > len(nodes):
            raise CodegenError("entry point command range exceeds full graph")

    outputs = [
        tensor["id"] for tensor in tensors if tensor.get("storage") == "output"
    ]
    if not outputs:
        raise CodegenError("full graph has no output tensor")
    return {
        "format": EXECUTION_GRAPH_FORMAT,
        "version": 1,
        "adapter": "apache-tvm-relax-byoc",
        "target": executable.get("target"),
        "execution_scope": executable.get("execution_scope"),
        "functional_graph_revision": revision,
        "source": dict(executable.get("source", {})),
        "source_digests": {
            "executable_sha256": _digest(executable),
            "tensor_plan_sha256": _digest(tensor_plan),
        },
        "entry_points": entry_points,
        "required_capabilities": sorted(required_capabilities),
        "runtime_symbols": ["runtime.batch", "runtime.sequence", "runtime.kv"],
        "tensor_count": len(tensors),
        "node_count": len(nodes),
        "operation_counts": dict(sorted(Counter(
            node["operation"] for node in nodes
        ).items())),
        "outputs": outputs,
        "tensors": tensors,
        "nodes": nodes,
    }


def validate_execution_graph(
    graph: dict[str, Any], executable: dict[str, Any], tensor_plan: dict[str, Any]
) -> None:
    """Reject stale or substituted graph artifacts at deployment time."""
    if graph.get("format") != EXECUTION_GRAPH_FORMAT:
        raise CodegenError(f"expected {EXECUTION_GRAPH_FORMAT}")
    expected = build_execution_graph(executable, tensor_plan)
    comparable = dict(graph)
    comparable.pop("relax_ir", None)
    if comparable != expected:
        raise CodegenError("execution graph is stale or does not match its sources")


def build_relax_execution_module(graph: dict[str, Any]):
    """Materialize the complete request DAG as a TVM Relax external-codegen graph."""
    try:
        from tvm import relax, tir
    except ImportError as error:
        raise CodegenError("Apache TVM with Relax is required") from error
    if graph.get("format") != EXECUTION_GRAPH_FORMAT:
        raise CodegenError(f"expected {EXECUTION_GRAPH_FORMAT}")

    symbols: dict[str, Any] = {}

    def tensor_sinfo(record: dict[str, Any]):
        dimensions = []
        for dimension in record.get("shape", []):
            if isinstance(dimension, int) and dimension > 0:
                dimensions.append(dimension)
            elif isinstance(dimension, str) and dimension in graph["runtime_symbols"]:
                symbols.setdefault(dimension, tir.Var(dimension.replace(".", "_"), "int64"))
                dimensions.append(symbols[dimension])
            else:
                raise CodegenError(
                    f"tensor {record.get('id')} has unsupported dimension {dimension!r}"
                )
        dtype = record.get("data_type")
        if dtype not in {"float32", "int32"}:
            raise CodegenError(
                f"tensor {record.get('id')} has unsupported dtype {dtype!r}"
            )
        return relax.TensorStructInfo(dimensions, dtype)

    tensors = {record["id"]: record for record in graph["tensors"]}
    parameter_ids = [
        tensor_id for tensor_id, record in tensors.items()
        if record.get("storage") in {"input", "persistent"}
    ]
    parameters = [
        relax.Var(f"tensor_{tensor_id}", tensor_sinfo(tensors[tensor_id]))
        for tensor_id in parameter_ids
    ]
    values = dict(zip(parameter_ids, parameters))
    builder = relax.BlockBuilder()
    with builder.function(
        "main", parameters,
        attrs={"Codegen": "opennpux", "global_symbol": "opennpux_full_model"},
    ):
        with builder.dataflow():
            for node in graph["nodes"]:
                try:
                    arguments = [values[tensor_id] for tensor_id in node["inputs"]]
                except KeyError as error:
                    raise CodegenError(
                        f"command {node['id']} consumes tensor {error.args[0]} before definition"
                    ) from error
                output_info = [tensor_sinfo(tensors[value]) for value in node["outputs"]]
                result_info = output_info[0] if len(output_info) == 1 else relax.TupleStructInfo(output_info)
                descriptor = relax.StringImm(json.dumps({
                    key: value for key, value in node.items()
                    if key not in {"inputs", "outputs"}
                }, sort_keys=True, separators=(",", ":")))
                call = relax.call_pure_packed(
                    f"opennpux.request.{node['opcode'].lower()}",
                    *arguments, descriptor,
                    sinfo_args=result_info,
                )
                result = builder.emit(call, name_hint=f"command_{node['id']}")
                if len(node["outputs"]) == 1:
                    values[node["outputs"][0]] = result
                else:
                    for index, tensor_id in enumerate(node["outputs"]):
                        values[tensor_id] = builder.emit(
                            relax.TupleGetItem(result, index),
                            name_hint=f"tensor_{tensor_id}",
                        )
            results = [values[tensor_id] for tensor_id in graph["outputs"]]
            output = results[0] if len(results) == 1 else relax.Tuple(results)
            output = builder.emit_output(output)
        builder.emit_func_output(output)
    return builder.get()


def save_relax_execution_module(graph: dict[str, Any], output: Path) -> str:
    """Write canonical TVM JSON and return its SHA-256 identity."""
    try:
        import tvm
    except ImportError as error:
        raise CodegenError("Apache TVM with Relax is required") from error
    encoded = tvm.ir.save_json(build_relax_execution_module(graph))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(encoded.encode()).hexdigest()
