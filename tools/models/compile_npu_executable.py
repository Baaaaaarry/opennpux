#!/usr/bin/env python3
"""Lower frontend execution-plan phases into a generic NPU executable."""

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


FORMAT = "OPENNPUX_NPU_EXECUTABLE_V2"
TENSOR_PLAN_FORMAT = "OPENNPUX_NPU_TENSOR_PLAN_V1"
HEADER = struct.Struct("<8I3Q2I3Q")
ENTRY = struct.Struct("<4I4Q")
COMMAND = struct.Struct("<8I4Q2IQ")
OPERATOR_PARAMETERS = struct.Struct("<16I")
TENSOR_PLAN_HEADER = struct.Struct("<12I4Q")
TENSOR_PLAN_TENSOR = struct.Struct("<16I2Q")
TENSOR_PLAN_COMMAND = struct.Struct("<12I")
TENSOR_PLAN_SLOT = struct.Struct("<2IQ")
MAGIC = 0x4558504E
TENSOR_PLAN_MAGIC = 0x5054504E
INVOCATION_HEADER_SIZE = 144
TENSOR_BINDING_SIZE = 112
INVOCATION_COMMAND_SIZE = 112
RECORD_ALIGNMENT = 64
OPERATOR_PARAMETERS_MAGIC = 0x5058504E

OPCODE = {
    "EMBED": 1, "MATMUL": 2, "ADD": 3, "MUL": 4, "NORMALIZE": 5,
    "ROPE": 6, "SOFTMAX": 7, "TOPK": 8, "CONVOLUTION": 9,
    "CAUSAL_CONVOLUTION": 10, "RECURRENT_UPDATE": 11, "ROUTER": 12,
    "EXPERT": 13, "DMA": 14, "ATTENTION": 15, "ACTIVATION": 16,
    "COMBINE": 17,
}

PHASE_OPCODE = {
    "attention_norm": "NORMALIZE",
    "qkv_projection": "MATMUL",
    "rope": "ROPE",
    "paged_kv_cache_update": "DMA",
    "scaled_dot_product_attention": "ATTENTION",
    "attention_output_projection": "MATMUL",
    "linear_attention_projection": "MATMUL",
    "causal_depthwise_conv": "CAUSAL_CONVOLUTION",
    "recurrent_state_update": "RECURRENT_UPDATE",
    "linear_attention_gate_norm": "NORMALIZE",
    "linear_attention_output_projection": "MATMUL",
    "residual_add": "ADD",
    "ffn_norm": "NORMALIZE",
    "router_topk": "ROUTER",
    "routed_experts_active_only": "EXPERT",
    "shared_expert": "EXPERT",
    "moe_combine": "COMBINE",
}

STATE_PHASES = {"paged_kv_cache_update", "recurrent_state_update"}

WEIGHT_PHASES = {
    "token_embedding", "attention_norm", "qkv_projection",
    "attention_output_projection", "linear_attention_projection",
    "causal_depthwise_conv", "linear_attention_gate_norm",
    "linear_attention_output_projection", "ffn_norm", "router_topk",
    "routed_experts_active_only", "shared_expert", "final_norm", "lm_head",
}

MODEL_PHASE_OPCODE = {
    "token_embedding": "EMBED",
    "final_norm": "NORMALIZE",
    "lm_head": "MATMUL",
    "token_selection": "TOPK",
}

PHASE_KIND = {
    "token_embedding": 1,
    "attention_norm": 2,
    "ffn_norm": 2,
    "linear_attention_gate_norm": 2,
    "final_norm": 2,
    "qkv_projection": 3,
    "attention_output_projection": 3,
    "linear_attention_projection": 3,
    "linear_attention_output_projection": 3,
    "lm_head": 3,
    "scaled_dot_product_attention": 4,
    "paged_kv_cache_update": 5,
    "recurrent_state_update": 5,
    "router_topk": 6,
    "routed_experts_active_only": 7,
    "shared_expert": 7,
    "moe_combine": 8,
    "token_selection": 9,
}


def operator_parameters(manifest: dict[str, Any], phase: str, opcode: str) -> dict[str, int]:
    hidden = max(1, int(manifest.get("hidden_size", 1)))
    heads = max(1, int(manifest.get("head_count", 1)))
    kv_heads = max(1, int(manifest.get("kv_head_count", heads)))
    head_dim = max(1, int(manifest.get("head_dim", hidden // heads or 1)))
    experts = max(1, int(manifest.get("expert_count", 1)))
    moe = max(1, int(manifest.get("moe_intermediate_size", hidden)))
    shared = max(1, int(manifest.get("shared_expert_intermediate_size", moe)))
    vocab = max(1, int(manifest.get("vocab_size", 1)))
    input_features = hidden
    output_features = hidden
    intermediate = 0
    if phase == "qkv_projection":
        output_features = (heads + 2 * kv_heads) * head_dim
    elif phase == "router_topk":
        output_features = experts
    elif phase == "routed_experts_active_only":
        intermediate = moe
    elif phase == "shared_expert":
        intermediate = shared
    elif phase == "lm_head":
        output_features = vocab
    elif phase == "token_embedding":
        input_features = 1
    elif phase == "token_selection":
        input_features = vocab
        output_features = 1
    quantization = manifest.get("quantization", {})
    quant_bits = int(manifest.get("quantization_bits", quantization.get("bits", 0)))
    group_size = int(manifest.get(
        "quantization_group_size", quantization.get("group_size", 0)
    ))
    flags = 2
    if opcode in {"MATMUL", "EXPERT", "ROUTER", "EMBED"} and quant_bits == 4:
        flags |= 1
    return {
        "phase": PHASE_KIND.get(phase, 0),
        "flags": flags,
        "input_features": input_features,
        "output_features": output_features,
        "intermediate_features": intermediate,
        "head_count": heads,
        "kv_head_count": kv_heads,
        "head_dim": head_dim,
        "quantization_bits": quant_bits,
        "quantization_group_size": group_size,
        "scale_data_type": int(manifest.get("quantization_scale_data_type", 6)),
        "quantized_zero_bias": int(manifest.get("quantization_zero_bias", 0)),
    }


def estimate_phase_work(
    manifest: dict[str, Any], phase: str
) -> tuple[int, int]:
    """Return architecture-neutral operations and bytes per token."""
    hidden = max(1, int(manifest.get("hidden_size", 1)))
    heads = max(1, int(manifest.get("head_count", 1)))
    kv_heads = max(1, int(manifest.get("kv_head_count", heads)))
    head_dim = max(1, int(manifest.get("head_dim", hidden // heads or 1)))
    experts = max(1, int(manifest.get("expert_count", 1)))
    active = max(1, int(manifest.get("experts_per_token", 1)))
    moe = max(1, int(manifest.get("moe_intermediate_size", hidden)))
    shared = max(1, int(manifest.get("shared_expert_intermediate_size", moe)))
    element_bytes = 2

    if phase == "qkv_projection":
        output = (heads + 2 * kv_heads) * head_dim
        return 2 * hidden * output, (hidden + output) * element_bytes
    if phase in {
        "attention_output_projection",
        "linear_attention_projection",
        "linear_attention_output_projection",
    }:
        return 2 * hidden * hidden, 2 * hidden * element_bytes
    if phase == "scaled_dot_product_attention":
        return 4 * heads * head_dim, 3 * heads * head_dim * element_bytes
    if phase == "router_topk":
        return 2 * hidden * experts, (hidden + experts) * element_bytes
    if phase == "routed_experts_active_only":
        return 6 * hidden * moe * active, (hidden + moe * active) * element_bytes
    if phase == "shared_expert":
        return 6 * hidden * shared, (hidden + shared) * element_bytes
    if phase in {"attention_norm", "ffn_norm", "linear_attention_gate_norm"}:
        return 5 * hidden, 2 * hidden * element_bytes
    if phase == "rope":
        return 4 * heads * head_dim, 2 * heads * head_dim * element_bytes
    if phase == "causal_depthwise_conv":
        return 8 * hidden, 3 * hidden * element_bytes
    if phase == "recurrent_state_update":
        return 8 * hidden, 3 * hidden * element_bytes
    if phase == "moe_combine":
        return active * hidden, (active + 1) * hidden * element_bytes
    if phase == "residual_add":
        return hidden, 3 * hidden * element_bytes
    if phase == "paged_kv_cache_update":
        return hidden, 2 * hidden * element_bytes
    if phase == "token_embedding":
        return hidden, hidden * element_bytes
    if phase == "final_norm":
        return 5 * hidden, 2 * hidden * element_bytes
    if phase == "lm_head":
        vocab = max(1, int(manifest.get("vocab_size", 1)))
        return 2 * hidden * vocab, (hidden + vocab) * element_bytes
    if phase == "token_selection":
        vocab = max(1, int(manifest.get("vocab_size", 1)))
        return 3 * vocab, vocab * element_bytes
    return hidden, 2 * hidden * element_bytes


def load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def lower_commands(plan: dict[str, Any]) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    dependency = 0

    def append_command(
        opcode: str, parameter_symbol: str, profiling_tag: int,
        attributes: dict[str, Any]
    ) -> None:
        nonlocal dependency
        phase = str(attributes["phase"])
        completion = len(commands) + 1
        estimated_operations, estimated_bytes = estimate_phase_work(
            plan.get("model", {}), phase
        )
        commands.append(
            {
                "command_id": len(commands),
                "opcode": opcode,
                "flags": 1 if phase in WEIGHT_PHASES else 0,
                "capability": f"op.{opcode.lower()}.v1",
                "dependency_token": dependency,
                "completion_token": completion,
                "parameter_symbol": parameter_symbol,
                "profiling_tag": profiling_tag,
                "estimated_operations": estimated_operations,
                "estimated_bytes": estimated_bytes,
                "attributes": attributes,
                "parameters": operator_parameters(
                    plan.get("model", {}), phase, opcode
                ),
            }
        )
        dependency = completion

    append_command(
        MODEL_PHASE_OPCODE["token_embedding"], "model.token_embedding", 0xff000001,
        {"scope": "model", "phase": "token_embedding", "persistent_state": False},
    )
    for layer in plan.get("layers", []):
        layer_index = int(layer["index"])
        layer_type = str(layer["type"])
        for phase_index, phase_value in enumerate(layer.get("phases", [])):
            phase = str(phase_value)
            opcode = PHASE_OPCODE.get(phase)
            if opcode is None:
                raise ValueError(
                    f"no generic opcode for layer {layer_index} phase {phase}"
                )
            append_command(
                opcode, f"layer.{layer_index}.{phase}",
                (layer_index << 16) | phase_index,
                {
                    "scope": "layer",
                    "layer": layer_index,
                    "layer_type": layer_type,
                    "phase": phase,
                    "persistent_state": phase in STATE_PHASES,
                },
            )
    for index, phase in enumerate(("final_norm", "lm_head", "token_selection")):
        append_command(
            MODEL_PHASE_OPCODE[phase], f"model.{phase}", 0xff000010 + index,
            {"scope": "model", "phase": phase, "persistent_state": False},
        )
    if not commands:
        raise ValueError("execution plan contains no commands")
    return commands


def build_executable(
    manifest: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    if plan.get("format") != "OPENNPUX_QWEN_EXECUTION_PLAN_V1":
        raise ValueError("unsupported frontend execution plan")
    if int(plan.get("observed_layer_count", 0)) != int(
        manifest.get("layer_count", -1)
    ):
        raise ValueError("execution plan does not cover every decoder layer")
    unclassified_layers = [
        int(layer["index"])
        for layer in plan.get("layers", [])
        if layer.get("type") == "unclassified_moe"
    ]
    if unclassified_layers:
        raise ValueError(
            "unclassified decoder layers: "
            + ",".join(str(layer) for layer in unclassified_layers)
        )
    unknown_decoder = plan.get("unknown_decoder_tensor_patterns", {})
    if unknown_decoder:
        patterns = ", ".join(list(unknown_decoder)[:8])
        raise ValueError(
            f"decoder has {len(unknown_decoder)} unclassified tensor patterns: "
            f"{patterns}"
        )
    plan_with_model = dict(plan)
    plan_with_model["model"] = manifest
    commands = lower_commands(plan_with_model)
    capabilities = sorted({command["capability"] for command in commands})
    return {
        "format": FORMAT,
        "version": 2,
        "default_active_experts": int(manifest.get("experts_per_token", 1)),
        "target": "opennpux-coral-generic-v1",
        "source": {
            "model_manifest": plan["model_manifest"],
            "architecture": plan["architecture"],
            "frontend_plan_format": plan["format"],
        },
        "memory_requirements": {
            "weights": "paged-external",
            "scratch": "runtime-sized",
            "persistent_state": "runtime-bound",
            "alignment": 64,
        },
        "execution_scope": "token-to-next-token",
        "logical_bindings": [
            {"id": 0, "name": "input", "access": "read", "dynamic": True},
            {"id": 1, "name": "output", "access": "write", "dynamic": True},
            {"id": 2, "name": "weights", "access": "read", "dynamic": True},
            {
                "id": 3,
                "name": "persistent_state",
                "access": "read_write",
                "dynamic": True,
            },
            {"id": 4, "name": "scratch", "access": "read_write", "dynamic": True},
        ],
        "entry_points": [
            {"id": 1, "name": "prefill", "first_command": 0, "command_count": len(commands)},
            {"id": 2, "name": "decode", "first_command": 0, "command_count": len(commands)},
        ],
        "required_capabilities": capabilities,
        "commands": commands,
    }


def build_tensor_plan(executable: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Build a model-independent SSA tensor graph for executable commands."""
    hidden = max(1, int(manifest.get("hidden_size", 1)))
    heads = max(1, int(manifest.get("head_count", 1)))
    kv_heads = max(1, int(manifest.get("kv_head_count", heads)))
    head_dim = max(1, int(manifest.get("head_dim", hidden // heads or 1)))
    active_experts = max(1, int(manifest.get("experts_per_token", 1)))
    vocab = max(1, int(manifest.get("vocab_size", 1)))
    tensors: list[dict[str, Any]] = []
    command_io: list[dict[str, Any]] = []
    consumers: dict[int, list[int]] = {}

    def tensor(
        name: str, storage: str, shape: list[int | str], producer: int | None,
        data_type: str = "float32",
    ) -> int:
        tensor_id = len(tensors)
        tensors.append({
            "id": tensor_id,
            "name": name,
            "storage": storage,
            "data_type": data_type,
            "shape": shape,
            "producer_command": producer,
        })
        consumers[tensor_id] = []
        return tensor_id

    def emit(command: dict[str, Any], inputs: list[int], outputs: list[int]) -> None:
        command_id = int(command["command_id"])
        if not inputs or not outputs:
            raise ValueError(f"command {command_id} has incomplete tensor IO")
        for tensor_id in inputs:
            consumers[tensor_id].append(command_id)
        command_io.append({
            "command_id": command_id,
            "input_tensor_ids": inputs,
            "output_tensor_ids": outputs,
        })

    rows: list[int | str] = ["runtime.batch", "runtime.sequence"]
    token_ids = tensor("invocation.token_ids", "input", rows, None, "int32")
    final_output = tensor(
        "invocation.next_token", "output", ["runtime.batch"], None, "int32"
    )
    current = token_ids
    layer_input = token_ids
    post_attention = token_ids
    route_indices: int | None = None
    route_weights: int | None = None
    routed_output: int | None = None
    shared_output: int | None = None
    query_tensor: int | None = None
    key_tensor: int | None = None
    value_tensor: int | None = None
    layer_residual_count: dict[int, int] = {}

    for command in executable["commands"]:
        command_id = int(command["command_id"])
        attributes = command["attributes"]
        phase = str(attributes["phase"])
        layer = attributes.get("layer")
        prefix = "model" if layer is None else f"layer.{int(layer)}"

        if phase == "token_embedding":
            output = tensor(f"{prefix}.hidden", "scratch", rows + [hidden], command_id)
            emit(command, [current], [output])
            current = output
            continue
        if layer is not None and phase == "attention_norm":
            layer_input = current
        if phase in {"attention_norm", "ffn_norm", "linear_attention_gate_norm",
                     "final_norm"}:
            output = tensor(f"{prefix}.{phase}", "scratch", rows + [hidden], command_id)
            emit(command, [current], [output])
            current = output
            continue
        if phase == "qkv_projection":
            query_tensor = tensor(
                f"{prefix}.query", "scratch", rows + [heads * head_dim], command_id
            )
            key_tensor = tensor(
                f"{prefix}.key", "scratch", rows + [kv_heads * head_dim], command_id
            )
            value_tensor = tensor(
                f"{prefix}.value", "scratch", rows + [kv_heads * head_dim], command_id
            )
            emit(command, [current], [query_tensor, key_tensor, value_tensor])
            current = query_tensor
            continue
        if phase in {"attention_output_projection", "linear_attention_projection",
                     "linear_attention_output_projection"}:
            output = tensor(f"{prefix}.{phase}", "scratch", rows + [hidden], command_id)
            emit(command, [current], [output])
            current = output
            continue
        if phase == "rope":
            if query_tensor is None or key_tensor is None:
                raise ValueError(f"command {command_id} has no Q/K tensors")
            rotated_query = tensor(
                f"{prefix}.query_rope", "scratch",
                list(tensors[query_tensor]["shape"]), command_id,
            )
            rotated_key = tensor(
                f"{prefix}.key_rope", "scratch", list(tensors[key_tensor]["shape"]),
                command_id,
            )
            emit(command, [query_tensor, key_tensor], [rotated_query, rotated_key])
            query_tensor = rotated_query
            key_tensor = rotated_key
            current = rotated_query
            continue
        if phase == "paged_kv_cache_update":
            if key_tensor is None or value_tensor is None:
                raise ValueError(f"command {command_id} has no K/V tensors")
            state = tensor(
                f"{prefix}.kv_cache", "persistent",
                [2, "runtime.batch", "runtime.kv", kv_heads, head_dim], command_id,
            )
            emit(command, [key_tensor, value_tensor], [state])
            continue
        if phase == "scaled_dot_product_attention":
            output = tensor(f"{prefix}.attention", "scratch", rows + [hidden], command_id)
            state = next(
                item["id"] for item in reversed(tensors)
                if item["name"] == f"{prefix}.kv_cache"
            )
            emit(command, [current, state], [output])
            current = output
            continue
        if phase == "causal_depthwise_conv":
            output = tensor(f"{prefix}.conv", "scratch", rows + [hidden], command_id)
            emit(command, [current], [output])
            current = output
            continue
        if phase == "recurrent_state_update":
            state = tensor(
                f"{prefix}.recurrent_state", "persistent",
                ["runtime.batch", hidden], command_id,
            )
            output = tensor(f"{prefix}.recurrent", "scratch", rows + [hidden], command_id)
            emit(command, [current], [output, state])
            current = output
            continue
        if phase == "router_topk":
            route_indices = tensor(
                f"{prefix}.route_indices", "scratch", rows + [active_experts],
                command_id, "int32",
            )
            route_weights = tensor(
                f"{prefix}.route_weights", "scratch", rows + [active_experts], command_id
            )
            emit(command, [current], [route_indices, route_weights])
            continue
        if phase == "routed_experts_active_only":
            if route_indices is None or route_weights is None:
                raise ValueError(f"command {command_id} has no router outputs")
            routed_output = tensor(
                f"{prefix}.routed_expert", "scratch", rows + [hidden], command_id
            )
            emit(command, [current, route_indices, route_weights], [routed_output])
            continue
        if phase == "shared_expert":
            shared_output = tensor(
                f"{prefix}.shared_expert", "scratch", rows + [hidden], command_id
            )
            emit(command, [current], [shared_output])
            continue
        if phase == "moe_combine":
            if routed_output is None or shared_output is None or route_weights is None:
                raise ValueError(f"command {command_id} has incomplete expert inputs")
            output = tensor(f"{prefix}.moe", "scratch", rows + [hidden], command_id)
            emit(command, [routed_output, shared_output, route_weights], [output])
            current = output
            continue
        if phase == "residual_add":
            layer_index = int(layer)
            residual_index = layer_residual_count.get(layer_index, 0)
            residual = layer_input if residual_index == 0 else post_attention
            output = tensor(
                f"{prefix}.residual.{residual_index}", "scratch", rows + [hidden],
                command_id,
            )
            emit(command, [residual, current], [output])
            current = output
            layer_residual_count[layer_index] = residual_index + 1
            if residual_index == 0:
                post_attention = output
            continue
        if phase == "lm_head":
            output = tensor(f"{prefix}.logits", "scratch", rows + [vocab], command_id)
            emit(command, [current], [output])
            current = output
            continue
        if phase == "token_selection":
            tensors[final_output]["producer_command"] = command_id
            emit(command, [current], [final_output])
            current = final_output
            continue
        raise ValueError(f"command {command_id} has no tensor lowering for phase {phase}")

    if len(command_io) != len(executable["commands"]):
        raise ValueError("tensor plan does not cover every command")
    for item in tensors:
        item_consumers = consumers[item["id"]]
        item["consumer_commands"] = item_consumers
        item["last_consumer_command"] = max(
            item_consumers,
            default=item["producer_command"] if item["producer_command"] is not None else 0,
        )

    scratch_tensors = [item for item in tensors if item["storage"] == "scratch"]
    slots: list[dict[str, int]] = []
    active: list[tuple[int, int]] = []
    for item in sorted(scratch_tensors, key=lambda value: int(value["producer_command"])):
        producer = int(item["producer_command"])
        active = [(end, slot) for end, slot in active if end >= producer]
        used = {slot for _, slot in active}
        slot = next((index for index in range(len(slots)) if index not in used), len(slots))
        static_elements = 1
        for dimension in item["shape"]:
            if isinstance(dimension, int):
                static_elements *= dimension
        bytes_per_row = static_elements * 4
        if slot == len(slots):
            slots.append({"id": slot, "bytes_per_runtime_row": bytes_per_row})
        else:
            slots[slot]["bytes_per_runtime_row"] = max(
                slots[slot]["bytes_per_runtime_row"], bytes_per_row
            )
        item["allocation_slot"] = slot
        item["bytes_per_runtime_row"] = bytes_per_row
        active.append((int(item["last_consumer_command"]), slot))

    persistent_tensors = [
        item for item in tensors if item["storage"] == "persistent"
    ]
    for slot, item in enumerate(persistent_tensors):
        item["allocation_slot"] = slot

    return {
        "format": TENSOR_PLAN_FORMAT,
        "version": 1,
        "execution_scope": executable["execution_scope"],
        "runtime_row_expression": "runtime.batch * runtime.sequence",
        "tensor_count": len(tensors),
        "command_count": len(command_io),
        "scratch_slot_count": len(slots),
        "persistent_slot_count": len(persistent_tensors),
        "scratch_bytes_per_runtime_row": sum(slot["bytes_per_runtime_row"] for slot in slots),
        "scratch_slots": slots,
        "tensors": tensors,
        "command_io": command_io,
    }


def hash64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def checksum(data: bytes, checksum_offset: int) -> int:
    mutable = bytearray(data)
    mutable[checksum_offset : checksum_offset + 4] = b"\0\0\0\0"
    value = 2166136261
    for byte in mutable:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def align(value: int, alignment: int = RECORD_ALIGNMENT) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def write_binary(executable: dict[str, Any], path: Path) -> None:
    entries = executable["entry_points"]
    commands = executable["commands"]
    entry_offset = HEADER.size
    command_offset = entry_offset + len(entries) * ENTRY.size
    parameter_offset = align(command_offset + len(commands) * COMMAND.size)
    parameter_size = len(commands) * OPERATOR_PARAMETERS.size
    total_size = parameter_offset + parameter_size
    executable_id = hash64(json.dumps(executable["source"], sort_keys=True)) or 1
    header = HEADER.pack(
        MAGIC, 2, HEADER.size, total_size, len(entries), len(commands),
        ENTRY.size, COMMAND.size, entry_offset, command_offset, executable_id,
        0, int(executable["default_active_experts"]), parameter_offset,
        parameter_size, 0,
    )
    entry_data = b"".join(
        ENTRY.pack(
            int(entry["id"]), int(entry["first_command"]),
            int(entry["command_count"]), 0, 0, 0, 0, 0,
        )
        for entry in entries
    )
    command_data = b"".join(
        COMMAND.pack(
            int(command["command_id"]), OPCODE[command["opcode"]],
            int(command["flags"]),
            hash64(command["capability"]) & 0xFFFFFFFF, 0, 5,
            int(command["dependency_token"]), int(command["completion_token"]),
            hash64(command["parameter_symbol"]),
            int(command["estimated_operations"]),
            int(command["estimated_bytes"]),
            int(command["profiling_tag"]),
            index * OPERATOR_PARAMETERS.size,
            OPERATOR_PARAMETERS.size,
            2 | (3 << 16) | (4 << 32),
        )
        for index, command in enumerate(commands)
    )
    parameter_data = b"".join(
        OPERATOR_PARAMETERS.pack(
            OPERATOR_PARAMETERS_MAGIC, 2, OPERATOR_PARAMETERS.size,
            OPCODE[command["opcode"]], int(command["parameters"]["phase"]),
            int(command["parameters"]["flags"]),
            int(command["parameters"]["input_features"]),
            int(command["parameters"]["output_features"]),
            int(command["parameters"]["intermediate_features"]),
            int(command["parameters"]["head_count"]),
            int(command["parameters"]["kv_head_count"]),
            int(command["parameters"]["head_dim"]),
            int(command["parameters"]["quantization_bits"]),
            int(command["parameters"]["quantization_group_size"]),
            int(command["parameters"]["scale_data_type"]),
            int(command["parameters"]["quantized_zero_bias"]),
        )
        for command in commands
    )
    data = header + entry_data + command_data
    data += b"\0" * (parameter_offset - len(data)) + parameter_data
    # offsetof(opennpux_npu_executable_header, checksum)
    checksum_offset = 56
    value = checksum(data, checksum_offset)
    data = data[:checksum_offset] + struct.pack("<I", value) + data[checksum_offset + 4 :]
    path.write_bytes(data)


def write_tensor_plan_binary(plan: dict[str, Any], path: Path) -> None:
    storage_ids = {"input": 1, "output": 2, "scratch": 3, "persistent": 4}
    data_type_ids = {"int32": 3, "float32": 6}
    dimension_ids = {
        "runtime.batch": 1,
        "runtime.sequence": 2,
        "runtime.kv": 3,
        "runtime.active_experts": 4,
    }
    tensor_offset = align(TENSOR_PLAN_HEADER.size)
    command_offset = align(tensor_offset + len(plan["tensors"]) * TENSOR_PLAN_TENSOR.size)
    slot_offset = align(command_offset + len(plan["command_io"]) * TENSOR_PLAN_COMMAND.size)
    total_size = slot_offset + len(plan["scratch_slots"]) * TENSOR_PLAN_SLOT.size
    header = TENSOR_PLAN_HEADER.pack(
        TENSOR_PLAN_MAGIC, 1, TENSOR_PLAN_HEADER.size, total_size,
        len(plan["tensors"]), len(plan["command_io"]), len(plan["scratch_slots"]),
        TENSOR_PLAN_TENSOR.size, TENSOR_PLAN_COMMAND.size, TENSOR_PLAN_SLOT.size,
        0, 0, tensor_offset, command_offset, slot_offset,
        int(plan["scratch_bytes_per_runtime_row"]),
    )
    tensor_data = bytearray()
    for tensor in plan["tensors"]:
        dimensions = [0] * 8
        dimension_symbols = 0
        shape = tensor["shape"]
        if len(shape) > len(dimensions):
            raise ValueError(f"tensor {tensor['id']} rank exceeds binary ABI")
        for index, dimension in enumerate(shape):
            if isinstance(dimension, int):
                dimensions[index] = dimension
            else:
                symbol = dimension_ids.get(str(dimension))
                if symbol is None:
                    raise ValueError(f"unknown runtime dimension {dimension}")
                dimension_symbols |= symbol << (index * 4)
        producer = tensor["producer_command"]
        tensor_data += TENSOR_PLAN_TENSOR.pack(
            int(tensor["id"]), storage_ids[str(tensor["storage"])],
            data_type_ids[str(tensor["data_type"])], len(shape),
            0xFFFFFFFF if producer is None else int(producer),
            int(tensor["last_consumer_command"]),
            int(tensor.get("allocation_slot", 0xFFFFFFFF)), dimension_symbols,
            *dimensions, int(tensor.get("bytes_per_runtime_row", 0)), 0,
        )
    command_data = bytearray()
    for record in plan["command_io"]:
        inputs = [int(value) for value in record["input_tensor_ids"]]
        outputs = [int(value) for value in record["output_tensor_ids"]]
        if len(inputs) > 4 or len(outputs) > 3:
            raise ValueError(f"command {record['command_id']} exceeds tensor IO ABI")
        command_data += TENSOR_PLAN_COMMAND.pack(
            int(record["command_id"]), len(inputs), len(outputs), 0,
            *(inputs + [0xFFFFFFFF] * (4 - len(inputs))),
            *(outputs + [0xFFFFFFFF] * (3 - len(outputs))), 0,
        )
    slot_data = b"".join(
        TENSOR_PLAN_SLOT.pack(
            int(slot["id"]), 0, int(slot["bytes_per_runtime_row"])
        )
        for slot in plan["scratch_slots"]
    )
    data = header
    data += b"\0" * (tensor_offset - len(data)) + tensor_data
    data += b"\0" * (command_offset - len(data)) + command_data
    data += b"\0" * (slot_offset - len(data)) + slot_data
    if len(data) != total_size:
        raise ValueError("tensor plan binary layout mismatch")
    value = checksum(data, 40)
    data = data[:40] + struct.pack("<I", value) + data[44:]
    path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = load_object(args.manifest)
    executable = build_executable(manifest, load_object(args.plan))
    args.output.write_text(json.dumps(executable, indent=2, sort_keys=True) + "\n")
    tensor_plan = build_tensor_plan(executable, manifest)
    tensor_plan_output = args.output.with_suffix(".npxt")
    tensor_plan_output.write_text(json.dumps(tensor_plan, indent=2, sort_keys=True) + "\n")
    tensor_plan_binary = args.output.with_suffix(".npxtb")
    write_tensor_plan_binary(tensor_plan, tensor_plan_binary)
    binary_output = args.output.with_suffix(".npxc")
    write_binary(executable, binary_output)
    print(f"npu_executable={args.output}")
    print(f"npu_command_template={binary_output}")
    print(f"npu_tensor_plan={tensor_plan_output}")
    print(f"npu_tensor_plan_binary={tensor_plan_binary}")
    print(f"npu_tensor_plan_binary_bytes={tensor_plan_binary.stat().st_size}")
    print(f"npu_tensor_plan_tensors={tensor_plan['tensor_count']}")
    print(f"npu_tensor_plan_scratch_slots={tensor_plan['scratch_slot_count']}")
    print(
        "npu_tensor_plan_scratch_bytes_per_runtime_row="
        f"{tensor_plan['scratch_bytes_per_runtime_row']}"
    )
    print(f"npu_executable_commands={len(executable['commands'])}")
    print(f"npu_command_template_bytes={binary_output.stat().st_size}")
    binding_offset = align(INVOCATION_HEADER_SIZE)
    command_offset = align(binding_offset + 5 * TENSOR_BINDING_SIZE)
    invocation_bytes = align(
        command_offset + len(executable["commands"]) * INVOCATION_COMMAND_SIZE
    ) + len(executable["commands"]) * OPERATOR_PARAMETERS.size
    invocation_bytes = align(invocation_bytes)
    print(f"npu_invocation_bytes_upper_bound={invocation_bytes}")
    print(
        "npu_executable_capabilities="
        + ",".join(executable["required_capabilities"])
    )
    print("npu_executable=PASS")


if __name__ == "__main__":
    main()
