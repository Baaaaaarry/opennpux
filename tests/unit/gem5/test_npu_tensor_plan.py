import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load_compiler():
    path = ROOT / "tools/models/compile_npu_executable.py"
    spec = importlib.util.spec_from_file_location("compile_npu_executable", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qwen35b_shape_has_complete_model_independent_tensor_plan():
    compiler = load_compiler()
    manifest = {
        "layer_count": 40,
        "hidden_size": 2048,
        "head_count": 16,
        "kv_head_count": 2,
        "head_dim": 256,
        "expert_count": 256,
        "experts_per_token": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "vocab_size": 248320,
        "quantization_bits": 4,
        "quantization_group_size": 128,
        "architecture": "Qwen3_5MoeForConditionalGeneration",
    }
    full_attention = [
        "attention_norm", "qkv_projection", "rope", "paged_kv_cache_update",
        "scaled_dot_product_attention", "attention_output_projection",
        "residual_add", "ffn_norm", "router_topk",
        "routed_experts_active_only", "shared_expert", "moe_combine",
        "residual_add",
    ]
    linear_attention = [
        "attention_norm", "linear_attention_projection", "causal_depthwise_conv",
        "recurrent_state_update", "linear_attention_gate_norm",
        "linear_attention_output_projection", "residual_add", "ffn_norm",
        "router_topk", "routed_experts_active_only", "shared_expert",
        "moe_combine", "residual_add",
    ]
    layers = []
    for index in range(40):
        layer_type = "full_attention_moe" if index % 4 == 3 else "linear_attention_moe"
        layers.append({
            "index": index,
            "type": layer_type,
            "phases": full_attention if index % 4 == 3 else linear_attention,
        })
    frontend_plan = {
        "format": "OPENNPUX_QWEN_EXECUTION_PLAN_V1",
        "model_manifest": "model.npxm",
        "architecture": "Qwen3_5MoeForConditionalGeneration",
        "observed_layer_count": 40,
        "unknown_decoder_tensor_patterns": {},
        "layers": layers,
    }

    executable = compiler.build_executable(manifest, frontend_plan)
    tensor_plan = compiler.build_tensor_plan(executable, manifest)

    assert len(executable["commands"]) == 524
    assert tensor_plan["command_count"] == 524
    assert tensor_plan["tensor_count"] == 695
    assert tensor_plan["scratch_slot_count"] == 7
    gated_qkv = [record for record in tensor_plan["command_io"]
                 if len(record["output_tensor_ids"]) == 4]
    assert len(gated_qkv) == 10
    assert all(record["input_tensor_ids"] for record in tensor_plan["command_io"])
    assert all(record["output_tensor_ids"] for record in tensor_plan["command_io"])


def test_tensor_plan_preserves_attention_and_residual_dependencies():
    compiler = load_compiler()
    manifest = {
        "layer_count": 1,
        "hidden_size": 8,
        "head_count": 2,
        "kv_head_count": 1,
        "head_dim": 4,
        "expert_count": 4,
        "experts_per_token": 2,
        "moe_intermediate_size": 12,
        "shared_expert_intermediate_size": 12,
        "vocab_size": 16,
    }
    phases = [
        "attention_norm", "qkv_projection", "rope", "paged_kv_cache_update",
        "scaled_dot_product_attention", "attention_output_projection",
        "residual_add", "ffn_norm", "router_topk",
        "routed_experts_active_only", "shared_expert", "moe_combine",
        "residual_add",
    ]
    frontend_plan = {
        "format": "OPENNPUX_QWEN_EXECUTION_PLAN_V1",
        "model_manifest": "model.npxm",
        "architecture": "test",
        "observed_layer_count": 1,
        "unknown_decoder_tensor_patterns": {},
        "layers": [{"index": 0, "type": "full_attention_moe", "phases": phases}],
    }
    executable = compiler.build_executable(manifest, frontend_plan)
    tensor_plan = compiler.build_tensor_plan(executable, manifest)
    tensors = {tensor["name"]: tensor for tensor in tensor_plan["tensors"]}
    io = {record["command_id"]: record for record in tensor_plan["command_io"]}

    attention_command = next(
        command for command in executable["commands"]
        if command["attributes"]["phase"] == "scaled_dot_product_attention"
    )
    assert tensors["layer.0.kv_cache"]["id"] in io[attention_command["command_id"]][
        "input_tensor_ids"
    ]
    kv_command = next(
        command for command in executable["commands"]
        if command["attributes"]["phase"] == "paged_kv_cache_update"
    )
    assert tensors["layer.0.key_rope"]["id"] in io[kv_command["command_id"]][
        "input_tensor_ids"
    ]
    assert tensors["layer.0.value"]["id"] in io[kv_command["command_id"]][
        "input_tensor_ids"
    ]
    residuals = [
        command for command in executable["commands"]
        if command["attributes"]["phase"] == "residual_add"
    ]
    assert io[residuals[0]["command_id"]]["input_tensor_ids"] != io[
        residuals[1]["command_id"]
    ]["input_tensor_ids"]
