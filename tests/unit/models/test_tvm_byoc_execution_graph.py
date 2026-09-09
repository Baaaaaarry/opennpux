import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/models"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qwen_graph():
    compiler = load_module(
        ROOT / "tools/models/compile_npu_executable.py", "npu_executable"
    )
    manifest = {
        "layer_count": 40, "hidden_size": 2048, "head_count": 16,
        "kv_head_count": 2, "head_dim": 256, "expert_count": 256,
        "experts_per_token": 8, "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512, "vocab_size": 248320,
        "quantization_bits": 4, "quantization_group_size": 128,
        "architecture": "Qwen3_5MoeForConditionalGeneration",
    }
    full = [
        "attention_norm", "qkv_projection", "rope", "paged_kv_cache_update",
        "scaled_dot_product_attention", "attention_output_projection",
        "residual_add", "ffn_norm", "router_topk",
        "routed_experts_active_only", "shared_expert", "moe_combine",
        "residual_add",
    ]
    linear = [
        "attention_norm", "linear_attention_projection", "causal_depthwise_conv",
        "recurrent_state_update", "linear_attention_gate_norm",
        "linear_attention_output_projection", "residual_add", "ffn_norm",
        "router_topk", "routed_experts_active_only", "shared_expert",
        "moe_combine", "residual_add",
    ]
    plan = {
        "format": "OPENNPUX_QWEN_EXECUTION_PLAN_V1",
        "model_manifest": "model.npxm",
        "architecture": manifest["architecture"],
        "observed_layer_count": 40,
        "unknown_decoder_tensor_patterns": {},
        "layers": [
            {"index": index,
             "type": "full_attention_moe" if index % 4 == 3 else "linear_attention_moe",
             "phases": full if index % 4 == 3 else linear}
            for index in range(40)
        ],
    }
    executable = compiler.build_executable(manifest, plan)
    tensor_plan = compiler.build_tensor_plan(executable, manifest)
    return executable, tensor_plan


class TvmByocExecutionGraphTest(unittest.TestCase):
    def test_tvm_024_tirx_namespace_is_supported(self):
        from opennpux_tvm_byoc.execution_graph import _tvm_modules

        fake_tvm = types.ModuleType("tvm")
        fake_tvm.__version__ = "0.24.0"
        fake_tvm.relax = object()
        fake_tvm.tirx = types.SimpleNamespace(Var=object())
        with mock.patch.dict(sys.modules, {"tvm": fake_tvm}):
            tvm, relax, tir_module = _tvm_modules()
        self.assertIs(tvm, fake_tvm)
        self.assertIs(relax, fake_tvm.relax)
        self.assertIs(tir_module, fake_tvm.tirx)

    def test_real_qwen_shape_forms_closed_relocatable_byoc_graph(self):
        from opennpux_tvm_byoc.execution_graph import build_execution_graph

        executable, tensor_plan = qwen_graph()
        graph = build_execution_graph(executable, tensor_plan)
        self.assertEqual(graph["adapter"], "apache-tvm-relax-byoc")
        self.assertEqual(graph["node_count"], 524)
        self.assertEqual(graph["tensor_count"], 725)
        self.assertEqual(len(graph["outputs"]), 1)
        self.assertEqual({node["operation"] for node in graph["nodes"]}, {
            "EMBED", "MATMUL", "ADD", "NORMALIZE", "ROPE", "TOPK",
            "CAUSAL_CONVOLUTION", "RECURRENT_UPDATE", "ROUTER", "EXPERT",
            "DMA", "ATTENTION", "COMBINE",
        })

    def test_execution_graph_rejects_stale_source(self):
        from opennpux_tvm_byoc.execution_graph import (
            build_execution_graph, validate_execution_graph,
        )
        from opennpux_tvm_byoc.xgraph_codegen import CodegenError

        executable, tensor_plan = qwen_graph()
        graph = build_execution_graph(executable, tensor_plan)
        stale = json.loads(json.dumps(graph))
        stale["nodes"][0]["opcode"] = 99
        with self.assertRaisesRegex(CodegenError, "stale"):
            validate_execution_graph(stale, executable, tensor_plan)

    def test_cli_emits_auditable_full_graph_without_tvm(self):
        import subprocess

        executable, tensor_plan = qwen_graph()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            executable_path = directory / "model.npxe"
            tensor_plan_path = directory / "model.npxt"
            output = directory / "model.npxtvm"
            command_template = directory / "model.tvm.npxc"
            tensor_plan_binary = directory / "model.tvm.npxtb"
            executable_path.write_text(json.dumps(executable), encoding="utf-8")
            tensor_plan_path.write_text(json.dumps(tensor_plan), encoding="utf-8")
            result = subprocess.run([
                sys.executable,
                str(ROOT / "tools/models/compile_tvm_byoc_execution_graph.py"),
                str(executable_path), str(tensor_plan_path), str(output),
                "--require-node-count", "524",
                "--command-template-output", str(command_template),
                "--tensor-plan-binary-output", str(tensor_plan_binary),
            ], check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("tvm_byoc_execution_graph=PASS", result.stdout)
            graph = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(graph["node_count"], 524)
            self.assertEqual(len(graph["source_digests"]["executable_sha256"]), 64)
            self.assertGreater(command_template.stat().st_size, 0)
            self.assertGreater(tensor_plan_binary.stat().st_size, 0)

    def test_require_tvm_requires_relax_output(self):
        import subprocess

        executable, tensor_plan = qwen_graph()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            executable_path = directory / "model.npxe"
            tensor_plan_path = directory / "model.npxt"
            executable_path.write_text(json.dumps(executable), encoding="utf-8")
            tensor_plan_path.write_text(json.dumps(tensor_plan), encoding="utf-8")
            result = subprocess.run([
                sys.executable,
                str(ROOT / "tools/models/compile_tvm_byoc_execution_graph.py"),
                str(executable_path), str(tensor_plan_path),
                str(directory / "model.npxtvm"), "--require-tvm",
            ], check=False, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires --relax-output", result.stderr)


if __name__ == "__main__":
    unittest.main()
