import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/models"))

from opennpux_backend import compile_frontend  # noqa: E402
from opennpux_backend.compiler import compile_graph, compile_module  # noqa: E402
from opennpux_mojo import MAX_GRAPH_EXPORT_FORMAT, MaxGraphAdapter  # noqa: E402


def max_export(graph):
    operations = []
    names = {
        "relax.matmul": "max.matmul",
        "relax.add": "max.add",
        "relax.nn.silu": "max.silu",
        "relax.nn.softmax": "max.softmax",
        "relax.topk": "max.topk",
    }
    for index, node in enumerate(graph["nodes"]):
        operation = copy.deepcopy(node)
        operation["name"] = f"operation_{index}"
        operation["op"] = names[operation["op"]]
        operations.append(operation)
    return {
        "format": MAX_GRAPH_EXPORT_FORMAT,
        "kind": "graph",
        "tensors": copy.deepcopy(graph["tensors"]),
        "operations": operations,
        "outputs": copy.deepcopy(graph["outputs"]),
    }


class OpenNPUXMojoAdapterTest(unittest.TestCase):
    def setUp(self):
        fixture = ROOT / "tests/fixtures/models/tvm_byoc_basic.json"
        self.graph = json.loads(fixture.read_text(encoding="utf-8"))

    def test_max_export_matches_tvm_xgraph(self):
        tvm_artifact, tvm_metadata = compile_graph(self.graph)
        mojo = compile_frontend(max_export(self.graph), MaxGraphAdapter())
        self.assertEqual(mojo["artifact"], tvm_artifact)
        self.assertEqual(mojo["metadata"]["commands"], tvm_metadata["commands"])
        self.assertEqual(mojo["metadata"]["frontend_adapter"], "mojo-max")

    def test_adapter_accepts_sdk_independent_export_protocol(self):
        export = max_export(self.graph)

        class MaxGraph:
            def to_opennpux_export(self):
                return export

        direct = compile_frontend(export, MaxGraphAdapter())
        object_result = compile_frontend(MaxGraph(), MaxGraphAdapter())
        self.assertEqual(direct["artifact"], object_result["artifact"])

    def test_symbolic_max_export_uses_common_specializer(self):
        export = max_export(self.graph)
        export["shape_symbols"] = {"batch": {"min": 1, "max": 8}}
        for tensor in export["tensors"]:
            if tensor["shape"][0] == 2:
                tensor["shape"][0] = "batch"
        symbolic = compile_frontend(
            export, MaxGraphAdapter(), shape_bindings={"batch": 2}
        )
        static = compile_frontend(max_export(self.graph), MaxGraphAdapter())
        self.assertEqual(symbolic["artifact"], static["artifact"])

    def test_unknown_max_operation_is_rejected(self):
        export = max_export(self.graph)
        export["operations"][0]["op"] = "max.model_specific_qwen_matmul"
        with self.assertRaisesRegex(ValueError, "unsupported MAX operation"):
            MaxGraphAdapter().to_backend_ir(export)

    def test_max_module_reuses_common_region_and_module_codegen(self):
        fixture = ROOT / "tests/fixtures/models/tvm_byoc_module.json"
        module = json.loads(fixture.read_text(encoding="utf-8"))
        export = copy.deepcopy(module)
        export["format"] = MAX_GRAPH_EXPORT_FORMAT
        export["kind"] = "module"
        reverse_names = {
            "add": "max.add",
            "relax.add": "max.add",
            "silu": "max.silu",
            "relax.nn.silu": "max.silu",
        }
        for region in export["regions"]:
            graph = region["graph"]
            graph["operations"] = graph.pop("nodes")
            for index, operation in enumerate(graph["operations"]):
                operation["name"] = f"{region['name']}_{index}"
                operation["op"] = reverse_names[operation["op"]]
        expected_artifacts, expected_manifest = compile_module(module)
        compiled = compile_frontend(export, MaxGraphAdapter())
        self.assertEqual(compiled["artifacts"], expected_artifacts)
        self.assertEqual(
            compiled["manifest"]["total_commands"],
            expected_manifest["total_commands"],
        )


if __name__ == "__main__":
    unittest.main()
