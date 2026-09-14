import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/models"))

from opennpux_backend import compile_frontend  # noqa: E402
from opennpux_backend.compiler import compile_graph, compile_module  # noqa: E402
from opennpux_mojo import (  # noqa: E402
    MAX_GRAPH_EXPORT_FORMAT,
    MaxExportBuilder,
    MaxGraphAdapter,
)


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

    def test_export_builder_records_public_max_values_and_calls(self):
        class Value:
            def __init__(self, shape, dtype="DType.float32"):
                self.shape = shape
                self.dtype = dtype

        lhs = Value([2, 4])
        rhs = Value([4, 3])
        output = Value([2, 3])
        builder = MaxExportBuilder("projection")
        builder.add_tensor("lhs", lhs, storage="input")
        builder.add_tensor("rhs", rhs, storage="constant")
        result = builder.call(
            "max.matmul",
            lambda left, right: output,
            [lhs, rhs],
            ["projected"],
            name="projection",
        )
        builder.add_output(result)

        compiled = compile_frontend(builder, MaxGraphAdapter())
        expected = {
            "format": "OPENNPUX_BACKEND_GRAPH_V1",
            "tensors": [
                {"name": "lhs", "shape": [2, 4], "dtype": "float32", "storage": "input"},
                {"name": "rhs", "shape": [4, 3], "dtype": "float32", "storage": "constant"},
                {"name": "projected", "shape": [2, 3], "dtype": "float32", "storage": "output"},
            ],
            "nodes": [
                {"op": "matmul", "inputs": ["lhs", "rhs"], "outputs": ["projected"]}
            ],
            "outputs": ["projected"],
        }
        expected_artifact, _ = compile_graph(expected)
        self.assertEqual(compiled["artifact"], expected_artifact)

    def test_export_builder_requires_bounds_for_max_symbolic_dims(self):
        class Value:
            shape = ["sequence", 4]
            dtype = "float32"

        builder = MaxExportBuilder()
        with self.assertRaisesRegex(ValueError, "requires declared bounds"):
            builder.add_tensor("input", Value(), storage="input")
        builder.declare_symbol("sequence", 1, 4096)
        builder.add_tensor("input", Value(), storage="input")
        builder.add_output("input")
        export = builder.to_opennpux_export()
        self.assertEqual(
            export["shape_symbols"]["sequence"], {"min": 1, "max": 4096}
        )


if __name__ == "__main__":
    unittest.main()
