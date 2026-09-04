import copy
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/models"))

from opennpux_tvm_byoc import CodegenError  # noqa: E402
from opennpux_tvm_byoc.module_codegen import compile_module  # noqa: E402
from opennpux_tvm_byoc.module_runtime import ModuleRuntime  # noqa: E402


class XGraphModuleCodegenTest(unittest.TestCase):
    def load_fixture(self):
        return json.loads(
            (ROOT / "tests/fixtures/models/tvm_byoc_module.json").read_text(
                encoding="utf-8"
            )
        )

    def test_compiles_regions_in_dependency_order(self):
        artifacts, manifest = compile_module(self.load_fixture())
        self.assertEqual(manifest["execution_order"], ["residual", "activation"])
        self.assertEqual(manifest["region_count"], 2)
        self.assertEqual(manifest["total_commands"], 2)
        self.assertEqual(set(artifacts), {"residual", "activation"})
        self.assertEqual(manifest["regions"][0]["external_inputs"], ["lhs", "rhs"])
        self.assertEqual(manifest["regions"][1]["external_inputs"], [])
        self.assertEqual(manifest["edges"][0]["bytes"], 32)

    def test_rejects_incompatible_edge_types(self):
        module = self.load_fixture()
        module["regions"][0]["graph"]["tensors"][0]["shape"] = [1, 4]
        with self.assertRaisesRegex(CodegenError, "tensor type mismatch"):
            compile_module(module)

    def test_rejects_multiple_producers(self):
        module = self.load_fixture()
        module["edges"].append(copy.deepcopy(module["edges"][0]))
        with self.assertRaisesRegex(CodegenError, "multiple producers"):
            compile_module(module)

    def test_rejects_region_cycles(self):
        module = self.load_fixture()
        activation_graph = module["regions"][0]["graph"]
        activation_graph["tensors"].append(
            {"name": "feedback", "shape": [2, 4], "dtype": "float32", "storage": "input"}
        )
        module["edges"].append(
            {
                "from": {"region": "activation", "tensor": "output"},
                "to": {"region": "residual", "tensor": "lhs"},
            }
        )
        with self.assertRaisesRegex(CodegenError, "contains a cycle"):
            compile_module(module)

    def test_runtime_binds_edges_and_executes_in_order(self):
        artifacts, manifest = compile_module(self.load_fixture())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "module.npxgm.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            for region in manifest["regions"]:
                binary, metadata = artifacts[region["name"]]
                path = directory / region["artifact"]
                path.write_bytes(binary)
                Path(f"{path}.json").write_text(json.dumps(metadata), encoding="utf-8")

            executed = []

            def execute(name, _artifact, arena, metadata):
                tensors = {tensor["name"]: tensor for tensor in metadata["tensors"]}

                def values(tensor_name):
                    tensor = tensors[tensor_name]
                    return struct.unpack_from(
                        "<8f", arena, tensor["offset"]
                    )

                if name == "residual":
                    result = [a + b for a, b in zip(values("lhs"), values("rhs"))]
                    struct.pack_into("<8f", arena, tensors["sum"]["offset"], *result)
                else:
                    result = [value / (1.0 + math.exp(-value)) for value in values("input")]
                    struct.pack_into(
                        "<8f", arena, tensors["output"]["offset"], *result
                    )
                executed.append(name)

            runtime = ModuleRuntime(directory, execute)
            runtime.bind("residual", "lhs", struct.pack("<8f", *range(8)))
            runtime.bind("residual", "rhs", struct.pack("<8f", *([1.0] * 8)))
            outputs = runtime.run()
            actual = struct.unpack("<8f", outputs["activation.output"])
            expected = [value / (1.0 + math.exp(-value)) for value in range(1, 9)]
            self.assertEqual(executed, ["residual", "activation"])
            for lhs, rhs in zip(actual, expected):
                self.assertAlmostEqual(lhs, rhs, places=6)

    def test_runtime_rejects_missing_external_binding(self):
        artifacts, manifest = compile_module(self.load_fixture())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "module.npxgm.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            for region in manifest["regions"]:
                binary, metadata = artifacts[region["name"]]
                path = directory / region["artifact"]
                path.write_bytes(binary)
                Path(f"{path}.json").write_text(json.dumps(metadata), encoding="utf-8")
            runtime = ModuleRuntime(directory, lambda *_: None)
            with self.assertRaisesRegex(CodegenError, "unbound external Tensors"):
                runtime.run()


if __name__ == "__main__":
    unittest.main()
