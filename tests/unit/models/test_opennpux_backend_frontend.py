import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/models"))

from opennpux_backend import (  # noqa: E402
    FrontendAdapter,
    GRAPH_FORMAT,
    compile_frontend,
    capability_manifest,
    supports_operation_contract,
    write_graph_artifact,
)
from opennpux_backend.compiler import CodegenError  # noqa: E402


class LegacyTvmAdapter:
    name = "tvm-relax-test"

    def to_backend_ir(self, source):
        return copy.deepcopy(source)


class FakeMojoAdapter:
    name = "mojo-max-test"

    _operations = {
        "max.matmul": "matmul",
        "max.add": "add",
        "max.silu": "silu",
        "max.softmax": "softmax",
        "max.topk": "topk",
    }

    def to_backend_ir(self, source):
        graph = copy.deepcopy(source)
        graph["format"] = GRAPH_FORMAT
        for node in graph["nodes"]:
            node["op"] = self._operations[node["op"]]
        return graph


class OpenNPUXFrontendContractTest(unittest.TestCase):
    def load_fixture(self):
        path = ROOT / "tests/fixtures/models/tvm_byoc_basic.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_tvm_and_mojo_adapters_share_byte_identical_backend(self):
        tvm_graph = self.load_fixture()
        mojo_graph = copy.deepcopy(tvm_graph)
        replacements = {
            "relax.matmul": "max.matmul",
            "relax.add": "max.add",
            "relax.nn.silu": "max.silu",
            "relax.nn.softmax": "max.softmax",
            "relax.topk": "max.topk",
        }
        for node in mojo_graph["nodes"]:
            node["op"] = replacements[node["op"]]

        tvm = compile_frontend(tvm_graph, LegacyTvmAdapter())
        mojo = compile_frontend(mojo_graph, FakeMojoAdapter())

        self.assertEqual(tvm["kind"], "graph")
        self.assertEqual(tvm["artifact"], mojo["artifact"])
        self.assertEqual(
            tvm["metadata"]["commands"], mojo["metadata"]["commands"]
        )
        self.assertEqual(tvm["backend_ir"], mojo["backend_ir"])
        self.assertEqual(tvm["metadata"]["frontend_adapter"], "tvm-relax-test")
        self.assertEqual(mojo["metadata"]["frontend_adapter"], "mojo-max-test")

    def test_backend_import_does_not_load_frontend_packages(self):
        script = """
import sys
import opennpux_backend
assert "tvm" not in sys.modules
assert not any(name.startswith("opennpux_tvm_byoc") for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "tools/models")},
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_adapter_identity_is_required(self):
        class InvalidAdapter:
            def to_backend_ir(self, source):
                return source

        with self.assertRaisesRegex(CodegenError, "non-empty name"):
            compile_frontend(self.load_fixture(), InvalidAdapter())

    def test_tvm_adapter_implements_common_protocol_without_loading_tvm(self):
        from opennpux_tvm_byoc.relax_backend import RelaxFrontendAdapter

        adapter = RelaxFrontendAdapter(module=False, partitioned=True)
        self.assertIsInstance(adapter, FrontendAdapter)
        self.assertEqual(adapter.name, "tvm-relax-byoc")
        self.assertNotIn("tvm", sys.modules)

    def test_common_artifact_writer_is_frontend_independent(self):
        compilation = compile_frontend(self.load_fixture(), LegacyTvmAdapter())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.npxg"
            sidecar = write_graph_artifact(
                output, compilation["artifact"], compilation["metadata"]
            )
            self.assertEqual(output.read_bytes(), compilation["artifact"])
            self.assertEqual(
                json.loads(sidecar.read_text(encoding="utf-8")),
                compilation["metadata"],
            )

    def test_capability_contract_is_frontend_neutral_and_shape_aware(self):
        tensor = lambda shape, dtype="float32": {"shape": shape, "dtype": dtype}
        self.assertTrue(supports_operation_contract(
            "relax.matmul",
            [tensor([2, 4]), tensor([4, 8])],
            [tensor([2, 8])],
        ))
        self.assertFalse(supports_operation_contract(
            "matmul",
            [tensor([2, 4]), tensor([3, 8])],
            [tensor([2, 8])],
        ))
        manifest = capability_manifest()
        self.assertEqual(manifest["operations"]["matmul"]["arity"], [2, 1])
        self.assertEqual(
            manifest["operations"]["matmul"]["shape_relation"], "matmul"
        )


if __name__ == "__main__":
    unittest.main()
