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
from opennpux_tvm_byoc.module_runtime import (  # noqa: E402
    CoralCtlExecutor,
    ModuleRuntime,
)


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

    def test_runtime_resolves_host_tensor_between_npu_regions(self):
        module = self.load_fixture()
        module["regions"].reverse()
        module["edges"] = []
        artifacts, manifest = compile_module(module)
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

            def execute(name, _artifact, arena, metadata):
                tensors = {tensor["name"]: tensor for tensor in metadata["tensors"]}
                source_name = "lhs" if name == "residual" else "input"
                source = struct.unpack_from("<8f", arena, tensors[source_name]["offset"])
                if name == "residual":
                    rhs = struct.unpack_from("<8f", arena, tensors["rhs"]["offset"])
                    result = [a + b for a, b in zip(source, rhs)]
                    output_name = "sum"
                else:
                    result = [value / (1.0 + math.exp(-value)) for value in source]
                    output_name = "output"
                struct.pack_into("<8f", arena, tensors[output_name]["offset"], *result)

            resolved = []

            def resolve(region, tensor, available):
                self.assertEqual((region, tensor), ("activation", "input"))
                values = struct.unpack("<8f", available["residual.sum"])
                resolved.append((region, tensor))
                return struct.pack("<8f", *(max(value, 0.0) for value in values))

            runtime = ModuleRuntime(directory, execute, resolve)
            runtime.bind("residual", "lhs", struct.pack("<8f", *range(-4, 4)))
            runtime.bind("residual", "rhs", struct.pack("<8f", *([1.0] * 8)))
            outputs = runtime.run()
            actual = struct.unpack("<8f", outputs["activation.output"])
            expected_input = [max(float(value + 1), 0.0) for value in range(-4, 4)]
            expected = [value / (1.0 + math.exp(-value)) for value in expected_input]
            self.assertEqual(resolved, [("activation", "input")])
            for lhs, rhs in zip(actual, expected):
                self.assertAlmostEqual(lhs, rhs, places=6)

    def test_coralctl_executor_chains_verified_region_outputs(self):
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
            fake = directory / "coralctl"
            fake.write_text(
                """#!/usr/bin/env python3
import math, os, struct, sys
graph = open(sys.argv[2], 'rb').read()
arena = bytearray(open(sys.argv[3], 'rb').read())
command = struct.unpack_from('<16I', graph, 96)
opcode, destination, source0, source1 = command[0], command[2], command[3], command[4]
count = command[5] * command[6]
lhs = struct.unpack_from(f'<{count}f', arena, source0)
if opcode == 2:
    rhs = struct.unpack_from(f'<{count}f', arena, source1)
    result = [a + b for a, b in zip(lhs, rhs)]
elif opcode == 7:
    result = [value / (1.0 + math.exp(-value)) for value in lhs]
else:
    raise SystemExit(2)
open(os.environ['OPENNPUX_XGRAPH_OUTPUT_PATH'], 'wb').write(struct.pack(f'<{count}f', *result))
print('xgraph_output_readback=PASS')
""",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            executor = CoralCtlExecutor(fake)
            runtime = ModuleRuntime(directory, executor)
            runtime.bind("residual", "lhs", struct.pack("<8f", *range(8)))
            runtime.bind("residual", "rhs", struct.pack("<8f", *([1.0] * 8)))
            outputs = runtime.run()
            actual = struct.unpack("<8f", outputs["activation.output"])
            expected = [value / (1.0 + math.exp(-value)) for value in range(1, 9)]
            self.assertEqual(len(executor.logs), 2)
            self.assertTrue(all("xgraph_output_readback=PASS" in log for log in executor.logs))
            for lhs, rhs in zip(actual, expected):
                self.assertAlmostEqual(lhs, rhs, places=6)


if __name__ == "__main__":
    unittest.main()
