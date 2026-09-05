import copy
import json
import math
import struct
import subprocess
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
    HostPipelineExecutor,
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

    def test_runtime_executes_manifest_host_binding(self):
        module = self.load_fixture()
        module["regions"].reverse()
        module["edges"] = []
        module["host_bindings"] = [
            {
                "from": {"region": "residual", "tensor": "sum"},
                "to": {"region": "activation", "tensor": "input"},
                "pipeline": [{"op": "relax.nn.relu", "attrs": {}}],
            }
        ]
        artifacts, manifest = compile_module(module)
        self.assertEqual(manifest["execution_order"], ["residual", "activation"])
        self.assertEqual(manifest["regions"][1]["external_bindings"], [])
        self.assertEqual(manifest["host_bindings"][0]["shape"], [2, 4])
        self.assertEqual(manifest["host_bindings"][0]["dtype"], "float32")
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

            host = HostPipelineExecutor()
            runtime = ModuleRuntime(directory, execute, host_executor=host)
            runtime.bind("residual", "lhs", struct.pack("<8f", *range(-4, 4)))
            runtime.bind("residual", "rhs", struct.pack("<8f", *([1.0] * 8)))
            outputs = runtime.run()
            actual = struct.unpack("<8f", outputs["activation.output"])
            expected_input = [max(float(value + 1), 0.0) for value in range(-4, 4)]
            expected = [value / (1.0 + math.exp(-value)) for value in expected_input]
            for lhs, rhs in zip(actual, expected):
                self.assertAlmostEqual(lhs, rhs, places=6)
            self.assertEqual(host.completed_bindings, 1)
            self.assertEqual(host.completed_operations, 1)
            self.assertEqual(host.completed_elements, 8)

    def test_host_pipeline_rejects_unsupported_operation(self):
        executor = HostPipelineExecutor()
        binding = {
            "dtype": "float32",
            "pipeline": [{"op": "relax.nn.gelu", "attrs": {}}],
        }
        with self.assertRaisesRegex(CodegenError, "unsupported Host pipeline"):
            executor(binding, struct.pack("<f", 1.0))

    def test_runtime_rejects_missing_host_executor(self):
        module = self.load_fixture()
        module["regions"].reverse()
        module["edges"] = []
        module["host_bindings"] = [
            {
                "from": {"region": "residual", "tensor": "sum"},
                "to": {"region": "activation", "tensor": "input"},
                "pipeline": [{"op": "relax.nn.relu", "attrs": {}}],
            }
        ]
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
            runtime = ModuleRuntime(directory, lambda *_: None)
            runtime.bind("residual", "lhs", bytes(32))
            runtime.bind("residual", "rhs", bytes(32))
            with self.assertRaisesRegex(CodegenError, "requires a Host executor"):
                runtime.run()

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

    def test_module_runner_executes_manifest_host_pipeline(self):
        module = self.load_fixture()
        module["regions"].reverse()
        module["edges"] = []
        module["host_bindings"] = [
            {
                "from": {"region": "residual", "tensor": "sum"},
                "to": {"region": "activation", "tensor": "input"},
                "pipeline": [{"op": "relax.nn.relu", "attrs": {}}],
            }
        ]
        artifacts, manifest = compile_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            module_dir = directory / "module"
            output_dir = directory / "outputs"
            module_dir.mkdir()
            (module_dir / "module.npxgm.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            for region in manifest["regions"]:
                binary, metadata = artifacts[region["name"]]
                path = module_dir / region["artifact"]
                path.write_bytes(binary)
                Path(f"{path}.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
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
            lhs_path = directory / "lhs.bin"
            rhs_path = directory / "rhs.bin"
            lhs_path.write_bytes(struct.pack("<8f", *range(-4, 4)))
            rhs_path.write_bytes(struct.pack("<8f", *([1.0] * 8)))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/models/run_tvm_byoc_module.py"),
                    str(module_dir),
                    "--coralctl",
                    str(fake),
                    "--bind",
                    f"residual.lhs={lhs_path}",
                    "--bind",
                    f"residual.rhs={rhs_path}",
                    "--output-dir",
                    str(output_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("xgraph_module_host_bindings_completed=1", completed.stdout)
            self.assertIn("xgraph_module_run=PASS", completed.stdout)
            actual = struct.unpack(
                "<8f", (output_dir / "activation.output.bin").read_bytes()
            )
            expected_input = [max(float(value + 1), 0.0) for value in range(-4, 4)]
            expected = [value / (1.0 + math.exp(-value)) for value in expected_input]
            for lhs, rhs in zip(actual, expected):
                self.assertAlmostEqual(lhs, rhs, places=6)

    def test_builds_static_guest_module_package(self):
        module = self.load_fixture()
        module["regions"].reverse()
        module["edges"] = []
        module["host_bindings"] = [
            {
                "from": {"region": "residual", "tensor": "sum"},
                "to": {"region": "activation", "tensor": "input"},
                "pipeline": [{"op": "relax.nn.relu", "attrs": {}}],
            }
        ]
        artifacts, manifest = compile_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            module_dir = directory / "module"
            module_dir.mkdir()
            (module_dir / "module.npxgm.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            command = [
                sys.executable,
                str(ROOT / "tools/models/build_tvm_byoc_module_package.py"),
                str(module_dir),
                str(directory / "module.npxgm"),
                "--clear-external-bindings",
            ]
            arena_arguments = []
            for region in manifest["regions"]:
                binary, metadata = artifacts[region["name"]]
                artifact_path = module_dir / region["artifact"]
                artifact_path.write_bytes(binary)
                Path(f"{artifact_path}.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                arena_path = directory / f"{region['name']}.arena.bin"
                arena_data = bytes([region["sequence"] + 1]) * region["arena_size"]
                arena_path.write_bytes(arena_data)
                arena_arguments.extend(
                    ["--arena", f"{region['name']}={arena_path}"]
                )
            command.extend(arena_arguments)
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("xgraph_module_package=PASS", completed.stdout)
            package = (directory / "module.npxgm").read_bytes()
            header = struct.unpack_from("<16I", package)
            self.assertEqual(header[0], 0x4D47584E)
            self.assertEqual(header[1], 1)
            self.assertEqual(header[3], len(package))
            self.assertEqual(header[4], 2)
            self.assertEqual(header[5], 0)
            self.assertEqual(header[6], 1)
            self.assertEqual(header[7], 1)
            self.assertEqual(header[8], 1)
            self.assertEqual(header[9:14], (32, 24, 28, 8, 16))
            self.assertEqual(header[14] % 64, 0)
            first_region = struct.unpack_from("<8I", package, 64)
            first_arena_offset = first_region[2]
            first_metadata = artifacts[manifest["regions"][0]["name"]][1]
            first_tensors = {
                tensor["name"]: tensor for tensor in first_metadata["tensors"]
            }
            for tensor_name in manifest["regions"][0]["external_bindings"]:
                tensor = first_tensors[tensor_name]
                begin = first_arena_offset + tensor["offset"]
                end = begin + tensor["byte_size"]
                self.assertEqual(package[begin:end], bytes(tensor["byte_size"]))

            invocation_path = directory / "invocation.npxmi"
            invocation = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/models/build_tvm_byoc_invocation.py"),
                    str(module_dir),
                    str(invocation_path),
                    *arena_arguments,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(invocation.returncode, 0, invocation.stderr)
            self.assertIn("xgraph_invocation_bindings=2", invocation.stdout)
            invocation_image = invocation_path.read_bytes()
            invocation_header = struct.unpack_from("<8I", invocation_image)
            self.assertEqual(invocation_header[0], 0x4958504E)
            self.assertEqual(invocation_header[1], 1)
            self.assertEqual(invocation_header[3], len(invocation_image))
            self.assertEqual(invocation_header[4], 2)
            self.assertEqual(invocation_header[5], 24)


if __name__ == "__main__":
    unittest.main()
