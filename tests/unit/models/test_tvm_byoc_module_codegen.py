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
from opennpux_tvm_byoc.relax_backend import PATTERN_OPS  # noqa: E402
from opennpux_tvm_byoc.storage_policy import (  # noqa: E402
    apply_parameter_aliases,
    apply_parameter_storage,
    apply_state_updates,
)


class XGraphModuleCodegenTest(unittest.TestCase):
    def load_fixture(self):
        return json.loads(
            (ROOT / "tests/fixtures/models/tvm_byoc_module.json").read_text(
                encoding="utf-8"
            )
        )

    def test_decomposed_silu_pattern_precedes_generic_multiply(self):
        patterns = list(PATTERN_OPS)
        self.assertLess(
            patterns.index("opennpux.silu_decomposed"),
            patterns.index("opennpux.multiply"),
        )

    def test_compiles_regions_in_dependency_order(self):
        artifacts, manifest = compile_module(self.load_fixture())
        self.assertEqual(manifest["execution_order"], ["residual", "activation"])
        self.assertEqual(manifest["region_count"], 2)
        self.assertEqual(manifest["total_commands"], 2)
        self.assertEqual(set(artifacts), {"residual", "activation"})
        self.assertEqual(manifest["regions"][0]["external_inputs"], ["lhs", "rhs"])
        self.assertEqual(
            manifest["regions"][0]["invocation_bindings"], ["lhs", "rhs"]
        )
        self.assertEqual(manifest["regions"][0]["constant_bindings"], [])
        self.assertEqual(manifest["regions"][1]["external_inputs"], [])
        self.assertEqual(manifest["edges"][0]["bytes"], 32)

    def test_separates_module_constants_from_invocation_bindings(self):
        module = self.load_fixture()
        apply_parameter_storage(module, ["rhs"], [])
        _, manifest = compile_module(module)
        region = manifest["regions"][0]
        self.assertEqual(region["external_bindings"], ["lhs", "rhs"])
        self.assertEqual(region["invocation_bindings"], ["lhs"])
        self.assertEqual(region["constant_bindings"], ["rhs"])

    def test_parameter_aliases_restore_frontend_names(self):
        module = self.load_fixture()
        residual = next(
            region for region in module["regions"] if region["name"] == "residual"
        )
        residual["binding_sources"] = {
            "lhs": "lhs",
            "rhs": "utput_weight",
        }
        apply_parameter_aliases(module, {"utput_weight": "output_weight"})
        apply_parameter_storage(module, ["output_weight"], [])
        _, manifest = compile_module(module)
        region = manifest["regions"][0]
        self.assertEqual(
            region["binding_sources"]["rhs"], "output_weight"
        )
        self.assertEqual(region["constant_bindings"], ["rhs"])

    def test_storage_policy_rejects_unknown_and_conflicting_parameters(self):
        module = self.load_fixture()
        with self.assertRaisesRegex(CodegenError, "were not found"):
            apply_parameter_storage(module, ["missing"], [])
        with self.assertRaisesRegex(CodegenError, "both constant and state"):
            apply_parameter_storage(module, ["lhs"], ["lhs"])

    def test_state_update_is_persistent_and_omitted_from_invocation(self):
        module = {
            "format": "OPENNPUX_TVM_BYOC_MODULE_V1",
            "regions": [{
                "name": "decode",
                "graph": {
                    "format": "OPENNPUX_TVM_BYOC_GRAPH_V1",
                    "tensors": [
                        {"name": "token", "shape": [2], "dtype": "float32",
                         "storage": "input"},
                        {"name": "kv_state", "shape": [2], "dtype": "float32",
                         "storage": "state"},
                        {"name": "updated_state", "shape": [2],
                         "dtype": "float32", "storage": "output"},
                    ],
                    "nodes": [{"op": "add", "inputs": ["token", "kv_state"],
                               "outputs": ["updated_state"]}],
                    "outputs": ["updated_state"],
                },
            }],
            "edges": [],
            "state_updates": [{
                "from": {"region": "decode", "tensor": "updated_state"},
                "to": {"region": "decode", "tensor": "kv_state"},
            }],
        }
        artifacts, manifest = compile_module(module)
        region = manifest["regions"][0]
        self.assertEqual(region["invocation_bindings"], ["token"])
        self.assertEqual(region["state_bindings"], ["kv_state"])
        self.assertEqual(manifest["state_updates"][0]["bytes"], 8)
        self.assertEqual(manifest["module_outputs"][0]["tensor"], "updated_state")

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            module_dir = directory / "module"
            module_dir.mkdir()
            (module_dir / "module.npxgm.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            binary, metadata = artifacts["decode"]
            artifact = module_dir / region["artifact"]
            artifact.write_bytes(binary)
            Path(f"{artifact}.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            arena = bytearray(region["arena_size"])
            tensors = {tensor["name"]: tensor for tensor in metadata["tensors"]}
            state = tensors["kv_state"]
            arena[state["offset"]:state["offset"] + state["byte_size"]] = (
                struct.pack("<2f", 1.0, 2.0)
            )
            arena_path = directory / "decode.arena.bin"
            arena_path.write_bytes(arena)
            package_path = directory / "module.npxgm"
            package = subprocess.run([
                sys.executable,
                str(ROOT / "tools/models/build_tvm_byoc_module_package.py"),
                str(module_dir), str(package_path), "--clear-external-bindings",
                "--arena", f"decode={arena_path}",
            ], check=False, capture_output=True, text=True)
            self.assertEqual(package.returncode, 0, package.stderr)
            image = package_path.read_bytes()
            header = struct.unpack_from("<16I", image)
            self.assertEqual(header[5], 1)
            edge = struct.unpack_from("<6I", image, 64 + 32)
            self.assertEqual(edge[4:], (8, 1))
            arena_offset = struct.unpack_from("<8I", image, 64)[2]
            self.assertEqual(
                image[arena_offset + state["offset"]:
                      arena_offset + state["offset"] + state["byte_size"]],
                struct.pack("<2f", 1.0, 2.0),
            )
            invocation = subprocess.run([
                sys.executable,
                str(ROOT / "tools/models/build_tvm_byoc_invocation.py"),
                str(module_dir), str(directory / "request.npxmi"),
                "--arena", f"decode={arena_path}",
            ], check=False, capture_output=True, text=True)
            self.assertEqual(invocation.returncode, 0, invocation.stderr)
            self.assertIn("xgraph_invocation_bindings=1", invocation.stdout)

    def test_state_update_requires_output_to_state(self):
        module = self.load_fixture()
        module["state_updates"] = [{
            "from": {"region": "residual", "tensor": "lhs"},
            "to": {"region": "activation", "tensor": "input"},
        }]
        with self.assertRaisesRegex(CodegenError, "produced Tensor to state storage"):
            compile_module(module)

    def test_state_append_encodes_stride_and_capacity(self):
        module = json.loads(
            (ROOT / "tests/fixtures/models/tvm_byoc_state_append_module.json")
            .read_text(encoding="utf-8")
        )
        artifacts, manifest = compile_module(module)
        update = manifest["state_updates"][0]
        self.assertEqual(update["mode"], "append")
        self.assertEqual(update["bytes"], 8)
        self.assertEqual(update["stride"], 8)
        self.assertEqual(update["capacity"], 4)
        self.assertEqual(manifest["scalar_bindings"], [{
            "name": "decode_position",
            "region": "decode",
            "command": 0,
            "field": "reserved4",
            "minimum": 0,
            "maximum": 3,
        }])

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            module_dir = directory / "module"
            module_dir.mkdir()
            (module_dir / "module.npxgm.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            binary, metadata = artifacts["decode"]
            artifact = module_dir / manifest["regions"][0]["artifact"]
            artifact.write_bytes(binary)
            Path(f"{artifact}.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            arena_path = directory / "decode.arena.bin"
            arena_path.write_bytes(bytes(manifest["regions"][0]["arena_size"]))
            package_path = directory / "module.npxgm"
            package = subprocess.run([
                sys.executable,
                str(ROOT / "tools/models/build_tvm_byoc_module_package.py"),
                str(module_dir), str(package_path),
                "--arena", f"decode={arena_path}",
            ], check=False, capture_output=True, text=True)
            self.assertEqual(package.returncode, 0, package.stderr)
            image = package_path.read_bytes()
            edge = struct.unpack_from("<6I", image, 64 + 32)
            self.assertEqual(edge[4], 8)
            self.assertEqual(edge[5], 2 | (2 << 2) | (4 << 16))
            invocation_path = directory / "request.npxmi"
            invocation = subprocess.run([
                sys.executable,
                str(ROOT / "tools/models/build_tvm_byoc_invocation.py"),
                str(module_dir), str(invocation_path),
                "--arena", f"decode={arena_path}",
                "--scalar", "decode_position=2",
            ], check=False, capture_output=True, text=True)
            self.assertEqual(invocation.returncode, 0, invocation.stderr)
            invocation_image = invocation_path.read_bytes()
            header = struct.unpack_from("<8I", invocation_image)
            self.assertEqual(header[4], 2)
            scalar = struct.unpack_from("<6I", invocation_image, 32 + 24)
            self.assertEqual(scalar[0], 0)
            self.assertEqual(scalar[1], 96 + 60)
            self.assertEqual(scalar[2], 4)
            self.assertEqual(scalar[5], 1)
            self.assertEqual(
                struct.unpack_from("<I", invocation_image, scalar[3])[0], 2
            )

            invalid = subprocess.run([
                sys.executable,
                str(ROOT / "tools/models/build_tvm_byoc_invocation.py"),
                str(module_dir), str(directory / "invalid.npxmi"),
                "--arena", f"decode={arena_path}",
                "--scalar", "decode_position=4",
            ], check=False, capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("outside its declared range", invalid.stderr)

    def test_planar_kv_append_orders_attention_consumer(self):
        module = json.loads((
            ROOT / "tests/fixtures/models/tvm_byoc_kv_attention_module.json"
        ).read_text(encoding="utf-8"))
        _, manifest = compile_module(module)
        self.assertEqual(manifest["execution_order"], ["qkv_projection", "attention"])
        self.assertEqual(manifest["module_outputs"], [
            {"region": "qkv_projection", "tensor": "produced_kv"},
            {"region": "attention", "tensor": "context"},
        ])
        self.assertEqual(manifest["state_updates"], [{
            "from_region": "qkv_projection",
            "from_tensor": "produced_kv",
            "to_region": "attention",
            "to_tensor": "kv_cache",
            "bytes": 32,
            "mode": "append_planar2",
            "stride": 16,
            "capacity": 2,
        }])

        module["regions"][0]["graph"]["tensors"][7]["shape"] = [1, 2, 2]
        with self.assertRaisesRegex(CodegenError, "planar append"):
            compile_module(module)

    def test_state_append_rejects_incompatible_state_shape(self):
        module = json.loads(
            (ROOT / "tests/fixtures/models/tvm_byoc_state_append_module.json")
            .read_text(encoding="utf-8")
        )
        module["regions"][0]["graph"]["tensors"][1]["shape"] = [4, 3]
        with self.assertRaisesRegex(CodegenError, "shape"):
            compile_module(module)

    def test_state_update_cli_policy_resolves_typed_endpoints(self):
        module = self.load_fixture()
        apply_parameter_storage(module, [], ["lhs"])
        apply_state_updates(module, ["sum=lhs"])
        update = module["state_updates"][0]
        self.assertEqual(update["from"], {"region": "residual", "tensor": "sum"})
        self.assertEqual(update["to"], {"region": "residual", "tensor": "lhs"})

    def test_state_update_cli_policy_resolves_unique_graph_output(self):
        module = self.load_fixture()
        apply_parameter_storage(module, [], ["lhs"])
        apply_state_updates(module, ["@output=lhs"])
        self.assertEqual(
            module["state_updates"][0]["from"],
            {"region": "activation", "tensor": "output"},
        )

    def test_state_update_cli_policy_resolves_state_consumer(self):
        module = self.load_fixture()
        apply_parameter_storage(module, [], ["lhs"])
        apply_state_updates(module, ["@update=lhs"])
        self.assertEqual(
            module["state_updates"][0]["from"],
            {"region": "residual", "tensor": "sum"},
        )

    def test_state_update_cli_policy_rejects_multiple_state_consumers(self):
        module = self.load_fixture()
        apply_parameter_storage(module, [], ["lhs"])
        graph = module["regions"][1]["graph"]
        graph["nodes"].append({
            "op": "add", "inputs": ["lhs", "rhs"], "outputs": ["sum2"]
        })
        graph["tensors"].append({
            "name": "sum2", "shape": [2, 4], "dtype": "float32",
            "storage": "scratch",
        })
        with self.assertRaisesRegex(CodegenError, "one direct consumer output"):
            apply_state_updates(module, ["@update=lhs"])

    def test_state_update_cli_policy_rejects_ambiguous_endpoint(self):
        module = self.load_fixture()
        apply_parameter_storage(module, [], ["lhs"])
        module["regions"][0]["graph"]["tensors"][1]["name"] = "sum"
        module["regions"][0]["graph"]["tensors"][1]["storage"] = "output"
        with self.assertRaisesRegex(CodegenError, "exactly one produced Tensor"):
            apply_state_updates(module, ["sum=lhs"])

    def test_package_preserves_constants_and_invocation_omits_them(self):
        module = self.load_fixture()
        apply_parameter_storage(module, ["rhs"], [])
        artifacts, manifest = compile_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            module_dir = directory / "module"
            module_dir.mkdir()
            (module_dir / "module.npxgm.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            arena_arguments = []
            first_arena = None
            first_metadata = None
            for region in manifest["regions"]:
                binary, metadata = artifacts[region["name"]]
                artifact = module_dir / region["artifact"]
                artifact.write_bytes(binary)
                Path(f"{artifact}.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                arena = bytearray(region["arena_size"])
                tensors = {tensor["name"]: tensor for tensor in metadata["tensors"]}
                if region["name"] == "residual":
                    lhs = tensors["lhs"]
                    rhs = tensors["rhs"]
                    arena[lhs["offset"]:lhs["offset"] + lhs["byte_size"]] = (
                        bytes([0x11]) * lhs["byte_size"]
                    )
                    arena[rhs["offset"]:rhs["offset"] + rhs["byte_size"]] = (
                        bytes([0x22]) * rhs["byte_size"]
                    )
                    first_arena = arena
                    first_metadata = metadata
                arena_path = directory / f"{region['name']}.arena.bin"
                arena_path.write_bytes(arena)
                arena_arguments.extend(["--arena", f"{region['name']}={arena_path}"])

            package_path = directory / "module.npxgm"
            package = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/models/build_tvm_byoc_module_package.py"),
                    str(module_dir),
                    str(package_path),
                    "--clear-external-bindings",
                    *arena_arguments,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(package.returncode, 0, package.stderr)
            image = package_path.read_bytes()
            first_region = struct.unpack_from("<8I", image, 64)
            arena_offset = first_region[2]
            self.assertIsNotNone(first_arena)
            self.assertIsNotNone(first_metadata)
            tensors = {
                tensor["name"]: tensor for tensor in first_metadata["tensors"]
            }
            lhs = tensors["lhs"]
            rhs = tensors["rhs"]
            self.assertEqual(
                image[arena_offset + lhs["offset"]:
                      arena_offset + lhs["offset"] + lhs["byte_size"]],
                bytes(lhs["byte_size"]),
            )
            self.assertEqual(
                image[arena_offset + rhs["offset"]:
                      arena_offset + rhs["offset"] + rhs["byte_size"]],
                bytes([0x22]) * rhs["byte_size"],
            )

            invocation_path = directory / "module.npxmi"
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
            invocation_header = struct.unpack_from(
                "<8I", invocation_path.read_bytes()
            )
            self.assertEqual(invocation_header[4], 1)

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
            self.assertNotEqual(header[15], 0)
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
            self.assertEqual(invocation_header[7], header[15])


if __name__ == "__main__":
    unittest.main()
