import json
import struct
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/models"))

from opennpux_tvm_byoc.xgraph_codegen import (  # noqa: E402
    COMMAND,
    HEADER,
    CodegenError,
    compile_graph,
)


class XGraphCodegenTest(unittest.TestCase):
    def load_fixture(self):
        path = ROOT / "tests/fixtures/models/tvm_byoc_basic.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_compiles_exact_xgraph_v2_abi(self):
        binary, metadata = compile_graph(self.load_fixture())
        header = HEADER.unpack_from(binary)

        self.assertEqual(HEADER.size, 96)
        self.assertEqual(COMMAND.size, 64)
        self.assertEqual(header[0], 0x5847504E)
        self.assertEqual(header[1], 2)
        self.assertEqual(header[2], 96)
        self.assertEqual(header[3], 64)
        self.assertEqual(header[4], 5)
        self.assertEqual(header[5], len(binary))
        self.assertEqual(header[6], 1)
        self.assertEqual(metadata["command_count"], 5)

        commands = [
            COMMAND.unpack_from(binary, HEADER.size + index * COMMAND.size)
            for index in range(5)
        ]
        self.assertEqual([command[0] for command in commands], [1, 2, 7, 5, 9])
        self.assertEqual(commands[0][5:8], (2, 3, 4))
        self.assertEqual(commands[4][1], 1)
        self.assertEqual(commands[4][8], 1)
        self.assertEqual(commands[4][11], metadata["tensors"][-1]["offset"])
        self.assertEqual([command[10] for command in commands], list(range(5)))

    def test_arena_is_aligned_and_non_overlapping(self):
        _, metadata = compile_graph(self.load_fixture())
        prior_end = 0
        for tensor in metadata["tensors"]:
            self.assertEqual(tensor["offset"] % 64, 0)
            self.assertGreaterEqual(tensor["offset"], prior_end)
            prior_end = tensor["offset"] + tensor["byte_size"]

    def test_rejects_broadcast_until_semantics_are_explicit(self):
        graph = self.load_fixture()
        graph["tensors"][2]["shape"] = [3]
        with self.assertRaisesRegex(CodegenError, "non-broadcast"):
            compile_graph(graph)

    def test_rejects_matmul_requiring_tiled_lowering(self):
        graph = {
            "format": "OPENNPUX_TVM_BYOC_GRAPH_V1",
            "tensors": [
                {
                    "name": "a",
                    "shape": [1, 2048],
                    "dtype": "float32",
                    "storage": "input",
                },
                {
                    "name": "b",
                    "shape": [2048, 8],
                    "dtype": "float32",
                    "storage": "constant",
                },
                {
                    "name": "c",
                    "shape": [1, 8],
                    "dtype": "float32",
                    "storage": "output",
                },
            ],
            "nodes": [{"op": "matmul", "inputs": ["a", "b"], "outputs": ["c"]}],
            "outputs": ["c"],
        }
        with self.assertRaisesRegex(CodegenError, "tiled lowering"):
            compile_graph(graph)

    def test_rejects_overlapping_explicit_offsets(self):
        graph = self.load_fixture()
        graph["tensors"][0]["offset"] = 0x20000
        graph["tensors"][1]["offset"] = 0x20000
        with self.assertRaisesRegex(CodegenError, "overlaps"):
            compile_graph(graph)

    def test_encodes_remaining_stage_one_primitives(self):
        graph = {
            "format": "OPENNPUX_TVM_BYOC_GRAPH_V1",
            "tensors": [
                {
                    "name": "table",
                    "shape": [4, 2],
                    "dtype": "float32",
                    "storage": "constant",
                },
                {"name": "indices", "shape": [2], "dtype": "int32", "storage": "input"},
                {"name": "gathered", "shape": [2, 2], "dtype": "float32"},
                {
                    "name": "norm_weight",
                    "shape": [2],
                    "dtype": "float32",
                    "storage": "constant",
                },
                {"name": "normalized", "shape": [2, 2], "dtype": "float32"},
                {
                    "name": "rope_table",
                    "shape": [2, 2],
                    "dtype": "float32",
                    "storage": "constant",
                },
                {"name": "rotated", "shape": [2, 2], "dtype": "float32"},
                {
                    "name": "scale",
                    "shape": [2, 2],
                    "dtype": "float32",
                    "storage": "constant",
                },
                {"name": "scaled", "shape": [2, 2], "dtype": "float32"},
                {
                    "name": "output",
                    "shape": [2, 2],
                    "dtype": "float32",
                    "storage": "output",
                },
            ],
            "nodes": [
                {"op": "take", "inputs": ["table", "indices"], "outputs": ["gathered"]},
                {
                    "op": "rms_norm",
                    "inputs": ["gathered", "norm_weight"],
                    "outputs": ["normalized"],
                },
                {
                    "op": "rope",
                    "inputs": ["normalized", "rope_table"],
                    "outputs": ["rotated"],
                    "attrs": {"layout": "half_split"},
                },
                {"op": "multiply", "inputs": ["rotated", "scale"], "outputs": ["scaled"]},
                {"op": "copy", "inputs": ["scaled"], "outputs": ["output"]},
            ],
            "outputs": ["output"],
        }
        binary, _ = compile_graph(graph)
        commands = [
            COMMAND.unpack_from(binary, HEADER.size + index * COMMAND.size)
            for index in range(5)
        ]
        self.assertEqual([command[0] for command in commands], [8, 4, 6, 3, 11])
        self.assertEqual(commands[2][8], 1)


if __name__ == "__main__":
    unittest.main()
