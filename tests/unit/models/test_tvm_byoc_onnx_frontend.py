import argparse
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "tools/models/import_onnx_to_relax.py"
SPEC = importlib.util.spec_from_file_location("import_onnx_to_relax", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TvmByocOnnxFrontendTest(unittest.TestCase):
    def test_preserves_onnx_parameter_order_as_source_names(self):
        model = SimpleNamespace(
            graph=SimpleNamespace(
                input=[SimpleNamespace(name="hidden_states")],
                initializer=[
                    SimpleNamespace(name="query_shape"),
                    SimpleNamespace(name="output_weight"),
                ],
            )
        )
        self.assertEqual(
            MODULE.source_parameter_names(model, ["query_shape"]),
            ["hidden_states", "output_weight"],
        )

    def test_parses_shape(self):
        self.assertEqual(MODULE.parse_shape("tokens=1,16,2048"),
                         ("tokens", [1, 16, 2048]))

    def test_rejects_dynamic_or_zero_shape(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            MODULE.parse_shape("tokens=1,0,2048")

    def test_rejects_duplicate_overrides(self):
        with self.assertRaisesRegex(ValueError, "duplicate shape"):
            MODULE.unique_map([("tokens", [1]), ("tokens", [2])], "shape")

    def test_classifies_only_reshape_shape_as_compile_time(self):
        model = SimpleNamespace(
            graph=SimpleNamespace(
                initializer=[
                    SimpleNamespace(name="query_shape"),
                    SimpleNamespace(name="weight"),
                ],
                node=[
                    SimpleNamespace(
                        op_type="Reshape", input=["query_flat", "query_shape"]
                    ),
                    SimpleNamespace(op_type="MatMul", input=["query", "weight"]),
                ],
            )
        )
        self.assertEqual(
            MODULE.compile_time_initializer_names(model), ["query_shape"]
        )

    def test_ignores_runtime_reshape_shape(self):
        model = SimpleNamespace(
            graph=SimpleNamespace(
                initializer=[SimpleNamespace(name="weight")],
                node=[
                    SimpleNamespace(
                        op_type="Reshape", input=["query_flat", "runtime_shape"]
                    )
                ],
            )
        )
        self.assertEqual(MODULE.compile_time_initializer_names(model), [])


if __name__ == "__main__":
    unittest.main()
