import argparse
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "tools/models/import_onnx_to_relax.py"
SPEC = importlib.util.spec_from_file_location("import_onnx_to_relax", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TvmByocOnnxFrontendTest(unittest.TestCase):
    def test_parses_shape(self):
        self.assertEqual(MODULE.parse_shape("tokens=1,16,2048"),
                         ("tokens", [1, 16, 2048]))

    def test_rejects_dynamic_or_zero_shape(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            MODULE.parse_shape("tokens=1,0,2048")

    def test_rejects_duplicate_overrides(self):
        with self.assertRaisesRegex(ValueError, "duplicate shape"):
            MODULE.unique_map([("tokens", [1]), ("tokens", [2])], "shape")


if __name__ == "__main__":
    unittest.main()
