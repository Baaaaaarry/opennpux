import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COMPILER = ROOT / "tools/models/compile_tvm_byoc_deployment.py"
MODEL = ROOT / "tests/fixtures/models/tvm_byoc_module.json"


def deployment() -> dict:
    return {
        "format": "OPENNPUX_TVM_BYOC_DEPLOYMENT_V1",
        "constant_parameters": ["rhs"],
        "state_parameters": [],
        "state_updates": [],
        "state_appends": [],
        "module_values": {"rhs": {"fill": 1.0}},
        "invocations": [{
            "name": "request-000",
            "values": {"lhs": {"repeat": [-1.0, 0.0, 1.0, 2.0]}},
            "scalars": {},
        }],
    }


class TvmByocDeploymentTest(unittest.TestCase):
    def run_compiler(self, spec: dict):
        temporary = tempfile.TemporaryDirectory()
        directory = Path(temporary.name)
        spec_path = directory / "deployment.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(COMPILER), str(MODEL), str(spec_path),
             str(directory / "output")],
            check=False, capture_output=True, text=True,
        )
        return temporary, directory, result

    def test_compiles_reusable_module_and_invocation(self):
        temporary, directory, result = self.run_compiler(deployment())
        with temporary:
            self.assertEqual(result.returncode, 0, result.stderr)
            output = directory / "output"
            self.assertTrue((output / "model.npxgm").is_file())
            self.assertTrue((output / "request-000.npxmi").is_file())
            manifest = json.loads(
                (output / "deployment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["region_count"], 2)
            self.assertEqual(manifest["command_count"], 2)
            self.assertEqual(manifest["invocations"], [{
                "name": "request-000", "artifact": "request-000.npxmi"
            }])
            self.assertIn("tvm_byoc_deployment=PASS", result.stdout)

    def test_rejects_missing_constant(self):
        spec = deployment()
        spec["module_values"] = {}
        temporary, _, result = self.run_compiler(spec)
        with temporary:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("module values missing residual.rhs", result.stderr)

    def test_rejects_missing_invocation_input(self):
        spec = deployment()
        spec["invocations"][0]["values"] = {}
        temporary, _, result = self.run_compiler(spec)
        with temporary:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invocation values missing residual.lhs", result.stderr)


if __name__ == "__main__":
    unittest.main()
