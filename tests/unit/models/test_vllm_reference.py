import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/models"))

from run_vllm_next_token import _safe_gpu_memory_utilization  # noqa: E402


class VllmReferenceTest(unittest.TestCase):
    def test_caps_budget_at_normal_default(self):
        gib = 1024**3
        self.assertEqual(_safe_gpu_memory_utilization(79 * gib, 80 * gib), 0.9)

    def test_scales_budget_to_current_free_memory(self):
        gib = 1024**3
        self.assertEqual(_safe_gpu_memory_utilization(39 * gib, 83 * gib), 0.44)

    def test_rejects_exhausted_device(self):
        gib = 1024**3
        with self.assertRaisesRegex(RuntimeError, "insufficient free GPU memory"):
            _safe_gpu_memory_utilization(5 * gib, 83 * gib)


if __name__ == "__main__":
    unittest.main()
