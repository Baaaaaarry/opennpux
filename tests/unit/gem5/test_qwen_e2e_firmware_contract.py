from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_qwen_device_firmware_receives_device_request_only():
    scripts = (
        ROOT / "runtime/host/bootscripts/coral-qwen-e2e-test.rcS",
        ROOT / "sim/gem5/configs/coralnpu/coral-qwen-e2e-test.rcS",
    )
    for script in scripts:
        source = script.read_text(encoding="utf-8")
        assert "qwen-device-run" in source
        assert "run_and_print qwen-run-tcb" not in source


def test_qwen_runner_uses_runtime_bootscript_source_of_truth():
    source = (
        ROOT / "tools/coralnpu/run_qwen_e2e_test.sh"
    ).read_text(encoding="utf-8")
    assert (
        'TEST_SCRIPT="${ROOT_DIR}/runtime/host/bootscripts/'
        'coral-qwen-e2e-test.rcS"'
    ) in source
