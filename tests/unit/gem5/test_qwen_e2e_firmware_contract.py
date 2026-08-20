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


def test_qwen35b_weight_plan_rebuild_uses_json_executable_plan():
    source = (
        ROOT / "tools/coralnpu/run_qwen35b_real_weights_test.sh"
    ).read_text(encoding="utf-8")
    assert (
        'EXECUTABLE_PLAN_NAME="${CORAL_NPU_EXECUTABLE_PLAN_NAME:-model.npxe}"'
        in source
    )
    compile_call = source.split(
        '"${ROOT_DIR}/tools/models/compile_npu_weight_plan.py"', 1
    )[1].split("--require-complete", 1)[0]
    assert '"$MODEL_DIR/$EXECUTABLE_PLAN_NAME"' in compile_call
    assert '"$MODEL_DIR/$EXECUTABLE_NAME"' not in compile_call
