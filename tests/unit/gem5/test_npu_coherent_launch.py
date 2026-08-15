from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "sim/gem5/src/dev/npu/npu_device.cc"


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : index]
    raise AssertionError(f"unterminated function {signature}")


def test_local_extmem_launch_uses_coherent_dma_before_reset_release():
    source = SOURCE.read_text(encoding="utf-8")
    start = function_body(source, "NPUDevice::startHostToLocalExtmemSync()")
    complete = function_body(
        source, "NPUDevice::completeHostToLocalExtmemSync()"
    )

    assert "dmaRead(" in start
    assert "functionalMemoryRange(" not in start
    assert "releaseBackendReset();" not in start.split("dmaRead(", 1)[1]
    assert complete.index("writeLocalExtmem(") < complete.index(
        "releaseBackendReset();"
    )
    assert "first_word=%#010x" in complete
