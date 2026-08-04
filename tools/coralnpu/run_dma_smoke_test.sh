#!/bin/sh
# Run the coherent DMA smoke test with the Verilated Coral RTL backend.
#
# Prerequisites:
#   - Bridge and firmware must be built (tools/coralnpu/phase2_build_bridge.sh)
#   - gem5 overlay must be applied (sim/gem5/apply_patchset.sh)
#   - coralctl must be installed in the guest image
#
# Expected output: dma_result=42 dma_magic=0x4e505544 dma_test=PASS

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BRIDGE="${ROOT_DIR}/build/coralnpu/libcoralnpu_gem5_bridge.so"
FIRMWARE="${ROOT_DIR}/build/coralnpu/gem5_dma_smoke.elf"
TEST_SCRIPT="${ROOT_DIR}/thirdparty/gem5/configs/coralnpu/coral-dma-test.rcS"

if [ ! -f "${BRIDGE}" ]; then
    echo "error: Coral RTL bridge not found: ${BRIDGE}" >&2
    echo "run ./tools/coralnpu/phase2_build_bridge.sh first" >&2
    exit 1
fi
if [ ! -f "${FIRMWARE}" ]; then
    echo "error: DMA smoke firmware not found: ${FIRMWARE}" >&2
    echo "run ./tools/coralnpu/phase2_build_bridge.sh first" >&2
    exit 1
fi

"${ROOT_DIR}/sim/gem5/apply_patchset.sh"

CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_BRIDGE="${BRIDGE}" \
CORAL_RTL_FIRMWARE="${FIRMWARE}" \
CORAL_RESUME_BOOTSCRIPT="${TEST_SCRIPT}" \
GEM5_OPTIONS="${GEM5_OPTIONS:---debug-flags=NPUDevice}" \
exec "${ROOT_DIR}/thirdparty/gem5/run_multicore.sh"
