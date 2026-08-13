#!/bin/sh
# Run the Phase-3 driver-backed DMA smoke test.
#
# Validates the Linux kernel driver (/dev/opennpux-coral) with coherent DMA
# without requiring /dev/mem access.
#
# Prerequisites:
#   - DMA firmware must be built (tools/coralnpu/phase2_build_bridge.sh)
#   - Validated 4.19 kernel with opennpux_coral.ko must be installed in guest image
#
# Expected output: dma_test=PASS

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FIRMWARE="${ROOT_DIR}/build/coralnpu/gem5_dma_smoke.elf"
TEST_SCRIPT="${ROOT_DIR}/thirdparty/gem5/configs/coralnpu/coral-driver-dma-test.rcS"

if [ ! -f "${FIRMWARE}" ]; then
    echo "error: DMA smoke firmware not found: ${FIRMWARE}" >&2
    echo "run ./tools/coralnpu/build_rtl_bridge.sh first" >&2
    exit 1
fi

"${ROOT_DIR}/sim/gem5/apply_patchset.sh"

CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_FIRMWARE="${FIRMWARE}" \
CORAL_RESUME_BOOTSCRIPT="${TEST_SCRIPT}" \
GEM5_OPTIONS="${GEM5_OPTIONS:---debug-flags=NPUDevice}" \
exec "${ROOT_DIR}/thirdparty/gem5/run_multicore.sh"
