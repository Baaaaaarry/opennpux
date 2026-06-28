#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FIRMWARE="${ROOT_DIR}/build/coralnpu/gem5_command_smoke.elf"
TEST_SCRIPT="${ROOT_DIR}/thirdparty/gem5/configs/coralnpu/coral-command-test.rcS"

if [ ! -f "${FIRMWARE}" ]; then
    echo "error: command firmware not found: ${FIRMWARE}" >&2
    echo "run ./tools/coralnpu/phase2_build_bridge.sh first" >&2
    exit 1
fi

"${ROOT_DIR}/sim/gem5/apply_patchset.sh"

CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_FIRMWARE="${FIRMWARE}" \
CORAL_RESUME_BOOTSCRIPT="${TEST_SCRIPT}" \
GEM5_OPTIONS="${GEM5_OPTIONS:---debug-flags=NPUDevice}" \
exec "${ROOT_DIR}/thirdparty/gem5/run_multicore.sh"
