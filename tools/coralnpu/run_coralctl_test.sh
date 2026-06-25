#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
TEST_SCRIPT="${ROOT_DIR}/thirdparty/gem5/configs/coralnpu/coralctl-test.rcS"

"${ROOT_DIR}/sim/gem5/apply_patchset.sh"

if [ ! -f "${TEST_SCRIPT}" ]; then
    echo "error: Coral control test script was not installed: ${TEST_SCRIPT}" >&2
    exit 1
fi

echo "[coralctl-test] resume script: ${TEST_SCRIPT}"
CORAL_NPU_BACKEND="${CORAL_NPU_BACKEND:-verilated-coral}" \
CORAL_RESUME_BOOTSCRIPT="${TEST_SCRIPT}" \
exec "${ROOT_DIR}/thirdparty/gem5/run_multicore.sh"
