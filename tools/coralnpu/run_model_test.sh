#!/bin/sh
# Run the Phase-4/5 heterogeneous model dispatch test.
#
# Dispatches a .npxm model container with mixed official/custom operators
# through the versioned command ABI.
#
# Prerequisites:
#   - Bridge and command firmware must be built (tools/coralnpu/phase2_build_bridge.sh)
#   - Guest assets must be installed (tools/coralnpu/phase45_prepare_guest_assets.sh)
#   - Boot checkpoint must be rebuilt after installing guest assets

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BRIDGE="${ROOT_DIR}/build/coralnpu/libcoralnpu_gem5_bridge.so"
FIRMWARE="${ROOT_DIR}/build/coralnpu/gem5_command_smoke.elf"
TEST_SCRIPT="${ROOT_DIR}/thirdparty/gem5/configs/coralnpu/coral-model-test.rcS"

[ -f "${BRIDGE}" ] || {
    echo "error: Coral RTL bridge not found: ${BRIDGE}" >&2
    echo "run ./tools/coralnpu/phase2_build_bridge.sh first" >&2
    exit 1
}
[ -f "${FIRMWARE}" ] || {
    echo "error: command firmware not found: ${FIRMWARE}" >&2
    echo "run ./tools/coralnpu/phase2_build_bridge.sh first" >&2
    exit 1
}
"${ROOT_DIR}/sim/gem5/apply_patchset.sh"
CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_BRIDGE="${BRIDGE}" \
CORAL_RTL_FIRMWARE="${FIRMWARE}" \
CORAL_RESUME_BOOTSCRIPT="${TEST_SCRIPT}" \
exec "${ROOT_DIR}/thirdparty/gem5/run_multicore.sh"
