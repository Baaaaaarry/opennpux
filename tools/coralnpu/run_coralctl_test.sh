#!/bin/sh
# Validate the coralctl tool inside a gem5 full-system simulation.
#
# Prerequisites:
#   - coralctl must be installed in the guest disk image
#     (tools/guest_tools/build_coralctl.sh && install_coralctl_to_image.sh)
#   - When CORAL_NPU_BACKEND=verilated-coral, the bridge must be built
#     (tools/coralnpu/phase2_build_bridge.sh)
#
# Environment:
#   CORAL_NPU_BACKEND       default: verilated-coral
#   CORAL_DISK_IMG          path to the ARM64 disk image
#   CORAL_KERNEL_IMAGE      path to the vmlinux kernel ELF

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
TEST_SCRIPT="${ROOT_DIR}/thirdparty/gem5/configs/coralnpu/coralctl-test.rcS"
BRIDGE="${ROOT_DIR}/build/coralnpu/libcoralnpu_gem5_bridge.so"
BACKEND="${CORAL_NPU_BACKEND:-verilated-coral}"

"${ROOT_DIR}/sim/gem5/apply_patchset.sh"

if [ ! -f "${TEST_SCRIPT}" ]; then
    echo "error: Coral control test script was not installed: ${TEST_SCRIPT}" >&2
    exit 1
fi

# When using the RTL backend, the bridge shared library must exist.
if [ "${BACKEND}" = "verilated-coral" ] && [ ! -f "${BRIDGE}" ]; then
    echo "error: Coral RTL bridge not found: ${BRIDGE}" >&2
    echo "run ./tools/coralnpu/phase2_build_bridge.sh first" >&2
    exit 1
fi

echo "[coralctl-test] resume script: ${TEST_SCRIPT}"
CORAL_NPU_BACKEND="${BACKEND}" \
CORAL_RTL_BRIDGE="${BRIDGE}" \
CORAL_RESUME_BOOTSCRIPT="${TEST_SCRIPT}" \
exec "${ROOT_DIR}/thirdparty/gem5/run_multicore.sh"
