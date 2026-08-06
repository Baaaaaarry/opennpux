#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
GEM5_ROOT="${ROOT_DIR}/thirdparty/gem5"
TEST_SCRIPT="${GEM5_ROOT}/configs/coralnpu/coral-qwen-info-test.rcS"
KERNEL_RELEASE_FILE="${ROOT_DIR}/build/kernel/kernel.release"
VALIDATED_DISK_DEFAULT="/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img"

if [ -z "${CORAL_KERNEL_IMAGE:-}" ]; then
    if [ -f "${KERNEL_RELEASE_FILE}" ]; then
        KERNEL_RELEASE="$(cat "${KERNEL_RELEASE_FILE}")"
        CORAL_KERNEL_IMAGE="${ROOT_DIR}/build/kernel/vmlinux-${KERNEL_RELEASE}"
    else
        CORAL_KERNEL_IMAGE="/home/barry/wlk/gem5_arm_linux_images/vmlinux.arm64"
    fi
fi

CORAL_KERNEL_INIT="${CORAL_KERNEL_INIT:-/sbin/opennpux-init.sh}"
CORAL_DISK_IMG="${CORAL_DISK_IMG:-${VALIDATED_DISK_DEFAULT}}"

[ -f "${CORAL_DISK_IMG}" ] || {
    echo "error: disk image not found: ${CORAL_DISK_IMG}" >&2
    exit 1
}
[ -f "${CORAL_KERNEL_IMAGE}" ] || {
    echo "error: kernel image not found: ${CORAL_KERNEL_IMAGE}" >&2
    exit 1
}

"${ROOT_DIR}/sim/gem5/apply_patchset.sh"
cd "${GEM5_ROOT}"
CORAL_NPU_BACKEND=stage-a \
CORAL_DISK_IMG="${CORAL_DISK_IMG}" \
CORAL_KERNEL_IMAGE="${CORAL_KERNEL_IMAGE}" \
CORAL_KERNEL_INIT="${CORAL_KERNEL_INIT}" \
CORAL_RESUME_BOOTSCRIPT="${TEST_SCRIPT}" \
./run_multicore.sh
