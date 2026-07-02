#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BRIDGE="${ROOT_DIR}/build/coralnpu/libcoralnpu_gem5_rvv_highmem_bridge.so"
FIRMWARE="${ROOT_DIR}/build/coralnpu/gem5_mobilenet.elf"
TEST_SCRIPT="${ROOT_DIR}/thirdparty/gem5/configs/coralnpu/coral-mobilenet-test.rcS"
GEM5_OPTIONS_VALUE="${GEM5_OPTIONS:-}"
RTL_CYCLES_PER_EVENT="${CORAL_RTL_CYCLES_PER_EVENT:-1}"
KERNEL_RELEASE_FILE="${ROOT_DIR}/build/kernel/kernel.release"
VALIDATED_DISK_DEFAULT="/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img"

if [ -z "${CORAL_KERNEL_IMAGE:-}" ]; then
    [ -f "${KERNEL_RELEASE_FILE}" ] || {
        echo "error: kernel release metadata not found: ${KERNEL_RELEASE_FILE}" >&2
        echo "build the validated 4.19 kernel before running MobileNet" >&2
        exit 1
    }
    KERNEL_RELEASE="$(cat "${KERNEL_RELEASE_FILE}")"
    CORAL_KERNEL_IMAGE="${ROOT_DIR}/build/kernel/vmlinux-${KERNEL_RELEASE}"
fi
[ -f "${CORAL_KERNEL_IMAGE}" ] || {
    echo "error: validated gem5 kernel not found: ${CORAL_KERNEL_IMAGE}" >&2
    exit 1
}
CORAL_KERNEL_INIT="${CORAL_KERNEL_INIT:-/sbin/opennpux-init.sh}"
CORAL_DISK_IMG="${CORAL_DISK_IMG:-${VALIDATED_DISK_DEFAULT}}"
[ -f "${CORAL_DISK_IMG}" ] || {
    echo "error: validated gem5 disk image not found: ${CORAL_DISK_IMG}" >&2
    echo "set CORAL_DISK_IMG to the Phase-3 validated image" >&2
    exit 1
}

echo "[coral-mobilenet] kernel: ${CORAL_KERNEL_IMAGE}"
echo "[coral-mobilenet] init: ${CORAL_KERNEL_INIT}"
echo "[coral-mobilenet] disk: ${CORAL_DISK_IMG}"

if [ "${CORAL_MOBILENET_DEBUG:-0}" = "1" ]; then
    DEBUG_LOG="${CORAL_MOBILENET_DEBUG_LOG:-${ROOT_DIR}/simout/coral-mobilenet.debug}"
    mkdir -p "$(dirname "${DEBUG_LOG}")"
    GEM5_OPTIONS_VALUE="${GEM5_OPTIONS_VALUE} --debug-flags=NPUDevice --debug-file=${DEBUG_LOG}"
    echo "[coral-mobilenet] debug log: ${DEBUG_LOG}"
fi

echo "[coral-mobilenet] RTL cycles per event: ${RTL_CYCLES_PER_EVENT}"
[ -z "${GEM5_OPTIONS_VALUE}" ] || \
    echo "[coral-mobilenet] gem5 options: ${GEM5_OPTIONS_VALUE}"

[ -f "${BRIDGE}" ] || {
    echo "error: RVV highmem bridge not found: ${BRIDGE}" >&2
    exit 1
}
[ -f "${FIRMWARE}" ] || {
    echo "error: MobileNet firmware not found: ${FIRMWARE}" >&2
    exit 1
}

"${ROOT_DIR}/sim/gem5/apply_patchset.sh"

CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_BRIDGE="${BRIDGE}" \
CORAL_RTL_FIRMWARE="${FIRMWARE}" \
CORAL_RTL_CYCLES_PER_EVENT="${RTL_CYCLES_PER_EVENT}" \
CORAL_KERNEL_IMAGE="${CORAL_KERNEL_IMAGE}" \
CORAL_KERNEL_INIT="${CORAL_KERNEL_INIT}" \
CORAL_DISK_IMG="${CORAL_DISK_IMG}" \
CORAL_AUTO_RESUME_AFTER_CKPT=1 \
CORAL_CKPT_ROOT="${CORAL_MOBILENET_CKPT_ROOT:-${ROOT_DIR}/m5out/coralnpu_mobilenet_ckpt}" \
CORAL_RESUME_BOOTSCRIPT="${TEST_SCRIPT}" \
CORAL_CONFIG_OPTIONS="${CORAL_CONFIG_OPTIONS:-} --npu-dma-shared-size=8MiB" \
GEM5_OPTIONS="${GEM5_OPTIONS_VALUE}" \
exec "${ROOT_DIR}/thirdparty/gem5/run_multicore.sh"
