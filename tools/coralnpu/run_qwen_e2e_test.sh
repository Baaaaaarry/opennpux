#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
GEM5_ROOT="${ROOT_DIR}/thirdparty/gem5"
TEST_SCRIPT="${ROOT_DIR}/runtime/host/bootscripts/coral-qwen-e2e-test.rcS"
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
CORAL_RTL_BRIDGE="${CORAL_RTL_BRIDGE:-${ROOT_DIR}/build/coralnpu/libcoralnpu_gem5_rvv_highmem_bridge.so}"
CORAL_RTL_FIRMWARE="${CORAL_RTL_FIRMWARE:-${ROOT_DIR}/build/coralnpu/gem5_qwen_device_infer.elf}"
CORAL_CKPT_ROOT="${CORAL_CKPT_ROOT:-${ROOT_DIR}/checkpoint/coralnpu_qwen_device_ckpt}"

# Pulling an overlay commit does not always make the mirrored source newer
# than an existing gem5 binary (for example after preserving timestamps while
# copying a worktree). Require the coherent-launch capability in the binary;
# run_multicore.sh will relink it when the marker is absent.
if [ -x "${GEM5_ROOT}/build/ARM/gem5.opt" ] &&
   ! strings "${GEM5_ROOT}/build/ARM/gem5.opt" 2>/dev/null |
       grep -q 'Coral coherent host-to-EXTMEM sync complete'; then
    echo "[coral-qwen-e2e] gem5 lacks coherent launch sync; forcing rebuild"
    GEM5_REBUILD=1
    export GEM5_REBUILD
fi

[ -f "${CORAL_DISK_IMG}" ] || {
    echo "error: disk image not found: ${CORAL_DISK_IMG}" >&2
    exit 1
}
[ -f "${CORAL_KERNEL_IMAGE}" ] || {
    echo "error: kernel image not found: ${CORAL_KERNEL_IMAGE}" >&2
    exit 1
}
[ -f "${CORAL_RTL_BRIDGE}" ] || {
    echo "error: RTL bridge not found: ${CORAL_RTL_BRIDGE}" >&2
    echo "hint: run ./tools/coralnpu/build_rtl_bridge.sh" >&2
    exit 1
}
[ -f "${CORAL_RTL_FIRMWARE}" ] || {
    echo "error: Qwen device inference firmware not found: ${CORAL_RTL_FIRMWARE}" >&2
    echo "hint: run ./tools/coralnpu/build_rtl_bridge.sh" >&2
    exit 1
}

"${ROOT_DIR}/sim/gem5/apply_patchset.sh"
cd "${GEM5_ROOT}"
CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_BRIDGE="${CORAL_RTL_BRIDGE}" \
CORAL_RTL_FIRMWARE="${CORAL_RTL_FIRMWARE}" \
CORAL_DISK_IMG="${CORAL_DISK_IMG}" \
CORAL_KERNEL_IMAGE="${CORAL_KERNEL_IMAGE}" \
CORAL_KERNEL_INIT="${CORAL_KERNEL_INIT}" \
CORAL_CKPT_ROOT="${CORAL_CKPT_ROOT}" \
CORAL_RESUME_BOOTSCRIPT="${TEST_SCRIPT}" \
CORAL_CONFIG_OPTIONS="${CORAL_CONFIG_OPTIONS:-} --npu-dma-shared-size=8MiB --npu-operator-mode=hybrid --npu-fast-dma --npu-fast-dma-sync-offset=0 --npu-fast-dma-sync-size=8KiB" \
./run_multicore.sh
