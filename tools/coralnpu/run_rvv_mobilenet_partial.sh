#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BRIDGE="${ROOT_DIR}/build/coralnpu/libcoralnpu_gem5_rvv_highmem_bridge.so"
FIRMWARE="${ROOT_DIR}/build/coralnpu/gem5_mobilenet_partial.elf"
TEST_SCRIPT="${ROOT_DIR}/thirdparty/gem5/configs/coralnpu/coral-mobilenet-partial-test.rcS"
GEM5_OPTIONS_VALUE="${GEM5_OPTIONS:-}"
RTL_CYCLES_PER_EVENT="${CORAL_RTL_CYCLES_PER_EVENT:-1000}"
FAST_DMA_EVENT_BATCH="${CORAL_FAST_DMA_EVENT_BATCH:-4096}"
OPERATOR_MODE="${CORAL_OPERATOR_MODE:-hybrid}"
FAST_DMA_OPTION=""
[ "${CORAL_FAST_DMA:-1}" != "1" ] || FAST_DMA_OPTION=" --npu-fast-dma"
KERNEL_RELEASE_FILE="${ROOT_DIR}/build/kernel/kernel.release"
VALIDATED_DISK_DEFAULT="/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img"
CKPT_ROOT="${CORAL_MOBILENET_PARTIAL_CKPT_ROOT:-${ROOT_DIR}/m5out/coralnpu_mobilenet_partial_ckpt}"
DMA_SHARED_BASE="${CORAL_MOBILENET_PARTIAL_SHARED_BASE:-0x8f000000}"
DMA_SHARED_BASE_META="${CKPT_ROOT}.shared_base"

case "${OPERATOR_MODE}" in
    rtl|hybrid|sampled) ;;
    *) echo "error: CORAL_OPERATOR_MODE must be rtl, hybrid, or sampled" >&2; exit 1 ;;
esac

if [ -z "${CORAL_KERNEL_IMAGE:-}" ]; then
    [ -f "${KERNEL_RELEASE_FILE}" ] || {
        echo "error: kernel release metadata not found: ${KERNEL_RELEASE_FILE}" >&2
        echo "build the validated 4.19 kernel before running MobileNet partial" >&2
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

echo "[coral-mobilenet-partial] kernel: ${CORAL_KERNEL_IMAGE}"
echo "[coral-mobilenet-partial] init: ${CORAL_KERNEL_INIT}"
echo "[coral-mobilenet-partial] disk: ${CORAL_DISK_IMG}"
echo "[coral-mobilenet-partial] operator mode: ${OPERATOR_MODE}"
echo "[coral-mobilenet-partial] RTL cycles per event: ${RTL_CYCLES_PER_EVENT}"
if [ -n "${FAST_DMA_OPTION}" ]; then
    echo "[coral-mobilenet-partial] DMA mode: functional-fast"
    echo "[coral-mobilenet-partial] fast DMA event batch: ${FAST_DMA_EVENT_BATCH}"
else
    echo "[coral-mobilenet-partial] DMA mode: timing"
fi

if [ "${CORAL_MOBILENET_PARTIAL_DEBUG:-${CORAL_MOBILENET_DEBUG:-0}}" = "1" ]; then
    DEBUG_LOG="${CORAL_MOBILENET_PARTIAL_DEBUG_LOG:-${ROOT_DIR}/simout/coral-mobilenet-partial.debug}"
    mkdir -p "$(dirname "${DEBUG_LOG}")"
    GEM5_OPTIONS_VALUE="${GEM5_OPTIONS_VALUE} --debug-flags=NPUDevice --debug-file=${DEBUG_LOG}"
    echo "[coral-mobilenet-partial] debug log: ${DEBUG_LOG}"
fi
[ -z "${GEM5_OPTIONS_VALUE}" ] || \
    echo "[coral-mobilenet-partial] gem5 options: ${GEM5_OPTIONS_VALUE}"

[ -f "${BRIDGE}" ] || {
    echo "error: RVV highmem bridge not found: ${BRIDGE}" >&2
    echo "run ./tools/coralnpu/build_rvv_mobilenet_partial.sh first" >&2
    exit 1
}
[ -f "${FIRMWARE}" ] || {
    echo "error: partial MobileNet firmware not found: ${FIRMWARE}" >&2
    echo "run ./tools/coralnpu/build_rvv_mobilenet_partial.sh first" >&2
    exit 1
}

"${ROOT_DIR}/sim/gem5/apply_patchset.sh"

if [ -f "${CKPT_ROOT}/booted/m5.cpt" ] &&
   { [ ! -f "${DMA_SHARED_BASE_META}" ] ||
     [ "$(cat "${DMA_SHARED_BASE_META}")" != "${DMA_SHARED_BASE}" ]; }; then
    echo "[coral-mobilenet-partial] shared DMA base changed; rebuilding checkpoint"
    rm -rf "${CKPT_ROOT}"
fi
mkdir -p "$(dirname "${DMA_SHARED_BASE_META}")"
printf '%s\n' "${DMA_SHARED_BASE}" > "${DMA_SHARED_BASE_META}"

CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_BRIDGE="${BRIDGE}" \
CORAL_RTL_FIRMWARE="${FIRMWARE}" \
CORAL_RTL_CYCLES_PER_EVENT="${RTL_CYCLES_PER_EVENT}" \
CORAL_KERNEL_IMAGE="${CORAL_KERNEL_IMAGE}" \
CORAL_KERNEL_INIT="${CORAL_KERNEL_INIT}" \
CORAL_DISK_IMG="${CORAL_DISK_IMG}" \
CORAL_AUTO_RESUME_AFTER_CKPT=1 \
CORAL_CKPT_ROOT="${CKPT_ROOT}" \
CORAL_RESUME_BOOTSCRIPT="${TEST_SCRIPT}" \
CORAL_CONFIG_OPTIONS="${CORAL_CONFIG_OPTIONS:-} --npu-dma-shared-base=${DMA_SHARED_BASE} --npu-dma-shared-size=8MiB --npu-operator-mode=${OPERATOR_MODE}${FAST_DMA_OPTION} --npu-fast-dma-event-batch=${FAST_DMA_EVENT_BATCH}" \
GEM5_OPTIONS="${GEM5_OPTIONS_VALUE}" \
exec "${ROOT_DIR}/thirdparty/gem5/run_multicore.sh"
