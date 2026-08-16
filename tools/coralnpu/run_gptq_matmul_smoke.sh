#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
GEM5_ROOT="${ROOT_DIR}/thirdparty/gem5"
BRIDGE="${CORAL_RTL_BRIDGE:-${ROOT_DIR}/build/coralnpu/libcoralnpu_gem5_rvv_highmem_bridge.so}"
FIRMWARE="${CORAL_RTL_FIRMWARE:-${ROOT_DIR}/build/coralnpu/gem5_gptq_matmul_smoke.elf}"
TEST_SCRIPT="${ROOT_DIR}/thirdparty/gem5/configs/coralnpu/coralctl-test.rcS"
HOST_LOG="${CORAL_GPTQ_HOST_LOG:-${ROOT_DIR}/simout/coral-gptq-host.log}"
DEBUG_LOG="${CORAL_GPTQ_DEBUG_LOG:-${ROOT_DIR}/simout/coral-gptq.debug}"
KERNEL_RELEASE_FILE="${ROOT_DIR}/build/kernel/kernel.release"

if [ -z "${CORAL_KERNEL_IMAGE:-}" ]; then
    if [ -f "${KERNEL_RELEASE_FILE}" ]; then
        KERNEL_RELEASE="$(cat "${KERNEL_RELEASE_FILE}")"
        CORAL_KERNEL_IMAGE="${ROOT_DIR}/build/kernel/vmlinux-${KERNEL_RELEASE}"
    else
        CORAL_KERNEL_IMAGE="/home/barry/wlk/gem5_arm_linux_images/vmlinux.arm64"
    fi
fi
CORAL_KERNEL_INIT="${CORAL_KERNEL_INIT:-/sbin/opennpux-init.sh}"
CORAL_DISK_IMG="${CORAL_DISK_IMG:-/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img}"
CORAL_CKPT_ROOT="${CORAL_CKPT_ROOT:-${ROOT_DIR}/checkpoint/coralnpu_gptq_ckpt}"

for path in "${BRIDGE}" "${FIRMWARE}" "${CORAL_KERNEL_IMAGE}" \
            "${CORAL_DISK_IMG}"; do
    [ -f "${path}" ] || { echo "error: required file not found: ${path}" >&2; exit 1; }
done
mkdir -p "$(dirname "${HOST_LOG}")" "$(dirname "${DEBUG_LOG}")"
"${ROOT_DIR}/sim/gem5/apply_patchset.sh"

cd "${GEM5_ROOT}"
CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_BRIDGE="${BRIDGE}" \
CORAL_RTL_FIRMWARE="${FIRMWARE}" \
CORAL_DISK_IMG="${CORAL_DISK_IMG}" \
CORAL_KERNEL_IMAGE="${CORAL_KERNEL_IMAGE}" \
CORAL_KERNEL_INIT="${CORAL_KERNEL_INIT}" \
CORAL_CKPT_ROOT="${CORAL_CKPT_ROOT}" \
CORAL_RESUME_BOOTSCRIPT="${TEST_SCRIPT}" \
CORAL_CONFIG_OPTIONS="${CORAL_CONFIG_OPTIONS:-} --npu-dma-shared-size=8MiB --npu-operator-mode=hybrid --npu-fast-dma --npu-fast-dma-sync-offset=0 --npu-fast-dma-sync-size=8KiB" \
GEM5_OPTIONS="${GEM5_OPTIONS:-} --debug-flags=NPUDevice --debug-file=${DEBUG_LOG}" \
./run_multicore.sh 2>&1 | tee "${HOST_LOG}"

grep -q 'operator_opcode=10 source=custom-instruction' "${HOST_LOG}" || {
    echo "error: GPTQ CUSTOM_0 submission was not observed" >&2
    exit 1
}
for micro_op in fetch-descriptor read-operands execute-operator writeback complete; do
    grep -q "micro_op=${micro_op}.*ok=1" "${HOST_LOG}" || {
        echo "error: GPTQ micro-op did not complete: ${micro_op}" >&2
        exit 1
    }
done
grep -q 'name=gptq_matmul_int4' "${HOST_LOG}" || {
    echo "error: GPTQ numerical kernel did not complete" >&2
    exit 1
}

echo "GPTQ custom-instruction end-to-end smoke: PASS"
