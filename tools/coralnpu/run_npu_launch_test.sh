#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BRIDGE="${ROOT_DIR}/build/coralnpu/libcoralnpu_gem5_rvv_highmem_bridge.so"
FIRMWARE="${CORAL_NPU_LAUNCH_FIRMWARE:-${ROOT_DIR}/build/coralnpu/gem5_npu_launch_smoke.elf}"
TEST_SCRIPT="${ROOT_DIR}/thirdparty/gem5/configs/coralnpu/coral-npu-launch-test.rcS"
HOST_LOG="${CORAL_NPU_LAUNCH_HOST_LOG:-${ROOT_DIR}/simout/coral-npu-launch-host.log}"
DEBUG_LOG="${CORAL_NPU_LAUNCH_DEBUG_LOG:-${ROOT_DIR}/simout/coral-npu-launch.debug}"
CKPT_ROOT="${CORAL_NPU_LAUNCH_CKPT_ROOT:-${ROOT_DIR}/checkpoint/coralnpu_mobilenet_ckpt}"
SHARED_BASE="${CORAL_NPU_LAUNCH_SHARED_BASE:-0x8f000000}"
KERNEL_RELEASE_FILE="${ROOT_DIR}/build/kernel/kernel.release"

[ -f "${BRIDGE}" ] || {
    echo "error: RVV highmem bridge not found: ${BRIDGE}" >&2
    echo "run ./tools/coralnpu/build_rvv_mobilenet.sh -c opt first" >&2
    exit 1
}
[ -f "${FIRMWARE}" ] || {
    echo "error: NPU_LAUNCH firmware not found: ${FIRMWARE}" >&2
    echo "run ./tools/coralnpu/build_npu_launch_smoke.sh first" >&2
    exit 1
}

if [ -z "${CORAL_KERNEL_IMAGE:-}" ]; then
    [ -f "${KERNEL_RELEASE_FILE}" ] || {
        echo "error: set CORAL_KERNEL_IMAGE or build the validated kernel" >&2
        exit 1
    }
    KERNEL_RELEASE="$(cat "${KERNEL_RELEASE_FILE}")"
    CORAL_KERNEL_IMAGE="${ROOT_DIR}/build/kernel/vmlinux-${KERNEL_RELEASE}"
fi
CORAL_KERNEL_INIT="${CORAL_KERNEL_INIT:-/sbin/opennpux-init.sh}"
CORAL_DISK_IMG="${CORAL_DISK_IMG:-${HOME}/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img}"
[ -f "${CORAL_KERNEL_IMAGE}" ] || {
    echo "error: kernel not found: ${CORAL_KERNEL_IMAGE}" >&2
    exit 1
}
[ -f "${CORAL_DISK_IMG}" ] || {
    echo "error: disk image not found: ${CORAL_DISK_IMG}" >&2
    exit 1
}

mkdir -p "$(dirname "${HOST_LOG}")" "$(dirname "${DEBUG_LOG}")"
"${ROOT_DIR}/sim/gem5/apply_patchset.sh"

STATUS_FILE="$(mktemp)"
trap 'rm -f "${STATUS_FILE}"' EXIT
set +e
(
    CORAL_NPU_BACKEND=verilated-coral \
    CORAL_RTL_BRIDGE="${BRIDGE}" \
    CORAL_RTL_FIRMWARE="${FIRMWARE}" \
    CORAL_RTL_CYCLES_PER_EVENT="${CORAL_RTL_CYCLES_PER_EVENT:-1000}" \
    CORAL_KERNEL_IMAGE="${CORAL_KERNEL_IMAGE}" \
    CORAL_KERNEL_INIT="${CORAL_KERNEL_INIT}" \
    CORAL_DISK_IMG="${CORAL_DISK_IMG}" \
    CORAL_AUTO_RESUME_AFTER_CKPT=1 \
    CORAL_CKPT_ROOT="${CKPT_ROOT}" \
    CORAL_RESUME_BOOTSCRIPT="${TEST_SCRIPT}" \
    CORAL_CONFIG_OPTIONS="${CORAL_CONFIG_OPTIONS:-} --npu-dma-shared-base=${SHARED_BASE} --npu-dma-shared-size=8MiB --npu-operator-mode=hybrid --npu-fast-dma" \
    GEM5_OPTIONS="${GEM5_OPTIONS:-} --debug-flags=NPUDevice --debug-file=${DEBUG_LOG}" \
    "${ROOT_DIR}/thirdparty/gem5/run_multicore.sh"
    printf '%s\n' "$?" > "${STATUS_FILE}"
) 2>&1 | tee "${HOST_LOG}"
PIPE_STATUS=$?
set -e
RUN_STATUS="${PIPE_STATUS}"
[ ! -s "${STATUS_FILE}" ] || RUN_STATUS="$(cat "${STATUS_FILE}")"
[ "${RUN_STATUS}" -eq 0 ] || exit "${RUN_STATUS}"

grep -q 'source=custom-instruction' "${HOST_LOG}" || {
    echo "error: no custom-instruction submission observed" >&2
    exit 1
}
for micro_op in fetch-descriptor read-operands execute-operator writeback complete; do
    grep -q "micro_op=${micro_op}" "${HOST_LOG}" || {
        echo "error: missing micro-op completion: ${micro_op}" >&2
        exit 1
    }
done
grep -q 'kernel=done' "${HOST_LOG}" || {
    echo "error: hybrid ADD kernel did not complete" >&2
    exit 1
}
if [ -n "${CORAL_NPU_LAUNCH_EXPECTED_OPCODE:-}" ]; then
    grep -q "operator_opcode=${CORAL_NPU_LAUNCH_EXPECTED_OPCODE}" \
        "${HOST_LOG}" || {
        echo "error: expected operator opcode was not executed: ${CORAL_NPU_LAUNCH_EXPECTED_OPCODE}" >&2
        exit 1
    }
fi

# gem5term records CRLF on some hosts. Also tolerate the legacy output
# locations used before run_multicore.sh standardized logs/sim/m5out.
TERMINAL=""
for candidate in \
    "${ROOT_DIR}/logs/sim/m5out/system.terminal" \
    "${ROOT_DIR}/m5out/system.terminal" \
    "${ROOT_DIR}/thirdparty/gem5/m5out/system.terminal"; do
    if [ -f "${candidate}" ] && awk '
        { sub(/\r$/, "") }
        $0 == "npu_launch_test=PASS" { found = 1 }
        END { exit found ? 0 : 1 }
    ' "${candidate}"; then
        TERMINAL="${candidate}"
        break
    fi
done
[ -n "${TERMINAL}" ] || {
    echo "error: guest NPU_LAUNCH verdict missing" >&2
    echo "checked guest terminal logs:" >&2
    for candidate in \
        "${ROOT_DIR}/logs/sim/m5out/system.terminal" \
        "${ROOT_DIR}/m5out/system.terminal" \
        "${ROOT_DIR}/thirdparty/gem5/m5out/system.terminal"; do
        [ -f "${candidate}" ] || continue
        echo "--- ${candidate}" >&2
        tail -n 30 "${candidate}" | tr -d '\r' >&2
    done
    exit 1
}

echo "guest verdict: ${TERMINAL}"
echo "NPU_LAUNCH end-to-end test: PASS"
