#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
GEM5_ROOT="${ROOT_DIR}/thirdparty/gem5"
BRIDGE="${CORAL_RTL_BRIDGE:-${ROOT_DIR}/build/coralnpu/libcoralnpu_gem5_bridge.so}"
FIRMWARE="${CORAL_RTL_FIRMWARE:-${ROOT_DIR}/build/coralnpu/gem5_tmma_smoke.elf}"
TEST_SCRIPT="${ROOT_DIR}/thirdparty/gem5/configs/coralnpu/coralctl-test.rcS"
HOST_LOG="${CORAL_TMMA_HOST_LOG:-${ROOT_DIR}/simout/coral-tmma-host.log}"
DEBUG_LOG="${CORAL_TMMA_DEBUG_LOG:-${ROOT_DIR}/simout/coral-tmma.debug}"

for path in "${BRIDGE}" "${FIRMWARE}" "${TEST_SCRIPT}"; do
    [ -f "${path}" ] || {
        echo "error: required TMMA test file not found: ${path}" >&2
        echo "run ./tools/coralnpu/build_rtl_bridge.sh first" >&2
        exit 1
    }
done

mkdir -p "$(dirname "${HOST_LOG}")" "$(dirname "${DEBUG_LOG}")"
rm -f "${HOST_LOG}" "${DEBUG_LOG}"
"${ROOT_DIR}/sim/gem5/apply_patchset.sh"

cd "${GEM5_ROOT}"
CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_BRIDGE="${BRIDGE}" \
CORAL_RTL_FIRMWARE="${FIRMWARE}" \
CORAL_RESUME_BOOTSCRIPT="${TEST_SCRIPT}" \
CORAL_CKPT_ROOT="${CORAL_CKPT_ROOT:-${ROOT_DIR}/checkpoint/coralnpu_tmma_ckpt}" \
CORAL_AUTO_RESUME_AFTER_CKPT="${CORAL_AUTO_RESUME_AFTER_CKPT:-1}" \
CORAL_RTL_CYCLES_PER_EVENT="${CORAL_RTL_CYCLES_PER_EVENT:-1}" \
GEM5_OPTIONS="${GEM5_OPTIONS:-} --debug-flags=NPUDevice --debug-file=${DEBUG_LOG}" \
./run_multicore.sh 2>&1 | tee "${HOST_LOG}"

require_log() {
    pattern="$1"
    description="$2"
    if ! grep -Eq "${pattern}" "${HOST_LOG}"; then
        echo "error: TMMA validation missing ${description}" >&2
        grep -E 'Coral XOpenNPU (dispatch|accepted|complete|writeback|reject)|Coral RTL core fault' \
            "${HOST_LOG}" >&2 || true
        echo "host log: ${HOST_LOG}" >&2
        exit 1
    fi
}

require_log 'Coral XOpenNPU dispatch sequence=0 operation=tmma' \
    'TMMA L2 dispatch'
require_log 'Coral XOpenNPU complete sequence=0 .*error=0 .*macs=8 cycles=8' \
    'successful MMA completion'
require_log 'Coral XOpenNPU writeback sequence=0 destination=0x20000200 bytes=16 checksum=0xe6d7ed59 words=0x41980000/0x41b00000/0x422c0000/0x42480000' \
    'independently checked matrix writeback'
require_log 'Coral XOpenNPU dispatch sequence=1 operation=tfence' \
    'tfence completion ordering'
require_log '\[coralctl-test\] PASS' 'Guest completion verdict'

echo "TMMA custom-instruction full-system test: PASS"
