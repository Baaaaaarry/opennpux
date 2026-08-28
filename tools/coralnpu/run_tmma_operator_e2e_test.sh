#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
GEM5_ROOT="${ROOT_DIR}/thirdparty/gem5"
BRIDGE="${CORAL_RTL_BRIDGE:-${ROOT_DIR}/build/coralnpu/libcoralnpu_gem5_bridge.so}"
FIRMWARE="${CORAL_RTL_FIRMWARE:-${ROOT_DIR}/build/coralnpu/gem5_tmma_operator_e2e.elf}"
TEST_SCRIPT="${ROOT_DIR}/thirdparty/gem5/configs/coralnpu/coralctl-test.rcS"
HOST_LOG="${CORAL_TMMA_HOST_LOG:-${ROOT_DIR}/simout/coral-tmma-operator-e2e-host.log}"
DEBUG_LOG="${CORAL_TMMA_DEBUG_LOG:-${ROOT_DIR}/simout/coral-tmma-operator-e2e.debug}"

for path in "${BRIDGE}" "${FIRMWARE}" "${TEST_SCRIPT}"; do
    [ -f "${path}" ] || {
        echo "error: required TMMA operator test file not found: ${path}" >&2
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
CORAL_CKPT_ROOT="${CORAL_CKPT_ROOT:-${ROOT_DIR}/checkpoint/coralnpu_tmma_operator_ckpt}" \
CORAL_AUTO_RESUME_AFTER_CKPT="${CORAL_AUTO_RESUME_AFTER_CKPT:-1}" \
CORAL_RTL_CYCLES_PER_EVENT="${CORAL_RTL_CYCLES_PER_EVENT:-1}" \
CORAL_CONFIG_OPTIONS="${CORAL_CONFIG_OPTIONS:-} --npu-dma-shared-size=8MiB --npu-fast-dma --npu-fast-dma-event-batch=1" \
GEM5_OPTIONS="${GEM5_OPTIONS:-} --debug-flags=NPUDevice --debug-file=${DEBUG_LOG}" \
./run_multicore.sh 2>&1 | tee "${HOST_LOG}"

require_log() {
    pattern="$1"
    description="$2"
    if ! grep -Eq "${pattern}" "${HOST_LOG}"; then
        echo "error: TMMA operator validation missing ${description}" >&2
        grep -E 'Coral XOpenNPU (dispatch|accepted|complete|writeback|reject)|Coral RTL core fault' \
            "${HOST_LOG}" >&2 || true
        echo "host log: ${HOST_LOG}" >&2
        exit 1
    fi
}

reject_fault_log() {
    pattern="$1"
    description="$2"
    if grep -Eq "${pattern}" "${HOST_LOG}"; then
        echo "error: TMMA operator validation observed ${description}" >&2
        grep -E "${pattern}" "${HOST_LOG}" >&2 || true
        exit 1
    fi
}

require_log 'Coral XOpenNPU complete sequence=0 .*operation=tmma .*error=0 .*macs=12 .*cycles=12' \
    'case 0 completion'
require_log 'Coral XOpenNPU writeback sequence=0 destination=0x20000200 bytes=16 checksum=0xe6084308 words=0x42680000/0x42800000/0x430b0000/0x431a0000' \
    'case 0 rectangular matrix result'
require_log 'Coral XOpenNPU dispatch sequence=1 operation=tfence' \
    'case 0 completion fence'

require_log 'Coral XOpenNPU complete sequence=2 .*operation=tmma .*error=0 .*macs=24 .*cycles=24' \
    'case 1 completion'
require_log 'Coral XOpenNPU writeback sequence=2 destination=0x20000600 bytes=48 checksum=0x515811d8 words=0xc0c00000/0xc0000000/0xc0a00000/0x40e00000' \
    'case 1 signed rectangular matrix result'
require_log 'Coral XOpenNPU dispatch sequence=3 operation=tfence' \
    'case 1 completion fence'

require_log 'Coral XOpenNPU complete sequence=4 .*operation=tmma .*error=0 .*macs=12 .*cycles=12' \
    'case 2 completion'
require_log 'Coral XOpenNPU writeback sequence=4 destination=0x20000a00 bytes=12 checksum=0xacc1ee78 words=0x3f800000/0x40000000/0xc0600000/0000000000' \
    'case 2 vector-by-matrix result'
require_log 'Coral XOpenNPU dispatch sequence=5 operation=tfence' \
    'case 2 completion fence'

require_log 'Coral XOpenNPU dispatch sequence=6 operation=tadd' \
    'TADD L2 dispatch'
require_log 'Coral XOpenNPU complete sequence=6 .*operation=tadd .*error=0 .*elements=8 cycles=8' \
    'TADD functional completion'
require_log 'Coral XOpenNPU writeback sequence=6 destination=0x20000f00 bytes=32 checksum=0x16ace36b words=0x40400000/0xbf800000/0000000000/0x41000000' \
    'TADD independently checked result'
require_log 'Coral XOpenNPU dispatch sequence=7 operation=tfence' \
    'TADD completion fence'

require_log 'Coral XOpenNPU dispatch sequence=8 operation=tmul' \
    'TMUL L2 dispatch'
require_log 'Coral XOpenNPU complete sequence=8 .*operation=tmul .*error=0 .*elements=8 cycles=8' \
    'TMUL functional completion'
require_log 'Coral XOpenNPU writeback sequence=8 destination=0x20001200 bytes=32 checksum=0x2ac700dc words=0x40000000/0xc0000000/0xc1100000/0x41800000' \
    'TMUL independently checked result'
require_log 'Coral XOpenNPU dispatch sequence=9 operation=tfence' \
    'TMUL completion fence'

# Guest UART output can be redirected to system.terminal rather than the host
# log. The m5 exit event is the reliable host-side evidence that the restored
# guest script completed after the firmware self-check.
require_log 'm5_exit instruction encountered' 'guest-driven simulator exit'
reject_fault_log 'Coral XOpenNPU reject|Coral RTL core fault' 'reject or core fault'

echo "TMMA operator end-to-end test: PASS"
