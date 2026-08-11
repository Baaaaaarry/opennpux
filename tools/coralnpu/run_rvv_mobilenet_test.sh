#!/bin/sh
#
# run_rvv_mobilenet_test.sh — Run the RVV Highmem MobileNet acceptance test.
#
# Launches gem5 ARM full-system simulation with the RvvCoreMiniHighmemAxi
# Verilated Coral backend and executes the full mobilenet_v1_0.25_224_int8
# graph against LiteRT Micro reference kernels.
#
# The first invocation creates a dedicated boot checkpoint with an 8 MiB
# coherent DMA window at CORAL_MOBILENET_SHARED_BASE.  The second invocation
# (launched automatically via CORAL_AUTO_RESUME_AFTER_CKPT=1) restores from
# that checkpoint and executes the MobileNet firmware.
#
# Operator execution modes (CORAL_OPERATOR_MODE):
#   rtl      All operators run on the Verilated RISC-V RTL core.
#            Cycle counts are authoritative for NPU performance analysis.
#            Hours-long runtime for a full MobileNet graph.
#
#   hybrid   Supported compute operators (Conv2D, DepthwiseConv2D, etc.) are
#            dispatched through an EXTMEM doorbell to x86 host TFLite kernels.
#            Firmware control flow and tensor management still run through RTL.
#            Minutes-long runtime; modeled_cycles are estimates, not RTL cycles.
#
#   sampled  Firmware, driver, and doorbell path all run through RTL, but
#            long-running operators use hybrid host kernels so the full graph
#            completes in minutes.  Use CORAL_SAMPLED_RTL_OPS to force specific
#            operator classes back onto real RTL.
#
# The default mode is rtl (see CORAL_OPERATOR_MODE below), kept for backward
# compatibility; hybrid/sampled are the fast bring-up modes.
#
# Pipeline:
#   apply_patchset.sh  → overlay sim/gem5 into thirdparty/gem5
#     → run_multicore.sh  → scons + gem5 ARM FS boot
#       → checkpoint bootstrap (stage-a) → checkpoint restore (verilated-coral)
#         → coral-mobilenet-test.rcS → coralctl run
#
# Environment:
#   CORAL_OPERATOR_MODE        rtl | hybrid | sampled (default: rtl)
#   CORAL_SAMPLED_RTL_OPS      ops to force to RTL in sampled mode (default: none)
#   CORAL_FAST_DMA             1 = functional fast path, 0 = timing (default: 1)
#   CORAL_FAST_DMA_EVENT_BATCH batches per gem5 event (default: 4096)
#   CORAL_RTL_CYCLES_PER_EVENT cycles per Verilator step (default: 1000)
#   CORAL_HYBRID_OPS_PER_CYCLE hybrid latency model (default: 1)
#   CORAL_HYBRID_BYTES_PER_CYCLE  hybrid latency model (default: 16)
#   CORAL_HYBRID_FIXED_CYCLES  hybrid latency model fixed cost (default: 0)
#   CORAL_KERNEL_IMAGE         vmlinux ELF (auto-detected from build/kernel/)
#   CORAL_KERNEL_INIT          guest init path (default: /sbin/opennpux-init.sh)
#   CORAL_DISK_IMG             ARM64 disk image path
#   CORAL_MOBILENET_DEBUG      1 = enable NPUDevice debug trace
#   CORAL_MOBILENET_HOST_LOG   host output log (default: logs/sim/coral-mobilenet-host-<ts>.log)
#   CORAL_MOBILENET_CKPT_ROOT  checkpoint directory (default: checkpoint/coralnpu_mobilenet_ckpt)
#   CORAL_MOBILENET_SHARED_BASE  DMA window base address (default: 0x8f000000)
#
# Expected output:
#   mobilenet_test=PASS
#   mobilenet_output_checksum=<non-zero>
#   mobilenet_dma_errors=0
#
# @coral-build-spec  v1  2025-07-29

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

# ---------------------------------------------------------------------------
# Host-side output capture and wall-clock timing.
#   Re-exec through tee so every byte of console output (this script,
#   apply_patchset, run_multicore.sh, gem5, and the bridge's stderr) lands in
#   CORAL_MOBILENET_HOST_LOG while still printing to the console.  On exit —
#   normal, error, or Ctrl+C / SIGINT — a footer with the start/end wall
#   times and elapsed seconds is appended to both destinations.
#   To interrupt: press Ctrl+C, or run: pkill -INT -f 'build/ARM/gem5.opt'
#   (gem5 dumps stats on SIGINT; the footer prints in both cases).
# ---------------------------------------------------------------------------
if [ -z "${CORAL_MOBILENET_TEE_ACTIVE:-}" ]; then
    LOG_DIR="${ROOT_DIR}/logs/sim"
    mkdir -p "${LOG_DIR}"
    HOST_LOG="${CORAL_MOBILENET_HOST_LOG:-${LOG_DIR}/coral-mobilenet-host-$(date +%Y%m%d-%H%M%S).log}"
    ln -sfn "$(basename "${HOST_LOG}")" "${LOG_DIR}/coral-mobilenet-host.log"
    _start_epoch="$(date +%s)"
    _status_file="$(mktemp)"
    # Snapshot the guest terminal's mtime so the footer only reports a
    # verdict produced by THIS run. gem5 rewrites (truncates) the file at
    # startup, so a line-count baseline breaks when the new log is shorter;
    # an mtime comparison is robust for both rewrite and preflight-failure.
    _term="${LOG_DIR}/m5out/system.terminal"
    _term_mtime=0
    [ -f "${_term}" ] && _term_mtime="$(stat -c %Y "${_term}")"
    {
        printf '[coral-mobilenet] host log: %s\n' "${HOST_LOG}"
        printf '[coral-mobilenet] start: %s\n' "$(date -Is)"
        printf '[coral-mobilenet] interrupt: Ctrl+C, or: pkill -INT -f build/ARM/gem5.opt\n'
    } | tee "${HOST_LOG}"
    trap '' INT   # gem5 gets Ctrl+C itself via the process group; survive to print the footer
    (
        trap 'printf "%s\n" "$?" > "${_status_file}"' EXIT
        CORAL_MOBILENET_TEE_ACTIVE=1 "$0" "$@"
    ) 2>&1 | (trap '' INT; exec tee -a "${HOST_LOG}")
    if [ -s "${_status_file}" ]; then
        _rc="$(cat "${_status_file}")"
    else
        _rc=130    # interrupted before the inner script could report
    fi
    rm -f "${_status_file}"
    _end_epoch="$(date +%s)"
    _elapsed=$(( _end_epoch - _start_epoch ))
    _verdict=""
    if [ -f "${_term}" ] && \
       [ "$(stat -c %Y "${_term}")" -gt "${_term_mtime}" ]; then
        _verdict="$(grep -m1 'mobilenet_test=' "${_term}" || true)"
    fi
    {
        printf '[coral-mobilenet] start: %s\n' "$(date -Is -d "@${_start_epoch}")"
        printf '[coral-mobilenet] end: %s elapsed=%ss (%dh %dm %ds) exit=%s\n' \
            "$(date -Is -d "@${_end_epoch}")" "${_elapsed}" \
            "$((_elapsed / 3600))" "$((_elapsed % 3600 / 60))" \
            "$((_elapsed % 60))" "${_rc}"
        if [ -n "${_verdict}" ]; then
            printf '[coral-mobilenet] guest verdict: %s (from system.terminal)\n' "${_verdict}"
        else
            printf '[coral-mobilenet] guest verdict: (no new terminal output this run)\n'
        fi
        printf '[coral-mobilenet] gem5 stats: %s\n' "${LOG_DIR}/m5out/stats.txt"
    } | tee -a "${HOST_LOG}"
    exit "${_rc}"
fi

# ---------------------------------------------------------------------------
# Fixed paths: bridge, firmware, and resume script.
# ---------------------------------------------------------------------------
BRIDGE="${ROOT_DIR}/build/coralnpu/libcoralnpu_gem5_rvv_highmem_bridge.so"
FIRMWARE="${ROOT_DIR}/build/coralnpu/gem5_mobilenet.elf"
TEST_SCRIPT="${ROOT_DIR}/thirdparty/gem5/configs/coralnpu/coral-mobilenet-test.rcS"

# ---------------------------------------------------------------------------
# Tunable parameters — all overridable via environment.
# ---------------------------------------------------------------------------
GEM5_OPTIONS_VALUE="${GEM5_OPTIONS:-}"
RTL_CYCLES_PER_EVENT="${CORAL_RTL_CYCLES_PER_EVENT:-1000}"
FAST_DMA_EVENT_BATCH="${CORAL_FAST_DMA_EVENT_BATCH:-4096}"
OPERATOR_MODE="${CORAL_OPERATOR_MODE:-rtl}"
SAMPLED_RTL_OPS="${CORAL_SAMPLED_RTL_OPS:-none}"
HYBRID_OPS_PER_CYCLE="${CORAL_HYBRID_OPS_PER_CYCLE:-1}"
HYBRID_BYTES_PER_CYCLE="${CORAL_HYBRID_BYTES_PER_CYCLE:-16}"
HYBRID_FIXED_CYCLES="${CORAL_HYBRID_FIXED_CYCLES:-0}"

# Fast DMA is enabled by default for functional inference throughput.
# Set CORAL_FAST_DMA=0 to select timing-DMA mode for cycle studies.
FAST_DMA_OPTION=""
[ "${CORAL_FAST_DMA:-1}" != "1" ] || FAST_DMA_OPTION=" --npu-fast-dma"

# ---------------------------------------------------------------------------
# Resolve kernel image from build/kernel/kernel.release if not provided.
# ---------------------------------------------------------------------------
KERNEL_RELEASE_FILE="${ROOT_DIR}/build/kernel/kernel.release"
VALIDATED_DISK_DEFAULT="${HOME}/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img"
CKPT_ROOT="${CORAL_MOBILENET_CKPT_ROOT:-${ROOT_DIR}/checkpoint/coralnpu_mobilenet_ckpt}"
DMA_SHARED_BASE="${CORAL_MOBILENET_SHARED_BASE:-0x8f000000}"
DMA_SHARED_BASE_META="${CKPT_ROOT}.shared_base"
DT_ABI_VERSION=2
DT_ABI_META="${CKPT_ROOT}.dt_abi"

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

# ---------------------------------------------------------------------------
# Report the effective configuration.
# ---------------------------------------------------------------------------
echo "[coral-mobilenet] kernel: ${CORAL_KERNEL_IMAGE}"
echo "[coral-mobilenet] init: ${CORAL_KERNEL_INIT}"
echo "[coral-mobilenet] disk: ${CORAL_DISK_IMG}"

if [ "${CORAL_MOBILENET_DEBUG:-0}" = "1" ]; then
    DEBUG_LOG="${CORAL_MOBILENET_DEBUG_LOG:-${ROOT_DIR}/logs/sim/coral-mobilenet.debug}"
    mkdir -p "$(dirname "${DEBUG_LOG}")"
    GEM5_OPTIONS_VALUE="${GEM5_OPTIONS_VALUE} --debug-flags=NPUDevice --debug-file=${DEBUG_LOG}"
    echo "[coral-mobilenet] debug log: ${DEBUG_LOG}"
fi

echo "[coral-mobilenet] RTL cycles per event: ${RTL_CYCLES_PER_EVENT}"
case "${OPERATOR_MODE}" in
    rtl|hybrid|sampled) ;;
    *) echo "error: CORAL_OPERATOR_MODE must be rtl, hybrid, or sampled" >&2; exit 1 ;;
esac
echo "[coral-mobilenet] operator mode: ${OPERATOR_MODE}"
if [ "${OPERATOR_MODE}" = "hybrid" ] || [ "${OPERATOR_MODE}" = "sampled" ]; then
    echo "[coral-mobilenet] hybrid latency: ops_per_cycle=${HYBRID_OPS_PER_CYCLE} bytes_per_cycle=${HYBRID_BYTES_PER_CYCLE} fixed_cycles=${HYBRID_FIXED_CYCLES}"
fi
if [ "${OPERATOR_MODE}" = "sampled" ]; then
    echo "[coral-mobilenet] sampled RTL ops: ${SAMPLED_RTL_OPS}"
fi
if [ -n "${FAST_DMA_OPTION}" ]; then
    echo "[coral-mobilenet] DMA mode: functional-fast"
    echo "[coral-mobilenet] fast DMA event batch: ${FAST_DMA_EVENT_BATCH}"
else
    echo "[coral-mobilenet] DMA mode: timing"
fi
[ -z "${GEM5_OPTIONS_VALUE}" ] || \
    echo "[coral-mobilenet] gem5 options: ${GEM5_OPTIONS_VALUE}"

# ---------------------------------------------------------------------------
# Preflight: bridge and firmware must have been built.
# ---------------------------------------------------------------------------
[ -f "${BRIDGE}" ] || {
    echo "error: RVV highmem bridge not found: ${BRIDGE}" >&2
    exit 1
}
[ -f "${FIRMWARE}" ] || {
    echo "error: MobileNet firmware not found: ${FIRMWARE}" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Checkpoint management: invalidate when the shared DMA base changes.
# The MobileNet configuration uses a dedicated 8 MiB window at a different
# address from the standard 4 KiB command-test window.
# ---------------------------------------------------------------------------
if [ -f "${CKPT_ROOT}/booted/m5.cpt" ] &&
   { [ ! -f "${DMA_SHARED_BASE_META}" ] ||
     [ "$(cat "${DMA_SHARED_BASE_META}")" != "${DMA_SHARED_BASE}" ]; }; then
    echo "[coral-mobilenet] shared DMA base changed; rebuilding checkpoint"
    rm -rf "${CKPT_ROOT}"
fi
if [ -f "${CKPT_ROOT}/booted/m5.cpt" ] &&
   { [ ! -f "${DT_ABI_META}" ] ||
     [ "$(cat "${DT_ABI_META}")" != "${DT_ABI_VERSION}" ]; }; then
    echo "[coral-mobilenet] Coral DT ABI changed; rebuilding checkpoint"
    rm -rf "${CKPT_ROOT}"
fi
mkdir -p "$(dirname "${DMA_SHARED_BASE_META}")"
printf '%s\n' "${DMA_SHARED_BASE}" > "${DMA_SHARED_BASE_META}"

# ---------------------------------------------------------------------------
# Apply gem5 overlay and launch.
#   CORAL_AUTO_RESUME_AFTER_CKPT=1 tells run_multicore.sh to automatically
#   resume from a freshly-created checkpoint, so MobileNet execution begins
#   without a second manual invocation.
# ---------------------------------------------------------------------------
"${ROOT_DIR}/sim/gem5/apply_patchset.sh"

# Wall-clock timing is reported by the tee wrapper's footer at the top of
# this script; no separate timing file is needed here.
CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_BRIDGE="${BRIDGE}" \
CORAL_RTL_FIRMWARE="${FIRMWARE}" \
CORAL_RTL_CYCLES_PER_EVENT="${RTL_CYCLES_PER_EVENT}" \
CORAL_HYBRID_OPS_PER_CYCLE="${HYBRID_OPS_PER_CYCLE}" \
CORAL_HYBRID_BYTES_PER_CYCLE="${HYBRID_BYTES_PER_CYCLE}" \
CORAL_HYBRID_FIXED_CYCLES="${HYBRID_FIXED_CYCLES}" \
CORAL_SAMPLED_RTL_OPS="${SAMPLED_RTL_OPS}" \
CORAL_KERNEL_IMAGE="${CORAL_KERNEL_IMAGE}" \
CORAL_KERNEL_INIT="${CORAL_KERNEL_INIT}" \
CORAL_DISK_IMG="${CORAL_DISK_IMG}" \
CORAL_AUTO_RESUME_AFTER_CKPT=1 \
CORAL_CKPT_ROOT="${CKPT_ROOT}" \
CORAL_RESUME_BOOTSCRIPT="${TEST_SCRIPT}" \
CORAL_CONFIG_OPTIONS="${CORAL_CONFIG_OPTIONS:-} --npu-dma-shared-base=${DMA_SHARED_BASE} --npu-dma-shared-size=8MiB --npu-operator-mode=${OPERATOR_MODE}${FAST_DMA_OPTION} --npu-fast-dma-event-batch=${FAST_DMA_EVENT_BATCH}" \
GEM5_OPTIONS="${GEM5_OPTIONS_VALUE}" \
"${ROOT_DIR}/thirdparty/gem5/run_multicore.sh"
printf '%s\n' "${DT_ABI_VERSION}" > "${DT_ABI_META}"
# No exec: the script's exit status must propagate run_multicore.sh's status
# to the tee wrapper at the top of this file.
