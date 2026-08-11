#!/bin/sh
#
# phase2_build_bridge.sh — Build the standard Coral gem5 bridge and firmware.
#
# Compiles four Bazel targets from the Coral submodule using the base CoreMiniAxi
# configuration: one gem5 bridge shared library and three firmware ELF images.
# This is the standard (non-RVV) bridge used by most Phase 2–5 tests.
#
# For the RVV Highmem MobileNet variant, see build_rvv_mobilenet.sh.
#
# Targets:
#   //hw_sim:libcoralnpu_gem5_bridge.so    gem5 ↔ Verilator C ABI bridge
#   //hw_sim:gem5_smoke_halt.elf           minimal "run to halt" firmware
#   //hw_sim:gem5_dma_smoke.elf            coherent DMA test firmware
#   //hw_sim:gem5_command_smoke.elf        command descriptor firmware
#
# Pipeline:
#   phase2_check_abi.sh                  ← ABI header consistency
#   check_command_abi.sh                 ← command ABI consistency
#   apply_patchset.sh (coral)            ← overlay sim/coralnpu → thirdparty/coralnpu
#   phase2_check_overlay_boundary.sh     ← verify no upstream conflicts
#   phase2_test_axi_adapter.sh           ← AXI master adapter unit tests
#   phase5_test_custom_rtl.sh            ← custom RTL accelerator tests
#     → Bazel build (4 targets)
#       → build/coralnpu/libcoralnpu_gem5_bridge.so
#       → build/coralnpu/gem5_smoke_halt.elf
#       → build/coralnpu/gem5_dma_smoke.elf
#       → build/coralnpu/gem5_command_smoke.elf
#
# Post-build checks:
#   - No unresolved sram_(init|read|write) backdoor symbols (stale bridge)
#   - No libsystemc linkage (would conflict with gem5's own SystemC)
#
# Environment:
#   BAZEL                    Bazel executable (auto-detected)
#   CORAL_REPO               Coral submodule path (default: thirdparty/coralnpu)
#   PHASE2_BAZEL_OUTPUT_ROOT Bazel output base (default: .cache/coralnpu/bazel)
#   PHASE2_REPO_CACHE        Bazel repository cache
#   PHASE2_DISTDIR           Bazel distdir for offline deps
#
# Extra CLI args are forwarded to bazel build and cquery. The compilation
# mode defaults to "-c opt"; pass "-c fastbuild" or "-c dbg" to override.
#
# @coral-build-spec  v1  2025-07-29

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CORAL_REPO="${CORAL_REPO:-${ROOT_DIR}/thirdparty/coralnpu}"
TARGET="//hw_sim:libcoralnpu_gem5_bridge.so"
FIRMWARE_TARGET="//hw_sim:gem5_smoke_halt.elf"
DMA_FIRMWARE_TARGET="//hw_sim:gem5_dma_smoke.elf"
COMMAND_FIRMWARE_TARGET="//hw_sim:gem5_command_smoke.elf"
OUT_DIR="${ROOT_DIR}/build/coralnpu"
LOCAL_BAZEL="${ROOT_DIR}/.cache/coralnpu/bin/bazel"
BAZEL_OUTPUT_ROOT="${PHASE2_BAZEL_OUTPUT_ROOT:-${ROOT_DIR}/.cache/coralnpu/bazel}"
REPO_CACHE="${PHASE2_REPO_CACHE:-${ROOT_DIR}/.cache/coralnpu/repository}"
DISTDIR="${PHASE2_DISTDIR:-${CORAL_REPO}/distdir}"

# Default to an optimized build. The bridge contains the Verilated RTL model,
# and a default fastbuild (-O0) runs it ~14x slower; pass an explicit mode to
# override, e.g. "phase2_build_bridge.sh -c fastbuild" or "-c dbg".
case " $* " in
    *" -c "*|*" --compilation_mode"*) ;;
    *) set -- -c opt "$@" ;;
esac

# ---------------------------------------------------------------------------
# Resolve Bazel: explicit BAZEL env, system PATH, or local install.
# ---------------------------------------------------------------------------
if [ -n "${BAZEL:-}" ]; then
    :
elif command -v bazel >/dev/null 2>&1; then
    BAZEL="$(command -v bazel)"
elif [ -x "${LOCAL_BAZEL}" ]; then
    BAZEL="${LOCAL_BAZEL}"
else
    cat >&2 <<EOF
error: bazel not found

Prepare the repository-local Bazel executable first:
  ./tools/coralnpu/phase2_prepare_bazel.sh

For an offline/DNS-restricted host:
  BAZEL_BINARY=/path/to/bazel-\$(cat "${CORAL_REPO}/.bazelversion")-linux-x86_64 ./tools/coralnpu/phase2_prepare_bazel.sh
EOF
    exit 1
fi

# ---------------------------------------------------------------------------
# Pre-build checks: ABI consistency, Coral overlay, unit tests.
#   These guards catch mismatches before the (slow) Bazel build starts.
# ---------------------------------------------------------------------------
"${ROOT_DIR}/tools/coralnpu/phase2_check_abi.sh"
"${ROOT_DIR}/tools/coralnpu/check_command_abi.sh"
"${ROOT_DIR}/sim/coralnpu/apply_patchset.sh"
"${ROOT_DIR}/tools/coralnpu/phase2_check_overlay_boundary.sh"
"${ROOT_DIR}/tools/coralnpu/phase2_test_axi_adapter.sh"
"${ROOT_DIR}/tools/coralnpu/test_coprocessor_command.sh"
"${ROOT_DIR}/tools/coralnpu/phase5_test_custom_rtl.sh"

# ---------------------------------------------------------------------------
# Ensure Bazel cache and output directories exist.
#   Remove stale output files so a partial build cannot leave outdated
#   artifacts that downstream scripts mistakenly use.
# ---------------------------------------------------------------------------
mkdir -p "${BAZEL_OUTPUT_ROOT}" "${REPO_CACHE}" "${DISTDIR}"
rm -f \
    "${OUT_DIR}/libcoralnpu_gem5_bridge.so" \
    "${OUT_DIR}/gem5_smoke_halt.elf" \
    "${OUT_DIR}/gem5_dma_smoke.elf" \
    "${OUT_DIR}/gem5_command_smoke.elf" \
    "${OUT_DIR}/wfi_slot_0.elf"

# ---------------------------------------------------------------------------
# Build all four targets.
#
# Bazel's JVM-based downloader does not always respect HTTPS_PROXY.  On hosts
# behind a firewall, this causes "Connect timed out" failures.  When that
# happens, we pre-download every declared dependency into the distdir via wget
# (which does respect the proxy), then retry.
#
# Build output is saved to a timestamped log in logs/build/ for CI archiving.
# ---------------------------------------------------------------------------
BUILD_TS="$(date +%Y-%m-%dT%H-%M-%S)"
BUILD_LOG="${ROOT_DIR}/logs/build/coralnpu-bridge-build-${BUILD_TS}.log"
BUILD_LATEST="${ROOT_DIR}/logs/build/coralnpu-bridge-build.log"
mkdir -p "$(dirname "${BUILD_LOG}")"
ln -sfn "coralnpu-bridge-build-${BUILD_TS}.log" "${BUILD_LATEST}"

# Run a command, saving combined output to the build log while showing it on
# the terminal.  Returns the command's exit code.
_capture() {
    outfile="$1"; shift
    ecfile="$(mktemp)"
    { "$@" 2>&1; printf '%d\n' $? >"${ecfile}"; } | tee -a "${outfile}" >&2
    read -r ec <"${ecfile}"
    rm -f "${ecfile}"
    return "${ec}"
}

cd "${CORAL_REPO}"
bazel_ok=0

echo "[bazel] build started $(date -Iseconds)" | tee -a "${BUILD_LOG}"

if _capture "${BUILD_LOG}" \
       "${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" build \
       --repository_cache="${REPO_CACHE}" \
       --distdir="${DISTDIR}" \
       "${TARGET}" "${FIRMWARE_TARGET}" "${DMA_FIRMWARE_TARGET}" \
       "${COMMAND_FIRMWARE_TARGET}" "$@"; then
    bazel_ok=1
elif grep -qE 'Connect timed out|Closed by interrupt|Error downloading' "${BUILD_LOG}" 2>/dev/null; then
    echo "[bazel] network download failed; pre-fetching all dependencies with wget" | tee -a "${BUILD_LOG}" >&2
    "${ROOT_DIR}/tools/kernel/download_bazel_deps.sh" 2>&1 | tee -a "${BUILD_LOG}"
    echo "[bazel] retrying build with populated distdir" | tee -a "${BUILD_LOG}" >&2
    if _capture "${BUILD_LOG}" \
           "${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" build \
           --repository_cache="${REPO_CACHE}" \
           --distdir="${DISTDIR}" \
           "${TARGET}" "${FIRMWARE_TARGET}" "${DMA_FIRMWARE_TARGET}" \
           "${COMMAND_FIRMWARE_TARGET}" "$@"; then
        bazel_ok=1
    fi
fi

if [ "${bazel_ok}" -eq 0 ]; then
    echo "[bazel] BUILD FAILED $(date -Iseconds)" | tee -a "${BUILD_LOG}" >&2
    echo "error: Bazel build failed (see above)" >&2
    echo "build log: ${BUILD_LOG}" >&2
    exit 1
fi
echo "[bazel] build succeeded $(date -Iseconds)" | tee -a "${BUILD_LOG}"

# ---------------------------------------------------------------------------
# Resolve Bazel output paths.
#   cquery returns absolute or execution-root-relative paths.
#   BUILD_EXTRA_ARGS must mirror the build invocation so cquery resolves the
#   same configuration (e.g. "-c opt"); otherwise it silently returns the
#   default fastbuild output path and the build's extra flags are ignored.
# ---------------------------------------------------------------------------
BUILD_EXTRA_ARGS="$*"

EXEC_ROOT="$("${BAZEL}" \
    --output_user_root="${BAZEL_OUTPUT_ROOT}" \
    info execution_root)"

resolve_output()
{
    target="$1"
    output="$("${BAZEL}" \
        --output_user_root="${BAZEL_OUTPUT_ROOT}" \
        cquery \
        --repository_cache="${REPO_CACHE}" \
        --distdir="${DISTDIR}" \
        ${BUILD_EXTRA_ARGS} \
        --output=files \
        "${target}")"
    case "${output}" in
        /*) printf '%s\n' "${output}" ;;
        *) printf '%s\n' "${EXEC_ROOT}/${output}" ;;
    esac
}

BRIDGE="$(resolve_output "${TARGET}")"
FIRMWARE="$(resolve_output "${FIRMWARE_TARGET}")"
DMA_FIRMWARE="$(resolve_output "${DMA_FIRMWARE_TARGET}")"
COMMAND_FIRMWARE="$(resolve_output "${COMMAND_FIRMWARE_TARGET}")"

# ---------------------------------------------------------------------------
# Verify Bazel produced all four artifacts.
# ---------------------------------------------------------------------------
if [ ! -f "${BRIDGE}" ]; then
    echo "error: Bazel completed but bridge was not found: ${BRIDGE}" >&2
    exit 1
fi
if [ ! -f "${FIRMWARE}" ]; then
    echo "error: Bazel completed but firmware was not found: ${FIRMWARE}" >&2
    exit 1
fi
if [ ! -f "${DMA_FIRMWARE}" ]; then
    echo "error: Bazel completed but DMA firmware was not found: ${DMA_FIRMWARE}" >&2
    exit 1
fi
if [ ! -f "${COMMAND_FIRMWARE}" ]; then
    echo "error: Bazel completed but command firmware was not found: ${COMMAND_FIRMWARE}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Install artifacts into build/coralnpu/ with atomic temporary-file writes.
#   Writing to a temp file then mv-ing prevents a concurrent reader from
#   seeing a partially-written .so or .elf.
# ---------------------------------------------------------------------------
mkdir -p "${OUT_DIR}"
if [ ! -w "${OUT_DIR}" ]; then
    echo "error: output directory is not writable: ${OUT_DIR}" >&2
    echo "repair ownership once with:" >&2
    echo "  sudo chown -R \"${USER:-$(id -un)}:$(id -gn)\" \"${OUT_DIR}\"" >&2
    exit 1
fi

install_output()
{
    source_path="$1"
    destination_path="$2"
    mode="$3"
    temporary_path="$(mktemp "${OUT_DIR}/.$(basename "${destination_path}").XXXXXX")"
    if ! cp "${source_path}" "${temporary_path}" ||
       ! chmod "${mode}" "${temporary_path}" ||
       ! mv -f "${temporary_path}" "${destination_path}"; then
        rm -f "${temporary_path}"
        echo "error: unable to install artifact: ${destination_path}" >&2
        return 1
    fi
}

install_output "${BRIDGE}" "${OUT_DIR}/libcoralnpu_gem5_bridge.so" 0755
install_output "${FIRMWARE}" "${OUT_DIR}/gem5_smoke_halt.elf" 0644
install_output "${DMA_FIRMWARE}" "${OUT_DIR}/gem5_dma_smoke.elf" 0644
install_output "${COMMAND_FIRMWARE}" \
    "${OUT_DIR}/gem5_command_smoke.elf" 0644

# ---------------------------------------------------------------------------
# Post-build sanity checks.
# ---------------------------------------------------------------------------

# Bridge must not have unresolved SRAM backdoor symbols — these come from
# a stale build that wasn't rebuilt after a Coral overlay update.
if command -v nm >/dev/null 2>&1 &&
   nm -D --undefined-only \
       "${OUT_DIR}/libcoralnpu_gem5_bridge.so" 2>/dev/null |
       grep -Eq '[[:space:]]sram_(init|read|write)$'; then
    echo "error: Coral gem5 bridge has unresolved SRAM backdoor symbols" >&2
    exit 1
fi

# Bridge must not link libsystemc.  gem5 already links SystemC; a second
# copy loaded via dlopen would cause symbol conflicts and undefined behavior.
if command -v ldd >/dev/null 2>&1 &&
   ldd "${OUT_DIR}/libcoralnpu_gem5_bridge.so" 2>/dev/null |
       grep -qi systemc; then
    echo "error: Coral gem5 bridge unexpectedly links libsystemc" >&2
    echo "       rerun after applying the current Coral overlay" >&2
    exit 1
fi

echo "built: ${OUT_DIR}/libcoralnpu_gem5_bridge.so" | tee -a "${BUILD_LOG}"
echo "built: ${OUT_DIR}/gem5_smoke_halt.elf" | tee -a "${BUILD_LOG}"
echo "built: ${OUT_DIR}/gem5_dma_smoke.elf" | tee -a "${BUILD_LOG}"
echo "built: ${OUT_DIR}/gem5_command_smoke.elf" | tee -a "${BUILD_LOG}"
