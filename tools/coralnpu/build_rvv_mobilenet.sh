#!/bin/sh
#
# build_rvv_mobilenet.sh — Build the RVV Highmem bridge and MobileNet firmware.
#
# Compiles two Bazel targets from the Coral submodule using the RvvCoreMiniHighmemAxi
# configuration: a gem5 bridge shared library and an ELF firmware image containing
# the full upstream mobilenet_v1_0.25_224_int8_dummy.tflite graph.
#
# The RvvCoreMiniHighmemAxi configuration differs from the base CoreMiniAxi by
# enabling the RISC-V Vector (RVV) execution path and an 8 MiB coherent EXTMEM
# window.  The resulting bridge is a separate .so from the standard bridge
# (libcoralnpu_gem5_bridge.so) and the firmware is gem5_mobilenet.elf.
#
# Note: tflite_micro sources are compiled with vector code generation disabled
# (--per_file_copt, see TFLM_NOVEC_COPTS below) to work around a Coral RTL
# erratum: the core's external/ibus memory paths deadlock on vector
# loads/stores that span a 16-byte line boundary.
#
# Pipeline:
#   thirdparty/coralnpu (with sim/coralnpu overlay applied)
#     → Bazel build //hw_sim:libcoralnpu_gem5_rvv_highmem_bridge.so
#     → Bazel build //hw_sim:gem5_mobilenet.elf
#       → build/coralnpu/libcoralnpu_gem5_rvv_highmem_bridge.so
#       → build/coralnpu/gem5_mobilenet.elf
#
# Environment:
#   BAZEL                 Bazel executable (auto-detected from PATH or .cache)
#   CORAL_REPO            Coral submodule path (default: thirdparty/coralnpu)
#   PHASE2_BAZEL_OUTPUT_ROOT  Bazel output base (default: .cache/coralnpu/bazel)
#   PHASE2_REPO_CACHE     Bazel repository cache (default: .cache/coralnpu/repository)
#   PHASE2_DISTDIR        Bazel distdir for offline deps (default: thirdparty/coralnpu/distdir)
#
# Extra CLI args are forwarded to bazel build and cquery. The compilation
# mode defaults to "-c opt"; pass "-c fastbuild" or "-c dbg" to override.
#
# @coral-build-spec  v1  2025-07-28

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CORAL_REPO="${CORAL_REPO:-${ROOT_DIR}/thirdparty/coralnpu}"
BRIDGE_TARGET="//hw_sim:libcoralnpu_gem5_rvv_highmem_bridge.so"
FIRMWARE_TARGET="//hw_sim:gem5_mobilenet.elf"
OUT_DIR="${ROOT_DIR}/build/coralnpu"
LOCAL_BAZEL="${ROOT_DIR}/.cache/coralnpu/bin/bazel"
BAZEL_OUTPUT_ROOT="${PHASE2_BAZEL_OUTPUT_ROOT:-${ROOT_DIR}/.cache/coralnpu/bazel}"
REPO_CACHE="${PHASE2_REPO_CACHE:-${ROOT_DIR}/.cache/coralnpu/repository}"
DISTDIR="${PHASE2_DISTDIR:-${CORAL_REPO}/distdir}"

# Default to an optimized build. The bridge contains the Verilated RTL model,
# and a default fastbuild (-O0) runs it ~14x slower; pass an explicit mode to
# override, e.g. "build_rvv_mobilenet.sh -c fastbuild" (faster compile, debug
# info) or "-c dbg". The chosen flags also flow to the cquery resolution via
# BUILD_EXTRA_ARGS below.
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
    echo "error: bazel not found; run phase2_prepare_bazel.sh" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Pre-build checks: ABI consistency and Coral overlay.
# ---------------------------------------------------------------------------
"${ROOT_DIR}/tools/coralnpu/check_mobilenet_abi.sh"
"${ROOT_DIR}/sim/coralnpu/apply_patchset.sh"

# ---------------------------------------------------------------------------
# Ensure Bazel cache directories and output directory exist and are writable.
# ---------------------------------------------------------------------------
mkdir -p "${BAZEL_OUTPUT_ROOT}" "${REPO_CACHE}" "${DISTDIR}" "${OUT_DIR}"
if [ ! -w "${OUT_DIR}" ]; then
    echo "error: output directory is not writable: ${OUT_DIR}" >&2
    echo "repair ownership once with:" >&2
    echo "  sudo chown -R \"${USER:-$(id -un)}:$(id -gn)\" \"${OUT_DIR}\"" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build both targets: bridge .so and firmware .elf.
#
# Bazel's JVM-based downloader does not always respect HTTPS_PROXY.  On hosts
# behind a firewall, this causes "Connect timed out" failures.  When that
# happens, we pre-download every declared dependency into the distdir via wget
# (which does respect the proxy), then retry.  The retry finds everything
# locally and completes without network access.
# ---------------------------------------------------------------------------
# Build output (stdout + stderr) is saved to logs/build/coralnpu-build.log
# while also shown on the terminal in real time ("tee /dev/stderr" writes to
# both the log file and the terminal via stderr).
# Each build writes a timestamped log.  A convenience symlink points to the
# most recent one so "logs/build/coralnpu-build.log" always gives the latest.
BUILD_TS="$(date +%Y-%m-%dT%H-%M-%S)"
BUILD_LOG="${ROOT_DIR}/logs/build/coralnpu-build-${BUILD_TS}.log"
BUILD_LATEST="${ROOT_DIR}/logs/build/coralnpu-build.log"
mkdir -p "$(dirname "${BUILD_LOG}")"
ln -sfn "coralnpu-build-${BUILD_TS}.log" "${BUILD_LATEST}"

# Run a command, saving combined output to the build log while showing it on
# the terminal.  Returns the command's exit code (POSIX-compatible via a temp
# file, since we cannot use PIPESTATUS in /bin/sh).
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

# RTL erratum workaround: the Coral external/ibus memory paths deadlock on
# vector loads/stores that span a 16-byte line boundary. Auto-vectorized
# copies inside upstream TFLM reference kernels (PadEval PadParams copy,
# micro::GetTensorShape dims copy, ...) can hit such accesses on EXTMEM/ITCM
# addresses and wedge the core. Disable compiler vector code generation for
# tflite_micro sources only; coralnpu's own RVV kernels (intrinsics and
# rvv_opt helpers) stay vectorized. GCC emits vector code through three
# independent paths, so all must be suppressed: the loop/SLP vectorizers,
# loop-to-memcpy idiom recognition, and inline memcpy/memset expansion.
TFLM_NOVEC_COPTS="--per_file_copt=external/tflite_micro/.*@-fno-tree-vectorize"
TFLM_NOVEC_COPTS="${TFLM_NOVEC_COPTS} --per_file_copt=external/tflite_micro/.*@-fno-tree-slp-vectorize"
TFLM_NOVEC_COPTS="${TFLM_NOVEC_COPTS} --per_file_copt=external/tflite_micro/.*@-fno-tree-loop-distribute-patterns"
TFLM_NOVEC_COPTS="${TFLM_NOVEC_COPTS} --per_file_copt=external/tflite_micro/.*@-mstringop-strategy=scalar"

echo "[bazel] build started $(date -Iseconds)" | tee -a "${BUILD_LOG}"

if _capture "${BUILD_LOG}" \
       "${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" build \
       --repository_cache="${REPO_CACHE}" \
       --distdir="${DISTDIR}" \
       ${TFLM_NOVEC_COPTS} \
       "${BRIDGE_TARGET}" "${FIRMWARE_TARGET}" "$@"; then
    bazel_ok=1
elif grep -qE 'Connect timed out|Closed by interrupt|Error downloading' "${BUILD_LOG}" 2>/dev/null; then
    echo "[bazel] network download failed; pre-fetching all dependencies with wget" | tee -a "${BUILD_LOG}" >&2
    "${ROOT_DIR}/tools/kernel/download_bazel_deps.sh" 2>&1 | tee -a "${BUILD_LOG}"
    echo "[bazel] retrying build with populated distdir" | tee -a "${BUILD_LOG}" >&2
    if _capture "${BUILD_LOG}" \
           "${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" build \
           --repository_cache="${REPO_CACHE}" \
           --distdir="${DISTDIR}" \
           ${TFLM_NOVEC_COPTS} \
           "${BRIDGE_TARGET}" "${FIRMWARE_TARGET}" "$@"; then
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
# Resolve Bazel output paths (may be under the execution root or a convenience
# symlink tree — cquery tells us the absolute location).
# ---------------------------------------------------------------------------
EXEC_ROOT="$("${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" \
    info execution_root)"

# Extra CLI args ("$@") are forwarded to bazel build AND to the cquery
# resolution below, so the installed artifacts match the requested
# configuration (e.g. "-c opt" — otherwise cquery resolves the default
# fastbuild output path and the build's extra flags are silently ignored).
BUILD_EXTRA_ARGS="$*"

resolve_output()
{
    target="$1"
    output="$("${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" \
        cquery --repository_cache="${REPO_CACHE}" \
        --distdir="${DISTDIR}" ${BUILD_EXTRA_ARGS} --output=files "${target}")"
    case "${output}" in
        /*) printf '%s\n' "${output}" ;;
        *) printf '%s\n' "${EXEC_ROOT}/${output}" ;;
    esac
}

BRIDGE="$(resolve_output "${BRIDGE_TARGET}")"
FIRMWARE="$(resolve_output "${FIRMWARE_TARGET}")"

# ---------------------------------------------------------------------------
# Verify Bazel produced both artifacts.
# ---------------------------------------------------------------------------
[ -f "${BRIDGE}" ] || {
    echo "error: RVV highmem bridge output not found: ${BRIDGE}" >&2
    exit 1
}
[ -f "${FIRMWARE}" ] || {
    echo "error: MobileNet firmware output not found: ${FIRMWARE}" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Install artifacts into build/coralnpu/ with atomic temporary-file writes.
# ---------------------------------------------------------------------------
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

install_output "${BRIDGE}" \
    "${OUT_DIR}/libcoralnpu_gem5_rvv_highmem_bridge.so" 0755
install_output "${FIRMWARE}" "${OUT_DIR}/gem5_mobilenet.elf" 0644

# ---------------------------------------------------------------------------
# Print firmware ELF layout for diagnostics.
#   Entry point, LOAD segments, and key symbols are useful when debugging
#   ITCM/DTCM placement or BSS initialization issues.
# ---------------------------------------------------------------------------
if command -v readelf >/dev/null 2>&1; then
    {
        echo "MobileNet firmware ELF layout:"
        readelf -hW "${OUT_DIR}/gem5_mobilenet.elf" |
            grep -E 'Entry point address|Number of program headers'
        readelf -lW "${OUT_DIR}/gem5_mobilenet.elf" |
            grep -E '^[[:space:]]*LOAD'
        readelf -sW "${OUT_DIR}/gem5_mobilenet.elf" |
            grep -E '[[:space:]](_start|main|tensor_arena|__bss_start__|__bss_end__|__stack_start__|__stack_end__)$'
    } | tee -a "${BUILD_LOG}"
fi

# ---------------------------------------------------------------------------
# Sanity checks: the bridge must not have unresolved SRAM backdoor symbols
# or link against a second libsystemc (gem5 already provides one).
# ---------------------------------------------------------------------------
if command -v nm >/dev/null 2>&1 &&
   nm -D --undefined-only \
       "${OUT_DIR}/libcoralnpu_gem5_rvv_highmem_bridge.so" 2>/dev/null |
       grep -Eq '[[:space:]]sram_(init|read|write)$'; then
    echo "error: RVV highmem bridge has unresolved SRAM symbols" >&2
    exit 1
fi

if command -v ldd >/dev/null 2>&1 &&
   ldd "${OUT_DIR}/libcoralnpu_gem5_rvv_highmem_bridge.so" 2>/dev/null |
       grep -qi systemc; then
    echo "error: RVV highmem bridge unexpectedly links libsystemc" >&2
    exit 1
fi

echo "built: ${OUT_DIR}/libcoralnpu_gem5_rvv_highmem_bridge.so" | tee -a "${BUILD_LOG}"
echo "built: ${OUT_DIR}/gem5_mobilenet.elf" | tee -a "${BUILD_LOG}"
