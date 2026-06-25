#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CORAL_REPO="${CORAL_REPO:-${ROOT_DIR}/thirdparty/coralnpu}"
TARGET="//hw_sim:libcoralnpu_gem5_bridge.so"
FIRMWARE_TARGET="//hw_sim:gem5_smoke_halt.elf"
DMA_FIRMWARE_TARGET="//hw_sim:gem5_dma_smoke.elf"
OUT_DIR="${ROOT_DIR}/build/coralnpu"
LOCAL_BAZEL="${ROOT_DIR}/.cache/coralnpu/bin/bazel"
BAZEL_OUTPUT_ROOT="${PHASE2_BAZEL_OUTPUT_ROOT:-${ROOT_DIR}/.cache/coralnpu/bazel}"
REPO_CACHE="${PHASE2_REPO_CACHE:-${ROOT_DIR}/.cache/coralnpu/repository}"
DISTDIR="${PHASE2_DISTDIR:-${CORAL_REPO}/distdir}"

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
  BAZEL_BINARY=/path/to/bazel-$(cat "${CORAL_REPO}/.bazelversion")-linux-x86_64 ./tools/coralnpu/phase2_prepare_bazel.sh
EOF
    exit 1
fi

"${ROOT_DIR}/tools/coralnpu/phase2_check_abi.sh"
"${ROOT_DIR}/sim/coralnpu/apply_patchset.sh"
"${ROOT_DIR}/tools/coralnpu/phase2_check_overlay_boundary.sh"

mkdir -p "${BAZEL_OUTPUT_ROOT}" "${REPO_CACHE}" "${DISTDIR}"
rm -f \
    "${OUT_DIR}/libcoralnpu_gem5_bridge.so" \
    "${OUT_DIR}/gem5_smoke_halt.elf" \
    "${OUT_DIR}/gem5_dma_smoke.elf" \
    "${OUT_DIR}/wfi_slot_0.elf"

cd "${CORAL_REPO}"
"${BAZEL}" \
    --output_user_root="${BAZEL_OUTPUT_ROOT}" \
    build \
    --repository_cache="${REPO_CACHE}" \
    --distdir="${DISTDIR}" \
    "${TARGET}" "${FIRMWARE_TARGET}" "${DMA_FIRMWARE_TARGET}" "$@"
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

mkdir -p "${OUT_DIR}"
cp "${BRIDGE}" "${OUT_DIR}/libcoralnpu_gem5_bridge.so"
cp "${FIRMWARE}" "${OUT_DIR}/gem5_smoke_halt.elf"
cp "${DMA_FIRMWARE}" "${OUT_DIR}/gem5_dma_smoke.elf"
chmod 0755 "${OUT_DIR}/libcoralnpu_gem5_bridge.so"

if command -v ldd >/dev/null 2>&1 &&
   ldd "${OUT_DIR}/libcoralnpu_gem5_bridge.so" 2>/dev/null |
       grep -qi systemc; then
    echo "error: Coral gem5 bridge unexpectedly links libsystemc" >&2
    echo "       rerun after applying the current Coral overlay" >&2
    exit 1
fi

echo "built: ${OUT_DIR}/libcoralnpu_gem5_bridge.so"
echo "built: ${OUT_DIR}/gem5_smoke_halt.elf"
echo "built: ${OUT_DIR}/gem5_dma_smoke.elf"
