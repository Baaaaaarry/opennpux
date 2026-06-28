#!/bin/sh

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

"${ROOT_DIR}/tools/coralnpu/check_mobilenet_abi.sh"
"${ROOT_DIR}/sim/coralnpu/apply_patchset.sh"
mkdir -p "${BAZEL_OUTPUT_ROOT}" "${REPO_CACHE}" "${DISTDIR}" "${OUT_DIR}"

cd "${CORAL_REPO}"
"${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" build \
    --repository_cache="${REPO_CACHE}" \
    --distdir="${DISTDIR}" \
    "${BRIDGE_TARGET}" "${FIRMWARE_TARGET}" "$@"

EXEC_ROOT="$("${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" \
    info execution_root)"

resolve_output()
{
    target="$1"
    output="$("${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" \
        cquery --repository_cache="${REPO_CACHE}" \
        --distdir="${DISTDIR}" --output=files "${target}")"
    case "${output}" in
        /*) printf '%s\n' "${output}" ;;
        *) printf '%s\n' "${EXEC_ROOT}/${output}" ;;
    esac
}

BRIDGE="$(resolve_output "${BRIDGE_TARGET}")"
FIRMWARE="$(resolve_output "${FIRMWARE_TARGET}")"
[ -f "${BRIDGE}" ] || {
    echo "error: RVV highmem bridge output not found: ${BRIDGE}" >&2
    exit 1
}
[ -f "${FIRMWARE}" ] || {
    echo "error: MobileNet firmware output not found: ${FIRMWARE}" >&2
    exit 1
}

cp "${BRIDGE}" "${OUT_DIR}/libcoralnpu_gem5_rvv_highmem_bridge.so"
cp "${FIRMWARE}" "${OUT_DIR}/gem5_mobilenet.elf"
chmod 0755 "${OUT_DIR}/libcoralnpu_gem5_rvv_highmem_bridge.so"

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

echo "built: ${OUT_DIR}/libcoralnpu_gem5_rvv_highmem_bridge.so"
echo "built: ${OUT_DIR}/gem5_mobilenet.elf"
