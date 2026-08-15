#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CORAL_REPO="${CORAL_REPO:-${ROOT_DIR}/thirdparty/coralnpu}"
OUT_DIR="${ROOT_DIR}/build/coralnpu"
BAZEL_OUTPUT_ROOT="${CORAL_BAZEL_OUTPUT_ROOT:-${ROOT_DIR}/.cache/coralnpu/bazel}"
REPO_CACHE="${CORAL_REPO_CACHE:-${ROOT_DIR}/.cache/coralnpu/repository}"
DISTDIR="${CORAL_DISTDIR:-${CORAL_REPO}/distdir}"
BRIDGE_TARGET="//hw_sim:libcoralnpu_gem5_rvv_highmem_bridge.so"
FIRMWARE_TARGET="//hw_sim:gem5_qwen_device_infer.elf"

if [ -n "${BAZEL:-}" ]; then
    :
elif command -v bazel >/dev/null 2>&1; then
    BAZEL="$(command -v bazel)"
elif [ -x "${ROOT_DIR}/.cache/coralnpu/bin/bazel" ]; then
    BAZEL="${ROOT_DIR}/.cache/coralnpu/bin/bazel"
else
    echo "error: bazel not found; run prepare_coral_bazel.sh" >&2
    exit 1
fi

case " $* " in
    *" -c "*|*" --compilation_mode"*) ;;
    *) set -- -c opt "$@" ;;
esac

"${ROOT_DIR}/sim/coralnpu/apply_patchset.sh"
"${ROOT_DIR}/tools/coralnpu/check_rtl_bridge_abi.sh"
"${ROOT_DIR}/tools/coralnpu/test_coprocessor_command.sh"
"${ROOT_DIR}/tools/coralnpu/test_hybrid_kernels.sh"
mkdir -p "${BAZEL_OUTPUT_ROOT}" "${REPO_CACHE}" "${DISTDIR}" "${OUT_DIR}"
cd "${CORAL_REPO}"
"${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" build \
    --repository_cache="${REPO_CACHE}" --distdir="${DISTDIR}" \
    "${BRIDGE_TARGET}" "${FIRMWARE_TARGET}" "$@"

EXEC_ROOT="$("${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" info execution_root)"
resolve_output()
{
    target="$1"
    shift
    output="$("${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" cquery \
        --repository_cache="${REPO_CACHE}" --distdir="${DISTDIR}" "$@" \
        --output=files "${target}")"
    case "${output}" in
        /*) printf '%s\n' "${output}" ;;
        *) printf '%s\n' "${EXEC_ROOT}/${output}" ;;
    esac
}

BRIDGE="$(resolve_output "${BRIDGE_TARGET}" "$@")"
FIRMWARE="$(resolve_output "${FIRMWARE_TARGET}" "$@")"
[ -f "${BRIDGE}" ] || { echo "error: bridge output missing: ${BRIDGE}" >&2; exit 1; }
[ -f "${FIRMWARE}" ] || { echo "error: firmware output missing: ${FIRMWARE}" >&2; exit 1; }
cp "${BRIDGE}" "${OUT_DIR}/libcoralnpu_gem5_rvv_highmem_bridge.so"
chmod 0755 "${OUT_DIR}/libcoralnpu_gem5_rvv_highmem_bridge.so"
cp "${FIRMWARE}" "${OUT_DIR}/gem5_qwen_device_infer.elf"
chmod 0644 "${OUT_DIR}/gem5_qwen_device_infer.elf"
echo "built: ${OUT_DIR}/libcoralnpu_gem5_rvv_highmem_bridge.so"
echo "built: ${OUT_DIR}/gem5_qwen_device_infer.elf"
