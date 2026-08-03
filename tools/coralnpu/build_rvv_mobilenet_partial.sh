#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CORAL_REPO="${CORAL_REPO:-${ROOT_DIR}/thirdparty/coralnpu}"
BRIDGE_TARGET="//hw_sim:libcoralnpu_gem5_rvv_highmem_bridge.so"
FIRMWARE_TARGET="//hw_sim:gem5_mobilenet_partial.elf"
OUT_DIR="${ROOT_DIR}/build/coralnpu"
LOCAL_BAZEL="${ROOT_DIR}/.cache/coralnpu/bin/bazel"
BAZEL_OUTPUT_ROOT="${CORAL_BAZEL_OUTPUT_ROOT:-${ROOT_DIR}/.cache/coralnpu/bazel}"
REPO_CACHE="${CORAL_REPO_CACHE:-${ROOT_DIR}/.cache/coralnpu/repository}"
DISTDIR="${CORAL_DISTDIR:-${CORAL_REPO}/distdir}"

if [ -n "${BAZEL:-}" ]; then
    :
elif command -v bazel >/dev/null 2>&1; then
    BAZEL="$(command -v bazel)"
elif [ -x "${LOCAL_BAZEL}" ]; then
    BAZEL="${LOCAL_BAZEL}"
else
    echo "error: bazel not found; run prepare_coral_bazel.sh" >&2
    exit 1
fi

"${ROOT_DIR}/tools/coralnpu/check_mobilenet_abi.sh"
"${ROOT_DIR}/sim/coralnpu/apply_patchset.sh"
mkdir -p "${BAZEL_OUTPUT_ROOT}" "${REPO_CACHE}" "${DISTDIR}" "${OUT_DIR}"
if [ ! -w "${OUT_DIR}" ]; then
    echo "error: output directory is not writable: ${OUT_DIR}" >&2
    echo "repair ownership once with:" >&2
    echo "  sudo chown -R \"${USER:-$(id -un)}:$(id -gn)\" \"${OUT_DIR}\"" >&2
    exit 1
fi

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

BRIDGE="$(resolve_output "${BRIDGE_TARGET}")"
FIRMWARE="$(resolve_output "${FIRMWARE_TARGET}")"
[ -f "${BRIDGE}" ] || {
    echo "error: RVV highmem bridge output not found: ${BRIDGE}" >&2
    exit 1
}
[ -f "${FIRMWARE}" ] || {
    echo "error: partial MobileNet firmware output not found: ${FIRMWARE}" >&2
    exit 1
}

install_output "${BRIDGE}" \
    "${OUT_DIR}/libcoralnpu_gem5_rvv_highmem_bridge.so" 0755
install_output "${FIRMWARE}" "${OUT_DIR}/gem5_mobilenet_partial.elf" 0644

echo "built: ${OUT_DIR}/libcoralnpu_gem5_rvv_highmem_bridge.so"
echo "built: ${OUT_DIR}/gem5_mobilenet_partial.elf"
