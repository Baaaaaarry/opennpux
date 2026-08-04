#!/bin/sh

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
    cat >&2 <<EOF
error: bazel not found

Prepare the repository-local Bazel executable first:
  ./tools/coralnpu/prepare_coral_bazel.sh

For an offline/DNS-restricted host:
  BAZEL_BINARY=/path/to/bazel-$(cat "${CORAL_REPO}/.bazelversion")-linux-x86_64 ./tools/coralnpu/prepare_coral_bazel.sh
EOF
    exit 1
fi

"${ROOT_DIR}/tools/coralnpu/check_rtl_bridge_abi.sh"
"${ROOT_DIR}/tools/coralnpu/check_command_abi.sh"
"${ROOT_DIR}/sim/coralnpu/apply_patchset.sh"
"${ROOT_DIR}/tools/coralnpu/check_overlay_boundary.sh"
"${ROOT_DIR}/tools/coralnpu/test_axi_adapter.sh"
"${ROOT_DIR}/tools/coralnpu/test_custom_rtl.sh"

mkdir -p "${BAZEL_OUTPUT_ROOT}" "${REPO_CACHE}" "${DISTDIR}"
rm -f \
    "${OUT_DIR}/libcoralnpu_gem5_bridge.so" \
    "${OUT_DIR}/gem5_smoke_halt.elf" \
    "${OUT_DIR}/gem5_dma_smoke.elf" \
    "${OUT_DIR}/gem5_command_smoke.elf" \
    "${OUT_DIR}/wfi_slot_0.elf"

cd "${CORAL_REPO}"
"${BAZEL}" \
    --output_user_root="${BAZEL_OUTPUT_ROOT}" \
    build \
    --repository_cache="${REPO_CACHE}" \
    --distdir="${DISTDIR}" \
    "${TARGET}" "${FIRMWARE_TARGET}" "${DMA_FIRMWARE_TARGET}" \
    "${COMMAND_FIRMWARE_TARGET}" "$@"
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
COMMAND_FIRMWARE="$(resolve_output "${COMMAND_FIRMWARE_TARGET}")"

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

if command -v nm >/dev/null 2>&1 &&
   nm -D --undefined-only \
       "${OUT_DIR}/libcoralnpu_gem5_bridge.so" 2>/dev/null |
       grep -Eq '[[:space:]]sram_(init|read|write)$'; then
    echo "error: Coral gem5 bridge has unresolved SRAM backdoor symbols" >&2
    exit 1
fi

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
echo "built: ${OUT_DIR}/gem5_command_smoke.elf"
