#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CORAL_REPO="${CORAL_REPO:-${ROOT_DIR}/thirdparty/coralnpu}"
BRIDGE_TARGET="//hw_sim:libcoralnpu_gem5_rvv_highmem_bridge.so"
FIRMWARE_TARGET="//hw_sim:gem5_gptq_matmul_smoke.elf"
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

case " $* " in
    *" -c "*|*" --compilation_mode"*) ;;
    *) set -- -c opt "$@" ;;
esac

"${ROOT_DIR}/sim/coralnpu/apply_patchset.sh"
"${ROOT_DIR}/tools/coralnpu/test_coprocessor_command.sh"
"${ROOT_DIR}/tools/coralnpu/test_hybrid_kernels.sh"
mkdir -p "${BAZEL_OUTPUT_ROOT}" "${REPO_CACHE}" "${DISTDIR}" "${OUT_DIR}"

cd "${CORAL_REPO}"
"${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" build \
    --repository_cache="${REPO_CACHE}" --distdir="${DISTDIR}" \
    "${BRIDGE_TARGET}" "${FIRMWARE_TARGET}" "$@"

EXEC_ROOT="$("${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" \
    info execution_root)"
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

install_output()
{
    source_path="$1"
    destination_path="$2"
    mode="$3"
    temporary_path="$(mktemp "${OUT_DIR}/.$(basename "${destination_path}").XXXXXX")"
    cp "${source_path}" "${temporary_path}"
    chmod "${mode}" "${temporary_path}"
    mv -f "${temporary_path}" "${destination_path}"
}

BRIDGE="$(resolve_output "${BRIDGE_TARGET}" "$@")"
FIRMWARE="$(resolve_output "${FIRMWARE_TARGET}" "$@")"
[ -f "${BRIDGE}" ] || {
    echo "error: RVV highmem bridge output not found: ${BRIDGE}" >&2
    exit 1
}
[ -f "${FIRMWARE}" ] || {
    echo "error: GPTQ firmware output not found: ${FIRMWARE}" >&2
    exit 1
}

install_output "${BRIDGE}" \
    "${OUT_DIR}/libcoralnpu_gem5_rvv_highmem_bridge.so" 0755
install_output "${FIRMWARE}" "${OUT_DIR}/gem5_gptq_matmul_smoke.elf" 0644

python3 - "${OUT_DIR}/gem5_gptq_matmul_smoke.elf" <<'PY'
import struct
import sys

data = open(sys.argv[1], "rb").read()
if data[:6] != b"\x7fELF\x01\x01":
    raise SystemExit("error: expected a little-endian ELF32 firmware")
e_shoff = struct.unpack_from("<I", data, 32)[0]
e_shentsize, e_shnum = struct.unpack_from("<HH", data, 46)
matches = []
for index in range(e_shnum):
    offset = e_shoff + index * e_shentsize
    sh_type = struct.unpack_from("<I", data, offset + 4)[0]
    sh_flags = struct.unpack_from("<I", data, offset + 8)[0]
    sh_offset = struct.unpack_from("<I", data, offset + 16)[0]
    sh_size = struct.unpack_from("<I", data, offset + 20)[0]
    if sh_type != 1 or not (sh_flags & 0x4):
        continue
    for position in range(sh_offset, sh_offset + sh_size - 3, 2):
        word = struct.unpack_from("<I", data, position)[0]
        if word & 0xFE007FFF == 0x0000000B:
            matches.append(word)
if not matches:
    raise SystemExit("error: GPTQ firmware has no CUSTOM_0 instruction")
print("GPTQ CUSTOM_0 encoding=0x%08x count=%d" %
      (matches[0], len(matches)))
PY

echo "built: ${OUT_DIR}/libcoralnpu_gem5_rvv_highmem_bridge.so"
echo "built: ${OUT_DIR}/gem5_gptq_matmul_smoke.elf"
