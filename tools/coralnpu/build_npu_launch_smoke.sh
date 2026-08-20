#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CORAL_REPO="${CORAL_REPO:-${ROOT_DIR}/thirdparty/coralnpu}"
TARGET="${CORAL_NPU_LAUNCH_TARGET:-//hw_sim:gem5_npu_launch_smoke.elf}"
OUTPUT_NAME="${CORAL_NPU_LAUNCH_OUTPUT:-gem5_npu_launch_smoke.elf}"
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

case " $* " in
    *" -c "*|*" --compilation_mode"*) ;;
    *) set -- -c opt "$@" ;;
esac

"${ROOT_DIR}/sim/coralnpu/apply_patchset.sh"
mkdir -p "${BAZEL_OUTPUT_ROOT}" "${REPO_CACHE}" "${DISTDIR}" "${OUT_DIR}"

cd "${CORAL_REPO}"
"${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" build \
    --repository_cache="${REPO_CACHE}" \
    --distdir="${DISTDIR}" "${TARGET}" "$@"

EXEC_ROOT="$("${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" \
    info execution_root)"
OUTPUT="$("${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" cquery \
    --repository_cache="${REPO_CACHE}" --distdir="${DISTDIR}" \
    --output=files "${TARGET}" "$@")"
case "${OUTPUT}" in
    /*) SOURCE="${OUTPUT}" ;;
    *) SOURCE="${EXEC_ROOT}/${OUTPUT}" ;;
esac
[ -f "${SOURCE}" ] || {
    echo "error: NPU_LAUNCH firmware output not found: ${SOURCE}" >&2
    exit 1
}

DESTINATION="${OUT_DIR}/${OUTPUT_NAME}"
TEMPORARY="$(mktemp "${OUT_DIR}/.${OUTPUT_NAME}.XXXXXX")"
cp "${SOURCE}" "${TEMPORARY}"
chmod 0644 "${TEMPORARY}"
mv -f "${TEMPORARY}" "${DESTINATION}"

# CUSTOM_0 has opcode 0x0b in bits [6:0]. Search executable sections directly
# so validation does not depend on objdump knowing the custom mnemonic.
python3 - "${DESTINATION}" <<'PY'
import struct
import sys

path = sys.argv[1]
data = open(path, "rb").read()
if data[:4] != b"\x7fELF" or data[4] != 1 or data[5] != 1:
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
    for pos in range(sh_offset, sh_offset + sh_size - 3, 2):
        word = struct.unpack_from("<I", data, pos)[0]
        # funct7=0, funct3=0, rd=x0, opcode=CUSTOM_0. rs1/rs2 vary.
        if word & 0xfe007fff == 0x0000000b:
            matches.append(word)
if not matches:
    raise SystemExit("error: firmware has no CUSTOM_0/NPU_LAUNCH instruction")
print("NPU_LAUNCH encoding=0x%08x count=%d" % (matches[0], len(matches)))
PY

echo "built: ${DESTINATION}"
