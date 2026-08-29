#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CORAL_REPO="${CORAL_REPO:-${ROOT_DIR}/thirdparty/coralnpu}"
TARGET="${CORAL_NPU_LAUNCH_TARGET:-//hw_sim:gem5_npu_launch_smoke.elf}"
OUTPUT_NAME="${CORAL_NPU_LAUNCH_OUTPUT:-gem5_npu_launch_smoke.elf}"
BRIDGE_TARGET="${CORAL_NPU_LAUNCH_BRIDGE_TARGET:-}"
BRIDGE_OUTPUT_NAME="${CORAL_NPU_LAUNCH_BRIDGE_OUTPUT:-libcoralnpu_gem5_rvv_highmem_bridge.so}"
OUT_DIR="${ROOT_DIR}/build/coralnpu"
LOCAL_BAZEL="${ROOT_DIR}/.cache/coralnpu/bin/bazel"
BAZEL_OUTPUT_ROOT="${PHASE2_BAZEL_OUTPUT_ROOT:-${ROOT_DIR}/.cache/coralnpu/bazel}"
REPO_CACHE="${PHASE2_REPO_CACHE:-${ROOT_DIR}/.cache/coralnpu/repository}"
DISTDIR="${PHASE2_DISTDIR:-${CORAL_REPO}/distdir}"

if [ -n "${BAZEL:-}" ]; then
    :
elif [ -x "${LOCAL_BAZEL}" ]; then
    BAZEL="${LOCAL_BAZEL}"
elif command -v bazel >/dev/null 2>&1; then
    BAZEL="$(command -v bazel)"
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
if [ -n "${BRIDGE_TARGET}" ]; then
    "${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" build \
        --repository_cache="${REPO_CACHE}" \
        --distdir="${DISTDIR}" "${TARGET}" "${BRIDGE_TARGET}" "$@"
else
    "${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" build \
        --repository_cache="${REPO_CACHE}" \
        --distdir="${DISTDIR}" "${TARGET}" "$@"
fi

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

BRIDGE_SOURCE=""
if [ -n "${BRIDGE_TARGET}" ]; then
    BRIDGE_OUTPUT="$("${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" \
        cquery --repository_cache="${REPO_CACHE}" --distdir="${DISTDIR}" \
        --output=files "${BRIDGE_TARGET}" "$@")"
    case "${BRIDGE_OUTPUT}" in
        /*) BRIDGE_SOURCE="${BRIDGE_OUTPUT}" ;;
        *) BRIDGE_SOURCE="${EXEC_ROOT}/${BRIDGE_OUTPUT}" ;;
    esac
    [ -f "${BRIDGE_SOURCE}" ] || {
        echo "error: NPU_LAUNCH bridge output not found: ${BRIDGE_SOURCE}" >&2
        exit 1
    }
fi

DESTINATION="${OUT_DIR}/${OUTPUT_NAME}"
TEMPORARY="$(mktemp "${OUT_DIR}/.${OUTPUT_NAME}.XXXXXX")"
cp "${SOURCE}" "${TEMPORARY}"
chmod 0644 "${TEMPORARY}"
mv -f "${TEMPORARY}" "${DESTINATION}"

if [ -n "${BRIDGE_SOURCE}" ]; then
    BRIDGE_DESTINATION="${OUT_DIR}/${BRIDGE_OUTPUT_NAME}"
    BRIDGE_TEMPORARY="$(mktemp "${OUT_DIR}/.${BRIDGE_OUTPUT_NAME}.XXXXXX")"
    cp "${BRIDGE_SOURCE}" "${BRIDGE_TEMPORARY}"
    chmod 0755 "${BRIDGE_TEMPORARY}"
    mv -f "${BRIDGE_TEMPORARY}" "${BRIDGE_DESTINATION}"
    echo "built: ${BRIDGE_DESTINATION}"
fi

# Search executable sections directly so validation does not depend on objdump
# knowing the project-specific instruction mnemonics.
python3 - "${DESTINATION}" \
    "${CORAL_NPU_LAUNCH_INSTRUCTION_SET:-npu-launch}" <<'PY'
import struct
import sys

path = sys.argv[1]
instruction_set = sys.argv[2]
data = open(path, "rb").read()
if data[:4] != b"\x7fELF" or data[4] != 1 or data[5] != 1:
    raise SystemExit("error: expected a little-endian ELF32 firmware")
e_shoff = struct.unpack_from("<I", data, 32)[0]
e_shentsize, e_shnum = struct.unpack_from("<HH", data, 46)
words = []
for index in range(e_shnum):
    offset = e_shoff + index * e_shentsize
    sh_type = struct.unpack_from("<I", data, offset + 4)[0]
    sh_flags = struct.unpack_from("<I", data, offset + 8)[0]
    sh_offset = struct.unpack_from("<I", data, offset + 16)[0]
    sh_size = struct.unpack_from("<I", data, offset + 20)[0]
    if sh_type != 1 or not (sh_flags & 0x4):
        continue
    for pos in range(sh_offset, sh_offset + sh_size - 3, 2):
        words.append(struct.unpack_from("<I", data, pos)[0])

if instruction_set == "npu-launch":
    # funct7=0, funct3=0, rd=x0, opcode=CUSTOM_0. rs1/rs2 vary.
    matches = [word for word in words
               if word & 0xfe007fff == 0x0000000b]
    if not matches:
        raise SystemExit(
            "error: firmware has no CUSTOM_0/NPU_LAUNCH instruction")
    print("NPU_LAUNCH encoding=0x%08x count=%d" %
          (matches[0], len(matches)))
elif instruction_set == "xopennpux-qwen":
    expected = {
        "tmma": (0, 0x00),
        "tadd": (1, 0x01),
        "tmul": (1, 0x02),
        "trmsnorm": (2, 0x31),
        "tsoftmax": (2, 0x32),
        "trope": (2, 0x33),
        "tsilu": (2, 0x46),
        "tgather": (3, 0x10),
        "ttopk": (4, 0x00),
        "tfence": (6, 0x00),
    }
    observed = {
        ((word >> 12) & 0x7, (word >> 25) & 0x7f)
        for word in words if word & 0x7f == 0x7b
    }
    missing = [name for name, encoding in expected.items()
               if encoding not in observed]
    if missing:
        raise SystemExit(
            "error: firmware is missing XOpenNPUX instructions: " +
            ", ".join(missing))
    print("XOpenNPUX Qwen instruction set: %s" %
          ",".join(expected))
else:
    raise SystemExit("error: unknown firmware instruction set: " +
                     instruction_set)
PY

echo "built: ${DESTINATION}"
