#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CORAL_REPO="${CORAL_REPO:-${ROOT_DIR}/thirdparty/coralnpu}"
LOCAL_BAZEL="${ROOT_DIR}/.cache/coralnpu/bin/bazel"
BAZEL_OUTPUT_ROOT="${CORAL_BAZEL_OUTPUT_ROOT:-${ROOT_DIR}/.cache/coralnpu/bazel}"
REPO_CACHE="${CORAL_REPO_CACHE:-${ROOT_DIR}/.cache/coralnpu/repository}"
DISTDIR="${CORAL_DISTDIR:-${CORAL_REPO}/distdir}"
EXPECTED_VERSION="$(head -n 1 "${CORAL_REPO}/.bazelversion" | tr -d '\r')"

if [ -n "${BAZEL:-}" ]; then
    :
elif [ -x "${LOCAL_BAZEL}" ]; then
    BAZEL="${LOCAL_BAZEL}"
elif command -v bazel >/dev/null 2>&1; then
    BAZEL="$(command -v bazel)"
else
    BAZEL=""
fi

if [ -z "${BAZEL}" ] ||
   ! ACTUAL_VERSION="$(${BAZEL} --version 2>/dev/null)" ||
   ! printf '%s\n' "${ACTUAL_VERSION}" | grep -Fq "${EXPECTED_VERSION}"; then
    echo "error: CoralNPU requires Bazel ${EXPECTED_VERSION}" >&2
    echo "run ./tools/coralnpu/prepare_coral_bazel.sh first" >&2
    exit 1
fi

"${ROOT_DIR}/sim/coralnpu/apply_patchset.sh"
mkdir -p "${BAZEL_OUTPUT_ROOT}" "${REPO_CACHE}" "${DISTDIR}"

cd "${CORAL_REPO}"
"${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" build \
    --repository_cache="${REPO_CACHE}" --distdir="${DISTDIR}" \
    //hw_sim:gem5_tmma_operator_e2e.elf "$@"

EXEC_ROOT="$(${BAZEL} --output_user_root="${BAZEL_OUTPUT_ROOT}" \
    info execution_root)"
OUTPUT="$(${BAZEL} --output_user_root="${BAZEL_OUTPUT_ROOT}" cquery \
    --repository_cache="${REPO_CACHE}" --distdir="${DISTDIR}" \
    --output=files //hw_sim:gem5_tmma_operator_e2e.elf "$@")"
case "${OUTPUT}" in
    /*) FIRMWARE="${OUTPUT}" ;;
    *) FIRMWARE="${EXEC_ROOT}/${OUTPUT}" ;;
esac

python3 - "${FIRMWARE}" <<'PY'
import struct
import sys

path = sys.argv[1]
data = open(path, "rb").read()
if data[:4] != b"\x7fELF" or data[4] != 1 or data[5] != 1:
    raise SystemExit("error: expected a little-endian ELF32 firmware")

section_offset = struct.unpack_from("<I", data, 32)[0]
section_entry_size, section_count = struct.unpack_from("<HH", data, 46)
words = []
for index in range(section_count):
    offset = section_offset + index * section_entry_size
    section_type = struct.unpack_from("<I", data, offset + 4)[0]
    section_flags = struct.unpack_from("<I", data, offset + 8)[0]
    payload_offset = struct.unpack_from("<I", data, offset + 16)[0]
    payload_size = struct.unpack_from("<I", data, offset + 20)[0]
    if section_type != 1 or not (section_flags & 0x4):
        continue
    for position in range(payload_offset, payload_offset + payload_size - 3, 2):
        words.append(struct.unpack_from("<I", data, position)[0])

tmma_count = sum((word & 0xFE00707F) == 0x0000007B for word in words)
tadd_count = sum((word & 0xFE00707F) == 0x0200107B for word in words)
tmul_count = sum((word & 0xFE00707F) == 0x0400107B for word in words)
trmsnorm_count = sum((word & 0xFE00707F) == 0x6200207B for word in words)
tfence_count = sum(word == 0x0000607B for word in words)
shape_writes = sum((word & 0xFFF07FFF) == 0x80001073 for word in words)
dtype_writes = sum((word & 0xFFF07FFF) == 0x80101073 for word in words)
tensor_shape_writes = sum((word & 0xFFF07FFF) == 0x80201073 for word in words)
tensor_dtype_writes = sum((word & 0xFFF07FFF) == 0x80601073 for word in words)
scalar_param_writes = sum((word & 0xFFF07FFF) == 0x80B01073 for word in words)
checks = {
    "tmma_encoding_count": (tmma_count, 1),
    "tadd_encoding_count": (tadd_count, 1),
    "tmul_encoding_count": (tmul_count, 1),
    "trmsnorm_encoding_count": (trmsnorm_count, 1),
    "tfence_encoding_count": (tfence_count, 1),
    "shape_write_encoding_count": (shape_writes, 1),
    "dtype_write_encoding_count": (dtype_writes, 1),
    "tensor_shape_write_encoding_count": (tensor_shape_writes, 1),
    "tensor_dtype_write_encoding_count": (tensor_dtype_writes, 1),
    "scalar_param_write_encoding_count": (scalar_param_writes, 1),
}
for name, (actual, expected) in checks.items():
    if actual < expected:
        raise SystemExit(f"error: {name}={actual}, expected at least {expected}")
    print(f"{name}={actual}")
print(f"tmma_operator_firmware={path}")
print("tmma_operator_encoding=PASS")
PY
