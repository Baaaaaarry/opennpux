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
    cat >&2 <<EOF
error: CoralNPU requires Bazel ${EXPECTED_VERSION}

Prepare the repository-local executable, then rerun this script:
  ./tools/coralnpu/prepare_coral_bazel.sh

Offline host:
  BAZEL_BINARY=/path/to/bazel-${EXPECTED_VERSION}-linux-x86_64 \\
    ./tools/coralnpu/prepare_coral_bazel.sh
EOF
    exit 1
fi

"${ROOT_DIR}/sim/coralnpu/apply_patchset.sh"
mkdir -p "${BAZEL_OUTPUT_ROOT}" "${REPO_CACHE}" "${DISTDIR}"

cd "${CORAL_REPO}"
"${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" test \
    --repository_cache="${REPO_CACHE}" --distdir="${DISTDIR}" \
    //hw_sim:gem5_tmma_coprocessor_test "$@"
"${BAZEL}" --output_user_root="${BAZEL_OUTPUT_ROOT}" build \
    --repository_cache="${REPO_CACHE}" --distdir="${DISTDIR}" \
    //hw_sim:gem5_tmma_smoke.elf "$@"

EXEC_ROOT="$(${BAZEL} --output_user_root="${BAZEL_OUTPUT_ROOT}" \
    info execution_root)"
OUTPUT="$(${BAZEL} --output_user_root="${BAZEL_OUTPUT_ROOT}" cquery \
    --repository_cache="${REPO_CACHE}" --distdir="${DISTDIR}" \
    --output=files //hw_sim:gem5_tmma_smoke.elf "$@")"
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

checks = {
    "csrw_mma_shape": any((word & 0xFFF07FFF) == 0x80001073 for word in words),
    "csrw_mma_data_type": any((word & 0xFFF07FFF) == 0x80101073 for word in words),
    "tmma_custom3": any((word & 0xFE00707F) == 0x0000007B for word in words),
    "tfence": 0x0000607B in words,
}
missing = [name for name, present in checks.items() if not present]
if missing:
    raise SystemExit("error: firmware encoding missing: " + ", ".join(missing))
for name in checks:
    print(f"{name}=PASS")
print(f"tmma_firmware={path}")
print("tmma_instruction_test=PASS")
PY
