#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CORAL_REPO="${CORAL_REPO:-${ROOT_DIR}/thirdparty/coralnpu}"
BAZEL="${BAZEL:-bazel}"
TARGET="//hw_sim/gem5_bridge:libcoralnpu_gem5_bridge.so"
OUT_DIR="${ROOT_DIR}/build/coralnpu"

"${ROOT_DIR}/sim/coralnpu/apply_patchset.sh"

cd "${CORAL_REPO}"
"${BAZEL}" build "${TARGET}" "$@"
BAZEL_BIN="$("${BAZEL}" info bazel-bin)"
BRIDGE="${BAZEL_BIN}/hw_sim/gem5_bridge/libcoralnpu_gem5_bridge.so"

if [ ! -f "${BRIDGE}" ]; then
    echo "error: Bazel completed but bridge was not found: ${BRIDGE}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
cp "${BRIDGE}" "${OUT_DIR}/libcoralnpu_gem5_bridge.so"
chmod 0755 "${OUT_DIR}/libcoralnpu_gem5_bridge.so"

echo "built: ${OUT_DIR}/libcoralnpu_gem5_bridge.so"
