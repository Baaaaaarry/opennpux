#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
OUT_DIR="${ROOT_DIR}/build/guest-tools"
CROSS="${CROSS:-aarch64-linux-gnu-}"
CC="${CC:-${CROSS}gcc}"
HOST_CC="${HOST_CC:-cc}"

if ! command -v "${CC}" >/dev/null 2>&1; then
    echo "error: ${CC} not found; install gcc-aarch64-linux-gnu or set CC" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}" "${ROOT_DIR}/build/local-tests"
"${HOST_CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/qwen_model.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/qwen_model_test.c" \
    -o "${ROOT_DIR}/build/local-tests/qwen_model_test"
"${CC}" -O2 -static -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/qwen_model.c" \
    "${ROOT_DIR}/runtime/host/tools/qwen_inspect.c" \
    -o "${OUT_DIR}/qwen-inspect-aarch64"
echo "built: ${OUT_DIR}/qwen-inspect-aarch64"
