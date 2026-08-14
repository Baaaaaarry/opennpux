#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
MODEL="${1:-${ROOT_DIR}/build/models/qwen-tiny.npxm}"
OUT_DIR="${ROOT_DIR}/build/local-tests"
CC="${CC:-cc}"

"${SCRIPT_DIR}/run_qwen_golden.sh" "${MODEL}"
mkdir -p "${OUT_DIR}"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/qwen_model.c" \
    "${ROOT_DIR}/runtime/host/tools/qwen_inspect.c" \
    -lm \
    -o "${OUT_DIR}/qwen-inspect"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/qwen_model.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/qwen_model_test.c" \
    -lm \
    -o "${OUT_DIR}/qwen_model_test"
"${OUT_DIR}/qwen-inspect" "${MODEL}"
"${OUT_DIR}/qwen_model_test" "${MODEL}"
echo "qwen_loader=PASS"
