#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
OUT_DIR="${ROOT_DIR}/build/local-tests"
OUT="${OUT_DIR}/coral_runtime_test"
CC="${CC:-cc}"

mkdir -p "${OUT_DIR}"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/coral_runtime.c" \
    "${ROOT_DIR}/runtime/host/src/npu_gptq_tile_plan.c" \
    "${ROOT_DIR}/runtime/host/src/npu_xgraph_lowering.c" \
    "${ROOT_DIR}/runtime/host/src/model_package.c" \
    "${ROOT_DIR}/runtime/host/src/npu_submission.c" \
    "${ROOT_DIR}/runtime/host/src/qwen_model.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/coral_runtime_test.c" \
    -o "${OUT}"
"${OUT}"

"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/npu_submission.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_submission_test.c" \
    -o "${OUT_DIR}/npu_submission_test"
"${OUT_DIR}/npu_submission_test"

"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/npu_submission.c" \
    "${ROOT_DIR}/runtime/host/src/npu_executable.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_executable_test.c" \
    -o "${OUT_DIR}/npu_executable_test"
