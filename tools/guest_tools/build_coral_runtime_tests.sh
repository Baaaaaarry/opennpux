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
    "${ROOT_DIR}/tests/unit/runtime_host/coral_runtime_test.c" \
    -o "${OUT}"
"${OUT}"
