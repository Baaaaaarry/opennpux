#!/bin/sh

set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 <huggingface-model-directory> [output.npxm]" >&2
    exit 2
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
MODEL_DIR="$(CDPATH= cd -- "$1" && pwd -P)"
OUTPUT="${2:-${MODEL_DIR}/model.npxm}"
OUT_DIR="${ROOT_DIR}/build/local-tests"
CC="${CC:-cc}"

mkdir -p "${OUT_DIR}"
"${SCRIPT_DIR}/import_hf_model.py" "${MODEL_DIR}" "${OUTPUT}"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/model_package.c" \
    "${ROOT_DIR}/runtime/host/tools/model_inspect.c" \
    -o "${OUT_DIR}/model-inspect"
"${OUT_DIR}/model-inspect" "${OUTPUT}"
echo "hf_model_package_prepare=PASS"
