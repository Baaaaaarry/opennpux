#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <model.npxm> <model.npxr>" >&2
    exit 2
fi

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
OUT_DIR="${ROOT_DIR}/build/local-tests/npu-weight-inspect"
CC="${CC:-cc}"
mkdir -p "$OUT_DIR"
"$CC" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/model_package.c" \
    "${ROOT_DIR}/runtime/host/src/npu_weight_ranges.c" \
    "${ROOT_DIR}/runtime/host/src/npu_weight_pager.c" \
    "${ROOT_DIR}/runtime/host/tools/npu_weight_inspect.c" \
    -o "${OUT_DIR}/npu-weight-inspect"
exec "${OUT_DIR}/npu-weight-inspect" "$1" "$2"
