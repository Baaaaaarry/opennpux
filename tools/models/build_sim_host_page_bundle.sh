#!/bin/sh
set -eu

if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
    echo "usage: $0 <model.npxm> <model.npxr> <output.npxb> [transfer-bytes] [max-pages-per-range]" >&2
    exit 2
fi

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
OUT_DIR="${ROOT_DIR}/build/host-tools"
CC="${CC:-cc}"
mkdir -p "$OUT_DIR" "$(dirname -- "$3")"
"$CC" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/model_package.c" \
    "${ROOT_DIR}/runtime/host/src/npu_weight_ranges.c" \
    "${ROOT_DIR}/runtime/host/src/npu_weight_pager.c" \
    "${ROOT_DIR}/runtime/host/tools/npu_page_bundle.c" \
    -o "${OUT_DIR}/npu-page-bundle"

TMP="${3}.tmp.$$"
trap 'rm -f "$TMP"' EXIT HUP INT TERM
"${OUT_DIR}/npu-page-bundle" "$1" "$2" "$TMP" "${4:-65536}" "${5:-1}"
mv -f "$TMP" "$3"
trap - EXIT HUP INT TERM
