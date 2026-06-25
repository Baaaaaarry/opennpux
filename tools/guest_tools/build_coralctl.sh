#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
SRC="${ROOT_DIR}/runtime/host/tools/coralctl.c"
OUT_DIR="${ROOT_DIR}/build/guest-tools"
OUT="${OUT_DIR}/coralctl-aarch64"
CROSS="${CROSS:-aarch64-linux-gnu-}"
CC="${CC:-${CROSS}gcc}"

if ! command -v "${CC}" >/dev/null 2>&1; then
    echo "error: ${CC} not found; install gcc-aarch64-linux-gnu or set CC" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
"${CC}" -O2 -static -Wall -Wextra -Werror -std=c11 "${SRC}" -o "${OUT}"
echo "built: ${OUT}"
