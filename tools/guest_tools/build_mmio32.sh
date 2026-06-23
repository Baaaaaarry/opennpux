#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
SRC="${ROOT_DIR}/runtime/host/tools/mmio32.c"
OUT_DIR="${ROOT_DIR}/build/guest-tools"
OUT="${OUT_DIR}/mmio32-aarch64"
CROSS="${CROSS:-aarch64-linux-gnu-}"
CC="${CC:-${CROSS}gcc}"

mkdir -p "${OUT_DIR}"

if ! command -v "${CC}" >/dev/null 2>&1; then
    echo "error: ${CC} not found; install gcc-aarch64-linux-gnu or set CC" >&2
    exit 1
fi

if "${CC}" -O2 -static -s -Wall -Wextra -o "${OUT}" "${SRC}"; then
    :
else
    echo "warning: static build failed; retrying dynamic build" >&2
    "${CC}" -O2 -s -Wall -Wextra -o "${OUT}" "${SRC}"
fi

file "${OUT}" 2>/dev/null || true
echo "built: ${OUT}"
