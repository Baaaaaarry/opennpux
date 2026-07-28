#!/bin/sh
#
# build_mmio32.sh — Cross-compile the minimal mmio32 MMIO helper for aarch64.
#
# mmio32 is a tiny /dev/mem utility used during early NPU bring-up, before
# the kernel driver (opennpux-coral) is available.  It reads and writes
# 32-bit values at physical addresses through /dev/mem.
#
# This tool is superseded by coralctl once the Phase-3 driver is loaded,
# but remains useful for debugging the MMIO aperture before the driver
# stack is functional.
#
# The build tries static linking first.  Some cross-compiler installations
# lack a static libc for aarch64; in that case it retries with dynamic
# linking.  The resulting binary is small enough (~10 KiB) that either
# variant works in the guest.
#
# Pipeline:
#   runtime/host/tools/mmio32.c
#     → aarch64-linux-gnu-gcc -static (fallback: dynamic)
#       → build/guest-tools/mmio32-aarch64
#
# Output:
#   build/guest-tools/mmio32-aarch64    aarch64 guest binary
#
# Environment:
#   CROSS      cross-compiler prefix (default: aarch64-linux-gnu-)
#   CC         compiler (default: ${CROSS}gcc)
#
# @guest-tools-spec  v1  2025-07-28

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
SRC="${ROOT_DIR}/runtime/host/tools/mmio32.c"
OUT_DIR="${ROOT_DIR}/build/guest-tools"
OUT="${OUT_DIR}/mmio32-aarch64"
CROSS="${CROSS:-aarch64-linux-gnu-}"
CC="${CC:-${CROSS}gcc}"

mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# Preflight: the aarch64 cross-compiler must be installed.
# ---------------------------------------------------------------------------
if ! command -v "${CC}" >/dev/null 2>&1; then
    echo "error: ${CC} not found; install gcc-aarch64-linux-gnu or set CC" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build: try static first, fall back to dynamic.
#   Static linking is preferred for guest binaries — no libc dependency
#   in the minimal Ubuntu image.  However, some distros ship
#   gcc-aarch64-linux-gnu without a static libc, so we retry dynamically.
# ---------------------------------------------------------------------------
if "${CC}" -O2 -static -s -Wall -Wextra -o "${OUT}" "${SRC}"; then
    :
else
    echo "warning: static build failed; retrying dynamic build" >&2
    "${CC}" -O2 -s -Wall -Wextra -o "${OUT}" "${SRC}"
fi

# ---------------------------------------------------------------------------
# Verify the output binary was produced.
# ---------------------------------------------------------------------------
if [ ! -x "${OUT}" ]; then
    echo "error: build completed but ${OUT} was not produced" >&2
    exit 1
fi

file "${OUT}" 2>/dev/null || true
echo "built: ${OUT}"
