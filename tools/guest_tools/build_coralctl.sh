#!/bin/sh
#
# build_coralctl.sh — Cross-compile the coralctl NPU control utility for aarch64.
#
# Builds a static aarch64 binary that links the reusable host runtime
# (coral_runtime.c) with the CLI frontend (coralctl.c).  The binary is
# installed into the guest disk image and runs inside the gem5 ARM Linux
# simulation.
#
# coralctl is the primary guest-side NPU interface.  It auto-selects the
# transport: /dev/opennpux-coral driver when available, /dev/mem otherwise.
# Supported commands: info, run, dma-test, vector-add, mem-info, mem-clear,
# mem-read32, mem-write32.
#
# Pipeline:
#   runtime/host/src/coral_runtime.c  ─┐
#   runtime/host/tools/coralctl.c     ─┤
#   runtime/host/include/opennpux/*.h ─┘
#     → aarch64-linux-gnu-gcc -static
#       → build/guest-tools/coralctl-aarch64
#
# Output:
#   build/guest-tools/coralctl-aarch64    static aarch64 guest binary
#
# Environment:
#   CROSS      cross-compiler prefix (default: aarch64-linux-gnu-)
#   CC         compiler (default: ${CROSS}gcc)
#
# @guest-tools-spec  v1  2025-07-28

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
OUT_DIR="${ROOT_DIR}/build/guest-tools"
OUT="${OUT_DIR}/coralctl-aarch64"
CROSS="${CROSS:-aarch64-linux-gnu-}"
CC="${CC:-${CROSS}gcc}"

# ---------------------------------------------------------------------------
# Preflight: the aarch64 cross-compiler must be installed.
#   On Debian/Ubuntu: sudo apt-get install gcc-aarch64-linux-gnu
# ---------------------------------------------------------------------------
if ! command -v "${CC}" >/dev/null 2>&1; then
    echo "error: ${CC} not found; install gcc-aarch64-linux-gnu or set CC" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build: static aarch64 binary.
#   -O2 -static: small, self-contained binary (no guest libc dependency).
#   -Wall -Wextra -Werror: strict warnings to catch ABI mismatches early.
#   Two source files: coral_runtime.c (transport abstraction) + coralctl.c (CLI).
# ---------------------------------------------------------------------------
mkdir -p "${OUT_DIR}"
"${CC}" -O2 -static -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/coral_runtime.c" \
    "${ROOT_DIR}/runtime/host/tools/coralctl.c" \
    -o "${OUT}"

# ---------------------------------------------------------------------------
# Verify the output binary was produced.
# ---------------------------------------------------------------------------
if [ ! -x "${OUT}" ]; then
    echo "error: build completed but ${OUT} was not produced" >&2
    exit 1
fi

echo "built: ${OUT}"
