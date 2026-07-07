#!/bin/sh

set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
LOG="${1:-${ROOT_DIR}/simout/coral-mobilenet.debug}"

[ -f "${LOG}" ] || {
    echo "error: MobileNet log not found: ${LOG}" >&2
    exit 1
}

echo "MobileNet progress: ${LOG}"
grep 'Coral firmware progress=' "${LOG}" | tail -n 10 || true

latest_stats="$(grep 'Coral phase stats marker=' "${LOG}" | tail -n 1 || true)"
if [ -n "${latest_stats}" ]; then
    echo "Latest phase statistics:"
    printf '%s\n' "${latest_stats}"
else
    echo "No phase statistics found; rebuild the RVV bridge."
fi

latest_extmem="$(grep 'Coral local EXTMEM accesses=' "${LOG}" | tail -n 1 || true)"
if [ -n "${latest_extmem}" ]; then
    echo "Latest EXTMEM sample:"
    printf '%s\n' "${latest_extmem}"
fi
