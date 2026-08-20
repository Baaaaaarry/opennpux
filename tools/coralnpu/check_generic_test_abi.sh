#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
HOST_HEADER="${ROOT_DIR}/runtime/host/include/opennpux/coral_generic_test.h"
FIRMWARE_HEADER="${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/coral_generic_test.h"

if ! cmp -s "${HOST_HEADER}" "${FIRMWARE_HEADER}"; then
    echo "error: Coral generic test mailbox ABI headers differ" >&2
    diff -u "${HOST_HEADER}" "${FIRMWARE_HEADER}" || true
    exit 1
fi

echo "Coral generic test mailbox ABI headers match"
