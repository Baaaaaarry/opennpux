#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
HOST_HEADER="${ROOT_DIR}/runtime/host/include/opennpux/coral_command.h"
FIRMWARE_HEADER="${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/coral_command.h"

if ! cmp -s "${HOST_HEADER}" "${FIRMWARE_HEADER}"; then
    echo "error: host and Coral command ABI headers differ" >&2
    diff -u "${HOST_HEADER}" "${FIRMWARE_HEADER}" >&2 || true
    exit 1
fi

echo "Coral command ABI headers match"
