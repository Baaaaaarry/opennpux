#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CORAL_NPU_LAUNCH_FIRMWARE="${ROOT_DIR}/build/coralnpu/gem5_generic_launch_smoke.elf" \
CORAL_NPU_LAUNCH_HOST_LOG="${ROOT_DIR}/simout/coral-generic-launch-host.log" \
CORAL_NPU_LAUNCH_DEBUG_LOG="${ROOT_DIR}/simout/coral-generic-launch.debug" \
    exec "${SCRIPT_DIR}/run_npu_launch_test.sh" "$@"
