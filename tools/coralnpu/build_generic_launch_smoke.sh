#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
"${ROOT_DIR}/tools/coralnpu/check_generic_test_abi.sh"
CORAL_NPU_LAUNCH_TARGET="//hw_sim:gem5_generic_launch_smoke.elf" \
CORAL_NPU_LAUNCH_OUTPUT="gem5_generic_launch_smoke.elf" \
    exec "${SCRIPT_DIR}/build_npu_launch_smoke.sh" "$@"
