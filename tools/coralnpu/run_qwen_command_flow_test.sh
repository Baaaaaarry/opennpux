#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CORAL_NPU_LAUNCH_FIRMWARE="${ROOT_DIR}/build/coralnpu/gem5_qwen_command_flow_smoke.elf" \
CORAL_NPU_LAUNCH_TEST_SCRIPT="${ROOT_DIR}/thirdparty/gem5/configs/coralnpu/coral-qwen-command-flow-test.rcS" \
CORAL_NPU_LAUNCH_HOST_LOG="${ROOT_DIR}/simout/coral-qwen-command-flow-host.log" \
CORAL_NPU_LAUNCH_DEBUG_LOG="${ROOT_DIR}/simout/coral-qwen-command-flow.debug" \
CORAL_NPU_LAUNCH_XOPENNPUX=1 \
CORAL_NPU_LAUNCH_EXPECTED_GUEST_VERDICT="qwen_command_flow=PASS" \
CORAL_NPU_LAUNCH_EXPECTED_XOPENNPUX_OPS="tgather tmma tadd tmul trmsnorm trope tsilu tsoftmax ttopk" \
    exec "${SCRIPT_DIR}/run_npu_launch_test.sh" "$@"
