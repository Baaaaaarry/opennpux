#!/bin/sh
set -eu
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
HOST="${ROOT_DIR}/runtime/host/include/opennpux/npu_submission.h"
RTL="${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/npu_submission.h"
for token in OPENNPUX_NPU_INVOCATION_MAGIC OPENNPUX_NPU_INVOCATION_VERSION \
    OPENNPUX_NPU_INVOCATION_HEADER_SIZE OPENNPUX_NPU_TENSOR_BINDING_SIZE \
    OPENNPUX_NPU_COMMAND_SIZE OPENNPUX_NPU_COMPLETION_SIZE \
    OPENNPUX_NPU_TRACE_MAGIC OPENNPUX_NPU_TRACE_VERSION \
    OPENNPUX_NPU_TRACE_HEADER_SIZE OPENNPUX_NPU_TRACE_RECORD_SIZE \
    OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC \
    OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION \
    OPENNPUX_NPU_OPERATOR_PARAMETERS_SIZE; do
    host_value="$(awk -v t="$token" '$2 == t {print $3; exit}' "$HOST")"
    rtl_value="$(awk -v t="$token" '$2 == t {print $3; exit}' "$RTL")"
    [ -n "$host_value" ] && [ "$host_value" = "$rtl_value" ] || {
        echo "error: generic NPU ABI mismatch for $token" >&2
        exit 1
    }
done
echo "Generic NPU submission ABI headers match"
