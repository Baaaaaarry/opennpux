#!/bin/sh
set -eu
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
HOST="${ROOT_DIR}/runtime/host/include/opennpux/npu_submission.h"
RTL="${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/npu_submission.h"
HOST_ROUTE="${ROOT_DIR}/runtime/host/include/opennpux/npu_route_table.h"
RTL_ROUTE="${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/npu_route_table.h"
HOST_INFERENCE="${ROOT_DIR}/runtime/host/include/opennpux/npu_inference_io.h"
RTL_INFERENCE="${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/npu_inference_io.h"
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
for token in OPENNPUX_NPU_ROUTE_TABLE_MAGIC \
    OPENNPUX_NPU_ROUTE_TABLE_VERSION \
    OPENNPUX_NPU_ROUTE_TABLE_HEADER_SIZE OPENNPUX_NPU_ROUTE_RECORD_SIZE \
    OPENNPUX_NPU_MAX_ACTIVE_EXPERTS; do
    host_value="$(awk -v t="$token" '$2 == t {print $3; exit}' "$HOST_ROUTE")"
    rtl_value="$(awk -v t="$token" '$2 == t {print $3; exit}' "$RTL_ROUTE")"
    [ -n "$host_value" ] && [ "$host_value" = "$rtl_value" ] || {
        echo "error: NPU route table ABI mismatch for $token" >&2
        exit 1
    }
done
for token in OPENNPUX_NPU_INFERENCE_IO_MAGIC \
    OPENNPUX_NPU_INFERENCE_IO_VERSION \
    OPENNPUX_NPU_INFERENCE_PROMPT_BYTES \
    OPENNPUX_NPU_INFERENCE_MAX_RESULT_TOKENS \
    OPENNPUX_NPU_INFERENCE_TOKEN_IDS_OFFSET; do
    host_value="$(awk -v t="$token" '$2 == t {print $3; exit}' \
        "$HOST_INFERENCE")"
    rtl_value="$(awk -v t="$token" '$2 == t {print $3; exit}' \
        "$RTL_INFERENCE")"
    [ -n "$host_value" ] && [ "$host_value" = "$rtl_value" ] || {
        echo "error: NPU inference I/O ABI mismatch for $token" >&2
        exit 1
    }
done
echo "Generic NPU submission ABI headers match"
