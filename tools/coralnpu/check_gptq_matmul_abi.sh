#!/bin/sh
# Verify that the host runtime and Coral firmware share one GPTQ MatMul request
# ABI. The bridge, the firmware, the host reference, and the offline
# materializer all decode the same 128-byte record.
#
# Callers: docs/runbooks/qwen35b_model_package.md, manual ABI review
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
HOST_HEADER="${ROOT_DIR}/runtime/host/include/opennpux/coral_gptq_matmul.h"
FIRMWARE_HEADER="${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/coral_gptq_matmul.h"
MATERIALIZER="${ROOT_DIR}/tools/models/materialize_gptq_projection.py"

if ! cmp -s "${HOST_HEADER}" "${FIRMWARE_HEADER}"; then
    echo "error: host and Coral GPTQ MatMul ABI headers differ" >&2
    diff -u "${HOST_HEADER}" "${FIRMWARE_HEADER}" >&2 || true
    exit 1
fi

for token in CORAL_GPTQ_MATMUL_MAGIC:REQUEST_MAGIC \
             CORAL_GPTQ_MATMUL_VERSION:REQUEST_VERSION; do
    header_name="${token%%:*}"
    python_name="${token##*:}"
    header_value="$(awk -v t="${header_name}" \
        '$2 == t {gsub(/UINT32_C\(|\)/, "", $3); print tolower($3); exit}' \
        "${HOST_HEADER}")"
    python_value="$(awk -v t="${python_name}" \
        '$1 == t && $2 == "=" {print tolower($3); exit}' "${MATERIALIZER}")"
    [ -n "${header_value}" ] && [ "${header_value}" = "${python_value}" ] || {
        echo "error: GPTQ MatMul ABI mismatch for ${header_name}" >&2
        exit 1
    }
done

echo "Coral GPTQ MatMul ABI headers match"
