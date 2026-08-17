#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
HOST_HEADER="${ROOT_DIR}/runtime/host/include/opennpux/coral_gptq_expert.h"
FIRMWARE_HEADER="${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/coral_gptq_expert.h"
MATERIALIZER="${ROOT_DIR}/tools/models/materialize_gptq_expert.py"

extract_structs()
{
    awk '
        /^struct coral_gptq_projection_weights/ {capture=1}
        /^struct coral_gptq_expert_request/ {capture=1}
        capture {gsub(/[[:space:]]/, ""); print}
        capture && /^};/ {capture=0}
    ' "$1"
}

HOST_LAYOUT="$(extract_structs "$HOST_HEADER")"
FIRMWARE_LAYOUT="$(extract_structs "$FIRMWARE_HEADER")"
[ "$HOST_LAYOUT" = "$FIRMWARE_LAYOUT" ] || {
    echo "error: host and Coral GPTQ Expert ABI layouts differ" >&2
    exit 1
}

for token in CORAL_GPTQ_EXPERT_MAGIC:REQUEST_MAGIC \
             CORAL_GPTQ_EXPERT_VERSION:REQUEST_VERSION; do
    header_name="${token%%:*}"
    python_name="${token##*:}"
    header_value="$(awk -v t="$header_name" \
        '$2 == t {gsub(/UINT32_C\(|\)/, "", $3); print tolower($3); exit}' \
        "$HOST_HEADER")"
    python_value="$(awk -v t="$python_name" \
        '$1 == t && $2 == "=" {print tolower($3); exit}' "$MATERIALIZER")"
    [ -n "$header_value" ] && [ "$header_value" = "$python_value" ] || {
        echo "error: GPTQ Expert ABI mismatch for $header_name" >&2
        exit 1
    }
done

echo "Coral GPTQ Expert ABI headers match"
