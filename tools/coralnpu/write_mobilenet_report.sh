#!/bin/sh

set -eu

usage() {
    echo "usage: $0 <compare-log> <report.md>" >&2
}

[ "$#" -eq 2 ] || {
    usage
    exit 2
}

COMPARE_LOG="$1"
REPORT="$2"

[ -f "${COMPARE_LOG}" ] || {
    echo "error: compare log not found: ${COMPARE_LOG}" >&2
    exit 2
}

value() {
    key="$1"
    awk -F= -v key="${key}" '$1 == key { value=$2 } END { if (value != "") print value }' "${COMPARE_LOG}"
}

value_or_na() {
    key="$1"
    result="$(value "${key}")"
    if [ -z "${result}" ]; then
        printf 'n/a\n'
    else
        printf '%s\n' "${result}"
    fi
}

mkdir -p "$(dirname "${REPORT}")"

{
    echo "# MobileNet RTL/Hybrid Report"
    echo
    echo "- compare: $(value_or_na mobilenet_compare)"
    echo "- scope: $(value_or_na mobilenet_compare_scope)"
    echo "- hybrid checksum: $(value_or_na hybrid_checksum)"
    echo "- rtl checksum: $(value_or_na rtl_checksum)"
    echo
    echo "| Metric | Hybrid | RTL |"
    echo "| --- | ---: | ---: |"
    echo "| NPU cycles | $(value_or_na hybrid_npu_cycles) | $(value_or_na rtl_npu_cycles) |"
    echo "| Operation count | $(value_or_na hybrid_operation_count) | $(value_or_na rtl_operation_count) |"
    echo "| Bytes read | $(value_or_na hybrid_bytes_read) | $(value_or_na rtl_bytes_read) |"
    echo "| Bytes written | $(value_or_na hybrid_bytes_written) | $(value_or_na rtl_bytes_written) |"
    echo "| Total bytes | $(value_or_na hybrid_total_bytes) | $(value_or_na rtl_total_bytes) |"
    echo "| Cycles/op | $(value_or_na hybrid_cycles_per_operation) | $(value_or_na rtl_cycles_per_operation) |"
    echo "| Bytes/cycle | $(value_or_na hybrid_bytes_per_cycle) | $(value_or_na rtl_bytes_per_cycle) |"
    echo
    echo "- RTL/hybrid cycle ratio: $(value_or_na rtl_to_hybrid_cycle_ratio)"
} > "${REPORT}"

echo "mobilenet_report=${REPORT}"
