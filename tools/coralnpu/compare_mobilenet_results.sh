#!/bin/sh

set -eu

usage() {
    echo "usage: $0 <hybrid-log> <rtl-log>" >&2
}

[ "$#" -eq 2 ] || {
    usage
    exit 2
}

HYBRID_LOG="$1"
RTL_LOG="$2"

[ -f "${HYBRID_LOG}" ] || {
    echo "error: hybrid log not found: ${HYBRID_LOG}" >&2
    exit 2
}
[ -f "${RTL_LOG}" ] || {
    echo "error: RTL log not found: ${RTL_LOG}" >&2
    exit 2
}

last_value() {
    key="$1"
    file="$2"
    awk -F= -v key="${key}" '
        $1 == key {
            value=$2
            gsub(/\r/, "", value)
        }
        END { if (value != "") print value }
    ' "${file}"
}

require_value() {
    key="$1"
    file="$2"
    value="$(last_value "${key}" "${file}")"
    if [ -z "${value}" ]; then
        echo "error: ${key} missing from ${file}" >&2
        exit 1
    fi
    printf '%s\n' "${value}"
}

compare_value() {
    name="$1"
    hybrid_value="$2"
    rtl_value="$3"
    echo "hybrid_${name}=${hybrid_value}"
    echo "rtl_${name}=${rtl_value}"
    if [ "${hybrid_value}" != "${rtl_value}" ]; then
        echo "mobilenet_compare=FAIL"
        echo "error: ${name} mismatch" >&2
        exit 1
    fi
}

hybrid_pass="$(last_value mobilenet_test "${HYBRID_LOG}")"
rtl_pass="$(last_value mobilenet_test "${RTL_LOG}")"
[ "${hybrid_pass}" = "PASS" ] || {
    echo "error: hybrid MobileNet did not report PASS" >&2
    exit 1
}
[ "${rtl_pass}" = "PASS" ] || {
    echo "error: RTL MobileNet did not report PASS" >&2
    exit 1
}

hybrid_cycles="$(require_value mobilenet_npu_cycles "${HYBRID_LOG}")"
rtl_cycles="$(require_value mobilenet_npu_cycles "${RTL_LOG}")"

echo "hybrid_npu_cycles=${hybrid_cycles}"
echo "rtl_npu_cycles=${rtl_cycles}"

for key in mobilenet_operation_count mobilenet_bytes_read mobilenet_bytes_written; do
    hybrid_value="$(last_value "${key}" "${HYBRID_LOG}")"
    rtl_value="$(last_value "${key}" "${RTL_LOG}")"
    if [ -n "${hybrid_value}" ] || [ -n "${rtl_value}" ]; then
        echo "hybrid_${key#mobilenet_}=${hybrid_value:-missing}"
        echo "rtl_${key#mobilenet_}=${rtl_value:-missing}"
        if [ -n "${hybrid_value}" ] && [ -n "${rtl_value}" ] &&
           [ "${hybrid_value}" != "${rtl_value}" ]; then
            echo "mobilenet_compare=FAIL"
            echo "error: ${key} mismatch" >&2
            exit 1
        fi
    fi
done

hybrid_checksum="$(last_value mobilenet_output_checksum "${HYBRID_LOG}")"
rtl_checksum="$(last_value mobilenet_output_checksum "${RTL_LOG}")"
hybrid_bytes="$(last_value mobilenet_output_bytes "${HYBRID_LOG}")"
rtl_bytes="$(last_value mobilenet_output_bytes "${RTL_LOG}")"

if [ -n "${hybrid_checksum}" ] && [ -n "${rtl_checksum}" ] &&
   [ -n "${hybrid_bytes}" ] && [ -n "${rtl_bytes}" ]; then
    echo "mobilenet_compare_scope=full-output-checksum"
    compare_value checksum "${hybrid_checksum}" "${rtl_checksum}"
    compare_value output_bytes "${hybrid_bytes}" "${rtl_bytes}"
else
    hybrid_output="$(require_value mobilenet_output "${HYBRID_LOG}")"
    rtl_output="$(require_value mobilenet_output "${RTL_LOG}")"
    echo "mobilenet_compare_scope=sample-output"
    echo "warning: mobilenet_output_checksum missing; rebuild/install the latest guest coralctl and rebuild the checkpoint for full-output comparison" >&2
    compare_value output "${hybrid_output}" "${rtl_output}"
fi

echo "mobilenet_compare=PASS"
