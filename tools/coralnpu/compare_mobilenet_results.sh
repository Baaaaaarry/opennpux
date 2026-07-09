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
    awk -F= -v key="${key}" '$1 == key { value=$2 } END { if (value != "") print value }' "${file}"
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

hybrid_checksum="$(require_value mobilenet_output_checksum "${HYBRID_LOG}")"
rtl_checksum="$(require_value mobilenet_output_checksum "${RTL_LOG}")"
hybrid_bytes="$(require_value mobilenet_output_bytes "${HYBRID_LOG}")"
rtl_bytes="$(require_value mobilenet_output_bytes "${RTL_LOG}")"
hybrid_cycles="$(require_value mobilenet_npu_cycles "${HYBRID_LOG}")"
rtl_cycles="$(require_value mobilenet_npu_cycles "${RTL_LOG}")"

echo "hybrid_checksum=${hybrid_checksum}"
echo "rtl_checksum=${rtl_checksum}"
echo "hybrid_output_bytes=${hybrid_bytes}"
echo "rtl_output_bytes=${rtl_bytes}"
echo "hybrid_npu_cycles=${hybrid_cycles}"
echo "rtl_npu_cycles=${rtl_cycles}"

if [ "${hybrid_checksum}" != "${rtl_checksum}" ]; then
    echo "mobilenet_compare=FAIL"
    echo "error: output checksum mismatch" >&2
    exit 1
fi

if [ "${hybrid_bytes}" != "${rtl_bytes}" ]; then
    echo "mobilenet_compare=FAIL"
    echo "error: output byte count mismatch" >&2
    exit 1
fi

echo "mobilenet_compare=PASS"
