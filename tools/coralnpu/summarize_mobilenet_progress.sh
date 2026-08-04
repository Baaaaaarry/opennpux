#!/bin/sh

set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
LOG="${1:-${ROOT_DIR}/logs/sim/coral-mobilenet.debug}"

[ -f "${LOG}" ] || {
    echo "error: MobileNet log not found: ${LOG}" >&2
    exit 1
}

marker_name() {
    case "$1" in
        0x4d4e0100) echo "crt-ready" ;;
        0x4d4e0200) echo "main" ;;
        0x4d4e0201) echo "mailbox-ready" ;;
        0x4d4e0300) echo "allocate-begin" ;;
        0x4d4e0301) echo "allocate-end" ;;
        0x4d4e0400) echo "input-ready" ;;
        0x4d4e0500) echo "invoke-begin" ;;
        0x4d4e0501) echo "invoke-end" ;;
        0x4d4e0600) echo "conv-begin" ;;
        0x4d4e0601) echo "conv-end" ;;
        0x4d4e0700) echo "depthwise-begin" ;;
        0x4d4e0701) echo "depthwise-end" ;;
        0x4d4eff01) echo "exception-mepc-next" ;;
        0x4d4eff02) echo "exception-mtval-next" ;;
        0x4d4eff03) echo "exception-mcause-next" ;;
        *) echo "unknown" ;;
    esac
}

value_of() {
    key="$1"
    line="$2"
    printf '%s\n' "${line}" | tr ' ' '\n' | awk -F= -v key="${key}" '$1 == key { print $2 }'
}

last_line_matching() {
    pattern="$1"
    grep "${pattern}" "${LOG}" | tail -n 1 || true
}

echo "mobilenet_log=${LOG}"

last_progress="$(last_line_matching 'Coral firmware progress=')"
if [ -n "${last_progress}" ]; then
    marker="${last_progress##*=}"
    echo "last_progress=${marker}"
    echo "last_progress_name=$(marker_name "${marker}")"
fi

fault="$(last_line_matching 'Coral RTL core fault:')"
if [ -n "${fault}" ]; then
    echo "rtl_fault=1"
    printf '%s\n' "${fault}" |
        sed -n 's/.*mepc=\([^ ]*\).*mtval=\([^ ]*\).*mcause=\([^ ]*\).*/rtl_fault_mepc=\1\nrtl_fault_mtval=\2\nrtl_fault_mcause=\3/p'
else
    echo "rtl_fault=0"
fi

last_heartbeat="$(last_line_matching 'Coral RTL heartbeat event=')"
if [ -n "${last_heartbeat}" ]; then
    echo "last_heartbeat_event=$(value_of event "${last_heartbeat}")"
    echo "last_heartbeat_requested_cycles=$(value_of requested_cycles "${last_heartbeat}")"
    echo "last_heartbeat_step_result=$(value_of step_result "${last_heartbeat}")"
fi

last_extmem="$(last_line_matching 'Coral local EXTMEM accesses=')"
if [ -n "${last_extmem}" ]; then
    echo "extmem_accesses=$(value_of accesses "${last_extmem}")"
    echo "extmem_reads=$(value_of reads "${last_extmem}")"
    echo "extmem_writes=$(value_of writes "${last_extmem}")"
    echo "extmem_bytes=$(value_of bytes "${last_extmem}")"
    echo "extmem_last=$(value_of last "${last_extmem}")"
fi

last_dma_wait="$(last_line_matching 'Coral RTL waiting for DMA')"
if [ -n "${last_dma_wait}" ]; then
    echo "last_dma_wait_count=$(value_of count "${last_dma_wait}")"
    echo "last_dma_wait_type=$(value_of type "${last_dma_wait}")"
    echo "last_dma_wait_addr=$(value_of addr "${last_dma_wait}")"
    echo "last_dma_wait_size=$(value_of size "${last_dma_wait}")"
fi

last_dma_complete="$(last_line_matching 'Coral DMA complete count=')"
if [ -n "${last_dma_complete}" ]; then
    echo "last_dma_complete_count=$(value_of count "${last_dma_complete}")"
    echo "last_dma_complete_mode=$(value_of mode "${last_dma_complete}")"
fi

echo
echo "Recent progress markers:"
grep 'Coral firmware progress=' "${LOG}" | tail -n 12 | while IFS= read -r line; do
    marker="${line##*=}"
    printf '%s %s\n' "${marker}" "$(marker_name "${marker}")"
done

echo
echo "Phase statistics:"
awk '
    /Coral phase stats marker=/ {
        marker = cycles = delta = wall = accesses = daccesses = bytes = dbytes = ""
        for (i = 1; i <= NF; ++i) {
            split($i, kv, "=")
            if (kv[1] == "marker") marker = kv[2]
            if (kv[1] == "cycles") cycles = kv[2]
            if (kv[1] == "delta_cycles") delta = kv[2]
            if (kv[1] == "wall_ms") wall = kv[2]
            if (kv[1] == "accesses") accesses = kv[2]
            if (kv[1] == "delta_accesses") daccesses = kv[2]
            if (kv[1] == "bytes") bytes = kv[2]
            if (kv[1] == "delta_bytes") dbytes = kv[2]
        }
        printf "%s cycles=%s delta_cycles=%s wall_ms=%s accesses=%s delta_accesses=%s bytes=%s delta_bytes=%s\n", marker, cycles, delta, wall, accesses, daccesses, bytes, dbytes
    }
' "${LOG}" | tail -n 20

echo
echo "Operator phase summary:"
grep 'Coral operator phase summary' "${LOG}" | tail -n 20 || true

echo
echo "Hybrid operator summary:"
grep 'Coral hybrid operator summary' "${LOG}" | tail -n 20 || true
