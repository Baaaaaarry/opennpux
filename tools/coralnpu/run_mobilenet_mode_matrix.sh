#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
OUT_DIR="${CORAL_MOBILENET_MATRIX_OUT:-${ROOT_DIR}/simout/mobilenet-matrix}"
RUNNER="${ROOT_DIR}/tools/coralnpu/run_rvv_mobilenet_test.sh"
COMPARE="${ROOT_DIR}/tools/coralnpu/compare_mobilenet_results.sh"
WRITE_REPORT="${ROOT_DIR}/tools/coralnpu/write_mobilenet_report.sh"
SUMMARIZE_PROGRESS="${ROOT_DIR}/tools/coralnpu/summarize_mobilenet_progress.sh"
TERMINAL_CANDIDATES="
${ROOT_DIR}/thirdparty/gem5/m5out/system.terminal
${ROOT_DIR}/m5out/system.terminal
"

mkdir -p "${OUT_DIR}"

clear_terminal_logs() {
    for terminal in ${TERMINAL_CANDIDATES}; do
        [ ! -f "${terminal}" ] || : > "${terminal}"
    done
}

append_terminal_log() {
    mode="$1"
    log="$2"
    found=0
    for terminal in ${TERMINAL_CANDIDATES}; do
        if [ -s "${terminal}" ]; then
            {
                echo
                echo "[mobilenet-matrix] guest terminal for mode=${mode}: ${terminal}"
                cat "${terminal}"
            } >> "${log}"
            found=1
        fi
    done
    if [ "${found}" -eq 0 ]; then
        echo "[mobilenet-matrix] warning: no guest terminal log found for mode=${mode}" | tee -a "${log}" >&2
    fi
}

run_mode() {
    mode="$1"
    log="${OUT_DIR}/mobilenet-${mode}.log"
    debug_log="${OUT_DIR}/mobilenet-${mode}.debug"
    summary="${OUT_DIR}/mobilenet-${mode}.summary"
    echo "[mobilenet-matrix] running mode=${mode}, log=${log}"
    clear_terminal_logs
    CORAL_OPERATOR_MODE="${mode}" \
    CORAL_MOBILENET_DEBUG_LOG="${debug_log}" \
        "${RUNNER}" 2>&1 | tee "${log}"
    append_terminal_log "${mode}" "${log}"
    if [ "${CORAL_MOBILENET_DEBUG:-0}" = "1" ] && [ -f "${debug_log}" ]; then
        "${SUMMARIZE_PROGRESS}" "${debug_log}" | tee "${summary}"
    fi
}

run_mode hybrid
run_mode rtl

"${COMPARE}" \
    "${OUT_DIR}/mobilenet-hybrid.log" \
    "${OUT_DIR}/mobilenet-rtl.log" | tee "${OUT_DIR}/mobilenet-compare.log"

"${WRITE_REPORT}" \
    "${OUT_DIR}/mobilenet-compare.log" \
    "${OUT_DIR}/mobilenet-report.md"
