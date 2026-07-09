#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
OUT_DIR="${CORAL_MOBILENET_MATRIX_OUT:-${ROOT_DIR}/simout/mobilenet-matrix}"
RUNNER="${ROOT_DIR}/tools/coralnpu/run_rvv_mobilenet_test.sh"
COMPARE="${ROOT_DIR}/tools/coralnpu/compare_mobilenet_results.sh"

mkdir -p "${OUT_DIR}"

run_mode() {
    mode="$1"
    log="${OUT_DIR}/mobilenet-${mode}.log"
    echo "[mobilenet-matrix] running mode=${mode}, log=${log}"
    CORAL_OPERATOR_MODE="${mode}" "${RUNNER}" 2>&1 | tee "${log}"
}

run_mode hybrid
run_mode rtl

"${COMPARE}" \
    "${OUT_DIR}/mobilenet-hybrid.log" \
    "${OUT_DIR}/mobilenet-rtl.log" | tee "${OUT_DIR}/mobilenet-compare.log"
