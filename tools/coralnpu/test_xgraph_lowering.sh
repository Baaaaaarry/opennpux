#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BUILD_DIR="${ROOT_DIR}/build/local-tests/xgraph-lowering"
CC="${CC:-cc}"

mkdir -p "${BUILD_DIR}"
"${CC}" -std=c11 -Wall -Wextra -Werror -pedantic \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/npu_xgraph_lowering.c" \
    "${ROOT_DIR}/runtime/host/src/npu_gptq_tile_plan.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_xgraph_lowering_test.c" \
    -o "${BUILD_DIR}/npu_xgraph_lowering_test"
"${BUILD_DIR}/npu_xgraph_lowering_test"

"${CC}" -std=c11 -Wall -Wextra -Werror -pedantic \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/npu_gptq_tile_plan.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_gptq_tile_plan_test.c" \
    -o "${BUILD_DIR}/npu_gptq_tile_plan_test"
"${BUILD_DIR}/npu_gptq_tile_plan_test"

"${CC}" -std=c11 -Wall -Wextra -Werror -pedantic \
    -I"${ROOT_DIR}/runtime/host/include" \
    -c "${ROOT_DIR}/runtime/host/src/npu_xgraph_lowering.c" \
    -o "${BUILD_DIR}/npu_xgraph_lowering.o"
"${CC}" -std=c11 -Wall -Wextra -Werror -pedantic \
    -I"${ROOT_DIR}/runtime/host/include" \
    -c "${ROOT_DIR}/runtime/host/src/npu_gptq_tile_plan.c" \
    -o "${BUILD_DIR}/npu_gptq_tile_plan.o"
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
    -I"${ROOT_DIR}/sim/coralnpu" \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_xgraph_lowering_audit.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_xgraph_lowering_audit_test.cc" \
    "${BUILD_DIR}/npu_xgraph_lowering.o" \
    "${BUILD_DIR}/npu_gptq_tile_plan.o" \
    -o "${BUILD_DIR}/gem5_xgraph_lowering_audit_test"
"${BUILD_DIR}/gem5_xgraph_lowering_audit_test"
echo "XGraph materialized-request audit test: PASS"
