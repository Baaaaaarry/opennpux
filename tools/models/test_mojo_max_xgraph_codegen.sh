#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BUILD_DIR="${ROOT_DIR}/build/local-tests/mojo-max-xgraph"
CC="${CC:-cc}"
PYTHON="${PYTHON:-python3}"

mkdir -p "${BUILD_DIR}"
LOWERING_LIB="${BUILD_DIR}/libopennpux_xgraph_codegen.so"
CC_ARCH_FLAGS=""
if [ "$(uname -s)" = Darwin ]; then
    CC_ARCH_FLAGS="-arch $("${PYTHON}" -c 'import platform; print(platform.machine())')"
fi
# shellcheck disable=SC2086
"${CC}" ${CC_ARCH_FLAGS} -std=c11 -Wall -Wextra -Werror -pedantic -fPIC -shared \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/npu_xgraph_codegen_ffi.c" \
    "${ROOT_DIR}/runtime/host/src/npu_xgraph_lowering.c" \
    "${ROOT_DIR}/runtime/host/src/npu_gptq_tile_plan.c" \
    -lm -o "${LOWERING_LIB}"

"${PYTHON}" -m unittest \
    tests.unit.models.test_opennpux_mojo_adapter \
    tests.unit.models.test_opennpux_backend_frontend
"${PYTHON}" "${SCRIPT_DIR}/compile_mojo_max_xgraph.py" \
    "${ROOT_DIR}/tests/fixtures/models/mojo_max_basic.json" \
    "${BUILD_DIR}/mojo-basic.npxg" \
    --lowering-library "${LOWERING_LIB}"
"${PYTHON}" "${SCRIPT_DIR}/create_mojo_max_basic_arena.py" \
    "${BUILD_DIR}/mojo-basic.npxg.json" \
    "${BUILD_DIR}/mojo-basic.arena.bin"
"${PYTHON}" "${SCRIPT_DIR}/compile_opennpux_backend.py" \
    "${ROOT_DIR}/tests/fixtures/models/tvm_byoc_basic.json" \
    "${BUILD_DIR}/backend-basic.npxg" \
    --lowering-library "${LOWERING_LIB}"
cmp "${BUILD_DIR}/mojo-basic.npxg" "${BUILD_DIR}/backend-basic.npxg"

"${CC}" -std=c11 -Wall -Wextra -Werror -pedantic \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/tests/unit/models/tvm_byoc_xgraph_artifact_test.c" \
    -o "${BUILD_DIR}/xgraph_artifact_test"
"${BUILD_DIR}/xgraph_artifact_test" "${BUILD_DIR}/mojo-basic.npxg"
"${PYTHON}" "${SCRIPT_DIR}/test_max_sdk_opennpux_export.py" \
    "${BUILD_DIR}/max-sdk-projection.npxg" \
    --lowering-library "${LOWERING_LIB}"

echo "mojo_max_tvm_artifact_equivalence=PASS"
echo "mojo_max_xgraph_codegen=PASS"
