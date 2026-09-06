#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BUILD_DIR="${ROOT_DIR}/build/local-tests/tvm-byoc-xgraph"
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
export OPENNPUX_XGRAPH_LOWERING_LIB="${LOWERING_LIB}"
"${PYTHON}" -m unittest discover \
    -s "${ROOT_DIR}/tests/unit/models" \
    -p 'test_tvm_byoc*_codegen.py'
"${PYTHON}" "${SCRIPT_DIR}/compile_tvm_byoc_xgraph.py" \
    "${ROOT_DIR}/tests/fixtures/models/tvm_byoc_basic.json" \
    "${BUILD_DIR}/basic.npxg"
"${PYTHON}" "${SCRIPT_DIR}/compile_tvm_byoc_module.py" \
    "${ROOT_DIR}/tests/fixtures/models/tvm_byoc_module.json" \
    "${BUILD_DIR}/module"
"${PYTHON}" - "${BUILD_DIR}/module/module.npxgm.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["execution_order"] == ["residual", "activation"]
assert manifest["region_count"] == 2
assert manifest["total_commands"] == 2
PY
"${PYTHON}" - "${BUILD_DIR}/basic.npxg" <<'PY'
import struct
import sys

data = open(sys.argv[1], "rb").read()
header = struct.unpack_from("<12I2Q8I", data)
assert header[0] == 0x5847504E
assert header[1] == 2
assert header[4] == 5
assert len(data) == 96 + 5 * 64
PY
"${CC}" -std=c11 -Wall -Wextra -Werror -pedantic \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/tests/unit/models/tvm_byoc_xgraph_artifact_test.c" \
    -o "${BUILD_DIR}/tvm_byoc_xgraph_artifact_test"
"${BUILD_DIR}/tvm_byoc_xgraph_artifact_test" "${BUILD_DIR}/basic.npxg"
echo "TVM BYOC XGraph codegen test: PASS"

if [ -z "${TVM_HOME:-}" ] && [ -f "${ROOT_DIR}/.cache/tvm/env.sh" ]; then
    # shellcheck disable=SC1091
    . "${ROOT_DIR}/.cache/tvm/env.sh"
fi
if [ -n "${TVM_HOME:-}" ]; then
    TVM_PYTHON="${TVM_PYTHON:-${PYTHON}}"
    TVM_BUILD_DIR="${TVM_BUILD_DIR:-${TVM_HOME}/build}"
    export PYTHONPATH="${TVM_HOME}/python:${ROOT_DIR}/tools/models${PYTHONPATH:+:${PYTHONPATH}}"
    export TVM_LIBRARY_PATH="${TVM_BUILD_DIR}/lib"
    export LD_LIBRARY_PATH="${TVM_BUILD_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export DYLD_LIBRARY_PATH="${TVM_BUILD_DIR}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
    "${TVM_PYTHON}" "${SCRIPT_DIR}/create_tvm_byoc_relax_e2e.py" \
        "${BUILD_DIR}/relax-model.json"
    "${TVM_PYTHON}" "${SCRIPT_DIR}/compile_tvm_byoc_xgraph.py" \
        "${BUILD_DIR}/relax-model.json" "${BUILD_DIR}/relax-model.npxg" \
        --dump-byoc-graph "${BUILD_DIR}/relax-model.byoc.json"
    "${TVM_PYTHON}" "${SCRIPT_DIR}/create_tvm_byoc_multi_region.py" \
        "${BUILD_DIR}/multi-region-model.json"
    "${TVM_PYTHON}" "${SCRIPT_DIR}/create_tvm_transformer_block.py" \
        "${BUILD_DIR}/transformer-block.json"
    "${TVM_PYTHON}" "${SCRIPT_DIR}/compile_tvm_byoc_xgraph.py" \
        "${BUILD_DIR}/transformer-block.json" \
        "${BUILD_DIR}/transformer-block.npxg" \
        --dump-byoc-graph "${BUILD_DIR}/transformer-block.byoc.json"
    "${TVM_PYTHON}" "${SCRIPT_DIR}/compile_tvm_byoc_module.py" \
        "${BUILD_DIR}/multi-region-model.json" \
        "${BUILD_DIR}/multi-region-module" \
        --dump-byoc-module "${BUILD_DIR}/multi-region-module.json"
    "${TVM_PYTHON}" - "${BUILD_DIR}/multi-region-module/module.npxgm.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["region_count"] == 2
assert manifest["total_commands"] == 2
assert len(manifest["execution_order"]) == 2
assert manifest["edges"] == []
assert len(manifest["host_bindings"]) == 1
host_binding = manifest["host_bindings"][0]
assert host_binding["pipeline"] == [{"op": "relax.nn.relu", "attrs": {}}]
assert host_binding["bytes"] == 32
print("xgraph_module_direct_edges=0")
print("xgraph_module_host_bindings=1")
print("xgraph_module_host_pipeline=relax.nn.relu")
PY
    "${TVM_PYTHON}" "${SCRIPT_DIR}/build_xgraph_tensor_image.py" \
        "${BUILD_DIR}/relax-model.npxg.json" \
        "${ROOT_DIR}/tests/fixtures/models/tvm_byoc_relax_values.json" \
        "${BUILD_DIR}/relax-model.arena.bin"
    "${TVM_PYTHON}" "${SCRIPT_DIR}/build_xgraph_tensor_image.py" \
        "${BUILD_DIR}/transformer-block.npxg.json" \
        "${ROOT_DIR}/tests/fixtures/models/tvm_transformer_block_values.json" \
        "${BUILD_DIR}/transformer-block.arena.bin"
    "${CC}" -std=c11 -Wall -Wextra -Werror -pedantic \
        -I"${ROOT_DIR}/runtime/host/include" \
        "${ROOT_DIR}/tests/unit/models/tvm_byoc_relax_e2e_test.c" \
        -lm -o "${BUILD_DIR}/tvm_byoc_relax_e2e_test"
    "${BUILD_DIR}/tvm_byoc_relax_e2e_test" \
        "${BUILD_DIR}/relax-model.npxg" \
        "${BUILD_DIR}/relax-model.arena.bin"
    "${CC}" -std=c11 -Wall -Wextra -Werror -pedantic \
        -I"${ROOT_DIR}/runtime/host/include" \
        "${ROOT_DIR}/tests/unit/models/tvm_transformer_block_e2e_test.c" \
        -lm -o "${BUILD_DIR}/tvm_transformer_block_e2e_test"
    "${BUILD_DIR}/tvm_transformer_block_e2e_test" \
        "${BUILD_DIR}/transformer-block.npxg" \
        "${BUILD_DIR}/transformer-block.arena.bin"
    "${CC}" -std=c11 -Wall -Wextra -Werror -pedantic \
        -I"${ROOT_DIR}/runtime/host/include" \
        "${ROOT_DIR}/runtime/host/src/xgraph_artifact.c" \
        "${ROOT_DIR}/tests/unit/models/tvm_byoc_xgraph_loader_test.c" \
        -o "${BUILD_DIR}/tvm_byoc_xgraph_loader_test"
    "${BUILD_DIR}/tvm_byoc_xgraph_loader_test" \
        "${BUILD_DIR}/relax-model.npxg" \
        "${BUILD_DIR}/relax-model.arena.bin" 6 64
    "${BUILD_DIR}/tvm_byoc_xgraph_loader_test" \
        "${BUILD_DIR}/transformer-block.npxg" \
        "${BUILD_DIR}/transformer-block.arena.bin" 4 512
    echo "tvm_transformer_block_byoc_e2e=PASS"
    echo "tvm_relax_byoc_e2e=PASS"
else
    echo "TVM Relax end-to-end test: SKIP (set TVM_HOME and TVM_PYTHON)"
    if [ "${OPENNPUX_REQUIRE_TVM:-0}" = 1 ]; then
        echo "TVM Relax end-to-end test: FAIL (TVM is required)" >&2
        exit 1
    fi
fi
