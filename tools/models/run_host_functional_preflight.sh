#!/bin/sh
set -eu

# Purpose: build and run the bridge's Host C++ functional graph without gem5.
# Inputs: model directory, CSV token IDs, and generation count.
# Output: build/host-tools/gem5_host_functional_runner and generated token IDs.

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
MODEL_DIR=${1:?usage: run_host_functional_preflight.sh MODEL_DIR TOKEN_IDS MAX_NEW_TOKENS}
TOKEN_IDS=${2:?usage: run_host_functional_preflight.sh MODEL_DIR TOKEN_IDS MAX_NEW_TOKENS}
MAX_NEW_TOKENS=${3:?usage: run_host_functional_preflight.sh MODEL_DIR TOKEN_IDS MAX_NEW_TOKENS}
BUILD_DIR="${ROOT_DIR}/build/host-tools"
WORK_DIR="${ROOT_DIR}/build/host-functional-preflight"
CXX=${CXX:-c++}
CC=${CC:-cc}
mkdir -p "$BUILD_DIR" "$WORK_DIR"

for source in npu_submission npu_executable npu_functional_materializer \
              npu_tensor_plan model_package npu_weight_ranges; do
    "$CC" -O2 -Wall -Wextra -Werror -std=c11 \
        -I"${ROOT_DIR}/runtime/host/include" \
        -c "${ROOT_DIR}/runtime/host/src/${source}.c" \
        -o "${WORK_DIR}/${source}.o"
done

"$CXX" -O2 -Wall -Wextra -Werror -std=c++17 \
    -I"${ROOT_DIR}/sim/coralnpu" \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_gptq_kernels.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_generic_gptq_executor.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_transformer_kernels.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_host_functional_backend.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_generic_command_dispatch.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_host_tensor_arena.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_host_weight_provider.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_host_routed_expert.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_host_functional_graph.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_host_functional_runner.cc" \
    "${WORK_DIR}/npu_submission.o" \
    "${WORK_DIR}/npu_executable.o" \
    "${WORK_DIR}/npu_functional_materializer.o" \
    "${WORK_DIR}/npu_tensor_plan.o" \
    "${WORK_DIR}/model_package.o" \
    "${WORK_DIR}/npu_weight_ranges.o" \
    -o "${BUILD_DIR}/gem5_host_functional_runner"

exec "${BUILD_DIR}/gem5_host_functional_runner" \
    "$MODEL_DIR/model.npxc" "$MODEL_DIR/model.npxtb" \
    "$MODEL_DIR/model.npxm" "$MODEL_DIR/model.npxr" \
    "$TOKEN_IDS" "$MAX_NEW_TOKENS"
