#!/bin/sh
#
# test_host_functional_backend.sh - Test native Host C++ tensor kernels.
#
# Builds the model-independent functional backend directly with the host C++
# compiler. This is a fast pre-Bazel correctness gate for Transformer kernels;
# it does not run gem5, firmware, RTL, or publish an external numerical result.
#
# Pipeline:
#   gem5_transformer_kernels_test
#     -> gem5_host_functional_backend_test
#     -> gem5_generic_command_dispatch_test
#
# Environment:
#   CXX  Host C++ compiler (default: c++)
#
# Output:
#   build/local-tests/host-functional/*
#
# @coral-build-spec  v1  2026-08-20

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
OUT_DIR="${ROOT_DIR}/build/local-tests/host-functional"
CXX="${CXX:-c++}"
SOURCE_DIR="${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge"

mkdir -p "${OUT_DIR}"

"${CXX}" -O2 -Wall -Wextra -Werror -std=c++17 \
    -I"${ROOT_DIR}/sim/coralnpu" \
    "${SOURCE_DIR}/gem5_transformer_kernels.cc" \
    "${SOURCE_DIR}/gem5_transformer_kernels_test.cc" \
    -o "${OUT_DIR}/gem5_transformer_kernels_test"
"${OUT_DIR}/gem5_transformer_kernels_test"

"${CXX}" -O2 -Wall -Wextra -Werror -std=c++17 \
    -I"${ROOT_DIR}/sim/coralnpu" \
    "${SOURCE_DIR}/gem5_gptq_kernels.cc" \
    "${SOURCE_DIR}/gem5_generic_gptq_executor.cc" \
    "${SOURCE_DIR}/gem5_transformer_kernels.cc" \
    "${SOURCE_DIR}/gem5_host_functional_backend.cc" \
    "${SOURCE_DIR}/gem5_host_functional_backend_test.cc" \
    -o "${OUT_DIR}/gem5_host_functional_backend_test"
"${OUT_DIR}/gem5_host_functional_backend_test"

"${CXX}" -O2 -Wall -Wextra -Werror -std=c++17 \
    -I"${ROOT_DIR}/sim/coralnpu" \
    "${SOURCE_DIR}/gem5_gptq_kernels.cc" \
    "${SOURCE_DIR}/gem5_generic_gptq_executor.cc" \
    "${SOURCE_DIR}/gem5_transformer_kernels.cc" \
    "${SOURCE_DIR}/gem5_host_functional_backend.cc" \
    "${SOURCE_DIR}/gem5_generic_command_dispatch.cc" \
    "${SOURCE_DIR}/gem5_generic_command_dispatch_test.cc" \
    -o "${OUT_DIR}/gem5_generic_command_dispatch_test"
exec "${OUT_DIR}/gem5_generic_command_dispatch_test"
