#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
OUT_DIR="${ROOT_DIR}/build/local-tests"
CXX="${CXX:-c++}"

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <gptq-expert.bin>" >&2
    exit 2
fi
mkdir -p "$OUT_DIR"
"$CXX" -O2 -std=c++17 -Wall -Wextra -Werror -ffp-contract=off \
    -I"${ROOT_DIR}/sim/coralnpu" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_gptq_kernels.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_hybrid_kernels.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_hybrid_operator.cc" \
    "${ROOT_DIR}/runtime/host/tools/gptq_expert_reference.cc" \
    -o "${OUT_DIR}/gptq-expert-reference"
exec "${OUT_DIR}/gptq-expert-reference" "$1"
