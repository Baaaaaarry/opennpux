#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
OUT_DIR="${ROOT_DIR}/build/local-tests/sim-host-pager"
CXX="${CXX:-c++}"
mkdir -p "$OUT_DIR"
"$CXX" -O2 -Wall -Wextra -Werror -std=c++17 \
    -I"${ROOT_DIR}/sim/coralnpu" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_sim_host_pager.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_sim_host_pager_test.cc" \
    -o "${OUT_DIR}/gem5_sim_host_pager_test"
"${OUT_DIR}/gem5_sim_host_pager_test"
"$CXX" -O2 -Wall -Wextra -Werror -std=c++17 \
    -I"${ROOT_DIR}/sim/coralnpu" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_sim_host_numerical.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_sim_host_numerical_test.cc" \
    -o "${OUT_DIR}/gem5_sim_host_numerical_test"
exec "${OUT_DIR}/gem5_sim_host_numerical_test"
