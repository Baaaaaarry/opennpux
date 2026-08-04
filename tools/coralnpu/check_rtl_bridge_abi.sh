#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CANONICAL="${ROOT_DIR}/rtl/wrappers/coralnpu_gem5_abi.h"

cmp "${CANONICAL}" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/coralnpu_gem5_abi.h"
cmp "${CANONICAL}" \
    "${ROOT_DIR}/sim/gem5/src/dev/npu/coralnpu_gem5_abi.h"

echo "Coral gem5 ABI headers match"
