#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CORAL_REPO="${CORAL_REPO:-${ROOT_DIR}/thirdparty/coralnpu}"

for path in hw_sim/hw_primitives.h hw_sim/core_mini_axi_wrapper.h; do
    if ! git -C "${CORAL_REPO}" diff --quiet HEAD -- "${path}"; then
        echo "error: gem5 integration overrides upstream Coral ${path}" >&2
        exit 1
    fi
done

grep -q 'Gem5CoreMiniAxiWrapper' \
    "${CORAL_REPO}/hw_sim/gem5_bridge/coralnpu_gem5_abi.cc"
grep -q 'Gem5AxiMasterReadDriver' \
    "${CORAL_REPO}/hw_sim/gem5_bridge/gem5_axi_master_drivers.h"
grep -q 'Gem5AxiMasterWriteDriver' \
    "${CORAL_REPO}/hw_sim/gem5_bridge/gem5_axi_master_drivers.h"

echo "Coral gem5 adapter is isolated from upstream AXI drivers"
