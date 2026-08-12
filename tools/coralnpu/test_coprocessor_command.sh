#!/bin/sh
#
# Purpose: validate the OpenNPUX coprocessor command front-end without building
# the full Coral RTL bridge.
# Pipeline: sync sim/coralnpu overlay -> compile the host C++ command adapter
# unit test -> check the Coral scalar custom-0 decode hook exists.
# Environment:
#   CORAL_REPO: CoralNPU source tree to validate; defaults to thirdparty/coralnpu.
#   CXX: host C++ compiler; defaults to c++.
# Output:
#   A temporary host executable under TMPDIR and a PASS line on success.

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CORAL_REPO="${CORAL_REPO:-${ROOT_DIR}/thirdparty/coralnpu}"
CXX="${CXX:-c++}"
OUTPUT="${TMPDIR:-/tmp}/opennpux-coprocessor-command-test"

"${ROOT_DIR}/sim/coralnpu/apply_patchset.sh"

"${CXX}" -std=c++17 -Wall -Wextra -Werror \
    -I"${CORAL_REPO}" \
    "${CORAL_REPO}/hw_sim/gem5_bridge/gem5_coprocessor_command.cc" \
    "${CORAL_REPO}/hw_sim/gem5_bridge/gem5_coprocessor_command_test.cc" \
    -o "${OUTPUT}"
"${OUTPUT}"

grep -q 'val npuLaunch = Bool()' \
    "${CORAL_REPO}/hdl/chisel/src/coralnpu/scalar/Decode.scala"
grep -q '00000_0001011' \
    "${CORAL_REPO}/hdl/chisel/src/coralnpu/scalar/Decode.scala"

echo "coprocessor command decode/queue/scoreboard test: PASS"
