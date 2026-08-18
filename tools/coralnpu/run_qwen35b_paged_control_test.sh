#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
MODEL_DIR="${CORAL_MODEL_DIR:-/data/models/Qwen3.5-35B}"
EXECUTABLE="${CORAL_NPU_EXECUTABLE:-${MODEL_DIR}/model.npxc}"
TERMINAL="${ROOT_DIR}/logs/sim/m5out/system.terminal"
NORMALIZED_TERMINAL="$(mktemp)"
trap 'rm -f "$NORMALIZED_TERMINAL"' EXIT

[ -r "$EXECUTABLE" ] || {
    echo "error: Qwen executable missing: $EXECUTABLE" >&2
    echo "prepare it with: ./tools/models/prepare_hf_model_package.sh $MODEL_DIR" >&2
    exit 1
}

CORAL_NPU_EXECUTABLE="$EXECUTABLE" \
CORAL_NPU_WEIGHT_PAGE="${CORAL_NPU_WEIGHT_PAGE:-$EXECUTABLE}" \
CORAL_CKPT_ROOT="${CORAL_CKPT_ROOT:-${ROOT_DIR}/checkpoint/coralnpu_paged_8m_8f000000_ckpt}" \
    "${ROOT_DIR}/tools/coralnpu/run_paged_executable_test.sh"

[ -r "$TERMINAL" ] || {
    echo "error: guest terminal log missing: $TERMINAL" >&2
    exit 1
}
tr -d '\r' <"$TERMINAL" >"$NORMALIZED_TERMINAL"

require_line()
{
    pattern="$1"
    description="$2"
    if ! grep -q "^${pattern}$" "$NORMALIZED_TERMINAL"; then
        echo "error: missing ${description}: ${pattern}" >&2
        exit 1
    fi
}

require_line 'submitted_commands=524' 'full command submission'
require_line 'completed_commands=524' 'full command completion'
require_line 'paging_faults_serviced=343' 'weight-bearing command faults'
require_line 'paging_queue_producer=343' 'queue producer count'
require_line 'paging_queue_service=343' 'queue service count'
require_line 'paging_queue_retire=343' 'queue retire count'
require_line 'paging_queue_backpressure=0' 'zero queue backpressure'
require_line 'executable_run=PASS' 'device executable verdict'
require_line '\[coral-paged-executable-test\] PASS' 'guest test verdict'

echo 'qwen35b_paged_commands=524'
echo 'qwen35b_paged_faults=343'
echo 'qwen35b_paged_control=PASS'
