#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
MODEL="${1:-${ROOT_DIR}/build/models/qwen-tiny.npxm}"

"${SCRIPT_DIR}/create_qwen_tiny_model.py" "${MODEL}" --print-summary
"${SCRIPT_DIR}/inspect_qwen_model.py" "${MODEL}"
echo "qwen_golden=PASS"
