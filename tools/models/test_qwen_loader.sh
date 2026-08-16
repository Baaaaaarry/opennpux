#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
MODEL="${1:-${ROOT_DIR}/build/models/qwen-tiny.npxm}"
OUT_DIR="${ROOT_DIR}/build/local-tests"
CC="${CC:-cc}"

"${SCRIPT_DIR}/run_qwen_golden.sh" "${MODEL}"
mkdir -p "${OUT_DIR}"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/qwen_model.c" \
    "${ROOT_DIR}/runtime/host/tools/qwen_inspect.c" \
    -lm \
    -o "${OUT_DIR}/qwen-inspect"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/qwen_model.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/qwen_model_test.c" \
    -lm \
    -o "${OUT_DIR}/qwen_model_test"
"${OUT_DIR}/qwen-inspect" "${MODEL}"
"${OUT_DIR}/qwen_model_test" "${MODEL}"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    -c "${ROOT_DIR}/runtime/host/src/qwen_model.c" \
    -o "${OUT_DIR}/qwen_model.o"
"${CXX:-c++}" -O2 -Wall -Wextra -Werror -std=c++17 \
    -I"${ROOT_DIR}/runtime/host/include" \
    -I"${ROOT_DIR}/sim/coralnpu" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_gptq_kernels.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_hybrid_kernels.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_hybrid_operator.cc" \
    "${ROOT_DIR}/tests/unit/runtime_host/qwen_device_inference_test.cc" \
    "${OUT_DIR}/qwen_model.o" -lm \
    -o "${OUT_DIR}/qwen_device_inference_test"
"${OUT_DIR}/qwen_device_inference_test" "${MODEL}"
echo "qwen_device_inference_kernel=PASS"
CORRUPT_MODEL="${OUT_DIR}/qwen-tiny-corrupt.npxm"
python3 - "${MODEL}" "${CORRUPT_MODEL}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    package = json.load(source)
# Token 1 is consumed by the fixed prompt, so this must alter the logits.
package["numeric_reference"]["token_embedding"][8] += 1.0
with open(sys.argv[2], "w", encoding="utf-8") as output:
    json.dump(package, output)
PY
if "${OUT_DIR}/qwen_model_test" "${CORRUPT_MODEL}" >/dev/null 2>&1; then
    echo "error: numerical Qwen path accepted corrupted weights" >&2
    exit 1
fi
echo "qwen_numeric_corruption_rejected=PASS"
echo "qwen_loader=PASS"
