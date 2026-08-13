#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
WORK_DIR="${ROOT_DIR}/build/local-tests/model-package"
MODEL_DIR="${WORK_DIR}/hf-model"
MANIFEST="${MODEL_DIR}/model.npxm"
CC="${CC:-cc}"

rm -rf "${WORK_DIR}"
mkdir -p "${MODEL_DIR}"
printf '%s\n' '{"architectures":["Qwen3_5MoeForCausalLM"],"model_type":"qwen3_5_moe","name_or_path":"qwen-test","text_config":{"dtype":"bfloat16","num_hidden_layers":2,"vocab_size":32,"hidden_size":18,"intermediate_size":48,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":6,"max_position_embeddings":4096,"num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":24,"shared_expert_intermediate_size":32}}' > "${MODEL_DIR}/config.json"
python3 - "${MODEL_DIR}" <<'PY'
import json
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name, tensors in (
    ("model-00001-of-00002.safetensors", {"tensor.a": bytes(range(8))}),
    ("model-00002-of-00002.safetensors", {
        "tensor.b": bytes(range(10, 18)), "tensor.c": bytes(range(20, 28))
    }),
):
    offset = 0
    header = {}
    payload = b""
    for tensor_name, data in tensors.items():
        header[tensor_name] = {
            "dtype": "U8", "shape": [len(data)],
            "data_offsets": [offset, offset + len(data)]
        }
        payload += data
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    (root / name).write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
PY
printf '%s\n' '{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors","c":"model-00002-of-00002.safetensors"}}' > "${MODEL_DIR}/model.safetensors.index.json"

"${SCRIPT_DIR}/import_hf_model.py" "${MODEL_DIR}" "${MANIFEST}"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/model_package.c" \
    "${ROOT_DIR}/runtime/host/tools/model_inspect.c" \
    -o "${WORK_DIR}/model-inspect"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/model_package.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/model_package_test.c" \
    -o "${WORK_DIR}/model_package_test"
"${WORK_DIR}/model-inspect" "${MANIFEST}"
"${WORK_DIR}/model_package_test" "${MANIFEST}"
echo "model_package_loader=PASS"
