#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
WORK_DIR="${ROOT_DIR}/build/local-tests/model-package"
MODEL_DIR="${WORK_DIR}/hf-model"
MANIFEST="${MODEL_DIR}/model.npxm"
CC="${CC:-cc}"
CXX="${CXX:-c++}"

rm -rf "${WORK_DIR}"
mkdir -p "${MODEL_DIR}"
printf '%s\n' '{"architectures":["Qwen3_5MoeForConditionalGeneration"],"model_type":"qwen3_5_moe","name_or_path":"qwen-test","quantization_config":{"quant_method":"gptq","bits":4,"group_size":128,"desc_act":false,"sym":true},"text_config":{"dtype":"bfloat16","num_hidden_layers":2,"vocab_size":32,"hidden_size":18,"intermediate_size":48,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":6,"max_position_embeddings":4096,"num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":24,"shared_expert_intermediate_size":32}}' > "${MODEL_DIR}/config.json"
python3 - "${MODEL_DIR}" <<'PY'
import json
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1])
weight_map = {}
for name, tensors in (
    ("model-00001-of-00002.safetensors", {
        "model.language_model.embed_tokens.weight": bytes(range(8)),
        "model.language_model.layers.0.self_attn.q_proj.qweight": bytes(
            index % 256 for index in range(288)),
        "model.language_model.layers.0.self_attn.q_proj.qzeros": bytes(range(12)),
        "model.language_model.layers.0.self_attn.q_proj.scales": bytes(range(48)),
        # 18 input columns at group size 128 all belong to group 0, so the
        # numerical reference can consume this g_idx instead of rejecting it.
        "model.language_model.layers.0.self_attn.q_proj.g_idx": bytes(72),
        "model.language_model.layers.0.self_attn.k_proj.qweight": bytes(range(8)),
        "model.language_model.layers.0.self_attn.q_norm.weight": bytes(range(8)),
    }),
    ("model-00002-of-00002.safetensors", {
        "model.language_model.layers.0.mlp.experts.0.gate_proj.qweight": bytes(288),
        "model.language_model.layers.0.mlp.experts.0.gate_proj.qzeros": bytes(12),
        "model.language_model.layers.0.mlp.experts.0.gate_proj.scales": bytes(48),
        "model.language_model.layers.0.mlp.experts.0.gate_proj.g_idx": bytes(72),
        "model.language_model.layers.0.mlp.experts.0.up_proj.qweight": bytes(288),
        "model.language_model.layers.0.mlp.experts.0.up_proj.qzeros": bytes(12),
        "model.language_model.layers.0.mlp.experts.0.up_proj.scales": bytes(48),
        "model.language_model.layers.0.mlp.experts.0.up_proj.g_idx": bytes(72),
        "model.language_model.layers.0.mlp.experts.0.down_proj.qweight": bytes(216),
        "model.language_model.layers.0.mlp.experts.0.down_proj.qzeros": bytes(12),
        "model.language_model.layers.0.mlp.experts.0.down_proj.scales": bytes(36),
        "model.language_model.layers.0.mlp.experts.0.down_proj.g_idx": bytes(96),
        "model.language_model.layers.1.mlp.experts.7.down_proj.qweight": bytes(range(10, 18)),
        "model.language_model.layers.1.mlp.router.weight": bytes(range(20, 28)),
        "model.language_model.layers.1.linear_attn.in_proj_qkv.qweight": bytes(range(30, 38)),
        "model.language_model.norm.weight": bytes(range(40, 48)),
    }),
):
    offset = 0
    header = {}
    payload = b""
    for tensor_name, data in tensors.items():
        weight_map[tensor_name] = name
        dtype = "U8"
        if tensor_name.endswith(".scales"):
            dtype = "F16"
        elif tensor_name.endswith(".g_idx") or tensor_name.endswith(".qzeros"):
            dtype = "I32"
        header[tensor_name] = {
            "dtype": dtype, "shape": [len(data)],
            "data_offsets": [offset, offset + len(data)]
        }
        payload += data
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    (root / name).write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
(root / "model.safetensors.index.json").write_text(
    json.dumps({"weight_map": weight_map}, separators=(",", ":"))
)
PY

"${SCRIPT_DIR}/import_hf_model.py" "${MODEL_DIR}" "${MANIFEST}"
"${SCRIPT_DIR}/build_qwen_execution_plan.py" "${MANIFEST}"
"${SCRIPT_DIR}/compile_npu_executable.py" \
    "${MANIFEST}" "${MODEL_DIR}/execution-plan.npxp" \
    "${MODEL_DIR}/model.npxe"
"${SCRIPT_DIR}/inspect_npu_tensor_plan.py" "${MODEL_DIR}/model.npxt"
"${SCRIPT_DIR}/compile_npu_weight_plan.py" \
    "${MANIFEST}" "${MODEL_DIR}/model.npxe" "${MODEL_DIR}/model.npxw"
python3 - "${MODEL_DIR}/model.npxw" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    plan = json.load(source)
assert plan["format"] == "OPENNPUX_NPU_WEIGHT_PLAN_V1"
assert plan["command_count"] == 30
assert plan["mapped_command_count"] > 0
assert plan["matched_tensor_range_count"] > 0
for command in plan["commands"]:
    primary = command["primary_range"]
    if primary is not None:
        assert primary["size"] > 0
        assert primary["offset"] >= 0
print("npu_weight_plan_loader=PASS")
PY
python3 - "${MODEL_DIR}/model.npxw" "${MODEL_DIR}/model.npxr" <<'PY'
import json
import struct
import sys

header_struct = struct.Struct("<8I4Q")
record_struct = struct.Struct("<4I6Q")
with open(sys.argv[1], encoding="utf-8") as source:
    plan = json.load(source)
data = open(sys.argv[2], "rb").read()
header = header_struct.unpack_from(data)
assert header[0] == 0x5258504E and header[1] == 1
assert header[2] == header_struct.size and header[3] == record_struct.size
assert header[4] == plan["command_count"]
assert header[5] == plan["matched_tensor_range_count"]
assert header[9] == len(data)
mutable = bytearray(data)
mutable[28:32] = bytes(4)
checksum = 2166136261
for byte in mutable:
    checksum = ((checksum ^ byte) * 16777619) & 0xffffffff
assert checksum == header[7]
for command in plan["commands"]:
    assert command["range_count"] == command["matched_tensor_count"]
    assert command["range_start"] + command["range_count"] <= header[5]
records = [
    record_struct.unpack_from(data, header[8] + index * header[3])
    for index in range(header[5])
]
assert any(record[8] == 7 for record in records)
assert any(
    record[2] == 0xA3F281BA and record[3] == 0x40406979 and
    record[9] & 0xffff == 1 and record[9] >> 16 & 0xff == 1
    for record in records
)
assert any(record[8] == 7 and record[9] & 0xffff == 7 for record in records)
print("npu_weight_range_index=PASS")
PY
"${SCRIPT_DIR}/inspect_gptq_bindings.py" "${MODEL_DIR}/model.npxr" \
    | tee "${WORK_DIR}/gptq-bindings.log"
grep -q '^gptq_binding_complete=4$' "${WORK_DIR}/gptq-bindings.log"
grep -q '^gptq_binding_incomplete=3$' "${WORK_DIR}/gptq-bindings.log"
grep -q '^gptq_binding_duplicate=0$' "${WORK_DIR}/gptq-bindings.log"
grep -q '^gptq_binding_scales_dtypes=float16:4$' \
    "${WORK_DIR}/gptq-bindings.log"
"${SCRIPT_DIR}/materialize_gptq_projection.py" \
    "${MANIFEST}" "${MODEL_DIR}/model.npxw" "${MODEL_DIR}/model.npxr" \
    "${MODEL_DIR}/gptq-projection.bin" --layer 0 --phase qkv_projection \
    --role attention_q_proj --expert -1 --slot q_proj \
    | tee "${WORK_DIR}/gptq-projection.log"
grep -q '^gptq_projection_shape=1x18x24$' \
    "${WORK_DIR}/gptq-projection.log"
grep -q '^gptq_projection_scale_dtype=4$' \
    "${WORK_DIR}/gptq-projection.log"
grep -q '^gptq_projection_materialize=PASS$' \
    "${WORK_DIR}/gptq-projection.log"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 -ffp-contract=off \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/npu_gptq_reference.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_gptq_reference_test.c" \
    -o "${WORK_DIR}/npu_gptq_reference_test" -lm
"${WORK_DIR}/npu_gptq_reference_test" "${MODEL_DIR}/gptq-projection.bin"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 -ffp-contract=off \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/model_package.c" \
    "${ROOT_DIR}/runtime/host/src/npu_weight_ranges.c" \
    "${ROOT_DIR}/runtime/host/src/npu_gptq_weights.c" \
    "${ROOT_DIR}/runtime/host/src/npu_gptq_reference.c" \
    "${ROOT_DIR}/runtime/host/src/npu_gptq_request.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_gptq_request_test.c" \
    -o "${WORK_DIR}/npu_gptq_request_test" -lm
"${WORK_DIR}/npu_gptq_request_test" \
    "${MANIFEST}" "${MODEL_DIR}/model.npxr" \
    "${MODEL_DIR}/gptq-projection.bin"
"${SCRIPT_DIR}/materialize_gptq_expert.py" \
    "${MANIFEST}" "${MODEL_DIR}/model.npxw" "${MODEL_DIR}/model.npxr" \
    "${MODEL_DIR}/gptq-expert.bin" --layer 0 --expert 0 \
    | tee "${WORK_DIR}/gptq-expert.log"
grep -q '^gptq_expert_shape=1x18x24$' "${WORK_DIR}/gptq-expert.log"
grep -q '^gptq_expert_materialize=PASS$' "${WORK_DIR}/gptq-expert.log"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 -ffp-contract=off \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/model_package.c" \
    "${ROOT_DIR}/runtime/host/src/npu_weight_ranges.c" \
    "${ROOT_DIR}/runtime/host/src/npu_gptq_weights.c" \
    "${ROOT_DIR}/runtime/host/src/npu_gptq_request.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_gptq_expert_request_test.c" \
    -o "${WORK_DIR}/npu_gptq_expert_request_test"
"${WORK_DIR}/npu_gptq_expert_request_test" \
    "${MANIFEST}" "${MODEL_DIR}/model.npxr" \
    "${MODEL_DIR}/gptq-expert.bin"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/npu_router.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_router_test.c" \
    -o "${WORK_DIR}/npu_router_test" -lm
"${WORK_DIR}/npu_router_test"
echo "PASS: NPU stable top-K router tests"
"${SCRIPT_DIR}/gptq_reference.sh" "${MODEL_DIR}/gptq-projection.bin" \
    | tee "${WORK_DIR}/gptq-reference.log"
grep -q '^gptq_reference_shape=1x18x24$' "${WORK_DIR}/gptq-reference.log"
grep -q '^gptq_reference_scale_dtype=4$' "${WORK_DIR}/gptq-reference.log"
grep -q '^gptq_reference=PASS$' "${WORK_DIR}/gptq-reference.log"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/npu_weight_ranges.c" \
    "${ROOT_DIR}/runtime/host/src/model_package.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_weight_ranges_test.c" \
    -o "${WORK_DIR}/npu_weight_ranges_test"
"${WORK_DIR}/npu_weight_ranges_test" \
    "${MANIFEST}" "${MODEL_DIR}/model.npxr"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/model_package.c" \
    "${ROOT_DIR}/runtime/host/src/npu_weight_ranges.c" \
    "${ROOT_DIR}/runtime/host/src/npu_gptq_weights.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_gptq_weights_test.c" \
    -o "${WORK_DIR}/npu_gptq_weights_test"
"${WORK_DIR}/npu_gptq_weights_test" \
    "${MANIFEST}" "${MODEL_DIR}/model.npxr"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/model_package.c" \
    "${ROOT_DIR}/runtime/host/src/npu_weight_ranges.c" \
    "${ROOT_DIR}/runtime/host/src/npu_weight_pager.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_weight_pager_test.c" \
    -o "${WORK_DIR}/npu_weight_pager_test"
"${WORK_DIR}/npu_weight_pager_test" \
    "${MANIFEST}" "${MODEL_DIR}/model.npxr"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/model_package.c" \
    "${ROOT_DIR}/runtime/host/src/npu_weight_ranges.c" \
    "${ROOT_DIR}/runtime/host/src/npu_weight_pager.c" \
    "${ROOT_DIR}/runtime/host/src/npu_weight_queue.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_weight_queue_test.c" \
    -o "${WORK_DIR}/npu_weight_queue_test"
"${WORK_DIR}/npu_weight_queue_test" \
    "${MANIFEST}" "${MODEL_DIR}/model.npxr"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/model_package.c" \
    "${ROOT_DIR}/runtime/host/src/npu_weight_ranges.c" \
    "${ROOT_DIR}/runtime/host/src/npu_weight_pager.c" \
    "${ROOT_DIR}/runtime/host/src/npu_weight_queue.c" \
    "${ROOT_DIR}/runtime/host/src/npu_paging_layout.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_paging_layout_test.c" \
    -o "${WORK_DIR}/npu_paging_layout_test"
"${WORK_DIR}/npu_paging_layout_test"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/coral_runtime.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/coral_async_runtime_test.c" \
    -o "${WORK_DIR}/coral_async_runtime_test"
"${WORK_DIR}/coral_async_runtime_test"
"${SCRIPT_DIR}/inspect_npu_weight_pages.sh" \
    "${MANIFEST}" "${MODEL_DIR}/model.npxr"
"${SCRIPT_DIR}/materialize_npu_weight_page.py" \
    "${MODEL_DIR}/model.npxw" "${MODEL_DIR}/weight-page.bin"
python3 - "${MODEL_DIR}/model.npxw" "${MODEL_DIR}/weight-page.bin" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    plan = json.load(source)
command = next(item for item in plan["commands"] if item["primary_range"])
tensor = command["primary_range"]
with open(f"{sys.argv[1].rsplit('/', 1)[0]}/{tensor['shard']}", "rb") as source:
    source.seek(tensor["offset"])
    expected = source.read(min(4096, tensor["size"]))
with open(sys.argv[2], "rb") as source:
    actual = source.read()
assert len(actual) == 4096
assert actual[:len(expected)] == expected
assert not any(actual[len(expected):])
print("npu_weight_page_materialize=PASS")
PY
"${SCRIPT_DIR}/materialize_npu_weight_samples.py" \
    "${MODEL_DIR}/model.npxw" "${MODEL_DIR}/weight-samples.bin" \
    --allow-unresolved
python3 - "${MODEL_DIR}/model.npxw" "${MODEL_DIR}/weight-samples.bin" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    plan = json.load(source)
with open(sys.argv[2], "rb") as source:
    samples = source.read()
assert len(samples) == 4096
for command in plan["commands"]:
    tensor = command["primary_range"]
    begin = command["command_id"] * 4
    actual = samples[begin:begin + 4]
    if tensor is None:
        assert actual == bytes(4)
        continue
    shard = f"{sys.argv[1].rsplit('/', 1)[0]}/{tensor['shard']}"
    with open(shard, "rb") as source:
        source.seek(tensor["offset"])
        expected = source.read(min(4, tensor["size"])).ljust(4, bytes(1))
    assert actual == expected
print("npu_weight_samples_materialize=PASS")
PY
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/npu_submission.c" \
    "${ROOT_DIR}/runtime/host/src/npu_executable.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_executable_test.c" \
    -o "${WORK_DIR}/npu_executable_test"
"${WORK_DIR}/npu_executable_test" "${MODEL_DIR}/model.npxc"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/npu_tensor_plan.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_tensor_plan_test.c" \
    -o "${WORK_DIR}/npu_tensor_plan_test"
"${WORK_DIR}/npu_tensor_plan_test" "${MODEL_DIR}/model.npxtb"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    -c "${ROOT_DIR}/runtime/host/src/npu_tensor_plan.c" \
    -o "${WORK_DIR}/npu_tensor_plan.o"
"${CXX}" -O2 -Wall -Wextra -Werror -std=c++17 \
    -I"${ROOT_DIR}/sim/coralnpu" \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_host_tensor_arena.cc" \
    "${ROOT_DIR}/sim/coralnpu/hw_sim/gem5_bridge/gem5_host_tensor_arena_test.cc" \
    "${WORK_DIR}/npu_tensor_plan.o" \
    -o "${WORK_DIR}/gem5_host_tensor_arena_test"
"${WORK_DIR}/gem5_host_tensor_arena_test" "${MODEL_DIR}/model.npxtb"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/npu_submission.c" \
    "${ROOT_DIR}/runtime/host/src/npu_tensor_plan.c" \
    "${ROOT_DIR}/runtime/host/src/npu_functional_materializer.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_functional_materializer_test.c" \
    -o "${WORK_DIR}/npu_functional_materializer_test"
"${WORK_DIR}/npu_functional_materializer_test"
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/npu_submission.c" \
    "${ROOT_DIR}/runtime/host/src/npu_executable.c" \
    "${ROOT_DIR}/runtime/host/src/npu_tensor_plan.c" \
    "${ROOT_DIR}/runtime/host/src/npu_functional_materializer.c" \
    "${ROOT_DIR}/tests/unit/runtime_host/npu_functional_program_test.c" \
    -o "${WORK_DIR}/npu_functional_program_test"
"${WORK_DIR}/npu_functional_program_test" \
    "${MODEL_DIR}/model.npxc" "${MODEL_DIR}/model.npxtb"
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
