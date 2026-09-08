#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BUILD_DIR="${ROOT_DIR}/build/local-tests/tvm-byoc-xgraph"
GRAPH="${BUILD_DIR}/relax-model.npxg"
ARENA="${BUILD_DIR}/relax-model.arena.bin"
TRANSFORMER_GRAPH="${BUILD_DIR}/transformer-block.npxg"
TRANSFORMER_ARENA="${BUILD_DIR}/transformer-block.arena.bin"
TRANSFORMER_MODULE_DIR="${BUILD_DIR}/transformer-block-module"
TRANSFORMER_DEPLOYMENT_DIR="${BUILD_DIR}/transformer-deployment"
TRANSFORMER_MODULE_PACKAGE="${TRANSFORMER_DEPLOYMENT_DIR}/model.npxgm"
TRANSFORMER_MODULE_INVOCATION="${TRANSFORMER_DEPLOYMENT_DIR}/decode-000.npxmi"
TRANSFORMER_EXPECTED="${BUILD_DIR}/transformer-block.expected.bin"
ONNX_DEPLOYMENT_DIR="${BUILD_DIR}/projection-residual-deployment"
ONNX_MODULE_PACKAGE="${ONNX_DEPLOYMENT_DIR}/deployment/model.npxgm"
ONNX_MODULE_INVOCATION="${ONNX_DEPLOYMENT_DIR}/deployment/request-000.npxmi"
ONNX_EXPECTED="${BUILD_DIR}/projection-residual.expected.bin"
ONNX_STATE_DEPLOYMENT_DIR="${BUILD_DIR}/stateful-decode-deployment/deployment"
ONNX_STATE_MODULE_PACKAGE="${ONNX_STATE_DEPLOYMENT_DIR}/model.npxgm"
ONNX_STATE_INVOCATION0="${ONNX_STATE_DEPLOYMENT_DIR}/decode-000.npxmi"
ONNX_STATE_INVOCATION1="${ONNX_STATE_DEPLOYMENT_DIR}/decode-001.npxmi"
ONNX_STATE_EXPECTED="${BUILD_DIR}/stateful-decode.expected.bin"
STATE_MODULE_DIR="${BUILD_DIR}/stateful-transformer-module"
STATE_MODULE_ARENA="${BUILD_DIR}/stateful-transformer.arena.bin"
STATE_MODULE_PACKAGE="${BUILD_DIR}/stateful-transformer.npxgm"
STATE_MODULE_INVOCATION="${BUILD_DIR}/stateful-transformer.npxmi"
STATE_EXPECTED="${BUILD_DIR}/stateful-transformer.expected.bin"
APPEND_MODULE_DIR="${BUILD_DIR}/state-append-module"
APPEND_MODULE_PACKAGE="${BUILD_DIR}/state-append.npxgm"
APPEND_MODULE_INVOCATION="${BUILD_DIR}/state-append.npxmi"
APPEND_EXPECTED="${BUILD_DIR}/state-append.expected.bin"
ATTENTION_MODULE_DIR="${BUILD_DIR}/dynamic-attention-module"
ATTENTION_MODULE_PACKAGE="${BUILD_DIR}/dynamic-attention.npxgm"
ATTENTION_INVOCATION1="${BUILD_DIR}/dynamic-attention-kv1.npxmi"
ATTENTION_INVOCATION2="${BUILD_DIR}/dynamic-attention-kv2.npxmi"
ATTENTION_EXPECTED1="${BUILD_DIR}/dynamic-attention-kv1.expected.bin"
ATTENTION_EXPECTED2="${BUILD_DIR}/dynamic-attention-kv2.expected.bin"
KV_ATTENTION_MODULE_DIR="${BUILD_DIR}/kv-attention-module"
KV_ATTENTION_PACKAGE="${BUILD_DIR}/kv-attention.npxgm"
KV_ATTENTION_INVOCATION1="${BUILD_DIR}/kv-attention-step1.npxmi"
KV_ATTENTION_INVOCATION2="${BUILD_DIR}/kv-attention-step2.npxmi"
KV_ATTENTION_EXPECTED="${BUILD_DIR}/kv-attention.expected.bin"
KV_ATTENTION_STATE_EXPECTED="${BUILD_DIR}/kv-attention-state.expected.bin"
MODULE_DIR="${BUILD_DIR}/multi-region-module"
MODULE_PACKAGE="${BUILD_DIR}/tvm-mixed-module.npxgm"
MODULE_INVOCATION="${BUILD_DIR}/tvm-mixed-module.npxmi"
MODULE_INVOCATION2="${BUILD_DIR}/tvm-mixed-module-second.npxmi"
MODULE_MISMATCH="${BUILD_DIR}/tvm-mixed-module-mismatch.npxmi"
LOCAL_LOG="${ROOT_DIR}/simout/tvm-byoc-xgraph-local.log"
HOST_LOG="${ROOT_DIR}/simout/tvm-byoc-xgraph-host.log"
DEBUG_LOG="${ROOT_DIR}/simout/tvm-byoc-xgraph.debug"
FIRMWARE="${ROOT_DIR}/build/coralnpu/gem5_qwen_command_flow_smoke.elf"

mkdir -p "${ROOT_DIR}/simout"
if ! OPENNPUX_REQUIRE_TVM=1 \
    "${ROOT_DIR}/tools/models/test_tvm_byoc_xgraph_codegen.sh" \
    >"${LOCAL_LOG}" 2>&1; then
    cat "${LOCAL_LOG}"
    echo "error: TVM BYOC local generation failed" >&2
    exit 1
fi
cat "${LOCAL_LOG}"
[ -f "${GRAPH}" ] && [ -f "${ARENA}" ] &&
    [ -f "${TRANSFORMER_GRAPH}" ] && [ -f "${TRANSFORMER_ARENA}" ] || {
    echo "error: TVM BYOC XGraph artifacts were not generated" >&2
    exit 1
}
MODULE_ARENA_BINDINGS="$("${TVM_PYTHON:-python3}" - \
    "${MODULE_DIR}" <<'PY'
import json
import math
import struct
import sys
from pathlib import Path

directory = Path(sys.argv[1])
manifest = json.load(open(directory / "module.npxgm.json", encoding="utf-8"))
assert len(manifest["regions"]) == 2
assert len(manifest["host_bindings"]) == 1
region0, region1 = manifest["regions"]
metadata0 = json.load(open(directory / f"{region0['artifact']}.json", encoding="utf-8"))
metadata1 = json.load(open(directory / f"{region1['artifact']}.json", encoding="utf-8"))
tensors0 = {tensor["name"]: tensor for tensor in metadata0["tensors"]}
tensors1 = {tensor["name"]: tensor for tensor in metadata1["tensors"]}
inputs0 = [tensor for tensor in metadata0["tensors"] if tensor["storage"] == "input"]
assert len(inputs0) == 2
lhs = [float(value) for value in range(-4, 4)]
rhs = [1.0] * 8
summed = [a + b for a, b in zip(lhs, rhs)]
host_relu = [max(value, 0.0) for value in summed]
activated = [value / (1.0 + math.exp(-value)) for value in host_relu]
(directory / "invocation.expected.bin").write_bytes(
    struct.pack("<8f", *activated)
)
arena0 = bytearray(metadata0["arena_size"])
arena1 = bytearray(metadata1["arena_size"])
for arena, table, name, values in (
    (arena0, tensors0, inputs0[0]["name"], lhs),
    (arena0, tensors0, inputs0[1]["name"], rhs),
    (arena0, tensors0, metadata0["output"], summed),
    (arena1, tensors1, metadata1["output"], activated),
):
    struct.pack_into(f"<{len(values)}f", arena, table[name]["offset"], *values)
arena0_path = directory / "region-000.invocation.bin"
arena1_path = directory / "region-001.invocation.bin"
arena0_path.write_bytes(arena0)
arena1_path.write_bytes(arena1)
print(f"{region0['name']}={arena0_path}")
print(f"{region1['name']}={arena1_path}")

second_lhs = [float(value) for value in range(4, 12)]
second_rhs = [2.0] * 8
second_summed = [a + b for a, b in zip(second_lhs, second_rhs)]
second_relu = [max(value, 0.0) for value in second_summed]
second_activated = [
    value / (1.0 + math.exp(-value)) for value in second_relu
]
(directory / "invocation2.expected.bin").write_bytes(
    struct.pack("<8f", *second_activated)
)
second_arena0 = bytearray(metadata0["arena_size"])
second_arena1 = bytearray(metadata1["arena_size"])
struct.pack_into("<8f", second_arena0, tensors0[inputs0[0]["name"]]["offset"],
                 *second_lhs)
struct.pack_into("<8f", second_arena0, tensors0[inputs0[1]["name"]]["offset"],
                 *second_rhs)
second_arena0_path = directory / "region-000.invocation2.bin"
second_arena1_path = directory / "region-001.invocation2.bin"
second_arena0_path.write_bytes(second_arena0)
second_arena1_path.write_bytes(second_arena1)
(directory / "invocation2.bindings").write_text(
    f"{region0['name']}={second_arena0_path}\n"
    f"{region1['name']}={second_arena1_path}\n",
    encoding="utf-8",
)
PY
)"
set --
while IFS= read -r binding; do
    [ -n "${binding}" ] && set -- "$@" --arena "${binding}"
done <<EOF
${MODULE_ARENA_BINDINGS}
EOF
"${TVM_PYTHON:-python3}" \
    "${ROOT_DIR}/tools/models/build_tvm_byoc_module_package.py" \
    "${MODULE_DIR}" "${MODULE_PACKAGE}" --clear-external-bindings "$@"
"${TVM_PYTHON:-python3}" \
    "${ROOT_DIR}/tools/models/build_tvm_byoc_invocation.py" \
    "${MODULE_DIR}" "${MODULE_INVOCATION}" "$@"
set --
while IFS= read -r binding; do
    [ -n "${binding}" ] && set -- "$@" --arena "${binding}"
done <"${MODULE_DIR}/invocation2.bindings"
"${TVM_PYTHON:-python3}" \
    "${ROOT_DIR}/tools/models/build_tvm_byoc_invocation.py" \
    "${MODULE_DIR}" "${MODULE_INVOCATION2}" "$@"
TRANSFORMER_REGION="$("${TVM_PYTHON:-python3}" - \
    "${TRANSFORMER_MODULE_DIR}/module.npxgm.json" \
    "${TRANSFORMER_MODULE_DIR}" "${TRANSFORMER_ARENA}" \
    "${TRANSFORMER_EXPECTED}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
module_dir = Path(sys.argv[2])
arena_path = Path(sys.argv[3])
expected_path = Path(sys.argv[4])
manifest = json.load(open(manifest_path, encoding="utf-8"))
assert manifest["region_count"] == 1
region = manifest["regions"][0]
assert len(region["invocation_bindings"]) == 2
assert len(region["constant_bindings"]) == 2
metadata = json.load(
    open(module_dir / f"{region['artifact']}.json", encoding="utf-8")
)
output = next(
    tensor for tensor in metadata["tensors"]
    if tensor["name"] == metadata["output"]
)
arena = arena_path.read_bytes()
begin = output["offset"]
expected_path.write_bytes(arena[begin:begin + output["byte_size"]])
print(region["name"])
PY
)"
STATE_REGION="$("${TVM_PYTHON:-python3}" - \
    "${STATE_MODULE_DIR}" \
    "${STATE_MODULE_ARENA}" "${STATE_EXPECTED}" <<'PY'
import json
import math
import struct
import sys
from pathlib import Path

directory = Path(sys.argv[1])
arena_path = Path(sys.argv[2])
expected_path = Path(sys.argv[3])
manifest = json.load(open(directory / "module.npxgm.json", encoding="utf-8"))
region = manifest["regions"][0]
metadata_path = directory / f"{region['artifact']}.json"
metadata = json.load(open(metadata_path, encoding="utf-8"))
tensors = {tensor["name"]: tensor for tensor in metadata["tensors"]}
arena = arena_path.read_bytes()

def read(name):
    tensor = tensors[name]
    count = tensor["byte_size"] // 4
    return list(struct.unpack_from(f"<{count}f", arena, tensor["offset"]))

def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]

hidden = read("hidden")
state = read("recurrent_state")
norm_weight = read("norm_weight")
weight = read("projection_weight")
residual = read("residual")
output = []
for _ in range(2):
    normalized = []
    for row in range(2):
        values = hidden[row * 64:(row + 1) * 64]
        mean_square = sum(value * value for value in values) / 64
        scale = 1.0 / math.sqrt(mean_square + 1.0e-5)
        normalized.extend(
            f32(value * scale * norm_weight[column])
            for column, value in enumerate(values)
        )
    projected = []
    for row in range(2):
        for column in range(64):
            accumulator = 0.0
            for inner in range(64):
                accumulator = f32(
                    accumulator
                    + f32(normalized[row * 64 + inner]
                          * weight[inner * 64 + column])
                )
            projected.append(accumulator)
    state = [f32(projected[index] + state[index]) for index in range(128)]
    residual_sum = [
        f32(state[index] + residual[index])
        for index in range(128)
    ]
    output = [
        f32(value / (1.0 + math.exp(-value))) for value in residual_sum
    ]
expected_path.write_bytes(struct.pack("<128f", *output))
print(f"{region['name']}={arena_path}")
PY
)"
"${TVM_PYTHON:-python3}" \
    "${ROOT_DIR}/tools/models/build_tvm_byoc_module_package.py" \
    "${STATE_MODULE_DIR}" "${STATE_MODULE_PACKAGE}" \
    --clear-external-bindings --arena "${STATE_REGION}"
"${TVM_PYTHON:-python3}" \
    "${ROOT_DIR}/tools/models/build_tvm_byoc_invocation.py" \
    "${STATE_MODULE_DIR}" "${STATE_MODULE_INVOCATION}" \
    --arena "${STATE_REGION}"
APPEND_REGION="$("${TVM_PYTHON:-python3}" - \
    "${APPEND_MODULE_DIR}" "${APPEND_EXPECTED}" <<'PY'
import json
import math
import struct
import sys
from pathlib import Path

directory = Path(sys.argv[1])
expected_path = Path(sys.argv[2])
manifest = json.load(open(directory / "module.npxgm.json", encoding="utf-8"))
region = manifest["regions"][0]
metadata = json.load(
    open(directory / f"{region['artifact']}.json", encoding="utf-8")
)
tensors = {tensor["name"]: tensor for tensor in metadata["tensors"]}
arena = bytearray(region["arena_size"])
struct.pack_into("<2f", arena, tensors["token"]["offset"], 2.0, 3.0)
arena_path = directory / "decode.arena.bin"
arena_path.write_bytes(arena)
update = [value / (1.0 + math.exp(-value)) for value in (2.0, 3.0)]
expected_path.write_bytes(struct.pack("<8f", *(update + update + [0.0] * 4)))
print(f"{region['name']}={arena_path}")
PY
)"
"${TVM_PYTHON:-python3}" \
    "${ROOT_DIR}/tools/models/build_tvm_byoc_module_package.py" \
    "${APPEND_MODULE_DIR}" "${APPEND_MODULE_PACKAGE}" \
    --clear-external-bindings --arena "${APPEND_REGION}"
"${TVM_PYTHON:-python3}" \
    "${ROOT_DIR}/tools/models/build_tvm_byoc_invocation.py" \
    "${APPEND_MODULE_DIR}" "${APPEND_MODULE_INVOCATION}" \
    --arena "${APPEND_REGION}" --scalar decode_position=0
ATTENTION_REGION="$("${TVM_PYTHON:-python3}" - \
    "${ATTENTION_MODULE_DIR}" "${ATTENTION_EXPECTED1}" \
    "${ATTENTION_EXPECTED2}" <<'PY'
import json
import math
import struct
import sys
from pathlib import Path

directory = Path(sys.argv[1])
manifest = json.load(open(directory / "module.npxgm.json", encoding="utf-8"))
region = manifest["regions"][0]
metadata = json.load(open(directory / f"{region['artifact']}.json", encoding="utf-8"))
tensors = {tensor["name"]: tensor for tensor in metadata["tensors"]}
query = [1.0, 0.0, 0.0, 1.0]
state = [1.0, 0.0, 0.0, 1.0, 10.0, 20.0, 30.0, 40.0]
arena = bytearray(region["arena_size"])
struct.pack_into("<4f", arena, tensors["query"]["offset"], *query)
struct.pack_into("<8f", arena, tensors["kv_cache"]["offset"], *state)
arena_path = directory / "decode-attention.arena.bin"
arena_path.write_bytes(arena)
Path(sys.argv[2]).write_bytes(struct.pack("<4f", 10.0, 20.0, 10.0, 20.0))
scale = 1.0 / math.sqrt(2.0)
p = math.exp(scale) / (math.exp(scale) + 1.0)
expected = [p * 10.0 + (1.0 - p) * 30.0,
            p * 20.0 + (1.0 - p) * 40.0,
            (1.0 - p) * 10.0 + p * 30.0,
            (1.0 - p) * 20.0 + p * 40.0]
Path(sys.argv[3]).write_bytes(struct.pack("<4f", *expected))
print(f"{region['name']}={arena_path}")
PY
)"
"${TVM_PYTHON:-python3}" \
    "${ROOT_DIR}/tools/models/build_tvm_byoc_module_package.py" \
    "${ATTENTION_MODULE_DIR}" "${ATTENTION_MODULE_PACKAGE}" \
    --clear-external-bindings --arena "${ATTENTION_REGION}"
"${TVM_PYTHON:-python3}" \
    "${ROOT_DIR}/tools/models/build_tvm_byoc_invocation.py" \
    "${ATTENTION_MODULE_DIR}" "${ATTENTION_INVOCATION1}" \
    --arena "${ATTENTION_REGION}" --scalar kv_length=1
"${TVM_PYTHON:-python3}" \
    "${ROOT_DIR}/tools/models/build_tvm_byoc_invocation.py" \
    "${ATTENTION_MODULE_DIR}" "${ATTENTION_INVOCATION2}" \
    --arena "${ATTENTION_REGION}" --scalar kv_length=2
"${TVM_PYTHON:-python3}" - \
    "${KV_ATTENTION_MODULE_DIR}" "${KV_ATTENTION_EXPECTED}" \
    "${KV_ATTENTION_STATE_EXPECTED}" <<'PY'
import json
import math
import struct
import sys
from pathlib import Path

directory = Path(sys.argv[1])
manifest = json.load(open(directory / "module.npxgm.json", encoding="utf-8"))
regions = {region["name"]: region for region in manifest["regions"]}
metadata = {
    name: json.load(open(directory / f"{region['artifact']}.json", encoding="utf-8"))
    for name, region in regions.items()
}
tensors = {
    name: {tensor["name"]: tensor for tensor in record["tensors"]}
    for name, record in metadata.items()
}
hidden_steps = ([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0])
for step, hidden in enumerate(hidden_steps, 1):
    writer = bytearray(regions["qkv_projection"]["arena_size"])
    attention = bytearray(regions["attention"]["arena_size"])
    struct.pack_into("<4f", writer,
                     tensors["qkv_projection"]["hidden"]["offset"], *hidden)
    (directory / f"qkv-step{step}.arena.bin").write_bytes(writer)
    (directory / f"attention-step{step}.arena.bin").write_bytes(attention)
writer = bytearray(regions["qkv_projection"]["arena_size"])
for weight in ("q_weight", "k_weight"):
    struct.pack_into("<4f", writer,
                     tensors["qkv_projection"][weight]["offset"],
                     1.0, 0.0, 0.0, 1.0)
struct.pack_into("<4f", writer,
                 tensors["qkv_projection"]["v_weight"]["offset"],
                 10.0, 0.0, 0.0, 20.0)
(directory / "qkv-base.arena.bin").write_bytes(writer)
(directory / "attention-base.arena.bin").write_bytes(
    bytes(regions["attention"]["arena_size"]))
scale = 1.0 / math.sqrt(2.0)
p = math.exp(scale) / (math.exp(scale) + 1.0)
expected = [(1.0 - p) * 10.0, p * 20.0,
            p * 10.0, (1.0 - p) * 20.0]
Path(sys.argv[2]).write_bytes(struct.pack("<4f", *expected))
Path(sys.argv[3]).write_bytes(struct.pack(
    "<16f",
    1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
    10.0, 0.0, 0.0, 20.0, 0.0, 20.0, 10.0, 0.0))
PY
"${TVM_PYTHON:-python3}" \
    "${ROOT_DIR}/tools/models/build_tvm_byoc_module_package.py" \
    "${KV_ATTENTION_MODULE_DIR}" "${KV_ATTENTION_PACKAGE}" \
    --clear-external-bindings \
    --arena "qkv_projection=${KV_ATTENTION_MODULE_DIR}/qkv-base.arena.bin" \
    --arena "attention=${KV_ATTENTION_MODULE_DIR}/attention-base.arena.bin"
"${TVM_PYTHON:-python3}" \
    "${ROOT_DIR}/tools/models/build_tvm_byoc_invocation.py" \
    "${KV_ATTENTION_MODULE_DIR}" "${KV_ATTENTION_INVOCATION1}" \
    --arena "qkv_projection=${KV_ATTENTION_MODULE_DIR}/qkv-step1.arena.bin" \
    --arena "attention=${KV_ATTENTION_MODULE_DIR}/attention-step1.arena.bin" \
    --scalar kv_length=1
"${TVM_PYTHON:-python3}" \
    "${ROOT_DIR}/tools/models/build_tvm_byoc_invocation.py" \
    "${KV_ATTENTION_MODULE_DIR}" "${KV_ATTENTION_INVOCATION2}" \
    --arena "qkv_projection=${KV_ATTENTION_MODULE_DIR}/qkv-step2.arena.bin" \
    --arena "attention=${KV_ATTENTION_MODULE_DIR}/attention-step2.arena.bin" \
    --scalar kv_length=2
"${TVM_PYTHON:-python3}" - "${MODULE_INVOCATION}" "${MODULE_MISMATCH}" <<'PY'
import sys

image = bytearray(open(sys.argv[1], "rb").read())
assert len(image) >= 32
image[28] ^= 1
open(sys.argv[2], "wb").write(image)
PY
[ -f "${MODULE_PACKAGE}" ] && [ -f "${MODULE_INVOCATION}" ] &&
    [ -f "${MODULE_INVOCATION2}" ] && [ -f "${MODULE_MISMATCH}" ] || {
    echo "error: TVM BYOC module or invocation was not generated" >&2
    exit 1
}
[ -f "${TRANSFORMER_MODULE_PACKAGE}" ] &&
    [ -f "${TRANSFORMER_MODULE_INVOCATION}" ] &&
    [ -f "${TRANSFORMER_EXPECTED}" ] || {
    echo "error: Transformer module package was not generated" >&2
    exit 1
}
[ -f "${ONNX_MODULE_PACKAGE}" ] && [ -f "${ONNX_MODULE_INVOCATION}" ] &&
    [ -f "${ONNX_EXPECTED}" ] || {
    echo "error: ONNX frontend deployment was not generated" >&2
    exit 1
}
[ -f "${ONNX_STATE_MODULE_PACKAGE}" ] &&
    [ -f "${ONNX_STATE_INVOCATION0}" ] &&
    [ -f "${ONNX_STATE_INVOCATION1}" ] && [ -f "${ONNX_STATE_EXPECTED}" ] || {
    echo "error: ONNX stateful deployment was not generated" >&2
    exit 1
}
[ -f "${STATE_MODULE_PACKAGE}" ] && [ -f "${STATE_MODULE_INVOCATION}" ] &&
    [ -f "${STATE_EXPECTED}" ] || {
    echo "error: stateful module package was not generated" >&2
    exit 1
}
[ -f "${APPEND_MODULE_PACKAGE}" ] && [ -f "${APPEND_MODULE_INVOCATION}" ] &&
    [ -f "${APPEND_EXPECTED}" ] || {
    echo "error: state append module package was not generated" >&2
    exit 1
}
[ -f "${ATTENTION_MODULE_PACKAGE}" ] && [ -f "${ATTENTION_INVOCATION1}" ] &&
    [ -f "${ATTENTION_INVOCATION2}" ] && [ -f "${ATTENTION_EXPECTED1}" ] &&
    [ -f "${ATTENTION_EXPECTED2}" ] || {
    echo "error: dynamic attention module package was not generated" >&2
    exit 1
}
[ -f "${KV_ATTENTION_PACKAGE}" ] && [ -f "${KV_ATTENTION_INVOCATION1}" ] &&
    [ -f "${KV_ATTENTION_INVOCATION2}" ] &&
    [ -f "${KV_ATTENTION_EXPECTED}" ] &&
    [ -f "${KV_ATTENTION_STATE_EXPECTED}" ] || {
    echo "error: KV append-attention module package was not generated" >&2
    exit 1
}
EXPECTED_CHECKSUM="$(sed -n \
    's/^xgraph_output_checksum=\(0x[0-9a-fA-F]*\)$/\1/p' \
    "${LOCAL_LOG}" | tail -n 1)"
[ -n "${EXPECTED_CHECKSUM}" ] || {
    echo "error: host reference output checksum missing" >&2
    exit 1
}
EXPECTED_COMMANDS="$("${TVM_PYTHON:-python3}" - "${GRAPH}" <<'PY'
import struct
import sys

with open(sys.argv[1], "rb") as source:
    header = source.read(96)
if len(header) != 96:
    raise SystemExit("compiled XGraph header is truncated")
fields = struct.unpack("<12I2Q8I", header)
if fields[0] != 0x5847504E or fields[1] != 2:
    raise SystemExit("compiled XGraph header has an invalid identity")
print(fields[4])
PY
)"
[ -n "${EXPECTED_COMMANDS}" ] || {
    echo "error: compiled XGraph command count missing" >&2
    exit 1
}

"${ROOT_DIR}/tools/coralnpu/build_qwen_command_flow_smoke.sh"

TEST_SCRIPT="$(mktemp)"
trap 'rm -f "${TEST_SCRIPT}"' EXIT
cat >"${TEST_SCRIPT}" <<'EOF'
#!/bin/sh
set -eu
mkdir -p /proc /sys /tmp /dev
[ -r /proc/mounts ] || mount -t proc proc /proc 2>/dev/null || true
is_mounted()
{
    while read -r _source target _rest; do
        [ "${target}" = "$1" ] && return 0
    done </proc/mounts
    return 1
}
is_mounted /sys || mount -t sysfs sysfs /sys 2>/dev/null || true
is_mounted /tmp || mount -t tmpfs tmpfs /tmp 2>/dev/null || true
is_mounted /dev || mount -t devtmpfs devtmpfs /dev 2>/dev/null || true
fail()
{
    echo "[tvm-byoc-xgraph] FAIL: $*"
    command -v m5 >/dev/null 2>&1 && m5 --inst exit
    exit 1
}
decode_base64()
{
    if command -v base64 >/dev/null 2>&1; then base64 -d
    elif [ -x /bin/busybox ]; then /bin/busybox base64 -d
    elif [ -x /tmp/busybox ]; then /tmp/busybox base64 -d
    else fail 'base64 decoder missing'
    fi
}
[ -f /tmp/coralnpu-baseline-preload.ready ] || fail 'stale checkpoint'
[ -f /tmp/opennpux_coral.ko ] || fail 'kernel module missing'
[ -x /tmp/busybox ] || fail 'BusyBox missing'
if [ ! -c /dev/opennpux-coral ]; then
    /tmp/busybox insmod /tmp/opennpux_coral.ko \
        2>/tmp/opennpux-coral-insmod.err || true
fi
if [ ! -c /dev/opennpux-coral ]; then
    if [ -r /tmp/opennpux-coral-insmod.err ]; then
        while IFS= read -r line; do
            echo "[tvm-byoc-xgraph] insmod: ${line}"
        done </tmp/opennpux-coral-insmod.err
    fi
    fail '/dev/opennpux-coral missing; module and vmlinux must match'
fi
decode_base64 >/tmp/tvm-model.npxg <<'OPENNPUX_TVM_GRAPH_EOF'
EOF
base64 "${GRAPH}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_GRAPH_EOF
decode_base64 >/tmp/tvm-model.arena.bin <<'OPENNPUX_TVM_ARENA_EOF'
EOF
base64 "${ARENA}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<EOF
OPENNPUX_TVM_ARENA_EOF
decode_base64 >/tmp/tvm-transformer-block.npxg <<'OPENNPUX_TVM_TRANSFORMER_GRAPH_EOF'
EOF
base64 "${TRANSFORMER_GRAPH}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_TRANSFORMER_GRAPH_EOF
decode_base64 >/tmp/tvm-transformer-block.arena.bin <<'OPENNPUX_TVM_TRANSFORMER_ARENA_EOF'
EOF
base64 "${TRANSFORMER_ARENA}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<EOF
OPENNPUX_TVM_TRANSFORMER_ARENA_EOF
decode_base64 >/tmp/tvm-transformer-block.npxgm <<'OPENNPUX_TVM_TRANSFORMER_MODULE_EOF'
EOF
base64 "${TRANSFORMER_MODULE_PACKAGE}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_TRANSFORMER_MODULE_EOF
decode_base64 >/tmp/tvm-transformer-block.npxmi <<'OPENNPUX_TVM_TRANSFORMER_INVOCATION_EOF'
EOF
base64 "${TRANSFORMER_MODULE_INVOCATION}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_TRANSFORMER_INVOCATION_EOF
decode_base64 >/tmp/tvm-transformer-block.expected.bin <<'OPENNPUX_TVM_TRANSFORMER_EXPECTED_EOF'
EOF
base64 "${TRANSFORMER_EXPECTED}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<EOF
OPENNPUX_TVM_TRANSFORMER_EXPECTED_EOF
decode_base64 >/tmp/tvm-onnx-projection.npxgm <<'OPENNPUX_TVM_ONNX_MODULE_EOF'
EOF
base64 "${ONNX_MODULE_PACKAGE}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_ONNX_MODULE_EOF
decode_base64 >/tmp/tvm-onnx-projection.npxmi <<'OPENNPUX_TVM_ONNX_INVOCATION_EOF'
EOF
base64 "${ONNX_MODULE_INVOCATION}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_ONNX_INVOCATION_EOF
decode_base64 >/tmp/tvm-onnx-projection.expected.bin <<'OPENNPUX_TVM_ONNX_EXPECTED_EOF'
EOF
base64 "${ONNX_EXPECTED}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<EOF
OPENNPUX_TVM_ONNX_EXPECTED_EOF
decode_base64 >/tmp/tvm-onnx-state.npxgm <<'OPENNPUX_TVM_ONNX_STATE_MODULE_EOF'
EOF
base64 "${ONNX_STATE_MODULE_PACKAGE}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_ONNX_STATE_MODULE_EOF
decode_base64 >/tmp/tvm-onnx-state-0.npxmi <<'OPENNPUX_TVM_ONNX_STATE_INVOCATION0_EOF'
EOF
base64 "${ONNX_STATE_INVOCATION0}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_ONNX_STATE_INVOCATION0_EOF
decode_base64 >/tmp/tvm-onnx-state-1.npxmi <<'OPENNPUX_TVM_ONNX_STATE_INVOCATION1_EOF'
EOF
base64 "${ONNX_STATE_INVOCATION1}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_ONNX_STATE_INVOCATION1_EOF
decode_base64 >/tmp/tvm-onnx-state.expected.bin <<'OPENNPUX_TVM_ONNX_STATE_EXPECTED_EOF'
EOF
base64 "${ONNX_STATE_EXPECTED}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<EOF
OPENNPUX_TVM_ONNX_STATE_EXPECTED_EOF
decode_base64 >/tmp/tvm-stateful-module.npxgm <<'OPENNPUX_TVM_STATE_MODULE_EOF'
EOF
base64 "${STATE_MODULE_PACKAGE}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_STATE_MODULE_EOF
decode_base64 >/tmp/tvm-stateful-module.npxmi <<'OPENNPUX_TVM_STATE_INVOCATION_EOF'
EOF
base64 "${STATE_MODULE_INVOCATION}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_STATE_INVOCATION_EOF
decode_base64 >/tmp/tvm-stateful-module.expected.bin <<'OPENNPUX_TVM_STATE_EXPECTED_EOF'
EOF
base64 "${STATE_EXPECTED}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<EOF
OPENNPUX_TVM_STATE_EXPECTED_EOF
decode_base64 >/tmp/tvm-state-append.npxgm <<'OPENNPUX_TVM_APPEND_MODULE_EOF'
EOF
base64 "${APPEND_MODULE_PACKAGE}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_APPEND_MODULE_EOF
decode_base64 >/tmp/tvm-state-append.npxmi <<'OPENNPUX_TVM_APPEND_INVOCATION_EOF'
EOF
base64 "${APPEND_MODULE_INVOCATION}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_APPEND_INVOCATION_EOF
decode_base64 >/tmp/tvm-state-append.expected.bin <<'OPENNPUX_TVM_APPEND_EXPECTED_EOF'
EOF
base64 "${APPEND_EXPECTED}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<EOF
OPENNPUX_TVM_APPEND_EXPECTED_EOF
decode_base64 >/tmp/tvm-dynamic-attention.npxgm <<'OPENNPUX_TVM_ATTENTION_MODULE_EOF'
EOF
base64 "${ATTENTION_MODULE_PACKAGE}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_ATTENTION_MODULE_EOF
decode_base64 >/tmp/tvm-dynamic-attention-kv1.npxmi <<'OPENNPUX_TVM_ATTENTION_INVOCATION1_EOF'
EOF
base64 "${ATTENTION_INVOCATION1}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_ATTENTION_INVOCATION1_EOF
decode_base64 >/tmp/tvm-dynamic-attention-kv2.npxmi <<'OPENNPUX_TVM_ATTENTION_INVOCATION2_EOF'
EOF
base64 "${ATTENTION_INVOCATION2}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_ATTENTION_INVOCATION2_EOF
decode_base64 >/tmp/tvm-dynamic-attention-kv1.expected.bin <<'OPENNPUX_TVM_ATTENTION_EXPECTED1_EOF'
EOF
base64 "${ATTENTION_EXPECTED1}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_ATTENTION_EXPECTED1_EOF
decode_base64 >/tmp/tvm-dynamic-attention-kv2.expected.bin <<'OPENNPUX_TVM_ATTENTION_EXPECTED2_EOF'
EOF
base64 "${ATTENTION_EXPECTED2}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<EOF
OPENNPUX_TVM_ATTENTION_EXPECTED2_EOF
decode_base64 >/tmp/tvm-kv-attention.npxgm <<'OPENNPUX_TVM_KV_ATTENTION_MODULE_EOF'
EOF
base64 "${KV_ATTENTION_PACKAGE}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_KV_ATTENTION_MODULE_EOF
decode_base64 >/tmp/tvm-kv-attention-step1.npxmi <<'OPENNPUX_TVM_KV_ATTENTION_INVOCATION1_EOF'
EOF
base64 "${KV_ATTENTION_INVOCATION1}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_KV_ATTENTION_INVOCATION1_EOF
decode_base64 >/tmp/tvm-kv-attention-step2.npxmi <<'OPENNPUX_TVM_KV_ATTENTION_INVOCATION2_EOF'
EOF
base64 "${KV_ATTENTION_INVOCATION2}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_KV_ATTENTION_INVOCATION2_EOF
decode_base64 >/tmp/tvm-kv-attention.expected.bin <<'OPENNPUX_TVM_KV_ATTENTION_EXPECTED_EOF'
EOF
base64 "${KV_ATTENTION_EXPECTED}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_TVM_KV_ATTENTION_EXPECTED_EOF
decode_base64 >/tmp/tvm-kv-attention-state.expected.bin <<'OPENNPUX_TVM_KV_ATTENTION_STATE_EXPECTED_EOF'
EOF
base64 "${KV_ATTENTION_STATE_EXPECTED}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<EOF
OPENNPUX_TVM_KV_ATTENTION_STATE_EXPECTED_EOF
decode_base64 >/tmp/tvm-mixed-module.npxgm <<'OPENNPUX_TVM_MODULE_EOF'
EOF
base64 "${MODULE_PACKAGE}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<EOF
OPENNPUX_TVM_MODULE_EOF
decode_base64 >/tmp/tvm-mixed-module.npxmi <<'OPENNPUX_TVM_INVOCATION_EOF'
EOF
base64 "${MODULE_INVOCATION}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<EOF
OPENNPUX_TVM_INVOCATION_EOF
decode_base64 >/tmp/tvm-mixed-module-second.npxmi <<'OPENNPUX_TVM_INVOCATION2_EOF'
EOF
base64 "${MODULE_INVOCATION2}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<EOF
OPENNPUX_TVM_INVOCATION2_EOF
decode_base64 >/tmp/tvm-mixed-module-mismatch.npxmi <<'OPENNPUX_TVM_MISMATCH_EOF'
EOF
base64 "${MODULE_MISMATCH}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<EOF
OPENNPUX_TVM_MISMATCH_EOF
decode_base64 >/tmp/tvm-module.expected.bin <<'OPENNPUX_TVM_EXPECTED_EOF'
EOF
base64 "${MODULE_DIR}/invocation.expected.bin" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<EOF
OPENNPUX_TVM_EXPECTED_EOF
decode_base64 >/tmp/tvm-module-second.expected.bin <<'OPENNPUX_TVM_EXPECTED2_EOF'
EOF
base64 "${MODULE_DIR}/invocation2.expected.bin" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<EOF
OPENNPUX_TVM_EXPECTED2_EOF
OUTPUT="\$(OPENNPUX_CORAL_TRANSPORT=driver \
    OPENNPUX_XGRAPH_OUTPUT_TOLERANCE=0.00005 \
    /tmp/coralctl xgraph-run \
    /tmp/tvm-model.npxg /tmp/tvm-model.arena.bin 0x1d000000 1000000)" || {
    printf '%s\n' "\${OUTPUT}"
    fail 'artifact execution failed'
}
printf '%s\n' "\${OUTPUT}"
has_output_line()
{
    expected="\$1"
    while IFS= read -r line; do
        case "\${line}" in
            "\${expected}"*) return 0 ;;
        esac
    done <<OPENNPUX_OUTPUT_EOF
\${OUTPUT}
OPENNPUX_OUTPUT_EOF
    return 1
}
has_output_line 'xgraph_completed_commands=${EXPECTED_COMMANDS}' ||
    fail 'unexpected command completion count'
has_output_line 'xgraph_output_readback=PASS' ||
    fail 'Local EXTMEM output was not synchronized to the host window'
has_output_line 'xgraph_reference_checksum=${EXPECTED_CHECKSUM}' ||
    fail 'staged reference checksum differs from host reference'
has_output_line 'xgraph_output_reference=PASS' ||
    fail 'output differs numerically from host reference'
has_output_line 'xgraph_artifact_run=PASS' ||
    fail 'runtime PASS verdict missing'
TRANSFORMER_OUTPUT="\$(OPENNPUX_CORAL_TRANSPORT=driver \
    OPENNPUX_XGRAPH_OUTPUT_TOLERANCE=0.00005 \
    /tmp/coralctl xgraph-run /tmp/tvm-transformer-block.npxg \
    /tmp/tvm-transformer-block.arena.bin 0x1d000000 1000000)" || {
    printf '%s\n' "\${TRANSFORMER_OUTPUT}"
    fail 'Transformer block artifact execution failed'
}
printf '%s\n' "\${TRANSFORMER_OUTPUT}"
OUTPUT="\${TRANSFORMER_OUTPUT}"
has_output_line 'xgraph_completed_commands=4' ||
    fail 'Transformer block command count mismatch'
has_output_line 'xgraph_output_reference=PASS' ||
    fail 'Transformer block output differs from independent reference'
has_output_line 'xgraph_artifact_run=PASS' ||
    fail 'Transformer block runtime PASS verdict missing'
echo 'tvm_transformer_block_xgraph=PASS'
TRANSFORMER_MODULE_OUTPUT="\$(OPENNPUX_CORAL_TRANSPORT=driver \
    OPENNPUX_XGRAPH_MODULE_OUTPUT_PATH=/tmp/tvm-transformer-module.output.bin \
    OPENNPUX_XGRAPH_MODULE_INVOCATION_PATH=/tmp/tvm-transformer-block.npxmi \
    /tmp/coralctl xgraph-module-run /tmp/tvm-transformer-block.npxgm \
    0x1d000000 1000000)" || {
    printf '%s\n' "\${TRANSFORMER_MODULE_OUTPUT}"
    fail 'Transformer module execution failed'
}
printf '%s\n' "\${TRANSFORMER_MODULE_OUTPUT}"
OUTPUT="\${TRANSFORMER_MODULE_OUTPUT}"
has_output_line 'xgraph_module_regions_completed=1' ||
    fail 'Transformer module region count mismatch'
has_output_line 'xgraph_module_commands_completed=4' ||
    fail 'Transformer module command count mismatch'
has_output_line 'xgraph_module_invocation_bindings=2' ||
    fail 'Transformer module dynamic binding count mismatch'
has_output_line 'xgraph_module_run=PASS' ||
    fail 'Transformer module runtime PASS verdict missing'
/tmp/coralctl tensor-compare-fp32 \
    /tmp/tvm-transformer-module.output.bin \
    /tmp/tvm-transformer-block.expected.bin 0.00005 ||
    fail 'Transformer module output differs from independent reference'
echo 'tvm_transformer_module_storage=PASS'
ONNX_OUTPUT="\$(OPENNPUX_CORAL_TRANSPORT=driver \
    OPENNPUX_XGRAPH_MODULE_OUTPUT_PATH=/tmp/tvm-onnx-projection.output.bin \
    OPENNPUX_XGRAPH_MODULE_INVOCATION_PATH=/tmp/tvm-onnx-projection.npxmi \
    /tmp/coralctl xgraph-module-run /tmp/tvm-onnx-projection.npxgm \
    0x1d000000 1000000)" || {
    printf '%s\n' "\${ONNX_OUTPUT}"
    fail 'ONNX frontend module execution failed'
}
printf '%s\n' "\${ONNX_OUTPUT}"
OUTPUT="\${ONNX_OUTPUT}"
has_output_line 'xgraph_module_regions_completed=2' ||
    fail 'ONNX frontend module region count mismatch'
has_output_line 'xgraph_module_commands_completed=2' ||
    fail 'ONNX frontend module command count mismatch'
has_output_line 'xgraph_module_host_operations_completed=1' ||
    fail 'ONNX frontend Host partition was not executed'
has_output_line 'xgraph_module_invocation_bindings=2' ||
    fail 'ONNX frontend invocation binding count mismatch'
has_output_line 'xgraph_module_run=PASS' ||
    fail 'ONNX frontend module runtime PASS verdict missing'
/tmp/coralctl tensor-compare-fp32 \
    /tmp/tvm-onnx-projection.output.bin \
    /tmp/tvm-onnx-projection.expected.bin 0.00001 ||
    fail 'ONNX frontend output differs from independent reference'
echo 'tvm_onnx_relax_byoc_xgraph=PASS'
ONNX_STATE_OUTPUT="\$(OPENNPUX_CORAL_TRANSPORT=driver \
    OPENNPUX_XGRAPH_MODULE_OUTPUT_PATH=/tmp/tvm-onnx-state.output.bin \
    OPENNPUX_XGRAPH_MODULE_INVOCATION_SEQUENCE=/tmp/tvm-onnx-state-0.npxmi:/tmp/tvm-onnx-state-1.npxmi \
    /tmp/coralctl xgraph-module-run /tmp/tvm-onnx-state.npxgm \
    0x1d000000 1000000)" || {
    printf '%s\n' "\${ONNX_STATE_OUTPUT}"
    fail 'ONNX stateful decode execution failed'
}
printf '%s\n' "\${ONNX_STATE_OUTPUT}"
OUTPUT="\${ONNX_STATE_OUTPUT}"
has_output_line 'xgraph_module_commands_completed=4' ||
    fail 'ONNX stateful decode command count mismatch'
has_output_line 'xgraph_module_invocations_completed=2' ||
    fail 'ONNX stateful decode invocation count mismatch'
has_output_line 'xgraph_module_state_updates_completed=2' ||
    fail 'ONNX stateful decode update count mismatch'
has_output_line 'xgraph_module_run=PASS' ||
    fail 'ONNX stateful decode runtime PASS verdict missing'
/tmp/coralctl tensor-compare-fp32 /tmp/tvm-onnx-state.output.bin \
    /tmp/tvm-onnx-state.expected.bin 0.00001 ||
    fail 'ONNX stateful decode output differs from independent reference'
echo 'tvm_onnx_stateful_decode=PASS'
STATE_OUTPUT="\$(OPENNPUX_CORAL_TRANSPORT=driver \
    OPENNPUX_XGRAPH_MODULE_OUTPUT_PATH=/tmp/tvm-stateful-module.output.bin \
    OPENNPUX_XGRAPH_MODULE_INVOCATION_SEQUENCE=/tmp/tvm-stateful-module.npxmi:/tmp/tvm-stateful-module.npxmi \
    /tmp/coralctl xgraph-module-run /tmp/tvm-stateful-module.npxgm \
    0x1d000000 1000000)" || {
    printf '%s\n' "\${STATE_OUTPUT}"
    fail 'stateful module sequence failed'
}
printf '%s\n' "\${STATE_OUTPUT}"
OUTPUT="\${STATE_OUTPUT}"
has_output_line 'xgraph_module_commands_completed=10' ||
    fail 'stateful module did not execute both decode steps'
has_output_line 'xgraph_module_invocation_bindings=4' ||
    fail 'stateful module input bindings were not applied per step'
has_output_line 'xgraph_module_invocations_completed=2' ||
    fail 'stateful module invocation count mismatch'
has_output_line 'xgraph_module_run=PASS' ||
    fail 'stateful module runtime PASS verdict missing'
/tmp/coralctl tensor-compare-fp32 \
    /tmp/tvm-stateful-module.output.bin \
    /tmp/tvm-stateful-module.expected.bin 0.00005 ||
    fail 'device-resident state was not preserved across decode steps'
echo 'xgraph_module_state_updates=2'
echo 'tvm_stateful_transformer_decode=PASS'
APPEND_OUTPUT="\$(OPENNPUX_CORAL_TRANSPORT=driver \
    OPENNPUX_XGRAPH_MODULE_STATE_PREFIX=/tmp/tvm-state-append \
    OPENNPUX_XGRAPH_MODULE_INVOCATION_SEQUENCE=/tmp/tvm-state-append.npxmi:/tmp/tvm-state-append.npxmi \
    /tmp/coralctl xgraph-module-run /tmp/tvm-state-append.npxgm \
    0x1d000000 1000000)" || {
    printf '%s\n' "\${APPEND_OUTPUT}"
    fail 'state append module sequence failed'
}
printf '%s\n' "\${APPEND_OUTPUT}"
OUTPUT="\${APPEND_OUTPUT}"
has_output_line 'xgraph_module_state=0 mode=append' ||
    fail 'state append mode was not reported'
has_output_line 'xgraph_module_commands_completed=2' ||
    fail 'state append module did not execute both decode steps'
has_output_line 'xgraph_module_invocations_completed=2' ||
    fail 'state append invocation count mismatch'
has_output_line 'xgraph_module_scalar_bindings=2' ||
    fail 'state append dynamic scalar binding count mismatch'
has_output_line 'xgraph_module_state_updates_completed=2' ||
    fail 'state append update count mismatch'
has_output_line 'xgraph_module_state_bytes=32' ||
    fail 'state append capacity byte count mismatch'
/tmp/coralctl tensor-compare-fp32 /tmp/tvm-state-append.0.bin \
    /tmp/tvm-state-append.expected.bin 0.000001 ||
    fail 'state append window differs from independent reference'
echo 'tvm_kv_state_append=PASS'
if OPENNPUX_CORAL_TRANSPORT=driver \
    OPENNPUX_XGRAPH_MODULE_INVOCATION_SEQUENCE=/tmp/tvm-state-append.npxmi:/tmp/tvm-state-append.npxmi:/tmp/tvm-state-append.npxmi:/tmp/tvm-state-append.npxmi:/tmp/tvm-state-append.npxmi \
    /tmp/coralctl xgraph-module-run /tmp/tvm-state-append.npxgm \
    0x1d000000 1000000 >/tmp/tvm-state-capacity.log 2>&1; then
    fail 'state append accepted an invocation beyond capacity'
fi
CAPACITY_OUTPUT="\$(cat /tmp/tvm-state-capacity.log)"
case "\${CAPACITY_OUTPUT}" in
    *'xgraph-module-run state capacity'*) ;;
    *)
        printf '%s\n' "\${CAPACITY_OUTPUT}"
        fail 'state append capacity rejection diagnostic missing'
        ;;
esac
echo 'tvm_kv_state_capacity_rejection=PASS'
ATTENTION_OUTPUT1="\$(OPENNPUX_CORAL_TRANSPORT=driver \
    OPENNPUX_XGRAPH_MODULE_OUTPUT_PATH=/tmp/tvm-dynamic-attention-kv1.output.bin \
    OPENNPUX_XGRAPH_MODULE_INVOCATION_PATH=/tmp/tvm-dynamic-attention-kv1.npxmi \
    /tmp/coralctl xgraph-module-run /tmp/tvm-dynamic-attention.npxgm \
    0x1d000000 1000000)" || {
    printf '%s\n' "\${ATTENTION_OUTPUT1}"
    fail 'dynamic attention kv_length=1 execution failed'
}
printf '%s\n' "\${ATTENTION_OUTPUT1}"
OUTPUT="\${ATTENTION_OUTPUT1}"
has_output_line 'xgraph_module_scalar_bindings=1' ||
    fail 'dynamic attention kv_length=1 scalar was not applied'
/tmp/coralctl tensor-compare-fp32 \
    /tmp/tvm-dynamic-attention-kv1.output.bin \
    /tmp/tvm-dynamic-attention-kv1.expected.bin 0.00001 ||
    fail 'dynamic attention kv_length=1 output mismatch'
ATTENTION_OUTPUT2="\$(OPENNPUX_CORAL_TRANSPORT=driver \
    OPENNPUX_XGRAPH_MODULE_OUTPUT_PATH=/tmp/tvm-dynamic-attention-kv2.output.bin \
    OPENNPUX_XGRAPH_MODULE_INVOCATION_PATH=/tmp/tvm-dynamic-attention-kv2.npxmi \
    /tmp/coralctl xgraph-module-run /tmp/tvm-dynamic-attention.npxgm \
    0x1d000000 1000000)" || {
    printf '%s\n' "\${ATTENTION_OUTPUT2}"
    fail 'dynamic attention kv_length=2 execution failed'
}
printf '%s\n' "\${ATTENTION_OUTPUT2}"
OUTPUT="\${ATTENTION_OUTPUT2}"
has_output_line 'xgraph_module_scalar_bindings=1' ||
    fail 'dynamic attention kv_length=2 scalar was not applied'
/tmp/coralctl tensor-compare-fp32 \
    /tmp/tvm-dynamic-attention-kv2.output.bin \
    /tmp/tvm-dynamic-attention-kv2.expected.bin 0.00001 ||
    fail 'dynamic attention kv_length=2 output mismatch'
cmp -s /tmp/tvm-dynamic-attention-kv1.output.bin \
    /tmp/tvm-dynamic-attention-kv2.output.bin &&
    fail 'dynamic attention ignored kv_length relocation'
echo 'tvm_dynamic_kv_attention=PASS'
OUTPUT="\$(OPENNPUX_CORAL_TRANSPORT=driver \
    OPENNPUX_XGRAPH_MODULE_OUTPUT_PREFIX=/tmp/tvm-kv-attention-output \
    OPENNPUX_XGRAPH_MODULE_STATE_PREFIX=/tmp/tvm-kv-attention-state \
    OPENNPUX_XGRAPH_MODULE_INVOCATION_SEQUENCE=/tmp/tvm-kv-attention-step1.npxmi:/tmp/tvm-kv-attention-step2.npxmi \
    /tmp/coralctl xgraph-module-run /tmp/tvm-kv-attention.npxgm \
    0x1d000000 1000000)" || {
    printf '%s\n' "\${OUTPUT}"
    fail 'KV append-attention sequence failed'
}
printf '%s\n' "\${OUTPUT}"
has_output_line 'xgraph_module_commands_completed=12' ||
    fail 'KV append-attention command count mismatch'
has_output_line 'xgraph_module_scalar_bindings=2' ||
    fail 'KV append-attention scalar count mismatch'
has_output_line 'xgraph_module_invocations_completed=2' ||
    fail 'KV append-attention invocation count mismatch'
has_output_line 'xgraph_module_state_updates_completed=2' ||
    fail 'KV append-attention state update count mismatch'
has_output_line 'xgraph_module_state=0 mode=append_planar2' ||
    fail 'KV append-attention planar state was not published'
/tmp/coralctl tensor-compare-fp32 /tmp/tvm-kv-attention-output.1.bin \
    /tmp/tvm-kv-attention.expected.bin 0.00001 ||
    fail 'KV append-attention context mismatch'
/tmp/coralctl tensor-compare-fp32 /tmp/tvm-kv-attention-state.0.bin \
    /tmp/tvm-kv-attention-state.expected.bin 0 ||
    fail 'KV append-attention state mismatch'
echo 'tvm_kv_append_attention_sequence=PASS'
if OPENNPUX_CORAL_TRANSPORT=driver \
    OPENNPUX_XGRAPH_MODULE_INVOCATION_PATH=/tmp/tvm-mixed-module-mismatch.npxmi \
    /tmp/coralctl xgraph-module-run \
    /tmp/tvm-mixed-module.npxgm 0x1d000000 1000000 \
    >/tmp/tvm-module-mismatch.log 2>&1; then
    fail 'module accepted an invocation for a different identity'
fi
MISMATCH_OUTPUT="\$(cat /tmp/tvm-module-mismatch.log)"
case "\${MISMATCH_OUTPUT}" in
    *'xgraph-module-run invocation'*) ;;
    *)
        printf '%s\n' "\${MISMATCH_OUTPUT}"
        fail 'module identity rejection diagnostic missing'
        ;;
esac
echo 'xgraph_module_identity_rejection=PASS'
OUTPUT="\$(OPENNPUX_CORAL_TRANSPORT=driver \
    OPENNPUX_XGRAPH_OUTPUT_TOLERANCE=0.00005 \
    OPENNPUX_XGRAPH_MODULE_OUTPUT_PATH=/tmp/tvm-module.output.bin \
    OPENNPUX_XGRAPH_MODULE_OUTPUT_PREFIX=/tmp/tvm-module-output \
    OPENNPUX_XGRAPH_MODULE_INVOCATION_PATH=/tmp/tvm-mixed-module.npxmi \
    /tmp/coralctl xgraph-module-run \
    /tmp/tvm-mixed-module.npxgm 0x1d000000 1000000)" || {
    printf '%s\n' "\${OUTPUT}"
    fail 'compiled TVM module execution failed'
}
printf '%s\n' "\${OUTPUT}"
has_output_line 'xgraph_module_regions_completed=2' ||
    fail 'module region completion count mismatch'
has_output_line 'xgraph_module_commands_completed=2' ||
    fail 'module command completion count mismatch'
has_output_line 'xgraph_module_host_operations_completed=1' ||
    fail 'compiled Host pipeline was not executed'
has_output_line 'xgraph_module_invocation_bindings=2' ||
    fail 'dynamic invocation bindings were not applied'
has_output_line 'xgraph_module_outputs_completed=1' ||
    fail 'module output completion count mismatch'
has_output_line 'xgraph_module_output_bytes=32' ||
    fail 'module aggregate output size mismatch'
has_output_line 'xgraph_module_run=PASS' ||
    fail 'module runtime PASS verdict missing'
[ "\$(wc -c </tmp/tvm-module.output.bin)" -eq 32 ] ||
    fail 'module output size mismatch'
[ "\$(wc -c </tmp/tvm-module-output.0.bin)" -eq 32 ] ||
    fail 'indexed module output size mismatch'
cmp /tmp/tvm-module.output.bin /tmp/tvm-module-output.0.bin ||
    fail 'compatibility and indexed module outputs differ'
/tmp/coralctl tensor-compare-fp32 /tmp/tvm-module.output.bin \
    /tmp/tvm-module.expected.bin 0.00005 ||
    fail 'first module invocation differs from independent reference'
SECOND_OUTPUT="\$(OPENNPUX_CORAL_TRANSPORT=driver \
    OPENNPUX_XGRAPH_MODULE_OUTPUT_PATH=/tmp/tvm-module-second.output.bin \
    OPENNPUX_XGRAPH_MODULE_INVOCATION_PATH=/tmp/tvm-mixed-module-second.npxmi \
    /tmp/coralctl xgraph-module-run \
    /tmp/tvm-mixed-module.npxgm 0x1d000000 1000000)" || {
    printf '%s\n' "\${SECOND_OUTPUT}"
    fail 'second module invocation failed'
}
printf '%s\n' "\${SECOND_OUTPUT}"
case "\${SECOND_OUTPUT}" in
    *'xgraph_module_invocation_bindings=2'*'xgraph_module_run=PASS'*) ;;
    *) fail 'second module invocation verdict missing' ;;
esac
[ "\$(wc -c </tmp/tvm-module-second.output.bin)" -eq 32 ] ||
    fail 'second module output size mismatch'
/tmp/coralctl tensor-compare-fp32 /tmp/tvm-module-second.output.bin \
    /tmp/tvm-module-second.expected.bin 0.00005 ||
    fail 'second module invocation differs from independent reference'
if cmp -s /tmp/tvm-module.output.bin /tmp/tvm-module-second.output.bin; then
    fail 'different module invocations produced identical output'
fi
echo 'xgraph_module_reused_invocations=2'
echo 'xgraph_module_reuse=PASS'
echo 'xgraph_module_chain=PASS'
echo 'tvm_onnx_relax_byoc_xgraph=PASS'
echo 'tvm_byoc_xgraph=PASS'
echo '[tvm-byoc-xgraph] PASS'
command -v m5 >/dev/null 2>&1 && m5 --inst exit
exit 0
EOF

CORAL_NPU_LAUNCH_FIRMWARE="${FIRMWARE}" \
CORAL_NPU_LAUNCH_TEST_SCRIPT="${TEST_SCRIPT}" \
CORAL_NPU_LAUNCH_HOST_LOG="${HOST_LOG}" \
CORAL_NPU_LAUNCH_DEBUG_LOG="${DEBUG_LOG}" \
CORAL_NPU_LAUNCH_XOPENNPUX=1 \
CORAL_NPU_LAUNCH_EXPECTED_GUEST_VERDICT="tvm_byoc_xgraph=PASS" \
CORAL_NPU_LAUNCH_EXPECTED_XOPENNPUX_OPS="tmma tadd trmsnorm tsilu tsoftmax tattention" \
    "${ROOT_DIR}/tools/coralnpu/run_npu_launch_test.sh"

echo "TVM BYOC Guest -> Coral firmware -> XGraph test: PASS"
