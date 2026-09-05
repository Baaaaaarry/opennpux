#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BUILD_DIR="${ROOT_DIR}/build/local-tests/tvm-byoc-xgraph"
GRAPH="${BUILD_DIR}/relax-model.npxg"
ARENA="${BUILD_DIR}/relax-model.arena.bin"
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
OPENNPUX_REQUIRE_TVM=1 \
    "${ROOT_DIR}/tools/models/test_tvm_byoc_xgraph_codegen.sh" \
    2>&1 | tee "${LOCAL_LOG}"
[ -f "${GRAPH}" ] && [ -f "${ARENA}" ] || {
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
EXPECTED_CHECKSUM="$(sed -n \
    's/^xgraph_output_checksum=\(0x[0-9a-fA-F]*\)$/\1/p' \
    "${LOCAL_LOG}" | tail -n 1)"
[ -n "${EXPECTED_CHECKSUM}" ] || {
    echo "error: host reference output checksum missing" >&2
    exit 1
}
EXPECTED_COMMANDS="$(sed -n \
    's/^xgraph_commands=\([0-9][0-9]*\)$/\1/p' \
    "${LOCAL_LOG}" | tail -n 1)"
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
if cmp -s /tmp/tvm-module.output.bin /tmp/tvm-module-second.output.bin; then
    fail 'different module invocations produced identical output'
fi
echo 'xgraph_module_reused_invocations=2'
echo 'xgraph_module_reuse=PASS'
echo 'xgraph_module_chain=PASS'
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
CORAL_NPU_LAUNCH_EXPECTED_XOPENNPUX_OPS="tmma tadd tsilu tsoftmax" \
    "${ROOT_DIR}/tools/coralnpu/run_npu_launch_test.sh"

echo "TVM BYOC Guest -> Coral firmware -> XGraph test: PASS"
