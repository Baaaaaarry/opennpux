#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BUILD_DIR="${ROOT_DIR}/build/local-tests/tvm-byoc-xgraph"
GRAPH="${BUILD_DIR}/relax-model.npxg"
ARENA="${BUILD_DIR}/relax-model.arena.bin"
MODULE_DIR="${BUILD_DIR}/module"
REGION0_GRAPH="${MODULE_DIR}/region-000-residual.npxg"
REGION1_GRAPH="${MODULE_DIR}/region-001-activation.npxg"
REGION0_ARENA="${MODULE_DIR}/region-000-residual.arena.bin"
REGION1_ARENA="${MODULE_DIR}/region-001-activation.arena.bin"
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
"${TVM_PYTHON:-python3}" - \
    "${REGION0_GRAPH}.json" "${REGION0_ARENA}" \
    "${REGION1_GRAPH}.json" "${REGION1_ARENA}" <<'PY'
import json
import math
import struct
import sys

metadata0 = json.load(open(sys.argv[1], encoding="utf-8"))
metadata1 = json.load(open(sys.argv[3], encoding="utf-8"))
tensors0 = {tensor["name"]: tensor for tensor in metadata0["tensors"]}
tensors1 = {tensor["name"]: tensor for tensor in metadata1["tensors"]}
lhs = [float(value) for value in range(-4, 4)]
rhs = [1.0] * 8
summed = [a + b for a, b in zip(lhs, rhs)]
host_relu = [max(value, 0.0) for value in summed]
activated = [value / (1.0 + math.exp(-value)) for value in host_relu]
arena0 = bytearray(metadata0["arena_size"])
arena1 = bytearray(metadata1["arena_size"])
for arena, table, name, values in (
    (arena0, tensors0, "lhs", lhs),
    (arena0, tensors0, "rhs", rhs),
    (arena0, tensors0, "sum", summed),
    (arena1, tensors1, "output", activated),
):
    struct.pack_into(f"<{len(values)}f", arena, table[name]["offset"], *values)
open(sys.argv[2], "wb").write(arena0)
open(sys.argv[4], "wb").write(arena1)
print(f"module_region1_input_offset={tensors1['input']['offset']}")
PY
REGION1_INPUT_OFFSET="$("${TVM_PYTHON:-python3}" -c \
    'import json,sys; print(next(t["offset"] for t in json.load(open(sys.argv[1]))["tensors"] if t["name"] == "input"))' \
    "${REGION1_GRAPH}.json")"
REGION1_ARENA_SIZE="$(wc -c <"${REGION1_ARENA}")"
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
decode_base64 >/tmp/tvm-region0.npxg <<'OPENNPUX_REGION0_GRAPH_EOF'
EOF
base64 "${REGION0_GRAPH}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_REGION0_GRAPH_EOF
decode_base64 >/tmp/tvm-region0.arena.bin <<'OPENNPUX_REGION0_ARENA_EOF'
EOF
base64 "${REGION0_ARENA}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_REGION0_ARENA_EOF
decode_base64 >/tmp/tvm-region1.npxg <<'OPENNPUX_REGION1_GRAPH_EOF'
EOF
base64 "${REGION1_GRAPH}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<'EOF'
OPENNPUX_REGION1_GRAPH_EOF
decode_base64 >/tmp/tvm-region1.arena.bin <<'OPENNPUX_REGION1_ARENA_EOF'
EOF
base64 "${REGION1_ARENA}" >>"${TEST_SCRIPT}"
cat >>"${TEST_SCRIPT}" <<EOF
OPENNPUX_REGION1_ARENA_EOF
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
OUTPUT="\$(OPENNPUX_CORAL_TRANSPORT=driver \
    OPENNPUX_XGRAPH_OUTPUT_TOLERANCE=0.00005 \
    OPENNPUX_XGRAPH_OUTPUT_PATH=/tmp/tvm-region0.output.bin \
    /tmp/coralctl xgraph-run \
    /tmp/tvm-region0.npxg /tmp/tvm-region0.arena.bin \
    0x1d000000 1000000)" || {
    printf '%s\n' "\${OUTPUT}"
    fail 'module region 0 execution failed'
}
printf '%s\n' "\${OUTPUT}"
has_output_line 'xgraph_completed_commands=1' ||
    fail 'module region 0 command count mismatch'
has_output_line 'xgraph_output_readback=PASS' ||
    fail 'module region 0 readback failed'
has_output_line 'xgraph_output_reference=PASS' ||
    fail 'module region 0 reference mismatch'
[ "\$(wc -c </tmp/tvm-region0.output.bin)" -eq 32 ] ||
    fail 'module region 0 output size mismatch'
HOST_OUTPUT="\$(/tmp/coralctl host-tensor-unary relu \
    /tmp/tvm-region0.output.bin /tmp/tvm-region0.host.bin)" || {
    printf '%s\n' "\${HOST_OUTPUT}"
    fail 'module Host ReLU execution failed'
}
printf '%s\n' "\${HOST_OUTPUT}"
case "\${HOST_OUTPUT}" in
    *'host_tensor_run=PASS'*) ;;
    *) fail 'module Host ReLU PASS verdict missing' ;;
esac
[ "\$(wc -c </tmp/tvm-region0.host.bin)" -eq 32 ] ||
    fail 'module Host ReLU output size mismatch'
if command -v dd >/dev/null 2>&1; then
    DD_BIN="\$(command -v dd)"
    DD_APPLET=
elif /tmp/busybox dd --help >/dev/null 2>&1; then
    DD_BIN=/tmp/busybox
    DD_APPLET=dd
else
    fail 'module Tensor edge copy requires dd'
fi
if ! "\${DD_BIN}" \${DD_APPLET} if=/tmp/tvm-region0.host.bin \
    of=/tmp/tvm-region1.arena.bin bs=32 count=1 \
    seek=$((REGION1_INPUT_OFFSET / 32)) conv=notrunc \
    2>/tmp/tvm-edge-copy.err; then
    [ ! -r /tmp/tvm-edge-copy.err ] ||
        while IFS= read -r line; do
            echo "[tvm-byoc-xgraph] edge-copy: \${line}"
        done </tmp/tvm-edge-copy.err
    fail 'module Tensor edge copy failed'
fi
[ "\$(wc -c </tmp/tvm-region1.arena.bin)" -eq ${REGION1_ARENA_SIZE} ] ||
    fail 'module region 1 arena size changed during edge copy'
OUTPUT="\$(OPENNPUX_CORAL_TRANSPORT=driver \
    OPENNPUX_XGRAPH_OUTPUT_TOLERANCE=0.00005 \
    OPENNPUX_XGRAPH_OUTPUT_PATH=/tmp/tvm-region1.output.bin \
    /tmp/coralctl xgraph-run \
    /tmp/tvm-region1.npxg /tmp/tvm-region1.arena.bin \
    0x1d000000 1000000)" || {
    printf '%s\n' "\${OUTPUT}"
    fail 'module region 1 execution failed'
}
printf '%s\n' "\${OUTPUT}"
has_output_line 'xgraph_completed_commands=1' ||
    fail 'module region 1 command count mismatch'
has_output_line 'xgraph_output_readback=PASS' ||
    fail 'module region 1 readback failed'
has_output_line 'xgraph_output_reference=PASS' ||
    fail 'module region 1 reference mismatch'
echo 'xgraph_module_regions_completed=2'
echo 'xgraph_module_direct_edges=0'
echo 'xgraph_module_host_bindings=1'
echo 'xgraph_module_host_pipeline=relax.nn.relu'
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
