#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BUILD_DIR="${ROOT_DIR}/build/local-tests/tvm-byoc-xgraph"
GRAPH="${BUILD_DIR}/relax-model.npxg"
ARENA="${BUILD_DIR}/relax-model.arena.bin"
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
has_output_line 'xgraph_reference_checksum=${EXPECTED_CHECKSUM}' ||
    fail 'staged reference checksum differs from host reference'
has_output_line 'xgraph_output_reference=PASS' ||
    fail 'output differs numerically from host reference'
has_output_line 'xgraph_artifact_run=PASS' ||
    fail 'runtime PASS verdict missing'
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
