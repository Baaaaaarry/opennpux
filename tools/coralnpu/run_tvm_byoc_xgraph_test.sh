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

"${ROOT_DIR}/tools/coralnpu/build_qwen_command_flow_smoke.sh"

TEST_SCRIPT="$(mktemp)"
trap 'rm -f "${TEST_SCRIPT}"' EXIT
cat >"${TEST_SCRIPT}" <<'EOF'
#!/bin/sh
set -eu
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
OUTPUT="\$(OPENNPUX_CORAL_TRANSPORT=driver /tmp/coralctl xgraph-run \
    /tmp/tvm-model.npxg /tmp/tvm-model.arena.bin 0x1d000000 1000000)" || {
    printf '%s\n' "\${OUTPUT}"
    fail 'artifact execution failed'
}
printf '%s\n' "\${OUTPUT}"
printf '%s\n' "\${OUTPUT}" | grep -qx 'xgraph_completed_commands=4' ||
    fail 'unexpected command completion count'
printf '%s\n' "\${OUTPUT}" | grep -qx 'xgraph_output_checksum=${EXPECTED_CHECKSUM}' ||
    fail 'output checksum differs from host reference'
printf '%s\n' "\${OUTPUT}" | grep -qx 'xgraph_artifact_run=PASS' ||
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
