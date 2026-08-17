#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
GEM5_ROOT="${ROOT_DIR}/thirdparty/gem5"
MODEL_DIR="${CORAL_MODEL_DIR:-/data/models/Qwen3.5-35B}"
MANIFEST="${CORAL_MODEL_MANIFEST:-${MODEL_DIR}/model.npxm}"
WEIGHT_PLAN="${CORAL_NPU_WEIGHT_PLAN:-${MODEL_DIR}/model.npxw}"
RANGE_INDEX="${CORAL_NPU_WEIGHT_RANGES:-${MODEL_DIR}/model.npxr}"
BRIDGE="${CORAL_RTL_BRIDGE:-${ROOT_DIR}/build/coralnpu/libcoralnpu_gem5_rvv_highmem_bridge.so}"
FIRMWARE="${CORAL_RTL_FIRMWARE:-${ROOT_DIR}/build/coralnpu/gem5_gptq_projection.elf}"
CORALCTL="${ROOT_DIR}/build/guest-tools/coralctl-aarch64"
IMAGE="$(mktemp)"
SCRIPT="$(mktemp)"
trap 'rm -f "$IMAGE" "$SCRIPT"' EXIT

for path in "$MANIFEST" "$WEIGHT_PLAN" "$RANGE_INDEX" "$BRIDGE" \
            "$FIRMWARE" "$CORALCTL"; do
    [ -r "$path" ] || { echo "error: required file missing: $path" >&2; exit 1; }
done

"${ROOT_DIR}/tools/models/materialize_gptq_projection.py" \
    "$MANIFEST" "$WEIGHT_PLAN" "$RANGE_INDEX" "$IMAGE" \
    --layer "${CORAL_GPTQ_LAYER:-0}" \
    --expert "${CORAL_GPTQ_EXPERT:-0}" \
    --slot "${CORAL_GPTQ_SLOT:-gate_proj}"
IMAGE_BYTES="$(wc -c < "$IMAGE" | tr -d ' ')"

cat >"$SCRIPT" <<'EOF'
#!/bin/sh
decode_base64()
{
    if command -v base64 >/dev/null 2>&1; then base64 -d
    elif [ -x /bin/busybox ]; then /bin/busybox base64 -d
    elif [ -x /tmp/busybox ]; then /tmp/busybox base64 -d
    else return 1
    fi
}
decode_base64 >/tmp/coralctl <<'OPENNPUX_CORALCTL_EOF'
EOF
base64 "$CORALCTL" >>"$SCRIPT"
cat >>"$SCRIPT" <<'EOF'
OPENNPUX_CORALCTL_EOF
chmod 0755 /tmp/coralctl
decode_base64 >/tmp/gptq-projection.bin <<'OPENNPUX_GPTQ_EOF'
EOF
base64 "$IMAGE" >>"$SCRIPT"
cat >>"$SCRIPT" <<'EOF'
OPENNPUX_GPTQ_EOF
echo '[coral-gptq-projection-test] started'
/tmp/coralctl mem-load /tmp/gptq-projection.bin || {
    echo '[coral-gptq-projection-test] FAIL: EXTMEM image load failed'
    m5 --inst exit
    exit 1
}
/tmp/coralctl run || {
    echo '[coral-gptq-projection-test] FAIL: NPU execution failed'
    m5 --inst exit
    exit 1
}
read_value()
{
    /tmp/coralctl mem-read32 "$1" | sed -n 's/.*=//p'
}
STATE="$(read_value 12)"
ERROR="$(read_value 16)"
OPERATIONS="$(read_value 64)"
CHECKSUM="$(read_value 96)"
echo "gptq_projection_state=$STATE"
echo "gptq_projection_error=$ERROR"
echo "gptq_projection_operations_low=$OPERATIONS"
echo "gptq_projection_checksum=$CHECKSUM"
if [ "$STATE" = 0x00000002 ] && [ "$ERROR" = 0x00000000 ] &&
   [ "$OPERATIONS" != 0x00000000 ] && [ "$CHECKSUM" != 0x00000000 ]; then
    echo 'gptq_projection_run=PASS'
    echo '[coral-gptq-projection-test] PASS'
else
    echo '[coral-gptq-projection-test] FAIL: invalid projection result'
fi
m5 --inst exit
exit 0
EOF

"${ROOT_DIR}/sim/gem5/apply_patchset.sh"
cd "$GEM5_ROOT"
CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_BRIDGE="$BRIDGE" \
CORAL_RTL_FIRMWARE="$FIRMWARE" \
CORAL_CKPT_ROOT="${CORAL_CKPT_ROOT:-${ROOT_DIR}/checkpoint/coralnpu_gptq_projection_ckpt}" \
CORAL_CONFIG_OPTIONS="${CORAL_CONFIG_OPTIONS:-} --npu-dma-shared-size=8MiB --npu-operator-mode=hybrid --npu-fast-dma --npu-fast-dma-sync-offset=0 --npu-fast-dma-sync-size=${IMAGE_BYTES}B" \
CORAL_REBUILD_CKPT="${CORAL_REBUILD_CKPT:-0}" \
CORAL_RESUME_BOOTSCRIPT="$SCRIPT" \
./run_multicore.sh
