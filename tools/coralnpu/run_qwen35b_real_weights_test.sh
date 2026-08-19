#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
KERNEL_RELEASE_FILE="${ROOT_DIR}/build/kernel/kernel.release"
MODEL_DIR="${CORAL_MODEL_DIR:-/data/models/Qwen3.5-35B}"
EXECUTABLE_NAME="${CORAL_NPU_EXECUTABLE_NAME:-model.npxc}"
MANIFEST_NAME="${CORAL_NPU_MANIFEST_NAME:-model.npxm}"
RANGE_NAME="${CORAL_NPU_RANGE_NAME:-model.npxr}"
POLL_COUNT="${CORAL_PAGED_POLL_COUNT:-100000000}"
BASE="${CORAL_NPU_BASE:-0x1d000000}"
PROMPT="${CORAL_QWEN_PROMPT:-OpenNPUX heterogeneous inference}"
NUMERICAL_ENV=""
SIM_HOST_PAGING="${CORAL_SIM_HOST_PAGING:-1}"
[ "${CORAL_QWEN35B_NUMERICAL:-0}" = 0 ] ||
    NUMERICAL_ENV="OPENNPUX_NPU_NUMERICAL=1"

[ "${#PROMPT}" -lt 128 ] || {
    echo "error: CORAL_QWEN_PROMPT must be shorter than 128 bytes" >&2
    exit 1
}
case "$PROMPT" in
    *"'"*)
        echo "error: CORAL_QWEN_PROMPT cannot contain a single quote" >&2
        exit 1
        ;;
esac
case "$MODEL_DIR" in
    *,*)
        echo "error: CORAL_MODEL_DIR cannot contain a comma (9P mount option separator)" >&2
        exit 1
        ;;
esac

for asset in "$EXECUTABLE_NAME" "$MANIFEST_NAME" "$RANGE_NAME"; do
    [ -r "$MODEL_DIR/$asset" ] || {
        echo "error: Qwen model asset missing: $MODEL_DIR/$asset" >&2
        echo "prepare it with: ./tools/models/prepare_hf_model_package.sh $MODEL_DIR" >&2
        exit 1
    }
done
command -v diod >/dev/null 2>&1 || {
    echo "error: diod is required for the guest model mount" >&2
    echo "Ubuntu: sudo apt-get install diod" >&2
    exit 1
}

SIM_HOST_ENV=""
SIM_HOST_BUNDLE=""
if [ "$SIM_HOST_PAGING" != 0 ]; then
    if [ "${CORAL_QWEN35B_NUMERICAL:-0}" = 0 ]; then
        BUNDLE_MODE="functional"
        BUNDLE_MAX_PAGES=1
    else
        BUNDLE_MODE="numerical"
        BUNDLE_MAX_PAGES=0
    fi
    SIM_HOST_BUNDLE="${CORAL_SIM_HOST_PAGE_BUNDLE:-${ROOT_DIR}/build/model-pages/qwen35b-${BUNDLE_MODE}-64k.npxb}"
    bundle_stale=0
    [ -r "$SIM_HOST_BUNDLE" ] || bundle_stale=1
    if [ "$bundle_stale" -eq 0 ] &&
       { [ "$MODEL_DIR/$MANIFEST_NAME" -nt "$SIM_HOST_BUNDLE" ] ||
         [ "$MODEL_DIR/$RANGE_NAME" -nt "$SIM_HOST_BUNDLE" ]; }; then
        bundle_stale=1
    fi
    if [ "$bundle_stale" -eq 0 ] &&
       find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' \
           -newer "$SIM_HOST_BUNDLE" -print -quit | grep -q .; then
        bundle_stale=1
    fi
    if [ "$bundle_stale" -eq 1 ] ||
       [ "${CORAL_REBUILD_SIM_HOST_BUNDLE:-0}" != 0 ]; then
        echo "[coral-qwen35b-real-weights-test] building sim-host page bundle" >&2
        "${ROOT_DIR}/tools/models/build_sim_host_page_bundle.sh" \
            "$MODEL_DIR/$MANIFEST_NAME" "$MODEL_DIR/$RANGE_NAME" \
            "$SIM_HOST_BUNDLE" 65536 "$BUNDLE_MAX_PAGES"
    fi
    SIM_HOST_ENV="OPENNPUX_SIM_HOST_PAGING=1"
    echo "[coral-qwen35b-real-weights-test] sim-host bundle: $SIM_HOST_BUNDLE" >&2
fi

if [ -z "${CORAL_KERNEL_IMAGE:-}" ]; then
    if [ ! -r "$KERNEL_RELEASE_FILE" ]; then
        echo "error: kernel release metadata missing: $KERNEL_RELEASE_FILE" >&2
        echo "build the 9P-enabled kernel with: ./tools/kernel/build_arm64_kernel.sh" >&2
        exit 1
    fi
    KERNEL_RELEASE="$(cat "$KERNEL_RELEASE_FILE")"
    CORAL_KERNEL_IMAGE="${ROOT_DIR}/build/kernel/vmlinux-${KERNEL_RELEASE}"
fi
[ -r "$CORAL_KERNEL_IMAGE" ] || {
    echo "error: kernel image missing: $CORAL_KERNEL_IMAGE" >&2
    echo "build it with: ./tools/kernel/build_arm64_kernel.sh" >&2
    exit 1
}

CORALCTL="${ROOT_DIR}/build/guest-tools/coralctl-aarch64"
FIRMWARE="${CORAL_RTL_FIRMWARE:-${ROOT_DIR}/build/coralnpu/gem5_npu_command_processor_smoke.elf}"
BRIDGE="${CORAL_RTL_BRIDGE:-${ROOT_DIR}/build/coralnpu/libcoralnpu_gem5_bridge.so}"
coralctl_stale=0
if [ ! -x "$CORALCTL" ]; then
    coralctl_stale=1
elif find "${ROOT_DIR}/runtime/host" -type f -newer "$CORALCTL" \
     -print -quit | grep -q .; then
    coralctl_stale=1
fi
if [ "$coralctl_stale" -eq 1 ]; then
    echo "[coral-qwen35b-real-weights-test] coralctl missing or stale; rebuilding" >&2
    "${ROOT_DIR}/tools/guest_tools/build_coralctl.sh"
fi
[ -r "$FIRMWARE" ] || "${ROOT_DIR}/tools/coralnpu/build_rtl_bridge.sh"
[ -r "$BRIDGE" ] || { echo "error: RTL bridge missing: $BRIDGE" >&2; exit 1; }

TMP_SCRIPT="$(mktemp)"
trap 'rm -f "$TMP_SCRIPT"' EXIT

cat >"$TMP_SCRIPT" <<EOF
#!/bin/sh
set -u
mkdir -p /proc /sys /dev /tmp /mnt/opennpux-model
mount -t proc proc /proc 2>/dev/null || true
mount -t sysfs sysfs /sys 2>/dev/null || true
mount -t devtmpfs devtmpfs /dev 2>/dev/null || true
decode_base64()
{
    if command -v base64 >/dev/null 2>&1; then
        base64 -d
    elif [ -x /bin/busybox ]; then
        /bin/busybox base64 -d
    elif [ -x /tmp/busybox ]; then
        /tmp/busybox base64 -d
    else
        return 1
    fi
}
decode_base64 >/tmp/coralctl <<'OPENNPUX_CORALCTL_EOF'
EOF
base64 "$CORALCTL" >>"$TMP_SCRIPT"
cat >>"$TMP_SCRIPT" <<EOF
OPENNPUX_CORALCTL_EOF
chmod 0755 /tmp/coralctl
echo '[coral-qwen35b-real-weights-test] started'
if ! grep -qw 9p /proc/filesystems; then
    echo '[coral-qwen35b-real-weights-test] FAIL: kernel lacks CONFIG_9P_FS'
    m5 --inst exit
    exit 1
fi
if ! mount -t 9p \
    -o 'trans=virtio,version=9p2000.L,ro,aname=$MODEL_DIR' \
    gem5 /mnt/opennpux-model; then
    echo '[coral-qwen35b-real-weights-test] FAIL: VirtIO 9P model mount failed'
    dmesg 2>/dev/null | tail -n 20 || true
    m5 --inst exit
    exit 1
fi
for asset in '$EXECUTABLE_NAME' '$MANIFEST_NAME' '$RANGE_NAME'; do
    if [ ! -r "/mnt/opennpux-model/\$asset" ]; then
        echo "[coral-qwen35b-real-weights-test] FAIL: mounted asset missing: \$asset"
        m5 --inst exit
        exit 1
    fi
done
env $NUMERICAL_ENV $SIM_HOST_ENV \
    OPENNPUX_PROMPT='$PROMPT' \
    OPENNPUX_MODEL_ROOT=/mnt/opennpux-model \
    /tmp/coralctl executable-run-paged \
    /mnt/opennpux-model/$EXECUTABLE_NAME decode \
    /mnt/opennpux-model/$MANIFEST_NAME \
    /mnt/opennpux-model/$RANGE_NAME \
    $BASE $POLL_COUNT || {
    echo '[coral-qwen35b-real-weights-test] FAIL: real weight execution failed'
    m5 --inst exit
    exit 1
}
echo '[coral-qwen35b-real-weights-test] PASS'
m5 --inst exit
exit 0
EOF

"${ROOT_DIR}/sim/gem5/apply_patchset.sh"
cd "${ROOT_DIR}/thirdparty/gem5"
CORAL_NPU_BACKEND=verilated-coral \
CORAL_SIM_HOST_PAGE_BUNDLE="$SIM_HOST_BUNDLE" \
CORAL_RTL_BRIDGE="$BRIDGE" \
CORAL_RTL_FIRMWARE="$FIRMWARE" \
CORAL_RTL_CYCLES_PER_EVENT="${CORAL_RTL_CYCLES_PER_EVENT:-1000}" \
CORAL_KERNEL_IMAGE="$CORAL_KERNEL_IMAGE" \
CORAL_AUTO_RESUME_AFTER_CKPT="${CORAL_AUTO_RESUME_AFTER_CKPT:-1}" \
CORAL_CKPT_ROOT="${CORAL_CKPT_ROOT:-${ROOT_DIR}/checkpoint/coralnpu_qwen35b_real_9p_ckpt}" \
CORAL_CONFIG_OPTIONS="${CORAL_CONFIG_OPTIONS:-} --vio-9p --vio-9p-root=$MODEL_DIR --npu-operator-mode=hybrid --npu-dma-shared-base=0x8f000000 --npu-dma-shared-size=8MiB --npu-fast-dma --npu-fast-dma-event-batch=${CORAL_FAST_DMA_EVENT_BATCH:-1} --npu-fast-dma-sync-offset=0 --npu-fast-dma-sync-size=64KiB" \
CORAL_RESUME_BOOTSCRIPT="$TMP_SCRIPT" \
./run_multicore.sh
