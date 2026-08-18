#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
GEM5_ROOT="${ROOT_DIR}/thirdparty/gem5"
EXECUTABLE="${CORAL_NPU_EXECUTABLE:-${ROOT_DIR}/build/local-tests/model-package/hf-model/model.npxc}"
FIRMWARE="${CORAL_RTL_FIRMWARE:-${ROOT_DIR}/build/coralnpu/gem5_npu_command_processor_smoke.elf}"
BRIDGE="${CORAL_RTL_BRIDGE:-${ROOT_DIR}/build/coralnpu/libcoralnpu_gem5_bridge.so}"
CORALCTL="${ROOT_DIR}/build/guest-tools/coralctl-aarch64"
WEIGHT_PAGE_SOURCE="${CORAL_NPU_WEIGHT_PAGE:-${EXECUTABLE}}"
SHARED_BASE="${CORAL_PAGED_SHARED_BASE:-0x8f000000}"
CKPT_ROOT="${CORAL_CKPT_ROOT:-${ROOT_DIR}/checkpoint/coralnpu_paged_8m_8f000000_ckpt}"
POLL_COUNT="${CORAL_PAGED_POLL_COUNT:-1000000}"

case "$POLL_COUNT" in
    ''|*[!0-9]*)
        echo "error: CORAL_PAGED_POLL_COUNT must be a positive integer" >&2
        exit 1
        ;;
    0)
        echo "error: CORAL_PAGED_POLL_COUNT must be greater than zero" >&2
        exit 1
        ;;
esac

[ -r "$EXECUTABLE" ] || { echo "error: executable missing: $EXECUTABLE" >&2; exit 1; }

rtl_stale=0
if [ ! -r "$FIRMWARE" ] || [ ! -r "$BRIDGE" ]; then
    rtl_stale=1
elif [ -z "${CORAL_RTL_FIRMWARE:-}" ] &&
     find "${ROOT_DIR}/sim/coralnpu" -type f \( -newer "$FIRMWARE" -o -newer "$BRIDGE" \) \
         -print -quit | grep -q .; then
    rtl_stale=1
fi
if [ "$rtl_stale" -eq 1 ] &&
   [ -z "${CORAL_RTL_FIRMWARE:-}" ] &&
   [ -z "${CORAL_RTL_BRIDGE:-}" ] &&
   [ "${CORAL_AUTO_BUILD_FIRMWARE:-1}" = 1 ]; then
    echo "[coral-paged-executable-test] RTL artifacts missing or stale; rebuilding" >&2
    "${ROOT_DIR}/tools/coralnpu/build_rtl_bridge.sh"
fi

if [ ! -r "$FIRMWARE" ]; then
    echo "error: firmware missing: $FIRMWARE" >&2
    echo "build the command processor firmware with:" >&2
    echo "  ./tools/coralnpu/build_rtl_bridge.sh" >&2
    echo "or select an existing image with CORAL_RTL_FIRMWARE=/path/to/firmware.elf" >&2
    exit 1
fi
coralctl_stale=0
if [ ! -x "$CORALCTL" ]; then
    coralctl_stale=1
elif find "${ROOT_DIR}/runtime/host" -type f -newer "$CORALCTL" -print -quit |
     grep -q .; then
    coralctl_stale=1
fi
if [ "$coralctl_stale" -eq 1 ] &&
   [ "${CORAL_AUTO_BUILD_GUEST_TOOLS:-1}" = 1 ]; then
    echo "[coral-paged-executable-test] coralctl missing or stale; rebuilding" >&2
    "${ROOT_DIR}/tools/guest_tools/build_coralctl.sh"
fi
[ -x "$CORALCTL" ] || { echo "error: coralctl missing: $CORALCTL" >&2; exit 1; }
[ -r "$BRIDGE" ] || { echo "error: RTL bridge missing: $BRIDGE" >&2; exit 1; }
[ -r "$WEIGHT_PAGE_SOURCE" ] || { echo "error: weight page source missing: $WEIGHT_PAGE_SOURCE" >&2; exit 1; }

TMP_SCRIPT="$(mktemp)"
TMP_WEIGHT_PAGE="$(mktemp)"
trap 'rm -f "$TMP_SCRIPT" "$TMP_WEIGHT_PAGE"' EXIT
dd if="$WEIGHT_PAGE_SOURCE" of="$TMP_WEIGHT_PAGE" \
    bs=65536 count=1 conv=sync 2>/dev/null

cat >"$TMP_SCRIPT" <<EOF
#!/bin/sh
CORAL_PAGED_POLL_COUNT=$POLL_COUNT
EOF
cat >>"$TMP_SCRIPT" <<'EOF'
mkdir -p /proc /sys /tmp /dev
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
        echo '[coral-paged-executable-test] FAIL: base64 decoder missing' >&2
        return 1
    fi
}
decode_base64 >/tmp/coralctl <<'OPENNPUX_CORALCTL_EOF'
EOF
base64 "$CORALCTL" >>"$TMP_SCRIPT"
cat >>"$TMP_SCRIPT" <<'EOF'
OPENNPUX_CORALCTL_EOF
chmod 0755 /tmp/coralctl
decode_base64 >/tmp/weight-page.bin <<'OPENNPUX_WEIGHT_PAGE_EOF'
EOF
base64 "$TMP_WEIGHT_PAGE" >>"$TMP_SCRIPT"
cat >>"$TMP_SCRIPT" <<'EOF'
OPENNPUX_WEIGHT_PAGE_EOF
decode_base64 >/tmp/model.npxc <<'OPENNPUX_EXECUTABLE_EOF'
EOF
base64 "$EXECUTABLE" >>"$TMP_SCRIPT"
cat >>"$TMP_SCRIPT" <<'EOF'
OPENNPUX_EXECUTABLE_EOF
echo '[coral-paged-executable-test] started'
/tmp/coralctl executable-run-paged /tmp/model.npxc decode /tmp/weight-page.bin 0x1d000000 ${CORAL_PAGED_POLL_COUNT:-1000000} || {
    echo '[coral-paged-executable-test] FAIL: paged executable submission failed'
    m5 --inst exit
    exit 1
}
echo '[coral-paged-executable-test] PASS'
m5 --inst exit
exit 0
EOF

if [ "${CORAL_REBUILD_CKPT:-0}" != 1 ]; then
    echo "note: using checkpoint with NPU shared window ${SHARED_BASE}/8MiB" >&2
fi

"${ROOT_DIR}/sim/gem5/apply_patchset.sh"
cd "$GEM5_ROOT"
CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_BRIDGE="$BRIDGE" \
CORAL_RTL_FIRMWARE="$FIRMWARE" \
CORAL_RTL_CYCLES_PER_EVENT="${CORAL_RTL_CYCLES_PER_EVENT:-1000}" \
CORAL_CKPT_ROOT="$CKPT_ROOT" \
CORAL_CONFIG_OPTIONS="${CORAL_CONFIG_OPTIONS:-} --npu-dma-shared-base=${SHARED_BASE} --npu-dma-shared-size=8MiB" \
CORAL_REBUILD_CKPT="${CORAL_REBUILD_CKPT:-0}" \
CORAL_RESUME_BOOTSCRIPT="$TMP_SCRIPT" \
./run_multicore.sh
