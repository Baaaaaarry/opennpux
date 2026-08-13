#!/bin/sh
set -eu
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
GEM5_ROOT="${ROOT_DIR}/thirdparty/gem5"
EXECUTABLE="${CORAL_NPU_EXECUTABLE:-${ROOT_DIR}/build/local-tests/model-package/hf-model/model.npxc}"
FIRMWARE="${CORAL_RTL_FIRMWARE:-${ROOT_DIR}/build/coralnpu/gem5_npu_command_processor_smoke.elf}"
CORALCTL="${ROOT_DIR}/build/guest-tools/coralctl-aarch64"

[ -r "$EXECUTABLE" ] || { echo "error: executable missing: $EXECUTABLE" >&2; exit 1; }
[ -r "$FIRMWARE" ] || { echo "error: firmware missing: $FIRMWARE" >&2; exit 1; }
[ -x "$CORALCTL" ] || { echo "error: coralctl missing: $CORALCTL" >&2; exit 1; }

if [ "${CORAL_REBUILD_CKPT:-0}" != 1 ]; then
    cat >&2 <<EOF
note: this test requires a checkpoint whose DT reserves a 64KiB NPU shared window.
      Set CORAL_REBUILD_CKPT=1 for the first run after this change.
EOF
fi

TMP_SCRIPT="$(mktemp)"
trap 'rm -f "$TMP_SCRIPT"' EXIT
cat >"$TMP_SCRIPT" <<EOF
#!/bin/sh
mkdir -p /proc /sys /tmp /dev
mount -t proc proc /proc 2>/dev/null || true
mount -t sysfs sysfs /sys 2>/dev/null || true
mount -t devtmpfs devtmpfs /dev 2>/dev/null || true
if command -v base64 >/dev/null 2>&1; then
    BASE64_DECODE='base64 -d'
elif [ -x /bin/busybox ]; then
    BASE64_DECODE='/bin/busybox base64 -d'
elif [ -x /tmp/busybox ]; then
    BASE64_DECODE='/tmp/busybox base64 -d'
else
    echo '[coral-executable-run-test] FAIL: base64 decoder missing'
    m5 --inst exit
    exit 1
fi
\$BASE64_DECODE >/tmp/coralctl <<'OPENNPUX_CORALCTL_EOF'
EOF
base64 "$CORALCTL" >>"$TMP_SCRIPT"
cat >>"$TMP_SCRIPT" <<'EOF'
OPENNPUX_CORALCTL_EOF
chmod 0755 /tmp/coralctl
CORALCTL=/tmp/coralctl
\$BASE64_DECODE >/tmp/model.npxc <<'OPENNPUX_EXECUTABLE_EOF'
EOF
base64 "$EXECUTABLE" >>"$TMP_SCRIPT"
cat >>"$TMP_SCRIPT" <<'EOF'
OPENNPUX_EXECUTABLE_EOF
echo '[coral-executable-run-test] started'
"$CORALCTL" executable-run /tmp/model.npxc decode || {
    echo '[coral-executable-run-test] FAIL: generic executable submission failed'
    m5 --inst exit
    exit 1
}
echo '[coral-executable-run-test] PASS'
m5 --inst exit
exit 0
EOF

"${ROOT_DIR}/sim/gem5/apply_patchset.sh"
cd "$GEM5_ROOT"
CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_FIRMWARE="$FIRMWARE" \
CORAL_CONFIG_OPTIONS="${CORAL_CONFIG_OPTIONS:-} --npu-dma-shared-size=64KiB" \
CORAL_REBUILD_CKPT="${CORAL_REBUILD_CKPT:-0}" \
CORAL_RESUME_BOOTSCRIPT="$TMP_SCRIPT" \
./run_multicore.sh
