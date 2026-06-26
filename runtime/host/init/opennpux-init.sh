#!/bin/sh

# Minimal PID 1 for gem5/OpenNPUX bring-up. It deliberately avoids systemd and
# the legacy image-specific gem5 init script so rebuilt kernels can execute the
# current gem5 readfile script through a stable path.

mkdir -p /proc /sys /sys/kernel/debug /tmp /dev /data
mount -t proc proc /proc 2>/dev/null || true
mount -t sysfs sysfs /sys 2>/dev/null || true
mount -t debugfs debugfs /sys/kernel/debug 2>/dev/null || true
mount -t tmpfs tmpfs /tmp 2>/dev/null || true
mount -t devtmpfs devtmpfs /dev 2>/dev/null || true

echo "[opennpux-init] started"
echo "[opennpux-init] kernel=$(uname -r)"

for dev in /dev/vda1 /dev/vda /dev/sda1 /dev/sda; do
    if [ -b "$dev" ] && ! mountpoint -q /data 2>/dev/null; then
        mount "$dev" /data 2>/dev/null || true
    fi
done

READFILE=/tmp/opennpux-readfile.rcS
rm -f "$READFILE"
if [ -x /sbin/m5 ]; then
    /sbin/m5 --inst readfile > "$READFILE" 2>/tmp/opennpux-m5-readfile.err || \
        /sbin/m5 readfile > "$READFILE" 2>>/tmp/opennpux-m5-readfile.err || true
elif command -v m5 >/dev/null 2>&1; then
    m5 --inst readfile > "$READFILE" 2>/tmp/opennpux-m5-readfile.err || \
        m5 readfile > "$READFILE" 2>>/tmp/opennpux-m5-readfile.err || true
else
    echo "[opennpux-init] m5 tool not found"
fi

if [ -s "$READFILE" ]; then
    chmod 0700 "$READFILE"
    echo "[opennpux-init] executing gem5 readfile script"
    /bin/sh "$READFILE"
    rc="$?"
    echo "[opennpux-init] readfile script exited rc=$rc"
else
    echo "[opennpux-init] no readfile script available"
    if [ -s /tmp/opennpux-m5-readfile.err ]; then
        cat /tmp/opennpux-m5-readfile.err
    fi
fi

echo "[opennpux-init] dropping to emergency shell"
exec /bin/sh
