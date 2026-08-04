#!/bin/sh
#
# install_opennpux_init_to_image.sh — Install the OpenNPUX init script into a gem5
#                                     disk image.
#
# Replaces /sbin/opennpux-init.sh inside the guest image.  gem5 passes
# init=/sbin/opennpux-init.sh on the kernel command line, so this script is
# the first userspace process after boot.  It loads the opennpux-coral driver
# module, mounts necessary pseudo-filesystems, and launches the test payload.
#
# Installed path inside the image:
#   /sbin/opennpux-init.sh              boot init script
#
# Usage:
#   sudo $0 <disk-image> [init-script]
#
# The optional [init-script] argument overrides the default
# runtime/host/init/opennpux-init.sh.
#
# @kernel-install-spec  v1  2025-07-28

set -eu

usage() {
    echo "usage: $0 <disk-image> [init-script]" >&2
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
    exit 2
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
IMAGE="$1"
INIT_SCRIPT="${2:-${ROOT_DIR}/runtime/host/init/opennpux-init.sh}"

# ---------------------------------------------------------------------------
# Preflight: verify inputs exist.
# ---------------------------------------------------------------------------
if [ ! -f "${IMAGE}" ]; then
    echo "error: image not found: ${IMAGE}" >&2
    exit 1
fi
if [ ! -f "${INIT_SCRIPT}" ]; then
    echo "error: init script not found: ${INIT_SCRIPT}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Detect the first partition's start sector (same logic as install_kernel_to_image.sh).
# ---------------------------------------------------------------------------
start_sector=""
if command -v partx >/dev/null 2>&1; then
    start_sector="$(partx -g -o START "${IMAGE}" 2>/dev/null | awk '
        $1 ~ /^[0-9]+$/ { print $1; exit }
    ')"
fi
if [ -z "${start_sector}" ]; then
    start_sector="$(LC_ALL=C fdisk -l "${IMAGE}" 2>/dev/null | awk '
        $1 ~ /[0-9]$/ || $1 ~ /p[0-9]$/ {
            for (i = 2; i <= NF; ++i) {
                if ($i ~ /^[0-9]+$/) {
                    print $i
                    exit
                }
            }
        }')"
fi
if [ -z "${start_sector}" ]; then
    start_sector=2048
    echo "warning: could not detect first partition start; assuming sector ${start_sector}" >&2
fi
case "${start_sector}" in
    *[!0-9]*)
        echo "error: invalid partition start sector: ${start_sector}" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Mount via loopback offset and install.
# ---------------------------------------------------------------------------
offset=$((start_sector * 512))
mnt="$(mktemp -d)"
cleanup() {
    if mount | grep -q " on ${mnt} "; then
        sudo umount "${mnt}"
    fi
    rmdir "${mnt}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sudo mount -o loop,offset="${offset}" "${IMAGE}" "${mnt}"
sudo install -D -m 0755 "${INIT_SCRIPT}" "${mnt}/sbin/opennpux-init.sh"
sync

echo "installed init=/sbin/opennpux-init.sh"
