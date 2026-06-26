#!/bin/sh

set -eu

usage() {
    echo "usage: $0 <disk-image> <aarch64-busybox-or-insmod>" >&2
}

if [ "$#" -ne 2 ]; then
    usage
    exit 2
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
IMAGE="$1"
LOADER="$2"

if [ ! -f "${IMAGE}" ]; then
    echo "error: image not found: ${IMAGE}" >&2
    exit 1
fi
if [ ! -x "${LOADER}" ]; then
    echo "error: loader not found or not executable: ${LOADER}" >&2
    echo "hint: pass an aarch64 busybox binary or an aarch64 insmod binary" >&2
    exit 1
fi

"${ROOT_DIR}/tools/kernel/install_opennpux_init_to_image.sh" "${IMAGE}" >/dev/null

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
sudo install -D -m 0755 "${LOADER}" "${mnt}/sbin/insmod"

name="$(basename "${LOADER}")"
case "${name}" in
    busybox*)
        sudo install -D -m 0755 "${LOADER}" "${mnt}/bin/busybox"
        sudo ln -sf /bin/busybox "${mnt}/sbin/insmod"
        sudo ln -sf /bin/busybox "${mnt}/sbin/modprobe"
        ;;
esac
sync

echo "installed module loader into ${IMAGE}"
echo "guest paths: /sbin/insmod"
