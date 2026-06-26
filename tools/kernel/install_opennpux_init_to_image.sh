#!/bin/sh

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

if [ ! -f "${IMAGE}" ]; then
    echo "error: image not found: ${IMAGE}" >&2
    exit 1
fi
if [ ! -f "${INIT_SCRIPT}" ]; then
    echo "error: init script not found: ${INIT_SCRIPT}" >&2
    exit 1
fi

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
sudo install -D -m 0755 "${INIT_SCRIPT}" "${mnt}/sbin/opennpux-init.sh"
sync

echo "installed init=/sbin/opennpux-init.sh"
