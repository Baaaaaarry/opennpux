#!/bin/sh

set -eu

usage() {
    echo "usage: $0 <disk-image> <kernel-build-dir> [kernel-image] [opennpux-coral.ko]" >&2
}

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    usage
    exit 2
fi

IMAGE="$1"
LINUX_BUILD="$2"
KERNEL_IMAGE="${3:-${LINUX_BUILD}/arch/arm64/boot/Image}"
KO="${4:-}"

if [ ! -f "${IMAGE}" ]; then
    echo "error: image not found: ${IMAGE}" >&2
    exit 1
fi
if [ ! -f "${KERNEL_IMAGE}" ]; then
    echo "error: kernel image not found: ${KERNEL_IMAGE}" >&2
    exit 1
fi
if [ ! -f "${LINUX_BUILD}/include/config/kernel.release" ]; then
    echo "error: kernel release not found in ${LINUX_BUILD}" >&2
    exit 1
fi

kernel_release="$(cat "${LINUX_BUILD}/include/config/kernel.release")"
if [ -z "${KO}" ]; then
    candidate="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)/build/kernel/opennpux_coral-${kernel_release}.ko"
    if [ -f "${candidate}" ]; then
        KO="${candidate}"
    fi
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
        echo "hint: pass an already partition-mounted image manually or fix fdisk/partx output" >&2
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
sudo install -D -m 0644 "${KERNEL_IMAGE}" "${mnt}/boot/Image-${kernel_release}"
sudo ln -sf "Image-${kernel_release}" "${mnt}/boot/Image"

sudo make -C "${LINUX_BUILD}" INSTALL_MOD_PATH="${mnt}" modules_install
if [ -n "${KO}" ]; then
    if [ ! -f "${KO}" ]; then
        echo "error: ko not found: ${KO}" >&2
        exit 1
    fi
    sudo install -D -m 0644 "${KO}" \
        "${mnt}/lib/modules/${kernel_release}/extra/opennpux_coral.ko"
    sudo mkdir -p "${mnt}/etc/modules-load.d"
    printf '%s\n' opennpux_coral | \
        sudo tee "${mnt}/etc/modules-load.d/opennpux-coral.conf" >/dev/null
fi
sudo depmod -b "${mnt}" "${kernel_release}" 2>/dev/null || true
sync

echo "installed kernel_release=${kernel_release}"
echo "installed image boot path=/boot/Image-${kernel_release}"
if [ -n "${KO}" ]; then
    echo "installed module=/lib/modules/${kernel_release}/extra/opennpux_coral.ko"
fi
