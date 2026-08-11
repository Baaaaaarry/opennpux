#!/bin/sh
#
# install_kernel_to_image.sh — Install a compiled arm64 kernel into a gem5 disk image.
#
# Mounts the Ubuntu ARM64 disk image (with automatic partition-offset detection),
# installs the kernel Image, out-of-tree modules, and the opennpux-coral driver,
# then unmounts.  The image is modified in-place; gem5 will use the installed
# kernel on next boot.
#
# Installed paths inside the image:
#   /boot/Image-<release>                  kernel image
#   /boot/Image → Image-<release>           convenience symlink
#   /lib/modules/<release>/                 in-tree modules (modules_install)
#   /lib/modules/<release>/extra/opennpux_coral.ko   NPU driver
#   /etc/modules-load.d/opennpux-coral.conf          auto-load on boot
#
# Usage:
#   sudo $0 <disk-image> <kernel-build-dir> [kernel-image] [opennpux-coral.ko]
#
# The optional [kernel-image] and [opennpux-coral.ko] arguments override the
# default paths under build/linux-arm64 and build/kernel/ respectively.
#
# @kernel-install-spec  v1  2025-07-28

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

usage() {
    echo "usage: $0 <disk-image> <kernel-build-dir> [kernel-image] [opennpux-coral.ko]" >&2
}

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    usage
    exit 2
fi

IMAGE="$1"
LINUX_BUILD="$2"
LINUX_SRC="${LINUX_SRC:-${ROOT_DIR}/.cache/linux-src}"
KERNEL_IMAGE="${3:-${LINUX_BUILD}/arch/arm64/boot/Image}"
KO="${4:-}"

# ---------------------------------------------------------------------------
# Preflight: verify required inputs exist.
# ---------------------------------------------------------------------------
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
if [ ! -f "${LINUX_SRC}/Makefile" ]; then
    echo "error: kernel source tree not found: ${LINUX_SRC}" >&2
    echo "hint: set LINUX_SRC to the source tree used by this build" >&2
    exit 1
fi

kernel_release="$(cat "${LINUX_BUILD}/include/config/kernel.release")"

# ---------------------------------------------------------------------------
# Resolve the driver .ko path if not explicitly provided.
# ---------------------------------------------------------------------------
if [ -z "${KO}" ]; then
    candidate="${ROOT_DIR}/build/kernel/opennpux_coral-${kernel_release}.ko"
    if [ -f "${candidate}" ]; then
        KO="${candidate}"
    fi
fi

# ---------------------------------------------------------------------------
# Detect the first partition's start sector.
#   Tries partx (util-linux) first, then fdisk, then falls back to sector 2048
#   (the conventional start for EXT4 images created by gem5 / qemu-img).
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
        echo "hint: pass an already partition-mounted image manually or fix fdisk/partx output" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Mount the image's first partition via loopback offset.
#   offset = start_sector × 512 (standard sector size).
#   A cleanup trap ensures the mount and any stray loop device are released
#   even if an installation step fails.
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

# ---------------------------------------------------------------------------
# Install: kernel image, modules, driver module, and auto-load configuration.
# ---------------------------------------------------------------------------
sudo install -D -m 0644 "${KERNEL_IMAGE}" "${mnt}/boot/Image-${kernel_release}"
sudo ln -sf "Image-${kernel_release}" "${mnt}/boot/Image"

sudo make -C "${LINUX_SRC}" O="${LINUX_BUILD}" \
    INSTALL_MOD_PATH="${mnt}" modules_install

if [ -n "${KO}" ]; then
    if [ ! -f "${KO}" ]; then
        echo "error: driver module not found: ${KO}" >&2
        exit 1
    fi
    sudo install -D -m 0644 "${KO}" \
        "${mnt}/lib/modules/${kernel_release}/extra/opennpux_coral.ko"
    sudo mkdir -p "${mnt}/etc/modules-load.d"
    printf '%s\n' opennpux_coral | \
        sudo tee "${mnt}/etc/modules-load.d/opennpux-coral.conf" >/dev/null
fi

# ---------------------------------------------------------------------------
# Regenerate module dependencies so the guest kernel can resolve symbols.
#   depmod may fail if the host and guest kernel versions differ widely;
#   this is usually harmless for our single out-of-tree module.
# ---------------------------------------------------------------------------
sudo depmod -b "${mnt}" "${kernel_release}" 2>/dev/null || true
sync

echo "installed kernel_release=${kernel_release}"
echo "installed image boot path=/boot/Image-${kernel_release}"
if [ -n "${KO}" ]; then
    echo "installed module=/lib/modules/${kernel_release}/extra/opennpux_coral.ko"
fi
