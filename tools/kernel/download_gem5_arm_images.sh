#!/bin/sh
#
# download_gem5_arm_images.sh — Download gem5 ARM64 kernel + disk image.
#
# Fetches the pre-built ARM full-system binaries and Ubuntu 18.04 disk image
# from the official gem5 distribution mirror.  These are the minimum assets
# needed to boot a gem5 ARM full-system simulation before any custom kernel
# or Coral NPU work.
#
# The assets are placed under IMAGE_PATH (default: $HOME/wlk/gem5_arm_linux_images)
# where run_multicore.sh expects to find them.
#
# Output (under IMAGE_PATH):
#   binaries/vmlinux.arm64                   default ARM kernel ELF
#   ubuntu-18.04-arm64-docker.img            root filesystem disk image
#
# Environment:
#   IMAGE_PATH          destination directory (default: $HOME/wlk/gem5_arm_linux_images)
#   GEM5_DIST_YEAR      gem5 release year for download URL (default: 2022)
#   GEM5_DIST_MONTH     gem5 release month (default: 07)
#
# @bootstrap-assets-spec  v1  2025-07-28

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

IMAGE_PATH="${IMAGE_PATH:-$HOME/wlk/gem5_arm_linux_images}"
GEM5_DIST_YEAR="${GEM5_DIST_YEAR:-2022}"
GEM5_DIST_MONTH="${GEM5_DIST_MONTH:-07}"
GEM5_DIST_BASE="http://dist.gem5.org/dist/v22-0/arm"

# Filenames (stable — these have not changed across gem5 22.x releases).
SYSTEM_TARBALL="aarch-system-${GEM5_DIST_YEAR}${GEM5_DIST_MONTH}${GEM5_DIST_MONTH}.tar.bz2"
DISK_IMAGE="ubuntu-18.04-arm64-docker.img"
DISK_ARCHIVE="${DISK_IMAGE}.bz2"

mkdir -p "${IMAGE_PATH}"

# ---------------------------------------------------------------------------
# Step 1: Download and extract ARM system binaries (vmlinux.arm64 + bootloader).
# ---------------------------------------------------------------------------
if [ -f "${IMAGE_PATH}/binaries/vmlinux.arm64" ]; then
    echo "[download] binaries already present: ${IMAGE_PATH}/binaries/vmlinux.arm64"
else
    cd "${IMAGE_PATH}"
    if [ ! -f "${SYSTEM_TARBALL}" ]; then
        echo "[download] fetching ${GEM5_DIST_BASE}/${SYSTEM_TARBALL}"
        if command -v wget >/dev/null 2>&1; then
            wget "${GEM5_DIST_BASE}/${SYSTEM_TARBALL}"
        elif command -v curl >/dev/null 2>&1; then
            curl -O "${GEM5_DIST_BASE}/${SYSTEM_TARBALL}"
        else
            echo "error: install wget or curl" >&2
            exit 1
        fi
    fi
    echo "[download] extracting ${SYSTEM_TARBALL}"
    tar -xvjf "${SYSTEM_TARBALL}"
    rm -f "${SYSTEM_TARBALL}"
    echo "[download] binaries ready"
fi

# ---------------------------------------------------------------------------
# Step 2: Download and decompress the Ubuntu 18.04 ARM64 disk image.
# ---------------------------------------------------------------------------
if [ -f "${IMAGE_PATH}/${DISK_IMAGE}" ]; then
    echo "[download] disk image already present: ${IMAGE_PATH}/${DISK_IMAGE}"
else
    cd "${IMAGE_PATH}"
    if [ ! -f "${DISK_ARCHIVE}" ]; then
        echo "[download] fetching ${GEM5_DIST_BASE}/disks/${DISK_ARCHIVE}"
        if command -v wget >/dev/null 2>&1; then
            wget "${GEM5_DIST_BASE}/disks/${DISK_ARCHIVE}"
        elif command -v curl >/dev/null 2>&1; then
            curl -O "${GEM5_DIST_BASE}/disks/${DISK_ARCHIVE}"
        else
            echo "error: install wget or curl" >&2
            exit 1
        fi
    fi
    echo "[download] decompressing ${DISK_ARCHIVE}"
    bunzip2 "${DISK_ARCHIVE}"
    echo "[download] disk image ready"
fi

# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------
echo
echo "Assets installed under ${IMAGE_PATH}:"
echo
ls -lh "${IMAGE_PATH}/binaries/vmlinux.arm64" 2>/dev/null || \
    echo "  binaries/vmlinux.arm64  — MISSING"
ls -lh "${IMAGE_PATH}/${DISK_IMAGE}" 2>/dev/null || \
    echo "  ${DISK_IMAGE}  — MISSING"
echo
echo "Export IMAGE_PATH for subsequent steps:"
echo "  export IMAGE_PATH=\"${IMAGE_PATH}\""
