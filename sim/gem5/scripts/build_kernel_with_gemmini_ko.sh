#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: build_kernel_with_gemmini_ko.sh \
  [--kernel-src <path>] \
  [--kernel-out <path>] \
  [--driver-dir <path>] \
  [--rootfs <mount-point>] \
  [--host-dir <path>] \
  [--image-name <file>] \
  [--kernel-dest <file>] \
  [--mount-point <path>] \
  [--arch arm64] \
  [--cross aarch64-linux-gnu-] \
  [--jobs <n>] \
  [--defconfig <target>] \
  [--config <file>] \
  [--kernel-url <url>] \
  [--kernel-tag <tag>] \
  [--install-deps] \
  [--build-user] \
  [--kernel-update] \
  [--kernel-remote <name>] \
  [--kernel-branch <name>]

Examples:
  ./scripts/build_kernel_with_gemmini_ko.sh \
    --kernel-src /path/to/linux \
    --kernel-out /path/to/linux/build \
    --driver-dir /path/to/gem5/tests/test-progs/gemmini-apps/driver \
    --rootfs /mnt/rootfs \
    --defconfig defconfig

Defaults:
  --kernel-url  git@github.com:torvalds/linux.git
  --kernel-tag  v4.18
  --kernel-src  <repo-root>/.tmp/linux-v4.18
  --kernel-out  <kernel-src>/build
  --driver-dir  <repo-root>/tests/test-progs/gemmini-apps/driver
  --defconfig   defconfig
  --image-name  ubuntu-18.04-arm64-docker.img
  --kernel-dest vmlinux.arm64
  --mount-point /mnt/gem5_rootfs
  --build-user  enabled

Notes:
  - Ensure CONFIG_MODULES=y in the kernel .config.
  - --config copies a prepared .config and runs olddefconfig.
  - If --kernel-src does not exist, the script clones --kernel-url at --kernel-tag.
  - If --kernel-update is set, the script runs git fetch/pull in --kernel-src.
  - If --rootfs is set, modules are installed and the Gemmini .ko is copied
    into /root/gemmini inside the rootfs.
  - If --host-dir is set, the script mounts the rootfs image, installs
    modules and the .ko into it, replaces the kernel image, then unmounts.
EOF
}

KERNEL_SRC=""
KERNEL_OUT=""
DRIVER_DIR=""
ROOTFS=""
HOST_DIR="/hdd_8T/barry/gem5_arm_linux_images/"
IMAGE_NAME="ubuntu-18.04-arm64-docker.img"
KERNEL_DEST="vmlinux.arm64"
MOUNT_POINT="/home/barry/mnt/gem5_rootfs"
ARCH="arm64"
CROSS="aarch64-linux-gnu-"
JOBS=""
DEFCONFIG=""
CONFIG_FILE=""
KERNEL_UPDATE="false"
KERNEL_REMOTE="origin"
KERNEL_BRANCH=""
KERNEL_URL="git@github.com:torvalds/linux.git"
KERNEL_TAG="v4.18"
INSTALL_DEPS="false"
BUILD_USER="true"

while [ $# -gt 0 ]; do
    case "$1" in
        --kernel-src) KERNEL_SRC="$2"; shift 2 ;;
        --kernel-out) KERNEL_OUT="$2"; shift 2 ;;
        --driver-dir) DRIVER_DIR="$2"; shift 2 ;;
        --rootfs) ROOTFS="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        --cross) CROSS="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --defconfig) DEFCONFIG="$2"; shift 2 ;;
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --kernel-url) KERNEL_URL="$2"; shift 2 ;;
        --kernel-tag) KERNEL_TAG="$2"; shift 2 ;;
        --host-dir) HOST_DIR="$2"; shift 2 ;;
        --image-name) IMAGE_NAME="$2"; shift 2 ;;
        --kernel-dest) KERNEL_DEST="$2"; shift 2 ;;
        --mount-point) MOUNT_POINT="$2"; shift 2 ;;
        --install-deps) INSTALL_DEPS="true"; shift ;;
        --build-user) BUILD_USER="true"; shift ;;
        --kernel-update) KERNEL_UPDATE="true"; shift ;;
        --kernel-remote) KERNEL_REMOTE="$2"; shift 2 ;;
        --kernel-branch) KERNEL_BRANCH="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1"; usage; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -z "$DEFCONFIG" ]; then
    DEFCONFIG="defconfig"
fi

if [ -z "$KERNEL_SRC" ]; then
    KERNEL_SRC="${REPO_ROOT}/.tmp/linux-${KERNEL_TAG}"
fi
if [ -z "$KERNEL_OUT" ]; then
    KERNEL_OUT="${KERNEL_SRC}/build"
fi
if [ -z "$DRIVER_DIR" ]; then
    DRIVER_DIR="${REPO_ROOT}/tests/test-progs/gemmini-apps/driver"
fi

if [ -n "$CROSS" ] && ! command -v "${CROSS}gcc" >/dev/null 2>&1; then
    echo "warning: cross compiler '${CROSS}gcc' not found in PATH"
fi

if [ -n "$JOBS" ]; then
    MAKE_JOBS="-j$JOBS"
else
    MAKE_JOBS="-j$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi

if [ "$INSTALL_DEPS" = "true" ]; then
    if command -v apt-get >/dev/null 2>&1; then
        SUDO=""
        if command -v sudo >/dev/null 2>&1; then
            SUDO="sudo"
        fi
        $SUDO apt-get update
        $SUDO apt-get install -y \
            build-essential bc bison flex libssl-dev libelf-dev gcc-aarch64-linux-gnu
    else
        echo "warning: --install-deps requested but apt-get not found"
    fi
fi

if [ ! -d "$KERNEL_SRC" ]; then
    mkdir -p "$(dirname "$KERNEL_SRC")"
    if [ -n "$KERNEL_TAG" ]; then
        git clone --depth 1 --branch "$KERNEL_TAG" "$KERNEL_URL" "$KERNEL_SRC"
    else
        git clone "$KERNEL_URL" "$KERNEL_SRC"
    fi
fi

if [ "$KERNEL_UPDATE" = "true" ]; then
    if [ -d "$KERNEL_SRC/.git" ]; then
        git -C "$KERNEL_SRC" fetch "$KERNEL_REMOTE" --tags
        if [ -n "$KERNEL_BRANCH" ]; then
            git -C "$KERNEL_SRC" checkout "$KERNEL_BRANCH"
        elif [ -n "$KERNEL_TAG" ]; then
            git -C "$KERNEL_SRC" checkout "$KERNEL_TAG"
        fi
        git -C "$KERNEL_SRC" pull --ff-only "$KERNEL_REMOTE" ${KERNEL_BRANCH:-}
    else
        echo "warning: --kernel-update set but $KERNEL_SRC is not a git repo"
    fi
fi

if [ -n "$CONFIG_FILE" ]; then
    mkdir -p "$KERNEL_OUT"
    cp "$CONFIG_FILE" "$KERNEL_OUT/.config"
    make -C "$KERNEL_SRC" O="$KERNEL_OUT" ARCH="$ARCH" \
        CROSS_COMPILE="$CROSS" olddefconfig
else
    if [ ! -f "$KERNEL_OUT/.config" ]; then
        make -C "$KERNEL_SRC" O="$KERNEL_OUT" ARCH="$ARCH" \
            CROSS_COMPILE="$CROSS" "$DEFCONFIG"
    fi
fi

make -C "$KERNEL_SRC" O="$KERNEL_OUT" ARCH="$ARCH" \
    CROSS_COMPILE="$CROSS" $MAKE_JOBS Image modules dtbs

make -C "$DRIVER_DIR" KDIR="$KERNEL_OUT" ARCH="$ARCH" CROSS_COMPILE="$CROSS"

if [ "$BUILD_USER" = "true" ]; then
    if command -v "${CROSS}gcc" >/dev/null 2>&1; then
        "${CROSS}gcc" -O2 -Wall -Wextra -o \
            "$DRIVER_DIR/gemmini_dev_a_user" \
            "$DRIVER_DIR/gemmini_dev_a_user.c"
    else
        echo "warning: ${CROSS}gcc not found; skipping user app build"
    fi
fi

SUDO=""
if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
fi

cleanup_mount() {
    if [ -n "${MOUNT_POINT_ACTIVE:-}" ]; then
        $SUDO umount "$MOUNT_POINT_ACTIVE" || true
        MOUNT_POINT_ACTIVE=""
    fi
    if [ -n "${LOOP_DEV:-}" ]; then
        $SUDO losetup -d "$LOOP_DEV" || true
        LOOP_DEV=""
    fi
}

if [ -n "$HOST_DIR" ]; then
    IMG_PATH="${HOST_DIR}/${IMAGE_NAME}"
    if [ ! -f "$IMG_PATH" ]; then
        echo "error: rootfs image not found at $IMG_PATH"
        exit 1
    fi
    if ! command -v losetup >/dev/null 2>&1; then
        echo "error: losetup not found (host-dir mount requires Linux)"
        exit 1
    fi
    mkdir -p "$MOUNT_POINT"
    LOOP_DEV="$($SUDO losetup --find --show -P "$IMG_PATH")"
    PART="${LOOP_DEV}p1"
    if [ -b "$PART" ]; then
        $SUDO mount -o rw "$PART" "$MOUNT_POINT"
    else
        $SUDO mount -o rw "$LOOP_DEV" "$MOUNT_POINT"
    fi
    MOUNT_POINT_ACTIVE="$MOUNT_POINT"
    trap cleanup_mount EXIT
    ROOTFS="$MOUNT_POINT_ACTIVE"
fi

if [ -n "$ROOTFS" ]; then
    if [ -z "$SUDO" ]; then
        echo "warning: no sudo found; rootfs write may fail"
    fi
    $SUDO make -C "$KERNEL_SRC" O="$KERNEL_OUT" ARCH="$ARCH" \
        CROSS_COMPILE="$CROSS" INSTALL_MOD_PATH="$ROOTFS" modules_install

    $SUDO mkdir -p "$ROOTFS/root/gemmini"
    $SUDO cp -f "$DRIVER_DIR/gemmini_dev_a_drv.ko" "$ROOTFS/root/gemmini/"
    if [ -f "$DRIVER_DIR/gemmini_dev_a_user" ]; then
        $SUDO cp -f "$DRIVER_DIR/gemmini_dev_a_user" "$ROOTFS/root/gemmini/"
        $SUDO chmod 0755 "$ROOTFS/root/gemmini/gemmini_dev_a_user"
    fi
fi

if [ -n "$HOST_DIR" ]; then
    KERNEL_SRC_PATH="$KERNEL_OUT/vmlinux"
    if [ ! -f "$KERNEL_SRC_PATH" ]; then
        KERNEL_SRC_PATH="$KERNEL_OUT/arch/arm64/boot/Image"
    fi
    if [ -f "$KERNEL_SRC_PATH" ]; then
        cp -f "$KERNEL_SRC_PATH" "${HOST_DIR}/${KERNEL_DEST}"
    else
        echo "warning: kernel image not found in $KERNEL_OUT"
    fi
fi
