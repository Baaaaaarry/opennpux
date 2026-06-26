#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
LINUX_SRC="${LINUX_SRC:-${ROOT_DIR}/.cache/linux-src}"
LINUX_BUILD="${LINUX_BUILD:-${ROOT_DIR}/build/linux-arm64}"
LINUX_BRANCH="${LINUX_BRANCH:-linux-4.19.y}"
LINUX_REPO="${LINUX_REPO:-https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git}"
ARCH="${ARCH:-arm64}"
CROSS_COMPILE="${CROSS_COMPILE:-aarch64-linux-gnu-}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
LOCALVERSION="${LOCALVERSION:--opennpux}"
KERNEL_BASE_CONFIG="${KERNEL_BASE_CONFIG:-}"

if ! command -v git >/dev/null 2>&1; then
    echo "error: git not found" >&2
    exit 1
fi
if ! command -v "${CROSS_COMPILE}gcc" >/dev/null 2>&1; then
    echo "error: ${CROSS_COMPILE}gcc not found; install gcc-aarch64-linux-gnu" >&2
    exit 1
fi

mkdir -p "$(dirname "${LINUX_SRC}")" "${LINUX_BUILD}"
if [ ! -d "${LINUX_SRC}/.git" ]; then
    git clone --depth 1 --branch "${LINUX_BRANCH}" "${LINUX_REPO}" "${LINUX_SRC}"
else
    git -C "${LINUX_SRC}" fetch --depth 1 origin "${LINUX_BRANCH}"
    git -C "${LINUX_SRC}" checkout FETCH_HEAD
fi

if [ -n "${KERNEL_BASE_CONFIG}" ]; then
    if [ ! -f "${KERNEL_BASE_CONFIG}" ]; then
        echo "error: KERNEL_BASE_CONFIG not found: ${KERNEL_BASE_CONFIG}" >&2
        exit 1
    fi
    cp "${KERNEL_BASE_CONFIG}" "${LINUX_BUILD}/.config"
    echo "using base config: ${KERNEL_BASE_CONFIG}"
    : "${OPENNPUX_PRESERVE_CONFIG_CMDLINE:=1}"
else
    make -C "${LINUX_SRC}" O="${LINUX_BUILD}" ARCH="${ARCH}" \
        CROSS_COMPILE="${CROSS_COMPILE}" defconfig
    : "${OPENNPUX_PRESERVE_CONFIG_CMDLINE:=0}"
fi

OPENNPUX_PRESERVE_CONFIG_CMDLINE="${OPENNPUX_PRESERVE_CONFIG_CMDLINE}" \
    "${SCRIPT_DIR}/configure_arm64_gem5_kernel.sh" "${LINUX_BUILD}/.config"

make -C "${LINUX_SRC}" O="${LINUX_BUILD}" ARCH="${ARCH}" \
    CROSS_COMPILE="${CROSS_COMPILE}" LOCALVERSION="${LOCALVERSION}" \
    olddefconfig
make -C "${LINUX_SRC}" O="${LINUX_BUILD}" ARCH="${ARCH}" \
    CROSS_COMPILE="${CROSS_COMPILE}" LOCALVERSION="${LOCALVERSION}" \
    -j"${JOBS}" Image modules

kernel_release="$(make -s -C "${LINUX_SRC}" O="${LINUX_BUILD}" ARCH="${ARCH}" \
    CROSS_COMPILE="${CROSS_COMPILE}" LOCALVERSION="${LOCALVERSION}" kernelrelease)"
mkdir -p "${ROOT_DIR}/build/kernel"
cp "${LINUX_BUILD}/arch/arm64/boot/Image" \
   "${ROOT_DIR}/build/kernel/Image-${kernel_release}"
cp "${LINUX_BUILD}/vmlinux" "${ROOT_DIR}/build/kernel/vmlinux-${kernel_release}"
printf '%s\n' "${kernel_release}" > "${ROOT_DIR}/build/kernel/kernel.release"

echo "kernel_release=${kernel_release}"
echo "guest_boot_image=${ROOT_DIR}/build/kernel/Image-${kernel_release}"
echo "gem5_kernel=${ROOT_DIR}/build/kernel/vmlinux-${kernel_release}"
echo "kernel_build=${LINUX_BUILD}"
