#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
LINUX_BUILD="${LINUX_BUILD:-${ROOT_DIR}/build/linux-arm64}"
ARCH="${ARCH:-arm64}"
CROSS_COMPILE="${CROSS_COMPILE:-aarch64-linux-gnu-}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

if [ ! -f "${LINUX_BUILD}/include/config/kernel.release" ]; then
    echo "error: kernel build tree not prepared: ${LINUX_BUILD}" >&2
    echo "hint: run tools/kernel/build_arm64_kernel.sh first or set LINUX_BUILD" >&2
    exit 1
fi

make -C "${LINUX_BUILD}" M="${ROOT_DIR}/runtime/kernel" \
    OPENNPUX_ROOT="${ROOT_DIR}" \
    ARCH="${ARCH}" CROSS_COMPILE="${CROSS_COMPILE}" -j"${JOBS}" modules

kernel_release="$(cat "${LINUX_BUILD}/include/config/kernel.release")"
mkdir -p "${ROOT_DIR}/build/kernel"
cp "${ROOT_DIR}/runtime/kernel/opennpux_coral.ko" \
   "${ROOT_DIR}/build/kernel/opennpux_coral-${kernel_release}.ko"
echo "kernel_release=${kernel_release}"
echo "module=${ROOT_DIR}/build/kernel/opennpux_coral-${kernel_release}.ko"
