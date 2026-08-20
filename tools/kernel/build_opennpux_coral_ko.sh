#!/bin/sh
#
# build_opennpux_coral_ko.sh — Build the opennpux-coral kernel driver module.
#
# Compiles the out-of-tree Linux kernel module at runtime/kernel/ against
# a previously built arm64 kernel tree.  The kernel tree must already exist
# (build_arm64_kernel.sh produces it).  The resulting .ko is staged into
# build/kernel/ for installation into the guest disk image.
#
# Pipeline:
#   build_arm64_kernel.sh           ← must run first (creates build/linux-arm64)
#     │
#     └── build_opennpux_coral_ko.sh
#           → make -C build/linux-arm64 M=runtime/kernel modules
#           → cp opennpux_coral.ko → build/kernel/opennpux_coral-<release>.ko
#
# Output:
#   build/kernel/opennpux_coral-<kernel_release>.ko   driver module for guest
#
# Environment:
#   LINUX_SRC        kernel source tree (default: .cache/linux-src)
#   LINUX_BUILD      kernel build tree (default: build/linux-arm64)
#   ARCH             target architecture (default: arm64)
#   CROSS_COMPILE    cross-compiler prefix (default: aarch64-linux-gnu-)
#   JOBS             parallel build jobs (default: nproc)
#
# @kernel-build-spec  v1  2025-07-28
# @synchronized-with  tools/kernel/build_arm64_kernel.sh

set -eu

# ---------------------------------------------------------------------------
# Resolve paths relative to this script's location in the project tree.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

# ---------------------------------------------------------------------------
# Build parameters — all overridable via environment.
# ---------------------------------------------------------------------------
LINUX_SRC="${LINUX_SRC:-${ROOT_DIR}/.cache/linux-src}"
LINUX_BUILD="${LINUX_BUILD:-${ROOT_DIR}/build/linux-arm64}"
ARCH="${ARCH:-arm64}"
CROSS_COMPILE="${CROSS_COMPILE:-aarch64-linux-gnu-}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

# ---------------------------------------------------------------------------
# Preflight: the kernel build tree must exist.
#   build_arm64_kernel.sh creates build/linux-arm64/include/config/kernel.release
#   as part of its normal output.  Its absence means the kernel was never built
#   or the build tree was deleted.
# ---------------------------------------------------------------------------
if [ ! -f "${LINUX_BUILD}/include/config/kernel.release" ]; then
    echo "error: kernel build tree not prepared: ${LINUX_BUILD}" >&2
    echo "hint: run tools/kernel/build_arm64_kernel.sh first or set LINUX_BUILD" >&2
    exit 1
fi
if [ ! -f "${LINUX_SRC}/Makefile" ]; then
    echo "error: kernel source tree not found: ${LINUX_SRC}" >&2
    echo "hint: run tools/kernel/build_arm64_kernel.sh first or set LINUX_SRC" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Build the out-of-tree module.
#   The kernel Makefile system compiles runtime/kernel/opennpux_coral.c against
#   the headers and Module.symvers of the previously built kernel.  OPENNPUX_ROOT
#   is passed so the module Makefile can find the UAPI header at
#   runtime/host/include/opennpux/coral_uapi.h.
# ---------------------------------------------------------------------------
# Module .cmd files live in runtime/kernel, not under the selected kernel
# output directory. Clean them so switching kernel trees or ARM64 atomic/OF
# configuration cannot reuse an incompatible object file.
make -C "${LINUX_SRC}" O="${LINUX_BUILD}" \
    M="${ROOT_DIR}/runtime/kernel" \
    OPENNPUX_ROOT="${ROOT_DIR}" \
    ARCH="${ARCH}" CROSS_COMPILE="${CROSS_COMPILE}" clean
make -C "${LINUX_SRC}" O="${LINUX_BUILD}" \
    M="${ROOT_DIR}/runtime/kernel" \
    OPENNPUX_ROOT="${ROOT_DIR}" \
    ARCH="${ARCH}" CROSS_COMPILE="${CROSS_COMPILE}" -j"${JOBS}" modules

# ---------------------------------------------------------------------------
# Step 2: Collect the kernel release string from the build tree.
#   This must match the kernel that will boot in gem5.  The release string
#   includes the LOCALVERSION suffix (e.g. "4.19.325-opennpux").
# ---------------------------------------------------------------------------
kernel_release="$(cat "${LINUX_BUILD}/include/config/kernel.release")"

# ---------------------------------------------------------------------------
# Step 3: Stage the module into build/kernel/ with the release-qualified name.
#   install_kernel_to_image.sh expects the .ko at this path.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Step 4: Verify the module was produced, then stage it.
# ---------------------------------------------------------------------------
MODULE_SRC="${ROOT_DIR}/runtime/kernel/opennpux_coral.ko"
if [ ! -f "${MODULE_SRC}" ]; then
    echo "error: module build completed but ${MODULE_SRC} was not produced" >&2
    exit 1
fi
MODULE_DST="${ROOT_DIR}/build/kernel/opennpux_coral-${kernel_release}.ko"
mkdir -p "${ROOT_DIR}/build/kernel"
cp "${MODULE_SRC}" "${MODULE_DST}"
echo "kernel_release=${kernel_release}"
echo "module=${MODULE_DST}"
