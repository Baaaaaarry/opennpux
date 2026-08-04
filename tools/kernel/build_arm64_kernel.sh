#!/bin/sh
#
# build_arm64_kernel.sh — Build an arm64 Linux kernel for gem5 full-system simulation.
#
# This script clones (or updates) a pinned Linux stable branch, applies the
# gem5-required config overlay via configure_arm64_gem5_kernel.sh, and builds
# the kernel Image and modules with an aarch64 cross-compiler.
#
# Pipeline:
#   1. Clone/fetch linux-stable (default: linux-4.19.y)
#   2. Seed .config from KERNEL_BASE_CONFIG, or defconfig with a warning
#   3. Run configure_arm64_gem5_kernel.sh to set/unset gem5-specific options
#   4. Run olddefconfig to resolve any new symbols
#   5. Build Image + modules
#   6. Stage artifacts into build/kernel/
#
# Output artifacts (under build/):
#   build/kernel/Image-<release>       guest kernel image (for /boot)
#   build/kernel/vmlinux-<release>     ELF kernel (for gem5 --kernel)
#   build/kernel/kernel.release        single-line release string
#   build/linux-arm64/                 kernel build tree (for out-of-tree modules)
#
# Environment variables:
#   KERNEL_BASE_CONFIG    path to seed .config (default: tools/kernel/gem5-4.18.config)
#                         set to "" to force arm64 defconfig
#   LINUX_BRANCH          git branch to build (default: linux-4.19.y)
#   LINUX_SRC             kernel source location (default: .cache/linux-src)
#   LINUX_BUILD           O= build directory (default: build/linux-arm64)
#   CROSS_COMPILE         cross-compiler prefix (default: aarch64-linux-gnu-)
#   LOCALVERSION          kernel LOCALVERSION suffix (default: -opennpux)
#   JOBS                  parallel build jobs (default: nproc)
#
# @kernel-build-spec  v1  2025-07-28
# @synchronized-with  tools/kernel/configure_arm64_gem5_kernel.sh
# @synchronized-with  tools/kernel/check_gem5_kernel_config.sh

set -eu

# ---------------------------------------------------------------------------
# Resolve paths relative to this script's location in the project tree.
# SCRIPT_DIR = tools/kernel/,  ROOT_DIR = superproject root.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

# ---------------------------------------------------------------------------
# Build parameters — all overridable via environment.
# ---------------------------------------------------------------------------
LINUX_SRC="${LINUX_SRC:-${ROOT_DIR}/.cache/linux-src}"
LINUX_BUILD="${LINUX_BUILD:-${ROOT_DIR}/build/linux-arm64}"
LINUX_BRANCH="${LINUX_BRANCH:-linux-4.19.y}"
LINUX_REPO="${LINUX_REPO:-https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git}"
ARCH="${ARCH:-arm64}"
CROSS_COMPILE="${CROSS_COMPILE:-aarch64-linux-gnu-}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
LOCALVERSION="${LOCALVERSION:--opennpux}"

# Default: the known-good gem5 4.18 config bundled in tools/kernel/.
# Set KERNEL_BASE_CONFIG="" to force arm64 defconfig instead (may not boot in gem5).
KERNEL_BASE_CONFIG="${KERNEL_BASE_CONFIG:-${ROOT_DIR}/tools/kernel/gem5-4.18.config}"

# ---------------------------------------------------------------------------
# Preflight: verify required host tools are available.
# ---------------------------------------------------------------------------
if ! command -v git >/dev/null 2>&1; then
    echo "error: git not found" >&2
    exit 1
fi
if ! command -v "${CROSS_COMPILE}gcc" >/dev/null 2>&1; then
    echo "error: ${CROSS_COMPILE}gcc not found; install gcc-aarch64-linux-gnu" >&2
    exit 1
fi
for tool in make bc bison flex; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "error: ${tool} not found; install build-essential bison flex" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Step 1: Clone or update the Linux kernel source tree.
#   Shallow clone (--depth 1) saves ~2 GB of transfer and disk.
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "${LINUX_SRC}")" "${LINUX_BUILD}"
if [ ! -d "${LINUX_SRC}/.git" ]; then
    git clone --depth 1 --branch "${LINUX_BRANCH}" "${LINUX_REPO}" "${LINUX_SRC}"
else
    git -C "${LINUX_SRC}" fetch --depth 1 origin "${LINUX_BRANCH}"
    git -C "${LINUX_SRC}" checkout FETCH_HEAD
fi

# ---------------------------------------------------------------------------
# Step 2: Seed the kernel .config.
#
#   Two paths:
#   (A) KERNEL_BASE_CONFIG is set (default):
#         Copy the known-good config into the build tree.  The bundled
#         gem5-4.18.config was extracted from a booting gem5 guest and is
#         validated to reach the PL011 console with linux-4.19.y.
#         OPENNPUX_PRESERVE_CONFIG_CMDLINE=1 tells the config overlay
#         to keep the base CMDLINE rather than overwriting it.
#
#   (B) KERNEL_BASE_CONFIG is explicitly empty (KERNEL_BASE_CONFIG=""):
#         Use arm64 defconfig and warn the user.  This is a last resort;
#         defconfig kernels typically fail to boot in gem5 because they
#         lack VExpress, PL011, and virtio support.
# ---------------------------------------------------------------------------
if [ -n "${KERNEL_BASE_CONFIG}" ]; then
    if [ ! -f "${KERNEL_BASE_CONFIG}" ]; then
        echo "error: KERNEL_BASE_CONFIG not found: ${KERNEL_BASE_CONFIG}" >&2
        exit 1
    fi
    cp "${KERNEL_BASE_CONFIG}" "${LINUX_BUILD}/.config"
    echo "using base config: ${KERNEL_BASE_CONFIG}"
    : "${OPENNPUX_PRESERVE_CONFIG_CMDLINE:=1}"
else
    cat >&2 <<'EOF'
warning: building from arm64 defconfig (KERNEL_BASE_CONFIG is empty).
A defconfig-based kernel may not produce serial output in gem5.
For a known-good baseline, use the reference config bundled at:
  tools/kernel/gem5-4.18.config
To extract a config from a booting gem5 guest:
  zcat /proc/config.gz > /tmp/gem5-4.18.config
EOF
    make -C "${LINUX_SRC}" O="${LINUX_BUILD}" ARCH="${ARCH}" \
        CROSS_COMPILE="${CROSS_COMPILE}" defconfig
    : "${OPENNPUX_PRESERVE_CONFIG_CMDLINE:=0}"
fi

# ---------------------------------------------------------------------------
# Step 3: Apply the gem5 config overlay.
#   Sets VExpress, PL011, virtio, GIC, modules, and other required options.
#   Disables KASLR, BTI, MTE, PTR_AUTH, and other features incompatible
#   with gem5's ARM model.
# ---------------------------------------------------------------------------
OPENNPUX_PRESERVE_CONFIG_CMDLINE="${OPENNPUX_PRESERVE_CONFIG_CMDLINE}" \
    "${SCRIPT_DIR}/configure_arm64_gem5_kernel.sh" "${LINUX_BUILD}/.config"

# ---------------------------------------------------------------------------
# Step 4: Resolve new config symbols (olddefconfig) then build.
#   olddefconfig sets every symbol introduced since the base config's kernel
#   version to its default value, avoiding interactive prompts.
# ---------------------------------------------------------------------------
make -C "${LINUX_SRC}" O="${LINUX_BUILD}" ARCH="${ARCH}" \
    CROSS_COMPILE="${CROSS_COMPILE}" LOCALVERSION="${LOCALVERSION}" \
    olddefconfig
make -C "${LINUX_SRC}" O="${LINUX_BUILD}" ARCH="${ARCH}" \
    CROSS_COMPILE="${CROSS_COMPILE}" LOCALVERSION="${LOCALVERSION}" \
    -j"${JOBS}" Image modules

# ---------------------------------------------------------------------------
# Step 5: Verify build outputs and stage them for gem5 consumption.
#   - Image-<release>:   raw kernel image, installed into guest /boot
#   - vmlinux-<release>:  ELF with debug symbols, passed to gem5 --kernel
#   - kernel.release:     single-line release string for scripts to consume
# ---------------------------------------------------------------------------
kernel_release="$(make -s -C "${LINUX_SRC}" O="${LINUX_BUILD}" ARCH="${ARCH}" \
    CROSS_COMPILE="${CROSS_COMPILE}" LOCALVERSION="${LOCALVERSION}" kernelrelease)"

IMAGE_SRC="${LINUX_BUILD}/arch/arm64/boot/Image"
VMLINUX_SRC="${LINUX_BUILD}/vmlinux"
if [ ! -f "${IMAGE_SRC}" ]; then
    echo "error: kernel build completed but ${IMAGE_SRC} was not produced" >&2
    exit 1
fi
if [ ! -f "${VMLINUX_SRC}" ]; then
    echo "error: kernel build completed but ${VMLINUX_SRC} was not produced" >&2
    exit 1
fi

mkdir -p "${ROOT_DIR}/build/kernel"
cp "${IMAGE_SRC}" "${ROOT_DIR}/build/kernel/Image-${kernel_release}"
cp "${VMLINUX_SRC}" "${ROOT_DIR}/build/kernel/vmlinux-${kernel_release}"
printf '%s\n' "${kernel_release}" > "${ROOT_DIR}/build/kernel/kernel.release"

echo "kernel_release=${kernel_release}"
echo "guest_boot_image=${ROOT_DIR}/build/kernel/Image-${kernel_release}"
echo "gem5_kernel=${ROOT_DIR}/build/kernel/vmlinux-${kernel_release}"
echo "kernel_build=${LINUX_BUILD}"
