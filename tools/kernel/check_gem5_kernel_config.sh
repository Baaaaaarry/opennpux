#!/bin/sh
#
# check_gem5_kernel_config.sh — Report gem5-relevant kernel config options.
#
# This is the read-only counterpart of configure_arm64_gem5_kernel.sh.
# It prints the current value of each config option that matters for gem5
# ARM full-system boot, one per line in KEY=VALUE format.
#
# Call chain:
#   phase3_validate_4.19_kernel.sh
#     → check_gem5_kernel_config.sh <kernel-build/.config>
#
# Manual usage (from the superproject root):
#   ./tools/kernel/check_gem5_kernel_config.sh build/linux-arm64/.config
#
# Options checked here are a SUBSET of the options set by the configure
# script.  This script focuses on the "must-have" options whose absence
# causes silent boot failure (no serial output).  Options that are
# convenience or optimization only (CONFIG_DRM, CONFIG_USB, etc.) are
# intentionally not listed here.
#
# When the kernel config requirements change, update both this script and
# configure_arm64_gem5_kernel.sh together.
#
# @kernel-config-spec  v2  2026-08-18
# @synchronized-with  tools/kernel/configure_arm64_gem5_kernel.sh

set -eu

CONFIG="${1:-build/linux-arm64/.config}"
if [ ! -f "${CONFIG}" ]; then
    echo "error: config not found: ${CONFIG}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Helper: grep_key KEY
#   Print the config line for KEY.  If KEY is absent from the file entirely
#   (not even "# KEY is not set"), print "KEY=<missing>" so the omission
#   is visible rather than silently skipped.
# ---------------------------------------------------------------------------
grep_key() {
    key="$1"
    grep -E "^${key}=|^# ${key} is not set" "${CONFIG}" || \
        echo "${key}=<missing>"
}

# ===========================================================================
# Category 1: Core boot infrastructure
#   Without these the kernel cannot initialize the VExpress platform or
#   mount the root filesystem.
# ===========================================================================
for key in \
    CONFIG_ARCH_VEXPRESS \
    CONFIG_VEXPRESS_CONFIG \
    CONFIG_MFD_VEXPRESS_SYSREG \
    CONFIG_CLK_VEXPRESS_OSC \
    CONFIG_SERIAL_AMBA_PL011 \
    CONFIG_SERIAL_AMBA_PL011_CONSOLE \
    CONFIG_SERIAL_EARLYCON \
    CONFIG_PRINTK \
    CONFIG_DEVTMPFS \
    CONFIG_DEVTMPFS_MOUNT \
    CONFIG_EXT4_FS
do
    grep_key "$key"
done

# ===========================================================================
# Category 2: Virtio and disk
#   Required for the virtio-blk root device and partition detection.
# ===========================================================================
for key in \
    CONFIG_VIRTIO \
    CONFIG_VIRTIO_MMIO \
    CONFIG_VIRTIO_PCI \
    CONFIG_VIRTIO_BLK \
    CONFIG_NET_9P \
    CONFIG_NET_9P_VIRTIO \
    CONFIG_9P_FS \
    CONFIG_MSDOS_PARTITION \
    CONFIG_EFI_PARTITION
do
    grep_key "$key"
done

# ===========================================================================
# Category 3: Interrupt controller and timer
# ===========================================================================
for key in \
    CONFIG_ARM_GIC \
    CONFIG_ARM_ARCH_TIMER \
    CONFIG_ARM_TIMER_SP804
do
    grep_key "$key"
done

# ===========================================================================
# Category 4: Module support
#   CONFIG_MODULES is required for opennpux_coral.ko.
# ===========================================================================
grep_key CONFIG_MODULES

# ===========================================================================
# Category 5: Features known to break gem5 boot
#   These should all read "# CONFIG_* is not set".  If any are "=y" the
#   kernel may hang before reaching the PL011 console.
# ===========================================================================
for key in \
    CONFIG_RANDOMIZE_BASE \
    CONFIG_ARM64_BTI \
    CONFIG_ARM64_MTE \
    CONFIG_ARM64_PTR_AUTH
do
    grep_key "$key"
done

# ===========================================================================
# Category 6: Kernel command line
#   CONFIG_CMDLINE_BOOL must be set, and CONFIG_CMDLINE should contain
#   the gem5 earlycon/console parameters.
# ===========================================================================
for key in \
    CONFIG_CMDLINE_BOOL \
    CONFIG_CMDLINE
do
    grep_key "$key"
done
