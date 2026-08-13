#!/bin/sh
#
# configure_arm64_gem5_kernel.sh — Apply gem5-required kernel config options.
#
# This script overwrites specific CONFIG_* entries in a Linux kernel .config
# so the resulting arm64 kernel boots under gem5 full-system simulation.
# It is called by build_arm64_kernel.sh after the base config (defconfig or
# a user-supplied KERNEL_BASE_CONFIG) has been copied into the build tree.
#
# Call chain:
#   build_arm64_kernel.sh
#     → configure_arm64_gem5_kernel.sh <kernel-build/.config>
#     → make olddefconfig
#     → make Image modules
#
# The symmetric read-only checker is check_gem5_kernel_config.sh.  When
# options are added or removed here, update that script as well.
#
# @kernel-config-spec  v1  2025-07-28
# @synchronized-with  tools/kernel/check_gem5_kernel_config.sh

set -eu

CONFIG="${1:?usage: $0 <kernel-build/.config>}"

# ---------------------------------------------------------------------------
# Helper: set_config KEY VALUE
#   Ensure KEY=VALUE exists in the .config, replacing any previous entry
#   (including "# KEY is not set").  If KEY is completely absent, append it.
# ---------------------------------------------------------------------------
set_config() {
    key="$1"
    value="$2"
    if grep -q "^${key}=" "${CONFIG}" || grep -q "^# ${key} is not set" "${CONFIG}"; then
        tmp="${CONFIG}.tmp.$$"
        awk -v key="${key}" -v value="${value}" '
            $0 ~ "^" key "=" || $0 == "# " key " is not set" {
                print key "=" value
                next
            }
            { print }
        ' "${CONFIG}" > "${tmp}"
        mv "${tmp}" "${CONFIG}"
    else
        printf '%s=%s\n' "${key}" "${value}" >> "${CONFIG}"
    fi
}

# ---------------------------------------------------------------------------
# Helper: unset_config KEY
#   Ensure "# KEY is not set" exists in the .config.  Used to disable
#   features that are known to break gem5 (KASLR, BTI, MTE, etc.).
# ---------------------------------------------------------------------------
unset_config() {
    key="$1"
    if grep -q "^${key}=" "${CONFIG}" || grep -q "^# ${key} is not set" "${CONFIG}"; then
        tmp="${CONFIG}.tmp.$$"
        awk -v key="${key}" '
            $0 ~ "^" key "=" || $0 == "# " key " is not set" {
                print "# " key " is not set"
                next
            }
            { print }
        ' "${CONFIG}" > "${tmp}"
        mv "${tmp}" "${CONFIG}"
    else
        printf '# %s is not set\n' "${key}" >> "${CONFIG}"
    fi
}

# ===========================================================================
# Category 1: Core boot infrastructure
#   These are required for gem5's VExpress-based ARM full-system model to
#   reach the PL011 serial console and mount the root filesystem.
# ===========================================================================
set_config CONFIG_BLK_DEV_INITRD y
set_config CONFIG_DEVTMPFS y
set_config CONFIG_DEVTMPFS_MOUNT y
set_config CONFIG_ARCH_VEXPRESS y
set_config CONFIG_VEXPRESS_CONFIG y
set_config CONFIG_MFD_VEXPRESS_SYSREG y
set_config CONFIG_CLK_VEXPRESS_OSC y
set_config CONFIG_PROC_FS y
set_config CONFIG_SYSFS y
set_config CONFIG_TMPFS y
set_config CONFIG_EXT4_FS y
set_config CONFIG_TTY y
set_config CONFIG_PRINTK y
set_config CONFIG_BUG y
set_config CONFIG_DEBUG_KERNEL y

# ===========================================================================
# Category 2: Virtio block and PCI transport
#   gem5 provides a virtio-blk device for the root disk image.  Without
#   these the kernel cannot find /dev/vda and will panic.
# ===========================================================================
set_config CONFIG_VIRTIO y
set_config CONFIG_VIRTIO_MMIO y
set_config CONFIG_VIRTIO_PCI y
set_config CONFIG_VIRTIO_BLK y
set_config CONFIG_BLK_MQ_VIRTIO y

# ===========================================================================
# Category 3: Serial console (PL011)
#   gem5's ARM model exposes a PL011 UART at 0x1c090000.
# ===========================================================================
set_config CONFIG_SERIAL_AMBA_PL011 y
set_config CONFIG_SERIAL_AMBA_PL011_CONSOLE y
set_config CONFIG_SERIAL_EARLYCON y

# ===========================================================================
# Category 4: Interrupt controller and timer
#   GIC (Generic Interrupt Controller) and arch timer are part of the
#   gem5 ARM SoC model.
# ===========================================================================
set_config CONFIG_ARM_GIC y
set_config CONFIG_ARM_GIC_V2M y
set_config CONFIG_ARM_GIC_V3 y
set_config CONFIG_ARM_ARCH_TIMER y
set_config CONFIG_ARM_TIMER_SP804 y

# ===========================================================================
# Category 5: Module and Device Tree support
#   CONFIG_MODULES is required for opennpux_coral.ko.
#   CONFIG_OF_* enables Device Tree parsing for the reserved-memory region.
#   CONFIG_UIO is needed for userspace MMIO access during bring-up.
# ===========================================================================
set_config CONFIG_MODULES y
set_config CONFIG_MODULE_UNLOAD y
set_config CONFIG_OF y
set_config CONFIG_OF_ADDRESS y
set_config CONFIG_UIO y

# ===========================================================================
# Category 6: Kernel command line
#   Gem5 injects the command line through the DTB.  CONFIG_CMDLINE_BOOL
#   makes the built-in fallback available.  The default preserves the
#   base config's CMDLINE when OPENNPUX_PRESERVE_CONFIG_CMDLINE=1
#   (useful for reusing a booting 4.18 config verbatim).
#   CONFIG_STRICT_DEVMEM must be disabled so /dev/mem works for MMIO
#   bring-up before the driver is available.
# ===========================================================================
set_config CONFIG_CMDLINE_BOOL y
if [ "${OPENNPUX_PRESERVE_CONFIG_CMDLINE:-0}" = "1" ] &&
   grep -q '^CONFIG_CMDLINE=' "${CONFIG}"; then
    echo "preserving base CONFIG_CMDLINE from ${CONFIG}"
else
    set_config CONFIG_CMDLINE '"earlycon=pl011,0x1c090000 console=ttyAMA0 keep_bootcon ignore_loglevel loglevel=8 nokaslr root=/dev/vda1 rw init=/sbin/init"'
fi
unset_config CONFIG_STRICT_DEVMEM

# ===========================================================================
# Category 7: Disable features incompatible with gem5's ARM model
#   - KASLR/relocation: gem5 does not model the required hardware RNG.
#   - BTI/MTE/PTR_AUTH: ARMv8.5+ features not modelled by the current gem5
#     CPU configuration.  Enabling them causes silent boot failures.
# ===========================================================================
unset_config CONFIG_RELOCATABLE
unset_config CONFIG_RANDOMIZE_BASE
unset_config CONFIG_RANDOMIZE_MODULE_REGION_FULL
unset_config CONFIG_RANDOMIZE_KSTACK_OFFSET
unset_config CONFIG_ARM64_BTI
unset_config CONFIG_ARM64_MTE
unset_config CONFIG_ARM64_PTR_AUTH

# ===========================================================================
# Category 8: Keep the guest lean
#   These subsystems are not used in simulation and add build time.
# ===========================================================================
unset_config CONFIG_DRM
unset_config CONFIG_SOUND
unset_config CONFIG_WLAN
unset_config CONFIG_BT
unset_config CONFIG_USB
unset_config CONFIG_USB_DWC2
unset_config CONFIG_USB_DWC2_HOST
unset_config CONFIG_USB_DWC2_PERIPHERAL
unset_config CONFIG_USB_DWC2_DUAL_ROLE
unset_config CONFIG_USB_GADGET
unset_config CONFIG_USB_SUPPORT
unset_config CONFIG_USB_COMMON
unset_config CONFIG_USB_PHY
unset_config CONFIG_NOP_USB_XCEIV
unset_config CONFIG_KVM
