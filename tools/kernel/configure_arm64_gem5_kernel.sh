#!/bin/sh

set -eu

CONFIG="${1:?usage: $0 <kernel-build/.config>}"

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

# Required for gem5 ARM full-system boot and runtime debugging.
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
set_config CONFIG_VIRTIO y
set_config CONFIG_VIRTIO_MMIO y
set_config CONFIG_VIRTIO_PCI y
set_config CONFIG_VIRTIO_BLK y
set_config CONFIG_BLK_MQ_VIRTIO y
set_config CONFIG_SERIAL_AMBA_PL011 y
set_config CONFIG_SERIAL_AMBA_PL011_CONSOLE y
set_config CONFIG_ARM_GIC y
set_config CONFIG_ARM_GIC_V2M y
set_config CONFIG_ARM_GIC_V3 y
set_config CONFIG_ARM_ARCH_TIMER y
set_config CONFIG_ARM_TIMER_SP804 y
set_config CONFIG_MODULES y
set_config CONFIG_MODULE_UNLOAD y
set_config CONFIG_OF y
set_config CONFIG_OF_ADDRESS y
set_config CONFIG_UIO y
set_config CONFIG_CMDLINE_BOOL y
set_config CONFIG_CMDLINE '"earlycon=pl011,0x1c090000 console=ttyAMA0 loglevel=8 root=/dev/vda1 rw init=/sbin/init"'
unset_config CONFIG_STRICT_DEVMEM

# gem5's ARM model and boot path are easier to debug with fixed placement and
# without newer architectural extensions that may not be fully modelled.
unset_config CONFIG_RELOCATABLE
unset_config CONFIG_RANDOMIZE_BASE
unset_config CONFIG_RANDOMIZE_MODULE_REGION_FULL
unset_config CONFIG_RANDOMIZE_KSTACK_OFFSET
unset_config CONFIG_ARM64_BTI
unset_config CONFIG_ARM64_MTE
unset_config CONFIG_ARM64_PTR_AUTH

# Keep the guest lean. The image is not a package host or desktop.
unset_config CONFIG_DRM
unset_config CONFIG_SOUND
unset_config CONFIG_WLAN
unset_config CONFIG_BT
unset_config CONFIG_USB
unset_config CONFIG_KVM
