#!/bin/sh

set -eu

CONFIG="${1:?usage: $0 <kernel-build/.config>}"

set_config() {
    key="$1"
    value="$2"
    if grep -q "^${key}=" "${CONFIG}" || grep -q "^# ${key} is not set" "${CONFIG}"; then
        sed -i.bak -e "s/^${key}=.*/${key}=${value}/" \
            -e "s/^# ${key} is not set/${key}=${value}/" "${CONFIG}"
        rm -f "${CONFIG}.bak"
    else
        printf '%s=%s\n' "${key}" "${value}" >> "${CONFIG}"
    fi
}

unset_config() {
    key="$1"
    if grep -q "^${key}=" "${CONFIG}" || grep -q "^# ${key} is not set" "${CONFIG}"; then
        sed -i.bak -e "s/^${key}=.*/# ${key} is not set/" "${CONFIG}"
        rm -f "${CONFIG}.bak"
    else
        printf '# %s is not set\n' "${key}" >> "${CONFIG}"
    fi
}

# Required for gem5 ARM full-system boot and runtime debugging.
set_config CONFIG_BLK_DEV_INITRD y
set_config CONFIG_DEVTMPFS y
set_config CONFIG_DEVTMPFS_MOUNT y
set_config CONFIG_PROC_FS y
set_config CONFIG_SYSFS y
set_config CONFIG_TMPFS y
set_config CONFIG_EXT4_FS y
set_config CONFIG_VIRTIO y
set_config CONFIG_VIRTIO_MMIO y
set_config CONFIG_VIRTIO_BLK y
set_config CONFIG_SERIAL_AMBA_PL011 y
set_config CONFIG_SERIAL_AMBA_PL011_CONSOLE y
set_config CONFIG_MODULES y
set_config CONFIG_MODULE_UNLOAD y
set_config CONFIG_OF y
set_config CONFIG_OF_ADDRESS y
set_config CONFIG_UIO y
unset_config CONFIG_STRICT_DEVMEM

# Keep the guest lean. The image is not a package host or desktop.
unset_config CONFIG_DRM
unset_config CONFIG_SOUND
unset_config CONFIG_WLAN
unset_config CONFIG_BT
unset_config CONFIG_USB
unset_config CONFIG_KVM
