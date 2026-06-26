#!/bin/sh

set -eu

CONFIG="${1:-build/linux-arm64/.config}"
if [ ! -f "${CONFIG}" ]; then
    echo "error: config not found: ${CONFIG}" >&2
    exit 1
fi

grep_key() {
    key="$1"
    grep -E "^${key}=|^# ${key} is not set" "${CONFIG}" || \
        echo "${key}=<missing>"
}

for key in \
    CONFIG_ARCH_VEXPRESS \
    CONFIG_VEXPRESS_CONFIG \
    CONFIG_MFD_VEXPRESS_SYSREG \
    CONFIG_CLK_VEXPRESS_OSC \
    CONFIG_SERIAL_AMBA_PL011 \
    CONFIG_SERIAL_AMBA_PL011_CONSOLE \
    CONFIG_SERIAL_EARLYCON \
    CONFIG_PRINTK \
    CONFIG_VIRTIO \
    CONFIG_VIRTIO_MMIO \
    CONFIG_VIRTIO_BLK \
    CONFIG_ARM_GIC \
    CONFIG_ARM_ARCH_TIMER \
    CONFIG_ARM_TIMER_SP804 \
    CONFIG_MODULES \
    CONFIG_RANDOMIZE_BASE \
    CONFIG_ARM64_BTI \
    CONFIG_ARM64_MTE \
    CONFIG_ARM64_PTR_AUTH \
    CONFIG_CMDLINE_BOOL \
    CONFIG_CMDLINE
do
    grep_key "$key"
done
