# Rebuild Kernel And Coral Driver For gem5

Use this flow when the original 18.04 kernel source or build tree is missing.
The root filesystem can stay lightweight: keep the Linux source and build tree
on the x86 host, and install only the kernel image, modules, `coralctl`, and
`opennpux_coral.ko` into the guest image.

## Why Rebuild

Out-of-tree modules require a kernel build tree that exactly matches the guest
kernel release. If the original tree for `4.18.0+` is gone, rebuilding a newer
kernel is cleaner than trying to guess the old config.

The image does not need to become heavy. Do not copy kernel source or headers
into the image.

## Build A New arm64 Kernel

On the x86 Linux host:

```sh
sudo apt-get install -y \
  bc bison flex git libssl-dev make gcc-aarch64-linux-gnu

cd ~/code/opennpux
./tools/kernel/build_arm64_kernel.sh
```

Outputs:

```text
build/kernel/Image-<kernel-release>
build/kernel/vmlinux-<kernel-release>
build/kernel/kernel.release
build/linux-arm64/
```

`build/linux-arm64` is the matching build tree for out-of-tree modules.

## Reuse The Booting 4.18 Config

If a kernel built from `defconfig` produces no serial output, use the old
booting guest config as the base. Boot the known-good 4.18 system and run:

```sh
zcat /proc/config.gz > /tmp/gem5-4.18.config
```

Copy it to the x86 host:

```sh
mkdir -p ~/code/opennpux/build/kernel
# Use whatever transfer path is available in the guest: 9p, mounted data disk,
# scp, or paste through the terminal.
cp /path/from/guest/gem5-4.18.config \
  ~/code/opennpux/build/kernel/gem5-4.18.config
```

Then rebuild the new kernel using that config as the base:

```sh
cd ~/code/opennpux
KERNEL_BASE_CONFIG="$PWD/build/kernel/gem5-4.18.config" \
LINUX_BRANCH=linux-4.19.y \
./tools/kernel/build_arm64_kernel.sh
```

When `KERNEL_BASE_CONFIG` is set, the script preserves the base
`CONFIG_CMDLINE` by default. For the known-good 4.18 gem5 image this keeps:

```text
CONFIG_CMDLINE="console=ttyAMA0"
```

The script still overlays the required OpenNPUX options after copying the base
config, including modules, early console support, VExpress, virtio, ext4, and
NPU driver support. To force the longer repository command line instead, set:

```sh
OPENNPUX_PRESERVE_CONFIG_CMDLINE=0
```

Check the resulting config:

```sh
./tools/kernel/check_gem5_kernel_config.sh build/linux-arm64/.config
```

Start with `linux-4.19.y` for bring-up. It is close to the known-good 4.18
gem5 kernel but still provides a reproducible build tree for out-of-tree
modules. Treat `linux-6.6.y` as a later compatibility target; if it reaches no
serial output while the old `vmlinux.arm64` works, the failure is before Linux
reaches the PL011 console.

The repository default is therefore `linux-4.19.y`. Override `LINUX_BRANCH`
only after the 4.19 driver path is accepted.

Compare a non-booting kernel against the known-good kernel:

```sh
./tools/kernel/diagnose_gem5_kernel.sh \
  "$PWD/build/kernel/vmlinux-$(cat build/kernel/kernel.release)" \
  /home/barry/wlk/gem5_arm_linux_images/vmlinux.arm64
```

## Build The Coral Driver Module

```sh
cd ~/code/opennpux
./tools/kernel/build_opennpux_coral_ko.sh
```

Output:

```text
build/kernel/opennpux_coral-<kernel-release>.ko
```

## Install Into The Lightweight Image

```sh
IMG=/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img

sudo ./tools/kernel/install_kernel_to_image.sh \
  "$IMG" \
  "$PWD/build/linux-arm64" \
  "$PWD/build/kernel/Image-$(cat build/kernel/kernel.release)" \
  "$PWD/build/kernel/opennpux_coral-$(cat build/kernel/kernel.release).ko"
sudo ./tools/kernel/install_opennpux_init_to_image.sh "$IMG"
```

This installs:

- `/boot/Image-<kernel-release>`
- `/boot/Image` symlink
- `/lib/modules/<kernel-release>/*`
- `/lib/modules/<kernel-release>/extra/opennpux_coral.ko`
- `/etc/modules-load.d/opennpux-coral.conf`

The source and build tree remain on the host.

## Boot gem5 With The New Kernel

The gem5 run script accepts a kernel override. Use the ELF `vmlinux-*` file for
gem5. The raw `Image-*` file is installed into the guest filesystem for
completeness, but this ARM full-system config loads the host-side ELF through
`--kernel`.

```sh
cd thirdparty/gem5
CORAL_KERNEL_IMAGE=/home/barry/code/opennpux/build/kernel/vmlinux-$(cat ../../build/kernel/kernel.release) \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
CORAL_REBUILD_CKPT=1 \
./run_multicore.sh
```

The checkpoint metadata includes the kernel image path. Changing
`CORAL_KERNEL_IMAGE` rebuilds the boot checkpoint once.

If gem5 reports:

```text
Could not load kernel file .../Image-<release>
```

you passed the raw arm64 `Image` instead of `vmlinux`. Re-run with
`build/kernel/vmlinux-<release>`.

## No Serial Output

If `m5out/system.terminal` stays empty, Linux did not reach the PL011 console.
First rebuild with the repository config script so `CONFIG_SERIAL_EARLYCON` and
the gem5-safe command line are applied:

```sh
./tools/kernel/build_arm64_kernel.sh
```

Then boot with an explicit early console override:

```sh
CORAL_KERNEL_IMAGE=/home/barry/code/opennpux/build/kernel/vmlinux-$(cat ../../build/kernel/kernel.release) \
CORAL_KERNEL_CMDLINE="earlycon=pl011,mmio32,0x1c090000 console=ttyAMA0 keep_bootcon ignore_loglevel loglevel=8 nokaslr root=/dev/vda1 rw init=/bin/sh" \
CORAL_REBUILD_CKPT=1 \
./run_multicore.sh
```

If the terminal is still empty, inspect whether the new config actually
contains:

```sh
grep -E 'CONFIG_SERIAL_EARLYCON|CONFIG_PRINTK|CONFIG_SERIAL_AMBA_PL011|CONFIG_RANDOMIZE_BASE|CONFIG_ARM64_BTI|CONFIG_ARM64_MTE|CONFIG_ARM64_PTR_AUTH' \
  /home/barry/code/opennpux/build/linux-arm64/.config
```

Expected:

```text
CONFIG_SERIAL_EARLYCON=y
CONFIG_PRINTK=y
CONFIG_SERIAL_AMBA_PL011=y
CONFIG_SERIAL_AMBA_PL011_CONSOLE=y
# CONFIG_RANDOMIZE_BASE is not set
# CONFIG_ARM64_BTI is not set
# CONFIG_ARM64_MTE is not set
# CONFIG_ARM64_PTR_AUTH is not set
```

## Validate Driver Path

Use the dedicated driver-info resume script. Do not switch the existing DMA
smoke to the driver path until shared-window mmap is implemented.

```sh
cd thirdparty/gem5
CORAL_RESUME_BOOTSCRIPT="$PWD/configs/coralnpu/coral-driver-info-test.rcS" \
CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_BRIDGE="$PWD/../../build/coralnpu/libcoralnpu_gem5_bridge.so" \
CORAL_RTL_FIRMWARE="$PWD/../../build/coralnpu/gem5_smoke_halt.elf" \
./run_multicore.sh
```

Manual guest commands are:

```sh
ls -l /dev/opennpux-coral
OPENNPUX_CORAL_TRANSPORT=driver coralctl info
```

Expected:

```text
transport=driver
backend=verilated-coral
```

The current driver scaffold supports `info` and `run`. Driver-backed
`dma-test` requires the next increment: shared-window `mmap`.
