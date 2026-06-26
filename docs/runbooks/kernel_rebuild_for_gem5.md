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
LINUX_BRANCH=linux-6.6.y ./tools/kernel/build_arm64_kernel.sh
```

Outputs:

```text
build/kernel/Image-<kernel-release>
build/kernel/vmlinux-<kernel-release>
build/kernel/kernel.release
build/linux-arm64/
```

`build/linux-arm64` is the matching build tree for out-of-tree modules.

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
```

This installs:

- `/boot/Image-<kernel-release>`
- `/boot/Image` symlink
- `/lib/modules/<kernel-release>/*`
- `/lib/modules/<kernel-release>/extra/opennpux_coral.ko`
- `/etc/modules-load.d/opennpux-coral.conf`

The source and build tree remain on the host.

## Boot gem5 With The New Kernel

The gem5 run script accepts a kernel override:

```sh
cd thirdparty/gem5
CORAL_KERNEL_IMAGE=/home/barry/code/opennpux/build/kernel/Image-$(cat ../../build/kernel/kernel.release) \
CORAL_REBUILD_CKPT=1 \
./run_multicore.sh
```

The checkpoint metadata includes the kernel image path. Changing
`CORAL_KERNEL_IMAGE` rebuilds the boot checkpoint once.

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
