# Phase 3 Runtime Acceptance Runbook

This runbook validates the first Phase-3 increment: the stable host runtime API
that sits between Linux userspace tools and the future Coral kernel driver.

## Scope

This increment keeps the Phase-2 `/dev/mem` backend so the existing gem5
checkpoint flow remains usable, but moves Coral control and shared-buffer logic
out of the `coralctl` command-line tool into a reusable C API:

- `runtime/host/include/opennpux/coral_runtime.h`
- `runtime/host/src/coral_runtime.c`
- `runtime/host/tools/coralctl.c`

The intended next backend swap is:

- current: runtime API -> `/dev/mem` MMIO and shared-window mmap
- next: runtime API -> `/dev/opennpux-coral` ioctl and mmap

`coralctl` is now only a CLI frontend. System-level output remains compatible
with the Phase-2 scripts.

The repository now carries the first driver boundary scaffold:

- `runtime/host/include/opennpux/coral_uapi.h`
- `runtime/kernel/opennpux_coral.c`
- `runtime/kernel/Makefile`

The runtime auto-selects `/dev/opennpux-coral` when it exists and falls back to
`/dev/mem` otherwise. Set `OPENNPUX_CORAL_TRANSPORT=driver` to require the
driver path, or `OPENNPUX_CORAL_TRANSPORT=devmem` to force the bring-up path.

## Host Validation

Run from the superproject root:

```sh
./tools/guest_tools/build_coral_runtime_tests.sh
cc -O2 -Wall -Wextra -Werror -std=c11 \
  -Iruntime/host/include \
  runtime/host/src/coral_runtime.c \
  runtime/host/tools/coralctl.c \
  -o build/local-tests/coralctl
```

Expected:

```text
PASS: coral runtime host unit tests
```

The driver scaffold can be syntax-built on a Linux host with matching kernel
headers:

```sh
make -C runtime/kernel
```

This first driver increment provides `OPENNPUX_CORAL_IOC_GET_INFO` and
`OPENNPUX_CORAL_IOC_RUN`. Shared-window mmap is the next driver increment.

## 4.19 Kernel And Driver Validation

Use Linux 4.19 for the first kernel-driver acceptance. It has been confirmed to
reach the gem5 PL011 console with the 4.18 base config, while 6.6 currently
requires separate early-boot compatibility work.

Prepare the base config once:

```sh
mkdir -p build/kernel
cp /path/to/gem5-4.18.config build/kernel/gem5-4.18.config
```

Build the 4.19 kernel, build `opennpux_coral.ko`, and install both into the
lightweight image:

```sh
IMAGE=/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img \
./tools/kernel/phase3_validate_4.19_kernel.sh
```

The guest image must also contain a module loader. If the driver-info test
prints `insmod: not found` or `no insmod/modprobe found in guest`, install an
aarch64 BusyBox or kmod-provided `insmod` binary:

```sh
./tools/guest_tools/install_module_loader_to_image.sh \
  /home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img \
  /path/to/aarch64/busybox
```

The script prints the `CORAL_KERNEL_IMAGE=.../vmlinux-<release>` value to use
with gem5. First rebuild the checkpoint with the 4.19 kernel:

```sh
cd thirdparty/gem5
CORAL_KERNEL_IMAGE=/home/barry/code/opennpux/build/kernel/vmlinux-$(cat ../../build/kernel/kernel.release) \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
CORAL_REBUILD_CKPT=1 \
./run_multicore.sh
```

Then validate the driver-backed info path:

```sh
CORAL_KERNEL_IMAGE=/home/barry/code/opennpux/build/kernel/vmlinux-$(cat ../../build/kernel/kernel.release) \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
CORAL_RESUME_BOOTSCRIPT="$PWD/configs/coralnpu/coral-driver-info-test.rcS" \
./run_multicore.sh
```

Expected:

```text
[coral-driver-info-test] kernel=4.19...
transport=driver
backend=stage-a
[coral-driver-info-test] PASS
```

## Guest Tool Build

On the x86 Linux development host:

```sh
./tools/guest_tools/build_coralctl.sh
sudo ./tools/guest_tools/install_coralctl_to_image.sh \
  /home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img
```

The build script now compiles both the runtime library source and the CLI
frontend into the static aarch64 guest binary.

## System-Level Regression

Reuse the Phase-2 boot checkpoint. Do not rebuild the checkpoint for this
runtime-only change.

```sh
cd thirdparty/gem5
CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_BRIDGE="$PWD/../../build/coralnpu/libcoralnpu_gem5_bridge.so" \
CORAL_RTL_FIRMWARE="$PWD/../../build/coralnpu/gem5_smoke_halt.elf" \
CORAL_RESUME_BOOTSCRIPT="$PWD/configs/coralnpu/coralctl-test.rcS" \
./run_multicore.sh
```

Expected:

```text
[coralctl-test] PASS
```

Then run DMA smoke:

```sh
cd thirdparty/gem5
CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_BRIDGE="$PWD/../../build/coralnpu/libcoralnpu_gem5_bridge.so" \
CORAL_RTL_FIRMWARE="$PWD/../../build/coralnpu/gem5_dma_smoke.elf" \
CORAL_RESUME_BOOTSCRIPT="$PWD/configs/coralnpu/coral-dma-test.rcS" \
./run_multicore.sh
```

Expected:

```text
status=0x00000001
dma_result=42
dma_magic=0x4e505544
dma_requests=4
dma_completions=4
dma_errors=0
dma_state=0x00000000
dma_test=PASS
[coral-dma-test] PASS
```

## Acceptance Criteria

Phase-3 runtime increment passes when all of these are true:

- host runtime unit test passes
- host `coralctl` compile passes with `-Wall -Wextra -Werror`
- aarch64 `coralctl` guest binary builds and installs into the image
- `coralctl-test` still passes from the existing checkpoint
- `coral-dma-test` still passes from the existing checkpoint
- no `coralctl` command output used by Phase-2 scripts regresses

## Next Increment

Complete the minimal Linux device boundary behind this runtime API:

- `/dev/opennpux-coral` character device
- ioctl for info, reset/start, and status
- mmap for the shared DMA window
- poll or interrupt-driven completion path

The runtime API should remain the userspace contract while the backend changes
from `/dev/mem` to the kernel driver.
