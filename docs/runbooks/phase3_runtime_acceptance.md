# Phase 3 Runtime Acceptance Runbook

This runbook validates the Phase-3 Linux driver and stable host runtime API.

## Scope

The Phase-2 `/dev/mem` backend remains available for bring-up compatibility,
but normal Phase-3 operation uses `/dev/opennpux-coral` for control and shared
memory:

- `runtime/host/include/opennpux/coral_runtime.h`
- `runtime/host/src/coral_runtime.c`
- `runtime/host/tools/coralctl.c`

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

The driver provides:

- `GET_INFO` and backward-compatible synchronous `RUN` ioctls
- capability negotiation and asynchronous `START`
- bounded mmap of the DT-reserved shared DMA window
- poll-based completion notification with timeout recovery through `RESET`

The shared mapping is non-cached. This is conservative for both coherent gem5
DMA and future non-coherent platforms; userspace cannot mmap CSR or arbitrary
physical-memory ranges through this device.

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
prints `insmod: not found` or `no insmod/modprobe found in guest`, build the
minimal static aarch64 BusyBox. The build enables only `insmod` and the small
module-loading implementation:

```sh
sudo apt-get install gcc-aarch64-linux-gnu libc6-dev-arm64-cross \
  make bzip2 curl qemu-user
./tools/guest_tools/build_busybox_aarch64.sh
```

The source archive is downloaded from `busybox.net`, cached under
`.cache/busybox`, and checked against the pinned SHA-256. For an offline build,
set `BUSYBOX_TARBALL=/path/to/busybox-1.36.1.tar.bz2`.

Install the resulting multicall binary:

```sh
./tools/guest_tools/install_module_loader_to_image.sh \
  /home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img \
  ./build/guest-tools/busybox-aarch64
```

Because this changes the guest disk, rebuild the boot checkpoint once before
running the driver-info resume test.

The checkpoint bootstrap copies `coralctl`, `opennpux_coral.ko`, and BusyBox
into tmpfs. Resume tests use those copies so they do not issue virtio-blk reads
after restore; the current gem5 virtio queue can otherwise report
`req.0:id ... is not a head` and deliver `SIGBUS` while loading the module.
Resume scripts must not mount a second tmpfs over `/tmp`, because that would
hide the preloaded checkpoint files. All Coral resume scripts therefore check
`/proc/mounts` before mounting pseudo filesystems.

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

## Driver DMA Acceptance

After rebuilding and installing the Phase-3 module and `coralctl`, rebuild the
boot checkpoint once so both files are preloaded into tmpfs. Then run:

```sh
CORAL_KERNEL_IMAGE=/home/barry/code/opennpux/build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
./tools/coralnpu/run_driver_dma_test.sh
```

In addition to the normal DMA result, expected output includes:

```text
transport=driver
driver_abi=1
driver_features=0x0000000f
shared_clear=PASS
[coral-driver-dma-test] PASS
```

## Acceptance Criteria

Phase-3 runtime increment passes when all of these are true:

- host runtime unit test passes
- host `coralctl` compile passes with `-Wall -Wextra -Werror`
- aarch64 `coralctl` guest binary builds and installs into the image
- `coralctl-test` still passes from the existing checkpoint
- `coral-dma-test` still passes from the existing checkpoint
- `coral-driver-dma-test` passes without opening `/dev/mem`
- no `coralctl` command output used by Phase-2 scripts regresses

## Next Increment

Define the inference submission contract on top of the completed device
boundary: versioned command descriptors, runtime-owned tensor buffers, and a
real interrupt source when the platform model exposes one.
