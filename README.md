# gem5 Coral x86 Superproject

This repository is the x86-oriented superproject for system-level Coral NPU
simulation work.

## Quick Start

Set the path to your gem5 ARM disk images and kernels once per session:

```sh
export IMAGE_PATH="${IMAGE_PATH:-$HOME/wlk/gem5_arm_linux_images}"
```

If your images live elsewhere, set `IMAGE_PATH` explicitly. The examples below use `$IMAGE_PATH` for all disk image paths.

The gem5 run script (`run_multicore.sh`) also respects `IMAGE_PATH`, and most test wrappers accept `CORAL_DISK_IMG` and `CORAL_KERNEL_IMAGE` overrides.

## Layout

- `thirdparty/gem5`: pinned gem5 submodule
- `thirdparty/coralnpu`: pinned official Coral submodule
- `sim/gem5`: gem5-side local source deltas, stored with the same directory
  structure as upstream gem5
- `sim/coralnpu`: Coral-side local source deltas, stored with the same
  directory structure as upstream Coral
- `runtime/host`: host-side bootscripts and runtime bring-up assets
- `runtime/npu`: NPU-side program placeholders and notes
- `rtl/wrappers`: wrapper code and integration shims for RTL co-simulation
- `docs`: ADRs, design notes, and runbooks
- `tools/coralnpu`: phase-1 Coral validation helpers

## Workflow

1. Update submodules to the pinned commits recorded in
   `thirdparty/PINNED_COMMITS.md`.
2. Keep local gem5 and Coral modifications under `sim/gem5/` and
   `sim/coralnpu/` using the same relative paths as their upstream trees.
3. Merge `sim/gem5` into `thirdparty/gem5` and `sim/coralnpu` into
   `thirdparty/coralnpu` before compilation.
4. Use the runbooks under `docs/runbooks` to validate phase-1 and later system
   flows on an x86 Linux host.

## Guest MMIO Tool

The Coral bring-up scripts need a guest-side MMIO helper because `dd` reads from
`/dev/mem` can fail on device memory with `Bad address`. On the x86 Linux host:

```sh
tools/guest_tools/build_mmio32.sh
sudo tools/guest_tools/install_mmio32_to_image.sh \
  $IMAGE_PATH/ubuntu-18.04-arm64-docker.img
```

Rebuild the boot checkpoint after modifying the disk image so Linux sees the
updated contents.

The userspace control utility can be built and installed similarly:

```sh
tools/guest_tools/build_coralctl.sh
sudo tools/guest_tools/install_coralctl_to_image.sh \
  $IMAGE_PATH/ubuntu-18.04-arm64-docker.img
```

Use `coralctl info` to inspect the active backend and `coralctl run` to start
the staged firmware and wait for completion.
Use `coralctl dma-test` with the DMA smoke firmware to verify coherent
shared-memory reads and writes.

## Phase 2: Coral RTL bridge

On the supported x86 Linux host, build the official `CoreMiniAxi` Verilator
model and the OpenNPUX C ABI adapter:

```sh
./tools/coralnpu/phase2_prepare_bazel.sh
./tools/coralnpu/phase2_check_abi.sh
./tools/coralnpu/phase2_build_bridge.sh
```

If the x86 host cannot resolve `releases.bazel.build`, download
`bazel-8.6.0-linux-x86_64` on another machine and install it with:

```sh
BAZEL_BINARY=/path/to/bazel-8.6.0-linux-x86_64 \
  ./tools/coralnpu/phase2_prepare_bazel.sh
```

The staged Phase-2 artifacts are:

```sh
build/coralnpu/libcoralnpu_gem5_bridge.so
build/coralnpu/gem5_smoke_halt.elf
build/coralnpu/gem5_dma_smoke.elf
build/coralnpu/gem5_command_smoke.elf
```

Apply the gem5 overlay and run the existing full-system flow with the RTL
backend:

```sh
./sim/gem5/apply_patchset.sh
CORAL_NPU_BACKEND=verilated-coral ./thirdparty/gem5/run_multicore.sh
```

The default remains `stage-a`. Set `CORAL_RTL_TICK_PERIOD` and
`CORAL_RTL_CYCLES_PER_EVENT` to tune the RTL scheduling quantum.

The boot checkpoint is stored at `checkpoint/coralnpu_ckpt` relative to the
superproject root. Pulling code or changing the injected resume script does
not rebuild it. Use `CORAL_REBUILD_CKPT=1` only when the booted guest state
must be recreated.

## Phase 4: Command submission

Build the bridge and command firmware, rebuild/install `coralctl`, and run the
driver-backed multi-buffer command test:

```sh
./tools/coralnpu/phase2_build_bridge.sh
./tools/guest_tools/build_coralctl.sh
sudo ./tools/guest_tools/install_coralctl_to_image.sh \
  $IMAGE_PATH/ubuntu-18.04-arm64-docker.img

cd thirdparty/gem5
CORAL_KERNEL_IMAGE=../../build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
CORAL_REBUILD_CKPT=1 \
./run_multicore.sh
cd ../..

CORAL_KERNEL_IMAGE=./build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
./tools/coralnpu/run_command_test.sh
```

The acceptance result is `vector_add=PASS`, `completed_elements=16`, and
`output_checksum=0x00000198`.

## Phase 4/5: Heterogeneous platform

The generic model runtime and custom RTL accelerator use the same driver and
command ABI as the official Coral path. Build and install the sample assets:

```sh
./tools/coralnpu/phase2_build_bridge.sh
IMAGE=$IMAGE_PATH/ubuntu-18.04-arm64-docker.img \
  ./tools/coralnpu/phase45_prepare_guest_assets.sh
```

After rebuilding the boot checkpoint once, run:

```sh
CORAL_KERNEL_IMAGE=./build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
./tools/coralnpu/run_model_test.sh
```

See `docs/runbooks/phase45_platform_acceptance.md` for the model and explicit
official/custom RTL acceptance criteria.

## RVV Highmem MobileNet

The optional highmem path retains the standard bridge and adds the official
`RvvCoreMiniHighmemAxi` configuration with LiteRT Micro MobileNet firmware:

```sh
./tools/coralnpu/build_rvv_mobilenet.sh
IMAGE=$IMAGE_PATH/ubuntu-18.04-arm64-docker.img \
./tools/coralnpu/phase45_prepare_guest_assets.sh
CORAL_KERNEL_IMAGE=./build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
./tools/coralnpu/run_rvv_mobilenet_test.sh
```

The first run creates a dedicated 8 MiB-window checkpoint; the second restores
it and executes MobileNet. See
`docs/runbooks/rvv_mobilenet_acceptance.md` for the complete procedure.
