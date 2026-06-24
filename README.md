# gem5 Coral x86 Superproject

This repository is the x86-oriented superproject for system-level Coral NPU
simulation work.

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
  /home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img
```

Rebuild the boot checkpoint after modifying the disk image so Linux sees the
updated contents.

## Phase 2: Coral RTL bridge

On the supported x86 Linux host, build the official `CoreMiniAxi` Verilator
model and the OpenNPUX C ABI adapter:

```bash
./tools/coralnpu/phase2_prepare_bazel.sh
./tools/coralnpu/phase2_check_abi.sh
./tools/coralnpu/phase2_build_bridge.sh
```

If the x86 host cannot resolve `releases.bazel.build`, download
`bazel-8.6.0-linux-x86_64` on another machine and install it with:

```bash
BAZEL_BINARY=/path/to/bazel-8.6.0-linux-x86_64 \
  ./tools/coralnpu/phase2_prepare_bazel.sh
```

The staged shared library is:

```text
build/coralnpu/libcoralnpu_gem5_bridge.so
```

Apply the gem5 overlay and run the existing full-system flow with the RTL
backend:

```bash
./sim/gem5/apply_patchset.sh
CORAL_NPU_BACKEND=verilated-coral ./thirdparty/gem5/run_multicore.sh
```

The default remains `stage-a`. Set `CORAL_RTL_TICK_PERIOD` and
`CORAL_RTL_CYCLES_PER_EVENT` to tune the RTL scheduling quantum.

The boot checkpoint is stored at `m5out/coralnpu_ckpt` relative to the
superproject root. Pulling code or changing the injected resume script does
not rebuild it. Use `CORAL_REBUILD_CKPT=1` only when the booted guest state
must be recreated.

## Scope

This superproject intentionally excludes macOS-specific build caches, ARM guest
 image build helpers, and transient simulation outputs from the tracked tree.
