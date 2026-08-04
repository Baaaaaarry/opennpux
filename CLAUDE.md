# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a superproject for x86-hosted gem5 + Google Coral NPU system-level simulation. It simulates an ARM heterogeneous SoC (D9200/D9300) running Linux, with a Coral NPU integrated via a Verilator RTL bridge. The NPU executes official Coral firmware and custom accelerator RTL, exchanging commands and tensors with the CPU through coherent shared memory.

## Repository Layout

```
thirdparty/gem5          # pinned upstream gem5 submodule
thirdparty/coralnpu       # pinned upstream google-coral/coralnpu submodule
sim/gem5/                 # local gem5 source deltas (mirrored directory structure)
sim/coralnpu/             # local Coral source deltas (mirrored directory structure)
runtime/host/             # guest-side bootscripts, userspace runtime, tools (coralctl, mmio32)
runtime/kernel/           # Linux kernel driver (opennpux-coral)
runtime/npu/              # NPU-side program notes
rtl/wrappers/             # versioned C ABI header shared between gem5 and Coral bridge
docs/                     # ADRs, design notes, runbooks
tools/                    # build helpers, validation scripts, phase-specific test runners
tests/                    # unit tests and simulation test docs
```

## Superproject Pattern

Upstream submodules stay clean. All local modifications live under `sim/gem5/` and `sim/coralnpu/` using the **same relative paths** as the upstream trees. Before compilation, overlay scripts merge them into the submodules:

- `./sim/gem5/apply_patchset.sh` — syncs `sim/gem5/` deltas into `thirdparty/gem5/`
- `./sim/coralnpu/apply_patchset.sh` — syncs `sim/coralnpu/` deltas into `thirdparty/coralnpu/` (also handles `overlay_delete.txt` and `overlay_restore.txt`)

Pinned submodule commits are documented in `thirdparty/PINNED_COMMITS.md`.

## Core Architecture

### gem5 NPU Device (`sim/gem5/src/dev/npu/`)

`NPUDevice` is a `DmaVirtDevice` SimObject at MMIO address `0x1d000000`. It delegates execution to a pluggable backend:

- **`CoralStageABackend`** — transaction-level run-to-halt stub for smoke testing. Used during checkpoint bootstrapping (`--npu-backend=stage-a`).
- **`CoralVerilatedBackend`** — bridges to official Coral `CoreMiniAxi` Verilator model through a versioned C ABI shared library (`libcoralnpu_gem5_bridge.so`). Activated with `CORAL_NPU_BACKEND=verilated-coral`.

The backend interface (`coral_backend.hh`) defines MMIO read/write, DMA request/complete, event scheduling, and optional local EXTMEM.

### Coral Bridge (`sim/coralnpu/hw_sim/gem5_bridge/`)

Built with Bazel. Key components:
- **`coralnpu_gem5_abi.h/.cc`** — versioned C ABI (currently v6) between gem5 and the Verilator wrapper. This header is synchronized across `rtl/wrappers/`, `sim/coralnpu/`, and `sim/gem5/`.
- **`gem5_hybrid_kernels.h/.cc`** — host-side functional operator implementations (Conv2D, DepthwiseConv2D, MatMul, FullyConnected, Add, Softmax, LayerNorm) for hybrid/sampled execution modes.
- **`gem5_core_mini_axi_wrapper.h`** — wraps the Verilated `CoreMiniAxi` and drives it tick-by-tick.
- **`coral_operator.h`** / **`coral_operator_client.h`** — operator descriptor ABI and client library.

### Coral Operator ABI (`docs/design/coral_operator_abi.md`)

Three execution modes, configured before RTL starts:
- **`rtl`** — firmware calls Coral RVV/TFLM kernel directly. RTL cycle counters are authoritative.
- **`hybrid`** — firmware submits an EXTMEM descriptor through a doorbell; the bridge executes a host functional kernel and returns results. Host time != NPU latency; `modeled_cycles` uses a configurable linear model.
- **`sampled`** — firmware and driver run through Verilated Coral, but supported long operators execute via hybrid host kernels. Default bring-up mode for large graphs.

### Linux Driver (`runtime/kernel/opennpux_coral.c`)

Minimal platform driver exposing `/dev/opennpux-coral` with ioctls:
- `GET_INFO` — MMIO base, backend ID, firmware entry, DMA counters
- `GET_CAPS` — ABI version and feature negotiation
- `START` — asynchronous firmware launch
- `RUN` — poll-based completion wait
- `RESET` — device reset

Provides bounded non-cached mmap of the DT-reserved shared DMA window.

### Userspace Runtime (`runtime/host/`)

- **`coral_runtime.c`** — transport abstraction with auto-selection: prefers `/dev/opennpux-coral` driver, falls back to `/dev/mem` for legacy checkpoint tests.
- **`coralctl.c`** — CLI frontend for the runtime (`info`, `run`, `dma-test`, `vector-add`, `mem-info`, etc.).
- **`mmio32.c`** — minimal `/dev/mem` MMIO helper for early bring-up.

### Checkpoint System

Simulation uses a two-phase flow:
1. **Bootstrap**: Boot ARM Linux with `--npu-backend=stage-a`, run a minimal init script, take a checkpoint at `checkpoint/coralnpu_ckpt/booted/`.
2. **Restore**: Restore from checkpoint with the real NPU backend and test bootscript.

Checkpoint invalidation is automatic when the disk image, kernel image, kernel init path, or kernel cmdline changes. Force rebuild with `CORAL_REBUILD_CKPT=1`.

## Common Commands

### Build

```bash
# Build the Coral RTL bridge and firmware (required before any RTL run)
./tools/coralnpu/phase2_prepare_bazel.sh
./tools/coralnpu/phase2_build_bridge.sh

# Build only the RVV Highmem MobileNet bridge variant
./tools/coralnpu/build_rvv_mobilenet.sh

# Apply gem5 overlay and compile gem5
./sim/gem5/apply_patchset.sh
cd thirdparty/gem5 && scons build/ARM/gem5.opt -j$(nproc)

# Build guest userspace tools
./tools/guest_tools/build_coralctl.sh
./tools/guest_tools/build_mmio32.sh

# Build the ARM64 Linux kernel
./tools/kernel/build_arm64_kernel.sh

# Build the kernel driver module
./tools/kernel/build_opennpux_coral_ko.sh

# Install guest tools into the ARM64 disk image
sudo ./tools/guest_tools/install_coralctl_to_image.sh /path/to/arm64.img
sudo ./tools/guest_tools/install_mmio32_to_image.sh /path/to/arm64.img
```

### Run Simulation

```bash
# Stage-A smoke test (default backend)
./thirdparty/gem5/run_multicore.sh

# RTL backend (requires built bridge)
CORAL_NPU_BACKEND=verilated-coral ./thirdparty/gem5/run_multicore.sh

# Force checkpoint rebuild
CORAL_REBUILD_CKPT=1 ./thirdparty/gem5/run_multicore.sh

# With custom kernel
CORAL_KERNEL_IMAGE=/path/to/vmlinux-4.19.325-opennpux \
  CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
  CORAL_NPU_BACKEND=verilated-coral \
  ./thirdparty/gem5/run_multicore.sh
```

### Test Runners (all under `tools/coralnpu/`)

```bash
./tools/coralnpu/run_dma_smoke_test.sh       # Phase 2 DMA smoke
./tools/coralnpu/run_driver_dma_test.sh       # Phase 3 driver DMA
./tools/coralnpu/run_coralctl_test.sh         # coralctl integration
./tools/coralnpu/run_command_test.sh          # Phase 4 command submission
./tools/coralnpu/run_model_test.sh            # Phase 4/5 model dispatch
./tools/coralnpu/run_custom_rtl_test.sh       # Phase 5 custom RTL A/B
./tools/coralnpu/run_rvv_mobilenet_test.sh    # RVV Highmem MobileNet
./tools/coralnpu/test_hybrid_kernels.sh       # Hybrid kernel unit tests
```

### ABI Validation

```bash
./tools/coralnpu/phase2_check_abi.sh          # Verify bridge ABI consistency
./tools/coralnpu/check_command_abi.sh         # Verify command descriptor ABI
./tools/coralnpu/check_mobilenet_abi.sh       # Verify MobileNet operator ABI
```

### Unit Tests

```bash
# Build and run host-side runtime tests
./tools/guest_tools/build_coral_runtime_tests.sh
./build/tests/unit/runtime_host/coral_runtime_test
```

## Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `CORAL_NPU_BACKEND` | `stage-a` | NPU backend selection (`stage-a` or `verilated-coral`) |
| `CORAL_RTL_BRIDGE` | `build/coralnpu/libcoralnpu_gem5_bridge.so` | Path to the Verilator bridge shared library |
| `CORAL_RTL_FIRMWARE` | `build/coralnpu/gem5_smoke_halt.elf` | Firmware ELF loaded into Coral TCM |
| `CORAL_RTL_TICK_PERIOD` | `1ns` | gem5 time per RTL cycle |
| `CORAL_RTL_CYCLES_PER_EVENT` | `1` | RTL cycles per gem5 event quantum |
| `CORAL_REBUILD_CKPT` | `0` | Force checkpoint rebuild |
| `CORAL_OPERATOR_MODE` | `rtl` | Operator execution mode (`rtl`, `hybrid`, `sampled`) |
| `CORAL_HYBRID_OPS_PER_CYCLE` | `1` | Hybrid latency model: operations per modeled cycle |
| `CORAL_HYBRID_BYTES_PER_CYCLE` | `16` | Hybrid latency model: memory bytes per modeled cycle |
| `CORAL_HYBRID_FIXED_CYCLES` | `0` | Hybrid latency model: fixed overhead cycles |
| `CORAL_FAST_DMA_EVENT_BATCH` | — | Batch size for fast-DMA event coalescing |
| `CORAL_SAMPLED_RTL_OPS` | — | Force specific ops to RTL in sampled mode |

## Development Phases

The project follows a phased bring-up plan documented in `docs/design/system_level_plan.md`:
- **Phase 1** (complete): Standalone Coral build environment validation
- **Phase 2** (complete): gem5 SoC bridge to official Coral RTL (MMIO, DMA, firmware loading)
- **Phase 3** (complete): Linux kernel driver (`/dev/opennpux-coral`) and userspace runtime with transport abstraction
- **Phase 4** (in progress): Versioned command descriptors, `.npxm` model container, multi-operator dispatch
- **Phase 5** (in progress): Custom RTL accelerator (3-cycle MAC), operator A/B testing against official Coral
- **RVV Highmem MobileNet** (in progress): `RvvCoreMiniHighmemAxi` with 8 MiB EXTMEM for full MobileNet V1 execution

Detailed progress is tracked in `docs/design/current_progress.md`.

## Coding Standards

**[Shell Scripting Standard](docs/standards/shell-scripting.md)** — mandatory template for production shell scripts. Key rules:

- Resolve the superproject root via `SCRIPT_DIR`/`ROOT_DIR`, never `$PWD`
- Use `set -eu`, never `set -e` alone
- Tag interdependent scripts with `@kernel-config-spec` / `@synchronized-with` version labels
- Document environment variables, output artifacts, and the call chain in every script header
- Never hardcode user home directories; use `$IMAGE_PATH` with a default for disk images
- Use `$ROOT_DIR` or relative paths (`./` / `../../`) for build artifacts, never `$PWD/build/`

When modifying a script, add the standard header if it is missing.

## Important Constraints

- The build host is **x86-64 Linux**. ARM64 cross-compilation is used for guest binaries.
- The bridge shared library must **not** link `libsystemc` — gem5 already links its own SystemC, and a second copy causes conflicts. The build script uses a non-SystemC Verilator runtime.
- The canonical ABI header (`coralnpu_gem5_abi.h`) exists in three places that must stay synchronized. Run `phase2_check_abi.sh` after any ABI change.
- Checkpoints carry format version metadata. Changing the resume mechanism requires a version bump and forced rebuild.
- Upstream submodules should never carry long-lived local product code — all local changes belong in `sim/`.
