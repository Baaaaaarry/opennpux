# OpenNPUX Codex Collaboration Guide

This repository is developed by multiple people with separate Codex accounts.
The core rule is:

**Do not let multiple accounts edit the same working directory or the same
branch. Share repository rules, task plans, and interface contracts. Work in
independent branches or worktrees, then merge through Pull Requests.**

## Repository Model

- `thirdparty/gem5` and `thirdparty/coralnpu` are upstream submodules.
- Local gem5 changes live under `sim/gem5` with the same relative paths as
  `thirdparty/gem5`.
- Local CoralNPU changes live under `sim/coralnpu` with the same relative paths
  as `thirdparty/coralnpu`.
- Build scripts may sync overlays into `thirdparty/*` worktrees. Do not commit
  generated or synced submodule dirtiness in the superproject.
- Architecture, ABI, memory-map, task descriptor, and runbook changes belong in
  `docs/`.
- Runtime host tools live under `runtime/host` and `tools/`.
- NPU-side firmware, bridge, and RTL integration overlays live under
  `sim/coralnpu`.

## Team Workflow

1. Create or claim a GitHub Issue before development.
2. For non-trivial work, write or update a design note under `docs/design/`.
3. Get the design reviewed before changing shared interfaces.
4. Create a dedicated branch or worktree for the Issue.
5. Keep Codex scoped to the claimed area.
6. Commit small, reviewable increments.
7. Open a Pull Request with validation logs.
8. Run Codex review plus human review.
9. Merge only after CI or documented manual acceptance passes.

Recommended branch naming:

- `feature/<issue-id>-<short-name>`
- `fix/<issue-id>-<short-name>`
- `docs/<issue-id>-<short-name>`
- `experiment/<issue-id>-<short-name>` for disposable prototypes

Use a separate worktree when working on multiple tasks:

```bash
git fetch origin
git worktree add ../opennpux-<task> -b feature/<issue-id>-<short-name> origin/main
```

## Codex Operating Rules

- Read this file and the nearest nested `AGENTS.md` before editing.
- Inspect existing code first; do not infer interfaces from memory.
- Prefer `rg` and `rg --files` for search.
- Use `apply_patch` for manual file edits.
- Do not modify `thirdparty/gem5` or `thirdparty/coralnpu` as the source of
  truth. Put lasting changes in `sim/gem5` or `sim/coralnpu`.
- Do not revert user or teammate changes unless explicitly requested.
- Do not use destructive git commands such as `git reset --hard` or
  `git checkout --` without explicit approval.
- If an interface changes, update both code and docs in the same PR.
- If a test cannot be run locally, state why and provide the exact command for
  the x86/GB10 validation machine.

## Shared Interface Change Rules

These changes require a design note and reviewer agreement before code merge:

- SoC memory map or address aperture changes.
- NPU MMIO CSR layout changes.
- Shared DMA window base, size, synchronization, or coherency semantics.
- TCB/operator descriptor layout changes.
- Kernel UAPI or `coralctl` CLI behavior changes.
- gem5 checkpoint flow, guest image, or kernel boot contract changes.
- RTL bridge C ABI changes.
- MobileNet/Transformer operator semantics or checksum acceptance changes.

Design notes should be placed under `docs/design/` and, once accepted, linked
from the Issue and PR. If a design is superseded, add a short note instead of
rewriting history silently.

## Build And Validation Baseline

Apply overlays before building:

```bash
./sim/gem5/apply_patchset.sh
./sim/coralnpu/apply_patchset.sh
```

Core validation commands:

```bash
./tools/coralnpu/build_rtl_bridge.sh
./tools/coralnpu/build_rvv_mobilenet_partial.sh
CORAL_MOBILENET_PARTIAL_DEBUG=1 ./tools/coralnpu/run_rvv_mobilenet_partial.sh
CORAL_OPERATOR_MODE=hybrid ./tools/coralnpu/run_rvv_mobilenet_test.sh
```

Use the runbooks under `docs/runbooks/` for baseline acceptance. PRs must
include the exact commands run, host type, commit, and PASS/FAIL output.

## Ownership Boundaries

- gem5 SoC/CPU/device modeling: `sim/gem5`, `docs/design`, `docs/runbooks`.
- CoralNPU RTL/bridge/firmware overlays: `sim/coralnpu`, `rtl/wrappers`.
- Runtime and guest tools: `runtime/host`, `tools/guest_tools`,
  `tools/coralnpu`.
- Kernel driver/UAPI: `runtime/kernel`, `runtime/host/include/opennpux`.
- Tests: `tests/unit`, `tests/sim`, and corresponding runbooks.

Cross-boundary changes are allowed, but the PR description must explain the
interface and validation impact.

## Commit And PR Standards

- Keep commits focused. Avoid mixing refactors, generated files, and behavior
  changes.
- Do not commit `m5out`, `simout`, Bazel caches, build outputs, or local disk
  images.
- Do not commit submodule working tree dirtiness unless the submodule pointer is
  intentionally updated and `thirdparty/PINNED_COMMITS.md` is updated.
- PRs must include validation evidence or an explicit reason validation is
  deferred.
- PRs touching ABI, memory map, checkpoint, kernel, or bridge code require one
  reviewer familiar with that subsystem.


---

*The section above is the team collaboration guide (from origin/main); the section below is the repository guide for AI coding agents (local branch). Keep both.*


# Repository Guide for AI Coding Agents

Guidance for AI coding agents working in this repository. Assumes no prior
knowledge of the project.

## Project Overview

This is the **gem5 Coral x86 Superproject** (package name `gem5-coral-x86`): a
superproject for x86-hosted system-level simulation of an ARM heterogeneous SoC
with a Google Coral NPU. The simulated platform is a D9200/D9300 SoC booting
ARM64 Linux, with a Coral NPU integrated via a Verilator RTL bridge. The NPU
executes official Coral firmware and custom accelerator RTL, exchanging
commands and tensors with the CPU through coherent shared memory.

Key facts:

- The build host is **x86-64 Linux**; guest binaries are ARM64 cross-compiled.
- The project is mostly C/C++ (gem5 SimObjects, RTL bridge, kernel driver,
  userspace runtime), POSIX shell scripts, and a small amount of Python
  (>= 3.11, e.g. `tools/models/create_sample_model.py`). Ruff is configured in
  `pyproject.toml` (line-length 100, target py311) for the Python parts.
- Build systems in play: **scons** (gem5), **Bazel 8.6.0** (Coral bridge),
  **make** (kernel module), and plain shell for everything else.
- Development follows a phased bring-up plan
  (`docs/design/system_level_plan.md`); live status is tracked in
  `docs/design/current_progress.md`. Phases 1–3 are complete; Phase 4
  (versioned command descriptors, `.npxm` model container, multi-operator
  dispatch), Phase 5 (custom RTL accelerator, A/B testing vs. official Coral),
  and RVV Highmem MobileNet are in progress.

## Repository Layout

```
thirdparty/gem5          # pinned upstream gem5 submodule (github.com/gem5/gem5, stable @ c8222cc6)
thirdparty/coralnpu      # pinned upstream google-coral/coralnpu submodule (@ 8baac418)
sim/gem5/                # local gem5 source deltas (mirrored directory structure)
sim/coralnpu/            # local Coral source deltas (mirrored directory structure)
runtime/host/            # guest-side bootscripts, userspace runtime, tools (coralctl, mmio32)
runtime/kernel/          # Linux kernel driver (opennpux_coral.c -> opennpux-coral)
runtime/npu/             # NPU-side program notes
rtl/wrappers/            # versioned C ABI header shared between gem5 and the Coral bridge
docs/                    # ADRs, design notes, runbooks, standards
tools/coralnpu/          # phase-specific build helpers and test runners
tools/guest_tools/       # guest userspace build/install scripts
tools/kernel/            # ARM64 kernel and driver build scripts
tools/models/            # sample model generation (Python)
tests/unit/              # host-side unit tests (e.g. runtime_host/coral_runtime_test.c)
tests/sim/               # simulation test docs
build/                   # build artifacts (bridge .so, firmware ELFs, kernel, images)
checkpoint/              # gem5 boot checkpoints (e.g. checkpoint/coralnpu_ckpt)
logs/, simout/           # build and simulation logs
```

## Superproject Pattern (most important convention)

Upstream submodules stay clean. **All local modifications live under
`sim/gem5/` and `sim/coralnpu/` using the same relative paths as the upstream
trees.** Before compilation, overlay scripts merge them into the submodules:

- `./sim/gem5/apply_patchset.sh` — syncs `sim/gem5/` deltas into
  `thirdparty/gem5/`
- `./sim/coralnpu/apply_patchset.sh` — syncs `sim/coralnpu/` deltas into
  `thirdparty/coralnpu/` (also handles `overlay_delete.txt` and
  `overlay_restore.txt`)

Never carry long-lived local product code inside either submodule. Pinned
submodule commits and the update policy are documented in
`thirdparty/PINNED_COMMITS.md`.

## Core Architecture

### gem5 NPU device (`sim/gem5/src/dev/npu/`)

`NPUDevice` is a `DmaVirtDevice` SimObject at MMIO address `0x1d000000`. It
delegates execution to a pluggable backend (`coral_backend.hh` defines MMIO
read/write, DMA request/complete, event scheduling, optional local EXTMEM):

- `CoralStageABackend` — transaction-level run-to-halt stub for smoke testing;
  default backend, used during checkpoint bootstrapping.
- `CoralVerilatedBackend` — bridges to the official Coral `CoreMiniAxi`
  Verilator model through a versioned C ABI shared library
  (`libcoralnpu_gem5_bridge.so`); activated with
  `CORAL_NPU_BACKEND=verilated-coral`.

### Coral bridge (`sim/coralnpu/hw_sim/gem5_bridge/`)

Built with Bazel. Key components:

- `coralnpu_gem5_abi.h/.cc` — versioned C ABI (currently v6) between gem5 and
  the Verilator wrapper. This header exists in **three places** that must stay
  synchronized: `rtl/wrappers/`, `sim/coralnpu/`, and `sim/gem5/`. Run
  `./tools/coralnpu/phase2_check_abi.sh` after any ABI change.
- `gem5_hybrid_kernels.h/.cc` — host-side functional operator implementations
  (Conv2D, DepthwiseConv2D, MatMul, FullyConnected, Add, Softmax, LayerNorm)
  for hybrid/sampled execution modes.
- `gem5_core_mini_axi_wrapper.h` — wraps the Verilated `CoreMiniAxi` and drives
  it tick-by-tick.
- `coral_operator.h` / `coral_operator_client.h` — operator descriptor ABI and
  client library (see `docs/design/coral_operator_abi.md`).

### Operator execution modes (`CORAL_OPERATOR_MODE`)

- `rtl` — firmware calls the Coral RVV/TFLM kernel directly; RTL cycle counters
  are authoritative.
- `hybrid` — firmware submits an EXTMEM descriptor through a doorbell; the
  bridge executes a host functional kernel; `modeled_cycles` uses a
  configurable linear latency model.
- `sampled` — firmware and driver run through Verilated Coral, but supported
  long operators execute via hybrid host kernels. Default bring-up mode for
  large graphs.

### Linux driver (`runtime/kernel/opennpux_coral.c`)

Minimal platform driver exposing `/dev/opennpux-coral` with ioctls `GET_INFO`,
`GET_CAPS`, `START` (async firmware launch), `RUN` (poll-based completion
wait), and `RESET`. Provides bounded non-cached mmap of the DT-reserved shared
DMA window.

### Userspace runtime (`runtime/host/`)

- `src/coral_runtime.c` — transport abstraction with auto-selection: prefers
  the `/dev/opennpux-coral` driver, falls back to `/dev/mem` for legacy
  checkpoint tests.
- `tools/coralctl.c` — CLI frontend (`info`, `run`, `dma-test`, `vector-add`,
  `mem-info`, etc.).
- `tools/mmio32.c` — minimal `/dev/mem` MMIO helper for early bring-up (needed
  because `dd` on `/dev/mem` can fail on device memory with `Bad address`).
- `bootscripts/*.rcS` — guest-side boot/test scripts; `init/opennpux-init.sh`
  is the custom init used with `CORAL_KERNEL_INIT=/sbin/opennpux-init.sh`.

### Checkpoint system

Two-phase flow:

1. **Bootstrap**: boot ARM Linux with `--npu-backend=stage-a`, run a minimal
   init script, take a checkpoint at `checkpoint/coralnpu_ckpt/booted/`.
2. **Restore**: restore from the checkpoint with the real NPU backend and the
   test bootscript.

Checkpoint invalidation is automatic when the disk image, kernel image, kernel
init path, or kernel cmdline changes. Force a rebuild with
`CORAL_REBUILD_CKPT=1`. Checkpoints carry format-version metadata — changing
the resume mechanism requires a version bump and a forced rebuild.

## Build and Test Commands

### Build

```bash
# Build the Coral RTL bridge and firmware (required before any RTL run)
./tools/coralnpu/phase2_prepare_bazel.sh     # supports BAZEL_BINARY=/path/to/bazel-8.6.0-linux-x86_64
./tools/coralnpu/phase2_check_abi.sh
./tools/coralnpu/phase2_build_bridge.sh

# Build only the RVV Highmem MobileNet bridge variant
./tools/coralnpu/build_rvv_mobilenet.sh

# Apply the gem5 overlay and compile gem5
./sim/gem5/apply_patchset.sh
cd thirdparty/gem5 && scons build/ARM/gem5.opt -j$(nproc)

# Build guest userspace tools
./tools/guest_tools/build_coralctl.sh
./tools/guest_tools/build_mmio32.sh

# Build the ARM64 Linux kernel and the driver module
./tools/kernel/build_arm64_kernel.sh
./tools/kernel/build_opennpux_coral_ko.sh

# Install guest tools into the ARM64 disk image (rebuild the boot checkpoint afterwards)
sudo ./tools/guest_tools/install_coralctl_to_image.sh $IMAGE_PATH/ubuntu-18.04-arm64-docker.img
sudo ./tools/guest_tools/install_mmio32_to_image.sh  $IMAGE_PATH/ubuntu-18.04-arm64-docker.img
```

Staged Phase-2 artifacts land in `build/coralnpu/`
(`libcoralnpu_gem5_bridge.so`, `gem5_smoke_halt.elf`, `gem5_dma_smoke.elf`,
`gem5_command_smoke.elf`).

### Run simulation

```bash
export IMAGE_PATH="${IMAGE_PATH:-$HOME/wlk/gem5_arm_linux_images}"  # once per session

# Stage-A smoke test (default backend)
./thirdparty/gem5/run_multicore.sh

# RTL backend (requires the built bridge)
CORAL_NPU_BACKEND=verilated-coral ./thirdparty/gem5/run_multicore.sh

# Force checkpoint rebuild
CORAL_REBUILD_CKPT=1 ./thirdparty/gem5/run_multicore.sh

# With custom kernel + init
CORAL_KERNEL_IMAGE=./build/kernel/vmlinux-4.19.325-opennpux \
  CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
  ./thirdparty/gem5/run_multicore.sh
```

### Test runners (all under `tools/coralnpu/`)

```bash
./tools/coralnpu/run_dma_smoke_test.sh       # Phase 2 DMA smoke
./tools/coralnpu/run_driver_dma_test.sh      # Phase 3 driver DMA
./tools/coralnpu/run_coralctl_test.sh        # coralctl integration
./tools/coralnpu/run_command_test.sh         # Phase 4 command submission (acceptance: vector_add=PASS, completed_elements=16, output_checksum=0x00000198)
./tools/coralnpu/run_model_test.sh           # Phase 4/5 model dispatch
./tools/coralnpu/run_custom_rtl_test.sh      # Phase 5 custom RTL A/B
./tools/coralnpu/run_rvv_mobilenet_test.sh   # RVV Highmem MobileNet
./tools/coralnpu/test_hybrid_kernels.sh      # hybrid kernel unit tests
```

### ABI validation

```bash
./tools/coralnpu/phase2_check_abi.sh         # bridge ABI consistency (run after any ABI change)
./tools/coralnpu/check_command_abi.sh        # command descriptor ABI
./tools/coralnpu/check_mobilenet_abi.sh      # MobileNet operator ABI
```

### Unit tests

```bash
./tools/guest_tools/build_coral_runtime_tests.sh
./build/tests/unit/runtime_host/coral_runtime_test
```

Acceptance procedures with expected outputs live in `docs/runbooks/` (e.g.
`phase2_acceptance.md`, `phase4_command_acceptance.md`,
`phase45_platform_acceptance.md`, `rvv_mobilenet_acceptance.md`).

## Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `IMAGE_PATH` | `$HOME/wlk/gem5_arm_linux_images` | Location of ARM64 disk images/kernels; most scripts also accept `CORAL_DISK_IMG` / `CORAL_KERNEL_IMAGE` overrides |
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
| `CORAL_AXI_WATCHDOG_CYCLES` | `5000000` | Bridge stall watchdog: dump AXI master channel handshake levels after this many RTL cycles without master activity |

## Code Style Guidelines

**Shell scripts** must follow
[docs/standards/shell-scripting.md](docs/standards/shell-scripting.md) —
a mandatory header template for production scripts. Key rules:

- Use `#!/bin/sh` (POSIX-compatible), not `#!/bin/bash`.
- Resolve the superproject root via `SCRIPT_DIR`/`ROOT_DIR`, never `$PWD`.
- Use `set -eu`, never `set -e` alone.
- Every script header documents: one-line purpose, the pipeline/call chain,
  environment variables, and output artifacts.
- Tag interdependent scripts with `@kernel-config-spec` / `@synchronized-with`
  version labels.
- Never hardcode user home directories; use `$IMAGE_PATH` with a default for
  disk images.
- Use `$ROOT_DIR` or relative paths (`./` / `../../`) for build artifacts,
  never `$PWD/build/`.

When modifying a script that lacks the standard header, add it.

**C/C++**: match the surrounding style of the tree you are editing — gem5-side
code follows gem5 conventions, Coral-side code follows Coral conventions. Keep
local changes inside `sim/` mirrors rather than patching submodules directly.

**Python**: Ruff, line-length 100, Python >= 3.11 (`pyproject.toml`).

## Important Constraints / Security Considerations

- **The bridge shared library must not link `libsystemc`** — gem5 already links
  its own SystemC, and a second copy causes conflicts. The bridge build uses a
  non-SystemC Verilator runtime.
- The canonical ABI header `coralnpu_gem5_abi.h` exists in three places that
  must stay synchronized (`rtl/wrappers/`, `sim/coralnpu/`, `sim/gem5/`).
  Always run `phase2_check_abi.sh` after an ABI change.
- Checkpoint format carries version metadata; changing the resume mechanism
  requires a version bump and forced rebuild (`CORAL_REBUILD_CKPT=1`).
- Guest-tool installation scripts mount and modify ARM64 disk images and
  require `sudo` — they mutate real image files under `$IMAGE_PATH`.
- Guest-side tests deliberately access `/dev/mem` and device MMIO
  (`mmio32`, runtime `/dev/mem` fallback). This is expected inside the
  simulated guest, but treat such code carefully: never run these helpers on
  the host.
- Rebuild the boot checkpoint after modifying the disk image so Linux sees the
  updated contents.
- `.gitmodules` sets `ignore = dirty` for both submodules — the overlay merge
  intentionally dirties them before builds; this is normal.

## Where to Look for More Detail

- `README.md` — quick start, phased flows, guest MMIO tool
- `CLAUDE.md` — sibling guidance file with equivalent content
- `docs/design/system_level_plan.md` — phased bring-up plan
- `docs/design/current_progress.md` — live progress tracker
- `docs/adr/0001-superproject-layout.md` — rationale for the superproject layout
- `docs/design/coral_operator_abi.md` — operator descriptor ABI and execution modes
- `docs/runbooks/` — step-by-step acceptance procedures per phase
