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

## Scope

This superproject intentionally excludes macOS-specific build caches, ARM guest
 image build helpers, and transient simulation outputs from the tracked tree.
