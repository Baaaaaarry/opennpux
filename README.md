# gem5 Coral x86 Superproject

This repository is the x86-oriented superproject for system-level Coral NPU
simulation work.

## Layout

- `thirdparty/gem5`: pinned gem5 submodule
- `thirdparty/coralnpu`: pinned official Coral submodule
- `sim/gem5`: gem5-side patches, overlays, and wrapper notes
- `sim/coralnpu_sim`: Coral standalone simulation notes and scripts
- `runtime/host`: host-side bootscripts and runtime bring-up assets
- `runtime/npu`: NPU-side program placeholders and notes
- `rtl/wrappers`: wrapper code and integration shims for RTL co-simulation
- `docs`: ADRs, design notes, and runbooks
- `tools/coralnpu`: phase-1 Coral validation helpers

## Workflow

1. Update submodules to the pinned commits recorded in
   `thirdparty/PINNED_COMMITS.md`.
2. Apply the gem5 patch set in `sim/gem5/patches`.
3. Copy or merge new-file overlays from `sim/gem5/overlay` into the gem5
   worktree when a patch alone is insufficient.
4. Use the runbooks under `docs/runbooks` to validate phase-1 and later system
   flows on an x86 Linux host.

## Scope

This superproject intentionally excludes macOS-specific build caches, ARM guest
 image build helpers, and transient simulation outputs from the tracked tree.
