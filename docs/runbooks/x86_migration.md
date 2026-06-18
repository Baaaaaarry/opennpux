# x86 Migration Runbook

## Goal

Move the active Coral + gem5 system simulation flow to an x86 Linux host.

## Steps

1. Clone this superproject and initialize submodules.
2. Check out the pinned commits from `thirdparty/PINNED_COMMITS.md`.
3. Keep local gem5 and Coral source changes under `sim/gem5/` and
   `sim/coralnpu/` using the same relative paths as upstream.
4. Run `sim/gem5/apply_patchset.sh` and `sim/coralnpu/apply_patchset.sh` to
   merge those local source deltas into `thirdparty/gem5` and
   `thirdparty/coralnpu`.
5. Run the Coral phase-1 tooling from `tools/coralnpu/`.
6. Continue RTL bridge and runtime work from this repository, not directly in
   the submodules.

## Excluded From Migration

- macOS-only caches and failed Bazel state
- ARM guest image builder artifacts from the exploratory bring-up phase
- transient `build/`, `m5out/`, and `simout/` outputs

## macOS-specific limitation being left behind

The current macOS workspace shows a case-sensitivity conflict inside the Coral
tree (`Sram.scala` vs `SRAM.scala`). That leaves the submodule dirty on this
host even after reverting local edits. This is one of the reasons the
authoritative build and validation environment is being moved to x86 Linux.
