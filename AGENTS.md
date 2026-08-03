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

