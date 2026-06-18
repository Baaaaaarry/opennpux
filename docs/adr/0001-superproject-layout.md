# ADR 0001: Superproject Layout

## Status

Accepted

## Context

The original working tree was a modified gem5 repository with Coral-related
source edits, scripts, docs, caches, and host-specific artifacts mixed into the
upstream tree. That made upstream sync and x86 migration harder.

## Decision

Use a superproject layout:

- Track `gem5` and `coralnpu` as submodules under `thirdparty/`.
- Keep local integration logic outside those submodules.
- Represent gem5 deltas as patches plus overlays under `sim/gem5/`.
- Keep runtime scripts, RTL wrappers, and validation tooling in top-level
  project directories.

## Consequences

- Upstream sync is explicit and reviewable.
- Phase-specific tooling becomes host-agnostic and easier to move to x86.
- Existing local modifications require patch/overlay application before use.
