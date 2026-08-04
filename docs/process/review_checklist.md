# Review Checklist

Use this checklist for human review and Codex review.

## Correctness

- Does the change implement the Issue scope?
- Are edge cases and error paths handled?
- Does the code preserve existing Stage-A, driver, DMA, partial MobileNet, and
  hybrid MobileNet flows unless intentionally changed?

## Architecture And Interfaces

- Are memory-map changes documented?
- Are CSR/MMIO offsets and semantics compatible?
- Are shared DMA base, size, and coherency assumptions explicit?
- Are TCB/operator descriptor changes versioned or backward-compatible?
- Are bridge C ABI changes reflected on both gem5 and Coral sides?

## Overlay Discipline

- Are lasting gem5 changes under `sim/gem5`?
- Are lasting CoralNPU changes under `sim/coralnpu`?
- Are `thirdparty/*` changes only synced worktree changes or intentional
  submodule pointer updates?
- Is `thirdparty/PINNED_COMMITS.md` updated if submodule pointers changed?

## Validation

- Are exact commands and outputs included?
- Is the validation host stated, for example x86 Linux or GB10?
- If full RTL is too slow, is partial/sampled/hybrid validation provided?
- Are checkpoint and image assumptions documented?

## Maintainability

- Is the code scoped and readable?
- Are new debug logs actionable and not excessive by default?
- Are follow-up items captured in Issues rather than hidden in comments?

