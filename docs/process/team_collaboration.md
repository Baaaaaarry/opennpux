# Team Collaboration Process

## Principle

Multiple developers and Codex accounts must not operate on the same working
directory or the same branch at the same time. The team shares the repository,
issues, design documents, interface contracts, and validation rules. Each
developer works in an independent branch or worktree and merges through PR.

## Recommended Flow

```text
Product requirement / technical proposal
        |
GitHub Epic / Issue
        |
Design note / interface contract review
        |
+-------+----------------+----------------+
|                        |                |
Developer A             Developer B      Developer C
Codex account A         Codex account B  Codex account C
feature/gem5-device     feature/rtl-op   feature/tests
|                        |                |
+----------- Pull Request ----------------+
        |
Codex review + human review
        |
CI / manual acceptance on x86 or GB10
        |
main
```

## Issue Lifecycle

1. Create an Epic for a large phase, for example `Phase 5: Transformer operator
   support`.
2. Split the Epic into implementation Issues with a clear owner and boundary.
3. Add labels for subsystem and risk:
   `gem5`, `coralnpu`, `runtime`, `kernel`, `rtl`, `docs`, `tests`, `abi`,
   `memmap`, `checkpoint`, `performance`.
4. Before implementation, add or update the design note if the task changes a
   shared interface.
5. Move completed decisions into the repository. Do not rely on chat history as
   the source of truth.

## Design Review Flow

Use this flow for non-trivial tasks:

1. Write a draft under `docs/design/`.
2. Include scope, non-goals, affected files, interface changes, validation
   plan, and rollback strategy.
3. Request review in the Issue.
4. After agreement, mark the design as reviewed in the document header or link
   the accepted comment/PR.
5. Break the design into concrete Issues.
6. Each developer implements only the Issue they claimed.

Use ADRs under `docs/adr/` for decisions that should be stable over time, such
as repository layout, ABI strategy, or simulation mode policy.

## Interface Contract Ownership

| Contract | Owner Area | Required Docs |
| --- | --- | --- |
| SoC memory map | gem5 modeling | `README.md`, relevant runbook |
| NPU CSR/MMIO layout | gem5 + Coral bridge | design note, runtime header |
| Shared DMA window | gem5 + runtime + firmware | design note, runbook |
| TCB/operator descriptor | runtime + firmware | `docs/design/coral_operator_abi.md` |
| Kernel UAPI | kernel + runtime | UAPI header and driver runbook |
| Verilated bridge C ABI | gem5 + Coral bridge | bridge design note |
| Operator semantics | Coral firmware/RTL/runtime | operator table and acceptance test |

Any PR changing these contracts must include both implementation and
documentation updates.

## Codex Usage Rules For Team Members

Each developer should start a Codex task with:

```text
Read AGENTS.md and the nearest subsystem AGENTS.md. Work only on Issue #<id>.
Do not edit thirdparty directly as source of truth. Put lasting gem5 changes in
sim/gem5 and CoralNPU changes in sim/coralnpu. Provide validation commands and
do not touch unrelated files.
```

When using Codex for review, ask it to focus on:

- Behavioral regressions.
- ABI or memory-map incompatibility.
- Missing validation.
- Incorrect overlay/submodule handling.
- Checkpoint or image assumptions.
- Performance regressions in full RTL, hybrid, or sampled modes.

## Branch And Worktree Policy

- One task, one branch.
- One branch, one active owner.
- Use worktrees for parallel tasks.
- Rebase or merge `origin/main` before opening a PR if the branch is old.
- Do not force-push shared branches unless the branch owner explicitly
  coordinates it.

Example:

```bash
git fetch origin
git worktree add ../opennpux-123-rvv-softmax \
  -b feature/123-rvv-softmax origin/main
```

## PR Acceptance Checklist

A PR is ready only when it contains:

- Linked Issue.
- Scope and non-goals.
- Affected contracts.
- Exact validation commands and outputs.
- Any checkpoint/image/kernel assumptions.
- Documentation updates for interface changes.
- No unrelated submodule dirtiness.
- Clear follow-up work if full validation is deferred.

## Recommended Ownership Split

- gem5 SoC and NPUDevice modeling.
- CoralNPU RTL bridge and Verilator integration.
- NPU firmware/operator runtime.
- Host runtime, `coralctl`, kernel driver/UAPI.
- Guest image and kernel toolchain.
- Tests, CI, and runbook automation.
- Performance modeling and hybrid/sampled operator modes.

This split reduces conflicts because most work stays within one overlay or one
runtime area.

