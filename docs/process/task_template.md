# Development Task Template

Use this template in GitHub Issues or as the first message to Codex.

## Task

- Issue:
- Owner:
- Branch:
- Subsystem: `gem5` / `coralnpu` / `runtime` / `kernel` / `tools` / `docs` /
  `tests`
- Priority:

## Goal

Describe the expected user-visible or platform-visible result.

## Scope

- In scope:
- Out of scope:

## Interface Impact

Mark all that apply:

- SoC memory map:
- NPU CSR/MMIO:
- Shared DMA window:
- Invocation/command/tensor descriptor:
- Kernel UAPI:
- Verilated bridge C ABI:
- Checkpoint/image/kernel boot contract:
- Operator semantics:

## Expected Files

- Files/directories Codex may edit:
- Files/directories Codex must not edit:

## Validation Plan

Commands:

```bash
# fill in exact commands
```

Expected output:

```text
# fill in key PASS lines or metrics
```

## Risks

- Compatibility:
- Performance:
- Debuggability:
- Rollback:

## Codex Prompt

```text
Read AGENTS.md and the nearest subsystem AGENTS.md. Work only on this Issue.
Keep thirdparty as upstream source; put lasting changes under sim/gem5 or
sim/coralnpu. Do not touch unrelated files. Implement the requested scope,
update docs if an interface changes, run validation when possible, and report
exact commands and outputs.
```
