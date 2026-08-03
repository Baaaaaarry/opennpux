# gem5 Overlay Rules

`sim/gem5` is the source of truth for all local gem5 changes. Keep the upstream
submodule clean and reproducible.

## Scope

- Add or modify gem5 files here using the same relative path as
  `thirdparty/gem5`.
- Sync into the submodule with:

```bash
./sim/gem5/apply_patchset.sh
```

- Build from `thirdparty/gem5` after syncing.

## Required Care

- Memory-map changes must update `README.md` and the relevant runbook.
- NPUDevice parameter changes must update CLI docs and checkpoint compatibility
  notes.
- Checkpoint behavior changes must preserve the ability to restore without
  rebuilding when only the resume script changes.
- Debug flag changes must document expected log files and useful commands.
- Do not directly edit `thirdparty/gem5` as the lasting source of truth.

## Validation

For D9200/D9300 or NPU integration changes, provide at least one of:

- Stage-A MMIO PASS.
- Verilated Coral DMA PASS.
- Driver path PASS.
- Partial MobileNet PASS.
- Full MobileNet Hybrid/Sampled PASS.

Include exact commands and relevant guest output in the PR.

