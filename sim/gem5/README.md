# gem5 Integration Assets

- `configs/`, `src/`, `tests/`, `scripts/`, ...: local gem5 deltas kept in the
  same directory structure as upstream `thirdparty/gem5`
- `apply_patchset.sh`: sync helper that merges these mirrored directories into
  the gem5 submodule worktree for compilation

This directory is the authoritative home for all local gem5 deltas. The
`thirdparty/gem5` submodule should stay aligned with official upstream gem5,
and custom SoC work such as D9200/D9300 integration should be represented here
instead of being committed directly inside the submodule.

Workflow:

1. Update `thirdparty/gem5` by submodule.
2. Place any added or modified gem5 files under `sim/gem5/` using the same
   relative paths they have in upstream gem5.
   Top-level files such as `SConstruct` or `run_multicore.sh` can also be
   mirrored here when they are part of the local gem5 delta.
3. Run `apply_patchset.sh` to merge `sim/gem5` into `thirdparty/gem5`.
4. Build from `thirdparty/gem5`.
