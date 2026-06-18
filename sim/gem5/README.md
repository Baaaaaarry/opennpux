# gem5 Integration Assets

- `patches/`: tracked diffs against `thirdparty/gem5`
- `overlay/`: new files that are not representable by the tracked patch alone

This directory is the authoritative home for all local gem5 deltas. The
`thirdparty/gem5` submodule should stay aligned with official upstream gem5,
and custom SoC work such as D9200/D9300 integration should be represented here
instead of being committed directly inside the submodule.

Apply patches first, then merge overlays.
