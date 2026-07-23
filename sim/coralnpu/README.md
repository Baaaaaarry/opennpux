# Coral NPU Integration Assets

This directory is the authoritative home for all local Coral source deltas.
The `thirdparty/coralnpu` submodule should stay aligned with official upstream
Coral, and local simulation-oriented source changes should be represented here
using the same relative paths they have in upstream Coral.

Workflow:

1. Update `thirdparty/coralnpu` by submodule.
2. Place any added or modified Coral files under `sim/coralnpu/` using the
   same relative paths they have in upstream Coral.
3. Run `apply_patchset.sh` to merge `sim/coralnpu` into `thirdparty/coralnpu`.
4. Build from `thirdparty/coralnpu`.

difference between CoreMini & RvvCoreMiniHignMemAxi

CoreMiniAxi / coralmini
  |
  ├── CoreMini RTL
  ├── AXI Slave
  ├── AXI Master
  ├── ITCM
  ├── DTCM
  └── CSR

RvvCoreMiniHighmemAxi
  |
  ├── RvvCoreMini RTL
  ├── RVV / Vector execution path
  ├── AXI Slave
  ├── AXI Master
  ├── ITCM
  ├── DTCM
  ├── CSR
  ├── Highmem address mapping
  └── Local EXTMEM