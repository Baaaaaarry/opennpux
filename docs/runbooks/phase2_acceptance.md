# Phase 2 Acceptance Runbook

This runbook validates the Phase-2 gem5 + Coral RTL integration on the x86
Linux development host.

## Scope

Phase 2 is accepted when the system demonstrates:

- upstream submodules remain overlaid by `sim/gem5` and `sim/coralnpu`
- Coral bridge and smoke firmware build reproducibly
- gem5 D9300 full-system boot checkpoint can be created and reused
- Linux can control the NPU through `coralctl`
- Verilated Coral can run to halt
- coherent CPU/NPU shared-memory DMA passes
- shared-window commands can inspect and update the reserved buffer
- unsupported in-flight checkpointing and invalid DMA windows fail safely
- AXI master adapter enforces single-outstanding ordering with burst support

Multiple outstanding AXI IDs are intentionally out of Phase-2 acceptance. The
Phase-2 bridge backpressures `AR`, `AW`, and `W` while a request is outstanding.

## Host Build

Run from the superproject root:

```sh
git submodule update --init --recursive
./sim/gem5/apply_patchset.sh
./sim/coralnpu/apply_patchset.sh
./tools/coralnpu/phase2_prepare_bazel.sh
./tools/coralnpu/phase2_check_abi.sh
./tools/coralnpu/phase2_check_overlay_boundary.sh
./tools/coralnpu/phase2_test_axi_adapter.sh
./tools/coralnpu/phase2_build_bridge.sh
```

Expected:

```text
Coral gem5 ABI headers match
Coral gem5 adapter is isolated from upstream AXI drivers
PASS: gem5 Coral AXI master adapter
```

Build guest tools and install them into the runtime image:

```sh
./tools/guest_tools/build_coralctl.sh
sudo ./tools/guest_tools/install_coralctl_to_image.sh \
  $IMAGE_PATH/ubuntu-18.04-arm64-docker.img
```

Build gem5:

```sh
cd thirdparty/gem5
scons build/ARM/gem5.opt -j"$(nproc)"
```

## Boot Checkpoint

Create or reuse the boot checkpoint:

```sh
cd thirdparty/gem5
CORAL_REBUILD_CKPT=1 CORAL_NPU_BACKEND=stage-a ./run_multicore.sh
```

Expected terminal output:

```text
Boot checkpoint saved at .../checkpoint/coralnpu_ckpt/booted
```

For ordinary Phase-2 tests, do not set `CORAL_REBUILD_CKPT=1`; the resume
script is injected through `m5 readfile`, so guest test scripts can change
without rebuilding the boot checkpoint.

## MMIO And Shared Window Smoke

Use the `coralctl` resume script:

```sh
cd thirdparty/gem5
CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_BRIDGE="$PWD/../../build/coralnpu/libcoralnpu_gem5_bridge.so" \
CORAL_RTL_FIRMWARE="$PWD/../../build/coralnpu/gem5_smoke_halt.elf" \
CORAL_RESUME_BOOTSCRIPT="$PWD/configs/coralnpu/coralctl-test.rcS" \
./run_multicore.sh
```

Expected guest output:

```text
backend=verilated-coral
status=0x00000001
[coralctl-test] PASS
```

Shared-window commands should work in the guest:

```sh
coralctl mem-info
coralctl mem-clear
coralctl mem-write32 0x0 0x2a
coralctl mem-read32 0x0
```

Expected:

```text
shared_base=0x8ff00000
shared_size=0x00001000
shared_clear=PASS
shared[0x00000000]=0x0000002a
```

## Coherent DMA Smoke

Run with the DMA smoke firmware and resume script:

```sh
cd thirdparty/gem5
CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_BRIDGE="$PWD/../../build/coralnpu/libcoralnpu_gem5_bridge.so" \
CORAL_RTL_FIRMWARE="$PWD/../../build/coralnpu/gem5_dma_smoke.elf" \
CORAL_RESUME_BOOTSCRIPT="$PWD/configs/coralnpu/dma-smoke.rcS" \
./run_multicore.sh
```

Expected guest output:

```text
dma_result=42
dma_magic=0x4e505544
dma_requests=4
dma_completions=4
dma_errors=0
dma_state=0x00000000
dma_test=PASS
```

Expected host debug output with `--debug-flags=NPUDevice`:

```text
Coral DMA start count=1 type=read
Coral DMA complete count=1
Coral DMA start count=4 type=write
Coral DMA complete count=4
```

## Acceptance Criteria

Phase 2 passes when all of these are true:

- `phase2_build_bridge.sh` completes without unresolved `sram_*` symbols and
  without linking `libsystemc`
- `phase2_test_axi_adapter.sh` passes
- `coralctl-test` passes from a restored boot checkpoint
- `dma-test` passes from a restored boot checkpoint
- `coralctl mem-*` commands can inspect and modify the shared window
- `dma_errors=0` for the normal smoke path
- `dma_state=0x00000000` after tests complete
- no new boot checkpoint is required for resume-script-only changes

## Failure Triage

- `Unable to load Coral RTL bridge`: rebuild with `phase2_build_bridge.sh` and
  verify `CORAL_RTL_BRIDGE`.
- `undefined symbol: sram_init`: stale bridge artifact; rerun
  `phase2_build_bridge.sh`.
- `Coral NPU did not halt`: verify `CORAL_RTL_FIRMWARE` points at the intended
  ELF and that the bridge ABI check passes.
- `dma_errors > 0`: Coral firmware accessed outside the configured EXTMEM
  window; inspect `shared_base`, `shared_size`, and firmware addresses.
- `dma_state != 0`: a request is still pending or an active DMA has not
  completed; do not checkpoint this run.
- `Cannot serialize Coral NPU while RTL/DMA is in flight`: expected guard; let
  the test reach quiescence before checkpointing.
