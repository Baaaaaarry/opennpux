# Phase 2: gem5 to Coral RTL Bridge

## Implemented milestone

The first Phase-2 milestone replaces the placeholder backend with a
runtime-loaded bridge to the official Coral `CoreMiniAxiWrapper`.

The split is:

- `sim/coralnpu/hw_sim/gem5_bridge`
  - Bazel shared-library adapter built with the official Verilated model
- `rtl/wrappers/coralnpu_gem5_abi.h`
  - versioned C ABI contract between Coral and gem5
- `sim/gem5/src/dev/npu/CoralVerilatedBackend`
  - `dlopen` loader, AXI slave/MMIO forwarding, and gem5 event scheduling

The bridge currently supports:

- ITCM, DTCM, and CSR accesses through the Coral AXI slave interface
- reset/start control through the real Coral CSR implementation
- periodic RTL evaluation on the gem5 event queue
- halted/WFI detection
- the official simulator mailbox behavior for external AXI accesses

## Build

Run on x86 Linux:

```bash
./tools/coralnpu/phase2_prepare_bazel.sh
./tools/coralnpu/phase2_check_abi.sh
./tools/coralnpu/phase2_build_bridge.sh
./sim/gem5/apply_patchset.sh
```

Then build gem5 normally:

```bash
cd thirdparty/gem5
scons build/ARM/gem5.opt -j"$(nproc)"
```

## Full-system smoke

The existing boot checkpoint can be reused because the resume script is
injected with `m5 readfile`. The backend is selected when the restored gem5
configuration is instantiated:

```bash
CORAL_NPU_BACKEND=verilated-coral \
CORAL_RTL_BRIDGE="$PWD/build/coralnpu/libcoralnpu_gem5_bridge.so" \
./thirdparty/gem5/run_multicore.sh
```

Use `--debug-flags=NPUDevice` when invoking gem5 directly to see bridge load,
MMIO, stepping, and halt messages.

## Remaining Phase-2 work

The official wrapper invokes AXI master callbacks synchronously. gem5 coherent
DMA completes asynchronously. The next Phase-2 increment must therefore add a
request/completion queue that pauses RTL evaluation while a DMA is outstanding,
then resumes the AXI response after the gem5 DMA callback fires.

Do not implement this path with `PhysicalMemory::read` or `/dev/mem`-style
backdoor accesses. Those paths bypass CPU caches and would invalidate the
memory-consistency model. Until the queued DMA bridge is implemented, external
AXI accesses retain the official mailbox-only simulator behavior.
