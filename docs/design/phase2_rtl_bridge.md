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
- host-side ELF loading into Coral TCM before reset release
- periodic RTL evaluation on the gem5 event queue
- halted/WFI detection
- single-outstanding external AXI reads and writes through gem5 coherent DMA

The bridge uses the non-SystemC Verilator C++ model and runtime. Linking
Accellera SystemC into `gem5.opt` through the shared library is prohibited
because gem5 already provides its own SystemC implementation and the two
lifecycles conflict during process teardown.

## Build

Run on x86 Linux:

```bash
./tools/coralnpu/phase2_prepare_bazel.sh
./tools/coralnpu/phase2_check_abi.sh
./tools/coralnpu/phase2_check_overlay_boundary.sh
./tools/coralnpu/phase2_test_axi_adapter.sh
./tools/coralnpu/phase2_build_bridge.sh
./sim/gem5/apply_patchset.sh
```

The build stages both `build/coralnpu/libcoralnpu_gem5_bridge.so` and the
minimal smoke firmware `build/coralnpu/gem5_smoke_halt.elf`. The firmware
returns from `main`, allowing the Coral CRT to execute `mpause` and assert the
real halted status bit without requiring an external interrupt.

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

## AXI master DMA bridge

The gem5 bridge uses dedicated asynchronous AXI master drivers rather than
modifying the official wrapper's synchronous callbacks. When Coral issues an
external access:

1. the wrapper captures one AXI request and deasserts its ready signal
2. `coral_gem5_step` returns a DMA-wait result
3. `NPUDevice` submits `dmaReadVirt` or `dmaWriteVirt`
4. the RTL event stream remains paused while gem5 memory timing completes
5. the completion callback injects the AXI response and resumes RTL

The DMA port is connected on the SLC-side coherent path in the D9200/D9300
configurations. The implementation does not use `PhysicalMemory::read` or
other cache-bypassing backdoors.

The current transport intentionally supports one outstanding AXI request and
one beat of up to 16 bytes. Burst splitting, multiple IDs, error propagation,
and checkpointing an in-flight transaction remain future increments.

`phase2_test_axi_adapter.sh` runs a signal-level regression without Linux or
the Coral core. It covers independent `AW`/`W` arrival, held-valid replay
prevention, deferred responses, and response-ready changes after a rising-edge
handshake.

The detailed failure analysis and AXI timing resolution are documented in
`docs/design/phase2_dma_root_cause_report.md`.

## Coherent DMA smoke

The full-system test reserves `0x8ff00000-0x8ff00fff` in the generated device
tree. Coral EXTMEM address `0x20000000` maps onto that SoC physical window.
`coralctl dma-test` writes operands `7` and `35`, starts
`gem5_dma_smoke.elf`, and verifies that Coral writes back `42` and
`0x4e505544`.

## Remaining Phase-2 work

- promote the fixed reserved-memory smoke page into driver-managed allocation
- add burst and multiple-outstanding support after the single-beat path is
  verified
