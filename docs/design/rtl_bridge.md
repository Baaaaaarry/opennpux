# gem5 to Coral RTL Bridge

## Implemented milestone

The RTL bridge baseline replaces the placeholder backend with a
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
- single-outstanding external AXI reads and writes through gem5 coherent DMA,
  including INCR bursts

The bridge uses the non-SystemC Verilator C++ model and runtime. Linking
Accellera SystemC into `gem5.opt` through the shared library is prohibited
because gem5 already provides its own SystemC implementation and the two
lifecycles conflict during process teardown.

## Build

Run on x86 Linux:

```bash
./tools/coralnpu/prepare_coral_bazel.sh
./tools/coralnpu/check_rtl_bridge_abi.sh
./tools/coralnpu/check_overlay_boundary.sh
./tools/coralnpu/test_axi_adapter.sh
./tools/coralnpu/build_rtl_bridge.sh
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

Checkpoint support is intentionally quiescent-only in the RTL bridge baseline. The NPU device
drains only when the RTL bridge has no scheduled event, no pending AXI/DMA
request, and no active gem5 DMA. Serializing with in-flight RTL/DMA state is
rejected instead of writing an unrecoverable checkpoint. Existing boot
checkpoints remain compatible; newly added DMA counters default to zero when
they are absent from an older checkpoint.

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

The current transport supports one outstanding AXI request. INCR read and
write bursts up to 256 beats and 4096 bytes are issued as one coherent gem5
DMA. Reads are returned to Coral as ordered AXI `R` beats with `RLAST` only on
the final beat. Writes accept independent `AW` and `W` arrival, cache all write
beats until `WLAST`, validate the burst shape and strobes, and then submit one
coherent DMA write. `AR`, `AW`, and `W` are backpressured while a request is
outstanding, making the single-outstanding RTL bridge contract explicit.
Multiple outstanding IDs remain a future performance extension.

`test_axi_adapter.sh` runs a signal-level regression without Linux or
the Coral core. It covers independent `AW`/`W` arrival, held-valid replay
prevention, deferred responses, response-ready changes after a rising-edge
handshake, read burst response retirement, and write burst collection when the
`W` channel arrives before `AW`. Unsupported burst types, transfers crossing
the 16-byte AXI data word, incorrect `WLAST`, short write bursts, 4 KiB
crossing bursts, and partial strobes are rejected with AXI `SLVERR` instead of
being forwarded as invalid gem5 DMA requests. The same test executes 64
deterministic randomized read transactions and 64 write transactions with
channel delays and response backpressure.

The detailed failure analysis and AXI timing resolution are documented in
`docs/design/dma_root_cause_report.md`.

## Coherent DMA smoke

The full-system test reserves `0x8ff00000-0x8ff00fff` in the generated device
tree. Coral EXTMEM address `0x20000000` maps onto that SoC physical window.
`coralctl dma-test` writes operands `7` and `35`, starts
`gem5_dma_smoke.elf`, and verifies that Coral writes back `42` and
`0x4e505544`.

The NPU shell also exposes a `dma_errors` CSR. Requests outside the configured
EXTMEM-to-shared-memory window are rejected by the shell and completed back to
Coral as AXI `SLVERR`; gem5 no longer terminates the whole simulation for this
class of firmware/runtime bug. Valid DMA requests and completions remain
tracked separately from rejected requests.

Before a Linux kernel driver is available, `coralctl` owns the RTL bridge
shared-buffer management contract. It discovers the reserved window through
the shell CSRs and provides bounded `mem-info`, `mem-clear`, `mem-read32`, and
`mem-write32` commands. The DMA smoke path uses the same mapper, so manual
buffer inspection and automated smoke tests exercise the same guest-side ABI.

## RTL bridge acceptance

Run `docs/runbooks/rtl_bridge_acceptance.md` on x86 Linux. The RTL bridge baseline is complete when
bridge build, adapter regression, restored-checkpoint MMIO, shared-window
commands, Verilated halt, and coherent DMA smoke all pass.
