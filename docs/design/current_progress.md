# OpenNPUX Current Progress

## Target

The target is an ARM heterogeneous SoC simulated by gem5:

- D9200/D9300 CPUs boot Linux and run the inference framework.
- Coral NPU executes official firmware and custom accelerator RTL.
- CPU and NPU exchange commands and tensors through coherent shared memory.
- System software remains usable while the Coral RTL is extended.

## Completed

### Repository and build structure

- `thirdparty/gem5` and `thirdparty/coralnpu` remain upstream submodules.
- Local additions and modifications are mirrored under `sim/gem5` and
  `sim/coralnpu`.
- Overlay scripts merge local sources into the submodules before compilation.
- The supported build host was moved to x86-64 Linux.
- Bazel caches, distdir, output root, and transitioned outputs are handled
  locally and repeatably.

### gem5 SoC bring-up

- D9200 and D9300 full-system configurations were preserved.
- The Coral NPU is instantiated as a `DmaVirtDevice`.
- Its MMIO aperture is visible at `0x1d000000`.
- Its DMA port is connected behind the SLC when that path exists.
- Linux boot checkpoints are reusable across source and guest-script updates.
- Guest MMIO access works through the minimal `mmio32` utility.

### Stage-A model

- Implemented reset, PC start, status, ITCM, and DTCM behavior.
- Verified Linux-to-NPU MMIO access and run-to-halt control.
- Used Stage A to isolate device-tree, address-map, image, and checkpoint
  problems before introducing RTL.

### Official Coral RTL bridge

- Built the official non-SystemC Verilated `CoreMiniAxi` model.
- Added a versioned C ABI shared library between Coral and gem5.
- Avoided linking a second SystemC runtime into gem5.
- Added RTL clock scheduling on the gem5 event queue.
- Added ELF `PT_LOAD` loading into Coral TCM.
- Added backend and firmware-entry discovery registers.
- Added minimal Coral firmware that returns through the official CRT and
  reaches the real `mpause` halted state.

### Verified result

The x86 full-system run now verifies:

```text
backend=verilated-coral
status(after)=0x00000001
PASS: Coral NPU MMIO is reachable and verilated-coral execution halted
```

This proves that Linux, gem5, the runtime-loaded bridge, official Coral RTL,
the Coral toolchain output, and RTL execution are connected end to end.

The coherent DMA smoke is also complete:

```text
dma_requests=4
dma_completions=4
dma_result=42
dma_magic=0x4e505544
dma_test=PASS
```

## Problems Resolved

- Missing and incompatible macOS Coral toolchains.
- Docker daemon, registry, DNS, credential-helper, and architecture issues.
- Bazel cache loss and repeated Bazel binary downloads.
- Corrupted external repositories and `rules_java` cache state.
- Bazel visibility, toolchain-transition output paths, and strict patch
  application failures.
- Duplicate gem5 overlay sources and duplicate MMIO responder ranges.
- Linux `/dev/mem` access without a usable `devmem` utility.
- Checkpoint invalidation caused by source or script timestamps.
- gem5 and Accellera SystemC teardown conflicts.
- Empty RTL TCM causing the NPU to run forever without halting.
- Valid Coral ELF entry address zero being treated as an error.

## Remaining Work

### Phase 2 completion

- The single-outstanding asynchronous AXI-to-gem5 DMA transport is verified.
- A DT reserved-memory buffer maps Coral EXTMEM onto SoC physical memory.
- DMA smoke firmware and `coralctl dma-test` verify coherent shared-memory
  reads and writes.
- gem5-specific asynchronous AXI handling is isolated from the official Coral
  wrapper and primitives.
- A signal-level adapter test covers independent write channels, held-valid
  request suppression, and delayed response handshakes without Linux boot.
- Deterministic randomized regressions exercise 64 reads and 64 writes with
  channel skew and response backpressure.
- Unsupported DMA request shapes return AXI `SLVERR` locally instead of
  reaching gem5 as zero-length DMA operations.
- INCR read bursts up to 256 beats and 4096 bytes use one coherent gem5 DMA
  and return ordered AXI response beats through `RLAST`.
- INCR write bursts up to 256 beats and 4096 bytes accept independent `AW` and
  `W` arrival, cache write beats through `WLAST`, validate strobes and burst
  boundaries, and submit one coherent gem5 DMA write.
- Add multiple outstanding IDs.
- Define checkpoint behavior for an in-flight RTL/DMA transaction.

### Phase 3

- Replace shell control with a stable userspace runtime and then a Linux driver.
- Allocate and pin shared input, output, command, and completion buffers.
- Add interrupts or an event-driven completion path.
- Define cache maintenance rules for non-coherent configurations.

### Phase 4

- Load a real model and tensors from Linux.
- Integrate a narrow inference runtime API.
- Measure CPU scheduling, memory traffic, NPU execution, and end-to-end latency.

### Phase 5

- Add the custom RTL accelerator inside the Coral source hierarchy.
- Add instructions or command descriptors, compiler/runtime support, and RTL
  verification.
- Compare official and custom execution using the same Linux workload.

## Immediate Development Order

1. Land the `coralctl` userspace control utility.
2. Implement the single-outstanding asynchronous AXI-to-DMA bridge.
3. Add a shared-memory read/write Coral firmware test.
4. Promote the control ABI into the minimal Linux runtime/driver.
