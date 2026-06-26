# Phase 2 Coral DMA Integration Report

## 1. Executive summary

Phase 2 connected the official Verilated Coral `CoreMiniAxi` RTL model to the
gem5 D9200/D9300 full-system platform. Linux controls the NPU through MMIO,
while Coral external-memory AXI requests are converted into coherent gem5 DMA
transactions behind the SLC.

The initial end-to-end DMA test stalled even though individual gem5 DMA
operations completed. The defect was not in gem5 memory access or cache
coherence. It was caused by adapting Coral's synchronous testbench AXI driver
to an asynchronous, event-driven gem5 memory system without preserving all
AXI channel and clock-edge semantics.

The final test completes two reads and two writes:

```text
read  0x20000000 -> 0x8ff00000
read  0x20000004 -> 0x8ff00004
write 0x20000000 -> 0x8ff00000
write 0x20000008 -> 0x8ff00008
dma_requests=4
dma_completions=4
dma_result=42
dma_magic=0x4e505544
dma_test=PASS
```

## 2. System background

The target is a heterogeneous full-system SoC:

- ARM D9200/D9300 CPUs run Linux, userspace control software, and inference
  framework code.
- Coral executes official firmware and later custom accelerator RTL.
- CPU and NPU exchange tensors and commands through shared physical memory.
- The NPU is a gem5 DMA device whose master port is connected to the coherent
  SLC-side path.

The verified smoke test reserves a 4 KiB page at SoC address `0x8ff00000`.
Coral address `0x20000000` maps to this page. The CPU writes operands 7 and 35;
Coral firmware reads them, writes back 42, writes a completion magic value,
and halts.

## 3. Integration model

The Coral bridge is runtime-loaded by gem5 through a versioned C ABI:

```text
Coral AXI master
    -> gem5-specific AXI adapter
    -> coral_gem5_dma_request
    -> NPUDevice dmaReadVirt/dmaWriteVirt
    -> coherent gem5 memory hierarchy
    -> coral_gem5_dma_complete
    -> AXI R/B response
    -> Coral RTL resumes
```

Only one request is outstanding in the current milestone. RTL stepping pauses
while gem5 services the DMA operation and resumes after the response is queued.

## 4. Failure symptoms

The problem appeared in stages:

1. DMA counters remained zero because AXI requests were not accepted reliably.
2. The first read was repeatedly submitted because a held `ARVALID` was
   interpreted as a new request on every simulated cycle.
3. Two reads completed, but the first write did not start because `AWVALID` and
   `WVALID` were incorrectly required in the same cycle.
4. The first write completed in gem5, but Coral never issued the second write
   because the AXI `B` response handshake was missed.

The decisive host trace was:

```text
Coral DMA complete count=3
```

with no subsequent write. This proved that memory completion occurred and the
remaining fault was between DMA completion and the Coral AXI state machine.

## 5. Root cause

### 5.1 Synchronous callback versus asynchronous DMA

The upstream Coral wrapper was designed primarily for standalone Verilator and
SystemC tests. Its AXI callbacks return data or write responses immediately in
the same host simulation flow.

gem5 timing DMA is asynchronous. A request leaves the RTL wrapper, advances
through gem5 event queues and the memory hierarchy, and completes at a later
gem5 tick. The adapter must therefore retain transaction state while RTL
execution is paused and inject the response later.

### 5.2 Independent AXI write channels

AXI defines write address (`AW`) and write data (`W`) as independent channels.
They may handshake in different cycles and in either order.

The first adapter inherited the stronger testbench assumption that both valid
signals would be present together. Coral did not guarantee that behavior, so
the adapter had to latch address and data independently and submit the DMA
write only after both had been captured.

### 5.3 Valid/ready persistence

An AXI transfer occurs once on a clock edge where both `VALID` and `READY` are
high. A master may keep `VALID` asserted until acceptance. Treating every host
evaluation with `VALID=1` as a new transaction replayed requests.

The adapter now records whether an address has already been accepted and does
not create another DMA request until the previous response is complete.

### 5.4 Response handshake edge mismatch

The wrapper drives inputs during a falling-edge observer callback. Coral
samples those signals at the following rising edge.

For the write response:

1. the adapter asserted `BVALID` on a falling edge;
2. Coral sampled `BVALID && BREADY` on the next rising edge;
3. Coral immediately changed `BREADY`;
4. the adapter checked `BREADY` only on the following falling edge.

The software observer therefore missed a handshake that had already occurred.
It retained the old response and never released the write transaction.

The solution is to evaluate the driven response signals and latch whether the
next rising edge will perform a handshake. The following falling edge consumes
that latched event instead of re-reading a signal that RTL may already have
changed. The read `RVALID/RREADY` path uses the same rule.

## 6. Protocol matching implemented

The gem5-specific adapter applies these AXI rules:

- `AR`, `AW`, and `W` transfers are recognized only on `VALID && READY`.
- A held valid signal cannot replay an already accepted request.
- `AW` and `W` are accepted and stored independently.
- DMA submission occurs only after the complete request is captured.
- `RVALID` and `BVALID` remain associated with the queued response until the
  corresponding handshake is known to have occurred.
- Response handshakes are latched according to the wrapper's falling-edge
  drive and rising-edge RTL sampling model.
- RTL stepping stops while a gem5 DMA request is outstanding.
- RTL resumes only after gem5 queues the matching AXI response.

This remains a single-outstanding implementation. INCR read and write bursts
are supported by collapsing one AXI burst into one coherent gem5 DMA
transaction and expanding the response back into AXI beats when required.
Multiple IDs require additional queues and ordering rules.

## 7. Why upstream did not expose the problem

The upstream implementation is valid for its original execution environment:

- callbacks are synchronous and return responses immediately;
- testbench queues and the DUT advance under one local clock scheduler;
- no external event-driven memory hierarchy introduces an unbounded delay;
- reference testbench behavior often presents address and data close together;
- the testbench observes and updates handshake signals in its expected phase.

Our integration changed the temporal contract, not the Coral RTL protocol.
Using gem5 timing DMA introduced a second scheduler and a delayed completion
boundary. The missing state retention and edge translation only became visible
at that boundary.

## 8. Engineering resolution

The functional corrections were first validated in the existing wrapper:

- deferred request and completion callbacks;
- duplicate request suppression;
- independent `AW`/`W` capture;
- response retention;
- rising-edge handshake latching.

After the DMA smoke passed, these changes were moved into dedicated
`Gem5AxiMasterReadDriver`, `Gem5AxiMasterWriteDriver`, and
`Gem5CoreMiniAxiWrapper` classes. The upstream Coral `hw_primitives.h` and
`CoreMiniAxiWrapper` are no longer overridden.

This preserves the superproject rule:

- `thirdparty/coralnpu` remains compatible with upstream;
- gem5-specific timing adaptation lives under
  `sim/coralnpu/hw_sim/gem5_bridge`;
- only the gem5 bridge target depends on the asynchronous adapter.

## 9. Verification and limits

Verified:

- Linux MMIO control of the real Verilated Coral model;
- firmware loading and execution;
- two coherent 32-bit reads;
- two coherent 32-bit writes;
- correct CPU-visible result and completion magic;
- four requests and four completions;
- normal Coral halt after firmware completion.

Not yet covered:

- multiple outstanding transactions and multiple IDs;
- sparse partial write strobes beyond the current full-byte-lane requirement;
- DMA error injection;
- interrupt-driven completion;
- checkpoint serialization during an in-flight DMA request;
- sustained randomized AXI backpressure.

## 10. Protocol regression gate

The adapter-level unit test now runs without booting Linux. It verifies
independent `AW` and `W` arrival in both orders, held-valid replay prevention,
deferred read and write responses, and a target that drops `RREADY` or
`BREADY` immediately after the rising-edge handshake.

Deterministic regressions now cover randomized backpressure and single
outstanding INCR read/write bursts. Multiple outstanding requests remain the
next protocol extension.
