# DS4 Readiness

This note tracks the gap between the current gem5 + Coral NPU environment and
running the DS4 end-to-end workload.

## Current State

Working:

- D9200/D9300 Linux full-system boot with reusable checkpoint.
- Coral NPU MMIO discovery through the guest.
- Official Verilated Coral backend loaded by gem5.
- Coherent NPU DMA into the reserved shared-memory window.
- `coralctl` runtime API and CLI smoke tests.
- `/dev/opennpux-coral` UAPI and loadable platform driver.
- Driver-bounded shared-window mmap.
- Asynchronous START and poll-based completion delivery.
- Versioned shared-memory command descriptor and tensor-region layout.
- End-to-end multi-buffer vector-add submission through Coral firmware.
- Generic model container with real tensors and mixed software/RTL nodes.
- Synthesizable custom MAC with system-level A/B validation.

Not yet complete:

- hardware interrupt delivery instead of the current kernel polling worker
- DS4 model artifact format and graph partitioning
- Coral/custom RTL firmware contract for DS4 operators

## Distance To DS4

The heterogeneous SoC platform path is complete through model loading and
custom RTL execution. Remaining DS4 work is workload-specific graph/operator
support rather than platform plumbing.

Estimated engineering chunks:

1. Define DS4 tensor metadata and required operator subset.
2. Add DS4 operator firmware/runtime tests.
3. Add DS4 graph partitioning to the generic model artifact.
4. Integrate the DS4 host inference adapter.

## Completed Platform Baseline

The driver-backed coherent DMA baseline is complete:

```text
transport=driver
backend=verilated-coral
status=0x00000001
dma_result=42
dma_magic=0x4e505544
dma_requests=4
dma_completions=4
dma_errors=0
dma_state=0x00000000
dma_test=PASS
```

DS4 integration can use the same runtime API without relying on privileged
`/dev/mem` or changing the gem5 SoC topology.
