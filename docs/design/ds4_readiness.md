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

Not yet complete:

- hardware interrupt delivery instead of the current kernel polling worker
- DS4 model artifact format and graph partitioning
- Coral/custom RTL firmware contract for DS4 operators

## Distance To DS4

The system-integration path is past bring-up and at the driver/runtime boundary.
The remaining work is not boot or DMA plumbing; it is turning the smoke runtime
into an inference submission runtime.

Estimated engineering chunks:

1. Pass the driver-only coherent DMA acceptance test.
2. Define DS4 tensor metadata and required operator subset.
3. Add DS4 operator firmware/runtime tests.
4. Define the model artifact and graph partitioning contract.
5. Integrate the DS4 host inference path.

## Next Acceptance Target

Before DS4, the required milestone is a driver-backed version of the current DMA
smoke:

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

After that passes, DS4 integration can use the same runtime API without relying
on privileged `/dev/mem`.
