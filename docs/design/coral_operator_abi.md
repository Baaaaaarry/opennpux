# Coral Operator ABI

The Coral operator ABI is shared by full-RTL and hybrid execution. It keeps
model/runtime code independent of the simulator backend and provides one
versioned contract for graph-level bring-up and future per-operator dispatch.

## Modes

- `rtl`: firmware calls the Coral RVV/TFLM kernel directly. RTL cycle counters
  are authoritative for NPU performance analysis.
- `hybrid`: firmware submits an EXTMEM descriptor through the operator
  doorbell. The bridge executes a functional host kernel and returns through
  the same descriptor. Host time is simulator performance, not NPU latency.

The mode register is read-only to firmware and is configured by gem5 before
the RTL starts. The default is `rtl`.

## Submission

1. Firmware clears a descriptor in the synchronized EXTMEM mailbox page.
2. Firmware writes magic, ABI version, descriptor size, opcode, tensors,
   quantization fields, and `SUBMITTED` state.
3. A `fence rw,rw` publishes the descriptor.
4. Firmware writes its Coral EXTMEM address to the doorbell register.
5. The bridge validates the complete descriptor before changing it to
   `RUNNING`.
6. Completion writes result statistics and either `COMPLETE` or `ERROR`.
7. Firmware performs another fence and validates both status and error.

Full-RTL execution initializes the same descriptor without ringing the hybrid
doorbell. Firmware sets `RUNNING` before the RVV kernel and writes its measured
cycles and `COMPLETE`/`ERROR` state afterward. Consumers therefore read one
execution record in both modes; only the producer and timing semantics differ.

Malformed descriptors and unsupported opcodes are explicit protocol errors;
they never silently fall back after submission. A capability register exposes
one bit per opcode. Firmware may choose an RTL fallback before submission when
the bit is absent and `ALLOW_RTL_FALLBACK` is set. Descriptor or execution
errors remain failures and cannot be converted into a fallback.

## Addressing

The descriptor stores Coral device addresses, not host pointers. Tensor ranges
must be validated against the configured EXTMEM window before a bridge kernel
uses them. DTCM tensors require explicit staging into EXTMEM until a separate
versioned TCM-backdoor interface is implemented.

The common client reserves a 3 MiB staging arena at EXTMEM offset `0x00500000`.
Its monotonic aligned allocator has no heap dependency and returns zero on
overflow. Tensor descriptors carry Coral addresses only. Per-channel
multiplier and shift arrays use `quantization_count` and are range-checked as
signed 32-bit arrays before dispatch.

The initial partial-MobileNet opcode has no external tensor descriptors because
the host reference runner owns its deterministic zero input. It validates the
control protocol. Conv2D, DepthwiseConv2D, and MatMul opcodes reserve the common
tensor, shape, quantization, and modeled-cycle fields needed for real operator
offload.

## Time

Bridge ABI v6 exposes the actual RTL cycle counter. In timing mode gem5
schedules the next backend event after `executed_cycles * npu_cycle_period`,
so batching no longer makes 1,000 cycles consume one cycle period. Functional
fast mode may execute many batches at one gem5 tick by design and remains
unsuitable for CPU/NPU overlap or SoC latency conclusions.
