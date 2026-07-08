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

Malformed descriptors and unsupported opcodes are explicit protocol errors;
they never silently fall back. Firmware may choose an RTL fallback before
submission when an operator shape is unsupported by the hybrid backend.

## Addressing

The descriptor stores Coral device addresses, not host pointers. Tensor ranges
must be validated against the configured EXTMEM window before a bridge kernel
uses them. DTCM tensors require explicit staging into EXTMEM until a separate
versioned TCM-backdoor interface is implemented.

The initial partial-MobileNet opcode has no external tensor descriptors because
the host reference runner owns its deterministic zero input. It validates the
control protocol. Conv2D, DepthwiseConv2D, and MatMul opcodes reserve the common
tensor, shape, quantization, and modeled-cycle fields needed for real operator
offload.
