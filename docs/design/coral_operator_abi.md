# Coral Operator ABI

> Scope: this per-operator descriptor is the Coral firmware bring-up ABI. The
> platform execution ABI is the model-independent `opennpux_npu_invocation_header`
> followed by tensor bindings, command records, parameter blocks and one
> completion record in `runtime/host/include/opennpux/npu_submission.h`. The
> historical Qwen tiny "TCB" is a frozen regression fixture, not a production
> interface for new models or hardware features.

The Coral operator ABI is shared by full-RTL and hybrid execution. It keeps
model/runtime code independent of the simulator backend and provides one
versioned contract for graph-level bring-up and future per-operator dispatch.

## Modes

- `rtl`: firmware calls the Coral RVV/TFLM kernel directly. RTL cycle counters
  are authoritative for NPU performance analysis.
- `hybrid`: firmware submits an EXTMEM descriptor through the operator
  doorbell. The bridge executes a functional host kernel and returns through
  the same descriptor. Host time is simulator performance, not NPU latency;
  `modeled_cycles` is produced by a configurable accelerator latency model.
- `sampled`: firmware and driver still run through the Verilated Coral path,
  but supported long operators are executed by the hybrid host kernel. This is
  the default bring-up mode for large graphs where full RVV RTL execution would
  take hours.

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
overflow. Tensor descriptors carry Coral addresses and an explicit rank of
one to four, so vectors such as bias are not encoded with ambiguous zero
dimensions. Per-channel
multiplier and shift arrays use `quantization_count` and are range-checked as
signed 32-bit arrays before dispatch.

Partial MobileNet now runs through per-operator Conv2D and DepthwiseConv2D
descriptors. `MicroInterpreter::Invoke()` remains in Coral firmware; wrappers
stage DTCM tensors and quantization arrays into EXTMEM, submit one operator,
copy its output back, and continue normal TFLM scheduling. The older host-side
whole-graph shortcut is unsupported.

The current hybrid kernel library covers these bring-up operators:

| Opcode | Tensor Types | Status |
|---|---|---|
| `CONV_2D_INT8` | int8 activations/weights, int32 bias | Implemented |
| `DEPTHWISE_CONV_2D_INT8` | int8 activations/weights, int32 bias | Implemented |
| `MATMUL_INT8` | int8 x int8 -> int8 | Implemented, symmetric/simple quantization |
| `FULLY_CONNECTED_INT8` | int8 input/weights, optional int32 bias | Implemented, symmetric/simple quantization |
| `ADD_INT8` | int8 elementwise | Implemented, same-shape only |
| `SOFTMAX` | float32 or int8 | Implemented for bring-up reference |
| `LAYER_NORM` | float32 input/scale/bias/output | Implemented for transformer bring-up |

MatMul, FullyConnected, Add, Softmax, and LayerNorm currently target functional
graph bring-up and hybrid/sampled correctness checks. Their quantized
production ABI still needs per-input scales, output scale, and richer
broadcast semantics before it can be treated as bit-exact TFLM coverage for
arbitrary models.

## Time

Bridge ABI v6 exposes the actual RTL cycle counter. In timing mode gem5
schedules the next backend event after `executed_cycles * npu_cycle_period`,
so batching no longer makes 1,000 cycles consume one cycle period. Functional
fast mode may execute many batches at one gem5 tick by design and remains
unsuitable for CPU/NPU overlap or SoC latency conclusions.

Hybrid kernels fill `operation_count`, memory traffic counters, host runtime,
and modeled accelerator cycles. The bridge currently uses a linear model:

```text
modeled_cycles =
  fixed_cycles +
  ceil(operation_count / ops_per_cycle) +
  ceil((bytes_read + bytes_written) / bytes_per_cycle)
```

The model is configured by `CORAL_HYBRID_OPS_PER_CYCLE`,
`CORAL_HYBRID_BYTES_PER_CYCLE`, and `CORAL_HYBRID_FIXED_CYCLES`. Defaults are
`1`, `16`, and `0`, respectively. These values are intentionally explicit and
calibratable; they provide deterministic end-to-end timing placeholders until a
validated hardware latency model is available.

## Generic NPU contract

The production path never submits a model name. A CPU compiler/runtime lowers
any supported graph into these hardware-visible records:

1. `invocation_header`: executable/context identity, entry point, priority,
   dependency fence and completion address.
2. `tensor_binding[]`: device address, byte size, data type, rank, dimensions,
   strides, access flags and memory-object identity.
3. `command[]`: generic opcode, binding range, dependency/completion tokens,
   scratch range, parameters and profiling tag.
4. `operator_parameters[]`: shape and quantization metadata used by a generic
   execution engine. Tensor data type and strided layout are carried by the
   bindings; an explicit accumulator data type remains a planned ABI extension.
5. `completion`: state/error, completed commands, cycles, DMA traffic and an
   optional trace location.

Model parsing, tokenization, graph partitioning and layout selection remain CPU
compiler/runtime responsibilities. NPU Modeling and RTL consume only this
versioned contract. New accelerators are exposed as generic capabilities or a
versioned `CUSTOM` opcode, never as model-specific TCB fields.
