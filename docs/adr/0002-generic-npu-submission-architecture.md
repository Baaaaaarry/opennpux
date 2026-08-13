# ADR 0002: Generic NPU Submission Architecture

## Status

Accepted.

## Context

The first Transformer bring-up uses a Qwen-compatible synthetic model and a
Qwen-specific TCB. That path proves descriptor staging, NPU-side validation,
completion reporting, and end-to-end numerical checks. It must not become the
platform contract: a fixed Qwen operator enum, fixed operator count, or an
offline-expanded decoder trace cannot represent arbitrary Transformer,
convolutional, multimodal, or future custom workloads.

Commercial heterogeneous SoCs separate model compilation from request
submission. Compilation may partition and schedule a graph offline, but the CPU
still submits every inference request to the NPU at runtime with live buffers,
shapes, sequence state, synchronization objects, and execution parameters.

## Decision

OpenNPUX uses four distinct contracts:

1. **Model package**: framework-independent tensor metadata and external weight
   shards. It contains no live device addresses.
2. **Compiled NPU executable**: immutable command templates, constants,
   relocation records, memory requirements, and required capability IDs. A
   model-family compiler plugin lowers source graphs into this generic format.
3. **Runtime submission ABI**: CPU-created jobs and command buffers containing
   live tensor bindings, dynamic dimensions, state/KV handles, execution mode,
   dependency fences, and completion records.
4. **Operator/ISA ABI**: NPU-visible operator descriptors, custom instructions,
   and backend capability IDs. These describe primitive work and are not named
   after a model family.

The CPU submits a coarse inference job or command buffer, not one syscall per
micro-operation. The NPU command processor fetches and validates the command
buffer, resolves bindings, schedules operators and DMA, and writes completion
records. The driver owns queue transport, memory mapping, fences, IRQ delivery,
and isolation; it does not understand Qwen, attention, or model topology.

Qwen3.5 remains the first compiler adapter and acceptance workload. Its tensor
inventory and heterogeneous decoder analysis belong in the Qwen frontend. They
may produce generic command templates for full attention, linear attention,
MoE, MTP, and vision domains, but no Qwen-specific type is added to the stable
driver or device ABI.

## Runtime Flow

```text
Inference framework
  -> model frontend/importer
  -> generic graph IR and NPU executable       (offline)

Application request
  -> runtime creates invocation bindings
  -> driver submit queue / doorbell
  -> NPU command processor fetches command buffer
  -> scheduler dispatches DMA + scalar/RVV/custom RTL operators
  -> completion queue + IRQ/fence
  -> runtime returns outputs                    (online)
```

## Consequences

- `OPENNPUX_QWEN_TCB_*` is a bring-up compatibility ABI, not the future device
  ABI. New functionality must not extend it with additional model-specific
  fields.
- `execution-plan.npxp` is compiler analysis metadata. It does not represent a
  submitted inference and must not contain final runtime addresses.
- Dynamic sequence length, batch, KV/recurrent state, selected MoE experts, and
  input/output buffers are bound by the CPU runtime for each invocation.
- Operator implementations can be RTL, RVV firmware, hybrid functional models,
  or sampled RTL without changing the framework-facing submission contract.
- A capability table identifies supported opcodes, dtypes, layouts, limits,
  custom ISA extensions, and timing models.

## Migration

1. Preserve the Qwen tiny TCB test as a regression fixture.
2. Define a generic executable and queue ABI with variable-length command
   buffers and 64-bit device virtual addresses.
3. Add runtime `load_executable`, `create_context`, `bind_tensor`, `submit`,
   `wait`, and `destroy` APIs.
4. Add an NPU command processor model with submission/completion queues.
5. Lower Qwen3.5 execution-plan phases into generic commands, then add other
   model frontends without changing the queue ABI.

