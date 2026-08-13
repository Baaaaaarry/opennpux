# Generic NPU Runtime And Modeling Architecture

## Scope

OpenNPUX models a reusable CPU+NPU SoC. Qwen3.5 is the current development
vehicle, not the architecture boundary. Model-family parsing, graph rewrites,
and operator fusion are compiler/frontend responsibilities. gem5 NPUDevice,
the Linux driver, queue transport, command processor, DMA, synchronization,
and performance model remain model-independent.

## Layered Architecture

| Layer | Generic responsibility | Qwen3.5 responsibility |
|---|---|---|
| Framework adapter | Import graph, tensors and dynamic inputs | Recognize Qwen3.5 MoE/attention naming |
| Compiler | Partition, fuse, tile, allocate and emit executable | Lower full/linear attention, MoE, MTP and vision subgraphs |
| CPU runtime | Load executable, bind live buffers/state, submit and wait | Construct prefill/decode invocation parameters |
| Kernel driver | Queue, memory map, IOMMU/pinning, fence and IRQ | None |
| NPU command processor | Fetch, validate, relocate and schedule commands | None |
| NPU operator pipe | DMA, tensor/vector/matrix/reduction/custom RTL execution | Execute primitive ops selected by compiled commands |
| Modeling | Resource contention, latency, bandwidth, occupancy and counters | Calibration workload only |

## Offline Artifacts

The deployment toolchain may perform expensive analysis once:

- parse framework and weight formats;
- normalize a generic graph IR;
- discover model-specific blocks and legal fusions;
- select supported NPU kernels and CPU fallback partitions;
- produce immutable command templates and relocations;
- record required capabilities and memory sizes;
- retain paged references to external weight shards.

This is compilation, not task submission. Offline artifacts use logical tensor
IDs and symbols. They cannot encode guest virtual pointers, current sequence
length, selected request buffers, active KV pages, queue sequence numbers, or
completion state.

## Online CPU Submission

For every inference request, the CPU runtime must:

1. select a loaded executable and execution context;
2. bind input/output tensors and dynamic dimensions;
3. bind or allocate KV cache, recurrent state, scratch, and weight pages;
4. choose prefill/decode or another entry point;
5. write an invocation descriptor and command-buffer bindings;
6. publish the submission queue entry with release ordering;
7. ring a doorbell or issue the submit ioctl;
8. wait on poll/IRQ/fence while other CPU work may continue;
9. consume the completion entry and make output buffers visible.

Submitting a command buffer rather than issuing one ioctl per operator preserves
realistic CPU/NPU asynchrony and allows the NPU scheduler to overlap DMA and
compute.

## Generic ABI Objects

### Executable Header

- ABI/version and target architecture ID;
- entry-point table, command-template table and relocation table;
- immutable constant/weight references;
- scratch, persistent-state and alignment requirements;
- required capability IDs and fallback partition metadata.

### Invocation Descriptor

- executable/context/entry-point IDs;
- request and queue sequence IDs;
- binding-table address and count;
- dynamic-dimension and scalar-parameter table;
- persistent-state handle for KV cache or recurrent state;
- dependency fence and completion address;
- priority, deadline, execution mode and profiling flags.

### Tensor Binding

- logical tensor ID;
- 64-bit device address or memory-object handle plus offset;
- byte size, dtype, rank, dimensions and strides;
- layout, quantization metadata and access flags.

### Command Record

- generic opcode/capability ID, flags and dependency tokens;
- input/output binding IDs;
- parameter-block offset and size;
- scratch range, tile information and optional custom instruction ID;
- profiling tag and completion token.

### Completion Record

- sequence ID, terminal state and error class;
- completed command count and faulting command;
- NPU cycles, DMA bytes, stalls and operator counters;
- output fence and optional trace-buffer range.

All queue-visible records require explicit size/version fields, bounded counts,
checksums where useful, and acquire/release ownership rules.

## NPU Modeling Work

The command processor and scheduler become first-class modeled resources:

- submission/completion queue depth and fetch latency;
- descriptor decode and relocation bandwidth;
- ready/dependency queues and issue width;
- DMA channels, outstanding transactions and bandwidth contention;
- tensor/local memory capacity, bank conflicts and allocation stalls;
- vector/matrix/reduction/custom accelerator pipelines;
- operator launch, setup, occupancy and pipeline drain costs;
- interrupt, fence and context-switch latency;
- per-context isolation and scheduling policy;
- hybrid/RTL/sampled implementations behind identical capability IDs.

The current linear hybrid latency formula remains a temporary backend. It must
be replaced or calibrated per resource and operator before performance claims.

## Qwen3.5 Adapter Boundary

`model.npxm` and `execution-plan.npxp` currently inventory Qwen3.5 tensors and
classify heterogeneous layers. The next Qwen increment should convert each
phase into generic executable templates. Runtime values remain unresolved:

- prompt tokens, batch and sequence length;
- prefill/decode selection;
- KV and linear-attention recurrent-state pages;
- router-selected experts and weight-page residency;
- input/output/logit buffers and synchronization objects.

Those values are supplied at invocation time. Future Llama, DeepSeek, Mixtral,
ViT, diffusion, and custom graph frontends can emit the same executable and
submission ABI while adding only model rewrites, kernels, or RTL capabilities.

## Development Order

1. Freeze generic executable, invocation, binding, command and completion ABI.
2. Implement CPU runtime and driver submission/completion rings.
3. Implement the NPU command processor and dependency scheduler model.
4. Route existing operator descriptors and custom instruction path behind
   generic command IDs.
5. Add memory objects, 64-bit addressing, paged weight/state management and
   coherency/fence rules.
6. Lower Qwen3.5 phases to generic commands and validate prefill plus iterative
   decode.
7. Add a second Transformer family as an ABI-generality gate before freezing
   version 1.

## Implemented Compiler/Runtime Boundary

The first generic-ABI increment adds:

- `model.npxe`: inspectable generic executable metadata;
- `model.npxc`: compact binary entry-point and command-template image;
- prefill and decode entry points over the same immutable templates;
- CPU-side runtime instantiation with live sequence/context IDs, tensor device
  addresses, memory objects and persistent-state binding;
- size, offset, record-count and checksum validation in the C loader;
- a host test that loads `.npxc`, creates a decode invocation and validates the
  resulting submission record.

This increment intentionally stops before device submission. The default 4KiB
shared window can validate small command-processor smoke workloads but cannot
hold a fully expanded 40-layer command buffer. The real-model compiler reports
`npu_command_template_bytes` and `npu_invocation_bytes_upper_bound`; those
measurements select either a larger queue memory object or paged command-buffer
fetch for the command-processor increment.
