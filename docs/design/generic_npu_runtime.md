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
   the currently provisional version 2 ABI.

## Implemented Compiler/Runtime Boundary

The generic-ABI implementation now adds:

- `model.npxe`: inspectable generic executable metadata;
- `model.npxc`: compact binary entry-point and command-template image;
- `model.npxw`: compiler-generated command-to-tensor weight paging plan;
- prefill and decode entry points over the same immutable templates;
- token-to-next-token command coverage, including embedding, all decoder
  layers, final normalization, LM head and token selection;
- CPU-side runtime instantiation with live sequence/context IDs, tensor device
  addresses, memory objects and persistent-state binding;
- per-command parameter-symbol relocation, compact runtime batch/sequence/KV/
  active-expert fields, and logical weight/state/scratch binding resolution;
- size, offset, record-count and checksum validation in the C loader;
- a host test that loads `.npxc`, creates a decode invocation and validates the
  resulting submission record.

The first command-processor smoke implementation uses a 64KiB contiguous
shared command buffer. The CPU stages one invocation containing all command
records, rings the existing START path once, and receives one completion
record. Firmware validates ABI/checksum/ranges, resolves every command's
parameter symbol and weight/state/scratch bindings, then walks the complete
command stream. The command processor also checks dependency/completion tokens,
dispatches capability IDs into generic opcode classes, applies runtime token
counts to compiler workload estimates, and returns per-opcode command,
operation and byte statistics in a trace buffer. This proves online CPU
submission and NPU-side relocation and scheduling control; it does not yet
execute command kernels or real model tensors. Because the shared window size
is represented in the device tree,
switching an existing 4KiB setup to 64KiB requires one new boot checkpoint.
Executable and invocation ABI version 2 identifies the inline relocation
fields. Version 1 `.npxc` files must be regenerated rather than interpreted
with implicit resource bindings.

The compiler now binds each command parameter symbol to tensor roles and real
safetensors shard/offset/size ranges through `model.npxw`. Static layer phases
use fixed ranges, routed-expert commands retain a router-selected policy, and
commands without parameters are explicitly marked weightless. The sidecar
keeps frontend tensor names out of the stable device ABI while supplying the
host pager with exact model-file locations.
`materialize_npu_weight_page.py` validates those locations and extracts tensor
payload bytes, zero-padding only beyond the selected tensor. The generic
executable smoke test uses this path by default when `model.npxw` is adjacent
to `model.npxc`; `CORAL_NPU_WEIGHT_PAGE` remains an explicit override.
For command-processor acceptance,
`materialize_npu_weight_samples.py` builds a 4KiB command-indexed probe table.
Each weight-bearing command receives the first 32-bit word from its own mapped
tensor range at offset `command_id * 4`; weightless slots remain zero. Firmware
uses the command ID directly, proving that relocation reaches distinct real
model tensors without pretending that a 4KiB window contains all GPTQ pages.
The companion `model.npxr` binary stores every matched tensor range as a fixed
64-byte record. Records carry command/shard IDs, role and component hashes,
file offset and size, tensor and parameter-symbol hashes, and the routed-expert
ID. This compact index is the source for the next bounded page-cache and
active-expert loading implementation; `model.npxw` retains only inspectable
per-command summaries and range spans.
The model-independent host loader in `runtime/host/src/npu_weight_ranges.c`
validates the binary ABI and checksum, rejects invalid command/shard/range
records, and performs binary-search lookup of all ranges for a command. It is
the runtime boundary between compiler-generated metadata and the future pager.
`opennpux_model_package_read_shard_range()` provides the corresponding bounded
data path from a validated range record to external shard bytes. It checks the
declared shard size and file range before I/O, so the pager does not need to
perform a tensor-name lookup or trust compiler offsets blindly.
`npu_weight_pager.c` converts command ranges into aligned transfer requests,
coalesces consecutive ranges that share a page, filters routed-expert records
against the runtime active-expert set, and reads short final pages with zero
padding. Existing callers retain a 4KiB default, while the sized API supports
power-of-two transfer blocks from 4KiB through 2MiB. This separates the model
shard read/DMA transaction size from the future IOMMU page size: larger blocks
reduce command-processor handshakes without requiring larger architectural
pages. The current implementation is a synchronous pager primitive; queueing,
overlap and command-processor pause/resume remain integration work.
The pager also provides a caller-owned fixed-slot LRU cache. It performs no
hidden allocation, records hit/miss/eviction/materialized-byte counters, and
returns a stable slot plus page pointer for DMA staging. This separates cache
policy from the eventual device page-fault transport and allows deterministic
unit and simulation sizing.
The 64-byte `opennpux_npu_page_fault` record defines the first resumable
paging handshake. A pending record identifies sequence, command, shard,
aligned file block, transfer size and optional expert; the CPU pager resolves it to a cache
slot and publishes READY or ERROR. The record is transport-neutral and will be
placed in the shared queue when command-processor pause/resume is integrated.
`npu_weight_queue.c` implements that shared single-producer/single-service/
single-retire ring contract. The NPU producer can publish multiple pending
faults, the CPU service side resolves them in order, and the NPU retires READY
or ERROR records before slots are reused. RV32-safe 32-bit producer, service
and retire indexes use wrap-safe unsigned differences; release/acquire
publication orders payload visibility, and an explicit backpressure counter
records queue-full stalls. This provides the
batchable control plane needed for 64KiB weight DMA transfers; mapping cache
slots into the shared DMA window and pausing/resuming device commands is
implemented by the paged executable smoke path. The RV32 command processor
publishes a page fault, stalls the affected command, consumes the CPU-filled
cache slot after READY, and retires the queue entry before continuing. The
guest `executable-run-paged` service uses the asynchronous Coral runtime loop
to fill 64KiB slots while the device is active. This closes the system control
loop; replacing the deterministic test page with `.npxr`-indexed safetensor
ranges is the next data-plane integration step.
The host runtime exposes nonblocking `start`, `status` and `reset` operations
plus `opennpux_coral_run_with_service()`. Its callback runs while the device is
non-terminal, allowing a CPU pager to drain the fault queue without waiting
for the final NPU completion interrupt. The legacy blocking run API remains
available for firmware that never yields on a host-managed resource.
`npu_paging_layout.c` reserves a control region for invocation/completion/
trace data, places the fault ring at the next 64-byte boundary, and aligns the
weight cache to the selected transfer size. The default 8MiB window uses a
64-entry ring and 64 64KiB cache slots: queue offset `0x10000`, cache offset
`0x20000`, and required size `0x420000`. Dedicated PAGE_QUEUE and PAGE_CACHE
binding flags expose these resources without embedding model-specific fields
in the invocation ABI.
The executable command flag `OPENNPUX_NPU_COMMAND_USES_WEIGHT` is emitted from
the lowered phase rather than inferred from the opcode. This prevents dynamic
attention compute from being counted as a static-weight DMA request while
retaining weight reads for projections, normalization, routing and experts.

The first data-plane probe stages one 4KiB page from the model weight source
behind binding 2. Weight-consuming command classes sample the page through the
Coral EXTMEM address, which exercises the RTL AXI Master and gem5 shared-memory
path. Trace version 2 reports page requests, actual DMA bytes and a content
checksum. This is deliberately a paging transport probe: it does not yet map
each command's selected range yet or execute arithmetic with the sampled
values. The next data-plane increment consumes `model.npxw` to populate a
bounded page cache and relocates weight binding 2 per command.
