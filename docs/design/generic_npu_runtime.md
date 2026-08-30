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

## Generic Command Execution Boundary

The bridge now exposes the first model-independent execution seam behind the
existing Coral control path:

1. Firmware submits a normal Coral descriptor with outer opcode
   `GENERIC_COMMAND` through the MMIO doorbell or `NPU_LAUNCH` custom
   instruction.
2. `reserved[0]` carries the generic executable opcode for second-level engine
   selection; the descriptor's sole tensor points to a versioned functional
   request in EXTMEM.
3. The request contains only NPU-visible addresses and role-tagged operands.
   The bridge validates all ranges before translating them to simulation-host
   pointers.
4. `Gem5HostFunctionalBackend` performs the real numerical kernel and returns
   operations, bytes read/written and modeled cycles through the normal
   asynchronous completion chain.

Current focused execution coverage is float Embedding/dense MatMul/ADD/MUL/
RMSNorm/RoPE/Softmax/TopK/SiLU and GPTQ MatMul. The nine-op Qwen command-flow
smoke submits each primitive through `NPU_LAUNCH`, preserves intermediate
Tensor dependencies in EXTMEM and verifies a deterministic TopK result.

The runtime tensor-plan API now resolves one `.npxe` command at a time into
bounded input/output Tensor views. Each view contains the stable Tensor ID,
storage class, data type, runtime-resolved dimensions, NPU-visible address and
byte size. Resolution applies live batch/sequence/KV/active-expert dimensions,
scratch-slot reuse and persistent-state layout before any functional request is
created. The next integration increment maps these model-independent views to
opcode-specific operand roles, adds multi-output operator support, and binds
GPTQ weight views before complete command-stream execution.

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

The command-processor smoke implementation uses the platform's 8MiB contiguous
shared window. The CPU stages one invocation containing all command
records, rings the existing START path once, and receives one completion
record. Firmware validates ABI/checksum/ranges, resolves every command's
parameter symbol, fixed-size numerical operator record and weight/state/scratch
bindings, then walks the complete command stream. The command processor also
checks dependency/completion tokens, dispatches capability IDs into generic
opcode classes, applies runtime token
counts to compiler workload estimates, and returns per-opcode command,
operation and byte statistics in a trace buffer. This proves online CPU
submission and NPU-side relocation and scheduling control; it does not yet
execute command kernels or real model tensors. Numerical parameter records make
the 524-command invocation about 93KiB, so the former 64KiB control window is
not sufficient. Because the shared window size is represented in the device
tree, switching an existing smaller setup to 8MiB requires one new boot
checkpoint.

### Host Functional Backend Contract

The fast functional NPU backend is a native C++ execution engine, not a token
publication shim and not a Guest CPU fallback. It consumes the same generic
command opcodes, real paged weights, runtime tensor bindings and persistent
state that timing and RTL engines consume. Each supported command must read its
actual input tensors, write its actual output tensors and make those outputs
visible to dependent commands. Returning estimated operations, bytes or cycles
without producing the output tensor is not numerical completion.

The first model-independent kernel layer implements float32 Add, Mul, SiLU,
RMSNorm, Softmax, RoPE and deterministic TopK. The functional dispatcher
returns `unsupported` for all other opcodes; it must never silently retire an
unsupported numerical command. Existing GPTQ INT4 MatMul and SwiGLU Expert
kernels are registered through the same dispatcher as the weight-bearing
execution foundation.

The current five logical executable bindings (`input`, `output`, `weights`,
`persistent_state`, `scratch`) describe memory classes, not every intermediate
tensor in the 524-command graph. Before the dispatcher can execute the complete
graph, the compiler/runtime contract must add:

- stable virtual tensor IDs for every command input and output;
- producer/consumer and liveness metadata;
- scratch offsets allocated from non-overlapping live ranges;
- persistent KV-cache and recurrent-state regions;
- runtime dimensions/strides for prefill and decode;
- explicit weight-role mappings for every numerical command.

This tensor allocation map is an executable side table rather than a
Qwen-specific ABI. Qwen3.5 is the first frontend that emits it; subsequent
Transformer frontends reuse the same contract.

The compiler now emits this first-stage side table as `model.npxt` with format
`OPENNPUX_NPU_TENSOR_PLAN_V1`. It is an SSA-style graph containing every
command's input/output tensor IDs, symbolic runtime shapes, producer/consumer
lifetimes, persistent KV/recurrent-state objects, and reusable scratch slots.
The 40-layer Qwen3.5 shape lowers 524 commands to 625 tensors and six scratch
slots. QKV projections have separate outputs; RoPE consumes and produces Q/K,
KV-cache update consumes K/V, and Attention consumes rotated Q plus cache.
`inspect_npu_tensor_plan.py` rejects incomplete command coverage,
read-before-produce dependencies, overlapping live ranges in one slot, and an
unproduced external output. This side table is still a compiler contract: the
next runtime increment must scale slots by live batch/sequence dimensions,
assign device addresses, and pass resolved tensor views to the Host C++
dispatcher before numerical execution can be claimed.

For runtime consumption, the same plan is serialized as `model.npxtb` using
the versioned `OPENNPUX_NPU_TENSOR_PLAN_V1` binary ABI. The C loader validates
the checksum, dense tensor/command IDs, producer-before-consumer ordering,
record bounds and slot capacity. Its resolver scales the six scratch slots by
live batch and sequence dimensions and maps a scratch tensor ID to a bounded
device address. Input/output and persistent-state address resolution, followed
by dispatcher integration, remain separate runtime steps.

vLLM remains an external oracle. A vLLM `.npxo` result may be used for system
transport regression, but it must not be copied into the device result during
Host C++ functional acceptance. Native acceptance requires the Host C++ engine
to produce logits and token IDs first, followed by an out-of-band comparison
against vLLM.
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
The binary range-index ABI now publishes stable semantic IDs and an exact
`command + role + component + expert` lookup API. A numerical backend can
therefore request `qweight`, `qzeros`, `scales` and `g_idx` independently
instead of treating a command's first tensor range as its complete weight.
Ambiguous matches are rejected, which preserves the distinction between MoE
experts and projection tensors before GPTQ execution.
A model-independent loader now materializes one exact GPTQ component set from
safetensors with caller-selected size limits and failure-safe cleanup. This is
the contiguous execution path for projection-sized tensors; the same component
contract will back tiled loading for oversized LM-head and expert matrices.
For command-processor acceptance,
`materialize_npu_weight_samples.py` builds a 4KiB command-indexed probe table.
Each weight-bearing command receives the first 32-bit word from its own mapped
tensor range at offset `command_id * 4`; weightless slots remain zero. Firmware
uses the command ID directly, proving that relocation reaches distinct real
model tensors without pretending that a 4KiB window contains all GPTQ pages.
The companion `model.npxr` binary stores every matched tensor range as a fixed
64-byte record. Records carry command/shard IDs, role and component hashes,
file offset and size, tensor and parameter-symbol hashes, and the routed-expert
ID. The flags field also carries the projection slot and normalized NPU data
type, allowing the runtime to distinguish packed int4, int32 metadata and
FP16/BF16/FP32 scales without model-specific naming rules. This compact index
is the source for the next bounded page-cache and
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
loop. The same command accepts a model manifest and `.npxr` range index; in
that mode it resolves the command's first active tensor page, reads the real
safetensors shard through the model package loader, and fills the shared LRU
cache rather than repeating a synthetic page. Multi-page command residency and
prefetch remain separate performance work.

MoE paging no longer requires a hard-coded expert order. The host runtime
provides `opennpux_npu_router_topk()`, which applies a stable descending Top-K
selection to runtime router logits, breaks ties by expert ID, and normalizes
the selected weights. For bring-up, `coralctl executable-run-paged` accepts a
binary float32 logits vector through `OPENNPUX_ROUTER_LOGITS`; its element count
must equal the manifest's expert count. The selected IDs drive the weight pager
and are printed with their normalized weights. Without this variable, the tool
uses the deterministic `0..K-1` fallback so existing smoke tests remain
reproducible. The file input is a test transport only; the production path will
bind the ROUTER operator output buffer directly to the same Top-K API.
The host runtime exposes nonblocking `start`, `status` and `reset` operations
plus `opennpux_coral_run_with_service()`. Its callback runs while the device is
non-terminal, allowing a CPU pager to drain the fault queue without waiting
for the final NPU completion interrupt. The legacy blocking run API remains
available for firmware that never yields on a host-managed resource.
`npu_paging_layout.c` reserves the actual invocation/completion/trace/I/O/route
control region, places the fault ring at the next 64-byte boundary, and aligns
the weight cache to the selected transfer size. A small control region retains
the default queue offset `0x10000` and cache offset `0x20000`; larger generated
invocations move the queue and cache forward automatically. This prevents a
large command stream, such as the 524-command Qwen decode invocation, from
overwriting the paging ring. Dedicated PAGE_QUEUE and PAGE_CACHE binding flags
expose these resources without embedding model-specific fields in the
invocation ABI.
Page-fault ABI version 4 also identifies the semantic tensor role, GPTQ
component, expert, and exact source range represented by every response. The
NPU can therefore reconstruct `qweight`, `qzeros`, `scales`, and `g_idx`
streams without parsing Qwen tensor names. Overlapping aligned transfers are
published once per semantic range; the bounded cache converts duplicate
physical reads into hits while preserving component identity for execution.
A PAGE_RESIDENCY binding publishes one fixed record per cache slot. Records
carry the currently resident command/component/expert and the exact tensor and
page ranges, with a generation counter for synchronization. Firmware validates
the record before consuming a READY page, and the Verilated bridge can convert
the same records into streamed-kernel page spans after the fault entry is
retired. This separates transient queue ownership from numerical operand
residency.
The executable command flag `OPENNPUX_NPU_COMMAND_USES_WEIGHT` is emitted from
the lowered phase rather than inferred from the opcode. This prevents dynamic
attention compute from being counted as a static-weight DMA request while
retaining weight reads for projections, normalization, routing and experts.

The host runtime now resolves each instantiated command against the binary
tensor plan and materializes the bridge's address-only functional request ABI.
This is a model-independent join of `.npxc` commands, dynamic runtime shape,
`.npxtb` SSA tensor views and caller-supplied weight-page operands. Up to four
inputs and three outputs retain distinct roles, so fused QKV projection, RoPE,
router Top-K and recurrent-state commands no longer collapse multiple tensors
into one output. `opennpux_npu_functional_program` iterates a validated
submission and relocates each operator parameter block from the submission's
device address. A build-time ABI probe rejects drift between the Linux runtime
header and the mirrored Coral bridge header before Verilator compilation.
The first stateful numerical kernels now share that request contract. KV cache
storage is explicitly `[K/V, batch, kv, kv_heads, head_dim]`; the update kernel
appends both planes, and the attention kernel performs grouped-query head
mapping, scaled score/softmax and value reduction. Recurrent update and MoE
combine also produce real output/state tensors. These paths execute on the
simulation host but consume the same tensor addresses and report the same
operation/byte/cycle statistics as future RTL engines. Fused QKV still requires
three separately bound projection weight sets and is intentionally not treated
as a valid single-output MatMul.
Fused QKV now has explicit Q, K and V GPTQ component roles. The runtime maps
compiler projection slots to those roles (and to gate/up/down expert roles),
while the Host backend runs three independently shaped projections and writes
three tensor-plan outputs. This preserves grouped-query dimensions where
`heads != kv_heads`; it also prevents one resident weight set from being
incorrectly reused for all three projections.

The first data-plane probe stages one 4KiB page from the model weight source
behind binding 2. Weight-consuming command classes sample the page through the
Coral EXTMEM address, which exercises the RTL AXI Master and gem5 shared-memory
path. Trace version 2 reports page requests, actual DMA bytes and a content
checksum. This is deliberately a paging transport probe: it does not yet map
each command's selected range yet or execute arithmetic with the sampled
values. The next data-plane increment consumes `model.npxw` to populate a
bounded page cache and relocates weight binding 2 per command.

## Guest-Supplied XOpenNPUX Graph

`OPENNPUX_XGRAPH_V1` is the functional bridge from the generic CPU runtime to
the custom-instruction operator library. The Guest writes a versioned graph
header, fixed 64-byte commands, and tensor payloads into the shared DMA window.
Each command carries a model-independent opcode, shared-window tensor offsets,
shape, scalar parameter, and data type. It does not carry Qwen layer names,
tensor names, or host pointers.

After coherent host-to-EXTMEM synchronization, Coral firmware validates the
buffer and lowers each record to XOpenNPUX CSR writes plus a 32-bit custom
instruction through `xopennpux_ops.h`. Coral performs L1 custom classification,
the NPU performs L2 operator decode, and `tfence` protects every dependent
Tensor edge. Completion state, command count, operation count, cycle estimate,
output checksum, and the final packed TopK result return through EXTMEM and the
shared window.

The initial full-system gate contains all nine functional primitive classes
plus one GPTQ MatMul request. The runtime intentionally submits them as two
batches containing 9 and 3 physical commands. This replaces the former
firmware-local descriptor smoke:
the command graph and input Tensor values now originate in the Linux Guest.
The next compiler increment lowers the existing generic `.npxc/.npxtb`
invocation into the same graph ABI, tiles dimensions that exceed CSR fields,
and binds paged GPTQ weights. `OPENNPUX_XGRAPH_V1` is therefore a functional
instruction-stream reference, not a replacement for the production generic
executable format.

The first part of that compiler boundary is implemented by
`opennpux_npu_xgraph_lower_primitive()`. It consumes the model-independent
functional request produced from `.npxc`, `.npxtb`, runtime shape and explicit
weight operands. Direct FP32 primitives lower without inspecting a model or
tensor name. RoPE layout, activation semantics and TopK packed scratch are
explicit options rather than model-family assumptions. Primitive lowering
rejects composite commands; batch lowering routes GPTQ MatMul through the
model-independent tile planner and expands it into `TDEQUANT/TMMA/TADD` before
instruction emission.

### Bounded XGraph batches

The runtime does not require a complete executable to fit in one command
buffer. It lowers only whole logical requests that fit in the current batch,
submits the batch, waits for completion, and resumes at the next request. A
composite request such as GPTQ MatMul is never split halfway across two
submissions: all of its `TDEQUANT/TMMA/TADD` records must fit together.

`opennpux_xgraph_header.reserved[0..4]` carry the batch sequence, first logical
request, request count, final-batch flag, and first global physical command.
Command IDs remain local and dense inside each batch because Coral firmware
retires a freshly loaded stream after every restart. The host runtime
aggregates completed logical requests, physical commands, operations, cycles,
and DMA statistics. Numerical validation remains a runtime/compiler concern;
generic firmware must not embed a model golden such as a fixed TopK index.
