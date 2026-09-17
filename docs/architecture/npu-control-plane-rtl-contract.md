# NPU Control Plane and RTL Adapter Contract

## Scope

This contract defines the control path between the Coral controller, the
functional NPU model, and future RTL engines. It is model-independent. Tensor
layout, tiling, quantization, and model topology remain compiler/runtime data;
the scheduler only handles tasks, dependencies, resources, and completion.

## Pipeline

1. L2 decode snapshots the custom CSRs and emits one `Gem5NpuTask`.
2. The dependency scoreboard checks the task's 64-entry dependency window.
3. The task scheduler checks dependency readiness, ordering epoch, engine
   credits, and completion-queue capacity.
4. The selected backend executes the issued task: C++ functional kernel, RTL,
   or shadow execution.
5. Execution produces one `Gem5NpuCompletion` containing status, fault address,
   operations, and modeled or measured cycles.
6. The completion queue is associative by sequence so engines may finish out
   of order without head-of-line deadlock. Retirement commits successful
   command IDs to the dependency scoreboard and returns status to Coral.

## Required invariants

- Issue never occurs before all dependency bits have retired successfully.
- A task consumes exactly one engine credit from issue through completion.
- Completion-queue backpressure prevents issue when completion storage is full.
- Ordering epochs form fences: a later epoch cannot issue while an older epoch
  has pending, issued, or unretired work.
- Failed commands retire an error but never satisfy dependent commands.
- Functional and RTL backends use the same task and completion structures.
- The current 64-command dependency window is an implementation limit, not an
  ABI promise. Graph batching or a future generation-tagged scoreboard must
  advance the window for larger graphs.

## RTL-facing signals

The eventual SystemVerilog adapter maps the structures to ready/valid channels:

- `task_valid/task_ready/task_payload`
- per-engine `credit_return`
- `completion_valid/completion_ready/completion_payload`
- memory request/response channels independent of completion
- reset, clock, fault, and performance-counter inputs

The C++ implementation is the executable reference for arbitration,
backpressure, ordering, and error propagation. It is not an RTL timing claim.

The Coral-to-NPU boundary snapshots scheduling state through custom CSRs:

| CSR | Purpose |
| --- | --- |
| `0x831` | dependency mask bits 31:0 |
| `0x832` | dependency mask bits 63:32 |
| `0x833` | ordering epoch |
| `0x834` | allowed engine mask |
| `0x835` | preferred engine in bits 7:0 and flags in bits 31:8 |

These values are part of the dispatch snapshot. Later CSR writes therefore
cannot alter an already accepted task.

## XGraph schedule table

XGraph v2 keeps the 64-byte command record stable and optionally appends one
16-byte `opennpux_xgraph_schedule` per command. Header reserved fields 5, 6,
and 7 carry the table offset, count, and record size. An all-zero tuple means a
legacy graph and requires conservative in-order execution.

The first compiler implementation emits a safe chain inside each 64-command
scoreboard window and increments `ordering_epoch` at every window boundary.
It also emits the allowed-engine mask and preferred engine derived from the
hardware opcode. This is deliberately equivalent to the old serial execution.
Later scheduling passes may remove unnecessary edges and widen allowed-engine
masks after tensor lifetime and memory-hazard analysis.
