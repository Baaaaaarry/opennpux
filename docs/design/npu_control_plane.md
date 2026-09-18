# NPU Control Plane: Dependency, Scheduling, and Completion

This document describes the current functional control-plane contract between
generic lowering, Coral firmware, and the NPU model. It is the reference for
replacing the C++ model with RTL without changing compiler artifacts.

## Schedule record

Generic lowering emits one `opennpux_xgraph_schedule` record for every emitted
XGraph command. The record contains:

- `dependency_mask`: dependencies in the current 64-command scoreboard window.
- `ordering_epoch`: an ordering domain; an older epoch must retire first.
- `allowed_engine_mask`: engines that may execute the command.
- `preferred_engine`: the compiler's preferred execution engine.
- `flags`: scheduling policy flags reserved by the ABI.

The artifact stores the schedule table separately from command semantics.
Commands describe what to calculate; schedule records describe when and where
they may execute.

## Dependency generation

`opennpux_npu_xgraph_build_request_schedule()` derives dependencies from the
address and byte range of every request operand:

- RAW: a reader waits for the previous writer.
- WAR: a writer waits for previous readers.
- WAW: a writer waits for the previous writer.
- Commands produced by one fused request remain ordered.
- Requests with disjoint Tensor ranges remain independent.

The dependency mask addresses the preceding 64 commands. A new epoch provides
the ordering boundary when the command stream moves beyond that window.

## Engine selection

Lowering assigns an allowed and preferred engine from operation semantics:

- TDMA: DMA and gather operations.
- Tensor: MMA, dequantization, attention, convolution, recurrent update, and
  routed-expert compute.
- Vector: elementwise operations, RoPE, SiLU, and combines.
- SFU: normalization, softmax, and TopK.

This is a portable hint, not a physical unit number. RTL may expose different
engine counts but must preserve the engine-class capability contract.

## Runtime transport

`coral_runtime.c` places command and schedule tables in the shared submission
window and publishes their offset, count, and record size. Firmware validates
the table and writes each record to the XOpenNPUX scheduling CSRs before it
dispatches the corresponding custom instruction. Firmware does not recompute
graph hazards.

## Functional model

`Gem5NpuDependencyScoreboard` tracks completed commands in the active 64-entry
window. A dependency is released only after successful retirement.

`Gem5NpuTaskScheduler` admits a ready task only when all dependencies are
complete, older epochs have retired, an allowed engine has credit, and the
completion queue has capacity. It tracks up to 64 in-flight tasks.

`Gem5NpuCompletionQueue` accepts out-of-order engine completion but retires the
oldest epoch and sequence first. Failed commands retire with an error and do
not satisfy dependent commands. Its current modeled capacity is 16 entries.

Telemetry reports submitted, issued, and retired tasks; dependency, epoch,
engine-credit, and completion-backpressure stalls; and maximum in-flight and
completion-queue occupancy.

## Engine adapter boundary

`Gem5NpuEngineAdapter` separates scheduling from execution. The scheduler uses
four operations only:

- `CanAccept(engine)` observes engine credit or downstream backpressure.
- `Submit(engine, packet)` transfers one accepted XOpenNPUX operation.
- `Poll(memory, base, completion)` returns any completed engine operation.
- `pending_count()` exposes outstanding work for drain and fence handling.

`Gem5NpuFunctionalEngineAdapter` implements this interface with the existing
C++ functional coprocessor. A Verilator/RTL adapter must implement the same
contract with ready/valid task submission and a completion channel. Neither
implementation owns graph dependency policy; the shared scheduler remains the
single source of issue and retirement decisions.

## Current boundary

The C++ control-plane model validates scheduling semantics and functional
execution. The Host XGraph executor issues the complete ready set before it
collects completions, so independent engine classes create observable in-flight
and completion-queue occupancy. Engine kernels still execute synchronously and
modeled cycles do not yet account for overlap. The Coral guest instruction
stream also remains program ordered. RTL integration must therefore replace
engine timing and the guest-side scheduler behind the same schedule/command
ABI, then demonstrate:

1. Concurrent issue of independent commands to different engines.
2. Correct RAW/WAR/WAW and epoch blocking.
3. Backpressure when engine credits or completion entries are exhausted.
4. Out-of-order completion with in-order retirement and precise errors.
5. Bit-identical operator results and unchanged XGraph artifacts.
