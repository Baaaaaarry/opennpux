# GEM5 + Coral NPU System-Level Plan

This tree now carries two distinct Coral integration layers:

- `sim/gem5/*`: the gem5-side local source deltas and integration notes.
- `sim/coralnpu/*`: the Coral-side local source deltas used to build on top of
  the official Coral submodule.
- `thirdparty/coralnpu`: the official RTL/software framework submodule to be
  modified for real execution and custom accelerator RTL work.

The current repository status is:

- ARM Linux boots on D9200/D9300 and restores reusable checkpoints.
- Official `CoreMiniAxi` executes firmware through the gem5 bridge.
- The Linux driver provides bounded shared mmap and asynchronous completion.
- Generic model files dispatch real tensors through versioned commands.
- Official software and custom RTL execution pass the same system workload.

## Official Coral Standalone Environment

Goal:
- Build the official `google-coral/coralnpu` repository outside gem5.
- Run the provided examples and `core_mini_axi_sim`.
- Establish a known-good RTL, toolchain, and software baseline before touching
  gem5 bridge logic.

Expected outputs:
- Verified external Coral checkout path.
- Successful standalone build logs for Coral examples and simulation targets.
- A documented command sequence for rebuilding Coral RTL artifacts.

## gem5 SoC Bridge to Official Coral RTL

Goal:
- Keep the current gem5 ARM FS environment and replace the stage-A execution
  model with a backend that talks to official Coral RTL artifacts.

Implementation split:
- `NPUDevice`: SoC shell with MMIO aperture, DMA port, backend selection, and
  event scheduling.
- `CoralStageABackend`: transaction-level smoke-test fallback.
- `CoralVerilatedBackend`: runtime-loaded bridge to Verilated Coral wrapper
  artifacts.

Baseline bridge capability:
- AXI slave path: host MMIO accesses from gem5 packets into Coral wrapper.
- AXI master path: Coral memory requests into gem5 DMA/packet transactions.
- Event loop: drive the RTL backend with a configurable tick quantum.
- Shared DMA window: exchange tensors, commands, status, and mailbox data.
- Operator modes: full RTL, hybrid modeling, and sampled RTL execution.

## Driver runtime baseline: Linux Driver and Minimal Runtime

Goal:
- Replace ad-hoc shell MMIO tests with a proper Linux-side control path.

Minimum software surface:
- load Coral ELF into ITCM/DTCM or shared memory
- write `PC_START` / `RESET_CONTROL`
- poll or interrupt on completion
- manage host-visible input/output buffers

This step should still avoid full framework integration. The objective is a
small, debuggable driver/runtime pair.

## Command runtime baseline: End-to-End Framework Integration

Goal:
- CPU runs real system software and a host runtime that dispatches inference
  jobs to Coral.

Recommended progression:
- start with a narrow userspace runtime API
- add model loading and buffer ownership rules
- then integrate the chosen inference framework

This is where CPU-side workload fidelity becomes important.

## Custom RTL baseline: Custom RTL Accelerator Unit

Goal:
- Modify the official Coral RTL to include custom accelerator logic while
  keeping the gem5 system environment unchanged.

Guideline:
- custom execution logic belongs in the external Coral RTL tree
- gem5 remains the SoC/integration environment
- the same Linux driver/runtime flow should continue to work across RTL
  revisions

## Repository Responsibilities

### Inside this superproject

- `sim/gem5/configs/*`
  - local board and configuration deltas for gem5
- `sim/gem5/src/dev/npu/*`
  - SoC shell and backend selection
- `sim/gem5/src/mem/*`
  - local gem5 memory-system fixes needed by checkpoint/restore
- `sim/coralnpu/*`
  - local Coral source deltas applied on top of official upstream
- `runtime/host/bootscripts/*`
  - temporary bring-up scripts
- `tools/coralnpu/*`
  - Coral standalone validation and migration helpers

### Repository-local submodule

- `thirdparty/coralnpu`
  - official Coral RTL/software checkout
  - authoritative location for RTL and software changes in baseline integration areas

### Outside this gem5 tree

- Verilated wrapper artifacts
- custom RTL modifications
- Coral standalone tests and examples

## Baseline Next Steps

1. Keep the baseline acceptance flows green: RTL bridge, driver DMA, partial
   MobileNet, hybrid MobileNet, and sampled MobileNet.
2. Improve full RTL performance around MobileNet tensor allocation and operator
   execution.
3. Extend the operator table toward Transformer workloads: MatMul, Linear,
   Softmax, LayerNorm/RMSNorm, GELU/SILU, RoPE/KV cache, and MoE routing.
4. Evolve the current operator modeling into a real NPU pipe model with queues,
   issue/execute/retire timing, memory bandwidth limits, and operator-level
   resource accounting.
