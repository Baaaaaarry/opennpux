# TVM BYOC to XGraph Codegen

## Objective

Use Apache TVM Relax and BYOC as the model-independent frontend and graph
partitioner, while preserving the existing OpenNPUX hardware/runtime contract.
The first increment compiles one fully offloaded, statically shaped BYOC region
into the exact XGraph v2 command layout already consumed by Coral firmware.

This is not a Qwen-specific compiler. Model frontends are responsible for
importing ONNX, PyTorch Export, or another supported model format into Relax.
OpenNPUX only owns backend capability matching, layout validation, Tensor arena
planning, XGraph command selection, and device/runtime integration. Hardware
tiling semantics live in the runtime C lowering library and are reused by the
compiler through a stable C ABI; they are not duplicated in Python Codegen.

## Stage-one flow

```text
Model frontend
    -> TVM Relax IR
    -> FuseOpsByPattern(OpenNPUX capability patterns)
    -> MergeCompositeFunctions(OpenNPUX regions)
    -> normalized OPENNPUX_TVM_BYOC_GRAPH_V1
    -> static shared-DMA Tensor arena planning
    -> XGraph v2 header + 64-byte commands (.npxg)
    -> existing Coral firmware XGraph decoder
    -> XOpenNPUX custom CSR + 32-bit instruction sequence
```

The implementation follows current TVM BYOC separation: pattern registration,
graph partitioning, backend code generation, then runtime execution. We keep
weights as region inputs (`bind_constants=False`) because OpenNPUX already has
a separate weight package, paging, and binding contract.

## Artifact contract

`compile_tvm_byoc_xgraph.py` writes:

- `.npxg`: one `opennpux_xgraph_header` followed by N
  `opennpux_xgraph_command` records.
- `.npxg.json`: an inspection manifest containing Tensor offsets, shapes,
  storage classes, arena size, and decoded command fields.

All command addresses are byte offsets in the shared DMA window. Tensor data
starts at or after `OPENNPUX_XGRAPH_DATA_OFFSET` and every allocation is
64-byte aligned. The compiler rejects overlaps, unsupported dtypes, symbolic
dimensions, broadcasts whose hardware semantics are not yet defined, and
single TMMA shapes above the 10-bit hardware dimension limit.

## Initial operator mapping

| Relax/BYOC operation | XGraph command | Initial restriction |
| --- | --- | --- |
| `relax.matmul` | one or more `TMMA` | FP32, static; C lowering tiles dimensions that exceed the instruction fields |
| `relax.add` | `TADD` | equal shapes, no broadcast |
| `relax.multiply` | `TMUL` | equal shapes, no broadcast |
| `relax.nn.rms_norm` | `TRMSNORM` | last-axis FP32 normalization |
| `relax.nn.softmax` | `TSOFTMAX` | last axis |
| `relax.nn.silu` | `TSILU` | FP32 |
| `relax.take` | `TGATHER` | embedding gather on axis 0 |
| `relax.topk` | `TTOPK` | static K, split values/indices |
| `opennpux.rope` | `TROPE` | explicit adjacent/half-split layout |
| `opennpux.copy` | `TDMA` | contiguous Tensor copy |

## Deliberate boundaries

The existing C lowering remains authoritative for GPTQ dequantization, N/K
tiling, routed experts, attention, recurrent state, and other multi-command
expansions. Reimplementing those algorithms inside Python Codegen would create
two incompatible hardware contracts. Dense FP32 MatMul is the first operation
to cross this boundary: Codegen passes Tensor offsets, shapes, RHS layout and
command ID through `opennpux_xgraph_codegen_dense_matmul()`, which invokes the
same C tiler used by the runtime. Both Relax `[K,N]` and model-runtime `[N,K]`
weight layouts are explicit rather than inferred.

The next compiler increment will generalize this bridge from dense MatMul to a
symbolic generic-request serialization, so GPTQ and fused operators can invoke
the same lowering after runtime addresses and dynamic dimensions are
materialized.

Multiple OpenNPUX regions mixed with host regions are split into two explicit
layers. The compiler layer exports every region
as an independent `.npxg` plus `module.npxgm.json`; the manifest records
topological submission order, external inputs, direct NPU-to-NPU Tensor edges,
and ordered Host pipelines. The runtime layer stages each region, executes Host
operations between NPU submissions, waits on completion, and publishes output
Tensor bindings. Keeping those responsibilities separate prevents compiler
artifacts from embedding a particular Host scheduler.

Compile a normalized multi-region module or a TVM IRModule with:

```bash
python3 tools/models/compile_tvm_byoc_module.py \
  model.relax.json build/model-regions \
  --dump-byoc-module build/model-regions.normalized.json
```

The compiler validates every cross-region edge as output-to-input, requires
identical shape and dtype, rejects multiple producers and cycles, and emits one
artifact per region in deterministic topological order. A supported region
separated from the next one by an unsupported Host operation intentionally has
no direct edge: its next input remains an external runtime binding produced by
the Host executor.

`ModuleRuntime` is the model-independent scheduler for this manifest. It owns a
separate Tensor arena per region, requires every external input, constant and
state binding before launch, copies direct NPU-to-NPU edges immediately before
the consumer executes, and returns all module-boundary outputs. Device execution
is an injected callback rather than Python operator code, so the same scheduler
contract can be tested with a functional executor and then implemented by the
Coral driver or TVM C++ runtime without changing artifacts.

`run_tvm_byoc_module.py` is the first generic manifest consumer. It binds raw
Tensor files, submits every NPU region through `coralctl xgraph-run`, executes
the manifest Host pipeline, and writes module outputs only after verified
device readback. For example:

```bash
python3 tools/models/run_tvm_byoc_module.py build/model-regions \
  --coralctl /usr/local/bin/coralctl \
  --transport driver \
  --bind residual.lhs=lhs.bin \
  --bind residual.rhs=rhs.bin \
  --output-dir build/model-output
```

The initial Host operation registry contains only FP32
`relax.nn.relu`. Unsupported operations and invalid dtype/byte-size contracts
fail explicitly. This Python runner is a reference scheduler for a Host OS
with Python; the production Guest path still requires an equivalent static C
runtime rather than embedding Python in the checkpoint.

The static Guest runtime now consumes a binary module invocation package. The
package is produced from the compiler manifest and invocation arenas with:

```bash
python3 tools/models/build_tvm_byoc_module_package.py \
  build/model-regions build/model-invocation.npxgm \
  --arena region0=region0.arena.bin \
  --arena region1=region1.arena.bin
```

Its versioned ABI contains fixed-size region, direct-edge, Host-binding,
Host-operation, and module-output tables followed by aligned XGraph and arena
payloads. `coralctl xgraph-module-run` validates every table and byte range,
applies incoming bindings, executes Host operations, submits each XGraph
region, imports only device-readback output, and publishes module outputs. A
package is an invocation artifact: it contains runtime Tensor values, while
`module.npxgm.json` plus the region `.npxg` files remain reusable compiled
artifacts. A later ABI revision can split invocation arenas from the package
without changing the region command format.

That split is now available without changing the XGraph region ABI. Build the
base module with external Tensor ranges cleared, then create a versioned
`.npxmi` overlay from one invocation's arenas:

```bash
python3 tools/models/build_tvm_byoc_module_package.py \
  build/model-regions build/model.npxgm \
  --clear-external-bindings \
  --arena region0=region0.arena.bin --arena region1=region1.arena.bin
python3 tools/models/build_tvm_byoc_invocation.py \
  build/model-regions build/request.npxmi \
  --arena region0=region0.arena.bin --arena region1=region1.arena.bin
OPENNPUX_XGRAPH_MODULE_INVOCATION_PATH=build/request.npxmi \
  coralctl xgraph-module-run build/model.npxgm
```

The Guest validates invocation magic/version, binding indices, source and
destination byte ranges, flags, and payload checksums before modifying any
region arena. This allows one compiled module package to be reused for
different requests while preserving the old embedded-arena path when no
invocation overlay is supplied.

The full-system acceptance executes the same cleared `.npxgm` twice with two
different `.npxmi` files during one Guest boot. Both invocations must apply two
bindings and complete the same region/Host topology, while their exported
outputs must differ. This separates dynamic rebinding coverage from the
single-graph numerical-reference test and detects runtimes that accidentally
retain the first request's input or output.

Every module output is reported with its region, byte range, name checksum,
and data checksum. `OPENNPUX_XGRAPH_MODULE_OUTPUT_PATH` writes output 0 for
backward compatibility. Setting `OPENNPUX_XGRAPH_MODULE_OUTPUT_PREFIX=/tmp/out`
writes all outputs as `/tmp/out.0.bin`, `/tmp/out.1.bin`, and so on. Completion
reports both `xgraph_module_outputs_completed` and aggregate
`xgraph_module_output_bytes`, so tuple and auxiliary outputs are not silently
discarded.

The normalized Codegen boundary already encodes `TOPK`, `ROPE`, and contiguous
copy commands. The initial automatic Relax pattern table excludes `TOPK`
because Relax represents its values/indices result as a tuple; tuple result
legalization is part of the next partitioning increment. `ROPE` and copy need
OpenNPUX legalization ops because they do not have a single portable Relax op
with all required hardware semantics.

## Usage and validation

Install the pinned Apache TVM 0.24.0 source build once, then load the generated
environment:

```bash
./tools/models/setup_tvm_byoc_env.sh
. .cache/tvm/env.sh
```

If direct GitHub access is unstable, route the main repository and all nested
submodules through the same GitHub-compatible mirror without modifying global
Git configuration:

```bash
TVM_GITHUB_BASE=https://gh-proxy.com/https://github.com \
TVM_BUILD_JOBS=12 ./tools/models/setup_tvm_byoc_env.sh
```

`TVM_GITHUB_BASE` must accept paths such as `apache/tvm.git`,
`apache/tvm-ffi`, `dmlc/dlpack`, and `ianlancetaylor/libbacktrace`. The setup
script retries interrupted clones and resumes an existing partial TVM tree.

Run the complete compiler-path acceptance test:

```bash
OPENNPUX_REQUIRE_TVM=1 ./tools/models/test_tvm_byoc_xgraph_codegen.sh
```

The test has two explicit levels:

- The dependency-free path compiles a normalized graph and checks the emitted
  bytes against the project C ABI.
- The real-TVM path constructs an actual Relax `IRModule`, runs OpenNPUX
  pattern fusion and BYOC region merging, extracts the merged region, lowers a
  `[2,2048] x [2048,8]` MatMul into three accumulating TMMA tiles plus
  `TADD/TSILU/TSOFTMAX`, executes all six commands through an ABI-level C
  consumer, and compares the output numerically with an independently evaluated
  graph.
- The multi-region path partitions `add -> Host relu -> silu`, verifies that
  the Host operation prevents incorrect region merging, and emits two ordered
  one-command XGraph artifacts plus a module manifest.

The real TVM multi-region compiler test is validated on GB10. Dependency-free
tests additionally execute a two-region `TADD -> TSILU` chain, bind both external
inputs, transfer the intermediate Tensor through the manifest edge, and compare
the final floating output.

The expected final line is:

```text
TVM Relax -> BYOC -> XGraph -> C execution: PASS
```

The current large-MatMul local acceptance reports:

```text
xgraph_commands=6
xgraph_arena_bytes=213312
xgraph_output_checksum=0x40f42b1d
xgraph_max_abs_error=1.60336494e-05
tvm_relax_byoc_e2e=PASS
```

For another serialized TVM `IRModule`, pass its JSON as the input. The CLI
runs OpenNPUX partitioning unless `--partitioned` is specified.
`--dump-byoc-graph` records the stable normalized boundary for review and
reproducibility:

```bash
python3 tools/models/compile_tvm_byoc_xgraph.py \
  model.relax.json build/model.npxg \
  --dump-byoc-graph build/model.byoc.json
```

`.npxg` intentionally contains reusable commands only. A live invocation binds
input, constant and state tensors through a separate raw Tensor arena image.
`build_xgraph_tensor_image.py` constructs that image from the generated
`.npxg.json` allocation table and a values JSON file; scratch and output ranges
remain runtime-owned.

The Guest runtime command is:

```bash
coralctl xgraph-run model.npxg invocation.arena.bin \
  0x1d000000 1000000
```

`xgraph-run` validates the artifact ABI and arena bounds, stages commands at
`OPENNPUX_XGRAPH_OFFSET`, stages Tensor data at
`OPENNPUX_XGRAPH_DATA_OFFSET`, starts the Coral command firmware and reports
the firmware completion state, command count, output checksum, operation count
and modeled cycles. This keeps executable and invocation state separate while
using the same shared DMA/EXTMEM protocol as other Guest submissions.

Run the complete compiler plus full-system acceptance on the x86/GB10 host:

```bash
. .cache/tvm/env.sh
./tools/coralnpu/run_tvm_byoc_xgraph_test.sh
```

The full-system test now packages the actual `multi-region-module` emitted
from the TVM Relax graph and invokes one Guest command:

```bash
OPENNPUX_CORAL_TRANSPORT=driver \
OPENNPUX_XGRAPH_OUTPUT_TOLERANCE=0.00005 \
coralctl xgraph-module-run model-invocation.npxgm
```

There is no test-script `dd`, Tensor offset, region order, or ReLU call in this
path. Those all come from the compiler-generated module ABI. Acceptance
requires two completed regions, two completed XGraph commands, one completed
Host operation, an exported 32-byte output, and the expected XOpenNPUX
instruction dispatches. The pre-existing six-command single-region path is
retained as an independent tiling and numerical regression.

The script first computes the host numerical reference and checksum, then
injects the same `.npxg` and arena into the Linux checkpoint. Acceptance checks
the generated command count dynamically, requires firmware-dispatched
XOpenNPUX `tmma`, `tadd`, `tsilu`, and `tsoftmax` operations, verifies that the
Local EXTMEM output checksum survives the explicit device-to-host transfer, and
then performs an independent elementwise numerical comparison.

The previous one-tile GB10 full-system acceptance baseline was:

```text
xgraph_state=0x00000003
xgraph_error=0
xgraph_completed_commands=4
xgraph_output_bytes=24
xgraph_output_checksum=0xbcd03dc5
xgraph_operation_count=72
xgraph_modeled_cycles=72
xgraph_artifact_run=PASS
tvm_byoc_xgraph=PASS
```

The launch wrapper rebuilds a stale `opennpux_coral.ko` and injects it into the
checkpoint tmpfs before the resume script opens `/dev/opennpux-coral`. The
module and simulated `vmlinux` must come from the same kernel build; otherwise
the test fails before artifact staging and prints the `insmod` error.

The six-command GB10 full-system acceptance now reports:

```text
xgraph_completed_commands=6
xgraph_output_bytes=64
xgraph_output_checksum=0xaeedc3a1
xgraph_output_readback_checksum=0xaeedc3a1
xgraph_output_readback=PASS
xgraph_reference_checksum=0x40f42b1d
xgraph_output_max_abs_error=1.60336494e-05
xgraph_output_reference=PASS
xgraph_operation_count=32896
xgraph_modeled_cycles=32896
xgraph_artifact_run=PASS
tvm_byoc_xgraph=PASS
```

The runtime explicitly
synchronizes Local EXTMEM back to the Shared DMA Window and requires the
readback checksum to equal the checksum produced by firmware. This prevents a
preloaded reference tensor in shared memory from creating a false numerical
PASS. It also explicitly publishes the invocation arena from the host window to
Local EXTMEM before launch. The firmware output checksum remains diagnostic:
optimized C++ execution may
differ from the independent Host C reference in low FP32 bits after tiled
accumulation and transcendental operations. Acceptance therefore compares all
output elements against the staged independent reference with a `5e-5`
absolute-error limit rather than requiring byte-identical floating-point data.
Operation and cycle counts are recorded from firmware rather than predicted by
the compiler test.

Multi-region execution uses the same artifact ABI. `CoralCtlExecutor` invokes
`coralctl xgraph-run` once per region, requests the verified device output with
`OPENNPUX_XGRAPH_OUTPUT_PATH`, and copies only that output range into the next
region's input binding. The output file is written only after Local EXTMEM has
been synchronized and its checksum has matched the firmware checksum, so an
edge cannot consume stale host arena contents. `ModuleRuntime` remains the
model-independent DAG scheduler; it validates external bindings and edge byte
ranges and does not encode model-specific operator order.

The full-system script additionally executes two independent one-command
regions with a CPU operation between them: `TADD -> Host ReLU -> TSILU`.
Region 0 deliberately produces negative values, so replacing ReLU with a raw
copy cannot pass the region 1 numerical reference. `coralctl
host-tensor-unary` is the initial CPU Host-pipeline entry point; it validates
the FP32 Tensor file and produces a separate output file. A successful mixed
runtime acceptance adds:

```text
xgraph_module_regions_completed=2
xgraph_module_direct_edges=0
xgraph_module_host_bindings=1
xgraph_module_host_pipeline=relax.nn.relu
xgraph_module_chain=PASS
```

This acceptance is now verified on GB10. Region 0 reported matching firmware,
readback and reference checksums (`0x119b1ae5`) with zero numerical error. The
CPU ReLU processed eight elements and reported `host_tensor_run=PASS`. Region 1
reported firmware/readback checksum `0x4983e4f0`, maximum absolute error
`2.38418579e-07`, `operation_count=24`, and `modeled_cycles=24`. The module
summary confirmed two regions, zero direct edges, one Host binding, and the
`relax.nn.relu` pipeline.

For a mixed TVM graph, a Host operation is not represented as a direct NPU
edge. `ModuleRuntime` accepts a model-independent binding resolver which runs
after producer regions complete and before a consumer with an unresolved
external binding is submitted. The resolver receives verified producer Tensor
bytes and returns the Host-runtime result. This preserves Host operations such
as the ReLU separating two BYOC regions without encoding ReLU or model logic in
the NPU scheduler.

The compiler now emits such paths explicitly as `host_bindings`. Each record
contains the producing region/Tensor, consuming region/Tensor, byte count, and
an ordered Host operation pipeline. Host bindings participate in the same DAG,
single-producer, shape and dtype validation as direct device edges, but require
an injected `host_executor` before the consumer can run. The real TVM
`add -> Host relu -> silu` regression must therefore contain zero direct edges
and exactly one `relax.nn.relu` Host binding.

The partition sequence follows the upstream
[Apache TVM BYOC documentation](https://tvm.apache.org/docs/how_to/tutorials/bring_your_own_codegen.html).
