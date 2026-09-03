# TVM BYOC to XGraph Codegen

## Objective

Use Apache TVM Relax and BYOC as the model-independent frontend and graph
partitioner, while preserving the existing OpenNPUX hardware/runtime contract.
The first increment compiles one fully offloaded, statically shaped BYOC region
into the exact XGraph v2 command layout already consumed by Coral firmware.

This is not a Qwen-specific compiler. Model frontends are responsible for
importing ONNX, PyTorch Export, or another supported model format into Relax.
OpenNPUX only owns backend capability matching, layout validation, Tensor arena
planning, XGraph command selection, and later device/runtime integration.

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
| `relax.matmul` | `TMMA` | FP32, static, one tile |
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
expansions. Reimplementing those algorithms inside an early Python Codegen
would create two incompatible hardware contracts. The next compiler increment
will define a symbolic generic-request serialization so TVM can invoke the same
lowering after runtime addresses and dynamic dimensions are materialized.

Multiple OpenNPUX regions mixed with host regions are also deferred. That step
requires a TVM runtime module which stages each region, submits the `.npxg`
batch, waits on completion/fences, and publishes output Tensor bindings.

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

Run the complete compiler-path acceptance test:

```bash
./tools/models/test_tvm_byoc_xgraph_codegen.sh
```

The test has two explicit levels:

- The dependency-free path compiles a normalized graph and checks the emitted
  bytes against the project C ABI.
- The real-TVM path constructs an actual Relax `IRModule`, runs OpenNPUX
  pattern fusion and BYOC region merging, extracts the merged region, emits an
  XGraph artifact, executes all four commands through an ABI-level C consumer,
  and compares the output numerically with an independently evaluated graph.

The expected final line is:

```text
TVM Relax -> BYOC -> XGraph -> C execution: PASS
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

This is a complete compiler and command-level numerical loop, not yet the
full-system Linux/Coral acceptance. `.npxg` intentionally contains commands
only. A live invocation must bind and stage input, constant, state, scratch,
and output tensors separately before placing the artifact at
`OPENNPUX_XGRAPH_OFFSET`. The full-system acceptance must then use the existing
shared DMA window and Coral command firmware; embedding test tensors in
`.npxg` would incorrectly merge executable and invocation state.

The partition sequence follows the upstream
[Apache TVM BYOC documentation](https://tvm.apache.org/docs/how_to/tutorials/bring_your_own_codegen.html).
