# Frontend-neutral OpenNPUX compiler backend

## Goal

OpenNPUX has one hardware compiler backend and multiple thin graph-compiler
adapters. TVM BYOC is the first adapter; it is not the backend ABI. A future
MAX/Mojo integration must reuse the same validation, storage planning, tiling,
XGraph encoding, package formats, Guest runtime, and device submission path.

## Layering

```text
ONNX/PyTorch -> TVM Relax -> TVM BYOC adapter ----\
MAX Graph -------------> future MAX/Mojo adapter --+-> OpenNPUX backend IR
Other graph IR -----------------> future adapter --/          |
                                                             v
                      capability and Tensor contract validation
                      static/dynamic storage and state planning
                      generic request lowering and hardware tiling
                      XGraph/XOpenNPUX command generation
                      .npxg/.npxgm/.npxmi artifact generation
                                                             |
                                                             v
                       Host runtime -> Coral driver/firmware -> NPU modeling
```

The dependency direction is always adapter to backend. Backend modules must not
import TVM, MAX, Mojo, ONNX, or model-specific Python classes.

## Backend IR contract

The public normalized identities are:

- `OPENNPUX_BACKEND_GRAPH_V1`: one device region containing typed Tensor
  records, generic operation nodes, outputs, and optional static arena policy.
- `OPENNPUX_BACKEND_MODULE_V1`: a DAG of regions plus direct Tensor edges,
  Host bindings, scalar bindings, state updates, and module outputs.

The low-level Python API is `opennpux_backend.compiler`. Frontends should
implement the `opennpux_backend.frontend.FrontendAdapter` protocol and call
`compile_frontend()`. The adapter has one required method:

```python
class MojoAdapter:
    name = "mojo-max"

    def to_backend_ir(self, max_graph):
        return lower_max_graph_to_opennpux_ir(max_graph)
```

The standalone contract tool is:

```bash
python3 tools/models/compile_opennpux_backend.py backend.json output
```

Compiled files are emitted through the backend-owned
`write_graph_artifact()` and `write_module_artifacts()` APIs. Frontend tools
must not duplicate `.npxg`/`.npxgm` naming or sidecar serialization rules.

Adapters query the same backend-owned operation table with
`opennpux_backend.supports_operation()` or as machine-readable JSON:

```bash
python3 tools/models/inspect_opennpux_backend.py
```

The capability table uses canonical operations such as `matmul` and
`attention`, not TVM or MAX spellings. It reports command families, input and
output arity, accepted dtypes, rank requirements, shape relations, attributes,
tiling support, and per-instruction extent limits. An adapter can call
`supports_operation_contract()` with concrete Tensor descriptors before
partitioning. Compilation calls the corresponding
`validate_operation_contract()` again before command emission, so an adapter
cannot bypass the backend ABI contract. Name-level support alone never makes an
unsupported Tensor invocation legal.

The contract intentionally separates two checks:

1. Capability preflight checks frontend-neutral invocation facts that TVM,
   MAX/Mojo, and other importers can all express: arity, static shape, dtype,
   layout attributes, and shape relations.
2. Final lowering checks command-specific address, arena, tile, scratch, and
   batch limits after storage planning.

This split lets frontends make consistent partition decisions without exposing
XGraph encoding details or duplicating hardware lowering policy.

Graph input produces `.npxg` and inspection metadata. Module input produces one
`.npxg` per region and `module.npxgm.json`. This entry point deliberately does
not import TVM and can consume IR emitted by TVM, MAX/Mojo, or another adapter.

Readers also accept `OPENNPUX_TVM_BYOC_GRAPH_V1` and
`OPENNPUX_TVM_BYOC_MODULE_V1` during migration. Writers always emit the neutral
identities, so the compatibility path is bounded and does not leak into new
frontends.

## Adapter responsibilities

Every frontend adapter must:

1. Match operations that the OpenNPUX backend capability table supports.
2. Partition supported regions while preserving unsupported Host operations.
3. Convert frontend dtype, shape, layout, attributes, constants, mutable state,
   and symbolic invocation parameters into explicit backend IR records.
4. Preserve stable model parameter names for runtime binding.
5. Reject ambiguous semantics instead of guessing layout or precision.

An adapter must not select hardware tiles, encode custom RISC-V instructions,
allocate the shared-DMA arena, implement GPTQ dequantization, or submit work to
the driver. These remain backend-owned behavior.

## TVM and Mojo integration

The TVM adapter is `RelaxFrontendAdapter`. It uses Relax pattern fusion and
BYOC partitioning, then emits the neutral IR through the common
`adapt_frontend()` boundary. Existing `compile_tvm_byoc_xgraph.py` and
`compile_tvm_byoc_module.py` remain convenience frontends and compatibility
entry points.

MAX/Mojo does not implement TVM BYOC. Its adapter will register MAX Graph custom
operations and/or Mojo custom kernels, map their graph values and attributes to
the same neutral IR, and call `opennpux_backend.compiler`. Mojo may later
provide native implementations of selected lowering functions, but those
implementations must obey the same backend IR and artifact ABI rather than
introducing a second command format.

## Migration plan

1. Stabilize neutral graph/module identities and generic compiler entry points. (Done)
2. Move backend implementation modules out of the historical
   `opennpux_tvm_byoc` namespace while retaining compatibility re-exports. (Done)
3. Define a versioned capability query so adapters can partition without
   duplicating support tables. (Done for operation identity, arity, dtype,
   static rank/shape relations, attributes, and instruction limits.)
4. Define shape-polymorphic constraints and invocation-time specialization in
   backend IR instead of frontend-specific flags.
5. Add a MAX/Mojo adapter conformance test that feeds the same normalized graph
   as TVM and requires byte-identical XGraph commands. (Done with a frontend-
   independent fake MAX/Mojo adapter; native MAX Graph extraction remains.)

The conformance boundary canonicalizes frontend aliases before compilation.
For example, `relax.matmul` and an adapter-emitted `matmul` reach exactly the
same tiler and XGraph encoder. Backend implementation modules do not import
`opennpux_tvm_byoc`, TVM, MAX, Mojo, ONNX, or model-specific code. The old TVM
module paths are compatibility re-exports only and may be removed after callers
migrate to `opennpux_backend`.

The first step is complete when legacy TVM artifacts remain readable, TVM emits
neutral formats, and the generic compiler can compile graph and module inputs
without importing TVM.
