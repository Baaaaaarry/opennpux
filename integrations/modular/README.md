# Modular/MAX source integration

This directory is the OpenNPUX-owned integration layer for the upstream
Modular source tree in `thirdparty/modular`. Do not add OpenNPUX device,
runtime, ABI, or lowering changes directly to the submodule.

The ownership boundary is:

```text
Modular MAX Graph / Mojo / KGEN source
                 |
                 v
OpenNPUX frontend adapter (`tools/models/opennpux_mojo`)
                 |
                 v
OpenNPUX Backend IR -> storage/tiling -> XGraph -> driver/device
```

The Modular source tree supplies the public graph construction, model and
kernel development environment. OpenNPUX owns device discovery, memory,
queues, synchronization, command ABI and execution. This keeps the backend
usable by TVM, MAX/Mojo, and future frontends, and avoids depending on MAX
runtime internals that are not part of the open-source tree.

## Setup

Initialize and validate the pinned source revision:

```bash
./tools/models/setup_modular_source.sh
```

Run the source integration contract without installing the `modular` wheel:

```bash
./tools/models/test_modular_source_integration.sh
```

To query or test upstream Bazel targets, opt in explicitly because the first
run may download Bazel and external dependencies:

```bash
./tools/models/setup_modular_source.sh --query //max/python/max/graph:all
./tools/models/setup_modular_source.sh --test //max/tests/tests:cpu_local_tests
```

Future MAX-facing code belongs in this directory or
`tools/models/opennpux_mojo`. If an upstream source change is unavoidable,
carry it as a reviewable patch under `integrations/modular/patches` and submit
it upstream instead of leaving the submodule dirty.

`MaxGraphExportSession` is the preferred public Graph entry point. It owns the
real `max.graph.Graph` context, registers declared inputs, records calls made
through its `call()` method, and finalizes MAX and OpenNPUX outputs together.
This avoids inspecting MAX private MLIR and prevents graph/export drift.
