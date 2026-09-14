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

Build a real public MAX Graph with the source tree's Bazel runtime, then lower
its stable export through the OpenNPUX backend without a wheel installation:

```bash
./tools/models/test_mojo_max_xgraph_codegen.sh
./tools/models/run_max_source_opennpux_export.sh
```

The runner defaults to Modular's prebuilt Mojo toolchain. Only developers who
are changing the Mojo compiler itself should select the source-built toolchain:

```bash
MODULAR_MOJO_CONFIG=build-mojo \
  ./tools/models/run_max_source_opennpux_export.sh
```

`rules_mojo` and `rules_cc` are pinned submodules matching Modular's archive
overrides. The runner creates a patched `rules_mojo` mirror under
`.cache/modular-deps` and passes both through Bazel `override_repository`, so
these dependencies do not require GitHub access during a GB10 build.
The transitive Bats test toolchain is handled the same way using the pinned
`bats-core` source and a generated BUILD equivalent to bazel-lib's repository
rule. Helper libraries are omitted because the export target does not run Bats
tests.
The generated mirror includes an explicit `REPO.bazel` marker required by
Bazel 8 `override_repository`; the runner repairs older cached mirrors in
place.

The runner copies the tracked overlay into the ignored submodule working tree,
runs `//max/opennpux:export_projection`, and compiles the resulting frontend
export outside Modular. The copied overlay is generated state; edit the tracked
files in `integrations/modular/source_export`, not the submodule copy.

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
