# Phase 1 Setup: Official Coral as a Submodule

The official Coral source tree is tracked in this repository as a git
submodule:

- `thirdparty/coralnpu`

## Initialize the submodule

```bash
cd /Users/libo/Work/gem5-coral-x86
git submodule update --init --recursive thirdparty/coralnpu
```

## Validate the checkout

```bash
cd /Users/libo/Work/gem5-coral-x86
./tools/coralnpu/phase1_bootstrap.sh
./tools/coralnpu/phase1_host_check.sh
```

The bootstrap script validates the expected official Coral repository layout,
including:

- `doc/integration_guide.md`
- `hdl/chisel/src/coralnpu/CoreAxi.scala`
- `hdl/chisel/src/coralnpu/CoreAxiCSR.scala`
- `hw_sim/core_mini_axi_wrapper.h`

The host-check script validates the local toolchain prerequisites visible to
the current shell, including:

- the Bazel version requested by `thirdparty/coralnpu/.bazelversion`
- `python3.12`
- `verilator`
- `srec_cat`

## Supported phase-1 host platform

The official Coral Phase-1 standalone builds are not host-agnostic. The
repository registers its Coral cross-toolchain and host-clang toolchain for an
`x86_64` Linux exec platform. As a result:

- `bazel build //examples:coralnpu_v2_hello_world_add_floats`
- `bazel build //tests/verilator_sim:core_mini_axi_sim`

are expected to work on `x86_64 Linux`, but not on `arm64 macOS` hosts.

For this gem5 integration tree, the supported path on macOS is the provided
Docker wrapper, which runs the official Coral builds inside a `linux/amd64`
container:

```bash
cd /Users/libo/Work/gem5-coral-x86
./tools/coralnpu/phase1_build_in_docker.sh
```

The wrapper reuses a locally cached Ubuntu base image and does not force a
fresh pull. If you need to switch the cached base image explicitly:

```bash
BASE_IMAGE=ubuntu:22.04 ./tools/coralnpu/phase1_build_in_docker.sh
```

On Apple Silicon hosts, the locally cached `ubuntu:*` image is often only the
`arm64` variant. That is not enough for Coral Phase 1, because the official
Coral toolchains require a `linux/amd64` exec platform. If Docker cannot run
your chosen base image under `--platform linux/amd64`, import a local amd64
rootfs tarball first and use that local tag instead:

```bash
docker import --platform linux/amd64 /absolute/path/ubuntu-amd64-rootfs.tar.gz local/ubuntu-amd64:noble
BASE_IMAGE=local/ubuntu-amd64:noble ./tools/coralnpu/phase1_build_in_docker.sh
```

## Using `distdir` when container DNS is incomplete

If Docker can start the container but Bazel fails on `Unknown host:
codeload.github.com` or similar repository fetch errors, seed Bazel's
`distdir` from the host side and retry.

The Phase-1 wrapper already mounts:

- `/Users/libo/Work/gem5-coral-x86/thirdparty/coralnpu/distdir`
- `/Users/libo/Work/gem5-coral-x86/.cache/coralnpu/bazelisk`
- `/Users/libo/Work/gem5-coral-x86/.cache/coralnpu/bazel`
- `/Users/libo/Work/gem5-coral-x86/.cache/coralnpu/repository`

into the container and passes it via `--distdir`.

The repository includes a seed URL list here:

- [`/Users/libo/Work/gem5-coral-x86/tools/coralnpu/phase1_distdir_urls.txt`](</Users/libo/Work/gem5-coral-x86/tools/coralnpu/phase1_distdir_urls.txt:1>)

Download the needed files on the host, preserving the basename from each URL,
for example:

- `0.9.0.tar.gz` for `rules_foreign_cc`
- `rules_java-8.14.0.tar.gz`
- `rules_scala-v6.6.0.tar.gz`
- `cb68bd4a2cb80dea24d9760dc6397b5854ea41bd.tar.gz`

and place them under:

- `/Users/libo/Work/gem5-coral-x86/thirdparty/coralnpu/distdir`

Then rerun:

```bash
cd /Users/libo/Work/gem5-coral-x86
./tools/coralnpu/phase1_build_in_docker.sh
```

For `rules_java`, the phase-1 wrapper now goes one step further: if
`distdir/rules_java-8.14.0.tar.gz` exists, it pre-extracts that tarball into a
local override directory under the persisted Bazel output root and passes Bazel:

```text
--override_repository=rules_java=...
```

This avoids repeated failures caused by a half-populated `external/rules_java`
tree inside the Bazel output root.

Note that `distdir` only helps for Bazel `http_archive` / `http_file` style
downloads. Scala/Maven artifacts still require working network access unless
you separately provision a Maven cache.

## Why Bazel 8.6.0 was re-downloading every run

Inside the phase-1 container, `/usr/local/bin/bazel` is Bazelisk. Bazelisk
downloads the exact Bazel version requested by `.bazelversion` on first use and
caches it under the container user's cache directory. If that cache is not
persisted, each fresh container will download `bazel-8.6.0-linux-x86_64`
again.

The phase-1 wrapper now persists that cache under:

- `/Users/libo/Work/gem5-coral-x86/.cache/coralnpu/bazelisk`

If the persisted Bazel output root or repository cache becomes inconsistent
after interrupted fetches, force a clean rerun with:

```bash
cd /Users/libo/Work/gem5-coral-x86
PHASE1_CLEAN=1 ./tools/coralnpu/phase1_build_in_docker.sh
```

By default the wrapper builds both phase-1 targets:

- `//examples:coralnpu_v2_hello_world_add_floats`
- `//tests/verilator_sim:core_mini_axi_sim`

You can also pass an explicit target list:

```bash
./tools/coralnpu/phase1_build_in_docker.sh //tests/verilator_sim:core_mini_axi_sim
```

## Phase 1 target

At the end of phase 1, the submodule should be the authoritative RTL/software
tree used for:

- official standalone builds
- official examples
- future Verilated wrapper generation for gem5 phase 2

## Recommended next commands

Run the Docker wrapper once the submodule and Docker runtime are ready:

```bash
./tools/coralnpu/phase1_build_in_docker.sh
```

Those artifacts are the expected inputs for the future `verilated-coral`
backend in this gem5 tree.
