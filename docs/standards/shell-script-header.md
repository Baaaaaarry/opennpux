# Shell Script Header Standard

Canonical template for production shell scripts in this repository.
New scripts follow this template; existing scripts are incrementally
migrated toward it.

---

## Mandatory Sections

Every script must include the following sections in order.

### 1. Shebang

```sh
#!/bin/sh
```

Use `/bin/sh`, not `/bin/bash`.  The scripts in this repository target a
POSIX-compatible shell with only widely-supported extensions (local,
`$()` substitution, `${VAR:-default}`).

### 2. Purpose (one line)

A single sentence describing what the script does, using the imperative
mood and the script's own filename:

```sh
#
# build_arm64_kernel.sh — Build an arm64 Linux kernel for gem5 full-system simulation.
#
```

### 3. Pipeline / Call Chain

Who calls this script, and what does it call in turn?  Use a compact
ASCII-art diagram when the chain is non-trivial:

```sh
#
# Pipeline:
#   1. Clone/fetch linux-stable
#   2. Seed .config from KERNEL_BASE_CONFIG
#   3. Run configure_arm64_gem5_kernel.sh  ← called by this script
#   4. Build Image + modules
#   5. Stage artifacts into build/kernel/
#
```

### 4. Output Artifacts

Every file the script produces, with its purpose:

```sh
#
# Output artifacts:
#   build/kernel/vmlinux-<release>     ELF kernel (for gem5 --kernel)
#   build/kernel/kernel.release        single-line release string
```

### 5. Environment Variables

Every overridable variable, its default, and what it controls.  Use a
table-like format for readability:

```sh
#
# Environment variables:
#   KERNEL_BASE_CONFIG    path to seed .config (default: tools/kernel/gem5-4.18.config)
#   LINUX_BRANCH          git branch to build (default: linux-4.19.y)
#   JOBS                  parallel build jobs (default: nproc)
```

### 6. Version Tags

Two kinds of tags:

| Tag | Meaning | Format |
|---|---|---|
| `@`_domain_`-spec` | Semantic version of this script's output contract | `@kernel-config-spec v1 YYYY-MM-DD` |
| `@synchronized-with` | Scripts whose output must stay consistent with this one | `@synchronized-with tools/kernel/check_...` |

When you change the output or the options a script controls, increment the
spec version and verify the synchronized scripts.  Examples from this repo:

```sh
# @kernel-build-spec  v1  2025-07-28
# @synchronized-with  tools/kernel/configure_arm64_gem5_kernel.sh
# @synchronized-with  tools/kernel/check_gem5_kernel_config.sh
```

```sh
# @kernel-config-spec  v1  2025-07-28
# @synchronized-with  tools/kernel/check_gem5_kernel_config.sh
```

The `synchronized-with` relationship is **directional**: if `configure`
adds a new `set_config` call, `check_gem5_kernel_config.sh` may need a
matching `grep_key`.

### 7. Error Handling

Always use `set -eu`.  The `-u` flag catches typos in variable names
before they cause silent failures deep in a pipeline:

```sh
set -eu
```

### 8. Inline Section Comments

Long scripts are divided into numbered steps.  Each step comment answers
**what** and **why**:

```sh
# ---------------------------------------------------------------------------
# Step 2: Seed the kernel .config.
#
#   Two paths:
#   (A) KERNEL_BASE_CONFIG is set (default):
#         Copy the known-good config into the build tree.  ...
#   (B) KERNEL_BASE_CONFIG is explicitly empty:
#         Use arm64 defconfig and warn. ...
# ---------------------------------------------------------------------------
```

Use `---` separators to visually group sections.  Do not use `====`
separators outside the file header.

---

## Path Conventions

### Script-relative root resolution

Every script resolves the superproject root from its own location, never
from `$PWD` or a hardcoded path:

```sh
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
```

`SCRIPT_DIR` is the directory containing the script.  `ROOT_DIR` is the
superproject root, computed by walking up the tree.  The number of `..`
steps depends on the script's depth:

| Script location | Path to root |
|---|---|
| `tools/kernel/build_arm64_kernel.sh` | `../..` |
| `tools/coralnpu/phase2_build_bridge.sh` | `../..` |
| `sim/gem5/run_multicore.sh` | N/A (starts from gem5 root) |

All paths inside the script use `${ROOT_DIR}` or `${SCRIPT_DIR}`, never
`$PWD`.  See [ADR 0001: Superproject Layout](docs/adr/0001-superproject-layout.md)
for the directory structure rationale.

### Image and kernel paths

Disk image paths use `$IMAGE_PATH` with a default, never a hardcoded
user directory:

```sh
export IMAGE_PATH="${IMAGE_PATH:-$HOME/wlk/gem5_arm_linux_images}"
```

Kernel paths reference build artifacts relative to `ROOT_DIR` or by
using `./` / `../../` relative paths that are correct from the expected
working directory.

### Anti-patterns

| Avoid | Because |
|---|---|
| `/home/barry/...` | Hardcoded user path — fails on every other machine |
| `$PWD/build/...` after `cd` | `$PWD` changes after `cd`; use `$ROOT_DIR` instead |
| `set -e` without `-u` | Unset variables silently expand to empty strings |
| `sudo mount ...` without `||` | Mount failure proceeds to modify the wrong filesystem |
| No `@synchronized-with` tags | Maintainer doesn't know which scripts to update together |

---

## Test Runner Scripts

Scripts under `tools/coralnpu/run_*_test.sh` use a lighter header:

```sh
#!/bin/sh
# Run the Phase-N <description>.
#
# <one-line description of what the test validates>
#
# Prerequisites:
#   - <artifact that must be built first>
#   - <guest tool that must be installed>
#
# Expected output: <key acceptance criteria>
#
# Environment:
#   CORAL_xxx    description (default: value)
```

They do not need the full Pipeline / Artifacts / Version sections unless
they grow complex enough to warrant them.

---

## Migration Notes

Existing scripts are being migrated incrementally.  When you touch a
script for any functional change, add the header if it is missing.  Do
not add a header to a script you are not otherwise changing — that can
wait for its next functional update.

Scripts already conforming (as of 2025-07-28):

- `tools/kernel/build_arm64_kernel.sh`
- `tools/kernel/configure_arm64_gem5_kernel.sh`
- `tools/kernel/check_gem5_kernel_config.sh`
- `tools/coralnpu/run_coralctl_test.sh`
- `tools/coralnpu/run_dma_smoke_test.sh`
- `tools/coralnpu/run_command_test.sh`
- `tools/coralnpu/run_model_test.sh`
- `tools/coralnpu/run_custom_rtl_test.sh`
- `tools/coralnpu/run_driver_dma_test.sh`
