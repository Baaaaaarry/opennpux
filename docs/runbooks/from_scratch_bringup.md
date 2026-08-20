# From-Scratch Bring-up: Zero to MobileNet Simulation

This runbook takes a fresh clone to a passing MobileNet NPU simulation.
Steps 1–8 are repository-wide (needed by every test); steps 9–10 are
MobileNet-specific. Each step lists its verification point — do not move on
until the check passes.

Prerequisites: x86-64 Linux host with the toolchain from
`docs/runbooks/phase1_setup.md`, and sudo for the image-mount steps.

```sh
export IMAGE_PATH="${IMAGE_PATH:-$HOME/wlk/gem5_arm_linux_images}"
cd $HOME/wlk/open-npux
```

## The overlay pattern (you never run these by hand)

Local changes live in `sim/gem5/` and `sim/coralnpu/` mirrors and are merged
into the submodules at build/run time:

- `sim/coralnpu/apply_patchset.sh` — invoked automatically by
  `build_rvv_mobilenet.sh` and `phase2_build_bridge.sh` before every build.
- `sim/gem5/apply_patchset.sh` — invoked automatically by
  `run_rvv_mobilenet_test.sh` before every simulation.

Consequences: always build/run through the official scripts; building
directly inside `thirdparty/` may use a stale overlay; edits made under
`thirdparty/` are overwritten by the next sync.

## Step 1: Submodules

```sh
git submodule update --init --recursive
```

Verify: `thirdparty/gem5/` and `thirdparty/coralnpu/` are non-empty.

## Step 2: ARM64 kernel

The known-good gem5 kernel config is bundled in the repo
(`tools/kernel/gem5-4.18.config`, the default `KERNEL_BASE_CONFIG`), so no
config extraction is needed:

```sh
./tools/kernel/build_arm64_kernel.sh
./tools/kernel/check_gem5_kernel_config.sh tools/kernel/gem5-4.18.config
```

Verify: `build/kernel/kernel.release` and
`build/kernel/vmlinux-$(cat build/kernel/kernel.release)` exist.

## Step 3: NPU driver module

```sh
./tools/kernel/build_opennpux_coral_ko.sh
```

Verify: `build/kernel/opennpux_coral-$(cat build/kernel/kernel.release).ko`
exists.

## Step 4: Guest disk images

```sh
./tools/kernel/download_gem5_arm_images.sh
```

Verify: `$IMAGE_PATH/ubuntu-18.04-arm64-docker.img` exists.

## Step 5: Guest tools → disk image (sudo)

```sh
./tools/guest_tools/build_coralctl.sh
./tools/guest_tools/build_mmio32.sh

sudo ./tools/kernel/install_kernel_to_image.sh \
  "$IMAGE_PATH/ubuntu-18.04-arm64-docker.img" \
  "$PWD/build/linux-arm64" \
  "$PWD/build/kernel/Image-$(cat build/kernel/kernel.release)" \
  "$PWD/build/kernel/opennpux_coral-$(cat build/kernel/kernel.release).ko"
sudo ./tools/kernel/install_opennpux_init_to_image.sh \
  "$IMAGE_PATH/ubuntu-18.04-arm64-docker.img"
sudo ./tools/guest_tools/install_coralctl_to_image.sh \
  "$IMAGE_PATH/ubuntu-18.04-arm64-docker.img"
sudo ./tools/guest_tools/install_mmio32_to_image.sh \
  "$IMAGE_PATH/ubuntu-18.04-arm64-docker.img"
```

## Step 6: Coral bridge + MobileNet firmware (Bazel)

```sh
./tools/coralnpu/phase2_prepare_bazel.sh
./tools/coralnpu/build_rvv_mobilenet.sh
```

Notes: the compilation mode defaults to `-c opt` (a fastbuild/-O0 bridge
runs ~14x slower); builds are offline-reproducible from the local
repository cache + distdir. The script runs the overlay sync and ABI
checks (`phase2_check_abi.sh`, mailbox header check) automatically.

Verify: `build/coralnpu/libcoralnpu_gem5_rvv_highmem_bridge.so` (~8 MB for
opt, ~21 MB with debug info) and `build/coralnpu/gem5_mobilenet.elf`.

## Step 7: Phase-4/5 guest assets (optional for MobileNet)

```sh
IMAGE=$IMAGE_PATH/ubuntu-18.04-arm64-docker.img \
  ./tools/coralnpu/phase45_prepare_guest_assets.sh
```

This builds coralctl, generates the `.npxm` heterogeneous sample model via
`tools/models/create_sample_model.py`, and installs both into the image.
Required by `run_model_test.sh` / `run_custom_rtl_test.sh`; the MobileNet
test does not use the `.npxm`.

## Step 8: BusyBox → disk image (sudo)

```sh
./tools/guest_tools/build_busybox_aarch64.sh
sudo ./tools/guest_tools/install_module_loader_to_image.sh \
  "$IMAGE_PATH/ubuntu-18.04-arm64-docker.img" \
  ./build/guest-tools/busybox-aarch64
```

The MobileNet boot payload (busybox, coralctl, driver `.ko`) is preloaded
into a tmpfs during checkpoint bootstrap, so BusyBox must be in the image
before the first simulation run.

## Step 9: First run builds gem5 automatically

`run_multicore.sh` runs `scons build/ARM/gem5.opt` on first use — a full
gem5 build takes 30–60 minutes. Subsequent runs skip it via mtime checks
(`GEM5_REBUILD=1` forces a rebuild).

## Step 10: Run MobileNet

```sh
# First verified mode; ~15 min. Also creates/restores the checkpoint
# (checkpoint/coralnpu_mobilenet_ckpt) automatically.
CORAL_MOBILENET_DEBUG=1 CORAL_OPERATOR_MODE=hybrid \
  ./tools/coralnpu/run_rvv_mobilenet_test.sh

# sampled (hybrid + CORAL_SAMPLED_RTL_OPS ops forced onto RTL)
CORAL_MOBILENET_DEBUG=1 CORAL_OPERATOR_MODE=sampled \
  ./tools/coralnpu/run_rvv_mobilenet_test.sh

# rtl (full RTL, ~17 h; the only source of authoritative cycle counts)
CORAL_MOBILENET_DEBUG=1 CORAL_OPERATOR_MODE=rtl \
  ./tools/coralnpu/run_rvv_mobilenet_test.sh
```

Checkpoint rebuilds are automatic when the disk image/kernel/init/cmdline
change; `CORAL_REBUILD_CKPT=1` forces it. See the Regression Pre-flight
Checklist in `rvv_mobilenet_acceptance.md` for the two staleness traps
(guest /tmp payload, checkpoint format version).

## When to rebuild or reinstall what

| You changed | Then run | Checkpoint rebuild? |
|---|---|---|
| `sim/gem5/` or gem5 sources | nothing — next run rebuilds gem5.opt by mtime (`GEM5_REBUILD=1` forces) | only if NPU device state/params changed (bump format version + force) |
| `sim/coralnpu/`, `rtl/wrappers/` (bridge, firmware) | `./tools/coralnpu/build_rvv_mobilenet.sh` | no — bridge and firmware load at restore time |
| kernel sources / `tools/kernel/*.config` | `build_arm64_kernel.sh`, then Step 3 and Step 5 kernel install | yes (kernel mtime rule fires automatically) |
| `runtime/kernel/` (driver) | `build_opennpux_coral_ko.sh`, then re-run Step 5's `install_kernel_to_image.sh` | yes — see Trap #1 below |
| `runtime/host/` (coralctl, runtime) or busybox | rebuild the tool, re-run its Step 5/8 installer | yes — Trap #1 |
| guest init (`runtime/host/init/`) | re-run `install_opennpux_init_to_image.sh` | yes (init path rule) |
| `.npxm` model / Phase-4/5 assets | re-run Step 7 | yes (image mtime rule) |

**Trap #1**: the `/tmp` payload (busybox, coralctl, driver `.ko`) is copied
into tmpfs during checkpoint bootstrap and frozen into the checkpoint. The
automatic invalidation does NOT watch guest-tool sources — after changing
`runtime/host/` or `runtime/kernel/`, you must reinstall into the image AND
`CORAL_REBUILD_CKPT=1`, or the simulation silently tests the old binaries.

**Trap #2**: changing the NPU device's serialized state or parameters
requires a checkpoint format-version bump plus a forced rebuild; restoring
otherwise can mis-restore without an error.

## Cleaning generated artifacts

Cleanup is on-demand (disk pressure, pre-release audit), not a ritual —
offline from-scratch reproducibility is already proven, so nothing needs to
be deleted periodically to stay confident. Sizes measured on this repo:

**Safe to delete (regenerate in minutes):**

| Path | Size | Restored by |
|---|---|---|
| `.cache/coralnpu/bazel/<hash>/` (bazel output base) | ~7.5 G | Step 6 (~2–4 min, offline) |
| `build/coralnpu/` (bridge + firmware) | ~36 M | Step 6 |
| `checkpoint/coralnpu_mobilenet_ckpt` | ~26 M | next run re-bootstraps automatically |
| `build/busybox-aarch64/`, `build/guest-tools/`, `build/models/` | ~18 M | their build scripts (seconds) |

**Deletable but costly to regenerate:**

| Path | Size | Regeneration cost |
|---|---|---|
| `thirdparty/gem5/build/` (gem5.opt + objects) | ~5.5 G | Step 9: full scons build, 30–60 min |
| `build/linux-arm64/` + `build/kernel/` | ~1.1 G | Steps 2+3 (10–30 min) + redo Step 5 (sudo) |

**Never delete (require network or rework):**

- `.cache/coralnpu/repository/` and `thirdparty/coralnpu/distdir/` — the
  offline dependency archives;
- `.cache/coralnpu/bin/` — the bazel executable;
- `$IMAGE_PATH/ubuntu-18.04-arm64-docker.img` — has the guest tools
  installed; replacing it means redoing Steps 5/7/8.

After deleting only the two build trees (bazel output base, gem5 build,
`build/coralnpu/`), re-run just Step 6 and Step 10 — Step 9's gem5 rebuild
happens automatically inside the run.

**Warning**: do not use `git clean -fdx` for cleanup. `build/`, `logs/`,
and `checkpoint/` are ignored directories it would wipe, along with any
untracked local notes.


## Acceptance checks (all three must pass)

```sh
tail -5 logs/sim/coral-mobilenet-host.log
```

Expect a footer like:

```text
[coral-mobilenet] end: ... elapsed=~900s (0h 15m 0s) exit=0
[coral-mobilenet] guest verdict: mobilenet_test=PASS (from system.terminal)
```

`exit=0` means the runner and gem5 finished cleanly; the `guest verdict`
line is the actual test result read back from the guest serial log — the
runner exits 0 even on some failures, so PASS must be confirmed explicitly.
If the verdict reads `(no new terminal output this run)`, the test never
produced a result — treat as failure and inspect the log.

```sh
grep mobilenet_output_checksum logs/sim/m5out/system.terminal
```

Expect `mobilenet_output_checksum=0x38c2a6a4` (and
`mobilenet_output=-85,-69,-78,-89,-64`). This is the FNV-1a checksum of the
full output tensor produced by the firmware inside the NPU. Because the
model ships with dummy weights and a zero-filled input, the result is
deterministic — every mode (rtl/hybrid/sampled) must produce exactly this
checksum. A different value means the computation itself diverged, even if
the run says PASS.

```sh
grep -c "AXI watchdog\|Coral AXI reject" logs/sim/coral-mobilenet-host.log
```

Expect `0`. The watchdog fires when the RTL core shows no bus activity for
5M cycles (a stall signature), and a reject line means the bridge refused a
transaction with SLVERR. Either one indicates an infrastructure bug even on
an otherwise passing run — do not ignore a PASS that comes with either.
## Troubleshooting: Kernel/Module ABI Pair

The guest `vmlinux` and `opennpux_coral.ko` must be produced from the same
kernel build directory and `.config`, not merely share the same `uname -r`.
Unknown symbols such as `of_node_put` together with
`__ll_sc___cmpxchg_case_mb_32` indicate a Device Tree or ARM64 atomic
configuration mismatch. Rebuild the module against `build/linux-arm64`,
install that release-qualified module into the disk image, and rebuild the
checkpoint because `/tmp/opennpux_coral.ko` is checkpoint-resident.
