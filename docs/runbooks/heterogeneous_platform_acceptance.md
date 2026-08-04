# Heterogeneous Platform Acceptance

## Build RTL And Firmware

```sh
git pull
./tools/coralnpu/build_rtl_bridge.sh
```

This runs the AXI adapter regression and standalone custom RTL test, then
builds the official Coral bridge and command firmware.

## Install Generic Model Runtime

```sh
IMAGE=/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img \
./tools/coralnpu/prepare_guest_assets.sh
./sim/gem5/apply_patchset.sh
```

Rebuild the checkpoint once:

```sh
cd thirdparty/gem5
CORAL_KERNEL_IMAGE=/home/barry/code/opennpux/build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
CORAL_REBUILD_CKPT=1 \
./run_multicore.sh
cd ../..
```

## Model Execution

```sh
CORAL_KERNEL_IMAGE=/home/barry/code/opennpux/build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
./tools/coralnpu/run_model_test.sh
```

Expected:

```text
model_commands=2
completed_commands=2
model_checksum=0x00000330
accelerator_cycles=48
model_dma_requests=<non-zero>
model_dma_completions=<same-as-requests>
model_dma_errors=0
host_elapsed_ns=<simulated-time>
model_run=PASS
[coral-model-test] PASS
```

## Explicit Software/RTL Comparison

```sh
CORAL_KERNEL_IMAGE=/home/barry/code/opennpux/build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
./tools/coralnpu/run_custom_rtl_test.sh
```

Both paths must report checksum `0x00000198`; the software path reports zero
accelerator cycles and the custom path reports 48.

## Back-To-Back Submission Upgrade

If an older checkpoint completes the first model command and fails the second
with `Device or resource busy`, rebuild and install `coralctl`, then rebuild the
checkpoint once. The runtime now acknowledges each completion before submitting
the next command, so this update works with the previous baseline kernel module.

Rebuilding `opennpux_coral.ko` is still recommended: the current driver also
consumes terminal status in `poll()` and cancels stale completion work before a
new `START`. Neither fix requires rebuilding gem5 or the Coral bridge.
