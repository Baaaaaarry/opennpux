# RVV Highmem MobileNet Acceptance

## Build

Run on x86 Linux:

```sh
git pull
./tools/coralnpu/build_rvv_mobilenet.sh
```

The build script applies the Coral overlay that enables the pinned
`@tflite_micro` repository, native workspace dependencies, and the pinned
`@tflm_pip_deps` code-generation environment. No separate TFLite or Python
package installation is required.

Expected artifacts:

```text
build/coralnpu/libcoralnpu_gem5_rvv_highmem_bridge.so
build/coralnpu/gem5_mobilenet.elf
```

Update the guest `coralctl` before creating the dedicated checkpoint:

```sh
IMAGE=/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img \
./tools/coralnpu/phase45_prepare_guest_assets.sh
./sim/gem5/apply_patchset.sh
```

## Create Dedicated Checkpoint

The MobileNet configuration reserves an 8 MiB coherent window, so it uses a
separate checkpoint from the 4 KiB command tests:

```sh
CORAL_KERNEL_IMAGE=/home/barry/code/opennpux/build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
./tools/coralnpu/run_rvv_mobilenet_test.sh
```

The first invocation creates `m5out/coralnpu_mobilenet_ckpt`. Run the same
command again to restore it and execute MobileNet.

For a functional run with NPU progress tracing, use:

```sh
CORAL_MOBILENET_DEBUG=1 \
CORAL_RTL_CYCLES_PER_EVENT=1000 \
CORAL_KERNEL_IMAGE=/home/barry/code/opennpux/build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
./tools/coralnpu/run_rvv_mobilenet_test.sh

tail -F simout/coral-mobilenet.debug
```

The wrapper prints the effective gem5 options and RTL cycle batch before it
starts gem5. Neither option requires rebuilding the boot checkpoint.

`run_rvv_mobilenet_test.sh` passes the window size through
`CORAL_CONFIG_OPTIONS`, after the gem5 Python configuration path. Keep
`GEM5_OPTIONS` for gem5-global flags such as `--debug-flags`.

## Expected Result

```text
[coral-mobilenet-test] started
mobilenet_prepare=mailbox-only
mobilenet_run=started
status=0x00000001
mobilenet_state=0x00000003
mobilenet_error=0
mobilenet_npu_cycles=<non-zero>
mobilenet_dma_requests=<non-zero>
mobilenet_dma_completions=<same-as-requests>
mobilenet_dma_errors=0
mobilenet_output=<five signed int8 values>
mobilenet_test=PASS
[coral-mobilenet-test] PASS
```

`mobilenet_run=started` is flushed immediately before reset is released. If
RTL inference is slow, this distinguishes it from guest-side shared-memory
initialization.

This acceptance proves that ARM Linux dispatches an official LiteRT Micro graph
to the official RVV highmem Coral RTL. It does not claim classification
accuracy because the upstream test artifact uses dummy MobileNet weights and a
zero-filled input tensor.
