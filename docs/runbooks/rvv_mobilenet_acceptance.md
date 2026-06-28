# RVV Highmem MobileNet Acceptance

## Build

Run on x86 Linux:

```sh
git pull
./tools/coralnpu/build_rvv_mobilenet.sh
```

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

## Expected Result

```text
[coral-mobilenet-test] started
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

This acceptance proves that ARM Linux dispatches an official LiteRT Micro graph
to the official RVV highmem Coral RTL. It does not claim classification
accuracy because the upstream test artifact uses dummy MobileNet weights and a
zero-filled input tensor.
