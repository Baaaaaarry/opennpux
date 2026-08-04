# Phase 4 Command Acceptance

Run all commands from the superproject root on the x86 Linux host.

## Build

```sh
git pull
./tools/coralnpu/phase2_build_bridge.sh
./tools/guest_tools/build_coralctl.sh
sudo ./tools/guest_tools/install_coralctl_to_image.sh \
  $IMAGE_PATH/ubuntu-18.04-arm64-docker.img
./sim/gem5/apply_patchset.sh
```

The Coral build must produce:

```text
build/coralnpu/libcoralnpu_gem5_bridge.so
build/coralnpu/gem5_command_smoke.elf
```

## Refresh Checkpoint

The new `coralctl` must be captured in tmpfs. Rebuild the checkpoint once:

```sh
cd thirdparty/gem5
CORAL_KERNEL_IMAGE=../../build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
CORAL_REBUILD_CKPT=1 \
./run_multicore.sh
cd ../..
```

## End-To-End Test

```sh
CORAL_KERNEL_IMAGE=../../build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
./tools/coralnpu/run_command_test.sh
```

Expected guest result:

```text
transport=driver
backend=verilated-coral
command_status=2
command_error=0
completed_elements=16
element_count=16
output_checksum=0x00000198
vector_add=PASS
[coral-command-test] PASS
```

This proves CPU descriptor construction, driver mmap and submission, Coral
firmware parsing, three coherent tensor buffers, and completion delivery.
