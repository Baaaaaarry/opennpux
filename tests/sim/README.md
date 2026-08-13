# Simulation Tests

Add smoke and end-to-end simulation tests here.

RTL bridge smoke sequence:

```bash
./tools/coralnpu/prepare_coral_bazel.sh
./tools/coralnpu/check_rtl_bridge_abi.sh
./tools/coralnpu/build_rtl_bridge.sh
./sim/gem5/apply_patchset.sh
CORAL_NPU_BACKEND=verilated-coral ./thirdparty/gem5/run_multicore.sh
```

After installing `coralctl` in the guest image, the resume script
automatically prefers it over the legacy shell/MMIO sequence:

```bash
./tools/guest_tools/build_coralctl.sh
sudo ./tools/guest_tools/install_coralctl_to_image.sh /path/to/arm64.img
CORAL_NPU_BACKEND=verilated-coral ./thirdparty/gem5/run_multicore.sh
```

Expected runtime output includes `backend=verilated-coral` and
`status=0x00000001`.

To test only `coralctl` and fail instead of falling back to `mmio32`, restore
the existing checkpoint with the dedicated resume script:

```bash
./tools/coralnpu/run_coralctl_test.sh
```

This wrapper applies the current gem5 overlay before launching, so the
submodule cannot silently use an older `run_multicore.sh` or resume script.

The first run after introducing the dynamic resume trampoline rebuilds the
boot checkpoint and exits. Run the same command a second time; subsequent
resume-script changes do not require another checkpoint rebuild.

The checkpoint script copies `coralctl` into tmpfs and waits for the virtio
block queue to become idle before checkpointing. This avoids post-restore disk
reads for the test binary and stale virtqueue descriptors.

The RTL bridge ABI is now version 2 and carries deferred external AXI
requests into gem5 coherent DMA. Rebuild both the bridge and gem5 after pulling
this change:

```bash
./tools/coralnpu/build_rtl_bridge.sh
./sim/gem5/apply_patchset.sh
cd thirdparty/gem5
scons build/ARM/gem5.opt -j"$(nproc)"
```

## Coherent DMA smoke

Build and install the updated guest tool:

```bash
./tools/guest_tools/build_coralctl.sh
sudo ./tools/guest_tools/install_coralctl_to_image.sh \
  $IMAGE_PATH/ubuntu-18.04-arm64-docker.img
```

Build the bridge, DMA firmware, and gem5 as shown above. Then run the DMA test
twice. The first invocation creates the format-v4 checkpoint with the
reserved-memory DT and updated `coralctl`; the second restores and tests:

```bash
./tools/coralnpu/run_dma_smoke_test.sh
./tools/coralnpu/run_dma_smoke_test.sh
```

Expected output:

```text
[coral-dma-test] tool=/tmp/coralctl
shared_base=0x8ff00000
shared_size=0x00001000
dma_result=42
dma_magic=0x4e505544
dma_test=PASS
[coral-dma-test] PASS
```
