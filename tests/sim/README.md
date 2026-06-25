# Simulation Tests

Add smoke and end-to-end simulation tests here.

Phase-2 smoke sequence:

```bash
./tools/coralnpu/phase2_prepare_bazel.sh
./tools/coralnpu/phase2_check_abi.sh
./tools/coralnpu/phase2_build_bridge.sh
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

The Phase-2 bridge ABI is now version 2 and carries deferred external AXI
requests into gem5 coherent DMA. Rebuild both the bridge and gem5 after pulling
this change:

```bash
./tools/coralnpu/phase2_build_bridge.sh
./sim/gem5/apply_patchset.sh
cd thirdparty/gem5
scons build/ARM/gem5.opt -j"$(nproc)"
```
