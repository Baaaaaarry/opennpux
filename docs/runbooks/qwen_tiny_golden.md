# Qwen Tiny Golden Acceptance

This runbook validates the first E2E Qwen development milestone without
touching gem5, Coral RTL, kernel, or guest images.

The goal is to create a deterministic tiny Qwen-compatible package that later
runtime, firmware, hybrid, sampled, and RTL work can use as a shared contract.

## Build The Package

```bash
./tools/models/prepare_qwen_tiny.sh
```

Default output:

```text
build/models/qwen-tiny.npxm
```

The file is a JSON model package for the Q0 milestone. It contains:

- model metadata;
- prompt/input ids;
- deterministic golden logits;
- next-token id;
- logits and weight checksums;
- ordered Transformer operator trace.

## Inspect The Package

```bash
./tools/models/inspect_qwen_model.py build/models/qwen-tiny.npxm
```

Expected output includes:

```text
qwen_format=OPENNPUX_QWEN_TINY_V1
qwen_operator_count=19
qwen_ops=ADD,EMBED,MATMUL,MUL,RMS_NORM,ROPE,SILU,SOFTMAX,TOPK
qwen_inspect=PASS
```

## One-Shot Golden Test

```bash
./tools/models/run_qwen_golden.sh
```

Expected output ends with:

```text
qwen_inspect=PASS
qwen_golden=PASS
```

The exact `qwen_logits_checksum` and `qwen_next_token` values are the acceptance
contract for subsequent Qwen runtime and full-system work. If they change, the
PR must explain whether the model definition changed intentionally.

## Host Loader Test

Validate the C model loader and host-side inspect tool:

```bash
./tools/models/test_qwen_loader.sh
```

Expected output includes:

```text
qwen_op_mask=0x000001ff
PASS: qwen model host unit tests
qwen_loader=PASS
```

Build the guest/aarch64 inspect binary:

```bash
./tools/guest_tools/build_qwen_inspect.sh
```

Expected output:

```text
built: build/guest-tools/qwen-inspect-aarch64
```

## coralctl Qwen Info

Build `coralctl` with Qwen package inspect support:

```bash
./tools/guest_tools/build_coralctl.sh
```

Host-side smoke check:

```bash
cc -O2 -Wall -Wextra -Werror -std=c11 \
  -Iruntime/host/include \
  runtime/host/src/coral_runtime.c \
  runtime/host/src/qwen_model.c \
  runtime/host/tools/coralctl.c \
  -o build/local-tests/coralctl-host

build/local-tests/coralctl-host qwen-info build/models/qwen-tiny.npxm
```

Expected output includes:

```text
qwen_op_mask=0x000001ff
qwen_next_token=7
qwen_logits_checksum=0x829e9f00
qwen_info=PASS
```

## Guest Asset Install

Install `coralctl`, `qwen-inspect`, and `qwen-tiny.npxm` into the guest image:

```bash
CORAL_DISK_IMG=/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img \
  ./tools/coralnpu/prepare_qwen_guest_assets.sh
```

Rebuild the boot checkpoint once after changing the image:

```bash
cd thirdparty/gem5
CORAL_NPU_BACKEND=stage-a \
CORAL_REBUILD_CKPT=1 \
CORAL_DISK_IMG=/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
./run_multicore.sh
```

The checkpoint boot log should include:

```text
[coralnpu] preloaded qwen-tiny into checkpoint tmpfs
```

If the restored test reports `qwen model missing`, the running checkpoint was
created before `qwen-tiny.npxm` was installed, or it was created from a
different disk image. Re-run the asset install command against the same image
used by `CORAL_DISK_IMG`, then rebuild the checkpoint with `CORAL_REBUILD_CKPT=1`.

Then validate Qwen package visibility inside the restored guest:

```bash
cd ../..
./tools/coralnpu/run_qwen_info_test.sh
```

Expected guest output:

```text
[coral-qwen-info-test] started
qwen_model_path=/tmp/qwen-tiny.npxm
qwen_model=qwen-tiny-synthetic
qwen_info=PASS
[coral-qwen-info-test] PASS
```

Validate the checkpoint-aware Qwen runtime command path:

```bash
CORAL_DISK_IMG=/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img \
  ./tools/coralnpu/run_qwen_e2e_test.sh
```

Expected guest output:

```text
[coral-qwen-test] started
qwen_model_path=/tmp/qwen-tiny.npxm
qwen_model=qwen-tiny-synthetic
qwen_mode=hybrid-sim
qwen_prefill=PASS
qwen_decode=PASS
qwen_completed_operators=19
qwen_op_00=EMBED layer=none shape=none ...
qwen_op_01=RMS_NORM layer=0 shape=4x8 ...
qwen_operation_count=<non-zero>
qwen_modeled_cycles=<non-zero>
qwen_logits_checksum=0x829e9f00
qwen_next_token=7
qwen_run=PASS
[coral-qwen-test] PASS
```

## Follow-Up Integration

This milestone validates the guest command path inside gem5 using the package
golden result. The next milestones should replace the `golden-package`
execution mode with real operator dispatch while keeping the same acceptance
fields:

- firmware operator scheduler;
- hybrid/sampled operator tests;
- full-system `run_qwen_e2e_test.sh`.
