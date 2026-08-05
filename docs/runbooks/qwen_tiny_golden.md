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

## Follow-Up Integration

This milestone does not execute inside gem5 yet. The next milestones should
consume `build/models/qwen-tiny.npxm` from:

- host-side runtime loader;
- `qwen-inspect-aarch64` inside the guest image;
- `coralctl qwen-info` / `coralctl qwen-run`;
- firmware operator scheduler;
- hybrid/sampled operator tests;
- full-system `run_qwen_e2e_test.sh`.
