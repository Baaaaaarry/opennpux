# Task: E2E Qwen On OpenNPUX Heterogeneous SoC

## Task

- Issue: create an Epic named `E2E Qwen bring-up`
- Owner: TBD
- Branch: `feature/<issue-id>-e2e-qwen-*`
- Subsystem: `gem5`, `coralnpu`, `runtime`, `tools`, `docs`, `tests`
- Priority: high

## Goal

Run a small Qwen-family Transformer model end-to-end on the OpenNPUX
gem5+CoralNPU heterogeneous SoC platform:

- ARM Linux guest runs tokenizer, model runtime, scheduler, and system control.
- CPU submits Transformer operator tasks to CoralNPU through driver/runtime.
- NPU executes tasks through hybrid/sampled/full-RTL modes depending on
  operator readiness.
- The flow produces deterministic logits/token output, operator statistics,
  checksum, and a reproducible acceptance report.

The first target is functional correctness and platform completeness, not final
NPU performance.

## Scope

In scope:

- Define Qwen model subset and graph/operator contract.
- Extend model packaging beyond the current smoke `.npxm` path.
- Support Transformer descriptors for MatMul/FC/Add/Mul/Softmax/LayerNorm or
  RMSNorm/RoPE/KV cache/GELU or SiLU/TopK as required by the selected model.
- Implement hybrid functional kernels first.
- Add sampled mode for selected RTL-backed operators.
- Add run scripts, guest assets, golden-output comparison, and reports.
- Keep current MobileNet, DMA, driver, and command tests passing.

Out of scope for the first milestone:

- Full production tokenizer coverage for every Qwen variant.
- Large Qwen deployment such as multi-billion parameter models.
- Bit-exact optimized quantization for arbitrary external model exports.
- Final high-performance RTL for every Transformer operator.
- Linux userspace integration with a full upstream inference framework unless
  a minimal runtime path is already stable.

## Interface Impact

- SoC memory map: likely no change for first milestone; revisit if Qwen weights
  exceed current shared/EXTMEM window.
- NPU CSR/MMIO: no change expected.
- Shared DMA window: may need larger configurable size and chunked staging.
- TCB/operator descriptor: yes, extend for batched MatMul, broadcast,
  normalization, RoPE, KV cache, and TopK metadata.
- Kernel UAPI: no new ioctl expected initially; may add model-buffer pinning or
  larger shared mapping later.
- Verilated bridge C ABI: may need capability bits and operator mode reporting
  for new Transformer opcodes.
- Checkpoint/image/kernel boot contract: guest image must include new runtime
  assets and model files; checkpoint rebuild required after image updates.
- Operator semantics: yes, must define quantization, shape, axis, broadcast,
  cache layout, and numerical tolerance.

## Expected Files

Files/directories Codex may edit:

- `docs/design/`
- `docs/runbooks/`
- `docs/process/`
- `runtime/host/`
- `runtime/kernel/` only if UAPI is reviewed
- `tools/coralnpu/`
- `tools/models/`
- `tests/unit/`
- `tests/sim/`
- `sim/gem5/`
- `sim/coralnpu/`

Files/directories Codex must not edit as source of truth:

- `thirdparty/gem5`
- `thirdparty/coralnpu`
- build outputs, caches, disk images, `m5out`, `simout`

## Validation Plan

Baseline commands:

```bash
./sim/gem5/apply_patchset.sh
./sim/coralnpu/apply_patchset.sh
./tools/coralnpu/check_overlay_boundary.sh
./tools/coralnpu/build_rtl_bridge.sh
./tools/coralnpu/build_rvv_mobilenet_partial.sh
CORAL_MOBILENET_PARTIAL_DEBUG=1 ./tools/coralnpu/run_rvv_mobilenet_partial.sh
CORAL_OPERATOR_MODE=hybrid ./tools/coralnpu/run_rvv_mobilenet_test.sh
```

Qwen target commands to add:

```bash
./tools/models/prepare_qwen_tiny.sh
./tools/coralnpu/build_qwen_firmware.sh
./tools/coralnpu/prepare_qwen_guest_assets.sh
CORAL_QWEN_MODE=hybrid ./tools/coralnpu/run_qwen_e2e_test.sh
CORAL_QWEN_MODE=sampled CORAL_SAMPLED_RTL_OPS=matmul,fc \
  ./tools/coralnpu/run_qwen_e2e_test.sh
```

Expected output:

```text
[coral-qwen-test] started
qwen_model=<selected-model>
qwen_prompt_checksum=...
qwen_prefill=PASS
qwen_decode=PASS
qwen_logits_checksum=...
qwen_next_token=<id>
qwen_operator_summary=...
[coral-qwen-test] PASS
```

## Risks

- Compatibility: Qwen exported graph may not map cleanly to current TCB v6.
- Performance: full RTL Transformer execution is expected to be too slow until
  tiled MatMul and memory movement are modeled/accelerated.
- Debuggability: need phase markers per layer/operator and host-side summaries.
- Memory: weights and KV cache may exceed current guest shared window.
- Rollback: keep MobileNet acceptance as baseline and gate all new Qwen paths
  behind new scripts/options.

## Codex Prompt

```text
Read AGENTS.md and the nearest subsystem AGENTS.md. Work only on the assigned
E2E Qwen Issue. Keep thirdparty as upstream source; put lasting gem5 changes in
sim/gem5 and CoralNPU changes in sim/coralnpu. Do not touch unrelated files.
If changing ABI/memmap/DMA/UAPI/operator semantics, update docs and stop if no
reviewed design exists. Preserve MobileNet hybrid/sampled validation. Run the
Issue validation commands when possible and report exact commands and outputs.
```

