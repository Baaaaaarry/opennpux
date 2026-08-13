# E2E Qwen Development Plan

## 1. Objective

Bring up a small Qwen-family Transformer model end-to-end on the OpenNPUX
heterogeneous SoC platform. The target execution model is:

```text
ARM Linux guest
  tokenizer / runtime / scheduler
  driver + shared DMA window
        |
        v
gem5 NPUDevice + Coral Verilated bridge
        |
        v
CoralNPU firmware + operator runtime
        |
        +-- hybrid host kernels for functional bring-up
        +-- sampled RTL operators for selected hardware paths
        +-- full RTL operators for performance studies
```

The first acceptance target is one deterministic prompt producing matching
logits checksum and next-token id against a golden host reference. Performance
optimization follows after functional E2E correctness is stable.

## 2. Current Foundation

Already available:

- D9200/D9300 ARM Linux full-system boot with checkpoint flow.
- `NPUDevice` MMIO aperture, shared DMA window, fast DMA, and statistics.
- Kernel driver `/dev/opennpux-coral` and userspace `coralctl`.
- Verilated Coral bridge loaded through C ABI and `dlopen`.
- Stage-A, DMA smoke, driver DMA, command runtime, and model smoke tests.
- RVV Highmem MobileNet path with partial, hybrid, sampled, and full RTL modes.
- Operator descriptor ABI shared by RTL/hybrid/sampled paths.
- Hybrid functional kernels for Conv2D, DepthwiseConv2D, MatMul, FullyConnected,
  Add, Softmax, and LayerNorm.

Main gaps for Qwen:

- No Qwen model packaging/export path.
- No Transformer graph executor in the guest runtime.
- Operator ABI is not yet rich enough for all Qwen shape, broadcast, cache, and
  quantization cases.
- RMSNorm, RoPE, KV cache update, GELU/SiLU, TopK/sampling are not yet complete
  end-to-end operators.
- Full RTL execution of Transformer operators is not performance-viable yet.

## 3. Model Strategy

Use a staged model ladder to avoid blocking on full Qwen complexity:

| Stage | Model | Purpose |
| --- | --- | --- |
| Q0 | synthetic Transformer block | Validate descriptors, tensors, and operator order |
| Q1 | tiny Qwen-compatible graph | Validate prefill/decode shape and KV cache |
| Q2 | quantized small Qwen variant | Validate real tokenizer + real weights |
| Q3 | larger Qwen target | Stress memory, scheduling, and sampled/full RTL |

Initial recommendation: use a tiny Qwen-compatible graph generated from a known
host reference, with one or two layers, small hidden size, fixed sequence
length, and int8 or float32 bring-up tensors. This prevents multi-hour
simulation while preserving the Qwen execution pattern.

### General Model Package

`OPENNPUX_MODEL_PACKAGE_V2` is the scale-out package boundary for real models:

- `tools/models/import_hf_model.py` imports Hugging Face `config.json` and
  safetensors metadata without copying or loading weight payloads.
- The `.npxm` manifest records architecture, dimensions, dtype, required
  operators, tensor count, external shards, and total weight bytes.
- `tensor-index.npxi` maps a tensor name to shard, payload offset, dtype, and
  shape. `runtime/host/src/model_package.c` performs bounded tensor range reads.
- Existing `OPENNPUX_QWEN_TINY_V1` remains the deterministic golden/TCB package;
  it is not presented as a real pretrained model.

This makes 35B-class assets loadable with bounded host memory. It does not yet
make them executable: layer graph lowering, quantized kernels, paged weight
DMA, KV-cache management, tokenizer integration, and multi-token validation
remain subsequent milestones.

## 4. Execution Modes

### Hybrid Mode

Purpose: fastest E2E functional bring-up.

- CPU guest submits model/layer/operator descriptors.
- Coral firmware runs the control path and rings operator doorbells.
- gem5 bridge executes functional host kernels.
- The bridge returns modeled cycles, memory traffic, and output tensors.

Hybrid is the first acceptance mode for Qwen.

### Sampled Mode

Purpose: validate selected RTL hardware paths without running the whole model in
full RTL.

- Same guest/runtime/firmware path as hybrid.
- Selected operators run through RTL or RTL-like sampled implementation.
- Remaining operators use hybrid kernels.
- Compare logits checksum and operator-level checksums against hybrid.

Sampled mode should initially target MatMul/FullyConnected and then RMSNorm or
RoPE.

### Full RTL Mode

Purpose: hardware realism and performance studies.

- Entire operator implementation runs on Coral RTL/RVV/custom RTL path.
- Expected to be slow until tiled MatMul, vector reductions, and burst memory
  movement are implemented.
- Not a daily CI gate in the first Qwen milestone.

## 5. Operator Requirements

Qwen-family inference requires these operator groups:

| Operator | Use | First Implementation | Later RTL Direction |
| --- | --- | --- | --- |
| MatMul/BatchMatMul | Q/K/V, attention score, FFN projections | hybrid int8/fp32 | tiled systolic or vector MAC pipeline |
| FullyConnected/Linear | projection and MLP layers | hybrid int8 | shared MatMul backend |
| Add | residual, bias, mask | hybrid with broadcast | vector ALU pipeline |
| Mul | scale, gate, RoPE components | hybrid with broadcast | vector ALU pipeline |
| RMSNorm/LayerNorm | block normalization | hybrid fp32/int8 bring-up | reduction + rsqrt pipeline |
| Softmax | attention probability | hybrid functional | reduce-max/sum/exp approximation pipeline |
| RoPE | position encoding | hybrid functional | vector sin/cos or lookup pipeline |
| KV cache | decode state update/read | runtime + descriptor | DMA/cache-aware layout |
| GELU/SiLU/SwiGLU | FFN activation/gate | hybrid functional | piecewise/LUT/vector pipeline |
| TopK/Sampling | token selection | CPU fallback first | optional NPU assist |

The first Qwen E2E milestone may keep TopK/sampling on CPU because it is not the
main NPU acceleration target.

## 6. TCB And ABI Extensions

> Architecture boundary: the Qwen TCB below is a bring-up compatibility format,
> not the general SoC submission ABI. New model support must target the generic
> executable, invocation, command-buffer and completion contracts described in
> `docs/adr/0002-generic-npu-submission-architecture.md` and
> `docs/design/generic_npu_runtime.md`. Qwen graph analysis belongs to the
> compiler frontend; the CPU still submits every live inference request.

The current `coral_operator_descriptor` should be extended only through a
reviewed ABI version bump or reserved fields with documented semantics.

Needed fields:

- `graph_id`, `layer_id`, `op_id` for traceability.
- `sequence_length`, `kv_length`, `head_count`, `head_dim`.
- `axis` for Softmax/Norm.
- Broadcast flags for Add/Mul.
- Per-input scale/zero-point and output scale/zero-point.
- Optional fp32/int8 dtype per tensor.
- Scratch arena and tile descriptor.
- KV cache base, stride, and update mode.
- Operator checksum and debug marker.

Implemented Qwen TCB v1 host/runtime staging:

- Header: `magic/version/total_size/op_count`, model shape, prompt checksum,
  expected logits checksum, next token, aggregate ops/bytes/cycles, and TCB
  checksum.
- Per-op descriptor: op index, kind, layer, rank/shape, input/weight/output
  offsets, scratch offset, operations, bytes read/written, and modeled cycles.
- Current builder emits the TCB from parsed `operator_trace` in
  `opennpux_qwen_build_tcb()` and validates it in host unit tests.
- `coralctl qwen-stage-tcb` writes the generated TCB into the NPU shared DMA
  window and validates the staged header through the mapped guest window.
- The generated tiny Qwen TCB is currently 2088 bytes, so it fits in the
  default 4 KiB shared DMA window. Larger models will need chunking or a larger
  `--npu-dma-shared-size`.
- `gem5_qwen_tcb_smoke.elf` walks each descriptor on the NPU side, validates
  opcode/rank bounds, accumulates modeled cycles, computes a device op mask and
  trace checksum, and marks every descriptor complete through reserved fields.
  This keeps the TCB v1 structure size unchanged while adding stronger
  firmware-side execution observability.
- `coralctl qwen-run-tcb` validates device completed-op count, device op mask,
  trace checksum, modeled cycles, and per-descriptor completion markers before
  reporting `qwen_tcb_run=PASS`.
- The host runtime also computes `qwen_expected_trace_checksum` from the staged
  TCB descriptors and requires the firmware-reported
  `qwen_device_trace_checksum` to match exactly. This detects descriptor order,
  opcode, layer/rank, operation count, or modeled-cycle mismatches even when
  aggregate completion status is `PASS`.

Required docs:

- Update `docs/design/coral_operator_abi.md`.
- Add a Qwen-specific model ABI section or new `docs/design/qwen_model_abi.md`.
- Update runtime and firmware headers together.

## 7. Memory Plan

Short-term:

- Keep weights/model assets in the guest filesystem.
- Runtime stages required tensors into shared DMA/EXTMEM windows.
- Use chunked execution for layers when tensors exceed the current shared
  window.

Medium-term:

- Add explicit model weight arena and KV cache arena.
- Add ownership/fence rules between CPU runtime, driver, and NPU firmware.
- Add per-layer reuse to avoid repeatedly copying static weights.

Potential memory windows:

- Shared DMA window: CPU/NPU exchange, descriptors, inputs, outputs, stats.
- Local EXTMEM: NPU tensor arena and operator staging.
- Guest filesystem: model package and tokenizer assets.

## 8. Development Phases

### Phase Q0: Specification And Golden Reference

Deliverables:

- `docs/design/qwen_model_abi.md`.
- Golden reference script for a tiny Qwen-compatible graph.
- Fixed prompt, input ids, expected logits checksum, and next token.
- Operator list extracted from the graph.

Validation:

```bash
./tools/models/prepare_qwen_tiny.sh
./tools/models/run_qwen_golden.sh
```

Acceptance:

```text
qwen_golden=PASS
qwen_logits_checksum=<stable>
qwen_next_token=<stable>
```

Implemented Q0 scripts:

- `tools/models/create_qwen_tiny_model.py`
- `tools/models/prepare_qwen_tiny.sh`
- `tools/models/inspect_qwen_model.py`
- `tools/models/run_qwen_golden.sh`
- `docs/runbooks/qwen_tiny_golden.md`

### Phase Q1: Model Package And Runtime Loader

Deliverables:

- Qwen `.npxm` or equivalent package format.
- Host-side model loader with tensors, weights, graph metadata, and tokenizer
  metadata.
- `coralctl qwen-info` and `coralctl qwen-run` skeleton.
- Standalone `qwen-inspect` host/guest tool for package validation before NPU
  execution.

Validation:

```bash
./tools/models/inspect_qwen_model.sh build/models/qwen-tiny.npxm
./tools/guest_tools/build_coralctl.sh
```

Implemented Q1 loader checks:

```bash
./tools/models/test_qwen_loader.sh
./tools/guest_tools/build_qwen_inspect.sh
./tools/guest_tools/build_coralctl.sh
```

Implemented Q1/Q2 package visibility path:

```bash
IMAGE=/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img \
  ./tools/coralnpu/prepare_qwen_guest_assets.sh
./tools/coralnpu/run_qwen_info_test.sh
```

### Phase Q2: Hybrid Transformer Kernels

Deliverables:

- MatMul/FC broadcast and quantization extensions.
- Add/Mul broadcast.
- RMSNorm or LayerNorm matching the selected graph.
- Softmax axis support.
- RoPE and KV cache functional kernels.
- Unit tests for each kernel.

Validation:

```bash
./tools/coralnpu/test_hybrid_kernels.sh
```

Acceptance:

```text
gem5_hybrid_kernels_test PASS
qwen_operator_kernels=PASS
```

### Phase Q3: Firmware Operator Runtime

Deliverables:

- Qwen firmware entrypoint.
- Layer loop: prefill and single-token decode.
- Descriptor submission per operator.
- Progress markers and operator summaries.
- Error handling and checksum reporting.

Validation:

```bash
./tools/coralnpu/build_qwen_firmware.sh
```

### Phase Q4: Full-System Hybrid E2E

Deliverables:

- Guest asset installer.
- gem5 run script.
- Checkpoint-aware Qwen boot/resume script.
- Host log summarizer and report writer.

Validation:

```bash
./tools/coralnpu/prepare_qwen_guest_assets.sh
./tools/coralnpu/build_rtl_bridge.sh
./tools/coralnpu/run_qwen_e2e_test.sh
```

Acceptance:

```text
[coral-qwen-e2e-test] PASS
qwen_prefill=PASS
qwen_decode=PASS
qwen_logits_checksum=<golden>
qwen_next_token=<golden>
qwen_tcb_run=PASS
qwen_e2e=PASS
```

### Phase Q5: Sampled RTL Bring-up

Deliverables:

- `CORAL_SAMPLED_RTL_OPS=matmul,fc,...` support for Qwen.
- Per-operator mode summary.
- Hybrid-vs-sampled checksum comparison.

Validation:

```bash
CORAL_QWEN_MODE=sampled CORAL_SAMPLED_RTL_OPS=matmul,fc \
  ./tools/coralnpu/run_qwen_e2e_test.sh
```

Acceptance:

```text
qwen_sampled_compare=PASS
sampled_operator_summary=...
```

### Phase Q6: Full RTL Performance Path

Deliverables:

- Tiled MatMul/FC RTL or RVV optimized kernel path.
- Burst EXTMEM access and weight/activation tiling.
- Reduction pipeline for RMSNorm/Softmax.
- Performance counters: cycles, stalls, bytes, burst length, utilization.

Validation:

```bash
CORAL_QWEN_MODE=rtl CORAL_QWEN_DEBUG=1 \
  ./tools/coralnpu/run_qwen_e2e_test.sh
```

Acceptance is not required for daily CI until runtime is practical.

## 9. Issue Split

Recommended parallel Issues:

1. `Q0 golden and tiny Qwen model export`
2. `Qwen model package ABI and loader`
3. `Transformer operator ABI extension`
4. `Hybrid MatMul/FC/Add/Mul broadcast and quantization`
5. `Hybrid RMSNorm/LayerNorm/Softmax/RoPE/KV cache`
6. `Qwen firmware layer scheduler`
7. `coralctl qwen-run and guest runtime`
8. `gem5 Qwen run scripts and checkpoint flow`
9. `Qwen E2E report and log summarizer`
10. `Sampled RTL operator selection for Qwen`
11. `RTL tiled MatMul design`
12. `RTL memory/burst optimization`

Each Issue must include exact allowed directories and validation commands.

## 10. Acceptance Matrix

| Milestone | Mode | Required Result |
| --- | --- | --- |
| Q0 | host golden | deterministic checksum and token |
| Q1 | host loader | model package parsed |
| Q2 | unit hybrid | each Transformer kernel passes |
| Q3 | firmware | descriptors emitted and completed |
| Q4 | gem5 hybrid | E2E Qwen PASS |
| Q5 | gem5 sampled | sampled checksum matches hybrid |
| Q6 | gem5 full RTL | selected layer/operator completes with useful stats |

## 11. Debug Requirements

Every Qwen run must report:

- model name/version/checksum;
- prompt/input checksum;
- mode: `hybrid`, `sampled`, or `rtl`;
- prefill/decode phase markers;
- layer id and operator id;
- operator opcode, shape, dtype, and execution mode;
- cycles, modeled cycles, host ns, bytes read/written;
- output/logits checksum;
- error code and failing descriptor address on failure.

## 12. Non-Negotiable Compatibility Gates

These must keep passing while Qwen work proceeds:

```bash
./tools/coralnpu/build_rtl_bridge.sh
./tools/coralnpu/build_rvv_mobilenet_partial.sh
CORAL_MOBILENET_PARTIAL_DEBUG=1 ./tools/coralnpu/run_rvv_mobilenet_partial.sh
CORAL_OPERATOR_MODE=hybrid ./tools/coralnpu/run_rvv_mobilenet_test.sh
```

If a Qwen PR breaks MobileNet, the PR must either fix it or explicitly update
the shared ABI with a migration plan and reviewer approval.
