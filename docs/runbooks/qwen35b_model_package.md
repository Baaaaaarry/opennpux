# Qwen 35B General Model Package

This gate validates real Hugging Face model assets before graph lowering or
gem5 execution. Qwen3.5 is the first frontend and acceptance model for the
generic NPU platform; it is not the driver, queue, or device ABI. This step
does not copy weight shards, load them into RAM, or submit an NPU task.

## GB10 Validation

Assume the downloaded model directory contains `config.json` and either
`model.safetensors` or `model.safetensors.index.json` plus all referenced
shards:

```sh
cd /home/barry/code/opennpux
./tools/models/prepare_hf_model_package.sh \
  /data/models/Qwen3.5-35B \
  /data/models/Qwen3.5-35B/model.npxm
```

`/data/models/Qwen3.5-35B` is an example only. Replace it with the actual
download location. To find an existing Hugging Face cache snapshot:

```sh
find "${HF_HOME:-$HOME/.cache/huggingface}/hub" \
  -path '*/snapshots/*/config.json' -print
```

The preparation script also accepts a Hugging Face repository id and resolves
it from the local cache without network access:

```sh
./tools/models/prepare_hf_model_package.sh ORGANIZATION/MODEL_NAME
```

For an explicit directory that does not yet contain assets, the preparation
script downloads `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` from `hf-mirror.com` by
default. Downloads use `.part` files, resume with HTTP ranges, and are renamed
atomically after completion:

```sh
./tools/models/prepare_hf_model_package.sh /data/models/Qwen3.5-35B
```

The destination must be writable by the invoking user. For a root-owned
`/data/models` mount, initialize the model directory once:

```sh
sudo install -d -o "$(id -un)" -g "$(id -gn)" \
  /data/models/Qwen3.5-35B
```

Configuration:

```text
OPENNPUX_HF_AUTO_DOWNLOAD=0     disable automatic download
OPENNPUX_HF_MODEL_REPO=...     override repository id
OPENNPUX_HF_REVISION=...       override revision (default: main)
HF_ENDPOINT=...                 override endpoint
```

The downloader checks index-declared weight size against available disk space
and fetches only config, tokenizer files, the safetensors index, and referenced
weight shards.

Expected final lines:

```text
model_package=PASS
hf_model_package_prepare=PASS
```

Retain the complete output. In particular, report:

```text
model_architecture=...
model_dtype=...
model_layers=...
model_hidden=...
model_intermediate=...
model_heads=...
model_kv_heads=...
model_head_dim=...
model_experts=...
model_experts_per_token=...
model_moe_intermediate=...
model_shared_expert_intermediate=...
model_tensors=...
model_shards=...
model_weight_bytes=...
model_quantization=...
model_quantization_bits=...
model_quantization_group_size=...
```

The generated files are small offline compiler metadata artifacts:

- `model.npxm`: versioned architecture and shard manifest.
- `tensor-index.npxi`: tensor name to shard/offset/shape index.
- `execution-plan.npxp`: Qwen frontend tensor-role and decoder-lowering
  inventory. It contains no live invocation or device addresses.
- `model.npxe`: inspectable generic NPU executable metadata.
- `model.npxc`: binary prefill/decode command templates loaded by the CPU
  runtime. Runtime tensor addresses and state are deliberately unresolved.
- Original `*.safetensors`: unchanged external weight payloads.

## Current Boundary

Passing this gate proves that the deployment toolchain can ingest and
range-address the real 35B model without whole-model memory allocation. The
next implementation lowers the observed graph into a generic NPU executable.
For every prefill/decode request, the CPU runtime must still bind live tensors,
dynamic dimensions, KV/recurrent state and synchronization, then submit a
command-buffer job to the NPU. The existing Qwen tiny TCB remains a bring-up
regression fixture and must not be extended into the stable platform ABI.

## Execution Plan Inventory

After updating the repository, regenerate only metadata; downloaded shards are
reused:

```sh
./tools/models/prepare_hf_model_package.sh \
  /data/models/Qwen3.5-35B \
  /data/models/Qwen3.5-35B/model.npxm

cat /data/models/Qwen3.5-35B/execution-plan.npxp
```

Expected terminal verdict:

```text
qwen_plan_layers=40/40
qwen_plan_experts=256/256
qwen_plan_layer_types=...
qwen_plan_domains=...
qwen_plan_unknown_patterns=...
qwen_plan_unknown_decoder_patterns=0
npu_executable_commands=...
npu_command_template_bytes=...
npu_invocation_bytes_upper_bound=...
npu_executable=PASS
npu_weight_plan_commands=524
npu_weight_plan_mapped_commands=343
npu_weight_plan_weightless_commands=181
npu_weight_plan_unresolved_weight_commands=0
npu_weight_range_records=...
npu_weight_range_bytes=...
npu_weight_plan=PASS
qwen_execution_plan=PASS
```

Inspect the complete range index and estimate runtime paging with only the
router-selected experts active. The tool enumerates metadata for every page,
but reads only the first page of each mapped command:

```sh
./tools/models/inspect_npu_weight_pages.sh \
  /data/models/Qwen3.5-35B/model.npxm \
  /data/models/Qwen3.5-35B/model.npxr
```

The verdict must include `weight_inspect_commands=524`,
`weight_inspect_mapped_commands=343`, `weight_inspect_active_experts=8`, and
`weight_inspect=PASS`. `weight_inspect_page_requests` is the first measured
full-weight paging requirement and is used to size cache and DMA modeling.

Compare 4KiB paging against a 64KiB transfer block before selecting the DMA
protocol granularity:

```sh
./tools/models/inspect_npu_weight_pages.sh \
  /data/models/Qwen3.5-35B/model.npxm \
  /data/models/Qwen3.5-35B/model.npxr 4096
./tools/models/inspect_npu_weight_pages.sh \
  /data/models/Qwen3.5-35B/model.npxm \
  /data/models/Qwen3.5-35B/model.npxr 65536
```

Compare `weight_inspect_page_requests` for metadata/interrupt pressure,
`weight_inspect_request_bytes` for padded transfer volume, and
`weight_inspect_bytes_read` for sampled host I/O. The 64KiB mode is a transfer
coalescing experiment, not a declaration that the device page size is 64KiB.

The plan classifies the text decoder independently from vision and MTP tensors,
so similarly numbered auxiliary layers cannot contaminate the 40 decoder-layer
templates. Each text layer is currently classified as `full_attention_moe`,
`linear_attention_moe`, or `unclassified_moe`. Full-attention layers receive
paged KV-cache phases; linear-attention layers receive causal-convolution and
recurrent-state phases.

The exact `qwen_plan_tensor_roles`, `qwen_plan_layer_types`, and
`qwen_plan_unknown_patterns` output is required for the graph-lowering
increment. Text-domain unknown patterns are inventory diagnostics because
legal graph-level constants may not belong to a decoder layer. A nonzero
`qwen_plan_unknown_decoder_patterns` count or an `unclassified_moe` layer is a
hard executable-compilation failure. The scheduler uses paged range reads and
router-TopK active experts only; it never schedules all 256 experts as resident
state.

## Generic Command Processor Smoke

After the real model produces `model.npxc`, build the command processor and
guest tool. The first run must create a checkpoint whose device tree reserves
the standard 8MiB NPU shared window. The compiled 524-command invocation is
larger than 64KiB once numerical operator parameter records are included:

Before simulation, validate that every packed GPTQ tensor has an unambiguous
`qweight/qzeros/scales[/g_idx]` binding:

```sh
./tools/models/inspect_gptq_bindings.py \
  /data/models/Qwen3.5-35B/model.npxr --require-complete
```

Re-run `prepare_hf_model_package.sh` after pulling a runtime ABI update. The
current operator-parameter ABI is version 2 and records the GPTQ scale data
type; stale `model.npxc` and `model.npxr` files must be regenerated so FP16
scales are not interpreted as FP32.

## Real GPTQ Projection

Build the bridge, external-request firmware, and current guest tool, then run
one real layer-0 routed-expert gate projection:

```sh
./tools/coralnpu/build_gptq_matmul_smoke.sh
./tools/guest_tools/build_coralctl.sh

CORAL_MODEL_DIR=/data/models/Qwen3.5-35B \
  ./tools/coralnpu/run_gptq_projection_test.sh
```

The first run needs an 8MiB-window checkpoint; set `CORAL_REBUILD_CKPT=1` only
when that checkpoint does not already exist. The expected verdict includes
`gptq_projection_scale_dtype=4`, `gptq_projection_state=0x00000002`, a zero
error, `gptq_projection_reference=PASS`, and `gptq_projection_run=PASS`.
Select another real projection with `CORAL_GPTQ_LAYER`, `CORAL_GPTQ_EXPERT`
and `CORAL_GPTQ_SLOT=gate_proj|up_proj|down_proj`.

## Fast Full-System Paging And Real Token

The Qwen35B full-system test defaults to sim-host direct paging. The host
materializes a streaming `.npxb` page bundle once; the Verilated bridge then
services the firmware page-fault queue directly in Local EXTMEM. The NPU still
publishes and retires the same queue records, but the simulated ARM CPU no
longer reads safetensors over 9P or copies each 64KiB page.

The default acceptance path also runs real Hugging Face autoregressive decode
on the simulation host. By default it applies the tokenizer's own chat
template with a user message and generation prompt before tokenization; set
`CORAL_QWEN_PROMPT_FORMAT=raw` only when deliberately testing plain-text
continuation. It uses the model KV cache and defaults to eight new tokens. The
generated token IDs, text and per-step logits checksum are
cached in a 256-byte `.npxo` v2 result record. The bridge validates the executable
ID, checksum of the original CPU prompt, vocabulary size and requested token
limit, then publishes
that result only after the NPU firmware has completed the entire 524-command
invocation. This is a hybrid
numerical path: command scheduling, paging and completion remain NPU-visible,
while the current full-model numerical kernel runs on the host. It must not be
reported as full-RTL numerical execution.

The fixed 256-byte inference I/O header carries result metadata and token text.
The complete token-ID sequence is published in the output binding at
`OPENNPUX_NPU_INFERENCE_TOKEN_IDS_OFFSET`; header fields describe its offset and
byte length. This supports all 32 result tokens without truncating Guest output
to the former 12 inline IDs. `coralctl` validates the external array bounds,
every token against the vocabulary, and the first token against `next_token`.

The numerical generator also honors the model's local `generation_config.json`
by default. For the pinned Qwen3.5 GPTQ package this enables sampling with the
model-provided temperature, top-k and top-p values. A fixed seed, default 42,
makes the resulting golden reproducible. Set
`CORAL_QWEN_GENERATION_SEED=<n>` to select another deterministic sample or
`CORAL_QWEN_DECODE_MODE=greedy` for a strict argmax regression. Decode mode and
seed are part of the cache key. Model-mode generation rejects an all-identical
multi-token result instead of accepting a degenerate loop as a golden; use
`OPENNPUX_ALLOW_DEGENERATE_OUTPUT=1` only for diagnosis.

The default `CORAL_QWEN_MODEL_LOADER=transformers` follows the model card's
multimodal text path with `AutoModelForMultimodalLM` and `AutoProcessor`, even
for a text-only prompt. This matters because Qwen3.5 packages text and vision
under a conditional-generation architecture. The former direct
`GPTQModel.load` path bypassed the official processor/model pairing and produced
multilingual token fragments despite a valid transport golden. It remains
available as `CORAL_QWEN_MODEL_LOADER=gptqmodel` for backend diagnosis only.
The loader name and quantization backend are part of the result cache key.

Install a recent model-compatible `torch`, `transformers`, `accelerate` and
`numpy` environment before the first run. The GPTQ model may additionally need
the quantization backend required by the selected Transformers release.

Create the repository-local environment once:

```sh
./tools/models/setup_hf_numerical_env.sh
```

The setup uses `--system-site-packages`, so an optimized PyTorch already
installed on GB10 is reused. If PyTorch is absent, it installs the default pip
build. Set `OPENNPUX_TORCH_INDEX_URL` when the platform requires a specific
PyTorch wheel index. If Transformers reports that the GPTQ backend is missing,
rerun the setup script; it installs Optimum and GPTQModel by default. Set
`OPENNPUX_INSTALL_GPTQMODEL=0` only for non-GPTQ models. Override the
interpreter used by the simulation with `CORAL_HF_PYTHON=/path/to/python`.
Qwen3.5 imports vision utilities even for the text next-token path, so the
setup also installs a `torchvision` build from the same PyTorch wheel index.
The model directory must also contain `preprocessor_config.json`; the runner
detects old downloads missing that file and uses `download_hf_model.sh` to add
the processor metadata without downloading existing weight shards again.
The real-token generator defaults GPTQ loading to the portable
`gptq_torch` backend because Qwen3.5 contains projections whose output width is
not divisible by 64 and therefore cannot use Marlin. The official Transformers
loader receives this setting through an explicit `GPTQConfig`; relying on auto
selection can instantiate `MarlinLinear` and fail on an `out_features=32`
projection. Override it only after validating every model shape with
`CORAL_QWEN_GPTQ_BACKEND=<backend>`. A successful load reports
`hf_numerical_backend=transformers-multimodal:gptq_torch` and
`hf_numerical_quant_backend=gptq_torch`.
PyTorch 2.10.0 has a known Python 3.12 TorchInductor `CSE` generic annotation
regression and has no 2.10.1 bug-fix release. The generator detects that exact
API mismatch and applies a process-local second-parameter default before
GPTQModel imports TorchInductor; it reports
`hf_numerical_torch_cse_compat=1` when the workaround is active.

CUDA is not a gem5, Verilator, Coral RTL, AXI or DMA dependency. It is used only
to accelerate the optional host-side Hugging Face forward that creates the
real-token `.npxo` reference. CPU-only PyTorch is valid but substantially
slower for the 35B model. Once an `.npxo` record exists, subsequent runs reuse
it without importing PyTorch or requiring CUDA. Full-RTL runs can disable this
path with `CORAL_SIM_HOST_NUMERICAL=0`; their numerical result must then come
from implemented NPU firmware/RTL kernels.

```sh
CORAL_MODEL_DIR=/data/models/Qwen3.5-35B \
  CORAL_QWEN_PROMPT='Explain heterogeneous computing in one sentence.' \
  CORAL_QWEN_PROMPT_FORMAT=chat \
  CORAL_QWEN_DECODE_MODE=model \
  CORAL_QWEN_MODEL_LOADER=transformers \
  CORAL_QWEN_GENERATION_SEED=42 \
  CORAL_QWEN_MAX_NEW_TOKENS=8 \
  CORAL_SIM_HOST_PAGING=1 \
  CORAL_SIM_HOST_NUMERICAL=1 \
  ./tools/coralnpu/run_qwen35b_real_weights_test.sh
```

The first run prints `sim_host_bundle=PASS`; later runs reuse
`build/model-pages/qwen35b-functional-64k.npxb`. It also prints
`hf_numerical=PASS` and caches the prompt-specific result under
`build/model-results/`. Set
`CORAL_REBUILD_SIM_HOST_BUNDLE=1` to force regeneration, or
`CORAL_SIM_HOST_PAGING=0` to retain the cycle-slow Guest paging reference.
Set `CORAL_REBUILD_SIM_HOST_RESULT=1` to rerun the real model forward, or
`CORAL_SIM_HOST_NUMERICAL=0` to disable host numerical publication.
Set `CORAL_QWEN_MAX_NEW_TOKENS=1..32` to control decode length. The token
count is part of the cache filename, so results for different lengths cannot be
reused accidentally. The prompt format is also part of the cache filename, so a
legacy raw-continuation result cannot be reused by the default chat path. The
generator reports both `hf_numerical_prompt_checksum` for the original CPU ABI
payload and `hf_numerical_formatted_prompt_checksum` for the actual model input.

The runner reads the tokenizer-derived input length from `.npxo` v2 and passes
it to Guest runtime as `OPENNPUX_INPUT_TOKEN_COUNT`. Decode invocation metadata
therefore reports `runtime_sequence=<prompt tokens>` and
`runtime_kv=<prompt tokens + generated tokens - 1>` instead of the former
fixed `1/1` shape. The bridge rejects a numerical result whose input-token
count differs from the submitted inference request.

The RV32 command processor replays the executable command template once per
requested output token. Step zero is the prefill step and uses the submitted
prompt sequence length; subsequent steps use sequence length one while retaining
the final KV length in runtime metadata. For the current 524-command template
and eight generated tokens, acceptance therefore requires
`execution_steps=8`, `expected_executed_commands=4192`, and
`completed_commands=4192`. Trace command counts, page requests and modeled
cycles cover all eight steps rather than one template traversal.

Hybrid sim-host numerical runs default to immutable decode-weight reuse. The
prefill step services the complete page-fault stream once; later decode steps
reuse that modeled weight residency while still executing and accounting every
command. This avoids seven redundant transfers of identical read-only weights.
Set `CORAL_REUSE_DECODE_WEIGHTS=0` to retain the hardware-reference behavior
that pages weights on every token. The Guest reports
`inference_decode_weight_policy=reuse-after-prefill|page-every-step`.

Acceptance requires `paging_source=sim-host-direct`, equal
`paging_queue_producer/service/retire` values, `inference_mode=numerical`,
`inference_result_source=sim-host-numerical`, a nonempty
`inference_token_text`, a nonzero `inference_generated_tokens`,
an untruncated `inference_token_ids` sequence whose first element equals
`inference_next_token`,
and token IDs/text equal to the Hugging Face golden printed when the `.npxo`
record was generated,
`inference_run=PASS`, `executable_run=PASS`, and
`[coral-qwen35b-real-weights-test] PASS`. The generated Guest script now
compares the complete returned token-ID sequence and generated-token count with
the host `.npxo` record; successful automated comparison prints
`[coral-qwen35b-real-weights-test] token_golden=PASS`.

Before booting, the runner recomputes the staged image on the host and injects
the result into the guest script, so the device must match
`gptq_projection_expected_checksum` and
`gptq_projection_expected_operations_low` exactly. Inspect the same expectation
without a simulation run:

```sh
./tools/models/materialize_gptq_projection.py \
  /data/models/Qwen3.5-35B/model.npxm \
  /data/models/Qwen3.5-35B/model.npxw \
  /data/models/Qwen3.5-35B/model.npxr \
  /tmp/gptq-projection.bin --layer 0 --expert 0 --slot gate_proj
./tools/models/gptq_reference.sh /tmp/gptq-projection.bin
./tools/coralnpu/check_gptq_matmul_abi.sh
```

`gptq_reference.sh` compiles `runtime/host/src/npu_gptq_reference.c` with
`-ffp-contract=off`. That flag is mandatory: the reference accumulates in
float32 in the same order as the bridge kernel, and a fused multiply-add would
change the rounding and therefore the checksum.

## Real GPTQ Gated MLP Expert

After the individual projection gate passes, validate one complete routed
expert. This path consumes the real `gate_proj`, `up_proj`, and `down_proj`
GPTQ tensors and executes the model-independent topology
`SiLU(gate(input)) * up(input) -> down`:

```sh
./tools/coralnpu/build_gptq_matmul_smoke.sh
./tools/guest_tools/build_coralctl.sh

CORAL_MODEL_DIR=/data/models/Qwen3.5-35B \
  CORAL_GPTQ_LAYER=0 \
  CORAL_GPTQ_EXPERT=0 \
  ./tools/coralnpu/run_gptq_expert_test.sh
```

The expected guest verdict is `[coral-gptq-expert-test] PASS`. The runner
checks completion/error state, the exact analytical operation count, non-zero
checksums at all four stage boundaries, and modeled cycles. Select another
expert or token batch with `CORAL_GPTQ_LAYER`, `CORAL_GPTQ_EXPERT`, and
`CORAL_GPTQ_ROWS`. The implementation is a hybrid functional/timing model of a
generic gated MLP expert, not a Qwen-specific kernel or dedicated RTL unit.

The CPU runtime can generate the same request online with
`opennpux_npu_gptq_expert_stage()`. It resolves the selected command, role and
active expert from `model.npxr`, binds the caller's live input tensor, and
produces the same 256-byte request plus payload layout as
`materialize_gptq_expert.py`. `test_model_package.sh` enforces byte-for-byte
equivalence between these two producers. Run the ABI guard independently with:

```sh
./tools/coralnpu/check_gptq_expert_abi.sh
```

```sh
./tools/coralnpu/build_rtl_bridge.sh
./tools/guest_tools/build_coralctl.sh
./tools/coralnpu/test_hybrid_kernels.sh
./tools/models/compile_npu_executable.py \
  /data/models/Qwen3.5-35B/model.npxm \
  /data/models/Qwen3.5-35B/execution-plan.npxp \
  /data/models/Qwen3.5-35B/model.npxe

CORAL_NPU_EXECUTABLE=/data/models/Qwen3.5-35B/model.npxc \
CORAL_REBUILD_CKPT=1 \
  ./tools/coralnpu/run_generic_executable_test.sh

CORAL_NPU_EXECUTABLE=/data/models/Qwen3.5-35B/model.npxc \
  ./tools/coralnpu/run_generic_executable_test.sh
```

The first command only boots Linux and saves the new checkpoint. The second
command restores it and runs the smoke test. The run script injects the current
host-built `coralctl` into `/tmp`, so later runtime-only changes do not require
editing the disk image or rebuilding the checkpoint. Later runs need only the
second command. Expected guest verdict:

```text
submitted_commands=524
completion_state=2
completion_error=0
completed_commands=524
runtime_batch=1
runtime_sequence=1
runtime_kv=1
runtime_active_experts=8
weight_binding=2
state_binding=3
scratch_binding=4
relocated_commands=524
parameter_checksum=0x...
dispatch_dependency_edges=523
dispatch_estimated_operations=...
dispatch_estimated_bytes=...
dispatch_weight_page_requests=343
dispatch_weight_dma_bytes=1372
dispatch_weight_checksum=0x...
dispatch_modeled_cycles=...
dispatch_op_MATMUL=count:...,operations:...,bytes:...
dispatch_op_EXPERT=count:...,operations:...,bytes:...
executable_run=PASS
[coral-executable-run-test] PASS
```

This is a control-path milestone: CPU runtime submission, NPU command fetch,
validation, compact dynamic-parameter relocation, resource-binding resolution,
dependency scheduling, capability dispatch, workload accounting, traversal
and completion are real. Numerical Qwen inference still requires variable-size
operator parameter blocks, complete GPTQ page streaming, operator execution
and prefill/decode state management. Parameter symbols are now mapped to real
safetensors ranges in `model.npxw`; the smoke test materializes the first
mapped tensor payload by default. Set
`CORAL_NPU_WEIGHT_PAGE=/path/to/page.bin` to override it.

## Full Paged Control-Plane Acceptance

After the small 30-command paging smoke passes, run the complete 524-command
Qwen3.5 decode schedule through the same RV32 firmware and shared fault ring:

```sh
CORAL_MODEL_DIR=/data/models/Qwen3.5-35B \
  ./tools/coralnpu/run_qwen35b_paged_control_test.sh
```

This gate deliberately uses one repeated 64KiB page per weight-bearing command.
It validates command-scale control flow, not Qwen numerical correctness. The
script requires 524 submitted and completed commands, 343 serviced/retired page
faults, zero queue backpressure, and the final executable PASS verdict. Real
GPTQ range streaming and numerical kernels are the following data-plane gate.
