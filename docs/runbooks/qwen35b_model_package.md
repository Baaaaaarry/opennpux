# Qwen 35B General Model Package

This gate validates real Hugging Face model assets before graph lowering or
gem5 execution. It does not copy weight shards and does not load them into RAM.

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

The generated files are small metadata artifacts:

- `model.npxm`: versioned architecture and shard manifest.
- `tensor-index.npxi`: tensor name to shard/offset/shape index.
- `execution-plan.npxp`: compact tensor-role and decoder scheduling inventory.
- Original `*.safetensors`: unchanged external weight payloads.

## Current Boundary

Passing this gate proves that the platform can ingest and range-address the
real 35B model without whole-model memory allocation. The next implementation
uses the observed tensor names to lower each Transformer layer into TCBs and
adds paged weight DMA, quantized MatMul, KV-cache allocation, tokenizer input,
and iterative decode. The existing tiny Qwen golden/TCB path remains the
functional control-flow reference.

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
qwen_execution_plan=PASS
```

The plan classifies the text decoder independently from vision and MTP tensors,
so similarly numbered auxiliary layers cannot contaminate the 40 decoder-layer
templates. Each text layer is currently classified as `full_attention_moe`,
`linear_attention_moe`, or `unclassified_moe`. Full-attention layers receive
paged KV-cache phases; linear-attention layers receive causal-convolution and
recurrent-state phases.

The exact `qwen_plan_tensor_roles`, `qwen_plan_layer_types`, and
`qwen_plan_unknown_patterns` output is required for the graph-lowering
increment. Any nonzero text-domain unknown count must be reviewed before it is
treated as executable coverage. The scheduler uses paged range reads and
router-TopK active experts only; it never schedules all 256 experts as resident
state.
