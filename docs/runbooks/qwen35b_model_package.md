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
model_tensors=...
model_shards=...
model_weight_bytes=...
```

The generated files are small metadata artifacts:

- `model.npxm`: versioned architecture and shard manifest.
- `tensor-index.npxi`: tensor name to shard/offset/shape index.
- Original `*.safetensors`: unchanged external weight payloads.

## Current Boundary

Passing this gate proves that the platform can ingest and range-address the
real 35B model without whole-model memory allocation. The next implementation
uses the observed tensor names to lower each Transformer layer into TCBs and
adds paged weight DMA, quantized MatMul, KV-cache allocation, tokenizer input,
and iterative decode. The existing tiny Qwen golden/TCB path remains the
functional control-flow reference.
