#!/usr/bin/env python3
"""Create an OpenNPUX v2 manifest for external Hugging Face safetensors.

The converter records model configuration and shard metadata but never copies
or loads tensor payloads. This keeps import memory bounded for 35B-class models.
"""

import argparse
import json
import struct
from pathlib import Path
from typing import Any


FORMAT = "OPENNPUX_MODEL_PACKAGE_V2"
REQUIRED_QWEN_OP_MASK = 0x1FF
TENSOR_INDEX_MAGIC = 0x5458504E
TENSOR_INDEX_VERSION = 1
TENSOR_INDEX_HEADER = struct.Struct("<8I")
TENSOR_INDEX_RECORD = struct.Struct("<160s4I8Q2Q")
DTYPES = {
    "BOOL": 1,
    "U8": 2,
    "I8": 3,
    "I16": 4,
    "I32": 5,
    "I64": 6,
    "F16": 7,
    "BF16": 8,
    "F32": 9,
    "F64": 10,
}

NPU_DTYPES = {"F16": 4, "BF16": 5, "F32": 6}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        raw_size = source.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"truncated safetensors file: {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        if header_size == 0 or header_size > 100 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header size: {path}")
        header = source.read(header_size)
        if len(header) != header_size:
            raise ValueError(f"truncated safetensors header: {path}")
    value = json.loads(header)
    if not isinstance(value, dict):
        raise ValueError(f"invalid safetensors header: {path}")
    return value


def discover_shards(model_dir: Path) -> tuple[list[Path], int]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index has no weight_map")
        names = sorted({str(value) for value in weight_map.values()})
        return [model_dir / name for name in names], len(weight_map)

    single = model_dir / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(
            "expected model.safetensors or model.safetensors.index.json"
        )
    header = safetensors_header(single)
    return [single], sum(name != "__metadata__" for name in header)


def build_tensor_index(shards: list[Path], output: Path) -> tuple[int, int]:
    records: list[bytes] = []
    scale_dtypes: set[str] = set()
    for shard_index, shard in enumerate(shards):
        header = safetensors_header(shard)
        with shard.open("rb") as source:
            header_size = struct.unpack("<Q", source.read(8))[0]
        payload_base = 8 + header_size
        for name in sorted(key for key in header if key != "__metadata__"):
            tensor = header[name]
            if not isinstance(tensor, dict):
                raise ValueError(f"invalid tensor record {name!r} in {shard}")
            encoded_name = name.encode("utf-8")
            shape = tensor.get("shape")
            offsets = tensor.get("data_offsets")
            dtype = str(tensor.get("dtype", ""))
            if name.endswith(".scales"):
                scale_dtypes.add(dtype)
            if len(encoded_name) >= 160 or not isinstance(shape, list) or len(shape) > 8:
                raise ValueError(f"unsupported tensor name/rank: {name}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
            ):
                raise ValueError(f"invalid tensor offsets: {name}")
            dims = [int(value) for value in shape]
            if any(value < 0 or value > 0xFFFFFFFFFFFFFFFF for value in dims):
                raise ValueError(f"invalid tensor shape: {name}")
            records.append(
                TENSOR_INDEX_RECORD.pack(
                    encoded_name,
                    shard_index,
                    DTYPES.get(dtype, 0),
                    len(dims),
                    0,
                    *(dims + [0] * (8 - len(dims))),
                    payload_base + offsets[0],
                    offsets[1] - offsets[0],
                )
            )
    output.write_bytes(
        TENSOR_INDEX_HEADER.pack(
            TENSOR_INDEX_MAGIC,
            TENSOR_INDEX_VERSION,
            TENSOR_INDEX_HEADER.size,
            TENSOR_INDEX_RECORD.size,
            len(records),
            0,
            0,
            0,
        )
        + b"".join(records)
    )
    if len(scale_dtypes) > 1:
        raise ValueError(f"mixed GPTQ scale dtypes are unsupported: {scale_dtypes}")
    scale_dtype = NPU_DTYPES.get(next(iter(scale_dtypes), ""), 0)
    return len(records), scale_dtype


def config_u32(config: dict[str, Any], key: str, fallback: int = 0) -> int:
    value = config.get(key, fallback)
    if not isinstance(value, int) or value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"invalid config field {key}: {value!r}")
    return value


def text_model_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    if nested is None:
        return config
    if not isinstance(nested, dict):
        raise ValueError("config field text_config must be an object")
    return nested


def quantization_info(config: dict[str, Any]) -> dict[str, Any]:
    quant = config.get("quantization_config", {})
    if not isinstance(quant, dict):
        raise ValueError("config field quantization_config must be an object")
    method = str(quant.get("quant_method", "none"))
    bits = config_u32(quant, "bits")
    group_size = config_u32(quant, "group_size")
    return {
        "quantization_method": method,
        "quantization_bits": bits,
        "quantization_group_size": group_size,
        "quantization_desc_act": int(bool(quant.get("desc_act", False))),
        "quantization_sym": int(bool(quant.get("sym", False))),
        "quantization_zero_bias": 1 if method.lower() == "gptq" else 0,
    }


def build_manifest(
    model_dir: Path, name: str | None, tensor_index_name: str
) -> dict[str, Any]:
    config = read_json(model_dir / "config.json")
    model_config = text_model_config(config)
    shards, declared_tensor_count = discover_shards(model_dir)
    tensor_count, scale_data_type = build_tensor_index(
        shards, model_dir / tensor_index_name
    )
    if tensor_count != declared_tensor_count:
        raise ValueError("safetensors index and shard headers disagree")
    shard_entries = []
    total_size = 0
    for shard in shards:
        if not shard.is_file():
            raise FileNotFoundError(shard)
        size = shard.stat().st_size
        total_size += size
        shard_entries.append({"path": shard.name, "size": size})

    architectures = config.get(
        "architectures", model_config.get("architectures", [])
    )
    architecture = (
        str(architectures[0])
        if isinstance(architectures, list) and architectures
        else str(model_config.get("model_type", config.get("model_type", "unknown")))
    )
    hidden = config_u32(model_config, "hidden_size")
    heads = config_u32(model_config, "num_attention_heads")
    explicit_head_dim = config_u32(model_config, "head_dim")
    if hidden == 0 or heads == 0:
        raise ValueError("hidden_size and num_attention_heads must be non-zero")
    if explicit_head_dim == 0:
        if hidden % heads != 0:
            raise ValueError(
                "head_dim is absent and hidden_size is not divisible by "
                "num_attention_heads"
            )
        explicit_head_dim = hidden // heads
    dtype = str(
        model_config.get(
            "torch_dtype",
            model_config.get(
                "dtype", config.get("torch_dtype", config.get("dtype", "unknown"))
            ),
        )
    )

    manifest = {
        "format": FORMAT,
        "version": 2,
        "name": name or str(config.get("name_or_path", model_dir.name)),
        "architecture": architecture,
        "dtype": dtype,
        "layer_count": config_u32(model_config, "num_hidden_layers"),
        "vocab_size": config_u32(model_config, "vocab_size"),
        "hidden_size": hidden,
        "intermediate_size": config_u32(model_config, "intermediate_size"),
        "head_count": heads,
        "kv_head_count": config_u32(model_config, "num_key_value_heads", heads),
        "head_dim": explicit_head_dim,
        "max_sequence_length": config_u32(
            model_config, "max_position_embeddings", 32768
        ),
        "expert_count": config_u32(model_config, "num_experts"),
        "experts_per_token": config_u32(model_config, "num_experts_per_tok"),
        "moe_intermediate_size": config_u32(
            model_config, "moe_intermediate_size"
        ),
        "shared_expert_intermediate_size": config_u32(
            model_config, "shared_expert_intermediate_size"
        ),
        "weight_format": "safetensors",
        "tensor_index": tensor_index_name,
        "tensor_count": tensor_count,
        "shard_count": len(shard_entries),
        "total_weight_bytes": total_size,
        "required_op_mask": REQUIRED_QWEN_OP_MASK,
        "quantization_scale_data_type": scale_data_type,
        "shards": shard_entries,
    }
    manifest.update(quantization_info(config))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--tensor-index", default="tensor-index.npxi")
    args = parser.parse_args()

    manifest = build_manifest(
        args.model_dir.resolve(), args.name, args.tensor_index
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"model_package={args.output}")
    print(f"model_architecture={manifest['architecture']}")
    print(f"model_tensors={manifest['tensor_count']}")
    print(f"model_shards={manifest['shard_count']}")
    print(f"model_weight_bytes={manifest['total_weight_bytes']}")
    print(f"model_experts={manifest['expert_count']}")
    print(f"model_experts_per_token={manifest['experts_per_token']}")
    print(f"model_quantization={manifest['quantization_method']}")
    print(f"model_quantization_bits={manifest['quantization_bits']}")
    print(
        "model_quantization_scale_data_type="
        f"{manifest['quantization_scale_data_type']}"
    )


if __name__ == "__main__":
    main()
