#!/usr/bin/env python3
"""Summarize a Qwen model tensor index into a layer execution plan."""

import argparse
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HEADER = struct.Struct("<8I")
RECORD = struct.Struct("<160s4I8Q2Q")
MAGIC = 0x5458504E
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
EXPERT_RE = re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def tensor_names(index_path: Path, expected_count: int) -> list[str]:
    data = index_path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("truncated tensor index")
    fields = HEADER.unpack_from(data)
    magic, version, header_size, record_size, count = fields[:5]
    if (
        magic != MAGIC
        or version != 1
        or header_size != HEADER.size
        or record_size != RECORD.size
        or count != expected_count
        or len(data) != header_size + count * record_size
    ):
        raise ValueError("invalid tensor index ABI")
    names = []
    for index in range(count):
        offset = header_size + index * record_size
        raw_name = RECORD.unpack_from(data, offset)[0]
        names.append(raw_name.split(b"\0", 1)[0].decode("utf-8"))
    return names


def tensor_role(name: str) -> str:
    if "embed_tokens" in name or "token_embedding" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "input_layernorm" in name:
        return "attention_norm"
    if "post_attention_layernorm" in name:
        return "ffn_norm"
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if projection in name:
            return f"attention_{projection}"
    if "shared_expert" in name:
        return "shared_expert"
    if EXPERT_RE.search(name):
        return "routed_expert"
    if "router" in name or ".gate." in name:
        return "router"
    if "gate_proj" in name or "up_proj" in name or "down_proj" in name:
        return "dense_ffn"
    if "vision" in name or "visual" in name:
        return "vision"
    return "other"


def tensor_component(name: str) -> str:
    suffix = name.rsplit(".", 1)[-1]
    if suffix in {"qweight", "qzeros", "scales", "g_idx", "bias", "weight"}:
        return suffix
    return "other"


def build_plan(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    index_path = manifest_path.parent / str(manifest["tensor_index"])
    names = tensor_names(index_path, int(manifest["tensor_count"]))
    roles: Counter[str] = Counter()
    components: Counter[str] = Counter()
    layer_roles: dict[int, Counter[str]] = defaultdict(Counter)
    experts: set[int] = set()
    prefixes: Counter[str] = Counter()

    for name in names:
        role = tensor_role(name)
        roles[role] += 1
        components[tensor_component(name)] += 1
        prefixes[name.split(".", 1)[0]] += 1
        layer_match = LAYER_RE.search(name)
        if layer_match:
            layer = int(layer_match.group(1))
            if layer >= int(manifest["layer_count"]):
                raise ValueError(f"tensor layer {layer} exceeds model layer count")
            layer_roles[layer][role] += 1
        expert_match = EXPERT_RE.search(name)
        if expert_match:
            expert = int(expert_match.group(1))
            if expert >= int(manifest["expert_count"]):
                raise ValueError(f"tensor expert {expert} exceeds expert count")
            experts.add(expert)

    observed_layers = sorted(layer_roles)
    plan = {
        "format": "OPENNPUX_QWEN_EXECUTION_PLAN_V1",
        "model_manifest": manifest_path.name,
        "architecture": manifest["architecture"],
        "layer_count": manifest["layer_count"],
        "observed_layer_count": len(observed_layers),
        "first_observed_layer": observed_layers[0] if observed_layers else None,
        "last_observed_layer": observed_layers[-1] if observed_layers else None,
        "expert_count": manifest["expert_count"],
        "experts_per_token": manifest["experts_per_token"],
        "observed_expert_count": len(experts),
        "quantization": {
            "method": manifest["quantization_method"],
            "bits": manifest["quantization_bits"],
            "group_size": manifest["quantization_group_size"],
        },
        "tensor_count": len(names),
        "tensor_roles": dict(sorted(roles.items())),
        "tensor_components": dict(sorted(components.items())),
        "top_level_prefixes": dict(prefixes.most_common(16)),
        "layer_zero_template": dict(sorted(layer_roles.get(0, {}).items())),
        "scheduler": {
            "weight_policy": "paged-range-read",
            "expert_policy": "router-topk-active-only",
            "active_experts_per_token": manifest["experts_per_token"],
            "kv_cache_policy": "paged-per-layer-kv-head",
            "tcb_granularity": "decoder-layer-phase",
        },
    }
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    output = args.output or manifest.with_name("execution-plan.npxp")
    plan = build_plan(manifest)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"qwen_execution_plan={output}")
    print(f"qwen_plan_layers={plan['observed_layer_count']}/{plan['layer_count']}")
    print(
        f"qwen_plan_experts={plan['observed_expert_count']}/"
        f"{plan['expert_count']}"
    )
    print(f"qwen_plan_tensor_roles={','.join(plan['tensor_roles'])}")
    print("qwen_execution_plan=PASS")


if __name__ == "__main__":
    main()
