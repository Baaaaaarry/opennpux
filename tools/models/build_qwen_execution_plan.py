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
NUMBER_RE = re.compile(r"(?<=\.)\d+(?=\.)")


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
    lowered = name.lower()
    if "embed_tokens" in name or "token_embedding" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "rotary_emb" in lowered:
        return "rotary_embedding"
    if ".q_norm." in lowered or lowered.endswith(".q_norm.weight"):
        return "attention_q_norm"
    if ".k_norm." in lowered or lowered.endswith(".k_norm.weight"):
        return "attention_k_norm"
    if "input_layernorm" in name:
        return "attention_norm"
    if "post_attention_layernorm" in name:
        return "ffn_norm"
    if ("language_model.norm." in lowered or "model.norm." in lowered or
            lowered.endswith(".final_layernorm.weight")):
        return "final_norm"
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if projection in name:
            return f"attention_{projection}"
    linear_attention_roles = {
        "in_proj_qkv": "linear_attention_qkv",
        "in_proj_z": "linear_attention_gate",
        "in_proj_b": "linear_attention_beta",
        "in_proj_a": "linear_attention_alpha",
        "conv1d": "linear_attention_conv",
        "dt_bias": "linear_attention_decay",
        "a_log": "linear_attention_decay",
        "out_proj": "linear_attention_output",
    }
    if "linear_attn" in lowered or "linear_attention" in lowered:
        for token, role in linear_attention_roles.items():
            if token in lowered:
                return role
        if ".norm." in lowered or lowered.endswith(".norm.weight"):
            return "linear_attention_norm"
        return "linear_attention_other"
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


def tensor_domain(name: str) -> str:
    if name.startswith("mtp.") or ".mtp." in name:
        return "mtp"
    if "vision" in name or "visual" in name:
        return "vision"
    return "text"


def normalized_pattern(name: str) -> str:
    return NUMBER_RE.sub("{index}", name)


def layer_type(roles: Counter[str]) -> str:
    if roles["attention_q_proj"]:
        return "full_attention_moe"
    if any(role.startswith("linear_attention_") for role in roles):
        return "linear_attention_moe"
    return "unclassified_moe"


def layer_phases(kind: str) -> list[str]:
    common_tail = [
        "residual_add",
        "ffn_norm",
        "router_topk",
        "routed_experts_active_only",
        "shared_expert",
        "moe_combine",
        "residual_add",
    ]
    if kind == "full_attention_moe":
        return [
            "attention_norm",
            "qkv_projection",
            "rope",
            "paged_kv_cache_update",
            "scaled_dot_product_attention",
            "attention_output_projection",
        ] + common_tail
    if kind == "linear_attention_moe":
        return [
            "attention_norm",
            "linear_attention_projection",
            "causal_depthwise_conv",
            "recurrent_state_update",
            "linear_attention_gate_norm",
            "linear_attention_output_projection",
        ] + common_tail
    return ["unclassified_attention"] + common_tail


def tensor_component(name: str) -> str:
    suffix = name.rsplit(".", 1)[-1]
    if suffix.lower() in {"a_log", "dt_bias"}:
        return "weight"
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
    domains: Counter[str] = Counter()
    domain_roles: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_patterns: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_samples: dict[str, list[str]] = defaultdict(list)
    unknown_decoder_patterns: Counter[str] = Counter()
    unknown_decoder_samples: list[str] = []

    for name in names:
        role = tensor_role(name)
        roles[role] += 1
        domain = tensor_domain(name)
        domains[domain] += 1
        domain_roles[domain][role] += 1
        components[tensor_component(name)] += 1
        prefixes[name.split(".", 1)[0]] += 1
        layer_match = LAYER_RE.search(name)
        if layer_match and domain == "text":
            layer = int(layer_match.group(1))
            if layer >= int(manifest["layer_count"]):
                raise ValueError(f"tensor layer {layer} exceeds model layer count")
            layer_roles[layer][role] += 1
        if role in {"other", "linear_attention_other"}:
            unknown_patterns[domain][normalized_pattern(name)] += 1
            if len(unknown_samples[domain]) < 16:
                unknown_samples[domain].append(name)
            if layer_match and domain == "text":
                unknown_decoder_patterns[normalized_pattern(name)] += 1
                if len(unknown_decoder_samples) < 32:
                    unknown_decoder_samples.append(name)
        expert_match = EXPERT_RE.search(name)
        if expert_match and domain == "text":
            expert = int(expert_match.group(1))
            if expert >= int(manifest["expert_count"]):
                raise ValueError(f"tensor expert {expert} exceeds expert count")
            experts.add(expert)

    observed_layers = sorted(layer_roles)
    layers = []
    layer_type_counts: Counter[str] = Counter()
    for layer in observed_layers:
        kind = layer_type(layer_roles[layer])
        layer_type_counts[kind] += 1
        layers.append(
            {
                "index": layer,
                "type": kind,
                "tensor_roles": dict(sorted(layer_roles[layer].items())),
                "phases": layer_phases(kind),
            }
        )
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
        "tensor_domains": dict(sorted(domains.items())),
        "domain_tensor_roles": {
            domain: dict(sorted(domain_roles[domain].items()))
            for domain in sorted(domain_roles)
        },
        "top_level_prefixes": dict(prefixes.most_common(16)),
        "layer_zero_template": dict(sorted(layer_roles.get(0, {}).items())),
        "layer_type_counts": dict(sorted(layer_type_counts.items())),
        "layers": layers,
        "unknown_tensor_patterns": {
            domain: dict(unknown_patterns[domain].most_common(32))
            for domain in sorted(unknown_patterns)
        },
        "unknown_tensor_samples": {
            domain: unknown_samples[domain] for domain in sorted(unknown_samples)
        },
        "unknown_decoder_tensor_patterns": dict(
            unknown_decoder_patterns.most_common(64)
        ),
        "unknown_decoder_tensor_samples": unknown_decoder_samples,
        "scheduler": {
            "weight_policy": "paged-range-read",
            "expert_policy": "router-topk-active-only",
            "active_experts_per_token": manifest["experts_per_token"],
            "full_attention_state": "paged-per-layer-kv-head",
            "linear_attention_state": "paged-recurrent-state",
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
    print(
        "qwen_plan_layer_types="
        + ",".join(
            f"{name}:{count}" for name, count in plan["layer_type_counts"].items()
        )
    )
    print(
        "qwen_plan_domains="
        + ",".join(
            f"{name}:{count}" for name, count in plan["tensor_domains"].items()
        )
    )
    print(
        "qwen_plan_unknown_patterns="
        + ",".join(
            f"{domain}:{len(patterns)}"
            for domain, patterns in plan["unknown_tensor_patterns"].items()
        )
    )
    print(
        "qwen_plan_unknown_decoder_patterns="
        f"{len(plan['unknown_decoder_tensor_patterns'])}"
    )
    print("qwen_execution_plan=PASS")


if __name__ == "__main__":
    main()
