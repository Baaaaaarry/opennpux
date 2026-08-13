#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


EXPECTED_FORMAT = "OPENNPUX_QWEN_TINY_V1"
REQUIRED_OPS = {
    "ADD",
    "MATMUL",
    "MUL",
    "RMS_NORM",
    "ROPE",
    "SILU",
    "SOFTMAX",
    "TOPK",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    model_path = Path(args.model)
    package = json.loads(model_path.read_text())
    if package.get("format") != EXPECTED_FORMAT:
        raise SystemExit(f"unsupported qwen package format: {package.get('format')}")
    if package.get("version") != 1:
        raise SystemExit(f"unsupported qwen package version: {package.get('version')}")

    model = package["model"]
    golden = package["golden"]
    trace = package["operator_trace"]
    present_ops = {entry["op"] for entry in trace}
    missing = sorted(REQUIRED_OPS - present_ops)
    if missing:
        raise SystemExit(f"missing required qwen ops: {','.join(missing)}")

    logits = golden["logits"]
    next_token = golden["next_token"]
    if next_token < 0 or next_token >= model["vocab_size"]:
        raise SystemExit("golden next token is outside vocab")
    if len(logits) != model["vocab_size"]:
        raise SystemExit("golden logits size does not match vocab")

    print(f"qwen_model={model['name']}")
    print(f"qwen_format={package['format']}")
    print(f"qwen_layers={model['layer_count']}")
    print(f"qwen_hidden={model['hidden_size']}")
    print(f"qwen_heads={model['head_count']}")
    print(f"qwen_vocab={model['vocab_size']}")
    print(f"qwen_operator_count={len(trace)}")
    print(f"qwen_ops={','.join(sorted(present_ops))}")
    print(f"qwen_next_token={next_token}")
    print(f"qwen_logits_checksum={golden['logits_checksum']}")
    print("qwen_inspect=PASS")


if __name__ == "__main__":
    main()
