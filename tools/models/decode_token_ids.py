#!/usr/bin/env python3
"""Decode NPU-produced token IDs with the model's CPU-side tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_token_ids(value: str) -> list[int]:
    tokens: list[int] = []
    for field in value.split(","):
        field = field.strip()
        if not field:
            raise ValueError("empty token ID")
        token = int(field, 10)
        if token < 0 or token > 0xFFFFFFFF:
            raise ValueError(f"token ID outside uint32 range: {token}")
        tokens.append(token)
    if not tokens:
        raise ValueError("no token IDs supplied")
    return tokens


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--token-ids", required=True)
    parser.add_argument(
        "--include-special-tokens",
        action="store_true",
        help="retain special tokens in decoded text",
    )
    args = parser.parse_args()

    token_ids = parse_token_ids(args.token_ids)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    decoded = tokenizer.decode(
        token_ids, skip_special_tokens=not args.include_special_tokens
    )
    print("inference_text_source=cpu-tokenizer")
    print(f"inference_token_text={decoded}")


if __name__ == "__main__":
    main()
