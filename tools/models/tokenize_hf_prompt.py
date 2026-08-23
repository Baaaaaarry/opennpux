#!/usr/bin/env python3
"""Tokenize a CPU-side prompt for the generic NPU inference ABI."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

def normalize_token_ids(encoded: Any) -> list[int]:
    """Convert tokenizer/processor output into one flat token-ID sequence."""
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("tokenizer result does not contain input_ids")
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids

    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if (
        isinstance(encoded, Sequence)
        and not isinstance(encoded, (str, bytes))
        and len(encoded) == 1
        and isinstance(encoded[0], Sequence)
        and not isinstance(encoded[0], (str, bytes))
    ):
        encoded = encoded[0]
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
        raise ValueError("tokenizer input_ids is not a sequence")

    token_ids: list[int] = []
    for token in encoded:
        if hasattr(token, "item"):
            token = token.item()
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError(f"invalid token ID type: {type(token).__name__}")
        if token < 0 or token > 0xFFFFFFFF:
            raise ValueError(f"token ID outside uint32 range: {token}")
        token_ids.append(token)
    if not token_ids:
        raise ValueError("tokenizer produced no input tokens")
    return token_ids


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="raw")
    parser.add_argument("--thinking-mode", choices=("off", "on"), default="off")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    if args.prompt_format == "chat":
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=args.thinking_mode == "on",
        )
    else:
        encoded = tokenizer.encode(args.prompt, add_special_tokens=True)
    token_ids = normalize_token_ids(encoded)
    print(f"input_token_count={len(token_ids)}")
    print("input_token_ids=" + ",".join(str(token) for token in token_ids))


if __name__ == "__main__":
    main()
