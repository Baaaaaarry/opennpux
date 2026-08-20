#!/usr/bin/env python3
"""Tokenize a CPU-side prompt for the generic NPU inference ABI."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="raw")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    if args.prompt_format == "chat":
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        token_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    if not token_ids:
        raise ValueError("tokenizer produced no input tokens")
    print(f"input_token_count={len(token_ids)}")
    print("input_token_ids=" + ",".join(str(token) for token in token_ids))


if __name__ == "__main__":
    main()
