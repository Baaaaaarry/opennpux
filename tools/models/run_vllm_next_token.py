#!/usr/bin/env python3
"""Generate an OpenNPUX numerical reference with the vLLM offline engine."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

from run_hf_next_token import (
    HEADER,
    MAGIC,
    MAX_RESULT_TOKENS,
    VERSION,
    executable_id,
    fnv1a,
)


def vocabulary_size(config: dict) -> int:
    for candidate in (
        config,
        config.get("text_config", {}),
        config.get("language_config", {}),
    ):
        value = candidate.get("vocab_size")
        if value:
            return int(value)
    raise ValueError("model config does not define a vocabulary size")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-format", choices=("chat", "raw"), default="chat")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--decode-mode", choices=("model", "greedy"), default="model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= MAX_RESULT_TOKENS:
        parser.error(f"--max-new-tokens must be between 1 and {MAX_RESULT_TOKENS}")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    tokenizer = processor.tokenizer
    if args.prompt_format == "chat":
        messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
        formatted_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = args.prompt

    generation_path = args.model_dir / "generation_config.json"
    generation_policy = json.loads(generation_path.read_text()) if generation_path.is_file() else {}
    sampling_arguments = {
        "max_tokens": args.max_new_tokens,
        "seed": args.seed,
        "temperature": 0.0,
    }
    if args.decode_mode == "model":
        for name in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "repetition_penalty",
        ):
            if name in generation_policy:
                sampling_arguments[name] = generation_policy[name]
    sampling = SamplingParams(**sampling_arguments)
    quantization = os.environ.get("OPENNPUX_VLLM_QUANTIZATION", "moe_wna16")
    llm = LLM(
        model=str(args.model_dir),
        trust_remote_code=True,
        quantization=quantization,
        language_model_only=True,
        max_model_len=int(os.environ.get("OPENNPUX_VLLM_MAX_MODEL_LEN", "4096")),
        gpu_memory_utilization=float(
            os.environ.get("OPENNPUX_VLLM_GPU_MEMORY_UTILIZATION", "0.9")
        ),
    )
    request = llm.generate([formatted_prompt], sampling, use_tqdm=True)[0]
    completion = request.outputs[0]
    generated_ids = [int(value) for value in completion.token_ids]
    if not generated_ids:
        raise RuntimeError("vLLM did not generate any tokens")
    input_tokens = len(request.prompt_token_ids or ())
    if input_tokens == 0:
        input_tokens = len(tokenizer.encode(formatted_prompt))
    decoded_text = completion.text
    token_text = decoded_text.encode("utf-8")[:64]
    token_text = token_text.decode("utf-8", errors="ignore").encode("utf-8")
    config_path = args.model_dir / "config.json"
    config = json.loads(config_path.read_text())
    vocab_size = vocabulary_size(config)
    for token in generated_ids:
        if not 0 <= token < vocab_size:
            raise RuntimeError(f"vLLM returned token {token} outside the vocabulary")
    token_checksum = fnv1a(struct.pack(f"<{len(generated_ids)}I", *generated_ids))
    stop_reason = int(completion.finish_reason == "stop")
    prompt_bytes = args.prompt.encode("utf-8")
    record = HEADER.pack(
        MAGIC,
        VERSION,
        HEADER.size,
        1,
        executable_id(args.executable),
        fnv1a(prompt_bytes),
        generated_ids[0],
        vocab_size,
        token_checksum,
        input_tokens,
        len(token_text),
        fnv1a(config_path.read_bytes()),
        vocab_size,
        token_text.ljust(64, b"\0"),
        len(generated_ids),
        stop_reason,
        *(generated_ids + [0] * (MAX_RESULT_TOKENS - len(generated_ids))),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(record)
    temporary.replace(args.output)
    print("hf_numerical_backend=vllm:" + quantization)
    print("hf_numerical_model_loader=vllm")
    print(f"hf_numerical_prompt_checksum=0x{fnv1a(prompt_bytes):08x}")
    print(f"hf_numerical_input_tokens={input_tokens}")
    print(f"hf_numerical_generated_tokens={len(generated_ids)}")
    print("hf_numerical_token_ids=" + ",".join(str(value) for value in generated_ids))
    print(f"hf_numerical_token_text={decoded_text!r}")
    print("hf_numerical_reference=vllm-offline")
    print("hf_numerical=PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"hf_numerical=FAIL: {error}", file=sys.stderr)
        raise
