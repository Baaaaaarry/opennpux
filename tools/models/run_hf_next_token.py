#!/usr/bin/env python3
"""Run Hugging Face autoregressive decode and emit an OpenNPUX result record."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any


MAGIC = 0x5258504E
VERSION = 2
MAX_RESULT_TOKENS = 32
HEADER = struct.Struct("<IIIIQIIIIIIII64sII32I")
EXECUTABLE_MAGIC = 0x4558504E

assert HEADER.size == 256


def fnv1a(data: bytes, value: int = 2166136261) -> int:
    for byte in data:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def executable_id(path: Path) -> int:
    data = path.read_bytes()[:88]
    if len(data) != 88 or struct.unpack_from("<I", data)[0] != EXECUTABLE_MAGIC:
        raise ValueError(f"invalid NPU executable: {path}")
    return struct.unpack_from("<Q", data, 48)[0]


def patch_torch_cse_generic() -> bool:
    """Work around the PyTorch 2.10 CSE one-vs-two generic parameter bug."""
    from torch._inductor.codegen.common import CSE

    try:
        CSE[Any]
        return False
    except TypeError as error:
        if "Too few arguments" not in str(error):
            raise

    original = CSE.__class_getitem__

    def compatible_class_getitem(cls, parameters):
        del cls
        if not isinstance(parameters, tuple):
            parameters = (parameters, Any)
        return original(parameters)

    CSE.__class_getitem__ = classmethod(compatible_class_getitem)
    CSE[Any]
    return True


def load_model(model_dir: Path, backend_name: str):
    config = json.loads((model_dir / "config.json").read_text())
    quantization = config.get("quantization_config", {})
    quant_method = str(
        quantization.get("quant_method", quantization.get("method", ""))
    ).lower()
    if quant_method == "gptq":
        cse_compat = patch_torch_cse_generic()
        from gptqmodel import BACKEND, GPTQModel

        try:
            backend = BACKEND(backend_name)
        except ValueError as error:
            choices = ", ".join(item.value for item in BACKEND)
            raise ValueError(
                f"unknown GPTQ backend {backend_name!r}; choose one of: {choices}"
            ) from error
        loaded = GPTQModel.load(
            str(model_dir),
            backend=backend,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        model = getattr(loaded, "model", loaded)
        return model, backend.value, cse_compat

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    return model, "transformers-auto", False


def format_prompt(tokenizer, prompt: str, prompt_format: str) -> str:
    if prompt_format == "raw":
        return prompt
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "chat prompt format requested, but the tokenizer has no chat template"
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--prompt-format",
        choices=("chat", "raw"),
        default=os.environ.get("OPENNPUX_PROMPT_FORMAT", "chat"),
        help="format the CPU prompt with the model chat template or use it verbatim",
    )
    parser.add_argument(
        "--gptq-backend",
        default=os.environ.get("OPENNPUX_GPTQ_BACKEND", "gptq_torch"),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("OPENNPUX_MAX_NEW_TOKENS", "8")),
    )
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 32:
        parser.error("--max-new-tokens must be between 1 and 32")

    try:
        import numpy as np
        import torch
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "real-token generation requires torch, transformers, accelerate "
            f"and numpy: {error}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    formatted_prompt = format_prompt(tokenizer, args.prompt, args.prompt_format)
    model, numerical_backend, cse_compat = load_model(
        args.model_dir, args.gptq_backend
    )
    model.eval()
    encoded = tokenizer(formatted_prompt, return_tensors="pt")
    input_device = next(model.parameters()).device
    encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    input_tokens = int(encoded["input_ids"].shape[-1])
    generated_ids = output.sequences[0, input_tokens:].detach().cpu().tolist()
    if not generated_ids or not output.scores:
        raise RuntimeError("model did not generate a token or return logits")
    token = int(generated_ids[0])
    vocabulary_size = int(output.scores[0].shape[-1])
    logits_checksum = 2166136261
    for score in output.scores:
        logits = score[0].detach().float().cpu().numpy().astype("<f4")
        if int(logits.shape[0]) != vocabulary_size:
            raise RuntimeError("generation score vocabulary size changed")
        logits_checksum = fnv1a(logits.tobytes(), logits_checksum)
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if not decoded_text:
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    token_text = decoded_text.encode("utf-8")[:64]
    token_text = token_text.decode("utf-8", errors="ignore").encode("utf-8")
    generation_config = getattr(model, "generation_config", None)
    eos_token_ids = getattr(generation_config, "eos_token_id", None)
    if eos_token_ids is None:
        eos_token_ids = tokenizer.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    stop_reason = int(bool(eos_token_ids) and generated_ids[-1] in eos_token_ids)
    prompt_bytes = args.prompt.encode("utf-8")
    model_checksum = fnv1a((args.model_dir / "config.json").read_bytes())
    record = HEADER.pack(
        MAGIC,
        VERSION,
        HEADER.size,
        1,
        executable_id(args.executable),
        fnv1a(prompt_bytes),
        token,
        vocabulary_size,
        logits_checksum,
        input_tokens,
        len(token_text),
        model_checksum,
        vocabulary_size,
        token_text.ljust(64, b"\0"),
        len(generated_ids),
        stop_reason,
        *(generated_ids + [0] * (MAX_RESULT_TOKENS - len(generated_ids))),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(record)
    temporary.replace(args.output)
    print(f"hf_numerical_executable_id=0x{executable_id(args.executable):016x}")
    print(f"hf_numerical_backend={numerical_backend}")
    print(f"hf_numerical_torch_cse_compat={int(cse_compat)}")
    print(f"hf_numerical_prompt_format={args.prompt_format}")
    print(f"hf_numerical_prompt_checksum=0x{fnv1a(prompt_bytes):08x}")
    print(
        "hf_numerical_formatted_prompt_checksum="
        f"0x{fnv1a(formatted_prompt.encode('utf-8')):08x}"
    )
    print(f"hf_numerical_input_tokens={input_tokens}")
    print(f"hf_numerical_logits={vocabulary_size}")
    print(f"hf_numerical_logits_checksum=0x{logits_checksum:08x}")
    print(f"hf_numerical_next_token={token}")
    print(f"hf_numerical_generated_tokens={len(generated_ids)}")
    print("hf_numerical_token_ids=" + ",".join(str(value) for value in generated_ids))
    print(f"hf_numerical_stop_reason={stop_reason}")
    print(f"hf_numerical_token_text={decoded_text!r}")
    print("hf_numerical=PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"hf_numerical=FAIL: {error}", file=sys.stderr)
        raise
