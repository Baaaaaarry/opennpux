#!/usr/bin/env python3
"""Run one real Hugging Face forward pass and emit an OpenNPUX result record."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path


MAGIC = 0x5258504E
VERSION = 1
HEADER = struct.Struct("<IIIIQIIIIIIII64sQ")
EXECUTABLE_MAGIC = 0x4558504E

assert HEADER.size == 128


def fnv1a(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def executable_id(path: Path) -> int:
    data = path.read_bytes()[:88]
    if len(data) != 88 or struct.unpack_from("<I", data)[0] != EXECUTABLE_MAGIC:
        raise ValueError(f"invalid NPU executable: {path}")
    return struct.unpack_from("<Q", data, 48)[0]


def load_model(model_dir: Path, backend_name: str):
    config = json.loads((model_dir / "config.json").read_text())
    quantization = config.get("quantization_config", {})
    quant_method = str(
        quantization.get("quant_method", quantization.get("method", ""))
    ).lower()
    if quant_method == "gptq":
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
        return model, backend.value

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    return model, "transformers-auto"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--gptq-backend",
        default=os.environ.get("OPENNPUX_GPTQ_BACKEND", "gptq_torch"),
    )
    args = parser.parse_args()

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
    model, numerical_backend = load_model(args.model_dir, args.gptq_backend)
    model.eval()
    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_device = next(model.parameters()).device
    encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded, use_cache=False)
    logits = output.logits[0, -1].detach().float().cpu().numpy().astype("<f4")
    token = int(np.argmax(logits))
    token_text = tokenizer.decode([token], skip_special_tokens=False).encode("utf-8")
    if len(token_text) > 56:
        token_text = token_text[:56]
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
        int(logits.shape[0]),
        fnv1a(logits.tobytes()),
        int(encoded["input_ids"].shape[-1]),
        len(token_text),
        model_checksum,
        int(logits.shape[0]),
        token_text.ljust(64, b"\0"),
        0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(record)
    temporary.replace(args.output)
    print(f"hf_numerical_executable_id=0x{executable_id(args.executable):016x}")
    print(f"hf_numerical_backend={numerical_backend}")
    print(f"hf_numerical_prompt_checksum=0x{fnv1a(prompt_bytes):08x}")
    print(f"hf_numerical_input_tokens={encoded['input_ids'].shape[-1]}")
    print(f"hf_numerical_logits={logits.shape[0]}")
    print(f"hf_numerical_logits_checksum=0x{fnv1a(logits.tobytes()):08x}")
    print(f"hf_numerical_next_token={token}")
    print(f"hf_numerical_token_text={tokenizer.decode([token], skip_special_tokens=False)!r}")
    print("hf_numerical=PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"hf_numerical=FAIL: {error}", file=sys.stderr)
        raise
