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


def load_model(
    model_dir: Path,
    loader_name: str,
    backend_name: str,
    device_map_name: str,
):
    config = json.loads((model_dir / "config.json").read_text())
    quantization = config.get("quantization_config", {})
    quant_method = str(
        quantization.get("quant_method", quantization.get("method", ""))
    ).lower()
    device_map = (
        device_map_name if device_map_name == "auto" else {"": device_map_name}
    )
    if loader_name == "transformers":
        from transformers import AutoModelForMultimodalLM, GPTQConfig

        model_arguments = {}
        if quant_method == "gptq":
            # Do not let Transformers silently select Marlin. Qwen3.5 has
            # narrow projections that violate Marlin's output-width rules.
            quantization_arguments = dict(quantization)
            quantization_arguments.pop("quant_method", None)
            quantization_arguments.pop("method", None)
            quantization_arguments["backend"] = backend_name
            model_arguments["quantization_config"] = GPTQConfig(
                **quantization_arguments
            )

        model = AutoModelForMultimodalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            device_map=device_map,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            **model_arguments,
        )
        backend = backend_name if quant_method == "gptq" else "native"
        return model, f"transformers-multimodal:{backend}", False

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
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = getattr(loaded, "model", loaded)
        return model, backend.value, cse_compat

    raise ValueError("gptqmodel loader requires a GPTQ model package")


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


def prepare_inputs(model_dir: Path, prompt: str, prompt_format: str):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    tokenizer = processor.tokenizer
    if prompt_format == "raw":
        formatted_prompt = prompt
        encoded = processor(text=prompt, return_tensors="pt")
    else:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        formatted_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    return processor, tokenizer, formatted_prompt, encoded


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
        "--model-loader",
        choices=("transformers", "gptqmodel"),
        default=os.environ.get("OPENNPUX_MODEL_LOADER", "transformers"),
        help="use the official Transformers multimodal loader or GPTQModel",
    )
    parser.add_argument(
        "--device-map",
        choices=("auto", "cuda", "cpu"),
        default=os.environ.get("OPENNPUX_HF_DEVICE_MAP", "auto"),
        help="place the host numerical reference model automatically or on one device",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("OPENNPUX_MAX_NEW_TOKENS", "8")),
    )
    parser.add_argument(
        "--decode-mode",
        choices=("model", "greedy"),
        default=os.environ.get("OPENNPUX_DECODE_MODE", "model"),
        help="use the model generation_config or deterministic greedy decoding",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("OPENNPUX_GENERATION_SEED", "42")),
    )
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 32:
        parser.error("--max-new-tokens must be between 1 and 32")

    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise SystemExit(
            "real-token generation requires torch, transformers, accelerate "
            f"and numpy: {error}"
        )

    model, numerical_backend, cse_compat = load_model(
        args.model_dir,
        args.model_loader,
        args.gptq_backend,
        args.device_map,
    )
    model.eval()
    processor, tokenizer, formatted_prompt, encoded = prepare_inputs(
        args.model_dir, args.prompt, args.prompt_format
    )
    del processor
    input_device = next(model.parameters()).device
    encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}
    generation_arguments = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    generation_config_path = args.model_dir / "generation_config.json"
    generation_policy = {}
    if args.decode_mode == "model":
        if not generation_config_path.is_file():
            raise ValueError(
                "model decode mode requires generation_config.json"
            )
        generation_policy = json.loads(generation_config_path.read_text())
        for name in (
            "do_sample",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "eos_token_id",
            "pad_token_id",
            "bos_token_id",
        ):
            if name in generation_policy:
                generation_arguments[name] = generation_policy[name]
    else:
        generation_arguments["do_sample"] = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with torch.inference_mode():
        output = model.generate(**encoded, **generation_arguments)
    input_tokens = int(encoded["input_ids"].shape[-1])
    generated_ids = output.sequences[0, input_tokens:].detach().cpu().tolist()
    if not generated_ids or not output.scores:
        raise RuntimeError("model did not generate a token or return logits")
    if (args.decode_mode == "model" and len(generated_ids) > 1 and
            len(set(generated_ids)) == 1 and
            os.environ.get("OPENNPUX_ALLOW_DEGENERATE_OUTPUT") is None):
        raise RuntimeError(
            "model sampling generated one repeated token; refusing a "
            "degenerate numerical golden"
        )
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
    print(f"hf_numerical_model_loader={args.model_loader}")
    print(f"hf_numerical_quant_backend={args.gptq_backend}")
    print(f"hf_numerical_device_map={args.device_map}")
    print(f"hf_numerical_torch_cse_compat={int(cse_compat)}")
    print(f"hf_numerical_prompt_format={args.prompt_format}")
    print(f"hf_numerical_decode_mode={args.decode_mode}")
    print(f"hf_numerical_generation_seed={args.seed}")
    print(
        "hf_numerical_generation_policy="
        + json.dumps(generation_policy, sort_keys=True, separators=(",", ":"))
    )
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
    print(f"hf_numerical_unique_tokens={len(set(generated_ids))}")
    print("hf_numerical_token_ids=" + ",".join(str(value) for value in generated_ids))
    print(f"hf_numerical_stop_reason={stop_reason}")
    print(f"hf_numerical_token_text={decoded_text!r}")
    print("hf_numerical=PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"hf_numerical=FAIL: {error}", file=sys.stderr)
        if "illegal memory access" in str(error).lower():
            print(
                "hf_numerical_hint=CUDA kernel failed; rerun with "
                "CORAL_QWEN_HF_DEVICE=cpu and "
                "CORAL_QWEN_GPTQ_BACKEND=gptq_torch_aten, or diagnose with "
                "CUDA_LAUNCH_BLOCKING=1",
                file=sys.stderr,
            )
        raise
