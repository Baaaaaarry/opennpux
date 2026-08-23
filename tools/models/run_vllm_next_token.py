#!/usr/bin/env python3
"""Generate an OpenNPUX numerical reference with the vLLM offline engine."""

from __future__ import annotations

import argparse
import functools
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


def _install_layer_trace_hooks(model, *, output_path: str, trace_step: int) -> dict:
    """Install worker-local hooks without returning tensors through RPC."""
    import torch

    candidates = (
        ("language_model", "model", "layers"),
        ("model", "layers"),
        ("language_model", "language_model", "model", "layers"),
    )
    layers = None
    decoder_model = None
    selected_path = ""
    for path in candidates:
        value = model
        for component in path:
            value = getattr(value, component, None)
            if value is None:
                break
        if value is not None:
            layers = value
            decoder_model = model
            for component in path[:-1]:
                decoder_model = getattr(decoder_model, component)
            selected_path = ".".join(path)
            break
    if layers is None or decoder_model is None:
        raise RuntimeError("unable to locate decoder layers in vLLM model")

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    counters: dict[tuple[int, str], int] = {}

    def record_tensor(
        layer_index: int, point: str, tensor, *, width: int | None = None
    ) -> None:
        key = (layer_index, point)
        invocation = counters.get(key, 0)
        counters[key] = invocation + 1
        if invocation != trace_step or not isinstance(tensor, torch.Tensor):
            return
        if width is None:
            width = int(tensor.shape[-1])
        if width <= 0 or tensor.numel() % width != 0:
            return
        vector = tensor.detach().reshape(-1, width)[-1]
        values = vector.float().cpu().tolist()
        finite = [
            value
            for value in values
            if float("-inf") < value < float("inf")
        ]
        record = {
            "source": "vllm",
            "rank": rank,
            "step": invocation,
            "layer": layer_index,
            "point": point,
            "count": len(values),
            "minimum": min(finite) if finite else 0.0,
            "maximum": max(finite) if finite else 0.0,
            "values": values,
        }
        with open(output_path, "a", encoding="utf-8") as trace:
            trace.write(json.dumps(record, separators=(",", ":")) + "\n")

    def make_hook(layer_index: int):
        def trace_layer(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            residual = (
                output[1]
                if isinstance(output, tuple) and len(output) > 1
                else None
            )
            if not isinstance(hidden, torch.Tensor):
                return
            # vLLM carries the residual separately between decoder layers.
            # The next layer consumes their sum, which is the canonical layer
            # boundary corresponding to OpenNPUX's second residual_add.
            boundary = (
                hidden
                if not isinstance(residual, torch.Tensor)
                else hidden + residual
            )
            record_tensor(layer_index, "layer_boundary", boundary)

        return trace_layer

    def make_tensor_hook(
        layer_index: int,
        point: str,
        output_index: int = 0,
        *,
        width: int | None = None,
    ):
        def trace_tensor(_module, _inputs, output):
            tensor = output
            if isinstance(output, tuple):
                if output_index >= len(output):
                    return
                tensor = output[output_index]
            record_tensor(layer_index, point, tensor, width=width)

        return trace_tensor

    def make_input_hook(
        layer_index: int, point: str, *, width: int | None = None
    ):
        def trace_input(_module, inputs):
            if inputs:
                record_tensor(layer_index, point, inputs[0], width=width)

        return trace_input

    def make_tail_hook(
        layer_index: int, point: str, *, width: int
    ):
        def trace_tail(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor) or tensor.shape[-1] < width:
                return
            record_tensor(layer_index, point, tensor[..., -width:], width=width)

        return trace_tail

    registered_points: set[str] = set()

    def register_output(
        module, layer_index: int, point: str, output_index: int = 0,
        *, width: int | None = None,
    ) -> None:
        if module is None:
            return
        module.register_forward_hook(
            make_tensor_hook(
                layer_index, point, output_index, width=width
            )
        )
        registered_points.add(point)

    def register_input(
        module, layer_index: int, point: str, *, width: int | None = None
    ) -> None:
        if module is None:
            return
        module.register_forward_pre_hook(
            make_input_hook(layer_index, point, width=width)
        )
        registered_points.add(point)

    decoder_model.embed_tokens.register_forward_hook(
        make_tensor_hook(-1, "embedding")
    )
    for layer_index, layer in enumerate(layers):
        layer.input_layernorm.register_forward_hook(
            make_tensor_hook(layer_index, "attention_norm")
        )
        mixer = getattr(layer, "linear_attn", None)
        if mixer is None:
            mixer = getattr(layer, "self_attn", None)
        if mixer is None:
            raise RuntimeError(f"decoder layer {layer_index} has no token mixer")
        mixer.register_forward_hook(make_tensor_hook(layer_index, "token_mixer"))
        if getattr(layer, "self_attn", None) is mixer:
            query_width = int(getattr(mixer, "q_size", 0)) or None
            key_width = int(getattr(mixer, "kv_size", 0)) or None
            register_output(
                getattr(mixer, "q_norm", None), layer_index,
                "qkv_query", width=query_width,
            )
            register_output(
                getattr(mixer, "k_norm", None), layer_index,
                "qkv_key", width=key_width,
            )
            register_output(
                getattr(mixer, "rotary_emb", None), layer_index,
                "rope_query", 0, width=query_width,
            )
            register_output(
                getattr(mixer, "rotary_emb", None), layer_index,
                "rope_key", 1, width=key_width,
            )
            register_input(
                getattr(mixer, "o_proj", None), layer_index,
                "attention_core", width=query_width,
            )
            register_output(
                getattr(mixer, "o_proj", None), layer_index,
                "attention_output_projection",
            )
        else:
            value_width = int(getattr(mixer, "value_dim", 0)) or None
            if value_width is not None and getattr(mixer, "in_proj_qkvz", None):
                mixer.in_proj_qkvz.register_forward_hook(
                    make_tail_hook(
                        layer_index, "linear_attention_gate_projection",
                        width=value_width,
                    )
                )
                registered_points.add("linear_attention_gate_projection")
            register_output(
                getattr(mixer, "chunk_gated_delta_rule", None), layer_index,
                "recurrent_state_update", 0, width=value_width,
            )
            # Optimized GDN paths may invoke the fused gate/norm kernel
            # without passing through nn.Module.__call__, so a hook on
            # mixer.norm is not reliable. The out-projection input is the
            # architectural boundary immediately after gate/norm in every
            # backend path.
            register_input(
                getattr(mixer, "out_proj", None), layer_index,
                "linear_attention_gate_norm", width=value_width,
            )
            register_output(
                getattr(mixer, "out_proj", None), layer_index,
                "linear_attention_output_projection",
            )
        layer.post_attention_layernorm.register_forward_hook(
            make_tensor_hook(layer_index, "ffn_norm")
        )
        layer.post_attention_layernorm.register_forward_hook(
            make_tensor_hook(layer_index, "attention_residual", 1)
        )
        layer.mlp.register_forward_hook(make_tensor_hook(layer_index, "moe"))
        layer.register_forward_hook(make_hook(layer_index))
    return {
        "rank": rank,
        "layers": len(layers),
        "path": selected_path,
        "points": sorted(registered_points),
    }


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
    parser.add_argument("--layer-trace", type=Path)
    parser.add_argument("--layer-trace-step", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= MAX_RESULT_TOKENS:
        parser.error(f"--max-new-tokens must be between 1 and {MAX_RESULT_TOKENS}")
    if args.layer_trace_step < 0:
        parser.error("--layer-trace-step must be non-negative")

    # FlashInfer's sampler architecture gate misclassifies some SM120/SM121
    # wheel combinations. vLLM's native sampler is the supported fallback.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

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
    attention_backend = os.environ.get(
        "OPENNPUX_VLLM_ATTENTION_BACKEND", "TRITON_ATTN"
    )
    enforce_eager = os.environ.get("OPENNPUX_VLLM_ENFORCE_EAGER", "1") != "0"
    if args.layer_trace is not None:
        # Callable RPC serialization is opt-in in current vLLM releases and
        # must be enabled before worker processes are created.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(
        model=str(args.model_dir),
        trust_remote_code=True,
        quantization=quantization,
        language_model_only=True,
        attention_backend=attention_backend,
        enforce_eager=enforce_eager,
        max_model_len=int(os.environ.get("OPENNPUX_VLLM_MAX_MODEL_LEN", "4096")),
        gpu_memory_utilization=float(
            os.environ.get("OPENNPUX_VLLM_GPU_MEMORY_UTILIZATION", "0.9")
        ),
    )
    if args.layer_trace is not None:
        args.layer_trace.parent.mkdir(parents=True, exist_ok=True)
        args.layer_trace.unlink(missing_ok=True)
        registrations = llm.apply_model(
            functools.partial(
                _install_layer_trace_hooks,
                output_path=str(args.layer_trace),
                trace_step=args.layer_trace_step,
            )
        )
        print("hf_numerical_layer_trace=" + str(args.layer_trace))
        print(f"hf_numerical_layer_trace_step={args.layer_trace_step}")
        print("hf_numerical_layer_trace_workers=" + json.dumps(registrations))
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
    print("hf_numerical_vllm_attention_backend=" + attention_backend)
    print(f"hf_numerical_vllm_enforce_eager={int(enforce_eager)}")
    print(
        "hf_numerical_vllm_flashinfer_sampler="
        + os.environ["VLLM_USE_FLASHINFER_SAMPLER"]
    )
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
