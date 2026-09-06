#!/usr/bin/env python3
"""Create a model-like Relax Transformer block for OpenNPUX BYOC tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import tvm
from tvm import relax


def build_module():
    hidden = relax.Var("hidden", relax.TensorStructInfo([2, 64], "float32"))
    norm_weight = relax.Var(
        "norm_weight", relax.TensorStructInfo([64], "float32")
    )
    projection_weight = relax.Var(
        "projection_weight", relax.TensorStructInfo([64, 64], "float32")
    )
    residual = relax.Var(
        "residual", relax.TensorStructInfo([2, 64], "float32")
    )
    builder = relax.BlockBuilder()
    with builder.function(
        "main", [hidden, norm_weight, projection_weight, residual]
    ):
        with builder.dataflow():
            normalized = builder.emit(
                relax.op.nn.rms_norm(
                    hidden, norm_weight, axes=[-1], epsilon=1.0e-5
                ),
                name_hint="normalized",
            )
            projected = builder.emit(
                relax.op.matmul(normalized, projection_weight),
                name_hint="projected",
            )
            residual_sum = builder.emit(
                relax.op.add(projected, residual), name_hint="residual_sum"
            )
            activated = builder.emit(
                relax.op.nn.silu(residual_sum), name_hint="activated"
            )
            output = builder.emit_output(activated, name_hint="output")
        builder.emit_func_output(output)
    return relax.transform.CanonicalizeBindings()(builder.get())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tvm.ir.save_json(build_module()), encoding="utf-8")
    print(f"tvm_version={tvm.__version__}")
    print("tvm_transformer_block=PASS")


if __name__ == "__main__":
    main()
