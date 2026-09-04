#!/usr/bin/env python3
"""Create a real Relax graph used by the OpenNPUX BYOC end-to-end test."""

from __future__ import annotations

import argparse
from pathlib import Path

import tvm
from tvm import relax


def build_module():
    input_tensor = relax.Var(
        "input", relax.TensorStructInfo([2, 2048], "float32")
    )
    weight = relax.Var("weight", relax.TensorStructInfo([2048, 8], "float32"))
    bias = relax.Var("bias", relax.TensorStructInfo([2, 8], "float32"))
    builder = relax.BlockBuilder()
    with builder.function("main", [input_tensor, weight, bias]):
        with builder.dataflow():
            matmul = builder.emit(
                relax.op.matmul(input_tensor, weight), name_hint="matmul"
            )
            biased = builder.emit(relax.op.add(matmul, bias), name_hint="biased")
            activated = builder.emit(
                relax.op.nn.silu(biased), name_hint="activated"
            )
            probabilities = builder.emit(
                relax.op.nn.softmax(activated, axis=-1),
                name_hint="probabilities",
            )
            output = builder.emit_output(probabilities, name_hint="output")
        builder.emit_func_output(output)
    return relax.transform.CanonicalizeBindings()(builder.get())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tvm.ir.save_json(build_module()), encoding="utf-8")
    print(f"tvm_version={tvm.__version__}")
    print("tvm_relax_model=PASS")


if __name__ == "__main__":
    main()
