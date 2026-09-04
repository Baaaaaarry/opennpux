#!/usr/bin/env python3
"""Create a Relax graph with two NPU regions separated by one Host op."""

from __future__ import annotations

import argparse
from pathlib import Path

import tvm
from tvm import relax


def build_module():
    lhs = relax.Var("lhs", relax.TensorStructInfo([2, 4], "float32"))
    rhs = relax.Var("rhs", relax.TensorStructInfo([2, 4], "float32"))
    builder = relax.BlockBuilder()
    with builder.function("main", [lhs, rhs]):
        with builder.dataflow():
            added = builder.emit(relax.op.add(lhs, rhs), name_hint="added")
            host_relu = builder.emit(relax.op.nn.relu(added), name_hint="host_relu")
            activated = builder.emit(
                relax.op.nn.silu(host_relu), name_hint="activated"
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
    print("tvm_multi_region_model=PASS")


if __name__ == "__main__":
    main()
