#!/usr/bin/env python3
"""Create an independently evaluated arena for the MAX adapter smoke graph."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    tensors = {tensor["name"]: tensor for tensor in metadata["tensors"]}
    arena = bytearray(metadata["arena_size"])
    inputs = [1.0, 2.0, 3.0, 4.0, -4.0, 3.0, -2.0, 1.0]
    weights = [
        1.0, 0.0, -1.0,
        0.0, 1.0, -1.0,
        1.0, 1.0, 0.0,
        0.0, -1.0, 1.0,
    ]
    bias = [0.25, -0.5, 0.75, -0.25, 0.5, -0.75]

    def write(name: str, fmt: str, values: list[float] | list[int]) -> None:
        struct.pack_into(
            f"<{len(values)}{fmt}", arena, tensors[name]["offset"], *values
        )

    write("input", "f", inputs)
    write("weight", "f", weights)
    write("bias", "f", bias)
    top_values = []
    top_indices = []
    for row in range(2):
        projected = []
        for column in range(3):
            value = bias[row * 3 + column]
            for inner in range(4):
                value += inputs[row * 4 + inner] * weights[inner * 3 + column]
            projected.append(value / (1.0 + math.exp(-value)))
        maximum = max(range(3), key=projected.__getitem__)
        exponential = [math.exp(value - max(projected)) for value in projected]
        denominator = sum(exponential)
        top_values.append(exponential[maximum] / denominator)
        top_indices.append(maximum)
    write("top_values", "f", top_values)
    write("top_indices", "i", top_indices)
    args.output.write_bytes(arena)
    print(f"mojo_max_arena={args.output}")
    print(f"mojo_max_reference_indices={','.join(map(str, top_indices))}")
    print("mojo_max_arena=PASS")


if __name__ == "__main__":
    main()
