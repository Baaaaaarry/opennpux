#!/usr/bin/env python3

import argparse
import struct
from pathlib import Path

MODEL_MAGIC = 0x4E50584D
MODEL_VERSION = 1
OP_VECTOR_ADD = 1
OP_VECTOR_ADD_CUSTOM = 2
ELEMENTS = 16
HEADER_SIZE = 32
COMMAND_SIZE = 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output", nargs="?", default="build/models/heterogeneous-smoke.npxm"
    )
    args = parser.parse_args()

    input0 = list(range(1, ELEMENTS + 1))
    input1 = [value * 2 for value in input0]
    expected_checksum = sum(a + b for a, b in zip(input0, input1))
    command_count = 2
    command_offset = HEADER_SIZE
    tensor_offset = HEADER_SIZE + command_count * COMMAND_SIZE
    input0_offset = tensor_offset
    input1_offset = input0_offset + ELEMENTS * 4
    file_size = input1_offset + ELEMENTS * 4

    header = struct.pack(
        "<8I",
        MODEL_MAGIC,
        MODEL_VERSION,
        HEADER_SIZE,
        file_size,
        command_count,
        command_offset,
        0,
        0,
    )
    commands = b"".join(
        struct.pack(
            "<8I",
            opcode,
            ELEMENTS,
            input0_offset,
            input1_offset,
            expected_checksum,
            0,
            0,
            0,
        )
        for opcode in (OP_VECTOR_ADD, OP_VECTOR_ADD_CUSTOM)
    )
    tensors = struct.pack("<16I", *input0) + struct.pack("<16I", *input1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(header + commands + tensors)
    print(f"built: {output} ({file_size} bytes)")


if __name__ == "__main__":
    main()
