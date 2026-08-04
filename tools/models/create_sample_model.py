#!/usr/bin/env python3
"""Generate the heterogeneous smoke-test .npxm model container.

.npxm is the OpenNPUX binary model format.  A file consists of:

    [header 32 B] [command 0 32 B] ... [command N 32 B] [tensor data ...]

The header (8 × uint32 little-endian):
    0  magic           0x4E50584D ("NPXM")
    1  version         1
    2  header_size     32
    3  file_size       total bytes
    4  command_count   number of operator descriptors
    5  command_offset  byte offset to first command (usually 32)
    6  reserved
    7  reserved

Each command descriptor (8 × uint32 little-endian):
    0  opcode          operator ID (1 = VECTOR_ADD, 2 = VECTOR_ADD_CUSTOM)
    1  element_count   number of uint32 elements per tensor
    2  input0_offset   byte offset to first input tensor
    3  input1_offset   byte offset to second input tensor
    4  checksum        expected output checksum (sum of all output elements)
    5  reserved
    6  reserved
    7  reserved

The generated model contains two operators:
    - VECTOR_ADD (opcode 1): official Coral software path
    - VECTOR_ADD_CUSTOM (opcode 2): custom RTL accelerator path

Both operate on the same input tensors (1..16 and 2..32), so the model
validates that software and custom RTL produce identical results.

Usage:
    ./tools/models/create_sample_model.py [output.npxm]
"""

import argparse
import struct
from pathlib import Path

MODEL_MAGIC = 0x4E50584D   # "NPXM"
MODEL_VERSION = 1
OP_VECTOR_ADD = 1           # official Coral software operator
OP_VECTOR_ADD_CUSTOM = 2    # custom RTL accelerator operator
ELEMENTS = 16
HEADER_SIZE = 32
COMMAND_SIZE = 32


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the heterogeneous smoke-test .npxm model."
    )
    parser.add_argument(
        "output", nargs="?", default="build/models/heterogeneous-smoke.npxm"
    )
    args = parser.parse_args()

    # Create two input tensors: [1..16] and [2,4,6,...,32].
    input0 = list(range(1, ELEMENTS + 1))
    input1 = [value * 2 for value in input0]
    expected_checksum = sum(a + b for a, b in zip(input0, input1))

    # Layout: header → commands → tensor data.
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

    # Both operators share the same tensor layout and expected checksum.
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
