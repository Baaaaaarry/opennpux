#!/usr/bin/env python3
"""Build one runtime Tensor arena for a compiled XGraph artifact."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


PACKERS = {
    "float32": "f",
    "int32": "i",
}
RUNTIME_STORAGES = {"input", "constant", "state"}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def build_image(metadata: dict[str, Any], values: dict[str, Any]) -> bytes:
    arena_size = metadata.get("arena_size")
    tensors = metadata.get("tensors")
    if not isinstance(arena_size, int) or arena_size < 0x20000:
        raise ValueError("metadata has an invalid arena_size")
    if not isinstance(tensors, list):
        raise ValueError("metadata has no tensor table")
    image = bytearray(arena_size)
    consumed: set[str] = set()
    for tensor in tensors:
        if not isinstance(tensor, dict):
            raise ValueError("tensor metadata must contain objects")
        name = tensor.get("name")
        storage = tensor.get("storage")
        dtype = tensor.get("dtype")
        offset = tensor.get("offset")
        byte_size = tensor.get("byte_size")
        if storage not in RUNTIME_STORAGES:
            if name in values:
                raise ValueError(f"tensor {name}: values are only valid for runtime inputs")
            continue
        if not isinstance(name, str) or name not in values:
            raise ValueError(f"tensor {name}: runtime values are required")
        if dtype not in PACKERS or not isinstance(offset, int) or not isinstance(byte_size, int):
            raise ValueError(f"tensor {name}: invalid metadata")
        tensor_values = values[name]
        if not isinstance(tensor_values, list):
            raise ValueError(f"tensor {name}: values must be an array")
        element_size = struct.calcsize("<" + PACKERS[dtype])
        if len(tensor_values) * element_size != byte_size:
            raise ValueError(
                f"tensor {name}: expected {byte_size // element_size} values, "
                f"got {len(tensor_values)}"
            )
        end = offset + byte_size
        if offset < 0x20000 or end > arena_size:
            raise ValueError(f"tensor {name}: range is outside the Tensor arena")
        image[offset:end] = struct.pack(
            f"<{len(tensor_values)}{PACKERS[dtype]}", *tensor_values
        )
        consumed.add(name)
    unknown = set(values) - consumed
    if unknown:
        raise ValueError(f"values contain unknown runtime tensors: {', '.join(sorted(unknown))}")
    return bytes(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path, help="XGraph .npxg.json metadata")
    parser.add_argument("values", type=Path, help="runtime Tensor values JSON")
    parser.add_argument("output", type=Path, help="output raw Tensor arena image")
    args = parser.parse_args()
    try:
        image = build_image(_load_object(args.metadata), _load_object(args.values))
    except (OSError, ValueError, struct.error) as error:
        raise SystemExit(f"xgraph_tensor_image=FAIL: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(image)
    print(f"xgraph_tensor_arena={args.output}")
    print(f"xgraph_tensor_arena_bytes={len(image)}")
    print("xgraph_tensor_image=PASS")


if __name__ == "__main__":
    main()
