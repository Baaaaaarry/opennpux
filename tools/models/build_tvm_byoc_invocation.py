#!/usr/bin/env python3
"""Build a dynamic Tensor binding image for a packaged BYOC module."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

from opennpux_tvm_byoc import CodegenError


MAGIC = 0x4958504E
VERSION = 1
HEADER = struct.Struct("<8I")
BINDING = struct.Struct("<6I")


def align(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) & -alignment


def checksum(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def module_identity(manifest: dict) -> int:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return checksum(canonical.encode("utf-8"))


def parse_arena(value: str) -> tuple[str, Path]:
    name, separator, filename = value.partition("=")
    if not separator or not name or not filename:
        raise argparse.ArgumentTypeError("arena must use REGION=FILE syntax")
    return name, Path(filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arena", action="append", default=[], type=parse_arena)
    args = parser.parse_args()
    try:
        manifest = json.loads(
            (args.module / "module.npxgm.json").read_text(encoding="utf-8")
        )
        arena_paths = dict(args.arena)
        payload_offset = align(
            HEADER.size
            + sum(len(region.get("external_bindings", []))
                  for region in manifest["regions"]) * BINDING.size
        )
        cursor = payload_offset
        records: list[tuple[int, int, int, int, int, int]] = []
        payloads: list[tuple[int, bytes]] = []
        for region_index, region in enumerate(manifest["regions"]):
            name = region["name"]
            if name not in arena_paths:
                raise CodegenError(f"missing invocation arena for region {name}")
            arena = arena_paths[name].read_bytes()
            if len(arena) != region["arena_size"]:
                raise CodegenError(f"region {name} arena size mismatch")
            metadata = json.loads(
                (args.module / f"{region['artifact']}.json").read_text(
                    encoding="utf-8"
                )
            )
            tensors = {tensor["name"]: tensor for tensor in metadata["tensors"]}
            for tensor_name in region.get("external_bindings", []):
                tensor = tensors[tensor_name]
                target_offset = int(tensor["offset"])
                byte_size = int(tensor["byte_size"])
                data = arena[target_offset:target_offset + byte_size]
                data_offset = cursor
                records.append((region_index, target_offset, byte_size,
                                data_offset, checksum(data), 0))
                payloads.append((data_offset, data))
                cursor = align(cursor + byte_size)
        if not records:
            raise CodegenError("module invocation has no external bindings")
        if cursor > 0xFFFFFFFF:
            raise CodegenError("module invocation exceeds 32-bit ABI")
        image = bytearray(cursor)
        image[:HEADER.size] = HEADER.pack(
            MAGIC, VERSION, HEADER.size, len(image), len(records),
            BINDING.size, payload_offset, module_identity(manifest),
        )
        offset = HEADER.size
        for record in records:
            image[offset:offset + BINDING.size] = BINDING.pack(*record)
            offset += BINDING.size
        for payload_position, payload in payloads:
            image[payload_position:payload_position + len(payload)] = payload
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(image)
    except (OSError, KeyError, ValueError, CodegenError) as error:
        print(f"xgraph_invocation=FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"xgraph_invocation={args.output}")
    print(f"xgraph_invocation_bytes={len(image)}")
    print(f"xgraph_invocation_bindings={len(records)}")
    print(f"xgraph_invocation_module_identity=0x{module_identity(manifest):08x}")
    print("xgraph_invocation=PASS")


if __name__ == "__main__":
    main()
