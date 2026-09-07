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
COMMAND_SIZE = 64
COMMAND_OFFSET = 96
COMMAND_FIELDS = {
    "flags": 4,
    "dim0": 20,
    "dim1": 24,
    "dim2": 28,
    "scalar0": 32,
    "reserved0": 44,
    "reserved1": 48,
    "reserved2": 52,
    "reserved3": 56,
    "reserved4": 60,
}
COMMAND_U32 = 1


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


def parse_scalar(value: str) -> tuple[str, int]:
    name, separator, raw_value = value.partition("=")
    if not separator or not name or not raw_value:
        raise argparse.ArgumentTypeError("scalar must use NAME=VALUE syntax")
    try:
        parsed = int(raw_value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError("scalar value must be an integer") from error
    if not 0 <= parsed <= 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("scalar value must be a uint32")
    return name, parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arena", action="append", default=[], type=parse_arena)
    parser.add_argument("--scalar", action="append", default=[], type=parse_scalar)
    args = parser.parse_args()
    try:
        manifest = json.loads(
            (args.module / "module.npxgm.json").read_text(encoding="utf-8")
        )
        arena_paths = dict(args.arena)
        scalar_values = dict(args.scalar)
        if len(scalar_values) != len(args.scalar):
            raise CodegenError("duplicate scalar binding value")
        scalar_bindings = manifest.get("scalar_bindings", [])
        declared_scalars = {binding["name"] for binding in scalar_bindings}
        if set(scalar_values) != declared_scalars:
            missing = sorted(declared_scalars - set(scalar_values))
            unexpected = sorted(set(scalar_values) - declared_scalars)
            raise CodegenError(
                f"scalar bindings mismatch missing={missing} unexpected={unexpected}"
            )
        payload_offset = align(
            HEADER.size
            + sum(len(region.get(
                "invocation_bindings", region.get("external_bindings", [])
            ))
                  for region in manifest["regions"]) * BINDING.size
            + len(scalar_bindings) * BINDING.size
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
            for tensor_name in region.get(
                "invocation_bindings", region.get("external_bindings", [])
            ):
                tensor = tensors[tensor_name]
                target_offset = int(tensor["offset"])
                byte_size = int(tensor["byte_size"])
                data = arena[target_offset:target_offset + byte_size]
                data_offset = cursor
                records.append((region_index, target_offset, byte_size,
                                data_offset, checksum(data), 0))
                payloads.append((data_offset, data))
                cursor = align(cursor + byte_size)
        region_indices = {
            region["name"]: index for index, region in enumerate(manifest["regions"])
        }
        for binding in scalar_bindings:
            value = scalar_values[binding["name"]]
            if not binding["minimum"] <= value <= binding["maximum"]:
                raise CodegenError(
                    f"scalar {binding['name']} is outside its declared range"
                )
            data = struct.pack("<I", value)
            data_offset = cursor
            target_offset = (
                COMMAND_OFFSET + binding["command"] * COMMAND_SIZE
                + COMMAND_FIELDS[binding["field"]]
            )
            records.append((region_indices[binding["region"]], target_offset,
                            len(data), data_offset, checksum(data), COMMAND_U32))
            payloads.append((data_offset, data))
            cursor = align(cursor + len(data))
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
