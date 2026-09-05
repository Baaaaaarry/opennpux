#!/usr/bin/env python3
"""Bundle compiled BYOC regions and invocation arenas for the Guest runtime."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

from opennpux_tvm_byoc import CodegenError


MAGIC = 0x4D47584E
VERSION = 1
HEADER = struct.Struct("<16I")
REGION = struct.Struct("<8I")
EDGE = struct.Struct("<6I")
HOST_BINDING = struct.Struct("<7I")
HOST_OPERATION = struct.Struct("<2I")
OUTPUT = struct.Struct("<4I")
HOST_OPCODES = {"relax.nn.relu": 1}


def align(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) & -alignment


def checksum(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def parse_arena(value: str) -> tuple[str, Path]:
    name, separator, filename = value.partition("=")
    if not separator or not name or not filename:
        raise argparse.ArgumentTypeError("arena must use REGION=FILE syntax")
    return name, Path(filename)


def tensor_offset(metadata: dict, name: str) -> int:
    for tensor in metadata.get("tensors", []):
        if tensor.get("name") == name:
            return int(tensor["offset"])
    raise CodegenError(f"Tensor metadata missing {name}")


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
        arenas = dict(args.arena)
        regions = manifest["regions"]
        region_index = {region["name"]: index for index, region in enumerate(regions)}
        metadata = {
            region["name"]: json.loads(
                (args.module / f"{region['artifact']}.json").read_text(encoding="utf-8")
            )
            for region in regions
        }
        graphs = [(args.module / region["artifact"]).read_bytes() for region in regions]
        arena_images = []
        for region in regions:
            name = region["name"]
            if name not in arenas:
                raise CodegenError(f"missing invocation arena for region {name}")
            image = arenas[name].read_bytes()
            if len(image) != region["arena_size"]:
                raise CodegenError(f"region {name} arena size mismatch")
            arena_images.append(image)

        operations = []
        host_bindings = []
        for binding in manifest.get("host_bindings", []):
            if binding.get("dtype") != "float32" or binding["bytes"] % 4 != 0:
                raise CodegenError("packaged Host operations require float32 Tensors")
            first_operation = len(operations)
            for operation in binding["pipeline"]:
                opcode = HOST_OPCODES.get(operation["op"])
                if opcode is None:
                    raise CodegenError(
                        f"unsupported packaged Host operation {operation['op']}"
                    )
                operations.append((opcode, 0))
            host_bindings.append((
                region_index[binding["from_region"]],
                region_index[binding["to_region"]],
                tensor_offset(metadata[binding["from_region"]], binding["from_tensor"]),
                tensor_offset(metadata[binding["to_region"]], binding["to_tensor"]),
                binding["bytes"], first_operation, len(binding["pipeline"]),
            ))
        edges = [(
            region_index[edge["from_region"]],
            region_index[edge["to_region"]],
            tensor_offset(metadata[edge["from_region"]], edge["from_tensor"]),
            tensor_offset(metadata[edge["to_region"]], edge["to_tensor"]),
            edge["bytes"], 0,
        ) for edge in manifest.get("edges", [])]
        outputs = [(
            region_index[output["region"]],
            tensor_offset(metadata[output["region"]], output["tensor"]),
            next(
                tensor["byte_size"]
                for tensor in metadata[output["region"]]["tensors"]
                if tensor["name"] == output["tensor"]
            ),
            checksum(f"{output['region']}.{output['tensor']}"),
        ) for output in manifest.get("module_outputs", [])]
        if not outputs:
            raise CodegenError("module package has no outputs")

        table_size = (
            HEADER.size + len(regions) * REGION.size + len(edges) * EDGE.size
            + len(host_bindings) * HOST_BINDING.size
            + len(operations) * HOST_OPERATION.size + len(outputs) * OUTPUT.size
        )
        payload_offset = align(table_size)
        cursor = payload_offset
        region_records = []
        payloads = []
        for index, region in enumerate(regions):
            graph_offset = cursor
            payloads.append((graph_offset, graphs[index]))
            cursor = align(cursor + len(graphs[index]))
            arena_offset = cursor
            payloads.append((arena_offset, arena_images[index]))
            cursor = align(cursor + len(arena_images[index]))
            graph_metadata = metadata[region["name"]]
            output_name = graph_metadata["output"]
            output_bytes = next(
                tensor["byte_size"] for tensor in graph_metadata["tensors"]
                if tensor["name"] == output_name
            )
            region_records.append((
                graph_offset, len(graphs[index]), arena_offset,
                len(arena_images[index]), tensor_offset(graph_metadata, output_name),
                output_bytes, 0, 0,
            ))
        if cursor > 0xFFFFFFFF:
            raise CodegenError("module invocation package exceeds 32-bit ABI")
        image = bytearray(cursor)
        image[:HEADER.size] = HEADER.pack(
            MAGIC, VERSION, HEADER.size, len(image), len(regions), len(edges),
            len(host_bindings), len(operations), len(outputs), REGION.size,
            EDGE.size, HOST_BINDING.size, HOST_OPERATION.size, OUTPUT.size,
            payload_offset, 0,
        )
        offset = HEADER.size
        for record, layout in (
            (region_records, REGION), (edges, EDGE),
            (host_bindings, HOST_BINDING), (operations, HOST_OPERATION),
            (outputs, OUTPUT),
        ):
            for values in record:
                image[offset:offset + layout.size] = layout.pack(*values)
                offset += layout.size
        for payload_position, payload in payloads:
            image[payload_position:payload_position + len(payload)] = payload
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(image)
    except (OSError, KeyError, StopIteration, ValueError, CodegenError) as error:
        print(f"xgraph_module_package=FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"xgraph_module_package={args.output}")
    print(f"xgraph_module_package_bytes={len(image)}")
    print(f"xgraph_module_package_regions={len(regions)}")
    print(f"xgraph_module_package_host_operations={len(operations)}")
    print("xgraph_module_package=PASS")


if __name__ == "__main__":
    main()
