#!/usr/bin/env python3
"""Materialize one real GPTQ projection as a Coral EXTMEM image."""

import argparse
import json
import struct
from pathlib import Path
from typing import Any


EXTMEM_BASE = 0x20000000
REQUEST = struct.Struct("<16I4Q8I")
RANGE_HEADER = struct.Struct("<8I4Q")
RANGE_RECORD = struct.Struct("<4I6Q")
RANGE_MAGIC = 0x5258504E
REQUEST_MAGIC = 0x514D3447
REQUEST_VERSION = 2
COMPONENTS = {
    0x40406979: "qweight",
    0x5D11CEDB: "qzeros",
    0x76C0006C: "scales",
    0x1B2EDE4B: "g_idx",
}
ROLES = {
    "attention_q_proj": 0xA3F281BA,
    "attention_k_proj": 0xF91B83DD,
    "attention_v_proj": 0x77364AB0,
    "attention_o_proj": 0xFA6AFC40,
    "routed_expert": 0x676A07A0,
    "shared_expert": 0x06FC7F70,
}
SLOTS = {
    "q_proj": 1, "k_proj": 2, "v_proj": 3, "o_proj": 4,
    "gate_proj": 5, "up_proj": 6, "down_proj": 7, "qkv_proj": 8,
}
DTYPE_BYTES = {4: 2, 5: 2, 6: 4}


def align(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) & -alignment


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def command_for_layer(plan_path: Path, layer: int, phase: str) -> int:
    plan = load_json(plan_path)
    for command in plan.get("commands", []):
        command_layer = command.get("layer")
        if (
            command_layer is not None and int(command_layer) == layer
            and command.get("phase") == phase
        ):
            return int(command["command_id"])
    raise ValueError(f"no command for layer={layer} phase={phase}")


def load_ranges(path: Path) -> list[tuple[int, ...]]:
    data = path.read_bytes()
    if len(data) < RANGE_HEADER.size:
        raise ValueError("truncated NPU weight range index")
    header = RANGE_HEADER.unpack_from(data)
    if (
        header[0] != RANGE_MAGIC or header[1] != 1
        or header[2] != RANGE_HEADER.size or header[3] != RANGE_RECORD.size
        or header[8] + header[5] * header[3] != len(data)
    ):
        raise ValueError("invalid NPU weight range index")
    return [
        RANGE_RECORD.unpack_from(data, header[8] + index * header[3])
        for index in range(header[5])
    ]


def projection_shape(manifest: dict[str, Any], slot: str) -> tuple[int, int]:
    hidden = int(manifest["hidden_size"])
    heads = int(manifest["head_count"])
    kv_heads = int(manifest["kv_head_count"])
    head_dim = int(manifest["head_dim"])
    moe = int(manifest["moe_intermediate_size"])
    shapes = {
        "q_proj": (hidden, heads * head_dim),
        "k_proj": (hidden, kv_heads * head_dim),
        "v_proj": (hidden, kv_heads * head_dim),
        "o_proj": (heads * head_dim, hidden),
        "gate_proj": (hidden, moe),
        "up_proj": (hidden, moe),
        "down_proj": (moe, hidden),
        "qkv_proj": (hidden, (heads + 2 * kv_heads) * head_dim),
    }
    return shapes[slot]


def read_component(
    model_dir: Path, manifest: dict[str, Any], record: tuple[int, ...]
) -> bytes:
    shard_index, offset, size = record[1], record[4], record[5]
    shards = manifest.get("shards", [])
    if shard_index >= len(shards):
        raise ValueError("projection range has invalid shard index")
    shard = model_dir / str(shards[shard_index]["path"])
    if not shard.is_file() or offset + size > shard.stat().st_size:
        raise ValueError(f"projection range is outside shard: {shard}")
    with shard.open("rb") as source:
        source.seek(offset)
        payload = source.read(size)
    if len(payload) != size:
        raise ValueError(f"short projection component read: {shard}")
    return payload


def materialize(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    model_dir = manifest_path.parent
    manifest = load_json(manifest_path)
    command_id = args.command_id
    if command_id is None:
        command_id = command_for_layer(
            args.weight_plan.resolve(), args.layer, args.phase
        )
    role_id = ROLES[args.role]
    slot_id = SLOTS[args.slot]
    expert_id = (1 << 64) - 1 if args.expert == -1 else args.expert
    selected: dict[str, tuple[int, ...]] = {}
    for record in load_ranges(args.range_index.resolve()):
        flags = record[9]
        component = COMPONENTS.get(record[3])
        if (
            record[0] == command_id and record[2] == role_id
            and record[8] == expert_id and flags & 0xFFFF == slot_id
            and component is not None
        ):
            if component in selected:
                raise ValueError(f"duplicate projection component: {component}")
            selected[component] = record
    missing = {"qweight", "qzeros", "scales"} - selected.keys()
    if missing:
        raise ValueError(f"missing projection components: {sorted(missing)}")

    components = {
        name: read_component(model_dir, manifest, record)
        for name, record in selected.items()
    }
    scale_dtype = (selected["scales"][9] >> 16) & 0xFF
    if scale_dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported projection scale dtype: {scale_dtype}")
    input_columns, output_columns = projection_shape(manifest, args.slot)
    group_size = int(manifest["quantization_group_size"])
    groups = (input_columns + group_size - 1) // group_size
    expected = {
        "qweight": ((input_columns + 7) // 8) * output_columns * 4,
        "qzeros": groups * ((output_columns + 7) // 8) * 4,
        "scales": groups * output_columns * DTYPE_BYTES[scale_dtype],
        "g_idx": input_columns * 4,
    }
    for name in ("qweight", "qzeros", "scales"):
        if len(components[name]) != expected[name]:
            raise ValueError(
                f"{name} size mismatch: {len(components[name])} != {expected[name]}"
            )
    if "g_idx" in components and len(components["g_idx"]) != expected["g_idx"]:
        raise ValueError("g_idx size mismatch")

    offsets: dict[str, int] = {}
    cursor = align(REQUEST.size)
    input_data = b"".join(
        struct.pack("<f", ((index % 17) - 8) / 16.0)
        for index in range(input_columns * args.rows)
    )
    payloads = {"input": input_data, **components}
    payloads["output"] = bytes(args.rows * output_columns * 4)
    for name in ("input", "qweight", "qzeros", "scales", "g_idx", "output"):
        if name not in payloads:
            continue
        offsets[name] = cursor
        cursor = align(cursor + len(payloads[name]))
    image = bytearray(cursor)
    for name, payload in payloads.items():
        image[offsets[name] : offsets[name] + len(payload)] = payload
    address = lambda name: EXTMEM_BASE + offsets[name] if name in offsets else 0
    request = REQUEST.pack(
        REQUEST_MAGIC, REQUEST_VERSION, REQUEST.size, 0, 0,
        args.rows, input_columns, output_columns, group_size,
        int(manifest.get("quantization_zero_bias", 1)),
        address("input"), address("qweight"), address("qzeros"),
        address("scales"), address("g_idx"), address("output"),
        0, 0, 0, 0, 0, scale_dtype, 0, 0, 0, 0, 0, 0,
    )
    image[:REQUEST.size] = request
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(image)
    print(f"gptq_projection_command={command_id}")
    print(f"gptq_projection_layer={args.layer}")
    print(
        "gptq_projection_expert="
        + ("none" if args.expert == -1 else str(args.expert))
    )
    print(f"gptq_projection_slot={args.slot}")
    print(f"gptq_projection_shape={args.rows}x{input_columns}x{output_columns}")
    print(f"gptq_projection_scale_dtype={scale_dtype}")
    for name in ("qweight", "qzeros", "scales", "g_idx"):
        print(f"gptq_projection_{name}_bytes={len(components.get(name, b''))}")
    print(f"gptq_projection_image_bytes={len(image)}")
    print(f"gptq_projection_output_offset=0x{offsets['output']:x}")
    print("gptq_projection_materialize=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("weight_plan", type=Path)
    parser.add_argument("range_index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--phase", default="routed_experts_active_only")
    parser.add_argument("--role", choices=ROLES, default="routed_expert")
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--slot", choices=SLOTS, default="gate_proj")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--command-id", type=int)
    args = parser.parse_args()
    if args.layer < 0 or args.expert < -1 or args.rows <= 0:
        parser.error("layer must be non-negative, expert >= -1, and rows non-zero")
    materialize(args)


if __name__ == "__main__":
    main()
