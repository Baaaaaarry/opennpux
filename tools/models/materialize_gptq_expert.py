#!/usr/bin/env python3
"""Materialize one real GPTQ gated MLP expert as a Coral EXTMEM image."""

import argparse
import struct
from pathlib import Path

import materialize_gptq_projection as projection


REQUEST = struct.Struct("<15I18I5I8Q10I")
REQUEST_MAGIC = 0x45583447
REQUEST_VERSION = 1
SLOTS = ("gate_proj", "up_proj", "down_proj")


def select_components(
    records: list[tuple[int, ...]], command_id: int, role_id: int,
    expert_id: int, slot: str,
) -> dict[str, tuple[int, ...]]:
    selected: dict[str, tuple[int, ...]] = {}
    slot_id = projection.SLOTS[slot]
    for record in records:
        component = projection.COMPONENTS.get(record[3])
        if (
            record[0] == command_id and record[2] == role_id
            and record[8] == expert_id and record[9] & 0xFFFF == slot_id
            and component is not None
        ):
            if component in selected:
                raise ValueError(f"duplicate {slot} component: {component}")
            selected[component] = record
    missing = {"qweight", "qzeros", "scales"} - selected.keys()
    if missing:
        raise ValueError(f"missing {slot} components: {sorted(missing)}")
    return selected


def materialize(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    model_dir = manifest_path.parent
    manifest = projection.load_json(manifest_path)
    command_id = args.command_id
    if command_id is None:
        command_id = projection.command_for_layer(
            args.weight_plan.resolve(), args.layer, args.phase
        )
    records = projection.load_ranges(args.range_index.resolve())
    role_id = projection.ROLES[args.role]
    selections = {
        slot: select_components(records, command_id, role_id, args.expert, slot)
        for slot in SLOTS
    }
    components = {
        slot: {
            name: projection.read_component(model_dir, manifest, record)
            for name, record in selected.items()
        }
        for slot, selected in selections.items()
    }
    scale_dtypes = {
        (selected["scales"][9] >> 16) & 0xFF
        for selected in selections.values()
    }
    if len(scale_dtypes) != 1:
        raise ValueError(f"mixed expert scale dtypes: {sorted(scale_dtypes)}")
    scale_dtype = next(iter(scale_dtypes))
    if scale_dtype not in projection.DTYPE_BYTES:
        raise ValueError(f"unsupported expert scale dtype: {scale_dtype}")

    hidden, intermediate = projection.projection_shape(manifest, "gate_proj")
    if projection.projection_shape(manifest, "up_proj") != (
        hidden, intermediate
    ) or projection.projection_shape(manifest, "down_proj") != (
        intermediate, hidden
    ):
        raise ValueError("model does not describe a gated MLP projection trio")
    group_size = int(manifest["quantization_group_size"])

    def validate(slot: str, input_columns: int, output_columns: int) -> None:
        groups = (input_columns + group_size - 1) // group_size
        expected = {
            "qweight": ((input_columns + 7) // 8) * output_columns * 4,
            "qzeros": groups * ((output_columns + 7) // 8) * 4,
            "scales": groups * output_columns * projection.DTYPE_BYTES[scale_dtype],
            "g_idx": input_columns * 4,
        }
        for name in ("qweight", "qzeros", "scales"):
            actual = len(components[slot][name])
            if actual != expected[name]:
                raise ValueError(
                    f"{slot} {name} size mismatch: {actual} != {expected[name]}"
                )
        if "g_idx" in components[slot] and (
            len(components[slot]["g_idx"]) != expected["g_idx"]
        ):
            raise ValueError(f"{slot} g_idx size mismatch")

    validate("gate_proj", hidden, intermediate)
    validate("up_proj", hidden, intermediate)
    validate("down_proj", intermediate, hidden)

    payloads: dict[str, bytes] = {
        "input": b"".join(
            struct.pack("<f", ((index % 17) - 8) / 16.0)
            for index in range(args.rows * hidden)
        ),
        "gate_output": bytes(args.rows * intermediate * 4),
        "up_output": bytes(args.rows * intermediate * 4),
        "activated": bytes(args.rows * intermediate * 4),
        "output": bytes(args.rows * hidden * 4),
    }
    for slot in SLOTS:
        for name in ("qweight", "qzeros", "scales", "g_idx"):
            if name in components[slot]:
                payloads[f"{slot}_{name}"] = components[slot][name]

    offsets: dict[str, int] = {}
    cursor = projection.align(REQUEST.size)
    for name, payload in payloads.items():
        offsets[name] = cursor
        cursor = projection.align(cursor + len(payload))
    image = bytearray(cursor)
    for name, payload in payloads.items():
        image[offsets[name] : offsets[name] + len(payload)] = payload

    def address(name: str) -> int:
        return projection.EXTMEM_BASE + offsets[name] if name in offsets else 0

    header = [
        REQUEST_MAGIC, REQUEST_VERSION, REQUEST.size, 0, 0, args.rows,
        hidden, intermediate, group_size,
        int(manifest.get("quantization_zero_bias", 1)), address("input"),
        address("gate_output"), address("up_output"), address("activated"),
        address("output"),
    ]
    weight_fields: list[int] = []
    for slot in SLOTS:
        weight_fields.extend([
            address(f"{slot}_qweight"), address(f"{slot}_qzeros"),
            address(f"{slot}_scales"), address(f"{slot}_g_idx"),
            scale_dtype, 0,
        ])
    request = REQUEST.pack(
        *header, *weight_fields, *([0] * 5), *([0] * 8), *([0] * 10)
    )
    image[: REQUEST.size] = request
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(image)
    print(f"gptq_expert_command={command_id}")
    print(f"gptq_expert_layer={args.layer}")
    print(f"gptq_expert_id={args.expert}")
    print(f"gptq_expert_shape={args.rows}x{hidden}x{intermediate}")
    print(f"gptq_expert_scale_dtype={scale_dtype}")
    for slot in SLOTS:
        size = sum(len(value) for value in components[slot].values())
        print(f"gptq_expert_{slot}_weight_bytes={size}")
    print(f"gptq_expert_image_bytes={len(image)}")
    print(f"gptq_expert_output_offset=0x{offsets['output']:x}")
    print(
        "gptq_expert_expected_operations="
        f"{6 * args.rows * hidden * intermediate + 6 * args.rows * intermediate}"
    )
    print("gptq_expert_materialize=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("weight_plan", type=Path)
    parser.add_argument("range_index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--phase", default="routed_experts_active_only")
    parser.add_argument("--role", choices=projection.ROLES,
                        default="routed_expert")
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--command-id", type=int)
    args = parser.parse_args()
    if args.layer < 0 or args.expert < 0 or args.rows <= 0:
        parser.error("layer, expert, and rows must be non-negative/non-zero")
    materialize(args)


if __name__ == "__main__":
    main()
