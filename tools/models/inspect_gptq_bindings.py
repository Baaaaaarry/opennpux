#!/usr/bin/env python3
"""Validate model-independent GPTQ component groups in a range index."""

import argparse
import struct
from collections import Counter, defaultdict
from pathlib import Path


HEADER = struct.Struct("<8I4Q")
RECORD = struct.Struct("<4I6Q")
MAGIC = 0x5258504E
COMPONENTS = {
    0x40406979: "qweight",
    0x5D11CEDB: "qzeros",
    0x76C0006C: "scales",
    0x1B2EDE4B: "g_idx",
}
DTYPES = {
    0: "invalid", 1: "int4_packed", 2: "int8", 3: "int32",
    4: "float16", 5: "bfloat16", 6: "float32",
}
REQUIRED = frozenset({"qweight", "qzeros", "scales"})
SLOTS = {
    0: "default",
    1: "q_proj",
    2: "k_proj",
    3: "v_proj",
    4: "o_proj",
    5: "gate_proj",
    6: "up_proj",
    7: "down_proj",
    8: "qkv_proj",
}


def load_groups(path: Path):
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("truncated NPU weight range index")
    header = HEADER.unpack_from(data)
    if (
        header[0] != MAGIC
        or header[1] != 1
        or header[2] != HEADER.size
        or header[3] != RECORD.size
        or header[8] + header[5] * header[3] != len(data)
    ):
        raise ValueError("invalid NPU weight range index ABI")
    groups = defaultdict(lambda: defaultdict(list))
    for index in range(header[5]):
        record = RECORD.unpack_from(data, header[8] + index * header[3])
        component = COMPONENTS.get(record[3])
        if component is None:
            continue
        slot = record[9] & 0xFFFF
        key = (record[0], record[2], record[8], slot)
        dtype = (record[9] >> 16) & 0xFF
        groups[key][component].append((record[1], record[4], record[5], dtype))
    return header, groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("range_index", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    header, groups = load_groups(args.range_index)
    complete = 0
    incomplete = 0
    duplicate = 0
    component_bytes = Counter()
    component_dtypes = defaultdict(Counter)
    slot_counts = Counter()
    for (_, _, _, slot), components in groups.items():
        missing = REQUIRED - components.keys()
        duplicates = sum(len(records) - 1 for records in components.values())
        if missing:
            incomplete += 1
        elif duplicates:
            duplicate += 1
        else:
            complete += 1
            slot_counts[SLOTS.get(slot, f"slot_{slot}")] += 1
        for name, records in components.items():
            component_bytes[name] += sum(record[2] for record in records)
            component_dtypes[name].update(record[3] for record in records)

    print(f"gptq_binding_range_records={header[5]}")
    print(f"gptq_binding_groups={len(groups)}")
    print(f"gptq_binding_complete={complete}")
    print(f"gptq_binding_incomplete={incomplete}")
    print(f"gptq_binding_duplicate={duplicate}")
    for name in ("qweight", "qzeros", "scales", "g_idx"):
        print(f"gptq_binding_{name}_bytes={component_bytes[name]}")
        print(
            f"gptq_binding_{name}_dtypes="
            + ",".join(
                f"{DTYPES.get(dtype, f'dtype_{dtype}')}:{count}"
                for dtype, count in sorted(component_dtypes[name].items())
            )
        )
    print(
        "gptq_binding_slots="
        + ",".join(f"{name}:{slot_counts[name]}" for name in sorted(slot_counts))
    )
    if args.require_complete and (incomplete or duplicate or complete == 0):
        raise SystemExit("gptq_binding_inspect=FAIL")
    print("gptq_binding_inspect=PASS")


if __name__ == "__main__":
    main()
