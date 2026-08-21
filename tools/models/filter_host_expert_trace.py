#!/usr/bin/env python3
"""Filter anomalous routed-expert stages from Host C++ functional traces."""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO


TRACE_PREFIX = "host_functional_expert_trace="
PAIR_RE = re.compile(r"(?:^|,)([a-z_]+):([^,]+)")
STAGE_ORDER = ("input", "gate", "up", "activated", "down")


@dataclass
class RouteTrace:
    line: int
    row: int
    route: int
    expert: int
    route_weight: float
    stages: dict[str, dict[str, str]] = field(default_factory=dict)


def parse_trace(line: str) -> dict[str, str] | None:
    marker = line.find(TRACE_PREFIX)
    if marker < 0:
        return None
    payload = line[marker + len(TRACE_PREFIX):].strip()
    fields = dict(PAIR_RE.findall(payload))
    required = {"row", "route", "expert", "route_weight", "stage", "rms"}
    return fields if required.issubset(fields) else None


def is_suspicious(
    trace: RouteTrace,
    projection_rms: float,
    activation_rms: float,
) -> bool:
    for stage, fields in trace.stages.items():
        rms = float(fields["rms"])
        nonfinite = int(fields.get("nonfinite", "0"))
        threshold = projection_rms if stage in {"gate", "up"} else activation_rms
        if nonfinite != 0 or rms > threshold:
            return True
    return False


def print_trace(trace: RouteTrace, output: TextIO) -> None:
    stages = []
    for stage in STAGE_ORDER:
        fields = trace.stages.get(stage)
        if fields is None:
            continue
        stages.append(
            f"{stage}=rms:{float(fields['rms']):.6g}"
            f"/min:{float(fields.get('min', '0')):.6g}"
            f"/max:{float(fields.get('max', '0')):.6g}"
            f"/nf:{int(fields.get('nonfinite', '0'))}"
        )
    print(
        f"line={trace.line} row={trace.row} route={trace.route} "
        f"expert={trace.expert} weight={trace.route_weight:.9g} "
        + " ".join(stages),
        file=output,
    )


def filter_stream(
    source: TextIO,
    output: TextIO,
    projection_rms: float,
    activation_rms: float,
    limit: int,
) -> int:
    current: RouteTrace | None = None
    emitted = 0
    for line_number, line in enumerate(source, 1):
        fields = parse_trace(line)
        if fields is None:
            continue
        identity = (int(fields["row"]), int(fields["route"]), int(fields["expert"]))
        if current is None or identity != (current.row, current.route, current.expert):
            if current is not None and is_suspicious(
                current, projection_rms, activation_rms
            ):
                print_trace(current, output)
                emitted += 1
                if limit > 0 and emitted >= limit:
                    return emitted
            current = RouteTrace(
                line=line_number,
                row=identity[0],
                route=identity[1],
                expert=identity[2],
                route_weight=float(fields["route_weight"]),
            )
        current.stages[fields["stage"]] = fields

    if (
        current is not None
        and (limit == 0 or emitted < limit)
        and is_suspicious(current, projection_rms, activation_rms)
    ):
        print_trace(current, output)
        emitted += 1
    return emitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", type=Path, help="trace log; stdin if omitted")
    parser.add_argument(
        "--projection-rms", type=float, default=2.0,
        help="gate/up RMS threshold (default: 2.0)",
    )
    parser.add_argument(
        "--activation-rms", type=float, default=5.0,
        help="activated/down RMS threshold (default: 5.0)",
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="maximum route groups to print; 0 means unlimited (default: 20)",
    )
    args = parser.parse_args()
    if args.limit < 0 or args.projection_rms < 0 or args.activation_rms < 0:
        parser.error("thresholds and limit must be non-negative")

    if args.log is None:
        count = filter_stream(
            sys.stdin, sys.stdout, args.projection_rms,
            args.activation_rms, args.limit,
        )
    else:
        with args.log.open(encoding="utf-8", errors="replace") as source:
            count = filter_stream(
                source, sys.stdout, args.projection_rms,
                args.activation_rms, args.limit,
            )
    if count == 0:
        print("no suspicious routed-expert traces found", file=sys.stderr)


if __name__ == "__main__":
    main()
