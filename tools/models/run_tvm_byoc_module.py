#!/usr/bin/env python3
"""Execute a compiled OpenNPUX BYOC module through coralctl."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from opennpux_tvm_byoc import (
    CodegenError,
    CoralCtlExecutor,
    HostPipelineExecutor,
    ModuleRuntime,
)


def checksum(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def parse_binding(value: str) -> tuple[str, str, Path]:
    endpoint, separator, filename = value.partition("=")
    region, dot, tensor = endpoint.partition(".")
    if not separator or not dot or not region or not tensor or not filename:
        raise argparse.ArgumentTypeError(
            "binding must use REGION.TENSOR=FILE syntax"
        )
    return region, tensor, Path(filename)


def output_filename(endpoint: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", endpoint) + ".bin"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=Path, help="directory containing module.npxgm.json")
    parser.add_argument(
        "--bind",
        action="append",
        default=[],
        type=parse_binding,
        metavar="REGION.TENSOR=FILE",
        help="bind a module input, constant, or state Tensor",
    )
    parser.add_argument("--coralctl", default="coralctl")
    parser.add_argument("--base", type=lambda value: int(value, 0), default=0x1D000000)
    parser.add_argument("--poll-count", type=int, default=1000000)
    parser.add_argument("--transport", choices=("driver", "devmem"), default="driver")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.module / "outputs"
    try:
        device = CoralCtlExecutor(
            args.coralctl,
            base=args.base,
            polls=args.poll_count,
            environment={"OPENNPUX_CORAL_TRANSPORT": args.transport},
        )
        host = HostPipelineExecutor()
        runtime = ModuleRuntime(args.module, device, host_executor=host)
        for region, tensor, filename in args.bind:
            runtime.bind(region, tensor, filename.read_bytes())
        outputs = runtime.run()
        output_dir.mkdir(parents=True, exist_ok=True)
        for log in device.logs:
            print(log, end="" if log.endswith("\n") else "\n")
        for endpoint, data in outputs.items():
            path = output_dir / output_filename(endpoint)
            path.write_bytes(data)
            print(f"xgraph_module_output={endpoint}:{path}")
            print(f"xgraph_module_output_bytes={len(data)}")
            print(f"xgraph_module_output_checksum=0x{checksum(data):08x}")
        print(f"xgraph_module_regions_completed={len(runtime.regions)}")
        print(
            f"xgraph_module_host_bindings_completed={host.completed_bindings}"
        )
        print(
            f"xgraph_module_host_operations_completed={host.completed_operations}"
        )
        print(f"xgraph_module_host_elements={host.completed_elements}")
    except (OSError, ValueError, CodegenError) as error:
        print(f"xgraph_module_run=FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("xgraph_module_run=PASS")


if __name__ == "__main__":
    main()
