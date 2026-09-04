"""Model-independent scheduler for compiled OpenNPUX BYOC regions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable

from .module_codegen import MODULE_FORMAT
from .xgraph_codegen import CodegenError


RegionExecutor = Callable[[str, bytes, bytearray, dict[str, Any]], None]


class CoralCtlExecutor:
    """Execute one region through coralctl and import its verified output."""

    def __init__(
        self,
        coralctl: Path | str,
        base: int = 0x1D000000,
        polls: int = 1000000,
        environment: dict[str, str] | None = None,
    ):
        self.coralctl = str(coralctl)
        self.base = base
        self.polls = polls
        self.environment = dict(environment or {})
        self.logs: list[str] = []

    def __call__(
        self, name: str, artifact: bytes, arena: bytearray, metadata: dict[str, Any]
    ) -> None:
        output_name = metadata.get("output")
        tensors = {
            tensor.get("name"): tensor for tensor in metadata.get("tensors", [])
        }
        if output_name not in tensors:
            raise CodegenError(f"region {name} output metadata is missing")
        output = tensors[output_name]
        with tempfile.TemporaryDirectory(prefix="opennpux-region-") as temporary:
            directory = Path(temporary)
            graph_path = directory / "region.npxg"
            arena_path = directory / "region.arena.bin"
            output_path = directory / "region.output.bin"
            graph_path.write_bytes(artifact)
            arena_path.write_bytes(arena)
            environment = os.environ.copy()
            environment.update(self.environment)
            environment["OPENNPUX_XGRAPH_OUTPUT_PATH"] = str(output_path)
            completed = subprocess.run(
                [
                    self.coralctl,
                    "xgraph-run",
                    str(graph_path),
                    str(arena_path),
                    hex(self.base),
                    str(self.polls),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.logs.append(completed.stdout + completed.stderr)
            if completed.returncode != 0:
                raise CodegenError(
                    f"region {name} Coral execution failed with status {completed.returncode}"
                )
            if "xgraph_output_readback=PASS" not in completed.stdout:
                raise CodegenError(f"region {name} Coral output was not verified")
            if not output_path.is_file():
                raise CodegenError(f"region {name} Coral output file is missing")
            data = output_path.read_bytes()
            if len(data) != output.get("byte_size"):
                raise CodegenError(f"region {name} Coral output size mismatch")
            offset = output["offset"]
            arena[offset : offset + len(data)] = data


class ModuleRuntime:
    """Bind invocation data and execute XGraph regions in manifest order."""

    def __init__(self, directory: Path | str, executor: RegionExecutor):
        self.directory = Path(directory)
        self.executor = executor
        self.manifest = self._load_json(self.directory / "module.npxgm.json")
        if self.manifest.get("format") != MODULE_FORMAT:
            raise CodegenError("invalid XGraph module manifest format")
        self.regions: dict[str, dict[str, Any]] = {}
        self.arenas: dict[str, bytearray] = {}
        self.bound: set[tuple[str, str]] = set()
        for expected_sequence, region in enumerate(self.manifest.get("regions", [])):
            if not isinstance(region, dict) or region.get("sequence") != expected_sequence:
                raise CodegenError("module regions are not in canonical sequence order")
            name = region.get("name")
            artifact_name = region.get("artifact")
            if not isinstance(name, str) or not isinstance(artifact_name, str):
                raise CodegenError("module region record is invalid")
            metadata = self._load_json(self.directory / f"{artifact_name}.json")
            artifact = (self.directory / artifact_name).read_bytes()
            if metadata.get("command_count") != region.get("command_count"):
                raise CodegenError(f"region {name} command count mismatch")
            arena_size = metadata.get("arena_size")
            if not isinstance(arena_size, int) or arena_size != region.get("arena_size"):
                raise CodegenError(f"region {name} arena size mismatch")
            tensors = metadata.get("tensors")
            if not isinstance(tensors, list):
                raise CodegenError(f"region {name} tensor metadata is missing")
            tensor_table = {tensor.get("name"): tensor for tensor in tensors}
            if len(tensor_table) != len(tensors) or None in tensor_table:
                raise CodegenError(f"region {name} tensor metadata is invalid")
            self.regions[name] = {
                "record": region,
                "metadata": metadata,
                "artifact": artifact,
                "tensors": tensor_table,
            }
            self.arenas[name] = bytearray(arena_size)
        order = self.manifest.get("execution_order")
        if order != list(self.regions):
            raise CodegenError("module execution order does not match region records")

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise CodegenError(f"{path} must contain an object")
        return value

    def _tensor(self, region: str, tensor: str) -> dict[str, Any]:
        if region not in self.regions or tensor not in self.regions[region]["tensors"]:
            raise CodegenError(f"unknown module Tensor {region}.{tensor}")
        return self.regions[region]["tensors"][tensor]

    def bind(self, region: str, tensor: str, data: bytes) -> None:
        record = self._tensor(region, tensor)
        external = self.regions[region]["record"].get("external_bindings", [])
        if tensor not in external:
            raise CodegenError(f"Tensor {region}.{tensor} is not an external binding")
        if len(data) != record.get("byte_size"):
            raise CodegenError(f"Tensor {region}.{tensor} binding size mismatch")
        offset = record["offset"]
        self.arenas[region][offset : offset + len(data)] = data
        self.bound.add((region, tensor))

    def run(self) -> dict[str, bytes]:
        executed: set[str] = set()
        incoming: dict[str, list[dict[str, Any]]] = {
            name: [] for name in self.regions
        }
        for edge in self.manifest.get("edges", []):
            incoming[edge["to_region"]].append(edge)
        for name in self.manifest["execution_order"]:
            record = self.regions[name]["record"]
            missing = [
                tensor for tensor in record.get("external_bindings", [])
                if (name, tensor) not in self.bound
            ]
            if missing:
                raise CodegenError(
                    f"region {name} has unbound external Tensors: {', '.join(missing)}"
                )
            for edge in incoming[name]:
                source_region = edge["from_region"]
                if source_region not in executed:
                    raise CodegenError(f"region {name} depends on an unexecuted producer")
                source = self._tensor(source_region, edge["from_tensor"])
                target = self._tensor(name, edge["to_tensor"])
                size = edge["bytes"]
                source_offset = source["offset"]
                target_offset = target["offset"]
                self.arenas[name][target_offset : target_offset + size] = self.arenas[
                    source_region
                ][source_offset : source_offset + size]
            region = self.regions[name]
            self.executor(name, region["artifact"], self.arenas[name], region["metadata"])
            executed.add(name)

        outputs = {}
        for endpoint in self.manifest.get("module_outputs", []):
            region = endpoint["region"]
            tensor_name = endpoint["tensor"]
            tensor = self._tensor(region, tensor_name)
            offset = tensor["offset"]
            outputs[f"{region}.{tensor_name}"] = bytes(
                self.arenas[region][offset : offset + tensor["byte_size"]]
            )
        return outputs
