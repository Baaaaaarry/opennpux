"""Write compiled OpenNPUX backend artifacts without frontend dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .xgraph_codegen import CodegenError


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_graph_artifact(
    output: Path | str,
    artifact: bytes,
    metadata: dict[str, Any],
    metadata_path: Path | str | None = None,
) -> Path:
    """Write one `.npxg` and its metadata sidecar."""
    output = Path(output)
    sidecar = Path(metadata_path) if metadata_path is not None else Path(f"{output}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(artifact)
    _write_json(sidecar, metadata)
    return sidecar


def write_module_artifacts(
    output: Path | str,
    artifacts: dict[str, tuple[bytes, dict[str, Any]]],
    manifest: dict[str, Any],
) -> Path:
    """Write all region artifacts and one `.npxgm` manifest."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for region in manifest.get("regions", []):
        name = region.get("name") if isinstance(region, dict) else None
        artifact_name = region.get("artifact") if isinstance(region, dict) else None
        if not isinstance(name, str) or not isinstance(artifact_name, str):
            raise CodegenError("module manifest contains an invalid region record")
        if name not in artifacts:
            raise CodegenError(f"compiled artifact for region {name} is missing")
        binary, metadata = artifacts[name]
        write_graph_artifact(output / artifact_name, binary, metadata)
    manifest_path = output / "module.npxgm.json"
    _write_json(manifest_path, manifest)
    return manifest_path


__all__ = ["write_graph_artifact", "write_module_artifacts"]
