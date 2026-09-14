"""Record public MAX Graph values and calls into the stable export protocol."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Callable, Mapping, Sequence

from .graph_adapter import MAX_GRAPH_EXPORT_FORMAT


_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DTYPES = {
    "float32": "float32",
    "fp32": "float32",
    "int32": "int32",
    "sint32": "int32",
}


class MaxExportBuilder:
    """Sidecar recorder used while constructing a public MAX Graph."""

    def __init__(self, name: str = "main") -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("MAX export graph name must be non-empty")
        self.name = name
        self._tensors: list[dict[str, Any]] = []
        self._operations: list[dict[str, Any]] = []
        self._outputs: list[str] = []
        self._symbols: dict[str, dict[str, int]] = {}
        self._value_names: dict[int, str] = {}
        self._names: set[str] = set()

    def declare_symbol(self, name: str, minimum: int, maximum: int) -> None:
        if not _SYMBOL.fullmatch(name):
            raise ValueError(f"invalid MAX shape symbol {name!r}")
        if minimum <= 0 or maximum < minimum:
            raise ValueError(f"invalid bounds for MAX shape symbol {name}")
        constraint = {"min": int(minimum), "max": int(maximum)}
        previous = self._symbols.get(name)
        if previous is not None and previous != constraint:
            raise ValueError(f"conflicting bounds for MAX shape symbol {name}")
        self._symbols[name] = constraint

    def add_tensor(
        self,
        name: str,
        value: Any,
        *,
        storage: str = "scratch",
        source: str | None = None,
    ) -> str:
        if not isinstance(name, str) or not name or name in self._names:
            raise ValueError(f"duplicate or invalid MAX Tensor name {name!r}")
        shape = self._shape(value)
        dtype = self._dtype(value)
        record: dict[str, Any] = {
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "storage": storage,
        }
        if source is not None:
            record["source"] = source
        self._tensors.append(record)
        self._names.add(name)
        self._value_names[id(value)] = name
        return name

    def add_operation(
        self,
        op: str,
        inputs: Sequence[Any],
        outputs: Sequence[Any],
        *,
        name: str | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "op": op,
            "inputs": [self._reference(value) for value in inputs],
            "outputs": [self._reference(value) for value in outputs],
        }
        if name is not None:
            record["name"] = name
        if attrs:
            record["attrs"] = deepcopy(dict(attrs))
        self._operations.append(record)

    def call(
        self,
        op: str,
        function: Callable[..., Any],
        inputs: Sequence[Any],
        output_names: Sequence[str],
        *,
        name: str | None = None,
        attrs: Mapping[str, Any] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Invoke one MAX op and record its public TensorValue results."""
        result = function(*inputs, **dict(kwargs or {}))
        values = list(result) if isinstance(result, (list, tuple)) else [result]
        if len(values) != len(output_names):
            raise ValueError(f"MAX operation {op} output count mismatch")
        for output_name, value in zip(output_names, values):
            self.add_tensor(output_name, value)
        self.add_operation(op, inputs, values, name=name, attrs=attrs)
        return result

    def add_output(self, value: Any) -> None:
        name = self._reference(value)
        for tensor in self._tensors:
            if tensor["name"] == name and tensor["storage"] == "scratch":
                tensor["storage"] = "output"
                break
        if name not in self._outputs:
            self._outputs.append(name)

    def to_opennpux_export(self) -> dict[str, Any]:
        if not self._outputs:
            raise ValueError("MAX export requires at least one graph output")
        result: dict[str, Any] = {
            "format": MAX_GRAPH_EXPORT_FORMAT,
            "kind": "graph",
            "name": self.name,
            "tensors": deepcopy(self._tensors),
            "operations": deepcopy(self._operations),
            "outputs": list(self._outputs),
        }
        if self._symbols:
            result["shape_symbols"] = deepcopy(self._symbols)
        return result

    def _reference(self, value: Any) -> str:
        if isinstance(value, str) and value in self._names:
            return value
        name = self._value_names.get(id(value))
        if name is None:
            raise ValueError("MAX value must be registered before it is referenced")
        return name

    def _shape(self, value: Any) -> list[int | str]:
        shape = getattr(value, "shape", None)
        if shape is None:
            shape = getattr(getattr(value, "type", None), "shape", None)
        if shape is None:
            raise ValueError("MAX TensorValue/BufferValue has no public shape")
        result = []
        for dimension in shape:
            raw = getattr(dimension, "value", dimension)
            if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
                result.append(raw)
                continue
            symbol = raw if isinstance(raw, str) else str(raw)
            if not _SYMBOL.fullmatch(symbol) or symbol not in self._symbols:
                raise ValueError(
                    f"MAX dynamic dimension {symbol!r} requires declared bounds"
                )
            result.append(symbol)
        return result

    @staticmethod
    def _dtype(value: Any) -> str:
        dtype = getattr(value, "dtype", None)
        if dtype is None:
            dtype = getattr(getattr(value, "type", None), "dtype", None)
        spelling = str(dtype).rsplit(".", 1)[-1].lower()
        normalized = _DTYPES.get(spelling)
        if normalized is None:
            raise ValueError(f"unsupported MAX Tensor dtype {dtype!r}")
        return normalized


__all__ = ["MaxExportBuilder"]
