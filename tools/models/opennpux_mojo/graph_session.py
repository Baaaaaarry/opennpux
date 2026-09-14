"""Construct a MAX Graph and its OpenNPUX export as one frontend session."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Callable, Mapping, Sequence

from .export_builder import MaxExportBuilder


class MaxGraphExportSession:
    """Own a public MAX Graph context and record the same graph for OpenNPUX.

    The class intentionally accepts a graph factory instead of importing MAX.
    Production callers pass ``max.graph.Graph`` while unit tests and future
    Mojo bindings can supply any object with the same public context contract.
    """

    def __init__(
        self,
        graph_factory: Callable[..., Any],
        name: str,
        *,
        input_types: Sequence[Any],
        input_names: Sequence[str],
        input_storage: Sequence[str] | None = None,
        graph_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if len(input_types) != len(input_names):
            raise ValueError("MAX input type and name counts differ")
        storage = list(input_storage or ["input"] * len(input_names))
        if len(storage) != len(input_names):
            raise ValueError("MAX input storage and name counts differ")
        self.builder = MaxExportBuilder(name)
        self.graph = graph_factory(
            name, input_types=input_types, **dict(graph_kwargs or {})
        )
        self._input_names = list(input_names)
        self._input_storage = storage
        self._context: Any = None

    def __enter__(self) -> "MaxGraphExportSession":
        self._context = self.graph.__enter__()
        graph = self._context if self._context is not None else self.graph
        inputs = list(graph.inputs)
        if len(inputs) != len(self._input_names):
            raise ValueError("MAX Graph input count differs from declared inputs")
        for name, storage, value in zip(
            self._input_names, self._input_storage, inputs
        ):
            self.builder.add_tensor(
                name, getattr(value, "tensor", value), storage=storage
            )
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self.graph.__exit__(exception_type, exception, traceback)

    @property
    def inputs(self) -> Sequence[Any]:
        return tuple(getattr(value, "tensor", value) for value in self.graph.inputs)

    def declare_symbol(self, name: str, minimum: int, maximum: int) -> None:
        self.builder.declare_symbol(name, minimum, maximum)

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
        return self.builder.call(
            op,
            function,
            inputs,
            output_names,
            name=name,
            attrs=attrs,
            kwargs=kwargs,
        )

    def output(self, *values: Any) -> None:
        self.graph.output(*values)
        for value in values:
            self.builder.add_output(value)

    def to_opennpux_export(self) -> dict[str, Any]:
        return self.builder.to_opennpux_export()


__all__ = ["MaxGraphExportSession"]
