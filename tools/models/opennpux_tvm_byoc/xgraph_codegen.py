"""Lower a normalized TVM BYOC graph to the OpenNPUX XGraph v2 ABI."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any


FORMAT = "OPENNPUX_TVM_BYOC_GRAPH_V1"
MAGIC = 0x5847504E
VERSION = 2
STATE_READY = 1
HEADER = struct.Struct("<12I2Q8I")
COMMAND = struct.Struct("<16I")
HEADER_SIZE = 96
COMMAND_SIZE = 64
DATA_OFFSET = 0x00020000
MAX_COMMANDS = 768
ALIGNMENT = 64
DTYPE_FP32 = 2

OP_TMMA = 1
OP_TADD = 2
OP_TMUL = 3
OP_TRMSNORM = 4
OP_TSOFTMAX = 5
OP_TROPE = 6
OP_TSILU = 7
OP_TGATHER = 8
OP_TTOPK = 9
OP_TDMA = 11

TMMA_TRANSPOSE_RHS = 1
TTOPK_SPLIT_OUTPUT = 1

DTYPE_BYTES = {
    "float32": 4,
    "int32": 4,
}

OP_ALIASES = {
    "add": "add",
    "relax.add": "add",
    "multiply": "multiply",
    "relax.multiply": "multiply",
    "matmul": "matmul",
    "relax.matmul": "matmul",
    "rms_norm": "rms_norm",
    "relax.nn.rms_norm": "rms_norm",
    "softmax": "softmax",
    "relax.nn.softmax": "softmax",
    "silu": "silu",
    "relax.nn.silu": "silu",
    "take": "take",
    "relax.take": "take",
    "topk": "topk",
    "relax.topk": "topk",
    "rope": "rope",
    "opennpux.rope": "rope",
    "copy": "copy",
    "opennpux.copy": "copy",
}


class CodegenError(ValueError):
    """The BYOC graph cannot be represented by the current XGraph ABI."""


def _u32(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 0xFFFFFFFF:
        raise CodegenError(f"{field} must be a uint32")
    return value


def _align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _product(values: tuple[int, ...]) -> int:
    return math.prod(values) if values else 1


@dataclass
class Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    storage: str
    offset: int
    byte_size: int


@dataclass
class CommandRecord:
    opcode: int
    flags: int
    destination: int
    source0: int
    source1: int
    dim0: int
    dim1: int
    dim2: int
    scalar0: int
    command_id: int
    reserved: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)

    def encode(self) -> bytes:
        fields = (
            self.opcode,
            self.flags,
            self.destination,
            self.source0,
            self.source1,
            self.dim0,
            self.dim1,
            self.dim2,
            self.scalar0,
            DTYPE_FP32,
            self.command_id,
            *self.reserved,
        )
        return COMMAND.pack(*(_u32(value, "command field") for value in fields))

    def as_dict(self) -> dict[str, Any]:
        return {
            "opcode": self.opcode,
            "flags": self.flags,
            "destination_offset": self.destination,
            "source0_offset": self.source0,
            "source1_offset": self.source1,
            "dim0": self.dim0,
            "dim1": self.dim1,
            "dim2": self.dim2,
            "scalar0": self.scalar0,
            "data_type": DTYPE_FP32,
            "command_id": self.command_id,
            "reserved": list(self.reserved),
        }


def _parse_tensors(records: Any, data_offset: int) -> dict[str, Tensor]:
    if not isinstance(records, list) or not records:
        raise CodegenError("tensors must be a non-empty array")
    tensors: dict[str, Tensor] = {}
    cursor = _align(data_offset)
    occupied: list[tuple[int, int, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise CodegenError(f"tensor {index} must be an object")
        name = record.get("name")
        if not isinstance(name, str) or not name or name in tensors:
            raise CodegenError(f"tensor {index} has an invalid or duplicate name")
        raw_shape = record.get("shape")
        if not isinstance(raw_shape, list) or not raw_shape:
            raise CodegenError(f"tensor {name} must have a static, non-scalar shape")
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in raw_shape):
            raise CodegenError(f"tensor {name} has a dynamic or invalid shape")
        shape = tuple(raw_shape)
        dtype = record.get("dtype", "float32")
        if dtype not in DTYPE_BYTES:
            raise CodegenError(f"tensor {name} has unsupported dtype {dtype!r}")
        storage = record.get("storage", "scratch")
        if storage not in {"input", "output", "constant", "scratch", "state"}:
            raise CodegenError(f"tensor {name} has unsupported storage {storage!r}")
        byte_size = _product(shape) * DTYPE_BYTES[dtype]
        requested = record.get("offset")
        offset = cursor if requested is None else _u32(requested, f"tensor {name} offset")
        if offset < data_offset or offset % ALIGNMENT != 0:
            raise CodegenError(
                f"tensor {name} offset must be {ALIGNMENT}-byte aligned and >= 0x{data_offset:x}"
            )
        end = offset + byte_size
        if end > 0x100000000:
            raise CodegenError(f"tensor {name} exceeds the uint32 address space")
        for prior_start, prior_end, prior_name in occupied:
            if offset < prior_end and prior_start < end:
                raise CodegenError(f"tensor {name} overlaps tensor {prior_name}")
        occupied.append((offset, end, name))
        cursor = max(cursor, _align(end))
        tensors[name] = Tensor(name, shape, dtype, storage, offset, byte_size)
    return tensors


def _tensor(tensors: dict[str, Tensor], name: Any, node_index: int) -> Tensor:
    if not isinstance(name, str) or name not in tensors:
        raise CodegenError(f"node {node_index} references unknown tensor {name!r}")
    return tensors[name]


def _matrix_shape(tensor: Tensor, label: str) -> tuple[int, int]:
    if tensor.dtype != "float32" or len(tensor.shape) < 2:
        raise CodegenError(f"{label} must be a rank-2-or-higher float32 tensor")
    return _product(tensor.shape[:-1]), tensor.shape[-1]


def _elementwise_shape(tensor: Tensor) -> tuple[int, int]:
    if tensor.dtype != "float32":
        raise CodegenError(f"tensor {tensor.name} must use float32")
    return _product(tensor.shape[:-1]), tensor.shape[-1]


def _expect_count(values: Any, count: int, label: str, node_index: int) -> list[str]:
    if not isinstance(values, list) or len(values) != count:
        raise CodegenError(f"node {node_index} {label} must contain {count} tensors")
    return values


def _lower_node(
    node: dict[str, Any], tensors: dict[str, Tensor], command_id: int, node_index: int
) -> CommandRecord:
    raw_op = node.get("op")
    op = OP_ALIASES.get(raw_op)
    if op is None:
        raise CodegenError(f"node {node_index} has unsupported BYOC op {raw_op!r}")
    attrs = node.get("attrs", {})
    if not isinstance(attrs, dict):
        raise CodegenError(f"node {node_index} attrs must be an object")
    inputs = node.get("inputs")
    outputs = node.get("outputs")

    if op == "matmul":
        names = _expect_count(inputs, 2, "inputs", node_index)
        output_names = _expect_count(outputs, 1, "outputs", node_index)
        lhs = _tensor(tensors, names[0], node_index)
        rhs = _tensor(tensors, names[1], node_index)
        output = _tensor(tensors, output_names[0], node_index)
        rows, k = _matrix_shape(lhs, "matmul lhs")
        if rhs.dtype != "float32" or output.dtype != "float32" or len(rhs.shape) != 2:
            raise CodegenError("matmul rhs must be a rank-2 float32 tensor")
        transpose_rhs = bool(attrs.get("transpose_rhs", False))
        rhs_k, n = (rhs.shape[1], rhs.shape[0]) if transpose_rhs else rhs.shape
        if rhs_k != k or output.shape != lhs.shape[:-1] + (n,):
            raise CodegenError("matmul input/output shapes are inconsistent")
        if max(rows, n, k) > 1023:
            raise CodegenError("matmul exceeds one TMMA command; tiled lowering is required")
        return CommandRecord(
            OP_TMMA,
            TMMA_TRANSPOSE_RHS if transpose_rhs else 0,
            output.offset,
            lhs.offset,
            rhs.offset,
            rows,
            n,
            k,
            0,
            command_id,
        )

    if op in {"add", "multiply"}:
        names = _expect_count(inputs, 2, "inputs", node_index)
        output_names = _expect_count(outputs, 1, "outputs", node_index)
        lhs = _tensor(tensors, names[0], node_index)
        rhs = _tensor(tensors, names[1], node_index)
        output = _tensor(tensors, output_names[0], node_index)
        if (
            lhs.shape != rhs.shape
            or lhs.shape != output.shape
            or lhs.dtype != "float32"
            or rhs.dtype != "float32"
            or output.dtype != "float32"
        ):
            raise CodegenError(f"{op} currently requires equal, non-broadcast shapes")
        rows, features = _elementwise_shape(output)
        return CommandRecord(
            OP_TADD if op == "add" else OP_TMUL,
            0,
            output.offset,
            lhs.offset,
            rhs.offset,
            rows,
            features,
            1,
            0,
            command_id,
        )

    if op == "rms_norm":
        names = _expect_count(inputs, 2, "inputs", node_index)
        output_names = _expect_count(outputs, 1, "outputs", node_index)
        source = _tensor(tensors, names[0], node_index)
        weight = _tensor(tensors, names[1], node_index)
        output = _tensor(tensors, output_names[0], node_index)
        if (
            source.shape != output.shape
            or weight.shape != (source.shape[-1],)
            or source.dtype != "float32"
            or weight.dtype != "float32"
            or output.dtype != "float32"
        ):
            raise CodegenError("rms_norm expects input/output [...,K] and weight [K]")
        rows, features = _elementwise_shape(output)
        epsilon = attrs.get("epsilon", 1.0e-5)
        if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
            raise CodegenError("rms_norm epsilon must be positive and finite")
        epsilon_bits = struct.unpack("<I", struct.pack("<f", float(epsilon)))[0]
        return CommandRecord(
            OP_TRMSNORM,
            0,
            output.offset,
            source.offset,
            weight.offset,
            rows,
            features,
            1,
            epsilon_bits,
            command_id,
        )

    if op in {"softmax", "silu"}:
        names = _expect_count(inputs, 1, "inputs", node_index)
        output_names = _expect_count(outputs, 1, "outputs", node_index)
        source = _tensor(tensors, names[0], node_index)
        output = _tensor(tensors, output_names[0], node_index)
        if (
            source.shape != output.shape
            or source.dtype != "float32"
            or output.dtype != "float32"
        ):
            raise CodegenError(f"{op} input/output shapes must match")
        if op == "softmax" and int(attrs.get("axis", -1)) not in {-1, len(source.shape) - 1}:
            raise CodegenError("softmax currently requires the innermost axis")
        rows, features = _elementwise_shape(output)
        return CommandRecord(
            OP_TSOFTMAX if op == "softmax" else OP_TSILU,
            0,
            output.offset,
            source.offset,
            0,
            rows,
            features,
            1,
            0,
            command_id,
        )

    if op == "take":
        names = _expect_count(inputs, 2, "inputs", node_index)
        output_names = _expect_count(outputs, 1, "outputs", node_index)
        data = _tensor(tensors, names[0], node_index)
        indices = _tensor(tensors, names[1], node_index)
        output = _tensor(tensors, output_names[0], node_index)
        if (
            data.dtype != "float32"
            or len(data.shape) != 2
            or indices.dtype != "int32"
            or output.dtype != "float32"
        ):
            raise CodegenError("take expects float32 [vocabulary,K] data and int32 indices")
        if int(attrs.get("axis", 0)) != 0 or output.shape != indices.shape + (data.shape[1],):
            raise CodegenError("take supports embedding gather on axis 0 only")
        return CommandRecord(
            OP_TGATHER,
            0,
            output.offset,
            data.offset,
            indices.offset,
            _product(indices.shape),
            data.shape[1],
            1,
            data.shape[1],
            command_id,
        )

    if op == "topk":
        names = _expect_count(inputs, 1, "inputs", node_index)
        output_names = _expect_count(outputs, 2, "outputs", node_index)
        source = _tensor(tensors, names[0], node_index)
        values = _tensor(tensors, output_names[0], node_index)
        indices = _tensor(tensors, output_names[1], node_index)
        rows, features = _elementwise_shape(source)
        k = attrs.get("k")
        expected = source.shape[:-1] + (k,) if isinstance(k, int) else ()
        if not isinstance(k, int) or k <= 0 or k > features:
            raise CodegenError("topk k must be a positive static integer")
        if values.dtype != "float32" or indices.dtype != "int32":
            raise CodegenError("topk outputs must be float32 values and int32 indices")
        if values.shape != expected or indices.shape != expected:
            raise CodegenError("topk output shapes are inconsistent")
        return CommandRecord(
            OP_TTOPK,
            TTOPK_SPLIT_OUTPUT,
            values.offset,
            source.offset,
            0,
            rows,
            features,
            1,
            k,
            command_id,
            (indices.offset, 0, 0, 0, 0),
        )

    if op == "rope":
        names = _expect_count(inputs, 2, "inputs", node_index)
        output_names = _expect_count(outputs, 1, "outputs", node_index)
        source = _tensor(tensors, names[0], node_index)
        table = _tensor(tensors, names[1], node_index)
        output = _tensor(tensors, output_names[0], node_index)
        if source.shape != output.shape or source.dtype != "float32" or table.dtype != "float32":
            raise CodegenError("rope expects matching float32 input/output and a float32 table")
        rows, features = _elementwise_shape(output)
        layout = attrs.get("layout", "adjacent")
        if layout not in {"adjacent", "half_split"}:
            raise CodegenError("rope layout must be adjacent or half_split")
        return CommandRecord(
            OP_TROPE,
            0,
            output.offset,
            source.offset,
            table.offset,
            rows,
            features,
            1,
            int(layout == "half_split"),
            command_id,
        )

    names = _expect_count(inputs, 1, "inputs", node_index)
    output_names = _expect_count(outputs, 1, "outputs", node_index)
    source = _tensor(tensors, names[0], node_index)
    output = _tensor(tensors, output_names[0], node_index)
    if source.shape != output.shape or source.dtype != output.dtype:
        raise CodegenError("copy input/output tensor types must match")
    return CommandRecord(
        OP_TDMA,
        0,
        output.offset,
        source.offset,
        0,
        source.byte_size // DTYPE_BYTES[source.dtype],
        1,
        1,
        source.byte_size,
        command_id,
    )


def compile_graph(graph: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """Compile a normalized BYOC graph and return binary plus inspectable metadata."""
    if not isinstance(graph, dict) or graph.get("format") != FORMAT:
        raise CodegenError(f"graph format must be {FORMAT}")
    data_offset = _u32(graph.get("data_offset", DATA_OFFSET), "data_offset")
    if data_offset < DATA_OFFSET or data_offset % ALIGNMENT != 0:
        raise CodegenError("data_offset must be 64-byte aligned and not overlap commands")
    tensors = _parse_tensors(graph.get("tensors"), data_offset)
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise CodegenError("nodes must be a non-empty array")
    if len(nodes) > MAX_COMMANDS:
        raise CodegenError(f"graph exceeds the {MAX_COMMANDS}-command XGraph batch limit")
    commands = []
    available = {
        name
        for name, tensor in tensors.items()
        if tensor.storage in {"input", "constant", "state"}
    }
    produced: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise CodegenError(f"node {index} must be an object")
        inputs = node.get("inputs", [])
        outputs = node.get("outputs", [])
        if not isinstance(inputs, list) or any(name not in available for name in inputs):
            raise CodegenError(f"node {index} consumes a tensor before it is available")
        if not isinstance(outputs, list) or any(name in available for name in outputs):
            raise CodegenError(f"node {index} writes a tensor more than once")
        commands.append(_lower_node(node, tensors, index, index))
        produced.update(outputs)
        available.update(outputs)
    output_names = graph.get("outputs")
    if not isinstance(output_names, list) or len(output_names) != 1:
        raise CodegenError("the first-stage artifact requires exactly one graph output")
    output = _tensor(tensors, output_names[0], len(nodes))
    if output.name not in produced or output.storage != "output":
        raise CodegenError("graph output must be produced and use output storage")
    total_size = HEADER_SIZE + len(commands) * COMMAND_SIZE
    reserved = (0, 0, len(nodes), 1, 0, 0, 0, 0)
    header = HEADER.pack(
        MAGIC,
        VERSION,
        HEADER_SIZE,
        COMMAND_SIZE,
        len(commands),
        total_size,
        STATE_READY,
        0,
        0,
        output.offset,
        output.byte_size,
        0,
        0,
        0,
        *reserved,
    )
    binary = header + b"".join(command.encode() for command in commands)
    metadata = {
        "format": FORMAT,
        "xgraph_version": VERSION,
        "header_size": HEADER_SIZE,
        "command_size": COMMAND_SIZE,
        "command_count": len(commands),
        "binary_size": len(binary),
        "arena_size": _align(
            max(tensor.offset + tensor.byte_size for tensor in tensors.values())
        ),
        "output": output.name,
        "tensors": [
            {
                "name": tensor.name,
                "shape": list(tensor.shape),
                "dtype": tensor.dtype,
                "storage": tensor.storage,
                "offset": tensor.offset,
                "byte_size": tensor.byte_size,
            }
            for tensor in tensors.values()
        ],
        "commands": [command.as_dict() for command in commands],
    }
    return binary, metadata
