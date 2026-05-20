#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper


MAGIC = b"RVMMETAL"
VERSION = 2

OP = {
    "Conv": 1,
    "Relu": 2,
    "Sigmoid": 3,
    "Tanh": 4,
    "HardSigmoid": 5,
    "Add": 6,
    "Sub": 7,
    "Mul": 8,
    "Div": 9,
    "AveragePool": 10,
    "GlobalAveragePool": 11,
    "Resize": 12,
    "Concat": 13,
    "Split": 14,
    "Slice": 15,
    "Shape": 16,
    "Expand": 17,
    "Constant": 18,
    "Clip": 19,
    "ReduceMean": 20,
}

DTYPE_F32 = 1
DTYPE_F16 = 2
DTYPE_I64 = 3

HEADER_FIELDS = 16
RVM_INPUTS = ("src", "downsample_ratio", "r1i", "r2i", "r3i", "r4i")
RVM_OUTPUTS = ("pha", "r1o", "r2o", "r3o", "r4o")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Robust Video Matting ONNX graph to segmenter's experimental .rvmmetal format."
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--dtype",
        choices=("f16", "f32"),
        default="f16",
        help="Floating-point dtype used for exported float initializers.",
    )
    args = parser.parse_args()

    model = onnx.shape_inference.infer_shapes(onnx.load(args.model))
    graph = model.graph
    validate_rvm_io(graph)

    shapes = collect_shapes(graph)
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    value_ids: dict[str, int] = {}
    tensor_ids: dict[str, int] = {}
    tensors: list[tuple[int, int, tuple[int, tuple[int, int, int, int]], bytes]] = []
    nodes = []
    next_value_id = 0
    next_tensor_id = 1
    export_float_dtype = np.float16 if args.dtype == "f16" else np.float32
    export_dtype_id = DTYPE_F16 if args.dtype == "f16" else DTYPE_F32

    def value_id(name: str) -> int:
        nonlocal next_value_id
        if not name:
            return 0xFFFF_FFFF
        if name not in value_ids:
            value_ids[name] = next_value_id
            next_value_id += 1
        return value_ids[name]

    def add_tensor(name: str, array: np.ndarray) -> int:
        nonlocal next_tensor_id
        if name in tensor_ids:
            return tensor_ids[name]
        tensor_id = next_tensor_id
        next_tensor_id += 1
        dtype, rank_shape, data = encode_tensor(array, export_float_dtype, export_dtype_id)
        tensors.append((tensor_id, dtype, rank_shape, data))
        tensor_ids[name] = tensor_id
        return tensor_id

    for node_index, node in enumerate(graph.node):
        if node.op_type not in OP:
            raise SystemExit(f"unsupported op {node.op_type!r} at node {node_index} ({node.name})")
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        tensor_refs: list[int] = []
        inputs: list[int] = []
        if node.op_type == "Conv":
            if node.input[0] in initializers:
                raise SystemExit(f"Conv data input was unexpectedly constant at node {node_index}")
            inputs.append(value_id(node.input[0]))
            weight = initializers[node.input[1]]
            tensor_refs.append(add_tensor(node.input[1], weight))
            if len(node.input) > 2 and node.input[2] in initializers:
                bias = initializers[node.input[2]]
                if bias.ndim == 1:
                    bias = bias.reshape(1, bias.shape[0], 1, 1)
                tensor_refs.append(add_tensor(f"{node.input[2]}__nchw_bias", bias))
        else:
            for name in node.input:
                if name in initializers:
                    tensor_refs.append(add_tensor(name, initializers[name]))
                else:
                    inputs.append(value_id(name))

        if node.op_type == "Constant":
            constant = constant_value(attrs, node.name or f"Constant_{node_index}")
            tensor_refs.append(add_tensor(f"__constant_{node_index}", constant))

        node_attrs = encode_attrs(node.op_type, attrs, node)
        output_names = list(node.output) or [f"__node_{node_index}"]
        nodes.append(
            (
                OP[node.op_type],
                [value_id(name) for name in output_names],
                inputs,
                tensor_refs,
                node_attrs,
                [rank_and_shape4(shapes.get(name, [])) for name in output_names],
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        f.write(MAGIC)
        write_u32(f, VERSION)
        header = [
            value_id("src"),
            value_id("downsample_ratio"),
            value_id("pha"),
            value_id("r1i"),
            value_id("r2i"),
            value_id("r3i"),
            value_id("r4i"),
            value_id("r1o"),
            value_id("r2o"),
            value_id("r3o"),
            value_id("r4o"),
            export_dtype_id,
            0,
            0,
            0,
            0,
        ]
        if len(header) != HEADER_FIELDS:
            raise AssertionError("header field count drifted")
        for value in header:
            write_u32(f, value)

        write_u32(f, len(tensors))
        for tensor_id, dtype, rank_shape, data in tensors:
            rank, shape = rank_shape
            write_u32(f, tensor_id)
            write_u32(f, dtype)
            write_u32(f, rank)
            write_shape(f, shape)
            write_u64(f, len(data))
            f.write(data)

        write_u32(f, len(nodes))
        for op, outputs, inputs, tensor_refs, attrs, output_shapes in nodes:
            write_u32(f, op)
            write_u32(f, len(outputs))
            for output, (rank, shape) in zip(outputs, output_shapes):
                write_u32(f, output)
                write_u32(f, rank)
                write_shape(f, shape)
            write_u32(f, len(inputs))
            for value in inputs:
                write_u32(f, value)
            write_u32(f, len(tensor_refs))
            for value in tensor_refs:
                write_u32(f, value)
            write_u32(f, len(attrs))
            for value in attrs:
                write_i64(f, value)

    total_weights = sum(len(data) for _, _, _, data in tensors)
    print(
        f"wrote {args.out} dtype={args.dtype} "
        f"nodes={len(nodes)} tensors={len(tensors)} bytes={total_weights}"
    )


def validate_rvm_io(graph: onnx.GraphProto) -> None:
    inputs = {value.name for value in graph.input}
    outputs = {value.name for value in graph.output}
    missing_inputs = [name for name in RVM_INPUTS if name not in inputs]
    missing_outputs = [name for name in RVM_OUTPUTS if name not in outputs]
    if missing_inputs or missing_outputs:
        raise SystemExit(
            "model does not look like exported RVM; "
            f"missing inputs={missing_inputs}, missing outputs={missing_outputs}"
        )


def collect_shapes(graph: onnx.GraphProto) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for value in list(graph.input) + list(graph.value_info) + list(graph.output):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        for dim in tensor_type.shape.dim:
            dims.append(int(dim.dim_value) if dim.HasField("dim_value") else 0)
        shapes[value.name] = dims
    return shapes


def encode_tensor(
    array: np.ndarray,
    export_float_dtype: Any,
    export_dtype_id: int,
) -> tuple[int, tuple[int, tuple[int, int, int, int]], bytes]:
    if array.dtype.kind == "f":
        return (
            export_dtype_id,
            rank_and_shape4(array.shape),
            np.ascontiguousarray(array.astype(export_float_dtype)).tobytes(order="C"),
        )
    if array.dtype == np.int64:
        return (
            DTYPE_I64,
            rank_and_shape4(array.shape),
            np.ascontiguousarray(array.astype(np.int64)).tobytes(order="C"),
        )
    if array.dtype.kind in {"i", "u", "b"}:
        return (
            DTYPE_I64,
            rank_and_shape4(array.shape),
            np.ascontiguousarray(array.astype(np.int64)).tobytes(order="C"),
        )
    raise SystemExit(f"unsupported initializer dtype {array.dtype}")


def encode_attrs(op_type: str, attrs: dict[str, Any], node: onnx.NodeProto) -> list[int]:
    if op_type == "Conv":
        pads = list(attrs.get("pads", [0, 0, 0, 0]))
        strides = list(attrs.get("strides", [1, 1]))
        dilations = list(attrs.get("dilations", [1, 1]))
        group = int(attrs.get("group", 1))
        return [
            int(strides[1]),
            int(strides[0]),
            int(pads[1]),
            int(pads[0]),
            int(dilations[1]),
            int(dilations[0]),
            group,
        ]
    if op_type == "AveragePool":
        kernel = list(attrs.get("kernel_shape", [1, 1]))
        strides = list(attrs.get("strides", kernel))
        pads = list(attrs.get("pads", [0, 0, 0, 0]))
        return [
            int(kernel[-1]),
            int(kernel[-2]),
            int(strides[-1]),
            int(strides[-2]),
            int(pads[1]) if len(pads) > 1 else 0,
            int(pads[3]) if len(pads) > 3 else 0,
            int(pads[0]) if pads else 0,
            int(pads[2]) if len(pads) > 2 else 0,
            int(attrs.get("ceil_mode", 0)),
        ]
    if op_type == "ReduceMean":
        axes = attrs.get("axes", [])
        return [
            int(attrs.get("keepdims", 1)),
            *[int(axis) for axis in axes],
        ]
    if op_type == "Concat":
        return [int(attrs.get("axis", 0))]
    if op_type == "Split":
        return [int(attrs.get("axis", 0)), *[int(size) for size in attrs.get("split", [])]]
    if op_type == "HardSigmoid":
        return [float_to_i32(attrs.get("alpha", 0.2)), float_to_i32(attrs.get("beta", 0.5))]
    if op_type == "Resize":
        mode = attrs.get("mode", b"nearest")
        ctm = attrs.get("coordinate_transformation_mode", b"")
        if mode not in {b"linear", b"nearest"}:
            raise SystemExit(f"unsupported Resize mode {mode!r} in {node.name}")
        return [1 if mode == b"linear" else 0, stable_string_id(ctm)]
    return []


def constant_value(attrs: dict[str, Any], label: str) -> np.ndarray:
    if "value" in attrs:
        return numpy_helper.to_array(attrs["value"])
    if "value_float" in attrs:
        return np.array([attrs["value_float"]], dtype=np.float32)
    if "value_int" in attrs:
        return np.array([attrs["value_int"]], dtype=np.int64)
    if "value_floats" in attrs:
        return np.asarray(attrs["value_floats"], dtype=np.float32)
    if "value_ints" in attrs:
        return np.asarray(attrs["value_ints"], dtype=np.int64)
    raise SystemExit(f"Constant node {label!r} did not contain a supported value attribute")


def rank_and_shape4(shape: Any) -> tuple[int, tuple[int, int, int, int]]:
    dims = [int(v) for v in shape]
    if len(dims) > 4:
        raise SystemExit(f"expected tensor rank <= 4, got {dims}")
    return len(dims), tuple((dims + [1, 1, 1, 1])[:4])


def float_to_i32(value: float) -> int:
    return struct.unpack("<i", struct.pack("<f", float(value)))[0]


def stable_string_id(value: bytes | str) -> int:
    if isinstance(value, str):
        value = value.encode()
    result = 2166136261
    for byte in value:
        result ^= byte
        result = (result * 16777619) & 0xFFFF_FFFF
    return result


def write_u32(f, value: int) -> None:
    f.write(struct.pack("<I", int(value) & 0xFFFF_FFFF))


def write_u64(f, value: int) -> None:
    f.write(struct.pack("<Q", int(value)))


def write_i64(f, value: int) -> None:
    f.write(struct.pack("<q", int(value)))


def write_shape(f, shape: tuple[int, int, int, int]) -> None:
    for value in shape:
        write_u32(f, value)


if __name__ == "__main__":
    main()
