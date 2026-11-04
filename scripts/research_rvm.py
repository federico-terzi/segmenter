"""Offline graph experiments for the supplied RVM MobileNetV3 export.

Requires onnx 1.18.0 and numpy. Generates separate research artifacts; never
overwrites the source model. No download, training, or CUDA inference occurs.
See docs/rvm-performance.md for the shape-export and timing procedure.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def specialize_metadata(model, inferred):
    """Use CPU shape propagation, without retaining its FP32 graph rewrites."""
    inferred = onnx.shape_inference.infer_shapes(inferred)
    shapes = {}
    for v in [*inferred.graph.value_info, *inferred.graph.input, *inferred.graph.output]:
        shape = [d.dim_value for d in v.type.tensor_type.shape.dim]
        if all(d > 0 for d in shape):
            shapes[v.name.removeprefix("InsertedPrecisionFreeCast_")] = shape
    original = onnx.shape_inference.infer_shapes(model)
    types = {v.name: v.type.tensor_type.elem_type for v in [
        *original.graph.value_info, *original.graph.input, *original.graph.output
    ]}
    values = {
        name: helper.make_tensor_value_info(name, types[name], shape)
        for name, shape in shapes.items() if name in types
    }
    del model.graph.value_info[:]
    model.graph.value_info.extend(values.values())
    for v in [*model.graph.input, *model.graph.output]:
        v.CopyFrom(values[v.name])
    return values


def fragment(model, values, first, last, outputs=None):
    nodes = list(model.graph.node[first:last])
    produced = {v for node in nodes for v in node.output}
    used_later = {v for n in model.graph.node[last:] for v in n.input}
    used_later.update(v.name for v in model.graph.output)
    outputs = outputs or sorted(produced & used_later)
    constants = {n.output[0]: n for n in model.graph.node if n.op_type == "Constant"}
    initializers = {v.name: v for v in model.graph.initializer}
    by_output = {v: n for n in nodes for v in n.output}
    needed_nodes, needed_initializers, inputs = {}, {}, set()

    def visit(name):
        if not name:
            return
        if name in initializers:
            needed_initializers[name] = initializers[name]
        elif name in by_output or name in constants:
            n = by_output.get(name, constants.get(name))
            if n.name not in needed_nodes:
                for source in n.input:
                    visit(source)
                needed_nodes[n.name] = n
        else:
            inputs.add(name)

    for name in outputs:
        visit(name)
    graph = helper.make_graph(list(needed_nodes.values()), "rvm-research",
                              [values[n] for n in sorted(inputs)],
                              [values[n] for n in outputs],
                              list(needed_initializers.values()))
    result = helper.make_model(graph, opset_imports=model.opset_import, ir_version=model.ir_version)
    onnx.checker.check_model(result)
    return result


def macs(model, values):
    weights = {v.name: v for v in model.graph.initializer}
    total = 0
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        shape = [d.dim_value for d in values[node.output[0]].type.tensor_type.shape.dim]
        weight = weights[node.input[1]]
        total += int(np.prod(shape)) * int(np.prod(weight.dims[1:]))
    return total


def pad_channels(model, alignment):
    """Append zeros to unaligned Conv inputs and matching weight channels."""
    result = copy.deepcopy(model)
    weights = {v.name: v for v in result.graph.initializer}
    nodes = []
    changed = []
    for node in result.graph.node:
        if node.op_type == "Conv":
            groups = next((a.i for a in node.attribute if a.name == "group"), 1)
            weight = weights[node.input[1]]
            channels = weight.dims[1]
            extra = (-channels) % alignment
            if groups == 1 and extra and channels > 3:
                prefix = "research_pad_" + node.name
                result.graph.initializer.extend([
                    numpy_helper.from_array(np.array([0, 0, 0, 0, 0, extra, 0, 0], dtype=np.int64), prefix + "_pads"),
                ])
                nodes.append(helper.make_node("Pad", [node.input[0], prefix + "_pads"], [prefix], name=prefix))
                node.input[0] = prefix
                array = numpy_helper.to_array(weight)
                padded = np.pad(array, ((0, 0), (0, extra), (0, 0), (0, 0)))
                # New weights avoid changing another consumer of an initializer.
                weight_name = prefix + "_weights"
                result.graph.initializer.append(numpy_helper.from_array(padded, weight_name))
                node.input[1] = weight_name
                changed.append((node.name, channels, channels + extra))
        nodes.append(node)
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    del result.graph.value_info[:]
    onnx.checker.check_model(result)
    print("channel padding", alignment, changed)
    return result


def alpha_refiner(model):
    """Compute only the needed output channel AFTER all mixed-channel layers."""
    result = copy.deepcopy(model)
    by_name = {n.name: n for n in result.graph.node}
    weights = {v.name: v for v in result.graph.initializer}
    conv = by_name["Conv_316"]
    for name in conv.input[1:]:
        weights[name].CopyFrom(numpy_helper.from_array(numpy_helper.to_array(weights[name])[3:4].copy(), name))
    extra = [
        helper.make_node("Slice", ["760", "research_start", "research_end", "research_axis"], ["research_mean_x"], name="research_mean_x"),
        helper.make_node("Slice", ["761", "research_start", "research_end", "research_axis"], ["research_mean_y"], name="research_mean_y"),
    ]
    for name, value in [("research_start", 3), ("research_end", 4), ("research_axis", 1)]:
        result.graph.initializer.append(numpy_helper.from_array(np.array([value], dtype=np.int64), name))
    by_name["Mul_317"].input[1] = "research_mean_x"
    by_name["Sub_318"].input[0] = "research_mean_y"
    by_name["Mul_347"].input[1] = "755"
    by_name["Split_349"].CopyFrom(helper.make_node("Identity", ["809"], ["811"], name="research_alpha"))
    result.graph.node.extend(extra)
    return prune_sort(result)


def combine_box_filters(model):
    result = copy.deepcopy(model)
    result.graph.initializer.append(numpy_helper.from_array(
        np.tile(numpy_helper.to_array(next(v for v in result.graph.initializer if v.name == "refiner.box_filter.weight")), (4, 1, 1, 1)),
        "research_boxes_weight"))
    nodes = [n for n in result.graph.node if n.name not in ["Conv_301", "Conv_302", "Conv_304", "Conv_308"]]
    nodes += [
        helper.make_node("Concat", ["758", "759", "762", "766"], ["research_boxes_input"], axis=1, name="research_boxes_input"),
        helper.make_node("Conv", ["research_boxes_input", "research_boxes_weight"], ["research_boxes"], group=16, kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="research_boxes"),
        helper.make_node("Split", ["research_boxes"], ["760", "761", "763", "767"], axis=1, split=[4, 4, 4, 4], name="research_boxes_split"),
    ]
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    return prune_sort(result)


def pointwise_matmul(model):
    result = copy.deepcopy(model)
    weights = {v.name: v for v in result.graph.initializer}
    nodes = []
    for node in result.graph.node:
        if node.op_type == "Conv" and list(weights[node.input[1]].dims[2:]) == [1, 1]:
            assert next((a.i for a in node.attribute if a.name == "group"), 1) == 1
            prefix = "research_mm_" + node.name
            weight = numpy_helper.to_array(weights[node.input[1]])[:, :, 0, 0].T.copy()
            result.graph.initializer.append(numpy_helper.from_array(weight, prefix + "_w"))
            nodes += [
                helper.make_node("Transpose", [node.input[0]], [prefix + "_x"], perm=[0, 2, 3, 1], name=prefix + "_x"),
                helper.make_node("MatMul", [prefix + "_x", prefix + "_w"], [prefix + "_y"], name=prefix + "_y"),
            ]
            value = prefix + "_y"
            if len(node.input) == 3:
                nodes.append(helper.make_node("Add", [value, node.input[2]], [prefix + "_b"], name=prefix + "_b"))
                value = prefix + "_b"
            nodes.append(helper.make_node("Transpose", [value], node.output, perm=[0, 3, 1, 2], name=prefix + "_out"))
        else:
            nodes.append(node)
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    return prune_sort(result)


def prune_sort(model):
    """Remove unused branches, then topologically order a rewritten graph."""
    producers = {v: n for n in model.graph.node for v in n.output}
    initializers = {v.name: v for v in model.graph.initializer}
    nodes, weights = {}, {}

    def visit(name):
        if name in initializers:
            weights[name] = initializers[name]
        elif name in producers and producers[name].name not in nodes:
            node = producers[name]
            for source in node.input:
                visit(source)
            nodes[node.name] = node
    for output in model.graph.output:
        visit(output.name)
    del model.graph.node[:]
    model.graph.node.extend(nodes.values())
    del model.graph.initializer[:]
    model.graph.initializer.extend(weights.values())
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    return model


def unroll(model, count):
    """Exact sequential recurrence across multiple input frames in one graph.

    Each frame still has its complete network evaluation. Weights are shared.
    This experiment measures throughput with a larger submission, not live
    single-frame latency, and does not approximate or skip recurrent updates.
    """
    result = copy.deepcopy(model)
    del result.graph.node[:]
    weights = {v.name for v in model.graph.initializer}
    nodes = []
    for frame in range(count):
        prefix = f"frame{frame}_"
        def mapped(name):
            if not name or name in weights:
                return name
            if name in ["r1i", "r2i", "r3i", "r4i"]:
                return name if frame == 0 else f"frame{frame-1}_" + name.replace("i", "o")
            return prefix + name
        for suffix, value in [("start", frame), ("end", frame + 1), ("axis", 0)]:
            result.graph.initializer.append(numpy_helper.from_array(np.array([value], dtype=np.int64), prefix + suffix))
        nodes.append(helper.make_node("Slice", ["src", prefix + "start", prefix + "end", prefix + "axis"], [prefix + "src"], name=prefix + "src_slice"))
        for node in model.graph.node:
            n = copy.deepcopy(node)
            n.name = prefix + n.name
            for i, name in enumerate(n.input):
                n.input[i] = mapped(name)
            for i, name in enumerate(n.output):
                n.output[i] = mapped(name)
            nodes.append(n)
    nodes.append(helper.make_node("Concat", [f"frame{i}_pha" for i in range(count)], ["pha"], axis=0, name="sequence_alpha"))
    for name in ["r1o", "r2o", "r3o", "r4o"]:
        nodes.append(helper.make_node("Identity", [f"frame{count-1}_" + name], [name], name="sequence_" + name))
    result.graph.node.extend(nodes)
    for v in [*result.graph.input, *result.graph.output]:
        if v.name in ["src", "pha"]:
            v.type.tensor_type.shape.dim[0].dim_value = count
    return prune_sort(result)


def student_probe(model, rank, drop_residuals):
    """LOSSY, UNTRAINED architecture probe, not a deployable optimization.

    Approximate each decoder 3x3 Conv with per-input-channel spatial SVD and
    pointwise mixing. Recurrent tensor dimensions and frame order are retained.
    This gives a concrete latency screen before spending time on distillation.
    """
    result = copy.deepcopy(model)
    weights = {v.name: v for v in result.graph.initializer}
    nodes = []
    for node in result.graph.node:
        if node.name in drop_residuals:
            nodes.append(helper.make_node("Identity", [node.input[1]], node.output, name="student_skip_" + node.name))
            continue
        if node.op_type != "Conv" or not 176 <= int(node.name.split("_")[1]) <= 292:
            nodes.append(node)
            continue
        weight = numpy_helper.to_array(weights[node.input[1]]).astype(np.float32)
        outputs, inputs, kh, kw = weight.shape
        assert (kh, kw) == (3, 3)
        spatial = np.zeros((inputs * rank, 1, 3, 3), dtype=np.float16)
        pointwise = np.zeros((outputs, inputs * rank, 1, 1), dtype=np.float16)
        for channel in range(inputs):
            u, s, vh = np.linalg.svd(weight[:, channel].reshape(outputs, 9), full_matrices=False)
            spatial[channel*rank:(channel+1)*rank, 0] = vh[:rank].reshape(rank, 3, 3)
            pointwise[:, channel*rank:(channel+1)*rank, 0, 0] = u[:, :rank] * s[:rank]
        prefix = "student_" + node.name
        result.graph.initializer.extend([
            numpy_helper.from_array(spatial, prefix + "_dw"),
            numpy_helper.from_array(pointwise, prefix + "_pw"),
        ])
        nodes.extend([
            helper.make_node("Conv", [node.input[0], prefix + "_dw"], [prefix + "_features"],
                             group=inputs, kernel_shape=[3, 3], pads=[1, 1, 1, 1], name=prefix + "_spatial"),
            helper.make_node("Conv", [prefix + "_features", prefix + "_pw", *node.input[2:]], node.output,
                             kernel_shape=[1, 1], name=prefix + "_mix"),
        ])
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    return prune_sort(result)


def narrow_student(model, values, fraction):
    """UNTRAINED width probe with matching gates, skips, and recurrent states.

    Selects channels deterministically, not by learned importance. Never use
    the resulting alpha as a quality-preserving substitute without training.
    """
    result = copy.deepcopy(model)
    weights = {v.name: v for v in result.graph.initializer}
    producers = {v: n for n in result.graph.node for v in n.output}

    def channels(name):
        if name not in values:
            return None
        dims = values[name].type.tensor_type.shape.dim
        return dims[1].dim_value if len(dims) == 4 else None

    def reduced(count):
        return min(count, max(8, int(np.ceil(count * fraction / 8)) * 8))

    forced = {}
    def force(name, mask):
        if channels(name) is None:
            return
        if name in forced:
            assert forced[name] == mask, (name, forced[name], mask)
            return
        forced[name] = mask
        node = producers.get(name)
        if node and node.op_type in ["Mul", "Relu", "Sigmoid", "HardSigmoid", "Identity"]:
            for source in node.input:
                if channels(source) == channels(name):
                    force(source, mask)

    for node in result.graph.node:
        if node.op_type == "Split":
            sizes = next(list(a.ints) for a in node.attribute if a.name == "split")
            offset, mask = 0, []
            for size in sizes:
                mask.extend(range(offset, offset + reduced(size)))
                offset += size
            force(node.input[0], mask)
    masks = {v.name: list(range(channels(v.name) if v.name == "src" else reduced(channels(v.name)))) for v in result.graph.input}
    for v in result.graph.input:
        v.type.tensor_type.shape.dim[1].dim_value = len(masks[v.name])
    for node in result.graph.node:
        if node.op_type == "Conv":
            weight = numpy_helper.to_array(weights[node.input[1]])
            input_mask = masks[node.input[0]]
            group = next((a.i for a in node.attribute if a.name == "group"), 1)
            if group > 1:
                assert group == weight.shape[0] and weight.shape[1] == 1
                output_mask = input_mask
                new_weight = weight[output_mask]
                next(a for a in node.attribute if a.name == "group").i = len(input_mask)
            else:
                output_mask = forced.get(node.output[0], list(range(reduced(weight.shape[0]))))
                new_weight = weight[output_mask][:, input_mask]
            prefix = "narrow_" + node.name
            result.graph.initializer.append(numpy_helper.from_array(new_weight.copy(), prefix + "_w"))
            node.input[1] = prefix + "_w"
            if len(node.input) == 3:
                bias = numpy_helper.to_array(weights[node.input[2]])[output_mask].copy()
                result.graph.initializer.append(numpy_helper.from_array(bias, prefix + "_b"))
                node.input[2] = prefix + "_b"
            masks[node.output[0]] = output_mask
        elif node.op_type == "Split":
            attribute = next(a for a in node.attribute if a.name == "split")
            offset, sizes = 0, list(attribute.ints)
            for i, (name, size) in enumerate(zip(node.output, sizes)):
                masks[name] = [c - offset for c in masks[node.input[0]] if offset <= c < offset + size]
                attribute.ints[i] = len(masks[name])
                offset += size
        elif channels(node.output[0]) is not None:
            if node.op_type == "Concat":
                mask, offset = [], 0
                for source in node.input:
                    mask.extend(c + offset for c in masks[source])
                    offset += channels(source)
            elif node.op_type == "ReduceMean":
                mask = [0]
            else:
                matching = [masks[s] for s in node.input if s in masks and channels(s) == channels(node.output[0])]
                mask = matching[0] if matching else list(range(channels(node.output[0])))
                assert all(m == mask for m in matching), (node.name, matching)
            masks[node.output[0]] = mask
    for v in result.graph.output:
        v.type.tensor_type.shape.dim[1].dim_value = len(masks[v.name])
    return prune_sort(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("shape_model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = onnx.load(args.model)
    assert model.graph.node[169].name == "Split_172", "unexpected export; inspect before using these cuts"
    assert model.graph.node[293].name == "ReduceMean_296"
    values = specialize_metadata(model, onnx.load(args.shape_model))
    args.output.mkdir(parents=True, exist_ok=True)
    stages = [
        ("00_full", 0, 350, ["pha", "r1o", "r2o", "r3o", "r4o"]),
        ("01_encoder", 0, 169, None),
        ("02_bottleneck", 169, 186, None),
        ("03_decode3", 186, 216, None),
        ("04_decode2", 216, 246, None),
        ("05_decode1", 246, 276, None),
        ("06_output_head", 276, 293, None),
        ("07_refiner", 293, 350, ["pha"]),
        ("08_all_decoder", 169, 293, None),
    ]
    report = []
    for name, first, last, outputs in stages:
        block = fragment(model, values, first, last, outputs)
        onnx.save(block, args.output / (name + ".onnx"))
        row = dict(name=name, nodes=len(block.graph.node), conv_macs=macs(block, values),
                   inputs=[v.name for v in block.graph.input], outputs=[v.name for v in block.graph.output])
        report.append(row)
        print(json.dumps(row))
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2))
    diagnostics = args.output.parent / "rvm-refiner-diagnostics"
    diagnostics.mkdir(exist_ok=True)
    for name, first, last in [("01_statistics", 293, 309), ("02_mlp", 309, 314), ("03_apply", 314, 350)]:
        onnx.save(fragment(model, values, first, last, ["pha"] if last == 350 else None), diagnostics / (name + ".onnx"))
    variants = args.output.parent / "rvm-variants"
    variants.mkdir(exist_ok=True)
    full = fragment(model, values, 0, 350, ["pha", "r1o", "r2o", "r3o", "r4o"])
    onnx.save(full, variants / "00_unchanged.onnx")
    for alignment in [8, 16, 32]:
        onnx.save(pad_channels(full, alignment), variants / f"pad{alignment}.onnx")
    onnx.save(alpha_refiner(full), variants / "alpha_refiner.onnx")
    onnx.save(combine_box_filters(full), variants / "combined_boxes.onnx")
    onnx.save(combine_box_filters(alpha_refiner(full)), variants / "alpha_combined_boxes.onnx")
    onnx.save(pointwise_matmul(full), variants / "pointwise_matmul.onnx")
    identity_resizes = copy.deepcopy(full)
    for node in identity_resizes.graph.node:
        if node.name in ["Resize_332", "Resize_346"]:
            assert [d.dim_value for d in values[node.input[0]].type.tensor_type.shape.dim] == [d.dim_value for d in values[node.output[0]].type.tensor_type.shape.dim]
            node.CopyFrom(helper.make_node("Identity", [node.input[0]], node.output, name=node.name))
    onnx.save(prune_sort(identity_resizes), variants / "identity_refiner_resizes.onnx")
    slice_alpha = copy.deepcopy(full)
    for name, value in [("slice_start", 3), ("slice_end", 4), ("slice_axis", 1)]:
        slice_alpha.graph.initializer.append(numpy_helper.from_array(np.array([value], dtype=np.int64), name))
    for node in slice_alpha.graph.node:
        if node.name == "Split_349":
            node.CopyFrom(helper.make_node("Slice", ["809", "slice_start", "slice_end", "slice_axis"], ["811"], name="slice_alpha"))
    onnx.save(prune_sort(slice_alpha), variants / "slice_alpha.onnx")
    gather_alpha = copy.deepcopy(slice_alpha)
    node = next(n for n in gather_alpha.graph.node if n.name == "slice_alpha")
    node.CopyFrom(helper.make_node("Gather", ["809", "slice_start"], ["811"], axis=1, name="gather_alpha"))
    onnx.save(prune_sort(gather_alpha), variants / "gather_alpha.onnx")
    fp32 = copy.deepcopy(full)
    for tensor in fp32.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT16:
            tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float32), tensor.name))
    for node in fp32.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR and attribute.t.data_type == onnx.TensorProto.FLOAT16:
                attribute.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(attribute.t).astype(np.float32)))
    for value in [*fp32.graph.input, *fp32.graph.output, *fp32.graph.value_info]:
        if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    onnx.save(fp32, variants / "fp32.onnx")
    fp32.graph.initializer.append(numpy_helper.from_array(np.array([3], dtype=np.int64), "gather_index"))
    node = next(n for n in fp32.graph.node if n.name == "Split_349")
    node.CopyFrom(helper.make_node("Gather", ["809", "gather_index"], ["811"], axis=1, name="gather_alpha"))
    onnx.save(prune_sort(fp32), variants / "fp32_gather.onnx")
    batches = args.output.parent / "rvm-batches"
    batches.mkdir(exist_ok=True)
    encoder = fragment(model, values, 0, 169)
    for count in [1, 2, 4, 8]:
        batched = copy.deepcopy(encoder)
        for v in [*batched.graph.input, *batched.graph.output]:
            v.type.tensor_type.shape.dim[0].dim_value = count
        onnx.save(batched, batches / f"encoder_batch{count}.onnx")
        onnx.save(unroll(full, count), batches / f"sequence{count}.onnx")
    students = args.output.parent / "rvm-student-probes"
    students.mkdir(exist_ok=True)
    residuals = {n.name for n in model.graph.node[:166] if n.op_type == "Add"}
    for name, rank, dropped in [
        ("decoder_rank1", 1, set()),
        ("student_mild", 2, {"Add_48", "Add_83", "Add_118", "Add_159"}),
        ("student_fast", 1, residuals),
    ]:
        candidate = student_probe(full, rank, dropped)
        onnx.save(candidate, students / (name + ".onnx"))
        print("UNTRAINED STUDENT", name, "parameters", sum(int(np.prod(t.dims)) for t in candidate.graph.initializer))
    for fraction in [0.25, 0.5, 0.75]:
        candidate = narrow_student(full, values, fraction)
        onnx.save(candidate, students / f"narrow{fraction}.onnx")
        print("UNTRAINED NARROW STUDENT", fraction, "parameters", sum(int(np.prod(t.dims)) for t in candidate.graph.initializer))
        if fraction == 0.25:
            for node in candidate.graph.node:
                if node.name in residuals:
                    node.CopyFrom(helper.make_node("Identity", [node.input[1]], node.output, name="narrow_skip_" + node.name))
            onnx.save(prune_sort(candidate), students / "narrow0.25_short.onnx")


if __name__ == "__main__":
    main()
