"""Equivalent physical-layout experiments using ORT's registered NhwcConv.

No new dependencies; use the pinned research ONNX/NumPy environment. This is
specific to the inspected RVM export, and generates research artifacts only.
"""
from pathlib import Path
import copy
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper
from research_rvm import prune_sort, specialize_metadata


def continuous_layout(source):
    model = copy.deepcopy(source)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in [*model.graph.input, *model.graph.output, *model.graph.value_info]}
    weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    constants = dict(weights)
    for node in model.graph.node:
        if node.op_type == 'Constant':
            constants[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
    spatial = {name for name, s in shapes.items() if len(s) == 4}
    renamed = {name: name + '_nhwc' for name in spatial}
    axis = [0, 3, 1, 2]
    def nhwc(s):
        return [s[i] for i in [0, 2, 3, 1]]
    def constant(name, value):
        model.graph.initializer.append(numpy_helper.from_array(np.array(value, dtype=np.int64), name))
        return name
    nodes = []
    for value in model.graph.input:
        assert value.name in spatial
        nodes.append(helper.make_node('Transpose', [value.name], [renamed[value.name]],
                                      perm=[0, 2, 3, 1], name='layout_input_' + value.name))
    for original in model.graph.node:
        node = copy.deepcopy(original)
        if node.op_type == 'Constant':
            for a in node.attribute:
                if a.type == onnx.AttributeProto.TENSOR:
                    v = numpy_helper.to_array(a.t)
                    if v.ndim == 3 and v.dtype.kind == 'f':
                        a.t.CopyFrom(numpy_helper.from_array(v.transpose(1, 2, 0).copy()))
        if node.op_type == 'Shape':
            constant(node.output[0], shapes[node.input[0]])
            continue
        mapped_inputs = [renamed.get(v, v) for v in node.input]
        mapped_outputs = [renamed.get(v, v) for v in node.output]
        if node.op_type == 'Conv':
            w = node.input[1]
            mapped_inputs[1] = 'layout_' + node.name + '_w'
            model.graph.initializer.append(numpy_helper.from_array(
                weights[w].transpose(0, 2, 3, 1).copy(), mapped_inputs[1]))
            node.op_type = 'NhwcConv'
            node.domain = 'com.microsoft'
        elif node.op_type == 'GlobalAveragePool':
            node.op_type = 'ReduceMean'
            node.attribute.extend([helper.make_attribute('axes', [1, 2]), helper.make_attribute('keepdims', 1)])
        elif node.op_type == 'AveragePool':
            temp = node.name + '_pool_nchw'
            nodes.append(helper.make_node('Transpose', mapped_inputs, [temp + '_x'], perm=[0, 3, 1, 2], name=temp + '_in'))
            node.input[0] = temp + '_x'
            node.output[0] = temp + '_y'
            nodes.append(node)
            nodes.append(helper.make_node('Transpose', [temp + '_y'], mapped_outputs, perm=[0, 2, 3, 1], name=temp + '_out'))
            continue
        elif node.op_type in ['Concat', 'Split', 'ReduceMean'] and node.input[0] in spatial:
            for a in node.attribute:
                if a.name == 'axis':
                    a.i = axis[a.i % 4]
                elif a.name == 'axes':
                    mapped = [axis[v % 4] for v in a.ints]
                    del a.ints[:]
                    a.ints.extend(mapped)
        elif node.op_type == 'Resize':
            # Keep the original scale-based coordinate calculation when present.
            if len(node.input) < 4:
                mapped_inputs[2] = 'layout_' + node.name + '_scales'
                model.graph.initializer.append(numpy_helper.from_array(
                    constants[node.input[2]][[0, 2, 3, 1]].copy(), mapped_inputs[2]))
            else:
                mapped_inputs[3] = constant('layout_' + node.name + '_sizes', nhwc(shapes[node.output[0]]))
        elif node.op_type == 'Slice' and node.input[0] in spatial:
            mapped_inputs[3] = constant('layout_' + node.name + '_axes', [axis[int(v) % 4] for v in constants[node.input[3]]])
        elif node.op_type == 'Expand' and node.input[0] in spatial:
            mapped_inputs[1] = constant('layout_' + node.name + '_shape', nhwc(shapes[node.output[0]]))
        del node.input[:]
        node.input.extend(mapped_inputs)
        del node.output[:]
        node.output.extend(mapped_outputs)
        nodes.append(node)
    for value in model.graph.output:
        assert value.name in spatial
        nodes.append(helper.make_node('Transpose', [renamed[value.name]], [value.name],
                                      perm=[0, 3, 1, 2], name='layout_output_' + value.name))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    info = []
    for value in [*model.graph.value_info, *model.graph.input, *model.graph.output]:
        if value.name in spatial:
            info.append(helper.make_tensor_value_info(renamed[value.name], value.type.tensor_type.elem_type, nhwc(shapes[value.name])))
    del model.graph.value_info[:]
    model.graph.value_info.extend(info)
    model.opset_import.append(helper.make_opsetid('com.microsoft', 1))
    return prune_sort(model)


def channels_last(source, selected):
    model = copy.deepcopy(source)
    weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    nodes = []
    for node in model.graph.node:
        if node.op_type != 'Conv' or not selected(node, weights[node.input[1]]):
            nodes.append(node)
            continue
        x, w = node.input[:2]
        y = node.output[0]
        prefix = 'layout_' + node.name
        model.graph.initializer.append(numpy_helper.from_array(
            weights[w].transpose(0, 2, 3, 1).copy(), prefix + '_w'))
        nodes.append(helper.make_node('Transpose', [x], [prefix + '_x'],
                                      perm=[0, 2, 3, 1], name=prefix + '_in'))
        conv = helper.make_node('NhwcConv', [prefix + '_x', prefix + '_w', *node.input[2:]],
                                [prefix + '_y'], name=node.name, domain='com.microsoft')
        conv.attribute.extend(node.attribute)
        nodes.append(conv)
        nodes.append(helper.make_node('Transpose', [prefix + '_y'], [y],
                                      perm=[0, 3, 1, 2], name=prefix + '_out'))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    if not any(o.domain == 'com.microsoft' for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid('com.microsoft', 1))
    return prune_sort(model)


def main():
    source, shapes, directory = map(Path, sys.argv[1:4])
    model = onnx.load(source)
    specialize_metadata(model, onnx.load(shapes))
    directory.mkdir(parents=True, exist_ok=True)
    onnx.save(model, directory / '00_control.onnx')
    onnx.save(continuous_layout(model), directory / 'continuous.onnx')
    for name, predicate in [
        ('all', lambda n, w: True),
        ('dense', lambda n, w: w.shape[1] > 1),
        ('decoder', lambda n, w: 176 <= int(n.name.split('_')[1]) <= 294),
        ('encoder', lambda n, w: int(n.name.split('_')[1]) < 176),
    ]:
        candidate = channels_last(model, predicate)
        onnx.save(candidate, directory / (name + '.onnx'))
        print(name, sum(n.op_type == 'NhwcConv' for n in candidate.graph.node))


if __name__ == '__main__':
    main()
