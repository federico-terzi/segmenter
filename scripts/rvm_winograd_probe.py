"""Research F(2x2,3x3) convolution factorization; no training or downloads.

Uses exact Winograd transforms over real numbers, but FP16 intermediates can
change numerical output. Every candidate requires full-sequence quality checks.
"""
from pathlib import Path
import copy
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper
from research_rvm import specialize_metadata, prune_sort, fragment

BT = np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=np.float32)
G = np.array([[1, 0, 0], [.5, .5, .5], [.5, -.5, .5], [0, 0, 1]], dtype=np.float32)
AT = np.array([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=np.float32)


def convert(source, values, chosen, fp32=False):
    model = copy.deepcopy(source)
    weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    nodes = []
    for node in model.graph.node:
        if node.name not in chosen:
            nodes.append(node)
            continue
        w = weights[node.input[1]].astype(np.float32)
        outc, inc, kh, kw = w.shape
        assert (kh, kw) == (3, 3)
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        assert attrs.get('group', 1) == 1 and attrs['pads'] == [1, 1, 1, 1]
        assert attrs['strides'] == [1, 1] and attrs['dilations'] == [1, 1]
        dims = [d.dim_value for d in values[node.input[0]].type.tensor_type.shape.dim]
        assert dims[0] == 1 and dims[1] == inc
        _, _, h, width = dims
        th, tw = (h + 1)//2, (width + 1)//2
        p = 'winograd_' + node.name
        dtype = np.float32 if fp32 else np.float16
        def init(suffix, array, typ=None):
            name = p + '_' + suffix
            model.graph.initializer.append(numpy_helper.from_array(np.array(array, dtype=typ or dtype), name))
            return name
        def emit(op, inputs, suffix, **attrs):
            output = p + '_' + suffix
            nodes.append(helper.make_node(op, inputs, [output], name=output, **attrs))
            return output
        x = node.input[0]
        if fp32:
            x = emit('Cast', [x], 'cast_input', to=onnx.TensorProto.FLOAT)
        transform = np.einsum('ai,bj->abij', BT, BT).reshape(16, 1, 4, 4)
        transform = np.tile(transform, (inc, 1, 1, 1))
        x = emit('Conv', [x, init('bt', transform)], 'input_transform',
                 kernel_shape=[4, 4], strides=[2, 2], group=inc,
                 pads=[1, 1, 1+(h % 2), 1+(width % 2)])
        x = emit('Reshape', [x, init('shape_in', [1, inc, 16, th*tw], np.int64)], 'input_reshape')
        x = emit('Transpose', [x], 'input_transpose', perm=[0, 2, 1, 3])
        u = np.einsum('ai,ocij,bj->aboc', G, w, G).reshape(1, 16, outc, inc)
        x = emit('MatMul', [init('u', u), x], 'products')
        x = emit('Transpose', [x], 'output_transpose', perm=[0, 2, 1, 3])
        x = emit('Reshape', [x, init('shape_out', [1, outc*16, th, tw], np.int64)], 'output_reshape')
        inverse = np.einsum('ia,jb->abij', AT, AT).reshape(16, 1, 2, 2)
        inverse = np.tile(inverse, (outc, 1, 1, 1))
        args = [x, init('at', inverse)]
        if len(node.input) > 2:
            args.append(init('bias', weights[node.input[2]]))
        x = emit('ConvTranspose', args, 'output_transform', strides=[2, 2], group=outc, kernel_shape=[2, 2])
        if h % 2 or width % 2:
            x = emit('Slice', [x, init('starts', [0, 0], np.int64), init('ends', [h, width], np.int64),
                              init('axes', [2, 3], np.int64)], 'crop')
        nodes.append(helper.make_node('Cast' if fp32 else 'Identity', [x], node.output,
                                      name=node.name, **({'to': onnx.TensorProto.FLOAT16} if fp32 else {})))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return prune_sort(model)


def verify_transform():
    # Algebra check independent of ONNX's operators and GPU implementation.
    rng = np.random.default_rng(7)
    patch = rng.normal(size=(3, 4, 4))
    kernel = rng.normal(size=(2, 3, 3, 3))
    v = np.einsum('ai,cij,bj->cab', BT, patch, BT)
    u = np.einsum('ai,ocij,bj->ocab', G, kernel, G)
    actual = np.einsum('ia,ocab,cab,jb->oij', AT, u, v, AT)
    expected = np.empty((2, 2, 2))
    for i in range(2):
        for j in range(2):
            expected[:, i, j] = np.einsum('cij,ocij->o', patch[:, i:i+3, j:j+3], kernel)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def main():
    verify_transform()
    source, shapes, directory = map(Path, sys.argv[1:4])
    model = onnx.load(source)
    values = specialize_metadata(model, onnx.load(shapes))
    directory.mkdir(parents=True, exist_ok=True)
    high = {'Conv_260', 'Conv_266', 'Conv_271', 'Conv_290', 'Conv_292'}
    all_dense = {n.name for n in model.graph.node if n.op_type == 'Conv' and 176 <= int(n.name.split('_')[1]) <= 292}
    onnx.save(model, directory / '00_control.onnx')
    for name, chosen, fp32 in [('head', {'Conv_290', 'Conv_292'}, False),
                               ('high', high, False), ('decoder', all_dense, False),
                               ('head_fp32', {'Conv_290', 'Conv_292'}, True)]:
        candidate = convert(model, values, chosen, fp32)
        onnx.save(candidate, directory / (name + '.onnx'))
    isolated = fragment(model, values, 287, 289, ['748'])
    for fp32 in [False, True]:
        onnx.save(convert(isolated, values, {'Conv_290'}, fp32), directory.parent / ('rvm-winograd-conv32.onnx' if fp32 else 'rvm-winograd-conv16.onnx'))
    print('Winograd algebra verified; exported', directory)


if __name__ == '__main__':
    main()
