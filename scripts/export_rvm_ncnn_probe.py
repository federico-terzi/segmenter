"""Export the supplied static RVM graph to ncnn's documented parameter format.

This narrow converter preserves the original weights (FP16 values expanded
exactly to FP32 for serialization). It does not load a replacement checkpoint.
Unsupported operations/attributes fail rather than being approximated.
"""
from pathlib import Path
from collections import Counter
import json
import copy
import struct
import sys
import numpy as np
import onnx
from onnx import numpy_helper, helper
from research_rvm import specialize_metadata, prune_sort


def main():
    source, inferred, destination = map(Path, sys.argv[1:4])
    model = onnx.load(source)
    values = specialize_metadata(model, onnx.load(inferred))
    model = prune_sort(model)
    shapes = {name: [d.dim_value for d in v.type.tensor_type.shape.dim] for name, v in values.items()}
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    # Fold shape-only operations using the specialized graph's original shapes.
    remaining = []
    for node in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == 'Constant':
            constants[node.output[0]] = numpy_helper.to_array(attrs['value'])
        elif node.op_type == 'Shape':
            constants[node.output[0]] = np.array(shapes[node.input[0]], dtype=np.int64)
        elif node.op_type == 'Slice' and all(v in constants for v in node.input):
            x = constants[node.input[0]]
            slices = [slice(None)] * x.ndim
            for a, s, e in zip(constants[node.input[3]], constants[node.input[1]], constants[node.input[2]]):
                slices[int(a)] = slice(int(s), int(e))
            constants[node.output[0]] = x[tuple(slices)]
        elif node.op_type == 'Concat' and all(v in constants for v in node.input):
            constants[node.output[0]] = np.concatenate([constants[v] for v in node.input], axis=attrs['axis'])
        else:
            remaining.append(node)
    layers = []
    def add(kind, name, inputs, outputs, params=None, data=b''):
        layers.append(dict(kind=kind, name=name, inputs=list(inputs), outputs=list(outputs), params=params or {}, data=data))
    def raw(value):
        return np.asarray(value, dtype='<f4').tobytes()
    def tensor_constant(name):
        if any(name in layer['outputs'] for layer in layers):
            return
        value = constants[name]
        dims = list(value.shape)
        if len(dims) == 4:
            assert dims.pop(0) == 1
        assert 1 <= len(dims) <= 3
        params = {0: dims[-1]}
        if len(dims) >= 2: params[1] = dims[-2]
        if len(dims) >= 3: params[2] = dims[-3]
        add('MemoryData', 'constant_' + name, [], [name], params, raw(value))
    for v in model.graph.input:
        n, c, h, w = shapes[v.name]
        assert n == 1
        add('Input', 'input_' + v.name, [], [v.name], {0: w, 1: h, 2: c})
    for node in remaining:
        op, ins, outs = node.op_type, list(node.input), list(node.output)
        a = {v.name: helper.get_attribute_value(v) for v in node.attribute}
        data = b''
        params = {}
        if op == 'Conv':
            w = constants[ins[1]]
            oc, ic, kh, kw = w.shape
            group = a.get('group', 1)
            params = {0: oc, 1: kw, 11: kh, 2: a['dilations'][1], 12: a['dilations'][0],
                      3: a['strides'][1], 13: a['strides'][0], 4: a['pads'][1],
                      14: a['pads'][0], 15: a['pads'][3], 16: a['pads'][2],
                      5: int(len(ins) > 2), 6: w.size}
            if group > 1: params[7] = group
            kind = 'ConvolutionDepthWise' if group > 1 else 'Convolution'
            data = struct.pack('<I', 0) + raw(w)
            if len(ins) > 2: data += raw(constants[ins[2]])
            ins = ins[:1]
        elif op in ['Add', 'Sub', 'Mul', 'Div']:
            kind = 'BinaryOp'
            code = {'Add': 0, 'Sub': 1, 'Mul': 2, 'Div': 3}[op]
            if ins[1] in constants and constants[ins[1]].size == 1:
                params = {0: code, 1: 1, 2: float(constants[ins[1]].item())}
                ins = ins[:1]
            elif ins[0] in constants and constants[ins[0]].size == 1:
                params = {0: {0: 0, 1: 7, 2: 2, 3: 8}[code], 1: 1, 2: float(constants[ins[0]].item())}
                ins = ins[1:]
            else:
                params = {0: code}
                for v in ins:
                    if v in constants: tensor_constant(v)
        elif op in ['Relu', 'Sigmoid', 'Tanh']:
            kind = {'Relu': 'ReLU', 'Sigmoid': 'Sigmoid', 'Tanh': 'TanH'}[op]
        elif op == 'HardSigmoid':
            kind = 'HardSigmoid'
            params = {0: a.get('alpha', .2), 1: a.get('beta', .5)}
        elif op == 'Clip':
            kind = 'Clip'
            params = {0: float(constants[ins[1]].item()), 1: float(constants[ins[2]].item())}
            ins = ins[:1]
        elif op == 'Concat':
            kind = 'Concat'
            axis = a['axis'] % 4
            assert axis != 0
            params = {0: axis - 1}
        elif op == 'Split':
            kind = 'Slice'
            axis = a['axis'] % 4
            assert axis != 0
            params = {0: a['split'], 1: axis - 1}
        elif op == 'GlobalAveragePool':
            # ncnn's dedicated pooling is efficient and outputs a channel
            # vector; restore the original keepdims shape for downstream Conv.
            pooled = outs[0] + '_pooled'
            add('Pooling', node.name + '_pool', ins, [pooled], {0: 1, 4: 1})
            kind, ins = 'Reshape', [pooled]
            params = {0: 1, 1: 1, 2: shapes[outs[0]][1]}
        elif op == 'ReduceMean':
            # The generic Vulkan Reduction launches one group per output and
            # exceeds group-count Y on these full-resolution channel reductions.
            # A 1x1 channel projection has the same real-number mean. Keep its
            # coefficients in FP32 rather than rounding 1/3 to FP16.
            assert a['axes'] == [1] and a['keepdims'] == 1
            c = shapes[ins[0]][1]
            kind = 'Convolution'
            params = {0: 1, 1: 1, 5: 0, 6: c, 31: 1}
            data = struct.pack('<I', 0) + raw(np.full(c, 1.0/c))
        elif op == 'AveragePool':
            kind = 'Pooling'
            assert a['pads'] == [0, 0, 0, 0]
            params = {0: 1, 1: a['kernel_shape'][1], 11: a['kernel_shape'][0],
                      2: a['strides'][1], 12: a['strides'][0], 5: 0 if a.get('ceil_mode') else 1,
                      6: a.get('count_include_pad', 0)}
        elif op == 'Pad':
            p = constants[ins[1]].tolist()
            assert len(p) == 8 and p[0] == p[4] == 0 and a.get('mode', b'constant') == b'constant'
            value = float(constants[ins[2]].item()) if len(ins)>2 else 0.0
            params = {0: p[2], 1: p[6], 2: p[3], 3: p[7], 7: p[1], 8: p[5], 4: 0, 5: value}
            kind, ins = 'Padding', ins[:1]
        elif op == 'Expand':
            assert shapes[ins[0]] == shapes[outs[0]], node.name
            kind, ins = 'Split', ins[:1]
        elif op == 'Resize':
            assert a['mode'] == b'linear'
            assert a['coordinate_transformation_mode'] in [b'half_pixel', b'align_corners', b'pytorch_half_pixel']
            _, c, h, w = shapes[outs[0]]
            assert shapes[ins[0]][1] == c
            params = {0: 2, 3: h, 4: w, 6: int(a['coordinate_transformation_mode'] == b'align_corners')}
            kind, ins = 'Interp', ins[:1]
        elif op == 'Slice':
            assert len(shapes[ins[0]]) == 4
            axes = [int(v) % 4 for v in constants[ins[3]]]
            assert 0 not in axes
            params = {9: constants[ins[1]].tolist(), 10: constants[ins[2]].tolist(), 11: [axis - 1 for axis in axes]}
            kind, ins = 'Crop', ins[:1]
        elif op == 'Identity':
            kind, ins = 'Split', ins[:1]
        else:
            raise ValueError((node.name, op))
        add(kind, node.name, ins, outs, params, data)
    # ncnn's light-mode execution requires explicit fanout. Preserve external
    # outputs as a separate branch when they also feed later decoder stages.
    counts = Counter(v for layer in layers for v in layer['inputs'])
    external = {v.name for v in model.graph.output}
    counts.update(external)
    fanout = {name: [name if name in external and i == 0 else f'{name}_use{i}' for i in range(count)]
              for name, count in counts.items() if count > 1}
    used = Counter({name: int(name in external) for name in fanout})
    final = []
    for layer in layers:
        for i, name in enumerate(layer['inputs']):
            if name in fanout:
                layer['inputs'][i] = fanout[name][used[name]]
                used[name] += 1
        original_outputs = list(layer['outputs'])
        layer['outputs'] = [name + '_fanout' if name in fanout else name for name in original_outputs]
        final.append(layer)
        for name in original_outputs:
            if name in fanout:
                final.append(dict(kind='Split', name='fanout_' + name, inputs=[name + '_fanout'], outputs=fanout[name], params={}, data=b''))
    def parameter(key, value):
        if isinstance(value, (list, tuple)):
            return f'{-23300-key}=' + ','.join(map(str, [len(value), *value]))
        # ncnn's text loader distinguishes integer and floating-point literals.
        # Even integral float parameters must contain a decimal point/exponent.
        if isinstance(value, float): return f'{key}={value:.9e}'
        return f'{key}={value}'
    blobs = {name for layer in final for name in layer['outputs']}
    lines = ['7767517', f'{len(final)} {len(blobs)}']
    binary = bytearray()
    for layer in final:
        lines.append(' '.join(map(str, [layer['kind'], layer['name'], len(layer['inputs']), len(layer['outputs']),
                                       *layer['inputs'], *layer['outputs'],
                                       *(parameter(k, v) for k, v in layer['params'].items())])))
        binary.extend(layer['data'])
    destination.mkdir(parents=True, exist_ok=True)
    diagnostic = copy.deepcopy(model)
    available = {v for node in diagnostic.graph.node for v in node.output}
    present = {v.name for v in diagnostic.graph.output}
    for name in ['389','393','397','403','420','464','594','601','611','623','640',
                 '659','676','695','712','731','745','748','751','752','760','761',
                 '765','769','777','779']:
        if name in available and name in values and name not in present:
            diagnostic.graph.output.append(copy.deepcopy(values[name]))
    onnx.checker.check_model(diagnostic)
    onnx.save(diagnostic, destination/'diagnostic.onnx')
    (destination / 'rvm.param').write_text('\n'.join(lines) + '\n')
    (destination / 'rvm.bin').write_bytes(binary)
    (destination / 'shapes.json').write_text(json.dumps({v.name: shapes[v.name] for v in [*model.graph.input, *model.graph.output]}, indent=2))
    print('Exported ncnn', len(final), 'layers', len(blobs), 'blobs', len(binary), 'weight bytes')


if __name__ == '__main__':
    main()
