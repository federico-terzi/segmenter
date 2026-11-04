"""Weight-only SVD compression probes. No examples, training or downloads.

This is an approximation, not an equivalent graph rewrite. Full recurrent
sequence validation is required even when weight reconstruction error is small.
"""
from pathlib import Path
import copy
import json
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper
from research_rvm import specialize_metadata, prune_sort


def compress(source, selected, fraction):
    model = copy.deepcopy(source)
    weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    result, report = [], []
    for node in model.graph.node:
        if node.name not in selected:
            result.append(node)
            continue
        w = weights[node.input[1]].astype(np.float32)
        outc, inc, kh, kw = w.shape
        u, singular, vt = np.linalg.svd(w.reshape(outc, -1), full_matrices=False)
        rank = min(outc, max(4, int(outc * fraction) // 4 * 4))
        # Symmetric scaling prevents either factor from carrying all magnitude.
        root = np.sqrt(singular[:rank])
        first = (root[:, None] * vt[:rank]).reshape(rank, inc, kh, kw)
        second = (u[:, :rank] * root).reshape(outc, rank, 1, 1)
        p = 'svd_' + node.name
        model.graph.initializer.extend([
            numpy_helper.from_array(first.astype(np.float16), p + '_w1'),
            numpy_helper.from_array(second.astype(np.float16), p + '_w2')])
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        result.append(helper.make_node('Conv', [node.input[0], p + '_w1'], [p], name=p, **attrs))
        result.append(helper.make_node('Conv', [p, p + '_w2', *node.input[2:]], node.output,
                                      name=node.name, kernel_shape=[1, 1]))
        retained = float(np.sum(singular[:rank]**2) / np.sum(singular**2))
        report.append(dict(node=node.name, rank=rank, out_channels=outc, retained_energy=retained,
                           mac_ratio=(rank*inc*kh*kw+outc*rank)/(outc*inc*kh*kw)))
    del model.graph.node[:]
    model.graph.node.extend(result)
    model = prune_sort(model)
    onnx.checker.check_model(model)
    return model, report


def main():
    source, shapes, directory = map(Path, sys.argv[1:4])
    model = onnx.load(source)
    specialize_metadata(model, onnx.load(shapes))
    directory.mkdir(parents=True, exist_ok=True)
    onnx.save(prune_sort(copy.deepcopy(model)), directory/'00_control.onnx')
    decoder = {n.name for n in model.graph.node if n.op_type == 'Conv' and
               176 <= int(n.name.split('_')[-1]) <= 292}
    head = {'Conv_260', 'Conv_290', 'Conv_292'}
    reports = {}
    for name, selected, fraction in [('head_half', head, .5), ('head_threequarters', head, .75),
                                     ('decoder_quarter', decoder, .25), ('decoder_half', decoder, .5)]:
        candidate, reports[name] = compress(model, selected, fraction)
        onnx.save(candidate, directory/(name+'.onnx'))
    (directory/'report.json').write_text(json.dumps(reports, indent=2))
    print('Exported weight-only approximations:', directory)


if __name__ == '__main__':
    main()
