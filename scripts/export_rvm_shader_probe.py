"""Export actual RVM refiner weights and intermediate outputs for DirectX probes.
Uses the already selected research ONNX/NumPy versions; no downloads.
"""
from pathlib import Path
import copy
import sys
import numpy as np
import onnx
from onnx import numpy_helper, helper
from research_rvm import specialize_metadata, prune_sort, fragment

source, shapes, output = map(Path, sys.argv[1:4])
model = onnx.load(source)
values = specialize_metadata(model, onnx.load(shapes))
weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
shape = lambda name: [d.dim_value for d in values[name].type.tensor_type.shape.dim]
assert shape('389') == shape('src'), 'shader currently implements equal-size refinement only'
assert shape('751')[2:] == shape('src')[2:] == shape('752')[2:]
by_name = {n.name: n for n in model.graph.node}
parts = [weights['refiner.box_filter.weight'].reshape(-1)]
for name in ['Conv_312', 'Conv_314']:
    node = by_name[name]
    parts += [weights[node.input[1]].reshape(-1), weights[node.input[2]].reshape(-1)]
node = by_name['Conv_316']
parts += [weights[node.input[1]][3].reshape(-1), weights[node.input[2]][3:4]]
packed = np.concatenate(parts).astype(np.float32)
assert packed.size == 725
output.mkdir(parents=True, exist_ok=True)
packed.tofile(output / 'refiner-weights.f32')
refiner=fragment(model, values, 293, 350, ['pha'])
onnx.save(refiner, output / 'refiner.onnx')
def gather_alpha(graph):
    graph=copy.deepcopy(graph)
    graph.graph.initializer.append(numpy_helper.from_array(np.array([3], dtype=np.int64), 'probe_alpha_index'))
    node=next(n for n in graph.graph.node if n.name=='Split_349')
    node.CopyFrom(helper.make_node('Gather', ['809','probe_alpha_index'], ['811'], axis=1, name='probe_alpha_gather'))
    return prune_sort(graph)
onnx.save(gather_alpha(refiner), output / 'refiner-gather.onnx')
onnx.save(prune_sort(model), output / 'full.onnx')
onnx.save(gather_alpha(model), output / 'full-gather.onnx')
for name in ['389', '751', '752']:
    model.graph.output.append(copy.deepcopy(values[name]))
onnx.save(prune_sort(model), output / 'intermediates.onnx')
onnx.save(gather_alpha(model), output / 'intermediates-gather.onnx')
prefix = copy.deepcopy(model)
outputs = [v for v in prefix.graph.output if v.name != 'pha']
del prefix.graph.output[:]
prefix.graph.output.extend(outputs)
onnx.save(prune_sort(prefix), output / 'prefix.onnx')
for name in ['745','748']:
    model.graph.output.append(copy.deepcopy(values[name]))
onnx.save(prune_sort(model), output / 'convolution-intermediates.onnx')
conv=fragment(onnx.load(source), values, 287, 289, ['748'])
onnx.save(conv, output / 'conv290.onnx')
small = copy.deepcopy(conv)
for value in [*small.graph.input, *small.graph.output]:
    value.type.tensor_type.shape.dim[2].dim_value = 64
    value.type.tensor_type.shape.dim[3].dim_value = 64
onnx.save(small, output / 'conv290-small.onnx')
node=by_name['Conv_290']
np.concatenate([weights[node.input[1]].reshape(-1),weights[node.input[2]].reshape(-1)]).astype(np.float32).tofile(output / 'conv290-weights.f32')
g = np.array([[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]], dtype=np.float32)
winograd = np.einsum('ai,ocij,bj->ocab', g, weights[node.input[1]].astype(np.float32), g)
np.concatenate([winograd.reshape(-1), weights[node.input[2]].reshape(-1)]).astype(np.float32).tofile(output/'conv290-winograd.f32')
print('Exported weights and intermediates:', output)
