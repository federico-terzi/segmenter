"""Export the original ratio-.25 graph and weights for the two-pass refiner."""
from pathlib import Path
import copy
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper
from research_rvm import specialize_metadata, prune_sort

source, inferred, destination = map(Path, sys.argv[1:4])
model = onnx.load(source)
values = specialize_metadata(model, onnx.load(inferred))
shape = lambda name: [d.dim_value for d in values[name].type.tensor_type.shape.dim]
assert shape('389')[2:] == [v//4 for v in shape('src')[2:]]
assert shape('751')[2:] == shape('389')[2:]
weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
nodes = {n.name: n for n in model.graph.node}
parts = [weights['refiner.box_filter.weight'].ravel()]
for name in ['Conv_312','Conv_314']:
    n=nodes[name]
    parts += [weights[n.input[1]].ravel(),weights[n.input[2]].ravel()]
n=nodes['Conv_316']
parts += [weights[n.input[1]][3].ravel(),weights[n.input[2]][3:4]]
packed=np.concatenate(parts).astype(np.float32)
assert packed.size==725
destination.mkdir(parents=True,exist_ok=True)
packed.tofile(destination/'refiner-weights.f32')
onnx.save(prune_sort(copy.deepcopy(model)),destination/'full.onnx')
gather=copy.deepcopy(model)
gather.graph.initializer.append(numpy_helper.from_array(np.array([3],dtype=np.int64),'probe_alpha_index'))
next(n for n in gather.graph.node if n.name=='Split_349').CopyFrom(helper.make_node('Gather',['809','probe_alpha_index'],['811'],axis=1,name='probe_alpha_gather'))
onnx.save(prune_sort(gather),destination/'full-gather.onnx')
outputs=[v for v in model.graph.output if v.name!='pha']
outputs += [values[n] for n in ['389','751','752']]
fine=copy.deepcopy(values['src']);fine.name='fine_src';outputs.append(fine)
model.graph.node.append(helper.make_node('Identity',['src'],['fine_src'],name='probe_fine_src'))
del model.graph.output[:]
model.graph.output.extend(outputs)
model=prune_sort(model)
onnx.checker.check_model(model)
onnx.save(model,destination/'prefix.onnx')
source_resident=copy.deepcopy(model)
source_outputs=[v for v in source_resident.graph.output if v.name!='fine_src']
del source_resident.graph.output[:];source_resident.graph.output.extend(source_outputs)
onnx.save(prune_sort(source_resident),destination/'resident-source.onnx')
resident=copy.deepcopy(model)
inputs=[v for v in resident.graph.input if v.name!='src']+[values['389']]
outputs=[v for v in resident.graph.output if v.name not in ['389','fine_src']]
nodes=[v for v in resident.graph.node if '389' not in v.output]
del resident.graph.input[:];resident.graph.input.extend(inputs)
del resident.graph.output[:];resident.graph.output.extend(outputs)
del resident.graph.node[:];resident.graph.node.extend(nodes)
onnx.save(prune_sort(resident),destination/'resident-prefix.onnx')
print('Exported original ratio-.25 prefix and weights:',destination)
