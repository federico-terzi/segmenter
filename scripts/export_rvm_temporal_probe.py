"""Exact graph split: batch the frame-independent encoder, retain sequential GRUs."""
from pathlib import Path
import copy
import sys
import onnx
import numpy as np
from onnx import helper,numpy_helper
from research_rvm import specialize_metadata, fragment,prune_sort

source,inferred,destination=map(Path,sys.argv[1:4])
model=onnx.load(source)
values=specialize_metadata(model,onnx.load(inferred))
cut=next(i for i,n in enumerate(model.graph.node) if n.name=='Split_172')
end=next(i for i,n in enumerate(model.graph.node) if n.name=='Split_295')
encoder=fragment(model,values,0,cut,['389','403','420','464','601','602','603','604'])
decoder=fragment(model,values,cut,end,['751','752','r1o','r2o','r3o','r4o'])
destination.mkdir(parents=True,exist_ok=True)
onnx.save(decoder,destination/'decoder.onnx')
complete=fragment(model,values,cut,len(model.graph.node),['pha','r1o','r2o','r3o','r4o'])
onnx.save(complete,destination/'decoder-full.onnx')
features=['389','403','420','464','601','602','603','604']
sizes={name:int(np.prod([d.dim_value for d in values[name].type.tensor_type.shape.dim])) for name in features}
total=sum(sizes.values())
packed_decoder=copy.deepcopy(decoder)
inputs=[v for v in packed_decoder.graph.input if v.name not in features]
inputs.append(helper.make_tensor_value_info('packed',onnx.TensorProto.FLOAT16,[1,total]))
del packed_decoder.graph.input[:];packed_decoder.graph.input.extend(inputs)
prefix=[];offset=0
for name in features:
    p='unpack_'+name
    for suffix,array in [('start',[offset]),('end',[offset+sizes[name]]),('axis',[1]),('shape',[d.dim_value for d in values[name].type.tensor_type.shape.dim])]:
        packed_decoder.graph.initializer.append(numpy_helper.from_array(np.array(array,dtype=np.int64),p+'_'+suffix))
    prefix += [helper.make_node('Slice',['packed',p+'_start',p+'_end',p+'_axis'],[p],name=p),helper.make_node('Reshape',[p,p+'_shape'],[name],name=p+'_reshape')]
    offset+=sizes[name]
nodes=prefix+list(packed_decoder.graph.node)
del packed_decoder.graph.node[:];packed_decoder.graph.node.extend(nodes)
onnx.save(prune_sort(packed_decoder),destination/'decoder-packed.onnx')
for batch in [1,2,4,8,16]:
    candidate=copy.deepcopy(encoder)
    for value in [*candidate.graph.input,*candidate.graph.output]:value.type.tensor_type.shape.dim[0].dim_value=batch
    onnx.checker.check_model(candidate)
    onnx.save(candidate,destination/f'encoder{batch:02}.onnx')
    flat=[]
    for name in features:
        shape='pack_shape_'+name;output='pack_'+name
        candidate.graph.initializer.append(numpy_helper.from_array(np.array([batch,sizes[name]],dtype=np.int64),shape))
        candidate.graph.node.append(helper.make_node('Reshape',[name,shape],[output],name=output));flat.append(output)
    candidate.graph.node.append(helper.make_node('Concat',flat,['packed'],axis=1,name='packed_features'))
    del candidate.graph.output[:]
    candidate.graph.output.append(helper.make_tensor_value_info('packed',onnx.TensorProto.FLOAT16,[batch,total]))
    onnx.save(prune_sort(candidate),destination/f'encoder{batch:02}-packed.onnx')
print('Exported temporal encoder/decoder split:',destination)
