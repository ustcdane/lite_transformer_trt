#coding=utf8
import numpy as np
import os
import onnx
import onnx_graphsurgeon as gs
#from onnxsim import simplify
from collections import OrderedDict
from copy import deepcopy
import ctypes
import argparse

soFilePath      = './LayerNorm.so'
ctypes.cdll.LoadLibrary(soFilePath)
soFilePath2      = './DynamicConv.so'
ctypes.cdll.LoadLibrary(soFilePath2)


parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
    "--input",
    default=None,
    type=str,
    required=True,
    help="Input ONNX model ",
)

# Required parameters
parser.add_argument(
    "--output",
    default=None,
    type=str,
    required=True,
    help="Output ONNX model ",
)

args = parser.parse_args()

#onnx_type_mapping = {"int64": 7, "bool": 9, "uint32": 12, "uint64": 13}
def lite_transformer_encoder_plugin(inOnnx="./lite.onnx", outOnnx="./lite_clean.onnx"):    
    graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(inOnnx)))
    #plugin
    nLayerNormPlugin = 0
    nDynamicConvPlugin = 0
    nEq = 0
    node_replace_set = set()
    for node in graph.nodes:
        if node.name not in node_replace_set and node.op == 'ReduceMean' and \
            node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
            node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
            node.o().o(0).o().op == 'ReduceMean' and \
            node.o().o(0).o().o().op == 'Add' and \
            node.o().o(0).o().o().o().op == 'Sqrt' and \
            node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1):
            node_replace_set.add(node.name)
            inputTensor = node.inputs[0]
            lastDivNode = node.o().o(0).o().o().o().o()

            layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNormPlugin), inputs=[inputTensor], outputs=[lastDivNode.outputs[0]])
            graph.nodes.append(layerNormN)
            nLayerNormPlugin += 1
            lastDivNode.outputs = []
        if node.name not in node_replace_set and node.op == 'dynamicconv_forward':
            node_replace_set.add(node.name)
            v_l = [1]*node.attrs['padding_l']
            constant1 = gs.Constant(name='padding_l_' + str(nDynamicConvPlugin), \
            values=np.ascontiguousarray(np.array(deepcopy(v_l), dtype=np.int32).reshape(-1)))
            inputs_ = node.inputs + [constant1]
            layerDynamic = gs.Node("DynamicConv", "DynamicConvN-" + str(nDynamicConvPlugin), inputs=inputs_, \
            outputs=node.outputs,attrs=node.attrs)
            graph.nodes.append(layerDynamic)
            nDynamicConvPlugin += 1
            node.outputs = []
        if node.name not in node_replace_set and node.op == 'ScatterND' and node.i(0).op == 'Where' and node.i(1).op == 'Concat' and  node.o().op == 'ScatterElements':
            node_replace_set.add(node.name)
            cast_output = gs.Variable(name=node.name + "_Cast_output", dtype=None, shape=None)
            attrs_dict = {}
            attrs_dict['to'] = 7
            newNode = gs.Node(name= node.name + "_Cast", op="Cast", inputs=[node.i(0).outputs[0]],
                      outputs=[cast_output], attrs=attrs_dict)
            node.inputs[0] = cast_output
            graph.nodes.append(newNode)  # 记得把新节点加入计算图中
            
        if node.name not in node_replace_set and node.op == 'ScatterElements' :
            node_replace_set.add(node.name)
            #print(f'node:{node}')
            cast_output = gs.Variable(name=node.name + "_Cast_output", dtype=None, shape=None)
            attrs_dict = {}
            attrs_dict['to'] = 7
            newNode = gs.Node(name= node.name + "_Cast", op="Cast", inputs=[node.i(2).outputs[0]],
                      outputs=[cast_output], attrs=attrs_dict)
            node.inputs[2] = cast_output
            node.outputs[0].dtype = np.int32 # avoid equal type error
            graph.nodes.append(newNode)  # 记得把新节点加入计算图中
        if node.name not in node_replace_set and node.op == 'Equal' and node.name in ['Equal_507','Equal_510', 'Equal_549']:
            node_replace_set.add(node.name)
            #print(f'node:{node}')
            constant1 = gs.Constant(name='Equal_const_' + str(nEq), values=np.ones(shape=[1], dtype=np.int32))
            node.inputs[1] = constant1
            nEq += 1

    print("%4d LayerNormPlugin" %nLayerNormPlugin)
    print("%4d nDynamicConvPlugin" %nDynamicConvPlugin)
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), outOnnx)

def simple_onnx(inOnnx="./encoder_new.onnx", outOnnx="./encoder_simple.onnx"):
    # load your predefined ONNX model
    model = onnx.load(inOnnx)
    # convert model
    model_simp, check = simplify(model, dynamic_input_shape=True, )
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, outOnnx)

if __name__ == '__main__':

    lite_transformer_encoder_plugin(inOnnx=args.input, outOnnx=args.output)
    #lite_transformer_encoder_plugin(inOnnx="./lite_qat.onnx", outOnnx="./lite_qat_clean.onnx")
    #simple_onnx(inOnnx="./lite_plugin.onnx", outOnnx="./lite_new.onnx")

