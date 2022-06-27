#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
import os
import ctypes
import numpy as np
from cuda import cudart  # 使用 cuda runtime API
import tensorrt as trt

soFilePath      = './DynamicConv.so'

epsilon         = 5e-4
npDataType = np.float16
np.random.seed(97)

def check(a, b, weak = False):
    if weak:
        return np.all( np.abs(a - b) < epsilon)
    else:
        return np.all( a == b )

def getDynamicConvPlugin(pad_l):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'DynamicConv':
            pl = trt.PluginField('padding', np.int32(pad_l), trt.PluginFieldType.INT32)
            return c.create_plugin(c.name, trt.PluginFieldCollection([pl]))
    return None
#x:torch.Size([64, 512, 9]) weights:torch.Size([64, 8, 15, 9]) padding_l:7 outputs[0]:torch.Size([64, 512, 9])

def readData(file_npz):
    #for ioFile in sorted(glob(dataFilePath + "./encoder-*.npz")):
    if os.path.isfile(file_npz):
        ioData = np.load(file_npz)
        x = ioData['x']
        weights = ioData['weights']
        padding_l = ioData['padding_l']
        outputs = ioData['outputs']
        batchSize, emb_size, sequenceLength = x.shape
        print(f'data shape:\nx:{x.shape} weights:{weights.shape} pading_l:{padding_l} outputs:{outputs.shape}')
        
        return (x, weights, padding_l, outputs)

network_padding_l = 1
def run():
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    testCase = "fp%s" % ('16' if int(npDataType == np.float16) else '32')
    trtFile = "./model-" + testCase + ".plan"

    builder         = trt.Builder(logger)
    network         = builder.create_network(1<<0)
    config          = builder.create_builder_config()
    config.max_workspace_size = 6 << 30
    #config.flags    = 0
    config.flags = 1 << int(trt.BuilderFlag.FP16) if int(npDataType == np.float16) else 0

    inputTensorList = []
    trtDataType = trt.float16 if int(npDataType == np.float16) else trt.float32
    inputTensorList.append( network.add_input('inputX', trtDataType, [64, 512, -1]) )
    inputTensorList.append( network.add_input('inputW', trtDataType, [64, 8, -1, -1]) )
    #inputTensorList.append( network.add_input('inputP', trt.int32, [1]) )
    profile = builder.create_optimization_profile()
    profile.set_shape('inputX',[64,512,3],[64,512,12],[64,512,32])
    profile.set_shape('inputW',[64,8,1,3],[64,8,3,12],[64,8,15,32])
    #profile.set_shape('inputP',[1],[1],[1])
    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTensorList, getDynamicConvPlugin(network_padding_l))
    pluginLayer.get_output(0).dtype = trtDataType
    network.mark_output(pluginLayer.get_output(0))

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trtFile, 'wb') as f:
        f.write(engineString)

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    #sys.exit(0)
#########################################################################
    x, weights, padding_l, outputs = readData('dynamicconvFunction-64-8-3-12.npz')
    assert network_padding_l == padding_l
    context = engine.create_execution_context()

    # x: (64, 512, 12)
    context.set_binding_shape(0, [x.shape[0], x.shape[1], x.shape[2]])
    # weights: (64, 8, 3, 12)
    context.set_binding_shape(1, [weights.shape[0], weights.shape[1], weights.shape[2], weights.shape[3]])
    #context.set_binding_shape(2, [1])
    print("Binding all ? %s " % (["No","Yes"][int(context.all_binding_shapes_specified)]))
    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))

    bufferH = []
    print(x.astype(npDataType).reshape(-1).dtype)
    
    bufferH.append(x.astype(npDataType).reshape(-1))
    bufferH.append(weights.astype(npDataType).reshape(-1))
    #bufferH.append( padding_l.astype(np.int32))
    #nInput = len(bufferH)
    bufferH.append(outputs.astype(npDataType).reshape(-1))
    #nOutput = len(bufferH) - nInput
    bufferD = []
    for i in range(len(bufferH)):                
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)
    
    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    print("check result:")
    temp1 = bufferH[-1]
    temp2 = outputs.astype(npDataType).reshape(-1)
    print(check(temp1,temp2, True), "max diff=%f"%(np.max(np.abs(temp1 - temp2))) )
    
    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == '__main__':
    os.system("rm -f ./*.trt")
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    run()