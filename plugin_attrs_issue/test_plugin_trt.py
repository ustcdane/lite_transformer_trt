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

import os
import ctypes
import numpy as np
from cuda import cudart
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

soFile = "./AddScalarPlugin.so"
epsilon = 1.0e-6
np.random.seed(97)

def printArrayInfo(x, description=""):
    print( '%s: %s\n  Mean=%.5e,SumAbs=%.5e,Var=%.5e,Max=%.5f,Min=%.5f,SAD=%.5e'%( \
        description,str(x.shape),np.mean(x),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print("\t", x.reshape(-1)[:10])

def check(a, b, weak=False):
    print(f'a:{a}\nb:{b}')
    if weak:
        return np.all(np.abs(a - b) < epsilon)
    else:
        return np.all(a == b)

def addScalarCPU(inputH, scalar):
    return [inputH[0] + scalar]

def getAddScalarPlugin(scalar):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'AddScalar':
            parameterList = []
            parameterList.append(trt.PluginField("scalar", np.float32(scalar), trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

onnx_model_path="./model.onnx"

def run(shape, scalar):
    testCase = "<shape%s,scalar=%f>" % (shape, scalar)
    trtFile = "./model-Dims" + str(len(shape)) + ".plan"
    print("Test", testCase)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFile)
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser:
            with open(onnx_model_path, "rb") as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))

            profile = builder.create_optimization_profile()
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30

            network_inputs = [network.get_input(i) for i in range(network.num_inputs)]
            print(f'network_inputs:{network_inputs}')
            inputT0 = network_inputs[0]
            print(f'shape:{shape}')
            profile.set_shape(inputT0.name, (1, 1, 256), (16, 32, 256), (64, 32, 256))
            config.add_optimization_profile(profile)

            #pluginLayer = network.add_plugin_v2([inputT0], getAddScalarPlugin(scalar))
            #network.mark_output(pluginLayer.get_output(0))
            engineString = builder.build_serialized_network(network, config)
            if engineString == None:
                print("Failed building engine!")
                return
            print("Succeeded building engine!")
            with open(trtFile, 'wb') as f:
                f.write(engineString)
            engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0, shape)
    _, stream = cudart.cudaStreamCreate()
    #print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput


    bufferH = []
    bufferH.append(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    for i in range(nOutput):
        bufferH.append(np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMallocAsync(bufferH[i].nbytes, stream)[1])

    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    context.execute_async_v2(bufferD, stream)

    for i in range(nOutput):
        cudart.cudaMemcpyAsync(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    cudart.cudaStreamSynchronize(stream)

    outputCPU = addScalarCPU(bufferH[:nInput], scalar)
    '''
    for i in range(nInput):
        printArrayInfo(bufferH[i])
    for i in range(nOutput):
        printArrayInfo(bufferH[nInput+i])
    '''
    print("Test", testCase, check(bufferH[-1], outputCPU[0], True))

    cudart.cudaStreamDestroy(stream)
    for buffer in bufferD:
        cudart.cudaFree(buffer)
    print("Test", testCase, "finish!")

if __name__ == '__main__':
    os.system('rm ./*.plan')
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    run([2,16,256], 2)

    print("test finish!")
