# coding=utf-8
# Copyright 2021 NVIDIA Corporation. All rights reserved.
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
""" Build trt engine """
import argparse
import logging
import sys
import ctypes

import numpy as np

import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

soLNFilePath      = './LayerNorm.so'
soDynamicFilePath      = './DynamicConv.so'

#TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
#TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
    "--onnx_model_path",
    default=None,
    type=str,
    required=True,
    help="Path to ONNX model: ",
)

parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help="The output directory where the model checkpoints and predictions will be written.",
)

parser.add_argument(
    "--output_engine_name",
    default=None,
    type=str,
    required=True,
    help="The output directory where the engine will be written.",
)

parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision instead of 32-bit",
)
parser.add_argument(
    "--int8",
    action="store_true",
    help="Whether to use INT8",
)

parser.add_argument(
    "--isDynamic",
    action="store_true",
    help="Whether to use dynamic",
)

args = parser.parse_args()

logger.info("Parameters %s", args)

def getLayerNormPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'LayerNorm':
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

ctypes.cdll.LoadLibrary(soLNFilePath)
ctypes.cdll.LoadLibrary(soDynamicFilePath)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

#min_shape,common_shape,max_shape
# batch_size, max_seq_len

# input shape
SRC_TOKENS_MIN_INPUT_SHAPE = (1,32)
SRC_TOKENS_OPT_INPUT_SHAPE = (16,32)
SRC_TOKENS_MAX_INPUT_SHAPE = (32,32)

# TRT Engine properties
STRICT_TYPES = True
'''
engine_name = "temp_engine/bert-fp32.engine"
if args.fp16:
    engine_name = "temp_engine/bert-fp16.engine"
if args.int8:
    engine_name = "temp_engine/bert-int8.engine"
'''
engine_name = args.output_dir + '/' + args.output_engine_name

#dynameic batchs
#https://github.com/egbertYeah/simple_tensorrt_dynamic

print(f"Begin load ONNX model {args.onnx_model_path} to egine:{engine_name}")
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
    network, TRT_LOGGER
) as parser:
    with open(args.onnx_model_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))

    # Query input names and shapes from parsed TensorRT network
    network_inputs = [network.get_input(i) for i in range(network.num_inputs)]
    input_names = [_input.name for _input in network_inputs]
    for input_name in input_names:
        print(f'input name:{input_name}')
    #sys.exit(0)
    print(f'Complete parse onnx file {args.onnx_model_path}')
    with builder.create_builder_config() as config:
        # tensorrt 7.x
        #config.max_workspace_size = 1 << 40
        #tensorrt 8.x
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 40)
        if STRICT_TYPES:
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        if args.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if args.int8:
            config.set_flag(trt.BuilderFlag.INT8)

        config.clear_flag(trt.BuilderFlag.TF32) # disable TF32
        profile = builder.create_optimization_profile()
      
        if args.isDynamic:
            profile.set_shape('src_tokens', SRC_TOKENS_MIN_INPUT_SHAPE, SRC_TOKENS_OPT_INPUT_SHAPE, SRC_TOKENS_MAX_INPUT_SHAPE) # Dynamic Shape
        config.add_optimization_profile(profile)

        for i, name_ in enumerate(network):
            layer_type = name_.type
            layer = network[i]
            if name_.name.split('-')[0] == 'DynamicConvN':
                print(f'layer:{layer}')

        include_list = ['ConvTranspose'] #, 'Pow', 'Sqrt', 'LayerNormN'
        for i, name_ in enumerate(network):
            layer_type = name_.type
            layer = network[i]
            layer_precison = layer.precision
            #if layer_type == trt.LayerType.SHUFFLE :
            #    print(f'Network Layer:{i} {name_.name} {name_.type} {name_.precision}')
            #    continue
            #if layer_precison == trt.int32:
            #    print(f'Network Layer:{i} {name_.name} {name_.type} {name_.precision}')
            #    continue
            if name_.name.split('-')[0] in include_list:
                layer.precision = trt.float32 
                layer.get_output(0).dtype=trt.float32
            print(f'Network Layer:{i} {name_.name} {name_.type} {name_.precision} is set {name_.precision_is_set}')
        print(f'config.flags:{config.flags}')
        
        engineString  = builder.build_serialized_network(network, config) # old vesion build_engine will Bus Error!
        if engineString == None:
            print("Failed getting serialized engine!")
            sys.exit(1)
        print("Succeeded getting serialized engine!")
        # serialize_engine and store in file (can be directly loaded and deserialized):
        with open(engine_name, "wb") as f:
            f.write(engineString)
            print(f"Succeeded saving {engine_name} file!")