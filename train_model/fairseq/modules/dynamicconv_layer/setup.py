#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

#https://github.com/onnx/tutorials/tree/master/PyTorchCustomOperator

setup(
    name="dynamicconv_layer",
    ext_modules=[
        CUDAExtension(
            name="dynamicconv_cuda",
            sources=[
                "dynamicconv_cuda.cpp",
                "dynamicconv_cuda_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

import torch
import torch.utils.cpp_extension
# Define a C++ operator.
def test_custom_add():    
    op_source = """    
    #include <torch/script.h>    

    torch::Tensor custom_add(torch::Tensor self, torch::Tensor other) {
        return self + other;    
    }
    static auto registry = 
        torch::RegisterOperators("custom_namespace::custom_add",&custom_add);
    static auto registry =
  torch::RegisterOperators("DynamicconvSpace::dynamicconv_forward", &dynamicconv_forward);
    """
    torch.utils.cpp_extension.load_inline(
        name="dynamicconv_forward",
        cpp_sources=op_source,
        is_python_module=False,
        verbose=True,
    )

#test_custom_add()