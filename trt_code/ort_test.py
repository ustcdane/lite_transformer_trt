import numpy as np
import sys


import onnx
import onnxruntime
import numpy as np
import os
from onnxruntime.capi.onnxruntime_pybind11_state import Fail

import torch
input = torch.randint(low=3,high=20, size=(1,8))


input[:,0]=0
input[:,-1]=2
print(input)


#Error -> [ONNXRuntimeError] : 1 : FAIL : Failed to get symbol RegisterCustomOps with error
#https://github.com/YuxinWang6/pytorch2onnx_custom_op


shared_library = './dynamicconv_cuda.cpython-37m-x86_64-linux-gnu.so'
if not os.path.exists(shared_library):
    raise FileNotFoundError("Unable to find '{0}'".format(shared_library))

session_options = onnxruntime.SessionOptions()
session_options.register_custom_ops_library(shared_library) # Do not work

#session_options.register_custom_ops_library(_get_library_path()) # 

model_ = 'lite.onnx'
print(onnx.checker.check_model(model_))
sess = onnxruntime.InferenceSession(model_, session_options)
ort_output = sess.run(None, {'src_tokens': input.numpy()})[0]
print(ort_output)
#assert np.allclose(torch_output, ort_output)