from statistics import mode
import onnx_graphsurgeon as gs
import onnx
from collections import OrderedDict
import numpy as np


import ctypes
soFilePath      = './AddScalarPlugin.so'
ctypes.cdll.LoadLibrary(soFilePath)

inputs = gs.Variable(
    name="inputs", dtype=np.float32, shape=["batch", "seq", 256])
outputs = gs.Variable(
    name="outputs", dtype=np.float32, shape=["batch", "seq", 256])
nodes = [
    gs.Node(
        op="AddScalar",
        name="AddScalar_1",
        attrs=OrderedDict(scalar=np.array([2.0], dtype=np.float32)),
        inputs=[inputs],
        outputs=[outputs]
    )
]

graph = gs.Graph(
    nodes=nodes, inputs=[inputs], outputs=[outputs], opset=13,
    name="onnx")
model = gs.export_onnx(graph=graph)
onnx.save(model, "model.onnx")
print("model export success")