from polygraphy.backend.trt.loader import LoadPlugins
from polygraphy.backend.trt import (
    TrtRunner, EngineFromNetwork, NetworkFromOnnxPath,
    CreateConfig, Profile, SaveEngine
)
import os
import numpy as np


now_dir = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(now_dir, "model.onnx")
plugin_path = os.path.join(now_dir, "./AddScalarPlugin.so")
#plugin_path = os.path.join(now_dir, "./add_scalar_plugin.so")

assert os.path.exists(plugin_path), "插件目录不存在"
plugins = LoadPlugins(plugins=[plugin_path])()
print("插件导入完毕")


# 开始解析模型
profile = Profile()
profile.add(
    "inputs",
    min=(1, 1, 256),
    opt=(16, 32, 256),
    max=(32, 64, 256)
)
config = CreateConfig(
    max_workspace_size=2 << 30, profiles=[profile]
)
network = NetworkFromOnnxPath(onnx_path)
engine = EngineFromNetwork(network, config=config)

print(f'network:{network}')

#for i, name_ in enumerate(network):
#    layer_type = name_.type
#    layer = network[i]
#    print(f'layer-{i}\t{name_}')

# 准备好保存模型
engine = SaveEngine(engine=engine, path="model.engine")

# 运行模型
input_data = np.arange(256, dtype=np.float32).reshape(1, 1, 256)
input_data = input_data.reshape(1, 1, 256)
with TrtRunner(engine=engine) as runner:
    feed_dict = {
        "inputs": input_data
    }
    outputs = runner.infer(feed_dict=feed_dict)
    print(outputs.keys())
    print(f'input_data:{input_data}')
    print(outputs)