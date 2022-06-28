## 方法论
如果想简单的把PyTorch 转换ONNX ，这三部分必不可少：
- 算子在 PyTorch 中有实现
- 把 PyTorch 中算子映射成一个或多个 ONNX 算子的方法
- ONNX 此算子

但实际这三部分或多或少缺失。最差情况是（如我们的项目）：
需要定义了一个全新的算子，它不仅缺少 PyTorch 实现，还缺少 PyTorch 到 ONNX 的映射关系。

解决办法有：
- 增加 PyTorch 算子
* 组合现有算子
* 添加 TorchScript 算子
* 添加C++ 拓展算子

- 增加映射方法
- 为 ATen 算子添加符号函数
- 为 TorchScript 算子添加符号函数
- 封装成 torch.autograd.Function 并添加符号函数

- ONNX 算子
* 使用现有 ONNX 算子
* 定义新 ONNX 算子

## 解决方法

本项目使用 torch.autograd.Function 来解决 扩展算子及 onnx导出符号映射关系。
Function 类有一个很好的性质：如果它定义了 symbolic 静态方法，该 Function 在执行 torch.onnx.export() 时就可以根据 symbolic 中定义的规则转换成 ONNX 算子。

具体见代码：[dynamicconv_layer](https://github.com/ustcdane/lite_transformer_trt/tree/main/train_model/fairseq/modules/dynamicconv_layer)主要关注下dynamicconv_cuda.cpp 记得给torch注册自定义算子torch::RegisterOperators; 其次dynamicconv_layer.py 需要给torch.autograd.Function 类 dynamicconvFunction 增加symbolic函数，便于onnx导出该算子;此外 模型代码里也记得定义下这个[算子](https://github.com/ustcdane/lite_transformer_trt/blob/main/train_model/my_lite_plugins/models/liteConv_NAT.py#L39)。 代码目录如下:

```shell
dynamicconv_layer
├── build
├── cuda_function_gen.py
├── dist
├── dynamicconv_cuda_backward.cu
├── dynamicconv_cuda.cpp
├── dynamicconv_cuda.cuh
├── dynamicconv_cuda_forward.cu
├── dynamicconv_cuda_kernel.cu
├── dynamicconv_layer.py
├── dynamiconv_cpu.cpp
├── __init__.py
└── setup.py
```

## 导出onnx存在的问题

- 问题1 torch.arange 问题：[06/22/2022-04:54:18] [TRT] [V] Range_429 [Range] inputs: [785 -> ()[INT32]], [784 -> ()[INT32]], [786 -> ()[INT32]], 
[06/22/2022-04:54:18] [TRT] [V] Registering layer: Range_429 for ONNX node: Range_429
[06/22/2022-04:54:18] [TRT] [E] [graph.cpp::computeInputExecutionUses::553] Error Code 9: Internal Error (ArgMax_423: ITopKLayer cannot be used to compute a shape tensor)
In node 673 (parseGraph): INVALID_NODE: Invalid Node - Range_429
[graph.cpp::computeInputExecutionUses::553] Error Code 9: Internal Error (ArgMax_423: ITopKLayer cannot be used to compute a shape tensor)
解决：由于 torch.arange 接收的是一个tensor ，其range把这个tensor作为shape tensor，并且将这条路径上的tensor都标记成shape tensor，所以前面的layer作检查时报错了。办法是把模型代码中torch.arange输入 shape tensor某一维相关的值。

- 问题2 类型问题. In node 704 (parseGraph): INVALID_NODE: Invalid Node - ScatterND_479
ScatterND_479: Data and Updates tensors must have compatible types.
解决：修改计算图，加类型cast节点。


注意：ONNX 是一套标准，本身不包括实现