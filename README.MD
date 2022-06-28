---
## 总述
该项目为lite-transformer模型的TensorRT推理版本实现
- 原始模型名称及链接： https://github.com/mit-han-lab/lite-transformer
- 本项目使用模型： https://drive.google.com/drive/folders/1xDC8zLuH-a6Ws0l9ERsmX5UB1Ko2xkmf
- 本项目使用训练、测试数据集： https://statmt.org/wmt16/translation-task.html#Download
- 优化效果（精度和加速比,有干货：自定义pytorch算子、自定义算子ONNX导出方式、自定义算子plugin实现、FP16优化、int8QAT训练优化 

## 如何运行

### 模型训练
 如果想跳过模型训练直接转到[trt环境配置及运行](https://github.com/ustcdane/lite_transformer_trt#%E5%AE%89%E8%A3%85nvidia-docker)

- 模型训练

由于对原始模型进行了一些修改，如果想自己训练模型参见说明：https://github.com/ustcdane/lite_transformer_trt/tree/main/train_model

  本项目可用模型地址：https://drive.google.com/drive/folders/1xDC8zLuH-a6Ws0l9ERsmX5UB1Ko2xkmf

### 模型导出ONNX
  由于基于此模型自定义算子较多，一方面需要实现pytorch C++ 拓展算子(lite-transformer已有),另一方面需要实现onnx导出方式，对于未搞过pytorch或ONNX自定义算子，这部分其实耗费了一些精力。详细步骤及导出遇到的问题[参见](https://github.com/ustcdane/lite_transformer_trt/blob/main/doc/onnx_op.MD)：

### 安装nvidia-docker

为了在docker中正常使用GPU，请安装nvidia-docker。
- 请参考(unbuntu) [Installing Docker and The Docker Utility Engine for NVIDIA GPUs](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html) 安装nvidia-docker


### 启动环境

#### 首次运行此项目

``clone``本项目后，拉取项目运行所需的镜像

```commandline
docker pull registry.cn-hangzhou.aliyuncs.com/trt2022/trt-8.4-ga
```

启动容器，注意宿主机位置映射改下：
```commandline
nvidia-docker run -it --name trt2022  -p 8231:22  -v /root/workspace/lite_transformer_trt:/lite_transformer_trt registry.cn-hangzhou.aliyuncs.com/trt2022/trt-8.4-ga
```

在容器内拉取项目需要的模型文件：

执行``pip install gdown``

进入项目目录：
```commandline
 cd lite_transformer_trt
 ```

执行``download.sh`` 注意这块下载由于网络原因可能不好使,需要手动下载然后拷贝到data_model目录。

请确保以上是在trt2022容器内执行。

#### TRT性能测试

确保前面启动环境步骤成功且在trt2022容器内执行，运行如下命令测试模型效果及计算性能：
- 运行前确保已经有 model [onnx file](https://drive.google.com/drive/folders/1xDC8zLuH-a6Ws0l9ERsmX5UB1Ko2xkmf), 如果没有则下载 拷贝到  lite_transformer_trt/data_model 目录下。运行如下命令测试性能：

```commandline
cd trt_code
bash build.sh
```

## 原始模型
### 模型简介

transformer 模型提出时在NLP界大放异彩，其效果显著，当前在CV界也作为基本框架被学术、工业界二次开发、落地应用。 本项目为轻量自回归模型 lite-transformer-[Lite Transformer with Long-Short Range Attention paper](https://arxiv.org/abs/2004.11886), [Lite Transformer code](https://github.com/mit-han-lab/lite-transformer)落地探索提出 ，利用TensorRT对变种transformer模型的推理过程进行加速，目标是服务端高性能地运行lite-transformer模型。<br>

  为什么要为lite-transformer搞服务端优化:
- __轻量__  lite-transformer模型是对标准transformer模型的结构精简，在保持模型参数量减少的情况下效果依旧保持的比较好，更适合落地业务应用；
- __NLP类自定义算子难落地__ NLPer熟知的fairseq是众多NLP学术论文(自定义算子多，如本项目自定义算子dynamicconv)的开发框架 ，但由于对onnx导出困难，其项目issue挺多相关onnx问题，但大部分没很好解决，本项目其实想给给小例子，希望其它项目遇到问题时有个参考；
- __推广应用价值__  lite-transformer相关类(Cov+self-attn 的组合结构，能够有效捕获长短信息依赖) 的研究在学术界比较广泛，但相关模型的落地开源项目较少，因此考虑提出这样一个项目抛砖引玉，让更多此类模型可以参考地在TRT上跑起来；


## 原始网络结构改造

  由于lite-transformer是基于老版本fairseq开发，其代码融入到fairseq代码，整体代码缺乏一定可读性[如其主体代码 multibranch 在 fairseq/models/](https://github.com/mit-han-lab/lite-transformer/blob/master/fairseq/models/transformer_multibranch_v2.py)，此外lite-transformer原始模型[不能导出onnx模型](https://github.com/mit-han-lab/lite-transformer/issues/24)，结合以上原因我们对模型进行以下两点改造工作：<br>
  -  采用fairseq自定义plugin方式复现lite-transformer,见代码 train_model/my_lite_plugins
  -  实现了lite-transformer dynamicconv+self-attn的非回归版本([non-autoregressive 介绍](https://aclanthology.org/2020.acl-main.15.pdf) ) my_lite_plugins/models/liteConv_NAT.py
</br>

- 模型训练代码: [train code](https://github.com/ustcdane/lite_transformer_trt/tree/main/train_model)

## 优化过程

- 模型优化开发：实现了lite-transformer dynamicconv+self-attn的非回归版本([non-autoregressive 介绍](https://arxiv.org/pdf/1711.02281.pdf) ),详细见说明文档 [train_model/README.md](https://github.com/ustcdane/lite_transformer_trt/tree/main/train_model)
- 进行onnx模型转换：由于ONNX 没有算子DynamicConv，因此需要实现onnx 自定义算子 以便模型导出为onnx 具体解决方法见：doc/README.md
- DynamicConv复杂算子plugin实现：包括实现结果精度验证，详细见[DynamicConvPlugin](https://github.com/ustcdane/lite_transformer_trt/tree/main/trt_code/DynamicConvPlugin)
- 算子融合提升性能：对LayerNorm算子融合，进一步提升性能，详细见[LayerNormPlugin](https://github.com/ustcdane/lite_transformer_trt/tree/main/trt_code/LayerNormPlugin)
- onnxruntime自定义复杂算子还存在bug这里采用pytorch作为baseline, 延迟[测试地址](https://github.com/ustcdane/lite_transformer_trt/blob/main/train_model/test/conv_onnx_and_latent_time.sh)
- 计算图修改：对计算图不能转trt engine节点进行修改，[代码](https://github.com/ustcdane/lite_transformer_trt/blob/main/trt_code/onnx_surgeon.py), onnx部分节点导出问题[汇总](https://github.com/ustcdane/lite_transformer_trt/blob/main/doc/onnx_op.MD#%E5%AF%BC%E5%87%BAonnx%E5%AD%98%E5%9C%A8%E7%9A%84%E9%97%AE%E9%A2%98)
- 采用 FP32和FP16进行模型转换,当前int8转换存在问题，[engine转换代码](https://github.com/ustcdane/lite_transformer_trt/blob/main/trt_code/build_engine.py) 
- FP32 精度可接受，性能提升 81%(baseline seqlen=32)，详情见下表
- FP16 精度可接受，性能提升 59%(FP32 seqlen=32)， 详情见下表
- in8 QAT，目前开发完模型训练，但是onnx转int8 engine出现问题，[QAT地址](https://github.com/ustcdane/lite_transformer_trt/tree/main/train_model#qat%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83)

性能提升计算公式：（BASE-X）/BASE × 100%

| 批次         | BS=16 seqlen=32      | BS=16 seqlen=8 |  
|---------------|:-------|:-------|
| pytorch           | 88.35 |21.31|
| TRT（TF32）     | 16.71 | 7.11 |
| TRT（TF32）性能提升 | 81.09% | 66.64% |
| TRT（FP16）     | 6.85 |3.68 |
| TRT（FP16相对FP32）性能提升 | 59.01% | 48.24% |
| TRT（FP16相对baseline）性能提升 | 92.25% | 82.73% |

- 修正FP16精度溢出问题：

  FP16精度问题一般由于`pow sqt div`之类的算子，而这块主要集中在layernorm，通过编写插件`LayerNorm`确保精度溢出，另外layernorm Add算子加的超参是1e-5, 这块实现时也要注意下。

- int8 转trt engine问题：
   当前采QAT训练模型，但转onnx后再转trt engine出现问题，还在思考解决中。

## 精度与加速效果

因为pytorch的模型与onnx和trt的模型结构有调整。即在pytorch导出为onnx模型时模型结构有调整，因此仅对onnx和trt版本做比较。

- 在A10显卡上TRT执行效率如下

TF32 throughput 及 精度对比
```shell
bs: Batch Size
sl: Sequence Length
lt: Latency (ms)
tp: throughput (word/s)
acc: Predict accuracy
a0: maximum of absolute difference of output 0
r0: median of relative difference of output 0
----+----+--------+---------+---------+---------+---------+---------+-------------
  bs|  sl|      lt|       tp|       acc|       a0|       r0| output check
----+----+--------+---------+---------+---------+---------+---------+-------------

  16,  32,  16.715,3.063e+04,1.000e+00,0.000e+00,0.000e+00, Good

```

FP16 throughput 及 精度对比

```shell
bs: Batch Size
sl: Sequence Length
lt: Latency (ms)
tp: throughput (word/s)
acc: Predict accuracy
a0: maximum of absolute difference of output 0
r0: median of relative difference of output 0
----+----+--------+---------+---------+---------+---------+---------+-------------
  bs|  sl|      lt|       tp|       acc|       a0|       r0| output check
----+----+--------+---------+---------+---------+---------+---------+-------------

  16,  32,   6.853,7.471e+04,9.521e-01,8.415e-02,1.170e-04, Good

```

- 问题：目前由于onnx某些节点问题，目前自动评测性时只给了BS=16 seq=32情形，实际情况其它seq len需要转换onnx时指定seqlen为 8\16等情形（目前看着应该是计算图问题，还未来得及修改）。
- onnx 自定义算子，实现onnxruntime 相关参考不多，遇到问题还不知道如何解决 本项目实现初步的 ort_test.py 还未完全跑通

## 问题反馈(bug)
- onnx node attrs 在写plugin时不能有效识别(createPlugin 时PluginFieldCollection 无具体属性信息)，只能通过常量传参的方式给enqueue
- 相关验证[代码](https://github.com/ustcdane/lite_transformer_trt/tree/main/plugin_attrs_issue)


## 经验与体会
### 体会
- 这次比赛很有意义：一方面通过比赛大家积极讨论从中学到了TRT知识 感谢赛事组织方(由于近期在做服务端一些模型加速业务，机缘巧合看到TRT2022 hackathon，初赛入群比较晚，5月初才开始搞，群里大家讨论学到不少知识，另外玮神每天很晚、周末都在积极回答大家问题，给点个赞，辛苦了 ^_^，当然复赛导师也很积极帮助解决遇到的一些问题，在此也谢谢)；另一方面可以多向参赛其他同学学习更多优秀案例来提高trt技能点~~

### 经验
- 当前对于某些训练模型框架转onnx是使用TRT转起来的一大门槛，这块需要相关大厂多多支持模型转onnx解决方案,或者大家积极参与各种训练框架模型转onnx的问题解决方法分享
- TRT 技术为AI推理提供广阔的施展空间，trt相关技术文档可以提供更多使用样例(如有些难懂的技术只给出API说明，读者可能比较迷糊）让大家参考来写code，这样trt写起来更丝滑