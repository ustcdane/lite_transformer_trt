# lite_transformmer_trt

##简介
本开源内容为 TensorRT ，利用TensorRT对模型的推理过程进行加速

<br>

目录
=================
  * [项目缘由](#项目缘由)
  * [项目特色](#项目特色)
  * [依赖环境](#依赖环境)
  * [环境配置](#环境配置)
  * [快速上手](#快速上手)
  * [模型量化](#模型量化)
  * [端侧推理](#端侧推理)
  * [代码目录说明](#代码目录说明)

<br>

## 项目缘由
开发lite-nat项目有以下考量:
- __隐私考量__ 由于用户隐私安全风险，交互方式转变等多方面因素，当前包括Google、Meta、Tencent等国内外大厂布局AI离线能力，作为`AI交互文本输入研究团队`(支持搜狗输入法业务)理应提前布局，打造离线能力
- __AI硬件发展需要__ 当前AI硬件发现迅猛，算力不断大幅提升，如21年高通提出[骁龙8](https://mp.weixin.qq.com/s/pwp_cynFypt0th_LX0GocA) 其整个SoC多个计算单元实现了`6TOPS`的异构算力，可以预计未来移动端跑AI应用是个趋势，因此涌现了OEM AI应用需求(输入法中OEM安装量是绝对大头)，因此通过输入法AI能力和`OEM` 合作也是提升搜狗输入法用户量渠道之一。此外，全球智能手机已经超过[60亿](https://www.statista.com/statistics/330695/number-of-smartphone-users-worldwide/)部，如何让AI在指尖之上运行无疑是有`挑战且前景广阔`的

<br>

## 项目特色
由于lite-nat定位是:
- __易用__  依赖当前NLP知名开源库[fairseq](https://github.com/pytorch/fairseq/) 进行二次开发，易安装使用
- __可扩展__  可以参考本项目，扩展应用于NLP序列处理任务，另外可以根据本项目plugins模仿写自己的plugins 
- __小模型大效果__  本项目初衷是探索在`存储、计算`限制下的端到端文本输入的`效果`最大化

<br>

## 依赖环境
* Python >= 3.7
* [PyTorch](http://pytorch.org/) version >= 1.5.0
* configargparse >= 0.14
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* 如果使用混合精度，需要安装英伟达的[apex]( https://github.com/NVIDIA/apex)
* gcc >= 5.4 [gcc5.4](http://ftp.gnu.org/gnu/gcc/gcc-5.4.0/gcc-5.4.0.tar.gz)
* 语料处理过程中可能需要分词，可以使用[jieba](https://github.com/fxsjy/jieba)自定义词典，或者使用sentencepiece模型，需要安装[SentencePiece]( https://github.com/google/sentencepiece)

<br>

## 环境配置
*  GCC环境配置：
  如果GCC环境低于5.0建议安装一个高版本GCC环境，执行如下命令：
    ```wget  http://ftp.gnu.org/gnu/gcc/gcc-5.4.0/gcc-5.4.0.tar.gz
    tar -zxvf gcc-5.4.0.tar.gz
    cd  gcc-5.4.0/
        ./contrib/download_prerequisites
        mkdir build
    cd build/
    ../configure --prefix=/usr/local/gcc540 --enable-checking=release --enable-languages=c,c++,fortran --disable-multilib
    make -j
    make install
    ```
    导出gcc环境，vim gccenv.sh
    ```export PATH=/usr/local/gcc540/bin/:$PATH
    export CXX=g++
    export CC=gcc
    export LD_LIBRARY_PATH=/usr/local/gcc540/lib64/:$LD_LIBRARY_PATH
    ```
    source gccenv.sh 

*  Anaconda3虚拟环境配置：确保机器已经安装了CUDA驱动(我的机器版本是CUDA-11.1)，然后安装[conda3](https://www.anaconda.com/download/) (下载Anaconda3-5.3.1-Linux-x86_64.sh 运行命令  bash Anaconda3-5.3.1-Linux-x86_64.sh 另外可以网上搜索改下conda源, 这样conda下配置虚拟环境能快起来) 

* pytorch环境配置：运行如下命令配置pytorch虚拟环境 cu11.1_torch(注意需要我的环境是cuda-11.1，可以根据自己cuda版本来安装)
```bash
conda create -n cu11.1_torch python=3.7
conda activate cu11.1_torch
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

* 配置环境
```bash
    cd lite-nat
    pip install --editable .
 ```

配置卷积 `lightconv` and `dynamicconv` GPU支持 
Lightconv_layer
```bash
    cd fairseq/modules/lightconv_layer
    python cuda_function_gen.py
    python setup.py install
```

Dynamicconv_layer
```bash
  cd fairseq/modules/dynamicconv_layer
  python cuda_function_gen.py
  python setup.py install
```

* 混合精度加速
Apex环境配置：
```git clone https://github.com/NVIDIA/apex
source gccenv.sh
cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

<br>

## 快速上手

### 数据预处理
#### 预处理训练数据
 根据项目需要，抽取语料，进行处理。由于本项目主要目标是非自回归的端到端的生成技术，因此训练语料
 需要一份对齐的语料，语料编码为UTF8 每行token间空格分开。具体形式见corpora/train.clean.*

#### 生成训练数据的二进制文件
对预处理的训练语料生成binary文件，便于训练时快速迭代数据：
```bash
nohup sh run_scripts/mk_train_data_bin.sh &
```
这一步数据处理主要依赖宿主机CPU，如果训练数据比较大可以把脚本中的workers设置和机器CPU数一致，有关脚本中的各个参数的意义参见：
[fairseq-preprocess](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-preprocess)

### 模型训练
基于fairseq框架，我们开发了多种非自回归模型。下面按照我们当时开发lite-nat所做尝试实验的顺序来说明(会附上当时做的实验的参考论文)：

* 训练标准自回归端到端模型-[Transformer](https://arxiv.org/abs/1706.03762), [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)：
自回归端到端模型Transformer是当前众多NLP,CV等领域研究的基石，众多创新点也是基于AT-transformer进行探索而来。
```bash
 nohup sh run_scripts/ATstartTrain.sh 2>&1 > log.at &
```

* 轻量自回归端到端模型-[Lite Transformer with Long-Short Range Attention paper](https://arxiv.org/abs/2004.11886),
 [Lite Transformer code](https://github.com/mit-han-lab/lite-transformer)：
 这边paper主要思路是优化了attention结构及探索FFN参数缩减对模型效果，在保证模型效果的同时大幅减少模型参数量，从而提升模型推理性能。
```bash
 nohup sh run_scripts/lite_transfromer.sh 2>&1 > log.liteTransformer &
```

* SOTA非自回归端到端模型-[Glancing Transformer](https://arxiv.org/abs/2008.07905),
 [GLAT code](https://github.com/FLC777/GLAT)：
 字节跳动提出的一种NAT训练方式，其训练过程是两次解码，根据第一次解码结果和目标语句的差异，GLAT 会决定目标词的采样量(0.5->0.2)，
 差异越大采样数量就越多；在第二次解码中，GLAT 将被采样的目标词的向量表示替换到解码器输入中，然后让模型利用新的解码器输入学习预测剩余的目标词。
 论文得到的结论是模型效果在某些赛道上接近或超越自回归模型效果，这也是我们在探索端侧长句离线模型的一个重要启发点。有关非自回归的更多研究参见:[NonAutoregGenProgress](https://github.com/kahne/NonAutoregGenProgress)
```bash
 nohup sh run_scripts/glat.sh 2>&1 > log.glat &
```


* 轻量非自回归端到端模型- 其实是针对GLAT的`encoder`参考lite-transformer的设计思路进行升级(这里只升级了encoder)：
```bash
 nohup sh run_scripts/liteEncoder_NADecoder.sh 2>&1 > log.liteEncoderNAdecoder &
```

* 轻量非自回归端到端模型- 针对GLAT的`encoder、decoder`参考lite-transformer的设计思路进行升级：
```bash
 nohup sh run_scripts/liteEncoder_liteNADecoder.sh 2>&1 > log.liteEncoderLiteNAdecoder &
```


* 轻量非自回归端到端模型- 由于NAT通病是不能很好感知上下文，因此考虑通过引入CRF对目标结果关系进行建模。
(轻量attention-卷积部分为`动态权值卷积`+CRF目标关系建模)：
```bash
 nohup sh run_scripts/lite_nat_Attention_dynamicConv_crf.sh 2>&1 > log.liteEncoderLiteNAdecoder &
```
`注意`小数据量测试时指定一张卡，不然会出现"RuntimeError: CUDA error: device-side assert triggered"的报错。


* 轻量非自回归端到端模型- 由于NAT通病是不能很好感知上下文，因此考虑通过引入CRF对目标结果关系进行建模。
(轻量attention-卷积部分为`轻量线性权值卷积`+CRF目标关系建模)：
```bash
 nohup sh run_scripts/lite_nat_Attention_lightWeightConv_crf.sh 2>&1 > log.liteEncoderLiteNAdecoder &
```
`注意`小数据量测试时指定一张卡，不然会出现"RuntimeError: CUDA error: device-side assert triggered"的报错。


* 轻量非自回归端到端模型- 由于词表embedding空间占据整个模型比例较大(70%以上)，因此考虑对emb进行压缩，思路来源[Albert](https://arxiv.org/abs/1909.11942) embedding矩阵分解。
```bash
 nohup sh run_scripts/lite_nat_Attention_lightWeightConv_crf.sh 2>&1 > log.liteEncoderLiteNAdecoder &
```

更多实验：模型压缩方法，轻量卷积标注模型。


训练中的一些参数的意义参见：
[fairseq-train](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-train)


### 测试

* 二进制数据评测
 这样一步和训练数据制作一样。对预处理的训练语料生成binary文件，便于训练时快速迭代数据：
```bash
nohup sh run_scripts/mk__bin.sh &
```
这一步数据处理主要依赖宿主机CPU，如果训练数据比较大可以把脚本中的workers设置和机器CPU数一致，有关脚本中的各个参数的意义参见： 

把生成的测试二进制文件放到新建目录下，调用脚本进行测试：
```bash
 nohup sh run_scripts/multi_test.sh 2>&1 > log.test &
```
* 交互评测 
从标准输入输入(或文件内容重定向)：
```bash
 nohup sh run_scripts/interactive_test.sh 2>&1 > log.InterTest &
```

#### 效果

 后续补充...
 模型效果
 存储效果
 计算量变化

<br>


## 模型量化
    模型量化其实实现了浮点与定点的映射关系，另外定点计算往往比浮点快得多，量化模型能带来的收益主要有：

* 较小的模型存储，对于int8一般会达到4倍存储空间减少；
* 模型推理延迟大幅降低，由于更少的访存和更快速的int8(需要底层硬件支持)，一般可提速2~4倍；

模型量化入门建议看下这篇 paper： 神经网络量化白皮书 [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295)

总结起来有以下`两点`：

* 模型参数量化方式：对于浮点权重量化有很多方式，比如对称量化、非对称量化等等，量化对象有基于模型参数通道、Tensor等方式。
* 模型量化优化方式：当前模型量化方法大致可以分为两类：Post-Training Quantization (PTQ 训练后量化)和 Quantization-Aware-Training (QAT 量化感知训练，即训练时量化)。


下面给一个轻量卷积序列标注采用PTQ、QAT两种方式效果对比，可见QAT方式更接近FP32效果：

| 模型                                             | 9key    |            |        |           |
| -------------------------------------------------- | ------- | ---------- | ------ | --------- |
|                                                    | chat.2w | chat.2w.jp | bbs.2w | bbs.2w.jp |
| dynamic-conv-encoder-tagger FP32                   | 90.66%  | 73.07%     | 87.91% | 68.03%    |
| dynamic-conv-encoder-tagger int8 `PTQ`               | 88.27%  | 70.85%     | 84.76% | 65.24%    |
| dynamic-conv-encoder-tagger int8 `QAT` QuantNoise1.0 | 89.36%  | 71.33%     | 86.55% | 66.93%    |

训练方式参考(思路在 模型训练一定程度时 量化模型参数 回写 继续训练迭代)：
[Training with Quantization Noise for Extreme Model Compression](https://arxiv.org/abs/2004.07320)

<br>

## 端侧推理
移动端端侧推理相关研究也较为活跃，目前基于ARM NENO质量向量化矩阵乘 GEMM代表性开源库有：
* ruy(gemmlowp)
* QNNPACK


结合当前移动端AI计算推理库进行调研，汇总端侧推理库优缺点如下：

| 开源库 | 优点 | 缺点 | 地址 |
| ----------------------------------- | ---------------------------------- | ----------------------------- | ------------------------------------ |
|  NCNN/TNN (MNN...) |  Tencent(Alibaba)出品，社区成熟，适合CV领域， CNN卷积极致优化，底层硬件支持丰富 | int8 gemm支持不够，NLP类模型支持少且缺乏性能优化 | [NCNN](https://github.com/Tencent/NCNN) </br> [TNN](https://github.com/Tencent/TNN) </br> [MNN](https://github.com/alibaba/MNN) |
|  ruy/gemmlowp |  Google出品，int8矩阵乘性能优越，TensorFlowlite端侧底层低bit矩阵计算库 | 文档极匮乏，支持OP少，样例少，社区不活跃，缺乏NLP类模型支持 | [gemmlowp](https://github.com/google/gemmlowp) </br> [ruy](https://github.com/google/ruy) 
| QNNPACK | Facebook出品，int8矩阵计算性能优越，pytorch端侧底层低bit矩阵计算库 | 暂停维护，文档不够丰富，缺乏NLP类模型支持 | [QNNPACK](https://github.com/pytorch/QNNPACK)

<br>

## 代码目录说明

```
lite-nat
|-- fairseq # fairseq自带
|-- fairseq_cli
|-- fairseq.egg-info
|-- generate.py
|-- LICENSE
|-- m_free.sh
|-- my_lightweight_crf_plugins #轻量非自回归端到端，attention为Long-Short Range Attention 且卷积部分为优化后的轻量线性权值卷积+CRF目标关系建模
|-- my_lite_alnum_crf_plugins#attention为Long-Short Range Attention 且卷积部分为优化后的轻量线性权值卷积 + CRF目标关系建模
|-- my_lite_char_label_hz_join_plugins 字母-音节-汉字联合训练
|-- my_lite_crf_plugins
|-- my_lite_encoder_conv_tagging_plugins 轻量卷积encoder序列标注
|-- my_lite_encoder_NAT_decoder_plugins 轻量encoder nonautoregressive decoder 
|-- my_lite_nat_embC_plugins emb矩阵分解压缩实验
|-- my_lite_nat_embC_crf_plugins emb矩阵分解压缩实验+crf
|-- my_lite_plugins
|-- my_lite_sample_plugins
|-- preprocess.py
|-- py_scripts
|-- README.ZH.md
|-- run_scripts
|-- setup.py
|-- test 一些测试脚本
|-- lite_transfomer_configs
`-- train.py
```

### 脚本说明：
| scripts | Description |
| ----------------------------------- | --------------------------------------- |
| `run_scripts/ATstartTrain.sh` | 标准自回归transformer-encoder-decoder架构训练脚本 |
| `run_scripts/lite_transfromer.sh` |  轻量自回归[lite_transfromer](https://arxiv.org/abs/2004.11886) 训练脚本|  
|  `run_scripts/glat.sh` |  字节GLAT非自回归翻译 [Glancing Transformer](https://arxiv.org/abs/2008.07905) 训练脚本 | 
|  `run_scripts/lite_nat_Attention_dynamicConv_crf.sh` | 轻量非自回归端到端，attention为Long-Short Range Attention <br> 且卷积部分为`动态权值卷积`+CRF目标关系建模<br>| 
|  `run_scripts/lite_nat_Attention_lightWeightConv_crf.sh` | 轻量非自回归端到端，attention为Long-Short Range Attention <br> 且卷积部分为优化后的`轻量线性权值卷积`+CRF目标关系建模 | <br> 


继续完善...
