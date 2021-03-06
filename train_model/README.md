# lite transformer train code
<br>

目录
=================
  * [项目缘由](#项目缘由)
  * [项目特色](#项目特色)
  * [依赖环境](#依赖环境)
  * [环境配置](#环境配置)
  * [快速上手](#快速上手)
  * [模型量化](#模型量化)
  * [TRT推理](#TRT推理)
  * [代码目录说明](#代码目录说明)

<br>

## 项目缘由
本项目为轻量自回归模型 lite-transformer-[Lite Transformer with Long-Short Range Attention paper](https://arxiv.org/abs/2004.11886), [Lite Transformer code](https://github.com/mit-han-lab/lite-transformer)落地探索提出 ，利用TensorRT对模型的推理过程进行加速，目标是服务端高性能地运行lite-transformer模型。<br>
  为什么要为lite-transformer搞服务端优化:
- __轻量__  lite-transformer模型是对标准transformer模型的结构精简，在保持模型参数量减少的情况下效果依旧保持的比较好，更适合落地业务应用；
- __推广应用价值__  lite-transformer相关类(Cov+self-attn 的组合结构，能够有效捕获长短信息依赖) 的研究在学术界比较广泛，但相关模型的落地开源项目较少，因此考虑提出这样一个项目抛砖引玉，让更多此类模型可以参考地在TRT上跑起来；
- __其它__ ^_^

<br>

## 项目特色
由于lite-transformer_trt定位是:
- __易用__  依赖当前NLP知名开源库[fairseq](https://github.com/pytorch/fairseq/) ，易安装使用，还可以进行快速二次模型开发。 此外，依赖大名鼎鼎的[TensorRT](https://github.com/NVIDIA/TensorRT/)进行性能落地优化；
- __可扩展__  可以参考本项目，扩展应用于NLP序列处理任务，另外可以根据本项目plugins模仿写自己的plugins ，另外根据本项目实现基于trt的推理参考；

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
    cd lite_transformer_trt/train_model
    pip install --editable .
    
   如果想用最新版本fairseq 请自行下载最新版本fairseq 进行安装:
   git clone https://github.com/pytorch/fairseq
   cd fairseq
   pip install --editable ./
    
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

### 数据处理

bash configs/wmt16.en-de/prepare.sh
把生成的数据拷贝到 lite_transformer_trt/train_model, 数据目录格式如下:
```shell
data/binary/
└── wmt16_en_de_bpe32k
    ├── dict.de.txt
    ├── dict.en.txt
    ├── test.en-de.de.bin
    ├── test.en-de.de.idx
    ├── test.en-de.en.bin
    ├── test.en-de.en.idx
    ├── train.en-de.de.bin
    ├── train.en-de.de.idx
    ├── train.en-de.en.bin
    ├── train.en-de.en.idx
    ├── valid.en-de.de.bin
    ├── valid.en-de.de.idx
    ├── valid.en-de.en.bin
    └── valid.en-de.en.idx
```

### 模型训练

```commandline
nohub sh covCrfLiteNatStart.sh  2>&1 >> log.op_cov &
```

### 导出onnx & 测试

- 导出 onnx 和pytorch延迟评测:
```commandline
sh test/conv_onnx_and_latent_time.sh
```

- 效果评测，BLEU 
```commandline
lite_cov_test.sh
```

### QAT模型训练

- int8 训练
参考: Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)

https://github.com/facebookresearch/fairseq/tree/main/examples/quant_noise

由于fairseq 自带int8 emb不支持 LearnedPositionalEmbedding, 因此实现支持位置的int8 emb训练
具体代码：[train_model/fairseq/modules/quantization/scalar/modules/qposemb.py](https://github.com/ustcdane/lite_transformer_trt/blob/main/train_model/fairseq/modules/quantization/scalar/modules/qposemb.py)

训练命令
```commandline
nohub sh QATCovCrfLiteNatStart.sh.sh  2>&1 >> log.qat_op_cov &
```
## TRT推理
详见：
https://github.com/ustcdane/lite_transformer_trt/tree/main/trt_code

## 代码目录说明

```shell
train_model
├── build
├── configs
├── covCrfLiteNatStart.sh # 训练脚本
├── data
├── fairseq
├── fairseq_cli
├── fairseq.egg-info
├── gccenv.sh
├── generate.py
├── Lstop.sh
├── make_onnx.sh
├── myLiteConvOPExp 
├── my_lite_onnx_plugins #用户自定义插件
├── my_lite_plugins #用户自定义插件
├── preprocess.py
├── QATCovCrfLiteNatStart.sh # QAT训练脚本
├── README.md
├── setup.py
├── startTrain.sh # lite-transformer 自回归训练脚本
├── stop.sh
├── test 测试& onnx 生成脚本目录
└── train.py
```