## 运行方法
``clone``本项目后，拉取项目运行所需的镜像

```commandline
docker pull registry.cn-hangzhou.aliyuncs.com/trt2022/trt-8.4-ga
```

启动容器，注意宿主机位置映射改下：
```commandline
nvidia-docker run -it --name trt2022  -p 8231:22  -v /root/workspace/lite_transformer_trt:/target registry.cn-hangzhou.aliyuncs.com/trt2022/trt-8.4-ga
```

在容器内拉取项目需要的模型文件：

执行``pip install gdown``

执行``../download.sh``

请确保以上是在trt2022在容器内执行。

#### TRT性能测试

确保前面步骤成功且在trt2022容器内

```commandline
cd /workspace
bash build.sh
```