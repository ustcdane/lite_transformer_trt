
# 直接拷贝训练好的fairseq模型
cp ../data_model/checkpoint_last.pt  myLiteConvOPExp/
#或者自己基于数据进行训练 会生成 pt文件 把pt文件拷贝到myLiteConvOPExp 并命名为 checkpoint_last.pt

#生成onnx文件并测试该batch size seq len下的运行延迟
sh test/conv_onnx_and_latent_time.sh

#运行成功该目录会生成文件 lite.onnx , 或者直接去https://drive.google.com/drive/folders/1xDC8zLuH-a6Ws0l9ERsmX5UB1Ko2xkmf 下载相应文件