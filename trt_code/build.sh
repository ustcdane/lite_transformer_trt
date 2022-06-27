

cd ./LayerNormPlugin
make clean
make
cp LayerNorm.so ..
cd ..

cd ./DynamicConvPlugin
make clean
make
cp DynamicConv.so ..
cd ..

rm ./*.plan ./*.onnx

# onnx 如何生成参见: train_model/make_onnx.sh

onnx_surgeon_res='./lite_clean.onnx'
# 对onnx 计算图文件进行编辑，包括LayerNormPlugin  DynamicConvPlugin 节点替换、转换engine失败节点编辑等
python onnx_surgeon.py --input ../data_model/lite.onnx  --out $onnx_surgeon_res


#生成trt engine

#fp32
python build_engine.py --isDynamic --onnx_model_path $onnx_surgeon_res  --output_dir . --output_engine_name lite32.plan

#fp16
python build_engine.py --isDynamic --onnx_model_path $onnx_surgeon_res --output_dir . --fp16 --output_engine_name lite16.plan

#int8  目前生成int8 还有些问题 
#python build_engine.py --isDynamic --onnx_model_path $onnx_surgeon_res --output_dir . --int8 --output_engine_name lite16.plan


#测试TRT计算延迟性能
python testLiteConvTrt.py