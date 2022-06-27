#!/usr/bin/env bash

#rm -f *.plan 
onnx='./lite_clean.onnx'
#fp32
python build_engine.py --isDynamic --onnx_model_path $onnx  --output_dir . --output_engine_name lite32.plan

#fp16
python build_engine.py --isDynamic --onnx_model_path $onnx --output_dir . --fp16 --output_engine_name lite16.plan
#int8 
python build_engine.py --isDynamic --onnx_model_path $onnx --output_dir . --int8 --output_engine_name lite16.plan
