
rm -f *.onnx  *.engine *.txt
make clean

make
# 导出onnx计算图
python add_model_onnx_export.py

#polygraphy 跑trt 
python  test_plugin_polygraphy.py

# 正常流程 trt build engine 然后 trt跑
python test_plugin_trt.py