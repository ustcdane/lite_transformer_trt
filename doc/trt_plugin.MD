## LayerNormPlugin
 这部分其实是初赛的时候完成的，hackathon里其实也给了其它高性能的实现方式，可以参考解决学习。
 由于这块plugin enqueue 是直接传入的输入参数，实现其它也还好，注意要写Python脚本来验证正确性。

## DynamicConvPlugin

这部分由于是lite-transformer特有的结构，而且onnx无算子可以转化实现，因此是必须写plugin的，这部分其实也折腾挺久，
问题是 onnx 节点的attr常量属性值并不能被IPluginV2* createPlugin 自动识别，这部分目前是通过转化为Constant作为enqueue的一个输入来搞的， 实现中也遇到坑，比如这个常量值其实也是在GPU内存里，切记直接取解引用到CPU端。</br>
为了避免GPU、CPU内存拷贝在转化计算图时其实用到了一个小trick，由于我们的padding_l参数一般是一个比较小的数值(1~14),因此在进行计算图修改时,把这个常量值转化为一个一位向量shape属性，用的时候直接拿这个输入参数的shape即可。
```python
            v_l = [1]*node.attrs['padding_l']
            constant1 = gs.Constant(name='padding_l_' + str(nDynamicConvPlugin), \
            values=np.ascontiguousarray(np.array(deepcopy(v_l), dtype=np.int32).reshape(-1)))
```