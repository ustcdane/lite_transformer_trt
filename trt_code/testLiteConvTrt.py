#!/usr/bin/python

import os
import sys
import ctypes
import numpy as np
from glob import glob 
from time import time_ns
from datetime import datetime as dt
from cuda import cudart
import tensorrt as trt

dataFilePath = './'#"/workspace/data/"
planFilePath   = './'#"/target/"
liteConvPlan32File  = planFilePath + "lite32.plan" 
liteConvPlan16File  = planFilePath + "lite16.plan" 
liteConvPlan8File  = planFilePath + "lite8.plan" 
tokenScoreFile = planFilePath + "TokenScore"
soFileList = glob(planFilePath + "*.so")

tableHead = \
"""
bs: Batch Size
sl: Sequence Length
lt: Latency (ms)
tp: throughput (word/s)
acc:Predict accuracy
a0: maximum of absolute difference of output 0
r0: median of relative difference of output 0
----+----+--------+---------+---------+---------+---------+---------+-------------
  bs|  sl|      lt|       tp|       acc|       a0|       r0| output check
----+----+--------+---------+---------+---------+---------+---------+-------------
"""

def printArrayInfo(x, description=""):
    print( '%s: %s\n  Mean=%.5e,SumAbs=%.5e,Var=%.5e,Max=%.5f,Min=%.5f,SAD=%.5e'%( \
        description,str(x.shape),np.mean(x),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print("\t", x.reshape(-1)[:10])

def check(a, b, weak=False, epsilon = 1e-5):
    if weak:
        res = np.all( np.abs(a - b) < epsilon )
    else:
        res = np.all( a == b )
    diff0 = np.max(np.abs(a - b))/max(np.max(np.abs(a)), np.max(np.abs(b)))
    diff1 = np.median(np.abs(a - b) / (np.abs(b) + epsilon))
    #print("check:",res,diff0,diff1)
    return res,diff0,diff1

#accuracy
def check_token(a, b):
    return (a == b).astype(np.float32).mean().item()

#-------------------------------------------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')

if len(soFileList) > 0:
    print("Find Plugin %s!"%soFileList)
else:
    print("No Plugin!")
for soFile in soFileList:
    ctypes.cdll.LoadLibrary(soFile)

# Generate model result
def mkONNXResult(inputPath, liteConvPlanFile):
    if os.path.isfile(liteConvPlanFile):
        with open(liteConvPlanFile, 'rb') as encoderF:
            engine = trt.Runtime(logger).deserialize_cuda_engine(encoderF.read())
            if engine is None:
                print("Failed loading %s"%liteConvPlanFile)
                return
            print("Succeeded loading %s"%liteConvPlanFile)
    else:
        print("Failed finding %s"%liteConvPlanFile)
        return
    
    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    context = engine.create_execution_context()
    for ioFile in sorted(glob(inputPath + "/*.npz")):
        ioData = np.load(ioFile)
        src_tokens = ioData['src_tokens']
        #batchSize, sequenceLength = src_tokens.shape
        print(f'Input shape:{src_tokens.shape}')
        input_type = trt.nptype(engine.get_binding_dtype(0))
        context.set_binding_shape(0, src_tokens.shape)
        bufferH = []
        bufferH.append( src_tokens.astype(np.int32).reshape(-1) )
        for i in range(nInput, nInput + nOutput):                
            bufferH.append( np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))) )

        bufferD = []
        for i in range(nInput + nOutput):                
            bufferD.append( cudart.cudaMalloc(bufferH[i].nbytes)[1] )

        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        context.execute_v2(bufferD)

        for i in range(nInput, nInput + nOutput):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        indexOutputTokens = engine.get_binding_index('output_tokens')
        indexOutputScores = engine.get_binding_index('output_scores')
        fp32_data = {'src_tokens':ioData['src_tokens']}
        fp32_data['output_tokens'] = bufferH[indexOutputTokens]
        fp32_data['output_scores'] = bufferH[indexOutputScores]
        file_out = './test_data/' + ioFile.split('/')[-1].replace('.npz', '') + '.fp32.result.npz'
        np.savez(file_out , **fp32_data)
        for i in range(nInput + nOutput):                
            cudart.cudaFree(bufferD[i])

#-------------------------------------------------------------------------------
def testLiteTrt(liteConvPlanFile, filepath):
    print(f"Test {liteConvPlanFile}!")

    with open(tokenScoreFile, 'w') as f:
        if os.path.isfile(liteConvPlanFile):
            with open(liteConvPlanFile, 'rb') as encoderF:
                engine = trt.Runtime(logger).deserialize_cuda_engine(encoderF.read())
            if engine is None:
                print("Failed loading %s"%liteConvPlanFile)
                return
            print("Succeeded loading %s"%liteConvPlanFile)
        else:
            print("Failed finding %s"%liteConvPlanFile)
            return

        nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
        nOutput = engine.num_bindings - nInput
        context = engine.create_execution_context()
        #print(f'nInput:{nInput}\tnOutput:{nOutput}')
        print(tableHead)  # for standard output

        for ioFile in sorted(glob(filepath + "/*.npz")):
        #if os.path.isfile(inputfile):
            ioData = np.load(ioFile)
            src_tokens = ioData['src_tokens']
            batchSize, sequenceLength = src_tokens.shape
            #print(f'batchSize:{batchSize}\tsequenceLength:{sequenceLength}')
            input_type = trt.nptype(engine.get_binding_dtype(0))
            #print(f' src_tokens.shape:{ src_tokens.shape}')
            context.set_binding_shape(0, src_tokens.shape)
            #for i in range(nInput + nOutput):
            #    print("Input ->" if engine.binding_is_input(i) else "Output->", engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_dtype(i), engine.get_binding_name(i))
            #print("Finish all input binding: %s"%context.all_binding_shapes_specified)
            #print(f'src_tokens:{src_tokens}+++++++++++++++')
            bufferH = []
            bufferH.append( src_tokens.astype(np.int32).reshape(-1) )
            for i in range(nInput, nInput + nOutput):                
                bufferH.append( np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))) )

            bufferD = []
            for i in range(nInput + nOutput):                
                bufferD.append( cudart.cudaMalloc(bufferH[i].nbytes)[1] )

            for i in range(nInput):
                cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

            context.execute_v2(bufferD)

            for i in range(nInput, nInput + nOutput):
                cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

            # warm up
            for i in range(10):
                context.execute_v2(bufferD)

            # test infernece time
            t0 = time_ns()
            for i in range(30):
                context.execute_v2(bufferD)
            t1 = time_ns()
            timePerInference = (t1-t0)/1000/1000/30

            indexOutputTokens = engine.get_binding_index('output_tokens')
            indexOutputScores = engine.get_binding_index('output_scores')
            #print(f'indexOutputTokens:{indexOutputTokens} indexOutputScores:{indexOutputScores}')
            #oTk=ioData['output_tokens']
            #oSc=ioData['output_scores']
            #print(f'-----{oTk.shape}\ntarget:{oTk}predict:{bufferH[indexOutputTokens]}\n')
            #print(f'-----{oSc.shape}\ntarget:{oSc}predict:{bufferH[indexOutputScores]}\n')
            checkAcc = check_token(bufferH[indexOutputTokens], ioData['output_tokens'])
            check0 = check(bufferH[indexOutputScores], ioData['output_scores'],True,5e-3)
            string = "%4d,%4d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e, %s"%(batchSize,
                                                                        sequenceLength,
                                                                        timePerInference,
                                                                        batchSize*sequenceLength/timePerInference*1000,
                                                                        checkAcc,
                                                                        check0[1],
                                                                        check0[2],
                                                                        "Good" if check0[1] < 1 and check0[2] < 5e-3 and checkAcc > 0.94 else "Bad")
            print(string)
            f.write(string + "\n")

            for i in range(nInput + nOutput):                
                cudart.cudaFree(bufferD[i])

if __name__ == "__main__":
    mkONNXResult('./npz_input', liteConvPlan32File)
    testLiteTrt(liteConvPlan32File, './test_data')
    testLiteTrt(liteConvPlan16File, './test_data')
    #testLiteTrt(liteConvPlanint8File, './test_data')

