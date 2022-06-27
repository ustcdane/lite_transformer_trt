/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
 #include "LayerNormPlugin.h"

using namespace nvinfer1;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

template<typename T>
__global__ void layerNormKernel(T *pInput, T *pOutput)
{
    const int tx = threadIdx.x, index = blockIdx.x * 256 + threadIdx.x;

    __shared__ float temp[128];

    float value0 = pInput[index];
    float value1 = pInput[index + 128];

    temp[tx] = value0 + value1;
    __syncthreads();

    for (int stride = 64; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    float mean = temp[0] / 256;
    __syncthreads();

    temp[tx] = (value0 - mean) * (value0 - mean) + (value1 - mean) * (value1 - mean);
    __syncthreads();

    for (int stride = 64; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    float var = temp[0] / 256;

    pOutput[index]       = (value0 - mean) * rsqrtf(var + 6e-6);
    pOutput[index + 128] = (value1 - mean) * rsqrtf(var + 6e-6);
}

__global__ void layerNormKernelFast32(float* i_A, float* o_C, int vec_len, int bs, int dpt)
{
	//printf("vec_len:%d blockDim:%d blockIdx:%d threadIdx:%d\n", vec_len, blockDim.x, blockIdx.x, threadIdx.x);
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int base_idx = bx * vec_len + tx;
	float sum_aa = 0.0f;
	int i, pos;
	__shared__ float s_mean, s_variance;//var,

	for (i = 0; i < dpt; i++)
	{
		pos = base_idx + i * blockDim.x;
		sum_aa += i_A[pos];
	}

	for (int i = 16; i >= 1; i /= 2)
	{
		//sum_aa += __shfl_xor(sum_aa, i, 32);
		sum_aa += __shfl_xor_sync(0xffffffff, sum_aa, i, 32);
	}

	if (tx == 0)
	{
		s_mean = sum_aa / float(vec_len);
	}
	__syncthreads();

	sum_aa = 0.0f;
	float tmp;

	for (i = 0; i < dpt; i++)
	{
		pos = base_idx + i * blockDim.x;
		tmp = (i_A[pos] - s_mean);
		sum_aa += tmp * tmp;
	}
	for (int i = 16; i >= 1; i /= 2)
	{
		//sum_aa += __shfl_xor(sum_aa, i, 32);
		sum_aa += __shfl_xor_sync(0xffffffff, sum_aa, i, 32);
	}

	if (tx == 0)
	{
		//var = sum_aa / float(vec_len);
		s_variance = rsqrtf(sum_aa / float(vec_len) + 1e-5);
	}
	__syncthreads();

	//cal norm 
	for (i = 0; i < dpt; i++)
	{
		pos = base_idx + i * blockDim.x;
		o_C[pos] =  (i_A[pos] - s_mean) * s_variance; /// sqrtf(var + (6e-6));
	}

}

__global__ void layerNormKernelFast16(half* i_A, half* o_C, int vec_len, int bs, int dpt)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int base_idx = bx * vec_len + tx;
	float sum_aa = 0.0f;
	int i, pos;
	__shared__ float s_mean, s_variance;//var,
	//bool isFP16 = std::is_same<T, half>::value;

	for (i = 0; i < dpt; i++)
	{
		pos = base_idx + i * blockDim.x;
		//sum_aa += __half2float(i_A[pos]);
		sum_aa += (float)(__ldg(&i_A[pos])); // fp16 fp32可以共享代码了...
	}

	for (int i = 16; i >= 1; i /= 2)
	{
		sum_aa += __shfl_xor_sync(0xffffffff, sum_aa, i, 32);
	}

	if (tx == 0)
	{
		s_mean = sum_aa / float(vec_len);
	}
	__syncthreads();

	sum_aa = 0.0f;
	float tmp;

	for (i = 0; i < dpt; i++)
	{
		pos = base_idx + i * blockDim.x; 
		tmp = (__half2float(i_A[pos]) - s_mean);
		//tmp = ((float)(__ldg(&i_A[pos])) - s_mean);
		sum_aa += tmp * tmp;
	}
	for (int i = 16; i >= 1; i /= 2)
	{
		sum_aa += __shfl_xor_sync(0xffffffff, sum_aa, i, 32);
	}

	if (tx == 0)
	{
		//var = sum_aa / float(vec_len);
		s_variance = rsqrtf(sum_aa / float(vec_len) + 1e-5);
	}
	__syncthreads();

	//cal norm 
	for (i = 0; i < dpt; i++)
	{
		pos = base_idx + i * blockDim.x;
		float temp = (__half2float(i_A[pos]) - s_mean) * s_variance; // / sqrtf(var + (6e-6));
		//float temp = ((float)(__ldg(&i_A[pos])) - mean) * s_variance; 
		o_C[pos] = __float2half(temp);//(T) half(temp);
	}
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int p_batch_size = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    //printf("batch_size:%d d[0]:%d d[1]:%d d[2]:%d\n", p_batch_size, inputDesc[0].dims.d[0], inputDesc[0].dims.d[1],inputDesc[0].dims.d[2]);
	int num_thread = 32, p_vec_size = inputDesc[0].dims.d[2];
	int dpt = p_vec_size / num_thread;
	if (p_vec_size % num_thread != 0)
		dpt++;

    if (inputDesc[0].type == DataType::kFLOAT)
    {
		layerNormKernelFast32 << <p_batch_size, num_thread >> > ((float *)inputs[0], (float *)outputs[0], \
		p_vec_size, p_batch_size, dpt);

    }
    else
    {
		layerNormKernelFast16 << <p_batch_size, num_thread >> > ((half *)inputs[0], (half *)outputs[0], \
		p_vec_size, p_batch_size, dpt);
    }
    
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

