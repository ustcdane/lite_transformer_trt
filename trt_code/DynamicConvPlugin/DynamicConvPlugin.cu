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
 
 #include "DynamicConvPlugin.h"

using namespace nvinfer1;

PluginFieldCollection DynamicConvPluginCreator::fc_{};
std::vector<PluginField> DynamicConvPluginCreator::attr_;


// FS is filter size and kernels are specialized for filter sizes
template<int FS, int SB, int padding_l, typename scalar_t>
__global__ void dynamicconv_forward_kernel(const scalar_t* input,
                                const scalar_t* weight,
                                int minibatch,
                                int sequenceLength,
                                int numFeatures,
                                int numFiltersInBlock,
                                int numHeads,
                                scalar_t* output) {
  assert(blockDim.x == SB);

  const int tid = threadIdx.x;
  const int batchIdx = blockIdx.x;
  const int featureIdx = blockIdx.y;
  const int head = featureIdx / numFiltersInBlock;

  const int IOOffset = batchIdx * numFeatures * sequenceLength
                       + featureIdx * sequenceLength;
  const scalar_t* inputFeature = &input[IOOffset];
  scalar_t* outputFeature = &output[IOOffset];

  scalar_t filter[FS];

  __shared__ scalar_t tempInput[SB + FS];
  zeroSharedMem<FS, SB, padding_l>(tempInput);

  const int numIterations = divUp<int, int>(sequenceLength, SB);

  for (int i = 0; i < numIterations; ++i) {
    __syncthreads();
    const int inputOffset = i * SB;
    load_input_to_shared<FS, SB, padding_l>(inputFeature, inputOffset,
                                            sequenceLength, i,
                                            numIterations, false, tempInput);
    __syncthreads();
    if (inputOffset + tid < sequenceLength) {

      #pragma unroll
      for (int k = 0; k < FS; ++k) {
        const int filterOffset = batchIdx * numHeads * FS * sequenceLength
                                 + head * FS * sequenceLength
                                 + k * sequenceLength
                                 + i * SB + tid;
        filter[k] = weight[filterOffset];
      }

      scalar_t out = scalar_t(0.0);
      #pragma unroll
      for (int k = 0; k < FS; ++k) {
        out += filter[k] * tempInput[tid + k];
      }

      outputFeature[inputOffset + tid] = out;

    }
  }
}

template<typename scalar_t>
void dynamicconv_forward(const scalar_t *input, const scalar_t *weight, int padding_l,\
size_t minibatch, size_t numFeatures, size_t sequenceLength,\
size_t numHeads, size_t filterSize,\
scalar_t *output, cudaStream_t stream
) {

    /*at::DeviceGuard g(input.device());
    const auto minibatch = input.size(0);
    const auto numFeatures = input.size(1);
    const auto sequenceLength = input.size(2);

    const auto numHeads = weight.size(1);
    const auto filterSize = weight.size(2);
	*/
    const auto numFiltersInBlock = numFeatures / numHeads;
    const dim3 blocks(minibatch, numFeatures);

    //auto output = at::zeros_like(input);
    //auto stream = at::cuda::getCurrentCUDAStream();

    switch(filterSize) {

        case 3:

            if (padding_l == 1) 
			{
                    dynamicconv_forward_kernel<3, 32, 1, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input,
                            weight,
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output);
                
            } else if (padding_l == 2) {
                    dynamicconv_forward_kernel<3, 32, 2, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input,
                            weight,
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output);
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;


        case 5:

            if (padding_l == 2) {
                    dynamicconv_forward_kernel<5, 32, 2, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input,
                            weight,
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output);
            } else

            if (padding_l == 4) {
                    dynamicconv_forward_kernel<5, 32, 4, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input,
                            weight,
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output);
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;


        case 7:

            if (padding_l == 3) {
                    dynamicconv_forward_kernel<7, 32, 3, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input,
                            weight,
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output);
            } else

            if (padding_l == 6) {
                    dynamicconv_forward_kernel<7, 32, 6, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input,
                            weight,
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output);
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;


        case 15:

            if (padding_l == 7) {
                    dynamicconv_forward_kernel<15, 32, 7, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input,
                            weight,
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output);
            } else

            if (padding_l == 14) {
                    dynamicconv_forward_kernel<15, 32, 14, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input,
                            weight,
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output);
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;


        case 31:

            if (padding_l == 15) {
                    dynamicconv_forward_kernel<31, 32, 15, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input,
                            weight,
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output);
            } else

            if (padding_l == 30) {
                    dynamicconv_forward_kernel<31, 32, 30, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input,
                            weight,
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output);
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;


        case 63:

            if (padding_l == 31) {
                    dynamicconv_forward_kernel<63, 64, 31, scalar_t>
                    <<<blocks, 64, 0, stream>>>(
                            input,
                            weight,
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output);
            } else

            if (padding_l == 62) {
                    dynamicconv_forward_kernel<63, 64, 62, scalar_t>
                    <<<blocks, 64, 0, stream>>>(
                            input,
                            weight,
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output);
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;


        case 127:

            if (padding_l == 63) {
                    dynamicconv_forward_kernel<127, 128, 63, scalar_t>
                    <<<blocks, 128, 0, stream>>>(
                            input,
                            weight,
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output);
            } else

            if (padding_l == 126) {
                    dynamicconv_forward_kernel<127, 128, 126, scalar_t>
                    <<<blocks, 128, 0, stream>>>(
                            input,
                            weight,
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output);
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;


        case 255:

            if (padding_l == 127) {
                    dynamicconv_forward_kernel<255, 256, 127, scalar_t>
                    <<<blocks, 256, 0, stream>>>(
                            input,
                            weight,
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output);
            } else

            if (padding_l == 254) {
                    dynamicconv_forward_kernel<255, 256, 254, scalar_t>
                    <<<blocks, 256, 0, stream>>>(
                            input,
                            weight,
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output);
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;

        default:
            std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
    }
}

int32_t DynamicConvPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    size_t minibatch = inputDesc[0].dims.d[0], numFeatures = inputDesc[0].dims.d[1], sequenceLength = inputDesc[0].dims.d[2];
    size_t numHeads =  inputDesc[1].dims.d[1], filterSize = inputDesc[1].dims.d[2];
#ifdef DEBUG  
    printf("---------padding:%d\n",m_.padding);
#endif
    if(m_.padding == -1) {
#ifdef DEBUG
        printf("inputDesc[2].dims.d[0]:%d type:%d\n",  inputDesc[2].dims.d[0], inputDesc[2].type);
        printf("inputs[2]:%d\n", inputs[2]);
#endif
        //auto const_ptr = (const int*)inputs[2]; 
        //m_.padding  =  int(const_ptr[0]);// 这里不能这样搞 输入节点应在GPU内存
        m_.padding  =  inputDesc[2].dims.d[0];// 由于需要一次GPU到host的内存拷贝且padding数值一般不大, 这里用这个trick
    }

#ifdef DEBUG
    printf("padding_l:%d\n",  m_.padding);
    printf("minibatch:%d numFeatures:%d sequenceLength:%d numHeads:%d filterSize:%d\n",\
        minibatch, numFeatures, sequenceLength, numHeads, filterSize);
#endif

    if (inputDesc[0].type == DataType::kFLOAT)
    {
        dynamicconv_forward((float *)inputs[0], (float *)inputs[1], m_.padding,\
            minibatch, numFeatures, sequenceLength, numHeads, filterSize,\
            (float*)outputs[0], stream);
    }
    else if (inputDesc[0].type == DataType::kHALF)
    {
        dynamicconv_forward(reinterpret_cast<const __half *>(inputs[0]), reinterpret_cast<const __half *>(inputs[1]), m_.padding,\
            minibatch, numFeatures, sequenceLength, numHeads, filterSize,\
            reinterpret_cast<__half *>(outputs[0]), stream);
    }
    
    return 0;
}

REGISTER_TENSORRT_PLUGIN(DynamicConvPluginCreator);

