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
 
#include <vector>
#include <string>
#include <NvInfer.h>
#include <cuda_fp16.h>

//#include <ATen/ATen.h>
//#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "cuda_utils.cu"

// +------- Debug wrapper --------------------------------------------------------------------------
#if DEBUG
#define WHERE_AM_I() do {printf("[%s]: this=->%p\n",__func__,this);} while(0);
#else
#define WHERE_AM_I()
#endif // DEBUG


// +------- Plguin ---------------------------------------------------------------------------------
namespace
{
static const char* PLUGIN_NAME{"DynamicConv"};
static const char* PLUGIN_VERSION{"1"};
} // namespace

namespace nvinfer1
{

// +------- Plugin body ----------------------------------------------------------------------------
class DynamicConvPlugin: public IPluginV2DynamicExt
{
private:    
    std::string name_;
    std::string namespace_;
    //int padding_l_;
      struct
    {
        int padding;
    }m_;
public:
    DynamicConvPlugin(const std::string& name, int padding) : name_(name)
    {
        WHERE_AM_I();
        m_.padding = padding;
    }

    DynamicConvPlugin(const std::string& name, const void* data, size_t length) : name_(name)
    {
        WHERE_AM_I();
        memcpy(&m_, data, sizeof(m_));
    }
    
    DynamicConvPlugin() = delete;

    ~DynamicConvPlugin()
    {
        WHERE_AM_I();
    }

    size_t getSerializationSize() const noexcept override
    {
        WHERE_AM_I();
        return sizeof(m_);
    }
    
    void serialize(void *buffer) const noexcept override
    {
        WHERE_AM_I();
        memcpy(buffer, &m_, sizeof(m_));
    }
  
    IPluginV2DynamicExt* clone() const noexcept override
    {
        WHERE_AM_I();
        auto p = new DynamicConvPlugin(name_, &m_, sizeof(m_));
        p->setPluginNamespace(namespace_.c_str());
        return p;
    }

    int getNbOutputs() const noexcept override
    {
        WHERE_AM_I();
        return 1;
    }

    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override
    {
        WHERE_AM_I();
        return inputs[0];
    }

    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        WHERE_AM_I();
        if(inOut[pos].format != TensorFormat::kLINEAR)
        {
            return false;
        }
#ifdef DEBUG
        std::cout <<"--------pos " << pos << " format " << (int)inOut[pos].format << " type " << (int)inOut[pos].type << "\n"; 
#endif
        bool res = true;
        switch(pos)
        {
        case 0:
            res = (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF);
             break;
        case 1:
           res = (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF);
             break;
        /*case 2:
            res = (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF || \
            inOut[pos].type == DataType::kINT32);
            break;
        case 3:
            res = (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF);
            break;
        default:// should NOT be here
            res = false;
            */
        }
        return res;
    }
    
    DataType getOutputDataType(int outputIndex, const DataType* inputTypes, int nbInputs) const noexcept override
    {
        WHERE_AM_I();
        return inputTypes[0];
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override
    {
        WHERE_AM_I();
    }

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs,int32_t nbOutputs) const noexcept override
    {
        WHERE_AM_I();
        return 0;
    }

    void setPluginNamespace(const char* szNamespace) noexcept override
    {
        WHERE_AM_I();
        namespace_ = szNamespace;
    }
    const char* getPluginNamespace() const noexcept override
    {
        WHERE_AM_I();
        return namespace_.c_str();
    }
    const char* getPluginType() const noexcept override
    {
        WHERE_AM_I();
        return PLUGIN_NAME;
    }
    const char* getPluginVersion() const noexcept override
    {
        WHERE_AM_I();
        return PLUGIN_VERSION;
    }
    int initialize() noexcept override
    {
        WHERE_AM_I();
        return 0;
    }
    void terminate() noexcept override
    {
        WHERE_AM_I();
        return;
    }

    void destroy() noexcept override
    {
        WHERE_AM_I();
    }
    
    int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
}; // class DynamicConvPlugin

class DynamicConvPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection fc_;
    static std::vector<PluginField> attr_;
    std::string namespace_;

public:
    DynamicConvPluginCreator()
    {
        fc_.nbFields = attr_.size();
        fc_.fields = attr_.data();
    }

    ~DynamicConvPluginCreator() {}

    // 最重要的两个成员函数，分别用于“接受参数创建 Plugin” 和 “去序列化创建 Plugin”
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override
    {
        WHERE_AM_I();
        int padding_l = -1;
        for (int i = 0; i < fc->nbFields; i++)
        {
            PluginField field = fc->fields[i];
            std::string field_name(field.name);
            //std::cout << field_name << "---------\n";
            if (field_name.compare("padding") == 0)
            {
                padding_l = *reinterpret_cast<const int *>(field.data);
                std::cout << "padding_l ....... " << padding_l << "\n";
            }
        }
        auto pObj = new DynamicConvPlugin(name, padding_l);
        pObj->setPluginNamespace(namespace_.c_str());
        return pObj;
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override
    {
        WHERE_AM_I();
        auto pObj = new DynamicConvPlugin(name, serialData, serialLength);
        pObj->setPluginNamespace(namespace_.c_str());
        return pObj;
    }

    void setPluginNamespace(const char* szNamespace) noexcept override
    {
        WHERE_AM_I();
        namespace_ = szNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        WHERE_AM_I();
        return namespace_.c_str();
    }

    const char* getPluginName() const noexcept override
    {
        return PLUGIN_NAME;
    }

    const char* getPluginVersion() const noexcept override
    {
        return PLUGIN_VERSION;
    }

    const PluginFieldCollection* getFieldNames() noexcept override
    {
        return &fc_;
    }
}; // class DynamicConvPluginCreator

} // namespace nvinfer1

