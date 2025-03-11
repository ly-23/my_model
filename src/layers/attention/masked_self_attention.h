#pragma once
#include "src/weights/llama/attention_weights.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/kernels/linear.h" //1st/4th kernel of masked self attention, qkv gemm
#include "src/kernels/maskScaleSoftmax.h"
#include "src/kernels/padrope.h" //2nd rope
#include "src/kernels/decoderattn.h" //3rd kernel 
#include "src/utils/tensor.h"
#include "src/kernels/cublas_utils.h"
#include "src/models/llama/llama_param.h"
#include "src/utils/macro.h"


template<typename T>
class LLaMASelfAttentionLayer {
private:
   
    const int head_num;
    const int head_size;
    const int hidden_units;
    const int q_head_per_kv; 
    const int kv_head_num;
    float scale;
     
    LLaMAAttentionStaticParams attn_static_params;
    cudaStream_t stream;
    BaseAllocator* allocator;
    
    cublasWrapper* cublas_wrapper;

    
    TensorWrapper<T>* qkv_buf     = nullptr;
    TensorWrapper<T>* mha_output = nullptr; 

public:
    LLaMASelfAttentionLayer(int head_num,
                               int kv_head_num,
                               int head_size,
                               LLaMAAttentionStaticParams attn_params,
                               cudaStream_t stream,
                               cublasWrapper* cublas_wrapper,
                               BaseAllocator* allocator);
    
    LLaMAAttentionStaticParams& GetAttnStaticParams(){
        return attn_static_params;
    }
    void allocForForward(LLaMAAttentionDynParams& params);
    void freeBuf();
    void forward(TensorMap& inputs, TensorMap& outputs, LLaMAattentionWeights<T>& weights, LLaMAAttentionDynParams& params);
};