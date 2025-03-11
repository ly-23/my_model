#pragma once
#include "src/weights/llama/attention_weights.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/kernels/linear.h"
#include "src/kernels/maskScaleSoftmax.h"
#include "src/kernels/padrope.h"
#include "src/kernels/transremv.h"
#include "src/kernels/concat_kv.h"
#include "src/kernels/repeat_kv.h"
#include "src/utils/tensor.h"
#include "src/kernels/cublas_utils.h"
#include "src/model/llama/llama_param.h"
template<typename T>
class LLaMAContextAttentionLayer {
private:

    const int head_num;
    const int head_size;
    const int hidden_units;
    const int q_head_per_kv; //for GQA and MQA
    const int kv_head_num;
    float scale;

    LLaMAAttentionStaticParams attn_static_params;
    cudaStream_t stream;
    BaseAllocator* allocator;

    cublasWrapper* cublas_wrapper;

    TensorWrapper<T>*  qkv_buf_wo_pad = nullptr;      
    TensorWrapper<T>*  q_buf_w_pad = nullptr;
    TensorWrapper<T>*  k_buf_w_pad = nullptr;
    TensorWrapper<T>*  v_buf_w_pad = nullptr;
    TensorWrapper<T>*  k_cache_buf = nullptr;
    TensorWrapper<T>*  v_cache_buf = nullptr;
    TensorWrapper<T>*  qk_buf = nullptr;
    TensorWrapper<T>*  qkv_buf_w_pad = nullptr;
    TensorWrapper<T>*  qkv_buf_wo_pad_1 = nullptr;      

public:
    LLaMAContextAttentionLayer(int head_num,
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
    void forward(TensorMap& inputs, TensorMap& outputs, LLaMAattentionWeights<T>& weights, LLaMAAttentionDynParams& params, LLaMAAttentionStaticParams& static_params);

};