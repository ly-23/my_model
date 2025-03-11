#include <math.h>
#include "src/utils/macro.h"
#include "src/layers/attention/context_attention.h"

template<typename T>
LLaMAContextAttentionLayer<T>::LLaMAContextAttentionLayer(
                               int head_num,
                               int kv_head_num,
                               int head_size,
                               LLamaAttentionStaticParams attn_params,
                               cudaStream_t stream,
                               cublasWrapper* cublas_wrapper,
                               BaseAllocator* allocator):
    head_num(head_num),
    kv_head_num(kv_head_num),
    head_size(head_size),
    stream(stream),
    cublas_wrapper(cublas_wrapper),
    allocator(allocator), 
    hidden_units(head_num * head_size),
    attn_static_params(attn_params),
    // TODO: check kv_head_num is divided by haed_num
    q_head_per_kv(head_num / kv_head_num),
    scale(float(1 / sqrt(head_size))){}

template<typename T>    
void LLaMAContextAttentionLayer<T>::allocForForward(LLaMAAttentionDynParams& params) {
    int batch_size = params.batch_size;
    int num_tokens = params.num_tokens;
    int max_q_len = params.max_q_len;
    int max_k_len = params.max_k_len;
    DataType type = getTensorType<T>(); 
    const int qkv_head_num = head_num + 2 * kv_head_num;
    
    qkv_buf_wo_pad = new TensorWrapper<T>(Device::GPU, type, {num_tokens, qkv_head_num,  head_size});
    q_buf_w_pad = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_q_len, head_size}); 
    k_buf_w_pad = new TensorWrapper<T>(Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size}); 
    v_buf_w_pad = new TensorWrapper<T>(Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size});
    
    k_cache_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_k_len, head_size});
    v_cache_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_k_len, head_size});
    
    qk_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_q_len, max_k_len});
    
    qkv_buf_w_pad = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_q_len, head_size});
    
    qkv_buf_wo_pad_1 = new TensorWrapper<T>(Device::GPU, type, {num_tokens, head_num, head_size});
    
    qkv_buf_wo_pad->data = allocator->Malloc(qkv_buf_wo_pad->data, sizeof(T) * num_tokens * qkv_head_num * head_size, false);
    q_buf_w_pad->data = allocator->Malloc(
        q_buf_w_pad->data, sizeof(T) * qkv_head_num * batch_size * max_q_len * head_size, false);
    k_buf_w_pad->data = (T*)q_buf_w_pad->data + head_num * batch_size * max_q_len * head_size;
    v_buf_w_pad->data = (T*)k_buf_w_pad->data + kv_head_num * batch_size * max_q_len * head_size;
    k_cache_buf->data = allocator->Malloc(
        k_cache_buf->data, 2 * sizeof(T) * batch_size * head_num * max_k_len * head_size, false);
    v_cache_buf->data = (T*)k_cache_buf->data + batch_size * head_num * max_k_len * head_size;
    
    qk_buf->data =
        allocator->Malloc(qk_buf->data, sizeof(T) * batch_size * head_num * max_q_len * max_k_len, false);
    
    qkv_buf_w_pad->data = allocator->Malloc(
        qkv_buf_w_pad->data, sizeof(T) * batch_size * max_q_len * head_num * head_size, false);
    qkv_buf_wo_pad_1->data= allocator->Malloc(qkv_buf_wo_pad_1->data, sizeof(T) * num_tokens * head_num * head_size, false);
}

template<typename T>    
void LLaMAContextAttentionLayer<T>::freeBuf(){
    allocator->Free(qkv_buf_wo_pad->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(q_buf_w_pad->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(k_cache_buf->data);
    DeviceSyncAndCheckCudaError();

    allocator->Free(qk_buf->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(qkv_buf_w_pad->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(qkv_buf_wo_pad_1->data);
}


template<typename T>
void LLaMAContextAttentionLayer<T>::forward(TensorMap& inputs, TensorMap& outputs, LLaMAattentionWeights<T>& weights, LLaMAAttentionDynParams& params, LLaMAAttentionStaticParams& static_params)
{   
    
    allocForForward(params);
    
    Tensor* attention_input = inputs["attention_input"];
    launchLinearGemm(attention_input->as<T>(), weights.qkv, qkv_buf_wo_pad, cublas_wrapper, false, true);
    DeviceSyncAndCheckCudaError();
    
    Tensor* padding_offset = inputs["padding_offset"];
    Tensor* history_length = inputs["history_length"];
    Tensor* input_length = inputs["input_length"];
    Tensor* layer_id = inputs["layer_id"]; 
    launchPadRope(q_buf_w_pad, k_buf_w_pad, v_buf_w_pad, qkv_buf_wo_pad,
                                        weights.qkv, padding_offset->as<int>(), history_length->as<int>(), input_length->as<int>(), static_params);
#ifndef PERF
    DeviceSyncAndCheckCudaError();
#else
#endif
#ifdef SAVE_DATA
    save_tensor(q_buf_w_pad ,"q_buf_after_rope.bin", layer_id->as<int>()); 
#else
#endif
     Tensor* all_k_cache = outputs["all_k_cache"];
    Tensor* all_v_cache = outputs["all_v_cache"];
    launchConcatKV(k_buf_w_pad, v_buf_w_pad, layer_id->as<int>(), input_length->as<int>(), history_length->as<int>(), all_k_cache->as<T>(), all_v_cache->as<T>());
    DeviceSyncAndCheckCudaError();
   
    Tensor* context_length = inputs["context_length"];
    launchRepeatKV(all_k_cache->as<T>(), all_v_cache->as<T>(), context_length->as<int>(), 
                                layer_id->as<int>(), k_cache_buf, v_cache_buf);
    DeviceSyncAndCheckCudaError();
#ifdef SAVE_DATA
    save_tensor(k_cache_buf ,"k_buf_after_repeat.bin", layer_id->as<int>()); //{batch_size, head_num, max_k_len, head_size}
#else
#endif
    
    launchBatchLinearGemm(q_buf_w_pad, k_cache_buf, qk_buf, cublas_wrapper, false, true);
    DeviceSyncAndCheckCudaError();
    
    Tensor* attention_mask = inputs["attention_mask"];
    launchMaskScaleSoftmax(qk_buf, attention_mask->as<T>(), qk_buf, scale);
    DeviceSyncAndCheckCudaError();
    
    launchBatchLinearGemm(qk_buf, v_cache_buf, qkv_buf_w_pad, cublas_wrapper, false, false);
    DeviceSyncAndCheckCudaError();
#ifdef SAVE_DATA
    save_tensor(qkv_buf_w_pad ,"qk_v_buf_after_bmm.bin", layer_id->as<int>()); // {batch_size, head_num, max_q_len, head_size}
#else
#endif
launchTransRemv(qkv_buf_w_pad, padding_offset->as<int>(), qkv_buf_wo_pad_1);
    DeviceSyncAndCheckCudaError();
 
    Tensor* attention_output = outputs["attention_output"];
    launchLinearGemm(qkv_buf_wo_pad_1, weights.output, attention_output->as<T>(), cublas_wrapper, false, true);
#ifdef SAVE_DATA
    save_tensor(attention_output->as<T>() ,"out_linear_output.bin", layer_id->as<int>()); // {num_tokens, head_num, head_size}
#else
#endif
    DeviceSyncAndCheckCudaError();
    this->freeBuf();
}

template class LLaMAContextAttentionLayer<float>;
template class LLaMAContextAttentionLayer<half>;