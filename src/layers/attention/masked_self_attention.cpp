#include <math.h>
#include "src/utils/debug_utils.h"
#include "src/layers/attention/masked_self_attention.h"

template<typename T>
LLaMASelfAttentionLayer<T>::LLaMASelfAttentionLayer(
                               int head_num,
                               int kv_head_num,
                               int head_size,
                               LLaMAAttentionStaticParams attn_params,
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
    
    q_head_per_kv(head_num / kv_head_num),
    scale(float(1 / sqrt(head_size))){}

template<typename T>
void LLaMASelfAttentionLayer<T>::allocForForward(LLaMAAttentionDynParams& params) {
    int batch_size = params.batch_size;
    int num_tokens = params.num_tokens;
    int max_q_len = params.max_q_len;
    int max_k_len = params.max_k_len;
    DataType type = getTensorType<T>(); 
    const int qkv_head_num = head_num + 2 * kv_head_num;
    
    qkv_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, qkv_head_num, head_size}); 
    mha_output = new TensorWrapper<T>(Device::GPU, type, {batch_size, hidden_units});
    
    qkv_buf->data = allocator->Malloc(qkv_buf->data, sizeof(T) * batch_size * qkv_head_num * head_size, false);
    mha_output->data = allocator->Malloc(
        mha_output->data, sizeof(T) * batch_size * hidden_units, false);
}
template<typename T>
void LLaMASelfAttentionLayer<T>::freeBuf(){
    allocator->Free(qkv_buf->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(mha_output->data);
    DeviceSyncAndCheckCudaError();
}

template<typename T>
void LLaMASelfAttentionLayer<T>::forward(TensorMap& inputs, TensorMap& outputs, LLaMAattentionWeights<T>& weights, LLaMAAttentionDynParams& params)
{   
   
    allocForForward(params);
    
    Tensor* attention_input = inputs["attention_input"];
    launchLinearGemm(attention_input->as<T>(), weights.qkv, qkv_buf, cublas_wrapper, false, true);
    DeviceSyncAndCheckCudaError();
    
    Tensor* attention_output = outputs["attention_output"];
    
    Tensor* key_cache       = outputs["all_k_cache"];
    Tensor* value_cache     = outputs["all_v_cache"];
    Tensor* finished = inputs["finished"];
    Tensor* step = inputs["step"];//[1] onCPU
    Tensor* layer_id = inputs["layer_id"];//[1] onCPU
    launchRoPE(qkv_buf, step->as<int>(), attn_static_params);
    DeviceSyncAndCheckCudaError();
    
    launchdecoderattn<T>(qkv_buf, weights.qkv, layer_id->as<int>(), key_cache->as<T>(), value_cache->as<T>(), finished->as<bool>(), step->as<int>(), mha_output, attn_static_params);
    DeviceSyncAndCheckCudaError();
#ifdef SAVE_DATA
    save_tensor(mha_output ,"self_decoder_qk_v_after_bmm.bin", layer_id->as<int>());
#else
#endif
    
    launchLinearGemm(mha_output, weights.output, attention_output->as<T>(), cublas_wrapper, false, true);
    DeviceSyncAndCheckCudaError();
#ifdef SAVE_DATA
    save_tensor(mha_output ,"self_decoder_outlinear_out.bin", layer_id->as<int>());
#else
#endif
    this->freeBuf();
}

template class LLaMASelfAttentionLayer<float>;
template class LLaMASelfAttentionLayer<half>;