#include "src/kernels/repeat_kv.h"



template <typename T>
__global__ void repeatkv(T* k_src,T* v_src,T* k_dst,T* v_dst,int layer_num,int* context_length,int batch_size,int q_head_num,int kv_head_num,int max_seq_len,int head_size,int max_k_len){
    int batch_id=blockIdx.x;
    int head_num_id=blockIdx.y;
    int tid=threadIdx.x;
    for(int i=tid;i<context_length[batch_id]*head_size;i+=blockDim.x){
        int seq_id=i/head_size;
        int hid=i%head_size;
        int kv_src_offset=layer_num*batch_size*kv_head_num*max_seq_len*head_size+batch_id*kv_head_num*max_seq_len*head_size+head_num_id%kv_head_num*max_seq_len*head_size+seq_id*head_size+hid;
        int kv_dst_offset=batch_id*q_head_num*max_k_len*head_size+head_num_id*max_k_len*head_size+seq_id*head_size+hid;
        k_dst[kv_dst_offset]=k_src[kv_src_offset];
        v_dst[kv_dst_offset]=v_src[kv_src_offset];
    }
    

}






template <typename T>
void launchRepeatKV(TensorWrapper<T> *k_cache_src,
                    TensorWrapper<T> *v_cache_src,
                    TensorWrapper<T> *k_out,
                    TensorWrapper<T> *v_out,
                    TensorWrapper<int> *layer_id,
                    TensorWrapper<int> *context_length 
                    ){
    int batch_size=k_cache_src->shape[1];
    int q_head_num=k_out->shape[1];
    int max_k_len=k_out->shape[2];
    int kv_head_num=k_cache_src->shape[2];
    int max_seq_len=k_cache_src->shape[3];
    int head_size=k_cache_src->shape[4];
    int layer_num=layer_id->getVal();
    dim3 grid(batch_size,q_head_num);
    dim3 block(1024);
    repeatkv<T><<<grid,block>>>(k_cache_src->data,v_cache_src->data,k_out->data,v_out->data,layer_num,context_length->data,batch_size,q_head_num,kv_head_num,max_seq_len,head_size,max_k_len);
}

template void launchRepeatKV(TensorWrapper<float> *k_cache_src,
                    TensorWrapper<float> *v_cache_src,
                    TensorWrapper<float> *k_out,
                    TensorWrapper<float> *v_out,
                    TensorWrapper<int> *layer_id,
                    TensorWrapper<int> *context_length 
                    );