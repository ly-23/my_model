#include "src/kernels/concat_kv.h"



template<typename T>
__global__ void Concatkv(T* k_src,T* v_src,T* k_dst,T* v_dst,int* cur_query_lens,int* history_lens,int layer_num,int batch_size,
int head_num,int max_q_len,int head_size,int max_seq_len){
    int batch_id=blockIdx.x;
    int head_id=blockIdx.y;
    int q_id=blockIdx.z;
    int tid=threadIdx.x;

    if(q_id<cur_query_lens[batch_id])
    {int kv_dst_offset=layer_num*batch_size*head_num*max_seq_len*head_size+batch_id*head_num*max_seq_len*head_size+head_id*max_seq_len*head_size+(history_lens[batch_id]+q_id)*head_size+tid;
    int kv_src_offset=batch_id*head_num*max_q_len*head_size+head_id*max_q_len*head_size+q_id*head_size+tid;
        k_dst[kv_dst_offset]=k_src[kv_src_offset];
        v_dst[kv_dst_offset]=v_src[kv_src_offset];
    }





}

template<typename T>
void launchConcatKV(TensorWrapper<T> *k_src,
                    TensorWrapper<T> *v_src,
                    TensorWrapper<T> *k_dst,
                    TensorWrapper<T> *v_dst,
                    TensorWrapper<int> *cur_query_lens,
                    TensorWrapper<int> *history_lens,
                    TensorWrapper<int> *layer_id
                    ){
    int batch_size=k_src->shape[0];
    int head_num=k_src->shape[1];
    int max_q_len=k_src->shape[2];
    int head_size=k_src->shape[3];
    int max_seq_len=k_dst->shape[3];

    int layer_num=layer_id->getVal();
    dim3 grid(batch_size,head_num,max_q_len);
    dim3 block(head_size);
    Concatkv<T><<<grid,block>>>(k_src->data,v_src->data,k_dst->data,v_dst->data,cur_query_lens->data,history_lens->data,layer_num,batch_size,
        head_num,max_q_len,head_size,max_seq_len);


    }

template void launchConcatKV(TensorWrapper<float> *k_src,
                    TensorWrapper<float> *v_src,
                    TensorWrapper<float> *k_dst,
                    TensorWrapper<float> *v_dst,
                    TensorWrapper<int> *cur_query_lens,
                    TensorWrapper<int> *history_lens,
                    TensorWrapper<int> *layer_id
                    );