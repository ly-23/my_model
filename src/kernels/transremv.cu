#include "src/kernels/transremv.h"

template<typename T>
__global__ void transremv(T* in,T* out,int* padding_offset,int batch_size,int head_num,int max_q_len,int head_size,int num_tokens){
    int num_token_id=blockIdx.x;
    int batch_id=(num_token_id+padding_offset[num_token_id])/max_q_len;
    int q_id=(num_token_id+padding_offset[num_token_id])%max_q_len;
    int tid=threadIdx.x;
    for(int i=tid;i<head_size*head_num;i+=blockDim.x){
        int head_id=i/head_size;
        int hid=i%head_size;
        int in_offset=batch_id*head_num*max_q_len*head_size+head_id*max_q_len*head_size+q_id*head_size+hid;
        int out_offset=num_token_id*head_num*head_size+head_id*head_size+hid;
        out[out_offset]=in[in_offset];
    }
    
}


template<typename T>
void launchTransRemv(TensorWrapper<T> *in,TensorWrapper<T> *out,TensorWrapper<int> *padding_offset){
    int batch_size=in->shape[0];
    int head_num=in->shape[1];
    int max_q_len=in->shape[2];
    int head_size=in->shape[3];
    int num_tokens=out->shape[0];

    dim3 grid(num_tokens);
    dim3 block(min(1024,head_num*head_size));
    transremv<T><<<grid,block>>>(in->data,out->data,padding_offset->data,batch_size,head_num,max_q_len,head_size,num_tokens);

}

template void launchTransRemv(TensorWrapper<float> *in,TensorWrapper<float> *out,TensorWrapper<int> *padding_offset);