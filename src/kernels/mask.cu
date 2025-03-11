#include "src/kernels/mask.h"


template<typename T>
__global__ void masks(T* mask,int * q_lens,int* k_lens,int batch_size,int max_q_len,int max_k_len){
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    int q_len=q_lens[bid];
    int k_len=k_lens[bid];
    for(int i=tid;i<max_q_len*max_k_len;i+=blockDim.x){
        int q=i/max_k_len;
        int k=i%max_k_len;
        bool if_one=q<q_len&&k<k_len&&k<=q+k_len-q_len&&k>=k_len-q_len;
        mask[bid*max_q_len*max_k_len+i]=static_cast<T>(if_one);
    }
}

template <typename T>
void launchMasks(TensorWrapper<T>* mask,TensorWrapper<int>* q_lens,TensorWrapper<int>* k_lens){
    int max_q_len=mask->shape[1];
    int max_k_len=mask->shape[2];
    int batch_size=mask->shape[0];
    
    dim3 grid(batch_size);
    dim3 block(1024);

    masks<T><<<grid,block>>>(mask->data,q_lens->data,k_lens->data,batch_size,max_q_len,max_k_len);
}

template void launchMasks(TensorWrapper<float>* mask,TensorWrapper<int>* q_lens,TensorWrapper<int>* k_lens);