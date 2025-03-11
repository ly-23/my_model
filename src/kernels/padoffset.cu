#include "src/kernels/padoffset.h"

__global__ void padoffset(int* padding_offset,int* cum_seqlens,int* input_lengths,int batch_size,int max_q_len){
    int seq_len=0;
    for(int i=0;i<batch_size;i++){
        
        for(int j=0;j<input_lengths[i];j++){
            padding_offset[seq_len+j]=i*max_q_len-seq_len;
        }
        cum_seqlens[i]=seq_len;
        seq_len+=input_lengths[i];
    }
    cum_seqlens[batch_size]=seq_len;
}


void launchpadoffset(TensorWrapper<int>* padding_offset,TensorWrapper<int>* cum_seqlens,TensorWrapper<int>* input_lengths){
    int batch_size=padding_offset->shape[0];
    int max_q_len=padding_offset->shape[1];
    padoffset<<<1,1>>>(padding_offset->data,cum_seqlens->data,input_lengths->data,batch_size,max_q_len);

}