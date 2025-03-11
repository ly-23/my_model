#include "src/kernels/swiglu.h"

template <typename T>
__device__ __inline__ T swi(T x){
    return 1.0/(1.0+expf(-x));
}

template <typename T>
__global__ void swiglu(T* input,T* output,int batch_size,int inter_size){
    int batch_id=blockIdx.x;
    int tid=threadIdx.x;
    for(int i=tid;i<inter_size;i+=blockDim.x){
        int in_offset1=batch_id*2*inter_size+1*i;
        int in_offset2=batch_id*2*inter_size+1*i+inter_size;
        int out_offset=batch_id*inter_size+i;
        output[out_offset]=swi<T>(input[in_offset1])*input[in_offset2];
    }

}

template <typename T>
void launchSwiglu(TensorWrapper<T> *input,TensorWrapper<T> *output){
    int batch_size=input->shape[0];
    int inter_size=input->shape[2];
    dim3 grid(batch_size);
    dim3 block(1024);
    swiglu<T><<<grid,block>>>(input->data,output->data,batch_size,inter_size);
}

template void launchSwiglu(TensorWrapper<float> *input,TensorWrapper<float> *output);