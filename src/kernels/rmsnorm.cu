#include "src/kernels/rmsnorm.h"

template<typename T>
__device__ T warpSumReduce(T val){
    for(int i=16;i>0;i>>=1){
        val+=__shfl_xor_sync(0xffffffff,val,i);
    }
    return val;
}

template<typename T>
__device__ T blockSumReduce(T val){
    int tid=threadIdx.x;
    int wid=threadIdx.x/32;
    int lane_id=threadIdx.x%32;
    int warp_nums=(blockDim.x+31)/32;
    val=warpSumReduce<T>(val);
    static __shared__ float warps[64];
    if(lane_id==0){
        warps[wid]=val;
    }
    __syncthreads();
    T blocksum;
    blocksum=tid<warp_nums?warps[tid]:0;
    blocksum=warpSumReduce<T>(blocksum);
    return blocksum;

}

template<typename T>
__global__ void rmsnorm(T* output,T* residual ,T* scale, float eps,bool is_last,int num_tokens,int hidden_units)
{
    int bid=blockIdx.x;
    int tid=threadIdx.x;
    if(tid<hidden_units){
    T logit=(T)0;
    // if(tid==32&&bid==0)
    //     printf("logit的值为%f\n",logit);
    for(int i=tid;i<hidden_units;i+=blockDim.x){
        logit=logit+output[bid*hidden_units+i]*output[bid*hidden_units+i];
        residual[bid*hidden_units+i]=output[bid*hidden_units+i];               //这里并不对，因为output并没有更新。
    }
    // if(tid==32&&bid==0)
    //     printf("logit的值为%f\n",logit);
    logit=blockSumReduce<T>(logit);
    __shared__ float inv_mean;
    if(tid==0) inv_mean=rsqrtf(logit/hidden_units+eps);
    __syncthreads();                        //因为少了这一排导致的错误
    // if(tid==32&&bid==0)
    //     printf("inv_mean的值为%f\n",inv_mean);
    for(int i=tid;i<hidden_units;i+=blockDim.x){
        output[bid*hidden_units+i]=output[bid*hidden_units+i]*inv_mean*scale[i];
        // if(tid==32&&bid==0)
        // printf("output的值为%f\n", output[bid*hidden_units+i]);
    }}

}

template <typename T>
void launchRMSNorm(TensorWrapper<T>* output,TensorWrapper<T>* residual,rmsNormWeight<T>* scale,float eps,bool is_last){
    int num_tokens=residual->shape[0];
    int hidden_units=residual->shape[1];
    dim3 grid(num_tokens);
    dim3 block(1024);
    rmsnorm<T><<<grid,block>>>(output->data,residual->data,scale->gamma,eps,is_last,num_tokens,hidden_units);
}

template void launchRMSNorm(TensorWrapper<float>* output,TensorWrapper<float>* residual,rmsNormWeight<float>* scale,float eps,bool is_last);