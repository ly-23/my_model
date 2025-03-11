#include "src/kernels/maskScaleSoftmax.h"
#include <float.h>
template<typename T>
struct maxop{
    __device__ __forceinline__ T operator()(const T &a,const T &b){
        return max(a,b);
    }
};
template<typename T>
struct sumop{
    __device__ __forceinline__ T operator()(const T &a,const T &b){return a+b;}
};

template<typename T,template<typename> class reduceop>
__inline__ __device__ T warpreduce(T val){
    for(int i=16;i>0;i>>=1){
        val=reduceop<T>()(val,__shfl_xor_sync(0xffffffff,val,i));//注意前面的那个括号的含义是类的构建
    }
    return val;
}

template <typename T,template <typename> class reduceop>
__inline__ __device__ T blockreduce(T val){
    int tid=threadIdx.x;
    int wid=tid/32;
    int warpnum=(blockDim.x+31)/32;
    int lane_id=tid%32;
    static __shared__ T warps[64];
    val=warpreduce<T,reduceop>(val);
    if(lane_id==0) warps[wid]=val;
    __syncthreads();
    T out=tid<warpnum?warps[tid]:0;
    return warpreduce<T,reduceop>(out);
}


template <typename T>
__global__ void maskscalesoftmax(T* out,T* in ,T* mask,float scale,int batch_size,int head_num,int max_q_len,int max_k_len){
    int batch_id=blockIdx.x;
    int head_id=blockIdx.y;
    int q_id=blockIdx.z;
    T max_val=(T)FLT_MIN;
    int tid=threadIdx.x;
    if(tid>=max_k_len) return;
    for(int i=tid;i<max_k_len;i+=blockDim.x){
        int in_offset=batch_id*head_num*max_q_len*max_k_len+head_id*max_q_len*max_k_len+q_id*max_k_len+i;
        int mask_offset=batch_id*max_q_len*max_k_len+q_id*max_k_len+i;
        in[in_offset]=in[in_offset]*scale+(1-mask[mask_offset])*(-10000.0f);
        //if(batch_id==0&&head_id==0&&q_id==0)printf("mask*****%f\n",mask[mask_offset]);
        max_val=max(in[in_offset],max_val);
       // if(batch_id==0&&head_id==0&&q_id==0)printf("*****%f\n",max_val);
    }
    max_val=blockreduce<T,maxop>(max_val);
    __shared__ T fmax_val;
    if(tid==0){
        fmax_val=max_val;
      // printf("最大值为*****%f\n",fmax_val);
    }
    __syncthreads();
    T block_sum=(T)0;
    for(int i=tid;i<max_k_len;i+=blockDim.x){
        int in_offset=batch_id*head_num*max_q_len*max_k_len+head_id*max_q_len*max_k_len+q_id*max_k_len+i;
        //mask_offset=batch_id*max_q_len*max_k_len+q_id*max_k_len+i;
        in[in_offset]=expf(in[in_offset]-fmax_val);
        block_sum+=in[in_offset];}
    block_sum=blockreduce<T,sumop>(block_sum);
    __shared__ T fsum;
    if(tid==0){
        fsum=block_sum;
       // printf("和为*****%f\n",fsum);
    }
    __syncthreads();
    for(int i=tid;i<max_k_len;i+=blockDim.x){
        int offset=batch_id*head_num*max_q_len*max_k_len+head_id*max_q_len*max_k_len+q_id*max_k_len+i;
        //mask_offset=batch_id*max_q_len*max_k_len+q_id*max_k_len+i;
        out[offset]=in[offset]/(fsum+1e-6f);}
}

template <typename T>
void launchMaskScaleSoftmax(TensorWrapper<T> *out,
                            TensorWrapper<T> *in,
                            TensorWrapper<T> *mask,
                            float scale){
    int batch_size=out->shape[0];
    int head_num=out->shape[1];
    int max_q_len=out->shape[2];
    int max_k_len=out->shape[3];

    dim3 grid(batch_size,head_num,max_q_len);
    dim3 block(1024);
    maskscalesoftmax<T><<<grid,block>>>(out->data,in->data,mask->data,scale,batch_size,head_num,max_q_len,max_k_len);

            
                            }

template void launchMaskScaleSoftmax(TensorWrapper<float> *out,
                            TensorWrapper<float> *in,
                            TensorWrapper<float> *mask,
                            float scale);