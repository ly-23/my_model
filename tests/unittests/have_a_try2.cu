


template <typename T>
__device__ T warpReduceSum(T val){
    for(int i=16;i>0;i>>=1){
        val+=__shfl_xor_sync(0xffffffff,val,i);
    }
    return val;
}

template<typename T>
__device__ T blockReduceSum(T val);
    int tid=threadIdx.x;
    //int bid=blockIdx.x;
    int wid=threadIdx.x/32;
    int warp_num=(blockDim.x+31)/32;
    int lane_id=threadIdx.x%32;
    static __shared__ T warps[64];
    val=warpReduceSum(val);
    if(lane_id==0){
        warps[wid]=val;
    }
    __syncthreads();
    T out=tid<warp_num?warps[tid]:0;
    return warpReduceSum(out);