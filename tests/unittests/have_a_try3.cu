

template<typename T>
struct maxOp{
    __device__ __inline__ T operator()(T a,T b){
        return max(a,b);
    }
};

template<typename T>
struct addOp{
    __device__ __inline__ T operator()(T &a,T& b){
        return a+b;
    }
};

template<typename T,template <typename> class Op>
__device__ T warpReduceSum(T val){
        for(int i=16;i>0;i>>=1){
            val=Op<T>()(val,__shfl_xor_sync(0xffffffff,val,i));
        }
    return val;
}


template<typename T,template <typename> class Op>
__device__ T blockReduceOp(T val){
    int tid=threadIdx.x;
    //int bid=blockIdx.x;
    int wid=threadIdx.x/32;
    int warp_num=(blockDim.x+31)/32;
    int lane_id=threadIdx.x%32;
    static __shared__ T warps[64];
    val=warpReduceSum<T,Op>(val);
    if(lane_id==0){
        warps[wid]=val;
    }
    __syncthreads();
    T out=tid<warp_num?warps[tid]:0;
    return warpReduceSum(out);
}


//rmsnorm layernorm 
template <typename T>
__global__ void layernorm(T* in_data,T* out_data,int batch_size,int c,int h,int w,int eps,int* gama,int* beta){
    int bid=blockIdx.x;
    int tid=threadIdx.x;
    int c_id=threadIdx.x/(h*w);
    int hw_=tid%threadIdx.x;
    int h_id=hw/w;
    int w_id=hw%w;
    T val=(T)0;
    T sumval(T)0;//用于存储所有的
    for(int i=tid;i<)

}


template <typename T>
void launchLayernorm()