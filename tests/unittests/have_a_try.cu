#include<cuda.h>
#include<cuda_fp16.h>
#include<cuda_runtime.h>
#include<stdio.h>

enum type={
    int,
    float,
    half
}
template<typename T,type leixing>

template <typename T>
__global__ void vec_add(T* a,T* b,T* c){
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    if(tid*4<100){
        float4* a4=reinterpret_cast<float4*>(a+tid*4);
        float4* b4=reinterpret_cast<float4*>(b+tid*4);
        float4* c4=reinterpret_cast<float4*>(c+tid*4);
        (*c4).x=(*a4).x+(*b4).x;
        (*c4).y=(*a4).y+(*b4).y;
        (*c4).z=(*a4).z+(*b4).z;
        (*c4).w=(*a4).w+(*b4).w;
    }
    
}
template <>
__global__ void vec_add<half>(half* a, half* b,half* c){
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    if(tid*2<100){
        half2* cc=reinterpret_cast<half2*>(c+tid*2);
        half2* aa=reinterpret_cast<half2*>(a+tid*2);
        half2* bb=reinterpret_cast<half2*>(b+tid*2);
        *cc=__hadd2(*aa,*bb);
    }}
template <typename T>
__global__ void vec_add2(T* a,T* b,T* c){
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    for(int i=tid;i<100;i+=blockDim.x){
        c[i]=a[i]+b[i];
    }
    
}


int main(){
    half* a;
    half* b;
    half* c;
    int size=100*sizeof(half);
    a=(half*)malloc(size);
    b=(half*)malloc(size);
    c=(half*)malloc(size); 
    half* da;
    half* db;
    half* dc;
    cudaMalloc(&da,size);
    cudaMalloc(&db,size);
    cudaMalloc(&dc,size);
    for(int i=0;i<100;i++){
        a[i]=__float2half(1.0f);
        b[i]=__float2half(2.0f);
    }
    cudaMemcpy(da,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(db,b,size,cudaMemcpyHostToDevice);
    cudaMemcpy(dc,c,size,cudaMemcpyHostToDevice);
    dim3 grid(1);
    dim3 block(25);
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始事件
    cudaEventRecord(start, 0);

    // 调用核函数
    vec_add<half><<<grid, block>>>(da, db, dc);

    // 记录结束事件
    cudaEventRecord(stop, 0);

    // 等待结束事件完成
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    // 计算时间差
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("所花费的时间为：%f\n",milliseconds);
    cudaMemcpy(c,dc,size,cudaMemcpyDeviceToHost);
   // cudaMemory(b,db,size,cudaMemoryDeviceToHost);
    for(int i=0;i<100;i++){
        printf("%f  ",__half2float(c[i]));        
    }

    free(a);
    free(b);
    free(c);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return;

}

