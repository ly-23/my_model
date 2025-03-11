#pragma once 
#include<cuda.h>
#include<cuda_fp16.h>
#include<cuda_runtime.h>
#include "src/utils/tensor.h"
#include<float.h>
template <typename T,int K>
struct topk{
    T val[K];
    int id[K];
    
    __device__ void init(){
        for(int i=0;i<K;i++){
            id[i]=-1;
            val[i]=FLT_MIN;
        }
    }

    __device__ void insertHeap(T data,int data_id){
        if(id[K-1]==-1||val[K-1]<data){
            id[K-1]=data_id;//每次都是插入到第末尾
            val[K-1]=data;
        }
        for(int i=K-2;i>=0;i--){
            if(val[i+1]>val[i]||id[i]==-1){
                T tmp=val[i];
                val[i]=val[i+1];
                val[i+1]=tmp;
                int tmp_id=id[i];
                id[i]=id[i+1];
                id[i+1]=tmp_id;
            }
        }
    }
};


template <typename T>
void launchTopk(TensorWrapper<T> *probs,TensorWrapper<int> *temptopk_id,TensorWrapper<T> *temptopk_val,TensorWrapper<int> *fitopk_id,
                TensorWrapper<T> *fitopk_val);