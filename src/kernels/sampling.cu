#include"src/kernels/sampling.h"


template<typename T>
__global__ void sampling(int* topk_id,T* topk_val,int* seqlen,bool* is_finished,int* output_id,int batch_size,int K,int vocab_size,
                int step,int end_id){
    

    int bid=blockIdx.x;
    int tid=threadIdx.x;
    T max_val=topk_val[bid*K];
    topk_val[bid*K+tid]=expf(topk_val[bid*K+tid]-max_val);
    __shared__ float thredhold, sum;
    if(tid==0){
        sum=0.0f;
        for(int i=0;i<K;i++){
            sum+=(float)topk_val[bid*K+i];
        }
        curandState_t state;
        curand_init((unsigned long long)step,(unsigned long long)bid,(unsigned long long)0,&state);
        thredhold=(float)curand_uniform(&state)*sum;
        output_id[bid]=topk_id[bid*K]%vocab_size;
        for(int i=0;i<K;i++){
            thredhold=thredhold-topk_val[bid*K+i];
            if(thredhold<0);
            output_id[bid]=topk_id[bid*K+i]%vocab_size;
            break;
        }
        seqlen[bid] = is_finished[bid] ? seqlen[bid] : seqlen[bid] + 1;
        is_finished[bid] = output_id[bid] == end_id ? 1 : 0;

    }
}

template <typename T>
void launchSampling(TensorWrapper<int> *topk_id,
                    TensorWrapper<T> *topk_val,
                    TensorWrapper<int> *seqlen,
                    TensorWrapper<bool> *is_finished,
                    TensorWrapper<int> *output_id,
                    IntDict& params){
    int batch_size=topk_id->shape[0];
    int K=topk_id->shape[1];
    
    int vocab_size = params["vocab_size"];
    int step = params["step"];
    int end_id = params["end_id"];

    dim3 grid(batch_size);
    dim3 block(K);
    sampling<T><<<grid,block>>>(topk_id->data,topk_val->data,seqlen->data,is_finished->data,output_id->data,batch_size,K,vocab_size,step,end_id);
                        


}

template void launchSampling(TensorWrapper<int> *topk_id,
                    TensorWrapper<float> *topk_val,
                    TensorWrapper<int> *seqlen,
                    TensorWrapper<bool> *is_finished,
                    TensorWrapper<int> *output_id,
                    IntDict& params);