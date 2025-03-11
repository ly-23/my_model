#include "src/kernels/embedding.h"
#include<stdio.h>
template<typename T>
__global__ void Embedding(int* input,T* output,T* embed_table,int token_nums,int hidden_units){
    int bid=blockIdx.x;
    int tid=threadIdx.x;
    if(tid<hidden_units){
        for(int i=tid;i<hidden_units;i=i+blockDim.x){
            output[bid*hidden_units+i]=embed_table[input[bid]*hidden_units+i];
            //if(tid==0&&bid==0)printf("%f",output[bid*hidden_units+i]);
        }
    }
}


template<typename T>
void launchEmbedding(TensorWrapper<int>* input,TensorWrapper<T> *output,EmbeddingWeight<T> *embed_table){
    int token_nums=output->shape[0];//std::cout << "caaaaa55aaaaaaaaaaa" << std::endl;
   // int vocab_nums=embed_table.shape[0];std::cout << "caaaaa55aaaaaaaaaaa" << std::endl;
    int hidden_units=output->shape[1];
    //std::cout << "caaaaa55aaaaaaaaaaa" << std::endl;
    dim3 grid(token_nums);
    dim3 block(1024);
    //std::cout << "caaaaaaaa55aaaaaaaa" << std::endl;
    Embedding<T><<<grid,block>>>(input->data,output->data,embed_table->data,token_nums,hidden_units);
}

template void launchEmbedding(TensorWrapper<int>* input_ids,TensorWrapper<float>* output,EmbeddingWeight<float> *embed_table);

// #include<stdio.h>
// #include"src/kernels/embedding.h"

// template<typename T>
// __global__ void embeddingFunctor(const int* input_ids,
//                                 T* output,
//                                 const T* embed_table,
//                                 const int max_context_token_num,
//                                 const int hidden_size)
// {   
//     int index=blockIdx.x*blockDim.x+threadIdx.x;
//     while(index<max_context_token_num*hidden_size){
//         int id=input_ids[index/hidden_size];
//         output[index]=embed_table[id*hidden_size+index%hidden_size];
//         index+=blockDim.x*gridDim.x;
//     }
// }


//  template<typename T>   //这种不能出现在.cpp文件里面的
//  void launchEmbedding(TensorWrapper<int>* input_ids,TensorWrapper<T>* output,EmbeddingWeight<T>* embed_table){
//     const int blockSize=256;
//     const int max_context_token_num=output->shape[0];
//     const int hidden_size=output->shape[1];
//     const int gridSize=2048;
//     LLM_CHECK_WITH_INFO(max_context_token_num == input_ids->shape[0], "input ids 1st shape should equal to 1st shape of output");
//     embeddingFunctor<T><<<gridSize, blockSize>>>(input_ids->data,
//                                                  output->data,
//                                                  embed_table->data,
//                                                  max_context_token_num,
//                                                  hidden_size);

//  }

//  template void launchEmbedding(TensorWrapper<int>* input_ids,TensorWrapper<float>* output,EmbeddingWeight<float>* embed_table);

//  template void launchEmbedding(TensorWrapper<int>* input_ids,TensorWrapper<half>* output,EmbeddingWeight<half>* embed_table);
