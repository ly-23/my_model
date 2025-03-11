#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_fp16.h>
#include<src/utils/tensor.h>
#include<src/weights/llama/embedding_weights.h>

template<typename T>
void launchEmbedding(TensorWrapper<int>* input,TensorWrapper<T> *output,EmbeddingWeight<T> *embed_table);

