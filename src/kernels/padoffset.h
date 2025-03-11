#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_fp16.h>
#include<src/utils/tensor.h>


void launchpadoffset(TensorWrapper<int>* padding_offset,TensorWrapper<int>* cum_seqlens,TensorWrapper<int>* input_lengths);