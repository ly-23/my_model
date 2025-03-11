#pragma once 
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/weights/base_weight.h"
#include "src/model/llama/llama_param.h"


template<typename T>
void launchPadRope(TensorWrapper<T> *q_buf, 
                                   TensorWrapper<T> *k_buf,
                                   TensorWrapper<T> *v_buf,
                                   TensorWrapper<T> *QKV,
                                   BaseWeight<T>    &qkv,
                                   TensorWrapper<int> *padding_offset,
                                   TensorWrapper<int> *history_length, 
                                   TensorWrapper<int> *input_length,
                                   LLamaAttentionStaticParams &param   
                                );