#pragma once
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_fp16.h>
#include "src/utils/tensor.h"

template <typename T>
void launchRepeatKV(TensorWrapper<T> *k_cache_src,
                    TensorWrapper<T> *v_cache_src,
                    TensorWrapper<T> *k_out,
                    TensorWrapper<T> *v_out,
                    TensorWrapper<int> *layer_id,
                    TensorWrapper<int> *context_length 
                    );