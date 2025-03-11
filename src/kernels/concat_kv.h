#pragma once
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_fp16.h>
#include "src/utils/tensor.h"


template<typename T>
void launchConcatKV(TensorWrapper<T> *k_src,
                    TensorWrapper<T> *v_src,
                    TensorWrapper<T> *k_dst,
                    TensorWrapper<T> *v_dst,
                    TensorWrapper<int> *cur_query_lens,
                    TensorWrapper<int> *history_lens,
                    TensorWrapper<int> *layer_id
                    );
