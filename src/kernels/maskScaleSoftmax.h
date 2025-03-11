#pragma once 
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_fp16.h>
#include<src/utils/tensor.h>

template <typename T>
void launchMaskScaleSoftmax(TensorWrapper<T> *out,
                            TensorWrapper<T> *in,
                            TensorWrapper<T> *mask,
                            float scale);