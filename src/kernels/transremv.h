#pragma once 
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_fp16.h>
#include "src/utils/tensor.h"

template<typename T>
void launchTransRemv(TensorWrapper<T> *in,TensorWrapper<T> *out,TensorWrapper<int> *padding_offset);
