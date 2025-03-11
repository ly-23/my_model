#pragma once 
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_fp16.h>
#include "src/utils/tensor.h"

template <typename T>
void launchSwiglu(TensorWrapper<T> *input,TensorWrapper<T> *output);