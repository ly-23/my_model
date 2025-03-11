#pragma once
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/weights/llama/rmsnorm_weights.h"

template <typename T>
void launchRMSNorm(TensorWrapper<T>* output,TensorWrapper<T>* residual,rmsNormWeight<T>* scale,float eps,bool is_last=false);