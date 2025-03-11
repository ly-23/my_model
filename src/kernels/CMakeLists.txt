add_library(embeddingFunctor STATIC embedding.cu)
set_property(TARGET embeddingFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET embeddingFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET embeddingFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(padoffsetFunctor STATIC padoffset.cu)
set_property(TARGET padoffsetFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET padoffsetFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET padoffsetFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(rmsnormFunctor STATIC rmsnorm.cu)
set_property(TARGET rmsnormFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET rmsnormFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET rmsnormFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(maskFunctor STATIC mask.cu)
set_property(TARGET maskFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET maskFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET maskFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(cublasWrapper STATIC cublas_utils.cc)
set_property(TARGET cublasWrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasWrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(linearFunctor STATIC linear.cu)
set_property(TARGET linearFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET linearFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET linearFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(linearFunctor PUBLIC -lcudart -lcublas cublasWrapper)

add_library(padropeFunctor STATIC padrope.cu)
set_property(TARGET padropeFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET padropeFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET padropeFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(concatkvFunctor STATIC concat_kv.cu)
set_property(TARGET concatkvFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET concatkvFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET concatkvFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(repeatkvFunctor STATIC repeat_kv.cu)
set_property(TARGET repeatkvFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET repeatkvFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET repeatkvFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(maskscalesoftmaxFunctor STATIC maskScaleSoftmax.cu)
set_property(TARGET maskscalesoftmaxFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET maskscalesoftmaxFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET maskscalesoftmaxFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(transremvFunctor STATIC transremv.cu)
set_property(TARGET transremvFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET transremvFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET transremvFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(swigluFunctor STATIC swiglu.cu)
set_property(TARGET swigluFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET swigluFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET swigluFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(decoderattnFunctor STATIC decoderattn.cu)
set_property(TARGET decoderattnFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET decoderattnFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET decoderattnFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(topkFunctor STATIC topk.cu)
set_property(TARGET topkFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET topkFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET topkFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(samplingFunctor STATIC sampling.cu)
set_property(TARGET samplingFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET samplingFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET samplingFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
