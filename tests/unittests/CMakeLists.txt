add_executable(embedding test_embedding.cu)

target_link_libraries(
    embedding PUBLIC
    -lcudart
    -lcudadevrt
    embeddingFunctor
)


add_executable(padoffset test_padoffset.cu)

target_link_libraries(
    padoffset PUBLIC
    -lcudart
    -lcudadevrt
    padoffsetFunctor
)


add_executable(rmsnorm test_rmsnorm.cu)

target_link_libraries(
    rmsnorm PUBLIC
    -lcudart
    -lcudadevrt
    rmsnormFunctor
)

add_executable(mask test_mask.cu)

target_link_libraries(
    mask PUBLIC
    -lcudart
    -lcudadevrt
    maskFunctor
)

add_executable(linear test_linear.cu)

target_link_libraries(
   linear PUBLIC
   -lcudart
   -lcudadevrt
   linearFunctor
)

add_executable(padrope test_padrope.cu)

target_link_libraries(
   padrope PUBLIC
   -lcudart
   -lcudadevrt
   padropeFunctor
)

add_executable(concatkv test_concatkv.cu)

target_link_libraries(
   concatkv PUBLIC
   -lcudart
   -lcudadevrt
   concatkvFunctor
)

add_executable(repeatkv test_repeatkv.cu)

target_link_libraries(
   repeatkv PUBLIC
   -lcudart
   -lcudadevrt
   repeatkvFunctor
)

add_executable(maskscalesoftmax test_maskscalesoftmax.cu)

target_link_libraries(
   maskscalesoftmax PUBLIC
   -lcudart
   -lcudadevrt
   maskscalesoftmaxFunctor
)

add_executable(transremv test_transremv.cu)

target_link_libraries(
   transremv PUBLIC
   -lcudart
   -lcudadevrt
   transremvFunctor
)


add_executable(swiglu test_swiglu.cu)

target_link_libraries(
   swiglu PUBLIC
   -lcudart
   -lcudadevrt
   swigluFunctor
)

add_executable(decoderattn test_decoderattn.cu)

target_link_libraries(
   decoderattn PUBLIC
   -lcudart
   -lcudadevrt
   decoderattnFunctor
)

add_executable(topk test_topk.cu)

target_link_libraries(
   topk PUBLIC
   -lcudart
   -lcudadevrt
   topkFunctor
)

add_executable(sampling test_sampling.cu)

target_link_libraries(
   sampling PUBLIC
   -lcudart
   -lcudadevrt
   samplingFunctor
)