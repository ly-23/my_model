#pragma once 
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cublasLt.h>
#include "src/utils/macro.h"
class cublasWrapper{
    private:
        cublasHandle_t   cublas_handle;
        cublasLtHandle_t cublaslt_handle;
        cudaDataType_t Atype;
        cudaDataType_t Btype;
        cudaDataType_t Ctype;
        cublasComputeType_t computeType;
    
    public:
        cublasWrapper(cublasHandle_t cublas_handle,cublasLtHandle_t cublaslt_handle);
        ~cublasWrapper();
        void setFP32GemmConfig();
        void setFP16GemmConfig();
        void Gemm(cublasOperation_t transa,
                cublasOperation_t transb,
                const int m,
                const int n,
                const int k,
                const void* A,
                const int lda,
                const void* B,
                const int ldb,
                void* C,
                const int ldc,
                float alpha,
                float beta);
         
    
        void stridedBatchedGemm(cublasOperation_t transa,
                                cublasOperation_t transb,
                                const int m,
                                const int n,
                                const int k,
                                const void* A,
                                const int lda,
                                const int64_t strideA,
                                const void* B,
                                const int ldb,
                                const int64_t strideB,
                                void* C,
                                const int ldc,
                                const int64_t strideC,
                                const int batchCount,
                                float f_alpha,
                                float f_beta
                                    );
};