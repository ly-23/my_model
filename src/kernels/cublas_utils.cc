#include "src/kernels/cublas_utils.h"

 
cublasWrapper::cublasWrapper(cublasHandle_t cublas_handle,cublasLtHandle_t cublaslt_handle):cublas_handle(cublas_handle),cublaslt_handle(cublaslt_handle){}

cublasWrapper::~cublasWrapper(){}


void cublasWrapper::setFP32GemmConfig(){
    Atype=CUDA_R_32F;
    Btype=CUDA_R_32F;
    Ctype=CUDA_R_32F;
    computeType=CUBLAS_COMPUTE_32F;
}

void cublasWrapper::setFP16GemmConfig(){
    Atype=CUDA_R_16F;
    Btype=CUDA_R_16F;
    Ctype=CUDA_R_16F;
    computeType=CUBLAS_COMPUTE_16F;
}

void cublasWrapper::Gemm(cublasOperation_t transa,
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
                float f_alpha=1.0f,
                float f_beta=0.0f)
{
    half h_alpha=(half)(f_alpha);
    half h_beta=(half)(f_beta);
    int is_fp16_computeType=computeType==CUBLAS_COMPUTE_16F?1:0;
    const void* alpha=is_fp16_computeType?reinterpret_cast<void*>(&(h_alpha)):reinterpret_cast<void*>(&(f_alpha));
    const void* beta=is_fp16_computeType?reinterpret_cast<void*>(&(h_beta)):reinterpret_cast<void*>(&(f_beta));
    cublasGemmEx(cublas_handle,
                transa,
                transb, //用于表示a和b是否需要转置。
                m,      //行业潜规则：A(m,k) B(k,n) C(m,n)
                n,
                k,
                alpha,
                A,
                Atype,
                lda,  //全程叫做 leading dimension 此处为m.
                B,
                Btype,
                ldb,
                beta,
                C,
                Ctype,
                ldc,
                computeType,
            CUBLAS_GEMM_DEFAULT);
                
}
         
    
void cublasWrapper::stridedBatchedGemm(cublasOperation_t transa,
    cublasOperation_t transb,
    const int         m,
    const int         n,
    const int         k,
    const void*       A,
    const int         lda,
    const int64_t     strideA,
    const void*       B,
    const int         ldb,
    const int64_t     strideB,
    void*             C,
    const int         ldc,
    const int64_t     strideC,
    const int         batchCount,
    float       f_alpha = 1.0f,
    float       f_beta  = 0.0f)
{                                     //这部分代码有待商榷
int is_fp16_computeType = computeType == CUDA_R_16F ? 1 : 0;
const void* alpha =
is_fp16_computeType ? reinterpret_cast<void*>(&(f_alpha)) : reinterpret_cast<const void*>(&f_alpha);
const void* beta = is_fp16_computeType ? reinterpret_cast<void*>(&(f_beta)) : reinterpret_cast<const void*>(&f_beta);
CHECK_CUBLAS(cublasGemmStridedBatchedEx(cublas_handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A,
        Atype,
        lda,
        strideA, //batch dim 假说[1,2,3,4] 那么strideA=3*4;
        B,
        Btype,
        ldb,
        strideB,
        beta,
        C,
        Ctype,
        ldc,
        strideC,
        batchCount, //[1,2,3,4] batchcount=1*2;
        computeType,
        CUBLAS_GEMM_DEFAULT));
}