#include "src/kernels/linear.h"

template <typename T>
void launchLinearGemm(TensorWrapper<T> *input,
                        BaseWeight<T> &weight,
                        TensorWrapper<T> *output,
                        cublasWrapper *cublas_wrapper,
                        bool trans_a,
                        bool trans_b)
{       //y^T=w*x^T   这里的意思就是cublas一来会默认有一个转置
    int Am=weight.shape[1];
    int Ak=weight.shape[0];
    int Bk=input->shape[1];
    int Bn=input->shape[0];
    int Cm=output->shape[1];
    int Cn=output->shape[0];

    Bk=input->shape.size()==3?input->shape[1]*input->shape[2]:input->shape[1];
    Cm=output->shape.size()==3?output->shape[1]*output->shape[2]:output->shape[1];

    int lda=Am;    //这里有个很傻逼的点，就是m,n，k是需要去考虑转置，而lda，ldb,ldc则不需要考虑转置。
    int ldb=Bk;
    int ldc=Cm;

    cublasOperation_t transA=trans_b?CUBLAS_OP_T:CUBLAS_OP_N;
    cublasOperation_t transB=trans_a?CUBLAS_OP_T:CUBLAS_OP_N;
    
    cublas_wrapper->Gemm(transA,
                        transB,
                        trans_b?Ak:Am,
                        Cn,
                        Bk,
                        weight.data,
                        lda,
                        input->data,
                        ldb,
                        output->data,
                        ldc,
                        1.0f,
                        0.0f);
}
template void launchLinearGemm(TensorWrapper<float> *input,
    BaseWeight<float> &weight,
    TensorWrapper<float> *output,
    cublasWrapper *cublas_wrapper,
    bool trans_a,
    bool trans_b);


template <typename T>
void launchBatchLinearGemm(TensorWrapper<T> *input1,
                            TensorWrapper<T> *input2,
                            TensorWrapper<T> *output,
                            cublasWrapper *cublas_wrapper,
                            bool trans_a,
                            bool trans_b){
int Am=input2->shape[3];
int Ak=input2->shape[2];
int Bk=input1->shape[3];
int Bn=input1->shape[2];
int Cm=output->shape[3];
int Cn=output->shape[2];

int lda=Am;
int ldb=Bk;
int ldc=Cm;

int64_t strideA=Ak*Am;
int64_t strideB=Bk*Bn;
int64_t strideC=Cm*Cn;

int batchcount=input1->shape[0]*input1->shape[1];

cublasOperation_t transA=trans_b?CUBLAS_OP_T:CUBLAS_OP_N;
cublasOperation_t transB=trans_a?CUBLAS_OP_T:CUBLAS_OP_N;

cublas_wrapper->stridedBatchedGemm(
        transA,
        transB,
        Cm,
        Cn,
        Bk,
        input2->data,
        lda,
        strideA,
        input1->data,
        ldb,
        strideB,
        output->data,
        ldc,
        strideC,
        batchcount,
        1.0f,
        0.0f
);

}
template void launchBatchLinearGemm(TensorWrapper<float> *input1,
                            TensorWrapper<float> *input2,
                            TensorWrapper<float> *output,
                            cublasWrapper *cublas_wrapper,
                            bool trans_a,
                            bool trans_b);               




