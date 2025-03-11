#include"src/kernels/topk.h"
#include<cub/cub.cuh>

template <typename T,int K>
__device__ topk<T,K> reduce_functor(topk<T,K> &a,topk<T,K> &b){
    topk<T,K> ans=a;
    for(int i=0;i<K;i++){
        ans.insertHeap(b.val[i],b.id[i]);
    }
    return ans;
}

template<typename T,int K, int blocksize,int blockPerBeam>
__global__ void firstTopk(T* probs,int* temptopk_id,T* temptopk_val,const int vocab_size){
    int bid=blockIdx.x;
    int tid=threadIdx.x;
    int row_id=bid/blockPerBeam;
    int b_lane=bid%blockPerBeam;
    
    typedef cub::BlockReduce<topk<T,K>,blocksize> blockreduce;
    __shared__ typename blockreduce::TempStorage temp_storage;  //凡是用了模板参数的类的名字进行构造的时候我们需要考虑typename

    topk<T,K> thread_topk;
    thread_topk.init();
    for(int i=tid+b_lane*blocksize;i<vocab_size;i+=blockPerBeam*blocksize){
        int data_offset=row_id*vocab_size+i;
        T data=probs[data_offset];
        thread_topk.insertHeap(data,i);  //这里的i就是表示这一派的第几个；

    }
    topk<T,K> block_topk=blockreduce(temp_storage).Reduce(thread_topk,reduce_functor<T,K>);
    if(tid==0){
        for(int i=0;i<K;i++){
            temptopk_id[row_id*blockPerBeam*K+b_lane*K+i]=block_topk.id[i];
            temptopk_val[row_id*blockPerBeam*K+b_lane*K+i]=block_topk.val[i];
        }
    }
}

template <typename T,int K,int blocksize,int blockPerBeam>
__global__ void secondTopk(int *temptopk_id,T *temptopk_val,int* fitopk_id,T* fitopk_val){
    int bid=blockIdx.x;
    int tid=threadIdx.x;
    topk<T,K> thread_topk;
    thread_topk.init();
    for(int i=tid;i<K*blockPerBeam;i+=blocksize){
        int data_offset=bid*K*blockPerBeam+i;
        thread_topk.insertHeap(temptopk_val[data_offset],temptopk_id[data_offset]);
    }
    typedef cub::BlockReduce<topk<T,K>,blocksize> blockreduce;
    __shared__ typename blockreduce::TempStorage temp_storage;

    topk<T,K> block_topk=blockreduce(temp_storage).Reduce(thread_topk,reduce_functor<T,K>);

    if(tid==0){
        for(int i=0;i<K;i++){
            fitopk_id[bid*K+i]=block_topk.id[i];
            fitopk_val[bid*K+i]=block_topk.val[i];
        }
    }
}

template <typename T>
void launchTopk(TensorWrapper<T> *probs,TensorWrapper<int> *temptopk_id,TensorWrapper<T> *temptopk_val,TensorWrapper<int> *fitopk_id,
                TensorWrapper<T> *fitopk_val){
    int bswd=probs->shape[0];
    int vocab_size=probs->shape[1];
    constexpr int blockPerBeam=8;
    constexpr int K=5;
    dim3 grid1(min(bswd*blockPerBeam,1024));
    dim3 block1(256);

    dim3 grid2(min(bswd,1024));
    dim3 block2(256);

    firstTopk<T,K,256,blockPerBeam><<<grid1,block1>>>(probs->data,temptopk_id->data,temptopk_val->data,vocab_size);
    secondTopk<T,K,256,blockPerBeam><<<grid2,block2>>>(temptopk_id->data,temptopk_val->data,fitopk_id->data,fitopk_val->data);
}   

template void launchTopk(TensorWrapper<float> *probs,TensorWrapper<int> *temptopk_id,TensorWrapper<float> *temptopk_val,TensorWrapper<int> *fitopk_id,
                TensorWrapper<float> *fitopk_val);

// #include <float.h> //FLT_MIN
// #include <cuda.h>
// #include <iostream>
// #include "src/kernels/topk.h"
// #include <cub/cub.cuh>

// // Note: a b两个topK reduce输出一个topK
// template<typename T, int K>
// __device__ topk<T, K> reduce_functor(const topk<T, K>& a, const topk<T, K>& b) {
//     topk<T, K> res = a;
//     for(int i = 0; i < K; i++){
//         res.insertHeap(b.val[i], b.id[i]);
//     }
//     return res;
// }
// // gridsize:bs * beamwidth * BlockPerBeam 
// // blocksize: 256
// // shape infer: [bs, beamwidth, vocab size] => [bs, beamwidth, BlockPerBeam, K]
// template<typename T, int K, int blockSize, int BlockPerBeam>
// __global__ void topK_kernel_round1(const T* probs, const int vocab_size, 
//                                          int* topK_ids, T* topK_vals)
// {
//     typedef cub::BlockReduce<topk<T, K>, blockSize> blockreduce;
//     __shared__ typename blockreduce::TempStorage temp_storage;

//     int tid = threadIdx.x;
//     int bid = blockIdx.x;
//     int gid = blockIdx.x * blockDim.x + threadIdx.x;
//     int row_id = bid / BlockPerBeam;
//     int block_lane = bid % BlockPerBeam;
//     topk<T, K> thread_topK;
//     thread_topK.init();
//     // thread local reduce
//     for(int data_id = tid + block_lane * blockSize; data_id < vocab_size; data_id += BlockPerBeam * blockSize){
//         int data_offset = data_id + row_id * vocab_size;
//         T data = probs[data_offset];
//         //thread_topK.insertHeap(data, data_offset); // bug, id should be local in bsxbm, if use this line, assume bsxbm=2, prob=1-50000,the 2nd bsxbm res topk id will be 59999,59998..., but in bsxbm internal, this id will be 29999,29998... rather than not global id
//         thread_topK.insertHeap(data, data_id); 
//     }
//     //block local reduce
//     topk<T, K> block_topK = blockreduce(temp_storage).Reduce(thread_topK, reduce_functor<T, K>);

//     if(tid == 0){
//         for(int k_offset = 0; k_offset < K; k_offset++) {
//             // topK_vals[row_id * vocab_size + block_lane * blockSize + k_offset] = block_topK.val[k_offset]; //bug
//             topK_vals[row_id * BlockPerBeam * K + block_lane * K + k_offset] = block_topK.val[k_offset];
//             topK_ids[row_id * BlockPerBeam * K  + block_lane * K + k_offset] = block_topK.id[k_offset];//output offset要根据output buffer的shape来计算

//         }
//     }
// }
// // shape infer: [bs, beamwidth, BlockPerBeam, K] => [bs, beamwidth, K]
// // ids是beam width * vocab size中的全局word id
// // gridSize = bs
// // blockSize = 256
// template<typename T, int K, int blockSize, int BlockPerBeam>
// __global__ void topK_kernel_round2(const int* topK_ids, const T* topK_vals,
//                                     int* final_topK_ids, T* final_topK_vals)
// {
//     typedef cub::BlockReduce<topk<T, K>, blockSize> blockreduce;
//     __shared__ typename blockreduce::TempStorage temp_storage;

//     int tid = threadIdx.x;
//     int bid = blockIdx.x;
//     int gid = blockIdx.x * blockDim.x + threadIdx.x;
//     int row_id = bid;
//     topk<T, K> thread_topK;
//     // thread local reduce    
//     for(int i = tid; i < BlockPerBeam * K; i += blockDim.x) {
//         int data_offset = bid * BlockPerBeam * K + i;
//         thread_topK.insertHeap(topK_vals[data_offset], topK_ids[data_offset]);
//     }
//     // block reduce
//     topk<T, K> block_topK = blockreduce(temp_storage).Reduce(thread_topK, reduce_functor<T, K>);
//     if(tid == 0){
//         for(int k_offset = 0; k_offset < K; k_offset++) {
//             // topK_vals[row_id * vocab_size + block_lane * blockSize + k_offset] = block_topK.val[k_offset]; //bug
//             final_topK_vals[bid * K + k_offset] = block_topK.val[k_offset];
//             final_topK_ids[bid * K + k_offset] = block_topK.id[k_offset];
//         }
//     }    
// }

// template <typename T>
// void launchTopk(TensorWrapper<T> *probs,
//                              TensorWrapper<int> *topk_ids,
//                              TensorWrapper<T> *topk_vals,
//                              TensorWrapper<int> *final_topk_ids,
//                              TensorWrapper<T> *final_topk_vals)
// {
//     // support both beamserach and sampling topk by integrate beamwidth into batchsize, we get variable bsxbw = bs*bw, the probs shape is [bs*bw, vocabsize]
//     int bsxbm = probs->shape[0];
//     int vocab_size = probs->shape[1];
//     constexpr int BlockPerBeam = 8;
//     constexpr int beamwidth = 1;
//     constexpr int K = 5;
//     // buffer size
//     int topK_val_buf_size = bsxbm * BlockPerBeam * K;
//     int topK_ids_buf_size = bsxbm * BlockPerBeam * K;
//     int final_topK_val_buf_size = bsxbm * K;
    
//     T* topK_vals = topk_vals->data;
//     int* topK_ids = topk_ids->data;
//     T* final_topK_vals = final_topk_vals->data;
//     int* final_topK_ids = final_topk_ids->data;    
//     // prepare launch
//     // TODO: add GPUconfig API to easily get GPU config, ep: maxblocknums
//     // GPUConfig config;
//     // int maxBlockNums = config.getMaxBlockNums();
//     // TODO: how to alloc block nums more flexable according to shape
//     //constexpr int BlockPerBeam = 8;
//     int maxBlockNums = 1024;
//     int BlockNums1 = std::min(bsxbm * BlockPerBeam, maxBlockNums);
//     int BlockNums2 = std::min(bsxbm, maxBlockNums);
//     dim3 grid_round1(BlockNums1);
//     dim3 block_round1(256);
//     dim3 grid_round2(BlockNums2);
//     dim3 block_round2(256);
//     // debug info, better to retain: std::cout << "in cu file, before launch" << std::endl;
//     topK_kernel_round1<T, K, 256, BlockPerBeam>
//                         <<<grid_round1, block_round1>>>(probs->data, vocab_size, topK_ids, topK_vals);
//     topK_kernel_round2<T, K, 256, BlockPerBeam>
//                         <<<grid_round2, block_round2>>>(topK_ids, topK_vals, final_topK_ids, final_topK_vals);
//     // debug info, better to retain: std::cout << "in cu file, after launch" << std::endl;
// }

// template void launchTopk(TensorWrapper<float> *probs,
//                              TensorWrapper<int> *topk_ids,
//                              TensorWrapper<float> *topk_vals,
//                              TensorWrapper<int> *final_topk_ids,
//                              TensorWrapper<float> *final_topk_vals);

// template void launchTopk(TensorWrapper<half> *probs,
//                              TensorWrapper<int> *topk_ids,
//                              TensorWrapper<half> *topk_vals,
//                              TensorWrapper<int> *final_topk_ids,
//                              TensorWrapper<half> *final_topk_vals);