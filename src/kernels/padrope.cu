#include "src/kernels/padrope.h"


inline __device__ float2 GetRoPEfreq(int zid, int rot_embed_dim, float base, float t_step)
{
    // (RussWong) note: 每个token所属的id, 它的freq值都是固定的, id的上限为max position embedding
    // t_step表示token id（这里考虑了多轮对话历史上下文长度)
    // 每个freq值对应于zid = head size维度上0 2 4 6 ... 64带入下式计算
    const float inv_freq = t_step / powf(base, zid / (float)rot_embed_dim);  //t是指的对话长度
    return {cos(inv_freq), sin(inv_freq)};
}

inline __device__ float2 GetRoPEres(float data, float data_rotate, const float2 coef)
{
    float2 rot_v;
    rot_v.x = coef.x * data - coef.y * data_rotate;
    rot_v.y = coef.x * data_rotate + coef.y * data;
    return rot_v;
}

template <typename T>
__global__ void PadRope(T* q_buf,T* k_buf,T* v_buf,T* QKV,T* qkv,int* padding_offset,int* history_length,int* input_length,
    int num_tokens,int head_nums,int kv_head_nums,int max_q_len,int batch_size,int head_size,int rotary_embedding_dim, int rotary_embedding_base){
    
    int num_token_id=blockIdx.x;
    int head_num_id=blockIdx.y;
    int tid=threadIdx.x;
    int token_padding_offset=padding_offset[num_token_id];

    int dst_token_id=num_token_id+token_padding_offset;
    int batch_id=dst_token_id/max_q_len;
    int batch_token_id=dst_token_id%max_q_len;

    //注意这里并没有使用qkv，因为llama2中没有bias
    int rsc_q_id=num_token_id*(head_nums+2*kv_head_nums)*head_size+head_num_id*head_size+tid;
    int dst_q_id=batch_id*head_nums*max_q_len*head_size+head_num_id*max_q_len*head_size+batch_token_id*head_size+tid;

    int rsc_k_id=num_token_id*(head_nums+2*kv_head_nums)*head_size+(head_nums+head_num_id)*head_size+tid;
    int rsc_v_id=num_token_id*(head_nums+2*kv_head_nums)*head_size+(head_nums+kv_head_nums+head_num_id)*head_size+tid;

    int dst_kv_id=batch_id*kv_head_nums*max_q_len*head_size+head_num_id*max_q_len*head_size+batch_token_id*head_size+tid;

    //q_buf[dst_q_id]=QKV[rsc_q_id]
    if(head_num_id<kv_head_nums){  
        v_buf[dst_kv_id]=QKV[rsc_v_id];
    }
    const int cur_seq_history_len = history_length[batch_id];
    const int context_length = cur_seq_history_len + input_length[batch_id];
    //（RussWong)note: 多轮对话下要结合history length求得全局的cos和sin
    const int timestep = cur_seq_history_len + batch_token_id; 
    // (RussWong)note: timestep为cos(m*theta)中的m
    if (tid >rotary_embedding_dim / 2)
    {
        return;}
     // tid = [0,1,2,...,63]

    float2 cos_sin = GetRoPEfreq(tid * 2, rotary_embedding_dim, rotary_embedding_base, timestep);
    float2 q_rotate = GetRoPEres(QKV[rsc_q_id], QKV[rsc_q_id + head_size / 2], cos_sin);
    float2 k_rotate = GetRoPEres(QKV[rsc_k_id], QKV[rsc_k_id + head_size / 2], cos_sin);
    q_buf[dst_q_id] = q_rotate.x;
    q_buf[dst_q_id + head_size / 2] = q_rotate.y;
    if (head_num_id < kv_head_nums)
    { // for MQA and GQA
        k_buf[dst_kv_id] = k_rotate.x;
        k_buf[dst_kv_id + head_size / 2] = k_rotate.y;
    }
}







template <typename T>
void launchPadRope(TensorWrapper<T> *q_buf, 
                   TensorWrapper<T> *k_buf,
                   TensorWrapper<T> *v_buf,
                   TensorWrapper<T> *QKV,
                   BaseWeight<T>    &qkv,
                   TensorWrapper<int> *padding_offset,
                   TensorWrapper<int> *history_length, 
                   TensorWrapper<int> *input_length,
                   LLamaAttentionStaticParams &param   
                ){
    int batch_size=q_buf->shape[0];
    int head_nums=q_buf->shape[1];
    int max_q_len=q_buf->shape[2];
    int head_size=q_buf->shape[3];
    int num_tokens=QKV->shape[0];
    int kv_head_nums=k_buf->shape[1];
    int qkv_head_nums=QKV->shape[1];
    int rotary_embedding_dim=param.rotary_embedding_dim;
    float rotary_embedding_base=param.rotary_embedding_base;
    int max_position_embeddings=param.max_position_embeddings;
    
    dim3 grid(num_tokens,head_nums);
    dim3 block(head_size);
    PadRope<T><<<grid,block>>>(q_buf->data,k_buf->data,v_buf->data,QKV->data,qkv.bias,padding_offset->data,history_length->data,input_length->data,
         num_tokens, head_nums, kv_head_nums, max_q_len, batch_size, head_size, rotary_embedding_dim,  rotary_embedding_base);
}

template void launchPadRope(TensorWrapper<float> *q_buf, 
                   TensorWrapper<float> *k_buf,
                   TensorWrapper<float> *v_buf,
                   TensorWrapper<float> *QKV,
                   BaseWeight<float>    &qkv,
                   TensorWrapper<int> *padding_offset,
                   TensorWrapper<int> *history_length, 
                   TensorWrapper<int> *input_length,
                   LLamaAttentionStaticParams &param   
                );


                // #include <math.h>
                // #include <stdio.h>
                // //#include "src/utils/cuda_debug_utils.cuh"
                // #include "src/kernels/padrope.h"
                // // HF python code:
                // //    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
                // //         """Compute the inverse frequency."""
                // //         inv_freq = 1.0 / (base**(torch.arange(
                // //             0, self.rotary_dim, 2, dtype=torch.float, device="cuda") /
                // //                                  self.rotary_dim))
                // //         return inv_freq
                
                // //     def _compute_cos_sin_cache(self) -> torch.Tensor:
                // //         """Compute the cos and sin cache."""
                // //         inv_freq = self._compute_inv_freq(self.base)
                // //         t = torch.arange(self.max_position_embeddings,
                // //                          dtype=torch.float,
                // //                          device="cuda")
                
                // //         freqs = torch.einsum("i,j -> ij", t, inv_freq)
                // //         cos = freqs.cos() // 2048，64
                // //         sin = freqs.sin()
                // //         cache = torch.cat((cos, sin), dim=-1)
                // //         return cache
                // inline __device__ float2 GetRoPEfreq(int zid, int rot_embed_dim, float base, float t_step)
                // {
                //     // (RussWong) note: 每个token所属的id, 它的freq值都是固定的, id的上限为max position embedding
                //     // t_step表示token id（这里考虑了多轮对话历史上下文长度)
                //     // 每个freq值对应于zid = head size维度上0 2 4 6 ... 64带入下式计算
                //     const float inv_freq = t_step / powf(base, zid / (float)rot_embed_dim);
                //     return {cos(inv_freq), sin(inv_freq)};
                // }
                
                // inline __device__ float2 GetRoPEres(float data, float data_rotate, const float2 coef)
                // {
                //     float2 rot_v;
                //     rot_v.x = coef.x * data - coef.y * data_rotate;
                //     rot_v.y = coef.x * data_rotate + coef.y * data;
                //     return rot_v;
                // }
                // // inline __device__ float2 GetRoPEres(const float2 v, const float2 coef)
                // // {
                // //     float2 rot_v;
                // //     rot_v.x = coef.x * v.x - coef.y * v.y;
                // //     rot_v.y = coef.x * v.y + coef.y * v.x;
                // //     return rot_v;
                // // }
                
                // // inline __device__ half2 GetRoPEres(const half2 v, const float2 coef)
                // // {
                // //     float2 fv = __half22float2(v);
                // //     float2 rot_fv = GetRoPEres(fv, coef);
                // //     return __float22half2_rn(rot_fv);
                // // }
                
                // // inline __device__ void apply_RoPE(float q, float k, int tid, int rot_embed_dim, float base, int t_step)
                // // {
                // //     if (tid >= rot_embed_dim / 2)
                // //     {
                // //         return;
                // //     }
                
                // //     float2 coef0 = GetRoPEfreq(tid, rot_embed_dim, base, t_step);
                // //     q = GetRoPEres(q, coef0);
                // //     k = GetRoPEres(k, coef0);
                // // }
                
                // // inline __device__ void apply_RoPE(half2 &q, half2 &k, int tid, int rot_embed_dim, float base, int t_step)
                // // {
                // //     if (2 * tid >= rot_embed_dim)
                // //     {
                // //         return;
                // //     }
                // //     const auto coef = GetRoPEfreq(2 * tid, rot_embed_dim, base, t_step);
                // //     q = GetRoPEres(q, coef);
                // //     k = GetRoPEres(k, coef);
                // // }
                
                // // inline __device__ void apply_RoPE(float4 &q, float4 &k, int tid, int rot_embed_dim, float base, int t_step)
                // // {
                // //     if (4 * tid >= rot_embed_dim)
                // //     {
                // //         return;
                // //     }
                
                // //     TwoFloat2 &q_ = *reinterpret_cast<TwoFloat2 *>(&q);
                // //     TwoFloat2 &k_ = *reinterpret_cast<TwoFloat2 *>(&k);
                
                // //     float2 coef0 = GetRoPEfreq(4 * tid, rot_embed_dim, base, t_step);
                // //     q_.x = GetRoPEres(q_.x, coef0);
                // //     float2 coef1 = GetRoPEfreq(4 * tid + 2, rot_embed_dim, base, t_step);
                // //     q_.y = GetRoPEres(q_.y, coef1);
                // //     k_.x = GetRoPEres(k_.x, coef0);
                // //     k_.y = GetRoPEres(k_.y, coef1);
                // // }
                // template <typename T>
                // __global__ void add_fusedQKV_bias_transpose_kernel(T *q_buf,
                //                                                    T *k_buf,
                //                                                    T *v_buf,
                //                                                    T *QKV,
                //                                                    /*optional*/const T *qkv_bias,
                //                                                    const int *padding_offset, // created before qkv linear
                //                                                    const int *history_length,
                //                                                    const int *input_length, // actual length of each seq
                //                                                    const int batch_size,
                //                                                    const int seq_len, // max_seq_len to pad to
                //                                                    const int token_num,
                //                                                    const int head_num,
                //                                                    const int kv_head_num,
                //                                                    const int head_size,
                //                                                    const int rotary_embedding_dim,
                //                                                    float rotary_embedding_base, // default 10000 in llama
                //                                                    int max_position_embeddings, /*default 2048 in llama*/
                //                                                    bool use_dynamic_ntk /*placeholder for ntk RoPE*/)
                // {
                //     // (RussWong)note: in Llama, rotate size = 64, we are not able to vectorizedly read data.
                //     // int vec_size = Vec<T>::size;
                //     // using Vec_t = typename Vec<T>::Type;
                //     int token_id = blockIdx.x;
                //     int head_id = blockIdx.y;
                //     int tid = threadIdx.x;
                //     int token_padding_offset = padding_offset[token_id];
                //     // 1. prapare rebuilding , do rebuild padding and transpose when store
                //     int dst_token_id = token_id + token_padding_offset; // token id after rebuild padding
                
                //     int batch_id = dst_token_id / seq_len;       // seqlen is max_seq_len for padding used to unify all seq's length
                //     int local_token_id = dst_token_id % seq_len; // 每个seq中的局部token id
                //     // 2. bias add
                //     int qkv_head_num = head_num + 2 * kv_head_num;
                //     int q_id = token_id * qkv_head_num * head_size + head_id * head_size + tid;
                //     int k_id = token_id * qkv_head_num * head_size + head_id * head_size + tid + head_num * head_size;
                //     int v_id = token_id * qkv_head_num * head_size + head_id * head_size + tid + head_num * head_size + kv_head_num * head_size;
                
                //     float v = QKV[v_id];
                //     int dst_q_id = batch_id * seq_len * head_num * head_size +
                //                    head_id * seq_len * head_size +
                //                    local_token_id * head_size + tid;
                
                //     int dst_kv_id = batch_id * seq_len * kv_head_num * head_size +
                //                     head_id * seq_len * head_size +
                //                     local_token_id * head_size + tid;
                //     if (head_id < kv_head_num)
                //     { // (RussWong)note: for MQA and GQA
                //         v_buf[dst_kv_id] = v;
                //     }
                //     // 3. RoPE
                //     const int cur_seq_history_len = history_length[batch_id];
                //     const int context_length = cur_seq_history_len + input_length[batch_id];
                //     //（RussWong)note: 多轮对话下要结合history length求得全局的cos和sin
                //     const int timestep = cur_seq_history_len + local_token_id; 
                //     // (RussWong)note: timestep为cos(m*theta)中的m
                //     if (tid >= rotary_embedding_dim / 2)
                //     {
                //         return;
                //     } // tid = [0,1,2,...,63]
                
                //     float2 cos_sin = GetRoPEfreq(tid * 2, rotary_embedding_dim, rotary_embedding_base, timestep);
                //     // (RussWong)note: print cos and sin of each token id, this is nothing to do with concrete input id
                //     //if(token_id == 2 && head_id == 0 && tid == 0)
                //    // {
                //     //    printf("tokenid=2, cos_sin res:\n");
                //     //    printf("cos: %f, sin:%f\n", cos_sin.x, cos_sin.y);
                //     //}
                //     //if(token_id == 1 && head_id == 0 && tid == 0)
                //     //{
                //     //    printf("tokenid=1, cos_sin res:\n");
                //     //    printf("cos: %f, sin:%f\n", cos_sin.x, cos_sin.y);
                //     //}
                //     float2 q_rotate = GetRoPEres(QKV[q_id], QKV[q_id + head_size / 2], cos_sin);
                //     float2 k_rotate = GetRoPEres(QKV[k_id], QKV[k_id + head_size / 2], cos_sin);
                //     // (RussWong)note: write result back into q k v
                //     q_buf[dst_q_id] = q_rotate.x;
                //     q_buf[dst_q_id + head_size / 2] = q_rotate.y;
                //     if (head_id < kv_head_num)
                //     { // for MQA and GQA
                //         k_buf[dst_kv_id] = k_rotate.x;
                //         k_buf[dst_kv_id + head_size / 2] = k_rotate.y;
                //     }
                // }
                
                // template <>
                // __global__ void add_fusedQKV_bias_transpose_kernel(half *q_buf,
                //                                                    half *k_buf,
                //                                                    half *v_buf,
                //                                                    half *QKV,
                //                                                    /*optional*/const half *qkv_bias,
                //                                                    const int *padding_offset, // created before qkv linear
                //                                                    const int *history_length,
                //                                                    const int *input_length, // actual length of each seq
                //                                                    const int batch_size,
                //                                                    const int seq_len, // max_seq_len to pad to
                //                                                    const int token_num,
                //                                                    const int head_num,
                //                                                    const int kv_head_num,
                //                                                    const int head_size,
                //                                                    const int rotary_embedding_dim,
                //                                                    float rotary_embedding_base, // default 10000 in llama
                //                                                    int max_position_embeddings, /*default 2048 in llama, placeholder for ntk RoPE*/
                //                                                    bool use_dynamic_ntk /*placeholder for ntk RoPE*/)
                // {
                //     // int vec_size = Vec<half>::size;
                //     // using Vec_t = typename Vec<half>::Type;
                //     // int token_id = blockIdx.x;
                //     // int head_id = blockIdx.y;
                //     // int tid = threadIdx.x;
                //     // int token_padding_offset = padding_offset[token_id];
                //     // // 0. filter the redundant part, we'd better to allocate more threads than data to ensure all data can be vectorized
                //     // bool is_data = tid * vec_size < head_size;
                //     // // 1. prapare rebuilding , do rebuild padding and transpose when store
                //     // int dst_token_id = token_id + token_padding_offset; // token id after rebuild padding
                
                //     // int batch_id = dst_token_id / seq_len;       // seqlen is max_seq_len for padding used to unify all seq's length
                //     // int local_token_id = dst_token_id % seq_len; // 每个seq中的局部token id
                
                //     // // 2. bias add
                //     // int qkv_head_num = head_num + 2 * kv_head_num;
                //     // int q_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size;
                //     // int k_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size + head_num * head_size;
                //     // int v_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size;
                //     // // note: scalar add can be replaced by 3 overloaded function call, which is implemented by float add, float2 add and float4 add.
                //     // // TODO: reduce the pointer converter and fuse for loop
                //     // Vec_t q, k, v;
                //     // if (is_data)
                //     // {
                //     //     q = *reinterpret_cast<Vec_t *>(&QKV[q_id]);
                //     //     Vec_t q_bias = *reinterpret_cast<Vec_t *>(const_cast<half *>(&qkv_bias[head_id * head_size + tid * vec_size]));
                //     //     q = __hadd2(q, q_bias);
                //     // }
                //     // // note: kv judge condition is add a item that head_id<kv_head_id in case of GQA and MQA
                //     // if (is_data && head_id < kv_head_num)
                //     // {
                //     //     k = *reinterpret_cast<Vec_t *>(&QKV[k_id]);
                //     //     // note: I missed a vec_size about the bias offset causing memcpyd2h misaligned address
                //     //     Vec_t k_bias = *reinterpret_cast<Vec_t *>(const_cast<half *>(&qkv_bias[head_id * head_size + tid * vec_size + head_num * head_size]));
                //     //     k = __hadd2(k, k_bias);
                //     //     v = *reinterpret_cast<Vec_t *>(&QKV[v_id]);
                //     //     Vec_t v_bias = *reinterpret_cast<Vec_t *>(const_cast<half *>(&qkv_bias[head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size]));
                //     //     v = __hadd2(v, v_bias);
                //     // }
                
                //     // // 3. RoPE
                //     // const int cur_seq_history_len = history_length[batch_id]; // pay attention to where the history lenght cumsum
                //     // const int context_length = cur_seq_history_len + input_length[batch_id];
                //     // const int timestep = cur_seq_history_len + local_token_id; //+ local_token_id得到m，即要结合history length做全局位置编码
                //     // // timestep为cos(m*theta)中的m
                
                //     // apply_RoPE(q, k, tid, rotary_embedding_dim, rotary_embedding_base, timestep);
                //     // // 4.write back to gmem and do transpose
                //     // //  [bs, head num, seqlen, head size]
                //     // //  pay attention to local token id and kv head num and max_seq_len(seq_len)
                //     // int dst_q_id = batch_id * seq_len * head_num * head_size +
                //     //                head_id * seq_len * head_size +
                //     //                local_token_id * head_size + tid * vec_size;
                
                //     // int dst_kv_id = batch_id * seq_len * kv_head_num * head_size +
                //     //                 head_id * seq_len * head_size +
                //     //                 local_token_id * head_size + tid * vec_size;
                //     // if (is_data)
                //     // {
                //     //     *reinterpret_cast<Vec_t *>(&q_buf[dst_q_id]) = q; // remember to add & before q_buf[], cause q_buf[] is a scalar
                //     //     if (head_id < kv_head_num)
                //     //     { // for MQA and GQA
                //     //         *reinterpret_cast<Vec_t *>(&k_buf[dst_kv_id]) = k;
                //     //         *reinterpret_cast<Vec_t *>(&v_buf[dst_kv_id]) = v;
                //     //     }
                //     // }
                // }
                
                // // input: qkv_buf : qkv continouns buf when no padding
                // // shape = [num_tokens, qkv_head_num, head_size]
                // // output: q shape = [bs, head num, seqlen, head size], if k v is this shape, maybe need tranpose in successor steps, ep in cublas
                // //         k/v shape = [bs, kv head num, seqlen, head size]
                // template <typename T>
                // void launchPadRope(TensorWrapper<T> *q_buf,
                //                                            TensorWrapper<T> *k_buf,
                //                                            TensorWrapper<T> *v_buf,
                //                                            TensorWrapper<T> *QKV,
                //                                            BaseWeight<T> &qkv,
                //                                            // Tensor* qkv_bias,
                //                                            TensorWrapper<int> *padding_offset,
                //                                            TensorWrapper<int> *history_length,
                //                                            TensorWrapper<int> *input_length,
                //                                            LLamaAttentionStaticParams &params)
                // {
                //     int token_num = QKV->shape[0];
                //     int qkv_head_num = QKV->shape[1];
                //     int head_size = QKV->shape[2];
                //     int batch_size = q_buf->shape[0];
                //     int head_num = q_buf->shape[1];
                //     int seq_len = q_buf->shape[2];
                //     int kv_head_num = (qkv_head_num - head_num) / 2;
                
                //     dim3 grid(token_num, head_num);
                //     dim3 block(head_size);
                //     add_fusedQKV_bias_transpose_kernel<T><<<grid, block>>>(q_buf->data,
                //                                                            k_buf->data,
                //                                                            v_buf->data,
                //                                                            QKV->data,
                //                                                            /*optional*/qkv.bias,
                //                                                            padding_offset->data,
                //                                                            history_length->data,
                //                                                            input_length->data,
                //                                                            batch_size,
                //                                                            seq_len,
                //                                                            token_num,
                //                                                            head_num,
                //                                                            kv_head_num,
                //                                                            head_size,
                //                                                            params.rotary_embedding_dim,
                //                                                            params.rotary_embedding_base,
                //                                                            params.max_position_embeddings,
                //                                                            params.use_dynamic_ntk);
                // #ifdef PRINT_DATA
                //     print_data<<<1, 1>>>(q_buf->data);
                // #else
                // #endif
                // }
                
                // template void launchPadRope(TensorWrapper<float> *q_buf,
                //                                                     TensorWrapper<float> *k_buf,
                //                                                     TensorWrapper<float> *v_buf,
                //                                                     TensorWrapper<float> *QKV,
                //                                                     BaseWeight<float> &qkv,
                //                                                     TensorWrapper<int> *padding_offset,
                //                                                     TensorWrapper<int> *history_length,
                //                                                     TensorWrapper<int> *input_length,
                //                                                     LLamaAttentionStaticParams &params);
                // template void launchPadRope(TensorWrapper<half> *q_buf,
                //                                                     TensorWrapper<half> *k_buf,
                //                                                     TensorWrapper<half> *v_buf,
                //                                                     TensorWrapper<half> *QKV,
                //                                                     BaseWeight<half> &qkv,
                //                                                     TensorWrapper<int> *padding_offset,
                //                                                     TensorWrapper<int> *history_length,
                //                                                     TensorWrapper<int> *input_length,
                //                                                     LLamaAttentionStaticParams &params);
                
               