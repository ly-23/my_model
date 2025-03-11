#include "src/models/basemodel.h"
#include "src/models/llama/llama_params.h"
#include "src/weights/llama/llama_weights.h"
#include "src/layers/decoder/context_decoder.h"
#include "src/layers/decoder/self_decoder.h"
#include "src/kernels/embedding.h" // embedding
#include "src/kernels/linear.h" // LM Head
#include "src/kernels/topk.h" // topK
#include "src/kernels/sampling.h" // sampling
#include "src/models/tokenizer.h"
#include "src/utils/debug_utils.h"
template<typename T>
class Llama: public BaseModel{
private:
    const int head_num;
    const int kv_head_num;
    const int head_size;
    const int inter_size;
    const int num_layers;
    int vocab_size;
    int vocab_size_padded;
    float rmsnorm_eps = 1e-5f;   
    const int hidden_units; 
    const int max_seq_len; // self defined
    int output_token_limit = 256; // self defined
    int pad_token_id = 0;
    int bos_token_id = 1;
    int eos_token_id = 2;
    int layer_id = 0;
    int batch_size = 1; 
    int beamwidth = 1; 
    int BlockPerBeam = 8; 
    int index = 0;
    std::string prompt = ""; 

    Tokenizer tokenizer;
    LlamaWeight<T>* llama_weights;
    LlamaSelfDecoder<T>* self_decoder;
    LlamaContextDecoder<T>* context_decoder;
    // int max_context_token_num_ = 32; // 

    int K = 4; // K of topK sort
    TensorWrapper<int>* step;
    TensorWrapper<T>* output_rmsnorm_weight;
    TensorWrapper<int>* layer;
    TensorWrapper<T>* context_decoder_input;
    TensorWrapper<T>* context_decoder_output;
    TensorWrapper<T>* context_decoder_lmhead_input;
    TensorWrapper<T>* decoder_input;
    TensorWrapper<T>* decoder_output;

    TensorWrapper<int>* input_ids;
    TensorWrapper<int>* input_length;
    TensorWrapper<int>* history_length;
    TensorWrapper<int>* context_length;

    TensorWrapper<T>* all_k_cache;
    TensorWrapper<T>* all_v_cache;
    TensorWrapper<T>* unused_residual;
    
    IntDict int_params_of_sample;
    TensorWrapper<T>* probs;
    TensorWrapper<int>* token_ids;
    TensorWrapper<int>* sequence_lengths; 
    TensorWrapper<bool>* is_finished;
    TensorWrapper<int>* topk_id;
    TensorWrapper<T>* topk_val;
    TensorWrapper<int>* final_topk_id;
    TensorWrapper<T>* final_topk_val;

 
    int* h_input_ids_buf_{};
    int* h_input_length_buf_{};
    int* h_history_length_buf_{};
    int* h_context_length_buf_{};
    int* h_sequence_lengths_{};
    bool* h_finished_buf_{};
    int* h_output_ids{};

public:
    Llama() = default;
    Llama(int head_num,
          int kv_head_num,
          int head_size,
          int inter_size,
          int num_layers,
          int vocab_size,
          const LLaMAAttentionStaticParams&  attn_static_params,
        // int max_context_token_num,
          int max_seq_len,
        //for base model
          cudaStream_t stream,
          cublasWrapper* cublas_wrapper,
          BaseAllocator* allocator,
          cudaDeviceProp* cuda_device_prop):
    BaseModel(stream, cublas_wrapper, allocator, cuda_device_prop),
    head_num(head_num),
    kv_head_num(kv_head_num),
    head_size(head_size),
    inter_size(inter_size),
    num_layers(num_layers),
    vocab_size(vocab_size),
    vocab_size_padded(vocab_size),
    hidden_units(head_num * head_size),
    max_seq_len(max_seq_len) {
        int_params_of_sample.insert({"vocab_size", vocab_size});
        int_params_of_sample.insert({"end_id", eos_token_id});
        layer = new TensorWrapper<int>(CPU, DataType::INT32, {1}, &layer_id);
        llama_weights = new LlamaWeight<T>(head_num,
                                          kv_head_num,
                                          head_size,
                                          inter_size,
                                          vocab_size,
                                          num_layers,
                                          /*attn_bias*/false,
                                          getWeightType<T>());

        self_decoder = new LlamaSelfDecoder<T>(head_num,
                                        kv_head_num,
                                        head_size,
                                        inter_size,
                                        num_layers,
                                        attn_static_params,
                                        rmsnorm_eps,
                                        stream,
                                        cublas_wrapper,
                                        allocator);

        context_decoder = new LlamaContextDecoder<T>(head_num,
                                                    kv_head_num,
                                                    head_size,
                                                    inter_size,
                                                    num_layers,
                                                    attn_static_params,
                                                    rmsnorm_eps,
                                                    stream,
                                                    cublas_wrapper,
                                                    allocator);

        allocateCPUBuffer(1); 
        allocateGPUBuffer(1);
    }

    ~Llama() {
        this->free();
    };
    void loadTokenizer(std::string file){
      tokenizer.Initialize(file);
    }
    void loadWeights(std::string file){
      llama_weights->loadWeights(file);
    }
    void loadWeightsFromDummy(){
      llama_weights->loadWeightsFromDummy();
    }
    void allocateCPUBuffer(int max_batch_size);
    void allocateGPUBuffer(int batch_size);
    void free();

    std::vector<std::string> MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

    std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前轮次回复更新history
    // single request response
    std::string Response(const std::vector<std::string>& input, CallBack PrintRes);

    int MakeOutput();

    void inputEmbedding(TensorWrapper<int>* input_ids, TensorWrapper<T>* decoder_input);
    void InitializeForContextDecoder(IntDict& int_params_first_token);
    int firstTokenGen(LLaMAAttentionDynParams& dparams, IntDict& int_params_first_token);
    void InitializeForSelfDecoder();
    int continueTokenGen(LLaMAAttentionDynParams& dparams);
    int LMHeadAndTopKSample(TensorMap& decoder_outputs);
};