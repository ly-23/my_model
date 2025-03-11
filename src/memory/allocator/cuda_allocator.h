#include "src/memory/allocator/base_allocator.h"
#include<map>
#include<vector>
struct CudaBigBlock{
    void* data;
    size_t size;
    bool is_allocated;
    CudaBigBlock()=default;
    CudaBigBlock(void* data,size_t size ,bool is_allocated):
        data(data),size(size),is_allocated(is_allocated){}

};

struct CudaSmallBlock{
    void* data;
    size_t size;
    bool is_allocated;
    CudaSmallBlock()=default;
    CudaSmallBlock(void* data,size_t size ,bool is_allocated):
        data(data),size(size),is_allocated(is_allocated){}
};



class CudaAllocator:public BaseAllocator{
private:
    std::map<int,std::vector<CudaBigBlock>> cudaBigBlocksMap;
    std::map<int,std::vector<CudaSmallBlock>> cudaSmallBlocksMap;
    std::map<int,size_t> FreeSize;
    size_t total_allocated_size=0;
    int dev_id;
public:
    CudaAllocator(){
        cudaGetDevice(&dev_id);  //后续切换dev_id的时候手动切换。
    }
    ~CudaAllocator(){
        
    }
    void* UnifyMalloc(void* ptr, size_t size,bool is_host){
        if(is_host){
            ptr=malloc(size);
            memory(ptr,0,size);
            return ptr;
        }
        if(size>1024*1024){
            auto &BigBlocks=cudaBigBlocksMap[dev_id];
            int block_id=-1;
            for(int i=0;i<BigBlocks.size();i++){
                if(BigBlocks[i].size>size&&BigBlocks[i].is_allocated&&BigBlocks[i].size-size<1024*1024){
                    if(block_id==-1||BigBlocks[i].size<BigBlocks[block_id].size){
                        block_id=i;
                    }
                }
            }
            if(block_id!=-1){
                BigBlocks[block_id].is_allocated=true;
                return BigBlocks[block_id].data;
            }
            
            void* new_ptr;
            cudaMalloc(&new_ptr,size);
            cudaBigBlocksMap[dev_id].push_back(CudaBigBlock(new_ptr,size,true));
            total_allocated_size+=size;
            return new_ptr;
        }
        auto &SmallBlocks=cudaSmallBlocksMap[dev_id];

        for(int i=0;i<SmallBlocks.size;i++){
            if(SmallBlocks[i]>size&&SmallBlocks[i].is_allocated==false){
                SmallBlocks[i].is_allocated=true;
                FreeSize[i]-=SmallBlocks[i].size;
                return SmallBlocks.data;
            }
        }
        void* new_ptr=(void*)ptr;
        cudaMalloc(new_ptr,size);
        cudaMemory(new_ptr,0,size);

        SmallBlocks.push_back(CudaSmallBlock(new_ptr,size,true));
        return new_ptr;
    }

    void* UnifyFree(void* ptr,bool is_host){
        if (ptr == nullptr) {
            return;
        }
        if (is_host) {
            free(ptr);
            return;
        }
        for (auto &it: cudaSmallBlocksMap) {
            if (FreeSize[it.first] > 1024 * 1024 * 1024) {
                auto &cudaBlocks = it.second;
                std::vector<CudaSmallBlock> temp;
                for (int i = 0; i < cudaBlocks.size(); i++) {
                    if (!cudaBlocks[i].is_allocated) {
                        cudaSetDevice(it.first);
                        cudaFree(cudaBlocks[i].data);
                    } else {
                        temp.push_back(cudaBlocks[i]);
                    }
                }
                cudaBlocks.clear();
                it.second = temp;
                FreeSize[it.first] = 0;
            }
        }
        for (auto &it: cudaSmallBlocksMap) {
            auto &cudaBlocks = it.second;
            for (int i = 0; i < cudaBlocks.size(); i++) {
                if (cudaBlocks[i].data == ptr) {
                    FreeSize[it.first] += cudaBlocks[i].size;
                    cudaBlocks[i].is_allocated = false;
        
                    return;
                }
            }
            //若是大block，那不归还到OS
            auto &bigBlocks = cudaBigBlocksMap[it.first];
            for (int i = 0; i < bigBlocks.size(); i++) {
                if (bigBlocks[i].data == ptr) {
                    bigBlocks[i].is_allocated = false;
                    return;
                }
            }
        }
        cudaFree(ptr);    
    }
    }   

