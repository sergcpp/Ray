#pragma once

#include <string>

#include "../FreelistAlloc.h"
#include "../SmallVector.h"
#include "BufferVK.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

namespace Ray {
namespace Vk {
class Buffer;
class MemoryAllocator;

struct MemAllocation {
    uint32_t offset = 0, block = 0;
    uint16_t pool = 0;
    MemoryAllocator *owner = nullptr;

    MemAllocation() = default;
    MemAllocation(const MemAllocation &rhs) = delete;
    MemAllocation(MemAllocation &&rhs) noexcept
        : offset(rhs.offset), block(rhs.block), pool(rhs.pool), owner(exchange(rhs.owner, nullptr)) {}

    MemAllocation &operator=(const MemAllocation &rhs) = delete;
    MemAllocation &operator=(MemAllocation &&rhs) noexcept {
        Release();

        offset = rhs.offset;
        block = rhs.block;
        pool = rhs.pool;
        owner = exchange(rhs.owner, nullptr);

        return (*this);
    }

    operator bool() const { return owner != nullptr; }

    ~MemAllocation() { Release(); }

    void Release();
};

class MemoryAllocator {
    char name_[32] = {};
    Context *ctx_ = nullptr;
    float growth_factor_;
    uint32_t max_pool_size_;

    struct MemPool {
        VkDeviceMemory mem;
        uint32_t size;
    };

    uint32_t mem_type_index_;
    FreelistAlloc alloc_;
    SmallVector<MemPool, 8> pools_;

    bool AllocateNewPool(uint32_t size);

  public:
    MemoryAllocator(const char name[32], Context *ctx, uint32_t initial_pool_size, uint32_t mem_type_index,
                    float growth_factor, uint32_t max_pool_size);
    ~MemoryAllocator();

    MemoryAllocator(const MemoryAllocator &rhs) = delete;
    MemoryAllocator(MemoryAllocator &&rhs) = default;

    MemoryAllocator &operator=(const MemoryAllocator &rhs) = delete;
    MemoryAllocator &operator=(MemoryAllocator &&rhs) = default;

    VkDeviceMemory mem(const int pool) const { return pools_[pool].mem; }
    uint32_t mem_type_index() const { return mem_type_index_; }

    MemAllocation Allocate(uint32_t alignment, uint32_t size);
    void Free(uint32_t block);

    void Print(ILog *log) const {}
};

class MemoryAllocators {
    char name_[16] = {};
    Context *ctx_;
    uint32_t initial_pool_size_;
    float growth_factor_;
    uint32_t max_pool_size_;
    SmallVector<MemoryAllocator, 4> allocators_;

  public:
    MemoryAllocators(const char name[16], Context *ctx, const uint32_t initial_pool_size, const float growth_factor,
                     const uint32_t max_pool_size)
        : ctx_(ctx), initial_pool_size_(initial_pool_size), growth_factor_(growth_factor),
          max_pool_size_(max_pool_size) {
        strcpy(name_, name);
    }

    MemAllocation Allocate(uint32_t alignment, uint32_t size, uint32_t mem_type_index);

    MemAllocation Allocate(const VkMemoryRequirements &mem_req, VkMemoryPropertyFlags desired_mem_flags);

    void Print(ILog *log);
};

inline VkDeviceSize AlignTo(VkDeviceSize size, VkDeviceSize alignment) {
    return alignment * ((size + alignment - 1) / alignment);
}
} // namespace Vk
} // namespace Ray

#ifdef _MSC_VER
#pragma warning(pop)
#endif
