#pragma once

#include <string>

#include "../FreelistAlloc.h"
#include "../SmallVector.h"
#include "BufferDX.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

struct ID3D12Heap;

enum D3D12_HEAP_TYPE;

namespace Ray {
namespace Dx {
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
        ID3D12Heap *heap;
        uint32_t size;
    };

    D3D12_HEAP_TYPE heap_type_;
    FreelistAlloc alloc_;
    SmallVector<MemPool, 8> pools_;

    bool AllocateNewPool(uint32_t size);

  public:
    MemoryAllocator(const char name[32], Context *ctx, uint32_t initial_pool_size, D3D12_HEAP_TYPE heap_type,
                    float growth_factor, uint32_t max_pool_size);
    ~MemoryAllocator();

    MemoryAllocator(const MemoryAllocator &rhs) = delete;
    MemoryAllocator(MemoryAllocator &&rhs) = default;

    MemoryAllocator &operator=(const MemoryAllocator &rhs) = delete;
    MemoryAllocator &operator=(MemoryAllocator &&rhs) = default;

    ID3D12Heap *heap(const int pool) const { return pools_[pool].heap; }
    D3D12_HEAP_TYPE heap_type() const { return heap_type_; }

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

    MemAllocation Allocate(const uint32_t alignment, const uint32_t size, const D3D12_HEAP_TYPE heap_type) {
        int alloc_index = -1;
        for (int i = 0; i < int(allocators_.size()); ++i) {
            if (allocators_[i].heap_type() == heap_type) {
                alloc_index = i;
                break;
            }
        }

        if (alloc_index == -1) {
            char name[32];
            snprintf(name, sizeof(name), "%s (type %i)", name_, int(heap_type));
            alloc_index = int(allocators_.size());
            allocators_.emplace_back(name, ctx_, initial_pool_size_, heap_type, growth_factor_, max_pool_size_);
        }

        return allocators_[alloc_index].Allocate(alignment, size);
    }

    void Print(ILog *log);
};
} // namespace Dx
} // namespace Ray

#ifdef _MSC_VER
#pragma warning(pop)
#endif
