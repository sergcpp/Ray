#pragma once

#include <memory>
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

namespace Ray::Dx {
class Buffer;
class MemAllocator;

struct MemAllocation {
    uint32_t offset = 0, block = 0;
    uint16_t pool = 0;
    MemAllocator *owner = nullptr;

    MemAllocation() = default;
    MemAllocation(const MemAllocation &rhs) = delete;
    MemAllocation(MemAllocation &&rhs) noexcept
        : offset(rhs.offset), block(rhs.block), pool(rhs.pool), owner(std::exchange(rhs.owner, nullptr)) {}

    MemAllocation &operator=(const MemAllocation &rhs) = delete;
    MemAllocation &operator=(MemAllocation &&rhs) noexcept {
        Release();

        offset = rhs.offset;
        block = rhs.block;
        pool = rhs.pool;
        owner = std::exchange(rhs.owner, nullptr);

        return (*this);
    }

    operator bool() const { return owner != nullptr; }

    ~MemAllocation() { Release(); }

    void Release();
};

class MemAllocator {
    std::string name_;
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
    MemAllocator(std::string_view name, Context *ctx, uint32_t initial_pool_size, D3D12_HEAP_TYPE heap_type,
                 float growth_factor, uint32_t max_pool_size);
    ~MemAllocator();

    MemAllocator(const MemAllocator &rhs) = delete;
    MemAllocator(MemAllocator &&rhs) = default;

    MemAllocator &operator=(const MemAllocator &rhs) = delete;
    MemAllocator &operator=(MemAllocator &&rhs) = default;

    ID3D12Heap *heap(const int pool) const { return pools_[pool].heap; }
    D3D12_HEAP_TYPE heap_type() const { return heap_type_; }

    MemAllocation Allocate(uint32_t alignment, uint32_t size);
    void Free(uint32_t block);

    void Print(ILog *log) const {}
};

class MemAllocators {
    std::string name_;
    Context *ctx_;
    uint32_t initial_pool_size_;
    float growth_factor_;
    uint32_t max_pool_size_;
    std::unique_ptr<MemAllocator> allocators_[8];

  public:
    MemAllocators(std::string_view name, Context *ctx, const uint32_t initial_pool_size, const float growth_factor,
                  const uint32_t max_pool_size)
        : name_(name), ctx_(ctx), initial_pool_size_(initial_pool_size), growth_factor_(growth_factor),
          max_pool_size_(max_pool_size) {}

    MemAllocation Allocate(const uint32_t alignment, const uint32_t size, const D3D12_HEAP_TYPE heap_type) {
        if (!allocators_[heap_type]) {
            std::string name = name_;
            name += " (type " + std::to_string(heap_type) + ")";
            allocators_[heap_type] = std::make_unique<MemAllocator>(name, ctx_, initial_pool_size_, heap_type,
                                                                    growth_factor_, max_pool_size_);
        }
        return allocators_[heap_type]->Allocate(alignment, size);
    }
};
} // namespace Ray::Dx

#ifdef _MSC_VER
#pragma warning(pop)
#endif
