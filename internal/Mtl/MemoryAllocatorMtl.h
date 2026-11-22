#pragma once

#include <cassert>
#include <memory>
#include <string>
#include <utility>

#include "../FreelistAlloc.h"
#include "../SmallVector.h"
#include "ApiMtl.h"

namespace Ray {
class ILog;
} // namespace Ray

namespace Ray::Mtl {
class Context;
class MemAllocator;

struct MemAllocation {
    uint32_t offset = 0, block = 0;
    uint16_t pool = 0;
    MemAllocator *owner = nullptr;
    bool reused = false; // false: block is above the pool's ever-used watermark (driver-guaranteed zero pages)

    MemAllocation() = default;
    MemAllocation(const MemAllocation &rhs) = delete;
    MemAllocation(MemAllocation &&rhs) noexcept
        : offset(rhs.offset), block(rhs.block), pool(rhs.pool), owner(std::exchange(rhs.owner, nullptr)),
          reused(rhs.reused) {}

    MemAllocation &operator=(const MemAllocation &rhs) = delete;
    MemAllocation &operator=(MemAllocation &&rhs) noexcept {
        Release();

        offset = rhs.offset;
        block = rhs.block;
        pool = rhs.pool;
        owner = std::exchange(rhs.owner, nullptr);
        reused = rhs.reused;

        return (*this);
    }

    operator bool() const { return owner != nullptr; }

    ~MemAllocation() { Release(); }

    void Release();
};
// MTL::Heap Placement pools sub-allocated by the shared FreelistAlloc (heap->newBuffer/newTexture at
// offset). No internal locking: like VK/DX, callers synchronize externally (scene mutex covers edits).
class MemAllocator {
    std::string name_;
    Context *ctx_ = nullptr;
    float growth_factor_;
    uint32_t max_pool_size_;

    struct MemPool {
        MTL::Heap *heap;
        uint32_t size;
    };

    MTL::StorageMode storage_mode_;
    FreelistAlloc alloc_;
    SmallVector<MemPool, 8> pools_;
    SmallVector<uint32_t, 8> pool_hwm_; // ever-allocated byte watermark per pool: below = recycled memory

    bool AllocateNewPool(uint32_t size);

  public:
    MemAllocator(std::string_view name, Context *ctx, uint32_t initial_pool_size, MTL::StorageMode storage_mode,
                 float growth_factor, uint32_t max_pool_size);
    ~MemAllocator();

    MemAllocator(const MemAllocator &rhs) = delete;
    MemAllocator(MemAllocator &&rhs) = delete;

    MemAllocator &operator=(const MemAllocator &rhs) = delete;
    MemAllocator &operator=(MemAllocator &&rhs) = delete;

    MTL::Heap *heap(const int pool) const { return pools_[pool].heap; }
    uint32_t pool_size(const int pool) const { return pools_[pool].size; }
    MTL::StorageMode storage_mode() const { return storage_mode_; }

    MemAllocation Allocate(uint32_t alignment, uint32_t size);
    // frees the block and returns a fully-idle pool to the OS (plain Metal resources
    // release eagerly; scene/atlas churn would otherwise pin pools at their high-water mark)
    void Free(uint32_t block, uint16_t pool);

    void Print(ILog *log) const {}
};

class MemAllocators {
    std::string name_;
    Context *ctx_;
    uint32_t initial_pool_size_;
    float growth_factor_;
    uint32_t max_pool_size_;
    std::unique_ptr<MemAllocator> allocators_[4];

  public:
    MemAllocators(std::string_view name, Context *ctx, const uint32_t initial_pool_size, const float growth_factor,
                  const uint32_t max_pool_size)
        : name_(name), ctx_(ctx), initial_pool_size_(initial_pool_size), growth_factor_(growth_factor),
          max_pool_size_(max_pool_size) {}

    MemAllocation Allocate(const uint32_t alignment, const uint32_t size, const MTL::StorageMode storage_mode) {
        const int key = int(storage_mode);
        assert(key >= 0 && key < 4);

        // oversized resources (multi-page atlas generations) get dedicated storage: pool
        // memory must stay bounded, and FreelistAlloc cannot index blocks >= 2^FL_INDEX_MAX
        if (size > max_pool_size_) {
            return {};
        }

        if (!allocators_[key]) {
            std::string name = name_;
            name += " (mode " + std::to_string(key) + ")";
            allocators_[key] = std::make_unique<MemAllocator>(name, ctx_, initial_pool_size_, storage_mode,
                                                              growth_factor_, max_pool_size_);
        }
        return allocators_[key]->Allocate(alignment, size);
    }
};
} // namespace Ray::Mtl
