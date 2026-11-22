#include "MemoryAllocatorMtl.h"

#include <algorithm>

#include "../../Log.h"
#include "ContextMtl.h"

void Ray::Mtl::MemAllocation::Release() {
    if (owner) {
        owner->Free(block, pool);
        owner = nullptr;
    }
}

Ray::Mtl::MemAllocator::MemAllocator(std::string_view name, Context *ctx, const uint32_t initial_pool_size,
                                     const MTL::StorageMode storage_mode, const float growth_factor,
                                     const uint32_t max_pool_size)
    : name_(name), ctx_(ctx), growth_factor_(growth_factor), max_pool_size_(max_pool_size),
      storage_mode_(storage_mode) {
    assert(growth_factor_ > 1.0f);
    AllocateNewPool(initial_pool_size);
}

Ray::Mtl::MemAllocator::~MemAllocator() {
    for (MemPool &pool : pools_) {
        if (pool.heap != nullptr) {
            pool.heap->release();
        }
    }
}

bool Ray::Mtl::MemAllocator::AllocateNewPool(const uint32_t size) {
    MTL::HeapDescriptor *desc = MTL::HeapDescriptor::alloc()->init();
    desc->setType(MTL::HeapTypePlacement);
    desc->setSize(NS::UInteger(size));
    desc->setStorageMode(storage_mode_);
    desc->setCpuCacheMode(MTL::CPUCacheModeDefaultCache);
    desc->setHazardTrackingMode(MTL::HazardTrackingModeDefault);
    desc->setResourceOptions(
        MTL::ResourceOptions(storage_mode_ | MTL::CPUCacheModeDefaultCache | MTL::HazardTrackingModeDefault));

    MTL::Heap *new_heap = ctx_->device()->newHeap(desc);
    desc->release();

    if (new_heap) {
        const uint16_t pool_ndx = alloc_.AddPool(size);
        if (pool_ndx < pools_.size()) {
            // AddPool recycles the index of a removed pool
            assert(pools_[pool_ndx].heap == nullptr);
            pools_[pool_ndx] = {new_heap, size};
            pool_hwm_[pool_ndx] = 0;
        } else {
            pools_.emplace_back(MemPool{new_heap, size});
            pool_hwm_.push_back(0);
        }
    }
    return new_heap != nullptr;
}

Ray::Mtl::MemAllocation Ray::Mtl::MemAllocator::Allocate(const uint32_t alignment, const uint32_t size) {
    auto allocation = alloc_.Alloc(alignment, size);

    if (allocation.block == 0xffffffff) {
        const uint32_t required_size = FreelistAlloc::rounded_size(size + alignment);
        const bool res = AllocateNewPool(
            std::max(required_size, std::min(max_pool_size_, uint32_t(pools_.back().size * growth_factor_))));
        if (!res) {
            // allocation failed (out of memory)
            return {};
        }
        allocation = alloc_.Alloc(alignment, size);
    }

    assert((allocation.offset % alignment) == 0);
    assert(alloc_.IntegrityCheck());

    MemAllocation new_alloc = {};
    new_alloc.offset = allocation.offset;
    new_alloc.block = allocation.block;
    new_alloc.pool = allocation.pool;
    new_alloc.owner = this;

    const FreelistAlloc::Range range = alloc_.GetBlockRange(allocation.block);
    new_alloc.reused = range.offset + range.size <= pool_hwm_[allocation.pool];
    pool_hwm_[allocation.pool] = std::max(pool_hwm_[allocation.pool], range.offset + range.size);

    return new_alloc;
}

void Ray::Mtl::MemAllocator::Free(const uint32_t block, const uint16_t pool) {
    alloc_.Free(block);

    if (alloc_.GetFirstOccupiedBlock(pool).size == 0) {
        alloc_.RemovePool(pool);
        pools_[pool].heap->release();
        pools_[pool] = {nullptr, 0};
        pool_hwm_[pool] = 0;
    }

    assert(alloc_.IntegrityCheck());
}
