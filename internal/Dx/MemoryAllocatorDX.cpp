#include "MemoryAllocatorDX.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12.h>

#include "ContextDX.h"

#pragma warning(push)
#pragma warning(disable : 4996) // function or variable may be unsafe

void Ray::Dx::MemAllocation::Release() {
    if (owner) {
        owner->Free(block);
        owner = nullptr;
    }
}

Ray::Dx::MemAllocator::MemAllocator(std::string_view name, Context *ctx, const uint32_t initial_pool_size,
                                    D3D12_HEAP_TYPE heap_type, const float growth_factor, const uint32_t max_pool_size)
    : name_(name), ctx_(ctx), growth_factor_(growth_factor), max_pool_size_(max_pool_size), heap_type_(heap_type) {
    assert(growth_factor_ > 1.0f);
    AllocateNewPool(initial_pool_size);
}

Ray::Dx::MemAllocator::~MemAllocator() {
    for (MemPool &pool : pools_) {
        pool.heap->Release();
    }
}

bool Ray::Dx::MemAllocator::AllocateNewPool(const uint32_t size) {
    D3D12_HEAP_DESC heap_desc = {};
    heap_desc.SizeInBytes = size;
    heap_desc.Flags = D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
    heap_desc.Properties.Type = heap_type_;
    heap_desc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_desc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_desc.Properties.CreationNodeMask = 1;
    heap_desc.Properties.VisibleNodeMask = 1;

    ID3D12Heap *dx_heap = nullptr;
    HRESULT hr = ctx_->device()->CreateHeap(&heap_desc, IID_PPV_ARGS(&dx_heap));
    if (SUCCEEDED(hr)) {
        pools_.emplace_back();
        MemPool &new_pool = pools_.back();
        new_pool.heap = dx_heap;
        new_pool.size = size;

        [[maybe_unused]] const uint16_t pool_ndx = alloc_.AddPool(size);
        assert(pool_ndx == pools_.size() - 1);
    }
    return SUCCEEDED(hr);
}

Ray::Dx::MemAllocation Ray::Dx::MemAllocator::Allocate(const uint32_t alignment, const uint32_t size) {
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

    return new_alloc;
}

void Ray::Dx::MemAllocator::Free(const uint32_t block) {
    alloc_.Free(block);
    assert(alloc_.IntegrityCheck());
}

#pragma warning(pop)