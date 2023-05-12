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
        owner->Free(block_ndx, alloc_off, alloc_size);
        owner = nullptr;
    }
}

Ray::Dx::MemoryAllocator::MemoryAllocator(const char name[32], Context *ctx, const uint32_t initial_block_size,
                                          D3D12_HEAP_TYPE heap_type, const float growth_factor)
    : ctx_(ctx), growth_factor_(growth_factor), heap_type_(heap_type) {
    strcpy(name_, name);

    assert(growth_factor_ > 1.0f);
    AllocateNewBlock(initial_block_size);
}

Ray::Dx::MemoryAllocator::~MemoryAllocator() {
    for (MemBlock &blk : blocks_) {
        blk.heap->Release();
    }
}

bool Ray::Dx::MemoryAllocator::AllocateNewBlock(const uint32_t size) {
    char buf_name[48];
    snprintf(buf_name, sizeof(buf_name), "%s block %i", name_, int(blocks_.size()));

    blocks_.emplace_back();
    MemBlock &new_block = blocks_.back();

    new_block.alloc = LinearAlloc{1024, size};

    D3D12_HEAP_DESC heap_desc = {};
    heap_desc.SizeInBytes = size;
    heap_desc.Flags = D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
    heap_desc.Properties.Type = heap_type_;
    heap_desc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_desc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_desc.Properties.CreationNodeMask = 1;
    heap_desc.Properties.VisibleNodeMask = 1;

    HRESULT hr = ctx_->device()->CreateHeap(&heap_desc, IID_PPV_ARGS(&new_block.heap));
    return SUCCEEDED(hr);
}

Ray::Dx::MemAllocation Ray::Dx::MemoryAllocator::Allocate(const uint32_t size, const uint32_t alignment, const char *tag) {
    while (true) {
        for (uint32_t i = 0; i < uint32_t(blocks_.size()); ++i) {
            if (size > blocks_[i].alloc.size()) {
                // can skip entire buffer
                continue;
            }

            const uint32_t alloc_off = blocks_[i].alloc.Alloc(size + alignment, tag);
            if (alloc_off != 0xffffffff) {
                // allocation succeded
                MemAllocation new_alloc = {};
                new_alloc.block_ndx = i;
                new_alloc.alloc_off = alloc_off;
                new_alloc.alloc_size = size + alignment;
                new_alloc.owner = this;
                return new_alloc;
            }
        }

        // allocation failed, add new buffer
        do {
            const bool res = AllocateNewBlock(uint32_t(blocks_.back().alloc.size() * growth_factor_));
            if (!res) {
                // allocation failed (out of memory)
                return {};
            }
        } while (blocks_.back().alloc.size() < size);
    }

    return {};
}

void Ray::Dx::MemoryAllocator::Free(const uint32_t block_ndx, const uint32_t alloc_off, const uint32_t alloc_size) {
    assert(block_ndx < blocks_.size());
    blocks_[block_ndx].alloc.Free(alloc_off, alloc_size);
}

void Ray::Dx::MemoryAllocators::Print(ILog *log) {
    /*log->Info("=================================================================");
    log->Info("MemAllocs %s", name_);
    for (const auto &alloc : allocators_) {
        alloc.Print(log);
    }
    log->Info("=================================================================");*/
}

#pragma warning(pop)