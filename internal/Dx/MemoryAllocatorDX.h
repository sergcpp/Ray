#pragma once

#include <string>

#include "../LinearAlloc.h"
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
    uint32_t block_ndx = 0;
    uint32_t alloc_off = 0, alloc_size = 0;
    MemoryAllocator *owner = nullptr;

    MemAllocation() = default;
    MemAllocation(const MemAllocation &rhs) = delete;
    MemAllocation(MemAllocation &&rhs) noexcept
        : block_ndx(rhs.block_ndx), alloc_off(rhs.alloc_off), alloc_size(rhs.alloc_size),
          owner(exchange(rhs.owner, nullptr)) {}

    MemAllocation &operator=(const MemAllocation &rhs) = delete;
    MemAllocation &operator=(MemAllocation &&rhs) noexcept {
        Release();

        block_ndx = rhs.block_ndx;
        alloc_off = rhs.alloc_off;
        alloc_size = rhs.alloc_size;
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

    struct MemBlock {
        ID3D12Heap *heap;
        LinearAlloc alloc;
    };

    D3D12_HEAP_TYPE heap_type_;
    SmallVector<MemBlock, 8> blocks_;

    bool AllocateNewBlock(uint32_t size);

  public:
    MemoryAllocator(const char name[32], Context *ctx, uint32_t initial_block_size, D3D12_HEAP_TYPE heap_type,
                    float growth_factor);
    ~MemoryAllocator();

    MemoryAllocator(const MemoryAllocator &rhs) = delete;
    MemoryAllocator(MemoryAllocator &&rhs) = default;

    MemoryAllocator &operator=(const MemoryAllocator &rhs) = delete;
    MemoryAllocator &operator=(MemoryAllocator &&rhs) = default;

    ID3D12Heap *heap(const int i) const { return blocks_[i].heap; }
    D3D12_HEAP_TYPE heap_type() const { return heap_type_; }

    MemAllocation Allocate(uint32_t size, uint32_t alignment, const char *tag);
    void Free(uint32_t block_ndx, uint32_t alloc_off, uint32_t alloc_size);

    void Print(ILog *log) const {}
};

class MemoryAllocators {
    char name_[16] = {};
    Context *ctx_;
    uint32_t initial_block_size_;
    float growth_factor_;
    SmallVector<MemoryAllocator, 4> allocators_;

  public:
    MemoryAllocators(const char name[16], Context *ctx, uint32_t initial_block_size, float growth_factor)
        : ctx_(ctx), initial_block_size_(initial_block_size), growth_factor_(growth_factor) {
        strcpy(name_, name);
    }

    MemAllocation Allocate(const uint32_t size, const uint32_t alignment, D3D12_HEAP_TYPE heap_type, const char *tag) {
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
            allocators_.emplace_back(name, ctx_, initial_block_size_, heap_type, growth_factor_);
        }

        return allocators_[alloc_index].Allocate(size, alignment, tag);
    }

    void Print(ILog *log);
};

inline uint64_t AlignTo(const uint64_t size, const uint64_t alignment) {
    return alignment * ((size + alignment - 1) / alignment);
}
} // namespace Dx
} // namespace Ray

#ifdef _MSC_VER
#pragma warning(pop)
#endif
