#include "MemoryAllocator.h"

#include "Context.h"

#pragma warning(push)
#pragma warning(disable : 4996) // function or variable may be unsafe

namespace Ray {
namespace Vk {
uint32_t FindMemoryType(const VkPhysicalDeviceMemoryProperties *mem_properties, uint32_t mem_type_bits,
                        VkMemoryPropertyFlags desired_mem_flags) {
    for (uint32_t i = 0; i < 32; i++) {
        const VkMemoryType mem_type = mem_properties->memoryTypes[i];
        if (mem_type_bits & 1u) {
            if ((mem_type.propertyFlags & desired_mem_flags) == desired_mem_flags) {
                return i;
            }
        }
        mem_type_bits = (mem_type_bits >> 1u);
    }
    return 0xffffffff;
}
} // namespace Vk
} // namespace Ray

void Ray::Vk::MemAllocation::Release() {
    if (owner) {
        owner->Free(block_ndx, alloc_off, alloc_size);
        owner = nullptr;
    }
}

Ray::Vk::MemoryAllocator::MemoryAllocator(const char name[32], Context *ctx, const uint32_t initial_block_size,
                                          uint32_t mem_type_index, const float growth_factor)
    : ctx_(ctx), growth_factor_(growth_factor), mem_type_index_(mem_type_index) {
    strcpy(name_, name);

    assert(growth_factor_ > 1.0f);
    AllocateNewBlock(initial_block_size);
}

Ray::Vk::MemoryAllocator::~MemoryAllocator() {
    for (MemBlock &blk : blocks_) {
        vkFreeMemory(ctx_->device(), blk.mem, nullptr);
    }
}

bool Ray::Vk::MemoryAllocator::AllocateNewBlock(const uint32_t size) {
    char buf_name[48];
    sprintf(buf_name, "%s block %i", name_, int(blocks_.size()));

    blocks_.emplace_back();
    MemBlock &new_block = blocks_.back();

    new_block.alloc = LinearAlloc{1024, size};

    VkMemoryAllocateInfo buf_alloc_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    buf_alloc_info.allocationSize = VkDeviceSize(size);
    buf_alloc_info.memoryTypeIndex = mem_type_index_;

    const VkResult res = vkAllocateMemory(ctx_->device(), &buf_alloc_info, nullptr, &new_block.mem);
    return res == VK_SUCCESS;
}

Ray::Vk::MemAllocation Ray::Vk::MemoryAllocator::Allocate(const uint32_t size, const uint32_t alignment, const char *tag) {
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
                new_alloc.owner = this;
                return new_alloc;
            }
        }

        // allocation failed, add new buffer
        do {
            const bool res = AllocateNewBlock(uint32_t(blocks_.back().alloc.size() * growth_factor_));
            assert(res);
        } while (blocks_.back().alloc.size() < size);
    }

    return {};
}

void Ray::Vk::MemoryAllocator::Free(const uint32_t block_ndx, const uint32_t alloc_off, const uint32_t alloc_size) {
    assert(block_ndx < blocks_.size());
    blocks_[block_ndx].alloc.Free(alloc_off, alloc_size);
}

void Ray::Vk::MemoryAllocators::Print(ILog *log) {
    /*log->Info("=================================================================");
    log->Info("MemAllocs %s", name_);
    for (const auto &alloc : allocators_) {
        alloc.Print(log);
    }
    log->Info("=================================================================");*/
}

#pragma warning(pop)