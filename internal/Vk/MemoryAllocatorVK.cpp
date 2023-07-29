#include "MemoryAllocatorVK.h"

#include "ContextVK.h"

#pragma warning(push)
#pragma warning(disable : 4996) // function or variable may be unsafe

namespace Ray {
namespace Vk {
uint32_t FindMemoryType(const VkPhysicalDeviceMemoryProperties *mem_properties, uint32_t mem_type_bits,
                        VkMemoryPropertyFlags desired_mem_flags, VkDeviceSize desired_size) {
    for (uint32_t i = 0; i < 32; i++) {
        const VkMemoryType mem_type = mem_properties->memoryTypes[i];
        if (mem_type_bits & 1u) {
            if ((mem_type.propertyFlags & desired_mem_flags) == desired_mem_flags &&
                mem_properties->memoryHeaps[mem_type.heapIndex].size >= desired_size) {
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
        owner->Free(block);
        owner = nullptr;
    }
}

Ray::Vk::MemoryAllocator::MemoryAllocator(const char name[32], Context *ctx, const uint32_t initial_block_size,
                                          uint32_t mem_type_index, const float growth_factor)
    : ctx_(ctx), growth_factor_(growth_factor), mem_type_index_(mem_type_index) {
    strcpy(name_, name);

    assert(growth_factor_ > 1.0f);
    AllocateNewPool(initial_block_size);
}

Ray::Vk::MemoryAllocator::~MemoryAllocator() {
    for (MemPool &pool : pools_) {
        ctx_->api().vkFreeMemory(ctx_->device(), pool.mem, nullptr);
    }
}

bool Ray::Vk::MemoryAllocator::AllocateNewPool(const uint32_t size) {
    char buf_name[48];
    snprintf(buf_name, sizeof(buf_name), "%s pool %i", name_, int(pools_.size()));

    pools_.emplace_back();
    MemPool &new_pool = pools_.back();
    new_pool.size = size;

    const uint16_t pool_ndx = alloc_.AddPool(size);
    assert(pool_ndx == pools_.size() - 1);

    VkMemoryAllocateInfo buf_alloc_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    buf_alloc_info.allocationSize = VkDeviceSize(size);
    buf_alloc_info.memoryTypeIndex = mem_type_index_;

    const VkResult res = ctx_->api().vkAllocateMemory(ctx_->device(), &buf_alloc_info, nullptr, &new_pool.mem);
    return res == VK_SUCCESS;
}

Ray::Vk::MemAllocation Ray::Vk::MemoryAllocator::Allocate(const uint32_t alignment, const uint32_t size) {
    auto allocation = alloc_.Alloc(alignment, size);

    if (allocation.block == 0xffffffff) {
        const bool res = AllocateNewPool(std::max(size, uint32_t(pools_.back().size * growth_factor_)));
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

void Ray::Vk::MemoryAllocator::Free(const uint32_t block) {
    alloc_.Free(block);
    assert(alloc_.IntegrityCheck());
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