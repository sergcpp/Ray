#include "MemoryAllocatorVK.h"

#include "ContextVK.h"

#pragma warning(push)
#pragma warning(disable : 4996) // function or variable may be unsafe

namespace Ray {
namespace Vk {
uint32_t FindMemoryType(uint32_t search_from, const VkPhysicalDeviceMemoryProperties *mem_properties,
                        uint32_t mem_type_bits, VkMemoryPropertyFlags desired_mem_flags, VkDeviceSize desired_size) {
    for (uint32_t i = search_from; i < 32; i++) {
        const VkMemoryType mem_type = mem_properties->memoryTypes[i];
        if (mem_type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD) {
            // skip for now
            continue;
        }
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

Ray::Vk::MemoryAllocator::MemoryAllocator(const char name[32], Context *ctx, const uint32_t initial_pool_size,
                                          uint32_t mem_type_index, const float growth_factor,
                                          const uint32_t max_pool_size)
    : ctx_(ctx), growth_factor_(growth_factor), max_pool_size_(max_pool_size), mem_type_index_(mem_type_index) {
    strcpy(name_, name);

    assert(growth_factor_ > 1.0f);
    AllocateNewPool(initial_pool_size);
}

Ray::Vk::MemoryAllocator::~MemoryAllocator() {
    for (MemPool &pool : pools_) {
        ctx_->api().vkFreeMemory(ctx_->device(), pool.mem, nullptr);
    }
}

bool Ray::Vk::MemoryAllocator::AllocateNewPool(const uint32_t size) {
    VkMemoryAllocateInfo buf_alloc_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    buf_alloc_info.allocationSize = VkDeviceSize(size);
    buf_alloc_info.memoryTypeIndex = mem_type_index_;

    VkDeviceMemory new_mem = {};
    const VkResult res = ctx_->api().vkAllocateMemory(ctx_->device(), &buf_alloc_info, nullptr, &new_mem);
    if (res == VK_SUCCESS) {
        pools_.emplace_back();
        MemPool &new_pool = pools_.back();
        new_pool.mem = new_mem;
        new_pool.size = size;

        const uint16_t pool_ndx = alloc_.AddPool(size);
        assert(pool_ndx == pools_.size() - 1);
    }
    return res == VK_SUCCESS;
}

Ray::Vk::MemAllocation Ray::Vk::MemoryAllocator::Allocate(const uint32_t alignment, const uint32_t size) {
    auto allocation = alloc_.Alloc(alignment, size);

    if (allocation.block == 0xffffffff) {
        const bool res =
            AllocateNewPool(std::max(size, std::min(max_pool_size_, uint32_t(pools_.back().size * growth_factor_))));
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

Ray::Vk::MemAllocation Ray::Vk::MemoryAllocators::Allocate(const uint32_t alignment, const uint32_t size,
                                                           const uint32_t mem_type_index) {
    if (mem_type_index == 0xffffffff) {
        return {};
    }

    int alloc_index = -1;
    for (int i = 0; i < int(allocators_.size()); ++i) {
        if (allocators_[i].mem_type_index() == mem_type_index) {
            alloc_index = i;
            break;
        }
    }

    if (alloc_index == -1) {
        char name[32];
        snprintf(name, sizeof(name), "%s (type %i)", name_, int(mem_type_index));
        alloc_index = int(allocators_.size());
        allocators_.emplace_back(name, ctx_, initial_pool_size_, mem_type_index, growth_factor_, max_pool_size_);
    }

    return allocators_[alloc_index].Allocate(alignment, size);
}

Ray::Vk::MemAllocation Ray::Vk::MemoryAllocators::Allocate(const VkMemoryRequirements &mem_req,
                                                           const VkMemoryPropertyFlags desired_mem_flags) {
    uint32_t mem_type_index =
        FindMemoryType(0, &ctx_->mem_properties(), mem_req.memoryTypeBits, desired_mem_flags, uint32_t(mem_req.size));
    while (mem_type_index != 0xffffffff) {
        MemAllocation alloc = Allocate(uint32_t(mem_req.alignment), uint32_t(mem_req.size), mem_type_index);
        if (alloc) {
            return alloc;
        }
        mem_type_index = FindMemoryType(mem_type_index + 1, &ctx_->mem_properties(), mem_req.memoryTypeBits,
                                        desired_mem_flags, uint32_t(mem_req.size));
    }
    return {};
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