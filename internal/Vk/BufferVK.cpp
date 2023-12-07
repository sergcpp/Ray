#include "BufferVK.h"

#include <algorithm>
#include <cassert>

#include "../../Log.h"
#include "ContextVK.h"

namespace Ray {
namespace Vk {
VkBufferUsageFlags GetVkBufferUsageFlags(const Context *ctx, const eBufType type) {
    VkBufferUsageFlags flags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    if (type == eBufType::VertexAttribs) {
        flags |= (VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    } else if (type == eBufType::VertexIndices) {
        flags |= (VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    } else if (type == eBufType::Texture) {
        flags |= (VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    } else if (type == eBufType::Uniform) {
        flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    } else if (type == eBufType::Storage) {
        flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    } else if (type == eBufType::Upload) {
    } else if (type == eBufType::Readback) {
    } else if (type == eBufType::AccStructure) {
        flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
    } else if (type == eBufType::ShaderBinding) {
        flags |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    } else if (type == eBufType::Indirect) {
        flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }

    if ((type == eBufType::VertexAttribs || type == eBufType::VertexIndices || type == eBufType::Storage ||
         type == eBufType::Indirect) &&
        ctx->raytracing_supported()) {
        flags |= (VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    }

    return flags;
}

VkMemoryPropertyFlags GetVkMemoryPropertyFlags(const eBufType type) {
    if (type == eBufType::Upload || type == eBufType::Readback) {
        return (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
}

uint32_t FindMemoryType(uint32_t search_from, const VkPhysicalDeviceMemoryProperties *mem_properties,
                        uint32_t mem_type_bits, VkMemoryPropertyFlags desired_mem_flags, VkDeviceSize desired_size);
} // namespace Vk
} // namespace Ray

int Ray::Vk::Buffer::g_GenCounter = 0;

Ray::Vk::Buffer::Buffer(const char *name, Context *ctx, const eBufType type, const uint32_t initial_size)
    : ctx_(ctx), name_(name), type_(type), size_(0) {
    Resize(initial_size);
}

Ray::Vk::Buffer::~Buffer() { Free(); }

Ray::Vk::Buffer &Ray::Vk::Buffer::operator=(Buffer &&rhs) noexcept {
    Free();

    assert(!mapped_ptr_);
    assert(mapped_offset_ == 0xffffffff);

    ctx_ = exchange(rhs.ctx_, nullptr);
    handle_ = exchange(rhs.handle_, {});
    name_ = std::move(rhs.name_);
    mem_ = exchange(rhs.mem_, {});

    type_ = exchange(rhs.type_, eBufType::Undefined);

    size_ = exchange(rhs.size_, 0);
    mapped_ptr_ = exchange(rhs.mapped_ptr_, nullptr);
    mapped_offset_ = exchange(rhs.mapped_offset_, 0xffffffff);

    resource_state = exchange(rhs.resource_state, eResState::Undefined);

    return (*this);
}

VkDeviceAddress Ray::Vk::Buffer::vk_device_address() const {
    VkBufferDeviceAddressInfo addr_info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addr_info.buffer = handle_.buf;
    return ctx_->api().vkGetBufferDeviceAddressKHR(ctx_->device(), &addr_info);
}

void Ray::Vk::Buffer::UpdateSubRegion(const uint32_t offset, const uint32_t size, const Buffer &init_buf,
                                      const uint32_t init_off, void *_cmd_buf) {
    assert(init_buf.type_ == eBufType::Upload || init_buf.type_ == eBufType::Readback);
    auto cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

    VkPipelineStageFlags src_stages = 0, dst_stages = 0;
    SmallVector<VkBufferMemoryBarrier, 2> barriers;

    if (init_buf.resource_state != eResState::Undefined && init_buf.resource_state != eResState::CopySrc) {
        auto &new_barrier = barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(init_buf.resource_state);
        new_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.buffer = init_buf.vk_handle();
        new_barrier.offset = VkDeviceSize{init_off};
        new_barrier.size = VkDeviceSize{size};

        src_stages |= VKPipelineStagesForState(init_buf.resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
    }

    if (this->resource_state != eResState::Undefined && this->resource_state != eResState::CopyDst) {
        auto &new_barrier = barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(this->resource_state);
        new_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.buffer = handle_.buf;
        new_barrier.offset = VkDeviceSize{offset};
        new_barrier.size = VkDeviceSize{size};

        src_stages |= VKPipelineStagesForState(this->resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
    }

    if (!barriers.empty()) {
        ctx_->api().vkCmdPipelineBarrier(cmd_buf, src_stages, dst_stages, 0, 0, nullptr, uint32_t(barriers.size()),
                                         barriers.cdata(), 0, nullptr);
    }

    const VkBufferCopy region_to_copy = {
        VkDeviceSize{init_off}, // srcOffset
        VkDeviceSize{offset},   // dstOffset
        VkDeviceSize{size}      // size
    };

    ctx_->api().vkCmdCopyBuffer(cmd_buf, init_buf.handle_.buf, handle_.buf, 1, &region_to_copy);

    init_buf.resource_state = eResState::CopySrc;
    this->resource_state = eResState::CopyDst;
}

void Ray::Vk::Buffer::Resize(const uint32_t new_size, const bool keep_content) {
    if (size_ >= new_size) {
        return;
    }

    const uint32_t old_size = size_;

    size_ = new_size;
    assert(size_ > 0);

    VkBufferCreateInfo buf_create_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buf_create_info.size = VkDeviceSize(AlignMapOffsetUp(size_));
    buf_create_info.usage = GetVkBufferUsageFlags(ctx_, type_);
    buf_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer new_buf = {};
    VkResult res = ctx_->api().vkCreateBuffer(ctx_->device(), &buf_create_info, nullptr, &new_buf);
    assert(res == VK_SUCCESS && "Failed to create vertex buffer!");

#ifdef ENABLE_OBJ_LABELS
    VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    name_info.objectType = VK_OBJECT_TYPE_BUFFER;
    name_info.objectHandle = uint64_t(new_buf);
    name_info.pObjectName = name_.c_str();
    ctx_->api().vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif

    VkMemoryRequirements memory_requirements = {};
    ctx_->api().vkGetBufferMemoryRequirements(ctx_->device(), new_buf, &memory_requirements);

    VkMemoryPropertyFlags memory_props = GetVkMemoryPropertyFlags(type_);

    VkMemoryAllocateInfo buf_alloc_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    buf_alloc_info.allocationSize = memory_requirements.size;
    buf_alloc_info.memoryTypeIndex = FindMemoryType(0, &ctx_->mem_properties(), memory_requirements.memoryTypeBits,
                                                    memory_props, buf_alloc_info.allocationSize);

    VkMemoryAllocateFlagsInfoKHR additional_flags = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR};
    additional_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

    if ((buf_create_info.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0) {
        buf_alloc_info.pNext = &additional_flags;
    }

    VkDeviceMemory buffer_mem = {};

    res = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    while (buf_alloc_info.memoryTypeIndex != 0xffffffff) {
        res = ctx_->api().vkAllocateMemory(ctx_->device(), &buf_alloc_info, nullptr, &buffer_mem);
        if (res == VK_SUCCESS) {
            break;
        }
        buf_alloc_info.memoryTypeIndex =
            FindMemoryType(buf_alloc_info.memoryTypeIndex + 1, &ctx_->mem_properties(),
                           memory_requirements.memoryTypeBits, memory_props, buf_alloc_info.allocationSize);
    }
    if (res == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        ctx_->log()->Warning("Not enough device memory, falling back to CPU RAM!");
        memory_props &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        buf_alloc_info.memoryTypeIndex = FindMemoryType(0, &ctx_->mem_properties(), memory_requirements.memoryTypeBits,
                                                        memory_props, buf_alloc_info.allocationSize);
        while (buf_alloc_info.memoryTypeIndex != 0xffffffff) {
            res = ctx_->api().vkAllocateMemory(ctx_->device(), &buf_alloc_info, nullptr, &buffer_mem);
            if (res == VK_SUCCESS) {
                break;
            }
            buf_alloc_info.memoryTypeIndex =
                FindMemoryType(buf_alloc_info.memoryTypeIndex + 1, &ctx_->mem_properties(),
                               memory_requirements.memoryTypeBits, memory_props, buf_alloc_info.allocationSize);
        }
    }
    assert(res == VK_SUCCESS && "Failed to allocate memory!");

    res = ctx_->api().vkBindBufferMemory(ctx_->device(), new_buf, buffer_mem, 0 /* offset */);
    assert(res == VK_SUCCESS && "Failed to bind memory!");

    if (handle_.buf != VK_NULL_HANDLE) {
        if (keep_content) {
            VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

            VkBufferCopy region_to_copy = {};
            region_to_copy.size = VkDeviceSize{old_size};

            ctx_->api().vkCmdCopyBuffer(cmd_buf, handle_.buf, new_buf, 1, &region_to_copy);

            EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf,
                                  ctx_->temp_command_pool());

            // destroy previous buffer
            ctx_->api().vkDestroyBuffer(ctx_->device(), handle_.buf, nullptr);
            ctx_->api().vkFreeMemory(ctx_->device(), mem_, nullptr);
        } else {
            // destroy previous buffer
            ctx_->bufs_to_destroy[ctx_->backend_frame].push_back(handle_.buf);
            ctx_->mem_to_free[ctx_->backend_frame].push_back(mem_);
        }
    }

    handle_.buf = new_buf;
    handle_.generation = g_GenCounter++;
    mem_ = buffer_mem;
}

void Ray::Vk::Buffer::Free() {
    assert(mapped_offset_ == 0xffffffff && !mapped_ptr_);
    if (handle_.buf != VK_NULL_HANDLE) {
        ctx_->bufs_to_destroy[ctx_->backend_frame].push_back(handle_.buf);
        ctx_->mem_to_free[ctx_->backend_frame].push_back(mem_);

        handle_ = {};
        size_ = 0;
    }
}

void Ray::Vk::Buffer::FreeImmediate() {
    assert(mapped_offset_ == 0xffffffff && !mapped_ptr_);
    if (handle_.buf != VK_NULL_HANDLE) {
        ctx_->api().vkDestroyBuffer(ctx_->device(), handle_.buf, nullptr);
        ctx_->api().vkFreeMemory(ctx_->device(), mem_, nullptr);

        handle_ = {};
        size_ = 0;
    }
}

uint32_t Ray::Vk::Buffer::AlignMapOffset(const uint32_t offset) const {
    const auto align_to = uint32_t(ctx_->device_properties().limits.nonCoherentAtomSize);
    return offset - (offset % align_to);
}

uint32_t Ray::Vk::Buffer::AlignMapOffsetUp(const uint32_t offset) const {
    const auto align_to = uint32_t(ctx_->device_properties().limits.nonCoherentAtomSize);
    return align_to * ((offset + align_to - 1) / align_to);
}

uint8_t *Ray::Vk::Buffer::MapRange(const uint32_t offset, const uint32_t size, const bool persistent) {
    assert(mapped_offset_ == 0xffffffff && !mapped_ptr_);
    assert(offset + size <= size_);
    assert(type_ == eBufType::Upload || type_ == eBufType::Readback);
    assert(offset == AlignMapOffset(offset));
    assert((offset + size) == size_ || (offset + size) == AlignMapOffset(offset + size));

    void *mapped = nullptr;
    const VkResult res = ctx_->api().vkMapMemory(ctx_->device(), mem_, VkDeviceSize(offset),
                                                 VkDeviceSize(AlignMapOffsetUp(size)), 0, &mapped);
    if (res != VK_SUCCESS) {
        ctx_->log()->Error("Failed to map memory!");
        return nullptr;
    }

    mapped_ptr_ = reinterpret_cast<uint8_t *>(mapped);
    mapped_offset_ = offset;
    return reinterpret_cast<uint8_t *>(mapped);
}

void Ray::Vk::Buffer::Unmap() {
    assert(mapped_offset_ != 0xffffffff && mapped_ptr_);
    ctx_->api().vkUnmapMemory(ctx_->device(), mem_);
    mapped_ptr_ = nullptr;
    mapped_offset_ = 0xffffffff;
}

void Ray::Vk::Buffer::Fill(const uint32_t dst_offset, const uint32_t size, const uint32_t data, void *_cmd_buf) {
    auto cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

    VkPipelineStageFlags src_stages = 0, dst_stages = 0;
    SmallVector<VkBufferMemoryBarrier, 1> barriers;

    if (resource_state != eResState::CopyDst) {
        auto &new_barrier = barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.buffer = handle_.buf;
        new_barrier.offset = VkDeviceSize{dst_offset};
        new_barrier.size = VkDeviceSize{size};

        src_stages |= VKPipelineStagesForState(resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
    }

    if (!barriers.empty()) {
        ctx_->api().vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                         dst_stages, 0, 0, nullptr, uint32_t(barriers.size()), barriers.cdata(), 0,
                                         nullptr);
    }

    ctx_->api().vkCmdFillBuffer(cmd_buf, handle_.buf, VkDeviceSize{dst_offset}, VkDeviceSize{size}, data);

    resource_state = eResState::CopyDst;
}

void Ray::Vk::Buffer::UpdateImmediate(uint32_t dst_offset, uint32_t size, const void *data, void *_cmd_buf) {
    auto cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

    VkPipelineStageFlags src_stages = 0, dst_stages = 0;
    SmallVector<VkBufferMemoryBarrier, 1> barriers;

    if (resource_state != eResState::CopyDst) {
        auto &new_barrier = barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.buffer = handle_.buf;
        new_barrier.offset = VkDeviceSize{dst_offset};
        new_barrier.size = VkDeviceSize{size};

        src_stages |= VKPipelineStagesForState(resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
    }

    if (!barriers.empty()) {
        ctx_->api().vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                         dst_stages, 0, 0, nullptr, uint32_t(barriers.size()), barriers.cdata(), 0,
                                         nullptr);
    }

    ctx_->api().vkCmdUpdateBuffer(cmd_buf, handle_.buf, VkDeviceSize{dst_offset}, VkDeviceSize{size}, data);

    resource_state = eResState::CopyDst;
}

void Ray::Vk::CopyBufferToBuffer(Buffer &src, const uint32_t src_offset, Buffer &dst, const uint32_t dst_offset,
                                 const uint32_t size, void *_cmd_buf) {
    auto cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

    VkPipelineStageFlags src_stages = 0, dst_stages = 0;
    SmallVector<VkBufferMemoryBarrier, 2> barriers;

    if (src.resource_state != eResState::Undefined && src.resource_state != eResState::CopySrc) {
        auto &new_barrier = barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(src.resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.buffer = src.vk_handle();
        new_barrier.offset = VkDeviceSize{src_offset};
        new_barrier.size = VkDeviceSize{size};

        src_stages |= VKPipelineStagesForState(src.resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
    }

    if (dst.resource_state != eResState::Undefined && dst.resource_state != eResState::CopyDst) {
        auto &new_barrier = barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(dst.resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.buffer = dst.vk_handle();
        new_barrier.offset = VkDeviceSize{dst_offset};
        new_barrier.size = VkDeviceSize{size};

        src_stages |= VKPipelineStagesForState(dst.resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
    }

    if (!barriers.empty()) {
        src.ctx()->api().vkCmdPipelineBarrier(cmd_buf, src_stages, dst_stages, 0, 0, nullptr, uint32_t(barriers.size()),
                                              barriers.cdata(), 0, nullptr);
    }

    VkBufferCopy region_to_copy = {};
    region_to_copy.srcOffset = VkDeviceSize{src_offset};
    region_to_copy.dstOffset = VkDeviceSize{dst_offset};
    region_to_copy.size = VkDeviceSize{size};

    src.ctx()->api().vkCmdCopyBuffer(cmd_buf, src.vk_handle(), dst.vk_handle(), 1, &region_to_copy);

    src.resource_state = eResState::CopySrc;
    dst.resource_state = eResState::CopyDst;
}
