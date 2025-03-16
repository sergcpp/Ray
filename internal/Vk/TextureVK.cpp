#include "TextureVK.h"

#include <memory>

#include "../../Config.h"
#include "../../Log.h"
#include "../TextureUtils.h"
#include "ContextVK.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#ifndef NDEBUG
// #define TEX_VERBOSE_LOGGING
#endif

namespace Ray {
namespace Vk {
extern const VkFilter g_min_mag_filter_vk[];
extern const VkSamplerAddressMode g_wrap_mode_vk[];
extern const VkSamplerMipmapMode g_mipmap_mode_vk[];
extern const VkCompareOp g_compare_ops_vk[];

extern const float AnisotropyLevel;

#define X(_0, _1, _2, _3, _4, _5, _6) _5,
extern const VkFormat g_formats_vk[] = {
#include "../TextureFormat.inl"
};
#undef X

uint32_t FindMemoryType(uint32_t search_from, const VkPhysicalDeviceMemoryProperties *mem_properties,
                        uint32_t mem_type_bits, VkMemoryPropertyFlags desired_mem_flags, VkDeviceSize desired_size);

uint32_t TextureHandleCounter = 0;

// make sure we can simply cast these
static_assert(VkSampleCountFlagBits::VK_SAMPLE_COUNT_1_BIT == 1, "!");
static_assert(VkSampleCountFlagBits::VK_SAMPLE_COUNT_2_BIT == 2, "!");
static_assert(VkSampleCountFlagBits::VK_SAMPLE_COUNT_4_BIT == 4, "!");
static_assert(VkSampleCountFlagBits::VK_SAMPLE_COUNT_8_BIT == 8, "!");

VkFormat ToSRGBFormat(const VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_UNORM:
        return VK_FORMAT_R8_SRGB;
    case VK_FORMAT_R8G8_UNORM:
        return VK_FORMAT_R8G8_SRGB;
    case VK_FORMAT_R8G8B8_UNORM:
        return VK_FORMAT_R8G8B8_SRGB;
    case VK_FORMAT_R8G8B8A8_UNORM:
        return VK_FORMAT_R8G8B8A8_SRGB;
    case VK_FORMAT_B8G8R8A8_UNORM:
        return VK_FORMAT_B8G8R8A8_SRGB;
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
        return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
    case VK_FORMAT_BC2_UNORM_BLOCK:
        return VK_FORMAT_BC2_SRGB_BLOCK;
    case VK_FORMAT_BC3_UNORM_BLOCK:
        return VK_FORMAT_BC3_SRGB_BLOCK;
    case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:
        return VK_FORMAT_ASTC_4x4_SRGB_BLOCK;
    case VK_FORMAT_ASTC_5x4_UNORM_BLOCK:
        return VK_FORMAT_ASTC_5x4_SRGB_BLOCK;
    case VK_FORMAT_ASTC_5x5_UNORM_BLOCK:
        return VK_FORMAT_ASTC_5x5_SRGB_BLOCK;
    case VK_FORMAT_ASTC_6x5_UNORM_BLOCK:
        return VK_FORMAT_ASTC_6x5_SRGB_BLOCK;
    case VK_FORMAT_ASTC_6x6_UNORM_BLOCK:
        return VK_FORMAT_ASTC_6x6_SRGB_BLOCK;
    case VK_FORMAT_ASTC_8x5_UNORM_BLOCK:
        return VK_FORMAT_ASTC_8x5_SRGB_BLOCK;
    case VK_FORMAT_ASTC_8x6_UNORM_BLOCK:
        return VK_FORMAT_ASTC_8x6_SRGB_BLOCK;
    case VK_FORMAT_ASTC_8x8_UNORM_BLOCK:
        return VK_FORMAT_ASTC_8x8_SRGB_BLOCK;
    case VK_FORMAT_ASTC_10x5_UNORM_BLOCK:
        return VK_FORMAT_ASTC_10x5_SRGB_BLOCK;
    case VK_FORMAT_ASTC_10x6_UNORM_BLOCK:
        return VK_FORMAT_ASTC_10x6_SRGB_BLOCK;
    case VK_FORMAT_ASTC_10x8_UNORM_BLOCK:
        return VK_FORMAT_ASTC_10x8_SRGB_BLOCK;
    case VK_FORMAT_ASTC_10x10_UNORM_BLOCK:
        return VK_FORMAT_ASTC_10x10_SRGB_BLOCK;
    case VK_FORMAT_ASTC_12x10_UNORM_BLOCK:
        return VK_FORMAT_ASTC_12x10_SRGB_BLOCK;
    case VK_FORMAT_ASTC_12x12_UNORM_BLOCK:
        return VK_FORMAT_ASTC_12x12_SRGB_BLOCK;
    default:
        return format;
    }
    return VK_FORMAT_UNDEFINED;
}

VkImageUsageFlags to_vk_image_usage(const Bitmask<eTexUsage> usage, const eTexFormat format) {
    VkImageUsageFlags ret = 0;
    if (usage & eTexUsage::Transfer) {
        ret |= (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    }
    if (usage & eTexUsage::Sampled) {
        ret |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if (usage & eTexUsage::Storage) {
        assert(!IsCompressedFormat(format));
        ret |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    if (usage & eTexUsage::RenderTarget) {
        assert(!IsCompressedFormat(format));
        if (IsDepthFormat(format)) {
            ret |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        } else {
            ret |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        }
    }
    return ret;
}

const eTexUsage g_tex_usage_per_state[] = {
    {},                      // Undefined
    {},                      // VertexBuffer
    {},                      // UniformBuffer
    {},                      // IndexBuffer
    eTexUsage::RenderTarget, // RenderTarget
    eTexUsage::Storage,      // UnorderedAccess
    eTexUsage::RenderTarget, // DepthRead
    eTexUsage::RenderTarget, // DepthWrite
    eTexUsage::RenderTarget, // StencilTestDepthFetch
    eTexUsage::Sampled,      // ShaderResource
    {},                      // IndirectArgument
    eTexUsage::Transfer,     // CopyDst
    eTexUsage::Transfer,     // CopySrc
    {},                      // BuildASRead
    {},                      // BuildASWrite
    {}                       // RayTracing
};
static_assert(sizeof(g_tex_usage_per_state) / sizeof(g_tex_usage_per_state[0]) == int(eResState::_Count), "!");
} // namespace Vk

bool EndsWith(const std::string &str1, const char *str2);
} // namespace Ray

Ray::eTexUsage Ray::Vk::TexUsageFromState(const eResState state) { return g_tex_usage_per_state[int(state)]; }

Ray::Vk::Texture::Texture(const char *name, Context *ctx, const TexParams &p, MemAllocators *mem_allocs, ILog *log)
    : ctx_(ctx), name_(name) {
    Init(p, mem_allocs, log);
}

Ray::Vk::Texture::Texture(const char *name, Context *ctx, const void *data, const uint32_t size, const TexParams &p,
                          Buffer &stage_buf, VkCommandBuffer cmd_buf, MemAllocators *mem_allocs,
                          eTexLoadStatus *load_status, ILog *log)
    : ctx_(ctx), name_(name) {
    Init(data, size, p, stage_buf, cmd_buf, mem_allocs, load_status, log);
}

Ray::Vk::Texture::~Texture() { Free(); }

Ray::Vk::Texture &Ray::Vk::Texture::operator=(Texture &&rhs) noexcept {
    if (this == &rhs) {
        return (*this);
    }

    Free();

    ctx_ = std::exchange(rhs.ctx_, nullptr);
    handle_ = std::exchange(rhs.handle_, {});
    alloc_ = std::exchange(rhs.alloc_, {});
    params = std::exchange(rhs.params, {});
    name_ = std::move(rhs.name_);

    resource_state = std::exchange(rhs.resource_state, eResState::Undefined);

    return (*this);
}

void Ray::Vk::Texture::Init(const TexParams &p, MemAllocators *mem_allocs, ILog *log) {
    InitFromRAWData(nullptr, 0, nullptr, mem_allocs, p, log);
}

void Ray::Vk::Texture::Init(const void *data, const uint32_t size, const TexParams &p, Buffer &sbuf,
                            VkCommandBuffer cmd_buf, MemAllocators *mem_allocs, eTexLoadStatus *load_status,
                            ILog *log) {
    uint8_t *stage_data = sbuf.Map();
    memcpy(stage_data, data, size);
    sbuf.Unmap();

    InitFromRAWData(&sbuf, 0, cmd_buf, mem_allocs, p, log);

    (*load_status) = eTexLoadStatus::CreatedFromData;
}

void Ray::Vk::Texture::Free() {
    if (params.format != eTexFormat::Undefined && !bool(params.flags & eTexFlags::NoOwnership)) {
        for (VkImageView view : handle_.views) {
            if (view) {
                ctx_->image_views_to_destroy[ctx_->backend_frame].push_back(view);
            }
        }
        ctx_->images_to_destroy[ctx_->backend_frame].push_back(handle_.img);
        ctx_->samplers_to_destroy[ctx_->backend_frame].push_back(handle_.sampler);
        ctx_->allocs_to_free[ctx_->backend_frame].emplace_back(std::move(alloc_));

        handle_ = {};
        params.format = eTexFormat::Undefined;
    }
}

bool Ray::Vk::Texture::Realloc(const int w, const int h, int mip_count, const int samples, const eTexFormat format,
                               const bool is_srgb, VkCommandBuffer cmd_buf, MemAllocators *mem_allocs, ILog *log) {
    VkImage new_image = VK_NULL_HANDLE;
    VkImageView new_image_view = VK_NULL_HANDLE;
    MemAllocation new_alloc = {};
    eResState new_resource_state = eResState::Undefined;

    { // create new image
        VkImageCreateInfo img_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        img_info.imageType = VK_IMAGE_TYPE_2D;
        img_info.extent.width = uint32_t(w);
        img_info.extent.height = uint32_t(h);
        img_info.extent.depth = 1;
        img_info.mipLevels = mip_count;
        img_info.arrayLayers = 1;
        img_info.format = g_formats_vk[size_t(format)];
        if (is_srgb) {
            img_info.format = ToSRGBFormat(img_info.format);
        }
        img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        img_info.usage = to_vk_image_usage(params.usage, format);

        img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        img_info.samples = VkSampleCountFlagBits(samples);
        img_info.flags = 0;

        VkResult res = ctx_->api().vkCreateImage(ctx_->device(), &img_info, nullptr, &new_image);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image!");
            return false;
        }

#ifdef ENABLE_GPU_DEBUG
        VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectType = VK_OBJECT_TYPE_IMAGE;
        name_info.objectHandle = uint64_t(new_image);
        name_info.pObjectName = name_.c_str();
        ctx_->api().vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif

        VkMemoryRequirements tex_mem_req;
        ctx_->api().vkGetImageMemoryRequirements(ctx_->device(), new_image, &tex_mem_req);

        VkMemoryPropertyFlags img_tex_desired_mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        new_alloc = mem_allocs->Allocate(tex_mem_req, img_tex_desired_mem_flags);
        if (!new_alloc) {
            ctx_->log()->Warning("Not enough device memory, falling back to CPU RAM!");
            img_tex_desired_mem_flags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            new_alloc = mem_allocs->Allocate(tex_mem_req, img_tex_desired_mem_flags);
        }

        if (!new_alloc) {
            log->Error("Failed to allocate memory!");
            return false;
        }

        res = ctx_->api().vkBindImageMemory(ctx_->device(), new_image, new_alloc.owner->mem(new_alloc.pool),
                                            VkDeviceSize(new_alloc.offset));
        if (res != VK_SUCCESS) {
            log->Error("Failed to bind memory!");
            return false;
        }
    }

    { // create new image view
        VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = new_image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = g_formats_vk[size_t(format)];
        if (is_srgb) {
            view_info.format = ToSRGBFormat(view_info.format);
        }
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = mip_count;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        if (GetChannelCount(format) == 1) {
            view_info.components.r = VK_COMPONENT_SWIZZLE_R;
            view_info.components.g = VK_COMPONENT_SWIZZLE_R;
            view_info.components.b = VK_COMPONENT_SWIZZLE_R;
            view_info.components.a = VK_COMPONENT_SWIZZLE_R;
        }

        const VkResult res = ctx_->api().vkCreateImageView(ctx_->device(), &view_info, nullptr, &new_image_view);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image view!");
            return false;
        }

#ifdef ENABLE_GPU_DEBUG
        VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
        name_info.objectHandle = uint64_t(new_image_view);
        name_info.pObjectName = name_.c_str();
        ctx_->api().vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif
    }

#ifdef TEX_VERBOSE_LOGGING
    if (params_.format != eTexFormat::Undefined) {
        log->Info("Realloc %s, %ix%i (%i mips) -> %ix%i (%i mips)", name_.c_str(), int(params_.w), int(params_.h),
                  int(params_.mip_count), w, h, mip_count);
    } else {
        log->Info("Alloc %s %ix%i (%i mips)", name_.c_str(), w, h, mip_count);
    }
#endif

    const TexHandle new_handle = {new_image, new_image_view, VK_NULL_HANDLE, std::exchange(handle_.sampler, {}),
                                  TextureHandleCounter++};

    // copy data from old texture
    if (params.format == format) {
        int src_mip = 0, dst_mip = 0;
        while (std::max(params.w >> src_mip, 1) != std::max(w >> dst_mip, 1) ||
               std::max(params.h >> src_mip, 1) != std::max(h >> dst_mip, 1)) {
            if (std::max(params.w >> src_mip, 1) > std::max(w >> dst_mip, 1) ||
                std::max(params.h >> src_mip, 1) > std::max(h >> dst_mip, 1)) {
                ++src_mip;
            } else {
                ++dst_mip;
            }
        }

        VkImageCopy copy_regions[16];
        uint32_t copy_regions_count = 0;

        for (; src_mip < int(params.mip_count) && dst_mip < mip_count; ++src_mip, ++dst_mip) {
            VkImageCopy &reg = copy_regions[copy_regions_count++];

            reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            reg.srcSubresource.baseArrayLayer = 0;
            reg.srcSubresource.layerCount = 1;
            reg.srcSubresource.mipLevel = src_mip;
            reg.srcOffset = {0, 0, 0};
            reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            reg.dstSubresource.baseArrayLayer = 0;
            reg.dstSubresource.layerCount = 1;
            reg.dstSubresource.mipLevel = dst_mip;
            reg.dstOffset = {0, 0, 0};
            reg.extent = {uint32_t(std::max(w >> dst_mip, 1)), uint32_t(std::max(h >> dst_mip, 1)), 1};

#ifdef TEX_VERBOSE_LOGGING
            log->Info("Copying data mip %i [old] -> mip %i [new]", src_mip, dst_mip);
#endif
        }

        if (copy_regions_count) {
            VkPipelineStageFlags src_stages = 0, dst_stages = 0;
            SmallVector<VkImageMemoryBarrier, 2> barriers;

            // src image barrier
            if (this->resource_state != eResState::CopySrc) {
                auto &new_barrier = barriers.emplace_back();
                new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                new_barrier.srcAccessMask = VKAccessFlagsForState(this->resource_state);
                new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
                new_barrier.oldLayout = VKImageLayoutForState(this->resource_state);
                new_barrier.newLayout = VKImageLayoutForState(eResState::CopySrc);
                new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                new_barrier.image = handle_.img;
                new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                new_barrier.subresourceRange.baseMipLevel = 0;
                new_barrier.subresourceRange.levelCount = params.mip_count; // transit the whole image
                new_barrier.subresourceRange.baseArrayLayer = 0;
                new_barrier.subresourceRange.layerCount = 1;

                src_stages |= VKPipelineStagesForState(this->resource_state);
                dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
            }

            // dst image barrier
            if (new_resource_state != eResState::CopyDst) {
                auto &new_barrier = barriers.emplace_back();
                new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                new_barrier.srcAccessMask = VKAccessFlagsForState(new_resource_state);
                new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
                new_barrier.oldLayout = VKImageLayoutForState(new_resource_state);
                new_barrier.newLayout = VKImageLayoutForState(eResState::CopyDst);
                new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                new_barrier.image = new_image;
                new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                new_barrier.subresourceRange.baseMipLevel = 0;
                new_barrier.subresourceRange.levelCount = mip_count; // transit the whole image
                new_barrier.subresourceRange.baseArrayLayer = 0;
                new_barrier.subresourceRange.layerCount = 1;

                src_stages |= VKPipelineStagesForState(new_resource_state);
                dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
            }

            src_stages &= ctx_->supported_stages_mask();
            dst_stages &= ctx_->supported_stages_mask();

            if (!barriers.empty()) {
                ctx_->api().vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                                 dst_stages, 0, 0, nullptr, 0, nullptr, barriers.size(),
                                                 barriers.cdata());
            }

            this->resource_state = eResState::CopySrc;
            new_resource_state = eResState::CopyDst;

            ctx_->api().vkCmdCopyImage(cmd_buf, handle_.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, new_image,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copy_regions_count, copy_regions);
        }
    }
    Free();

    handle_ = new_handle;
    alloc_ = std::move(new_alloc);
    params.w = w;
    params.h = h;
    if (is_srgb) {
        params.flags |= eTexFlags::SRGB;
    } else {
        params.flags &= ~Bitmask<eTexFlags>(eTexFlags::SRGB);
    }
    params.mip_count = mip_count;
    params.samples = samples;
    params.format = format;

    this->resource_state = new_resource_state;

    return true;
}

void Ray::Vk::Texture::InitFromRAWData(Buffer *sbuf, int data_off, VkCommandBuffer cmd_buf, MemAllocators *mem_allocs,
                                       const TexParams &p, ILog *log) {
    Free();

    handle_.generation = TextureHandleCounter++;
    params = p;

    { // create image
        VkImageCreateInfo img_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        img_info.imageType = p.d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        img_info.extent.width = p.w;
        img_info.extent.height = p.h;
        img_info.extent.depth = (p.d ? p.d : 1);
        img_info.mipLevels = params.mip_count;
        img_info.arrayLayers = 1;
        img_info.format = g_formats_vk[size_t(p.format)];
        if (p.flags & eTexFlags::SRGB) {
            img_info.format = ToSRGBFormat(img_info.format);
        }
        img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        img_info.usage = to_vk_image_usage(p.usage, p.format);

        img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        img_info.samples = VkSampleCountFlagBits(p.samples);
        img_info.flags = 0;

        VkResult res = ctx_->api().vkCreateImage(ctx_->device(), &img_info, nullptr, &handle_.img);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image!");
            return;
        }

#ifdef ENABLE_GPU_DEBUG
        VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectType = VK_OBJECT_TYPE_IMAGE;
        name_info.objectHandle = uint64_t(handle_.img);
        name_info.pObjectName = name_.c_str();
        ctx_->api().vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif

        VkMemoryRequirements tex_mem_req;
        ctx_->api().vkGetImageMemoryRequirements(ctx_->device(), handle_.img, &tex_mem_req);

        VkMemoryPropertyFlags img_tex_desired_mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        alloc_ = mem_allocs->Allocate(tex_mem_req, img_tex_desired_mem_flags);
        if (!alloc_) {
            ctx_->log()->Warning("Not enough device memory, falling back to CPU RAM!");
            img_tex_desired_mem_flags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            alloc_ = mem_allocs->Allocate(tex_mem_req, img_tex_desired_mem_flags);
        }

        if (!alloc_) {
            log->Error("Failed to allocate memory!");
            return;
        }

        res = ctx_->api().vkBindImageMemory(ctx_->device(), handle_.img, alloc_.owner->mem(alloc_.pool),
                                            VkDeviceSize(alloc_.offset));
        if (res != VK_SUCCESS) {
            log->Error("Failed to bind memory!");
            return;
        }
    }

    { // create default image view(s)
        VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = handle_.img;
        view_info.viewType = p.d ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = g_formats_vk[size_t(p.format)];
        if (p.flags & eTexFlags::SRGB) {
            view_info.format = ToSRGBFormat(view_info.format);
        }
        if (IsDepthStencilFormat(p.format)) {
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        } else if (IsDepthFormat(p.format)) {
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        } else {
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = params.mip_count;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        if (GetChannelCount(p.format) == 1 && !bool(p.usage & eTexUsage::Storage)) {
            view_info.components.r = VK_COMPONENT_SWIZZLE_R;
            view_info.components.g = VK_COMPONENT_SWIZZLE_R;
            view_info.components.b = VK_COMPONENT_SWIZZLE_R;
            view_info.components.a = VK_COMPONENT_SWIZZLE_R;
        }

        const VkResult res = ctx_->api().vkCreateImageView(ctx_->device(), &view_info, nullptr, &handle_.views[0]);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image view!");
            return;
        }

        if (IsDepthStencilFormat(p.format)) {
            // create additional depth-only image view
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            VkImageView depth_only_view;
            const VkResult res = ctx_->api().vkCreateImageView(ctx_->device(), &view_info, nullptr, &depth_only_view);
            if (res != VK_SUCCESS) {
                log->Error("Failed to create image view!");
                return;
            }
            handle_.views.push_back(depth_only_view);
        }

#ifdef ENABLE_GPU_DEBUG
        for (VkImageView view : handle_.views) {
            VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
            name_info.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
            name_info.objectHandle = uint64_t(view);
            name_info.pObjectName = name_.c_str();
            ctx_->api().vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
        }
#endif
    }

    this->resource_state = eResState::Undefined;

    if (sbuf) {
        assert(p.samples == 1);
        assert(sbuf && sbuf->type() == eBufType::Upload);

        VkPipelineStageFlags src_stages = 0, dst_stages = 0;
        SmallVector<VkBufferMemoryBarrier, 1> buf_barriers;
        SmallVector<VkImageMemoryBarrier, 1> img_barriers;

        if (sbuf->resource_state != eResState::Undefined && sbuf->resource_state != eResState::CopySrc) {
            auto &new_barrier = buf_barriers.emplace_back();
            new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            new_barrier.srcAccessMask = VKAccessFlagsForState(sbuf->resource_state);
            new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
            new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.buffer = sbuf->vk_handle();
            new_barrier.offset = VkDeviceSize(data_off);
            new_barrier.size = VkDeviceSize(sbuf->size() - data_off);

            src_stages |= VKPipelineStagesForState(sbuf->resource_state);
            dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
        }

        if (this->resource_state != eResState::CopyDst) {
            auto &new_barrier = img_barriers.emplace_back();
            new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            new_barrier.srcAccessMask = VKAccessFlagsForState(this->resource_state);
            new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
            new_barrier.oldLayout = VKImageLayoutForState(this->resource_state);
            new_barrier.newLayout = VKImageLayoutForState(eResState::CopyDst);
            new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.image = handle_.img;
            new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            new_barrier.subresourceRange.baseMipLevel = 0;
            new_barrier.subresourceRange.levelCount = params.mip_count; // transit whole image
            new_barrier.subresourceRange.baseArrayLayer = 0;
            new_barrier.subresourceRange.layerCount = 1;

            src_stages |= VKPipelineStagesForState(this->resource_state);
            dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
        }

        src_stages &= ctx_->supported_stages_mask();
        dst_stages &= ctx_->supported_stages_mask();

        if (!buf_barriers.empty() || !img_barriers.empty()) {
            ctx_->api().vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                             dst_stages, 0, 0, nullptr, buf_barriers.size(), buf_barriers.cdata(),
                                             img_barriers.size(), img_barriers.cdata());
        }

        sbuf->resource_state = eResState::CopySrc;
        this->resource_state = eResState::CopyDst;

        VkBufferImageCopy region = {};
        region.bufferOffset = VkDeviceSize(data_off);
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;

        region.imageOffset = {0, 0, 0};
        region.imageExtent = {uint32_t(p.w), uint32_t(p.h), 1};

        ctx_->api().vkCmdCopyBufferToImage(cmd_buf, sbuf->vk_handle(), handle_.img,
                                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }

    { // create new sampler
        VkSamplerCreateInfo sampler_info = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler_info.magFilter = g_min_mag_filter_vk[size_t(p.sampling.filter)];
        sampler_info.minFilter = g_min_mag_filter_vk[size_t(p.sampling.filter)];
        sampler_info.addressModeU = g_wrap_mode_vk[size_t(p.sampling.wrap)];
        sampler_info.addressModeV = g_wrap_mode_vk[size_t(p.sampling.wrap)];
        sampler_info.addressModeW = g_wrap_mode_vk[size_t(p.sampling.wrap)];
        sampler_info.anisotropyEnable = VK_TRUE;
        sampler_info.maxAnisotropy = AnisotropyLevel;
        sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        sampler_info.unnormalizedCoordinates = VK_FALSE;
        sampler_info.compareEnable = p.sampling.compare != eTexCompare::None ? VK_TRUE : VK_FALSE;
        sampler_info.compareOp = g_compare_ops_vk[size_t(p.sampling.compare)];
        sampler_info.mipmapMode = g_mipmap_mode_vk[size_t(p.sampling.filter)];
        sampler_info.mipLodBias = p.sampling.lod_bias.to_float();
        sampler_info.minLod = 0.0f;
        sampler_info.maxLod = VK_LOD_CLAMP_NONE;

        const VkResult res = ctx_->api().vkCreateSampler(ctx_->device(), &sampler_info, nullptr, &handle_.sampler);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create sampler!");
        }
    }
}

void Ray::Vk::Texture::SetSubImage(const int level, const int offsetx, const int offsety, const int offsetz,
                                   const int sizex, const int sizey, const int sizez, const eTexFormat format,
                                   const Buffer &sbuf, VkCommandBuffer cmd_buf, const int data_off,
                                   const int data_len) {
    assert(format == params.format);
    assert(params.samples == 1);
    assert(offsetx >= 0 && offsetx + sizex <= std::max(params.w >> level, 1));
    assert(offsety >= 0 && offsety + sizey <= std::max(params.h >> level, 1));
    assert(offsetz >= 0 && offsetz + sizez <= std::max(params.d >> level, 1));
    assert(sbuf.type() == eBufType::Upload);

    VkPipelineStageFlags src_stages = 0, dst_stages = 0;
    SmallVector<VkBufferMemoryBarrier, 1> buf_barriers;
    SmallVector<VkImageMemoryBarrier, 1> img_barriers;

    if (sbuf.resource_state != eResState::Undefined && sbuf.resource_state != eResState::CopySrc) {
        auto &new_barrier = buf_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(sbuf.resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.buffer = sbuf.vk_handle();
        new_barrier.offset = VkDeviceSize(0);
        new_barrier.size = VkDeviceSize(sbuf.size());

        src_stages |= VKPipelineStagesForState(sbuf.resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
    }

    if (this->resource_state != eResState::CopyDst) {
        auto &new_barrier = img_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(this->resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
        new_barrier.oldLayout = VKImageLayoutForState(this->resource_state);
        new_barrier.newLayout = VKImageLayoutForState(eResState::CopyDst);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.image = handle_.img;
        new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        new_barrier.subresourceRange.baseMipLevel = 0;
        new_barrier.subresourceRange.levelCount = params.mip_count; // transit whole image
        new_barrier.subresourceRange.baseArrayLayer = 0;
        new_barrier.subresourceRange.layerCount = 1;

        src_stages |= VKPipelineStagesForState(this->resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
    }

    src_stages &= ctx_->supported_stages_mask();
    dst_stages &= ctx_->supported_stages_mask();

    if (!buf_barriers.empty() || !img_barriers.empty()) {
        ctx_->api().vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                         dst_stages, 0, 0, nullptr, buf_barriers.size(), buf_barriers.cdata(),
                                         img_barriers.size(), img_barriers.cdata());
    }

    sbuf.resource_state = eResState::CopySrc;
    this->resource_state = eResState::CopyDst;

    VkBufferImageCopy region = {};

    region.bufferOffset = VkDeviceSize(data_off);
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = uint32_t(level);
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {int32_t(offsetx), int32_t(offsety), int32_t(offsetz)};
    region.imageExtent = {uint32_t(sizex), uint32_t(sizey), uint32_t(sizez)};

    ctx_->api().vkCmdCopyBufferToImage(cmd_buf, sbuf.vk_handle(), handle_.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                       &region);
}

void Ray::Vk::Texture::SetSampling(const SamplingParams s) {
    if (handle_.sampler) {
        ctx_->samplers_to_destroy[ctx_->backend_frame].emplace_back(handle_.sampler);
    }

    VkSamplerCreateInfo sampler_info = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_info.magFilter = g_min_mag_filter_vk[size_t(s.filter)];
    sampler_info.minFilter = g_min_mag_filter_vk[size_t(s.filter)];
    sampler_info.addressModeU = g_wrap_mode_vk[size_t(s.wrap)];
    sampler_info.addressModeV = g_wrap_mode_vk[size_t(s.wrap)];
    sampler_info.addressModeW = g_wrap_mode_vk[size_t(s.wrap)];
    sampler_info.anisotropyEnable = VK_TRUE;
    sampler_info.maxAnisotropy = AnisotropyLevel;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = s.compare != eTexCompare::None ? VK_TRUE : VK_FALSE;
    sampler_info.compareOp = g_compare_ops_vk[size_t(s.compare)];
    sampler_info.mipmapMode = g_mipmap_mode_vk[size_t(s.filter)];
    sampler_info.mipLodBias = s.lod_bias.to_float();
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = VK_LOD_CLAMP_NONE;

    const VkResult res = ctx_->api().vkCreateSampler(ctx_->device(), &sampler_info, nullptr, &handle_.sampler);
    if (res != VK_SUCCESS) {
        ctx_->log()->Error("Failed to create sampler!");
    }

    params.sampling = s;
}

void Ray::Vk::CopyImageToImage(VkCommandBuffer cmd_buf, Texture &src_tex, const uint32_t src_level,
                               const uint32_t src_x, const uint32_t src_y, Texture &dst_tex, const uint32_t dst_level,
                               const uint32_t dst_x, const uint32_t dst_y, const uint32_t width,
                               const uint32_t height) {
    assert(src_tex.resource_state == eResState::CopySrc);
    assert(dst_tex.resource_state == eResState::CopyDst);

    VkImageCopy reg;
    if (IsDepthFormat(src_tex.params.format)) {
        reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    } else {
        reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }
    reg.srcSubresource.baseArrayLayer = 0;
    reg.srcSubresource.layerCount = 1;
    reg.srcSubresource.mipLevel = src_level;
    reg.srcOffset = {int32_t(src_x), int32_t(src_y), 0};
    if (IsDepthFormat(dst_tex.params.format)) {
        reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    } else {
        reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }
    reg.dstSubresource.baseArrayLayer = 0;
    reg.dstSubresource.layerCount = 1;
    reg.dstSubresource.mipLevel = dst_level;
    reg.dstOffset = {int32_t(dst_x), int32_t(dst_y), 0};
    reg.extent = {width, height, 1};

    src_tex.ctx()->api().vkCmdCopyImage(cmd_buf, src_tex.handle().img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                        dst_tex.handle().img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
}

void Ray::Vk::CopyImageToBuffer(const Texture &src_tex, const int level, const int x, const int y, const int w,
                                const int h, const Buffer &dst_buf, VkCommandBuffer cmd_buf, const int data_off) {
    VkPipelineStageFlags src_stages = 0, dst_stages = 0;
    SmallVector<VkBufferMemoryBarrier, 1> buf_barriers;
    SmallVector<VkImageMemoryBarrier, 1> img_barriers;

    if (src_tex.resource_state != eResState::CopySrc) {
        auto &new_barrier = img_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(src_tex.resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
        new_barrier.oldLayout = VKImageLayoutForState(src_tex.resource_state);
        new_barrier.newLayout = VKImageLayoutForState(eResState::CopySrc);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.image = src_tex.handle().img;
        new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        new_barrier.subresourceRange.baseMipLevel = 0;
        new_barrier.subresourceRange.levelCount = src_tex.params.mip_count; // transit whole image
        new_barrier.subresourceRange.baseArrayLayer = 0;
        new_barrier.subresourceRange.layerCount = 1;

        src_stages |= VKPipelineStagesForState(src_tex.resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
    }

    if (dst_buf.resource_state != eResState::Undefined && dst_buf.resource_state != eResState::CopyDst) {
        auto &new_barrier = buf_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(dst_buf.resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.buffer = dst_buf.vk_handle();
        new_barrier.offset = VkDeviceSize(0);
        new_barrier.size = VkDeviceSize(dst_buf.size());

        src_stages |= VKPipelineStagesForState(dst_buf.resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
    }

    src_stages &= src_tex.ctx()->supported_stages_mask();
    dst_stages &= src_tex.ctx()->supported_stages_mask();

    if (!buf_barriers.empty() || !img_barriers.empty()) {
        src_tex.ctx()->api().vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                                  dst_stages, 0, 0, nullptr, buf_barriers.size(), buf_barriers.cdata(),
                                                  img_barriers.size(), img_barriers.cdata());
    }

    src_tex.resource_state = eResState::CopySrc;
    dst_buf.resource_state = eResState::CopyDst;

    VkBufferImageCopy region = {};

    region.bufferOffset = VkDeviceSize(data_off);
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = uint32_t(level);
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {int32_t(x), int32_t(y), 0};
    region.imageExtent = {uint32_t(w), uint32_t(h), 1};

    src_tex.ctx()->api().vkCmdCopyImageToBuffer(cmd_buf, src_tex.handle().img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                dst_buf.vk_handle(), 1, &region);
}

void Ray::Vk::ClearColorImage(Texture &tex, const float rgba[4], VkCommandBuffer cmd_buf) {
    assert(tex.resource_state == eResState::CopyDst);

    VkClearColorValue clear_val = {};
    memcpy(clear_val.float32, rgba, 4 * sizeof(float));

    VkImageSubresourceRange clear_range = {};
    clear_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clear_range.layerCount = 1;
    clear_range.levelCount = 1;

    tex.ctx()->api().vkCmdClearColorImage(cmd_buf, tex.handle().img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_val,
                                          1, &clear_range);
}

void Ray::Vk::ClearColorImage(Texture &tex, const uint32_t rgba[4], VkCommandBuffer cmd_buf) {
    assert(tex.resource_state == eResState::CopyDst);

    VkClearColorValue clear_val = {};
    memcpy(clear_val.uint32, rgba, 4 * sizeof(uint32_t));

    VkImageSubresourceRange clear_range = {};
    clear_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clear_range.layerCount = 1;
    clear_range.levelCount = 1;

    tex.ctx()->api().vkCmdClearColorImage(cmd_buf, tex.handle().img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_val,
                                          1, &clear_range);
}

////////////////////////////////////////////////////////////////////////////////////////

VkFormat Ray::Vk::VKFormatFromTexFormat(const eTexFormat format) { return g_formats_vk[size_t(format)]; }

bool Ray::Vk::RequiresManualSRGBConversion(const eTexFormat format) {
    const VkFormat vk_format = g_formats_vk[size_t(format)];
    return vk_format == ToSRGBFormat(vk_format);
}

bool Ray::Vk::CanBeBlockCompressed(int w, int h, int mip_count) {
    // assume non-multiple of block size resolutions are supported
    return true;
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
