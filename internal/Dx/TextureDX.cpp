#include "TextureDX.h"

#include <memory>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12.h>

#include "../../Log.h"
#include "../TextureUtils.h"
#include "ContextDX.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#ifndef NDEBUG
// #define TEX_VERBOSE_LOGGING
#endif

namespace Ray {
namespace Dx {
extern const D3D12_FILTER g_dx_filter[];
extern const D3D12_TEXTURE_ADDRESS_MODE g_dx_wrap_mode[];
extern const D3D12_COMPARISON_FUNC g_dx_compare_func[];

extern const float AnisotropyLevel;

extern const DXGI_FORMAT g_dx_formats[] = {
    DXGI_FORMAT_UNKNOWN,              // Undefined
    DXGI_FORMAT_UNKNOWN,              // RawRGB888
    DXGI_FORMAT_R8G8B8A8_UNORM,       // RawRGBA8888
    DXGI_FORMAT_R8G8B8A8_SNORM,       // RawRGBA8888Signed
    DXGI_FORMAT_B8G8R8A8_UNORM,       // RawBGRA8888
    DXGI_FORMAT_R32_FLOAT,            // RawR32F
    DXGI_FORMAT_R16_FLOAT,            // RawR16F
    DXGI_FORMAT_R8_UNORM,             // RawR8
    DXGI_FORMAT_R16_UINT,             // RawR16UI
    DXGI_FORMAT_R32_UINT,             // RawR32UI
    DXGI_FORMAT_R8G8_UNORM,           // RawRG88
    DXGI_FORMAT_R32G32B32_FLOAT,      // RawRGB32F
    DXGI_FORMAT_R32G32B32A32_FLOAT,   // RawRGBA32F
    DXGI_FORMAT_UNKNOWN,              // RawRGBE8888
    DXGI_FORMAT_UNKNOWN,              // RawRGB16F
    DXGI_FORMAT_R16G16B16A16_FLOAT,   // RawRGBA16F
    DXGI_FORMAT_R16G16_SNORM,         // RawRG16Snorm
    DXGI_FORMAT_R16G16_UNORM,         // RawRG16
    DXGI_FORMAT_R16G16_FLOAT,         // RawRG16F
    DXGI_FORMAT_R32G32_FLOAT,         // RawRG32F
    DXGI_FORMAT_R32G32_UINT,          // RawRG32U
    DXGI_FORMAT_R10G10B10A2_UNORM,    // RawRGB10_A2
    DXGI_FORMAT_R11G11B10_FLOAT,      // RawRG11F_B10F
    DXGI_FORMAT_D16_UNORM,            // Depth16
    DXGI_FORMAT_D24_UNORM_S8_UINT,    // Depth24Stencil8
    DXGI_FORMAT_D32_FLOAT_S8X24_UINT, // Depth32Stencil8
    DXGI_FORMAT_D32_FLOAT,            // Depth32
    DXGI_FORMAT_BC1_UNORM,            // BC1
    DXGI_FORMAT_BC2_UNORM,            // BC2
    DXGI_FORMAT_BC3_UNORM,            // BC3
    DXGI_FORMAT_BC4_UNORM,            // BC4
    DXGI_FORMAT_BC5_UNORM,            // BC5
    DXGI_FORMAT_UNKNOWN,              // ASTC
    DXGI_FORMAT_UNKNOWN,              // None
};
static_assert(sizeof(g_dx_formats) / sizeof(g_dx_formats[0]) == size_t(eTexFormat::_Count), "!");

uint32_t TextureHandleCounter = 0;

DXGI_FORMAT ToSRGBFormat(const DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
        return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case DXGI_FORMAT_BC1_UNORM:
        return DXGI_FORMAT_BC1_UNORM_SRGB;
    case DXGI_FORMAT_BC2_UNORM:
        return DXGI_FORMAT_BC2_UNORM_SRGB;
    case DXGI_FORMAT_BC3_UNORM:
        return DXGI_FORMAT_BC3_UNORM_SRGB;
    default:
        return format;
    }
    return DXGI_FORMAT_UNKNOWN;
}

D3D12_RESOURCE_FLAGS to_dx_image_flags(const eTexUsage usage, const eTexFormat format) {
    D3D12_RESOURCE_FLAGS ret = D3D12_RESOURCE_FLAG_NONE;
    if (uint8_t(usage & eTexUsage::Storage)) {
        assert(!IsCompressedFormat(format));
        ret |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    if (uint8_t(usage & eTexUsage::RenderTarget)) {
        assert(!IsCompressedFormat(format));
        if (IsDepthFormat(format)) {
            ret |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        } else {
            ret |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
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

uint32_t D3D12CalcSubresource(uint32_t MipSlice, uint32_t ArraySlice, uint32_t PlaneSlice, uint32_t MipLevels,
                              uint32_t ArraySize) {
    return MipSlice + ArraySlice * MipLevels + PlaneSlice * MipLevels * ArraySize;
}
} // namespace Dx

extern const int g_block_res[][2];

int round_up(int v, int align);

bool EndsWith(const std::string &str1, const char *str2);
} // namespace Ray

Ray::eTexUsage Ray::Dx::TexUsageFromState(const eResState state) { return g_tex_usage_per_state[int(state)]; }

Ray::Dx::Texture2D::Texture2D(const char *name, Context *ctx, const Tex2DParams &p, MemoryAllocators *mem_allocs,
                              ILog *log)
    : ctx_(ctx), name_(name) {
    Init(p, mem_allocs, log);
}

Ray::Dx::Texture2D::Texture2D(const char *name, Context *ctx, const void *data, const uint32_t size,
                              const Tex2DParams &p, Buffer &stage_buf, ID3D12GraphicsCommandList *cmd_buf, MemoryAllocators *mem_allocs,
                              eTexLoadStatus *load_status, ILog *log)
    : ctx_(ctx), name_(name) {
    Init(data, size, p, stage_buf, cmd_buf, mem_allocs, load_status, log);
}

Ray::Dx::Texture2D::Texture2D(const char *name, Context *ctx, const void *data[6], const int size[6],
                              const Tex2DParams &p, Buffer &stage_buf, ID3D12GraphicsCommandList *cmd_buf, MemoryAllocators *mem_allocs,
                              eTexLoadStatus *load_status, ILog *log)
    : ctx_(ctx), name_(name) {
    Init(data, size, p, stage_buf, cmd_buf, mem_allocs, load_status, log);
}

Ray::Dx::Texture2D::~Texture2D() { Free(); }

Ray::Dx::Texture2D &Ray::Dx::Texture2D::operator=(Texture2D &&rhs) noexcept {
    if (this == &rhs) {
        return (*this);
    }

    Free();

    ctx_ = std::exchange(rhs.ctx_, nullptr);
    handle_ = std::exchange(rhs.handle_, {});
    alloc_ = std::exchange(rhs.alloc_, {});
    params = std::exchange(rhs.params, {});
    ready_ = std::exchange(rhs.ready_, false);
    cubemap_ready_ = std::exchange(rhs.cubemap_ready_, 0);
    name_ = std::move(rhs.name_);

    resource_state = std::exchange(rhs.resource_state, eResState::Undefined);

    return (*this);
}

void Ray::Dx::Texture2D::Init(const Tex2DParams &p, MemoryAllocators *mem_allocs, ILog *log) {
    InitFromRAWData(nullptr, 0, nullptr, mem_allocs, p, log);
    ready_ = true;
}

void Ray::Dx::Texture2D::Init(const void *data, const uint32_t size, const Tex2DParams &p, Buffer &sbuf, ID3D12GraphicsCommandList *cmd_buf,
                              MemoryAllocators *mem_allocs, eTexLoadStatus *load_status, ILog *log) {
    if (!data) {
        uint8_t *stage_data = sbuf.Map();
        memcpy(stage_data, p.fallback_color, 4);
        sbuf.FlushMappedRange(0, sbuf.AlignMapOffset(4));
        sbuf.Unmap();

        Tex2DParams _p = p;
        _p.w = _p.h = 1;
        _p.mip_count = 1;
        _p.format = eTexFormat::RawRGBA8888;
        _p.usage = eTexUsage::Sampled | eTexUsage::Transfer;

        InitFromRAWData(&sbuf, 0, cmd_buf, mem_allocs, _p, log);
        // mark it as not ready
        ready_ = false;
        (*load_status) = eTexLoadStatus::CreatedDefault;
    } else {
        uint8_t *stage_data = sbuf.Map();
        memcpy(stage_data, data, size);
        sbuf.FlushMappedRange(0, sbuf.AlignMapOffset(size));
        sbuf.Unmap();

        InitFromRAWData(&sbuf, 0, cmd_buf, mem_allocs, p, log);

        ready_ = true;
        (*load_status) = eTexLoadStatus::CreatedFromData;
    }
}

void Ray::Dx::Texture2D::Init(const void *data[6], const int size[6], const Tex2DParams &p, Buffer &sbuf,
                              ID3D12GraphicsCommandList *cmd_buf, MemoryAllocators *mem_allocs, eTexLoadStatus *load_status, ILog *log) {
    if (!data) {
        uint8_t *stage_data = sbuf.Map();
        memcpy(stage_data, p.fallback_color, 4);
        sbuf.FlushMappedRange(0, sbuf.AlignMapOffset(4));
        sbuf.Unmap();

        int data_off[6] = {};

        Tex2DParams _p = p;
        _p.w = _p.h = 1;
        _p.format = eTexFormat::RawRGBA8888;
        _p.usage = eTexUsage::Sampled | eTexUsage::Transfer;

        InitFromRAWData(sbuf, data_off, cmd_buf, mem_allocs, _p, log);
        // mark it as not ready
        ready_ = false;
        cubemap_ready_ = 0;
        (*load_status) = eTexLoadStatus::CreatedDefault;
    } else {
        uint8_t *stage_data = sbuf.Map();
        uint32_t stage_off = 0;

        int data_off[6];
        for (int i = 0; i < 6; i++) {
            if (data[i]) {
                memcpy(&stage_data[stage_off], data[i], size[i]);
                data_off[i] = int(stage_off);
                stage_off += size[i];
            } else {
                data_off[i] = -1;
            }
        }
        sbuf.FlushMappedRange(0, sbuf.AlignMapOffset(4));
        sbuf.Unmap();

        InitFromRAWData(sbuf, data_off, cmd_buf, mem_allocs, p, log);

        ready_ = (cubemap_ready_ & (1u << 0u)) == 1;
        for (unsigned i = 1; i < 6; i++) {
            ready_ = ready_ && ((cubemap_ready_ & (1u << i)) == 1);
        }
        (*load_status) = eTexLoadStatus::CreatedFromData;
    }
}

void Ray::Dx::Texture2D::Free() {
    if (params.format != eTexFormat::Undefined && !bool(params.flags & eTexFlagBits::NoOwnership)) {
        ctx_->staging_descr_alloc()->Free(eDescrType::CBV_SRV_UAV, handle_.views_ref);
        ctx_->resources_to_destroy[ctx_->backend_frame].push_back(handle_.img);
        ctx_->staging_descr_alloc()->Free(eDescrType::Sampler, handle_.sampler_ref);
        ctx_->allocs_to_free[ctx_->backend_frame].emplace_back(std::move(alloc_));

        handle_ = {};
        params.format = eTexFormat::Undefined;
    }
}

bool Ray::Dx::Texture2D::Realloc(const int w, const int h, int mip_count, const int samples, const eTexFormat format,
                                 const eTexBlock block, const bool is_srgb, ID3D12GraphicsCommandList *cmd_buf,
                                 MemoryAllocators *mem_allocs, ILog *log) {
    ID3D12Resource *new_image = nullptr;
    // VkImageView new_image_view = VK_NULL_HANDLE;
    MemAllocation new_alloc = {};
    eResState new_resource_state = eResState::Undefined;

    mip_count = std::min(mip_count, CalcMipCount(w, h, 1, eTexFilter::Trilinear));

    { // create new image
        D3D12_RESOURCE_DESC image_desc = {};
        image_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        image_desc.Width = w;
        image_desc.Height = h;
        image_desc.DepthOrArraySize = 1;
        image_desc.MipLevels = mip_count;
        image_desc.Format = g_dx_formats[int(format)];
        if (is_srgb) {
            image_desc.Format = ToSRGBFormat(image_desc.Format);
        }
        image_desc.SampleDesc.Count = samples;
        image_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        image_desc.Flags = to_dx_image_flags(params.usage, format);

        (void)new_image;
#if 0
        VkResult res = vkCreateImage(ctx_->device(), &img_info, nullptr, &new_image);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image!");
            return false;
        }

#ifdef ENABLE_OBJ_LABELS
        VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectType = VK_OBJECT_TYPE_IMAGE;
        name_info.objectHandle = uint64_t(new_image);
        name_info.pObjectName = name_.c_str();
        vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif

        VkMemoryRequirements tex_mem_req;
        vkGetImageMemoryRequirements(ctx_->device(), new_image, &tex_mem_req);

        VkMemoryPropertyFlags img_tex_desired_mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        new_alloc = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                         FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                        img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                         name_.c_str());
        if (!alloc_) {
            ctx_->log()->Warning("Not enough device memory, falling back to CPU RAM!");
            img_tex_desired_mem_flags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                          FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                         img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                          name_.c_str());
            if (!alloc_) {
                img_tex_desired_mem_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

                alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                              FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                             img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                              name_.c_str());
            }
        }

        res = vkBindImageMemory(ctx_->device(), new_image, new_alloc.owner->mem(new_alloc.block_ndx), aligned_offset);
        if (res != VK_SUCCESS) {
            log->Error("Failed to bind memory!");
            return false;
        }
    }

    { // create new image view
        VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = new_image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = g_vk_formats[size_t(format)];
        if (is_srgb) {
            view_info.format = ToSRGBFormat(view_info.format);
        }
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = mip_count;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        if (GetColorChannelCount(format) == 1) {
            view_info.components.r = VK_COMPONENT_SWIZZLE_R;
            view_info.components.g = VK_COMPONENT_SWIZZLE_R;
            view_info.components.b = VK_COMPONENT_SWIZZLE_R;
            view_info.components.a = VK_COMPONENT_SWIZZLE_R;
        }

        const VkResult res = vkCreateImageView(ctx_->device(), &view_info, nullptr, &new_image_view);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image view!");
            return false;
        }

#ifdef ENABLE_OBJ_LABELS
        VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
        name_info.objectHandle = uint64_t(new_image_view);
        name_info.pObjectName = name_.c_str();
        vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif
#endif
    }

#if 0
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
    uint16_t new_initialized_mips = 0;

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
            if (initialized_mips_ & (1u << src_mip)) {
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

                new_initialized_mips |= (1u << dst_mip);
            }
        }

        if (copy_regions_count) {
            auto cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

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

            if (!barriers.empty()) {
                vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stages,
                                     0, 0, nullptr, 0, nullptr, uint32_t(barriers.size()), barriers.cdata());
            }

            this->resource_state = eResState::CopySrc;
            new_resource_state = eResState::CopyDst;

            vkCmdCopyImage(cmd_buf, handle_.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, new_image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copy_regions_count, copy_regions);
        }
    }
    Free();

    handle_ = new_handle;
    alloc_ = std::move(new_alloc);
    params.w = w;
    params.h = h;
    if (is_srgb) {
        params.flags |= eTexFlagBits::SRGB;
    } else {
        params.flags &= ~eTexFlagBits::SRGB;
    }
    params.mip_count = mip_count;
    params.samples = samples;
    params.format = format;
    params.block = block;
    initialized_mips_ = new_initialized_mips;
#endif
    this->resource_state = new_resource_state;

    return true;
}

void Ray::Dx::Texture2D::InitFromRAWData(Buffer *sbuf, int data_off, ID3D12GraphicsCommandList *cmd_buf, MemoryAllocators *mem_allocs,
                                         const Tex2DParams &p, ILog *log) {
    Free();

    handle_.generation = TextureHandleCounter++;
    params = p;
    initialized_mips_ = 0;

    int mip_count = params.mip_count;
    if (!mip_count) {
        mip_count = CalcMipCount(p.w, p.h, 4, p.sampling.filter);
    }

    { // create image
        D3D12_RESOURCE_DESC image_desc = {};
        image_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        image_desc.Width = p.w;
        image_desc.Height = p.h;
        image_desc.DepthOrArraySize = 1;
        image_desc.MipLevels = mip_count;
        image_desc.Format = g_dx_formats[int(p.format)];
        if (bool(p.flags & eTexFlagBits::SRGB)) {
            image_desc.Format = ToSRGBFormat(image_desc.Format);
        }
        image_desc.SampleDesc.Count = p.samples;
        image_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        image_desc.Flags = to_dx_image_flags(params.usage, p.format);

        const D3D12_RESOURCE_ALLOCATION_INFO alloc_info = ctx_->device()->GetResourceAllocationInfo(0, 1, &image_desc);

        alloc_ = mem_allocs->Allocate(uint32_t(alloc_info.Alignment), uint32_t(alloc_info.SizeInBytes),
                                      D3D12_HEAP_TYPE_DEFAULT);
        if (!alloc_) {
            log->Error("Failed to allocate memory!");
            return;
        }

        HRESULT hr =
            ctx_->device()->CreatePlacedResource(alloc_.owner->heap(alloc_.pool), UINT64(alloc_.offset), &image_desc,
                                                 D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&handle_.img));
        if (FAILED(hr)) {
            log->Error("Failed to create image!");
            return;
        }

#ifdef ENABLE_OBJ_LABELS
        std::wstring temp_str(name_.begin(), name_.end());
        handle_.img->SetName(temp_str.c_str());
#endif
    }

    const UINT CBV_SRV_UAV_INCR =
        ctx_->device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT SAMPLER_INCR = ctx_->device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

    const bool requires_uav = int(p.usage & eTexUsageBits::Storage) != 0;

    handle_.views_ref = ctx_->staging_descr_alloc()->Alloc(eDescrType::CBV_SRV_UAV, requires_uav ? 2 : 1);

    { // create default SRV
        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
        srv_desc.Format = g_dx_formats[int(p.format)];
        if (bool(p.flags & eTexFlagBits::SRGB)) {
            srv_desc.Format = ToSRGBFormat(srv_desc.Format);
        }
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Texture2D.MipLevels = p.mip_count;
        srv_desc.Texture2D.MostDetailedMip = 0;
        srv_desc.Texture2D.PlaneSlice = 0;
        srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;

        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (GetColorChannelCount(p.format) == 1 && int(p.usage & eTexUsageBits::Storage) == 0) {
            srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(0, 0, 0, 0);
        }

        D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = handle_.views_ref.heap->GetCPUDescriptorHandleForHeapStart();
        dest_handle.ptr += CBV_SRV_UAV_INCR * handle_.views_ref.offset;
        ctx_->device()->CreateShaderResourceView(handle_.img, &srv_desc, dest_handle);
    }
    if (requires_uav) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
        uav_desc.Format = g_dx_formats[int(p.format)];
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uav_desc.Texture2D.PlaneSlice = 0;
        uav_desc.Texture2D.MipSlice = 0;

        D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = handle_.views_ref.heap->GetCPUDescriptorHandleForHeapStart();
        dest_handle.ptr += CBV_SRV_UAV_INCR * (handle_.views_ref.offset + 1);
        ctx_->device()->CreateUnorderedAccessView(handle_.img, nullptr, &uav_desc, dest_handle);
    }

#if 0
    { // create default image view(s)
        ...

        if (IsDepthStencilFormat(p.format)) {
            // create additional depth-only image view
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            VkImageView depth_only_view;
            const VkResult res = vkCreateImageView(ctx_->device(), &view_info, nullptr, &depth_only_view);
            if (res != VK_SUCCESS) {
                log->Error("Failed to create image view!");
                return;
            }
            handle_.views.push_back(depth_only_view);
        }

#ifdef ENABLE_OBJ_LABELS
        for (VkImageView view : handle_.views) {
            VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
            name_info.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
            name_info.objectHandle = uint64_t(view);
            name_info.pObjectName = name_.c_str();
            vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
        }
#endif
    }

    this->resource_state = eResState::Undefined;

    if (sbuf) {
        assert(p.samples == 1);
        assert(sbuf && sbuf->type() == eBufType::Stage);
        auto cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

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
            new_barrier.subresourceRange.levelCount = mip_count; // transit whole image
            new_barrier.subresourceRange.baseArrayLayer = 0;
            new_barrier.subresourceRange.layerCount = 1;

            src_stages |= VKPipelineStagesForState(this->resource_state);
            dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
        }

        if (!buf_barriers.empty() || !img_barriers.empty()) {
            vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stages, 0, 0,
                                 nullptr, uint32_t(buf_barriers.size()), buf_barriers.cdata(),
                                 uint32_t(img_barriers.size()), img_barriers.cdata());
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

        vkCmdCopyBufferToImage(cmd_buf, sbuf->vk_handle(), handle_.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &region);

        initialized_mips_ |= (1u << 0);
    }
#endif

    { // create new sampler
        D3D12_SAMPLER_DESC sampler_desc = {};
        sampler_desc.Filter = g_dx_filter[size_t(p.sampling.filter)];
        sampler_desc.AddressU = g_dx_wrap_mode[size_t(p.sampling.wrap)];
        sampler_desc.AddressV = g_dx_wrap_mode[size_t(p.sampling.wrap)];
        sampler_desc.AddressW = g_dx_wrap_mode[size_t(p.sampling.wrap)];
        sampler_desc.MipLODBias = p.sampling.lod_bias.to_float();
        sampler_desc.MinLOD = p.sampling.min_lod.to_float();
        sampler_desc.MaxLOD = p.sampling.max_lod.to_float();
        sampler_desc.MaxAnisotropy = UINT(AnisotropyLevel);
        if (p.sampling.compare != eTexCompare::None) {
            sampler_desc.ComparisonFunc = g_dx_compare_func[size_t(p.sampling.compare)];
        }

        handle_.sampler_ref = ctx_->staging_descr_alloc()->Alloc(eDescrType::Sampler, 1);

        D3D12_CPU_DESCRIPTOR_HANDLE sampler_dest_handle =
            handle_.sampler_ref.heap->GetCPUDescriptorHandleForHeapStart();
        sampler_dest_handle.ptr += SAMPLER_INCR * handle_.sampler_ref.offset;
        ctx_->device()->CreateSampler(&sampler_desc, sampler_dest_handle);
    }
}

void Ray::Dx::Texture2D::InitFromRAWData(Buffer &sbuf, int data_off[6], ID3D12GraphicsCommandList *cmd_buf, MemoryAllocators *mem_allocs,
                                         const Tex2DParams &p, ILog *log) {
    assert(p.w > 0 && p.h > 0);
    Free();

    handle_.generation = TextureHandleCounter++;
    params = p;
    initialized_mips_ = 0;

    const int mip_count = CalcMipCount(p.w, p.h, 1, p.sampling.filter);
    (void)mip_count;
#if 0
    { // create image
        VkImageCreateInfo img_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        img_info.imageType = VK_IMAGE_TYPE_2D;
        img_info.extent.width = uint32_t(p.w);
        img_info.extent.height = uint32_t(p.h);
        img_info.extent.depth = 1;
        img_info.mipLevels = mip_count;
        img_info.arrayLayers = 1;
        img_info.format = g_vk_formats[size_t(p.format)];
        if (bool(p.flags & eTexFlagBits::SRGB)) {
            img_info.format = ToSRGBFormat(img_info.format);
        }
        img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        assert(uint8_t(p.usage) != 0);
        img_info.usage = to_vk_image_usage(p.usage, p.format);
        img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        img_info.samples = VkSampleCountFlagBits(p.samples);
        img_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

        VkResult res = vkCreateImage(ctx_->device(), &img_info, nullptr, &handle_.img);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image!");
            return;
        }

#ifdef ENABLE_OBJ_LABELS
        VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectType = VK_OBJECT_TYPE_IMAGE;
        name_info.objectHandle = uint64_t(handle_.img);
        name_info.pObjectName = name_.c_str();
        vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif

        VkMemoryRequirements tex_mem_req;
        vkGetImageMemoryRequirements(ctx_->device(), handle_.img, &tex_mem_req);

        VkMemoryPropertyFlags img_tex_desired_mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                      FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                     img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                      name_.c_str());
        if (!alloc_) {
            ctx_->log()->Warning("Not enough device memory, falling back to CPU RAM!");
            img_tex_desired_mem_flags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                          FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                         img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                          name_.c_str());
            if (!alloc_) {
                img_tex_desired_mem_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

                alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                              FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                             img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                              name_.c_str());
            }
        }

        res = vkBindImageMemory(ctx_->device(), handle_.img, alloc_.owner->mem(alloc_.block_ndx), aligned_offset);
        if (res != VK_SUCCESS) {
            log->Error("Failed to bind memory!");
            return;
        }
    }

    { // create default image view
        VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = handle_.img;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        view_info.format = g_vk_formats[size_t(p.format)];
        if (bool(p.flags & eTexFlagBits::SRGB)) {
            view_info.format = ToSRGBFormat(view_info.format);
        }
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = mip_count;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        if (GetColorChannelCount(p.format) == 1) {
            view_info.components.r = VK_COMPONENT_SWIZZLE_R;
            view_info.components.g = VK_COMPONENT_SWIZZLE_R;
            view_info.components.b = VK_COMPONENT_SWIZZLE_R;
            view_info.components.a = VK_COMPONENT_SWIZZLE_R;
        }

        const VkResult res = vkCreateImageView(ctx_->device(), &view_info, nullptr, &handle_.views[0]);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image view!");
            return;
        }

#ifdef ENABLE_OBJ_LABELS
        VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
        name_info.objectHandle = uint64_t(handle_.views[0]);
        name_info.pObjectName = name_.c_str();
        vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif
    }

    assert(p.samples == 1);
    assert(sbuf.type() == eBufType::Stage);
    auto cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

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
        new_barrier.subresourceRange.levelCount = mip_count; // transit whole image
        new_barrier.subresourceRange.baseArrayLayer = 0;
        new_barrier.subresourceRange.layerCount = 1;

        src_stages |= VKPipelineStagesForState(this->resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
    }

    if (!buf_barriers.empty() || !img_barriers.empty()) {
        vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stages, 0, 0,
                             nullptr, uint32_t(buf_barriers.size()), buf_barriers.cdata(),
                             uint32_t(img_barriers.size()), img_barriers.cdata());
    }

    sbuf.resource_state = eResState::CopySrc;
    this->resource_state = eResState::CopyDst;

    VkBufferImageCopy regions[6] = {};
    for (int i = 0; i < 6; i++) {
        regions[i].bufferOffset = VkDeviceSize(data_off[i]);
        regions[i].bufferRowLength = 0;
        regions[i].bufferImageHeight = 0;

        regions[i].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        regions[i].imageSubresource.mipLevel = 0;
        regions[i].imageSubresource.baseArrayLayer = i;
        regions[i].imageSubresource.layerCount = 1;

        regions[i].imageOffset = {0, 0, 0};
        regions[i].imageExtent = {uint32_t(p.w), uint32_t(p.h), 1};
    }

    vkCmdCopyBufferToImage(cmd_buf, sbuf.vk_handle(), handle_.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6, regions);
#endif
    initialized_mips_ |= (1u << 0);

    ApplySampling(p.sampling, log);
}

void Ray::Dx::Texture2D::SetSubImage(const int level, const int offsetx, const int offsety, const int sizex,
                                     const int sizey, const eTexFormat format, const Buffer &sbuf, ID3D12GraphicsCommandList *cmd_buf,
                                     const int data_off, const int data_len) {
    assert(format == params.format);
    assert(params.samples == 1);
    assert(offsetx >= 0 && offsetx + sizex <= std::max(params.w >> level, 1));
    assert(offsety >= 0 && offsety + sizey <= std::max(params.h >> level, 1));
    assert(sbuf.type() == eBufType::Upload);

    SmallVector<D3D12_RESOURCE_BARRIER, 2> barriers;

    if (/*sbuf.resource_state != eResState::Undefined &&*/ sbuf.resource_state != eResState::CopySrc) {
        auto &new_barrier = barriers.emplace_back();
        new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        new_barrier.Transition.pResource = sbuf.dx_resource();
        new_barrier.Transition.StateBefore = DXResourceState(sbuf.resource_state);
        new_barrier.Transition.StateAfter = DXResourceState(eResState::CopySrc);
        new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    }

    if (this->resource_state != eResState::CopyDst) {
        auto &new_barrier = barriers.emplace_back();
        new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        new_barrier.Transition.pResource = this->dx_resource();
        new_barrier.Transition.StateBefore = DXResourceState(this->resource_state);
        new_barrier.Transition.StateAfter = DXResourceState(eResState::CopyDst);
        new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    }

    if (!barriers.empty()) {
        cmd_buf->ResourceBarrier(UINT(barriers.size()), barriers.data());
    }

    sbuf.resource_state = eResState::CopySrc;
    this->resource_state = eResState::CopyDst;

    D3D12_TEXTURE_COPY_LOCATION src_loc = {};
    src_loc.pResource = sbuf.dx_resource();
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src_loc.PlacedFootprint.Offset = data_off;
    src_loc.PlacedFootprint.Footprint.Width = sizex;
    src_loc.PlacedFootprint.Footprint.Height = sizey;
    src_loc.PlacedFootprint.Footprint.Depth = 1;
    src_loc.PlacedFootprint.Footprint.Format = g_dx_formats[int(params.format)];
    if (bool(params.flags & eTexFlagBits::SRGB)) {
        src_loc.PlacedFootprint.Footprint.Format = ToSRGBFormat(src_loc.PlacedFootprint.Footprint.Format);
    }
    if (IsCompressedFormat(params.format)) {
        src_loc.PlacedFootprint.Footprint.RowPitch =
            round_up(GetBlockCount(sizex, 1, params.block) * GetBlockLenBytes(params.format, params.block),
                     TextureDataPitchAlignment);
    } else {
        src_loc.PlacedFootprint.Footprint.RowPitch =
            round_up(sizex * GetPerPixelDataLen(params.format), TextureDataPitchAlignment);
    }

    D3D12_TEXTURE_COPY_LOCATION dst_loc = {};
    dst_loc.pResource = dx_resource();
    dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst_loc.SubresourceIndex = D3D12CalcSubresource(level, 0, 0, params.mip_count, 1);

    cmd_buf->CopyTextureRegion(&dst_loc, offsetx, offsety, 0, &src_loc, nullptr);

    if (offsetx == 0 && offsety == 0 && sizex == std::max(params.w >> level, 1) &&
        sizey == std::max(params.h >> level, 1)) {
        // consider this level initialized
        initialized_mips_ |= (1u << level);
    }
}

void Ray::Dx::Texture2D::SetSampling(const SamplingParams s) {
    if (handle_.sampler_ref) {
        ctx_->staging_descr_alloc()->Free(eDescrType::Sampler, handle_.sampler_ref);
    }

    D3D12_SAMPLER_DESC sampler_desc = {};
    sampler_desc.Filter = g_dx_filter[size_t(s.filter)];
    sampler_desc.AddressU = g_dx_wrap_mode[size_t(s.wrap)];
    sampler_desc.AddressV = g_dx_wrap_mode[size_t(s.wrap)];
    sampler_desc.AddressW = g_dx_wrap_mode[size_t(s.wrap)];
    sampler_desc.MipLODBias = s.lod_bias.to_float();
    sampler_desc.MinLOD = s.min_lod.to_float();
    sampler_desc.MaxLOD = s.max_lod.to_float();
    sampler_desc.MaxAnisotropy = UINT(AnisotropyLevel);
    if (s.compare != eTexCompare::None) {
        sampler_desc.ComparisonFunc = g_dx_compare_func[size_t(s.compare)];
    }

    handle_.sampler_ref = ctx_->staging_descr_alloc()->Alloc(eDescrType::Sampler, 1);

    ID3D12Device *device = ctx_->device();

    D3D12_CPU_DESCRIPTOR_HANDLE sampler_dest_handle = handle_.sampler_ref.heap->GetCPUDescriptorHandleForHeapStart();
    sampler_dest_handle.ptr +=
        device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) * handle_.sampler_ref.offset;
    device->CreateSampler(&sampler_desc, sampler_dest_handle);

    params.sampling = s;
}

void Ray::Dx::CopyImageToImage(ID3D12GraphicsCommandList *cmd_buf, Texture2D &src_tex, const uint32_t src_level, const uint32_t src_x,
                               const uint32_t src_y, Texture2D &dst_tex, const uint32_t dst_level, const uint32_t dst_x,
                               const uint32_t dst_y, const uint32_t width, const uint32_t height) {
    assert(src_tex.resource_state == eResState::CopySrc);
    assert(dst_tex.resource_state == eResState::CopyDst);

    D3D12_TEXTURE_COPY_LOCATION src_loc = {};
    src_loc.pResource = src_tex.dx_resource();
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src_loc.SubresourceIndex = D3D12CalcSubresource(src_level, 0, 0, src_tex.params.mip_count, 1);

    D3D12_TEXTURE_COPY_LOCATION dst_loc = {};
    dst_loc.pResource = dst_tex.dx_resource();
    dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst_loc.SubresourceIndex = D3D12CalcSubresource(dst_level, 0, 0, dst_tex.params.mip_count, 1);

    D3D12_BOX src_region = {};
    src_region.left = src_x;
    src_region.right = src_x + width;
    src_region.top = src_y;
    src_region.bottom = src_y + height;
    src_region.front = 0;
    src_region.back = 1;

    cmd_buf->CopyTextureRegion(&dst_loc, dst_x, dst_y, 0, &src_loc, &src_region);
}

void Ray::Dx::CopyImageToBuffer(const Texture2D &src_tex, const int level, const int x, const int y, const int w,
                                const int h, const Buffer &dst_buf, ID3D12GraphicsCommandList *cmd_buf, const int data_off) {
    SmallVector<D3D12_RESOURCE_BARRIER, 2> barriers;

    if (src_tex.resource_state != eResState::CopySrc) {
        auto &new_barrier = barriers.emplace_back();
        new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        new_barrier.Transition.pResource = src_tex.dx_resource();
        new_barrier.Transition.StateBefore = DXResourceState(src_tex.resource_state);
        new_barrier.Transition.StateAfter = DXResourceState(eResState::CopySrc);
        new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    }

    if (/*dst_buf.resource_state != eResState::Undefined &&*/ dst_buf.resource_state != eResState::CopyDst) {
        auto &new_barrier = barriers.emplace_back();
        new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        new_barrier.Transition.pResource = dst_buf.dx_resource();
        new_barrier.Transition.StateBefore = DXResourceState(dst_buf.resource_state);
        new_barrier.Transition.StateAfter = DXResourceState(eResState::CopyDst);
        new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    }

    if (!barriers.empty()) {
        cmd_buf->ResourceBarrier(UINT(barriers.size()), barriers.data());
    }

    src_tex.resource_state = eResState::CopySrc;
    dst_buf.resource_state = eResState::CopyDst;

    D3D12_TEXTURE_COPY_LOCATION src_loc = {};
    src_loc.pResource = src_tex.dx_resource();
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src_loc.SubresourceIndex = D3D12CalcSubresource(level, 0, 0, src_tex.params.mip_count, 1);

    D3D12_TEXTURE_COPY_LOCATION dst_loc = {};
    dst_loc.pResource = dst_buf.dx_resource();
    dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst_loc.PlacedFootprint.Offset = data_off;
    dst_loc.PlacedFootprint.Footprint.Width = src_tex.params.w;
    dst_loc.PlacedFootprint.Footprint.Height = src_tex.params.h;
    dst_loc.PlacedFootprint.Footprint.Depth = 1;
    dst_loc.PlacedFootprint.Footprint.Format = g_dx_formats[int(src_tex.params.format)];
    if (IsCompressedFormat(src_tex.params.format)) {
        dst_loc.PlacedFootprint.Footprint.RowPitch =
            round_up(GetBlockCount(src_tex.params.w, 1, src_tex.params.block) *
                         GetBlockLenBytes(src_tex.params.format, src_tex.params.block),
                     TextureDataPitchAlignment);
    } else {
        dst_loc.PlacedFootprint.Footprint.RowPitch =
            round_up(src_tex.params.w * GetPerPixelDataLen(src_tex.params.format), TextureDataPitchAlignment);
    }

    D3D12_BOX src_region = {};
    src_region.left = x;
    src_region.right = x + w;
    src_region.top = y;
    src_region.bottom = y + h;
    src_region.front = 0;
    src_region.back = 1;

    cmd_buf->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, &src_region);
}

void Ray::Dx::_ClearColorImage(Texture2D &tex, const void *rgba, ID3D12GraphicsCommandList *cmd_buf) {
    assert(tex.resource_state == eResState::UnorderedAccess);

    Context *ctx = tex.ctx();
    ID3D12Device *device = ctx->device();

    D3D12_DESCRIPTOR_HEAP_DESC temp_cpu_descriptor_heap_desc = {};
    temp_cpu_descriptor_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    temp_cpu_descriptor_heap_desc.NumDescriptors = 1;
    temp_cpu_descriptor_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    ID3D12DescriptorHeap *temp_cpu_descriptor_heap = nullptr;
    HRESULT hr = device->CreateDescriptorHeap(&temp_cpu_descriptor_heap_desc, IID_PPV_ARGS(&temp_cpu_descriptor_heap));
    if (FAILED(hr)) {
        return;
    }

    D3D12_DESCRIPTOR_HEAP_DESC temp_gpu_descriptor_heap_desc = {};
    temp_gpu_descriptor_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    temp_gpu_descriptor_heap_desc.NumDescriptors = 1;
    temp_gpu_descriptor_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    ID3D12DescriptorHeap *temp_gpu_descriptor_heap = nullptr;
    hr = device->CreateDescriptorHeap(&temp_gpu_descriptor_heap_desc, IID_PPV_ARGS(&temp_gpu_descriptor_heap));
    if (FAILED(hr)) {
        return;
    }

    ID3D12DescriptorHeap *pp_descriptor_heaps[] = {temp_gpu_descriptor_heap};
    cmd_buf->SetDescriptorHeaps(1, pp_descriptor_heaps);

    D3D12_CPU_DESCRIPTOR_HANDLE temp_buffer_cpu_readable_UAV_handle =
        temp_cpu_descriptor_heap->GetCPUDescriptorHandleForHeapStart();

    D3D12_UNORDERED_ACCESS_VIEW_DESC buffer_UAV_desc = {};
    buffer_UAV_desc.Format = g_dx_formats[int(tex.params.format)];
    buffer_UAV_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    buffer_UAV_desc.Texture2D.PlaneSlice = 0;
    buffer_UAV_desc.Texture2D.MipSlice = 0;
    device->CreateUnorderedAccessView(tex.dx_resource(), nullptr, &buffer_UAV_desc,
                                      temp_buffer_cpu_readable_UAV_handle);

    D3D12_CPU_DESCRIPTOR_HANDLE temp_buffer_cpu_UAV_handle =
        temp_gpu_descriptor_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE temp_buffer_gpu_UAV_handle =
        temp_gpu_descriptor_heap->GetGPUDescriptorHandleForHeapStart();
    device->CopyDescriptorsSimple(1, temp_buffer_cpu_UAV_handle, temp_buffer_cpu_readable_UAV_handle,
                                  D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    if (IsUintFormat(tex.params.format)) {
        cmd_buf->ClearUnorderedAccessViewUint(temp_buffer_gpu_UAV_handle, temp_buffer_cpu_readable_UAV_handle,
                                              tex.dx_resource(), reinterpret_cast<const UINT *>(rgba), 0, nullptr);
    } else {
        cmd_buf->ClearUnorderedAccessViewFloat(temp_buffer_gpu_UAV_handle, temp_buffer_cpu_readable_UAV_handle,
                                               tex.dx_resource(), reinterpret_cast<const float *>(rgba), 0, nullptr);
    }

    ctx->descriptor_heaps_to_release[ctx->backend_frame].push_back(temp_cpu_descriptor_heap);
    ctx->descriptor_heaps_to_release[ctx->backend_frame].push_back(temp_gpu_descriptor_heap);
}

////////////////////////////////////////////////////////////////////////////////////////
#if 0
Ray::Vk::Texture1D::Texture1D(const char *name, Buffer *buf, const eTexFormat format, const uint32_t offset,
                              const uint32_t size, ILog *log)
    : name_(name) {
    Init(buf, format, offset, size, log);
}

Ray::Vk::Texture1D::~Texture1D() { Free(); }

Ray::Vk::Texture1D &Ray::Vk::Texture1D::operator=(Texture1D &&rhs) noexcept {
    if (this == &rhs) {
        return (*this);
    }

    Free();

    buf_ = std::exchange(rhs.buf_, nullptr);
    params_ = std::exchange(rhs.params_, {});
    name_ = std::move(rhs.name_);
    buf_view_ = std::exchange(rhs.buf_view_, {});

    return (*this);
}

void Ray::Vk::Texture1D::Init(Buffer *buf, const eTexFormat format, const uint32_t offset, const uint32_t size,
                              ILog *log) {
    Free();

    VkBufferViewCreateInfo view_info = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO};
    view_info.buffer = buf->vk_handle();
    view_info.format = g_vk_formats[size_t(format)];
    view_info.offset = VkDeviceSize(offset);
    view_info.range = VkDeviceSize(size);

    const VkResult res = vkCreateBufferView(buf->ctx()->device(), &view_info, nullptr, &buf_view_);
    if (res != VK_SUCCESS) {
        buf_->ctx()->log()->Error("Failed to create buffer view!");
    }

    buf_ = buf;
    params_.offset = offset;
    params_.size = size;
    params_.format = format;
}

void Ray::Vk::Texture1D::Free() {
    if (buf_) {
        buf_->ctx()->buf_views_to_destroy[buf_->ctx()->backend_frame].push_back(buf_view_);
        buf_view_ = {};
        buf_ = {};
    }
}
#endif
////////////////////////////////////////////////////////////////////////////////////////

Ray::Dx::Texture3D::Texture3D(const char *name, Context *ctx, const Tex3DParams &params, MemoryAllocators *mem_allocs,
                              ILog *log)
    : name_(name), ctx_(ctx) {
    Init(params, mem_allocs, log);
}

Ray::Dx::Texture3D::~Texture3D() { Free(); }

Ray::Dx::Texture3D &Ray::Dx::Texture3D::operator=(Texture3D &&rhs) noexcept {
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

void Ray::Dx::Texture3D::Init(const Tex3DParams &p, MemoryAllocators *mem_allocs, ILog *log) {
    Free();

    handle_.generation = TextureHandleCounter++;
    params = p;

    { // create image
        D3D12_RESOURCE_DESC image_desc = {};
        image_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        image_desc.Width = p.w;
        image_desc.Height = p.h;
        image_desc.DepthOrArraySize = p.d;
        image_desc.MipLevels = 1;
        image_desc.Format = g_dx_formats[int(p.format)];
        if (bool(p.flags & eTexFlagBits::SRGB)) {
            image_desc.Format = ToSRGBFormat(image_desc.Format);
        }
        image_desc.SampleDesc.Count = 1;
        image_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        image_desc.Flags = to_dx_image_flags(params.usage, p.format);

        const D3D12_RESOURCE_ALLOCATION_INFO alloc_info = ctx_->device()->GetResourceAllocationInfo(0, 1, &image_desc);

        alloc_ = mem_allocs->Allocate(uint32_t(alloc_info.Alignment), uint32_t(alloc_info.SizeInBytes),
                                      D3D12_HEAP_TYPE_DEFAULT);
        if (!alloc_) {
            log->Error("Failed to allocate memory!");
            return;
        }

        HRESULT hr =
            ctx_->device()->CreatePlacedResource(alloc_.owner->heap(alloc_.pool), UINT64(alloc_.offset), &image_desc,
                                                 D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&handle_.img));
        if (FAILED(hr)) {
            log->Error("Failed to create image!");
            return;
        }

#ifdef ENABLE_OBJ_LABELS
        std::wstring temp_str(name_.begin(), name_.end());
        handle_.img->SetName(temp_str.c_str());
#endif
    }

    this->resource_state = eResState::Undefined;

    const UINT CBV_SRV_UAV_INCR =
        ctx_->device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT SAMPLER_INCR = ctx_->device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

    handle_.views_ref = ctx_->staging_descr_alloc()->Alloc(eDescrType::CBV_SRV_UAV, 1);
    handle_.sampler_ref = ctx_->staging_descr_alloc()->Alloc(eDescrType::Sampler, 1);

    { // create default SRV
        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
        srv_desc.Format = g_dx_formats[int(p.format)];
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        srv_desc.Texture2D.MipLevels = 1;
        srv_desc.Texture2D.MostDetailedMip = 0;
        srv_desc.Texture2D.PlaneSlice = 0;
        srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;

        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (GetColorChannelCount(p.format) == 1) {
            srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(0, 0, 0, 0);
        }

        D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = handle_.views_ref.heap->GetCPUDescriptorHandleForHeapStart();
        dest_handle.ptr += CBV_SRV_UAV_INCR * handle_.views_ref.offset;
        ctx_->device()->CreateShaderResourceView(handle_.img, &srv_desc, dest_handle);
    }

    { // create new sampler
        D3D12_SAMPLER_DESC sampler_desc = {};
        sampler_desc.Filter = g_dx_filter[size_t(p.sampling.filter)];
        sampler_desc.AddressU = g_dx_wrap_mode[size_t(p.sampling.wrap)];
        sampler_desc.AddressV = g_dx_wrap_mode[size_t(p.sampling.wrap)];
        sampler_desc.AddressW = g_dx_wrap_mode[size_t(p.sampling.wrap)];
        sampler_desc.MipLODBias = p.sampling.lod_bias.to_float();
        sampler_desc.MinLOD = p.sampling.min_lod.to_float();
        sampler_desc.MaxLOD = p.sampling.max_lod.to_float();
        sampler_desc.MaxAnisotropy = UINT(AnisotropyLevel);
        if (p.sampling.compare != eTexCompare::None) {
            sampler_desc.ComparisonFunc = g_dx_compare_func[size_t(p.sampling.compare)];
        }

        D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = handle_.sampler_ref.heap->GetCPUDescriptorHandleForHeapStart();
        dest_handle.ptr += SAMPLER_INCR * handle_.sampler_ref.offset;
        ctx_->device()->CreateSampler(&sampler_desc, dest_handle);
    }
}

void Ray::Dx::Texture3D::Free() {
    if (params.format != eTexFormat::Undefined && !bool(params.flags & eTexFlagBits::NoOwnership)) {
        ctx_->staging_descr_alloc()->Free(eDescrType::CBV_SRV_UAV, handle_.views_ref);
        ctx_->resources_to_destroy[ctx_->backend_frame].push_back(handle_.img);
        ctx_->staging_descr_alloc()->Free(eDescrType::Sampler, handle_.sampler_ref);
        ctx_->allocs_to_free[ctx_->backend_frame].emplace_back(std::move(alloc_));

        handle_ = {};
        params.format = eTexFormat::Undefined;
    }
}

void Ray::Dx::Texture3D::SetSubImage(int offsetx, int offsety, int offsetz, int sizex, int sizey, int sizez,
                                     eTexFormat format, const Buffer &sbuf, ID3D12GraphicsCommandList *cmd_buf, int data_off,
                                     int data_len) {
    assert(format == params.format);
    assert(offsetx >= 0 && offsetx + sizex <= params.w);
    assert(offsety >= 0 && offsety + sizey <= params.h);
    assert(offsetz >= 0 && offsetz + sizez <= params.d);
    assert(sbuf.type() == eBufType::Upload);

    SmallVector<D3D12_RESOURCE_BARRIER, 2> barriers;

    if (/*sbuf.resource_state != eResState::Undefined &&*/ sbuf.resource_state != eResState::CopySrc) {
        auto &new_barrier = barriers.emplace_back();
        new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        new_barrier.Transition.pResource = sbuf.dx_resource();
        new_barrier.Transition.StateBefore = DXResourceState(sbuf.resource_state);
        new_barrier.Transition.StateAfter = DXResourceState(eResState::CopySrc);
        new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    }

    if (this->resource_state != eResState::CopyDst) {
        auto &new_barrier = barriers.emplace_back();
        new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        new_barrier.Transition.pResource = this->handle_.img;
        new_barrier.Transition.StateBefore = DXResourceState(this->resource_state);
        new_barrier.Transition.StateAfter = DXResourceState(eResState::CopyDst);
        new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    }

    if (!barriers.empty()) {
        cmd_buf->ResourceBarrier(UINT(barriers.size()), barriers.data());
    }

    sbuf.resource_state = eResState::CopySrc;
    this->resource_state = eResState::CopyDst;

    D3D12_TEXTURE_COPY_LOCATION src_loc = {};
    src_loc.pResource = sbuf.dx_resource();
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src_loc.PlacedFootprint.Offset = data_off;
    src_loc.PlacedFootprint.Footprint.Width = sizex;
    src_loc.PlacedFootprint.Footprint.Height = sizey;
    src_loc.PlacedFootprint.Footprint.Depth = sizez;
    src_loc.PlacedFootprint.Footprint.Format = g_dx_formats[int(params.format)];
    if (IsCompressedFormat(params.format)) {
        assert(false);
    } else {
        src_loc.PlacedFootprint.Footprint.RowPitch =
            round_up(sizex * GetPerPixelDataLen(params.format), TextureDataPitchAlignment);
    }

    D3D12_TEXTURE_COPY_LOCATION dst_loc = {};
    dst_loc.pResource = dx_resource();
    dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst_loc.SubresourceIndex = D3D12CalcSubresource(0, 0, 0, 1, 1);

    cmd_buf->CopyTextureRegion(&dst_loc, offsetx, offsety, offsetz, &src_loc, nullptr);
}

DXGI_FORMAT Ray::Dx::DXFormatFromTexFormat(eTexFormat format) { return g_dx_formats[size_t(format)]; }

bool Ray::Dx::RequiresManualSRGBConversion(const eTexFormat format) {
    const DXGI_FORMAT dxgi_format = g_dx_formats[size_t(format)];
    return dxgi_format == ToSRGBFormat(dxgi_format);
}

bool Ray::Dx::CanBeBlockCompressed(int w, int h, const int mip_count, const eTexBlock block) {
    bool ret = true;
    for (int i = 0; i < mip_count && ret; ++i) {
        // make sure resolution is multiple of block size
        ret &= (w % g_block_res[int(block)][0]) == 0 && (h % g_block_res[int(block)][1]) == 0;
        w /= 2;
        h /= 2;
    }
    return ret;
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
