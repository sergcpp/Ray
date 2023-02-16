#include "Sampler.h"

#include "Context.h"
#include "../../Log.h"

namespace Ray {
namespace Vk {
extern const VkFilter g_vk_min_mag_filter[] = {
    VK_FILTER_NEAREST, // NoFilter
    VK_FILTER_LINEAR,  // Bilinear
    VK_FILTER_LINEAR,  // Trilinear
    VK_FILTER_LINEAR,  // BilinearNoMipmap
    VK_FILTER_NEAREST, // NearestMipmap
};
static_assert(COUNT_OF(g_vk_min_mag_filter) == size_t(eTexFilter::_Count), "!");

extern const VkSamplerAddressMode g_vk_wrap_mode[] = {
    VK_SAMPLER_ADDRESS_MODE_REPEAT,          // Repeat
    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,   // ClampToEdge
    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, // ClampToBorder
};
static_assert(COUNT_OF(g_vk_wrap_mode) == size_t(eTexWrap::_Count), "!");

extern const VkSamplerMipmapMode g_vk_mipmap_mode[] = {
    VK_SAMPLER_MIPMAP_MODE_NEAREST, // NoFilter
    VK_SAMPLER_MIPMAP_MODE_NEAREST, // Bilinear
    VK_SAMPLER_MIPMAP_MODE_LINEAR,  // Trilinear
    VK_SAMPLER_MIPMAP_MODE_NEAREST, // BilinearNoMipmap
    VK_SAMPLER_MIPMAP_MODE_NEAREST, // NearestMipmap
};
static_assert(COUNT_OF(g_vk_mipmap_mode) == size_t(eTexFilter::_Count), "!");

extern const VkCompareOp g_vk_compare_ops[] = {
    VK_COMPARE_OP_NEVER,            // None
    VK_COMPARE_OP_LESS_OR_EQUAL,    // LEqual
    VK_COMPARE_OP_GREATER_OR_EQUAL, // GEqual
    VK_COMPARE_OP_LESS,             // Less
    VK_COMPARE_OP_GREATER,          // Greater
    VK_COMPARE_OP_EQUAL,            // Equal
    VK_COMPARE_OP_NOT_EQUAL,        // NotEqual
    VK_COMPARE_OP_ALWAYS,           // Always
    VK_COMPARE_OP_NEVER             // Never
};
static_assert(COUNT_OF(g_vk_compare_ops) == size_t(eTexCompare::_Count), "!");

extern const float AnisotropyLevel = 4.0f;
} // namespace Vk
} // namespace Ray

Ray::Vk::Sampler &Ray::Vk::Sampler::operator=(Sampler &&rhs) noexcept {
    if (&rhs == this) {
        return (*this);
    }

    Free();

    ctx_ = exchange(rhs.ctx_, nullptr);
    handle_ = exchange(rhs.handle_, {});
    params_ = exchange(rhs.params_, {});

    return (*this);
}

void Ray::Vk::Sampler::Free() {
    if (handle_) {
        ctx_->samplers_to_destroy[ctx_->backend_frame].emplace_back(handle_);
        handle_ = {};
    }
}

void Ray::Vk::Sampler::FreeImmediate() {
    if (handle_) {
        vkDestroySampler(ctx_->device(), handle_, nullptr);
        handle_ = {};
    }
}

void Ray::Vk::Sampler::Init(Context *ctx, const SamplingParams params) {
    Free();

    VkSamplerCreateInfo sampler_info = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_info.magFilter = g_vk_min_mag_filter[size_t(params.filter)];
    sampler_info.minFilter = g_vk_min_mag_filter[size_t(params.filter)];
    sampler_info.addressModeU = g_vk_wrap_mode[size_t(params.wrap)];
    sampler_info.addressModeV = g_vk_wrap_mode[size_t(params.wrap)];
    sampler_info.addressModeW = g_vk_wrap_mode[size_t(params.wrap)];
    sampler_info.anisotropyEnable = VK_TRUE;
    sampler_info.maxAnisotropy = AnisotropyLevel;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = (params.compare != eTexCompare::None) ? VK_TRUE : VK_FALSE;
    sampler_info.compareOp = g_vk_compare_ops[size_t(params.compare)];
    sampler_info.mipmapMode = g_vk_mipmap_mode[size_t(params.filter)];
    sampler_info.mipLodBias = params.lod_bias.to_float();
    sampler_info.minLod = params.min_lod.to_float();
    sampler_info.maxLod = params.max_lod.to_float();

    const VkResult res = vkCreateSampler(ctx->device(), &sampler_info, nullptr, &handle_);
    if (res != VK_SUCCESS) {
        ctx->log()->Error("Failed to create sampler!");
    }

    ctx_ = ctx;
    params_ = params;
}
