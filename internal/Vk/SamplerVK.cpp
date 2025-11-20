#include "SamplerVK.h"

#include "../../Log.h"
#include "ContextVK.h"

namespace Ray {
namespace Vk {
#define X(_0, _1, _2, _3) _1,
extern const VkFilter g_min_mag_filter_vk[] = {
#include "../Filter.inl"
};
#undef X

#define X(_0, _1, _2, _3) _2,
extern const VkSamplerMipmapMode g_mipmap_mode_vk[] = {
#include "../Filter.inl"
};
#undef X

#define X(_0, _1, _2) _1,
extern const VkSamplerAddressMode g_wrap_mode_vk[] = {
#include "../Wrap.inl"
};
#undef X

#define X(_0, _1, _2) _1,
extern const VkCompareOp g_compare_ops_vk[] = {
#include "../CompareOp.inl"
};
#undef X

extern const float AnisotropyLevel = 4.0f;
} // namespace Vk
} // namespace Ray

Ray::Vk::Sampler &Ray::Vk::Sampler::operator=(Sampler &&rhs) noexcept {
    if (&rhs == this) {
        return (*this);
    }

    Free();

    ctx_ = std::exchange(rhs.ctx_, nullptr);
    handle_ = std::exchange(rhs.handle_, {});
    params_ = std::exchange(rhs.params_, {});

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
        ctx_->api().vkDestroySampler(ctx_->device(), handle_, nullptr);
        handle_ = {};
    }
}

void Ray::Vk::Sampler::Init(Context *ctx, const SamplingParams params) {
    Free();

    VkSamplerCreateInfo sampler_info = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_info.magFilter = g_min_mag_filter_vk[size_t(params.filter)];
    sampler_info.minFilter = g_min_mag_filter_vk[size_t(params.filter)];
    sampler_info.addressModeU = g_wrap_mode_vk[size_t(params.wrap)];
    sampler_info.addressModeV = g_wrap_mode_vk[size_t(params.wrap)];
    sampler_info.addressModeW = g_wrap_mode_vk[size_t(params.wrap)];
    sampler_info.anisotropyEnable = (params.filter == eFilter::Nearest) ? VK_FALSE : VK_TRUE;
    sampler_info.maxAnisotropy = AnisotropyLevel;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = (params.compare != eCompareOp::None) ? VK_TRUE : VK_FALSE;
    sampler_info.compareOp = g_compare_ops_vk[size_t(params.compare)];
    sampler_info.mipmapMode = g_mipmap_mode_vk[size_t(params.filter)];
    sampler_info.mipLodBias = params.lod_bias.to_float();
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = VK_LOD_CLAMP_NONE;

    const VkResult res = ctx->api().vkCreateSampler(ctx->device(), &sampler_info, nullptr, &handle_);
    if (res != VK_SUCCESS) {
        ctx->log()->Error("Failed to create sampler!");
    }

    ctx_ = ctx;
    params_ = params;
}
