#include "SamplerMtl.h"

#include <utility>

#include "../../Log.h"
#include "ContextMtl.h"

namespace Ray::Mtl {
// Filter/Wrap/CompareOp.inl columns carry VK/DX enumerators - Metal tables are explicit,
// entry order must match the eFilter/eWrap/eCompareOp enum order in SamplingParams.h
extern const MTL::SamplerMinMagFilter g_min_mag_filter_mtl[] = {
    MTL::SamplerMinMagFilterNearest, MTL::SamplerMinMagFilterLinear, MTL::SamplerMinMagFilterLinear};
// eFilter::Nearest maps to mip NEAREST like VK/DX Filter.inl: NotMipmapped breaks bindless LOD sampling
extern const MTL::SamplerMipFilter g_mipmap_filter_mtl[] = {MTL::SamplerMipFilterNearest, MTL::SamplerMipFilterNearest,
                                                            MTL::SamplerMipFilterLinear};
extern const MTL::SamplerAddressMode g_wrap_mode_mtl[] = {
    MTL::SamplerAddressModeRepeat, MTL::SamplerAddressModeClampToEdge, MTL::SamplerAddressModeClampToBorderColor};
extern const MTL::CompareFunction g_compare_ops_mtl[] = {
    MTL::CompareFunctionNever,    MTL::CompareFunctionLessEqual, MTL::CompareFunctionGreaterEqual,
    MTL::CompareFunctionLess,     MTL::CompareFunctionGreater,   MTL::CompareFunctionEqual,
    MTL::CompareFunctionNotEqual, MTL::CompareFunctionAlways,    MTL::CompareFunctionNever};
} // namespace Ray::Mtl

Ray::Mtl::Sampler &Ray::Mtl::Sampler::operator=(Sampler &&rhs) noexcept {
    if (&rhs == this) {
        return (*this);
    }

    Free();

    ctx_ = std::exchange(rhs.ctx_, nullptr);
    handle_ = std::exchange(rhs.handle_, nullptr);
    params_ = std::exchange(rhs.params_, {});

    return (*this);
}

void Ray::Mtl::Sampler::Free() {
    if (handle_) {
        handle_->release();
        handle_ = nullptr;
    }
}

void Ray::Mtl::Sampler::FreeImmediate() {
    if (handle_) {
        handle_->release(); // Metal keeps the state alive while command buffers reference it
        handle_ = nullptr;
    }
}

void Ray::Mtl::Sampler::Init(Context *ctx, const SamplingParams params) {
    Free();

    MTL::SamplerDescriptor *sampler_desc = MTL::SamplerDescriptor::alloc()->init();
    sampler_desc->setMinFilter(g_min_mag_filter_mtl[size_t(params.filter)]);
    sampler_desc->setMagFilter(g_min_mag_filter_mtl[size_t(params.filter)]);
    sampler_desc->setMipFilter(g_mipmap_filter_mtl[size_t(params.filter)]);
    sampler_desc->setSAddressMode(g_wrap_mode_mtl[size_t(params.wrap)]);
    sampler_desc->setTAddressMode(g_wrap_mode_mtl[size_t(params.wrap)]);
    sampler_desc->setRAddressMode(g_wrap_mode_mtl[size_t(params.wrap)]);
    sampler_desc->setMaxAnisotropy((params.filter == eFilter::Nearest) ? 1 : 4);
    sampler_desc->setNormalizedCoordinates(true);
    sampler_desc->setCompareFunction((params.compare != eCompareOp::None) ? g_compare_ops_mtl[size_t(params.compare)]
                                                                          : MTL::CompareFunctionNever);

    handle_ = ctx->device()->newSamplerState(sampler_desc);
    sampler_desc->release();
    if (handle_ == nullptr) {
        ctx->log()->Error("Failed to create sampler!");
    }

    ctx_ = ctx;
    params_ = params;
}
