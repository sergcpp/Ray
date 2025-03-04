#include "SamplerDX.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12.h>

#include "../../Log.h"
#include "ContextDX.h"

namespace Ray {
namespace Dx {
#define X(_0, _1, _2, _3) _3,
extern const D3D12_FILTER g_dx_filter[] = {
#include "../TextureFilter.inl"
};
#undef X

#define X(_0, _1, _2) _2,
extern const D3D12_TEXTURE_ADDRESS_MODE g_dx_wrap_mode[] = {
#include "../TextureWrap.inl"
};
#undef X

#define X(_0, _1, _2) _2,
extern const D3D12_COMPARISON_FUNC g_dx_compare_func[] = {
#include "../TextureCompare.inl"
};
#undef X

extern const float AnisotropyLevel = 4.0f;
} // namespace Dx
} // namespace Ray

Ray::Dx::Sampler &Ray::Dx::Sampler::operator=(Sampler &&rhs) noexcept {
    if (&rhs == this) {
        return (*this);
    }

    Free();

    ctx_ = std::exchange(rhs.ctx_, nullptr);
    ref_ = std::exchange(rhs.ref_, {});
    params_ = std::exchange(rhs.params_, {});

    return (*this);
}

void Ray::Dx::Sampler::Free() {
    if (ref_) {
        ctx_->staging_descr_alloc()->Free(eDescrType::Sampler, ref_);
        ref_ = {};
    }
}

void Ray::Dx::Sampler::FreeImmediate() { Free(); }

void Ray::Dx::Sampler::Init(Context *ctx, const SamplingParams params) {
    Free();

    D3D12_SAMPLER_DESC sampler_desc = {};
    sampler_desc.Filter = g_dx_filter[size_t(params.filter)];
    sampler_desc.AddressU = g_dx_wrap_mode[size_t(params.wrap)];
    sampler_desc.AddressV = g_dx_wrap_mode[size_t(params.wrap)];
    sampler_desc.AddressW = g_dx_wrap_mode[size_t(params.wrap)];
    sampler_desc.MipLODBias = params.lod_bias.to_float();
    sampler_desc.MinLOD = params.min_lod.to_float();
    sampler_desc.MaxLOD = params.max_lod.to_float();
    sampler_desc.MaxAnisotropy = UINT(AnisotropyLevel);
    if (params.compare != eTexCompare::None) {
        sampler_desc.ComparisonFunc = g_dx_compare_func[size_t(params.compare)];
    }

    ref_ = ctx->staging_descr_alloc()->Alloc(eDescrType::Sampler, 1);

    ID3D12Device *device = ctx->device();

    D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = ref_.heap->GetCPUDescriptorHandleForHeapStart();
    dest_handle.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) * ref_.offset;
    device->CreateSampler(&sampler_desc, dest_handle);

    ctx_ = ctx;
    params_ = params;
}
