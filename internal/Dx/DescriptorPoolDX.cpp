#include "DescriptorPoolDX.h"

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
const D3D12_DESCRIPTOR_HEAP_TYPE g_descr_heap_types_dx[] = {
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, // CBV_SRV_UAV
    D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,     // Sampler
    D3D12_DESCRIPTOR_HEAP_TYPE_RTV,         // RTV
    D3D12_DESCRIPTOR_HEAP_TYPE_DSV          // DSV
};
static_assert(COUNT_OF(g_descr_heap_types_dx) == int(eDescrType::_Count), "!");
} // namespace Dx
} // namespace Ray

Ray::Dx::DescrPool &Ray::Dx::DescrPool::operator=(DescrPool &&rhs) noexcept {
    if (this == &rhs) {
        return (*this);
    }

    Destroy();

    ctx_ = exchange(rhs.ctx_, nullptr);
    type_ = rhs.type_;
    heap_ = exchange(rhs.heap_, nullptr);
    descr_count_ = exchange(rhs.descr_count_, 0);
    next_free_ = exchange(rhs.next_free_, 0);

    return (*this);
}

bool Ray::Dx::DescrPool::Init(const uint32_t descr_count, const bool shader_visible) {
    Destroy();

    ID3D12Device *device = ctx_->device();

    D3D12_DESCRIPTOR_HEAP_DESC temp_gpu_descriptor_heap_desc = {};
    temp_gpu_descriptor_heap_desc.Type = g_descr_heap_types_dx[int(type_)];
    temp_gpu_descriptor_heap_desc.NumDescriptors = descr_count;
    temp_gpu_descriptor_heap_desc.Flags =
        shader_visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    HRESULT hr = device->CreateDescriptorHeap(&temp_gpu_descriptor_heap_desc, IID_PPV_ARGS(&heap_));
    if (FAILED(hr)) {
        return false;
    }

    descr_count_ = descr_count;

    return true;
}

void Ray::Dx::DescrPool::Destroy() {
    if (heap_) {
        ctx_->descriptor_heaps_to_destroy[ctx_->backend_frame].emplace_back(heap_);
        heap_ = nullptr;
    }
}

uint32_t Ray::Dx::DescrPool::Alloc(const uint32_t descr_count) {
    if (next_free_ + descr_count >= descr_count_) {
        return {};
    }

    const uint32_t ret = next_free_;

    next_free_ += descr_count;

    return ret;
}

void Ray::Dx::DescrPool::Reset() { next_free_ = 0; }

/////////////////////////////////////////////////////////////////////////////////////////////////

Ray::Dx::PoolRef Ray::Dx::DescrPoolAlloc::Alloc(const uint32_t descr_count) {
    if (next_free_pool_ == -1 || pools_[next_free_pool_].free_count() < descr_count) {
        ++next_free_pool_;

        if (next_free_pool_ == pools_.size()) {
            // allocate twice more sets each time
            const uint32_t count_mul = (1u << pools_.size());

            DescrPool &new_pool = pools_.emplace_back(ctx_, type_);
            if (!new_pool.Init(count_mul * initial_descr_count_)) {
                return {};
            }
        }
    }
    return {pools_[next_free_pool_].heap(), pools_[next_free_pool_].Alloc(descr_count)};
}

void Ray::Dx::DescrPoolAlloc::Reset() {
    for (auto &pool : pools_) {
        pool.Reset();
    }
    next_free_pool_ = !pools_.empty() ? 0 : -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

Ray::Dx::DescrMultiPoolAlloc::DescrMultiPoolAlloc(Context *ctx, const uint32_t initial_descr_count) {
    for (int i = 0; i < int(eDescrType::_Count); ++i) {
        pools_.emplace_back(ctx, eDescrType(i), initial_descr_count);
    }
}

Ray::Dx::PoolRefs Ray::Dx::DescrMultiPoolAlloc::Alloc(const DescrSizes &sizes) {
    PoolRefs ret = {};
    for (int i = 0; i < int(eDescrType::_Count); ++i) {
        if (sizes.counts[i]) {
            ret.refs[i] = pools_[i].Alloc(sizes.counts[i]);
        }
    }
    return ret;
}

void Ray::Dx::DescrMultiPoolAlloc::Reset() {
    for (auto &pool : pools_) {
        pool.Reset();
    }
}
