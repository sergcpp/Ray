#include "DescriptorPoolDX.h"

#include <tuple>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12.h>

#include "../../Log.h"
#include "ContextDX.h"

namespace Ray::Dx {
const D3D12_DESCRIPTOR_HEAP_TYPE g_descr_heap_types_dx[] = {
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, // CBV_SRV_UAV
    D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,     // Sampler
    D3D12_DESCRIPTOR_HEAP_TYPE_RTV,         // RTV
    D3D12_DESCRIPTOR_HEAP_TYPE_DSV          // DSV
};
static_assert(std::size(g_descr_heap_types_dx) == int(eDescrType::_Count), "!");
} // namespace Ray::Dx

template <class Allocator>
Ray::Dx::DescrPool<Allocator> &Ray::Dx::DescrPool<Allocator>::operator=(DescrPool &&rhs) noexcept {
    if (this == &rhs) {
        return (*this);
    }

    Destroy();

    alloc_ = std::move(rhs.alloc_);
    ctx_ = std::exchange(rhs.ctx_, nullptr);
    type_ = rhs.type_;
    heap_ = std::exchange(rhs.heap_, nullptr);

    return (*this);
}

template <class Allocator>
bool Ray::Dx::DescrPool<Allocator>::Init(const uint32_t descr_count, const bool shader_visible) {
    Destroy();

    ID3D12Device *device = ctx_->device();

    D3D12_DESCRIPTOR_HEAP_DESC temp_gpu_descriptor_heap_desc = {};
    temp_gpu_descriptor_heap_desc.Type = g_descr_heap_types_dx[int(type_)];
    temp_gpu_descriptor_heap_desc.NumDescriptors = descr_count;
    if (type_ == eDescrType::Sampler) {
        temp_gpu_descriptor_heap_desc.NumDescriptors = std::min(temp_gpu_descriptor_heap_desc.NumDescriptors, 2048u);
    }
    temp_gpu_descriptor_heap_desc.Flags =
        shader_visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    HRESULT hr = device->CreateDescriptorHeap(&temp_gpu_descriptor_heap_desc, IID_PPV_ARGS(&heap_));
    if (FAILED(hr)) {
        return false;
    }

    alloc_ = Allocator(descr_count);

    return true;
}

template <class Allocator> void Ray::Dx::DescrPool<Allocator>::Destroy() {
    if (heap_) {
        const D3D12_DESCRIPTOR_HEAP_DESC descr = heap_->GetDesc();
        if (descr.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) {
            ctx_->descriptor_heaps_to_release[ctx_->backend_frame].emplace_back(heap_);
        } else {
            heap_->Release();
        }
        heap_ = nullptr;
    }
}

template <class Allocator> void Ray::Dx::DescrPool<Allocator>::Reset() { alloc_.Reset(); }

template class Ray::Dx::DescrPool<Ray::Dx::BumpAlloc>;
template class Ray::Dx::DescrPool<Ray::Dx::FreelistAllocAdapted>;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class Allocator> Ray::Dx::PoolRef Ray::Dx::DescrPoolAlloc<Allocator>::Alloc(const uint32_t descr_count) {
    PoolRef ref = {};
    for (auto &pool : pools_) {
        std::tie(ref.offset, ref.block) = pool.Alloc(descr_count);
        if (ref.offset != 0xffffffff) {
            ref.heap = pool.heap();
            break;
        }
    }

    if (ref.offset == 0xffffffff) {
        // allocate twice more sets each time
        const uint32_t count_mul = (1u << pools_.size());
        auto &new_pool = pools_.emplace_back(ctx_, type_);
        if (new_pool.Init(count_mul * initial_descr_count_, shader_visible_)) {
            std::tie(ref.offset, ref.block) = new_pool.Alloc(descr_count);
            if (ref.offset != 0xffffffff) {
                ref.heap = new_pool.heap();
            }
        }
    }

    return ref;
}

template <class Allocator> void Ray::Dx::DescrPoolAlloc<Allocator>::Free(const PoolRef &ref) {
    for (auto &pool : pools_) {
        if (pool.heap() == ref.heap) {
            pool.Free(ref.offset, ref.block);
            break;
        }
    }
}

template <class Allocator> void Ray::Dx::DescrPoolAlloc<Allocator>::Reset() {
    for (auto &pool : pools_) {
        pool.Reset();
    }
}

template class Ray::Dx::DescrPoolAlloc<Ray::Dx::BumpAlloc>;
template class Ray::Dx::DescrPoolAlloc<Ray::Dx::FreelistAllocAdapted>;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class Allocator>
Ray::Dx::DescrMultiPoolAlloc<Allocator>::DescrMultiPoolAlloc(Context *ctx, const bool shader_visible,
                                                             const uint32_t initial_descr_count) {
    for (int i = 0; i < int(eDescrType::_Count); ++i) {
        pools_.emplace_back(ctx, eDescrType(i), shader_visible, initial_descr_count);
    }
}

template <class Allocator>
Ray::Dx::PoolRef Ray::Dx::DescrMultiPoolAlloc<Allocator>::Alloc(const eDescrType type, const uint32_t descr_count) {
    return pools_[int(type)].Alloc(descr_count);
}

template <class Allocator> Ray::Dx::PoolRefs Ray::Dx::DescrMultiPoolAlloc<Allocator>::Alloc(const DescrSizes &sizes) {
    PoolRefs ret = {};
    for (int i = 0; i < int(eDescrType::_Count); ++i) {
        if (sizes.counts[i]) {
            ret.refs[i] = Alloc(eDescrType(i), sizes.counts[i]);
        }
    }
    return ret;
}

template <class Allocator>
void Ray::Dx::DescrMultiPoolAlloc<Allocator>::Free(const eDescrType type, const PoolRef &ref) {
    if (ref.heap) {
        pools_[int(type)].Free(ref);
    }
}

template <class Allocator> void Ray::Dx::DescrMultiPoolAlloc<Allocator>::Free(const PoolRefs &refs) {
    for (int i = 0; i < int(eDescrType::_Count); ++i) {
        Free(eDescrType(i), refs.refs[i]);
    }
}

template <class Allocator> void Ray::Dx::DescrMultiPoolAlloc<Allocator>::Reset() {
    for (auto &pool : pools_) {
        pool.Reset();
    }
}

template class Ray::Dx::DescrMultiPoolAlloc<Ray::Dx::BumpAlloc>;
template class Ray::Dx::DescrMultiPoolAlloc<Ray::Dx::FreelistAllocAdapted>;