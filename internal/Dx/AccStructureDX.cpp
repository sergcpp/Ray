#include "AccStructureDX.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12.h>

#include "ContextDX.h"

Ray::Dx::AccStructure::AccStructure(AccStructure &&rhs) noexcept
    : ctx_(exchange(rhs.ctx_, nullptr)), gpu_virtual_address_(exchange(rhs.gpu_virtual_address_, {})),
      view_ref_(exchange(rhs.view_ref_, {})) {}

Ray::Dx::AccStructure &Ray::Dx::AccStructure::operator=(AccStructure &&rhs) noexcept {
    Free();

    ctx_ = exchange(rhs.ctx_, nullptr);
    gpu_virtual_address_ = exchange(rhs.gpu_virtual_address_, 0);
    view_ref_ = exchange(rhs.view_ref_, {});
    resource_state = exchange(rhs.resource_state, eResState::Undefined);

    return (*this);
}

bool Ray::Dx::AccStructure::Init(Context *ctx, uint64_t gpu_virtual_address) {
    Free();

    ctx_ = ctx;
    gpu_virtual_address_ = gpu_virtual_address;
    view_ref_ = ctx_->staging_descr_alloc()->Alloc(eDescrType::CBV_SRV_UAV, 1);

    const UINT CBV_SRV_UAV_INCR =
        ctx_->device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    { // create default SRV
        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv_desc.Format = DXGI_FORMAT_UNKNOWN;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
        srv_desc.RaytracingAccelerationStructure.Location = gpu_virtual_address;

        D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = view_ref_.heap->GetCPUDescriptorHandleForHeapStart();
        dest_handle.ptr += CBV_SRV_UAV_INCR * view_ref_.offset;
        ctx_->device()->CreateShaderResourceView(nullptr, &srv_desc, dest_handle);
    }

    return true;
}

void Ray::Dx::AccStructure::FreeImmediate() {
    if (view_ref_) {
        ctx_->staging_descr_alloc()->Free(eDescrType::CBV_SRV_UAV, view_ref_);
    }
}