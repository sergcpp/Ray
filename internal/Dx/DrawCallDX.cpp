#include "DrawCallDX.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12.h>

#include "../../Log.h"
// #include "AccStructureVK.h"
#include "BufferDX.h"
#include "ContextDX.h"
#include "DescriptorPoolDX.h"
#include "PipelineDX.h"
#include "TextureDX.h"
// #include "TextureAtlasVK.h"
// #include "VK.h"

namespace Ray {
namespace Dx {
extern const DXGI_FORMAT g_dx_formats[];
}
} // namespace Ray

void Ray::Dx::PrepareDescriptors(Context *ctx, ID3D12GraphicsCommandList *cmd_buf, Span<const Binding> bindings,
                                 const void *uniform_data, const int uniform_data_len, const Program *prog,
                                 DescrMultiPoolAlloc *descr_alloc, ILog *log) {
    DescrSizes descr_sizes;
    for (const auto &b : bindings) {
        if (b.trg == eBindTarget::Tex2DSampled) {
            ++descr_sizes.cbv_srv_uav_count;
            ++descr_sizes.sampler_count;
        } else if (b.trg == eBindTarget::Tex2D || b.trg == eBindTarget::Tex3D || b.trg == eBindTarget::Tex2DArray ||
                   b.trg == eBindTarget::UBuf ||
                   b.trg == eBindTarget::TBuf || b.trg == eBindTarget::SBufRO || b.trg == eBindTarget::SBufRW ||
                   b.trg == eBindTarget::Image) {
            ++descr_sizes.cbv_srv_uav_count;
        } else if (b.trg == eBindTarget::DescrTable) {
            if (b.handle.descr_table->type == eDescrType::CBV_SRV_UAV) {
                descr_sizes.cbv_srv_uav_count += b.handle.descr_table->count;
            } else if (b.handle.descr_table->type == eDescrType::Sampler) {
                descr_sizes.sampler_count += b.handle.descr_table->count;
            }
        }
    }

    if (uniform_data) {
        ++descr_sizes.cbv_srv_uav_count;
    }

    const PoolRefs pool_refs = descr_alloc->Alloc(descr_sizes);

    SmallVector<ID3D12DescriptorHeap *, int(eDescrType::_Count)> descriptor_heaps;
    for (int i = 0; i < int(eDescrType::_Count); ++i) {
        if (pool_refs.refs[i].heap) {
            descriptor_heaps.push_back(pool_refs.refs[i].heap);
        }
    }
    cmd_buf->SetDescriptorHeaps(UINT(descriptor_heaps.size()), descriptor_heaps.data());

    ID3D12Device *device = ctx->device();
    const UINT cbv_srv_uav_incr = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT sampler_incr = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

    D3D12_CPU_DESCRIPTOR_HANDLE cbv_srv_uav_cpu_handle =
        pool_refs.cbv_srv_uav.heap->GetCPUDescriptorHandleForHeapStart();
    cbv_srv_uav_cpu_handle.ptr += pool_refs.cbv_srv_uav.offset * cbv_srv_uav_incr;
    D3D12_GPU_DESCRIPTOR_HANDLE cbv_srv_uav_gpu_handle =
        pool_refs.cbv_srv_uav.heap->GetGPUDescriptorHandleForHeapStart();
    cbv_srv_uav_gpu_handle.ptr += pool_refs.cbv_srv_uav.offset * cbv_srv_uav_incr;

    D3D12_CPU_DESCRIPTOR_HANDLE sampler_cpu_handle = pool_refs.sampler.heap
                                                         ? pool_refs.sampler.heap->GetCPUDescriptorHandleForHeapStart()
                                                         : D3D12_CPU_DESCRIPTOR_HANDLE{};
    sampler_cpu_handle.ptr += pool_refs.sampler.offset * sampler_incr;
    D3D12_GPU_DESCRIPTOR_HANDLE sampler_gpu_handle = pool_refs.sampler.heap
                                                         ? pool_refs.sampler.heap->GetGPUDescriptorHandleForHeapStart()
                                                         : D3D12_GPU_DESCRIPTOR_HANDLE{};
    sampler_gpu_handle.ptr += pool_refs.sampler.offset * sampler_incr;

    for (const auto &b : bindings) {
        const short param_index = prog->param_index(b.loc);
        if (param_index == -1) {
            continue;
        }

        if (b.trg == eBindTarget::Tex2D || b.trg == eBindTarget::Tex2DSampled) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
            srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv_desc.Format = g_dx_formats[int(b.handle.tex->params.format)];
            srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv_desc.Texture2D.MipLevels = b.handle.tex->params.mip_count;
            srv_desc.Texture2D.MostDetailedMip = 0;
            srv_desc.Texture2D.PlaneSlice = 0;
            srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;

            D3D12_CPU_DESCRIPTOR_HANDLE srv_dest_handle = cbv_srv_uav_cpu_handle;
            srv_dest_handle.ptr += cbv_srv_uav_incr * param_index;
            device->CreateShaderResourceView(b.handle.tex->dx_resource(), &srv_desc, srv_dest_handle);

            if (b.trg == eBindTarget::Tex2DSampled) {
                D3D12_SAMPLER_DESC sampler_desc = {};
                sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
                sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
                sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
                sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
                sampler_desc.MinLOD = 0.0f;
                sampler_desc.MaxLOD = 1000.0f;

                D3D12_CPU_DESCRIPTOR_HANDLE sampler_dest_handle = sampler_cpu_handle;
                sampler_dest_handle.ptr += sampler_incr * param_index;
                device->CreateSampler(&sampler_desc, sampler_dest_handle);
            }
        } else if (b.trg == eBindTarget::Tex3D) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
            srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv_desc.Format = g_dx_formats[int(b.handle.tex3d->params.format)];
            srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            srv_desc.Texture2D.MipLevels = 1;
            srv_desc.Texture2D.MostDetailedMip = 0;
            srv_desc.Texture2D.PlaneSlice = 0;
            srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;

            D3D12_CPU_DESCRIPTOR_HANDLE srv_dest_handle = cbv_srv_uav_cpu_handle;
            srv_dest_handle.ptr += cbv_srv_uav_incr * param_index;
            device->CreateShaderResourceView(b.handle.tex3d->dx_resource(), &srv_desc, srv_dest_handle);
        } else if (b.trg == eBindTarget::SBufRO) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
            srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
            srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            srv_desc.Buffer.FirstElement = 0;
            srv_desc.Buffer.NumElements = b.handle.buf->size() / sizeof(uint32_t);
            srv_desc.Buffer.StructureByteStride = 0;

            D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = cbv_srv_uav_cpu_handle;
            dest_handle.ptr += cbv_srv_uav_incr * param_index;
            device->CreateShaderResourceView(b.handle.buf->dx_resource(), &srv_desc, dest_handle);
        } else if (b.trg == eBindTarget::SBufRW) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
            uav_desc.Format = DXGI_FORMAT_R32_TYPELESS;
            uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            uav_desc.Buffer.FirstElement = 0;
            uav_desc.Buffer.NumElements = b.handle.buf->size() / sizeof(uint32_t);
            uav_desc.Buffer.StructureByteStride = 0;

            D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = cbv_srv_uav_cpu_handle;
            dest_handle.ptr += cbv_srv_uav_incr * param_index;
            device->CreateUnorderedAccessView(b.handle.buf->dx_resource(), nullptr, &uav_desc, dest_handle);
        } else if (b.trg == eBindTarget::Image) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
            uav_desc.Format = g_dx_formats[int(b.handle.tex->params.format)];
            uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            uav_desc.Texture2D.PlaneSlice = 0;
            uav_desc.Texture2D.MipSlice = 0;

            D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = cbv_srv_uav_cpu_handle;
            dest_handle.ptr += cbv_srv_uav_incr * param_index;
            device->CreateUnorderedAccessView(b.handle.tex->dx_resource(), nullptr, &uav_desc, dest_handle);
        } else if (b.trg == eBindTarget::DescrTable && b.handle.descr_table->count) {
            if (b.handle.descr_table->type == eDescrType::CBV_SRV_UAV) {
                D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = cbv_srv_uav_cpu_handle;
                dest_handle.ptr += cbv_srv_uav_incr * (descr_sizes.cbv_srv_uav_count - b.handle.descr_table->count);
                device->CopyDescriptorsSimple(b.handle.descr_table->count, dest_handle,
                                              D3D12_CPU_DESCRIPTOR_HANDLE{b.handle.descr_table->ptr},
                                              D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

                D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle = cbv_srv_uav_gpu_handle;
                gpu_handle.ptr += cbv_srv_uav_incr * (descr_sizes.cbv_srv_uav_count - b.handle.descr_table->count);
                cmd_buf->SetComputeRootDescriptorTable(b.loc, gpu_handle);
            } else if (b.handle.descr_table->type == eDescrType::Sampler) {
                D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = sampler_cpu_handle;
                dest_handle.ptr += sampler_incr * (descr_sizes.sampler_count - b.handle.descr_table->count);
                device->CopyDescriptorsSimple(b.handle.descr_table->count, dest_handle,
                                              D3D12_CPU_DESCRIPTOR_HANDLE{b.handle.descr_table->ptr},
                                              D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

                D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle = sampler_gpu_handle;
                gpu_handle.ptr += sampler_incr * (descr_sizes.sampler_count - b.handle.descr_table->count);
                cmd_buf->SetComputeRootDescriptorTable(b.loc, gpu_handle);
            }
        }
    }

    if (uniform_data) {
        // TODO: use single buffer for all uniform data
        Buffer temp_cb("Temp constant buffer", ctx, eBufType::Upload, 256);
        temp_cb.UpdateImmediate(0, uniform_data_len, uniform_data, cmd_buf);

        D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc = {};
        cbv_desc.BufferLocation = temp_cb.dx_resource()->GetGPUVirtualAddress();
        cbv_desc.SizeInBytes = 256;

        D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = cbv_srv_uav_cpu_handle;
        dest_handle.ptr += cbv_srv_uav_incr * prog->pc_param_index();
        device->CreateConstantBufferView(&cbv_desc, dest_handle);
    }

    if (descr_sizes.cbv_srv_uav_count) {
        cmd_buf->SetComputeRootDescriptorTable(0, cbv_srv_uav_gpu_handle);
    }
    if (descr_sizes.sampler_count) {
        cmd_buf->SetComputeRootDescriptorTable(1, sampler_gpu_handle);
    }
}

void Ray::Dx::DispatchCompute(ID3D12GraphicsCommandList *cmd_buf, const Pipeline &comp_pipeline,
                              const uint32_t grp_count[3], Span<const Binding> bindings, const void *uniform_data,
                              int uniform_data_len, DescrMultiPoolAlloc *descr_alloc, ILog *log) {
    Context *ctx = descr_alloc->ctx();

    cmd_buf->SetPipelineState(comp_pipeline.handle());
    cmd_buf->SetComputeRootSignature(comp_pipeline.prog()->root_signature());

    PrepareDescriptors(ctx, cmd_buf, bindings, uniform_data, uniform_data_len, comp_pipeline.prog(), descr_alloc, log);

    cmd_buf->Dispatch(grp_count[0], grp_count[1], grp_count[2]);
}

void Ray::Dx::DispatchComputeIndirect(ID3D12GraphicsCommandList *cmd_buf, const Pipeline &comp_pipeline,
                                      const Buffer &indir_buf, const uint32_t indir_buf_offset,
                                      Span<const Binding> bindings, const void *uniform_data, int uniform_data_len,
                                      DescrMultiPoolAlloc *descr_alloc, ILog *log) {
    Context *ctx = descr_alloc->ctx();

    cmd_buf->SetPipelineState(comp_pipeline.handle());
    cmd_buf->SetComputeRootSignature(comp_pipeline.prog()->root_signature());

    PrepareDescriptors(ctx, cmd_buf, bindings, uniform_data, uniform_data_len, comp_pipeline.prog(), descr_alloc, log);

    cmd_buf->ExecuteIndirect(comp_pipeline.cmd_signature(), 1, indir_buf.dx_resource(), indir_buf_offset, nullptr, 0);

    /*Context *ctx = descr_alloc->ctx();

    VkDescriptorSet descr_set =
        PrepareDescriptorSet(ctx, comp_pipeline.prog()->descr_set_layouts()[0], bindings, descr_alloc, log);
    if (!descr_set) {
        log->Error("Failed to allocate descriptor set, skipping draw call!");
        return;
    }

    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, comp_pipeline.handle());
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, comp_pipeline.layout(), 0, 1, &descr_set, 0,
                            nullptr);

    if (uniform_data && uniform_data_len) {
        vkCmdPushConstants(cmd_buf, comp_pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, uniform_data_len,
                           uniform_data);
    }

    vkCmdDispatchIndirect(cmd_buf, indir_buf.vk_handle(), VkDeviceSize(indir_buf_offset));*/
}

/*void Ray::Dx::TraceRays(VkCommandBuffer cmd_buf, const Pipeline &rt_pipeline, const uint32_t dims[3],
                        Span<const Binding> bindings, const void *uniform_data, const int uniform_data_len,
                        DescrMultiPoolAlloc *descr_alloc, ILog *log) {
    Context *ctx = descr_alloc->ctx();

    VkDescriptorSet descr_set =
        PrepareDescriptorSet(ctx, rt_pipeline.prog()->descr_set_layouts()[0], bindings, descr_alloc, log);
    if (!descr_set) {
        log->Error("Failed to allocate descriptor set, skipping draw call!");
        return;
    }

    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline.handle());
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline.layout(), 0, 1, &descr_set, 0,
                            nullptr);

    if (uniform_data) {
        // TODO: Properly determine required stages!
        vkCmdPushConstants(cmd_buf, rt_pipeline.layout(), VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, uniform_data_len,
                           uniform_data);
    }

    vkCmdTraceRaysKHR(cmd_buf, rt_pipeline.rgen_table(), rt_pipeline.miss_table(), rt_pipeline.hit_table(),
                      rt_pipeline.call_table(), dims[0], dims[1], dims[2]);
}

void Ray::Dx::TraceRaysIndirect(VkCommandBuffer cmd_buf, const Pipeline &rt_pipeline, const Buffer &indir_buf,
                                const uint32_t indir_buf_offset, Span<const Binding> bindings, const void *uniform_data,
                                int uniform_data_len, DescrMultiPoolAlloc *descr_alloc, ILog *log) {
    Context *ctx = descr_alloc->ctx();

    VkDescriptorSet descr_set =
        PrepareDescriptorSet(ctx, rt_pipeline.prog()->descr_set_layouts()[0], bindings, descr_alloc, log);
    if (!descr_set) {
        log->Error("Failed to allocate descriptor set, skipping draw call!");
        return;
    }

    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline.handle());
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline.layout(), 0, 1, &descr_set, 0,
                            nullptr);

    if (uniform_data && uniform_data_len) {
        // TODO: Properly determine required stages!
        vkCmdPushConstants(cmd_buf, rt_pipeline.layout(), VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, uniform_data_len,
                           uniform_data);
    }

    vkCmdTraceRaysIndirectKHR(cmd_buf, rt_pipeline.rgen_table(), rt_pipeline.miss_table(), rt_pipeline.hit_table(),
                              rt_pipeline.call_table(), indir_buf.vk_device_address() + indir_buf_offset);
}*/
