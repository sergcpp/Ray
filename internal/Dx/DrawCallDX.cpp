#include "DrawCallDX.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12.h>

#include "../../Log.h"
#include "AccStructureDX.h"
#include "BufferDX.h"
#include "ContextDX.h"
#include "DescriptorPoolDX.h"
#include "PipelineDX.h"
#include "TextureAtlasDX.h"
#include "TextureDX.h"

namespace Ray {
namespace Dx {
extern const DXGI_FORMAT g_formats_dx[];
}
} // namespace Ray

void Ray::Dx::PrepareDescriptors(Context *ctx, ID3D12GraphicsCommandList *cmd_buf, Span<const Binding> bindings,
                                 const void *uniform_data, const int uniform_data_len, const Program *prog,
                                 DescrMultiPoolAlloc<BumpAlloc> *descr_alloc, ILog *log) {
    ID3D12DescriptorHeap *descriptor_heaps[int(eDescrType::_Count)] = {};

    DescrSizes descr_sizes;
    for (const auto &b : bindings) {
        if (b.trg == eBindTarget::Tex2DSampled || b.trg == eBindTarget::Tex3DSampled ||
            b.trg == eBindTarget::Tex2DArraySampled) {
            descr_sizes.cbv_srv_uav_count += b.handle.count;
            descr_sizes.sampler_count += b.handle.count;
        } else if (b.trg == eBindTarget::Tex2D || b.trg == eBindTarget::Tex2DArray || b.trg == eBindTarget::Tex3D ||
                   b.trg == eBindTarget::UBuf || b.trg == eBindTarget::TBuf || b.trg == eBindTarget::SBufRO ||
                   b.trg == eBindTarget::SBufRW || b.trg == eBindTarget::Image || b.trg == eBindTarget::AccStruct) {
            descr_sizes.cbv_srv_uav_count += b.handle.count;
        } else if (b.trg == eBindTarget::Sampler) {
            descr_sizes.sampler_count += b.handle.count;
        } else if (b.trg == eBindTarget::DescrTable) {
            if (b.handle.descr_table->gpu_ptr) {
                descriptor_heaps[int(b.handle.descr_table->type)] = b.handle.descr_table->gpu_heap;
            } else if (b.handle.descr_table->type == eDescrType::CBV_SRV_UAV) {
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

    SmallVector<ID3D12DescriptorHeap *, int(eDescrType::_Count)> compacted_descriptor_heaps;
    for (int i = 0; i < int(eDescrType::_Count); ++i) {
        if (pool_refs.refs[i].heap) {
            assert(!descriptor_heaps[i] || pool_refs.refs[i].heap == descriptor_heaps[i]);
            compacted_descriptor_heaps.push_back(pool_refs.refs[i].heap);
        } else if (descriptor_heaps[i]) {
            compacted_descriptor_heaps.push_back(descriptor_heaps[i]);
        }
    }
    cmd_buf->SetDescriptorHeaps(UINT(compacted_descriptor_heaps.size()), compacted_descriptor_heaps.data());

    ID3D12Device *device = ctx->device();
    const UINT CBV_SRV_UAV_INCR = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT SAMPLER_INCR = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

    D3D12_CPU_DESCRIPTOR_HANDLE cbv_srv_uav_cpu_handle =
        pool_refs.cbv_srv_uav.heap->GetCPUDescriptorHandleForHeapStart();
    cbv_srv_uav_cpu_handle.ptr += pool_refs.cbv_srv_uav.offset * CBV_SRV_UAV_INCR;
    D3D12_GPU_DESCRIPTOR_HANDLE cbv_srv_uav_gpu_handle =
        pool_refs.cbv_srv_uav.heap->GetGPUDescriptorHandleForHeapStart();
    cbv_srv_uav_gpu_handle.ptr += pool_refs.cbv_srv_uav.offset * CBV_SRV_UAV_INCR;

    D3D12_CPU_DESCRIPTOR_HANDLE sampler_cpu_handle = pool_refs.sampler.heap
                                                         ? pool_refs.sampler.heap->GetCPUDescriptorHandleForHeapStart()
                                                         : D3D12_CPU_DESCRIPTOR_HANDLE{};
    sampler_cpu_handle.ptr += pool_refs.sampler.offset * SAMPLER_INCR;
    D3D12_GPU_DESCRIPTOR_HANDLE sampler_gpu_handle = pool_refs.sampler.heap
                                                         ? pool_refs.sampler.heap->GetGPUDescriptorHandleForHeapStart()
                                                         : D3D12_GPU_DESCRIPTOR_HANDLE{};
    sampler_gpu_handle.ptr += pool_refs.sampler.offset * SAMPLER_INCR;

    for (const auto &b : bindings) {
        const short descr_index = prog->descr_index(b.trg == eBindTarget::Sampler ? 1 : 0, b.loc);
        if (descr_index == -1 && b.trg != eBindTarget::DescrTable) {
            continue;
        }

        if (b.trg == eBindTarget::Tex2D || b.trg == eBindTarget::Tex2DSampled || b.trg == eBindTarget::Tex3D ||
            b.trg == eBindTarget::Tex3DSampled) {
            D3D12_CPU_DESCRIPTOR_HANDLE src_handle =
                b.handle.tex->handle().views_ref.heap->GetCPUDescriptorHandleForHeapStart();
            src_handle.ptr += CBV_SRV_UAV_INCR * b.handle.tex->handle().views_ref.offset;

            D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = cbv_srv_uav_cpu_handle;
            dest_handle.ptr += CBV_SRV_UAV_INCR * descr_index;

            device->CopyDescriptorsSimple(1, dest_handle, src_handle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

            if (b.trg == eBindTarget::Tex2DSampled || b.trg == eBindTarget::Tex3DSampled) {
                const short descr_index = prog->descr_index(1, b.loc);

                D3D12_CPU_DESCRIPTOR_HANDLE src_handle =
                    b.handle.tex->handle().sampler_ref.heap->GetCPUDescriptorHandleForHeapStart();
                src_handle.ptr += SAMPLER_INCR * b.handle.tex->handle().sampler_ref.offset;

                D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = sampler_cpu_handle;
                dest_handle.ptr += SAMPLER_INCR * descr_index;

                device->CopyDescriptorsSimple(1, dest_handle, src_handle, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
            }
        } else if (b.trg == eBindTarget::Tex2DArray || b.trg == eBindTarget::Tex2DArraySampled) {
            for (int i = 0; i < b.handle.count; ++i) {
                D3D12_CPU_DESCRIPTOR_HANDLE src_handle =
                    b.handle.tex_arr[i].srv_ref().heap->GetCPUDescriptorHandleForHeapStart();
                src_handle.ptr += CBV_SRV_UAV_INCR * b.handle.tex_arr[i].srv_ref().offset;

                D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = cbv_srv_uav_cpu_handle;
                dest_handle.ptr += CBV_SRV_UAV_INCR * (descr_index + i);

                device->CopyDescriptorsSimple(1, dest_handle, src_handle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

                if (b.trg == eBindTarget::Tex2DArraySampled) {
                    const short descr_index = prog->descr_index(1, b.loc);

                    D3D12_CPU_DESCRIPTOR_HANDLE src_handle =
                        b.handle.tex_arr[i].sampler_ref().heap->GetCPUDescriptorHandleForHeapStart();
                    src_handle.ptr += SAMPLER_INCR * b.handle.tex_arr[i].sampler_ref().offset;

                    D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = sampler_cpu_handle;
                    dest_handle.ptr += SAMPLER_INCR * (descr_index + i);

                    device->CopyDescriptorsSimple(1, dest_handle, src_handle, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
                }
            }
        } else if (b.trg == eBindTarget::UBuf || b.trg == eBindTarget::SBufRO) {
            if (b.offset == 0) {
                D3D12_CPU_DESCRIPTOR_HANDLE src_handle =
                    b.handle.buf->handle().cbv_srv_uav_ref.heap->GetCPUDescriptorHandleForHeapStart();
                src_handle.ptr += CBV_SRV_UAV_INCR * b.handle.buf->handle().cbv_srv_uav_ref.offset;

                D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = cbv_srv_uav_cpu_handle;
                dest_handle.ptr += CBV_SRV_UAV_INCR * descr_index;

                device->CopyDescriptorsSimple(1, dest_handle, src_handle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            } else {
                D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
                srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
                srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
                assert((b.offset % 4) == 0);
                srv_desc.Buffer.FirstElement = b.offset / sizeof(uint32_t);
                srv_desc.Buffer.NumElements = (b.size ? b.size : (b.handle.buf->size() - b.offset)) / sizeof(uint32_t);
                srv_desc.Buffer.StructureByteStride = 0;

                D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = cbv_srv_uav_cpu_handle;
                dest_handle.ptr += CBV_SRV_UAV_INCR * descr_index;

                device->CreateShaderResourceView(b.handle.buf->dx_resource(), &srv_desc, dest_handle);
            }
        } else if (b.trg == eBindTarget::SBufRW) {
            if (b.offset == 0) {
                D3D12_CPU_DESCRIPTOR_HANDLE src_handle =
                    b.handle.buf->handle().cbv_srv_uav_ref.heap->GetCPUDescriptorHandleForHeapStart();
                src_handle.ptr += CBV_SRV_UAV_INCR * (b.handle.buf->handle().cbv_srv_uav_ref.offset + 1);

                D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = cbv_srv_uav_cpu_handle;
                dest_handle.ptr += CBV_SRV_UAV_INCR * descr_index;

                device->CopyDescriptorsSimple(1, dest_handle, src_handle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            } else {
                D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
                uav_desc.Format = DXGI_FORMAT_R32_TYPELESS;
                uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
                assert((b.offset % 4) == 0);
                uav_desc.Buffer.FirstElement = b.offset / sizeof(uint32_t);
                uav_desc.Buffer.NumElements = (b.size ? b.size : (b.handle.buf->size() - b.offset)) / sizeof(uint32_t);
                uav_desc.Buffer.StructureByteStride = 0;

                D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = cbv_srv_uav_cpu_handle;
                dest_handle.ptr += CBV_SRV_UAV_INCR * descr_index;

                device->CreateUnorderedAccessView(b.handle.buf->dx_resource(), nullptr, &uav_desc, dest_handle);
            }
        } else if (b.trg == eBindTarget::Image) {
            D3D12_CPU_DESCRIPTOR_HANDLE src_handle =
                b.handle.tex->handle().views_ref.heap->GetCPUDescriptorHandleForHeapStart();
            src_handle.ptr += CBV_SRV_UAV_INCR * (b.handle.tex->handle().views_ref.offset + 1);

            D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = cbv_srv_uav_cpu_handle;
            dest_handle.ptr += CBV_SRV_UAV_INCR * descr_index;

            device->CopyDescriptorsSimple(1, dest_handle, src_handle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        } else if (b.trg == eBindTarget::Sampler) {
            D3D12_CPU_DESCRIPTOR_HANDLE src_handle = b.handle.sampler->ref().heap->GetCPUDescriptorHandleForHeapStart();
            src_handle.ptr += SAMPLER_INCR * b.handle.sampler->ref().offset;

            D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = sampler_cpu_handle;
            dest_handle.ptr += SAMPLER_INCR * descr_index;

            device->CopyDescriptorsSimple(1, dest_handle, src_handle, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        } else if (b.trg == eBindTarget::DescrTable && b.handle.descr_table->count) {
            if (b.handle.descr_table->gpu_ptr) {
                cmd_buf->SetComputeRootDescriptorTable(b.loc,
                                                       D3D12_GPU_DESCRIPTOR_HANDLE{b.handle.descr_table->gpu_ptr});
            } else if (b.handle.descr_table->type == eDescrType::CBV_SRV_UAV) {
                D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = cbv_srv_uav_cpu_handle;
                dest_handle.ptr += CBV_SRV_UAV_INCR * (descr_sizes.cbv_srv_uav_count - b.handle.descr_table->count);
                device->CopyDescriptorsSimple(b.handle.descr_table->count, dest_handle,
                                              D3D12_CPU_DESCRIPTOR_HANDLE{b.handle.descr_table->cpu_ptr},
                                              D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

                D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle = cbv_srv_uav_gpu_handle;
                gpu_handle.ptr += CBV_SRV_UAV_INCR * (descr_sizes.cbv_srv_uav_count - b.handle.descr_table->count);
                cmd_buf->SetComputeRootDescriptorTable(b.loc, gpu_handle);
            } else if (b.handle.descr_table->type == eDescrType::Sampler) {
                D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = sampler_cpu_handle;
                dest_handle.ptr += SAMPLER_INCR * (descr_sizes.sampler_count - b.handle.descr_table->count);
                device->CopyDescriptorsSimple(b.handle.descr_table->count, dest_handle,
                                              D3D12_CPU_DESCRIPTOR_HANDLE{b.handle.descr_table->cpu_ptr},
                                              D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

                D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle = sampler_gpu_handle;
                gpu_handle.ptr += SAMPLER_INCR * (descr_sizes.sampler_count - b.handle.descr_table->count);
                cmd_buf->SetComputeRootDescriptorTable(b.loc, gpu_handle);
            }
        } else if (b.trg == eBindTarget::AccStruct) {
            D3D12_CPU_DESCRIPTOR_HANDLE src_handle =
                b.handle.acc_struct->view_ref().heap->GetCPUDescriptorHandleForHeapStart();
            src_handle.ptr += CBV_SRV_UAV_INCR * b.handle.acc_struct->view_ref().offset;

            D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = cbv_srv_uav_cpu_handle;
            dest_handle.ptr += CBV_SRV_UAV_INCR * descr_index;

            device->CopyDescriptorsSimple(1, dest_handle, src_handle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }
    }

    if (uniform_data) {
        uint8_t *out_uniform_data =
            ctx->uniform_data_bufs[ctx->backend_frame].mapped_ptr() + ctx->uniform_data_buf_offs[ctx->backend_frame];
        memcpy(out_uniform_data, uniform_data, uniform_data_len);
        assert(ctx->uniform_data_buf_offs[ctx->backend_frame] + uniform_data_len <
               ctx->uniform_data_bufs[ctx->backend_frame].size());

        D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc = {};
        cbv_desc.BufferLocation = ctx->uniform_data_bufs[ctx->backend_frame].dx_resource()->GetGPUVirtualAddress() +
                                  ctx->uniform_data_buf_offs[ctx->backend_frame];
        cbv_desc.SizeInBytes = 256;

        D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = cbv_srv_uav_cpu_handle;
        dest_handle.ptr += CBV_SRV_UAV_INCR * prog->pc_param_index();
        device->CreateConstantBufferView(&cbv_desc, dest_handle);

        ctx->uniform_data_buf_offs[ctx->backend_frame] += 256;
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
                              int uniform_data_len, DescrMultiPoolAlloc<BumpAlloc> *descr_alloc, ILog *log) {
    Context *ctx = descr_alloc->ctx();

    cmd_buf->SetPipelineState(comp_pipeline.handle());
    cmd_buf->SetComputeRootSignature(comp_pipeline.prog()->root_signature());

    PrepareDescriptors(ctx, cmd_buf, bindings, uniform_data, uniform_data_len, comp_pipeline.prog(), descr_alloc, log);

    cmd_buf->Dispatch(grp_count[0], grp_count[1], grp_count[2]);
}

void Ray::Dx::DispatchComputeIndirect(ID3D12GraphicsCommandList *cmd_buf, const Pipeline &comp_pipeline,
                                      const Buffer &indir_buf, const uint32_t indir_buf_offset,
                                      Span<const Binding> bindings, const void *uniform_data, int uniform_data_len,
                                      DescrMultiPoolAlloc<BumpAlloc> *descr_alloc, ILog *log) {
    Context *ctx = descr_alloc->ctx();

    cmd_buf->SetPipelineState(comp_pipeline.handle());
    cmd_buf->SetComputeRootSignature(comp_pipeline.prog()->root_signature());

    PrepareDescriptors(ctx, cmd_buf, bindings, uniform_data, uniform_data_len, comp_pipeline.prog(), descr_alloc, log);

    cmd_buf->ExecuteIndirect(ctx->indirect_dispatch_cmd_signature(), 1, indir_buf.dx_resource(), indir_buf_offset,
                             nullptr, 0);
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
