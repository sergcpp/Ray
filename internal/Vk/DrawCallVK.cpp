#include "DrawCallVK.h"

#include "../../Log.h"
#include "AccStructureVK.h"
#include "BufferVK.h"
#include "ContextVK.h"
#include "DescriptorPoolVK.h"
#include "PipelineVK.h"
#include "TextureAtlasVK.h"
#include "TextureVK.h"

VkDescriptorSet Ray::Vk::PrepareDescriptorSet(Context *ctx, VkDescriptorSetLayout layout, Span<const Binding> bindings,
                                              DescrMultiPoolAlloc *descr_alloc, ILog *log) {
    VkDescriptorImageInfo sampler_infos[16] = {};
    VkDescriptorImageInfo img_sampler_infos[16] = {};
    VkDescriptorImageInfo img_storage_infos[16] = {};
    VkDescriptorBufferInfo ubuf_infos[16] = {};
    VkDescriptorBufferInfo sbuf_infos[32] = {};
    VkWriteDescriptorSetAccelerationStructureKHR desc_tlas_infos[16] = {};
    DescrSizes descr_sizes;

    SmallVector<VkWriteDescriptorSet, 48> descr_writes;

    for (const auto &b : bindings) {
        if (b.trg == eBindTarget::Tex || b.trg == eBindTarget::TexSampled) {
            auto &info = img_sampler_infos[descr_sizes.img_sampler_count++];
            if (b.trg == eBindTarget::TexSampled) {
                info.sampler = b.handle.tex->handle().sampler;
            }
            if (IsDepthStencilFormat(b.handle.tex->params.format)) {
                // use depth-only image view
                info.imageView = b.handle.tex->handle().views[1];
            } else {
                info.imageView = b.handle.tex->handle().views[0];
            }
            info.imageLayout = VKImageLayoutForState(b.handle.tex->resource_state);

            auto &new_write = descr_writes.emplace_back();
            new_write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            new_write.dstBinding = b.loc;
            new_write.dstArrayElement = b.offset;
            new_write.descriptorType = (b.trg == eBindTarget::TexSampled) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                                          : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            new_write.descriptorCount = 1;
            new_write.pImageInfo = &info;
        } else if (b.trg == eBindTarget::TexArray || b.trg == eBindTarget::TexArraySampled) {
            const uint32_t start_pos = descr_sizes.img_sampler_count;
            for (int i = 0; i < b.handle.count; ++i) {
                auto &info = img_sampler_infos[descr_sizes.img_sampler_count++];
                if (b.trg == eBindTarget::TexArraySampled) {
                    info.sampler = b.handle.tex_arr[i].vk_sampler();
                }
                info.imageView = b.handle.tex_arr[i].vk_imgage_view();
                info.imageLayout = VKImageLayoutForState(b.handle.tex_arr[i].resource_state);
            }

            auto &new_write = descr_writes.emplace_back();
            new_write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            new_write.dstBinding = b.loc;
            new_write.dstArrayElement = b.offset;
            new_write.descriptorType = (b.trg == eBindTarget::TexArraySampled)
                                           ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                           : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            new_write.descriptorCount = uint32_t(b.handle.count);
            new_write.pImageInfo = &img_sampler_infos[start_pos];
        } else if (b.trg == eBindTarget::UBuf) {
            auto &ubuf = ubuf_infos[descr_sizes.ubuf_count++];
            ubuf.buffer = b.handle.buf->vk_handle();
            ubuf.offset = b.offset;
            ubuf.range = b.offset ? b.size : VK_WHOLE_SIZE;

            auto &new_write = descr_writes.emplace_back();
            new_write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            new_write.dstBinding = b.loc;
            new_write.dstArrayElement = 0;
            new_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            new_write.descriptorCount = 1;
            new_write.pBufferInfo = &ubuf;
        } else if (b.trg == eBindTarget::SBufRO || b.trg == eBindTarget::SBufRW) {
            auto &sbuf = sbuf_infos[descr_sizes.sbuf_count++];
            sbuf.buffer = b.handle.buf->vk_handle();
            sbuf.offset = b.offset;
            sbuf.range = (b.offset || b.size) ? (b.size ? b.size : (b.handle.buf->size() - b.offset)) : VK_WHOLE_SIZE;

            auto &new_write = descr_writes.emplace_back();
            new_write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            new_write.dstBinding = b.loc;
            new_write.dstArrayElement = 0;
            new_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            new_write.descriptorCount = 1;
            new_write.pBufferInfo = &sbuf;
        } else if (b.trg == eBindTarget::Image) {
            auto &info = img_storage_infos[descr_sizes.store_img_count++];
            info.sampler = b.handle.tex->handle().sampler;
            if (IsDepthStencilFormat(b.handle.tex->params.format)) {
                // use depth-only image view
                info.imageView = b.handle.tex->handle().views[1];
            } else {
                info.imageView = b.handle.tex->handle().views[0];
            }
            info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            auto &new_write = descr_writes.emplace_back();
            new_write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            new_write.dstBinding = b.loc;
            new_write.dstArrayElement = 0;
            new_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            new_write.descriptorCount = 1;
            new_write.pImageInfo = &info;
        } else if (b.trg == eBindTarget::Sampler) {
            auto &info = sampler_infos[descr_sizes.sampler_count++];
            info.sampler = b.handle.sampler->vk_handle();

            auto &new_write = descr_writes.emplace_back();
            new_write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            new_write.dstBinding = b.loc;
            new_write.dstArrayElement = b.offset;
            new_write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            new_write.descriptorCount = 1;
            new_write.pImageInfo = &info;
        } else if (b.trg == eBindTarget::AccStruct) {
            auto &info = desc_tlas_infos[descr_sizes.acc_count++];
            info = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
            info.pAccelerationStructures = &b.handle.acc_struct->vk_handle();
            info.accelerationStructureCount = 1;

            auto &new_write = descr_writes.emplace_back();
            new_write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            new_write.dstBinding = b.loc;
            new_write.dstArrayElement = 0;
            new_write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            new_write.descriptorCount = 1;
            new_write.pNext = &info;
        }
    }

    VkDescriptorSet descr_set = descr_alloc->Alloc(descr_sizes, layout);
    if (!descr_set) {
        log->Error("Failed to allocate descriptor set!");
        return VK_NULL_HANDLE;
    }

    for (auto &d : descr_writes) {
        d.dstSet = descr_set;
    }

    ctx->api().vkUpdateDescriptorSets(ctx->device(), descr_writes.size(), descr_writes.data(), 0, nullptr);

    return descr_set;
}

void Ray::Vk::DispatchCompute(VkCommandBuffer cmd_buf, const Pipeline &comp_pipeline, const uint32_t grp_count[3],
                              Span<const Binding> bindings, const void *uniform_data, int uniform_data_len,
                              DescrMultiPoolAlloc *descr_alloc, ILog *log) {
    Context *ctx = descr_alloc->ctx();

    VkDescriptorSet descr_set =
        PrepareDescriptorSet(ctx, comp_pipeline.prog()->descr_set_layouts()[0], bindings, descr_alloc, log);
    if (!descr_set) {
        log->Error("Failed to allocate descriptor set, skipping draw call!");
        return;
    }

    ctx->api().vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, comp_pipeline.handle());
    ctx->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, comp_pipeline.layout(), 0, 1,
                                       &descr_set, 0, nullptr);

    if (uniform_data) {
        ctx->api().vkCmdPushConstants(cmd_buf, comp_pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, uniform_data_len,
                                      uniform_data);
    }

    ctx->api().vkCmdDispatch(cmd_buf, grp_count[0], grp_count[1], grp_count[2]);
}

void Ray::Vk::DispatchComputeIndirect(VkCommandBuffer cmd_buf, const Pipeline &comp_pipeline, const Buffer &indir_buf,
                                      const uint32_t indir_buf_offset, Span<const Binding> bindings,
                                      const void *uniform_data, int uniform_data_len, DescrMultiPoolAlloc *descr_alloc,
                                      ILog *log) {
    Context *ctx = descr_alloc->ctx();

    VkDescriptorSet descr_set =
        PrepareDescriptorSet(ctx, comp_pipeline.prog()->descr_set_layouts()[0], bindings, descr_alloc, log);
    if (!descr_set) {
        log->Error("Failed to allocate descriptor set, skipping draw call!");
        return;
    }

    ctx->api().vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, comp_pipeline.handle());
    ctx->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, comp_pipeline.layout(), 0, 1,
                                       &descr_set, 0, nullptr);

    if (uniform_data && uniform_data_len) {
        ctx->api().vkCmdPushConstants(cmd_buf, comp_pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, uniform_data_len,
                                      uniform_data);
    }

    ctx->api().vkCmdDispatchIndirect(cmd_buf, indir_buf.vk_handle(), VkDeviceSize(indir_buf_offset));
}

void Ray::Vk::TraceRays(VkCommandBuffer cmd_buf, const Pipeline &rt_pipeline, const uint32_t dims[3],
                        Span<const Binding> bindings, const void *uniform_data, const int uniform_data_len,
                        DescrMultiPoolAlloc *descr_alloc, ILog *log) {
    Context *ctx = descr_alloc->ctx();

    VkDescriptorSet descr_set =
        PrepareDescriptorSet(ctx, rt_pipeline.prog()->descr_set_layouts()[0], bindings, descr_alloc, log);
    if (!descr_set) {
        log->Error("Failed to allocate descriptor set, skipping draw call!");
        return;
    }

    ctx->api().vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline.handle());
    ctx->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline.layout(), 0, 1,
                                       &descr_set, 0, nullptr);

    if (uniform_data) {
        // TODO: Properly determine required stages!
        ctx->api().vkCmdPushConstants(cmd_buf, rt_pipeline.layout(), VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0,
                                      uniform_data_len, uniform_data);
    }

    ctx->api().vkCmdTraceRaysKHR(cmd_buf, rt_pipeline.rgen_table(), rt_pipeline.miss_table(), rt_pipeline.hit_table(),
                                 rt_pipeline.call_table(), dims[0], dims[1], dims[2]);
}

void Ray::Vk::TraceRaysIndirect(VkCommandBuffer cmd_buf, const Pipeline &rt_pipeline, const Buffer &indir_buf,
                                const uint32_t indir_buf_offset, Span<const Binding> bindings, const void *uniform_data,
                                int uniform_data_len, DescrMultiPoolAlloc *descr_alloc, ILog *log) {
    Context *ctx = descr_alloc->ctx();

    VkDescriptorSet descr_set =
        PrepareDescriptorSet(ctx, rt_pipeline.prog()->descr_set_layouts()[0], bindings, descr_alloc, log);
    if (!descr_set) {
        log->Error("Failed to allocate descriptor set, skipping draw call!");
        return;
    }

    ctx->api().vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline.handle());
    ctx->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline.layout(), 0, 1,
                                       &descr_set, 0, nullptr);

    if (uniform_data && uniform_data_len) {
        // TODO: Properly determine required stages!
        ctx->api().vkCmdPushConstants(cmd_buf, rt_pipeline.layout(), VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0,
                                      uniform_data_len, uniform_data);
    }

    ctx->api().vkCmdTraceRaysIndirectKHR(cmd_buf, rt_pipeline.rgen_table(), rt_pipeline.miss_table(),
                                         rt_pipeline.hit_table(), rt_pipeline.call_table(),
                                         indir_buf.vk_device_address() + indir_buf_offset);
}
