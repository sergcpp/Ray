#include "ResourceVK.h"

#include "ContextVK.h"
#include "TextureAtlasVK.h"
#include "TextureVK.h"

namespace Ray::Vk {
const Bitmask<eStage> g_stage_bits_per_state[] = {
    {},                  // Undefined
    eStage::VertexInput, // VertexBuffer
    Bitmask<eStage>{eStage::VertexShader} | eStage::FragmentShader | eStage::ComputeShader |
        eStage::RayTracingShader, // UniformBuffer
    eStage::VertexInput,          // IndexBuffer
    eStage::ColorAttachment,      // RenderTarget
    Bitmask<eStage>{eStage::VertexShader} | eStage::FragmentShader | eStage::ComputeShader |
        eStage::RayTracingShader,                                      // UnorderedAccess
    eStage::DepthAttachment,                                           // DepthRead
    eStage::DepthAttachment,                                           // DepthWrite
    Bitmask<eStage>{eStage::DepthAttachment} | eStage::FragmentShader, // StencilTestDepthFetch
    Bitmask<eStage>{eStage::VertexShader} | eStage::FragmentShader | eStage::ComputeShader |
        eStage::RayTracingShader, // ShaderResource
    eStage::DrawIndirect,         // IndirectArgument
    eStage::Transfer,             // CopyDst
    eStage::Transfer,             // CopySrc
    eStage::AccStructureBuild,    // BuildASRead
    eStage::AccStructureBuild,    // BuildASWrite
    eStage::RayTracingShader      // RayTracing
};
static_assert(std::size(g_stage_bits_per_state) == int(eResState::_Count), "!");

const VkPipelineStageFlags g_stage_flags_vk[] = {
    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,                                                     // VertexInput
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,                                                    // VertexShader
    VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT,                                      // TessCtrlShader
    VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT,                                   // TessEvalShader
    VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT,                                                  // GeometryShader
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,                                                  // FragmentShader
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,                                                   // ComputeShader
    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,                                           // RayTracingShader
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,                                          // ColorAttachment
    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, // DepthAttachment
    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,                                                    // DrawIndirect
    VK_PIPELINE_STAGE_TRANSFER_BIT,                                                         // Transfer
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR                                  // AccStructureBuild
};

static_assert(uint16_t(Bitmask<eStage>{eStage::VertexInput}) == 0b0000000000001u, "!");
static_assert(uint16_t(Bitmask<eStage>{eStage::VertexShader}) == 0b0000000000010u, "!");
static_assert(uint16_t(Bitmask<eStage>{eStage::TessCtrlShader}) == 0b0000000000100u, "!");
static_assert(uint16_t(Bitmask<eStage>{eStage::TessEvalShader}) == 0b0000000001000u, "!");
static_assert(uint16_t(Bitmask<eStage>{eStage::GeometryShader}) == 0b0000000010000u, "!");
static_assert(uint16_t(Bitmask<eStage>{eStage::FragmentShader}) == 0b0000000100000u, "!");
static_assert(uint16_t(Bitmask<eStage>{eStage::ComputeShader}) == 0b0000001000000u, "!");
static_assert(uint16_t(Bitmask<eStage>{eStage::RayTracingShader}) == 0b0000010000000u, "!");
static_assert(uint16_t(Bitmask<eStage>{eStage::ColorAttachment}) == 0b0000100000000u, "!");
static_assert(uint16_t(Bitmask<eStage>{eStage::DepthAttachment}) == 0b0001000000000u, "!");
static_assert(uint16_t(Bitmask<eStage>{eStage::DrawIndirect}) == 0b0010000000000u, "!");
static_assert(uint16_t(Bitmask<eStage>{eStage::Transfer}) == 0b0100000000000u, "!");
static_assert(uint16_t(Bitmask<eStage>{eStage::AccStructureBuild}) == 0b1000000000000u, "!");

VkPipelineStageFlags to_pipeline_stage_flags_vk(const Bitmask<eStage> stage_mask) {
    uint16_t mask_u16 = uint16_t(stage_mask);

    VkPipelineStageFlags ret = 0;
    for (int i = 0; mask_u16; mask_u16 >>= 1, i++) {
        if (mask_u16 & 1u) {
            ret |= g_stage_flags_vk[i];
        }
    }

    return ret;
}

const VkImageLayout g_image_layout_per_state_vk[] = {
    VK_IMAGE_LAYOUT_UNDEFINED,                        // Undefined
    VK_IMAGE_LAYOUT_UNDEFINED,                        // VertexBuffer
    VK_IMAGE_LAYOUT_UNDEFINED,                        // UniformBuffer
    VK_IMAGE_LAYOUT_UNDEFINED,                        // IndexBuffer
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,         // RenderTarget
    VK_IMAGE_LAYOUT_GENERAL,                          // UnorderedAccess,
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,  // DepthRead
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, // DepthWrite
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,  // StencilTestDepthFetch
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,         // ShaderResource
    VK_IMAGE_LAYOUT_UNDEFINED,                        // IndirectArgument
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,             // CopyDst
    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,             // CopySrc
    VK_IMAGE_LAYOUT_UNDEFINED,                        // BuildASRead
    VK_IMAGE_LAYOUT_UNDEFINED,                        // BuildASWrite
    VK_IMAGE_LAYOUT_UNDEFINED                         // RayTracing
};
static_assert(std::size(g_image_layout_per_state_vk) == int(eResState::_Count), "!");

const VkAccessFlags g_access_flags_per_state_vk[] = {
    0,                                                                                          // Undefined
    VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,                                                        // VertexBuffer
    VK_ACCESS_UNIFORM_READ_BIT,                                                                 // UniformBuffer
    VK_ACCESS_INDEX_READ_BIT,                                                                   // IndexBuffer
    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,                 // RenderTarget
    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,                                     // UnorderedAccess,
    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,                                                // DepthRead
    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, // DepthWrite
    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT,                    // StencilTestDepthFetch
    VK_ACCESS_SHADER_READ_BIT,                                                                  // ShaderResource
    VK_ACCESS_INDIRECT_COMMAND_READ_BIT,                                                        // IndirectArgument
    VK_ACCESS_TRANSFER_WRITE_BIT,                                                               // CopyDst
    VK_ACCESS_TRANSFER_READ_BIT,                                                                // CopySrc
    VK_ACCESS_SHADER_READ_BIT,                                                                  // BuildASRead
    VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, // BuildASWrite
    VK_ACCESS_SHADER_READ_BIT,                                                                      // RayTracing
};
static_assert(std::size(g_access_flags_per_state_vk) == int(eResState::_Count), "!");

const VkPipelineStageFlags g_pipeline_stages_per_state_vk[] = {
    {},                                 // Undefined
    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, // VertexBuffer
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, // UniformBuffer
    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,               // IndexBuffer
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,    // RenderTarget
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,                                       // UnorderedAccess
    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, // DepthRead
    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, // DepthWrite
    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, // StencilTestDepthFetch
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,       // ShaderResource
    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,                    // IndirectArgument
    VK_PIPELINE_STAGE_TRANSFER_BIT,                         // CopyDst
    VK_PIPELINE_STAGE_TRANSFER_BIT,                         // CopySrc
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, // BuildASRead
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, // BuildASWrite
    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR            // RayTracing
};
static_assert(std::size(g_pipeline_stages_per_state_vk) == int(eResState::_Count), "!");
} // namespace Ray::Vk

Ray::Bitmask<Ray::Vk::eStage> Ray::Vk::StagesForState(const eResState state) {
    return g_stage_bits_per_state[int(state)];
}

VkImageLayout Ray::Vk::VKImageLayoutForState(const eResState state) { return g_image_layout_per_state_vk[int(state)]; }

uint32_t Ray::Vk::VKAccessFlagsForState(const eResState state) { return g_access_flags_per_state_vk[int(state)]; }

uint32_t Ray::Vk::VKPipelineStagesForState(const eResState state) { return g_pipeline_stages_per_state_vk[int(state)]; }

void Ray::Vk::TransitionResourceStates(VkCommandBuffer cmd_buf, const Bitmask<eStage> src_stages_mask,
                                       const Bitmask<eStage> dst_stages_mask, Span<const TransitionInfo> transitions) {
    VkPipelineStageFlags src_stages = 0, dst_stages = 0;
    SmallVector<VkBufferMemoryBarrier, 32> buf_barriers;
    SmallVector<VkImageMemoryBarrier, 32> img_barriers;

    const Context *ctx = nullptr;

    for (const TransitionInfo &transition : transitions) {
        if (transition.p_tex && *transition.p_tex) {
            ctx = transition.p_tex->ctx();

            eResState old_state = transition.old_state;
            if (old_state == eResState::Undefined) {
                // take state from resource itself
                old_state = transition.p_tex->resource_state;
                if (old_state != eResState::Undefined && old_state == transition.new_state &&
                    old_state != eResState::UnorderedAccess) {
                    // transition is not needed
                    continue;
                }
            }

            auto &new_barrier = img_barriers.emplace_back();
            new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            new_barrier.srcAccessMask = VKAccessFlagsForState(old_state);
            new_barrier.dstAccessMask = VKAccessFlagsForState(transition.new_state);
            new_barrier.oldLayout = VKImageLayoutForState(old_state);
            new_barrier.newLayout = VKImageLayoutForState(transition.new_state);
            new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.image = transition.p_tex->handle().img;
            if (IsDepthStencilFormat(transition.p_tex->params.format)) {
                new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
            } else if (IsDepthFormat(transition.p_tex->params.format)) {
                new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            } else {
                new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            }
            // transition whole image for now
            new_barrier.subresourceRange.baseMipLevel = 0;
            new_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
            new_barrier.subresourceRange.baseArrayLayer = 0;
            new_barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

            src_stages |= VKPipelineStagesForState(old_state);
            dst_stages |= VKPipelineStagesForState(transition.new_state);

            if (transition.update_internal_state) {
                transition.p_tex->resource_state = transition.new_state;
            }
        } else if (transition.p_buf && *transition.p_buf) {
            ctx = transition.p_buf->ctx();

            eResState old_state = transition.old_state;
            if (old_state == eResState::Undefined) {
                // take state from resource itself
                old_state = transition.p_buf->resource_state;
                if (old_state == transition.new_state && old_state != eResState::UnorderedAccess) {
                    // transition is not needed
                    continue;
                }
            }

            auto &new_barrier = buf_barriers.emplace_back();
            new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            new_barrier.srcAccessMask = VKAccessFlagsForState(old_state);
            new_barrier.dstAccessMask = VKAccessFlagsForState(transition.new_state);
            new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.buffer = transition.p_buf->vk_handle();
            // transition whole buffer for now
            new_barrier.offset = 0;
            new_barrier.size = VK_WHOLE_SIZE;

            src_stages |= VKPipelineStagesForState(old_state);
            dst_stages |= VKPipelineStagesForState(transition.new_state);

            if (transition.update_internal_state) {
                transition.p_buf->resource_state = transition.new_state;
            }
        } else if (transition.p_tex_arr && transition.p_tex_arr->page_count()) {
            ctx = transition.p_tex_arr->ctx();

            eResState old_state = transition.old_state;
            if (old_state == eResState::Undefined) {
                // take state from resource itself
                old_state = transition.p_tex_arr->resource_state;
                if (old_state != eResState::Undefined && old_state == transition.new_state &&
                    old_state != eResState::UnorderedAccess) {
                    // transition is not needed
                    continue;
                }
            }

            auto &new_barrier = img_barriers.emplace_back();
            new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            new_barrier.srcAccessMask = VKAccessFlagsForState(old_state);
            new_barrier.dstAccessMask = VKAccessFlagsForState(transition.new_state);
            new_barrier.oldLayout = VKImageLayoutForState(old_state);
            new_barrier.newLayout = VKImageLayoutForState(transition.new_state);
            new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.image = transition.p_tex_arr->vk_image();
            new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

            // transition whole image for now
            new_barrier.subresourceRange.baseMipLevel = 0;
            new_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
            new_barrier.subresourceRange.baseArrayLayer = 0;
            new_barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

            src_stages |= VKPipelineStagesForState(old_state);
            dst_stages |= VKPipelineStagesForState(transition.new_state);

            if (transition.update_internal_state) {
                transition.p_tex_arr->resource_state = transition.new_state;
            }
        }
    }

    if (!ctx) {
        return;
    }

    src_stages &= to_pipeline_stage_flags_vk(src_stages_mask);
    dst_stages &= to_pipeline_stage_flags_vk(dst_stages_mask);

    if (!buf_barriers.empty() || !img_barriers.empty()) {
        src_stages &= ctx->supported_stages_mask();
        dst_stages &= ctx->supported_stages_mask();

        ctx->api().vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                        dst_stages, 0, 0, nullptr, buf_barriers.size(), buf_barriers.cdata(),
                                        img_barriers.size(), img_barriers.cdata());
    }
}