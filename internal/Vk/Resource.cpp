#include "Resource.h"

#include "Texture.h"
#include "TextureAtlas.h"

namespace Ray {
namespace Vk {
const eStageBits g_stage_bits_per_state[] = {
    {},                        // Undefined
    eStageBits::VertexInput,   // VertexBuffer
    eStageBits::VertexShader | /* eStageBits::TessCtrlShader | eStageBits::TessEvalShader | eStageBits::GeometryShader
                                  |*/
        eStageBits::FragmentShader | eStageBits::ComputeShader | eStageBits::RayTracingShader, // UniformBuffer
    eStageBits::VertexInput,                                                                   // IndexBuffer
    eStageBits::ColorAttachment,                                                               // RenderTarget
    eStageBits::VertexShader | /* eStageBits::TessCtrlShader | eStageBits::TessEvalShader | eStageBits::GeometryShader
                                  |*/
        eStageBits::FragmentShader | eStageBits::ComputeShader | eStageBits::RayTracingShader, // UnorderedAccess
    eStageBits::DepthAttachment,                                                               // DepthRead
    eStageBits::DepthAttachment,                                                               // DepthWrite
    eStageBits::DepthAttachment | eStageBits::FragmentShader,                                  // StencilTestDepthFetch
    eStageBits::VertexShader | /* eStageBits::TessCtrlShader | eStageBits::TessEvalShader | eStageBits::GeometryShader
                                  |*/
        eStageBits::FragmentShader | eStageBits::ComputeShader | eStageBits::RayTracingShader, // ShaderResource
    eStageBits::DrawIndirect,                                                                  // IndirectArgument
    eStageBits::Transfer,                                                                      // CopyDst
    eStageBits::Transfer,                                                                      // CopySrc
    eStageBits::AccStructureBuild,                                                             // BuildASRead
    eStageBits::AccStructureBuild,                                                             // BuildASWrite
    eStageBits::RayTracingShader                                                               // RayTracing
};
static_assert(sizeof(g_stage_bits_per_state) / sizeof(g_stage_bits_per_state[0]) == int(eResState::_Count), "!");

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

static_assert(uint16_t(eStageBits::VertexInput) == 0b0000000000001u, "!");
static_assert(uint16_t(eStageBits::VertexShader) == 0b0000000000010u, "!");
static_assert(uint16_t(eStageBits::TessCtrlShader) == 0b0000000000100u, "!");
static_assert(uint16_t(eStageBits::TessEvalShader) == 0b0000000001000u, "!");
static_assert(uint16_t(eStageBits::GeometryShader) == 0b0000000010000u, "!");
static_assert(uint16_t(eStageBits::FragmentShader) == 0b0000000100000u, "!");
static_assert(uint16_t(eStageBits::ComputeShader) == 0b0000001000000u, "!");
static_assert(uint16_t(eStageBits::RayTracingShader) == 0b0000010000000u, "!");
static_assert(uint16_t(eStageBits::ColorAttachment) == 0b0000100000000u, "!");
static_assert(uint16_t(eStageBits::DepthAttachment) == 0b0001000000000u, "!");
static_assert(uint16_t(eStageBits::DrawIndirect) == 0b0010000000000u, "!");
static_assert(uint16_t(eStageBits::Transfer) == 0b0100000000000u, "!");
static_assert(uint16_t(eStageBits::AccStructureBuild) == 0b1000000000000u, "!");

VkPipelineStageFlags to_pipeline_stage_flags_vk(const eStageBits stage_mask) {
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
static_assert(COUNT_OF(g_image_layout_per_state_vk) == int(eResState::_Count), "!");

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
static_assert(COUNT_OF(g_access_flags_per_state_vk) == int(eResState::_Count), "!");

const VkPipelineStageFlags g_pipeline_stages_per_state_vk[] = {
    {},                                   // Undefined
    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,   // VertexBuffer
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | /*VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
        VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT |*/
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, // UniformBuffer
    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,               // IndexBuffer
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,    // RenderTarget
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |             /*VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
                    VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT |*/
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,                                       // UnorderedAccess
    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, // DepthRead
    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, // DepthWrite
    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, // StencilTestDepthFetch
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |      /*VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
             VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT |*/
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,       // ShaderResource
    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,                    // IndirectArgument
    VK_PIPELINE_STAGE_TRANSFER_BIT,                         // CopyDst
    VK_PIPELINE_STAGE_TRANSFER_BIT,                         // CopySrc
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, // BuildASRead
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, // BuildASWrite
    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR            // RayTracing
};
static_assert(COUNT_OF(g_pipeline_stages_per_state_vk) == int(eResState::_Count), "!");
} // namespace Vk
} // namespace Ray

Ray::Vk::eStageBits Ray::Vk::StageBitsForState(const eResState state) { return g_stage_bits_per_state[int(state)]; }

VkImageLayout Ray::Vk::VKImageLayoutForState(const eResState state) { return g_image_layout_per_state_vk[int(state)]; }

uint32_t Ray::Vk::VKAccessFlagsForState(const eResState state) { return g_access_flags_per_state_vk[int(state)]; }

uint32_t Ray::Vk::VKPipelineStagesForState(const eResState state) { return g_pipeline_stages_per_state_vk[int(state)]; }

void Ray::Vk::TransitionResourceStates(void *_cmd_buf, const eStageBits src_stages_mask,
                                       const eStageBits dst_stages_mask, Span<const TransitionInfo> transitions) {
    VkCommandBuffer cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

    VkPipelineStageFlags src_stages = 0, dst_stages = 0;
    SmallVector<VkBufferMemoryBarrier, 32> buf_barriers;
    SmallVector<VkImageMemoryBarrier, 32> img_barriers;

    for (int i = 0; i < int(transitions.size()); i++) {
        if (transitions[i].p_tex) {
            eResState old_state = transitions[i].old_state;
            if (old_state == eResState::Undefined) {
                // take state from resource itself
                old_state = transitions[i].p_tex->resource_state;
                if (old_state != eResState::Undefined && old_state == transitions[i].new_state &&
                    old_state != eResState::UnorderedAccess) {
                    // transition is not needed
                    continue;
                }
            }

            auto &new_barrier = img_barriers.emplace_back();
            new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            new_barrier.srcAccessMask = VKAccessFlagsForState(old_state);
            new_barrier.dstAccessMask = VKAccessFlagsForState(transitions[i].new_state);
            new_barrier.oldLayout = VKImageLayoutForState(old_state);
            new_barrier.newLayout = VKImageLayoutForState(transitions[i].new_state);
            new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.image = transitions[i].p_tex->handle().img;
            if (IsDepthStencilFormat(transitions[i].p_tex->params.format)) {
                new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
            } else if (IsDepthFormat(transitions[i].p_tex->params.format)) {
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
            dst_stages |= VKPipelineStagesForState(transitions[i].new_state);

            if (transitions[i].update_internal_state) {
                transitions[i].p_tex->resource_state = transitions[i].new_state;
            }
        } else if (transitions[i].p_buf) {
            eResState old_state = transitions[i].old_state;
            if (old_state == eResState::Undefined) {
                // take state from resource itself
                old_state = transitions[i].p_buf->resource_state;
                if (old_state == transitions[i].new_state && old_state != eResState::UnorderedAccess) {
                    // transition is not needed
                    continue;
                }
            }

            auto &new_barrier = buf_barriers.emplace_back();
            new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            new_barrier.srcAccessMask = VKAccessFlagsForState(old_state);
            new_barrier.dstAccessMask = VKAccessFlagsForState(transitions[i].new_state);
            new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.buffer = transitions[i].p_buf->vk_handle();
            // transition whole buffer for now
            new_barrier.offset = 0;
            new_barrier.size = VK_WHOLE_SIZE;

            src_stages |= VKPipelineStagesForState(old_state);
            dst_stages |= VKPipelineStagesForState(transitions[i].new_state);

            if (transitions[i].update_internal_state) {
                transitions[i].p_buf->resource_state = transitions[i].new_state;
            }
        } else if (transitions[i].p_tex_arr && transitions[i].p_tex_arr->page_count()) {
            eResState old_state = transitions[i].old_state;
            if (old_state == eResState::Undefined) {
                // take state from resource itself
                old_state = transitions[i].p_tex_arr->resource_state;
                if (old_state != eResState::Undefined && old_state == transitions[i].new_state &&
                    old_state != eResState::UnorderedAccess) {
                    // transition is not needed
                    continue;
                }
            }

            auto &new_barrier = img_barriers.emplace_back();
            new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            new_barrier.srcAccessMask = VKAccessFlagsForState(old_state);
            new_barrier.dstAccessMask = VKAccessFlagsForState(transitions[i].new_state);
            new_barrier.oldLayout = VKImageLayoutForState(old_state);
            new_barrier.newLayout = VKImageLayoutForState(transitions[i].new_state);
            new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.image = transitions[i].p_tex_arr->vk_image();
            new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

            // transition whole image for now
            new_barrier.subresourceRange.baseMipLevel = 0;
            new_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
            new_barrier.subresourceRange.baseArrayLayer = 0;
            new_barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

            src_stages |= VKPipelineStagesForState(old_state);
            dst_stages |= VKPipelineStagesForState(transitions[i].new_state);

            if (transitions[i].update_internal_state) {
                transitions[i].p_tex_arr->resource_state = transitions[i].new_state;
            }
        }
    }

    src_stages &= to_pipeline_stage_flags_vk(src_stages_mask);
    dst_stages &= to_pipeline_stage_flags_vk(dst_stages_mask);

    if (!buf_barriers.empty() || !img_barriers.empty()) {
        vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stages, 0, 0,
                             nullptr, uint32_t(buf_barriers.size()), buf_barriers.cdata(),
                             uint32_t(img_barriers.size()), img_barriers.cdata());
    }
}