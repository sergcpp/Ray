#include "ResourceDX.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12.h>

#include "../SmallVector.h"
#include "BufferDX.h"
#include "TextureDX.h"
// #include "TextureAtlas.h"

namespace Ray {
namespace Dx {
#if 0
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
#endif

const D3D12_RESOURCE_STATES g_resource_states[] = {
    D3D12_RESOURCE_STATE_COMMON,                            // Undefined
    D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,        // VertexBuffer
    D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,        // UniformBuffer
    D3D12_RESOURCE_STATE_INDEX_BUFFER,                      // IndexBuffer
    D3D12_RESOURCE_STATE_RENDER_TARGET,                     // RenderTarget
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,                  // UnorderedAccess
    D3D12_RESOURCE_STATE_DEPTH_READ,                        // DepthRead
    D3D12_RESOURCE_STATE_DEPTH_WRITE,                       // DepthWrite
    D3D12_RESOURCE_STATE_DEPTH_READ,                        // StencilTestDepthFetch
    D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE,               // ShaderResource
    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,                 // IndirectArgument
    D3D12_RESOURCE_STATE_COPY_DEST,                         // CopyDst
    D3D12_RESOURCE_STATE_COPY_SOURCE,                       // CopySrc
    D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, // BuildASRead
    D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, // BuildASWrite
    D3D12_RESOURCE_STATE_GENERIC_READ                       // RayTracing
};
static_assert(COUNT_OF(g_resource_states) == int(eResState::_Count), "!");

} // namespace Dx
} // namespace Ray

D3D12_RESOURCE_STATES Ray::Dx::DXResourceState(const eResState state) { return g_resource_states[int(state)]; }

// Ray::Dx::eStageBits Ray::Dx::StageBitsForState(const eResState state) { return g_stage_bits_per_state[int(state)]; }

// VkImageLayout Ray::Vk::VKImageLayoutForState(const eResState state) { return g_image_layout_per_state_vk[int(state)];
// }

// uint32_t Ray::Vk::VKAccessFlagsForState(const eResState state) { return g_access_flags_per_state_vk[int(state)]; }

// uint32_t Ray::Vk::VKPipelineStagesForState(const eResState state) { return
// g_pipeline_stages_per_state_vk[int(state)]; }

void Ray::Dx::TransitionResourceStates(void *_cmd_buf, const eStageBits src_stages_mask,
                                       const eStageBits dst_stages_mask, Span<const TransitionInfo> transitions) {
    auto cmd_buf = reinterpret_cast<ID3D12GraphicsCommandList *>(_cmd_buf);

    SmallVector<D3D12_RESOURCE_BARRIER, 64> barriers;

    for (const TransitionInfo &transition : transitions) {
        if (transition.p_tex && transition.p_tex->ready()) {
            eResState old_state = transition.old_state;
            if (old_state == eResState::Undefined) {
                // take state from resource itself
                old_state = transition.p_tex->resource_state;
                if (old_state == transition.new_state &&
                    old_state != eResState::UnorderedAccess) {
                    // transition is not needed
                    continue;
                }
            }

            auto &new_barrier = barriers.emplace_back();
            if (old_state != transition.new_state) {
                new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                new_barrier.Transition.pResource = transition.p_tex->handle().img;
                new_barrier.Transition.StateBefore = DXResourceState(old_state);
                new_barrier.Transition.StateAfter = DXResourceState(transition.new_state);
                new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            } else {
                new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                new_barrier.UAV.pResource = transition.p_tex->handle().img;
            }

            if (transition.update_internal_state) {
                transition.p_tex->resource_state = transition.new_state;
            }
        } else /*if (transition.p_3dtex) {
            eResState old_state = transition.old_state;
            if (old_state == eResState::Undefined) {
                // take state from resource itself
                old_state = transition.p_3dtex->resource_state;
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
            new_barrier.image = transition.p_3dtex->handle().img;
            new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            // transition whole image for now
            new_barrier.subresourceRange.baseMipLevel = 0;
            new_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
            new_barrier.subresourceRange.baseArrayLayer = 0;
            new_barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

            src_stages |= VKPipelineStagesForState(old_state);
            dst_stages |= VKPipelineStagesForState(transition.new_state);

            if (transition.update_internal_state) {
                transition.p_3dtex->resource_state = transition.new_state;
            }
        } else*/
        if (transition.p_buf && *transition.p_buf) {
            eResState old_state = transition.old_state;
            if (old_state == eResState::Undefined) {
                // take state from resource itself
                old_state = transition.p_buf->resource_state;
                if (old_state == transition.new_state && old_state != eResState::UnorderedAccess) {
                    // transition is not needed
                    continue;
                }
            }

            auto &new_barrier = barriers.emplace_back();
            if (old_state != transition.new_state) {
                new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                new_barrier.Transition.pResource = transition.p_buf->dx_resource();
                new_barrier.Transition.StateBefore = DXResourceState(old_state);
                new_barrier.Transition.StateAfter = DXResourceState(transition.new_state);
                new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            } else {
                new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                new_barrier.UAV.pResource = transition.p_buf->dx_resource();
            }

            if (transition.update_internal_state) {
                transition.p_buf->resource_state = transition.new_state;
            }
        } /*else if (transition.p_tex_arr && transition.p_tex_arr->page_count()) {
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
        }*/
    }

    if (!barriers.empty()) {
        cmd_buf->ResourceBarrier(UINT(barriers.size()), barriers.data());
    }
}