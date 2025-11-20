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
#include "ImageAtlasDX.h"
#include "ImageDX.h"

namespace Ray::Dx {
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
static_assert(std::size(g_resource_states) == int(eResState::_Count), "!");
} // namespace Ray::Dx

D3D12_RESOURCE_STATES Ray::Dx::DXResourceState(const eResState state) { return g_resource_states[int(state)]; }

// Ray::Dx::eStageBits Ray::Dx::StageBitsForState(const eResState state) { return g_stage_bits_per_state[int(state)]; }

// VkImageLayout Ray::Vk::VKImageLayoutForState(const eResState state) { return g_image_layout_per_state_vk[int(state)];
// }

// uint32_t Ray::Vk::VKAccessFlagsForState(const eResState state) { return g_access_flags_per_state_vk[int(state)]; }

// uint32_t Ray::Vk::VKPipelineStagesForState(const eResState state) { return
// g_pipeline_stages_per_state_vk[int(state)]; }

void Ray::Dx::TransitionResourceStates(ID3D12GraphicsCommandList *cmd_buf, const Bitmask<eStage> src_stages_mask,
                                       const Bitmask<eStage> dst_stages_mask, Span<const TransitionInfo> transitions) {
    SmallVector<D3D12_RESOURCE_BARRIER, 64> barriers;

    for (const TransitionInfo &transition : transitions) {
        if (std::holds_alternative<const Image *>(transition.p_res) && *std::get<const Image *>(transition.p_res)) {
            eResState old_state = transition.old_state;
            if (old_state == eResState::Undefined) {
                // take state from resource itself
                old_state = std::get<const Image *>(transition.p_res)->resource_state;
                if (old_state == transition.new_state && old_state != eResState::UnorderedAccess) {
                    // transition is not needed
                    continue;
                }
            }

            auto &new_barrier = barriers.emplace_back();
            if (old_state != transition.new_state) {
                new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                new_barrier.Transition.pResource = std::get<const Image *>(transition.p_res)->handle().img;
                new_barrier.Transition.StateBefore = DXResourceState(old_state);
                new_barrier.Transition.StateAfter = DXResourceState(transition.new_state);
                new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            } else {
                new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                new_barrier.UAV.pResource = std::get<const Image *>(transition.p_res)->handle().img;
            }

            if (transition.update_internal_state) {
                std::get<const Image *>(transition.p_res)->resource_state = transition.new_state;
            }
        } else if (std::holds_alternative<const Buffer *>(transition.p_res) &&
                   *std::get<const Buffer *>(transition.p_res)) {
            eResState old_state = transition.old_state;
            if (old_state == eResState::Undefined) {
                // take state from resource itself
                old_state = std::get<const Buffer *>(transition.p_res)->resource_state;
                if (old_state == transition.new_state && old_state != eResState::UnorderedAccess) {
                    // transition is not needed
                    continue;
                }
            }

            auto &new_barrier = barriers.emplace_back();
            if (old_state != transition.new_state) {
                new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                new_barrier.Transition.pResource = std::get<const Buffer *>(transition.p_res)->dx_resource();
                new_barrier.Transition.StateBefore = DXResourceState(old_state);
                new_barrier.Transition.StateAfter = DXResourceState(transition.new_state);
                new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            } else {
                new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                new_barrier.UAV.pResource = std::get<const Buffer *>(transition.p_res)->dx_resource();
            }

            if (transition.update_internal_state) {
                std::get<const Buffer *>(transition.p_res)->resource_state = transition.new_state;
            }
        } else if (std::holds_alternative<const ImageAtlas *>(transition.p_res) &&
                   std::get<const ImageAtlas *>(transition.p_res)->page_count()) {
            eResState old_state = transition.old_state;
            if (old_state == eResState::Undefined) {
                // take state from resource itself
                old_state = std::get<const ImageAtlas *>(transition.p_res)->resource_state;
                if (old_state != eResState::Undefined && old_state == transition.new_state &&
                    old_state != eResState::UnorderedAccess) {
                    // transition is not needed
                    continue;
                }
            }

            auto &new_barrier = barriers.emplace_back();
            if (old_state != transition.new_state) {
                new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                new_barrier.Transition.pResource = std::get<const ImageAtlas *>(transition.p_res)->dx_resource();
                new_barrier.Transition.StateBefore = DXResourceState(old_state);
                new_barrier.Transition.StateAfter = DXResourceState(transition.new_state);
                new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            } else {
                new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                new_barrier.UAV.pResource = std::get<const ImageAtlas *>(transition.p_res)->dx_resource();
            }

            if (transition.update_internal_state) {
                std::get<const ImageAtlas *>(transition.p_res)->resource_state = transition.new_state;
            }
        }
    }

    if (!barriers.empty()) {
        cmd_buf->ResourceBarrier(UINT(barriers.size()), barriers.data());
    }
}