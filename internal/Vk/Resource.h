#pragma once

#include <cstdint>

#define COUNT_OF(x) ((sizeof(x) / sizeof(0 [x])) / ((size_t)(!(sizeof(x) % sizeof(0 [x])))))

#include "../Span.h"
#include "VK.h"

namespace Ray {
namespace Vk {
enum class eStageBits : uint16_t {
    None = 0,
    VertexInput = (1u << 0u),
    VertexShader = (1u << 1u),
    TessCtrlShader = (1u << 2u),
    TessEvalShader = (1u << 3u),
    GeometryShader = (1u << 4u),
    FragmentShader = (1u << 5u),
    ComputeShader = (1u << 6u),
    RayTracingShader = (1u << 7u),
    ColorAttachment = (1u << 8u),
    DepthAttachment = (1u << 9u),
    DrawIndirect = (1u << 10u),
    Transfer = (1u << 11u),
    AccStructureBuild = (1u << 12u)
};
inline eStageBits operator|(const eStageBits lhs, const eStageBits rhs) {
    return eStageBits(uint16_t(lhs) | uint16_t(rhs));
}
inline eStageBits operator&(const eStageBits lhs, const eStageBits rhs) {
    return eStageBits(uint16_t(lhs) & uint16_t(rhs));
}
inline eStageBits operator|=(eStageBits &lhs, const eStageBits rhs) {
    lhs = eStageBits(uint16_t(lhs) | uint16_t(rhs));
    return lhs;
}
inline eStageBits operator&=(eStageBits &lhs, const eStageBits rhs) {
    lhs = eStageBits(uint16_t(lhs) & uint16_t(rhs));
    return lhs;
}

const eStageBits AllStages = eStageBits::VertexInput | eStageBits::VertexShader | eStageBits::TessCtrlShader |
                             eStageBits::TessEvalShader | eStageBits::GeometryShader | eStageBits::FragmentShader |
                             eStageBits::ComputeShader | eStageBits::RayTracingShader | eStageBits::ColorAttachment |
                             eStageBits::DepthAttachment | eStageBits::DrawIndirect | eStageBits::Transfer |
                             eStageBits::AccStructureBuild;

enum class eResState : uint8_t {
    Undefined,
    VertexBuffer,
    UniformBuffer,
    IndexBuffer,
    RenderTarget,
    UnorderedAccess,
    DepthRead,
    DepthWrite,
    StencilTestDepthFetch,
    ShaderResource,
    IndirectArgument,
    CopyDst,
    CopySrc,
    BuildASRead,
    BuildASWrite,
    RayTracing,
    _Count
};

VkImageLayout VKImageLayoutForState(eResState state);
uint32_t VKAccessFlagsForState(eResState state);
uint32_t VKPipelineStagesForState(eResState state);

eStageBits StageBitsForState(eResState state);

class Buffer;
class Texture2D;
class TextureAtlas;

struct TransitionInfo {
    const Texture2D *p_tex = nullptr;
    const TextureAtlas *p_tex_arr = nullptr;
    const Buffer *p_buf = nullptr;

    eResState old_state = eResState::Undefined;
    eResState new_state = eResState::Undefined;

    bool update_internal_state = false;

    TransitionInfo() = default;
    TransitionInfo(const Texture2D *_p_tex, eResState _new_state)
        : p_tex(_p_tex), new_state(_new_state), update_internal_state(true) {}
    TransitionInfo(const TextureAtlas *_p_tex_arr, eResState _new_state)
        : p_tex_arr(_p_tex_arr), new_state(_new_state), update_internal_state(true) {}
    TransitionInfo(const Buffer *_p_buf, eResState _new_state)
        : p_buf(_p_buf), new_state(_new_state), update_internal_state(true) {}
};

void TransitionResourceStates(void *_cmd_buf, eStageBits src_stages_mask, eStageBits dst_stages_mask,
                              Span<const TransitionInfo> transitions);
} // namespace Vk
} // namespace Ray