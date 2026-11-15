#pragma once

#include <cstdint>
#include <variant>

#include "../../Bitmask.h"
#include "../../Span.h"
#include "ApiMtl.h"

namespace Ray::Mtl {
enum class eStage : uint16_t {
    VertexInput,
    VertexShader,
    TessCtrlShader,
    TessEvalShader,
    GeometryShader,
    FragmentShader,
    ComputeShader,
    RayTracingShader,
    ColorAttachment,
    DepthAttachment,
    DrawIndirect,
    Transfer,
    AccStructureBuild
};

const Bitmask<eStage> AllStages = Bitmask<eStage>{eStage::VertexInput} | eStage::VertexShader | eStage::TessCtrlShader |
                                  eStage::TessEvalShader | eStage::GeometryShader | eStage::FragmentShader |
                                  eStage::ComputeShader | eStage::RayTracingShader | eStage::ColorAttachment |
                                  eStage::DepthAttachment | eStage::DrawIndirect | eStage::Transfer |
                                  eStage::AccStructureBuild;

// Advisory-only states: kept so the shared kernels' Transition* calls port unchanged;
// Metal encoders resolve hazards natively and no barrier is ever emitted.
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

const eResState ResStateForClear = eResState::CopyDst;

Bitmask<eStage> StagesForState(eResState state);

class Buffer;
class Image;
class ImageAtlas;

struct TransitionInfo {
    std::variant<const Image *, const ImageAtlas *, const Buffer *> p_res;

    eResState old_state = eResState::Undefined;
    eResState new_state = eResState::Undefined;

    bool update_internal_state = false;

    TransitionInfo() = default;
    TransitionInfo(const Image *_p_tex, eResState _new_state)
        : p_res(_p_tex), new_state(_new_state), update_internal_state(true) {}
    TransitionInfo(const ImageAtlas *_p_tex_arr, eResState _new_state)
        : p_res(_p_tex_arr), new_state(_new_state), update_internal_state(true) {}
    TransitionInfo(const Buffer *_p_buf, eResState _new_state)
        : p_res(_p_buf), new_state(_new_state), update_internal_state(true) {}
};

void TransitionResourceStates(CommandBuffer cmd_buf, Bitmask<eStage> src_stages_mask, Bitmask<eStage> dst_stages_mask,
                              Span<const TransitionInfo> transitions);
} // namespace Ray::Mtl
