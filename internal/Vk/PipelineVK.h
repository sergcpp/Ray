#pragma once

#include <cstdint>

#include "../../Span.h"
#include "../ImageParams.h"
#include "../RastState.h"
#include "BufferVK.h"
#include "ProgramVK.h"

namespace Ray {
class ILog;
namespace Vk {
class Context;
class RenderPass;
class VertexInput;
struct RenderTarget;
struct RenderTargetInfo;

enum class ePipelineType : uint8_t { Undefined, Graphics, Compute, Raytracing };

struct DispatchIndirectCommand {
    uint32_t num_groups_x;
    uint32_t num_groups_y;
    uint32_t num_groups_z;
};

struct TraceRaysIndirectCommand {
    uint32_t width;
    uint32_t height;
    uint32_t depth;
};

class Pipeline {
    Context *ctx_ = nullptr;
    ePipelineType type_ = ePipelineType::Undefined;
    RastState rast_state_;
    const RenderPass *render_pass_ = nullptr;
    SmallVector<eFormat, 4> color_formats_;
    eFormat depth_format_ = eFormat::Undefined;
    Program *prog_ = nullptr;
    const VertexInput *vtx_input_ = nullptr;

    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline handle_ = VK_NULL_HANDLE;

    SmallVector<VkRayTracingShaderGroupCreateInfoKHR, 4> rt_shader_groups_;

    VkStridedDeviceAddressRegionKHR rgen_region_ = {};
    VkStridedDeviceAddressRegionKHR miss_region_ = {};
    VkStridedDeviceAddressRegionKHR hit_region_ = {};
    VkStridedDeviceAddressRegionKHR call_region_ = {};

    Buffer rt_sbt_buf_;

    void Destroy();

    bool Init(Context *ctx, const RastState &rast_state, Program *prog, const VertexInput *vtx_input,
              const RenderPass *render_pass, Span<const RenderTargetInfo> color_attachments,
              RenderTargetInfo depth_attachment, uint32_t subpass_index, ILog *log);

  public:
    Pipeline() = default;
    Pipeline(const Pipeline &rhs) = delete;
    Pipeline(Pipeline &&rhs) noexcept { (*this) = std::move(rhs); }
    ~Pipeline();

    Pipeline &operator=(const Pipeline &rhs) = delete;
    Pipeline &operator=(Pipeline &&rhs) noexcept;

    explicit operator bool() const { return ctx_ != nullptr; }

    ePipelineType type() const { return type_; }
    const RastState &rast_state() const { return rast_state_; }
    const RenderPass *render_pass() const { return render_pass_; }
    const Program *prog() const { return prog_; }
    const VertexInput *vtx_input() const { return vtx_input_; }

    const SmallVectorImpl<eFormat> &color_formats() const { return color_formats_; }
    eFormat depth_format() const { return depth_format_; }

    VkPipelineLayout layout() const { return layout_; }
    VkPipeline handle() const { return handle_; }

    const VkStridedDeviceAddressRegionKHR *rgen_table() const { return &rgen_region_; }
    const VkStridedDeviceAddressRegionKHR *miss_table() const { return &miss_region_; }
    const VkStridedDeviceAddressRegionKHR *hit_table() const { return &hit_region_; }
    const VkStridedDeviceAddressRegionKHR *call_table() const { return &call_region_; }

    bool Init(Context *ctx, const RastState &rast_state, Program *prog, const VertexInput *vtx_input,
              const RenderPass *render_pass, uint32_t subpass_index, ILog *log);
    bool Init(Context *ctx, const RastState &rast_state, Program *prog, const VertexInput *vtx_input,
              Span<const RenderTarget> color_attachments, RenderTarget depth_attachment, uint32_t subpass_index,
              ILog *log);
    bool Init(Context *ctx, Program *prog, ILog *log, int subgroup_size = -1, bool require_full_subgroup = false);
};
} // namespace Vk
} // namespace Ray
