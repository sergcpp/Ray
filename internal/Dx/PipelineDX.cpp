#include "PipelineDX.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <d3d12.h>

#include "../../Log.h"
#include "../RastState.h"
#include "ContextDX.h"
#include "ProgramDX.h"
#include "RenderPassDX.h"
#include "VertexInputDX.h"

namespace Ray {
namespace Dx {
// extern const VkShaderStageFlagBits g_shader_stages_vk[];

/*const VkCullModeFlagBits g_cull_modes_vk[] = {
    VK_CULL_MODE_NONE,      // None
    VK_CULL_MODE_FRONT_BIT, // Front
    VK_CULL_MODE_BACK_BIT   // Back
};
static_assert(std::size(g_cull_modes_vk) == int(eCullFace::_Count), "!");

#define X(_0, _1, _2) _1,
const VkCompareOp g_compare_op_vk[] = {
#include "../CompareOp.inl"
};
#undef X

const VkStencilOp g_stencil_op_vk[] = {
    VK_STENCIL_OP_KEEP,                // Keep
    VK_STENCIL_OP_ZERO,                // Zero
    VK_STENCIL_OP_REPLACE,             // Replace
    VK_STENCIL_OP_INCREMENT_AND_CLAMP, // Incr
    VK_STENCIL_OP_DECREMENT_AND_CLAMP, // Decr
    VK_STENCIL_OP_INVERT               // Invert
};
static_assert(std::size(g_stencil_op_vk) == int(eStencilOp::_Count), "!");

const VkPolygonMode g_poly_mode_vk[] = {
    VK_POLYGON_MODE_FILL, // Fill
    VK_POLYGON_MODE_LINE  // Line
};
static_assert(std::size(g_poly_mode_vk) == int(ePolygonMode::_Count), "!");

const VkBlendFactor g_blend_factor_vk[] = {
    VK_BLEND_FACTOR_ZERO,                // Zero
    VK_BLEND_FACTOR_ONE,                 // One
    VK_BLEND_FACTOR_SRC_COLOR,           // SrcColor
    VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, // OneMinusSrcColor
    VK_BLEND_FACTOR_DST_COLOR,           // DstColor
    VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR, // OneMinusDstColor
    VK_BLEND_FACTOR_SRC_ALPHA,           // SrcAlpha
    VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, // OneMinusSrcAlpha
    VK_BLEND_FACTOR_DST_ALPHA,           // DstAlpha
    VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA  // OneMinusDstAlpha
};
static_assert(std::size(g_blend_factor_vk) == int(eBlendFactor::_Count), "!");

uint32_t align_up(const uint32_t size, const uint32_t alignment) { return (size + alignment - 1) & ~(alignment - 1); }

static_assert(sizeof(TraceRaysIndirectCommand) == sizeof(VkTraceRaysIndirectCommandKHR), "!");*/
} // namespace Dx
} // namespace Ray

Ray::Dx::Pipeline &Ray::Dx::Pipeline::operator=(Pipeline &&rhs) noexcept {
    if (this == &rhs) {
        return (*this);
    }

    Destroy();

    ctx_ = std::exchange(rhs.ctx_, nullptr);
    type_ = std::exchange(rhs.type_, ePipelineType::Undefined);
    rast_state_ = std::exchange(rhs.rast_state_, {});
    render_pass_ = std::exchange(rhs.render_pass_, nullptr);
    prog_ = std::exchange(rhs.prog_, nullptr);
    vtx_input_ = std::exchange(rhs.vtx_input_, nullptr);
    // layout_ = std::exchange(rhs.layout_, {});
    handle_ = std::exchange(rhs.handle_, nullptr);

    // rt_shader_groups_ = std::move(rhs.rt_shader_groups_);

    // rgen_region_ = std::exchange(rhs.rgen_region_, {});
    // miss_region_ = std::exchange(rhs.miss_region_, {});
    // hit_region_ = std::exchange(rhs.hit_region_, {});
    // call_region_ = std::exchange(rhs.call_region_, {});

    // rt_sbt_buf_ = std::move(rhs.rt_sbt_buf_);

    return (*this);
}

Ray::Dx::Pipeline::~Pipeline() { Destroy(); }

void Ray::Dx::Pipeline::Destroy() {
    if (handle_) {
        ctx_->pipelines_to_destroy[ctx_->backend_frame].emplace_back(handle_);
        handle_ = nullptr;
    }

    color_formats_.clear();
    depth_format_ = eFormat::Undefined;

    // rt_shader_groups_.clear();

    // rgen_region_ = {};
    // miss_region_ = {};
    // hit_region_ = {};
    // call_region_ = {};

    // rt_sbt_buf_ = {};
}

bool Ray::Dx::Pipeline::Init(Context *ctx, const RastState &rast_state, Program *prog, const VertexInput *vtx_input,
                             const RenderPass *render_pass, Span<const RenderTargetInfo> color_attachments,
                             RenderTargetInfo depth_attachment, uint32_t subpass_index, ILog *log) {
    Destroy();

    /*SmallVector<VkPipelineShaderStageCreateInfo, int(eShaderType::_Count)> shader_stage_create_info;
    for (int i = 0; i < int(eShaderType::_Count); ++i) {
        const Shader *sh = prog->shader(eShaderType(i));
        if (!sh) {
            continue;
        }

        auto &stage_info = shader_stage_create_info.emplace_back();
        stage_info = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage_info.stage = g_shader_stages_vk[i];
        stage_info.module = prog->shader(eShaderType(i))->module();
        stage_info.pName = "main";
        stage_info.pSpecializationInfo = nullptr;
    }

    { // create pipeline layout
        VkPipelineLayoutCreateInfo layout_create_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_create_info.setLayoutCount = prog->descr_set_layouts_count();
        layout_create_info.pSetLayouts = prog->descr_set_layouts();
        layout_create_info.pushConstantRangeCount = prog->pc_range_count();
        layout_create_info.pPushConstantRanges = prog->pc_ranges();

        const VkResult res = vkCreatePipelineLayout(ctx->device(), &layout_create_info, nullptr, &layout_);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create pipeline layout!");
            return false;
        }
    }

    { // create graphics pipeline
        SmallVector<VkVertexInputBindingDescription, 8> bindings;
        SmallVector<VkVertexInputAttributeDescription, 8> attribs;
        vtx_input->FillVKDescriptions(bindings, attribs);

        VkPipelineVertexInputStateCreateInfo vtx_input_state_create_info = {
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vtx_input_state_create_info.vertexBindingDescriptionCount = uint32_t(bindings.size());
        vtx_input_state_create_info.pVertexBindingDescriptions = bindings.cdata();
        vtx_input_state_create_info.vertexAttributeDescriptionCount = uint32_t(attribs.size());
        vtx_input_state_create_info.pVertexAttributeDescriptions = attribs.cdata();

        VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info = {
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        input_assembly_state_create_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly_state_create_info.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = 1.0f;
        viewport.height = 1.0f;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissors = {};
        scissors.offset = {0, 0};
        scissors.extent = {1, 1};

        VkPipelineViewportStateCreateInfo viewport_state_ci = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport_state_ci.viewportCount = 1;
        viewport_state_ci.pViewports = &viewport;
        viewport_state_ci.scissorCount = 1;
        viewport_state_ci.pScissors = &scissors;

        VkPipelineRasterizationStateCreateInfo rasterization_state_ci = {
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rasterization_state_ci.depthClampEnable = VK_FALSE;
        rasterization_state_ci.rasterizerDiscardEnable = VK_FALSE;
        rasterization_state_ci.polygonMode = g_poly_mode_vk[rast_state.poly.mode];
        rasterization_state_ci.cullMode = g_cull_modes_vk[rast_state.poly.cull];

        rasterization_state_ci.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization_state_ci.depthBiasEnable =
            (eDepthBiasMode(rast_state.poly.depth_bias_mode) != eDepthBiasMode::Disabled) ? VK_TRUE : VK_FALSE;
        rasterization_state_ci.depthBiasConstantFactor = rast_state.depth_bias.constant_offset;
        rasterization_state_ci.depthBiasClamp = 0.0f;
        rasterization_state_ci.depthBiasSlopeFactor = rast_state.depth_bias.slope_factor;
        rasterization_state_ci.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisample_state_ci = {
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample_state_ci.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisample_state_ci.sampleShadingEnable = VK_FALSE;
        multisample_state_ci.minSampleShading = 0;
        multisample_state_ci.pSampleMask = nullptr;
        multisample_state_ci.alphaToCoverageEnable = VK_FALSE;
        multisample_state_ci.alphaToOneEnable = VK_FALSE;

        VkStencilOpState stencil_state = {};
        stencil_state.failOp = g_stencil_op_vk[int(rast_state.stencil.stencil_fail)];
        stencil_state.passOp = g_stencil_op_vk[int(rast_state.stencil.pass)];
        stencil_state.depthFailOp = g_stencil_op_vk[int(rast_state.stencil.depth_fail)];
        stencil_state.compareOp = g_compare_op_vk[int(rast_state.stencil.compare_op)];
        stencil_state.compareMask = rast_state.stencil.compare_mask;
        stencil_state.writeMask = rast_state.stencil.write_mask;
        stencil_state.reference = rast_state.stencil.reference;

        VkPipelineDepthStencilStateCreateInfo depth_state_ci = {
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth_state_ci.depthTestEnable = rast_state.depth.test_enabled ? VK_TRUE : VK_FALSE;
        depth_state_ci.depthWriteEnable = rast_state.depth.write_enabled ? VK_TRUE : VK_FALSE;
        depth_state_ci.depthCompareOp = g_compare_op_vk[int(rast_state.depth.compare_op)];
        depth_state_ci.depthBoundsTestEnable = VK_FALSE;
        depth_state_ci.stencilTestEnable = rast_state.stencil.enabled ? VK_TRUE : VK_FALSE;
        depth_state_ci.front = stencil_state;
        depth_state_ci.back = stencil_state;
        depth_state_ci.minDepthBounds = 0.0f;
        depth_state_ci.maxDepthBounds = 1.0f;

        VkPipelineColorBlendAttachmentState color_blend_attachment_states[Ray::Vk::MaxRTAttachments] = {};
        for (int i = 0; i < int(color_attachments.size()); ++i) {
            color_blend_attachment_states[i].blendEnable = rast_state.blend.enabled ? VK_TRUE : VK_FALSE;
            color_blend_attachment_states[i].colorBlendOp = VK_BLEND_OP_ADD;
            color_blend_attachment_states[i].srcColorBlendFactor = g_blend_factor_vk[int(rast_state.blend.src)];
            color_blend_attachment_states[i].dstColorBlendFactor = g_blend_factor_vk[int(rast_state.blend.dst)];
            color_blend_attachment_states[i].alphaBlendOp = VK_BLEND_OP_ADD;
            color_blend_attachment_states[i].srcAlphaBlendFactor = g_blend_factor_vk[int(rast_state.blend.src)];
            color_blend_attachment_states[i].dstAlphaBlendFactor = g_blend_factor_vk[int(rast_state.blend.dst)];
            color_blend_attachment_states[i].colorWriteMask = 0xf;
        }

        VkPipelineColorBlendStateCreateInfo color_blend_state_ci = {
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        color_blend_state_ci.logicOpEnable = VK_FALSE;
        color_blend_state_ci.logicOp = VK_LOGIC_OP_CLEAR;
        color_blend_state_ci.attachmentCount = uint32_t(color_attachments.size());
        color_blend_state_ci.pAttachments = color_blend_attachment_states;
        color_blend_state_ci.blendConstants[0] = 0.0f;
        color_blend_state_ci.blendConstants[1] = 0.0f;
        color_blend_state_ci.blendConstants[2] = 0.0f;
        color_blend_state_ci.blendConstants[3] = 0.0f;

        SmallVector<VkDynamicState, 8> dynamic_states;
        dynamic_states.push_back(VK_DYNAMIC_STATE_VIEWPORT);
        dynamic_states.push_back(VK_DYNAMIC_STATE_SCISSOR);
        if (eDepthBiasMode(rast_state.poly.depth_bias_mode) == eDepthBiasMode::Dynamic) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
        }

        VkPipelineDynamicStateCreateInfo dynamic_state_ci = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic_state_ci.dynamicStateCount = uint32_t(dynamic_states.size());
        dynamic_state_ci.pDynamicStates = dynamic_states.cdata();

        VkGraphicsPipelineCreateInfo pipeline_create_info = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipeline_create_info.stageCount = uint32_t(shader_stage_create_info.size());
        pipeline_create_info.pStages = shader_stage_create_info.cdata();
        pipeline_create_info.pVertexInputState = &vtx_input_state_create_info;
        pipeline_create_info.pInputAssemblyState = &input_assembly_state_create_info;
        pipeline_create_info.pTessellationState = nullptr;
        pipeline_create_info.pViewportState = &viewport_state_ci;
        pipeline_create_info.pRasterizationState = &rasterization_state_ci;
        pipeline_create_info.pMultisampleState = &multisample_state_ci;
        pipeline_create_info.pDepthStencilState = &depth_state_ci;
        pipeline_create_info.pColorBlendState = &color_blend_state_ci;
        pipeline_create_info.pDynamicState = &dynamic_state_ci;
        pipeline_create_info.layout = layout_;
        pipeline_create_info.renderPass = render_pass ? render_pass->handle() : VK_NULL_HANDLE;
        pipeline_create_info.subpass = subpass_index;
        pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
        pipeline_create_info.basePipelineIndex = 0;

        VkPipelineRenderingCreateInfoKHR pipeline_rendering_create_info{
            VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};

        SmallVector<VkFormat, MaxRTAttachments> color_attachment_formats;
        if (!render_pass) {
            for (const auto &att : color_attachments) {
                color_formats_.push_back(att.format);
                color_attachment_formats.push_back(VKFormatFromTexFormat(att.format));
            }
            depth_format_ = depth_attachment.format;

            pipeline_rendering_create_info.colorAttachmentCount = int(color_attachment_formats.size());
            pipeline_rendering_create_info.pColorAttachmentFormats = color_attachment_formats.data();
            pipeline_rendering_create_info.depthAttachmentFormat = VKFormatFromTexFormat(depth_attachment.format);
            pipeline_rendering_create_info.stencilAttachmentFormat =
                rast_state.stencil.enabled ? VKFormatFromTexFormat(depth_attachment.format) : VK_FORMAT_UNDEFINED;

            pipeline_create_info.pNext = &pipeline_rendering_create_info;
        }

        const VkResult res =
            vkCreateGraphicsPipelines(ctx->device(), VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &handle_);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create graphics pipeline!");
            return false;
        }
    }*/

    ctx_ = ctx;
    type_ = ePipelineType::Graphics;
    rast_state_ = rast_state;
    render_pass_ = render_pass;
    prog_ = prog;
    vtx_input_ = vtx_input;

    return true;
}

bool Ray::Dx::Pipeline::Init(Context *ctx, const RastState &rast_state, Program *prog, const VertexInput *vtx_input,
                             const RenderPass *render_pass, uint32_t subpass_index, ILog *log) {
    return Init(ctx, rast_state, prog, vtx_input, render_pass, render_pass->color_rts, render_pass->depth_rt,
                subpass_index, log);
}

bool Ray::Dx::Pipeline::Init(Context *ctx, const RastState &rast_state, Program *prog, const VertexInput *vtx_input,
                             Span<const RenderTarget> color_attachments, RenderTarget depth_attachment,
                             uint32_t subpass_index, ILog *log) {

    SmallVector<RenderTargetInfo, MaxRTAttachments> color_infos;
    for (const RenderTarget &attachment : color_attachments) {
        color_infos.emplace_back(attachment);
    }

    return Init(ctx, rast_state, prog, vtx_input, nullptr, color_infos, RenderTargetInfo{depth_attachment},
                subpass_index, log);
}

bool Ray::Dx::Pipeline::Init(Context *ctx, Program *prog, ILog *log) {
    Destroy();

    ID3D12Device *device = prog->shader(eShaderType::Comp)->device();
    const std::vector<uint8_t> &shader_code = prog->shader(eShaderType::Comp)->shader_code();

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {};
    pso_desc.pRootSignature = prog->root_signature();
    pso_desc.CS = {shader_code.data(), shader_code.size()};

    HRESULT hr = device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&handle_));
    if (FAILED(hr)) {
        log->Error("Failed to create pipeline state!");
        return false;
    }

    ctx_ = ctx;
    type_ = ePipelineType::Compute;
    prog_ = prog;

    return true;
}