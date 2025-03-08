#include "PipelineVK.h"

#include "../../Log.h"
#include "../RastState.h"
#include "ContextVK.h"
#include "ProgramVK.h"
#include "RenderPassVK.h"
#include "VertexInputVK.h"

namespace Ray {
namespace Vk {
extern const VkShaderStageFlagBits g_shader_stages_vk[];

const VkCullModeFlagBits g_cull_modes_vk[] = {
    VK_CULL_MODE_NONE,      // None
    VK_CULL_MODE_FRONT_BIT, // Front
    VK_CULL_MODE_BACK_BIT   // Back
};
static_assert(COUNT_OF(g_cull_modes_vk) == int(eCullFace::_Count), "!");

const VkCompareOp g_compare_op_vk[] = {
    VK_COMPARE_OP_ALWAYS,          // Always
    VK_COMPARE_OP_NEVER,           // Never
    VK_COMPARE_OP_LESS,            // Less
    VK_COMPARE_OP_EQUAL,           // Equal
    VK_COMPARE_OP_GREATER,         // Greater
    VK_COMPARE_OP_LESS_OR_EQUAL,   // LEqual
    VK_COMPARE_OP_NOT_EQUAL,       // NotEqual
    VK_COMPARE_OP_GREATER_OR_EQUAL // GEqual
};
static_assert(COUNT_OF(g_compare_op_vk) == int(eCompareOp::_Count), "!");

const VkStencilOp g_stencil_op_vk[] = {
    VK_STENCIL_OP_KEEP,                // Keep
    VK_STENCIL_OP_ZERO,                // Zero
    VK_STENCIL_OP_REPLACE,             // Replace
    VK_STENCIL_OP_INCREMENT_AND_CLAMP, // Incr
    VK_STENCIL_OP_DECREMENT_AND_CLAMP, // Decr
    VK_STENCIL_OP_INVERT               // Invert
};
static_assert(COUNT_OF(g_stencil_op_vk) == int(eStencilOp::_Count), "!");

const VkPolygonMode g_poly_mode_vk[] = {
    VK_POLYGON_MODE_FILL, // Fill
    VK_POLYGON_MODE_LINE  // Line
};
static_assert(COUNT_OF(g_poly_mode_vk) == int(ePolygonMode::_Count), "!");

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
static_assert(COUNT_OF(g_blend_factor_vk) == int(eBlendFactor::_Count), "!");

uint32_t align_up(const uint32_t size, const uint32_t alignment) { return (size + alignment - 1) & ~(alignment - 1); }

static_assert(sizeof(TraceRaysIndirectCommand) == sizeof(VkTraceRaysIndirectCommandKHR), "!");
} // namespace Vk
} // namespace Ray

Ray::Vk::Pipeline &Ray::Vk::Pipeline::operator=(Pipeline &&rhs) noexcept {
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
    layout_ = std::exchange(rhs.layout_, {});
    handle_ = std::exchange(rhs.handle_, {});

    rt_shader_groups_ = std::move(rhs.rt_shader_groups_);

    rgen_region_ = std::exchange(rhs.rgen_region_, {});
    miss_region_ = std::exchange(rhs.miss_region_, {});
    hit_region_ = std::exchange(rhs.hit_region_, {});
    call_region_ = std::exchange(rhs.call_region_, {});

    rt_sbt_buf_ = std::move(rhs.rt_sbt_buf_);

    return (*this);
}

Ray::Vk::Pipeline::~Pipeline() { Destroy(); }

void Ray::Vk::Pipeline::Destroy() {
    if (layout_ != VK_NULL_HANDLE) {
        ctx_->pipeline_layouts_to_destroy[ctx_->backend_frame].emplace_back(layout_);
        layout_ = VK_NULL_HANDLE;
    }
    if (handle_ != VK_NULL_HANDLE) {
        ctx_->pipelines_to_destroy[ctx_->backend_frame].emplace_back(handle_);
        handle_ = VK_NULL_HANDLE;
    }

    color_formats_.clear();
    depth_format_ = eTexFormat::Undefined;

    rt_shader_groups_.clear();

    rgen_region_ = {};
    miss_region_ = {};
    hit_region_ = {};
    call_region_ = {};

    rt_sbt_buf_ = {};
}

bool Ray::Vk::Pipeline::Init(Context *ctx, const RastState &rast_state, Program *prog, const VertexInput *vtx_input,
                             const RenderPass *render_pass, Span<const RenderTargetInfo> color_attachments,
                             RenderTargetInfo depth_attachment, uint32_t subpass_index, ILog *log) {
    Destroy();

    SmallVector<VkPipelineShaderStageCreateInfo, int(eShaderType::_Count)> shader_stage_create_info;
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

        const VkResult res = ctx->api().vkCreatePipelineLayout(ctx->device(), &layout_create_info, nullptr, &layout_);
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
        vtx_input_state_create_info.vertexBindingDescriptionCount = bindings.size();
        vtx_input_state_create_info.pVertexBindingDescriptions = bindings.cdata();
        vtx_input_state_create_info.vertexAttributeDescriptionCount = attribs.size();
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
        dynamic_state_ci.dynamicStateCount = dynamic_states.size();
        dynamic_state_ci.pDynamicStates = dynamic_states.cdata();

        VkGraphicsPipelineCreateInfo pipeline_create_info = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipeline_create_info.stageCount = shader_stage_create_info.size();
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

        const VkResult res = ctx->api().vkCreateGraphicsPipelines(ctx->device(), VK_NULL_HANDLE, 1,
                                                                  &pipeline_create_info, nullptr, &handle_);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create graphics pipeline!");
            return false;
        }
    }

    ctx_ = ctx;
    type_ = ePipelineType::Graphics;
    rast_state_ = rast_state;
    render_pass_ = render_pass;
    prog_ = prog;
    vtx_input_ = vtx_input;

    return true;
}

bool Ray::Vk::Pipeline::Init(Context *ctx, const RastState &rast_state, Program *prog, const VertexInput *vtx_input,
                             const RenderPass *render_pass, uint32_t subpass_index, ILog *log) {
    return Init(ctx, rast_state, prog, vtx_input, render_pass, render_pass->color_rts, render_pass->depth_rt,
                subpass_index, log);
}

bool Ray::Vk::Pipeline::Init(Context *ctx, const RastState &rast_state, Program *prog, const VertexInput *vtx_input,
                             Span<const RenderTarget> color_attachments, RenderTarget depth_attachment,
                             uint32_t subpass_index, ILog *log) {

    SmallVector<RenderTargetInfo, MaxRTAttachments> color_infos;
    for (const RenderTarget &attachment : color_attachments) {
        color_infos.emplace_back(attachment);
    }

    return Init(ctx, rast_state, prog, vtx_input, nullptr, color_infos, RenderTargetInfo{depth_attachment},
                subpass_index, log);
}

bool Ray::Vk::Pipeline::Init(Context *ctx, Program *prog, ILog *log) {
    Destroy();

    ePipelineType type = ePipelineType::Undefined;

    SmallVector<VkPipelineShaderStageCreateInfo, int(eShaderType::_Count)> shader_stage_create_info;
    int hit_group_index = -1;
    for (int i = 0; i < int(eShaderType::_Count); ++i) {
        const Shader *sh = prog->shader(eShaderType(i));
        if (!sh) {
            continue;
        }

        if (eShaderType(i) == eShaderType::Comp) {
            assert(type == ePipelineType::Undefined);
            type = ePipelineType::Compute;
        } else if (eShaderType(i) == eShaderType::RayGen) {
            assert(type == ePipelineType::Undefined);
            type = ePipelineType::Raytracing;

            auto &new_group = rt_shader_groups_.emplace_back();
            new_group = {VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
            new_group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
            new_group.generalShader = shader_stage_create_info.size();
            new_group.anyHitShader = VK_SHADER_UNUSED_KHR;
            new_group.closestHitShader = VK_SHADER_UNUSED_KHR;
            new_group.intersectionShader = VK_SHADER_UNUSED_KHR;
        } else if (eShaderType(i) == eShaderType::Miss) {
            auto &new_group = rt_shader_groups_.emplace_back();
            new_group = {VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
            new_group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
            new_group.generalShader = shader_stage_create_info.size();
            new_group.anyHitShader = VK_SHADER_UNUSED_KHR;
            new_group.closestHitShader = VK_SHADER_UNUSED_KHR;
            new_group.intersectionShader = VK_SHADER_UNUSED_KHR;
        } else if (eShaderType(i) == eShaderType::ClosestHit) {
            VkRayTracingShaderGroupCreateInfoKHR *hit_group = nullptr;
            if (hit_group_index == -1) {
                hit_group_index = int(rt_shader_groups_.size());
                hit_group = &rt_shader_groups_.emplace_back();
                (*hit_group) = {VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
                hit_group->type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
                hit_group->generalShader = VK_SHADER_UNUSED_KHR;
                hit_group->anyHitShader = VK_SHADER_UNUSED_KHR;
                hit_group->closestHitShader = VK_SHADER_UNUSED_KHR;
                hit_group->intersectionShader = VK_SHADER_UNUSED_KHR;
            } else {
                hit_group = &rt_shader_groups_[hit_group_index];
            }
            hit_group->closestHitShader = shader_stage_create_info.size();
        } else if (eShaderType(i) == eShaderType::AnyHit) {
            VkRayTracingShaderGroupCreateInfoKHR *hit_group = nullptr;
            if (hit_group_index == -1) {
                hit_group_index = int(rt_shader_groups_.size());
                hit_group = &rt_shader_groups_.emplace_back();
                (*hit_group) = {VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
                hit_group->type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
                hit_group->generalShader = VK_SHADER_UNUSED_KHR;
                hit_group->anyHitShader = VK_SHADER_UNUSED_KHR;
                hit_group->closestHitShader = VK_SHADER_UNUSED_KHR;
                hit_group->intersectionShader = VK_SHADER_UNUSED_KHR;
            } else {
                hit_group = &rt_shader_groups_[hit_group_index];
            }
            hit_group->anyHitShader = shader_stage_create_info.size();
        }

        auto &stage_info = shader_stage_create_info.emplace_back();
        stage_info = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage_info.stage = g_shader_stages_vk[i];
        stage_info.module = prog->shader(eShaderType(i))->module();
        stage_info.pName = "main";
        stage_info.pSpecializationInfo = nullptr;
    }

    if (type == ePipelineType::Undefined) {
        return false;
    }

    ctx_ = ctx;
    type_ = type;
    prog_ = prog;

    { // create pipeline layout
        VkPipelineLayoutCreateInfo layout_create_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_create_info.setLayoutCount = prog->descr_set_layouts_count();
        layout_create_info.pSetLayouts = prog->descr_set_layouts();
        layout_create_info.pushConstantRangeCount = prog->pc_range_count();
        layout_create_info.pPushConstantRanges = prog->pc_ranges();

        const VkResult res = ctx->api().vkCreatePipelineLayout(ctx->device(), &layout_create_info, nullptr, &layout_);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create pipeline layout!");
            return false;
        }
    }

    if (type == ePipelineType::Compute) {
        VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = shader_stage_create_info[0];
        info.layout = layout_;

        const VkResult res =
            ctx->api().vkCreateComputePipelines(ctx->device(), VK_NULL_HANDLE, 1, &info, nullptr, &handle_);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create pipeline!");
            return false;
        }
    } else /* if (type == ePipelineType::Raytracing) */ {
        VkRayTracingPipelineCreateInfoKHR info = {VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
        info.pStages = shader_stage_create_info.cdata();
        info.stageCount = shader_stage_create_info.size();
        info.layout = layout_;
        info.maxPipelineRayRecursionDepth = 1;
        info.groupCount = rt_shader_groups_.size();
        info.pGroups = rt_shader_groups_.cdata();

        const VkResult res = ctx->api().vkCreateRayTracingPipelinesKHR(ctx->device(), VK_NULL_HANDLE, VK_NULL_HANDLE, 1,
                                                                       &info, nullptr, &handle_);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create pipeline!");
            return false;
        }

        { // create shader binding table
            const int RgenCount = 1;
            const int MissCount = 1;
            const int HitCount = rt_shader_groups_.size() == 4 ? 2 : 1;

            const int HandleCount = RgenCount + MissCount + HitCount;

            const uint32_t handle_size = ctx->rt_props().shaderGroupHandleSize;
            const uint32_t handle_size_aligned = align_up(handle_size, ctx->rt_props().shaderGroupHandleAlignment);

            rgen_region_.stride = align_up(handle_size_aligned, ctx->rt_props().shaderGroupBaseAlignment);
            rgen_region_.size = rgen_region_.stride;
            miss_region_.stride = handle_size_aligned;
            miss_region_.size = align_up(MissCount * handle_size_aligned, ctx->rt_props().shaderGroupBaseAlignment);
            hit_region_.stride = handle_size_aligned;
            hit_region_.size = align_up(HitCount * handle_size_aligned, ctx->rt_props().shaderGroupBaseAlignment);

            const uint32_t data_size = HandleCount * handle_size;
            SmallVector<uint8_t, 128> handles_data(data_size);

            const VkResult res = ctx->api().vkGetRayTracingShaderGroupHandlesKHR(ctx->device(), handle_, 0, HandleCount,
                                                                                 data_size, &handles_data[0]);
            if (res != VK_SUCCESS) {
                log->Error("Failed to get shader group handles!");
                return false;
            }

            const VkDeviceSize sbt_size = rgen_region_.size + miss_region_.size + hit_region_.size;

            rt_sbt_buf_ = Buffer("SBT Buffer", ctx, eBufType::ShaderBinding, uint32_t(sbt_size));
            Buffer sbt_stage_buf = Buffer("SBT Staging Buffer", ctx, eBufType::Upload, uint32_t(sbt_size));

            const VkDeviceAddress sbt_address = rt_sbt_buf_.vk_device_address();
            rgen_region_.deviceAddress = sbt_address;
            miss_region_.deviceAddress = sbt_address + rgen_region_.size;
            hit_region_.deviceAddress = sbt_address + rgen_region_.size + miss_region_.size;

            { // Init staging buffer
                uint8_t *p_sbt_stage = sbt_stage_buf.Map();
                uint8_t *p_dst = p_sbt_stage;
                int handle_ndx = 0;
                // Copy raygen
                memcpy(p_dst, handles_data.cdata() + (handle_ndx++) * handle_size, handle_size);
                p_dst = p_sbt_stage + rgen_region_.size;
                // Copy miss
                for (int i = 0; i < MissCount; ++i) {
                    memcpy(p_dst, handles_data.cdata() + (handle_ndx++) * handle_size, handle_size);
                    p_dst += miss_region_.stride;
                }
                p_dst = p_sbt_stage + rgen_region_.size + miss_region_.size;
                // Copy hit
                for (int i = 0; i < HitCount; ++i) {
                    uint32_t off = (handle_ndx++) * handle_size;
                    uint8_t debug_value[32] = {};
                    memcpy(&debug_value, handles_data.cdata() + off, handle_size);
                    memcpy(p_dst, handles_data.cdata() + off, handle_size);
                    p_dst += hit_region_.stride;
                }
                // p_dst = p_sbt_stage + rgen_region_.size + miss_region_.size + hit_region_.size;

                sbt_stage_buf.Unmap();
            }

            { // Copy data
                VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx->api(), ctx->device(), ctx->temp_command_pool());

                VkBufferCopy region_to_copy = {};
                region_to_copy.srcOffset = VkDeviceSize{0};
                region_to_copy.dstOffset = VkDeviceSize{0};
                region_to_copy.size = VkDeviceSize{sbt_stage_buf.size()};

                ctx->api().vkCmdCopyBuffer(cmd_buf, sbt_stage_buf.vk_handle(), rt_sbt_buf_.vk_handle(), 1,
                                           &region_to_copy);

                sbt_stage_buf.resource_state = eResState::CopySrc;
                rt_sbt_buf_.resource_state = eResState::CopyDst;

                EndSingleTimeCommands(ctx->api(), ctx->device(), ctx->graphics_queue(), cmd_buf,
                                      ctx->temp_command_pool());
            }
        }
    }

    return true;
}