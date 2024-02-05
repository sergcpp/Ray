#include "RenderPassVK.h"

#include "../../Log.h"
#include "ContextVK.h"

#ifndef NDEBUG
#define VERBOSE_LOGGING
#endif

namespace Ray {
namespace Vk {
static_assert(int(eImageLayout::Undefined) == VK_IMAGE_LAYOUT_UNDEFINED, "!");
static_assert(int(eImageLayout::General) == VK_IMAGE_LAYOUT_GENERAL, "!");
static_assert(int(eImageLayout::ColorAttachmentOptimal) == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, "!");
static_assert(int(eImageLayout::DepthStencilAttachmentOptimal) == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
              "!");
static_assert(int(eImageLayout::DepthStencilReadOnlyOptimal) == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, "!");
static_assert(int(eImageLayout::ShaderReadOnlyOptimal) == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, "!");
static_assert(int(eImageLayout::TransferSrcOptimal) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, "!");
static_assert(int(eImageLayout::TransferDstOptimal) == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, "!");

extern const VkAttachmentLoadOp vk_load_ops[] = {
    VK_ATTACHMENT_LOAD_OP_LOAD,      // Load
    VK_ATTACHMENT_LOAD_OP_CLEAR,     // Clear
    VK_ATTACHMENT_LOAD_OP_DONT_CARE, // DontCare
    VK_ATTACHMENT_LOAD_OP_NONE_EXT   // None
};
static_assert((sizeof(vk_load_ops) / sizeof(vk_load_ops[0])) == int(eLoadOp::_Count), "!");

extern const VkAttachmentStoreOp vk_store_ops[] = {
    VK_ATTACHMENT_STORE_OP_STORE,     // Store
    VK_ATTACHMENT_STORE_OP_DONT_CARE, // DontCare
    VK_ATTACHMENT_STORE_OP_NONE_EXT   // None
};
static_assert((sizeof(vk_store_ops) / sizeof(vk_store_ops[0])) == int(eStoreOp::_Count), "!");

// make sure we can simply cast these
static_assert(VkSampleCountFlagBits::VK_SAMPLE_COUNT_1_BIT == 1, "!");
static_assert(VkSampleCountFlagBits::VK_SAMPLE_COUNT_2_BIT == 2, "!");
static_assert(VkSampleCountFlagBits::VK_SAMPLE_COUNT_4_BIT == 4, "!");
static_assert(VkSampleCountFlagBits::VK_SAMPLE_COUNT_8_BIT == 8, "!");

VkFormat ToSRGBFormat(VkFormat format);
} // namespace Vk
} // namespace Ray

Ray::Vk::RenderPass &Ray::Vk::RenderPass::operator=(RenderPass &&rhs) noexcept {
    if (this == &rhs) {
        return (*this);
    }

    Destroy();

    ctx_ = std::exchange(rhs.ctx_, nullptr);
    handle_ = std::exchange(rhs.handle_, {});
    color_rts = std::move(rhs.color_rts);
    depth_rt = std::exchange(rhs.depth_rt, {});

    return (*this);
}

bool Ray::Vk::RenderPass::Init(Context *ctx, Span<const RenderTargetInfo> _color_rts, RenderTargetInfo _depth_rt,
                               ILog *log) {
    Destroy();

    SmallVector<VkAttachmentDescription, MaxRTAttachments> pass_attachments;
    VkAttachmentReference color_attachment_refs[MaxRTAttachments];
    for (VkAttachmentReference &attachment_ref : color_attachment_refs) {
        attachment_ref = {VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED};
    }
    VkAttachmentReference depth_attachment_ref = {VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED};

    color_rts.resize(_color_rts.size());
    depth_rt = {};

    if (_depth_rt) {
        const auto att_index = uint32_t(pass_attachments.size());

        auto &att_desc = pass_attachments.emplace_back();
        att_desc.format = Ray::Vk::VKFormatFromTexFormat(_depth_rt.format);
        att_desc.samples = VkSampleCountFlagBits(_depth_rt.samples);
        att_desc.loadOp = vk_load_ops[int(_depth_rt.load)];
        att_desc.storeOp = vk_store_ops[int(_depth_rt.store)];
        att_desc.stencilLoadOp = vk_load_ops[int(_depth_rt.stencil_load)];
        att_desc.stencilStoreOp = vk_store_ops[int(_depth_rt.stencil_store)];
        att_desc.initialLayout = VkImageLayout(_depth_rt.layout);
        att_desc.finalLayout = att_desc.initialLayout;

        depth_attachment_ref.attachment = att_index;
        depth_attachment_ref.layout = att_desc.initialLayout;

        depth_rt = _depth_rt;
    }

    for (int i = 0; i < _color_rts.size(); ++i) {
        if (!_color_rts[i]) {
            continue;
        }

        const auto att_index = uint32_t(pass_attachments.size());

        auto &att_desc = pass_attachments.emplace_back();
        att_desc.format = VKFormatFromTexFormat(_color_rts[i].format);
        if (bool(_color_rts[i].flags & eTexFlagBits::SRGB)) {
            att_desc.format = ToSRGBFormat(att_desc.format);
        }
        att_desc.samples = VkSampleCountFlagBits(_color_rts[i].samples);
        if (VkImageLayout(_color_rts[i].layout) == VK_IMAGE_LAYOUT_UNDEFINED) {
            att_desc.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            att_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        } else {
            att_desc.loadOp = vk_load_ops[int(_color_rts[i].load)];
            att_desc.stencilLoadOp = vk_load_ops[int(_color_rts[i].load)];
        }
        att_desc.storeOp = vk_store_ops[int(_color_rts[i].store)];
        att_desc.stencilStoreOp = vk_store_ops[int(_color_rts[i].store)];
        att_desc.initialLayout = VkImageLayout(_color_rts[i].layout);
        att_desc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        color_attachment_refs[i].attachment = att_index;
        color_attachment_refs[i].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        color_rts[i] = _color_rts[i];
    }

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = uint32_t(_color_rts.size());
    subpass.pColorAttachments = color_attachment_refs;
    if (depth_attachment_ref.attachment != VK_ATTACHMENT_UNUSED) {
        subpass.pDepthStencilAttachment = &depth_attachment_ref;
    }

    VkRenderPassCreateInfo render_pass_create_info = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    render_pass_create_info.attachmentCount = uint32_t(pass_attachments.size());
    render_pass_create_info.pAttachments = pass_attachments.data();
    render_pass_create_info.subpassCount = 1;
    render_pass_create_info.pSubpasses = &subpass;

    const VkResult res = ctx->api().vkCreateRenderPass(ctx->device(), &render_pass_create_info, nullptr, &handle_);
    if (res != VK_SUCCESS) {
        log->Error("Failed to create render pass!");
        return false;
#ifdef VERBOSE_LOGGING
    } else {
        log->Info("RenderPass %p created", handle_);
#endif
    }

    ctx_ = ctx;
    return true;
}

void Ray::Vk::RenderPass::Destroy() {
    if (handle_ != VK_NULL_HANDLE) {
        ctx_->render_passes_to_destroy[ctx_->backend_frame].push_back(handle_);
        handle_ = VK_NULL_HANDLE;
    }
    color_rts.clear();
    depth_rt = {};
}

bool Ray::Vk::RenderPass::IsCompatibleWith(Span<const RenderTarget> _color_rts, RenderTarget _depth_rt) {
    if (_color_rts.size() != color_rts.size() || bool(_depth_rt) != bool(depth_rt)) {
        return false;
    }

    for (int i = 0; i < _color_rts.size(); ++i) {
        if (color_rts[i] != _color_rts[i]) {
            return false;
        }
    }

    if (depth_rt) {
        if (depth_rt != _depth_rt) {
            return false;
        }
    }

    return true;
}

bool Ray::Vk::RenderPass::Setup(Context *ctx, Span<const RenderTarget> _color_rts, const RenderTarget _depth_rt,
                                ILog *log) {
    if (_color_rts.size() == color_rts.size() &&
        std::equal(
            _color_rts.begin(), _color_rts.end(), color_rts.data(),
            [](const RenderTarget &rt, const RenderTargetInfo &i) { return (!rt.ref && !i) || (rt.ref && i == rt); }) &&
        ((!_depth_rt.ref && !depth_rt) || (_depth_rt.ref && depth_rt == _depth_rt))) {
        return true;
    }

    SmallVector<RenderTargetInfo, MaxRTAttachments> infos;
    for (const auto &color_rt : _color_rts) {
        infos.emplace_back(color_rt);
    }

    return Init(ctx, infos, RenderTargetInfo{_depth_rt}, log);
}

bool Ray::Vk::RenderPass::Setup(Context *ctx, Span<const RenderTargetInfo> _color_rts, RenderTargetInfo _depth_rt,
                                ILog *log) {
    if (_color_rts.size() == color_rts.size() && std::equal(_color_rts.begin(), _color_rts.end(), color_rts.data()) &&
        ((!_depth_rt && !depth_rt) || (_depth_rt && depth_rt == _depth_rt))) {
        return true;
    }
    return Init(ctx, _color_rts, _depth_rt, log);
}

#undef VERBOSE_LOGGING
