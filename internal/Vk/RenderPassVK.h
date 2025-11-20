#pragma once

#include "../../Span.h"
#include "ImageVK.h"

namespace Ray::Vk {
class Context;

const int MaxRTAttachments = 4;

enum class eImageLayout : uint8_t {
    Undefined,
    General,
    ColorAttachmentOptimal,
    DepthStencilAttachmentOptimal,
    DepthStencilReadOnlyOptimal,
    ShaderReadOnlyOptimal,
    TransferSrcOptimal,
    TransferDstOptimal,
    _Count
};

enum class eLoadOp : uint8_t { Load, Clear, DontCare, None, _Count };
enum class eStoreOp : uint8_t { Store, DontCare, None, _Count };

struct RenderTarget {
    Image *ref = nullptr;
    uint8_t view_index = 0;
    eLoadOp load = eLoadOp::DontCare;
    eStoreOp store = eStoreOp::DontCare;
    eLoadOp stencil_load = eLoadOp::DontCare;
    eStoreOp stencil_store = eStoreOp::DontCare;

    RenderTarget() = default;
    RenderTarget(Image *_ref, eLoadOp _load, eStoreOp _store, eLoadOp _stencil_load = eLoadOp::DontCare,
                 eStoreOp _stencil_store = eStoreOp::DontCare)
        : ref(_ref), load(_load), store(_store), stencil_load(_stencil_load), stencil_store(_stencil_store) {}
    RenderTarget(Image *_ref, uint8_t _view_index, eLoadOp _load, eStoreOp _store,
                 eLoadOp _stencil_load = eLoadOp::DontCare, eStoreOp _stencil_store = eStoreOp::DontCare)
        : ref(_ref), view_index(_view_index), load(_load), store(_store), stencil_load(_stencil_load),
          stencil_store(_stencil_store) {}

    operator bool() const { return bool(ref); }
};

inline bool operator==(const RenderTarget &lhs, const RenderTarget &rhs) {
    return lhs.ref == rhs.ref && lhs.view_index == rhs.view_index && lhs.load == rhs.load && lhs.store == rhs.store &&
           lhs.stencil_load == rhs.stencil_load && lhs.stencil_store == rhs.stencil_store;
}

struct RenderTargetInfo {
    eFormat format = eFormat::Undefined;
    uint8_t samples = 1;
    Bitmask<eImgFlags> flags;
    eImageLayout layout = eImageLayout::Undefined;
    eLoadOp load = eLoadOp::DontCare;
    eStoreOp store = eStoreOp::DontCare;
    eLoadOp stencil_load = eLoadOp::DontCare;
    eStoreOp stencil_store = eStoreOp::DontCare;

    RenderTargetInfo() = default;
    RenderTargetInfo(Image *_ref, eLoadOp _load, eStoreOp _store, eLoadOp _stencil_load = eLoadOp::DontCare,
                     eStoreOp _stencil_store = eStoreOp::DontCare)
        : format(_ref->params.format), samples(_ref->params.samples), flags(_ref->params.flags),
          layout(eImageLayout(VKImageLayoutForState(_ref->resource_state))), load(_load), store(_store),
          stencil_load(_stencil_load), stencil_store(_stencil_store) {}
    RenderTargetInfo(const Image *tex, eLoadOp _load, eStoreOp _store, eLoadOp _stencil_load = eLoadOp::DontCare,
                     eStoreOp _stencil_store = eStoreOp::DontCare)
        : format(tex->params.format), samples(tex->params.samples), flags(tex->params.flags),
          layout(eImageLayout(VKImageLayoutForState(tex->resource_state))), load(_load), store(_store),
          stencil_load(_stencil_load), stencil_store(_stencil_store) {}
    RenderTargetInfo(eFormat _format, uint8_t _samples, eImageLayout _layout, eLoadOp _load, eStoreOp _store,
                     eLoadOp _stencil_load = eLoadOp::DontCare, eStoreOp _stencil_store = eStoreOp::DontCare)
        : format(_format), samples(_samples), layout(_layout), load(_load), store(_store), stencil_load(_stencil_load),
          stencil_store(_stencil_store) {}
    explicit RenderTargetInfo(const RenderTarget &rt) {
        if (rt.ref) {
            format = rt.ref->params.format;
            samples = rt.ref->params.samples;
            flags = Bitmask<eImgFlags>{rt.ref->params.flags};
            layout = eImageLayout(VKImageLayoutForState(rt.ref->resource_state));
            load = rt.load;
            store = rt.store;
            stencil_load = rt.stencil_load;
            stencil_store = rt.stencil_store;
        }
    }

    operator bool() const { return format != eFormat::Undefined; }
};

inline bool operator==(const RenderTargetInfo &lhs, const RenderTarget &rhs) {
    const auto &p = rhs.ref->params;
    return lhs.format == p.format && lhs.samples == p.samples &&
           lhs.layout == eImageLayout(VKImageLayoutForState(rhs.ref->resource_state)) && lhs.load == rhs.load &&
           lhs.store == rhs.store && lhs.stencil_load == rhs.stencil_load && lhs.stencil_store == rhs.stencil_store;
}

inline bool operator!=(const RenderTargetInfo &lhs, const RenderTarget &rhs) { return !operator==(lhs, rhs); }

class RenderPass {
    Context *ctx_ = nullptr;
    VkRenderPass handle_ = VK_NULL_HANDLE;

    bool Init(Context *ctx, Span<const RenderTargetInfo> rts, RenderTargetInfo depth_rt, ILog *log);
    void Destroy();

  public:
    SmallVector<RenderTargetInfo, MaxRTAttachments> color_rts;
    RenderTargetInfo depth_rt;

    RenderPass() = default;
    RenderPass(const RenderPass &rhs) = delete;
    RenderPass(RenderPass &&rhs) noexcept { (*this) = std::move(rhs); }
    ~RenderPass() { Destroy(); }

    RenderPass &operator=(const RenderPass &rhs) = delete;
    RenderPass &operator=(RenderPass &&rhs) noexcept;

    VkRenderPass handle() const { return handle_; }

    bool IsCompatibleWith(Span<const RenderTarget> color_rts, RenderTarget depth_rt);

    bool Setup(Context *ctx, Span<const RenderTarget> rts, RenderTarget depth_rt, ILog *log);
    bool Setup(Context *ctx, Span<const RenderTargetInfo> rts, RenderTargetInfo depth_rt, ILog *log);
};
} // namespace Ray::Vk