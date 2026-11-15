#include "ResourceMtl.h"

#include <cassert>

#include "BufferMtl.h"
#include "ContextMtl.h"
#include "ImageAtlasMtl.h"
#include "ImageMtl.h"

namespace Ray::Mtl {
Bitmask<eStage> StagesForState(const eResState state) { return AllStages; }

void TransitionResourceStates(CommandBuffer cmd_buf, Bitmask<eStage>, Bitmask<eStage>,
                              Span<const TransitionInfo> transitions) {
    // the shared renderer brackets every dependent dispatch with transitions (VK pipeline
    // barriers); Metal B2+ may overlap dispatches in one compute encoder, so mirror each
    // call with an explicit memory barrier on the open encoder
    if (Context *ctx = Context::LookupForCommandBuffer(cmd_buf)) {
        if (MTL::ComputeCommandEncoder *enc = ctx->openComputeEncoder()) {
            enc->memoryBarrier(MTL::BarrierScopeBuffers | MTL::BarrierScopeTextures);
        }
    }
    for (const TransitionInfo &transition : transitions) {
        if (!transition.update_internal_state) {
            continue;
        }
        // record advisory state for all three resource kinds (ClearColorImage & friends assert on it)
        if (std::holds_alternative<const Buffer *>(transition.p_res)) {
            if (const Buffer *res = std::get<const Buffer *>(transition.p_res)) {
                res->resource_state = transition.new_state;
            }
        } else if (std::holds_alternative<const Image *>(transition.p_res)) {
            if (const Image *res = std::get<const Image *>(transition.p_res)) {
                res->resource_state = transition.new_state;
            }
        } else if (std::holds_alternative<const ImageAtlas *>(transition.p_res)) {
            if (const ImageAtlas *res = std::get<const ImageAtlas *>(transition.p_res)) {
                res->resource_state = transition.new_state;
            }
        }
    }
}
} // namespace Ray::Mtl
