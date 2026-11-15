#include "DrawCallMtl.h"

#include <cassert>
#include <cstring>
#include <set>

#include "../../Log.h"
#include "BufferMtl.h"
#include "ContextMtl.h"
#include "ImageAtlasMtl.h"
#include "ImageMtl.h"
#include "PipelineMtl.h"
#include "SamplerMtl.h"

namespace Ray::Mtl {
static void BindUniform(Context &ctx, MTL::ComputeCommandEncoder *enc, const Pipeline &pi, const void *uniform_data,
                        const int uniform_data_len, ILog *) {
    if (uniform_data_len <= 0) {
        return;
    }
    const MslBindingDesc *pc = pi.pushconst();
    assert(pc);
    MTL::Buffer *ring = nullptr;
    uint32_t offset = 0;
    if ((ring = ctx.UniformSlice(uint32_t(uniform_data_len), offset)) == nullptr) {
        assert(!"uniform params ring overflow - dispatch would run with stale params");
        return;
    }
    memcpy(static_cast<uint8_t *>(ring->contents()) + offset, uniform_data, size_t(uniform_data_len));
    enc->setBuffer(ring, offset, pc->msl_index);
}

static void BindAll(MTL::ComputeCommandEncoder *enc, const Pipeline &pi, Span<const Binding> bindings, ILog *log) {
    for (const Binding &b : bindings) {
        // standalone GLSL samplers only ever get an msl_k_sampler refl entry
        const MslBindingDesc *d = b.trg == eBindTarget::Sampler ? pi.find_sampler(b.loc) : pi.find(b.loc);
        if (d == nullptr) {
            // shared kernel bind lists include slots variants dead-strip (VK/DX ignore those silently) -
            // warn once per (pipeline, binding), dispatches would otherwise spam every bounce
            static thread_local std::set<std::pair<const void *, int>> warned_missing;
            if (warned_missing.emplace(&pi, int(b.loc)).second) {
                log->Warning("Metal pipeline %s has no refl entry for binding %d", pi.prog()->name().c_str(),
                             int(b.loc));
            }
            continue;
        }

        switch (b.trg) {
        case eBindTarget::UBuf:
        case eBindTarget::SBufRO:
        case eBindTarget::SBufRW: {
            assert(b.handle.buf);
            enc->setBuffer(b.handle.buf->mtl(), b.offset, static_cast<NS::UInteger>(d->msl_index));
        } break;
        case eBindTarget::Sampler: {
            assert(b.handle.sampler);
            if (!b.handle.sampler->mtl()) {
                log->Warning("Metal: nil sampler state at binding %d in %s", int(b.loc), pi.prog()->name().c_str());
                enc->setSamplerState(pi.ctx()->dummy_sampler(), static_cast<NS::UInteger>(d->msl_index));
            } else {
                enc->setSamplerState(b.handle.sampler->mtl(), static_cast<NS::UInteger>(d->msl_index));
            }
        } break;
        case eBindTarget::Tex:
        case eBindTarget::TexSampled:
        case eBindTarget::Image: {
            assert(b.handle.tex);
            enc->setTexture(b.handle.tex->mtl(), static_cast<NS::UInteger>(d->msl_index));
            // every Image carries its own Sampler; MSL splits it into a companion arg
            const MslBindingDesc *sd = pi.find_sampler(b.loc);
            if (sd) {
                if (b.handle.tex->sampler() && b.handle.tex->sampler()->mtl()) {
                    enc->setSamplerState(b.handle.tex->sampler()->mtl(), static_cast<NS::UInteger>(sd->msl_index));
                } else
                    log->Warning("Metal: nil sampler for binding %d in %s", int(b.loc), pi.prog()->name().c_str());
            }
        } break;
        // atlas mode: refl emits a count-9 tex entry + companion sampler array at the same loc
        case eBindTarget::TexArray:
        case eBindTarget::TexArraySampled: {
            assert(b.handle.tex_arr);
            const MslBindingDesc *sd = pi.find_sampler(b.loc);
            for (int i = 0; i < b.handle.count; ++i) {
                const ImageAtlas &atlas = b.handle.tex_arr[i];
                enc->setTexture(atlas.mtl(), static_cast<NS::UInteger>(d->msl_index) + i);
                if (sd && atlas.sampler()) {
                    enc->setSamplerState(atlas.sampler()->mtl(), static_cast<NS::UInteger>(sd->msl_index) + i);
                }
            }
        } break;
        // bindless arrays & RT targets not supported by this backend yet
        case eBindTarget::AccStruct:
        case eBindTarget::DescrTable:
            assert(false && "RT targets require HWRT support");
            break;
        default:
            break;
        }
    }
}

void DispatchCompute(CommandBuffer cmd_buf, const Pipeline &comp_pipeline, const uint32_t grp_count[3],
                     Span<const Binding> bindings, const void *uniform_data, const int uniform_data_len,
                     DescrMultiPoolAlloc *, ILog *log) {
    Context *ctx = comp_pipeline.ctx();
    MTL::ComputeCommandEncoder *enc = ctx->ComputeEncoder(cmd_buf);

    enc->setComputePipelineState(comp_pipeline.handle());
    BindUniform(*ctx, enc, comp_pipeline, uniform_data, uniform_data_len, log);
    BindAll(enc, comp_pipeline, bindings, log);

    const uint32_t *wg = comp_pipeline.workgroup_size();
    enc->dispatchThreadgroups(MTL::Size::Make(grp_count[0], grp_count[1], grp_count[2]),
                              MTL::Size::Make(wg[0], wg[1], wg[2]));
}

void DispatchComputeIndirect(CommandBuffer cmd_buf, const Pipeline &comp_pipeline, const Buffer &indir_buf,
                             const uint32_t indir_buf_offset, Span<const Binding> bindings, const void *uniform_data,
                             const int uniform_data_len, DescrMultiPoolAlloc *, ILog *log) {
    Context *ctx = comp_pipeline.ctx();
    MTL::ComputeCommandEncoder *enc = ctx->ComputeEncoder(cmd_buf);

    enc->setComputePipelineState(comp_pipeline.handle());
    BindUniform(*ctx, enc, comp_pipeline, uniform_data, uniform_data_len, log);
    BindAll(enc, comp_pipeline, bindings, log);

    // indir args are written by prepare_indir_args with the VK layout - structs are byte-identical
    const uint32_t *wg = comp_pipeline.workgroup_size();
    enc->dispatchThreadgroups(indir_buf.mtl(), indir_buf_offset, MTL::Size::Make(wg[0], wg[1], wg[2]));
}
} // namespace Ray::Mtl
