#include "ImageMtl.h"

#include <cassert>
#include <cstring>
#include <utility>

#include "../../Config.h"
#include "../../Log.h"
#include "../TextureUtils.h"
#include "ContextMtl.h"

#ifndef NDEBUG
// #define TEX_VERBOSE_LOGGING
#endif

namespace Ray::Mtl {
extern const MTL::PixelFormat g_formats_mtl[]; // referenced by ImageAtlasMtl.cpp: keep external linkage

#define X(_0, _1, _2, _3, _4, _5, _6, _7) _7,
const MTL::PixelFormat g_formats_mtl[] = {
#include "../Format.inl"
};
#undef X
static_assert(std::size(g_formats_mtl) == size_t(eFormat::_Count), "!");

int TextureHandleCounter = 0;

// same per-state usage table as VK: kept for the shared kernels' ImgUsageFromState() calls
const eImgUsage g_img_usage_per_state[] = {
    {},                      // Undefined
    {},                      // VertexBuffer
    {},                      // UniformBuffer
    {},                      // IndexBuffer
    eImgUsage::RenderTarget, // RenderTarget
    eImgUsage::Storage,      // UnorderedAccess
    eImgUsage::RenderTarget, // DepthRead
    eImgUsage::RenderTarget, // DepthWrite
    eImgUsage::RenderTarget, // StencilTestDepthFetch
    eImgUsage::Sampled,      // ShaderResource
    {},                      // IndirectArgument
    eImgUsage::Transfer,     // CopyDst
    eImgUsage::Transfer,     // CopySrc
    {},                      // BuildASRead
    {},                      // BuildASWrite
    {}                       // RayTracing
};
static_assert(std::size(g_img_usage_per_state) == int(eResState::_Count), "!");

MTL::PixelFormat MtlFormatFromFormat(const eFormat format) { return g_formats_mtl[size_t(format)]; }

eImgUsage ImgUsageFromState(const eResState state) { return g_img_usage_per_state[int(state)]; }

static MTL::TextureUsage to_mtl_texture_usage(const Bitmask<eImgUsage> usage, const eFormat real_format) {
    MTL::TextureUsage ret = MTL::TextureUsageUnknown;
    if (usage & eImgUsage::Sampled) {
        ret |= MTL::TextureUsageShaderRead;
    }
    if (usage & eImgUsage::Storage) {
        ret |= (MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead);
    }
    if (usage & eImgUsage::RenderTarget) {
        ret |= MTL::TextureUsageRenderTarget;
    }
    // the draw-free render pass is our only clear primitive, so clearable images need it;
    // Metal forbids RenderTarget on compressed formats
    if (bool(usage & eImgUsage::Transfer) && !IsCompressedFormat(real_format)) {
        ret |= MTL::TextureUsageRenderTarget;
    }
    return ret;
}

static MTL::Texture *CreateTexture(Context *ctx, const ImgParams &p, const eFormat real_format, ILog *log) {
    assert(!IsDepthFormat(p.format) && "the GPU renderer does not use depth imagery");
    if (g_formats_mtl[size_t(real_format)] == MTL::PixelFormatInvalid) {
        log->Error("No Metal pixel format for format %d!", int(p.format));
        return nullptr;
    }

    MTL::TextureDescriptor *desc = MTL::TextureDescriptor::alloc()->init();
    desc->setPixelFormat(g_formats_mtl[size_t(real_format)]);
    desc->setWidth(NS::UInteger(p.w));
    desc->setHeight(NS::UInteger(p.h));
    if (p.d) {
        desc->setTextureType(MTL::TextureType3D);
        desc->setDepth(NS::UInteger(p.d));
    } else {
        desc->setTextureType(p.samples > 1 ? MTL::TextureType2DMultisample : MTL::TextureType2D);
        desc->setDepth(1);
    }
    desc->setArrayLength(1);
    desc->setMipmapLevelCount(NS::UInteger(p.mip_count));
    desc->setSampleCount(NS::UInteger(p.samples));
    desc->setUsage(to_mtl_texture_usage(p.usage, real_format));
    desc->setStorageMode(MTL::StorageModeShared);

    MTL::Texture *tex = ctx->device()->newTexture(desc);
    desc->release();
    if (!tex) {
        log->Error("Failed to create Metal texture (%dx%dx%i, %i mips)!", int(p.w), int(p.h), int(p.d),
                   int(p.mip_count));
    }
    return tex;
}

void RepackToReal(const uint8_t *src, uint8_t *dst, const int px_count, const eFormat from, const eFormat to) {
    const int ch_src = GetChannelCount(from);
    const int esz = GetPerPixelDataLen(from) / ch_src;
    assert(GetPerPixelDataLen(to) == 4 * esz);

    for (int i = 0; i < px_count; ++i) {
        const uint8_t *s = src + size_t(i) * ch_src * esz;
        uint8_t *d = dst + size_t(i) * 4 * esz;
        if (ch_src == 1) {
            for (int c = 0; c < 4; ++c) {
                memcpy(d + c * esz, s, size_t(esz));
            }
        } else {
            memcpy(d, s, size_t(ch_src) * esz);
            memset(d + 3 * esz, 0xff, size_t(esz));
        }
    }
}

void Ray::Mtl::Image::InitFromRAWData(Buffer *sbuf, const int data_off, CommandBuffer cmd_buf,
                                      MemAllocators *mem_allocs, const ImgParams &p, ILog *log) {
    Free();

    handle_.generation = TextureHandleCounter++;
    params = p;
    real_format_ = MtlRealFormat(p.format, p.usage);

    handle_.tex = CreateTexture(ctx_, p, real_format_, log);
    if (!handle_.tex) {
        params.format = eFormat::Undefined;
        return;
    }

    sampler_.Init(ctx_, p.sampling);
    resource_state = eResState::Undefined;

    if (sbuf) {
        // the shared kernels may Init without a live command buffer; single-time commands
        // cover the upload then
        CommandBuffer upload_buf = cmd_buf ? cmd_buf : ctx_->BeginSingleTimeCommands();
        SetSubImage(0, 0, 0, 0, p.w, p.h, (p.d ? p.d : 1), p.format, *sbuf, upload_buf, data_off, 0);
        if (!cmd_buf) {
            ctx_->EndSingleTimeCommands(upload_buf);
        }
    }
}

Ray::Mtl::Image::Image(std::string_view name, Context *ctx, const ImgParams &p, MemAllocators *mem_allocs, ILog *log)
    : ctx_(ctx), name_(name) {
    Init(p, mem_allocs, log);
}

Ray::Mtl::Image::Image(std::string_view name, Context *ctx, MTL::Texture *tex, const ImgParams &_params, ILog *log)
    : ctx_(ctx), name_(name) {
    Init(tex, _params, log);
}

Ray::Mtl::Image::Image(std::string_view name, Context *ctx, const void *data, const uint32_t size, const ImgParams &p,
                       Buffer &stage_buf, CommandBuffer cmd_buf, MemAllocators *mem_allocs, eImgLoadStatus *load_status,
                       ILog *log)
    : ctx_(ctx), name_(name) {
    Init(data, size, p, stage_buf, cmd_buf, mem_allocs, load_status, log);
}

Ray::Mtl::Image::~Image() { Free(); }

Ray::Mtl::Image &Ray::Mtl::Image::operator=(Image &&rhs) noexcept {
    if (this == &rhs) {
        return (*this);
    }

    Free();

    ctx_ = std::exchange(rhs.ctx_, nullptr);
    handle_ = std::exchange(rhs.handle_, {});
    sampler_ = std::move(rhs.sampler_);
    name_ = std::move(rhs.name_);
    real_format_ = std::exchange(rhs.real_format_, eFormat::Undefined);
    params = std::exchange(rhs.params, {});
    resource_state = std::exchange(rhs.resource_state, eResState::Undefined);

    return (*this);
}

void Ray::Mtl::Image::Free() {
    if (params.format != eFormat::Undefined && !bool(Bitmask<eImgFlags>{params.flags} & eImgFlags::NoOwnership)) {
        if (handle_.tex) {
            handle_.tex->release();
            sampler_.Free();
        }

        handle_ = {};
        real_format_ = eFormat::Undefined;
        params.format = eFormat::Undefined;
    }
}

void Ray::Mtl::Image::Init(const ImgParams &p, MemAllocators *mem_allocs, ILog *log) {
    InitFromRAWData(nullptr, 0, nullptr, mem_allocs, p, log);
}

void Ray::Mtl::Image::Init(MTL::Texture *tex, const ImgParams &_params, ILog *) {
    Free();

    params = _params;
    real_format_ = MtlRealFormat(_params.format, _params.usage);
    handle_ = {tex->retain(), uint32_t(TextureHandleCounter++)}; // Free() releases: adopt = own
    sampler_.Init(ctx_, _params.sampling);
    resource_state = eResState::Undefined;
}

void Ray::Mtl::Image::Init(const void *data, const uint32_t size, const ImgParams &p, Buffer &sbuf,
                           CommandBuffer cmd_buf, MemAllocators *mem_allocs, eImgLoadStatus *load_status, ILog *log) {
    uint8_t *stage_data = sbuf.Map();
    memcpy(stage_data, data, size);
    sbuf.Unmap();

    InitFromRAWData(&sbuf, 0, cmd_buf, mem_allocs, p, log);

    if (load_status) {
        (*load_status) = eImgLoadStatus::CreatedFromData;
    }
}

bool Ray::Mtl::Image::Realloc(const int w, const int h, int mip_count, const int samples, const eFormat format,
                              CommandBuffer cmd_buf, MemAllocators *mem_allocs, ILog *log) {
    ImgParams p;
    p.w = uint16_t(w);
    p.h = uint16_t(h);
    p.d = params.d;
    p.mip_count = uint8_t(mip_count);
    p.samples = uint8_t(samples);
    p.flags = Bitmask<eImgFlags>{params.flags};
    p.usage = Bitmask<eImgUsage>{params.usage};
    p.format = format;
    p.sampling = SamplingParams{params.sampling};

    const eFormat new_real_format = MtlRealFormat(format, p.usage);
    MTL::Texture *new_tex = CreateTexture(ctx_, p, new_real_format, log);
    if (!new_tex) {
        return false;
    }

    // keep the content of matching mips, same mip-alignment walk as VK
    if (new_real_format == real_format_ && !IsCompressedFormat(new_real_format)) {
        int src_mip = 0, dst_mip = 0;
        while (std::max(int(params.w) >> src_mip, 1) != std::max(w >> dst_mip, 1) ||
               std::max(int(params.h) >> src_mip, 1) != std::max(h >> dst_mip, 1)) {
            if (std::max(int(params.w) >> src_mip, 1) > std::max(w >> dst_mip, 1) ||
                std::max(int(params.h) >> src_mip, 1) > std::max(h >> dst_mip, 1)) {
                ++src_mip;
            } else {
                ++dst_mip;
            }
        }

        MTL::BlitCommandEncoder *blit = ctx_->BlitEncoder(cmd_buf);
        for (; src_mip < int(params.mip_count) && dst_mip < mip_count; ++src_mip, ++dst_mip) {
            const int src_w = std::max(int(params.w) >> src_mip, 1), src_h = std::max(int(params.h) >> src_mip, 1);
            const int dst_w = std::max(w >> dst_mip, 1), dst_h = std::max(h >> dst_mip, 1);
            blit->copyFromTexture(handle_.tex, 0, NS::UInteger(src_mip), MTL::Origin::Make(0, 0, 0),
                                  MTL::Size::Make(NS::UInteger(src_w), NS::UInteger(src_h), 1), new_tex, 0,
                                  NS::UInteger(dst_mip), MTL::Origin::Make(0, 0, 0));
        }
    }

    const ImgHandle old_handle = std::exchange(handle_, {new_tex, uint32_t(TextureHandleCounter++)});
    if (old_handle.tex) {
        old_handle.tex->release();
    }

    params = p;
    real_format_ = new_real_format;
    resource_state = eResState::Undefined;

    return true;
}

void Ray::Mtl::Image::SetSubImage(const int level, const int offsetx, const int offsety, const int offsetz,
                                  const int sizex, const int sizey, const int sizez, const eFormat format,
                                  const Buffer &sbuf, CommandBuffer cmd_buf, const int data_off, const int) {
    assert(format == params.format);
    assert(handle_.tex);
    assert(sbuf.type() == eBufType::Upload);
    assert(offsetx >= 0 && offsetx + sizex <= std::max(int(params.w) >> level, 1));
    assert(offsety >= 0 && offsety + sizey <= std::max(int(params.h) >> level, 1));
    assert(offsetz >= 0 && offsetz + sizez <= std::max(int(params.d) >> level, 1));

    int ox = offsetx, oy = offsety, sx = sizex, sy = sizey;
    if (IsCompressedFormat(format)) {
        // Compressed copies use whole blocks: floor the offset to the block grid, ceil the extent
        // but clamp to the mip edge - same rule as the VK path.
        ox &= ~3;
        oy &= ~3;
        const int mip_w = std::max(int(params.w) >> level, 1), mip_h = std::max(int(params.h) >> level, 1);
        sx = std::min((sizex + 3) & ~3, mip_w - ox);
        sy = std::min((sizey + 3) & ~3, mip_h - oy);
    }

    // staged source rows are tightly packed in the logical format
    const int src_row = IsCompressedFormat(format) ? GetBlockCount(sizex, 1, format) * GetBlockLenBytes(format)
                                                   : sizex * GetPerPixelDataLen(format);
    const int rows = IsCompressedFormat(real_format_) ? (sy + 3) / 4 : sy;
    const int depth = std::max(sizez, 1);

    const Buffer *stage = &sbuf;
    Buffer temp_sbuf;
    if (real_format_ != format) {
        const int dst_row = sx * GetPerPixelDataLen(real_format_);
        temp_sbuf = Buffer("Temp Stage", ctx_, eBufType::Upload, dst_row * rows * depth);
        uint8_t *dst = temp_sbuf.Map();
        const uint8_t *src =
            sbuf.is_mapped() ? sbuf.mapped_ptr<uint8_t>() + data_off : static_cast<uint8_t *>(sbuf.mtl()->contents());
        if (!sbuf.is_mapped()) {
            src += data_off;
        }
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < rows; ++y) {
                RepackToReal(src + (size_t(z) * rows + y) * src_row, dst + (size_t(z) * rows + y) * dst_row, sx, format,
                             real_format_);
            }
        }
        temp_sbuf.Unmap();
        stage = &temp_sbuf;
    }
    const int dst_row = IsCompressedFormat(real_format_)
                            ? GetBlockCount(sx, 1, real_format_) * GetBlockLenBytes(real_format_)
                            : sx * GetPerPixelDataLen(real_format_);

    MTL::BlitCommandEncoder *blit = ctx_->BlitEncoder(cmd_buf);
    // sizez > 1 (3D textures, e.g. the glass energy LUTs) must copy all slices - depth==1 leaves the rest zeroed
    blit->copyFromBuffer(
        stage->mtl(), NS::UInteger(stage == &sbuf ? data_off : 0), NS::UInteger(dst_row), NS::UInteger(dst_row * rows),
        MTL::Size::Make(NS::UInteger(sx), NS::UInteger(sy), NS::UInteger(depth)), handle_.tex, 0, NS::UInteger(level),
        MTL::Origin::Make(NS::UInteger(ox), NS::UInteger(oy), NS::UInteger(offsetz)));

    sbuf.resource_state = eResState::CopySrc;
    resource_state = eResState::CopyDst;

    if (stage == &temp_sbuf) {
        temp_sbuf.FreeImmediate();
    }
}

void Ray::Mtl::Image::SetSampling(const SamplingParams s) {
    sampler_.Init(ctx_, s);
    params.sampling = s;
}

void CopyImageToImage(CommandBuffer cmd_buf, Image &src_tex, const uint32_t src_level, const uint32_t src_x,
                      const uint32_t src_y, Image &dst_tex, const uint32_t dst_level, const uint32_t dst_x,
                      const uint32_t dst_y, const uint32_t width, const uint32_t height) {
    assert(src_tex.real_format() == dst_tex.real_format());
    assert(!IsCompressedFormat(src_tex.real_format()));

    MTL::BlitCommandEncoder *blit = src_tex.ctx()->BlitEncoder(cmd_buf);
    blit->copyFromTexture(src_tex.mtl(), 0, src_level, MTL::Origin::Make(src_x, src_y, 0),
                          MTL::Size::Make(width, height, 1), dst_tex.mtl(), 0, dst_level,
                          MTL::Origin::Make(dst_x, dst_y, 0));

    src_tex.resource_state = eResState::CopySrc;
    dst_tex.resource_state = eResState::CopyDst;
}

void CopyImageToBuffer(const Image &src_tex, const int level, const int x, const int y, const int w, const int h,
                       const Buffer &dst_buf, CommandBuffer cmd_buf, const int data_off) {
    assert(!IsCompressedFormat(src_tex.real_format())); // Metal cannot blit compressed textures out

    const int row = w * GetPerPixelDataLen(src_tex.real_format());

    MTL::BlitCommandEncoder *blit = src_tex.ctx()->BlitEncoder(cmd_buf);
    blit->copyFromTexture(src_tex.mtl(), 0, NS::UInteger(level), MTL::Origin::Make(NS::UInteger(x), NS::UInteger(y), 0),
                          MTL::Size::Make(NS::UInteger(w), NS::UInteger(h), 1), dst_buf.mtl(), NS::UInteger(data_off),
                          NS::UInteger(row), NS::UInteger(row * h));

    src_tex.resource_state = eResState::CopySrc;
    dst_buf.resource_state = eResState::CopyDst;
}

static void ClearColorImageImpl(Image &tex, const void *rgba, CommandBuffer cmd_buf, const bool is_float) {
    assert(tex);
    assert(tex.resource_state == eResState::CopyDst);
    assert(!IsCompressedFormat(tex.real_format()));
    assert(tex.params.d == 0 && "the draw-free clear pass cannot attach 3D textures");

    MTL::RenderPassDescriptor *rpd = MTL::RenderPassDescriptor::alloc()->init();
    MTL::RenderPassColorAttachmentDescriptor *attachment = rpd->colorAttachments()->object(0);
    attachment->setTexture(tex.mtl());
    attachment->setLevel(0);
    attachment->setSlice(0);
    attachment->setLoadAction(MTL::LoadActionClear);
    attachment->setStoreAction(MTL::StoreActionStore);

    if (is_float) {
        const float *f = static_cast<const float *>(rgba);
        attachment->setClearColor(MTL::ClearColor::Make(double(f[0]), double(f[1]), double(f[2]), double(f[3])));
    } else {
        const uint32_t *u = static_cast<const uint32_t *>(rgba);
        // Metal render-pass clearColor doesn't work on integer attachments (0xffff comes out as 0):
        // blit a filled buffer instead - fine for the uniform-byte patterns actually used (0, 0xffff...).
        assert(u[0] == u[1] && u[1] == u[2] && u[2] == u[3] && (u[0] == 0u || uint8_t(u[0]) == uint8_t(u[0] >> 8)) &&
               "integer image clear only supports uniform-byte fill values");
        const size_t w = tex.mtl()->width(), h = tex.mtl()->height();
        const size_t bpr = (w * size_t(GetPerPixelDataLen(tex.real_format())) + 63ull) & ~63ull;
        MTL::Buffer *fb = tex.ctx()->device()->newBuffer(bpr * h, MTL::StorageModeShared);
        memset(fb->contents(), int(uint8_t(u[0])), bpr * h);
        MTL::BlitCommandEncoder *be = tex.ctx()->BlitEncoder(cmd_buf);
        be->copyFromBuffer(fb, 0, bpr, bpr * h, MTL::Size::Make(w, h, 1), tex.mtl(), 0, 0, MTL::Origin::Make(0, 0, 0));
        fb->release();
        rpd->release();
        tex.resource_state = eResState::CopyDst;
        return;
    }

    // draw-free pass: the clear load action does the work; Context closes it at the next
    // encoder switch or command-buffer end
    tex.ctx()->RenderEncoder(cmd_buf, rpd);
    rpd->release();

    tex.resource_state = eResState::CopyDst;
}

void ClearColorImage(Image &tex, const float rgba[4], CommandBuffer cmd_buf) {
    ClearColorImageImpl(tex, rgba, cmd_buf, true);
}

void ClearColorImage(Image &tex, const uint32_t rgba[4], CommandBuffer cmd_buf) {
    ClearColorImageImpl(tex, rgba, cmd_buf, false);
}

bool CanBeBlockCompressed(int w, int h, int mip_count) {
    // ASTC is mandatory on Apple7+, BC is gated by Context caps; block-edge mips are fine
    return true;
}
} // namespace Ray::Mtl
