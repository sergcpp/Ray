#pragma once

#include <cstdint>
#include <cstring>
#include <string>

#include "../ImageParams.h"
#include "BufferMtl.h"
#include "ContextMtl.h"
#include "MemoryAllocatorMtl.h"
#include "SamplerMtl.h"

namespace Ray {
class ILog;
namespace Mtl {
eImgUsage ImgUsageFromState(eResState state);

class Context;
class MemAllocators;

// Metal blits only require sourceBytesPerRow >= the row size, so no pitch rounding is
// needed - matches the VK path the shared scene code is written against.
const int ImageDataPitchAlignment = 1;

struct ImgHandle {
    MTL::Texture *tex = nullptr;
    uint32_t generation = 0; // used to identify unique texture (name can be reused)

    explicit operator bool() const { return tex != nullptr; }
};
static_assert(sizeof(ImgHandle) == 16, "!");

inline bool operator==(const ImgHandle &lhs, const ImgHandle &rhs) {
    return lhs.tex == rhs.tex && lhs.generation == rhs.generation;
}
inline bool operator!=(const ImgHandle &lhs, const ImgHandle &rhs) { return !operator==(lhs, rhs); }
inline bool operator<(const ImgHandle &lhs, const ImgHandle &rhs) {
    return lhs.tex < rhs.tex || (lhs.tex == rhs.tex && lhs.generation < rhs.generation);
}

// Metal texture format may differ from params.format: Metal has no RGB8 pixel format, and
// no view component swizzles, so sampled 1-channel uncompressed formats are stored 4-channel
// with R replicated (VK/DX views and the CPU storages all deliver (r,r,r,r) to shading).
inline eFormat MtlRealFormat(const eFormat format, const Bitmask<eImgUsage> usage) {
    if (format == eFormat::RGB8) {
        return eFormat::RGBA8;
    }
    if (format == eFormat::RGB8_srgb) {
        return eFormat::RGBA8_srgb;
    }
    if (GetChannelCount(format) == 1 && GetPerPixelDataLen(format) == 1 && !IsCompressedFormat(format) &&
        !bool(usage & eImgUsage::Storage)) {
        return eFormat::RGBA8;
    }
    return format;
}

// logical -> real texel repack for promoted formats (RGB8 gains A=255, 1-channel gets
// R replicated); defined in ImageMtl.cpp, shared with ImageAtlas
void RepackToReal(const uint8_t *src, uint8_t *dst, int px_count, eFormat from, eFormat to);

class Image {
    Context *ctx_ = nullptr;
    ImgHandle handle_;
    Sampler sampler_;
    std::string name_;
    eFormat real_format_ = eFormat::Undefined;

    void Free();

    void InitFromRAWData(Buffer *sbuf, int data_off, CommandBuffer cmd_buf, MemAllocators *mem_allocs,
                         const ImgParams &p, ILog *log);

  public:
    ImgParamsPacked params;

    mutable eResState resource_state = eResState::Undefined;

    Image() = default;
    Image(std::string_view name, Context *ctx, const ImgParams &params, MemAllocators *mem_allocs, ILog *log);
    Image(std::string_view name, Context *ctx, MTL::Texture *tex, const ImgParams &_params, ILog *log);
    Image(std::string_view name, Context *ctx, const void *data, uint32_t size, const ImgParams &p, Buffer &stage_buf,
          CommandBuffer cmd_buf, MemAllocators *mem_allocs, eImgLoadStatus *load_status, ILog *log);
    Image(const Image &rhs) = delete;
    Image(Image &&rhs) noexcept { (*this) = std::move(rhs); }
    ~Image();

    Image &operator=(const Image &rhs) = delete;
    Image &operator=(Image &&rhs) noexcept;

    operator bool() const { return (handle_.tex != nullptr); }

    void Init(const ImgParams &params, MemAllocators *mem_allocs, ILog *log);
    void Init(MTL::Texture *tex, const ImgParams &_params, ILog *log);
    void Init(const void *data, uint32_t size, const ImgParams &p, Buffer &stage_buf, CommandBuffer cmd_buf,
              MemAllocators *mem_allocs, eImgLoadStatus *load_status, ILog *log);

    bool Realloc(int w, int h, int mip_count, int samples, eFormat format, CommandBuffer cmd_buf,
                 MemAllocators *mem_allocs, ILog *log);

    Context *ctx() const { return ctx_; }
    const ImgHandle &handle() const { return handle_; }
    ImgHandle &handle() { return handle_; }
    MTL::Texture *mtl() const { return handle_.tex; }
    const Sampler *sampler() const { return &sampler_; }
    eFormat real_format() const { return real_format_; }
    uint32_t generation() const { return handle_.generation; }

    const std::string &name() const { return name_; }

    void SetSampling(SamplingParams sampling);
    void ApplySampling(const SamplingParams sampling, ILog *log) { SetSampling(sampling); }

    void SetSubImage(int level, int offsetx, int offsety, int offsetz, int sizex, int sizey, int sizez, eFormat format,
                     const Buffer &sbuf, CommandBuffer cmd_buf, int data_off, int data_len);
};

void CopyImageToImage(CommandBuffer cmd_buf, Image &src_tex, uint32_t src_level, uint32_t src_x, uint32_t src_y,
                      Image &dst_tex, uint32_t dst_level, uint32_t dst_x, uint32_t dst_y, uint32_t width,
                      uint32_t height);

void CopyImageToBuffer(const Image &src_tex, int level, int x, int y, int w, int h, const Buffer &dst_buf,
                       CommandBuffer cmd_buf, int data_off);

void ClearColorImage(Image &tex, const float rgba[4], CommandBuffer cmd_buf);
void ClearColorImage(Image &tex, const uint32_t rgba[4], CommandBuffer cmd_buf);

MTL::PixelFormat MtlFormatFromFormat(eFormat format);

bool CanBeBlockCompressed(int w, int h, int mip_count);

} // namespace Mtl
} // namespace Ray
