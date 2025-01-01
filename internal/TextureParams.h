#pragma once

#include <cstdint>

#include "../Bitmask.h"
#include "SamplingParams.h"

namespace Ray {
#define DECORATE(X, Y, Z, W, XX, YY, ZZ) X,
enum class eTexFormat : uint8_t {
#include "TextureFormat.inl"
    _Count
};
#undef DECORATE

inline bool IsDepthFormat(const eTexFormat format) {
    static_assert(int(eTexFormat::_Count) == 31, "Update the list below!");
    return format == eTexFormat::D16 || format == eTexFormat::D24_S8 || format == eTexFormat::D32_S8 ||
           format == eTexFormat::D32;
}

inline bool IsDepthStencilFormat(const eTexFormat format) {
    static_assert(int(eTexFormat::_Count) == 31, "Update the list below!");
    return format == eTexFormat::D24_S8 || format == eTexFormat::D32_S8;
}

inline bool IsCompressedFormat(const eTexFormat format) {
    static_assert(int(eTexFormat::_Count) == 31, "Update the list below!");
    switch (format) {
    case eTexFormat::BC1:
    case eTexFormat::BC2:
    case eTexFormat::BC3:
    case eTexFormat::BC4:
    case eTexFormat::BC5:
        return true;
    default:
        return false;
    }
    return false;
}

inline bool IsUintFormat(const eTexFormat format) {
    static_assert(int(eTexFormat::_Count) == 31, "Update the list below!");
    if (format == eTexFormat::R16UI || format == eTexFormat::R32UI || format == eTexFormat::RG32UI) {
        return true;
    }
    return false;
}

enum class eTexFlags : uint8_t { NoOwnership, SRGB };

enum class eTexUsage : uint8_t { Transfer, Sampled, Storage, RenderTarget };

struct Texture1DParams {
    uint16_t offset = 0, size = 0;
    eTexFormat format = eTexFormat::Undefined;
    uint8_t _padding = 0;
};
static_assert(sizeof(Texture1DParams) == 6, "!");

struct Tex2DParams {
    uint16_t w = 0, h = 0;
    Bitmask<eTexFlags> flags;
    uint8_t mip_count = 1;
    Bitmask<eTexUsage> usage;
    uint8_t samples = 1;
    uint8_t fallback_color[4] = {0, 255, 255, 255};
    eTexFormat format = eTexFormat::Undefined;
    SamplingParams sampling;
};
static_assert(sizeof(Tex2DParams) == 20, "!");

inline bool operator==(const Tex2DParams &lhs, const Tex2DParams &rhs) {
    return lhs.w == rhs.w && lhs.h == rhs.h && lhs.flags == rhs.flags && lhs.mip_count == rhs.mip_count &&
           lhs.usage == rhs.usage && lhs.samples == rhs.samples && lhs.fallback_color[0] == rhs.fallback_color[0] &&
           lhs.fallback_color[1] == rhs.fallback_color[1] && lhs.fallback_color[2] == rhs.fallback_color[2] &&
           lhs.fallback_color[3] == rhs.fallback_color[3] && lhs.format == rhs.format && lhs.sampling == rhs.sampling;
}
inline bool operator!=(const Tex2DParams &lhs, const Tex2DParams &rhs) { return !operator==(lhs, rhs); }

struct Tex3DParams {
    uint16_t w = 0, h = 0;
    uint8_t d = 0;
    Bitmask<eTexFlags> flags;
    Bitmask<eTexUsage> usage;
    eTexFormat format = eTexFormat::Undefined;
    SamplingParams sampling;
};
static_assert(sizeof(Tex3DParams) == 14, "!");

enum class eTexLoadStatus { Found, Reinitialized, CreatedDefault, CreatedFromData };

// const int TextureDataPitchAlignment = 256;

int GetChannelCount(eTexFormat format);
int GetPerPixelDataLen(eTexFormat format);
int GetBlockLenBytes(eTexFormat format);
int GetBlockCount(int w, int h, eTexFormat format);
inline int GetMipDataLenBytes(const int w, const int h, const eTexFormat format) {
    return GetBlockCount(w, h, format) * GetBlockLenBytes(format);
}
uint32_t EstimateMemory(const Tex2DParams &params);

eTexFormat FormatFromGLInternalFormat(uint32_t gl_internal_format, bool *is_srgb);
int BlockLenFromGLInternalFormat(uint32_t gl_internal_format);
} // namespace Ray