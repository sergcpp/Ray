#pragma once

#include <cstdint>

#include "../Bitmask.h"
#include "SamplingParams.h"

namespace Ray {
#define X(_0, ...) _0,
enum class eTexFormat : uint8_t {
#include "TextureFormat.inl"
    _Count
};
#undef X

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

struct TexParams {
    uint16_t w = 0, h = 0;
    uint8_t d = 0;
    uint8_t mip_count : 5;
    uint8_t samples : 3;
    Bitmask<eTexFlags> flags;
    Bitmask<eTexUsage> usage;
    eTexFormat format = eTexFormat::Undefined;
    SamplingParams sampling;

    TexParams() {
        mip_count = 1;
        samples = 1;
    }
};
static_assert(sizeof(TexParams) == 16, "!");

inline bool operator==(const TexParams &lhs, const TexParams &rhs) {
    return lhs.w == rhs.w && lhs.h == rhs.h && lhs.d == rhs.d && lhs.mip_count == rhs.mip_count &&
           lhs.samples == rhs.samples && lhs.flags == rhs.flags && lhs.usage == rhs.usage && lhs.format == rhs.format &&
           lhs.sampling == rhs.sampling;
}
inline bool operator!=(const TexParams &lhs, const TexParams &rhs) { return !operator==(lhs, rhs); }

enum class eTexLoadStatus { Found, Reinitialized, CreatedDefault, CreatedFromData };

// const int TextureDataPitchAlignment = 256;

int GetChannelCount(eTexFormat format);
int GetPerPixelDataLen(eTexFormat format);
int GetBlockLenBytes(eTexFormat format);
int GetBlockCount(int w, int h, eTexFormat format);
inline int GetMipDataLenBytes(const int w, const int h, const eTexFormat format) {
    return GetBlockCount(w, h, format) * GetBlockLenBytes(format);
}
uint32_t EstimateMemory(const TexParams &params);

eTexFormat FormatFromGLInternalFormat(uint32_t gl_internal_format, bool *is_srgb);
int BlockLenFromGLInternalFormat(uint32_t gl_internal_format);
} // namespace Ray