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

struct TexParams {
    uint16_t w = 0, h = 0;
    uint8_t d = 0;
    uint8_t mip_count : 5;
    uint8_t samples : 3;
    Bitmask<eTexFlags> flags;
    Bitmask<eTexUsage> usage;
    eTexFormat format = eTexFormat::Undefined;
    SamplingParams sampling;

    TexParams() : mip_count(1), samples(1) {
        assert(mip_count < 32);
        assert(samples < 8);
    }
    TexParams(const uint16_t _w, const uint16_t _h, const uint8_t _d, const uint8_t _mip_count, const uint8_t _samples,
              const Bitmask<eTexFlags> _flags, const Bitmask<eTexUsage> _usage, const eTexFormat _format,
              const SamplingParams _sampling)
        : w(_w), h(_h), d(_d), mip_count(_mip_count), samples(_samples), flags(_flags), usage(_usage), format(_format),
          sampling(_sampling) {}
};
static_assert(sizeof(TexParams) == 14, "!");

inline bool operator==(const TexParams &lhs, const TexParams &rhs) {
    return lhs.w == rhs.w && lhs.h == rhs.h && lhs.d == rhs.d && lhs.mip_count == rhs.mip_count &&
           lhs.samples == rhs.samples && lhs.flags == rhs.flags && lhs.usage == rhs.usage && lhs.format == rhs.format &&
           lhs.sampling == rhs.sampling;
}
inline bool operator!=(const TexParams &lhs, const TexParams &rhs) { return !operator==(lhs, rhs); }

struct TexParamsPacked {
    uint16_t w = 0, h = 0;
    uint8_t d = 0;
    uint8_t mip_count : 5;
    uint8_t samples : 3;
    uint8_t flags : 2;
    uint8_t usage : 4;
    uint8_t _unused : 2;
    eTexFormat format = eTexFormat::Undefined;
    SamplingParamsPacked sampling;

    TexParamsPacked() : TexParamsPacked(TexParams{}) {}
    TexParamsPacked(const TexParams &p)
        : w(p.w), h(p.h), d(p.d), mip_count(p.mip_count), samples(p.samples), flags(p.flags), usage(p.usage),
          format(p.format), sampling(p.sampling) {
        assert(uint8_t(p.flags) < 4);
        assert(uint8_t(p.usage) < 16);
    }

    operator TexParams() const {
        return TexParams(w, h, d, mip_count, samples, Bitmask<eTexFlags>(flags), Bitmask<eTexUsage>(usage), format,
                         SamplingParams{sampling});
    }
};
static_assert(sizeof(TexParamsPacked) == 10, "!");

inline bool operator==(const TexParamsPacked &lhs, const TexParamsPacked &rhs) {
    return lhs.w == rhs.w && lhs.h == rhs.h && lhs.d == rhs.d && lhs.mip_count == rhs.mip_count &&
           lhs.samples == rhs.samples && lhs.flags == rhs.flags && lhs.usage == rhs.usage && lhs.format == rhs.format &&
           lhs.sampling == rhs.sampling;
}
inline bool operator!=(const TexParamsPacked &lhs, const TexParamsPacked &rhs) { return !operator==(lhs, rhs); }
inline bool operator==(const TexParamsPacked &lhs, const TexParams &rhs) {
    return lhs.w == rhs.w && lhs.h == rhs.h && lhs.d == rhs.d && lhs.mip_count == rhs.mip_count &&
           lhs.samples == rhs.samples && lhs.flags == uint8_t(rhs.flags) && lhs.usage == uint8_t(rhs.usage) &&
           lhs.format == rhs.format && lhs.sampling == rhs.sampling;
}
inline bool operator!=(const TexParamsPacked &lhs, const TexParams &rhs) { return !operator==(lhs, rhs); }

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