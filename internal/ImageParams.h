#pragma once

#include <cstdint>

#include "../Bitmask.h"
#include "SamplingParams.h"

namespace Ray {
#define X(_0, ...) _0,
enum class eFormat : uint8_t {
#include "Format.inl"
    _Count
};
#undef X

inline bool IsDepthFormat(const eFormat format) {
    static_assert(int(eFormat::_Count) == 37, "Update the list below!");
    return format == eFormat::D16 || format == eFormat::D24_S8 || format == eFormat::D32_S8 || format == eFormat::D32;
}

inline bool IsDepthStencilFormat(const eFormat format) {
    static_assert(int(eFormat::_Count) == 37, "Update the list below!");
    return format == eFormat::D24_S8 || format == eFormat::D32_S8;
}

inline bool IsCompressedFormat(const eFormat format) {
    static_assert(int(eFormat::_Count) == 37, "Update the list below!");
    switch (format) {
    case eFormat::BC1:
    case eFormat::BC1_srgb:
    case eFormat::BC2:
    case eFormat::BC2_srgb:
    case eFormat::BC3:
    case eFormat::BC3_srgb:
    case eFormat::BC4:
    case eFormat::BC5:
        return true;
    default:
        return false;
    }
    return false;
}

inline bool IsUintFormat(const eFormat format) {
    static_assert(int(eFormat::_Count) == 37, "Update the list below!");
    if (format == eFormat::R16UI || format == eFormat::R32UI || format == eFormat::RG32UI) {
        return true;
    }
    return false;
}

inline eFormat ToSRGBFormat(const eFormat format) {
    static_assert(int(eFormat::_Count) == 37, "Update the list below!");
    switch (format) {
    case eFormat::BC1:
        return eFormat::BC1_srgb;
    case eFormat::BC2:
        return eFormat::BC2_srgb;
    case eFormat::BC3:
        return eFormat::BC3_srgb;
    default:
        return format;
    }
}

inline bool RequiresManualSRGBConversion(const eFormat format) { return format == ToSRGBFormat(format); }

enum class eImgFlags : uint8_t { NoOwnership };

enum class eImgUsage : uint8_t { Transfer, Sampled, Storage, RenderTarget };

struct ImgParams {
    uint16_t w = 0, h = 0;
    uint8_t d = 0;
    uint8_t mip_count : 5;
    uint8_t samples : 3;
    Bitmask<eImgFlags> flags;
    Bitmask<eImgUsage> usage;
    eFormat format = eFormat::Undefined;
    SamplingParams sampling;

    ImgParams() : mip_count(1), samples(1) {
        assert(mip_count < 32);
        assert(samples < 8);
    }
    ImgParams(const uint16_t _w, const uint16_t _h, const uint8_t _d, const uint8_t _mip_count, const uint8_t _samples,
              const Bitmask<eImgFlags> _flags, const Bitmask<eImgUsage> _usage, const eFormat _format,
              const SamplingParams _sampling)
        : w(_w), h(_h), d(_d), mip_count(_mip_count), samples(_samples), flags(_flags), usage(_usage), format(_format),
          sampling(_sampling) {}
};
static_assert(sizeof(ImgParams) == 14, "!");

inline bool operator==(const ImgParams &lhs, const ImgParams &rhs) {
    return lhs.w == rhs.w && lhs.h == rhs.h && lhs.d == rhs.d && lhs.mip_count == rhs.mip_count &&
           lhs.samples == rhs.samples && lhs.flags == rhs.flags && lhs.usage == rhs.usage && lhs.format == rhs.format &&
           lhs.sampling == rhs.sampling;
}
inline bool operator!=(const ImgParams &lhs, const ImgParams &rhs) { return !operator==(lhs, rhs); }

struct ImgParamsPacked {
    uint16_t w = 0, h = 0;
    uint8_t d = 0;
    uint8_t mip_count : 5;
    uint8_t samples : 3;
    uint8_t flags : 2;
    uint8_t usage : 4;
    uint8_t _unused : 2;
    eFormat format = eFormat::Undefined;
    SamplingParamsPacked sampling;

    ImgParamsPacked() : ImgParamsPacked(ImgParams{}) {}
    ImgParamsPacked(const ImgParams &p)
        : w(p.w), h(p.h), d(p.d), mip_count(p.mip_count), samples(p.samples), flags(p.flags), usage(p.usage),
          format(p.format), sampling(p.sampling) {
        assert(uint8_t(p.flags) < 4);
        assert(uint8_t(p.usage) < 16);
    }

    operator ImgParams() const {
        return ImgParams(w, h, d, mip_count, samples, Bitmask<eImgFlags>(flags), Bitmask<eImgUsage>(usage), format,
                         SamplingParams{sampling});
    }
};
static_assert(sizeof(ImgParamsPacked) == 10, "!");

inline bool operator==(const ImgParamsPacked &lhs, const ImgParamsPacked &rhs) {
    return lhs.w == rhs.w && lhs.h == rhs.h && lhs.d == rhs.d && lhs.mip_count == rhs.mip_count &&
           lhs.samples == rhs.samples && lhs.flags == rhs.flags && lhs.usage == rhs.usage && lhs.format == rhs.format &&
           lhs.sampling == rhs.sampling;
}
inline bool operator!=(const ImgParamsPacked &lhs, const ImgParamsPacked &rhs) { return !operator==(lhs, rhs); }
inline bool operator==(const ImgParamsPacked &lhs, const ImgParams &rhs) {
    return lhs.w == rhs.w && lhs.h == rhs.h && lhs.d == rhs.d && lhs.mip_count == rhs.mip_count &&
           lhs.samples == rhs.samples && lhs.flags == uint8_t(rhs.flags) && lhs.usage == uint8_t(rhs.usage) &&
           lhs.format == rhs.format && lhs.sampling == rhs.sampling;
}
inline bool operator!=(const ImgParamsPacked &lhs, const ImgParams &rhs) { return !operator==(lhs, rhs); }

enum class eImgLoadStatus { Found, Reinitialized, CreatedDefault, CreatedFromData };

// const int ImageDataPitchAlignment = 256;

int GetChannelCount(eFormat format);
int GetPerPixelDataLen(eFormat format);
int GetBlockLenBytes(eFormat format);
int GetBlockCount(int w, int h, eFormat format);
inline int GetMipDataLenBytes(const int w, const int h, const eFormat format) {
    return GetBlockCount(w, h, format) * GetBlockLenBytes(format);
}
uint32_t EstimateMemory(const ImgParams &params);

eFormat FormatFromGLInternalFormat(uint32_t gl_internal_format, bool *is_srgb);
int BlockLenFromGLInternalFormat(uint32_t gl_internal_format);
} // namespace Ray