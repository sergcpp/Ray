#pragma once

#include <cassert>
#include <cstdint>

#include "Fixed.h"

namespace Ray {
using Fixed8 = Fixed<int8_t, 3>;

#define X(_0, _1, _2, _3) _0,
enum class eTexFilter : uint8_t {
#include "TextureFilter.inl"
};
#undef X

#define X(_0, _1, _2) _0,
enum class eTexWrap : uint8_t {
#include "TextureWrap.inl"
};
#undef X

#define X(_0, _1, _2) _0,
enum class eTexCompare : uint8_t {
#include "TextureCompare.inl"
};
#undef X

struct SamplingParams {
    eTexFilter filter = eTexFilter::Nearest;
    eTexWrap wrap = eTexWrap::Repeat;
    eTexCompare compare = eTexCompare::None;
    Fixed8 lod_bias;

    SamplingParams() = default;
    SamplingParams(const eTexFilter _filter, const eTexWrap _wrap, const eTexCompare _compare, const Fixed8 _lod_bias)
        : filter(_filter), wrap(_wrap), compare(_compare), lod_bias(_lod_bias) {}
};
static_assert(sizeof(SamplingParams) == 4, "!");

inline bool operator==(const SamplingParams lhs, const SamplingParams rhs) {
    return lhs.filter == rhs.filter && lhs.wrap == rhs.wrap && lhs.compare == rhs.compare &&
           lhs.lod_bias == rhs.lod_bias;
}
inline bool operator!=(const SamplingParams lhs, const SamplingParams rhs) { return !operator==(lhs, rhs); }

struct SamplingParamsPacked {
    uint8_t filter : 2;
    uint8_t wrap : 2;
    uint8_t compare : 4;
    Fixed8 lod_bias;

    SamplingParamsPacked() : SamplingParamsPacked(SamplingParams{}) {}
    SamplingParamsPacked(const SamplingParams &p) {
        assert(uint8_t(p.filter) < 4);
        assert(uint8_t(p.wrap) < 4);
        assert(uint8_t(p.compare) < 16);
        filter = uint8_t(p.filter);
        wrap = uint8_t(p.wrap);
        compare = uint8_t(p.compare);
        lod_bias = p.lod_bias;
    }

    operator SamplingParams() const {
        return SamplingParams{eTexFilter(filter), eTexWrap(wrap), eTexCompare(compare), lod_bias};
    }
};
static_assert(sizeof(SamplingParamsPacked) == 2, "!");

inline bool operator==(const SamplingParamsPacked lhs, const SamplingParamsPacked rhs) {
    return lhs.filter == rhs.filter && lhs.wrap == rhs.wrap && lhs.compare == rhs.compare &&
           lhs.lod_bias == rhs.lod_bias;
}
inline bool operator!=(const SamplingParamsPacked lhs, const SamplingParamsPacked rhs) {
    return !operator==(lhs, rhs);
}
inline bool operator==(const SamplingParamsPacked lhs, const SamplingParams rhs) {
    return lhs.filter == uint8_t(rhs.filter) && lhs.wrap == uint8_t(rhs.wrap) && lhs.compare == uint8_t(rhs.compare) &&
           lhs.lod_bias == rhs.lod_bias;
}
inline bool operator!=(const SamplingParamsPacked lhs, const SamplingParams rhs) { return !operator==(lhs, rhs); }

int CalcMipCount(int w, int h, int min_res);
} // namespace Ray