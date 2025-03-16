#pragma once

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

int CalcMipCount(int w, int h, int min_res);
} // namespace Ray