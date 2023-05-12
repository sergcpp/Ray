#pragma once

#include <cstdint>

#include "Fixed.h"

namespace Ray {
using Fixed8 = Fixed<int8_t, 3>;

enum class eTexFilter : uint8_t { Nearest, Bilinear, Trilinear, BilinearNoMipmap, NearestMipmap, _Count };
enum class eTexWrap : uint8_t { Repeat, ClampToEdge, ClampToBorder, _Count };

#undef Always
enum class eTexCompare : uint8_t { None, LEqual, GEqual, Less, Greater, Equal, NotEqual, Always, Never, _Count };

struct SamplingParams {
    eTexFilter filter = eTexFilter::Nearest;
    eTexWrap wrap = eTexWrap::Repeat;
    eTexCompare compare = eTexCompare::None;
    Fixed8 lod_bias;
    Fixed8 min_lod = Fixed8::lowest(), max_lod = Fixed8::max();

    SamplingParams() = default;
    SamplingParams(const eTexFilter _filter, const eTexWrap _wrap, const eTexCompare _compare, const Fixed8 _lod_bias,
                   const Fixed8 _min_lod, const Fixed8 _max_lod)
        : filter(_filter), wrap(_wrap), compare(_compare), lod_bias(_lod_bias), min_lod(_min_lod), max_lod(_max_lod) {}
};
static_assert(sizeof(SamplingParams) == 6, "!");

inline bool operator==(const SamplingParams lhs, const SamplingParams rhs) {
    return lhs.filter == rhs.filter && lhs.wrap == rhs.wrap && lhs.compare == rhs.compare &&
           lhs.lod_bias == rhs.lod_bias && lhs.min_lod == rhs.min_lod && lhs.max_lod == rhs.max_lod;
}

enum class eSamplerLoadStatus { Found, Created };

int CalcMipCount(int w, int h, int min_res, eTexFilter filter);
} // namespace Ray