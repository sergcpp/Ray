#pragma once

#include <cmath>

#include <vector>

namespace Ray {
// This mostly taken from Cycles source code
template <typename Func> std::vector<float> CDFEvaluate(const int res, const float from, const float to, Func func) {
    const int cdf_count = res + 1;
    const float range = to - from;

    std::vector<float> cdf(cdf_count);
    cdf[0] = 0.0f;
    for (int i = 0; i < res; ++i) {
        float x = from + range * float(i) / float(res - 1);
        float y = func(x);
        cdf[i + 1] = cdf[i] + std::abs(y);
    }
    // Normalize
    float fac = (cdf[res] == 0.0f) ? 0.0f : 1.0f / cdf[res];
    for (int i = 0; i <= res; ++i) {
        cdf[i] *= fac;
    }
    cdf[res] = 1.0f;

    return cdf;
}

std::vector<float> CDFInvert(int res, float from, float to, const std::vector<float> &cdf, bool make_symmetric);

template <typename Func>
std::vector<float> CDFInverted(const int res, const float from, const float to, Func func, const bool make_symmetric) {
    std::vector<float> cdf = CDFEvaluate(res - 1, from, to, func);
    return CDFInvert(res, from, to, cdf, make_symmetric);
}
} // namespace Ray
