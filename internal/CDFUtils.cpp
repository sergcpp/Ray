#include "CDFUtils.h"

#include <algorithm>

#include <cassert>

std::vector<float> Ray::CDFInvert(const int res, const float from, const float to, const std::vector<float> &cdf,
                                  const bool make_symmetric) {
    const int cdf_size = int(cdf.size());
    assert(cdf[0] == 0.0f && cdf[cdf_size - 1] == 1.0f);

    const float inv_res = 1.0f / float(res);
    const float range = to - from;

    std::vector<float> inv_cdf(res);

    if (make_symmetric) {
        const int half_size = (res - 1) / 2;
        for (int i = 0; i <= half_size; ++i) {
            float x = float(i) / float(half_size);
            int index = int(std::upper_bound(begin(cdf), end(cdf), x) - begin(cdf));
            float t;
            if (index < cdf_size - 1) {
                t = (x - cdf[index]) / (cdf[index + 1] - cdf[index]);
            } else {
                t = 0.0f;
                index = cdf_size - 1;
            }
            float y = ((index + t) / (res - 1)) * 2.0f * range;
            inv_cdf[half_size + i] = 0.5f * (1.0f + y);
            inv_cdf[half_size - i] = 0.5f * (1.0f - y);
        }
    } else {
        for (int i = 0; i < res; ++i) {
            float x = (float(i) + 0.5f) * inv_res;
            int index = int(std::upper_bound(begin(cdf), end(cdf), x) - begin(cdf) - 1);
            float t;
            if (index < cdf_size - 1) {
                t = (x - cdf[index]) / (cdf[index + 1] - cdf[index]);
            } else {
                t = 0.0f;
                index = res;
            }
            inv_cdf[i] = from + range * (float(index) + t) * inv_res;
        }
    }

    return inv_cdf;
}