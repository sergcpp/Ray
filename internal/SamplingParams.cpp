#include "SamplingParams.h"

#include <algorithm>

int Ray::CalcMipCount(const int w, const int h, const int min_res) {
    int mip_count = 0;
    int min_dim = std::min(w, h);
    do {
        mip_count++;
    } while ((min_dim /= 2) >= min_res);
    return mip_count;
}