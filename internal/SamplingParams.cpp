#include "SamplingParams.h"

#include <algorithm>

int Ray::CalcMipCount(const int w, const int h, const int min_res, const eTexFilter filter) {
    int mip_count = 0;
    if (filter == eTexFilter::Trilinear || filter == eTexFilter::Bilinear) {
        int max_dim = std::max(w, h);
        do {
            mip_count++;
        } while ((max_dim /= 2) >= min_res);
    } else {
        mip_count = 1;
    }
    return mip_count;
}