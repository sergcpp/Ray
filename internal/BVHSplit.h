#pragma once

#include <vector>

#include "../SceneBase.h"
#include "CoreRef.h"

namespace Ray {
struct prim_t {
    uint32_t i0, i1, i2;
    Ref::simd_fvec4 bbox_min, bbox_max;
};

struct split_data_t {
    std::vector<uint32_t> left_indices, right_indices;
    Ref::simd_fvec4 left_bounds[2], right_bounds[2];
};

split_data_t SplitPrimitives_SAH(const prim_t *primitives, const std::vector<uint32_t> &prim_indices,
                                 const float *positions, size_t stride, const Ref::simd_fvec4 &bbox_min,
                                 const Ref::simd_fvec4 &bbox_max, const Ref::simd_fvec4 &root_min,
                                 const Ref::simd_fvec4 &root_max, const bvh_settings_t &s);

} // namespace Ray