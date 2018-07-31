#pragma once

#include <vector>

#include "CoreRef.h"
#include "../SceneBase.h"

namespace ray {
struct prim_t {
    uint32_t i0, i1, i2;
    ref::simd_fvec3 bbox_min, bbox_max;
};

struct split_data_t {
    std::vector<uint32_t> left_indices, right_indices;
    ref::simd_fvec3 left_bounds[2], right_bounds[2];
};

split_data_t SplitPrimitives_SAH(const prim_t *primitives, const std::vector<uint32_t> &tri_indices, const float *positions, size_t stride,
                                 const ref::simd_fvec3 &bbox_min, const ref::simd_fvec3 &bbox_max,
                                 const ref::simd_fvec3 &root_min, const ref::simd_fvec3 &root_max, bool use_spatial_splits);


}