#pragma once

#include <vector>

#include "CoreRef.h"
#include "../SceneBase.h"

namespace Ray {
struct prim_t {
    uint32_t i0, i1, i2;
    Ref::simd_fvec3 bbox_min, bbox_max;
};

struct split_data_t {
    std::vector<uint32_t> left_indices, right_indices;
    Ref::simd_fvec3 left_bounds[2], right_bounds[2];
};

struct split_settings_t {
    float oversplit_threshold = 0.95f;
    float node_traversal_cost = 0.025f;
    bool allow_spatial_splits = false;
};

split_data_t SplitPrimitives_SAH(const prim_t *primitives, const std::vector<uint32_t> &prim_indices, const float *positions, size_t stride,
                                 const Ref::simd_fvec3 &bbox_min, const Ref::simd_fvec3 &bbox_max,
                                 const Ref::simd_fvec3 &root_min, const Ref::simd_fvec3 &root_max, const split_settings_t &s);

}