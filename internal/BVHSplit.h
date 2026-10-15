#pragma once

#include <cfloat>
#include <vector>

#include "../SceneBase.h"
#include "../Span.h"
#include "CoreRef.h"
#include "SmallVector.h"

namespace Ray {
struct prim_t {
    uint32_t i0, i1, i2;
    Ref::fvec4 bbox_min, bbox_max;
};

struct bbox_t {
    Ref::fvec4 min = {FLT_MAX}, max = {-FLT_MAX};
    bbox_t() = default;
    bbox_t(const Ref::fvec4 &_min, const Ref::fvec4 &_max) : min(_min), max(_max) {}

    float surface_area() const { return surface_area(min, max); }

    static float surface_area(const Ref::fvec4 &min, const Ref::fvec4 &max) {
        const Ref::fvec4 e = max - min;
        return 2 * (e.get<0>() + e.get<1>() + e.get<2>());
    }
};

struct ref_t {
    Ref::fvec4 bbox_min, bbox_max;
    uint32_t prim_index;
};

// Per-build scratch buffers reused across nodes
struct scratch_t {
    std::vector<bbox_t> right_bounds;
};

struct split_result_t {
    bool is_leaf = false;
    bool spatial = false;
    uint32_t left_start = 0, left_end = 0, right_start = 0, right_end = 0;
    Ref::fvec4 left_bounds[2], right_bounds[2];
};

split_result_t SplitPrimitives_SAH(const prim_t *primitives, std::vector<ref_t> &refs, uint32_t start, uint32_t end,
                                   const vtx_attribute_t &positions, const Ref::fvec4 &bbox_min,
                                   const Ref::fvec4 &bbox_max, const Ref::fvec4 &root_min, const Ref::fvec4 &root_max,
                                   const bvh_settings_t &s, int depth, int spatial_depth, bool allow_spatial,
                                   scratch_t &scratch);

} // namespace Ray
