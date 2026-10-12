#pragma once

#include <cfloat>
#include <vector>

#include "../SceneBase.h"
#include "../Span.h"
#include "CoreRef.h"

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

struct split_data_t {
    std::vector<uint32_t> left_indices, right_indices;
    // Clipped reference bounds parallel to the index arrays - only populated when the parent
    // collection had prim_bounds (spatial-split mode); empty otherwise
    std::vector<bbox_t> left_prim_bounds, right_prim_bounds;
    Ref::fvec4 left_bounds[2], right_bounds[2];
    bool spatial = false;
};

// prim_bounds: per-primitive (possibly spatially clipped) reference bounds parallel to
// prim_indices, empty to use the full primitive bboxes. allow_spatial additionally gates on
// the caller's duplicate budget; spatial_depth counts spatial splits on the path to this node.
split_data_t SplitPrimitives_SAH(const prim_t *primitives, Span<const uint32_t> prim_indices,
                                 const vtx_attribute_t &positions, const Ref::fvec4 &bbox_min,
                                 const Ref::fvec4 &bbox_max, const Ref::fvec4 &root_min, const Ref::fvec4 &root_max,
                                 const bvh_settings_t &s, int depth, int spatial_depth, bool allow_spatial,
                                 Span<const bbox_t> prim_bounds);

} // namespace Ray
