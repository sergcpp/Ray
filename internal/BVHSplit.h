#pragma once

#include <vector>

#include <math/math.hpp>

#include "../SceneBase.h"

namespace ray {
struct prim_t {
    //uint32_t index;
    math::vec3 bbox_min, bbox_max;
};

struct split_data_t {
    std::vector<uint32_t> left_indices, right_indices;
    math::vec3 left_bounds[2], right_bounds[2];
};

split_data_t SplitPrimitives_SAH(const prim_t *primitives, const std::vector<uint32_t> &tri_indices, const math::vec3 &bbox_min, const math::vec3 &bbox_max);


}