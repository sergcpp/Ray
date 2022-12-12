#include "BVHSplit.h"

#include <algorithm>

#include "SmallVector.h"

namespace Ray {
const int BinningThreshold = 1024;
const int BinsCount = 256;

const float SpatialSplitAlpha = 0.00001f;
const int NumSpatialSplitBins = 256;

struct bbox_t {
    Ref::simd_fvec4 min = {std::numeric_limits<float>::max()}, max = {std::numeric_limits<float>::lowest()};
    bbox_t() = default;
    bbox_t(const Ref::simd_fvec4 &_min, const Ref::simd_fvec4 &_max) : min(_min), max(_max) {}

    float surface_area() const { return surface_area(min, max); }

    static float surface_area(const Ref::simd_fvec4 &min, const Ref::simd_fvec4 &max) {
        const Ref::simd_fvec4 e = max - min;
        return 2 * (e[0] + e[1] + e[2]);
        // return e[0] * e[1] + e[0] * e[2] + e[1] * e[2];
    }
};

// stolen from Mitsuba
static int sutherland_hodgman(const Ref::simd_dvec3 *input, const int in_count, Ref::simd_dvec3 *output, const int axis,
                              const double split_pos, const bool is_minimum) {
    if (in_count < 3) {
        return 0;
    }

    Ref::simd_dvec3 cur = input[0];
    const double sign = is_minimum ? 1.0 : -1.0;
    double distance = sign * (cur[axis] - split_pos);
    bool cur_is_inside = (distance >= 0);
    int out_count = 0;

    for (int i = 0; i < in_count; ++i) {
        int nextIdx = i + 1;
        if (nextIdx == in_count) {
            nextIdx = 0;
        }
        const Ref::simd_dvec3 &next = input[nextIdx];
        distance = sign * (next[axis] - split_pos);
        bool next_is_inside = (distance >= 0);

        if (cur_is_inside && next_is_inside) {
            // Both this and the next vertex are inside, add to the list
            output[out_count++] = next;
        } else if (cur_is_inside && !next_is_inside) {
            // Going outside -- add the intersection
            double t = (split_pos - cur[axis]) / (next[axis] - cur[axis]);
            Ref::simd_dvec3 p = cur + (next - cur) * t;
            p[axis] = split_pos; // Avoid roundoff errors
            output[out_count++] = p;
        } else if (!cur_is_inside && next_is_inside) {
            // Coming back inside -- add the intersection + next vertex
            double t = (split_pos - cur[axis]) / (next[axis] - cur[axis]);
            Ref::simd_dvec3 &p = output[out_count++];
            p = cur + (next - cur) * t;
            p[axis] = split_pos; // Avoid roundoff errors
            output[out_count++] = next;
        } else {
            // Entirely outside - do not add anything
        }
        cur = next;
        cur_is_inside = next_is_inside;
    }
    return out_count;
}

force_inline float castflt_down(const double val) {
    int32_t b;
    auto a = float(val);

    memcpy(&b, &a, sizeof(float));

    if (double(a) > val) {
        b += a > 0 ? -1 : 1;
    }

    memcpy(&a, &b, sizeof(float));

    return a;
}

force_inline float castflt_up(const double val) {
    int32_t b;
    auto a = float(val);

    memcpy(&b, &a, sizeof(float));

    if (double(a) < val) {
        b += a < 0 ? -1 : 1;
    }

    memcpy(&a, &b, sizeof(float));

    return a;
}

bbox_t GetClippedAABB(const Ref::simd_fvec3 &_v0, const Ref::simd_fvec3 &_v1, const Ref::simd_fvec3 &_v2,
                      const bbox_t &limits) {
    Ref::simd_dvec3 vertices1[9], vertices2[9];
    int vertex_count = 3;

    vertices1[0] = {double(_v0[0]), double(_v0[1]), double(_v0[2])};
    vertices1[1] = {double(_v1[0]), double(_v1[1]), double(_v1[2])};
    vertices1[2] = {double(_v2[0]), double(_v2[1]), double(_v2[2])};

    for (int axis = 0; axis < 3; axis++) {
        vertex_count = sutherland_hodgman(vertices1, vertex_count, vertices2, axis, limits.min[axis], true);
        vertex_count = sutherland_hodgman(vertices2, vertex_count, vertices1, axis, limits.max[axis], false);
    }

    bbox_t extends;

    for (int i = 0; i < vertex_count; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double pos = vertices1[i][j];
            extends.min[j] = std::min(extends.min[j], castflt_down(pos));
            extends.max[j] = std::max(extends.max[j], castflt_up(pos));
        }
    }

    return extends;
}
} // namespace Ray

Ray::split_data_t Ray::SplitPrimitives_SAH(const prim_t *primitives, Span<const uint32_t> prim_indices,
                                           const float *positions, const size_t stride, const Ref::simd_fvec4 &bbox_min,
                                           const Ref::simd_fvec4 &bbox_max, const Ref::simd_fvec4 &root_min,
                                           const Ref::simd_fvec4 &root_max, const bvh_settings_t &s) {
    const int num_prims = int(prim_indices.size());
    const bbox_t whole_box = {bbox_min, bbox_max};

    std::vector<bbox_t> modified_prim_bounds;

    if (s.allow_spatial_splits && positions) {
        modified_prim_bounds.resize(num_prims);

        for (int i = 0; i < num_prims; i++) {
            const prim_t &p = primitives[prim_indices[i]];

            const auto v0 = Ref::simd_fvec3{&positions[p.i0 * stride]}, v1 = Ref::simd_fvec3{&positions[p.i1 * stride]},
                       v2 = Ref::simd_fvec3{&positions[p.i2 * stride]};

            modified_prim_bounds[i] = GetClippedAABB(v0, v1, v2, whole_box);
        }
    }

    float res_sah = std::numeric_limits<float>::max();
    if (s.oversplit_threshold > 0.0f) {
        res_sah = s.oversplit_threshold *whole_box.surface_area() * float(num_prims);
    }
    int div_axis = -1;
    float div_pos = 0.0f;
    uint32_t div_index = 0;
    bbox_t res_left_bounds, res_right_bounds;

    if (num_prims > BinningThreshold) {
        for (int axis = 0; axis < 3; ++axis) {
            const float bounds_min = whole_box.min[axis];
            const float bounds_max = whole_box.max[axis];
            if (bounds_max - bounds_min < FLT_EPS) {
                // flat box
                continue;
            }

            const float Scale = float(BinsCount) / (bounds_max - bounds_min);

            struct bin_t {
                bbox_t bounds;
                int prim_count = 0;
            } bins[BinsCount];

            for (int i = 0; i < num_prims; ++i) {
                const prim_t &p = primitives[prim_indices[i]];
                const int bin_ndx = std::min(BinsCount - 1, int((p.bbox_min[axis] - bounds_min) * Scale));

                bins[bin_ndx].prim_count++;
                bins[bin_ndx].bounds.min = min(bins[bin_ndx].bounds.min, p.bbox_min);
                bins[bin_ndx].bounds.max = max(bins[bin_ndx].bounds.max, p.bbox_max);
            }

            float area_left[BinsCount - 1], area_right[BinsCount - 1];
            int count_left[BinsCount - 1], count_right[BinsCount - 1];

            bbox_t box_left, box_right;
            int prims_left = 0, prims_right = 0;
            for (int i = 0; i < BinsCount - 1; ++i) {
                prims_left += bins[i].prim_count;
                count_left[i] = prims_left;
                box_left.min = min(box_left.min, bins[i].bounds.min);
                box_left.max = max(box_left.max, bins[i].bounds.max);
                area_left[i] = prims_left ? box_left.surface_area() : 0.0f;

                prims_right += bins[BinsCount - i - 1].prim_count;
                count_right[BinsCount - i - 2] = prims_right;
                box_right.min = min(box_right.min, bins[BinsCount - i - 1].bounds.min);
                box_right.max = max(box_right.max, bins[BinsCount - i - 1].bounds.max);
                area_right[BinsCount - i - 2] = prims_right ? box_right.surface_area() : 0.0f;
            }

            const float scale = (bounds_max - bounds_min) / float(BinsCount);
            for (int i = 0; i < BinsCount - 1; ++i) {
                const float sah = float(count_left[i]) * area_left[i] + float(count_right[i]) * area_right[i];
                if (sah < res_sah) {
                    res_sah = sah;
                    div_axis = axis;
                    div_pos = bounds_min + scale * float(i + 1);
                }
            }
        }

        std::vector<uint32_t> left_indices, right_indices;
        if (div_axis != -1) {
            for (int i = 0; i < num_prims; ++i) {
                const prim_t &p = primitives[prim_indices[i]];
                if (p.bbox_min[div_axis] < div_pos) {
                    left_indices.push_back(prim_indices[i]);
                    res_left_bounds.min = min(res_left_bounds.min, p.bbox_min);
                    res_left_bounds.max = max(res_left_bounds.max, p.bbox_max);
                } else {
                    right_indices.push_back(prim_indices[i]);
                    res_right_bounds.min = min(res_right_bounds.min, p.bbox_min);
                    res_right_bounds.max = max(res_right_bounds.max, p.bbox_max);
                }
            }
        } else {
            left_indices.assign(prim_indices.begin(), prim_indices.end());
            res_left_bounds = whole_box;
        }

        return {std::move(left_indices),
                std::move(right_indices),
                {res_left_bounds.min, res_left_bounds.max},
                {res_right_bounds.min, res_right_bounds.max}};
    } else {
        SmallVector<uint32_t, 1024> axis_lists[3];

        if (num_prims > s.min_primitives_in_leaf) {
            for (auto &list : axis_lists) {
                list.reserve(num_prims);
            }

            for (uint32_t i = 0; i < uint32_t(num_prims); ++i) {
                axis_lists[0].push_back(i);
                axis_lists[1].push_back(i);
                axis_lists[2].push_back(i);
            }

            SmallVector<bbox_t, 1024> right_bounds(num_prims);

            for (int axis = 0; axis < 3; ++axis) {
                auto &list = axis_lists[axis];

                if (modified_prim_bounds.empty()) {
                    std::sort(list.begin(), list.end(),
                              [axis, primitives, &prim_indices](uint32_t p1, uint32_t p2) -> bool {
                                  return primitives[prim_indices[p1]].bbox_max[axis] <
                                         primitives[prim_indices[p2]].bbox_max[axis];
                              });
                } else {
                    std::sort(list.begin(), list.end(),
                              [axis, &modified_prim_bounds](uint32_t p1, uint32_t p2) -> bool {
                                  return modified_prim_bounds[p1].max[axis] < modified_prim_bounds[p2].max[axis];
                              });
                }

                bbox_t cur_right_bounds;
                if (modified_prim_bounds.empty()) {
                    for (size_t i = list.size() - 1; i > 0; --i) {
                        cur_right_bounds.min = min(cur_right_bounds.min, primitives[prim_indices[list[i]]].bbox_min);
                        cur_right_bounds.max = max(cur_right_bounds.max, primitives[prim_indices[list[i]]].bbox_max);
                        right_bounds[i - 1] = cur_right_bounds;
                    }
                } else {
                    for (size_t i = list.size() - 1; i > 0; --i) {
                        cur_right_bounds.min = min(cur_right_bounds.min, modified_prim_bounds[list[i]].min);
                        cur_right_bounds.max = max(cur_right_bounds.max, modified_prim_bounds[list[i]].max);
                        right_bounds[i - 1] = cur_right_bounds;
                    }
                }

                bbox_t left_bounds;
                for (size_t i = 1; i < list.size(); i++) {
                    if (modified_prim_bounds.empty()) {
                        left_bounds.min = min(left_bounds.min, primitives[prim_indices[list[i - 1]]].bbox_min);
                        left_bounds.max = max(left_bounds.max, primitives[prim_indices[list[i - 1]]].bbox_max);
                    } else {
                        left_bounds.min = min(left_bounds.min, modified_prim_bounds[list[i - 1]].min);
                        left_bounds.max = max(left_bounds.max, modified_prim_bounds[list[i - 1]].max);
                    }

                    const float sah =
                        left_bounds.surface_area() * float(i) + right_bounds[i - 1].surface_area() * float(list.size() - i);
                    if (sah < res_sah) {
                        res_sah = sah;
                        div_axis = axis;
                        div_index = uint32_t(i);
                        res_left_bounds = left_bounds;
                        res_right_bounds = right_bounds[i - 1];
                    }
                }
            }

            const bbox_t overlap = {max(res_left_bounds.min, res_right_bounds.min),
                                    min(res_left_bounds.max, res_right_bounds.max)};
            Ref::simd_ivec4 test = simd_cast(overlap.max <= overlap.min);
            test[3] = 0;

            if (s.allow_spatial_splits && test.all_zeros() &&
                overlap.surface_area() > SpatialSplitAlpha * bbox_t::surface_area(root_min, root_max)) {
                struct bin_t {
                    bbox_t extends, limits;
                    uint32_t enter_counter = 0, exit_counter = 0, prim_counter = 0;
                };

                int spatial_split = -1;

                for (int split_axis = 0; split_axis < 3; split_axis++) {
                    bin_t bins[NumSpatialSplitBins];
                    const double bin_size =
                        double(whole_box.max[split_axis] - whole_box.min[split_axis]) / NumSpatialSplitBins;

                    // skip this axis if bbox is flat
                    if (bin_size < FLT_EPS) {
                        continue;
                    }

                    for (int i = 0; i < NumSpatialSplitBins; i++) {
                        bins[i].limits.min = whole_box.min;
                        bins[i].limits.max = whole_box.max;
                        bins[i].limits.min[split_axis] = castflt_down(whole_box.min[split_axis] + i * bin_size);
                        bins[i].limits.max[split_axis] = castflt_up(whole_box.min[split_axis] + (i + 1) * bin_size);
                    }

                    bins[NumSpatialSplitBins - 1].limits.max[split_axis] = whole_box.max[split_axis];

                    const auto &list = axis_lists[split_axis];

                    for (const uint32_t i : list) {
                        const prim_t &p = primitives[prim_indices[i]];

                        float prim_min = p.bbox_min[split_axis], prim_max = p.bbox_max[split_axis];

                        if (!modified_prim_bounds.empty()) {
                            prim_min = modified_prim_bounds[i].min[split_axis];
                            prim_max = modified_prim_bounds[i].max[split_axis];
                        }

                        int enter_index = 0, exit_index = 0;

                        for (int j = 0; j < NumSpatialSplitBins; j++) {
                            if (prim_min >= bins[j].limits.min[split_axis]) {
                                enter_index = j;
                            } else {
                                break;
                            }
                        }

                        for (int j = enter_index; j < NumSpatialSplitBins; j++) {
                            if (prim_max >= bins[j].limits.min[split_axis]) {
                                exit_index = j;
                            } else {
                                break;
                            }
                        }

                        bins[enter_index].enter_counter++;
                        bins[exit_index].exit_counter++;

                        if (positions) {
                            auto v0 = Ref::simd_fvec3{&positions[p.i0 * stride]},
                                 v1 = Ref::simd_fvec3{&positions[p.i1 * stride]},
                                 v2 = Ref::simd_fvec3{&positions[p.i2 * stride]};

                            for (int j = enter_index; j <= exit_index; j++) {
                                bbox_t box = GetClippedAABB(v0, v1, v2, bins[j].limits);

                                bins[j].extends.min = min(bins[j].extends.min, box.min);
                                bins[j].extends.max = max(bins[j].extends.max, box.max);
                                bins[j].prim_counter++;
                            }
                        } else {
                            for (int j = enter_index; j <= exit_index; j++) {
                                bins[j].extends.min = min(bins[j].extends.min, p.bbox_min);
                                bins[j].extends.max = max(bins[j].extends.max, p.bbox_max);
                                bins[j].prim_counter++;
                            }
                        }
                    }

                    for (int i = 0; i < NumSpatialSplitBins; i++) {
                        if (!bins[i].prim_counter) {
                            bins[i].extends = bins[i].limits;
                        } else {
                            bins[i].extends.min = max(bins[i].extends.min, bins[i].limits.min);
                            bins[i].extends.max = min(bins[i].extends.max, bins[i].limits.max);
                        }
                    }

                    for (int split = 1; split < NumSpatialSplitBins; split++) {
                        bbox_t ext_left, ext_right;
                        uint32_t num_left = 0, num_right = 0;
                        for (int i = 0; i < split; i++) {
                            if (!bins[i].prim_counter)
                                continue;

                            ext_left.min = min(ext_left.min, bins[i].extends.min);
                            ext_left.max = max(ext_left.max, bins[i].extends.max);
                            num_left += bins[i].enter_counter;
                        }
                        for (int i = split; i < NumSpatialSplitBins; i++) {
                            if (!bins[i].prim_counter)
                                continue;

                            ext_right.min = min(ext_right.min, bins[i].extends.min);
                            ext_right.max = max(ext_right.max, bins[i].extends.max);
                            num_right += bins[i].exit_counter;
                        }

                        const float split_sah =
                            ext_left.surface_area() * float(num_left) + ext_right.surface_area() * float(num_right);
                        if (split_sah < res_sah) {
                            res_sah = split_sah;
                            spatial_split = split;
                            div_axis = split_axis;

                            res_left_bounds = ext_left;
                            res_right_bounds = ext_right;
                        }
                    }
                }

                if (spatial_split != -1) {
                    std::vector<uint32_t> left_indices, right_indices;

                    int num_in_both = 0;

                    const auto &list = axis_lists[div_axis];
                    for (const uint32_t i : list) {
                        bool b1 = false, b2 = false;
                        if (modified_prim_bounds[i].min[div_axis] <= res_left_bounds.max[div_axis]) {
                            left_indices.push_back(prim_indices[i]);
                            b1 = true;
                        }

                        if (modified_prim_bounds[i].max[div_axis] >= res_right_bounds.min[div_axis]) {
                            right_indices.push_back(prim_indices[i]);
                            b2 = true;
                        }

                        if (b1 && b2) {
                            num_in_both++;
                        }
                    }

                    printf("Spatial split: %i %i %i\n", (int)left_indices.size(), (int)right_indices.size(),
                           num_in_both);
                    printf("Extends: %f..%f %f..%f\n", res_left_bounds.min[div_axis], res_left_bounds.max[div_axis],
                           res_right_bounds.min[div_axis], res_right_bounds.max[div_axis]);

                    return {std::move(left_indices),
                            std::move(right_indices),
                            {res_left_bounds.min, res_left_bounds.max},
                            {res_right_bounds.min, res_right_bounds.max}};
                }
            }
        }

        std::vector<uint32_t> left_indices, right_indices;
        if (div_axis != -1) {
            left_indices.reserve(size_t(div_index));
            right_indices.reserve(prim_indices.size() - div_index);
            for (size_t i = 0; i < div_index; ++i) {
                left_indices.push_back(prim_indices[axis_lists[div_axis][i]]);
            }
            for (size_t i = div_index; i < axis_lists[div_axis].size(); ++i) {
                right_indices.push_back(prim_indices[axis_lists[div_axis][i]]);
            }
        } else {
            left_indices.assign(prim_indices.begin(), prim_indices.end());
            res_left_bounds = whole_box;
        }

        return {std::move(left_indices),
                std::move(right_indices),
                {res_left_bounds.min, res_left_bounds.max},
                {res_right_bounds.min, res_right_bounds.max}};
    }
}
