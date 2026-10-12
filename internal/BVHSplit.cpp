#include "BVHSplit.h"

#include <cfloat>
#include <climits>

#include <algorithm>
#include <numeric>

#include "SmallVector.h"

#pragma warning(push)
#pragma warning(disable : 6262) // Function uses a lot of stack

namespace Ray {
const int BinningThreshold = 1024;
const int BinsCount = 256;
const int MaxBuildDepth = 32;
const int MaxSpatialDepth = 24;

const float SpatialSplitAlpha = 0.00001f;
const int NumSpatialSplitBins = 32;

// stolen from Mitsuba
static int sutherland_hodgman(const Ref::dvec3 *input, const int in_count, Ref::dvec3 *output, const int axis,
                              const double split_pos, const bool is_minimum) {
    if (in_count < 3) {
        return 0;
    }

    Ref::dvec3 cur = input[0];
    const double sign = is_minimum ? 1.0 : -1.0;
    double distance = sign * (cur[axis] - split_pos);
    bool cur_is_inside = (distance >= 0);
    int out_count = 0;

    for (int i = 0; i < in_count; ++i) {
        int nextIdx = i + 1;
        if (nextIdx == in_count) {
            nextIdx = 0;
        }
        const Ref::dvec3 &next = input[nextIdx];
        distance = sign * (next[axis] - split_pos);
        bool next_is_inside = (distance >= 0);

        if (cur_is_inside && next_is_inside) {
            // Both this and the next vertex are inside, add to the list
            output[out_count++] = next;
        } else if (cur_is_inside && !next_is_inside) {
            // Going outside -- add the intersection
            double t = (split_pos - cur[axis]) / (next[axis] - cur[axis]);
            Ref::dvec3 p = cur + (next - cur) * t;
            p.set(axis, split_pos); // Avoid roundoff errors
            output[out_count++] = p;
        } else if (!cur_is_inside && next_is_inside) {
            // Coming back inside -- add the intersection + next vertex
            double t = (split_pos - cur[axis]) / (next[axis] - cur[axis]);
            Ref::dvec3 &p = output[out_count++];
            p = cur + (next - cur) * t;
            p.set(axis, split_pos); // Avoid roundoff errors
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

static void grow_from_poly(bbox_t &box, const Ref::dvec3 *poly, const int count) {
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < 3; ++j) {
            box.min.set(j, fminf(box.min[j], castflt_down(poly[i][j])));
            box.max.set(j, fmaxf(box.max[j], castflt_up(poly[i][j])));
        }
    }
}

static Ref::dvec3 prim_vertex(const vtx_attribute_t &positions, uint32_t vtx) {
    const Ref::fvec3 v{&positions.data[positions.offset + vtx * positions.stride]};
    return {double(v[0]), double(v[1]), double(v[2])};
}

// Balanced split by position along the widest axis - used whenever SAH would produce a leaf
// bigger than min_primitives_in_leaf (the compacted BVH form has no prim count per leaf, only
// the 8-prim block at prim_index, so oversized leaves make the extra prims unreachable) or when
// SAH keeps producing degenerate splits.
static split_data_t median_split(const prim_t *primitives, Span<const uint32_t> prim_indices, const bbox_t &whole_box,
                                 Span<const bbox_t> prim_bounds) {
    const int num_prims = int(prim_indices.size());

    const Ref::fvec4 ext = whole_box.max - whole_box.min;
    int axis = 0;
    if (ext[1] > ext[axis])
        axis = 1;
    if (ext[2] > ext[axis])
        axis = 2;

    std::vector<uint32_t> order(num_prims);
    std::iota(order.begin(), order.end(), 0u);
    const int mid = num_prims / 2;
    std::nth_element(
        order.begin(), order.begin() + mid, order.end(), [axis, primitives, &prim_indices](uint32_t p1, uint32_t p2) {
            return primitives[prim_indices[p1]].bbox_min[axis] < primitives[prim_indices[p2]].bbox_min[axis];
        });

    std::vector<uint32_t> left_indices, right_indices;
    std::vector<bbox_t> left_pb, right_pb;
    left_indices.reserve(mid);
    right_indices.reserve(num_prims - mid);
    if (!prim_bounds.empty()) {
        left_pb.reserve(mid);
        right_pb.reserve(num_prims - mid);
    }

    bbox_t left_bounds, right_bounds;
    for (int i = 0; i < mid; ++i) {
        const prim_t &p = primitives[prim_indices[order[i]]];
        left_indices.push_back(prim_indices[order[i]]);
        if (!prim_bounds.empty())
            left_pb.push_back(prim_bounds[order[i]]);
        left_bounds.min = min(left_bounds.min, p.bbox_min);
        left_bounds.max = max(left_bounds.max, p.bbox_max);
    }
    for (int i = mid; i < num_prims; ++i) {
        const prim_t &p = primitives[prim_indices[order[i]]];
        right_indices.push_back(prim_indices[order[i]]);
        if (!prim_bounds.empty())
            right_pb.push_back(prim_bounds[order[i]]);
        right_bounds.min = min(right_bounds.min, p.bbox_min);
        right_bounds.max = max(right_bounds.max, p.bbox_max);
    }

    return {std::move(left_indices),
            std::move(right_indices),
            std::move(left_pb),
            std::move(right_pb),
            {left_bounds.min, left_bounds.max},
            {right_bounds.min, right_bounds.max},
            false};
}

split_data_t SplitPrimitives_SAH(const prim_t *primitives, Span<const uint32_t> prim_indices,
                                 const vtx_attribute_t &positions, const Ref::fvec4 &bbox_min,
                                 const Ref::fvec4 &bbox_max, const Ref::fvec4 &root_min, const Ref::fvec4 &root_max,
                                 const bvh_settings_t &s, int depth, int spatial_depth, bool allow_spatial,
                                 Span<const bbox_t> prim_bounds) {
    const int num_prims = int(prim_indices.size());
    const bbox_t whole_box = {bbox_min, bbox_max};
    const bool use_pb = !prim_bounds.empty();

    if (depth >= MaxBuildDepth && num_prims > s.min_primitives_in_leaf) {
        // SAH keeps producing degenerate splits - force a balanced median split
        return median_split(primitives, prim_indices, whole_box, prim_bounds);
    }

    float res_sah = FLT_MAX;
    if (s.oversplit_threshold > 0.0f) {
        res_sah = s.oversplit_threshold * whole_box.surface_area() * float(num_prims);
    }
    int div_axis = -1;
    float div_pos = 0.0f;
    uint32_t div_index = 0;
    bbox_t res_left_bounds, res_right_bounds;

    if (num_prims > BinningThreshold || s.use_fast_bvh_build) {
        std::vector<uint32_t> left_indices, right_indices;
        std::vector<bbox_t> left_pb, right_pb;
        if (use_pb) {
            left_pb.reserve(size_t(num_prims));
            right_pb.reserve(size_t(num_prims));
        }

        if (num_prims <= s.min_primitives_in_leaf) {
            left_indices.assign(prim_indices.begin(), prim_indices.end());
            if (use_pb)
                left_pb.assign(prim_bounds.begin(), prim_bounds.end());
            return {std::move(left_indices),
                    std::move(right_indices),
                    std::move(left_pb),
                    std::move(right_pb),
                    {whole_box.min, whole_box.max},
                    {res_right_bounds.min, res_right_bounds.max},
                    false};
        }

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

        if (div_axis != -1) {
            for (int i = 0; i < num_prims; ++i) {
                const prim_t &p = primitives[prim_indices[i]];
                if (p.bbox_min[div_axis] < div_pos) {
                    left_indices.push_back(prim_indices[i]);
                    if (use_pb)
                        left_pb.push_back(prim_bounds[i]);
                    res_left_bounds.min = min(res_left_bounds.min, p.bbox_min);
                    res_left_bounds.max = max(res_left_bounds.max, p.bbox_max);
                } else {
                    right_indices.push_back(prim_indices[i]);
                    if (use_pb)
                        right_pb.push_back(prim_bounds[i]);
                    res_right_bounds.min = min(res_right_bounds.min, p.bbox_min);
                    res_right_bounds.max = max(res_right_bounds.max, p.bbox_max);
                }
            }

            if (left_indices.empty() || right_indices.empty())
                return median_split(primitives, prim_indices, whole_box, prim_bounds);
        } else if (num_prims > s.min_primitives_in_leaf) {
            return median_split(primitives, prim_indices, whole_box, prim_bounds);
        } else {
            left_indices.assign(prim_indices.begin(), prim_indices.end());
            if (use_pb)
                left_pb.assign(prim_bounds.begin(), prim_bounds.end());
            res_left_bounds = whole_box;
        }

        return {std::move(left_indices),
                std::move(right_indices),
                std::move(left_pb),
                std::move(right_pb),
                {res_left_bounds.min, res_left_bounds.max},
                {res_right_bounds.min, res_right_bounds.max},
                false};
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

                if (use_pb) {
                    std::stable_sort(list.begin(), list.end(), [axis, &prim_bounds](uint32_t p1, uint32_t p2) -> bool {
                        return prim_bounds[p1].max[axis] < prim_bounds[p2].max[axis];
                    });
                } else {
                    std::stable_sort(list.begin(), list.end(),
                                     [axis, primitives, &prim_indices](uint32_t p1, uint32_t p2) -> bool {
                                         return primitives[prim_indices[p1]].bbox_max[axis] <
                                                primitives[prim_indices[p2]].bbox_max[axis];
                                     });
                }

                bbox_t cur_right_bounds;
                if (use_pb) {
                    for (size_t i = list.size() - 1; i > 0; --i) {
                        const bbox_t &pb = prim_bounds[list[i]];
                        cur_right_bounds.min = min(cur_right_bounds.min, pb.min);
                        cur_right_bounds.max = max(cur_right_bounds.max, pb.max);
                        right_bounds[i - 1] = cur_right_bounds;
                    }
                } else {
                    for (size_t i = list.size() - 1; i > 0; --i) {
                        cur_right_bounds.min = min(cur_right_bounds.min, primitives[prim_indices[list[i]]].bbox_min);
                        cur_right_bounds.max = max(cur_right_bounds.max, primitives[prim_indices[list[i]]].bbox_max);
                        right_bounds[i - 1] = cur_right_bounds;
                    }
                }

                bbox_t left_bounds;
                for (size_t i = 1; i < list.size(); i++) {
                    if (use_pb) {
                        left_bounds.min = min(left_bounds.min, prim_bounds[list[i - 1]].min);
                        left_bounds.max = max(left_bounds.max, prim_bounds[list[i - 1]].max);
                    } else {
                        left_bounds.min = min(left_bounds.min, primitives[prim_indices[list[i - 1]]].bbox_min);
                        left_bounds.max = max(left_bounds.max, primitives[prim_indices[list[i - 1]]].bbox_max);
                    }

                    const float sah = left_bounds.surface_area() * float(i) +
                                      right_bounds[i - 1].surface_area() * float(list.size() - i);
                    if (sah < res_sah) {
                        res_sah = sah;
                        div_axis = axis;
                        div_index = uint32_t(i);
                        res_left_bounds = left_bounds;
                        res_right_bounds = right_bounds[i - 1];
                    }
                }
            }

            // Spatial split candidate search (SBVH): bin the (clipped) reference bounds, chop
            // polygons incrementally across the bin planes they span, then sweep the bins once
            // per axis for the best plane.
            int spatial_axis = -1;
            float spatial_pos = 0.0f;

            if (allow_spatial && use_pb && !positions.data.empty() && spatial_depth < MaxSpatialDepth &&
                div_axis != -1) {
                const bbox_t overlap = {max(res_left_bounds.min, res_right_bounds.min),
                                        min(res_left_bounds.max, res_right_bounds.max)};
                Ref::ivec4 test = simd_cast(overlap.max <= overlap.min);
                test.set<3>(0);

                if (test.all_zeros() &&
                    overlap.surface_area() > SpatialSplitAlpha * bbox_t::surface_area(root_min, root_max)) {
                    struct spatial_bin_t {
                        bbox_t bounds;
                        int enter = 0, exit = 0;
                    };
                    spatial_bin_t bins[3][NumSpatialSplitBins];
                    bool axis_ok[3] = {};

                    for (int axis = 0; axis < 3; ++axis) {
                        const float bmin = whole_box.min[axis];
                        const float bin_size = (whole_box.max[axis] - bmin) / float(NumSpatialSplitBins);
                        if (bin_size < FLT_EPS) {
                            continue;
                        }
                        axis_ok[axis] = true;
                        const float inv_bin_size = 1.0f / bin_size;

                        for (int i = 0; i < num_prims; ++i) {
                            const bbox_t &pb = prim_bounds[i];
                            const int first = std::min(NumSpatialSplitBins - 1,
                                                       std::max(0, int((pb.min[axis] - bmin) * inv_bin_size)));
                            const int last = std::min(NumSpatialSplitBins - 1,
                                                      std::max(first, int((pb.max[axis] - bmin) * inv_bin_size)));

                            bins[axis][first].enter++;
                            bins[axis][last].exit++;

                            Ref::dvec3 upper[64], lower[64], rest[64];
                            const prim_t &p = primitives[prim_indices[i]];
                            upper[0] = prim_vertex(positions, p.i0);
                            upper[1] = prim_vertex(positions, p.i1);
                            upper[2] = prim_vertex(positions, p.i2);
                            int upper_count = 3;

                            for (int j = first; j < last; ++j) {
                                const double plane = double(bmin) + double(bin_size) * double(j + 1);
                                const int lower_count =
                                    sutherland_hodgman(upper, upper_count, lower, axis, plane, false);
                                grow_from_poly(bins[axis][j].bounds, lower, lower_count);
                                const int rest_count = sutherland_hodgman(upper, upper_count, rest, axis, plane, true);
                                for (int k = 0; k < rest_count; ++k)
                                    upper[k] = rest[k];
                                upper_count = rest_count;
                            }
                            grow_from_poly(bins[axis][last].bounds, upper, upper_count);
                        }
                    }

                    for (int axis = 0; axis < 3; ++axis) {
                        if (!axis_ok[axis]) {
                            continue;
                        }

                        bbox_t rb_arr[NumSpatialSplitBins];
                        bbox_t rb;
                        for (int i = NumSpatialSplitBins - 1; i > 0; --i) {
                            rb.min = min(rb.min, bins[axis][i].bounds.min);
                            rb.max = max(rb.max, bins[axis][i].bounds.max);
                            rb_arr[i - 1] = rb;
                        }

                        bbox_t lb;
                        int left_num = 0, right_num = num_prims;
                        const float bmin = whole_box.min[axis];
                        const float bin_size = (whole_box.max[axis] - bmin) / float(NumSpatialSplitBins);

                        for (int i = 1; i < NumSpatialSplitBins; ++i) {
                            lb.min = min(lb.min, bins[axis][i - 1].bounds.min);
                            lb.max = max(lb.max, bins[axis][i - 1].bounds.max);
                            left_num += bins[axis][i - 1].enter;
                            right_num -= bins[axis][i - 1].exit;

                            const float sah =
                                lb.surface_area() * float(left_num) + rb_arr[i - 1].surface_area() * float(right_num);
                            if (sah < res_sah && left_num > 0 && right_num > 0) {
                                res_sah = sah;
                                spatial_axis = axis;
                                spatial_pos = bmin + bin_size * float(i);
                            }
                        }
                    }
                }
            }

            if (spatial_axis != -1) {
                std::vector<uint32_t> left_indices, right_indices;
                std::vector<bbox_t> left_pb, right_pb;
                left_indices.reserve(num_prims);
                right_indices.reserve(num_prims);
                left_pb.reserve(num_prims);
                right_pb.reserve(num_prims);

                bbox_t lb, rb;
                for (int i = 0; i < num_prims; ++i) {
                    const bbox_t &pb = prim_bounds[i];
                    if (pb.max[spatial_axis] <= spatial_pos) {
                        left_indices.push_back(prim_indices[i]);
                        left_pb.push_back(pb);
                        lb.min = min(lb.min, pb.min);
                        lb.max = max(lb.max, pb.max);
                    } else if (pb.min[spatial_axis] >= spatial_pos) {
                        right_indices.push_back(prim_indices[i]);
                        right_pb.push_back(pb);
                        rb.min = min(rb.min, pb.min);
                        rb.max = max(rb.max, pb.max);
                    }
                }

                // References intersecting both sides: unsplit to one side or duplicate, whichever
                // is cheapest by SAH (this also keeps the duplicate count low by itself).
                for (int i = 0; i < num_prims; ++i) {
                    const bbox_t &pb = prim_bounds[i];
                    if (pb.max[spatial_axis] <= spatial_pos || pb.min[spatial_axis] >= spatial_pos) {
                        continue;
                    }

                    Ref::dvec3 tri[3];
                    const prim_t &p = primitives[prim_indices[i]];
                    tri[0] = prim_vertex(positions, p.i0);
                    tri[1] = prim_vertex(positions, p.i1);
                    tri[2] = prim_vertex(positions, p.i2);

                    Ref::dvec3 lo[6], up[6];
                    const int lo_count = sutherland_hodgman(tri, 3, lo, spatial_axis, double(spatial_pos), false);
                    const int up_count = sutherland_hodgman(tri, 3, up, spatial_axis, double(spatial_pos), true);

                    bbox_t lref, rref;
                    if (lo_count)
                        grow_from_poly(lref, lo, lo_count);
                    else
                        lref = pb;
                    if (up_count)
                        grow_from_poly(rref, up, up_count);
                    else
                        rref = pb;
                    // keep the clipped pieces inside the reference's current bounds
                    lref.min = max(lref.min, pb.min);
                    lref.max = min(lref.max, pb.max);
                    rref.min = max(rref.min, pb.min);
                    rref.max = min(rref.max, pb.max);

                    const bool dup_valid = lref.min[0] <= lref.max[0] && lref.min[1] <= lref.max[1] &&
                                           lref.min[2] <= lref.max[2] && rref.min[0] <= rref.max[0] &&
                                           rref.min[1] <= rref.max[1] && rref.min[2] <= rref.max[2];

                    const bbox_t lub = {min(lb.min, pb.min), max(lb.max, pb.max)};
                    const bbox_t rub = {min(rb.min, pb.min), max(rb.max, pb.max)};
                    const bbox_t ldb = {min(lb.min, lref.min), max(lb.max, lref.max)};
                    const bbox_t rdb = {min(rb.min, rref.min), max(rb.max, rref.max)};

                    const float lac = float(left_indices.size()), rac = float(right_indices.size());
                    const float lbc = lac + 1.0f, rbc = rac + 1.0f;

                    const float unsplit_left_sah = lub.surface_area() * lbc + rb.surface_area() * rac;
                    const float unsplit_right_sah = lb.surface_area() * lac + rub.surface_area() * rbc;
                    const float duplicate_sah =
                        dup_valid ? ldb.surface_area() * lbc + rdb.surface_area() * rbc : FLT_MAX;

                    if (unsplit_left_sah <= unsplit_right_sah && unsplit_left_sah <= duplicate_sah) {
                        left_indices.push_back(prim_indices[i]);
                        left_pb.push_back(pb);
                        lb = lub;
                    } else if (unsplit_right_sah <= duplicate_sah) {
                        right_indices.push_back(prim_indices[i]);
                        right_pb.push_back(pb);
                        rb = rub;
                    } else {
                        left_indices.push_back(prim_indices[i]);
                        left_pb.push_back(lref);
                        right_indices.push_back(prim_indices[i]);
                        right_pb.push_back(rref);
                        lb = ldb;
                        rb = rdb;
                    }
                }

                if (!left_indices.empty() && !right_indices.empty() &&
                    int(left_indices.size() + right_indices.size()) <= 2 * num_prims) {
                    return {std::move(left_indices),
                            std::move(right_indices),
                            std::move(left_pb),
                            std::move(right_pb),
                            {lb.min, lb.max},
                            {rb.min, rb.max},
                            true};
                }
                // degenerate or over-duplicating spatial split - fall back to the object split
            }
        }

        std::vector<uint32_t> left_indices, right_indices;
        std::vector<bbox_t> left_pb, right_pb;
        if (div_axis != -1) {
            left_indices.reserve(size_t(div_index));
            right_indices.reserve(prim_indices.size() - div_index);
            if (use_pb) {
                left_pb.reserve(size_t(div_index));
                right_pb.reserve(prim_indices.size() - div_index);
            }
            for (size_t i = 0; i < div_index; ++i) {
                const uint32_t pi = axis_lists[div_axis][i];
                left_indices.push_back(prim_indices[pi]);
                if (use_pb)
                    left_pb.push_back(prim_bounds[pi]);
            }
            for (size_t i = div_index; i < axis_lists[div_axis].size(); ++i) {
                const uint32_t pi = axis_lists[div_axis][i];
                right_indices.push_back(prim_indices[pi]);
                if (use_pb)
                    right_pb.push_back(prim_bounds[pi]);
            }
        } else if (num_prims > s.min_primitives_in_leaf) {
            return median_split(primitives, prim_indices, whole_box, prim_bounds);
        } else {
            left_indices.assign(prim_indices.begin(), prim_indices.end());
            if (use_pb)
                left_pb.assign(prim_bounds.begin(), prim_bounds.end());
            res_left_bounds = whole_box;
        }

        return {std::move(left_indices),
                std::move(right_indices),
                std::move(left_pb),
                std::move(right_pb),
                {res_left_bounds.min, res_left_bounds.max},
                {res_right_bounds.min, res_right_bounds.max},
                false};
    }
}
} // namespace Ray

#pragma warning(pop)
