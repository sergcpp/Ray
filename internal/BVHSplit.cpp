#include "BVHSplit.h"

#include <cfloat>
#include <climits>

#include <algorithm>

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

static bbox_t range_bounds(const ref_t *rp, const int count) {
    bbox_t b;
    for (int i = 0; i < count; ++i) {
        b.min = min(b.min, rp[i].bbox_min);
        b.max = max(b.max, rp[i].bbox_max);
    }
    return b;
}

static int widest_axis(const bbox_t &box) {
    const Ref::fvec4 ext = box.max - box.min;
    int axis = 0;
    if (ext[1] > ext[axis])
        axis = 1;
    if (ext[2] > ext[axis])
        axis = 2;
    return axis;
}

// Balanced split by position along the widest axis
static split_result_t median_split(std::vector<ref_t> &refs, const uint32_t start, const uint32_t end,
                                   const bbox_t &whole_box) {
    const int num_prims = int(end - start);
    const int axis = widest_axis(whole_box);

    const uint32_t mid = start + num_prims / 2;
    std::nth_element(refs.begin() + start, refs.begin() + mid, refs.begin() + end,
                     [axis](const ref_t &r1, const ref_t &r2) { return r1.bbox_min[axis] < r2.bbox_min[axis]; });

    const bbox_t lb = range_bounds(&refs[start], int(mid - start));
    const bbox_t rb = range_bounds(&refs[mid], int(end - mid));

    split_result_t res;
    res.left_start = start;
    res.left_end = mid;
    res.right_start = mid;
    res.right_end = end;
    res.left_bounds[0] = lb.min;
    res.left_bounds[1] = lb.max;
    res.right_bounds[0] = rb.min;
    res.right_bounds[1] = rb.max;
    return res;
}

static split_result_t leaf_result() {
    split_result_t res;
    res.is_leaf = true;
    return res;
}

split_result_t SplitPrimitives_SAH(const prim_t *primitives, std::vector<ref_t> &refs, const uint32_t start,
                                   const uint32_t end, const vtx_attribute_t &positions, const Ref::fvec4 &bbox_min,
                                   const Ref::fvec4 &bbox_max, const Ref::fvec4 &root_min, const Ref::fvec4 &root_max,
                                   const bvh_settings_t &s, const int depth, const int spatial_depth,
                                   const bool allow_spatial, scratch_t &scratch) {
    const int num_prims = int(end - start);
    const bbox_t whole_box = {bbox_min, bbox_max};
    assert(end == refs.size()); // tail invariant: the node's refs are the last entries of refs

    if (depth >= MaxBuildDepth && num_prims > s.min_primitives_in_leaf) {
        // SAH keeps producing degenerate splits - force a balanced median split
        return median_split(refs, start, end, whole_box);
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
        if (num_prims <= s.min_primitives_in_leaf)
            return leaf_result();

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
                const ref_t &r = refs[start + i];
                const int bin_ndx = std::min(BinsCount - 1, int((r.bbox_min[axis] - bounds_min) * Scale));

                bins[bin_ndx].prim_count++;
                bins[bin_ndx].bounds.min = min(bins[bin_ndx].bounds.min, r.bbox_min);
                bins[bin_ndx].bounds.max = max(bins[bin_ndx].bounds.max, r.bbox_max);
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

        split_result_t result;
        if (div_axis != -1) {
            uint32_t lo = start, hi = end;
            while (lo < hi) {
                if (refs[lo].bbox_min[div_axis] < div_pos)
                    ++lo;
                else
                    std::swap(refs[lo], refs[--hi]);
            }

            if (lo == start || lo == end)
                return median_split(refs, start, end, whole_box);

            const bbox_t lb = range_bounds(&refs[start], int(lo - start));
            const bbox_t rb = range_bounds(&refs[lo], int(end - lo));
            result.left_start = start;
            result.left_end = lo;
            result.right_start = lo;
            result.right_end = end;
            result.left_bounds[0] = lb.min;
            result.left_bounds[1] = lb.max;
            result.right_bounds[0] = rb.min;
            result.right_bounds[1] = rb.max;
            return result;
        }

        if (num_prims > s.min_primitives_in_leaf)
            return median_split(refs, start, end, whole_box);
        return leaf_result();
    } else {
        if (num_prims > s.min_primitives_in_leaf) {
            // Sort the node's references in place along each axis and sweep them; after the
            // loop the range is in axis-2 order, so re-sort if another axis won.
            scratch.right_bounds.resize(num_prims);

            for (int axis = 0; axis < 3; ++axis) {
                std::sort(refs.begin() + start, refs.begin() + end, [axis](const ref_t &r1, const ref_t &r2) -> bool {
                    return r1.bbox_max[axis] < r2.bbox_max[axis];
                });

                bbox_t cur_right_bounds;
                for (int i = num_prims - 1; i > 0; --i) {
                    const ref_t &r = refs[start + i];
                    cur_right_bounds.min = min(cur_right_bounds.min, r.bbox_min);
                    cur_right_bounds.max = max(cur_right_bounds.max, r.bbox_max);
                    scratch.right_bounds[i - 1] = cur_right_bounds;
                }

                bbox_t left_bounds;
                for (int i = 1; i < num_prims; i++) {
                    const ref_t &r = refs[start + i - 1];
                    left_bounds.min = min(left_bounds.min, r.bbox_min);
                    left_bounds.max = max(left_bounds.max, r.bbox_max);

                    const float sah = left_bounds.surface_area() * float(i) +
                                      scratch.right_bounds[i - 1].surface_area() * float(num_prims - i);
                    if (sah < res_sah) {
                        res_sah = sah;
                        div_axis = axis;
                        div_index = uint32_t(i);
                        res_left_bounds = left_bounds;
                        res_right_bounds = scratch.right_bounds[i - 1];
                    }
                }
            }

            int spatial_axis = -1;
            float spatial_pos = 0.0f;

            if (allow_spatial && !positions.data.empty() && spatial_depth < MaxSpatialDepth && div_axis != -1) {
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
                            const ref_t &r = refs[start + i];
                            const int first = std::min(NumSpatialSplitBins - 1,
                                                       std::max(0, int((r.bbox_min[axis] - bmin) * inv_bin_size)));
                            const int last = std::min(NumSpatialSplitBins - 1,
                                                      std::max(first, int((r.bbox_max[axis] - bmin) * inv_bin_size)));

                            bins[axis][first].enter++;
                            bins[axis][last].exit++;

                            Ref::dvec3 upper[64], lower[64], rest[64];
                            const prim_t &p = primitives[r.prim_index];
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
                // Partition in place: [start, leftEnd) left, [leftEnd, rightStart) straddlers,
                // [rightStart, refs.size()) right (duplicates get appended at the end).
                uint32_t leftEnd = start, rightStart = end;
                bbox_t lb, rb;

                for (uint32_t i = start; i < rightStart;) {
                    const ref_t &r = refs[i];
                    if (r.bbox_max[spatial_axis] <= spatial_pos) {
                        lb.min = min(lb.min, r.bbox_min);
                        lb.max = max(lb.max, r.bbox_max);
                        if (i != leftEnd)
                            std::swap(refs[i], refs[leftEnd]);
                        ++leftEnd;
                        ++i;
                    } else if (r.bbox_min[spatial_axis] >= spatial_pos) {
                        rb.min = min(rb.min, r.bbox_min);
                        rb.max = max(rb.max, r.bbox_max);
                        std::swap(refs[i], refs[--rightStart]);
                    } else {
                        ++i;
                    }
                }

                // References intersecting both sides: unsplit to one side or duplicate, whichever
                // is cheapest by SAH (this also keeps the duplicate count low by itself).
                while (leftEnd < rightStart) {
                    const ref_t r = refs[leftEnd];

                    Ref::dvec3 tri[3];
                    const prim_t &p = primitives[r.prim_index];
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
                        lref = {r.bbox_min, r.bbox_max};
                    if (up_count)
                        grow_from_poly(rref, up, up_count);
                    else
                        rref = {r.bbox_min, r.bbox_max};
                    // keep the clipped pieces inside the reference's current bounds
                    lref.min = max(lref.min, r.bbox_min);
                    lref.max = min(lref.max, r.bbox_max);
                    rref.min = max(rref.min, r.bbox_min);
                    rref.max = min(rref.max, r.bbox_max);

                    const bool dup_valid = lref.min[0] <= lref.max[0] && lref.min[1] <= lref.max[1] &&
                                           lref.min[2] <= lref.max[2] && rref.min[0] <= rref.max[0] &&
                                           rref.min[1] <= rref.max[1] && rref.min[2] <= rref.max[2];
                    // never let the children hold more than twice the node's references -
                    // enforced here so we never have to abandon a partition that already
                    // clipped references
                    const bool dup_budget_ok =
                        int((leftEnd - start) + 1 + (refs.size() - rightStart) + 1) <= 2 * num_prims;

                    const bbox_t lub = {min(lb.min, r.bbox_min), max(lb.max, r.bbox_max)};
                    const bbox_t rub = {min(rb.min, r.bbox_min), max(rb.max, r.bbox_max)};
                    const bbox_t ldb = {min(lb.min, lref.min), max(lb.max, lref.max)};
                    const bbox_t rdb = {min(rb.min, rref.min), max(rb.max, rref.max)};

                    const float lac = float(leftEnd - start), rac = float(refs.size() - rightStart);
                    const float lbc = lac + 1.0f, rbc = rac + 1.0f;

                    const float unsplit_left_sah = lub.surface_area() * lbc + rb.surface_area() * rac;
                    const float unsplit_right_sah = lb.surface_area() * lac + rub.surface_area() * rbc;
                    const float duplicate_sah =
                        dup_valid && dup_budget_ok ? ldb.surface_area() * lbc + rdb.surface_area() * rbc : FLT_MAX;

                    if (unsplit_left_sah <= unsplit_right_sah && unsplit_left_sah <= duplicate_sah) {
                        lb = lub;
                        ++leftEnd;
                    } else if (unsplit_right_sah <= duplicate_sah) {
                        rb = rub;
                        std::swap(refs[leftEnd], refs[--rightStart]);
                    } else {
                        lb = ldb;
                        rb = rdb;
                        refs[leftEnd] = {lref.min, lref.max, r.prim_index};
                        ++leftEnd;
                        refs.push_back({rref.min, rref.max, r.prim_index});
                    }
                }

                if (leftEnd > start && rightStart < refs.size()) {
                    split_result_t result;
                    result.spatial = true;
                    result.left_start = start;
                    result.left_end = leftEnd;
                    result.right_start = rightStart;
                    result.right_end = uint32_t(refs.size());
                    result.left_bounds[0] = lb.min;
                    result.left_bounds[1] = lb.max;
                    result.right_bounds[0] = rb.min;
                    result.right_bounds[1] = rb.max;
                    return result;
                }
                // empty child - no reference was clipped in that case, so falling back to the
                // object split below is safe
            }
        }

        split_result_t result;
        if (div_axis != -1) {
            if (div_axis != 2) {
                std::sort(refs.begin() + start, refs.begin() + end,
                          [div_axis](const ref_t &r1, const ref_t &r2) -> bool {
                              return r1.bbox_max[div_axis] < r2.bbox_max[div_axis];
                          });
            }

            const uint32_t mid = start + div_index;
            result.left_start = start;
            result.left_end = mid;
            result.right_start = mid;
            result.right_end = end;
            // recompute from the actual partition - re-sorting can order ties differently than
            // the sweep did, and node boxes must contain everything below them
            const bbox_t lb = range_bounds(&refs[start], int(div_index));
            const bbox_t rb = range_bounds(&refs[mid], int(end - mid));
            result.left_bounds[0] = lb.min;
            result.left_bounds[1] = lb.max;
            result.right_bounds[0] = rb.min;
            result.right_bounds[1] = rb.max;
            return result;
        }

        if (num_prims > s.min_primitives_in_leaf)
            return median_split(refs, start, end, whole_box);
        return leaf_result();
    }
}
} // namespace Ray

#pragma warning(pop)
