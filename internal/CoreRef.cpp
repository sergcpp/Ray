#include "CoreRef.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <tuple>
#include <utility>

#include "RadCacheRef.h"
#include "TextureStorageCPU.h"

#pragma warning(push)
#pragma warning(disable : 6326) // potential comparison of a constant with another constant

namespace Ray {
#define VECTORIZE_BBOX_INTERSECTION 1
#define VECTORIZE_TRI_INTERSECTION 1
// #define FORCE_TEXTURE_LOD 0
#define USE_STOCH_TEXTURE_FILTERING 1

namespace Ref {
#define sign_of(f) (((f) >= 0) ? 1 : -1)
#define dot(x, y) ((x)[0] * (y)[0] + (x)[1] * (y)[1] + (x)[2] * (y)[2])
force_inline void IntersectTri(const float ro[3], const float rd[3], const tri_accel_t &tri, const uint32_t prim_index,
                               hit_data_t &inter) {
    const float det = dot(rd, tri.n_plane);
    const float dett = tri.n_plane[3] - dot(ro, tri.n_plane);
    if (det == 0.0f || sign_of(dett) != sign_of(det * inter.t - dett)) {
        return;
    }

    const float p[3] = {det * ro[0] + dett * rd[0], det * ro[1] + dett * rd[1], det * ro[2] + dett * rd[2]};

    const float detu = dot(p, tri.u_plane) + det * tri.u_plane[3];
    if (sign_of(detu) != sign_of(det - detu)) {
        return;
    }

    const float detv = dot(p, tri.v_plane) + det * tri.v_plane[3];
    if (sign_of(detv) != sign_of(det - detu - detv)) {
        return;
    }

    const float rdet = (1.0f / det);

    inter.prim_index = (det < 0.0f) ? int(prim_index) : -int(prim_index) - 1;
    inter.t = dett * rdet;
    inter.u = detu * rdet;
    inter.v = detv * rdet;
}
#undef dot
#undef sign_of

force_inline void IntersectTri(const float ro[3], const float rd[3], const mtri_accel_t &tri, const uint32_t prim_index,
                               hit_data_t &inter) {
#if VECTORIZE_TRI_INTERSECTION
    ivec4 _mask = 0, _prim_index = 0;
    fvec4 _t = inter.t, _u = 0.0f, _v = 0.0f;
    for (int i = 0; i < 8; i += 4) {
        fvec4 det = rd[0] * fvec4{&tri.n_plane[0][i], vector_aligned} +
                    rd[1] * fvec4{&tri.n_plane[1][i], vector_aligned} +
                    rd[2] * fvec4{&tri.n_plane[2][i], vector_aligned};
        const fvec4 dett =
            fvec4{&tri.n_plane[3][i], vector_aligned} - ro[0] * fvec4{&tri.n_plane[0][i], vector_aligned} -
            ro[1] * fvec4{&tri.n_plane[1][i], vector_aligned} - ro[2] * fvec4{&tri.n_plane[2][i], vector_aligned};

        // compare sign bits
        ivec4 is_active_lane = ~srai(simd_cast(dett ^ (det * _t - dett)), 31);
        if (is_active_lane.all_zeros()) {
            continue;
        }

        const fvec4 p[3] = {det * ro[0] + dett * rd[0], det * ro[1] + dett * rd[1], det * ro[2] + dett * rd[2]};

        const fvec4 detu =
            p[0] * fvec4{&tri.u_plane[0][i], vector_aligned} + p[1] * fvec4{&tri.u_plane[1][i], vector_aligned} +
            p[2] * fvec4{&tri.u_plane[2][i], vector_aligned} + det * fvec4{&tri.u_plane[3][i], vector_aligned};

        // compare sign bits
        is_active_lane &= ~srai(simd_cast(detu ^ (det - detu)), 31);
        if (is_active_lane.all_zeros()) {
            continue;
        }

        const fvec4 detv =
            p[0] * fvec4{&tri.v_plane[0][i], vector_aligned} + p[1] * fvec4{&tri.v_plane[1][i], vector_aligned} +
            p[2] * fvec4{&tri.v_plane[2][i], vector_aligned} + det * fvec4{&tri.v_plane[3][i], vector_aligned};

        // compare sign bits
        is_active_lane &= ~srai(simd_cast(detv ^ (det - detu - detv)), 31);
        if (is_active_lane.all_zeros()) {
            continue;
        }

        where(~is_active_lane, det) = FLT_EPS;
        const fvec4 rdet = (1.0f / det);

        ivec4 prim = -(int(prim_index) + i + ivec4{0, 1, 2, 3}) - 1;
        where(det < 0.0f, prim) = int(prim_index) + i + ivec4{0, 1, 2, 3};

        _mask |= is_active_lane;
        where(is_active_lane, _prim_index) = prim;
        where(is_active_lane, _t) = dett * rdet;
        where(is_active_lane, _u) = detu * rdet;
        where(is_active_lane, _v) = detv * rdet;
    }

    const float min_t = fminf(_t.get<0>(), fminf(_t.get<1>(), fminf(_t.get<2>(), _t.get<3>())));
    _mask &= simd_cast(_t == min_t);

    const long mask = _mask.movemask();
    if (mask) {
        const long i = GetFirstBit(mask);

        inter.prim_index = _prim_index[i];
        inter.t = _t[i];
        inter.u = _u[i];
        inter.v = _v[i];
    }
#else
#define _sign_of(f) (((f) >= 0) ? 1 : -1)
    for (int i = 0; i < 8; ++i) {
        const float det = rd[0] * tri.n_plane[0][i] + rd[1] * tri.n_plane[1][i] + rd[2] * tri.n_plane[2][i];
        const float dett =
            tri.n_plane[3][i] - ro[0] * tri.n_plane[0][i] - ro[1] * tri.n_plane[1][i] - ro[2] * tri.n_plane[2][i];
        if (_sign_of(dett) != _sign_of(det * inter.t - dett)) {
            continue;
        }

        const float p[3] = {det * ro[0] + dett * rd[0], det * ro[1] + dett * rd[1], det * ro[2] + dett * rd[2]};

        const float detu =
            p[0] * tri.u_plane[0][i] + p[1] * tri.u_plane[1][i] + p[2] * tri.u_plane[2][i] + det * tri.u_plane[3][i];
        if (_sign_of(detu) != _sign_of(det - detu)) {
            continue;
        }

        const float detv =
            p[0] * tri.v_plane[0][i] + p[1] * tri.v_plane[1][i] + p[2] * tri.v_plane[2][i] + det * tri.v_plane[3][i];
        if (_sign_of(detv) != _sign_of(det - detu - detv)) {
            continue;
        }

        const float rdet = (1.0f / det);

        inter.prim_index = (det < 0.0f) ? int(prim_index + i) : -int(prim_index + i) - 1;
        inter.t = dett * rdet;
        inter.u = detu * rdet;
        inter.v = detv * rdet;
    }
#undef _sign_of
#endif
}

force_inline uint32_t near_child(const float rd[3], const bvh_node_t &node) {
    return rd[node.prim_count >> 30] < 0 ? (node.right_child & RIGHT_CHILD_BITS) : node.left_child;
}

force_inline uint32_t far_child(const float rd[3], const bvh_node_t &node) {
    return rd[node.prim_count >> 30] < 0 ? node.left_child : (node.right_child & RIGHT_CHILD_BITS);
}

force_inline uint32_t other_child(const bvh_node_t &node, const uint32_t cur_child) {
    return (node.left_child == cur_child) ? (node.right_child & RIGHT_CHILD_BITS) : node.left_child;
}

force_inline bool is_leaf_node(const bvh_node_t &node) { return (node.prim_index & LEAF_NODE_BIT) != 0; }

force_inline bool is_leaf_node(const wbvh_node_t &node) { return (node.child[0] & LEAF_NODE_BIT) != 0; }

force_inline bool bbox_test(const float o[3], const float inv_d[3], const float t, const float bbox_min[3],
                            const float bbox_max[3], float &out_dist) {
    float lo_x = inv_d[0] * (bbox_min[0] - o[0]);
    float hi_x = inv_d[0] * (bbox_max[0] - o[0]);
    if (lo_x > hi_x) {
        const float tmp = lo_x;
        lo_x = hi_x;
        hi_x = tmp;
    }

    float lo_y = inv_d[1] * (bbox_min[1] - o[1]);
    float hi_y = inv_d[1] * (bbox_max[1] - o[1]);
    if (lo_y > hi_y) {
        const float tmp = lo_y;
        lo_y = hi_y;
        hi_y = tmp;
    }

    float lo_z = inv_d[2] * (bbox_min[2] - o[2]);
    float hi_z = inv_d[2] * (bbox_max[2] - o[2]);
    if (lo_z > hi_z) {
        const float tmp = lo_z;
        lo_z = hi_z;
        hi_z = tmp;
    }

    float tmin = lo_x > lo_y ? lo_x : lo_y;
    if (lo_z > tmin) {
        tmin = lo_z;
    }
    float tmax = hi_x < hi_y ? hi_x : hi_y;
    if (hi_z < tmax) {
        tmax = hi_z;
    }
    tmax *= 1.00000024f;

    out_dist = tmin;

    return tmin <= tmax && tmin <= t && tmax > 0;
}

force_inline bool bbox_test(const float o[3], const float inv_d[3], const float t, const float bbox_min[3],
                            const float bbox_max[3]) {
    float _unused;
    return bbox_test(o, inv_d, t, bbox_min, bbox_max, _unused);
}

force_inline bool bbox_test(const float p[3], const float bbox_min[3], const float bbox_max[3]) {
    return p[0] >= bbox_min[0] && p[0] <= bbox_max[0] && p[1] >= bbox_min[1] && p[1] <= bbox_max[1] &&
           p[2] >= bbox_min[2] && p[2] <= bbox_max[2];
}

force_inline bool bbox_test(const float o[3], const float inv_d[3], const float t, const bvh_node_t &node) {
    return bbox_test(o, inv_d, t, node.bbox_min, node.bbox_max);
}

force_inline bool bbox_test(const float p[3], const bvh_node_t &node) {
    return bbox_test(p, node.bbox_min, node.bbox_max);
}

force_inline long bbox_test_oct(const float p[3], const wbvh_node_t &node) {
    long mask = 0;
    UNROLLED_FOR_R(i, 2, { // NOLINT
        const fvec4 fmask = (fvec4{&node.bbox_min[0][4 * i], vector_aligned} <= p[0]) &
                            (fvec4{&node.bbox_min[1][4 * i], vector_aligned} <= p[1]) &
                            (fvec4{&node.bbox_min[2][4 * i], vector_aligned} <= p[2]) &
                            (fvec4{&node.bbox_max[0][4 * i], vector_aligned} >= p[0]) &
                            (fvec4{&node.bbox_max[1][4 * i], vector_aligned} >= p[1]) &
                            (fvec4{&node.bbox_max[2][4 * i], vector_aligned} >= p[2]);
        mask <<= 4;
        mask |= simd_cast(fmask).movemask();
    })
    return mask;
}

force_inline long bbox_test_oct(const float p[3], const cwbvh_node_t &node) {
    // Unpack bounds
    const float ext[3] = {(node.bbox_max[0] - node.bbox_min[0]) / 255.0f,
                          (node.bbox_max[1] - node.bbox_min[1]) / 255.0f,
                          (node.bbox_max[2] - node.bbox_min[2]) / 255.0f};
    alignas(16) float bbox_min[3][8], bbox_max[3][8];
    for (int i = 0; i < 8; ++i) {
        bbox_min[0][i] = bbox_min[1][i] = bbox_min[2][i] = -MAX_DIST;
        bbox_max[0][i] = bbox_max[1][i] = bbox_max[2][i] = MAX_DIST;
        if (node.ch_bbox_min[0][i] != 0xff || node.ch_bbox_max[0][i] != 0) {
            bbox_min[0][i] = node.bbox_min[0] + node.ch_bbox_min[0][i] * ext[0];
            bbox_min[1][i] = node.bbox_min[1] + node.ch_bbox_min[1][i] * ext[1];
            bbox_min[2][i] = node.bbox_min[2] + node.ch_bbox_min[2][i] * ext[2];

            bbox_max[0][i] = node.bbox_min[0] + node.ch_bbox_max[0][i] * ext[0];
            bbox_max[1][i] = node.bbox_min[1] + node.ch_bbox_max[1][i] * ext[1];
            bbox_max[2][i] = node.bbox_min[2] + node.ch_bbox_max[2][i] * ext[2];
        }
    }

    long mask = 0;
    UNROLLED_FOR_R(i, 2, { // NOLINT
        const fvec4 fmask = (fvec4{&bbox_min[0][4 * i], vector_aligned} <= p[0]) &
                            (fvec4{&bbox_min[1][4 * i], vector_aligned} <= p[1]) &
                            (fvec4{&bbox_min[2][4 * i], vector_aligned} <= p[2]) &
                            (fvec4{&bbox_max[0][4 * i], vector_aligned} >= p[0]) &
                            (fvec4{&bbox_max[1][4 * i], vector_aligned} >= p[1]) &
                            (fvec4{&bbox_max[2][4 * i], vector_aligned} >= p[2]);
        mask <<= 4;
        mask |= simd_cast(fmask).movemask();
    })
    return mask;
}

force_inline void bbox_test_oct(const float o[3], const float inv_d[3], const wbvh_node_t &node, int res[8],
                                float dist[8]){
    UNROLLED_FOR(i, 8,
                 { // NOLINT
                     float lo_x = inv_d[0] * (node.bbox_min[0][i] - o[0]);
                     float hi_x = inv_d[0] * (node.bbox_max[0][i] - o[0]);
                     if (lo_x > hi_x) {
                         const float tmp = lo_x;
                         lo_x = hi_x;
                         hi_x = tmp;
                     }

                     float lo_y = inv_d[1] * (node.bbox_min[1][i] - o[1]);
                     float hi_y = inv_d[1] * (node.bbox_max[1][i] - o[1]);
                     if (lo_y > hi_y) {
                         const float tmp = lo_y;
                         lo_y = hi_y;
                         hi_y = tmp;
                     }

                     float lo_z = inv_d[2] * (node.bbox_min[2][i] - o[2]);
                     float hi_z = inv_d[2] * (node.bbox_max[2][i] - o[2]);
                     if (lo_z > hi_z) {
                         const float tmp = lo_z;
                         lo_z = hi_z;
                         hi_z = tmp;
                     }

                     float tmin = lo_x > lo_y ? lo_x : lo_y;
                     if (lo_z > tmin) {
                         tmin = lo_z;
                     }
                     float tmax = hi_x < hi_y ? hi_x : hi_y;
                     if (hi_z < tmax) {
                         tmax = hi_z;
                     }
                     tmax *= 1.00000024f;

                     dist[i] = tmin;
                     res[i] = (tmin <= tmax && tmax > 0) ? 1 : 0;
                 }) // NOLINT
}

force_inline long bbox_test_oct(const float o[3], const float inv_d[3], const float t, const wbvh_node_t &node,
                                float out_dist[8]) {
    long mask = 0;
#if VECTORIZE_BBOX_INTERSECTION
    fvec4 lo, hi, tmin, tmax;
    UNROLLED_FOR_R(i, 2, { // NOLINT
        lo = inv_d[0] * (fvec4{&node.bbox_min[0][4 * i], vector_aligned} - o[0]);
        hi = inv_d[0] * (fvec4{&node.bbox_max[0][4 * i], vector_aligned} - o[0]);
        tmin = min(lo, hi);
        tmax = max(lo, hi);

        lo = inv_d[1] * (fvec4{&node.bbox_min[1][4 * i], vector_aligned} - o[1]);
        hi = inv_d[1] * (fvec4{&node.bbox_max[1][4 * i], vector_aligned} - o[1]);
        tmin = max(tmin, min(lo, hi));
        tmax = min(tmax, max(lo, hi));

        lo = inv_d[2] * (fvec4{&node.bbox_min[2][4 * i], vector_aligned} - o[2]);
        hi = inv_d[2] * (fvec4{&node.bbox_max[2][4 * i], vector_aligned} - o[2]);
        tmin = max(tmin, min(lo, hi));
        tmax = min(tmax, max(lo, hi));
        tmax *= 1.00000024f;

        const fvec4 fmask = (tmin <= tmax) & (tmin <= t) & (tmax > 0.0f);
        mask <<= 4;
        mask |= simd_cast(fmask).movemask();
        tmin.store_to(&out_dist[4 * i], vector_aligned);
    }) // NOLINT
#else
    UNROLLED_FOR(i, 8, { // NOLINT
        float lo_x = inv_d[0] * (node.bbox_min[0][i] - o[0]);
        float hi_x = inv_d[0] * (node.bbox_max[0][i] - o[0]);
        if (lo_x > hi_x) {
            const float tmp = lo_x;
            lo_x = hi_x;
            hi_x = tmp;
        }

        float lo_y = inv_d[1] * (node.bbox_min[1][i] - o[1]);
        float hi_y = inv_d[1] * (node.bbox_max[1][i] - o[1]);
        if (lo_y > hi_y) {
            const float tmp = lo_y;
            lo_y = hi_y;
            hi_y = tmp;
        }

        float lo_z = inv_d[2] * (node.bbox_min[2][i] - o[2]);
        float hi_z = inv_d[2] * (node.bbox_max[2][i] - o[2]);
        if (lo_z > hi_z) {
            const float tmp = lo_z;
            lo_z = hi_z;
            hi_z = tmp;
        }

        float tmin = lo_x > lo_y ? lo_x : lo_y;
        if (lo_z > tmin) {
            tmin = lo_z;
        }
        float tmax = hi_x < hi_y ? hi_x : hi_y;
        if (hi_z < tmax) {
            tmax = hi_z;
        }
        tmax *= 1.00000024f;

        out_dist[i] = tmin;
        mask |= ((tmin <= tmax && tmin <= t && tmax > 0) ? 1 : 0) << i;
    }) // NOLINT
#endif
    return mask;
}

force_inline long bbox_test_oct(const float o[3], const float inv_d[3], const float t, const cwbvh_node_t &node,
                                float out_dist[8]) {
    // Unpack bounds
    const float ext[3] = {(node.bbox_max[0] - node.bbox_min[0]) / 255.0f,
                          (node.bbox_max[1] - node.bbox_min[1]) / 255.0f,
                          (node.bbox_max[2] - node.bbox_min[2]) / 255.0f};
    alignas(16) float bbox_min[3][8], bbox_max[3][8];
    for (int i = 0; i < 8; ++i) {
        bbox_min[0][i] = bbox_min[1][i] = bbox_min[2][i] = -MAX_DIST;
        bbox_max[0][i] = bbox_max[1][i] = bbox_max[2][i] = MAX_DIST;
        if (node.ch_bbox_min[0][i] != 0xff || node.ch_bbox_max[0][i] != 0) {
            bbox_min[0][i] = node.bbox_min[0] + node.ch_bbox_min[0][i] * ext[0];
            bbox_min[1][i] = node.bbox_min[1] + node.ch_bbox_min[1][i] * ext[1];
            bbox_min[2][i] = node.bbox_min[2] + node.ch_bbox_min[2][i] * ext[2];

            bbox_max[0][i] = node.bbox_min[0] + node.ch_bbox_max[0][i] * ext[0];
            bbox_max[1][i] = node.bbox_min[1] + node.ch_bbox_max[1][i] * ext[1];
            bbox_max[2][i] = node.bbox_min[2] + node.ch_bbox_max[2][i] * ext[2];
        }
    }

    long mask = 0;
#if VECTORIZE_BBOX_INTERSECTION
    fvec4 lo, hi, tmin, tmax;
    UNROLLED_FOR_R(i, 2, { // NOLINT
        lo = inv_d[0] * (fvec4{&bbox_min[0][4 * i], vector_aligned} - o[0]);
        hi = inv_d[0] * (fvec4{&bbox_max[0][4 * i], vector_aligned} - o[0]);
        tmin = min(lo, hi);
        tmax = max(lo, hi);

        lo = inv_d[1] * (fvec4{&bbox_min[1][4 * i], vector_aligned} - o[1]);
        hi = inv_d[1] * (fvec4{&bbox_max[1][4 * i], vector_aligned} - o[1]);
        tmin = max(tmin, min(lo, hi));
        tmax = min(tmax, max(lo, hi));

        lo = inv_d[2] * (fvec4{&bbox_min[2][4 * i], vector_aligned} - o[2]);
        hi = inv_d[2] * (fvec4{&bbox_max[2][4 * i], vector_aligned} - o[2]);
        tmin = max(tmin, min(lo, hi));
        tmax = min(tmax, max(lo, hi));
        tmax *= 1.00000024f;

        const fvec4 fmask = (tmin <= tmax) & (tmin <= t) & (tmax > 0.0f);
        mask <<= 4;
        mask |= simd_cast(fmask).movemask();
        tmin.store_to(&out_dist[4 * i], vector_aligned);
    }) // NOLINT
#else
    UNROLLED_FOR(i, 8, { // NOLINT
        float lo_x = inv_d[0] * (bbox_min[0][i] - o[0]);
        float hi_x = inv_d[0] * (bbox_max[0][i] - o[0]);
        if (lo_x > hi_x) {
            const float tmp = lo_x;
            lo_x = hi_x;
            hi_x = tmp;
        }

        float lo_y = inv_d[1] * (bbox_min[1][i] - o[1]);
        float hi_y = inv_d[1] * (bbox_max[1][i] - o[1]);
        if (lo_y > hi_y) {
            const float tmp = lo_y;
            lo_y = hi_y;
            hi_y = tmp;
        }

        float lo_z = inv_d[2] * (bbox_min[2][i] - o[2]);
        float hi_z = inv_d[2] * (bbox_max[2][i] - o[2]);
        if (lo_z > hi_z) {
            const float tmp = lo_z;
            lo_z = hi_z;
            hi_z = tmp;
        }

        float tmin = lo_x > lo_y ? lo_x : lo_y;
        if (lo_z > tmin) {
            tmin = lo_z;
        }
        float tmax = hi_x < hi_y ? hi_x : hi_y;
        if (hi_z < tmax) {
            tmax = hi_z;
        }
        tmax *= 1.00000024f;

        out_dist[i] = tmin;
        mask |= ((tmin <= tmax && tmin <= t && tmax > 0) ? 1 : 0) << i;
    }) // NOLINT
#endif
    return mask;
}

struct stack_entry_t {
    uint32_t index;
    float dist;
};

struct light_stack_entry_t {
    uint32_t index;
    float dist;
    float factor;
};

template <int StackSize, typename T = stack_entry_t> struct TraversalStack {
    T stack[StackSize];
    uint32_t stack_size = 0;

    template <class... Args> force_inline void push(Args &&...args) {
        stack[stack_size++] = {std::forward<Args>(args)...};
        assert(stack_size < StackSize && "Traversal stack overflow!");
    }

    force_inline T pop() { return stack[--stack_size]; }

    force_inline uint32_t pop_index() { return stack[--stack_size].index; }

    force_inline bool empty() const { return stack_size == 0; }

    void sort_top3() {
        assert(stack_size >= 3);
        const uint32_t i = stack_size - 3;

        if (stack[i].dist > stack[i + 1].dist) {
            if (stack[i + 1].dist > stack[i + 2].dist) {
                return;
            } else if (stack[i].dist > stack[i + 2].dist) {
                std::swap(stack[i + 1], stack[i + 2]);
            } else {
                T tmp = stack[i];
                stack[i] = stack[i + 2];
                stack[i + 2] = stack[i + 1];
                stack[i + 1] = tmp;
            }
        } else {
            if (stack[i].dist > stack[i + 2].dist) {
                std::swap(stack[i], stack[i + 1]);
            } else if (stack[i + 2].dist > stack[i + 1].dist) {
                std::swap(stack[i], stack[i + 2]);
            } else {
                const T tmp = stack[i];
                stack[i] = stack[i + 1];
                stack[i + 1] = stack[i + 2];
                stack[i + 2] = tmp;
            }
        }

        assert(stack[stack_size - 3].dist >= stack[stack_size - 2].dist &&
               stack[stack_size - 2].dist >= stack[stack_size - 1].dist);
    }

    void sort_top4() {
        assert(stack_size >= 4);
        const uint32_t i = stack_size - 4;

        if (stack[i + 0].dist < stack[i + 1].dist) {
            std::swap(stack[i + 0], stack[i + 1]);
        }
        if (stack[i + 2].dist < stack[i + 3].dist) {
            std::swap(stack[i + 2], stack[i + 3]);
        }
        if (stack[i + 0].dist < stack[i + 2].dist) {
            std::swap(stack[i + 0], stack[i + 2]);
        }
        if (stack[i + 1].dist < stack[i + 3].dist) {
            std::swap(stack[i + 1], stack[i + 3]);
        }
        if (stack[i + 1].dist < stack[i + 2].dist) {
            std::swap(stack[i + 1], stack[i + 2]);
        }

        assert(stack[stack_size - 4].dist >= stack[stack_size - 3].dist &&
               stack[stack_size - 3].dist >= stack[stack_size - 2].dist &&
               stack[stack_size - 2].dist >= stack[stack_size - 1].dist);
    }

    void sort_topN(const int count) {
        assert(stack_size >= uint32_t(count));
        const int start = int(stack_size - count);

        for (int i = start + 1; i < int(stack_size); ++i) {
            const T key = stack[i];

            int j = i - 1;

            while (j >= start && stack[j].dist < key.dist) {
                stack[j + 1] = stack[j];
                j--;
            }

            stack[j + 1] = key;
        }

#ifndef NDEBUG
        for (int j = 0; j < count - 1; j++) {
            assert(stack[stack_size - count + j].dist >= stack[stack_size - count + j + 1].dist);
        }
#endif
    }
};

force_inline int clamp(const int val, const int min, const int max) {
    return val < min ? min : (val > max ? max : val);
}

force_inline uint32_t get_ray_hash(const ray_data_t &r, const float root_min[3], const float cell_size[3]) {
    int x = clamp(int((r.o[0] - root_min[0]) / cell_size[0]), 0, 255),
        y = clamp(int((r.o[1] - root_min[1]) / cell_size[1]), 0, 255),
        z = clamp(int((r.o[2] - root_min[2]) / cell_size[2]), 0, 255);

    // float omega = omega_table[int(r.d[2] / 0.0625f)];
    // float atan2f(r.d[1], r.d[0]);
    // int o = int(16 * omega / (PI)), p = int(16 * (phi + PI) / (2 * PI));

    x = morton_table_256[x];
    y = morton_table_256[y];
    z = morton_table_256[z];

    const int o = morton_table_16[int(omega_table[clamp(int((1.0f + r.d[2]) / omega_step), 0, 32)])];
    const int p = morton_table_16[int(
        phi_table[clamp(int((1.0f + r.d[1]) / phi_step), 0, 16)][clamp(int((1.0f + r.d[0]) / phi_step), 0, 16)])];

    return (o << 25) | (p << 24) | (y << 2) | (z << 1) | (x << 0);
}

force_inline void _radix_sort_lsb(ray_chunk_t *begin, ray_chunk_t *end, ray_chunk_t *begin1, unsigned maxshift) {
    ray_chunk_t *end1 = begin1 + (end - begin);

    for (unsigned shift = 0; shift <= maxshift; shift += 8) {
        size_t count[0x100] = {};
        for (ray_chunk_t *p = begin; p != end; p++) {
            count[(p->hash >> shift) & 0xFF]++;
        }
        ray_chunk_t *bucket[0x100], *q = begin1;
        for (int i = 0; i < 0x100; q += count[i++]) {
            bucket[i] = q;
        }
        for (ray_chunk_t *p = begin; p != end; p++) {
            *bucket[(p->hash >> shift) & 0xFF]++ = *p;
        }
        std::swap(begin, begin1);
        std::swap(end, end1);
    }
}

force_inline void radix_sort(ray_chunk_t *begin, ray_chunk_t *end, ray_chunk_t *begin1) {
    _radix_sort_lsb(begin, end, begin1, 24);
}

void create_tbn_matrix(const fvec4 &N, fvec4 out_TBN[3]) {
    fvec4 U;
    if (fabsf(N.get<1>()) < 0.999f) {
        U = {0.0f, 1.0f, 0.0f, 0.0f};
    } else {
        U = {1.0f, 0.0f, 0.0f, 0.0f};
    }

    fvec4 T = normalize(cross(U, N));
    U = cross(N, T);

    out_TBN[0].set<0>(T.get<0>());
    out_TBN[1].set<0>(T.get<1>());
    out_TBN[2].set<0>(T.get<2>());

    out_TBN[0].set<1>(U.get<0>());
    out_TBN[1].set<1>(U.get<1>());
    out_TBN[2].set<1>(U.get<2>());

    out_TBN[0].set<2>(N.get<0>());
    out_TBN[1].set<2>(N.get<1>());
    out_TBN[2].set<2>(N.get<2>());
}

void create_tbn_matrix(const fvec4 &N, fvec4 &T, fvec4 out_TBN[3]) {
    fvec4 U = normalize(cross(T, N));
    T = cross(N, U);

    out_TBN[0].set<0>(T.get<0>());
    out_TBN[1].set<0>(T.get<1>());
    out_TBN[2].set<0>(T.get<2>());

    out_TBN[0].set<1>(U.get<0>());
    out_TBN[1].set<1>(U.get<1>());
    out_TBN[2].set<1>(U.get<2>());

    out_TBN[0].set<2>(N.get<0>());
    out_TBN[1].set<2>(N.get<1>());
    out_TBN[2].set<2>(N.get<2>());
}

void create_tbn(const fvec4 &N, fvec4 &out_T, fvec4 &out_B) {
    fvec4 U;
    if (fabsf(N.get<1>()) < 0.999f) {
        U = {0.0f, 1.0f, 0.0f, 0.0f};
    } else {
        U = {1.0f, 0.0f, 0.0f, 0.0f};
    }

    out_T = normalize(cross(U, N));
    out_B = cross(N, out_T);
}

fvec4 map_to_cone(float r1, float r2, fvec4 N, float radius) {
    const fvec2 offset = 2.0f * fvec2(r1, r2) - fvec2(1.0f);
    if (offset.get<0>() == 0.0f && offset.get<1>() == 0.0f) {
        return N;
    }

    float theta, r;

    if (fabsf(offset.get<0>()) > fabsf(offset.get<1>())) {
        r = offset.get<0>();
        theta = 0.25f * PI * (offset.get<1>() / offset.get<0>());
    } else {
        r = offset.get<1>();
        theta = 0.5f * PI * (1.0f - 0.5f * (offset.get<0>() / offset.get<1>()));
    }

    const fvec2 uv = fvec2(radius * r * cosf(theta), radius * r * sinf(theta));

    fvec4 LT, LB;
    create_tbn(N, LT, LB);

    return N + uv.get<0>() * LT + uv.get<1>() * LB;
}

force_inline float sphere_intersection(const fvec4 &center, const float radius, const fvec4 &ro, const fvec4 &rd) {
    const fvec4 oc = ro - center;
    const float a = dot(rd, rd);
    const float b = 2 * dot(oc, rd);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = b * b - 4 * a * c;
    return (-b - sqrtf(fmaxf(discriminant, 0.0f))) / (2 * a);
}

void transpose(const fvec3 in_3x3[3], fvec3 out_3x3[3]) {
    out_3x3[0].set<0>(in_3x3[0].get<0>());
    out_3x3[0].set<1>(in_3x3[1].get<0>());
    out_3x3[0].set<2>(in_3x3[2].get<0>());

    out_3x3[1].set<0>(in_3x3[0].get<1>());
    out_3x3[1].set<1>(in_3x3[1].get<1>());
    out_3x3[1].set<2>(in_3x3[2].get<1>());

    out_3x3[2].set<0>(in_3x3[0].get<2>());
    out_3x3[2].set<1>(in_3x3[1].get<2>());
    out_3x3[2].set<2>(in_3x3[2].get<2>());
}

fvec3 mul(const fvec3 in_mat[3], const fvec3 &in_vec) {
    fvec3 out_vec;
    out_vec.set<0>(in_mat[0].get<0>() * in_vec.get<0>() + in_mat[1].get<0>() * in_vec.get<1>() +
                   in_mat[2].get<0>() * in_vec.get<2>());
    out_vec.set<1>(in_mat[0].get<1>() * in_vec.get<0>() + in_mat[1].get<1>() * in_vec.get<1>() +
                   in_mat[2].get<1>() * in_vec.get<2>());
    out_vec.set<2>(in_mat[0].get<2>() * in_vec.get<0>() + in_mat[1].get<2>() * in_vec.get<1>() +
                   in_mat[2].get<2>() * in_vec.get<2>());
    return out_vec;
}

force_inline bool quadratic(float a, float b, float c, float &t0, float &t1) {
    const float d = b * b - 4.0f * a * c;
    if (d < 0.0f) {
        return false;
    }
    const float sqrt_d = sqrtf(d);
    float q;
    if (b < 0.0f) {
        q = -0.5f * (b - sqrt_d);
    } else {
        q = -0.5f * (b + sqrt_d);
    }
    t0 = q / a;
    t1 = c / q;
    return true;
}

force_inline float ngon_rad(const float theta, const float n) {
    return cosf(PI / n) / cosf(theta - (2.0f * PI / n) * floorf((n * theta + PI) / (2.0f * PI)));
}

float approx_atan2(const float y, const float x) { // max error is 0.000004f
    float t0, t1, t3, t4;

    t3 = fabsf(x);
    t1 = fabsf(y);
    t0 = fmaxf(t3, t1);
    t1 = fminf(t3, t1);
    t3 = 1.0f / t0;
    t3 = t1 * t3;

    t4 = t3 * t3;
    t0 = -0.013480470f;
    t0 = t0 * t4 + 0.057477314f;
    t0 = t0 * t4 - 0.121239071f;
    t0 = t0 * t4 + 0.195635925f;
    t0 = t0 * t4 - 0.332994597f;
    t0 = t0 * t4 + 0.999995630f;
    t3 = t0 * t3;

    t3 = (fabsf(y) > fabsf(x)) ? 1.570796327f - t3 : t3;
    t3 = (x < 0) ? 3.141592654f - t3 : t3;
    t3 = (y < 0) ? -t3 : t3;

    return t3;
}

fvec4 approx_atan2(const fvec4 y, const fvec4 x) {
    fvec4 t0, t1, t3, t4;

    t3 = abs(x);
    t1 = abs(y);
    t0 = max(t3, t1);
    t1 = min(t3, t1);
    t3 = 1.0f / t0;
    t3 = t1 * t3;

    t4 = t3 * t3;
    t0 = -0.013480470f;
    t0 = t0 * t4 + 0.057477314f;
    t0 = t0 * t4 - 0.121239071f;
    t0 = t0 * t4 + 0.195635925f;
    t0 = t0 * t4 - 0.332994597f;
    t0 = t0 * t4 + 0.999995630f;
    t3 = t0 * t3;

    where(abs(y) > abs(x), t3) = 1.570796327f - t3;
    where(x < 0.0f, t3) = 3.141592654f - t3;
    where(y < 0.0f, t3) = -t3;

    return t3;
}

force_inline float approx_cos(float x) { // max error is 0.056010f
    const float tp = 1.0f / (2.0f * PI);
    x *= tp;
    x -= 0.25f + floorf(x + 0.25f);
    x *= 16.0f * (fabsf(x) - 0.5f);
    return x;
}

force_inline fvec4 approx_cos(fvec4 x) {
    const float tp = 1.0f / (2.0f * PI);
    x *= tp;
    x -= 0.25f + floor(x + 0.25f);
    x *= 16.0f * (abs(x) - 0.5f);
    return x;
}

force_inline float approx_acos(float x) { // max error is 0.000068f
    float negate = float(x < 0);
    x = fabsf(x);
    float ret = -0.0187293f;
    ret = ret * x;
    ret = ret + 0.0742610f;
    ret = ret * x;
    ret = ret - 0.2121144f;
    ret = ret * x;
    ret = ret + 1.5707288f;
    ret = ret * sqrtf(1.0f - x);
    ret = ret - 2 * negate * ret;
    return negate * PI + ret;
}

force_inline fvec4 approx_acos(fvec4 x) {
    fvec4 negate = 0.0f;
    where(x < 0.0f, negate) = 1.0f;
    x = abs(x);
    fvec4 ret = -0.0187293f;
    ret = ret * x;
    ret = ret + 0.0742610f;
    ret = ret * x;
    ret = ret - 0.2121144f;
    ret = ret * x;
    ret = ret + 1.5707288f;
    ret = ret * sqrt(1.0f - min(x, 1.0f));
    ret = ret - 2 * negate * ret;
    return negate * PI + ret;
}

float calc_lnode_importance(const light_bvh_node_t &n, const fvec4 &P) {
    float mul = 1.0f, v_len2 = 1.0f;
    if (n.bbox_min[0] > -MAX_DIST) { // check if this is a local light
        fvec4 v = P - 0.5f * (fvec4{n.bbox_min} + fvec4{n.bbox_max});
        v.set<3>(0.0f);

        fvec4 ext = fvec4{n.bbox_max} - fvec4{n.bbox_min};
        ext.set<3>(0.0f);

        const float extent = 0.5f * length(ext);
        v_len2 = dot(v, v);
        const float v_len = sqrtf(v_len2);
        const float omega_u = approx_atan2(extent, v_len) + 0.000005f;

        fvec4 axis = fvec4{n.axis};
        axis.set<3>(0.0f);

        const float omega = approx_acos(fminf(dot(axis, v / v_len), 1.0f)) - 0.00007f;
        const float omega_ = fmaxf(0.0f, omega - n.omega_n - omega_u);
        mul = omega_ < n.omega_e ? approx_cos(omega_) + 0.057f : 0.0f;
    }

    // TODO: account for normal dot product here
    return n.flux * mul / v_len2;
}

force_inline fvec4 dot3(const fvec4 v1[3], const fvec4 v2[3]) { return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]; }

force_inline fvec4 length(const fvec4 v[3]) { return sqrt(dot3(v, v)); }

force_inline fvec4 cos_sub_clamped(const fvec4 &sin_omega_a, const fvec4 &cos_omega_a, const fvec4 &sin_omega_b,
                                   const fvec4 &cos_omega_b) {
    fvec4 ret = cos_omega_a * cos_omega_b + sin_omega_a * sin_omega_b;
    where(cos_omega_a > cos_omega_b, ret) = 1.0f;
    return ret;
}

force_inline fvec4 sin_sub_clamped(const fvec4 &sin_omega_a, const fvec4 &cos_omega_a, const fvec4 &sin_omega_b,
                                   const fvec4 &cos_omega_b) {
    fvec4 ret = sin_omega_a * cos_omega_b - cos_omega_a * sin_omega_b;
    where(cos_omega_a > cos_omega_b, ret) = 0.0f;
    return ret;
}

force_inline fvec4 decode_oct_dir(const uint32_t oct) {
    fvec4 ret = 0.0f;
    ret.set<0>(-1.0f + 2.0f * float((oct >> 16) & 0x0000ffff) / 65535.0f);
    ret.set<1>(-1.0f + 2.0f * float(oct & 0x0000ffff) / 65535.0f);
    ret.set<2>(1.0f - fabsf(ret.get<0>()) - fabsf(ret.get<1>()));
    if (ret.get<2>() < 0.0f) {
        const float temp = ret.get<0>();
        ret.set<0>((1.0f - fabsf(ret.get<1>())) * copysignf(1.0f, temp));
        ret.set<1>((1.0f - fabsf(temp)) * copysignf(1.0f, ret.get<1>()));
    }
    return normalize(ret);
}

force_inline fvec4 normalize(std::array<fvec4, 3> &v) {
    const fvec4 l = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] = (v[0] / l);
    v[1] = (v[1] / l);
    v[2] = (v[2] / l);
    return l;
}

force_inline std::array<fvec4, 3> decode_oct_dir(const uvec4 &oct) {
    const uvec4 x = (oct >> 16) & 0x0000ffff;
    const uvec4 y = oct & 0x0000ffff;
    std::array<fvec4, 3> out_dir;
    out_dir[0] = -1.0f + 2.0f * fvec4(x) / 65535.0f;
    out_dir[1] = -1.0f + 2.0f * fvec4(y) / 65535.0f;
    out_dir[2] = 1.0f - abs(out_dir[0]) - abs(out_dir[1]);
    const fvec4 temp = out_dir[0];
    where(out_dir[2] < 0.0f, out_dir[0]) = (1.0f - abs(out_dir[1])) * copysign(1.0f, temp);
    where(out_dir[2] < 0.0f, out_dir[1]) = (1.0f - abs(temp)) * copysign(1.0f, out_dir[1]);
    normalize(out_dir);
    return out_dir;
}

force_inline std::array<fvec4, 2> decode_cosines(const uvec4 &val) {
    const uvec4 a = (val >> 16) & 0x0000ffff;
    const uvec4 b = (val & 0x0000ffff);
    std::array<fvec4, 2> ret;
    ret[0] = 2.0f * (fvec4(a) / 65534.0f) - 1.0f;
    ret[1] = 2.0f * (fvec4(b) / 65534.0f) - 1.0f;
    return ret;
}

void calc_lnode_importance(const light_wbvh_node_t &n, const fvec4 &P, float importance[8]) {
    for (int i = 0; i < 8; i += 4) {
        fvec4 mul = 1.0f, v_len2 = 1.0f;

        const ivec4 mask = simd_cast(fvec4{&n.bbox_min[0][i], vector_aligned} > -MAX_DIST);
        if (mask.not_all_zeros()) {
            const std::array<fvec4, 3> axis = decode_oct_dir(uvec4{&n.axis[i], vector_aligned});
            const fvec4 ext[3] = {fvec4{&n.bbox_max[0][i], vector_aligned} - fvec4{&n.bbox_min[0][i], vector_aligned},
                                  fvec4{&n.bbox_max[1][i], vector_aligned} - fvec4{&n.bbox_min[1][i], vector_aligned},
                                  fvec4{&n.bbox_max[2][i], vector_aligned} - fvec4{&n.bbox_min[2][i], vector_aligned}};
            const fvec4 extent = 0.5f * length(ext);

            const fvec4 pc[3] = {
                0.5f * (fvec4{&n.bbox_min[0][i], vector_aligned} + fvec4{&n.bbox_max[0][i], vector_aligned}),
                0.5f * (fvec4{&n.bbox_min[1][i], vector_aligned} + fvec4{&n.bbox_max[1][i], vector_aligned}),
                0.5f * (fvec4{&n.bbox_min[2][i], vector_aligned} + fvec4{&n.bbox_max[2][i], vector_aligned})};
            fvec4 wi[3] = {P.get<0>() - pc[0], P.get<1>() - pc[1], P.get<2>() - pc[2]};
            const fvec4 dist2 = dot3(wi, wi);
            const fvec4 dist = sqrt(dist2);
            UNROLLED_FOR(j, 3, { wi[j] /= dist; })

            where(mask, v_len2) = max(dist2, extent);

            const fvec4 cos_omega_w = dot3(axis.data(), wi);
            const fvec4 sin_omega_w = sqrt(max(1.0f - cos_omega_w * cos_omega_w, 0.0f));

            fvec4 cos_omega_b = sqrt(max(1.0f - (extent * extent) / dist2, 0.0f));
            where(dist2 < extent * extent, cos_omega_b) = -1.0f;
            const fvec4 sin_omega_b = sqrt(1.0f - cos_omega_b * cos_omega_b);

            const std::array<fvec4, 2> cos_omega_ne = decode_cosines(uvec4{&n.cos_omega_ne[i], vector_aligned});
            const fvec4 sin_omega_n = sqrt(1.0f - cos_omega_ne[0] * cos_omega_ne[0]);

            const fvec4 cos_omega_x = cos_sub_clamped(sin_omega_w, cos_omega_w, sin_omega_n, cos_omega_ne[0]);
            const fvec4 sin_omega_x = sin_sub_clamped(sin_omega_w, cos_omega_w, sin_omega_n, cos_omega_ne[0]);
            const fvec4 cos_omega = cos_sub_clamped(sin_omega_x, cos_omega_x, sin_omega_b, cos_omega_b);

            where(mask, mul) = 0.0f;
            where(mask & simd_cast(cos_omega > cos_omega_ne[1]), mul) = cos_omega;
        }

        const fvec4 imp = fvec4{&n.flux[i], vector_aligned} * mul / v_len2;
        imp.store_to(&importance[i], vector_aligned);
    }
}

void calc_lnode_importance(const light_cwbvh_node_t &n, const fvec4 &P, float importance[8]) {
    alignas(16) float bbox_min[3][8], bbox_max[3][8];
    { // Unpack bounds
        const float ext[3] = {(n.bbox_max[0] - n.bbox_min[0]) / 255.0f, (n.bbox_max[1] - n.bbox_min[1]) / 255.0f,
                              (n.bbox_max[2] - n.bbox_min[2]) / 255.0f};
        for (int i = 0; i < 8; ++i) {
            bbox_min[0][i] = bbox_min[1][i] = bbox_min[2][i] = -MAX_DIST;
            bbox_max[0][i] = bbox_max[1][i] = bbox_max[2][i] = MAX_DIST;
            if (n.ch_bbox_min[0][i] != 0xff || n.ch_bbox_max[0][i] != 0) {
                bbox_min[0][i] = n.bbox_min[0] + n.ch_bbox_min[0][i] * ext[0];
                bbox_min[1][i] = n.bbox_min[1] + n.ch_bbox_min[1][i] * ext[1];
                bbox_min[2][i] = n.bbox_min[2] + n.ch_bbox_min[2][i] * ext[2];

                bbox_max[0][i] = n.bbox_min[0] + n.ch_bbox_max[0][i] * ext[0];
                bbox_max[1][i] = n.bbox_min[1] + n.ch_bbox_max[1][i] * ext[1];
                bbox_max[2][i] = n.bbox_min[2] + n.ch_bbox_max[2][i] * ext[2];
            }
        }
    }

    for (int i = 0; i < 8; i += 4) {
        fvec4 imp = fvec4{&n.flux[i], vector_aligned};

        const ivec4 mask = simd_cast(fvec4{&bbox_min[0][i], vector_aligned} > -MAX_DIST);
        if (mask.not_all_zeros()) {
            const std::array<fvec4, 3> axis = decode_oct_dir(uvec4{&n.axis[i], vector_aligned});
            const fvec4 ext[3] = {fvec4{&bbox_max[0][i], vector_aligned} - fvec4{&bbox_min[0][i], vector_aligned},
                                  fvec4{&bbox_max[1][i], vector_aligned} - fvec4{&bbox_min[1][i], vector_aligned},
                                  fvec4{&bbox_max[2][i], vector_aligned} - fvec4{&bbox_min[2][i], vector_aligned}};
            const fvec4 extent = 0.5f * length(ext);

            const fvec4 pc[3] = {
                0.5f * (fvec4{&bbox_min[0][i], vector_aligned} + fvec4{&bbox_max[0][i], vector_aligned}),
                0.5f * (fvec4{&bbox_min[1][i], vector_aligned} + fvec4{&bbox_max[1][i], vector_aligned}),
                0.5f * (fvec4{&bbox_min[2][i], vector_aligned} + fvec4{&bbox_max[2][i], vector_aligned})};
            fvec4 wi[3] = {P.get<0>() - pc[0], P.get<1>() - pc[1], P.get<2>() - pc[2]};
            const fvec4 dist2 = dot3(wi, wi);
            const fvec4 dist = sqrt(dist2);
            UNROLLED_FOR(j, 3, { wi[j] /= dist; })

            const fvec4 v_len2 = max(dist2, extent);

            const fvec4 cos_omega_w = dot3(axis.data(), wi);
            const fvec4 sin_omega_w = sqrt(max(1.0f - cos_omega_w * cos_omega_w, 0.0f));

            fvec4 cos_omega_b = sqrt(max(1.0f - (extent * extent) / dist2, 0.0f));
            where(dist2 < extent * extent, cos_omega_b) = -1.0f;
            const fvec4 sin_omega_b = sqrt(1.0f - cos_omega_b * cos_omega_b);

            const std::array<fvec4, 2> cos_omega_ne = decode_cosines(uvec4{&n.cos_omega_ne[i], vector_aligned});
            const fvec4 sin_omega_n = sqrt(1.0f - cos_omega_ne[0] * cos_omega_ne[0]);

            const fvec4 cos_omega_x = cos_sub_clamped(sin_omega_w, cos_omega_w, sin_omega_n, cos_omega_ne[0]);
            const fvec4 sin_omega_x = sin_sub_clamped(sin_omega_w, cos_omega_w, sin_omega_n, cos_omega_ne[0]);
            const fvec4 cos_omega = cos_sub_clamped(sin_omega_x, cos_omega_x, sin_omega_b, cos_omega_b);

            fvec4 mul = 0.0f;
            where(cos_omega > cos_omega_ne[1], mul) = (cos_omega / v_len2);
            where(mask, imp) = (imp * mul);
        }
        imp.store_to(&importance[i], vector_aligned);
    }
}

force_inline uint32_t reverse_bits(uint32_t x) {
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return ((x >> 16) | (x << 16));
}

force_inline uint32_t laine_karras_permutation(uint32_t x, const uint32_t seed) {
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}

uint32_t nested_uniform_scramble_base2(uint32_t x, const uint32_t seed) {
    x = reverse_bits(x);
    x = laine_karras_permutation(x, seed);
    x = reverse_bits(x);
    return x;
}

force_inline float scramble_flt(const uint32_t seed, const float val) {
    uint32_t u = uint32_t(val * 16777216.0f) << 8;
    u = nested_uniform_scramble_base2(u, seed);
    return float(u >> 8) / 16777216.0f;
}

force_inline float scramble_unorm(const uint32_t seed, uint32_t val) {
    val = nested_uniform_scramble_base2(val, seed);
    return float(val >> 8) / 16777216.0f;
}

// Gram-Schmidt method
force_inline fvec4 orthogonalize(const fvec4 &a, const fvec4 &b) {
    // we assume that a is normalized
    return normalize(b - dot(a, b) * a);
}

force_inline fvec4 slerp(const fvec4 &start, const fvec4 &end, const float percent) {
    // Dot product - the cosine of the angle between 2 vectors.
    float cos_theta = dot(start, end);
    // Clamp it to be in the range of Acos()
    // This may be unnecessary, but floating point
    // precision can be a fickle mistress.
    cos_theta = clamp(cos_theta, -1.0f, 1.0f);
    // Acos(dot) returns the angle between start and end,
    // And multiplying that by percent returns the angle between
    // start and the final result.
    const float theta = acosf(cos_theta) * percent;
    fvec4 relative_vec = safe_normalize(end - start * cos_theta);
    // Orthonormal basis
    // The final result.
    return start * cosf(theta) + relative_vec * sinf(theta);
}

//
// asinf/acosf implemantation. Taken from apple libm source code
//

// Return arcsine(x) given that .57 < x
force_inline float asin_tail(const float x) {
    return (PI / 2) - ((x + 2.71745038f) * x + 14.0375338f) * (0.00440413551f * ((x - 8.31223679f) * x + 25.3978882f)) *
                          sqrtf(1 - x);
}

float portable_asinf(float x) {
    if (fabsf(x) > 0.57f) {
        const float ret = asin_tail(fabsf(x));
        return (x < 0.0f) ? -ret : ret;
    } else {
        const float x2 = x * x;
        return x + (0.0517513789f * ((x2 + 1.83372748f) * x2 + 1.56678128f)) * x *
                       (x2 * ((x2 - 1.48268414f) * x2 + 2.05554748f));
    }
}

force_inline float acos_positive_tail(const float x) {
    return (((x + 2.71850395f) * x + 14.7303705f)) * (0.00393401226f * ((x - 8.60734272f) * x + 27.0927486f)) *
           sqrtf(1 - x);
}

force_inline float acos_negative_tail(const float x) {
    return PI - (((x - 2.71850395f) * x + 14.7303705f)) * (0.00393401226f * ((x + 8.60734272f) * x + 27.0927486f)) *
                    sqrtf(1 + x);
}

float portable_acosf(float x) {
    if (x < -0.62f) {
        return acos_negative_tail(x);
    } else if (x <= 0.62f) {
        const float x2 = x * x;
        return (PI / 2) - x -
               (0.0700945929f * x * ((x2 + 1.57144082f) * x2 + 1.25210774f)) *
                   (x2 * ((x2 - 1.53757966f) * x2 + 1.89929986f));
    } else {
        return acos_positive_tail(x);
    }
}

// Equivalent to acosf(dot(a, b)), but more numerically stable
// Taken from PBRT source code
force_inline float angle_between(const fvec4 &v1, const fvec4 &v2) {
    if (dot(v1, v2) < 0) {
        return PI - 2 * portable_asinf(length(v1 + v2) / 2);
    } else {
        return 2 * portable_asinf(length(v2 - v1) / 2);
    }
}

} // namespace Ref
} // namespace Ray

// "An Area-Preserving Parametrization for Spherical Rectangles"
// https://www.arnoldrenderer.com/research/egsr2013_spherical_rectangle.pdf
// NOTE: no precomputation is done, everything is calculated in-place
float Ray::Ref::SampleSphericalRectangle(const fvec4 &P, const fvec4 &light_pos, const fvec4 &axis_u,
                                         const fvec4 &axis_v, const fvec2 Xi, fvec4 *out_p) {
    const fvec4 corner = light_pos - 0.5f * axis_u - 0.5f * axis_v;

    float axisu_len, axisv_len;
    const fvec4 x = normalize_len(axis_u, axisu_len), y = normalize_len(axis_v, axisv_len);
    fvec4 z = cross(x, y);

    // compute rectangle coords in local reference system
    const fvec4 dir = corner - P;
    float z0 = dot(dir, z);
    // flip z to make it point against Q
    if (z0 > 0.0f) {
        z = -z;
        z0 = -z0;
    }
    const float x0 = dot(dir, x);
    const float y0 = dot(dir, y);
    const float x1 = x0 + axisu_len;
    const float y1 = y0 + axisv_len;
    // compute internal angles (gamma_i)
    const fvec4 diff = fvec4{x0, y1, x1, y0} - fvec4{x1, y0, x0, y1};
    fvec4 nz = fvec4{y0, x1, y1, x0} * diff;
    nz = nz / sqrt(z0 * z0 * diff * diff + nz * nz);
    const float g0 = portable_acosf(clamp(-nz.get<0>() * nz.get<1>(), -1.0f, 1.0f));
    const float g1 = portable_acosf(clamp(-nz.get<1>() * nz.get<2>(), -1.0f, 1.0f));
    const float g2 = portable_acosf(clamp(-nz.get<2>() * nz.get<3>(), -1.0f, 1.0f));
    const float g3 = portable_acosf(clamp(-nz.get<3>() * nz.get<0>(), -1.0f, 1.0f));
    // compute predefined constants
    const float b0 = nz.get<0>();
    const float b1 = nz.get<2>();
    const float b0sq = b0 * b0;
    const float k = 2 * PI - g2 - g3;
    // compute solid angle from internal angles
    const float area = g0 + g1 - k;
    if (area <= SPHERICAL_AREA_THRESHOLD) {
        return 0.0f;
    }

    if (out_p) {
        // compute cu
        const float au = Xi.get<0>() * area + k;
        const float fu = safe_div((cosf(au) * b0 - b1), sinf(au));
        float cu = 1.0f / sqrtf(fu * fu + b0sq) * (fu > 0.0f ? 1.0f : -1.0f);
        cu = clamp(cu, -1.0f, 1.0f);
        // compute xu
        float xu = -(cu * z0) / fmaxf(sqrtf(1.0f - cu * cu), 1e-7f);
        xu = clamp(xu, x0, x1);
        // compute yv
        const float z0sq = z0 * z0;
        const float y0sq = y0 * y0;
        const float y1sq = y1 * y1;
        const float d = sqrtf(xu * xu + z0sq);
        const float h0 = y0 / sqrtf(d * d + y0sq);
        const float h1 = y1 / sqrtf(d * d + y1sq);
        const float hv = h0 + Xi.get<1>() * (h1 - h0), hv2 = hv * hv;
        const float yv = (hv2 < 1.0f - 1e-6f) ? (hv * d) / sqrtf(1.0f - hv2) : y1;

        // transform (xu, yv, z0) to world coords
        (*out_p) = P + xu * x + yv * y + z0 * z;
    }

    return (1.0f / area);
}

// "Stratified Sampling of Spherical Triangles" https://www.graphics.cornell.edu/pubs/1995/Arv95c.pdf
// Based on https://www.shadertoy.com/view/4tGGzd
float Ray::Ref::SampleSphericalTriangle(const fvec4 &P, const fvec4 &p1, const fvec4 &p2, const fvec4 &p3,
                                        const fvec2 Xi, fvec4 *out_dir) {
    // Setup spherical triangle
    const fvec4 A = normalize(p1 - P), B = normalize(p2 - P), C = normalize(p3 - P);

    // calculate internal angles of spherical triangle: alpha, beta and gamma
    const fvec4 BA = orthogonalize(A, B - A);
    const fvec4 CA = orthogonalize(A, C - A);
    const fvec4 AB = orthogonalize(B, A - B);
    const fvec4 CB = orthogonalize(B, C - B);
    const fvec4 BC = orthogonalize(C, B - C);
    const fvec4 AC = orthogonalize(C, A - C);

    const float alpha = angle_between(BA, CA);
    const float beta = angle_between(AB, CB);
    const float gamma = angle_between(BC, AC);

    const float area = alpha + beta + gamma - PI;
    if (area <= SPHERICAL_AREA_THRESHOLD) {
        return 0.0f;
    }

    if (out_dir) {
        // calculate arc lengths for edges of spherical triangle
        // const float a = portable_acosf(clamp(dot(B, C), -1.0f, 1.0f));
        const float b = portable_acosf(clamp(dot(C, A), -1.0f, 1.0f));
        const float c = portable_acosf(clamp(dot(A, B), -1.0f, 1.0f));

        // Use one random variable to select the new area
        const float area_S = Xi.get<0>() * area;

        // Save the sine and cosine of the angle delta
        const float p = sinf(area_S - alpha);
        const float q = cosf(area_S - alpha);

        // Compute the pair(u; v) that determines sin(beta_s) and cos(beta_s)
        const float u = q - cosf(alpha);
        const float v = p + sinf(alpha) * cosf(c);

        // Compute the s coordinate as normalized arc length from A to C_s
        const float denom = ((v * p + u * q) * sinf(alpha));
        const float s =
            (1.0f / b) * portable_acosf(clamp(safe_div(((v * q - u * p) * cosf(alpha) - v), denom), -1.0f, 1.0f));

        // Compute the third vertex of the sub - triangle
        const fvec4 C_s = slerp(A, C, s);

        // Compute the t coordinate using C_s and Xi[1]
        const float denom2 = portable_acosf(clamp(dot(C_s, B), -1.0f, 1.0f));
        const float t = safe_div(portable_acosf(clamp(1.0f - Xi.get<1>() * (1.0f - dot(C_s, B)), -1.0f, 1.0f)), denom2);

        // Construct the corresponding point on the sphere.
        (*out_dir) = slerp(B, C_s, t);
    }

    // return pdf
    return (1.0f / area);
}

Ray::Ref::fvec2 Ray::Ref::get_scrambled_2d_rand(const uint32_t dim, const uint32_t seed, const int sample,
                                                const uint32_t rand_seq[]) {
    const uint32_t shuffled_dim = nested_uniform_scramble_base2(dim, seed) & (RAND_DIMS_COUNT - 1);
    const uint32_t shuffled_i =
        nested_uniform_scramble_base2(sample, hash_combine(seed, dim)) & (RAND_SAMPLES_COUNT - 1);
    return fvec2{scramble_unorm(hash_combine(seed, 2 * dim + 0),
                                rand_seq[shuffled_dim * 2 * RAND_SAMPLES_COUNT + 2 * shuffled_i + 0]),
                 scramble_unorm(hash_combine(seed, 2 * dim + 1),
                                rand_seq[shuffled_dim * 2 * RAND_SAMPLES_COUNT + 2 * shuffled_i + 1])};
}

void Ray::Ref::GeneratePrimaryRays(const camera_t &cam, const rect_t &rect, const int w, const int h,
                                   const uint32_t rand_seq[], const uint32_t rand_seed, const float filter_table[],
                                   const int iteration, const uint16_t required_samples[],
                                   aligned_vector<ray_data_t> &out_rays, aligned_vector<hit_data_t> &out_inters) {
    const fvec4 cam_origin = make_fvec3(cam.origin), fwd = make_fvec3(cam.fwd), side = make_fvec3(cam.side),
                up = make_fvec3(cam.up);
    const float focus_distance = cam.focus_distance;

    const float k = float(w) / float(h);
    const float temp = tanf(0.5f * cam.fov * PI / 180.0f);
    const float fov_k = temp * focus_distance;
    const float spread_angle = atanf(2.0f * temp / float(h));

    auto get_pix_dir = [&](const float x, const float y, const fvec4 &origin) {
        fvec4 p(2 * fov_k * (float(x) / float(w) + cam.shift[0] / k) - fov_k,
                2 * fov_k * (float(-y) / float(h) + cam.shift[1]) + fov_k, focus_distance, 0.0f);
        p = cam_origin + k * p.get<0>() * side + p.get<1>() * up + p.get<2>() * fwd;
        return normalize(p - origin);
    };

    auto lookup_filter_table = [filter_table](float x) {
        x *= (FILTER_TABLE_SIZE - 1);

        const int index = std::min(int(x), FILTER_TABLE_SIZE - 1);
        const int nindex = std::min(index + 1, FILTER_TABLE_SIZE - 1);
        const float t = x - float(index);

        const float data0 = filter_table[index];
        if (t == 0.0f) {
            return data0;
        }

        float data1 = filter_table[nindex];
        return (1.0f - t) * data0 + t * data1;
    };

    size_t i = 0;
    out_rays.resize(size_t(rect.w) * rect.h);
    out_inters.resize(size_t(rect.w) * rect.h);

    for (int y = rect.y; y < rect.y + rect.h; ++y) {
        for (int x = rect.x; x < rect.x + rect.w; ++x) {
            if (required_samples && required_samples[y * w + x] < iteration) {
                continue;
            }

            auto fx = float(x), fy = float(y);

            const uint32_t px_hash = hash((x << 16) | y);
            const uint32_t rand_hash = hash_combine(px_hash, rand_seed);

            const fvec2 filter_rand = get_scrambled_2d_rand(RAND_DIM_FILTER, rand_hash, iteration - 1, rand_seq);
            float rx = filter_rand.get<0>(), ry = filter_rand.get<1>();

            if (cam.filter != ePixelFilter::Box) {
                rx = lookup_filter_table(rx);
                ry = lookup_filter_table(ry);
            }

            fx += rx;
            fy += ry;

            fvec2 offset = 0.0f;

            if (cam.fstop > 0.0f) {
                const fvec2 lens_rand = get_scrambled_2d_rand(RAND_DIM_LENS, rand_hash, iteration - 1, rand_seq);

                offset = 2.0f * lens_rand - fvec2{1.0f, 1.0f};
                if (offset.get<0>() != 0.0f && offset.get<1>() != 0.0f) {
                    float theta, r;
                    if (fabsf(offset.get<0>()) > fabsf(offset.get<1>())) {
                        r = offset.get<0>();
                        theta = 0.25f * PI * (offset.get<1>() / offset.get<0>());
                    } else {
                        r = offset.get<1>();
                        theta = 0.5f * PI - 0.25f * PI * (offset.get<0>() / offset.get<1>());
                    }

                    if (cam.lens_blades) {
                        r *= ngon_rad(theta, float(cam.lens_blades));
                    }

                    theta += cam.lens_rotation;

                    offset.set<0>(0.5f * r * cosf(theta) / cam.lens_ratio);
                    offset.set<1>(0.5f * r * sinf(theta));
                }

                const float coc = 0.5f * (cam.focal_length / cam.fstop);
                offset *= coc * cam.sensor_height;
            }

            ray_data_t &out_r = out_rays[i];

            const fvec4 _origin = cam_origin + side * offset.get<0>() + up * offset.get<1>();
            const fvec4 _d = get_pix_dir(fx, fy, _origin);
            const float clip_start = cam.clip_start / dot(_d, fwd);

            for (int j = 0; j < 3; j++) {
                out_r.o[j] = _origin[j] + _d[j] * clip_start;
                out_r.d[j] = _d[j];
                out_r.c[j] = 1.0f;
            }

            // air ior is implicit
            out_r.ior[0] = out_r.ior[1] = out_r.ior[2] = out_r.ior[3] = -1.0f;

            out_r.cone_width = 0.0f;
            out_r.cone_spread = spread_angle;

            out_r.pdf = 1e6f;
            out_r.xy = (x << 16) | y;
            out_r.depth = pack_ray_type(RAY_TYPE_CAMERA);
            out_r.depth |= pack_ray_depth(0, 0, 0, 0);

            hit_data_t &out_i = out_inters[i++];
            out_i = {};
            out_i.t = (cam.clip_end / dot(_d, fwd)) - clip_start;
        }
    }

    out_rays.resize(i);
    out_inters.resize(i);
}

void Ray::Ref::SampleMeshInTextureSpace(const int iteration, const int obj_index, const int uv_layer,
                                        const mesh_t &mesh, const mesh_instance_t &mi, const uint32_t *vtx_indices,
                                        const vertex_t *vertices, const rect_t &r, const int width, const int height,
                                        const uint32_t rand_seq[], aligned_vector<ray_data_t> &out_rays,
                                        aligned_vector<hit_data_t> &out_inters) {
    out_rays.resize(size_t(r.w) * r.h);
    out_inters.resize(out_rays.size());

    for (int y = r.y; y < r.y + r.h; ++y) {
        for (int x = r.x; x < r.x + r.w; ++x) {
            const int i = (y - r.y) * r.w + (x - r.x);

            ray_data_t &out_ray = out_rays[i];
            hit_data_t &out_inter = out_inters[i];

            out_ray.xy = (x << 16) | y;
            out_ray.c[0] = out_ray.c[1] = out_ray.c[2] = 1.0f;
            out_inter.v = -1.0f;
        }
    }

    const ivec2 irect_min = {r.x, r.y}, irect_max = {r.x + r.w - 1, r.y + r.h - 1};
    const fvec2 size = {float(width), float(height)};

    for (uint32_t tri = mesh.tris_index; tri < mesh.tris_index + mesh.tris_count; tri++) {
        const vertex_t &v0 = vertices[vtx_indices[tri * 3 + 0]];
        const vertex_t &v1 = vertices[vtx_indices[tri * 3 + 1]];
        const vertex_t &v2 = vertices[vtx_indices[tri * 3 + 2]];

        // TODO: use uv_layer
        const auto t0 = fvec2{v0.t[0], 1.0f - v0.t[1]} * size;
        const auto t1 = fvec2{v1.t[0], 1.0f - v1.t[1]} * size;
        const auto t2 = fvec2{v2.t[0], 1.0f - v2.t[1]} * size;

        fvec2 bbox_min = t0, bbox_max = t0;

        bbox_min = min(bbox_min, t1);
        bbox_min = min(bbox_min, t2);

        bbox_max = max(bbox_max, t1);
        bbox_max = max(bbox_max, t2);

        ivec2 ibbox_min = ivec2{bbox_min},
              ibbox_max = ivec2{int(roundf(bbox_max.get<0>())), int(roundf(bbox_max.get<1>()))};

        if (ibbox_max.get<0>() < irect_min.get<0>() || ibbox_max.get<1>() < irect_min.get<1>() ||
            ibbox_min.get<0>() > irect_max.get<0>() || ibbox_min.get<1>() > irect_max.get<1>()) {
            continue;
        }

        ibbox_min = max(ibbox_min, irect_min);
        ibbox_max = min(ibbox_max, irect_max);

        const fvec2 d01 = t0 - t1, d12 = t1 - t2, d20 = t2 - t0;

        const float area = d01.get<0>() * d20.get<1>() - d20.get<0>() * d01.get<1>();
        if (area < FLT_EPS) {
            continue;
        }

        const float inv_area = 1.0f / area;

        for (int y = ibbox_min.get<1>(); y <= ibbox_max.get<1>(); ++y) {
            for (int x = ibbox_min.get<0>(); x <= ibbox_max.get<0>(); ++x) {
                const int i = (y - r.y) * r.w + (x - r.x);
                ray_data_t &out_ray = out_rays[i];
                hit_data_t &out_inter = out_inters[i];

                if (out_inter.v >= 0.0f) {
                    continue;
                }

                const float _x = float(x); // + fract(rand_seq[RAND_DIM_FILTER_U] + construct_float(hash(out_ray.xy)));
                const float _y =
                    float(y); // + fract(rand_seq[RAND_DIM_FILTER_V] + construct_float(hash(hash(out_ray.xy))));

                float u = d01.get<0>() * (_y - t0.get<1>()) - d01.get<1>() * (_x - t0.get<0>()),
                      v = d12.get<0>() * (_y - t1.get<1>()) - d12.get<1>() * (_x - t1.get<0>()),
                      w = d20.get<0>() * (_y - t2.get<1>()) - d20.get<1>() * (_x - t2.get<0>());

                if (u >= -FLT_EPS && v >= -FLT_EPS && w >= -FLT_EPS) {
                    const auto p0 = fvec4{v0.p}, p1 = fvec4{v1.p}, p2 = fvec4{v2.p};
                    const auto n0 = fvec4{v0.n}, n1 = fvec4{v1.n}, n2 = fvec4{v2.n};

                    u *= inv_area;
                    v *= inv_area;
                    w *= inv_area;

                    const fvec4 p = TransformPoint(p0 * v + p1 * w + p2 * u, mi.xform),
                                n = TransformNormal(n0 * v + n1 * w + n2 * u, mi.inv_xform);

                    const fvec4 o = p + n, d = -n;

                    memcpy(&out_ray.o[0], value_ptr(o), 3 * sizeof(float));
                    memcpy(&out_ray.d[0], value_ptr(d), 3 * sizeof(float));

                    out_ray.cone_width = 0;
                    out_ray.cone_spread = 0;
                    out_ray.depth = pack_ray_type(RAY_TYPE_DIFFUSE);
                    out_ray.depth |= pack_ray_depth(0, 0, 0, 0);

                    out_inter.prim_index = int(tri);
                    out_inter.obj_index = obj_index;
                    out_inter.t = 1.0f;
                    out_inter.u = w;
                    out_inter.v = u;
                }
            }
        }
    }
}

int Ray::Ref::SortRays_CPU(Span<ray_data_t> rays, const float root_min[3], const float cell_size[3],
                           uint32_t *hash_values, uint32_t *scan_values, ray_chunk_t *chunks,
                           ray_chunk_t *chunks_temp) {
    // From "Fast Ray Sorting and Breadth-First Packet Traversal for GPU Ray Tracing" [2010]

    // compute ray hash values
    for (uint32_t i = 0; i < uint32_t(rays.size()); ++i) {
        hash_values[i] = get_ray_hash(rays[i], root_min, cell_size);
    }

    size_t chunks_count = 0;

    // compress codes into spans of indentical values (makes sorting stage faster)
    for (uint32_t start = 0, end = 1; end <= uint32_t(rays.size()); end++) {
        if (end == uint32_t(rays.size()) || (hash_values[start] != hash_values[end])) {
            chunks[chunks_count].hash = hash_values[start];
            chunks[chunks_count].base = start;
            chunks[chunks_count++].size = end - start;
            start = end;
        }
    }

    radix_sort(&chunks[0], &chunks[0] + chunks_count, &chunks_temp[0]);

    // decompress sorted spans
    size_t counter = 0;
    for (uint32_t i = 0; i < chunks_count; ++i) {
        for (uint32_t j = 0; j < chunks[i].size; j++) {
            scan_values[counter++] = chunks[i].base + j;
        }
    }

    // reorder rays
    for (uint32_t i = 0; i < uint32_t(rays.size()); ++i) {
        uint32_t j;
        while (i != (j = scan_values[i])) {
            const uint32_t k = scan_values[j];
            std::swap(rays[j], rays[k]);
            std::swap(scan_values[i], scan_values[j]);
        }
    }

    return int(rays.size());
}

int Ray::Ref::SortRays_GPU(Span<ray_data_t> rays, const float root_min[3], const float cell_size[3],
                           uint32_t *hash_values, int *head_flags, uint32_t *scan_values, ray_chunk_t *chunks,
                           ray_chunk_t *chunks_temp, uint32_t *skeleton) {
    // From "Fast Ray Sorting and Breadth-First Packet Traversal for GPU Ray Tracing" [2010]

    // compute ray hash values
    for (uint32_t i = 0; i < uint32_t(rays.size()); ++i) {
        hash_values[i] = get_ray_hash(rays[i], root_min, cell_size);
    }

    // set head flags
    head_flags[0] = 1;
    for (uint32_t i = 1; i < uint32_t(rays.size()); ++i) {
        head_flags[i] = hash_values[i] != hash_values[i - 1];
    }

    size_t chunks_count = 0;

    { // perform exclusive scan on head flags
        uint32_t cur_sum = 0;
        for (size_t i = 0; i < size_t(rays.size()); ++i) {
            scan_values[i] = cur_sum;
            cur_sum += head_flags[i];
        }
        chunks_count = cur_sum;
    }

    // init Ray chunks hash and base index
    for (uint32_t i = 0; i < uint32_t(rays.size()); ++i) {
        if (head_flags[i]) {
            chunks[scan_values[i]].hash = hash_values[i];
            chunks[scan_values[i]].base = uint32_t(i);
        }
    }

    // init ray chunks size
    if (chunks_count) {
        for (size_t i = 0; i < chunks_count - 1; ++i) {
            chunks[i].size = chunks[i + 1].base - chunks[i].base;
        }
        chunks[chunks_count - 1].size = uint32_t(rays.size()) - chunks[chunks_count - 1].base;
    }

    radix_sort(&chunks[0], &chunks[0] + chunks_count, &chunks_temp[0]);

    { // perform exclusive scan on chunks size
        uint32_t cur_sum = 0;
        for (size_t i = 0; i < chunks_count; ++i) {
            scan_values[i] = cur_sum;
            cur_sum += chunks[i].size;
        }
    }

    std::fill(skeleton, skeleton + rays.size(), 1);
    std::fill(head_flags, head_flags + rays.size(), 0);

    // init skeleton and head flags array
    for (size_t i = 0; i < chunks_count; ++i) {
        skeleton[scan_values[i]] = chunks[i].base;
        head_flags[scan_values[i]] = 1;
    }

    { // perform a segmented scan on skeleton array
        uint32_t cur_sum = 0;
        for (uint32_t i = 0; i < uint32_t(rays.size()); ++i) {
            if (head_flags[i]) {
                cur_sum = 0;
            }
            cur_sum += skeleton[i];
            scan_values[i] = cur_sum;
        }
    }

    // reorder rays
    for (uint32_t i = 0; i < uint32_t(rays.size()); ++i) {
        uint32_t j;
        while (i != (j = scan_values[i])) {
            const uint32_t k = scan_values[j];
            std::swap(rays[j], rays[k]);
            std::swap(scan_values[i], scan_values[j]);
        }
    }

    return int(rays.size());
}

bool Ray::Ref::IntersectTris_ClosestHit(const float ro[3], const float rd[3], const tri_accel_t *tris,
                                        const int tri_start, const int tri_end, const int obj_index,
                                        hit_data_t &out_inter) {
    hit_data_t inter{Uninitialize};
    inter.obj_index = obj_index;
    inter.t = out_inter.t;
    inter.v = -1.0f;

    for (int i = tri_start; i < tri_end; ++i) {
        IntersectTri(ro, rd, tris[i], i, inter);
    }

    out_inter.obj_index = inter.v >= 0.0f ? inter.obj_index : out_inter.obj_index;
    out_inter.prim_index = inter.v >= 0.0f ? inter.prim_index : out_inter.prim_index;
    out_inter.t = inter.t; // already contains min value
    out_inter.u = inter.v >= 0.0f ? inter.u : out_inter.u;
    out_inter.v = inter.v >= 0.0f ? inter.v : out_inter.v;

    return inter.v >= 0.0f;
}

bool Ray::Ref::IntersectTris_ClosestHit(const float ro[3], const float rd[3], const mtri_accel_t *mtris,
                                        const int tri_start, const int tri_end, const int obj_index,
                                        hit_data_t &out_inter) {
    hit_data_t inter{Uninitialize};
    inter.obj_index = obj_index;
    inter.t = out_inter.t;
    inter.v = -1.0f;

    for (int i = tri_start / 8; i < (tri_end + 7) / 8; ++i) {
        IntersectTri(ro, rd, mtris[i], i * 8, inter);
    }

    out_inter.obj_index = inter.v >= 0.0f ? inter.obj_index : out_inter.obj_index;
    out_inter.prim_index = inter.v >= 0.0f ? inter.prim_index : out_inter.prim_index;
    out_inter.t = inter.t; // already contains min value
    out_inter.u = inter.v >= 0.0f ? inter.u : out_inter.u;
    out_inter.v = inter.v >= 0.0f ? inter.v : out_inter.v;

    return inter.v >= 0.0f;
}

bool Ray::Ref::IntersectTris_AnyHit(const float ro[3], const float rd[3], const tri_accel_t *tris,
                                    const tri_mat_data_t *materials, const uint32_t *indices, const int tri_start,
                                    const int tri_end, const int obj_index, hit_data_t &out_inter) {
    hit_data_t inter{Uninitialize};
    inter.obj_index = obj_index;
    inter.t = out_inter.t;
    inter.v = -1.0f;

    for (int i = tri_start; i < tri_end; ++i) {
        IntersectTri(ro, rd, tris[i], i, inter);
        if (inter.v >= 0.0f && ((inter.prim_index > 0 && (materials[indices[i]].front_mi & MATERIAL_SOLID_BIT)) ||
                                (inter.prim_index < 0 && (materials[indices[i]].back_mi & MATERIAL_SOLID_BIT)))) {
            break;
        }
    }

    out_inter.obj_index = inter.v >= 0.0f ? inter.obj_index : out_inter.obj_index;
    out_inter.prim_index = inter.v >= 0.0f ? inter.prim_index : out_inter.prim_index;
    out_inter.t = inter.t; // already contains min value
    out_inter.u = inter.v >= 0.0f ? inter.u : out_inter.u;
    out_inter.v = inter.v >= 0.0f ? inter.v : out_inter.v;

    return inter.v >= 0.0f;
}

bool Ray::Ref::IntersectTris_AnyHit(const float ro[3], const float rd[3], const mtri_accel_t *mtris,
                                    const tri_mat_data_t *materials, const uint32_t *indices, const int tri_start,
                                    const int tri_end, const int obj_index, hit_data_t &out_inter) {
    hit_data_t inter{Uninitialize};
    inter.obj_index = obj_index;
    inter.t = out_inter.t;
    inter.v = -1.0f;

    for (int i = tri_start / 8; i < (tri_end + 7) / 8; ++i) {
        IntersectTri(ro, rd, mtris[i], i * 8, inter);
        if (inter.v >= 0.0f && ((inter.prim_index > 0 && (materials[indices[i]].front_mi & MATERIAL_SOLID_BIT)) ||
                                (inter.prim_index < 0 && (materials[indices[i]].back_mi & MATERIAL_SOLID_BIT)))) {
            break;
        }
    }

    out_inter.obj_index = inter.v >= 0.0f ? inter.obj_index : out_inter.obj_index;
    out_inter.prim_index = inter.v >= 0.0f ? inter.prim_index : out_inter.prim_index;
    out_inter.t = inter.t; // already contains min value
    out_inter.u = inter.v >= 0.0f ? inter.u : out_inter.u;
    out_inter.v = inter.v >= 0.0f ? inter.v : out_inter.v;

    return inter.v >= 0.0f;
}

bool Ray::Ref::Traverse_TLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], const uint32_t ray_flags,
                                                  const bvh_node_t *nodes, uint32_t root_index,
                                                  const mesh_instance_t *mesh_instances, const tri_accel_t *tris,
                                                  const uint32_t *tri_indices, hit_data_t &inter) {
    bool res = false;

    float inv_d[3];
    safe_invert(rd, inv_d);

    uint32_t stack[MAX_STACK_SIZE];
    uint32_t stack_size = 0;

    stack[stack_size++] = root_index;

    while (stack_size) {
        uint32_t cur = stack[--stack_size];

        if (!bbox_test(ro, inv_d, inter.t, nodes[cur])) {
            continue;
        }

        if (!is_leaf_node(nodes[cur])) {
            stack[stack_size++] = far_child(rd, nodes[cur]);
            stack[stack_size++] = near_child(rd, nodes[cur]);
        } else {
            assert(nodes[cur].prim_count == 1);
            const uint32_t prim_index = (nodes[cur].prim_index & PRIM_INDEX_BITS);

            const mesh_instance_t &mi = mesh_instances[prim_index];
            if ((mi.ray_visibility & ray_flags) == 0) {
                continue;
            }

            float _ro[3], _rd[3];
            TransformRay(ro, rd, mi.inv_xform, _ro, _rd);

            float _inv_d[3];
            safe_invert(_rd, _inv_d);
            res |= Traverse_BLAS_WithStack_ClosestHit(_ro, _rd, _inv_d, nodes, mi.node_index, tris, int(prim_index),
                                                      inter);
        }
    }

    // resolve primitive index indirection
    if (inter.prim_index < 0) {
        inter.prim_index = -int(tri_indices[-inter.prim_index - 1]) - 1;
    } else {
        inter.prim_index = int(tri_indices[inter.prim_index]);
    }

    return res;
}

bool Ray::Ref::Traverse_TLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], const uint32_t ray_flags,
                                                  const bvh2_node_t *nodes, uint32_t root_index,
                                                  const mesh_instance_t *mesh_instances, const tri_accel_t *tris,
                                                  const uint32_t *tri_indices, hit_data_t &inter) {
    bool res = false;

    float inv_d[3];
    safe_invert(rd, inv_d);

    uint32_t stack[MAX_STACK_SIZE];
    uint32_t stack_size = 0;

    stack[stack_size++] = 0x1fffffffu;

    uint32_t cur = root_index;
    while (stack_size) {
        uint32_t leaf_node = 0;
        while (stack_size && (cur & BVH2_PRIM_COUNT_BITS) == 0) {
            const bvh2_node_t &n = nodes[cur];

            uint32_t children[2] = {n.left_child, n.right_child};

            const float ch0_min[3] = {n.ch_data0[0], n.ch_data0[2], n.ch_data2[0]};
            const float ch0_max[3] = {n.ch_data0[1], n.ch_data0[3], n.ch_data2[1]};

            const float ch1_min[3] = {n.ch_data1[0], n.ch_data1[2], n.ch_data2[2]};
            const float ch1_max[3] = {n.ch_data1[1], n.ch_data1[3], n.ch_data2[3]};

            float ch0_dist, ch1_dist;
            const bool ch0_res = bbox_test(ro, inv_d, inter.t, ch0_min, ch0_max, ch0_dist);
            const bool ch1_res = bbox_test(ro, inv_d, inter.t, ch1_min, ch1_max, ch1_dist);

            if (!ch0_res && !ch1_res) {
                cur = stack[--stack_size];
            } else {
                cur = ch0_res ? children[0] : children[1];
                if (ch0_res && ch1_res) {
                    if (ch1_dist < ch0_dist) {
                        const uint32_t temp = cur;
                        cur = children[1];
                        children[1] = temp;
                    }
                    stack[stack_size++] = children[1];
                }
            }
            if ((cur & BVH2_PRIM_COUNT_BITS) != 0 && (leaf_node & BVH2_PRIM_COUNT_BITS) == 0) {
                leaf_node = cur;
                cur = stack[--stack_size];
            }
            if ((leaf_node & BVH2_PRIM_COUNT_BITS) != 0) {
                break;
            }
        }

        while ((leaf_node & BVH2_PRIM_COUNT_BITS) != 0) {
            assert(((leaf_node & BVH2_PRIM_COUNT_BITS) >> 29) == 1);
            const uint32_t mi_index = (leaf_node & BVH2_PRIM_INDEX_BITS);
            const mesh_instance_t &mi = mesh_instances[mi_index];
            if ((mi.ray_visibility & ray_flags) != 0) {
                float _ro[3], _rd[3];
                TransformRay(ro, rd, mi.inv_xform, _ro, _rd);

                float _inv_d[3];
                safe_invert(_rd, _inv_d);
                res |= Traverse_BLAS_WithStack_ClosestHit(_ro, _rd, _inv_d, nodes, mi.node_index, tris, int(mi_index),
                                                          inter);
            }
            leaf_node = cur;
            if ((cur & BVH2_PRIM_COUNT_BITS) != 0) {
                cur = stack[--stack_size];
            }
        }
    }

    // resolve primitive index indirection
    if (inter.prim_index < 0) {
        inter.prim_index = -int(tri_indices[-inter.prim_index - 1]) - 1;
    } else {
        inter.prim_index = int(tri_indices[inter.prim_index]);
    }

    return res;
}

bool Ray::Ref::Traverse_TLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], const uint32_t ray_flags,
                                                  const wbvh_node_t *nodes, uint32_t root_index,
                                                  const mesh_instance_t *mesh_instances, const mtri_accel_t *mtris,
                                                  const uint32_t *tri_indices, hit_data_t &inter) {
    bool res = false;

    float inv_d[3];
    safe_invert(rd, inv_d);

    TraversalStack<MAX_STACK_SIZE> st;
    st.push(root_index, 0.0f);

    while (!st.empty()) {
        stack_entry_t cur = st.pop();

        if (cur.dist > inter.t) {
            continue;
        }

    TRAVERSE:
        if (!is_leaf_node(nodes[cur.index])) {
            alignas(16) float dist[8];
            long mask = bbox_test_oct(ro, inv_d, inter.t, nodes[cur.index], dist);
            if (mask) {
                long i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                if (mask == 0) { // only one box was hit
                    cur.index = nodes[cur.index].child[i];
                    goto TRAVERSE;
                }

                long i2 = GetFirstBit(mask);
                mask = ClearBit(mask, i2);
                if (mask == 0) { // two boxes were hit
                    if (dist[i] < dist[i2]) {
                        st.push(nodes[cur.index].child[i2], dist[i2]);
                        cur.index = nodes[cur.index].child[i];
                    } else {
                        st.push(nodes[cur.index].child[i], dist[i]);
                        cur.index = nodes[cur.index].child[i2];
                    }
                    goto TRAVERSE;
                }

                st.push(nodes[cur.index].child[i], dist[i]);
                st.push(nodes[cur.index].child[i2], dist[i2]);

                i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                st.push(nodes[cur.index].child[i], dist[i]);
                if (mask == 0) { // three boxes were hit
                    st.sort_top3();
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }

                i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                st.push(nodes[cur.index].child[i], dist[i]);
                if (mask == 0) { // four boxes were hit
                    st.sort_top4();
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }

                const uint32_t size_before = st.stack_size;

                // from five to eight boxes were hit
                do {
                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], dist[i]);
                } while (mask != 0);

                const int count = int(st.stack_size - size_before + 4);
                st.sort_topN(count);
                cur.index = st.pop_index();
                goto TRAVERSE;
            }
        } else {
            assert(nodes[cur.index].child[1] == 1);
            const uint32_t prim_index = (nodes[cur.index].child[0] & PRIM_INDEX_BITS);

            const mesh_instance_t &mi = mesh_instances[prim_index];
            if ((mi.ray_visibility & ray_flags) == 0) {
                continue;
            }

            float _ro[3], _rd[3];
            TransformRay(ro, rd, mi.inv_xform, _ro, _rd);

            float _inv_d[3];
            safe_invert(_rd, _inv_d);
            res |= Traverse_BLAS_WithStack_ClosestHit(_ro, _rd, _inv_d, nodes, mi.node_index, mtris, int(prim_index),
                                                      inter);
        }
    }

    // resolve primitive index indirection
    if (inter.prim_index < 0) {
        inter.prim_index = -int(tri_indices[-inter.prim_index - 1]) - 1;
    } else {
        inter.prim_index = int(tri_indices[inter.prim_index]);
    }

    return res;
}

bool Ray::Ref::Traverse_TLAS_WithStack_AnyHit(const float ro[3], const float rd[3], const int ray_type,
                                              const bvh_node_t *nodes, const uint32_t root_index,
                                              const mesh_instance_t *mesh_instances, const tri_accel_t *tris,
                                              const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                              hit_data_t &inter) {
    const uint32_t ray_vismask = (1u << ray_type);

    float inv_d[3];
    safe_invert(rd, inv_d);

    uint32_t stack[MAX_STACK_SIZE];
    uint32_t stack_size = 0;

    stack[stack_size++] = root_index;

    while (stack_size) {
        const uint32_t cur = stack[--stack_size];

        if (!bbox_test(ro, inv_d, inter.t, nodes[cur])) {
            continue;
        }

        if (!is_leaf_node(nodes[cur])) {
            stack[stack_size++] = far_child(rd, nodes[cur]);
            stack[stack_size++] = near_child(rd, nodes[cur]);
        } else {
            assert(nodes[cur].prim_count == 1);
            const uint32_t prim_index = (nodes[cur].prim_index & PRIM_INDEX_BITS);

            const mesh_instance_t &mi = mesh_instances[prim_index];
            if ((mi.ray_visibility & ray_vismask) == 0) {
                continue;
            }

            float _ro[3], _rd[3];
            TransformRay(ro, rd, mi.inv_xform, _ro, _rd);

            float _inv_d[3];
            safe_invert(_rd, _inv_d);

            const bool solid_hit_found = Traverse_BLAS_WithStack_AnyHit(_ro, _rd, _inv_d, nodes, mi.node_index, tris,
                                                                        materials, tri_indices, int(prim_index), inter);
            if (solid_hit_found) {
                return true;
            }
        }
    }

    // resolve primitive index indirection
    if (inter.prim_index < 0) {
        inter.prim_index = -int(tri_indices[-inter.prim_index - 1]) - 1;
    } else {
        inter.prim_index = int(tri_indices[inter.prim_index]);
    }

    return false;
}

bool Ray::Ref::Traverse_TLAS_WithStack_AnyHit(const float ro[3], const float rd[3], const int ray_type,
                                              const bvh2_node_t *nodes, const uint32_t root_index,
                                              const mesh_instance_t *mesh_instances, const tri_accel_t *tris,
                                              const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                              hit_data_t &inter) {
    const uint32_t ray_vismask = (1u << ray_type);

    float inv_d[3];
    safe_invert(rd, inv_d);

    uint32_t stack[MAX_STACK_SIZE];
    uint32_t stack_size = 0;

    stack[stack_size++] = 0x1fffffffu;

    uint32_t cur = root_index;
    while (stack_size) {
        uint32_t leaf_node = 0;
        while (stack_size && (cur & BVH2_PRIM_COUNT_BITS) == 0) {
            const bvh2_node_t &n = nodes[cur];

            uint32_t children[2] = {n.left_child, n.right_child};

            const float ch0_min[3] = {n.ch_data0[0], n.ch_data0[2], n.ch_data2[0]};
            const float ch0_max[3] = {n.ch_data0[1], n.ch_data0[3], n.ch_data2[1]};

            const float ch1_min[3] = {n.ch_data1[0], n.ch_data1[2], n.ch_data2[2]};
            const float ch1_max[3] = {n.ch_data1[1], n.ch_data1[3], n.ch_data2[3]};

            float ch0_dist, ch1_dist;
            const bool ch0_res = bbox_test(ro, inv_d, inter.t, ch0_min, ch0_max, ch0_dist);
            const bool ch1_res = bbox_test(ro, inv_d, inter.t, ch1_min, ch1_max, ch1_dist);

            if (!ch0_res && !ch1_res) {
                cur = stack[--stack_size];
            } else {
                cur = ch0_res ? children[0] : children[1];
                if (ch0_res && ch1_res) {
                    if (ch1_dist < ch0_dist) {
                        const uint32_t temp = cur;
                        cur = children[1];
                        children[1] = temp;
                    }
                    stack[stack_size++] = children[1];
                }
            }
            if ((cur & BVH2_PRIM_COUNT_BITS) != 0 && (leaf_node & BVH2_PRIM_COUNT_BITS) == 0) {
                leaf_node = cur;
                cur = stack[--stack_size];
            }
            if ((leaf_node & BVH2_PRIM_COUNT_BITS) != 0) {
                break;
            }
        }

        while ((leaf_node & BVH2_PRIM_COUNT_BITS) != 0) {
            assert(((leaf_node & BVH2_PRIM_COUNT_BITS) >> 29) == 1);
            const uint32_t mi_index = (leaf_node & BVH2_PRIM_INDEX_BITS);
            const mesh_instance_t &mi = mesh_instances[mi_index];
            if ((mi.ray_visibility & ray_vismask) != 0) {
                float _ro[3], _rd[3];
                TransformRay(ro, rd, mi.inv_xform, _ro, _rd);

                float _inv_d[3];
                safe_invert(_rd, _inv_d);

                const bool solid_hit_found = Traverse_BLAS_WithStack_AnyHit(
                    _ro, _rd, _inv_d, nodes, mi.node_index, tris, materials, tri_indices, int(mi_index), inter);
                if (solid_hit_found) {
                    return true;
                }
            }
            leaf_node = cur;
            if ((cur & BVH2_PRIM_COUNT_BITS) != 0) {
                cur = stack[--stack_size];
            }
        }
    }

    // resolve primitive index indirection
    if (inter.prim_index < 0) {
        inter.prim_index = -int(tri_indices[-inter.prim_index - 1]) - 1;
    } else {
        inter.prim_index = int(tri_indices[inter.prim_index]);
    }

    return false;
}

bool Ray::Ref::Traverse_TLAS_WithStack_AnyHit(const float ro[3], const float rd[3], const int ray_type,
                                              const wbvh_node_t *nodes, const uint32_t root_index,
                                              const mesh_instance_t *mesh_instances, const mtri_accel_t *mtris,
                                              const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                              hit_data_t &inter) {
    const int ray_dir_oct = ((rd[2] > 0.0f) << 2) | ((rd[1] > 0.0f) << 1) | (rd[0] > 0.0f);
    const uint32_t ray_vismask = (1u << ray_type);

    int child_order[8];
    UNROLLED_FOR(i, 8, { child_order[i] = i ^ ray_dir_oct; })

    float inv_d[3];
    safe_invert(rd, inv_d);

    TraversalStack<MAX_STACK_SIZE> st;
    st.push(root_index, 0.0f);

    while (!st.empty()) {
        stack_entry_t cur = st.pop();

        if (cur.dist > inter.t) {
            continue;
        }

    TRAVERSE:
        if (!is_leaf_node(nodes[cur.index])) {
            alignas(16) float dist[8];
            long mask = bbox_test_oct(ro, inv_d, inter.t, nodes[cur.index], dist);
            if (mask) {
                long i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                if (mask == 0) { // only one box was hit
                    cur.index = nodes[cur.index].child[i];
                    goto TRAVERSE;
                }

                long i2 = GetFirstBit(mask);
                mask = ClearBit(mask, i2);
                if (mask == 0) { // two boxes were hit
                    if (dist[i] < dist[i2]) {
                        st.push(nodes[cur.index].child[i2], dist[i2]);
                        cur.index = nodes[cur.index].child[i];
                    } else {
                        st.push(nodes[cur.index].child[i], dist[i]);
                        cur.index = nodes[cur.index].child[i2];
                    }
                    goto TRAVERSE;
                }

                st.push(nodes[cur.index].child[i], dist[i]);
                st.push(nodes[cur.index].child[i2], dist[i2]);

                i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                st.push(nodes[cur.index].child[i], dist[i]);
                if (mask == 0) { // three boxes were hit
                    st.sort_top3();
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }

                i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                st.push(nodes[cur.index].child[i], dist[i]);
                if (mask == 0) { // four boxes were hit
                    st.sort_top4();
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }

                const uint32_t size_before = st.stack_size;

                // from five to eight boxes were hit
                do {
                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], dist[i]);
                } while (mask != 0);

                int count = int(st.stack_size - size_before + 4);
                st.sort_topN(count);
                cur.index = st.pop_index();
                goto TRAVERSE;
            }
        } else {
            assert(nodes[cur.index].child[1] == 1);
            const uint32_t prim_index = (nodes[cur.index].child[0] & PRIM_INDEX_BITS);

            const mesh_instance_t &mi = mesh_instances[prim_index];
            if ((mi.ray_visibility & ray_vismask) == 0) {
                continue;
            }

            float _ro[3], _rd[3];
            TransformRay(ro, rd, mi.inv_xform, _ro, _rd);

            float _inv_d[3];
            safe_invert(_rd, _inv_d);
            const bool solid_hit_found = Traverse_BLAS_WithStack_AnyHit(_ro, _rd, _inv_d, nodes, mi.node_index, mtris,
                                                                        materials, tri_indices, int(prim_index), inter);
            if (solid_hit_found) {
                return true;
            }
        }
    }

    // resolve primitive index indirection
    if (inter.prim_index < 0) {
        inter.prim_index = -int(tri_indices[-inter.prim_index - 1]) - 1;
    } else {
        inter.prim_index = int(tri_indices[inter.prim_index]);
    }

    return false;
}

bool Ray::Ref::Traverse_BLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], const float inv_d[3],
                                                  const bvh_node_t *nodes, const uint32_t root_index,
                                                  const tri_accel_t *tris, const int obj_index, hit_data_t &inter) {
    bool res = false;

    uint32_t stack[MAX_STACK_SIZE];
    uint32_t stack_size = 0;

    stack[stack_size++] = root_index;

    while (stack_size) {
        const uint32_t cur = stack[--stack_size];

        if (!bbox_test(ro, inv_d, inter.t, nodes[cur])) {
            continue;
        }

        if (!is_leaf_node(nodes[cur])) {
            stack[stack_size++] = far_child(rd, nodes[cur]);
            stack[stack_size++] = near_child(rd, nodes[cur]);
        } else {
            const int tri_start = int(nodes[cur].prim_index & PRIM_INDEX_BITS),
                      tri_end = int(tri_start + nodes[cur].prim_count);
            res |= IntersectTris_ClosestHit(ro, rd, tris, tri_start, tri_end, obj_index, inter);
        }
    }

    return res;
}

bool Ray::Ref::Traverse_BLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], const float inv_d[3],
                                                  const bvh2_node_t *nodes, const uint32_t root_index,
                                                  const tri_accel_t *tris, const int obj_index, hit_data_t &inter) {
    bool res = false;

    uint32_t stack[MAX_STACK_SIZE];
    uint32_t stack_size = 0;

    stack[stack_size++] = 0x1fffffffu;

    uint32_t cur = root_index;
    while (stack_size) {
        uint32_t leaf_node = 0;
        while (stack_size && (cur & BVH2_PRIM_COUNT_BITS) == 0) {
            const bvh2_node_t &n = nodes[cur];

            uint32_t children[2] = {n.left_child, n.right_child};

            const float ch0_min[3] = {n.ch_data0[0], n.ch_data0[2], n.ch_data2[0]};
            const float ch0_max[3] = {n.ch_data0[1], n.ch_data0[3], n.ch_data2[1]};

            const float ch1_min[3] = {n.ch_data1[0], n.ch_data1[2], n.ch_data2[2]};
            const float ch1_max[3] = {n.ch_data1[1], n.ch_data1[3], n.ch_data2[3]};

            float ch0_dist, ch1_dist;
            const bool ch0_res = bbox_test(ro, inv_d, inter.t, ch0_min, ch0_max, ch0_dist);
            const bool ch1_res = bbox_test(ro, inv_d, inter.t, ch1_min, ch1_max, ch1_dist);

            if (!ch0_res && !ch1_res) {
                cur = stack[--stack_size];
            } else {
                cur = ch0_res ? children[0] : children[1];
                if (ch0_res && ch1_res) {
                    if (ch1_dist < ch0_dist) {
                        const uint32_t temp = cur;
                        cur = children[1];
                        children[1] = temp;
                    }
                    stack[stack_size++] = children[1];
                }
            }
            if ((cur & BVH2_PRIM_COUNT_BITS) != 0 && (leaf_node & BVH2_PRIM_COUNT_BITS) == 0) {
                leaf_node = cur;
                cur = stack[--stack_size];
            }
            if ((leaf_node & BVH2_PRIM_COUNT_BITS) != 0) {
                break;
            }
        }

        while ((leaf_node & BVH2_PRIM_COUNT_BITS) != 0) {
            const int tri_start = int(leaf_node & BVH2_PRIM_INDEX_BITS),
                      tri_end = int(tri_start + ((leaf_node & BVH2_PRIM_COUNT_BITS) >> 29) + 1);
            assert((tri_start % 8) == 0);
            assert((tri_end - tri_start) <= 8);
            res |= IntersectTris_ClosestHit(ro, rd, tris, tri_start, tri_end, obj_index, inter);

            leaf_node = cur;
            if ((cur & BVH2_PRIM_COUNT_BITS) != 0) {
                cur = stack[--stack_size];
            }
        }
    }

    return res;
}

bool Ray::Ref::Traverse_BLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], const float inv_d[3],
                                                  const wbvh_node_t *nodes, const uint32_t root_index,
                                                  const mtri_accel_t *mtris, int obj_index, hit_data_t &inter) {
    bool res = false;

    TraversalStack<MAX_STACK_SIZE> st;
    st.push(root_index, 0.0f);

    while (!st.empty()) {
        stack_entry_t cur = st.pop();

        if (cur.dist > inter.t) {
            continue;
        }

    TRAVERSE:
        if (!is_leaf_node(nodes[cur.index])) {
            alignas(16) float dist[8];
            long mask = bbox_test_oct(ro, inv_d, inter.t, nodes[cur.index], dist);
            if (mask) {
                long i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                if (mask == 0) { // only one box was hit
                    cur.index = nodes[cur.index].child[i];
                    goto TRAVERSE;
                }

                const long i2 = GetFirstBit(mask);
                mask = ClearBit(mask, i2);
                if (mask == 0) { // two boxes were hit
                    if (dist[i] < dist[i2]) {
                        st.push(nodes[cur.index].child[i2], dist[i2]);
                        cur.index = nodes[cur.index].child[i];
                    } else {
                        st.push(nodes[cur.index].child[i], dist[i]);
                        cur.index = nodes[cur.index].child[i2];
                    }
                    goto TRAVERSE;
                }

                st.push(nodes[cur.index].child[i], dist[i]);
                st.push(nodes[cur.index].child[i2], dist[i2]);

                i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                st.push(nodes[cur.index].child[i], dist[i]);
                if (mask == 0) { // three boxes were hit
                    st.sort_top3();
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }

                i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                st.push(nodes[cur.index].child[i], dist[i]);
                if (mask == 0) { // four boxes were hit
                    st.sort_top4();
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }

                uint32_t size_before = st.stack_size;

                // from five to eight boxes were hit
                do {
                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], dist[i]);
                } while (mask != 0);

                const int count = int(st.stack_size - size_before + 4);
                st.sort_topN(count);
                cur.index = st.pop_index();
                goto TRAVERSE;
            }
        } else {
            const int tri_start = int(nodes[cur.index].child[0] & PRIM_INDEX_BITS),
                      tri_end = int(tri_start + nodes[cur.index].child[1]);
            res |= IntersectTris_ClosestHit(ro, rd, mtris, tri_start, tri_end, obj_index, inter);
        }
    }

    return res;
}

bool Ray::Ref::Traverse_BLAS_WithStack_AnyHit(const float ro[3], const float rd[3], const float inv_d[3],
                                              const bvh_node_t *nodes, uint32_t root_index, const tri_accel_t *tris,
                                              const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                              int obj_index, hit_data_t &inter) {
    uint32_t stack[MAX_STACK_SIZE];
    uint32_t stack_size = 0;

    stack[stack_size++] = root_index;

    while (stack_size) {
        const uint32_t cur = stack[--stack_size];

        if (!bbox_test(ro, inv_d, inter.t, nodes[cur])) {
            continue;
        }

        if (!is_leaf_node(nodes[cur])) {
            stack[stack_size++] = far_child(rd, nodes[cur]);
            stack[stack_size++] = near_child(rd, nodes[cur]);
        } else {
            const int tri_start = int(nodes[cur].prim_index & PRIM_INDEX_BITS),
                      tri_end = int(tri_start + nodes[cur].prim_count);
            const bool hit_found =
                IntersectTris_AnyHit(ro, rd, tris, materials, tri_indices, tri_start, tri_end, obj_index, inter);
            if (hit_found) {
                const bool is_backfacing = inter.prim_index < 0;
                const uint32_t prim_index = is_backfacing ? -inter.prim_index - 1 : inter.prim_index;

                if ((!is_backfacing && (materials[tri_indices[prim_index]].front_mi & MATERIAL_SOLID_BIT)) ||
                    (is_backfacing && (materials[tri_indices[prim_index]].back_mi & MATERIAL_SOLID_BIT))) {
                    return true;
                }
            }
        }
    }

    return false;
}

bool Ray::Ref::Traverse_BLAS_WithStack_AnyHit(const float ro[3], const float rd[3], const float inv_d[3],
                                              const bvh2_node_t *nodes, uint32_t root_index, const tri_accel_t *tris,
                                              const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                              int obj_index, hit_data_t &inter) {
    uint32_t stack[MAX_STACK_SIZE];
    uint32_t stack_size = 0;

    stack[stack_size++] = 0x1fffffffu;

    uint32_t cur = root_index;
    while (stack_size) {
        uint32_t leaf_node = 0;
        while (stack_size && (cur & BVH2_PRIM_COUNT_BITS) == 0) {
            const bvh2_node_t &n = nodes[cur];

            uint32_t children[2] = {n.left_child, n.right_child};

            const float ch0_min[3] = {n.ch_data0[0], n.ch_data0[2], n.ch_data2[0]};
            const float ch0_max[3] = {n.ch_data0[1], n.ch_data0[3], n.ch_data2[1]};

            const float ch1_min[3] = {n.ch_data1[0], n.ch_data1[2], n.ch_data2[2]};
            const float ch1_max[3] = {n.ch_data1[1], n.ch_data1[3], n.ch_data2[3]};

            float ch0_dist, ch1_dist;
            const bool ch0_res = bbox_test(ro, inv_d, inter.t, ch0_min, ch0_max, ch0_dist);
            const bool ch1_res = bbox_test(ro, inv_d, inter.t, ch1_min, ch1_max, ch1_dist);

            if (!ch0_res && !ch1_res) {
                cur = stack[--stack_size];
            } else {
                cur = ch0_res ? children[0] : children[1];
                if (ch0_res && ch1_res) {
                    if (ch1_dist < ch0_dist) {
                        const uint32_t temp = cur;
                        cur = children[1];
                        children[1] = temp;
                    }
                    stack[stack_size++] = children[1];
                }
            }
            if ((cur & BVH2_PRIM_COUNT_BITS) != 0 && (leaf_node & BVH2_PRIM_COUNT_BITS) == 0) {
                leaf_node = cur;
                cur = stack[--stack_size];
            }
            if ((leaf_node & BVH2_PRIM_COUNT_BITS) != 0) {
                break;
            }
        }

        while ((leaf_node & BVH2_PRIM_COUNT_BITS) != 0) {
            const int tri_start = int(leaf_node & BVH2_PRIM_INDEX_BITS),
                      tri_end = int(tri_start + ((leaf_node & BVH2_PRIM_COUNT_BITS) >> 29) + 1);
            assert((tri_start % 8) == 0);
            assert((tri_end - tri_start) <= 8);
            const bool hit_found =
                IntersectTris_AnyHit(ro, rd, tris, materials, tri_indices, tri_start, tri_end, obj_index, inter);
            if (hit_found) {
                const bool is_backfacing = inter.prim_index < 0;
                const uint32_t prim_index = is_backfacing ? -inter.prim_index - 1 : inter.prim_index;

                if ((!is_backfacing && (materials[tri_indices[prim_index]].front_mi & MATERIAL_SOLID_BIT)) ||
                    (is_backfacing && (materials[tri_indices[prim_index]].back_mi & MATERIAL_SOLID_BIT))) {
                    return true;
                }
            }

            leaf_node = cur;
            if ((cur & BVH2_PRIM_COUNT_BITS) != 0) {
                cur = stack[--stack_size];
            }
        }
    }

    return false;
}

bool Ray::Ref::Traverse_BLAS_WithStack_AnyHit(const float ro[3], const float rd[3], const float inv_d[3],
                                              const wbvh_node_t *nodes, const uint32_t root_index,
                                              const mtri_accel_t *mtris, const tri_mat_data_t *materials,
                                              const uint32_t *tri_indices, int obj_index, hit_data_t &inter) {
    TraversalStack<MAX_STACK_SIZE> st;
    st.push(root_index, 0.0f);

    while (!st.empty()) {
        stack_entry_t cur = st.pop();

        if (cur.dist > inter.t) {
            continue;
        }

    TRAVERSE:
        if (!is_leaf_node(nodes[cur.index])) {
            alignas(16) float dist[8];
            long mask = bbox_test_oct(ro, inv_d, inter.t, nodes[cur.index], dist);
            if (mask) {
                long i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                if (mask == 0) { // only one box was hit
                    cur.index = nodes[cur.index].child[i];
                    goto TRAVERSE;
                }

                const long i2 = GetFirstBit(mask);
                mask = ClearBit(mask, i2);
                if (mask == 0) { // two boxes were hit
                    if (dist[i] < dist[i2]) {
                        st.push(nodes[cur.index].child[i2], dist[i2]);
                        cur.index = nodes[cur.index].child[i];
                    } else {
                        st.push(nodes[cur.index].child[i], dist[i]);
                        cur.index = nodes[cur.index].child[i2];
                    }
                    goto TRAVERSE;
                }

                st.push(nodes[cur.index].child[i], dist[i]);
                st.push(nodes[cur.index].child[i2], dist[i2]);

                i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                st.push(nodes[cur.index].child[i], dist[i]);
                if (mask == 0) { // three boxes were hit
                    st.sort_top3();
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }

                i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                st.push(nodes[cur.index].child[i], dist[i]);
                if (mask == 0) { // four boxes were hit
                    st.sort_top4();
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }

                uint32_t size_before = st.stack_size;

                // from five to eight boxes were hit
                do {
                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], dist[i]);
                } while (mask != 0);

                const int count = int(st.stack_size - size_before + 4);
                st.sort_topN(count);
                cur.index = st.pop_index();
                goto TRAVERSE;
            }
        } else {
            const int tri_start = int(nodes[cur.index].child[0] & PRIM_INDEX_BITS),
                      tri_end = int(tri_start + nodes[cur.index].child[1]);
            const bool hit_found =
                IntersectTris_AnyHit(ro, rd, mtris, materials, tri_indices, tri_start, tri_end, obj_index, inter);
            if (hit_found) {
                const bool is_backfacing = inter.prim_index < 0;
                const uint32_t prim_index = is_backfacing ? -inter.prim_index - 1 : inter.prim_index;

                if ((!is_backfacing && (materials[tri_indices[prim_index]].front_mi & MATERIAL_SOLID_BIT)) ||
                    (is_backfacing && (materials[tri_indices[prim_index]].back_mi & MATERIAL_SOLID_BIT))) {
                    return true;
                }
            }
        }
    }

    return false;
}

void Ray::Ref::TransformRay(const float ro[3], const float rd[3], const float *xform, float out_ro[3],
                            float out_rd[3]) {
    out_ro[0] = xform[0] * ro[0] + xform[4] * ro[1] + xform[8] * ro[2] + xform[12];
    out_ro[1] = xform[1] * ro[0] + xform[5] * ro[1] + xform[9] * ro[2] + xform[13];
    out_ro[2] = xform[2] * ro[0] + xform[6] * ro[1] + xform[10] * ro[2] + xform[14];

    out_rd[0] = xform[0] * rd[0] + xform[4] * rd[1] + xform[8] * rd[2];
    out_rd[1] = xform[1] * rd[0] + xform[5] * rd[1] + xform[9] * rd[2];
    out_rd[2] = xform[2] * rd[0] + xform[6] * rd[1] + xform[10] * rd[2];
}

Ray::Ref::fvec4 Ray::Ref::TransformPoint(const fvec4 &p, const float *xform) {
    return fvec4{xform[0] * p.get<0>() + xform[4] * p.get<1>() + xform[8] * p.get<2>() + xform[12],
                 xform[1] * p.get<0>() + xform[5] * p.get<1>() + xform[9] * p.get<2>() + xform[13],
                 xform[2] * p.get<0>() + xform[6] * p.get<1>() + xform[10] * p.get<2>() + xform[14], 0.0f};
}

Ray::Ref::fvec4 Ray::Ref::TransformDirection(const fvec4 &p, const float *xform) {
    return fvec4{xform[0] * p.get<0>() + xform[4] * p.get<1>() + xform[8] * p.get<2>(),
                 xform[1] * p.get<0>() + xform[5] * p.get<1>() + xform[9] * p.get<2>(),
                 xform[2] * p.get<0>() + xform[6] * p.get<1>() + xform[10] * p.get<2>(), 0.0f};
}

Ray::Ref::fvec4 Ray::Ref::TransformNormal(const fvec4 &n, const float *inv_xform) {
    return fvec4{inv_xform[0] * n.get<0>() + inv_xform[1] * n.get<1>() + inv_xform[2] * n.get<2>(),
                 inv_xform[4] * n.get<0>() + inv_xform[5] * n.get<1>() + inv_xform[6] * n.get<2>(),
                 inv_xform[8] * n.get<0>() + inv_xform[9] * n.get<1>() + inv_xform[10] * n.get<2>(), 0.0f};
}

float Ray::Ref::get_texture_lod(const Cpu::TexStorageBase *const textures[], const uint32_t index, const fvec2 &duv_dx,
                                const fvec2 &duv_dy) {
#ifdef FORCE_TEXTURE_LOD
    const float lod = float(FORCE_TEXTURE_LOD);
#else
    fvec2 sz;
    textures[index >> 28]->GetFRes(index & 0x00ffffff, 0, value_ptr(sz));
    const fvec2 _duv_dx = duv_dx * sz, _duv_dy = duv_dy * sz;
    const fvec2 _diagonal = _duv_dx + _duv_dy;

    // Find minimal dimention of parallelogram
    const float min_length2 = fminf(fminf(_duv_dx.length2(), _duv_dy.length2()), _diagonal.length2());
    // Find lod
    float lod = fast_log2(min_length2);
    // Substruct 1 from lod to always have 4 texels for interpolation
    lod = clamp(0.5f * lod - 1.0f, 0.0f, float(MAX_MIP_LEVEL));
#endif
    return lod;
}

float Ray::Ref::get_texture_lod(const Cpu::TexStorageBase *const textures[], const uint32_t index, const float lambda) {
#ifdef FORCE_TEXTURE_LOD
    const float lod = float(FORCE_TEXTURE_LOD);
#else
    fvec2 res;
    textures[index >> 28]->GetFRes(index & 0x00ffffff, 0, value_ptr(res));
    // Find lod
    float lod = lambda + 0.5f * fast_log2(res.get<0>() * res.get<1>());
    // Substruct 1 from lod to always have 4 texels for interpolation
    lod = clamp(lod - 1.0f, 0.0f, float(MAX_MIP_LEVEL));
#endif
    return lod;
}

Ray::Ref::fvec4 Ray::Ref::SampleNearest(const Cpu::TexStorageBase *const textures[], const uint32_t index,
                                        const fvec2 &uvs, const int lod) {
    const Cpu::TexStorageBase &storage = *textures[index >> 28];
    const auto &pix = storage.Fetch(int(index & 0x00ffffff), uvs.get<0>(), uvs.get<1>(), lod);
    return fvec4{pix.v[0], pix.v[1], pix.v[2], pix.v[3]};
}

Ray::Ref::fvec4 Ray::Ref::SampleBilinear(const Cpu::TexStorageBase *const textures[], const uint32_t index,
                                         const fvec2 &uvs, const int lod, const fvec2 &rand) {
    const Cpu::TexStorageBase &storage = *textures[index >> 28];

    const int tex = int(index & 0x00ffffff);
    fvec2 img_size;
    storage.GetFRes(tex, lod, value_ptr(img_size));

    fvec2 _uvs = fract(uvs);
    _uvs = _uvs * img_size - 0.5f;

#if USE_STOCH_TEXTURE_FILTERING
    // Jitter UVs
    _uvs += rand;

    const auto &p00 = storage.Fetch(tex, int(_uvs.get<0>()), int(_uvs.get<1>()), lod);
    return fvec4{p00.v};
#else  // USE_STOCH_TEXTURE_FILTERING
    const auto &p00 = storage.Fetch(tex, int(_uvs.get<0>()) + 0, int(_uvs.get<1>()) + 0, lod);
    const auto &p01 = storage.Fetch(tex, int(_uvs.get<0>()) + 1, int(_uvs.get<1>()) + 0, lod);
    const auto &p10 = storage.Fetch(tex, int(_uvs.get<0>()) + 0, int(_uvs.get<1>()) + 1, lod);
    const auto &p11 = storage.Fetch(tex, int(_uvs.get<0>()) + 1, int(_uvs.get<1>()) + 1, lod);

    const float kx = fract(_uvs.get<0>()), ky = fract(_uvs.get<1>());

    const auto p0 = fvec4{p01.v[0] * kx + p00.v[0] * (1 - kx), p01.v[1] * kx + p00.v[1] * (1 - kx),
                          p01.v[2] * kx + p00.v[2] * (1 - kx), p01.v[3] * kx + p00.v[3] * (1 - kx)};

    const auto p1 = fvec4{p11.v[0] * kx + p10.v[0] * (1 - kx), p11.v[1] * kx + p10.v[1] * (1 - kx),
                          p11.v[2] * kx + p10.v[2] * (1 - kx), p11.v[3] * kx + p10.v[3] * (1 - kx)};

    return (p1 * ky + p0 * (1.0f - ky));
#endif // USE_STOCH_TEXTURE_FILTERING
}

Ray::Ref::fvec4 Ray::Ref::SampleBilinear(const Cpu::TexStorageBase &storage, const uint32_t tex, const fvec2 &iuvs,
                                         const int lod, const fvec2 &rand) {
#if USE_STOCH_TEXTURE_FILTERING
    // Jitter UVs
    fvec2 _uvs = iuvs + rand;

    const auto &p00 = storage.Fetch(tex, int(_uvs.get<0>()), int(_uvs.get<1>()), lod);
    return fvec4{p00.v};
#else  // USE_STOCH_TEXTURE_FILTERING
    const auto &p00 = storage.Fetch(int(tex), int(iuvs.get<0>()) + 0, int(iuvs.get<1>()) + 0, lod);
    const auto &p01 = storage.Fetch(int(tex), int(iuvs.get<0>()) + 1, int(iuvs.get<1>()) + 0, lod);
    const auto &p10 = storage.Fetch(int(tex), int(iuvs.get<0>()) + 0, int(iuvs.get<1>()) + 1, lod);
    const auto &p11 = storage.Fetch(int(tex), int(iuvs.get<0>()) + 1, int(iuvs.get<1>()) + 1, lod);

    const fvec2 k = fract(iuvs);

    const auto _p00 = fvec4{p00.v[0], p00.v[1], p00.v[2], p00.v[3]};
    const auto _p01 = fvec4{p01.v[0], p01.v[1], p01.v[2], p01.v[3]};
    const auto _p10 = fvec4{p10.v[0], p10.v[1], p10.v[2], p10.v[3]};
    const auto _p11 = fvec4{p11.v[0], p11.v[1], p11.v[2], p11.v[3]};

    const fvec4 p0X = _p01 * k.get<0>() + _p00 * (1 - k.get<0>());
    const fvec4 p1X = _p11 * k.get<0>() + _p10 * (1 - k.get<0>());

    return (p1X * k.get<1>() + p0X * (1 - k.get<1>()));
#endif // USE_STOCH_TEXTURE_FILTERING
}

Ray::Ref::fvec4 Ray::Ref::SampleTrilinear(const Cpu::TexStorageBase *const textures[], const uint32_t index,
                                          const fvec2 &uvs, const float lod, const fvec2 &rand) {
    const fvec4 col1 = SampleBilinear(textures, index, uvs, int(floorf(lod)), rand);
    const fvec4 col2 = SampleBilinear(textures, index, uvs, int(ceilf(lod)), rand);

    const float k = fract(lod);
    return col1 * (1 - k) + col2 * k;
}

Ray::Ref::fvec4 Ray::Ref::SampleAnisotropic(const Cpu::TexStorageBase *const textures[], const uint32_t index,
                                            const fvec2 &uvs, const fvec2 &duv_dx, const fvec2 &duv_dy) {
    const Cpu::TexStorageBase &storage = *textures[index >> 28];
    const int tex = int(index & 0x00ffffff);

    fvec2 sz;
    storage.GetFRes(tex, 0, value_ptr(sz));

    const fvec2 _duv_dx = abs(duv_dx * sz);
    const fvec2 _duv_dy = abs(duv_dy * sz);

    const float l1 = length(_duv_dx);
    const float l2 = length(_duv_dy);

    float lod, k;
    fvec2 step;

    if (l1 <= l2) {
        lod = fast_log2(fminf(_duv_dx.get<0>(), _duv_dx.get<1>()));
        k = l1 / l2;
        step = duv_dy;
    } else {
        lod = fast_log2(fminf(_duv_dy.get<0>(), _duv_dy.get<1>()));
        k = l2 / l1;
        step = duv_dx;
    }

    lod = clamp(lod, 0.0f, float(MAX_MIP_LEVEL));

    fvec2 _uvs = uvs - step * 0.5f;

    int num = int(2.0f / k);
    num = clamp(num, 1, 4);

    step = step / float(num);

    auto res = fvec4{0.0f};

    const int lod1 = int(floorf(lod));
    const int lod2 = int(ceilf(lod));

    fvec2 size1, size2;
    storage.GetFRes(tex, lod1, value_ptr(size1));
    storage.GetFRes(tex, lod2, value_ptr(size2));

    const float kz = fract(lod);

    for (int i = 0; i < num; ++i) {
        _uvs = fract(_uvs);

        const fvec2 _uvs1 = _uvs * size1;
        res += (1 - kz) * SampleBilinear(storage, tex, _uvs1, lod1, {});

        if (kz > 0.0001f) {
            const fvec2 _uvs2 = _uvs * size2;
            res += kz * SampleBilinear(storage, tex, _uvs2, lod2, {});
        }

        _uvs = _uvs + step;
    }

    return res / float(num);
}

Ray::Ref::fvec4 Ray::Ref::SampleLatlong_RGBE(const Cpu::TexStorageRGBA &storage, const uint32_t index, const fvec4 &dir,
                                             float y_rotation, const fvec2 &rand) {
    const float theta = acosf(clamp(dir.get<1>(), -1.0f, 1.0f)) / PI;
    float phi = atan2f(dir.get<2>(), dir.get<0>()) + y_rotation;
    if (phi < 0) {
        phi += 2 * PI;
    }
    if (phi > 2 * PI) {
        phi -= 2 * PI;
    }

    const float u = fract(0.5f * phi / PI);

    const int tex = int(index & 0x00ffffff);
    fvec2 size;
    storage.GetFRes(tex, 0, value_ptr(size));

    fvec2 uvs = fvec2{u, theta} * size;

#if USE_STOCH_TEXTURE_FILTERING
    // Jitter UVs
    uvs += rand;
    const ivec2 iuvs = ivec2(uvs);

    const auto &p00 = storage.Get(tex, iuvs.get<0>(), iuvs.get<1>(), 0);
    return rgbe_to_rgb(p00);
#else  // USE_STOCH_TEXTURE_FILTERING
    const ivec2 iuvs = ivec2(uvs);

    const auto &p00 = storage.Get(tex, iuvs.get<0>() + 0, iuvs.get<1>() + 0, 0);
    const auto &p01 = storage.Get(tex, iuvs.get<0>() + 1, iuvs.get<1>() + 0, 0);
    const auto &p10 = storage.Get(tex, iuvs.get<0>() + 0, iuvs.get<1>() + 1, 0);
    const auto &p11 = storage.Get(tex, iuvs.get<0>() + 1, iuvs.get<1>() + 1, 0);

    const fvec2 k = fract(uvs);

    const fvec4 _p00 = rgbe_to_rgb(p00), _p01 = rgbe_to_rgb(p01);
    const fvec4 _p10 = rgbe_to_rgb(p10), _p11 = rgbe_to_rgb(p11);

    const fvec4 p0X = _p01 * k.get<0>() + _p00 * (1 - k.get<0>());
    const fvec4 p1X = _p11 * k.get<0>() + _p10 * (1 - k.get<0>());

    return (p1X * k.get<1>() + p0X * (1 - k.get<1>()));
#endif // USE_STOCH_TEXTURE_FILTERING
}

void Ray::Ref::IntersectScene(Span<ray_data_t> rays, const int min_transp_depth, const int max_transp_depth,
                              const uint32_t rand_seq[], const uint32_t rand_seed, const int iteration,
                              const scene_data_t &sc, const uint32_t root_index,
                              const Cpu::TexStorageBase *const textures[], Span<hit_data_t> out_inter) {
    for (int i = 0; i < rays.size(); ++i) {
        ray_data_t &r = rays[i];
        hit_data_t &inter = out_inter[i];

        const fvec4 rd = make_fvec3(r.d);
        fvec4 ro = make_fvec3(r.o);

        const uint32_t ray_flags = (1u << get_ray_type(r.depth));

        const uint32_t px_hash = hash(r.xy);
        const uint32_t rand_hash = hash_combine(px_hash, rand_seed);

        uint32_t rand_dim = RAND_DIM_BASE_COUNT + get_total_depth(r.depth) * RAND_DIM_BOUNCE_COUNT;
        while (true) {
            const float t_val = inter.t;

            bool hit_found = false;
            if (sc.wnodes) {
                hit_found =
                    Traverse_TLAS_WithStack_ClosestHit(value_ptr(ro), value_ptr(rd), ray_flags, sc.wnodes, root_index,
                                                       sc.mesh_instances, sc.mtris, sc.tri_indices, inter);
            } else {
                hit_found =
                    Traverse_TLAS_WithStack_ClosestHit(value_ptr(ro), value_ptr(rd), ray_flags, sc.nodes, root_index,
                                                       sc.mesh_instances, sc.tris, sc.tri_indices, inter);
            }

            if (!hit_found) {
                break;
            }

            const bool is_backfacing = (inter.prim_index < 0);
            const uint32_t tri_index = is_backfacing ? -inter.prim_index - 1 : inter.prim_index;

            if ((!is_backfacing && (sc.tri_materials[tri_index].front_mi & MATERIAL_SOLID_BIT)) ||
                (is_backfacing && (sc.tri_materials[tri_index].back_mi & MATERIAL_SOLID_BIT))) {
                // solid hit found
                break;
            }

            const material_t *mat = is_backfacing
                                        ? &sc.materials[sc.tri_materials[tri_index].back_mi & MATERIAL_INDEX_BITS]
                                        : &sc.materials[sc.tri_materials[tri_index].front_mi & MATERIAL_INDEX_BITS];

            const vertex_t &v1 = sc.vertices[sc.vtx_indices[tri_index * 3 + 0]];
            const vertex_t &v2 = sc.vertices[sc.vtx_indices[tri_index * 3 + 1]];
            const vertex_t &v3 = sc.vertices[sc.vtx_indices[tri_index * 3 + 2]];

            const float w = 1.0f - inter.u - inter.v;
            const fvec2 uvs = fvec2(v1.t) * w + fvec2(v2.t) * inter.u + fvec2(v3.t) * inter.v;

            const fvec2 mix_term_rand =
                get_scrambled_2d_rand(rand_dim + RAND_DIM_BSDF_PICK, rand_hash, iteration - 1, rand_seq);
            const fvec2 tex_rand = get_scrambled_2d_rand(rand_dim + RAND_DIM_TEX, rand_hash, iteration - 1, rand_seq);

            float trans_r = mix_term_rand.get<0>();

            // resolve mix material
            while (mat->type == eShadingNode::Mix) {
                float mix_val = mat->strength;
                const uint32_t base_texture = mat->textures[BASE_TEXTURE];
                if (base_texture != 0xffffffff) {
                    fvec4 tex_color = SampleBilinear(textures, base_texture, uvs, 0, tex_rand);
                    if (base_texture & TEX_YCOCG_BIT) {
                        tex_color = YCoCg_to_RGB(tex_color);
                    }
                    if (base_texture & TEX_SRGB_BIT) {
                        tex_color = srgb_to_linear(tex_color);
                    }
                    mix_val *= tex_color.get<0>();
                }

                if (trans_r > mix_val) {
                    mat = &sc.materials[mat->textures[MIX_MAT1]];
                    trans_r = safe_div_pos(trans_r - mix_val, 1.0f - mix_val);
                } else {
                    mat = &sc.materials[mat->textures[MIX_MAT2]];
                    trans_r = safe_div_pos(trans_r, mix_val);
                }
            }

            if (mat->type != eShadingNode::Transparent) {
                break;
            }

            const bool can_terminate_path = USE_PATH_TERMINATION && get_transp_depth(r.depth) > min_transp_depth;

            const float lum = fmaxf(r.c[0], fmaxf(r.c[1], r.c[2]));
            const float p = mix_term_rand.get<1>();
            const float q = can_terminate_path ? fmaxf(0.05f, 1.0f - lum) : 0.0f;
            if (p < q || lum == 0.0f || get_transp_depth(r.depth) + 1 >= max_transp_depth) {
                // terminate ray
                r.c[0] = r.c[1] = r.c[2] = 0.0f;
                break;
            }

            r.c[0] *= mat->base_color[0] / (1.0f - q);
            r.c[1] *= mat->base_color[1] / (1.0f - q);
            r.c[2] *= mat->base_color[2] / (1.0f - q);

            const float t = inter.t + HIT_BIAS;
            ro += rd * t;

            // discard current intersection
            inter.v = -1.0f;
            inter.t = t_val - inter.t;

            r.depth += pack_ray_depth(0, 0, 0, 1);
            rand_dim += RAND_DIM_BOUNCE_COUNT;
        }

        inter.t += length(make_fvec3(r.o) - ro);
    }
}

Ray::Ref::fvec4 Ray::Ref::IntersectScene(const shadow_ray_t &r, const int max_transp_depth, const scene_data_t &sc,
                                         const uint32_t root_index, const uint32_t rand_seq[], const uint32_t rand_seed,
                                         const int iteration, const Cpu::TexStorageBase *const textures[]) {
    const fvec4 rd = make_fvec3(r.d);
    fvec4 ro = make_fvec3(r.o);
    fvec4 rc = make_fvec3(r.c);
    int depth = get_transp_depth(r.depth);

    const uint32_t px_hash = hash(r.xy);
    const uint32_t rand_hash = hash_combine(px_hash, rand_seed);

    uint32_t rand_dim = RAND_DIM_BASE_COUNT + get_total_depth(r.depth) * RAND_DIM_BOUNCE_COUNT;

    float dist = r.dist > 0.0f ? r.dist : MAX_DIST;
    while (dist > HIT_BIAS) {
        hit_data_t inter;
        inter.t = dist;

        bool solid_hit = false;
        if (sc.wnodes) {
            solid_hit =
                Traverse_TLAS_WithStack_AnyHit(value_ptr(ro), value_ptr(rd), RAY_TYPE_SHADOW, sc.wnodes, root_index,
                                               sc.mesh_instances, sc.mtris, sc.tri_materials, sc.tri_indices, inter);
        } else {
            solid_hit =
                Traverse_TLAS_WithStack_AnyHit(value_ptr(ro), value_ptr(rd), RAY_TYPE_SHADOW, sc.nodes, root_index,
                                               sc.mesh_instances, sc.tris, sc.tri_materials, sc.tri_indices, inter);
        }

        if (solid_hit || depth > max_transp_depth) {
            rc = 0.0f;
        }

        if (solid_hit || depth > max_transp_depth || inter.v < 0.0f) {
            break;
        }

        const bool is_backfacing = (inter.prim_index < 0);
        const uint32_t tri_index = is_backfacing ? -inter.prim_index - 1 : inter.prim_index;

        const uint32_t mat_index = is_backfacing ? (sc.tri_materials[tri_index].back_mi & MATERIAL_INDEX_BITS)
                                                 : (sc.tri_materials[tri_index].front_mi & MATERIAL_INDEX_BITS);

        const vertex_t &v1 = sc.vertices[sc.vtx_indices[tri_index * 3 + 0]];
        const vertex_t &v2 = sc.vertices[sc.vtx_indices[tri_index * 3 + 1]];
        const vertex_t &v3 = sc.vertices[sc.vtx_indices[tri_index * 3 + 2]];

        const float w = 1.0f - inter.u - inter.v;
        const fvec2 sh_uvs = fvec2(v1.t) * w + fvec2(v2.t) * inter.u + fvec2(v3.t) * inter.v;

        const fvec2 tex_rand = get_scrambled_2d_rand(rand_dim + RAND_DIM_TEX, rand_hash, iteration - 1, rand_seq);

        struct {
            uint32_t index;
            float weight;
        } stack[16];
        int stack_size = 0;

        stack[stack_size++] = {mat_index, 1.0f};

        fvec4 throughput = 0.0f;

        while (stack_size--) {
            const material_t *mat = &sc.materials[stack[stack_size].index];
            const float weight = stack[stack_size].weight;

            // resolve mix material
            if (mat->type == eShadingNode::Mix) {
                float mix_val = mat->strength;
                const uint32_t base_texture = mat->textures[BASE_TEXTURE];
                if (base_texture != 0xffffffff) {
                    fvec4 tex_color = SampleBilinear(textures, base_texture, sh_uvs, 0, tex_rand);
                    if (base_texture & TEX_YCOCG_BIT) {
                        tex_color = YCoCg_to_RGB(tex_color);
                    }
                    if (base_texture & TEX_SRGB_BIT) {
                        tex_color = srgb_to_linear(tex_color);
                    }
                    mix_val *= tex_color.get<0>();
                }

                stack[stack_size++] = {mat->textures[MIX_MAT1], weight * (1.0f - mix_val)};
                stack[stack_size++] = {mat->textures[MIX_MAT2], weight * mix_val};
            } else if (mat->type == eShadingNode::Transparent) {
                throughput += weight * make_fvec3(mat->base_color);
            }
        }

        rc *= throughput;
        if (lum(rc) < FLT_EPS) {
            break;
        }

        const float t = inter.t + HIT_BIAS;
        ro += rd * t;
        dist -= t;

        ++depth;
        rand_dim += RAND_DIM_BOUNCE_COUNT;
    }

    return rc;
}

void Ray::Ref::SampleLightSource(const fvec4 &P, const fvec4 &T, const fvec4 &B, const fvec4 &N, const scene_data_t &sc,
                                 const Cpu::TexStorageBase *const textures[], const float rand_pick_light,
                                 const fvec2 rand_light_uv, const fvec2 rand_tex_uv, light_sample_t &ls) {
    float u1 = rand_pick_light;

    uint32_t light_index;
    float factor;
    if (USE_HIERARCHICAL_NEE) {
        factor = 1.0f;
        uint32_t i = 0; // start from root
        while ((i & LEAF_NODE_BIT) == 0) {
            alignas(16) float importance[8];
            calc_lnode_importance(sc.light_cwnodes[i], P, importance);

            const float total_importance =
                hsum(fvec4{&importance[0], vector_aligned} + fvec4{&importance[4], vector_aligned});
            if (total_importance == 0.0f) {
                // failed to find lightsource for sampling
                return;
            }

            alignas(16) float factors[8];
            UNROLLED_FOR(j, 8, { factors[j] = importance[j] / total_importance; })

            float factors_cdf[9] = {};
            UNROLLED_FOR(j, 8, { factors_cdf[j + 1] = factors_cdf[j] + factors[j]; })
            // make sure cdf ends with 1.0
            UNROLLED_FOR(j, 8, {
                if (factors_cdf[j + 1] == factors_cdf[8]) {
                    factors_cdf[j + 1] = 1.01f;
                }
            })

            ivec4 less_eq[2] = {};
            where(fvec4{&factors_cdf[1]} <= u1, less_eq[0]) = 1;
            where(fvec4{&factors_cdf[5]} <= u1, less_eq[1]) = 1;

            const int next = hsum(less_eq[0] + less_eq[1]);
            assert(next < 8);

            u1 = fract((u1 - factors_cdf[next]) / factors[next]);
            i = sc.light_cwnodes[i].child[next];
            factor *= factors[next];
        }
        light_index = (i & PRIM_INDEX_BITS);
        factor = 1.0f / factor;
    } else {
        light_index = std::min(uint32_t(u1 * float(sc.li_indices.size())), uint32_t(sc.li_indices.size() - 1));
        u1 = u1 * float(sc.li_indices.size()) - float(light_index);
        light_index = sc.li_indices[light_index];
        factor = float(sc.li_indices.size());
    }
    const light_t &l = sc.lights[light_index];

    ls.col = make_fvec3(l.col);
    ls.cast_shadow = l.cast_shadow ? 1 : 0;
    ls.from_env = 0;

    if (l.type == LIGHT_TYPE_SPHERE) {
        const float r1 = rand_light_uv.get<0>(), r2 = rand_light_uv.get<1>();

        const fvec4 center = make_fvec3(l.sph.pos);
        float d;
        const fvec4 light_normal = normalize_len(center - P, d);

        const float temp = sqrtf(d * d - l.sph.radius * l.sph.radius);
        const float disk_radius = (temp * l.sph.radius) / d;
        float disk_dist = l.sph.radius > 0.0f ? ((temp * disk_radius) / l.sph.radius) : d;
        const fvec4 sampled_dir = normalize_len(map_to_cone(r1, r2, disk_dist * light_normal, disk_radius), disk_dist);

        if (l.sph.radius > 0.0f) {
            // TODO: Find better way to do this
            const float ls_dist = sphere_intersection(center, l.sph.radius, P, sampled_dir);

            const fvec4 light_surf_pos = P + sampled_dir * ls_dist;
            const fvec4 light_forward = normalize(light_surf_pos - center);

            const float sampled_area = PI * disk_radius * disk_radius;
            const float cos_theta = dot(sampled_dir, light_normal);

            ls.lp = offset_ray(light_surf_pos, light_forward);
            ls.pdf = (disk_dist * disk_dist) / (sampled_area * cos_theta);
        } else {
            ls.lp = center;
            ls.pdf = (disk_dist * disk_dist) / PI;
        }
        ls.L = sampled_dir;
        ls.area = PI * disk_radius * disk_radius;
        ls.ray_flags = l.ray_visibility;

        if (!l.visible) {
            ls.area = 0.0f;
        }

        if (l.sph.spot > 0.0f) {
            const float _dot = -dot(ls.L, fvec4{l.sph.dir});
            if (_dot > 0.0f) {
                const float _angle = acosf(saturate(_dot));
                ls.col *= saturate((l.sph.spot - _angle) / l.sph.blend);
            } else {
                ls.col *= 0.0f;
            }
        }
    } else if (l.type == LIGHT_TYPE_DIR) {
        ls.L = make_fvec3(l.dir.dir);
        ls.area = 0.0f;
        ls.pdf = 1.0f;
        if (l.dir.angle != 0.0f) {
            const float r1 = rand_light_uv.get<0>(), r2 = rand_light_uv.get<1>();

            const float radius = tanf(l.dir.angle);
            ls.L = normalize(map_to_cone(r1, r2, ls.L, radius));
            ls.area = PI * radius * radius;

            const float cos_theta = dot(ls.L, make_fvec3(l.dir.dir));
            ls.pdf = 1.0f / (ls.area * cos_theta);
        }
        ls.lp = P + ls.L;
        ls.dist_mul = MAX_DIST;
        ls.ray_flags = l.ray_visibility;

        if (!l.visible) {
            ls.area = 0.0f;
        }
    } else if (l.type == LIGHT_TYPE_RECT) {
        const fvec4 light_pos = make_fvec3(l.rect.pos);
        const fvec4 light_u = make_fvec3(l.rect.u), light_v = make_fvec3(l.rect.v);
        const fvec4 light_forward = normalize(cross(light_u, light_v));

        fvec4 lp;
        float pdf = 0.0f;

        if (USE_SPHERICAL_AREA_LIGHT_SAMPLING) {
            pdf = SampleSphericalRectangle(P, light_pos, light_u, light_v, rand_light_uv, &lp);
        }
        if (pdf <= 0.0f) {
            const float r1 = rand_light_uv.get<0>() - 0.5f, r2 = rand_light_uv.get<1>() - 0.5f;
            lp = light_pos + light_u * r1 + light_v * r2;
        }

        float ls_dist;
        ls.L = normalize_len(lp - P, ls_dist);
        ls.ray_flags = l.ray_visibility;

        const float cos_theta = dot(-ls.L, light_forward);
        if (cos_theta > 0.0f) {
            ls.lp = offset_ray(lp, light_forward);
            ls.pdf = (pdf > 0.0f) ? pdf : (ls_dist * ls_dist) / (l.rect.area * cos_theta);
            ls.area = l.visible ? l.rect.area : 0.0f;
            if (l.sky_portal != 0) {
                fvec4 env_col = make_fvec3(sc.env.env_col);
                if (sc.env.env_map != 0xffffffff) {
                    env_col *= SampleLatlong_RGBE(*static_cast<const Cpu::TexStorageRGBA *>(textures[0]),
                                                  sc.env.env_map, ls.L, sc.env.env_map_rotation, rand_tex_uv);
                }
                ls.col *= env_col;
                ls.from_env = 1;
            }
        }
    } else if (l.type == LIGHT_TYPE_DISK) {
        const fvec4 light_pos = make_fvec3(l.disk.pos);
        const fvec4 light_u = make_fvec3(l.disk.u), light_v = make_fvec3(l.disk.v);

        const float r1 = rand_light_uv.get<0>(), r2 = rand_light_uv.get<1>();

        fvec2 offset = 2.0f * fvec2{r1, r2} - fvec2{1.0f, 1.0f};
        if (offset.get<0>() != 0.0f && offset.get<1>() != 0.0f) {
            float theta, r;
            if (fabsf(offset.get<0>()) > fabsf(offset.get<1>())) {
                r = offset.get<0>();
                theta = 0.25f * PI * (offset.get<1>() / offset.get<0>());
            } else {
                r = offset.get<1>();
                theta = 0.5f * PI - 0.25f * PI * (offset.get<0>() / offset.get<1>());
            }

            offset.set(0, 0.5f * r * cosf(theta));
            offset.set(1, 0.5f * r * sinf(theta));
        }

        const fvec4 lp = light_pos + light_u * offset.get<0>() + light_v * offset.get<1>();
        const fvec4 light_forward = normalize(cross(light_u, light_v));

        ls.lp = offset_ray(lp, light_forward);
        float ls_dist;
        ls.L = normalize_len(lp - P, ls_dist);
        ls.area = l.disk.area;
        ls.ray_flags = l.ray_visibility;

        const float cos_theta = dot(-ls.L, light_forward);
        if (cos_theta > 0.0f) {
            ls.pdf = (ls_dist * ls_dist) / (ls.area * cos_theta);
        }

        if (!l.visible) {
            ls.area = 0.0f;
        }

        if (l.sky_portal != 0) {
            fvec4 env_col = make_fvec3(sc.env.env_col);
            if (sc.env.env_map != 0xffffffff) {
                env_col *= SampleLatlong_RGBE(*static_cast<const Cpu::TexStorageRGBA *>(textures[0]), sc.env.env_map,
                                              ls.L, sc.env.env_map_rotation, rand_tex_uv);
            }
            ls.col *= env_col;
            ls.from_env = 1;
        }
    } else if (l.type == LIGHT_TYPE_LINE) {
        const fvec4 light_pos = make_fvec3(l.line.pos);
        const fvec4 light_dir = make_fvec3(l.line.v);

        const float r1 = rand_light_uv.get<0>(), r2 = rand_light_uv.get<1>();

        const fvec4 center_to_surface = P - light_pos;

        fvec4 light_u = normalize(cross(center_to_surface, light_dir));
        fvec4 light_v = cross(light_u, light_dir);

        const float phi = PI * r1;
        const fvec4 normal = cosf(phi) * light_u + sinf(phi) * light_v;

        const fvec4 lp = light_pos + normal * l.line.radius + (r2 - 0.5f) * light_dir * l.line.height;

        ls.lp = lp;
        float ls_dist;
        ls.L = normalize_len(lp - P, ls_dist);
        ls.area = l.line.area;
        ls.ray_flags = l.ray_visibility;

        const float cos_theta = 1.0f - fabsf(dot(ls.L, light_dir));
        if (cos_theta != 0.0f) {
            ls.pdf = (ls_dist * ls_dist) / (ls.area * cos_theta);
        }

        if (!l.visible) {
            ls.area = 0.0f;
        }
    } else if (l.type == LIGHT_TYPE_TRI) {
        const mesh_instance_t &lmi = sc.mesh_instances[l.tri.mi_index];
        const uint32_t ltri_index = l.tri.tri_index;

        const vertex_t &v1 = sc.vertices[sc.vtx_indices[ltri_index * 3 + 0]],
                       &v2 = sc.vertices[sc.vtx_indices[ltri_index * 3 + 1]],
                       &v3 = sc.vertices[sc.vtx_indices[ltri_index * 3 + 2]];

        const fvec4 p1 = TransformPoint(fvec4(v1.p[0], v1.p[1], v1.p[2], 0.0f), lmi.xform),
                    p2 = TransformPoint(fvec4(v2.p[0], v2.p[1], v2.p[2], 0.0f), lmi.xform),
                    p3 = TransformPoint(fvec4(v3.p[0], v3.p[1], v3.p[2], 0.0f), lmi.xform);
        const fvec2 uv1 = fvec2(v1.t), uv2 = fvec2(v2.t), uv3 = fvec2(v3.t);

        const fvec4 e1 = p2 - p1, e2 = p3 - p1;
        float light_fwd_len;
        const fvec4 light_forward = normalize_len(cross(e1, e2), light_fwd_len);
        ls.area = 0.5f * light_fwd_len;
        ls.ray_flags = l.ray_visibility;

        fvec4 lp;
        fvec2 luvs;
        float pdf = 0.0f;

        if (USE_SPHERICAL_AREA_LIGHT_SAMPLING) {
            // Spherical triangle sampling
            pdf = SampleSphericalTriangle(P, p1, p2, p3, rand_light_uv, &ls.L);
        }
        if (pdf > 0.0f) {
            // find u, v of intersection point
            const fvec4 pvec = cross(ls.L, e2);
            const fvec4 tvec = P - p1, qvec = cross(tvec, e1);

            const float inv_det = 1.0f / dot(e1, pvec);
            const float tri_u = dot(tvec, pvec) * inv_det, tri_v = dot(ls.L, qvec) * inv_det;

            lp = (1.0f - tri_u - tri_v) * p1 + tri_u * p2 + tri_v * p3;
            luvs = (1.0f - tri_u - tri_v) * uv1 + tri_u * uv2 + tri_v * uv3;
        } else {
            // Simple area sampling
            const float r1 = sqrtf(rand_light_uv.get<0>()), r2 = rand_light_uv.get<1>();
            luvs = uv1 * (1.0f - r1) + r1 * (uv2 * (1.0f - r2) + uv3 * r2);
            lp = p1 * (1.0f - r1) + r1 * (p2 * (1.0f - r2) + p3 * r2);

            float ls_dist;
            ls.L = normalize_len(lp - P, ls_dist);

            const float cos_theta = -dot(ls.L, light_forward);
            pdf = safe_div_pos(ls_dist * ls_dist, ls.area * cos_theta);
        }

        float cos_theta = -dot(ls.L, light_forward);
        ls.lp = offset_ray(lp, cos_theta >= 0.0f ? light_forward : -light_forward);
        if (l.doublesided) {
            cos_theta = fabsf(cos_theta);
        }

        if (cos_theta > 0.0f) {
            ls.pdf = pdf;
            if (l.tri.tex_index != 0xffffffff) {
                fvec4 tex_color = SampleBilinear(textures, l.tri.tex_index, luvs, 0 /* lod */, rand_tex_uv);
                if (l.tri.tex_index & TEX_YCOCG_BIT) {
                    tex_color = YCoCg_to_RGB(tex_color);
                }
                if (l.tri.tex_index & TEX_SRGB_BIT) {
                    tex_color = srgb_to_linear(tex_color);
                }
                ls.col *= tex_color;
            }
        }
    } else if (l.type == LIGHT_TYPE_ENV) {
        const float rx = rand_light_uv.get<0>(), ry = rand_light_uv.get<1>();

        fvec4 dir_and_pdf;
        if (sc.env.qtree_levels) {
            // Sample environment using quadtree
            const auto *qtree_mips = reinterpret_cast<const fvec4 *const *>(sc.env.qtree_mips);
            dir_and_pdf = Sample_EnvQTree(sc.env.env_map_rotation, qtree_mips, sc.env.qtree_levels, u1, rx, ry);
        } else {
            // Sample environment as hemishpere
            const float phi = 2 * PI * ry;
            const float cos_phi = cosf(phi), sin_phi = sinf(phi);

            const float dir = sqrtf(1.0f - rx * rx);
            auto V = fvec4{dir * cos_phi, dir * sin_phi, rx, 0.0f}; // in tangent-space

            dir_and_pdf = world_from_tangent(T, B, N, V);
            dir_and_pdf.set<3>(0.5f / PI);
        }

        ls.L = fvec4{dir_and_pdf.get<0>(), dir_and_pdf.get<1>(), dir_and_pdf.get<2>(), 0.0f};
        ls.col *= {sc.env.env_col[0], sc.env.env_col[1], sc.env.env_col[2], 0.0f};

        if (sc.env.env_map != 0xffffffff) {
            ls.col *= SampleLatlong_RGBE(*static_cast<const Cpu::TexStorageRGBA *>(textures[0]), sc.env.env_map, ls.L,
                                         sc.env.env_map_rotation, rand_tex_uv);
        }

        ls.area = 1.0f;
        ls.lp = P + ls.L;
        ls.dist_mul = MAX_DIST;
        ls.pdf = dir_and_pdf.get<3>();
        ls.from_env = 1;
        ls.ray_flags = l.ray_visibility;
    }

    ls.pdf /= factor;
}

void Ray::Ref::IntersectAreaLights(Span<const ray_data_t> rays, Span<const light_t> lights,
                                   Span<const light_cwbvh_node_t> nodes, Span<hit_data_t> inout_inters) {
    for (int _i = 0; _i < rays.size(); ++_i) {
        const ray_data_t &ray = rays[_i];
        hit_data_t &inout_inter = inout_inters[_i];

        const fvec4 ro = make_fvec3(ray.o);
        const fvec4 rd = make_fvec3(ray.d);

        const uint32_t ray_flags = (1u << get_ray_type(ray.depth));

        float inv_d[3];
        safe_invert(value_ptr(rd), inv_d);

        ////

        TraversalStack<MAX_STACK_SIZE, light_stack_entry_t> st;
        st.push(0u /* root_index */, 0.0f /* distance */, 1.0f /* factor */);

        while (!st.empty()) {
            light_stack_entry_t cur = st.pop();

            if (cur.dist > inout_inter.t || cur.factor == 0.0f) {
                continue;
            }

        TRAVERSE:
            if ((cur.index & LEAF_NODE_BIT) == 0) {
                alignas(16) float dist[8];
                long mask = bbox_test_oct(value_ptr(ro), inv_d, inout_inter.t, nodes[cur.index], dist);
                if (mask) {
                    fvec4 importance[2];
                    calc_lnode_importance(nodes[cur.index], ro, value_ptr(importance[0]));

                    const float total_importance = hsum(importance[0] + importance[1]);
                    if (total_importance == 0.0f) {
                        continue;
                    }

                    importance[0] /= total_importance;
                    importance[1] /= total_importance;

                    alignas(16) float factors[8];
                    importance[0].store_to(&factors[0], vector_aligned);
                    importance[1].store_to(&factors[4], vector_aligned);

                    long i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    if (mask == 0) { // only one box was hit
                        cur.index = nodes[cur.index].child[i];
                        cur.factor *= factors[i];
                        goto TRAVERSE;
                    }

                    const long i2 = GetFirstBit(mask);
                    mask = ClearBit(mask, i2);
                    if (mask == 0) { // two boxes were hit
                        if (dist[i] < dist[i2]) {
                            st.push(nodes[cur.index].child[i2], dist[i2], cur.factor * factors[i2]);
                            cur.index = nodes[cur.index].child[i];
                            cur.factor *= factors[i];
                        } else {
                            st.push(nodes[cur.index].child[i], dist[i], cur.factor * factors[i]);
                            cur.index = nodes[cur.index].child[i2];
                            cur.factor *= factors[i2];
                        }
                        goto TRAVERSE;
                    }

                    st.push(nodes[cur.index].child[i], dist[i], cur.factor * factors[i]);
                    st.push(nodes[cur.index].child[i2], dist[i2], cur.factor * factors[i2]);

                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], dist[i], cur.factor * factors[i]);
                    if (mask == 0) { // three boxes were hit
                        st.sort_top3();
                        cur = st.pop();
                        goto TRAVERSE;
                    }

                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], dist[i], cur.factor * factors[i]);
                    if (mask == 0) { // four boxes were hit
                        st.sort_top4();
                        cur = st.pop();
                        goto TRAVERSE;
                    }

                    uint32_t size_before = st.stack_size;

                    // from five to eight boxes were hit
                    do {
                        i = GetFirstBit(mask);
                        mask = ClearBit(mask, i);
                        st.push(nodes[cur.index].child[i], dist[i], cur.factor * factors[i]);
                    } while (mask != 0);

                    const int count = int(st.stack_size - size_before + 4);
                    st.sort_topN(count);
                    cur = st.pop();
                    goto TRAVERSE;
                }
            } else {
                const int light_index = int(cur.index & PRIM_INDEX_BITS);
                const light_t &l = lights[light_index];
                if (!l.visible || (l.ray_visibility & ray_flags) == 0) {
                    continue;
                }
                if (l.sky_portal && inout_inter.v >= 0.0f) {
                    // Portal lights affect only missed rays
                    continue;
                }

                const bool no_shadow = (l.cast_shadow == 0);
                if (l.type == LIGHT_TYPE_SPHERE) {
                    const fvec4 light_pos = make_fvec3(l.sph.pos);
                    const fvec4 op = light_pos - ro;
                    const float b = dot(op, rd);
                    float det = b * b - dot(op, op) + l.sph.radius * l.sph.radius;
                    if (det >= 0.0f) {
                        det = sqrtf(det);
                        const float t1 = b - det, t2 = b + det;
                        if (t1 > HIT_EPS && (t1 < inout_inter.t || no_shadow)) {
                            bool accept = true;
                            if (l.sph.spot > 0.0f) {
                                const float _dot = -dot(rd, fvec4{l.sph.dir});
                                if (_dot > 0.0f) {
                                    const float _angle = acosf(saturate(_dot));
                                    accept &= (_angle <= l.sph.spot);
                                } else {
                                    accept = false;
                                }
                            }
                            if (accept) {
                                inout_inter.v = 0.0f;
                                inout_inter.obj_index = -int(light_index) - 1;
                                inout_inter.t = t1;
                                inout_inter.u = cur.factor;
                            }
                        } else if (t2 > HIT_EPS && (t2 < inout_inter.t || no_shadow)) {
                            inout_inter.v = 0.0f;
                            inout_inter.obj_index = -int(light_index) - 1;
                            inout_inter.t = t2;
                            inout_inter.u = cur.factor;
                        }
                    }
                } else if (l.type == LIGHT_TYPE_DIR) {
                    const fvec4 light_dir = make_fvec3(l.dir.dir);
                    const float cos_theta = dot(rd, light_dir);
                    if ((inout_inter.v < 0.0f || no_shadow) && cos_theta > cosf(l.dir.angle)) {
                        inout_inter.v = 0.0f;
                        inout_inter.obj_index = -int(light_index) - 1;
                        inout_inter.t = 1.0f / cos_theta;
                        inout_inter.u = cur.factor;
                    }
                } else if (l.type == LIGHT_TYPE_RECT) {
                    const fvec4 light_pos = make_fvec3(l.rect.pos);
                    fvec4 light_u = make_fvec3(l.rect.u), light_v = make_fvec3(l.rect.v);

                    const fvec4 light_forward = normalize(cross(light_u, light_v));

                    const float plane_dist = dot(light_forward, light_pos);
                    const float cos_theta = dot(rd, light_forward);
                    const float t = (plane_dist - dot(light_forward, ro)) / fminf(cos_theta, -FLT_EPS);

                    if (cos_theta < 0.0f && t > HIT_EPS && (t < inout_inter.t || no_shadow)) {
                        light_u /= dot(light_u, light_u);
                        light_v /= dot(light_v, light_v);

                        const auto p = ro + rd * t;
                        const fvec4 vi = p - light_pos;
                        const float a1 = dot(light_u, vi);
                        if (a1 >= -0.5f && a1 <= 0.5f) {
                            const float a2 = dot(light_v, vi);
                            if (a2 >= -0.5f && a2 <= 0.5f) {
                                inout_inter.v = 0.0f;
                                inout_inter.obj_index = -int(light_index) - 1;
                                inout_inter.t = t;
                                inout_inter.u = cur.factor;
                            }
                        }
                    }
                } else if (l.type == LIGHT_TYPE_DISK) {
                    const fvec4 light_pos = make_fvec3(l.disk.pos);
                    fvec4 light_u = make_fvec3(l.disk.u), light_v = make_fvec3(l.disk.v);

                    const fvec4 light_forward = normalize(cross(light_u, light_v));

                    const float plane_dist = dot(light_forward, light_pos);
                    const float cos_theta = dot(rd, light_forward);
                    const float t = safe_div_neg(plane_dist - dot(light_forward, ro), cos_theta);

                    if (cos_theta < 0.0f && t > HIT_EPS && (t < inout_inter.t || no_shadow)) {
                        light_u /= dot(light_u, light_u);
                        light_v /= dot(light_v, light_v);

                        const auto p = ro + rd * t;
                        const fvec4 vi = p - light_pos;
                        const float a1 = dot(light_u, vi);
                        const float a2 = dot(light_v, vi);

                        if (sqrtf(a1 * a1 + a2 * a2) <= 0.5f) {
                            inout_inter.v = 0.0f;
                            inout_inter.obj_index = -int(light_index) - 1;
                            inout_inter.t = t;
                            inout_inter.u = cur.factor;
                        }
                    }
                } else if (l.type == LIGHT_TYPE_LINE) {
                    const fvec4 light_pos = make_fvec3(l.line.pos);
                    const fvec4 light_u = make_fvec3(l.line.u), light_dir = make_fvec3(l.line.v);
                    const fvec4 light_v = cross(light_u, light_dir);

                    fvec4 _ro = ro - light_pos;
                    _ro = fvec4{dot(_ro, light_dir), dot(_ro, light_u), dot(_ro, light_v), 0.0f};

                    fvec4 _rd = rd;
                    _rd = fvec4{dot(_rd, light_dir), dot(_rd, light_u), dot(_rd, light_v), 0.0f};

                    const float A = _rd.get<2>() * _rd.get<2>() + _rd.get<1>() * _rd.get<1>();
                    const float B = 2.0f * (_rd.get<2>() * _ro.get<2>() + _rd.get<1>() * _ro.get<1>());
                    const float C = sqr(_ro.get<2>()) + sqr(_ro.get<1>()) - sqr(l.line.radius);

                    float t0, t1;
                    if (quadratic(A, B, C, t0, t1) && t0 > HIT_EPS && t1 > HIT_EPS) {
                        const float t = fminf(t0, t1);
                        const fvec4 p = _ro + t * _rd;
                        if (fabsf(p.get<0>()) < 0.5f * l.line.height && (t < inout_inter.t || no_shadow)) {
                            inout_inter.v = 0.0f;
                            inout_inter.obj_index = -int(light_index) - 1;
                            inout_inter.t = t;
                            inout_inter.u = cur.factor;
                        }
                    }
                } else if (l.type == LIGHT_TYPE_ENV && inout_inter.v < 0.0f) {
                    // NOTE: mask remains empty
                    inout_inter.obj_index = -int(light_index) - 1;
                    inout_inter.u = cur.factor;
                }
            }
        }
    }
}

void Ray::Ref::IntersectAreaLights(Span<const ray_data_t> rays, Span<const light_t> lights,
                                   Span<const light_wbvh_node_t> nodes, Span<hit_data_t> inout_inters) {
    for (int _i = 0; _i < rays.size(); ++_i) {
        const ray_data_t &ray = rays[_i];
        hit_data_t &inout_inter = inout_inters[_i];

        const fvec4 ro = make_fvec3(ray.o);
        const fvec4 rd = make_fvec3(ray.d);

        const uint32_t ray_flags = (1u << get_ray_type(ray.depth));

        float inv_d[3];
        safe_invert(value_ptr(rd), inv_d);

        ////

        TraversalStack<MAX_STACK_SIZE, light_stack_entry_t> st;
        st.push(0u /* root_index */, 0.0f /* distance */, 1.0f /* factor */);

        while (!st.empty()) {
            light_stack_entry_t cur = st.pop();

            if (cur.dist > inout_inter.t || cur.factor == 0.0f) {
                continue;
            }

        TRAVERSE:
            if ((cur.index & LEAF_NODE_BIT) == 0) {
                alignas(16) float dist[8];
                long mask = bbox_test_oct(value_ptr(ro), inv_d, inout_inter.t, nodes[cur.index], dist);
                if (mask) {
                    fvec4 importance[2];
                    calc_lnode_importance(nodes[cur.index], ro, value_ptr(importance[0]));

                    const float total_importance = hsum(importance[0] + importance[1]);
                    if (total_importance == 0.0f) {
                        continue;
                    }

                    importance[0] /= total_importance;
                    importance[1] /= total_importance;

                    alignas(16) float factors[8];
                    importance[0].store_to(&factors[0], vector_aligned);
                    importance[1].store_to(&factors[4], vector_aligned);

                    long i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    if (mask == 0) { // only one box was hit
                        cur.index = nodes[cur.index].child[i];
                        cur.factor *= factors[i];
                        goto TRAVERSE;
                    }

                    const long i2 = GetFirstBit(mask);
                    mask = ClearBit(mask, i2);
                    if (mask == 0) { // two boxes were hit
                        if (dist[i] < dist[i2]) {
                            st.push(nodes[cur.index].child[i2], dist[i2], cur.factor * factors[i2]);
                            cur.index = nodes[cur.index].child[i];
                            cur.factor *= factors[i];
                        } else {
                            st.push(nodes[cur.index].child[i], dist[i], cur.factor * factors[i]);
                            cur.index = nodes[cur.index].child[i2];
                            cur.factor *= factors[i2];
                        }
                        goto TRAVERSE;
                    }

                    st.push(nodes[cur.index].child[i], dist[i], cur.factor * factors[i]);
                    st.push(nodes[cur.index].child[i2], dist[i2], cur.factor * factors[i2]);

                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], dist[i], cur.factor * factors[i]);
                    if (mask == 0) { // three boxes were hit
                        st.sort_top3();
                        cur = st.pop();
                        goto TRAVERSE;
                    }

                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], dist[i], cur.factor * factors[i]);
                    if (mask == 0) { // four boxes were hit
                        st.sort_top4();
                        cur = st.pop();
                        goto TRAVERSE;
                    }

                    uint32_t size_before = st.stack_size;

                    // from five to eight boxes were hit
                    do {
                        i = GetFirstBit(mask);
                        mask = ClearBit(mask, i);
                        st.push(nodes[cur.index].child[i], dist[i], cur.factor * factors[i]);
                    } while (mask != 0);

                    const int count = int(st.stack_size - size_before + 4);
                    st.sort_topN(count);
                    cur = st.pop();
                    goto TRAVERSE;
                }
            } else {
                const int light_index = int(cur.index & PRIM_INDEX_BITS);
                const light_t &l = lights[light_index];
                if (!l.visible || (l.ray_visibility & ray_flags) == 0) {
                    continue;
                }
                if (l.sky_portal && inout_inter.v >= 0.0f) {
                    // Portal lights affect only missed rays
                    continue;
                }

                const bool no_shadow = (l.cast_shadow == 0);
                if (l.type == LIGHT_TYPE_SPHERE) {
                    const fvec4 light_pos = make_fvec3(l.sph.pos);
                    const fvec4 op = light_pos - ro;
                    const float b = dot(op, rd);
                    float det = b * b - dot(op, op) + l.sph.radius * l.sph.radius;
                    if (det >= 0.0f) {
                        det = sqrtf(det);
                        const float t1 = b - det, t2 = b + det;
                        if (t1 > HIT_EPS && (t1 < inout_inter.t || no_shadow)) {
                            bool accept = true;
                            if (l.sph.spot > 0.0f) {
                                const float _dot = -dot(rd, fvec4{l.sph.dir});
                                if (_dot > 0.0f) {
                                    const float _angle = acosf(saturate(_dot));
                                    accept &= (_angle <= l.sph.spot);
                                } else {
                                    accept = false;
                                }
                            }
                            if (accept) {
                                inout_inter.v = 0.0f;
                                inout_inter.obj_index = -int(light_index) - 1;
                                inout_inter.t = t1;
                                inout_inter.u = cur.factor;
                            }
                        } else if (t2 > HIT_EPS && (t2 < inout_inter.t || no_shadow)) {
                            inout_inter.v = 0.0f;
                            inout_inter.obj_index = -int(light_index) - 1;
                            inout_inter.t = t2;
                            inout_inter.u = cur.factor;
                        }
                    }
                } else if (l.type == LIGHT_TYPE_DIR) {
                    const fvec4 light_dir = make_fvec3(l.dir.dir);
                    const float cos_theta = dot(rd, light_dir);
                    if ((inout_inter.v < 0.0f || no_shadow) && cos_theta > cosf(l.dir.angle)) {
                        inout_inter.v = 0.0f;
                        inout_inter.obj_index = -int(light_index) - 1;
                        inout_inter.t = 1.0f / cos_theta;
                        inout_inter.u = cur.factor;
                    }
                } else if (l.type == LIGHT_TYPE_RECT) {
                    const fvec4 light_pos = make_fvec3(l.rect.pos);
                    fvec4 light_u = make_fvec3(l.rect.u), light_v = make_fvec3(l.rect.v);

                    const fvec4 light_forward = normalize(cross(light_u, light_v));

                    const float plane_dist = dot(light_forward, light_pos);
                    const float cos_theta = dot(rd, light_forward);
                    const float t = (plane_dist - dot(light_forward, ro)) / fminf(cos_theta, -FLT_EPS);

                    if (cos_theta < 0.0f && t > HIT_EPS && (t < inout_inter.t || no_shadow)) {
                        light_u /= dot(light_u, light_u);
                        light_v /= dot(light_v, light_v);

                        const auto p = ro + rd * t;
                        const fvec4 vi = p - light_pos;
                        const float a1 = dot(light_u, vi);
                        if (a1 >= -0.5f && a1 <= 0.5f) {
                            const float a2 = dot(light_v, vi);
                            if (a2 >= -0.5f && a2 <= 0.5f) {
                                inout_inter.v = 0.0f;
                                inout_inter.obj_index = -int(light_index) - 1;
                                inout_inter.t = t;
                                inout_inter.u = cur.factor;
                            }
                        }
                    }
                } else if (l.type == LIGHT_TYPE_DISK) {
                    const fvec4 light_pos = make_fvec3(l.disk.pos);
                    fvec4 light_u = make_fvec3(l.disk.u), light_v = make_fvec3(l.disk.v);

                    const fvec4 light_forward = normalize(cross(light_u, light_v));

                    const float plane_dist = dot(light_forward, light_pos);
                    const float cos_theta = dot(rd, light_forward);
                    const float t = safe_div_neg(plane_dist - dot(light_forward, ro), cos_theta);

                    if (cos_theta < 0.0f && t > HIT_EPS && (t < inout_inter.t || no_shadow)) {
                        light_u /= dot(light_u, light_u);
                        light_v /= dot(light_v, light_v);

                        const auto p = ro + rd * t;
                        const fvec4 vi = p - light_pos;
                        const float a1 = dot(light_u, vi);
                        const float a2 = dot(light_v, vi);

                        if (sqrtf(a1 * a1 + a2 * a2) <= 0.5f) {
                            inout_inter.v = 0.0f;
                            inout_inter.obj_index = -int(light_index) - 1;
                            inout_inter.t = t;
                            inout_inter.u = cur.factor;
                        }
                    }
                } else if (l.type == LIGHT_TYPE_LINE) {
                    const fvec4 light_pos = make_fvec3(l.line.pos);
                    const fvec4 light_u = make_fvec3(l.line.u), light_dir = make_fvec3(l.line.v);
                    const fvec4 light_v = cross(light_u, light_dir);

                    fvec4 _ro = ro - light_pos;
                    _ro = fvec4{dot(_ro, light_dir), dot(_ro, light_u), dot(_ro, light_v), 0.0f};

                    fvec4 _rd = rd;
                    _rd = fvec4{dot(_rd, light_dir), dot(_rd, light_u), dot(_rd, light_v), 0.0f};

                    const float A = _rd.get<2>() * _rd.get<2>() + _rd.get<1>() * _rd.get<1>();
                    const float B = 2.0f * (_rd.get<2>() * _ro.get<2>() + _rd.get<1>() * _ro.get<1>());
                    const float C = sqr(_ro.get<2>()) + sqr(_ro.get<1>()) - sqr(l.line.radius);

                    float t0, t1;
                    if (quadratic(A, B, C, t0, t1) && t0 > HIT_EPS && t1 > HIT_EPS) {
                        const float t = fminf(t0, t1);
                        const fvec4 p = _ro + t * _rd;
                        if (fabsf(p.get<0>()) < 0.5f * l.line.height && (t < inout_inter.t || no_shadow)) {
                            inout_inter.v = 0.0f;
                            inout_inter.obj_index = -int(light_index) - 1;
                            inout_inter.t = t;
                            inout_inter.u = cur.factor;
                        }
                    }
                } else if (l.type == LIGHT_TYPE_ENV && inout_inter.v < 0.0f) {
                    // NOTE: mask remains empty
                    inout_inter.obj_index = -int(light_index) - 1;
                    inout_inter.u = cur.factor;
                }
            }
        }
    }
}

void Ray::Ref::IntersectAreaLights(Span<const ray_data_t> rays, Span<const light_t> lights,
                                   Span<const light_bvh_node_t> nodes, Span<hit_data_t> inout_inters) {
    for (int i = 0; i < rays.size(); ++i) {
        const ray_data_t &ray = rays[i];
        hit_data_t &inout_inter = inout_inters[i];

        const fvec4 ro = make_fvec3(ray.o);
        const fvec4 rd = make_fvec3(ray.d);

        const uint32_t ray_flags = (1u << get_ray_type(ray.depth));

        float inv_d[3];
        safe_invert(value_ptr(rd), inv_d);

        ////

        uint32_t stack[MAX_STACK_SIZE];
        float stack_factors[MAX_STACK_SIZE];
        uint32_t stack_size = 0;

        stack_factors[stack_size] = 1.0f;
        stack[stack_size++] = 0;

        while (stack_size) {
            uint32_t cur = stack[--stack_size];
            float cur_factor = stack_factors[stack_size];

            // if (cur.dist > inout_inter.t) {
            //     continue;
            // }

            if (!bbox_test(value_ptr(ro), inv_d, inout_inter.t, nodes[cur])) {
                continue;
            }

            if (!is_leaf_node(nodes[cur])) {
                const uint32_t far = far_child(value_ptr(rd), nodes[cur]), near = near_child(value_ptr(rd), nodes[cur]);
                const light_bvh_node_t &f = nodes[far], &n = nodes[near];

                const float far_importance = calc_lnode_importance(f, ro),
                            near_importance = calc_lnode_importance(n, ro);
                const float total_importance = far_importance + near_importance;
                if (total_importance == 0.0f) {
                    continue;
                }

                if (far_importance > 0.0f) {
                    stack_factors[stack_size] = cur_factor * far_importance / total_importance;
                    stack[stack_size++] = far;
                }
                if (near_importance > 0.0f) {
                    stack_factors[stack_size] = cur_factor * near_importance / total_importance;
                    stack[stack_size++] = near;
                }
            } else {
                const int light_index = int(nodes[cur].prim_index & PRIM_INDEX_BITS);
                assert((nodes[cur].prim_count & PRIM_COUNT_BITS) == 1);

                ////

                const light_t &l = lights[light_index];
                if (!l.visible || (l.ray_visibility & ray_flags) == 0) {
                    continue;
                }
                if (l.sky_portal && inout_inter.v >= 0.0f) {
                    // Portal lights affect only missed rays
                    continue;
                }

                const bool no_shadow = (l.cast_shadow == 0);
                if (l.type == LIGHT_TYPE_SPHERE) {
                    const fvec4 light_pos = make_fvec3(l.sph.pos);
                    const fvec4 op = light_pos - ro;
                    const float b = dot(op, rd);
                    float det = b * b - dot(op, op) + l.sph.radius * l.sph.radius;
                    if (det >= 0.0f) {
                        det = sqrtf(det);
                        const float t1 = b - det, t2 = b + det;
                        if (t1 > HIT_EPS && (t1 < inout_inter.t || no_shadow)) {
                            bool accept = true;
                            if (l.sph.spot > 0.0f) {
                                const float _dot = -dot(rd, fvec4{l.sph.dir});
                                if (_dot > 0.0f) {
                                    const float _angle = acosf(saturate(_dot));
                                    accept &= (_angle <= l.sph.spot);
                                } else {
                                    accept = false;
                                }
                            }
                            if (accept) {
                                inout_inter.v = 0.0f;
                                inout_inter.obj_index = -int(light_index) - 1;
                                inout_inter.t = t1;
                                inout_inter.u = cur_factor;
                            }
                        } else if (t2 > HIT_EPS && (t2 < inout_inter.t || no_shadow)) {
                            inout_inter.v = 0.0f;
                            inout_inter.obj_index = -int(light_index) - 1;
                            inout_inter.t = t2;
                            inout_inter.u = cur_factor;
                        }
                    }
                } else if (l.type == LIGHT_TYPE_DIR) {
                    const fvec4 light_dir = make_fvec3(l.dir.dir);
                    const float cos_theta = dot(rd, light_dir);
                    if ((inout_inter.v < 0.0f || no_shadow) && cos_theta > cosf(l.dir.angle)) {
                        inout_inter.v = 0.0f;
                        inout_inter.obj_index = -int(light_index) - 1;
                        inout_inter.t = 1.0f / cos_theta;
                        inout_inter.u = cur_factor;
                    }
                } else if (l.type == LIGHT_TYPE_RECT) {
                    const fvec4 light_pos = make_fvec3(l.rect.pos);
                    fvec4 light_u = make_fvec3(l.rect.u), light_v = make_fvec3(l.rect.v);

                    const fvec4 light_forward = normalize(cross(light_u, light_v));

                    const float plane_dist = dot(light_forward, light_pos);
                    const float cos_theta = dot(rd, light_forward);
                    const float t = (plane_dist - dot(light_forward, ro)) / fminf(cos_theta, -FLT_EPS);

                    if (cos_theta < 0.0f && t > HIT_EPS && (t < inout_inter.t || no_shadow)) {
                        light_u /= dot(light_u, light_u);
                        light_v /= dot(light_v, light_v);

                        const auto p = ro + rd * t;
                        const fvec4 vi = p - light_pos;
                        const float a1 = dot(light_u, vi);
                        if (a1 >= -0.5f && a1 <= 0.5f) {
                            const float a2 = dot(light_v, vi);
                            if (a2 >= -0.5f && a2 <= 0.5f) {
                                inout_inter.v = 0.0f;
                                inout_inter.obj_index = -int(light_index) - 1;
                                inout_inter.t = t;
                                inout_inter.u = cur_factor;
                            }
                        }
                    }
                } else if (l.type == LIGHT_TYPE_DISK) {
                    const fvec4 light_pos = make_fvec3(l.disk.pos);
                    fvec4 light_u = make_fvec3(l.disk.u), light_v = make_fvec3(l.disk.v);

                    const fvec4 light_forward = normalize(cross(light_u, light_v));

                    const float plane_dist = dot(light_forward, light_pos);
                    const float cos_theta = dot(rd, light_forward);
                    const float t = safe_div_neg(plane_dist - dot(light_forward, ro), cos_theta);

                    if (cos_theta < 0.0f && t > HIT_EPS && (t < inout_inter.t || no_shadow)) {
                        light_u /= dot(light_u, light_u);
                        light_v /= dot(light_v, light_v);

                        const auto p = ro + rd * t;
                        const fvec4 vi = p - light_pos;
                        const float a1 = dot(light_u, vi);
                        const float a2 = dot(light_v, vi);

                        if (sqrtf(a1 * a1 + a2 * a2) <= 0.5f) {
                            inout_inter.v = 0.0f;
                            inout_inter.obj_index = -int(light_index) - 1;
                            inout_inter.t = t;
                            inout_inter.u = cur_factor;
                        }
                    }
                } else if (l.type == LIGHT_TYPE_LINE) {
                    const fvec4 light_pos = make_fvec3(l.line.pos);
                    const fvec4 light_u = make_fvec3(l.line.u), light_dir = make_fvec3(l.line.v);
                    const fvec4 light_v = cross(light_u, light_dir);

                    fvec4 _ro = ro - light_pos;
                    _ro = fvec4{dot(_ro, light_dir), dot(_ro, light_u), dot(_ro, light_v), 0.0f};

                    fvec4 _rd = rd;
                    _rd = fvec4{dot(_rd, light_dir), dot(_rd, light_u), dot(_rd, light_v), 0.0f};

                    const float A = _rd.get<2>() * _rd.get<2>() + _rd.get<1>() * _rd.get<1>();
                    const float B = 2.0f * (_rd.get<2>() * _ro.get<2>() + _rd.get<1>() * _ro.get<1>());
                    const float C = sqr(_ro.get<2>()) + sqr(_ro.get<1>()) - sqr(l.line.radius);

                    float t0, t1;
                    if (quadratic(A, B, C, t0, t1) && t0 > HIT_EPS && t1 > HIT_EPS) {
                        const float t = fminf(t0, t1);
                        const fvec4 p = _ro + t * _rd;
                        if (fabsf(p.get<0>()) < 0.5f * l.line.height && (t < inout_inter.t || no_shadow)) {
                            inout_inter.v = 0.0f;
                            inout_inter.obj_index = -int(light_index) - 1;
                            inout_inter.t = t;
                            inout_inter.u = cur_factor;
                        }
                    }
                } else if (l.type == LIGHT_TYPE_ENV && inout_inter.v < 0.0f) {
                    // NOTE: mask remains empty
                    inout_inter.obj_index = -int(light_index) - 1;
                    inout_inter.u = cur_factor;
                }
            }
        }
    }
}

float Ray::Ref::IntersectAreaLights(const shadow_ray_t &ray, Span<const light_t> lights,
                                    Span<const light_wbvh_node_t> nodes) {
    const float rdist = fabsf(ray.dist);

    const fvec4 ro = make_fvec3(ray.o);
    const fvec4 rd = make_fvec3(ray.d);

    float inv_d[3];
    safe_invert(value_ptr(rd), inv_d);

    ////

    TraversalStack<MAX_STACK_SIZE> st;
    st.push(0u /* root_index */, 0.0f);

    while (!st.empty()) {
        stack_entry_t cur = st.pop();

        if (cur.dist > rdist) {
            continue;
        }

    TRAVERSE:
        if ((cur.index & LEAF_NODE_BIT) == 0) {
            alignas(16) float dist[8];
            long mask = bbox_test_oct(value_ptr(ro), inv_d, rdist, nodes[cur.index], dist);
            if (mask) {
                long i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                if (mask == 0) { // only one box was hit
                    cur.index = nodes[cur.index].child[i];
                    goto TRAVERSE;
                }

                const long i2 = GetFirstBit(mask);
                mask = ClearBit(mask, i2);
                if (mask == 0) { // two boxes were hit
                    if (dist[i] < dist[i2]) {
                        st.push(nodes[cur.index].child[i2], dist[i2]);
                        cur.index = nodes[cur.index].child[i];
                    } else {
                        st.push(nodes[cur.index].child[i], dist[i]);
                        cur.index = nodes[cur.index].child[i2];
                    }
                    goto TRAVERSE;
                }

                st.push(nodes[cur.index].child[i], dist[i]);
                st.push(nodes[cur.index].child[i2], dist[i2]);

                i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                st.push(nodes[cur.index].child[i], dist[i]);
                if (mask == 0) { // three boxes were hit
                    st.sort_top3();
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }

                i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                st.push(nodes[cur.index].child[i], dist[i]);
                if (mask == 0) { // four boxes were hit
                    st.sort_top4();
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }

                uint32_t size_before = st.stack_size;

                // from five to eight boxes were hit
                do {
                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], dist[i]);
                } while (mask != 0);

                const int count = int(st.stack_size - size_before + 4);
                st.sort_topN(count);
                cur.index = st.pop_index();
                goto TRAVERSE;
            }
        } else {
            const int light_index = int(cur.index & PRIM_INDEX_BITS);
            const light_t &l = lights[light_index];
            if ((l.ray_visibility & RAY_TYPE_SHADOW_BIT) == 0) {
                continue;
            }
            if (l.sky_portal && ray.dist >= 0.0f) {
                continue;
            }
            if (l.type == LIGHT_TYPE_RECT) {
                const fvec4 light_pos = make_fvec3(l.rect.pos);
                fvec4 light_u = make_fvec3(l.rect.u), light_v = make_fvec3(l.rect.v);
                const fvec4 light_forward = normalize(cross(light_u, light_v));

                const float plane_dist = dot(light_forward, light_pos);
                const float cos_theta = dot(rd, light_forward);
                const float t = (plane_dist - dot(light_forward, ro)) / fminf(cos_theta, -FLT_EPS);

                if (cos_theta < 0.0f && t > HIT_EPS && t < rdist) {
                    light_u /= dot(light_u, light_u);
                    light_v /= dot(light_v, light_v);

                    const auto p = ro + rd * t;
                    const fvec4 vi = p - light_pos;
                    const float a1 = dot(light_u, vi);
                    if (a1 >= -0.5f && a1 <= 0.5f) {
                        const float a2 = dot(light_v, vi);
                        if (a2 >= -0.5f && a2 <= 0.5f) {
                            return 0.0f;
                        }
                    }
                }
            } else if (l.type == LIGHT_TYPE_DISK) {
                const fvec4 light_pos = make_fvec3(l.disk.pos);
                fvec4 light_u = make_fvec3(l.disk.u), light_v = make_fvec3(l.disk.v);

                const fvec4 light_forward = normalize(cross(light_u, light_v));

                const float plane_dist = dot(light_forward, light_pos);
                const float cos_theta = dot(rd, light_forward);
                const float t = safe_div_neg(plane_dist - dot(light_forward, ro), cos_theta);

                if (cos_theta < 0.0f && t > HIT_EPS && t < rdist) {
                    light_u /= dot(light_u, light_u);
                    light_v /= dot(light_v, light_v);

                    const auto p = ro + rd * t;
                    const fvec4 vi = p - light_pos;
                    const float a1 = dot(light_u, vi);
                    const float a2 = dot(light_v, vi);

                    if (sqrtf(a1 * a1 + a2 * a2) <= 0.5f) {
                        return 0.0f;
                    }
                }
            }
        }
    }
    return 1.0f;
}

float Ray::Ref::IntersectAreaLights(const shadow_ray_t &ray, Span<const light_t> lights,
                                    Span<const light_cwbvh_node_t> nodes) {
    const float rdist = fabsf(ray.dist);

    const fvec4 ro = make_fvec3(ray.o);
    const fvec4 rd = make_fvec3(ray.d);

    float inv_d[3];
    safe_invert(value_ptr(rd), inv_d);

    ////

    TraversalStack<MAX_STACK_SIZE> st;
    st.push(0u /* root_index */, 0.0f);

    while (!st.empty()) {
        stack_entry_t cur = st.pop();

        if (cur.dist > rdist) {
            continue;
        }

    TRAVERSE:
        if ((cur.index & LEAF_NODE_BIT) == 0) {
            alignas(16) float dist[8];
            long mask = bbox_test_oct(value_ptr(ro), inv_d, rdist, nodes[cur.index], dist);
            if (mask) {
                long i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                if (mask == 0) { // only one box was hit
                    cur.index = nodes[cur.index].child[i];
                    goto TRAVERSE;
                }

                const long i2 = GetFirstBit(mask);
                mask = ClearBit(mask, i2);
                if (mask == 0) { // two boxes were hit
                    if (dist[i] < dist[i2]) {
                        st.push(nodes[cur.index].child[i2], dist[i2]);
                        cur.index = nodes[cur.index].child[i];
                    } else {
                        st.push(nodes[cur.index].child[i], dist[i]);
                        cur.index = nodes[cur.index].child[i2];
                    }
                    goto TRAVERSE;
                }

                st.push(nodes[cur.index].child[i], dist[i]);
                st.push(nodes[cur.index].child[i2], dist[i2]);

                i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                st.push(nodes[cur.index].child[i], dist[i]);
                if (mask == 0) { // three boxes were hit
                    st.sort_top3();
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }

                i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                st.push(nodes[cur.index].child[i], dist[i]);
                if (mask == 0) { // four boxes were hit
                    st.sort_top4();
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }

                uint32_t size_before = st.stack_size;

                // from five to eight boxes were hit
                do {
                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], dist[i]);
                } while (mask != 0);

                const int count = int(st.stack_size - size_before + 4);
                st.sort_topN(count);
                cur.index = st.pop_index();
                goto TRAVERSE;
            }
        } else {
            const int light_index = int(cur.index & PRIM_INDEX_BITS);
            const light_t &l = lights[light_index];
            if ((l.ray_visibility & RAY_TYPE_SHADOW_BIT) == 0) {
                continue;
            }
            if (l.sky_portal && ray.dist >= 0.0f) {
                continue;
            }
            if (l.type == LIGHT_TYPE_RECT) {
                const fvec4 light_pos = make_fvec3(l.rect.pos);
                fvec4 light_u = make_fvec3(l.rect.u), light_v = make_fvec3(l.rect.v);
                const fvec4 light_forward = normalize(cross(light_u, light_v));

                const float plane_dist = dot(light_forward, light_pos);
                const float cos_theta = dot(rd, light_forward);
                const float t = (plane_dist - dot(light_forward, ro)) / fminf(cos_theta, -FLT_EPS);

                if (cos_theta < 0.0f && t > HIT_EPS && t < rdist) {
                    light_u /= dot(light_u, light_u);
                    light_v /= dot(light_v, light_v);

                    const auto p = ro + rd * t;
                    const fvec4 vi = p - light_pos;
                    const float a1 = dot(light_u, vi);
                    if (a1 >= -0.5f && a1 <= 0.5f) {
                        const float a2 = dot(light_v, vi);
                        if (a2 >= -0.5f && a2 <= 0.5f) {
                            return 0.0f;
                        }
                    }
                }
            } else if (l.type == LIGHT_TYPE_DISK) {
                const fvec4 light_pos = make_fvec3(l.disk.pos);
                fvec4 light_u = make_fvec3(l.disk.u), light_v = make_fvec3(l.disk.v);

                const fvec4 light_forward = normalize(cross(light_u, light_v));

                const float plane_dist = dot(light_forward, light_pos);
                const float cos_theta = dot(rd, light_forward);
                const float t = safe_div_neg(plane_dist - dot(light_forward, ro), cos_theta);

                if (cos_theta < 0.0f && t > HIT_EPS && t < rdist) {
                    light_u /= dot(light_u, light_u);
                    light_v /= dot(light_v, light_v);

                    const auto p = ro + rd * t;
                    const fvec4 vi = p - light_pos;
                    const float a1 = dot(light_u, vi);
                    const float a2 = dot(light_v, vi);

                    if (sqrtf(a1 * a1 + a2 * a2) <= 0.5f) {
                        return 0.0f;
                    }
                }
            }
        }
    }
    return 1.0f;
}

float Ray::Ref::EvalTriLightFactor(const fvec4 &P, const fvec4 &ro, const uint32_t tri_index,
                                   Span<const light_t> lights, Span<const light_bvh_node_t> nodes) {
    uint32_t stack[MAX_STACK_SIZE];
    float stack_factors[MAX_STACK_SIZE];
    uint32_t stack_size = 0;

    stack_factors[stack_size] = 1.0f;
    stack[stack_size++] = 0;

    while (stack_size) {
        const uint32_t cur = stack[--stack_size];
        const float cur_factor = stack_factors[stack_size];

        if (!bbox_test(value_ptr(P), nodes[cur])) {
            continue;
        }

        if (!is_leaf_node(nodes[cur])) {
            const uint32_t left_child = nodes[cur].left_child,
                           right_child = (nodes[cur].right_child & RIGHT_CHILD_BITS);
            const light_bvh_node_t &left = nodes[left_child], &right = nodes[right_child];

            const float left_importance = calc_lnode_importance(left, ro),
                        right_importance = calc_lnode_importance(right, ro);
            const float total_importance = left_importance + right_importance;
            if (total_importance == 0.0f) {
                continue;
            }

            if (left_importance > 0.0f) {
                stack_factors[stack_size] = cur_factor * left_importance / total_importance;
                stack[stack_size++] = left_child;
            }
            if (right_importance > 0.0f) {
                stack_factors[stack_size] = cur_factor * right_importance / total_importance;
                stack[stack_size++] = right_child;
            }
        } else {
            const int light_index = int(nodes[cur].prim_index & PRIM_INDEX_BITS);
            assert((nodes[cur].prim_count & PRIM_COUNT_BITS) == 1);

            const light_t &l = lights[light_index];
            if (l.type == LIGHT_TYPE_TRI && l.tri.tri_index == tri_index) {
                // needed triangle found
                return 1.0f / cur_factor;
            }
        }
    }

    return 1.0f;
}

float Ray::Ref::EvalTriLightFactor(const fvec4 &P, const fvec4 &ro, uint32_t tri_index, Span<const light_t> lights,
                                   Span<const light_wbvh_node_t> nodes) {
    uint32_t stack[MAX_STACK_SIZE];
    float stack_factors[MAX_STACK_SIZE];
    uint32_t stack_size = 0;

    stack_factors[stack_size] = 1.0f;
    stack[stack_size++] = 0;

    while (stack_size) {
        const uint32_t cur = stack[--stack_size];
        const float cur_factor = stack_factors[stack_size];

        if ((cur & LEAF_NODE_BIT) == 0) {
            long mask = bbox_test_oct(value_ptr(P), nodes[cur]);
            if (mask) {
                alignas(16) float importance[8];
                calc_lnode_importance(nodes[cur], ro, importance);

                const float total_importance =
                    hsum(fvec4{&importance[0], vector_aligned} + fvec4{&importance[4], vector_aligned});
                if (total_importance == 0.0f) {
                    continue;
                }

                do {
                    const long i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    if (importance[i] > 0.0f) {
                        stack_factors[stack_size] = cur_factor * importance[i] / total_importance;
                        stack[stack_size++] = nodes[cur].child[i];
                    }
                } while (mask != 0);
            }
        } else {
            const int light_index = int(cur & PRIM_INDEX_BITS);
            const light_t &l = lights[light_index];
            if (l.type == LIGHT_TYPE_TRI && l.tri.tri_index == tri_index) {
                // needed triangle found
                return 1.0f / cur_factor;
            }
        }
    }
    return 1.0f;
}

float Ray::Ref::EvalTriLightFactor(const fvec4 &P, const fvec4 &ro, uint32_t tri_index, Span<const light_t> lights,
                                   Span<const light_cwbvh_node_t> nodes) {
    uint32_t stack[MAX_STACK_SIZE];
    float stack_factors[MAX_STACK_SIZE];
    uint32_t stack_size = 0;

    stack_factors[stack_size] = 1.0f;
    stack[stack_size++] = 0;

    while (stack_size) {
        const uint32_t cur = stack[--stack_size];
        const float cur_factor = stack_factors[stack_size];

        if ((cur & LEAF_NODE_BIT) == 0) {
            long mask = bbox_test_oct(value_ptr(P), nodes[cur]);
            if (mask) {
                alignas(16) float importance[8];
                calc_lnode_importance(nodes[cur], ro, importance);

                const float total_importance =
                    hsum(fvec4{&importance[0], vector_aligned} + fvec4{&importance[4], vector_aligned});
                if (total_importance == 0.0f) {
                    continue;
                }

                do {
                    const long i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    if (importance[i] > 0.0f) {
                        stack_factors[stack_size] = cur_factor * importance[i] / total_importance;
                        stack[stack_size++] = nodes[cur].child[i];
                    }
                } while (mask != 0);
            }
        } else {
            const int light_index = int(cur & PRIM_INDEX_BITS);
            const light_t &l = lights[light_index];
            if (l.type == LIGHT_TYPE_TRI && l.tri.tri_index == tri_index) {
                // needed triangle found
                return 1.0f / cur_factor;
            }
        }
    }
    return 1.0f;
}

float Ray::Ref::Evaluate_EnvQTree(const float y_rotation, const fvec4 *const *qtree_mips, const int qtree_levels,
                                  const fvec4 &L) {
    int res = 2;
    int lod = qtree_levels - 1;

    fvec2 p;
    DirToCanonical(value_ptr(L), -y_rotation, value_ptr(p));
    float factor = 1.0f;

    while (lod >= 0) {
        const int x = clamp(int(p.get<0>() * float(res)), 0, res - 1);
        const int y = clamp(int(p.get<1>() * float(res)), 0, res - 1);

        int index = 0;
        index |= (x & 1) << 0;
        index |= (y & 1) << 1;

        const int qx = x / 2;
        const int qy = y / 2;

        const fvec4 quad = qtree_mips[lod][qy * res / 2 + qx];
        const float total = quad.get<0>() + quad.get<1>() + quad.get<2>() + quad.get<3>();
        if (total <= 0.0f) {
            break;
        }

        factor *= 4.0f * quad[index] / total;

        --lod;
        res *= 2;
    }

    return factor / (4.0f * PI);
}

Ray::Ref::fvec4 Ray::Ref::Sample_EnvQTree(const float y_rotation, const fvec4 *const *qtree_mips,
                                          const int qtree_levels, const float rand, const float rx, const float ry) {
    int res = 2;
    float step = 1.0f / float(res);

    float sample = rand;
    int lod = qtree_levels - 1;

    fvec2 origin = {0.0f, 0.0f};
    float factor = 1.0f;

    while (lod >= 0) {
        const int qx = int(origin.get<0>() * float(res)) / 2;
        const int qy = int(origin.get<1>() * float(res)) / 2;

        const fvec4 quad = qtree_mips[lod][qy * res / 2 + qx];

        const float top_left = quad.get<0>();
        const float top_right = quad.get<1>();
        float partial = top_left + quad.get<2>();
        const float total = partial + top_right + quad.get<3>();
        if (total <= 0.0f) {
            break;
        }

        float boundary = partial / total;

        int index = 0;
        if (sample < boundary) {
            assert(partial > 0.0f);
            sample /= boundary;
            boundary = top_left / partial;
        } else {
            partial = total - partial;
            assert(partial > 0.0f);
            origin.set<0>(origin.get<0>() + step);
            sample = (sample - boundary) / (1.0f - boundary);
            boundary = top_right / partial;
            index |= (1 << 0);
        }

        if (sample < boundary) {
            sample /= boundary;
        } else {
            origin.set<1>(origin.get<1>() + step);
            sample = (sample - boundary) / (1.0f - boundary);
            index |= (1 << 1);
        }

        factor *= 4.0f * quad[index] / total;

        --lod;
        res *= 2;
        step *= 0.5f;
    }

    origin += 2 * step * fvec2{rx, ry};

    // origin = fvec2{rx, ry};
    // factor = 1.0f;

    fvec4 dir_and_pdf;
    CanonicalToDir(value_ptr(origin), y_rotation, value_ptr(dir_and_pdf));
    dir_and_pdf.set<3>(factor / (4.0f * PI));

    return dir_and_pdf;
}

void Ray::Ref::TraceRays(Span<ray_data_t> rays, int min_transp_depth, int max_transp_depth, const scene_data_t &sc,
                         uint32_t node_index, bool trace_lights, const Cpu::TexStorageBase *const textures[],
                         const uint32_t rand_seq[], const uint32_t rand_seed, const int iteration,
                         Span<hit_data_t> out_inter) {
    IntersectScene(rays, min_transp_depth, max_transp_depth, rand_seq, rand_seed, iteration, sc, node_index, textures,
                   out_inter);
    if (trace_lights && sc.visible_lights_count) {
        if (!sc.light_cwnodes.empty()) {
            IntersectAreaLights(rays, sc.lights, sc.light_cwnodes, out_inter);
        } else {
            IntersectAreaLights(rays, sc.lights, sc.light_nodes, out_inter);
        }
    }
}

void Ray::Ref::TraceShadowRays(Span<const shadow_ray_t> rays, int max_transp_depth, float _clamp_val,
                               const scene_data_t &sc, uint32_t node_index, const uint32_t rand_seq[],
                               const uint32_t rand_seed, const int iteration,
                               const Cpu::TexStorageBase *const textures[], const int img_w, color_rgba_t *out_color) {
    const float limit = (_clamp_val != 0.0f) ? 3.0f * _clamp_val : FLT_MAX;
    for (int i = 0; i < int(rays.size()); ++i) {
        const shadow_ray_t &sh_r = rays[i];

        const int x = (sh_r.xy >> 16) & 0x0000ffff;
        const int y = sh_r.xy & 0x0000ffff;

        fvec4 rc = IntersectScene(sh_r, max_transp_depth, sc, node_index, rand_seq, rand_seed, iteration, textures);
        if (sc.blocker_lights_count) {
            rc *= IntersectAreaLights(sh_r, sc.lights, sc.light_cwnodes);
        }
        rc.set<3>(0.0f);

        const float sum = hsum(rc);
        if (sum > limit) {
            rc *= (limit / sum);
        }

        auto old_val = fvec4{out_color[y * img_w + x].v, vector_aligned};
        old_val += rc;
        old_val.store_to(out_color[y * img_w + x].v, vector_aligned);
    }
}

#pragma warning(pop)

#undef VECTORIZE_BBOX_INTERSECTION
#undef FORCE_TEXTURE_LOD
#undef USE_STOCH_TEXTURE_FILTERING
