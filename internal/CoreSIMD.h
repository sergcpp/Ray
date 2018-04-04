//#pragma once

#include <vector>

#include "Core.h"

#include "simd/simd_vec.h"

namespace ray {
namespace NS {

const int ray_packet_layout_x[] = { 0, 1, 0, 1,
                                    2, 3, 2, 3,
                                    0, 1, 0, 1,
                                    2, 3, 2, 3 };

const int ray_packet_layout_y[] = { 0, 0, 1, 1,
                                    0, 0, 1, 1,
                                    2, 2, 3, 3,
                                    2, 2, 3, 3 };

template <int S>
struct ray_packet_t {
    // directions of rays in packet
    simd_fvec<S> d[3];
    // origins of rays in packet
    simd_fvec<S> o[3];
    // left top corner coordinates of packet
    int x, y;

    // hint for math::aligned_vector
    static const size_t alignment = alignof(simd_fvec<S>);
};

template <int S>
struct hit_data_t {
    simd_ivec<S> mask;
    simd_ivec<S> obj_index;
    simd_ivec<S> prim_index;
    simd_fvec<S> t, u, v;
    // left top corner coordinates of packet
    int x, y;

    hit_data_t(eUninitialize) {}
    hit_data_t() {
        mask = { 0 };
        obj_index = { -1 };
        prim_index = { -1 };
        t = { MAX_DIST };
    }

    // hint for math::aligned_vector
    static const size_t alignment = alignof(simd_fvec16);
};

// Generating rays
template <int DimX, int DimY>
void ConstructRayPacket(const float *o, const float *d, int size, ray_packet_t<DimX * DimY> &out_r);
template <int DimX, int DimY>
void GeneratePrimaryRays(const camera_t &cam, const rect_t &r, int w, int h, math::aligned_vector<ray_packet_t<DimX * DimY>> &out_rays);

// Intersect primitives
template <int S>
bool IntersectTris(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const tri_accel_t *tris, uint32_t num_tris, uint32_t obj_index, hit_data_t<S> &out_inter);
template <int S>
bool IntersectTris(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const tri_accel_t *tris, const uint32_t *indices, uint32_t num_tris, uint32_t obj_index, hit_data_t<S> &out_inter);

// Traverse acceleration structure
// stack-less cpu-style traversal of outer nodes
template <int S>
bool Traverse_MacroTree_CPU(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const simd_fvec<S> inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                            const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                            const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t<S> &inter);
// stack-less cpu-style traversal of inner nodes
template <int S>
bool Traverse_MicroTree_CPU(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const simd_fvec<S> inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                            const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t<S> &inter);

// Transform
template <int S>
ray_packet_t<S> TransformRay(const ray_packet_t<S> &r, const float *xform);
}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <cassert>

namespace ray {
namespace NS {
template <int S>
force_inline void _IntersectTri(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const tri_accel_t &tri, uint32_t prim_index, hit_data_t<S> &inter) {
    const int _next_u[] = { 1, 0, 0 },
              _next_v[] = { 2, 2, 1 };

    int w = (tri.ci & TRI_W_BITS),
        u = _next_u[w],
        v = _next_v[w];

    // from "Ray-Triangle Intersection Algorithm for Modern CPU Architectures" [2007]

    simd_fvec<S> det = r.d[u] * tri.nu + r.d[v] * tri.nv + r.d[w];
    simd_fvec<S> dett = tri.np - (r.o[u] * tri.nu + r.o[v] * tri.nv + r.o[w]);
    simd_fvec<S> Du = r.d[u] * dett - (tri.pu - r.o[u]) * det;
    simd_fvec<S> Dv = r.d[v] * dett - (tri.pv - r.o[v]) * det;
    simd_fvec<S> detu = tri.e1v * Du - tri.e1u * Dv;
    simd_fvec<S> detv = tri.e0u * Dv - tri.e0v * Du;

    simd_fvec<S> tmpdet0 = det - detu - detv;

    //////////////////////////////////////////////////////////////////////////

    simd_fvec<S> mm = ((tmpdet0 > -HIT_EPS) & (detu > -HIT_EPS) & (detv > -HIT_EPS)) |
                      ((tmpdet0 < HIT_EPS) & (detu < HIT_EPS) & (detv < HIT_EPS));

    simd_ivec<S> imask = cast<simd_ivec<S>>(mm) & ray_mask;

    if (imask.all_zeros()) return; // no intersection found

    simd_fvec<S> rdet = 1.0f / det;
    simd_fvec<S> t = dett * rdet;

    simd_fvec<S> t_valid = (t < inter.t) & (t > 0.0f);
    imask = imask & cast<simd_ivec<S>>(t_valid);

    if (imask.all_zeros()) return; // all intersections further than needed

    simd_fvec<S> bar_u = detu * rdet;
    simd_fvec<S> bar_v = detv * rdet;

    const auto &fmask = cast<simd_fvec<S>>(imask);

    inter.mask = inter.mask | imask;

    where(imask, inter.prim_index) = simd_ivec<S>{ reinterpret_cast<const int&>(prim_index) };
    where(fmask, inter.t) = t;
    where(fmask, inter.u) = bar_u;
    where(fmask, inter.v) = bar_v;
}

template <int S>
force_inline simd_ivec<S> bbox_test(const simd_fvec<S> o[3], const simd_fvec<S> inv_d[3], const simd_fvec<S> &t, const float _bbox_min[3], const float _bbox_max[3]) {
    simd_fvec<S> low, high, tmin, tmax;
    
    low = inv_d[0] * (_bbox_min[0] - o[0]);
    high = inv_d[0] * (_bbox_max[0] - o[0]);
    tmin = min(low, high);
    tmax = max(low, high);

    low = inv_d[1] * (_bbox_min[1] - o[1]);
    high = inv_d[1] * (_bbox_max[1] - o[1]);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));

    low = inv_d[2] * (_bbox_min[2] - o[2]);
    high = inv_d[2] * (_bbox_max[2] - o[2]);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));

    simd_fvec<S> mask = (tmin <= tmax) & (tmin <= t) & (tmax > 0.0f);
    
    return cast<const simd_ivec<S>&>(mask);
}

template <int S>
force_inline simd_ivec<S> bbox_test(const simd_fvec<S> o[3], const simd_fvec<S> inv_d[3], const simd_fvec<S> &t, const bvh_node_t &node) {
    return bbox_test(o, inv_d, t, node.bbox[0], node.bbox[1]);
}

template <int S>
force_inline uint32_t near_child(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const bvh_node_t &node) {
    simd_ivec<S> mask = cast<simd_ivec<S>>(r.d[node.space_axis] < 0.0f);
    if (mask.all_zeros(ray_mask)) {
        return node.left_child;
    } else {
        assert(and_not(mask, ray_mask).all_zeros());
        return node.right_child;
    }
}

force_inline bool is_leaf_node(const bvh_node_t &node) {
    return node.prim_count != 0;
}

enum eTraversalSource { FromParent, FromChild, FromSibling };

template <int S>
struct TraversalState {
    struct {
        simd_ivec<S> mask;
        uint32_t cur;
        eTraversalSource src;
    } queue[S];

    int index = 0, num = 1;

    force_inline void select_near_child(const ray_packet_t<S> &r, const bvh_node_t &node) {
        auto mask1 = cast<simd_ivec<S>>(r.d[node.space_axis] < 0.0f);
        mask1 = mask1 & queue[index].mask;
        if (mask1.all_zeros()) {
            queue[index].cur = node.left_child;
        } else {
            simd_ivec<S> mask2 = and_not(mask1, queue[index].mask);
            if (mask2.all_zeros()) {
                queue[index].cur = node.right_child;
            } else {
                queue[num].cur = node.left_child;
                queue[num].mask = mask2;
                queue[num].src = queue[index].src;
                num++;
                queue[index].cur = node.right_child;
                queue[index].mask = mask1;
            }
        }
    }
};

template <int S>
force_inline void safe_invert(const simd_fvec<S> v[3], simd_fvec<S> inv_v[3]) {
    inv_v[0] = { 1.0f / v[0] };
    where(v[0] <= FLT_EPS & v[0] >= 0, inv_v[0]) = MAX_DIST;
    where(v[0] >= -FLT_EPS & v[0] < 0, inv_v[0]) = -MAX_DIST;

    inv_v[1] = { 1.0f / v[1] };
    where(v[1] <= FLT_EPS & v[1] >= 0, inv_v[1]) = MAX_DIST;
    where(v[1] >= -FLT_EPS & v[1] < 0, inv_v[1]) = -MAX_DIST;

    inv_v[2] = { 1.0f / v[2] };
    where(v[2] <= FLT_EPS & v[2] >= 0, inv_v[2]) = MAX_DIST;
    where(v[2] >= -FLT_EPS & v[2] < 0, inv_v[2]) = -MAX_DIST;
}

}
}

template <int DimX, int DimY>
void ray::NS::ConstructRayPacket(const float *o, const float *d, int size, ray_packet_t<DimX * DimY> &out_r) {
    /*assert(size <= S);

    if (size == RayPacketSize) {
    out_r.o[0] = { o[0],  o[3],  o[6],  o[9],  o[12], o[15], o[18], o[21],
    o[24], o[27], o[30], o[33], o[36], o[39], o[42], o[45] };
    out_r.o[1] = { o[1],  o[4],  o[7],  o[10], o[13], o[16], o[19], o[22],
    o[25], o[28], o[31], o[34], o[37], o[40], o[43], o[46] };
    out_r.o[2] = { o[2],  o[5],  o[8],  o[11], o[14], o[17], o[20], o[23],
    o[26], o[29], o[32], o[35], o[38], o[41], o[44], o[47] };
    out_r.d[0] = { d[0],  d[3],  d[6],  d[9],  d[12], d[15], d[18], d[21],
    d[24], d[27], d[30], d[33], d[36], d[39], d[42], d[45] };
    out_r.d[1] = { d[1],  d[4],  d[7],  d[10], d[13], d[16], d[19], d[22],
    d[25], d[28], d[31], d[34], d[37], d[40], d[43], d[46] };
    out_r.d[2] = { d[2],  d[5],  d[8],  d[11], d[14], d[17], d[20], d[23],
    d[26], d[29], d[32], d[35], d[38], d[41], d[44], d[47] };
    }*/
}

template <int DimX, int DimY>
void ray::NS::GeneratePrimaryRays(const camera_t &cam, const rect_t &r, int w, int h, math::aligned_vector<ray_packet_t<DimX * DimY>> &out_rays) {
    const int S = DimX * DimY;

    static_assert(S <= 16, "!");

    simd_fvec<S> ww = { (float)w }, hh = { (float)h };

    float k = float(h) / w;

    simd_fvec<S> fwd[3] = { { cam.fwd[0] },{ cam.fwd[1] },{ cam.fwd[2] } },
        side[3] = { { cam.side[0] },{ cam.side[1] },{ cam.side[2] } },
        up[3] = { { cam.up[0] * k },{ cam.up[1] * k },{ cam.up[2] * k } };

    auto get_pix_dirs = [&fwd, &side, &up, &ww, &hh](const simd_fvec<S> &x, const simd_fvec<S> &y, simd_fvec<S> d[3]) {
        auto _dx = x / ww - 0.5f;
        auto _dy = -y / hh + 0.5f;

        d[0] = _dx * side[0] + _dy * up[0] + fwd[0];
        d[1] = _dx * side[1] + _dy * up[1] + fwd[1];
        d[2] = _dx * side[2] + _dy * up[2] + fwd[2];

        simd_fvec<DimX * DimY> len = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        d[0] /= len;
        d[1] /= len;
        d[2] /= len;
    };

    simd_fvec<S> off_x, off_y;

    for (int i = 0; i < S; i++) {
        off_x[i] = (float)ray_packet_layout_x[i];
        off_y[i] = (float)ray_packet_layout_y[i];
    }

    size_t i = 0;
    out_rays.resize(r.w * r.h / S + ((r.w * r.h) % 4 != 0));

    for (int y = r.y; y < r.y + r.h - (r.h & (DimY - 1)); y += DimY) {
        simd_fvec<S> yy = { (float)y };
        yy += off_y;

        for (int x = r.x; x < r.x + r.w - (r.w & (DimX - 1)); x += DimX) {
            auto &out_r = out_rays[i++];

            simd_fvec<S> xx = { (float)x };
            xx += off_x;

            out_r.o[0] = { cam.origin[0] };
            out_r.o[1] = { cam.origin[1] };
            out_r.o[2] = { cam.origin[2] };

            get_pix_dirs(xx, yy, out_r.d);

            out_r.x = x;
            out_r.y = y;
        }
    }
}

template <int S>
bool ray::NS::IntersectTris(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const tri_accel_t *tris, uint32_t num_tris, uint32_t obj_index, hit_data_t<S> &out_inter) {
    hit_data_t<S> inter;
    inter.obj_index = { reinterpret_cast<const int&>(obj_index) };
    inter.t = out_inter.t;

    for (uint32_t i = 0; i < num_tris; i++) {
        _IntersectTri(r, ray_mask, tris[i], i, inter);
    }

    const auto &fmask = cast<simd_fvec<S>>(inter.mask);

    out_inter.mask = out_inter.mask | inter.mask;

    where(inter.mask, out_inter.obj_index) = inter.obj_index;
    where(inter.mask, out_inter.prim_index) = inter.prim_index;

    out_inter.t = inter.t; // already contains min value

    where(fmask, out_inter.u) = inter.u;
    where(fmask, out_inter.v) = inter.v;

    return inter.mask.not_all_zeros();
}

template <int S>
bool ray::NS::IntersectTris(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const tri_accel_t *tris, const uint32_t *indices, uint32_t num_tris, uint32_t obj_index, hit_data_t<S> &out_inter) {
    hit_data_t<S> inter;
    inter.obj_index = { reinterpret_cast<const int&>(obj_index) };
    inter.t = out_inter.t;

    for (uint32_t i = 0; i < num_tris; i++) {
        uint32_t index = indices[i];
        _IntersectTri(r, ray_mask, tris[index], index, inter);
    }

    const auto &fmask = cast<simd_fvec<S>>(inter.mask);

    out_inter.mask = out_inter.mask | inter.mask;

    where(inter.mask, out_inter.obj_index) = inter.obj_index;
    where(inter.mask, out_inter.prim_index) = inter.prim_index;

    out_inter.t = inter.t; // already contains min value

    where(fmask, out_inter.u) = inter.u;
    where(fmask, out_inter.v) = inter.v;

    return inter.mask.not_all_zeros();
}

template <int S>
bool ray::NS::Traverse_MacroTree_CPU(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const simd_fvec<S> inv_d[3], const bvh_node_t *nodes, uint32_t root_index,
                                     const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                     const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t<S> &inter) {
    bool res = false;

    TraversalState<S> st;

    st.queue[0].mask = ray_mask;

    st.queue[0].src = FromSibling;
    st.queue[0].cur = root_index;

    if (!is_leaf_node(nodes[root_index])) {
        st.queue[0].src = FromParent;
        st.select_near_child(r, nodes[root_index]);
    }

    while (st.index < st.num) {
        uint32_t &cur = st.queue[st.index].cur;
        eTraversalSource &src = st.queue[st.index].src;

        switch (src) {
        case FromChild:
            if (cur == root_index || cur == 0xffffffff) {
                st.index++;
                continue;
            }
            if (cur == near_child(r, st.queue[st.index].mask, nodes[nodes[cur].parent])) {
                cur = nodes[cur].sibling;
                src = FromSibling;
            } else {
                cur = nodes[cur].parent;
                src = FromChild;
            }
            break;
        case FromSibling: {
            auto mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]) & st.queue[st.index].mask;
            if (mask1.all_zeros()) {
                cur = nodes[cur].parent;
                src = FromChild;
            } else {
                auto mask2 = and_not(mask1, st.queue[st.index].mask);
                if (mask2.not_all_zeros()) {
                    st.queue[st.num].cur = nodes[cur].parent;
                    st.queue[st.num].mask = mask2;
                    st.queue[st.num].src = FromChild;
                    st.num++;
                    st.queue[st.index].mask = mask1;
                }

                if (is_leaf_node(nodes[cur])) {
                    // process leaf
                    for (uint32_t i = nodes[cur].prim_index; i < nodes[cur].prim_index + nodes[cur].prim_count; i++) {
                        const auto &mi = mesh_instances[mi_indices[i]];
                        const auto &m = meshes[mi.mesh_index];
                        const auto &tr = transforms[mi.tr_index];

                        auto bbox_mask = bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max) & st.queue[st.index].mask;
                        if (bbox_mask.all_zeros()) continue;

                        ray_packet_t<S> _r = TransformRay(r, tr.inv_xform);

                        simd_fvec<S> _inv_d[3];
                        safe_invert(_r.d, _inv_d);

                        res |= Traverse_MicroTree_CPU(_r, bbox_mask, _inv_d, nodes, m.node_index, tris, tri_indices, (int)mi_indices[i], inter);
                    }

                    cur = nodes[cur].parent;
                    src = FromChild;
                } else {
                    src = FromParent;
                    st.select_near_child(r, nodes[cur]);
                }
            }
        }
        break;
        case FromParent: {
            auto mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]) & st.queue[st.index].mask;
            if (mask1.all_zeros()) {
                cur = nodes[cur].sibling;
                src = FromSibling;
            } else {
                auto mask2 = and_not(mask1, st.queue[st.index].mask);
                if (mask2.not_all_zeros()) {
                    st.queue[st.num].cur = nodes[cur].sibling;
                    st.queue[st.num].mask = mask2;
                    st.queue[st.num].src = FromSibling;
                    st.num++;
                    st.queue[st.index].mask = mask1;
                }

                if (is_leaf_node(nodes[cur])) {
                    // process leaf
                    for (uint32_t i = nodes[cur].prim_index; i < nodes[cur].prim_index + nodes[cur].prim_count; i++) {
                        const auto &mi = mesh_instances[mi_indices[i]];
                        const auto &m = meshes[mi.mesh_index];
                        const auto &tr = transforms[mi.tr_index];

                        auto bbox_mask = bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max) & st.queue[st.index].mask;
                        if (bbox_mask.all_zeros()) continue;

                        ray_packet_t<S> _r = TransformRay(r, tr.inv_xform);

                        simd_fvec<S> _inv_d[3];
                        safe_invert(_r.d, _inv_d);

                        res |= Traverse_MicroTree_CPU(_r, bbox_mask, _inv_d, nodes, m.node_index, tris, tri_indices, (int)mi_indices[i], inter);
                    }

                    cur = nodes[cur].sibling;
                    src = FromSibling;
                } else {
                    src = FromParent;
                    st.select_near_child(r, nodes[cur]);
                }
            }
        }
        break;
        }
    }
    return res;
}

template <int S>
bool ray::NS::Traverse_MicroTree_CPU(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const simd_fvec<S> inv_d[3], const bvh_node_t *nodes, uint32_t root_index,
                                     const tri_accel_t *tris, const uint32_t *indices, int obj_index, hit_data_t<S> &inter) {
    bool res = false;

    TraversalState<S> st;

    st.queue[0].mask = ray_mask;

    st.queue[0].src = FromSibling;
    st.queue[0].cur = root_index;

    if (!is_leaf_node(nodes[root_index])) {
        st.queue[0].src = FromParent;
        st.select_near_child(r, nodes[root_index]);
    }

    while (st.index < st.num) {
        uint32_t &cur = st.queue[st.index].cur;
        eTraversalSource &src = st.queue[st.index].src;

        switch (src) {
        case FromChild:
            if (cur == root_index || cur == 0xffffffff) {
                st.index++;
                continue;
            }
            if (cur == near_child(r, st.queue[st.index].mask, nodes[nodes[cur].parent])) {
                cur = nodes[cur].sibling;
                src = FromSibling;
            } else {
                cur = nodes[cur].parent;
                src = FromChild;
            }
            break;
        case FromSibling: {
            auto mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]) & st.queue[st.index].mask;
            if (mask1.all_zeros()) {
                cur = nodes[cur].parent;
                src = FromChild;
            } else {
                auto mask2 = and_not(mask1, st.queue[st.index].mask);
                if (mask2.not_all_zeros()) {
                    st.queue[st.num].cur = nodes[cur].parent;
                    st.queue[st.num].mask = mask2;
                    st.queue[st.num].src = FromChild;
                    st.num++;
                    st.queue[st.index].mask = mask1;
                }

                if (is_leaf_node(nodes[cur])) {
                    // process leaf
                    res |= IntersectTris(r, st.queue[st.index].mask, tris, &indices[nodes[cur].prim_index], nodes[cur].prim_count, obj_index, inter);

                    cur = nodes[cur].parent;
                    src = FromChild;
                } else {
                    src = FromParent;
                    st.select_near_child(r, nodes[cur]);
                }
            }
        }
        break;
        case FromParent: {
            auto mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]) & st.queue[st.index].mask;
            if (mask1.all_zeros()) {
                cur = nodes[cur].sibling;
                src = FromSibling;
            } else {
                auto mask2 = and_not(mask1, st.queue[st.index].mask);
                if (mask2.not_all_zeros()) {
                    st.queue[st.num].cur = nodes[cur].sibling;
                    st.queue[st.num].mask = mask2;
                    st.queue[st.num].src = FromSibling;
                    st.num++;
                    st.queue[st.index].mask = mask1;
                }

                if (is_leaf_node(nodes[cur])) {
                    // process leaf
                    res |= IntersectTris(r, st.queue[st.index].mask, tris, &indices[nodes[cur].prim_index], nodes[cur].prim_count, obj_index, inter);

                    cur = nodes[cur].sibling;
                    src = FromSibling;
                } else {
                    src = FromParent;
                    st.select_near_child(r, nodes[cur]);
                }
            }
        }
        break;
        }
    }
    return res;
}

template <int S>
ray::NS::ray_packet_t<S> ray::NS::TransformRay(const ray_packet_t<S> &r, const float *xform) {
    ray_packet_t<S> _r = r;

    _r.o[0] = r.o[0] * xform[0] + r.o[1] * xform[4] + r.o[2] * xform[8] + xform[12];
    _r.o[1] = r.o[0] * xform[1] + r.o[1] * xform[5] + r.o[2] * xform[9] + xform[13];
    _r.o[2] = r.o[0] * xform[2] + r.o[1] * xform[6] + r.o[2] * xform[10] + xform[14];

    _r.d[0] = r.d[0] * xform[0] + r.d[1] * xform[4] + r.d[2] * xform[8];
    _r.d[1] = r.d[0] * xform[1] + r.d[1] * xform[5] + r.d[2] * xform[9];
    _r.d[2] = r.d[0] * xform[2] + r.d[1] * xform[6] + r.d[2] * xform[10];

    return _r;
}
