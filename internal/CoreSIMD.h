//#pragma once

#include <vector>

#include "TextureAtlas2.h"

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
    // color of ray
    simd_fvec<S> c[3];
    // derivatives
    simd_fvec<S> do_dx[3], dd_dx[3], do_dy[3], dd_dy[3];
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
    force_inline hit_data_t() {
        mask = { 0 };
        obj_index = { -1 };
        prim_index = { -1 };
        t = { MAX_DIST };
    }

    // hint for math::aligned_vector
    static const size_t alignment = alignof(simd_fvec16);
};

struct environment_t {
    float sun_dir[3];
    float sun_col[3];
    float sky_col[3];
    float sun_softness;
};

// Generating rays
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
bool Traverse_MacroTree_CPU(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                            const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                            const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t<S> &inter);
// stack-less cpu-style traversal of inner nodes
template <int S>
bool Traverse_MicroTree_CPU(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                            const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t<S> &inter);

// Transform
template <int S>
ray_packet_t<S> TransformRay(const ray_packet_t<S> &r, const float *xform);
simd_fvec2 TransformUVs(const simd_fvec2 &_uvs, const simd_fvec2 &atlas_size, const texture_t *t, int mip_level);

template <int S>
void TransformNormal(const simd_fvec<S> n[3], const float *inv_xform, simd_fvec<S> out_n[3]);

// Sample texture
simd_fvec4 SampleNearest(const ref::TextureAtlas2 &atlas, const texture_t &t, const simd_fvec2 &uvs, float lod);
simd_fvec4 SampleBilinear(const ref::TextureAtlas2 &atlas, const texture_t &t, const simd_fvec2 &uvs, int lod);
simd_fvec4 SampleTrilinear(const ref::TextureAtlas2 &atlas, const texture_t &t, const simd_fvec2 &uvs, float lod);
simd_fvec4 SampleAnisotropic(const ref::TextureAtlas2 &atlas, const texture_t &t, const simd_fvec2 &uvs, const simd_fvec2 &duv_dx, const simd_fvec2 &duv_dy);

// Shade
template <int S>
void ShadeSurface(const simd_ivec<S> &index, const int iteration, const float *halton, const hit_data_t<S> &inter, const ray_packet_t<S> &ray,
                  const environment_t &env, const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                  const mesh_t *meshes, const transform_t *transforms, const uint32_t *vtx_indices, const vertex_t *vertices,
                  const bvh_node_t *nodes, uint32_t node_index, const tri_accel_t *tris, const uint32_t *tri_indices,
                  const material_t *materials, const texture_t *textures, const ray::ref::TextureAtlas2 &tex_atlas, simd_fvec<S> out_rgba[4], simd_ivec<S> *out_secondary_masks, ray_packet_t<S> *out_secondary_rays, int *out_secondary_rays_count);
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

template <int S>
force_inline simd_fvec<S> dot(const simd_fvec<S> v1[3], const simd_fvec<S> v2[3]) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template <int S>
force_inline simd_fvec<S> abs(const simd_fvec<S> v) {
    // TODO: find faster implementation
    return max(v, -v);
}

template <int S>
force_inline void cross(const simd_fvec<S> v1[3], const simd_fvec<S> v2[3], simd_fvec<S> res[3]) {
    res[0] = v1[1] * v2[2] - v1[2] * v2[1];
    res[1] = v1[2] * v2[0] - v1[0] * v2[2];
    res[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

template <int S>
force_inline simd_fvec<S> clamp(const simd_fvec<S> &v, float min, float max) {
    simd_fvec<S> ret = v;
    where(ret < min, ret) = min;
    where(ret > max, ret) = max;
    return ret;
}

force_inline int hash(int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

template <int S>
force_inline simd_ivec<S> hash(const simd_ivec<S> &x) {
    simd_ivec<S> ret;
    ret = ((x >> 16) ^ x) * 0x45d9f3b;
    ret = ((ret >> 16) ^ ret) * 0x45d9f3b;
    ret = (ret >> 16) ^ ret;
    return ret;
}

force_inline float length(const simd_fvec2 &x) {
    return sqrtf(x[0] * x[0] + x[1] * x[1]);
}

force_inline float floor(float x) {
    return (float)((int)x - (x < 0.0f));
}

}
}

template <int DimX, int DimY>
void ray::NS::GeneratePrimaryRays(const camera_t &cam, const rect_t &r, int w, int h, math::aligned_vector<ray_packet_t<DimX * DimY>> &out_rays) {
    const int S = DimX * DimY;

    static_assert(S <= 16, "!");

    simd_fvec<S> ww = { (float)w }, hh = { (float)h };

    float k = float(h) / w;

    simd_fvec<S> fwd[3] = { { cam.fwd[0] }, { cam.fwd[1] }, { cam.fwd[2] } },
                 side[3] = { { cam.side[0] }, { cam.side[1] }, { cam.side[2] } },
                 up[3] = { { cam.up[0] * k }, { cam.up[1] * k }, { cam.up[2] * k } };

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

            

            simd_fvec<S> _d[3], _dx[3], _dy[3];
            get_pix_dirs(xx, yy, _d);
            get_pix_dirs(xx + 1.0f, yy, _dx);
            get_pix_dirs(xx, yy + 1.0f, _dy);

            for (int j = 0; j < 3; j++) {
                out_r.d[j] = _d[j];
                out_r.o[j] = { cam.origin[j] };
                out_r.c[j] = { 1.0f };

                out_r.do_dx[j] = { 0.0f };
                out_r.dd_dx[j] = _dx[j] - out_r.d[j];
                out_r.do_dy[j] = { 0.0f };
                out_r.dd_dy[j] = _dy[j] - out_r.d[j];
            }

            out_r.x = x;
            out_r.y = y;
        }
    }
}

template <int S>
bool ray::NS::IntersectTris(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const tri_accel_t *tris, uint32_t num_tris, uint32_t obj_index, hit_data_t<S> &out_inter) {
    hit_data_t<S> inter = { Uninitialize };
    inter.mask = { 0 };
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
    hit_data_t<S> inter = { Uninitialize };
    inter.mask = { 0 };
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
bool ray::NS::Traverse_MacroTree_CPU(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t root_index,
                                     const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                     const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t<S> &inter) {
    bool res = false;

    simd_fvec<S> inv_d[3];
    safe_invert(r.d, inv_d);

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

                        res |= Traverse_MicroTree_CPU(_r, bbox_mask, nodes, m.node_index, tris, tri_indices, (int)mi_indices[i], inter);
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

                        res |= Traverse_MicroTree_CPU(_r, bbox_mask, nodes, m.node_index, tris, tri_indices, (int)mi_indices[i], inter);
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
bool ray::NS::Traverse_MicroTree_CPU(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t root_index,
                                     const tri_accel_t *tris, const uint32_t *indices, int obj_index, hit_data_t<S> &inter) {
    bool res = false;

    simd_fvec<S> inv_d[3];
    safe_invert(r.d, inv_d);

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
force_inline ray::NS::ray_packet_t<S> ray::NS::TransformRay(const ray_packet_t<S> &r, const float *xform) {
    ray_packet_t<S> _r = r;

    _r.o[0] = r.o[0] * xform[0] + r.o[1] * xform[4] + r.o[2] * xform[8] + xform[12];
    _r.o[1] = r.o[0] * xform[1] + r.o[1] * xform[5] + r.o[2] * xform[9] + xform[13];
    _r.o[2] = r.o[0] * xform[2] + r.o[1] * xform[6] + r.o[2] * xform[10] + xform[14];

    _r.d[0] = r.d[0] * xform[0] + r.d[1] * xform[4] + r.d[2] * xform[8];
    _r.d[1] = r.d[0] * xform[1] + r.d[1] * xform[5] + r.d[2] * xform[9];
    _r.d[2] = r.d[0] * xform[2] + r.d[1] * xform[6] + r.d[2] * xform[10];

    return _r;
}

force_inline ray::NS::simd_fvec2 ray::NS::TransformUVs(const simd_fvec2 &_uvs, const simd_fvec2 &atlas_size, const texture_t *t, int mip_level) {
    simd_fvec2 pos = { (float)t->pos[mip_level][0], (float)t->pos[mip_level][1] };
    simd_fvec2 size = { (float)(t->size[0] >> mip_level), (float)(t->size[1] >> mip_level) };
    simd_fvec2 uvs = { _uvs[0] - floor(_uvs[0]), _uvs[1] - floor(_uvs[1]) };
    simd_fvec2 res = pos + uvs * size + 1.0f;

    return res / atlas_size;
}

template <int S>
void ray::NS::TransformNormal(const simd_fvec<S> n[3], const float *inv_xform, simd_fvec<S> out_n[3]) {
    out_n[0] = n[0] * inv_xform[0] + n[1] * inv_xform[1] + n[2] * inv_xform[2];
    out_n[1] = n[0] * inv_xform[4] + n[1] * inv_xform[5] + n[2] * inv_xform[6];
    out_n[2] = n[0] * inv_xform[8] + n[1] * inv_xform[9] + n[2] * inv_xform[10];
}

ray::NS::simd_fvec4 ray::NS::SampleNearest(const ref::TextureAtlas2 &atlas, const texture_t &t, const simd_fvec2 &uvs, float lod) {
    int _lod = (int)lod;

    simd_fvec2 atlas_size = { atlas.size_x(), atlas.size_y() };
    simd_fvec2 _uvs = TransformUVs(uvs, atlas_size, &t, _lod);

    if (_lod > MAX_MIP_LEVEL) _lod = MAX_MIP_LEVEL;

    int page = t.page[_lod];

    const auto &pix = atlas.Get(page, _uvs[0], _uvs[1]);

    const float k = 1.0f / 255.0f;
    return simd_fvec4{ pix.r * k, pix.g * k, pix.b * k, pix.a * k };
}

ray::NS::simd_fvec4 ray::NS::SampleBilinear(const ref::TextureAtlas2 &atlas, const texture_t &t, const simd_fvec2 &uvs, int lod) {
    simd_fvec2 atlas_size = { atlas.size_x(), atlas.size_y() };
    simd_fvec2 _uvs = TransformUVs(uvs, atlas_size, &t, lod);

    int page = t.page[lod];

    _uvs = _uvs * atlas_size - 0.5f;

    const auto &p00 = atlas.Get(page, int(_uvs[0]), int(_uvs[1]));
    const auto &p01 = atlas.Get(page, int(_uvs[0] + 1), int(_uvs[1]));
    const auto &p10 = atlas.Get(page, int(_uvs[0]), int(_uvs[1] + 1));
    const auto &p11 = atlas.Get(page, int(_uvs[0] + 1), int(_uvs[1] + 1));

    float kx = _uvs[0] - floor(_uvs[0]), ky = _uvs[1] - floor(_uvs[1]);

    const auto p0 = simd_fvec4{ p01.r * kx + p00.r * (1 - kx),
                                p01.g * kx + p00.g * (1 - kx),
                                p01.b * kx + p00.b * (1 - kx),
                                p01.a * kx + p00.a * (1 - kx) };

    const auto p1 = simd_fvec4{ p11.r * kx + p10.r * (1 - kx),
                                p11.g * kx + p10.g * (1 - kx),
                                p11.b * kx + p10.b * (1 - kx),
                                p11.a * kx + p10.a * (1 - kx) };

    const float k = 1.0f / 255.0f;
    return (p1 * ky + p0 * (1 - ky)) * k;
}

ray::NS::simd_fvec4 ray::NS::SampleTrilinear(const ref::TextureAtlas2 &atlas, const texture_t &t, const simd_fvec2 &uvs, float lod) {
    auto col1 = SampleBilinear(atlas, t, uvs, (int)floor(lod));
    auto col2 = SampleBilinear(atlas, t, uvs, (int)ceil(lod));

    const float k = lod - floor(lod);
    return col1 * (1 - k) + col2 * k;
}

ray::NS::simd_fvec4 ray::NS::SampleAnisotropic(const ref::TextureAtlas2 &atlas, const texture_t &t, const simd_fvec2 &uvs, const simd_fvec2 &duv_dx, const simd_fvec2 &duv_dy) {
    simd_fvec2 sz = { (float)t.size[0], (float)t.size[1] };
    
    float l1 = length(duv_dx * sz);
    float l2 = length(duv_dy * sz);

    float lod, k;
    simd_fvec2 step;

    if (l1 <= l2) {
        lod = log2(l1);
        k = l1 / l2;
        step = duv_dy;
    } else {
        lod = log2(l2);
        k = l2 / l1;
        step = duv_dx;
    }

    if (lod < 0.0f) lod = 0.0f;
    else if (lod > (float)MAX_MIP_LEVEL) lod = (float)MAX_MIP_LEVEL;

    simd_fvec2 _uvs = uvs - step * 0.5f;

    int num = (int)(2.0f / k);
    if (num < 1) num = 1;
    else if (num > 32) num = 32;

    step = step / float(num);

    simd_fvec4 res = { 0.0f };

    for (int i = 0; i < num; i++) {
        res += SampleTrilinear(atlas, t, _uvs, lod);
        _uvs += step;
    }

    return res / float(num);
}

template <int S>
__declspec(noinline) void ray::NS::ShadeSurface(const simd_ivec<S> &px_index, const int iteration, const float *halton, const hit_data_t<S> &inter, const ray_packet_t<S> &ray,
                           const environment_t &env, const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                           const mesh_t *meshes, const transform_t *transforms, const uint32_t *vtx_indices, const vertex_t *vertices,
                           const bvh_node_t *nodes, uint32_t node_index, const tri_accel_t *tris, const uint32_t *tri_indices,
                           const material_t *materials, const texture_t *textures, const ray::ref::TextureAtlas2 &tex_atlas, simd_fvec<S> out_rgba[4], simd_ivec<S> *out_secondary_masks, ray_packet_t<S> *out_secondary_rays, int *out_secondary_rays_count) {
    out_rgba[3] = { 1.0f };
    
    auto ino_hit = inter.mask ^ simd_ivec<S>(-1);
    auto no_hit = cast<simd_fvec<S>>(ino_hit);
    
    where(no_hit, out_rgba[0]) = ray.c[0] * env.sky_col[0];
    where(no_hit, out_rgba[1]) = ray.c[1] * env.sky_col[1];
    where(no_hit, out_rgba[2]) = ray.c[2] * env.sky_col[2];
    
    if (inter.mask.all_zeros()) return;

    const auto *I = ray.d;
    const simd_fvec<S> P[3] = { ray.o[0] + inter.t * I[0], ray.o[1] + inter.t * I[1], ray.o[2] + inter.t * I[2] };

    simd_fvec<S> w = simd_fvec<S>{ 1.0f } - inter.u - inter.v;

    simd_fvec<S> p1[3], p2[3], p3[3];
    simd_fvec<S> n1[3], n2[3], n3[3];
    simd_fvec<S> u1[2], u2[2], u3[2];
    simd_fvec<S> b1[3], b2[3], b3[3];

    simd_ivec<S> inter_prim_index = inter.prim_index;
    where(ino_hit, inter_prim_index) = { 0 };

    simd_ivec<S> mat_index = { -1 };

    simd_fvec<S> inv_xform1[3], inv_xform2[3], inv_xform3[3];

    for (int i = 0; i < S; i++) {
        const auto &v1 = vertices[vtx_indices[inter_prim_index[i] * 3 + 0]];
        const auto &v2 = vertices[vtx_indices[inter_prim_index[i] * 3 + 1]];
        const auto &v3 = vertices[vtx_indices[inter_prim_index[i] * 3 + 2]];

        p1[0][i] = v1.p[0]; p1[1][i] = v1.p[1]; p1[2][i] = v1.p[2];
        p2[0][i] = v2.p[0]; p2[1][i] = v2.p[1]; p2[2][i] = v2.p[2];
        p3[0][i] = v3.p[0]; p3[1][i] = v3.p[1]; p3[2][i] = v3.p[2];

        n1[0][i] = v1.n[0]; n1[1][i] = v1.n[1]; n1[2][i] = v1.n[2];
        n2[0][i] = v2.n[0]; n2[1][i] = v2.n[1]; n2[2][i] = v2.n[2];
        n3[0][i] = v3.n[0]; n3[1][i] = v3.n[1]; n3[2][i] = v3.n[2];

        u1[0][i] = v1.t0[0]; u1[1][i] = v1.t0[1];
        u2[0][i] = v2.t0[0]; u2[1][i] = v2.t0[1];
        u3[0][i] = v3.t0[0]; u3[1][i] = v3.t0[1];

        b1[0][i] = v1.b[0]; b1[1][i] = v1.b[1]; b1[2][i] = v1.b[2];
        b2[0][i] = v2.b[0]; b2[1][i] = v2.b[1]; b2[2][i] = v2.b[2];
        b3[0][i] = v3.b[0]; b3[1][i] = v3.b[1]; b3[2][i] = v3.b[2];

        mat_index[i] = reinterpret_cast<const int&>(tris[inter.prim_index[i]].mi);

        const auto *tr = &transforms[mesh_instances[inter.obj_index[i]].tr_index];

        inv_xform1[0][i] = tr->inv_xform[0]; inv_xform1[1][i] = tr->inv_xform[1]; inv_xform1[2][i] = tr->inv_xform[2];
        inv_xform2[0][i] = tr->inv_xform[4]; inv_xform2[1][i] = tr->inv_xform[5]; inv_xform2[2][i] = tr->inv_xform[6];
        inv_xform3[0][i] = tr->inv_xform[8]; inv_xform3[1][i] = tr->inv_xform[9]; inv_xform3[2][i] = tr->inv_xform[10];
    }

    simd_fvec<S> N[3] = { n1[0] * w + n2[0] * inter.u + n3[0] * inter.v,
                          n1[1] * w + n2[1] * inter.u + n3[1] * inter.v,
                          n1[2] * w + n2[2] * inter.u + n3[2] * inter.v };

    simd_fvec<S> uvs[2] = { u1[0] * w + u2[0] * inter.u + u3[0] * inter.v,
                            u1[1] * w + u2[1] * inter.u + u3[1] * inter.v };

    /////{
        // From 'Tracing Ray Differentials' [1999]

        simd_fvec<S> temp[3];

        temp[0] = ray.do_dx[0] + inter.t * ray.dd_dx[0];
        temp[1] = ray.do_dx[1] + inter.t * ray.dd_dx[1];
        temp[2] = ray.do_dx[2] + inter.t * ray.dd_dx[2];

        auto a = -dot(temp, N);
        auto b = dot(I, N);

        simd_fvec<S> dt_dx = -dot(temp, N) / dot(I, N);
        simd_fvec<S> do_dx[3] = { temp[0] + dt_dx * I[0], temp[1] + dt_dx * I[1], temp[2] + dt_dx * I[2] };
        simd_fvec<S> dd_dx[3] = { ray.dd_dx[0], ray.dd_dx[1], ray.dd_dx[2] };

        temp[0] = ray.do_dy[0] + inter.t * ray.dd_dy[0];
        temp[1] = ray.do_dy[1] + inter.t * ray.dd_dy[1];
        temp[2] = ray.do_dy[2] + inter.t * ray.dd_dy[2];

        simd_fvec<S> dt_dy = -dot(temp, N) / dot(I, N);
        simd_fvec<S> do_dy[3] = { temp[0] + dt_dy * I[0], temp[1] + dt_dy * I[1], temp[2] + dt_dy * I[2] };
        simd_fvec<S> dd_dy[3] = { ray.dd_dy[0], ray.dd_dy[1], ray.dd_dy[2] };

        // From 'Physically Based Rendering: ...' book

        simd_fvec<S> duv13[2] = { u1[0] - u3[0], u1[1] - u3[1] },
                     duv23[2] = { u2[0] - u3[0], u2[1] - u3[1] };
        simd_fvec<S> dp13[3] = { p1[0] - p3[0], p1[1] - p3[1], p1[2] - p3[2] },
                     dp23[3] = { p2[0] - p3[0], p2[1] - p3[1], p2[2] - p3[2] };

        simd_fvec<S> det_uv = duv13[0] * duv23[1] - duv13[1] * duv23[0];
        simd_fvec<S> inv_det_uv = 1.0f / det_uv;
        where(abs(det_uv) < FLT_EPS, inv_det_uv) = { 0.0f };

        const simd_fvec<S> dpdu[3] = { (duv23[1] * dp13[0] - duv13[1] * dp23[0]) * inv_det_uv,
                                       (duv23[1] * dp13[1] - duv13[1] * dp23[1]) * inv_det_uv,
                                       (duv23[1] * dp13[2] - duv13[1] * dp23[2]) * inv_det_uv };
        const simd_fvec<S> dpdv[3] = { (-duv23[0] * dp13[0] + duv13[0] * dp23[0]) * inv_det_uv,
                                       (-duv23[0] * dp13[1] + duv13[0] * dp23[1]) * inv_det_uv,
                                       (-duv23[0] * dp13[2] + duv13[0] * dp23[2]) * inv_det_uv };

        simd_fvec<S> A[2][2] = { { dpdu[0], dpdu[1] }, { dpdv[0], dpdv[1] } };
        simd_fvec<S> Bx[2] = { do_dx[0], do_dx[1] };
        simd_fvec<S> By[2] = { do_dy[0], do_dy[1] };

        auto mask1 = abs(N[0]) > abs(N[1]) & abs(N[0]) > abs(N[2]);
        where(mask1, A[0][0]) = dpdu[1];
        where(mask1, A[0][1]) = dpdu[2];
        where(mask1, A[1][0]) = dpdv[1];
        where(mask1, A[1][1]) = dpdv[2];
        where(mask1, Bx[0]) = do_dx[1];
        where(mask1, Bx[1]) = do_dx[2];
        where(mask1, By[0]) = do_dy[1];
        where(mask1, By[1]) = do_dy[2];

        auto mask2 = abs(N[1]) > abs(N[0]) & abs(N[1]) > abs(N[2]);
        where(mask2, A[0][1]) = dpdu[2];
        where(mask2, A[1][1]) = dpdv[2];
        where(mask2, Bx[1]) = do_dx[2];
        where(mask2, By[1]) = do_dy[2];

        simd_fvec<S> det = A[0][0] * A[1][1] - A[1][0] * A[0][1];
        simd_fvec<S> inv_det = 1.0f / det;
        where(abs(det) < FLT_EPS, inv_det) = { 0.0f };

        simd_fvec<S> duv_dx[2] = { (A[0][0] * Bx[0] - A[0][1] * Bx[1]) * inv_det,
                                   (A[1][0] * Bx[0] - A[1][1] * Bx[1]) * inv_det };
        simd_fvec<S> duv_dy[2] = { (A[0][0] * By[0] - A[0][1] * By[1]) * inv_det,
                                   (A[1][0] * By[0] - A[1][1] * By[1]) * inv_det };

        // Derivative for normal

        simd_fvec<S> dn1[3] = { n1[0] - n3[0], n1[1] - n3[1], n1[2] - n3[2] },
                     dn2[3] = { n2[0] - n3[0], n2[1] - n3[1], n2[2] - n3[2] };
        simd_fvec<S> dndu[3] = { (duv23[1] * dn1[0] - duv13[1] * dn2[0]) * inv_det_uv,
                                 (duv23[1] * dn1[1] - duv13[1] * dn2[1]) * inv_det_uv,
                                 (duv23[1] * dn1[2] - duv13[1] * dn2[2]) * inv_det_uv };
        simd_fvec<S> dndv[3] = { (-duv23[0] * dn1[0] + duv13[0] * dn2[0]) * inv_det_uv,
                                 (-duv23[0] * dn1[1] + duv13[0] * dn2[1]) * inv_det_uv,
                                 (-duv23[0] * dn1[2] + duv13[0] * dn2[2]) * inv_det_uv };

        simd_fvec<S> dndx[3] = { dndu[0] * duv_dx[0] + dndv[0] * duv_dx[1],
                                 dndu[1] * duv_dx[0] + dndv[1] * duv_dx[1],
                                 dndu[2] * duv_dx[0] + dndv[2] * duv_dx[1] };
        simd_fvec<S> dndy[3] = { dndu[0] * duv_dy[0] + dndv[0] * duv_dy[1],
                                 dndu[1] * duv_dy[0] + dndv[1] * duv_dy[1],
                                 dndu[2] * duv_dy[0] + dndv[2] * duv_dy[1] };

        simd_fvec<S> ddn_dx = dot(dd_dx, N) + dot(I, dndx);
        simd_fvec<S> ddn_dy = dot(dd_dy, N) + dot(I, dndy);

        ////////////////////////////////////////////////////////

        simd_fvec<S> B[3] = { b1[0] * w + b2[0] * inter.u + b3[0] * inter.v,
                              b1[1] * w + b2[1] * inter.u + b3[1] * inter.v,
                              b1[2] * w + b2[2] * inter.u + b3[2] * inter.v };

        simd_fvec<S> T[3];
        cross(B, N, T);
    /////}

    const simd_ivec<S> hi = (hash(px_index) + iteration) & (HaltonSeqLen - 1);

    simd_ivec<S> secondary_mask = { 0 };

    {
        simd_ivec<S> ray_queue[S];
        int index = 0;
        int num = 1;

        ray_queue[0] = inter.mask;

        while (index != num) {
            uint32_t first_oi = 0xffffffff, first_mi = 0xffffffff;

            for (int i = 0; i < S; i++) {
                if (!ray_queue[index][i]) continue;

                if (first_oi == 0xffffffff)
                    first_oi = inter.obj_index[i];

                const auto &tri = tris[inter.prim_index[i]];
                if (first_mi == 0xffffffff)
                    first_mi = tri.mi;
            }
    
            auto same_mi = mat_index == first_mi;
            auto diff_mi = and_not(same_mi, ray_queue[index]);

            if (diff_mi.not_all_zeros()) {
                ray_queue[num] = diff_mi;
                num++;
            }

            /////////////////////////////////////////

            const auto *mat = &materials[first_mi];

            simd_fvec<S> tex_normal[3], tex_albedo[3];

            for (int i = 0; i < S; i++) {
                if (!same_mi[i]) continue;

                simd_fvec2 _duv_dx = { duv_dx[0][i], duv_dx[1][i] };
                simd_fvec2 _duv_dy = { duv_dy[0][i], duv_dy[1][i] };

                simd_fvec2 _uvs = { uvs[0][i], uvs[1][i] };
                auto normals = SampleAnisotropic(tex_atlas, textures[mat->textures[NORMALS_TEXTURE]], _uvs, _duv_dx, _duv_dy);
            
                tex_normal[0][i] = normals[0] * 2.0f - 1.0f;
                tex_normal[1][i] = normals[1] * 2.0f - 1.0f;
                tex_normal[2][i] = normals[2] * 2.0f - 1.0f;

                auto albedo = SampleAnisotropic(tex_atlas, textures[mat->textures[MAIN_TEXTURE]], _uvs, _duv_dx, _duv_dy);

                tex_albedo[0][i] = pow(albedo[0] * mat->main_color[0], 2.2f);
                tex_albedo[1][i] = pow(albedo[1] * mat->main_color[1], 2.2f);
                tex_albedo[2][i] = pow(albedo[2] * mat->main_color[2], 2.2f);
            }

            simd_fvec<S> temp[3];
            temp[0] = tex_normal[0] * B[0] + tex_normal[2] * N[0] + tex_normal[1] * T[0];
            temp[1] = tex_normal[0] * B[1] + tex_normal[2] * N[1] + tex_normal[1] * T[1];
            temp[2] = tex_normal[0] * B[2] + tex_normal[2] * N[2] + tex_normal[1] * T[2];

            //////////////////////////////////////////

            simd_fvec<S> _N[3] = { dot(temp, inv_xform1), dot(temp, inv_xform2), dot(temp, inv_xform3) },
                         _B[3] = { dot(B, inv_xform1), dot(B, inv_xform2), dot(B, inv_xform3) },
                         _T[3] = { dot(T, inv_xform1), dot(T, inv_xform2), dot(T, inv_xform3) };

            //////////////////////////////////////////

            if (mat->type == DiffuseMaterial) {
                simd_fvec<S> k = _N[0] * env.sun_dir[0] + _N[1] * env.sun_dir[1] + _N[2] * env.sun_dir[2];
                simd_fvec<S> v = { 1.0f };

                const auto _mask = cast<simd_ivec<S>>(k > 0.0f) & ray_queue[index];

                if (_mask.not_all_zeros()) {
                    simd_fvec<S> V[3], TT[3], BB[3];

                    simd_fvec<S> temp[3] = { { env.sun_dir[0] }, { env.sun_dir[1] }, { env.sun_dir[2] } };

                    cross(temp, B, TT);
                    cross(temp, TT, BB);

                    for (int i = 0; i < S; i++) {
                        if (!_mask[i]) continue;

                        const float z = 1.0f - halton[hi[i] * 2] * env.sun_softness;
                        const float temp = std::sqrt(1.0f - z * z);

                        const float phi = halton[hi[i] * 2 + 1] * 2 * 3.14159265358979323846264338327950288f;
                        const float cos_phi = std::cos(phi);
                        const float sin_phi = std::sin(phi);

                        V[0][i] = temp * sin_phi * BB[0][i] + z * env.sun_dir[0] + temp * cos_phi * TT[0][i];
                        V[1][i] = temp * sin_phi * BB[1][i] + z * env.sun_dir[1] + temp * cos_phi * TT[1][i];
                        V[2][i] = temp * sin_phi * BB[2][i] + z * env.sun_dir[2] + temp * cos_phi * TT[2][i];
                    }

                    ray_packet_t<S> r;

                    r.o[0] = P[0] + HIT_BIAS * _N[0];
                    r.o[1] = P[1] + HIT_BIAS * _N[1];
                    r.o[2] = P[2] + HIT_BIAS * _N[2];

                    r.d[0] = V[0];
                    r.d[1] = V[1];
                    r.d[2] = V[2];

                    hit_data_t<S> inter;
                    Traverse_MacroTree_CPU(r, _mask, nodes, node_index, mesh_instances, mi_indices, meshes, transforms, tris, tri_indices, inter);
                    
                    where(cast<simd_fvec<S>>(inter.mask), v) = { 0.0f };
                }

                k = clamp(k, 0.0f, 1.0f);

                const auto &mask = cast<simd_fvec<S>>(same_mi);

                where(mask, out_rgba[0]) = ray.c[0] * tex_albedo[0] * env.sun_col[0] * v * k;
                where(mask, out_rgba[1]) = ray.c[1] * tex_albedo[1] * env.sun_col[1] * v * k;
                where(mask, out_rgba[2]) = ray.c[2] * tex_albedo[2] * env.sun_col[2] * v * k;

                // !!!!!!!!!!!!
                simd_fvec<S> rc[3] = { ray.c[0] * tex_albedo[0],
                                       ray.c[1] * tex_albedo[1],
                                       ray.c[2] * tex_albedo[2] };

                simd_fvec<S> V[3];

                for (int i = 0; i < S; i++) {
                    if (!same_mi[i]) continue;

                    const float z = 1.0f - halton[hi[i] * 2];
                    const float temp = std::sqrt(1.0f - z * z);

                    const float phi = halton[((hash(hi[i]) + iteration) & (HaltonSeqLen - 1)) * 2 + 0] * 2 * 3.14159265358979323846264338327950288f;
                    const float cos_phi = std::cos(phi);
                    const float sin_phi = std::sin(phi);

                    V[0][i] = temp * sin_phi * B[0][i] + z * _N[0][i] + temp * cos_phi * T[0][i];
                    V[1][i] = temp * sin_phi * B[1][i] + z * _N[1][i] + temp * cos_phi * T[1][i];
                    V[2][i] = temp * sin_phi * B[2][i] + z * _N[2][i] + temp * cos_phi * T[2][i];

                    rc[0][i] *= z;
                    rc[1][i] *= z;
                    rc[2][i] *= z;
                }

                simd_fvec<S> thres = dot(rc, rc);

                const auto new_ray_mask = (thres > 0.005f) & cast<simd_fvec<S>>(same_mi);

                if ((cast<simd_ivec<S>>(new_ray_mask)).not_all_zeros()) {
                    const int index = *out_secondary_rays_count;
                    auto &r = out_secondary_rays[index];

                    secondary_mask = secondary_mask | cast<simd_ivec<S>>(new_ray_mask);

                    where(new_ray_mask, r.o[0]) = P[0] + HIT_BIAS * _N[0];
                    where(new_ray_mask, r.o[1]) = P[1] + HIT_BIAS * _N[1];
                    where(new_ray_mask, r.o[2]) = P[2] + HIT_BIAS * _N[2];

                    where(new_ray_mask, r.d[0]) = V[0];
                    where(new_ray_mask, r.d[1]) = V[1];
                    where(new_ray_mask, r.d[2]) = V[2];

                    where(new_ray_mask, r.c[0]) = rc[0];
                    where(new_ray_mask, r.c[1]) = rc[1];
                    where(new_ray_mask, r.c[2]) = rc[2];

                    where(new_ray_mask, r.do_dx[0]) = do_dx[0];
                    where(new_ray_mask, r.do_dx[1]) = do_dx[1];
                    where(new_ray_mask, r.do_dx[2]) = do_dx[2];

                    where(new_ray_mask, r.do_dy[0]) = do_dy[0];
                    where(new_ray_mask, r.do_dy[1]) = do_dy[1];
                    where(new_ray_mask, r.do_dy[2]) = do_dy[2];

                    where(new_ray_mask, r.dd_dx[0]) = dd_dx[0] - 2 * (dot(I, _N) * dndx[0] + ddn_dx * _N[0]);
                    where(new_ray_mask, r.dd_dx[1]) = dd_dx[1] - 2 * (dot(I, _N) * dndx[1] + ddn_dx * _N[1]);
                    where(new_ray_mask, r.dd_dx[2]) = dd_dx[2] - 2 * (dot(I, _N) * dndx[2] + ddn_dx * _N[2]);

                    where(new_ray_mask, r.dd_dy[0]) = dd_dy[0] - 2 * (dot(I, _N) * dndy[0] + ddn_dy * _N[0]);
                    where(new_ray_mask, r.dd_dy[1]) = dd_dy[1] - 2 * (dot(I, _N) * dndy[1] + ddn_dy * _N[1]);
                    where(new_ray_mask, r.dd_dy[2]) = dd_dy[2] - 2 * (dot(I, _N) * dndy[2] + ddn_dy * _N[2]);
                }
            }

            index++;
        }
    }

    if (secondary_mask.not_all_zeros()) {
        const int index = *out_secondary_rays_count;
        out_secondary_masks[index] = secondary_mask;
        out_secondary_rays[index].x = ray.x;
        out_secondary_rays[index].y = ray.y;
        (*out_secondary_rays_count)++;
    }
}