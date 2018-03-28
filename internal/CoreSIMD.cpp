#include "CoreSIMD.h"

#include <cassert>
//#include <limits>

namespace ray {
namespace NS {
const simd_fvec16 _0_5 = { 0.5f };

const simd_fvec16 ZERO = { 0.0f };
const simd_fvec16 MINUS_ZERO = { -0.0f };

const simd_fvec16 ONE = { 1.0f };
const simd_fvec16 TWO = { 2.0f };
const simd_fvec16 FOUR = { 4.0f };

const simd_fvec16 HIT_EPS = { ray::HIT_EPS };
const simd_fvec16 M_HIT_EPS = { -ray::HIT_EPS };

const simd_ivec16 FF_MASK = { -1 };

force_inline void _IntersectTri(const ray_packet_t &r, const simd_ivec16 &ray_mask, const tri_accel_t &tri, uint32_t prim_index, hit_data_t &inter) {
    simd_fvec16 nu = { tri.nu }, nv = { tri.nv }, np = { tri.np };
    simd_fvec16 pu = { tri.pu }, pv = { tri.pv };

    simd_fvec16 e0u = { tri.e0u }, e0v = { tri.e0v };
    simd_fvec16 e1u = { tri.e1u }, e1v = { tri.e1v };

    const int _next_u[] = { 1, 0, 0 },
              _next_v[] = { 2, 2, 1 };

    int w = (tri.ci & TRI_W_BITS),
        u = _next_u[w],
        v = _next_v[w];

    // from "Ray-Triangle Intersection Algorithm for Modern CPU Architectures" [2007]

    simd_fvec16 det = r.d[u] * nu + r.d[v] * nv + r.d[w];
    simd_fvec16 dett = np - (r.o[u] * nu + r.o[v] * nv + r.o[w]);
    simd_fvec16 Du = r.d[u] * dett - (pu - r.o[u]) * det;
    simd_fvec16 Dv = r.d[v] * dett - (pv - r.o[v]) * det;
    simd_fvec16 detu = e1v * Du - e1u * Dv;
    simd_fvec16 detv = e0u * Dv - e0v * Du;

    simd_fvec16 tmpdet0 = det - detu - detv;

    //////////////////////////////////////////////////////////////////////////

    simd_fvec16 mm = ((tmpdet0 > M_HIT_EPS) & (detu > M_HIT_EPS) & (detv > M_HIT_EPS)) |
                     ((tmpdet0 < HIT_EPS) & (detu < HIT_EPS) & (detv < HIT_EPS));

    simd_ivec16 imask = reinterpret_cast<const simd_ivec16&>(mm) & ray_mask;

    //////////////////////////////////////////////////////////////////////////

    if (imask.all_zeros()) return; // no intersection found

    simd_fvec16 rdet = ONE / det;
    simd_fvec16 t = dett * rdet;

    simd_ivec16 t_valid = reinterpret_cast<const simd_ivec16&>((t < inter.t) & (t > ZERO));
    imask = imask & t_valid;

    if (imask.all_zeros()) return; // all intersections further than needed

    simd_fvec16 bar_u = detu * rdet;
    simd_fvec16 bar_v = detv * rdet;

    const auto &fmask = reinterpret_cast<const simd_fvec16&>(imask);

    inter.mask = inter.mask | imask;

    where(imask, inter.prim_index) = simd_ivec16{ reinterpret_cast<const int&>(prim_index) };
    where(fmask, inter.t) = t;
    where(fmask, inter.u) = bar_u;
    where(fmask, inter.v) = bar_v;
}

force_inline simd_ivec16 bbox_test(const simd_fvec16 o[3], const simd_fvec16 inv_d[3], const simd_fvec16 t, const float _bbox_min[3], const float _bbox_max[3]) {
    simd_fvec16 box_min[3] = { { _bbox_min[0] }, { _bbox_min[1] }, { _bbox_min[2] } },
                box_max[3] = { { _bbox_max[0] }, { _bbox_max[1] }, { _bbox_max[2] } };

    simd_fvec16 low, high, tmin, tmax;

    low = inv_d[0] * (box_min[0] - o[0]);
    high = inv_d[0] * (box_max[0] - o[0]);
    tmin = min(low, high);
    tmax = max(low, high);

    low = inv_d[1] * (box_min[1] - o[1]);
    high = inv_d[1] * (box_max[1] - o[1]);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));

    low = inv_d[2] * (box_min[2] - o[2]);
    high = inv_d[2] * (box_max[2] - o[2]);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));

    simd_fvec16 mask = (tmin <= tmax) & (tmin <= t) & (tmax > ZERO);

    return reinterpret_cast<const simd_ivec16&>(mask);
}

force_inline simd_ivec16 bbox_test(const simd_fvec16 o[3], const simd_fvec16 inv_d[3], const simd_fvec16 t, const bvh_node_t &node) {
    return bbox_test(o, inv_d, t, node.bbox[0], node.bbox[1]);
}

force_inline uint32_t near_child(const ray_packet_t &r, const simd_ivec16 ray_mask, const bvh_node_t &node) {
    simd_ivec16 mask = reinterpret_cast<const simd_ivec16 &>(r.d[node.space_axis] < ZERO);
    if (mask.all_zeros(ray_mask)) {
        return node.left_child;
    } else {
        assert(and_not(mask, FF_MASK).all_zeros(ray_mask));
        return node.right_child;
    }
}

force_inline bool is_leaf_node(const bvh_node_t &node) {
    return node.prim_count != 0;
}

enum eTraversalSource { FromParent, FromChild, FromSibling };

struct TraversalState {
    struct {
        simd_ivec16 mask;
        uint32_t cur;
        eTraversalSource src;
    } queue[16];

    int index = 0, num = 1;

    force_inline void select_near_child(const ray_packet_t &r, const bvh_node_t &node) {
        simd_fvec16 fmask = r.d[node.space_axis] < ZERO;
        const auto &mask1 = reinterpret_cast<const simd_ivec16&>(fmask);
        if (mask1.all_zeros()) {
            queue[index].cur = node.left_child;
        } else {
            simd_ivec16 mask2 = and_not(mask1, queue[index].mask);
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
}
}

ray::NS::hit_data_t::hit_data_t() {
    mask = { 0 };
    obj_index = { -1 };
    prim_index = { -1 };
    t = { std::numeric_limits<float>::max() };
}

void ray::NS::ConstructRayPacket(const float *o, const float *d, int size, ray_packet_t &out_r) {
    assert(size <= RayPacketSize);

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
    }
}

void ray::NS::GeneratePrimaryRays(const camera_t &cam, const rect_t &r, int w, int h, math::aligned_vector<ray_packet_t> &out_rays) {
    size_t i = 0;
    out_rays.resize(r.w * r.h / RayPacketSize + ((r.w * r.h) % 4 != 0));

    simd_fvec16 ww = { (float)w }, hh = { (float)h };

    float k = float(h) / w;

    simd_fvec16 fwd[3] = { { cam.fwd[0] },  { cam.fwd[1] }, { cam.fwd[2] } },
                side[3] = { { cam.side[0] }, { cam.side[1] }, { cam.side[2] } },
                up[3] = { { cam.up[0] * k }, { cam.up[1] * k }, { cam.up[2] * k } };

    for (int y = r.y; y < r.y + r.h - (r.h & (RayPacketDimY - 1)); y += RayPacketDimY) {
        simd_fvec16 xx = { float(r.x + 0), float(r.x + 1), float(r.x + 2), float(r.x + 3),
                           float(r.x + 0), float(r.x + 1), float(r.x + 2), float(r.x + 3),
                           float(r.x + 0), float(r.x + 1), float(r.x + 2), float(r.x + 3),
                           float(r.x + 0), float(r.x + 1), float(r.x + 2), float(r.x + 3) };
        float fy = float(y);
        simd_fvec16 yy = { -fy - 0, -fy - 0, -fy - 0, -fy - 0,
                           -fy - 1, -fy - 1, -fy - 1, -fy - 1,
                           -fy - 2, -fy - 2, -fy - 2, -fy - 2,
                           -fy - 3, -fy - 3, -fy - 3, -fy - 3 };
        yy /= hh;
        yy += _0_5;

        for (int x = r.x; x < r.x + r.w - (r.w & (RayPacketDimX - 1)); x += RayPacketDimX) {
            simd_fvec16 dd[3];

            // x / w - 0.5
            dd[0] /= ww;
            dd[0] -= _0_5;

            // -y / h + 0.5
            dd[1] = yy;

            dd[2] = ONE;

            // d = d.x * side + d.y * up + d.z * fwd
            simd_fvec16 temp1 = dd[0] * side[0] + dd[1] * up[0] + dd[2] * fwd[0];
            simd_fvec16 temp2 = dd[0] * side[1] + dd[1] * up[1] + dd[2] * fwd[1];
            simd_fvec16 temp3 = dd[0] * side[2] + dd[1] * up[2] + dd[2] * fwd[2];

            dd[0] = temp1;
            dd[1] = temp2;
            dd[2] = temp3;

            simd_fvec16 inv_l = dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2];
            inv_l = sqrt(inv_l);
            inv_l = ONE / inv_l;

            assert(i < out_rays.size());
            auto &r = out_rays[i++];

            r.d[0] = dd[0] * inv_l;
            r.d[1] = dd[1] * inv_l;
            r.d[2] = dd[2] * inv_l;

            r.o[0] = { cam.origin[0] };
            r.o[1] = { cam.origin[1] };
            r.o[2] = { cam.origin[2] };

            //r.id.x = x;
            //r.id.y = y;

            xx = xx + TWO;
        }
    }
}

bool ray::NS::IntersectTris(const ray_packet_t &r, const simd_ivec16 &ray_mask, const tri_accel_t *tris, uint32_t num_tris, uint32_t obj_index, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.obj_index = { reinterpret_cast<const int&>(obj_index) };
    inter.t = out_inter.t;

    for (uint32_t i = 0; i < num_tris; i++) {
        _IntersectTri(r, ray_mask, tris[i], i, inter);
    }

    const auto &fmask = reinterpret_cast<const simd_fvec16&>(inter.mask);

    out_inter.mask = out_inter.mask | inter.mask;

    where(inter.mask, out_inter.obj_index) = inter.obj_index;
    where(inter.mask, out_inter.prim_index) = inter.prim_index;

    out_inter.t = inter.t; // already contains min value

    where(fmask, out_inter.u) = inter.u;
    where(fmask, out_inter.v) = inter.v;

    return inter.mask.not_all_zeros();
}

bool ray::NS::IntersectTris(const ray_packet_t &r, const simd_ivec16 &ray_mask, const tri_accel_t *tris, const uint32_t *indices, uint32_t num_tris, uint32_t obj_index, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.obj_index = { reinterpret_cast<const int&>(obj_index) };
    inter.t = out_inter.t;

    for (uint32_t i = 0; i < num_tris; i++) {
        uint32_t index = indices[i];
        _IntersectTri(r, ray_mask, tris[index], index, inter);
    }

    const auto &fmask = reinterpret_cast<const simd_fvec16&>(inter.mask);

    out_inter.mask = out_inter.mask | inter.mask;

    where(inter.mask, out_inter.obj_index) = inter.obj_index;
    where(inter.mask, out_inter.prim_index) = inter.prim_index;

    out_inter.t = inter.t; // already contains min value

    where(fmask, out_inter.u) = inter.u;
    where(fmask, out_inter.v) = inter.v;

    return inter.mask.not_all_zeros();
}

bool ray::NS::IntersectCones(const ray_packet_t &r, const cone_accel_t *cones, uint32_t num_cones, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.t = out_inter.t;

    for (uint32_t i = 0; i < num_cones; i++) {
        const cone_accel_t &cone = cones[i];

        simd_fvec16 cone_o[3] = { { cone.o[0] }, { cone.o[1] }, { cone.o[2] } },
                    cone_v[3] = { { cone.v[0] }, { cone.v[1] }, { cone.v[0] } },
                    cone_cos_phi_sqr = { cone.cos_phi_sqr };

        simd_fvec16 co[3] = { { r.o[0] - cone_o[0] }, { r.o[1] - cone_o[1] }, { r.o[2] - cone_o[2] } };

        // a = dot(d, cone_v)
        simd_fvec16 a = r.d[0] * cone_v[0] + r.d[1] * cone_v[1] + r.d[2] * cone_v[2];

        // c = dot(co, cone_v)
        simd_fvec16 c = co[0] * cone_v[0] + co[1] * cone_v[1] + co[2] * cone_v[2];

        // b = 2 * (a * c - dot(d, co) * cone.cos_phi_sqr)
        simd_fvec16 b = TWO * (a * c - (r.d[0] * co[0] + r.d[1] * co[1] + r.d[2] * co[2]) * cone_cos_phi_sqr);

        // a = a * a - cone.cos_phi_sqr
        a = a * a - cone_cos_phi_sqr;

        // c = c * c - dot(co, co) * cone.cos_phi_sqr
        c = c * c - (co[0] * co[0] + co[1] * co[1] + co[2] * co[2]) * cone_cos_phi_sqr;

        // D = b * b - 4 * a * c
        simd_fvec16 D = b * b - FOUR * a * c;

        const auto &m = reinterpret_cast<const simd_ivec16&>(D >= ZERO);

        if (m.all_zeros()) continue;

        D = sqrt(D);

        a = a * TWO;
        b = b ^ MINUS_ZERO; // swap sign

        simd_fvec16 t1 = (b - D) / a,
                    t2 = (b + D) / a;

        simd_fvec16 mask1 = (t1 > ZERO) & (t1 < inter.t);
        simd_fvec16 mask2 = (t2 > ZERO) & (t2 < inter.t);
        auto mask = reinterpret_cast<const simd_ivec16&>(mask1 | mask2);

        if (mask.all_zeros()) continue;

        simd_fvec16 p1c[3] = { { cone_o[0] - r.o[0] + t1 * r.d[0] },
                               { cone_o[1] - r.o[1] + t1 * r.d[1] },
                               { cone_o[2] - r.o[2] + t1 * r.d[2] } },
                    p2c[3] = { { cone_o[0] - r.o[0] + t2 * r.d[0] },
                               { cone_o[1] - r.o[1] + t2 * r.d[1] },
                               { cone_o[2] - r.o[2] + t2 * r.d[2] } };

        simd_fvec16 dot1 = p1c[0] * cone_v[0] + p1c[1] * cone_v[1] + p1c[2] * cone_v[2];
        simd_fvec16 dot2 = p2c[0] * cone_v[0] + p2c[1] * cone_v[1] + p2c[2] * cone_v[2];

        simd_fvec16 cone_start = { cone.cone_start },
                    cone_end = { cone.cone_end };

        mask1 = (dot1 >= cone_start) & (dot1 <= cone_end);
        mask2 = (dot2 >= cone_start) & (dot2 <= cone_end);

        mask = reinterpret_cast<const simd_ivec16&>(mask1 | mask2);

        if (mask.all_zeros()) continue;

        inter.mask = inter.mask | mask;

        where(mask, inter.prim_index) = { reinterpret_cast<const int&>(i) };
    }

    out_inter.mask = out_inter.mask | inter.mask;

    where(inter.mask, out_inter.prim_index) = inter.prim_index;

    return inter.mask.not_all_zeros();
}

bool ray::NS::IntersectBoxes(const ray_packet_t &r, const aabox_t *boxes, uint32_t num_boxes, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.t = out_inter.t;

    simd_fvec16 inv_d[3] = { { ONE / r.d[0] }, { ONE / r.d[1] }, { ONE / r.d[2] } };

    for (uint32_t i = 0; i < num_boxes; i++) {
        const aabox_t &box = boxes[i];

        simd_fvec16 box_min[3] = { { box.min[0] }, { box.min[1] }, { box.min[2] } },
                    box_max[3] = { { box.max[0] }, { box.max[1] }, { box.max[2] } };

        simd_fvec16 low, high, tmin, tmax;

        low = inv_d[0] * (box_min[0] - r.o[0]);
        high = inv_d[0] * (box_max[0] - r.o[0]);
        tmin = min(low, high);
        tmax = max(low, high);

        low = inv_d[1] * (box_min[1] - r.o[1]);
        high = inv_d[1] * (box_max[1] - r.o[1]);
        tmin = max(tmin, min(low, high));
        tmax = min(tmax, max(low, high));

        low = inv_d[2] * (box_min[2] - r.o[2]);
        high = inv_d[2] * (box_max[2] - r.o[2]);
        tmin = max(tmin, min(low, high));
        tmax = min(tmax, max(low, high));

        simd_fvec16 mask = (tmin <= tmax) & (tmax > ZERO) & (tmin < inter.t);

        const auto &imask = reinterpret_cast<const simd_ivec16&>(mask);
        if (imask.all_zeros()) continue;

        inter.mask = inter.mask | imask;

        where(imask, inter.prim_index) = { reinterpret_cast<const int&>(i) };
        where(mask, inter.t) = tmin;
    }

    out_inter.mask = out_inter.mask | inter.mask;

    where(inter.mask, out_inter.prim_index) = inter.prim_index;
    //out_intersections.t = intersections.t; // already contains min value

    return inter.mask.not_all_zeros();
}

bool ray::NS::Traverse_MacroTree_CPU(const ray_packet_t &r, const simd_ivec16 &ray_mask, const simd_fvec16 inv_d[3], const bvh_node_t *nodes, uint32_t root_index,
                                      const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                      const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter) {
    bool res = false;

    TraversalState st;

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

                        ray_packet_t _r = TransformRay(r, tr.inv_xform);

                        simd_fvec16 _inv_d[3] = { { ONE / _r.d[0] }, { ONE / _r.d[1] }, { ONE / _r.d[2] } };

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

                        ray_packet_t _r = TransformRay(r, tr.inv_xform);

                        simd_fvec16 _inv_d[3] = { { ONE / _r.d[0] }, { ONE / _r.d[1] }, { ONE / _r.d[2] } };

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

bool ray::NS::Traverse_MicroTree_CPU(const ray_packet_t &r, const simd_ivec16 &ray_mask, const simd_fvec16 inv_d[3], const bvh_node_t *nodes, uint32_t root_index,
                                      const tri_accel_t *tris, const uint32_t *indices, int obj_index, hit_data_t &inter) {
    bool res = false;

    TraversalState st;

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

ray::NS::ray_packet_t ray::NS::TransformRay(const ray_packet_t &r, const float *xform) {
    ray_packet_t _r = r;

    _r.o[0] = r.o[0] * xform[0] + r.o[1] * xform[4] + r.o[2] * xform[8] + xform[12];
    _r.o[1] = r.o[0] * xform[1] + r.o[1] * xform[5] + r.o[2] * xform[9] + xform[13];
    _r.o[2] = r.o[0] * xform[2] + r.o[1] * xform[6] + r.o[2] * xform[10] + xform[14];

    _r.d[0] = r.d[0] * xform[0] + r.d[1] * xform[4] + r.d[2] * xform[8];
    _r.d[1] = r.d[0] * xform[1] + r.d[1] * xform[5] + r.d[2] * xform[9];
    _r.d[2] = r.d[0] * xform[2] + r.d[1] * xform[6] + r.d[2] * xform[10];

    return _r;
}
