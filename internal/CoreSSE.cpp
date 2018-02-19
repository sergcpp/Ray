#include "CoreSSE.h"

#include <limits>

#include <math/math.hpp>

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target ("sse3")
#endif

namespace ray {
namespace sse {
const __m128 _0_5 = _mm_set1_ps(0.5f);

const __m128 ZERO = _mm_set1_ps(0);
const __m128 MINUS_ZERO = _mm_set1_ps(-0.0f);

const __m128 ONE = _mm_set1_ps(1);
const __m128 TWO = _mm_set1_ps(2);
const __m128 FOUR = _mm_set1_ps(4);

const __m128 EPS = _mm_set1_ps(std::numeric_limits<float>::epsilon());
const __m128 M_EPS = _mm_set1_ps(-std::numeric_limits<float>::epsilon());

const __m128i FF_MASK = _mm_set1_epi32(0xffffffff);

force_inline void _IntersectTri(const ray_packet_t &r, const __m128i ray_mask, const tri_accel_t &tri, uint32_t prim_index, hit_data_t &inter) {
    __m128 nu = _mm_set1_ps(tri.nu), nv = _mm_set1_ps(tri.nv), np = _mm_set1_ps(tri.np);
    __m128 pu = _mm_set1_ps(tri.pu), pv = _mm_set1_ps(tri.pv);

    __m128 e0u = _mm_set1_ps(tri.e0u), e0v = _mm_set1_ps(tri.e0v);
    __m128 e1u = _mm_set1_ps(tri.e1u), e1v = _mm_set1_ps(tri.e1v);

    const int _next_u[] = { 1, 0, 0 },
                          _next_v[] = { 2, 2, 1 };

    int w = (tri.ci & W_BITS),
        u = _next_u[w],
        v = _next_v[w];

    // from "Ray-Triangle Intersection Algorithm for Modern CPU Architectures" [2007]

    //temporary variables
    __m128 det, dett, detu, detv, nrv, nru, du, dv, ou, ov, tmpdet0;//, tmpdet1;

    // ----ray-packet/triangle hit test----
    //dett = np -(ou*nu+ov*nv+ow)
    dett = np;
    dett = _mm_sub_ps(dett, r.o[w]);

    du = nu;
    dv = nv;

    ou = r.o[u];
    ov = r.o[v];

    du = _mm_mul_ps(du, ou);
    dv = _mm_mul_ps(dv, ov);
    dett = _mm_sub_ps(dett, du);
    dett = _mm_sub_ps(dett, dv);
    //det = du * nu + dv * nv + dw

    du = r.d[u];
    dv = r.d[v];

    ou = _mm_sub_ps(pu, ou);
    ov = _mm_sub_ps(pv, ov);

    det = nu;
    det = _mm_mul_ps(det, du);
    nrv = nv;
    nrv = _mm_mul_ps(nrv, dv);

    det = _mm_add_ps(det, r.d[w]);

    det = _mm_add_ps(det, nrv);
    //Du = du*dett - (pu-ou)*det
    nru = _mm_mul_ps(ou, det);

    du = _mm_mul_ps(du, dett);
    du = _mm_sub_ps(du, nru);
    //Dv = dv*dett - (pv-ov)*det
    nrv = _mm_mul_ps(ov, det);

    dv = _mm_mul_ps(dv, dett);
    dv = _mm_sub_ps(dv, nrv);
    //detu = (e1vDu ? e1u*Dv)
    nru = e1v;
    nrv = e1u;
    nru = _mm_mul_ps(nru, du);
    nrv = _mm_mul_ps(nrv, dv);
    detu = _mm_sub_ps(nru, nrv);
    //detv = (e0u*Dv ? e0v*Du)
    nrv = e0u;
    nrv = _mm_mul_ps(nrv, dv);
    dv = e0v;
    dv = _mm_mul_ps(dv, du);
    detv = _mm_sub_ps(nrv, dv);

    // same sign of 'det - detu - detv', 'detu', 'detv' indicates intersection

    tmpdet0 = _mm_sub_ps(_mm_sub_ps(det, detu), detv);
    /*tmpdet0 = _mm_xor_ps(tmpdet0, detu);
    tmpdet1 = _mm_xor_ps(detv, detu);
    //
    __m128 tmp = _mm_xor_ps(dett, det); // make sure t is positive
    //
    tmpdet1 = _mm_or_ps(tmpdet0, tmpdet1);
    tmpdet1 = _mm_or_ps(tmpdet1, tmp);
    // test
    __m128i mask = _mm_castps_si128(tmpdet1);
    mask = _mm_srai_epi32(mask, 31);
    mask = _mm_xor_si128(mask, FF_MASK);
    mask = _mm_and_si128(mask, ray_mask);*/

    //////////////////////////////////////////////////////////////////////////

    __m128 mm1 = _mm_mul_ps(tmpdet0, detu);
    __m128 mm = _mm_cmpgt_ps(mm1, M_EPS);
    __m128 mm2 = _mm_mul_ps(detu, detv);
    mm = _mm_and_ps(mm, _mm_cmpgt_ps(mm2, ZERO));
    __m128 mm3 = _mm_mul_ps(dett, det);
    mm = _mm_and_ps(mm, _mm_cmpgt_ps(mm3, ZERO));

    __m128i mask = _mm_castps_si128(mm);
    mask = _mm_and_si128(mask, ray_mask);

    //////////////////////////////////////////////////////////////////////////

    if (_mm_movemask_epi8(mask) == 0) return; // no intersection found
    //if (_mm_test_all_zeros(mask, FF_MASK)) continue; // no intersection found

    __m128 rdet = _mm_rcp_ps(det);  // 1 / det
    __m128 t = _mm_mul_ps(dett, rdet);

    __m128i t_valid = _mm_castps_si128(_mm_cmplt_ps(t, inter.t));
    if (_mm_test_all_zeros(t_valid, mask)) return; // all intersections further than needed

    __m128 bar_u = _mm_mul_ps(detu, rdet);
    __m128 bar_v = _mm_mul_ps(detv, rdet);

    mask = _mm_and_si128(mask, t_valid);
    __m128 mask_ps = _mm_castsi128_ps(mask);

    inter.mask = _mm_or_si128(inter.mask, mask);
    inter.prim_index = _mm_blendv_epi8(inter.prim_index, _mm_set1_epi32(prim_index), mask);
    inter.t = _mm_blendv_ps(inter.t, t, mask_ps);
    inter.u = _mm_blendv_ps(inter.u, bar_u, mask_ps);
    inter.v = _mm_blendv_ps(inter.v, bar_v, mask_ps);
}

__m128i bbox_test(const __m128 o[3], const __m128 inv_d[3], const __m128 t, const float _bbox_min[3], const float _bbox_max[3]) {
    __m128 box_min[3] = { _mm_set1_ps(_bbox_min[0]), _mm_set1_ps(_bbox_min[1]), _mm_set1_ps(_bbox_min[2]) },
                        box_max[3] = { _mm_set1_ps(_bbox_max[0]), _mm_set1_ps(_bbox_max[1]), _mm_set1_ps(_bbox_max[2]) };

    __m128 low, high, tmin, tmax;

    low = _mm_mul_ps(inv_d[0], _mm_sub_ps(box_min[0], o[0]));
    high = _mm_mul_ps(inv_d[0], _mm_sub_ps(box_max[0], o[0]));
    tmin = _mm_min_ps(low, high);
    tmax = _mm_max_ps(low, high);

    low = _mm_mul_ps(inv_d[1], _mm_sub_ps(box_min[1], o[1]));
    high = _mm_mul_ps(inv_d[1], _mm_sub_ps(box_max[1], o[1]));
    tmin = _mm_max_ps(tmin, _mm_min_ps(low, high));
    tmax = _mm_min_ps(tmax, _mm_max_ps(low, high));

    low = _mm_mul_ps(inv_d[2], _mm_sub_ps(box_min[2], o[2]));
    high = _mm_mul_ps(inv_d[2], _mm_sub_ps(box_max[2], o[2]));
    tmin = _mm_max_ps(tmin, _mm_min_ps(low, high));
    tmax = _mm_min_ps(tmax, _mm_max_ps(low, high));

    __m128 mask = _mm_cmple_ps(tmin, tmax);
    mask = _mm_and_ps(mask, _mm_cmple_ps(tmin, t));
    mask = _mm_and_ps(mask, _mm_cmpgt_ps(tmax, ZERO));

    return _mm_castps_si128(mask);
}

force_inline __m128i bbox_test(const __m128 o[3], const __m128 inv_d[3], const __m128 t, const bvh_node_t &node) {
    return bbox_test(o, inv_d, t, node.bbox[0], node.bbox[1]);
}

force_inline uint32_t near_child(const ray_packet_t &r, const __m128i ray_mask, const bvh_node_t &node) {
    __m128i mask = _mm_castps_si128(_mm_cmplt_ps(r.d[node.space_axis], ZERO));
    if (_mm_test_all_zeros(mask, ray_mask)) {
        return node.left_child;
    } else {
        assert(_mm_test_all_zeros(_mm_andnot_si128(mask, FF_MASK), ray_mask));
        return node.right_child;
    }
}

force_inline bool is_leaf_node(const bvh_node_t &node) {
    return node.prim_count != 0;
}

enum eTraversalSource { FromParent, FromChild, FromSibling };

struct TraversalState {
    struct {
        __m128i mask;
        uint32_t cur;
        eTraversalSource src;
    } queue[4];

    int index = 0, num = 1;

    force_inline void select_near_child(const ray_packet_t &r, const bvh_node_t &node) {
        __m128i mask1 = _mm_castps_si128(_mm_cmplt_ps(r.d[node.space_axis], ZERO));
        if (_mm_test_all_zeros(mask1, queue[index].mask)) {
            queue[index].cur = node.left_child;
        } else {
            __m128i mask2 = _mm_andnot_si128(mask1, FF_MASK);
            if (_mm_test_all_zeros(mask2, queue[index].mask)) {
                queue[index].cur = node.right_child;
            } else {
                queue[num].cur = node.left_child;
                queue[num].mask = _mm_and_si128(queue[index].mask, mask2);
                queue[num].src = queue[index].src;
                num++;
                queue[index].cur = node.right_child;
                queue[index].mask = _mm_and_si128(queue[index].mask, mask1);
            }
        }
    }
};
}
}

ray::sse::hit_data_t::hit_data_t() {
    mask = _mm_set1_epi32(0);
    obj_index = _mm_set1_epi32(-1);
    prim_index = _mm_set1_epi32(-1);
    t = _mm_set1_ps(std::numeric_limits<float>::max());
}

void ray::sse::ConstructRayPacket(const float *o, const float *d, int size, ray_packet_t &out_r) {
    assert(size <= RayPacketSize);

    // ray_packet_t
    // x0 x1 x2 x3
    // y0 y1 y2 y3
    // z0 z1 z2 z3

    if (size == 4) {
        out_r.o[0] = _mm_setr_ps(o[9], o[6], o[3], o[0]);
        out_r.o[1] = _mm_setr_ps(o[10], o[7], o[4], o[1]);
        out_r.o[2] = _mm_setr_ps(o[11], o[8], o[5], o[2]);
        out_r.d[0] = _mm_setr_ps(d[9], d[6], d[3], d[0]);
        out_r.d[1] = _mm_setr_ps(d[10], d[7], d[4], d[1]);
        out_r.d[2] = _mm_setr_ps(d[11], d[8], d[5], d[2]);
    } else if (size == 3) {
        out_r.o[0] = _mm_setr_ps(0, o[6], o[3], o[0]);
        out_r.o[1] = _mm_setr_ps(0, o[7], o[4], o[1]);
        out_r.o[2] = _mm_setr_ps(0, o[8], o[5], o[2]);
        out_r.d[0] = _mm_setr_ps(0, d[6], d[3], d[0]);
        out_r.d[1] = _mm_setr_ps(0, d[7], d[4], d[1]);
        out_r.d[2] = _mm_setr_ps(0, d[8], d[5], d[2]);
    } else if (size == 2) {
        out_r.o[0] = _mm_setr_ps(0, 0, o[3], o[0]);
        out_r.o[1] = _mm_setr_ps(0, 0, o[4], o[1]);
        out_r.o[2] = _mm_setr_ps(0, 0, o[5], o[2]);
        out_r.d[0] = _mm_setr_ps(0, 0, d[3], d[0]);
        out_r.d[1] = _mm_setr_ps(0, 0, d[4], d[1]);
        out_r.d[2] = _mm_setr_ps(0, 0, d[5], d[2]);
    } else if (size == 1) {
        out_r.o[0] = _mm_setr_ps(0, 0, 0, o[0]);
        out_r.o[1] = _mm_setr_ps(0, 0, 0, o[1]);
        out_r.o[2] = _mm_setr_ps(0, 0, 0, o[2]);
        out_r.d[0] = _mm_setr_ps(0, 0, 0, d[0]);
        out_r.d[1] = _mm_setr_ps(0, 0, 0, d[1]);
        out_r.d[2] = _mm_setr_ps(0, 0, 0, d[2]);
    }
}

void ray::sse::GeneratePrimaryRays(const camera_t &cam, const region_t &r, int w, int h, math::aligned_vector<ray_packet_t> &out_rays) {
    size_t i = 0;
    out_rays.resize(r.w * r.h / 4 + ((r.w * r.h) % 4 != 0));

    __m128 ww = _mm_set1_ps((float)w), hh = _mm_set1_ps((float)h);

    float k = float(h) / w;

    __m128 fwd[3] = { _mm_set1_ps(cam.fwd[0]), _mm_set1_ps(cam.fwd[1]), _mm_set1_ps(cam.fwd[2]) },
                    side[3] = { _mm_set1_ps(cam.side[0]), _mm_set1_ps(cam.side[1]), _mm_set1_ps(cam.side[2]) },
                              up[3] = { _mm_set1_ps(cam.up[0] * k), _mm_set1_ps(cam.up[1] * k), _mm_set1_ps(cam.up[2] * k) };


    for (int y = r.y; y < r.y + r.h - (r.h & (RayPacketDimY - 1)); y += RayPacketDimY) {
        __m128 xx = _mm_setr_ps(float(r.x), float(r.x + 1), float(r.x), float(r.x + 1));
        float fy = float(y);
        __m128 yy = _mm_setr_ps(-fy, -fy, -fy - 1, -fy - 1);
        yy = _mm_div_ps(yy, hh);
        yy = _mm_add_ps(yy, _0_5);

        for (int x = r.x; x < r.x + r.w - (r.w & (RayPacketDimX - 1)); x += RayPacketDimX) {
            __m128 dd[3];

            // x / w - 0.5
            dd[0] = _mm_div_ps(xx, ww);
            dd[0] = _mm_sub_ps(dd[0], _0_5);

            // -y / h + 0.5
            dd[1] = yy;

            dd[2] = ONE;

            // d = d.x * side + d.y * up + d.z * fwd
            __m128 temp1 = _mm_mul_ps(dd[0], side[0]);
            temp1 = _mm_add_ps(temp1, _mm_mul_ps(dd[1], up[0]));
            temp1 = _mm_add_ps(temp1, _mm_mul_ps(dd[2], fwd[0]));

            __m128 temp2 = _mm_mul_ps(dd[0], side[1]);
            temp2 = _mm_add_ps(temp2, _mm_mul_ps(dd[1], up[1]));
            temp2 = _mm_add_ps(temp2, _mm_mul_ps(dd[2], fwd[1]));

            __m128 temp3 = _mm_mul_ps(dd[0], side[2]);
            temp3 = _mm_add_ps(temp3, _mm_mul_ps(dd[1], up[2]));
            temp3 = _mm_add_ps(temp3, _mm_mul_ps(dd[2], fwd[2]));

            dd[0] = temp1;
            dd[1] = temp2;
            dd[2] = temp3;

            __m128 inv_l = _mm_mul_ps(dd[0], dd[0]);
            inv_l = _mm_add_ps(inv_l, _mm_mul_ps(dd[1], dd[1]));
            inv_l = _mm_add_ps(inv_l, _mm_mul_ps(dd[2], dd[2]));
            //inv_l = _mm_rsqrt_ps(inv_l);
            inv_l = _mm_sqrt_ps(inv_l);
            inv_l = _mm_div_ps(ONE, inv_l);

            assert(i < out_rays.size());
            auto &r = out_rays[i++];

            r.d[0] = _mm_mul_ps(dd[0], inv_l);
            r.d[1] = _mm_mul_ps(dd[1], inv_l);
            r.d[2] = _mm_mul_ps(dd[2], inv_l);

            r.o[0] = _mm_set1_ps(cam.origin[0]);
            r.o[1] = _mm_set1_ps(cam.origin[1]);
            r.o[2] = _mm_set1_ps(cam.origin[2]);

            r.id.x = x;
            r.id.y = y;

            xx = _mm_add_ps(xx, TWO);
        }
    }
}

bool ray::sse::IntersectTris(const ray_packet_t &r, const __m128i ray_mask, const tri_accel_t *tris, int num_tris, int obj_index, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.obj_index = _mm_set1_epi32(obj_index);
    inter.t = out_inter.t;

    for (int i = 0; i < num_tris; i++) {
        const tri_accel_t &tri = tris[i];

        _IntersectTri(r, ray_mask, tri, i, inter);
    }

    out_inter.mask = _mm_or_si128(out_inter.mask, inter.mask);
    out_inter.obj_index = _mm_blendv_epi8(out_inter.obj_index, inter.obj_index, inter.mask);
    out_inter.prim_index = _mm_blendv_epi8(out_inter.prim_index, inter.prim_index, inter.mask);
    out_inter.t = inter.t; // already contains min value

    __m128 mask_ps = _mm_castsi128_ps(inter.mask);
    out_inter.u = _mm_blendv_ps(out_inter.u, inter.u, mask_ps);
    out_inter.v = _mm_blendv_ps(out_inter.v, inter.v, mask_ps);

    return !_mm_test_all_zeros(inter.mask, FF_MASK);
}

bool ray::sse::IntersectTris(const ray_packet_t &r, const __m128i ray_mask, const tri_accel_t *tris, const uint32_t *indices, int num_tris, int obj_index, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.obj_index = _mm_set1_epi32(obj_index);
    inter.t = out_inter.t;

    for (int i = 0; i < num_tris; i++) {
        const tri_accel_t &tri = tris[indices[i]];

        _IntersectTri(r, ray_mask, tri, indices[i], inter);
    }

    out_inter.mask = _mm_or_si128(out_inter.mask, inter.mask);
    out_inter.obj_index = _mm_blendv_epi8(out_inter.obj_index, inter.obj_index, inter.mask);
    out_inter.prim_index = _mm_blendv_epi8(out_inter.prim_index, inter.prim_index, inter.mask);
    out_inter.t = inter.t; // already contains min value

    __m128 mask_ps = _mm_castsi128_ps(inter.mask);
    out_inter.u = _mm_blendv_ps(out_inter.u, inter.u, mask_ps);
    out_inter.v = _mm_blendv_ps(out_inter.v, inter.v, mask_ps);

    return !_mm_test_all_zeros(inter.mask, FF_MASK);
}

bool ray::sse::IntersectCones(const ray_packet_t &r, const cone_accel_t *cones, int num_cones, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.t = out_inter.t;

    for (int i = 0; i < num_cones; i++) {
        const cone_accel_t &cone = cones[i];

        __m128 cone_o[3], cone_v[3], cone_cos_phi_sqr;
        cone_o[0] = _mm_set1_ps(cone.o[0]);
        cone_o[1] = _mm_set1_ps(cone.o[1]);
        cone_o[2] = _mm_set1_ps(cone.o[2]);

        cone_v[0] = _mm_set1_ps(cone.v[0]);
        cone_v[1] = _mm_set1_ps(cone.v[1]);
        cone_v[2] = _mm_set1_ps(cone.v[2]);

        cone_cos_phi_sqr = _mm_set1_ps(cone.cos_phi_sqr);

        __m128 co[3];
        co[0] = _mm_sub_ps(r.o[0], cone_o[0]);
        co[1] = _mm_sub_ps(r.o[1], cone_o[1]);
        co[2] = _mm_sub_ps(r.o[2], cone_o[2]);

        // a = dot(d, cone_v)
        __m128 a = _mm_mul_ps(r.d[0], cone_v[0]);
        a = _mm_add_ps(a, _mm_mul_ps(r.d[1], cone_v[1]));
        a = _mm_add_ps(a, _mm_mul_ps(r.d[2], cone_v[2]));

        // c = dot(co, cone_v)
        __m128 c = _mm_mul_ps(co[0], cone_v[0]);
        c = _mm_add_ps(c, _mm_mul_ps(co[1], cone_v[1]));
        c = _mm_add_ps(c, _mm_mul_ps(co[2], cone_v[2]));

        // b = 2 * (a * c - dot(d, co) * cone.cos_phi_sqr)
        __m128 b = _mm_mul_ps(r.d[0], co[0]);
        b = _mm_add_ps(b, _mm_mul_ps(r.d[1], co[1]));
        b = _mm_add_ps(b, _mm_mul_ps(r.d[2], co[2]));
        b = _mm_mul_ps(b, cone_cos_phi_sqr);
        b = _mm_sub_ps(_mm_mul_ps(a, c), b);
        b = _mm_mul_ps(b, TWO);

        // a = a * a - cone.cos_phi_sqr
        a = _mm_mul_ps(a, a);
        a = _mm_sub_ps(a, cone_cos_phi_sqr);

        // c = c * c - dot(co, co) * cone.cos_phi_sqr
        c = _mm_mul_ps(c, c);

        __m128 temp = _mm_mul_ps(co[0], co[0]);
        temp = _mm_add_ps(temp, _mm_mul_ps(co[1], co[1]));
        temp = _mm_add_ps(temp, _mm_mul_ps(co[2], co[2]));

        temp = _mm_mul_ps(temp, cone_cos_phi_sqr);

        c = _mm_sub_ps(c, temp);

        // D = b * b - 4 * a * c
        __m128 D = _mm_mul_ps(b, b);

        temp = _mm_mul_ps(FOUR, a);
        temp = _mm_mul_ps(temp, c);

        D = _mm_sub_ps(D, temp);

        __m128i m = _mm_castps_si128(_mm_cmpge_ps(D, ZERO));

        if (_mm_test_all_zeros(m, FF_MASK)) continue;

        D = _mm_sqrt_ps(D);

        a = _mm_mul_ps(a, TWO);
        b = _mm_xor_ps(b, MINUS_ZERO); // swap sign

        __m128 t1, t2;
        t1 = _mm_sub_ps(b, D);
        t1 = _mm_div_ps(t1, a);
        t2 = _mm_add_ps(b, D);
        t2 = _mm_div_ps(t2, a);

        __m128 mask1 = _mm_cmpgt_ps(t1, ZERO);
        mask1 = _mm_and_ps(mask1, _mm_cmplt_ps(t1, inter.t));
        __m128 mask2 = _mm_cmpgt_ps(t2, ZERO);
        mask2 = _mm_and_ps(mask2, _mm_cmplt_ps(t2, inter.t));
        __m128i mask = _mm_castps_si128(_mm_or_ps(mask1, mask2));

        if (_mm_test_all_zeros(mask, FF_MASK)) continue;

        __m128 p1c[3], p2c[3];
        p1c[0] = _mm_add_ps(r.o[0], _mm_mul_ps(t1, r.d[0]));
        p1c[1] = _mm_add_ps(r.o[1], _mm_mul_ps(t1, r.d[1]));
        p1c[2] = _mm_add_ps(r.o[2], _mm_mul_ps(t1, r.d[2]));

        p2c[0] = _mm_add_ps(r.o[0], _mm_mul_ps(t2, r.d[0]));
        p2c[1] = _mm_add_ps(r.o[1], _mm_mul_ps(t2, r.d[1]));
        p2c[2] = _mm_add_ps(r.o[2], _mm_mul_ps(t2, r.d[2]));

        p1c[0] = _mm_sub_ps(cone_o[0], p1c[0]);
        p1c[1] = _mm_sub_ps(cone_o[1], p1c[1]);
        p1c[2] = _mm_sub_ps(cone_o[2], p1c[2]);

        p2c[0] = _mm_sub_ps(cone_o[0], p2c[0]);
        p2c[1] = _mm_sub_ps(cone_o[1], p2c[1]);
        p2c[2] = _mm_sub_ps(cone_o[2], p2c[2]);

        __m128 dot1 = _mm_mul_ps(p1c[0], cone_v[0]);
        dot1 = _mm_add_ps(dot1, _mm_mul_ps(p1c[1], cone_v[1]));
        dot1 = _mm_add_ps(dot1, _mm_mul_ps(p1c[2], cone_v[2]));

        __m128 dot2 = _mm_mul_ps(p2c[0], cone_v[0]);
        dot2 = _mm_add_ps(dot2, _mm_mul_ps(p2c[1], cone_v[1]));
        dot2 = _mm_add_ps(dot2, _mm_mul_ps(p2c[2], cone_v[2]));

        __m128 cone_start = _mm_set1_ps(cone.cone_start),
               cone_end = _mm_set1_ps(cone.cone_end);

        mask1 = _mm_cmpge_ps(dot1, cone_start);
        mask2 = _mm_cmpge_ps(dot2, cone_start);
        mask1 = _mm_and_ps(mask1, _mm_cmple_ps(dot1, cone_end));
        mask2 = _mm_and_ps(mask2, _mm_cmple_ps(dot2, cone_end));
        mask = _mm_castps_si128(_mm_or_ps(mask1, mask2));

        if (_mm_test_all_zeros(mask, FF_MASK)) continue;

        inter.mask = _mm_or_si128(inter.mask, mask);
        inter.prim_index = _mm_blendv_epi8(inter.prim_index, _mm_set1_epi32(i), mask);
        //intersections.t = _mm_or_ps(_mm_andnot_ps(mask_ps, intersections.t1), _mm_and_ps(mask_ps, t1));
    }

    out_inter.mask = _mm_or_si128(out_inter.mask, inter.mask);
    out_inter.prim_index = _mm_blendv_epi8(out_inter.prim_index, inter.prim_index, inter.mask);
    //out_intersections.t = intersections.t; // already contains min value

    return !_mm_test_all_zeros(inter.mask, FF_MASK);
}

bool ray::sse::IntersectBoxes(const ray_packet_t &r, const aabox_t *boxes, int num_boxes, hit_data_t &out_inter) {

    hit_data_t inter;
    inter.t = out_inter.t;

    __m128 inv_d[3] = { _mm_rcp_ps(r.d[0]), _mm_rcp_ps(r.d[1]), _mm_rcp_ps(r.d[2]) };

    for (int i = 0; i < num_boxes; i++) {
        const aabox_t &box = boxes[i];

        __m128 box_min[3] = { _mm_set1_ps(box.min[0]), _mm_set1_ps(box.min[1]), _mm_set1_ps(box.min[2]) },
                            box_max[3] = { _mm_set1_ps(box.max[0]), _mm_set1_ps(box.max[1]), _mm_set1_ps(box.max[2]) };

        __m128 low, high, tmin, tmax;

        low = _mm_mul_ps(inv_d[0], _mm_sub_ps(box_min[0], r.o[0]));
        high = _mm_mul_ps(inv_d[0], _mm_sub_ps(box_max[0], r.o[0]));
        tmin = _mm_min_ps(low, high);
        tmax = _mm_max_ps(low, high);

        low = _mm_mul_ps(inv_d[1], _mm_sub_ps(box_min[1], r.o[1]));
        high = _mm_mul_ps(inv_d[1], _mm_sub_ps(box_max[1], r.o[1]));
        tmin = _mm_max_ps(tmin, _mm_min_ps(low, high));
        tmax = _mm_min_ps(tmax, _mm_max_ps(low, high));

        low = _mm_mul_ps(inv_d[2], _mm_sub_ps(box_min[2], r.o[2]));
        high = _mm_mul_ps(inv_d[2], _mm_sub_ps(box_max[2], r.o[2]));
        tmin = _mm_max_ps(tmin, _mm_min_ps(low, high));
        tmax = _mm_min_ps(tmax, _mm_max_ps(low, high));

        __m128 mask = _mm_cmple_ps(tmin, tmax);
        mask = _mm_and_ps(mask, _mm_cmpgt_ps(tmax, ZERO));
        mask = _mm_and_ps(mask, _mm_cmplt_ps(tmin, inter.t));

        __m128i imask = _mm_castps_si128(mask);
        if (_mm_test_all_zeros(imask, FF_MASK)) continue;

        inter.mask = _mm_or_si128(inter.mask, imask);
        inter.prim_index = _mm_blendv_epi8(inter.prim_index, _mm_set1_epi32(i), imask);
        inter.t = _mm_or_ps(_mm_andnot_ps(mask, inter.t), _mm_and_ps(mask, tmin));
    }

    out_inter.mask = _mm_or_si128(out_inter.mask, inter.mask);
    out_inter.prim_index = _mm_blendv_epi8(out_inter.prim_index, inter.prim_index, inter.mask);
    //out_intersections.t = intersections.t; // already contains min value

    return !_mm_test_all_zeros(inter.mask, FF_MASK);
}

bool ray::sse::Traverse_MacroTree_CPU(const ray_packet_t &r, const __m128 inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                                      const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                      const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter) {
    bool res = false;

    TraversalState st;

    st.queue[0].mask = FF_MASK;

    st.queue[0].src = FromSibling;
    st.queue[0].cur = node_index;

    if (!is_leaf_node(nodes[node_index])) {
        st.queue[0].src = FromParent;
        st.select_near_child(r, nodes[node_index]);
    }

    while (st.index < st.num) {
        uint32_t &cur = st.queue[st.index].cur;
        eTraversalSource &state = st.queue[st.index].src;

        switch (state) {
        case FromChild:
            if (cur == node_index || cur == 0xffffffff) {
                st.index++;
                continue;
            }
            if (cur == near_child(r, st.queue[st.index].mask, nodes[nodes[cur].parent])) {
                cur = nodes[cur].sibling;
                state = FromSibling;
            } else {
                cur = nodes[cur].parent;
                state = FromChild;
            }
            break;
        case FromSibling: {
            __m128i mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]);
            if (_mm_test_all_zeros(mask1, st.queue[st.index].mask)) {
                cur = nodes[cur].parent;
                state = FromChild;
            } else {
                __m128i mask2 = _mm_andnot_si128(mask1, FF_MASK);
                if (!_mm_test_all_zeros(mask2, st.queue[st.index].mask)) {
                    st.queue[st.num].cur = nodes[cur].parent;
                    st.queue[st.num].mask = _mm_and_si128(st.queue[st.index].mask, mask2);
                    st.queue[st.num].src = FromChild;
                    st.num++;
                    st.queue[st.index].mask = _mm_and_si128(st.queue[st.index].mask, mask1);
                }

                if (is_leaf_node(nodes[cur])) {
                    // process leaf
                    for (uint32_t i = nodes[cur].prim_index; i < nodes[cur].prim_index + nodes[cur].prim_count; i++) {
                        const auto &mi = mesh_instances[mi_indices[i]];
                        const auto &m = meshes[mi.mesh_index];
                        const auto &tr = transforms[mi.tr_index];

                        auto bbox_mask = bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max);
                        if (_mm_test_all_zeros(bbox_mask, FF_MASK)) continue;

                        ray_packet_t _r = TransformRay(r, tr.inv_xform);

                        __m128 _inv_d[3] = { _mm_rcp_ps(_r.d[0]), _mm_rcp_ps(_r.d[1]), _mm_rcp_ps(_r.d[2]) };

                        if (Traverse_MicroTree_CPU(_r, _inv_d, nodes, m.node_index, tris, tri_indices, (int)mi_indices[i], inter)) {
                            res = true;
                        }
                    }

                    cur = nodes[cur].parent;
                    state = FromChild;
                } else {
                    state = FromParent;
                    st.select_near_child(r, nodes[cur]);
                }
            }
        }
        break;
        case FromParent: {
            __m128i mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]);
            if (_mm_test_all_zeros(mask1, st.queue[st.index].mask)) {
                cur = nodes[cur].sibling;
                state = FromSibling;
            } else {
                __m128i mask2 = _mm_andnot_si128(mask1, FF_MASK);
                if (!_mm_test_all_zeros(mask2, st.queue[st.index].mask)) {
                    st.queue[st.num].cur = nodes[cur].sibling;
                    st.queue[st.num].mask = _mm_and_si128(st.queue[st.index].mask, mask2);
                    st.queue[st.num].src = FromSibling;
                    st.num++;
                    st.queue[st.index].mask = _mm_and_si128(st.queue[st.index].mask, mask1);
                }

                if (is_leaf_node(nodes[cur])) {
                    // process leaf
                    for (uint32_t i = nodes[cur].prim_index; i < nodes[cur].prim_index + nodes[cur].prim_count; i++) {
                        const auto &mi = mesh_instances[mi_indices[i]];
                        const auto &m = meshes[mi.mesh_index];
                        const auto &tr = transforms[mi.tr_index];

                        auto bbox_mask = bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max);
                        if (_mm_test_all_zeros(bbox_mask, FF_MASK)) continue;

                        ray_packet_t _r = TransformRay(r, tr.inv_xform);

                        __m128 _inv_d[3] = { _mm_rcp_ps(_r.d[0]), _mm_rcp_ps(_r.d[1]), _mm_rcp_ps(_r.d[2]) };

                        if (Traverse_MicroTree_CPU(_r, _inv_d, nodes, m.node_index, tris, tri_indices, (int)mi_indices[i], inter)) {
                            res = true;
                        }
                    }

                    cur = nodes[cur].sibling;
                    state = FromSibling;
                } else {
                    state = FromParent;
                    st.select_near_child(r, nodes[cur]);
                }
            }
        }
        break;
        }
    }
    return res;
}

bool ray::sse::Traverse_MicroTree_CPU(const ray_packet_t &r, const __m128 inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                                      const tri_accel_t *tris, const uint32_t *indices, int obj_index, hit_data_t &inter) {
    bool res = false;

    TraversalState st;

    st.queue[0].mask = FF_MASK;

    st.queue[0].src = FromSibling;
    st.queue[0].cur = node_index;

    if (!is_leaf_node(nodes[node_index])) {
        st.queue[0].src = FromParent;
        st.select_near_child(r, nodes[node_index]);
    }

    while (st.index < st.num) {
        uint32_t &cur = st.queue[st.index].cur;
        eTraversalSource &state = st.queue[st.index].src;

        switch (state) {
        case FromChild:
            if (cur == node_index || cur == 0xffffffff) {
                st.index++;
                continue;
            }
            if (cur == near_child(r, st.queue[st.index].mask, nodes[nodes[cur].parent])) {
                cur = nodes[cur].sibling;
                state = FromSibling;
            } else {
                cur = nodes[cur].parent;
                state = FromChild;
            }
            break;
        case FromSibling: {
            __m128i mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]);
            if (_mm_test_all_zeros(mask1, st.queue[st.index].mask)) {
                cur = nodes[cur].parent;
                state = FromChild;
            } else {
                __m128i mask2 = _mm_andnot_si128(mask1, FF_MASK);
                if (!_mm_test_all_zeros(mask2, st.queue[st.index].mask)) {
                    st.queue[st.num].cur = nodes[cur].parent;
                    st.queue[st.num].mask = _mm_and_si128(st.queue[st.index].mask, mask2);
                    st.queue[st.num].src = FromChild;
                    st.num++;
                    st.queue[st.index].mask = _mm_and_si128(st.queue[st.index].mask, mask1);
                }

                if (is_leaf_node(nodes[cur])) {
                    // process leaf
                    if (IntersectTris(r, st.queue[st.index].mask, tris, &indices[nodes[cur].prim_index], nodes[cur].prim_count, obj_index, inter)) {
                        res = true;
                    }
                    cur = nodes[cur].parent;
                    state = FromChild;
                } else {
                    state = FromParent;
                    st.select_near_child(r, nodes[cur]);
                }
            }
        }
        break;
        case FromParent: {
            __m128i mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]);
            if (_mm_test_all_zeros(mask1, st.queue[st.index].mask)) {
                cur = nodes[cur].sibling;
                state = FromSibling;
            } else {
                __m128i mask2 = _mm_andnot_si128(mask1, FF_MASK);
                if (!_mm_test_all_zeros(mask2, st.queue[st.index].mask)) {
                    st.queue[st.num].cur = nodes[cur].sibling;
                    st.queue[st.num].mask = _mm_and_si128(st.queue[st.index].mask, mask2);
                    st.queue[st.num].src = FromSibling;
                    st.num++;
                    st.queue[st.index].mask = _mm_and_si128(st.queue[st.index].mask, mask1);
                }

                if (is_leaf_node(nodes[cur])) {
                    // process leaf
                    if (IntersectTris(r, st.queue[st.index].mask, tris, &indices[nodes[cur].prim_index], nodes[cur].prim_count, obj_index, inter)) {
                        res = true;

                        //intersections.index = _mm_or_si128(_mm_andnot_si128(intersections.mask, intersections.index),
                        //	_mm_and_si128(intersections.mask, _mm_set1_epi32(st.num_rays)));
                    }
                    cur = nodes[cur].sibling;
                    state = FromSibling;
                } else {
                    state = FromParent;
                    st.select_near_child(r, nodes[cur]);
                }
            }
        }
        break;
        }
    }
    return res;
}

ray::sse::ray_packet_t ray::sse::TransformRay(const ray_packet_t &r, const float *xform) {
    ray_packet_t _r = r;

    _r.o[0] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[0]), r.o[0]), _mm_mul_ps(_mm_set1_ps(xform[4]), r.o[1])), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[8]), r.o[2]), _mm_set1_ps(xform[12])));
    _r.o[1] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[1]), r.o[0]), _mm_mul_ps(_mm_set1_ps(xform[5]), r.o[1])), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[9]), r.o[2]), _mm_set1_ps(xform[13])));
    _r.o[2] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[2]), r.o[0]), _mm_mul_ps(_mm_set1_ps(xform[6]), r.o[1])), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[10]), r.o[2]), _mm_set1_ps(xform[14])));

    _r.d[0] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[0]), r.d[0]), _mm_mul_ps(_mm_set1_ps(xform[4]), r.d[1])), _mm_mul_ps(_mm_set1_ps(xform[8]), r.d[2]));
    _r.d[1] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[1]), r.d[0]), _mm_mul_ps(_mm_set1_ps(xform[5]), r.d[1])), _mm_mul_ps(_mm_set1_ps(xform[9]), r.d[2]));
    _r.d[2] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[2]), r.d[0]), _mm_mul_ps(_mm_set1_ps(xform[6]), r.d[1])), _mm_mul_ps(_mm_set1_ps(xform[10]), r.d[2]));

    return _r;
}

#ifdef __GNUC__
#pragma GCC pop_options
#endif