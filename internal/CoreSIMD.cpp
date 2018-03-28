#include "CoreSIMD.h"

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

const simd_ivec16 FF_MASK = { (int)0xffffffff };

/*force_inline __m128i _mm_blendv_si128(__m128i x, __m128i y, __m128i mask) {
    return _mm_blendv_epi8(x, y, mask);
}

force_inline bool _mm_all_zeroes(__m128i x) {
    return _mm_movemask_epi8(x) == 0;
}

force_inline bool _mm_not_all_zeroes(__m128i x) {
    return _mm_movemask_epi8(x) != 0;
}

force_inline void _IntersectTri(const ray_packet_t &r, const __m128i ray_mask, const tri_accel_t &tri, uint32_t prim_index, hit_data_t &inter) {
    __m128 nu = _mm_set1_ps(tri.nu), nv = _mm_set1_ps(tri.nv), np = _mm_set1_ps(tri.np);
    __m128 pu = _mm_set1_ps(tri.pu), pv = _mm_set1_ps(tri.pv);

    __m128 e0u = _mm_set1_ps(tri.e0u), e0v = _mm_set1_ps(tri.e0v);
    __m128 e1u = _mm_set1_ps(tri.e1u), e1v = _mm_set1_ps(tri.e1v);

    const int _next_u[] = { 1, 0, 0 },
                          _next_v[] = { 2, 2, 1 };

    int w = (tri.ci & TRI_W_BITS),
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

    //////////////////////////////////////////////////////////////////////////

    __m128 mm1 = _mm_cmpgt_ps(tmpdet0, M_HIT_EPS);
    mm1 = _mm_and_ps(mm1, _mm_cmpgt_ps(detu, M_HIT_EPS));
    mm1 = _mm_and_ps(mm1, _mm_cmpgt_ps(detv, M_HIT_EPS));

    __m128 mm2 = _mm_cmplt_ps(tmpdet0, HIT_EPS);
    mm2 = _mm_and_ps(mm2, _mm_cmplt_ps(detu, HIT_EPS));
    mm2 = _mm_and_ps(mm2, _mm_cmplt_ps(detv, HIT_EPS));

    __m128 mm = _mm_or_ps(mm1, mm2);

    __m128i mask = _mm_castps_si128(mm);
    mask = _mm_and_si128(mask, ray_mask);

    //////////////////////////////////////////////////////////////////////////

    if (_mm_all_zeroes(mask)) return; // no intersection found

    __m128 rdet = _mm_div_ps(ONE, det);
    __m128 t = _mm_mul_ps(dett, rdet);

    __m128i t_valid = _mm_castps_si128(_mm_and_ps(_mm_cmplt_ps(t, inter.t), _mm_cmpgt_ps(t, ZERO)));
    mask = _mm_and_si128(mask, t_valid);
    if (_mm_all_zeroes(mask)) return; // all intersections further than needed

    __m128 bar_u = _mm_mul_ps(detu, rdet);
    __m128 bar_v = _mm_mul_ps(detv, rdet);

    __m128 mask_ps = _mm_castsi128_ps(mask);

    inter.mask = _mm_or_si128(inter.mask, mask);
    inter.prim_index = _mm_blendv_si128(inter.prim_index, _mm_set1_epi32(prim_index), mask);
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
        mask1 = _mm_and_si128(mask1, queue[index].mask);
        if (_mm_all_zeroes(mask1)) {
            queue[index].cur = node.left_child;
        } else {
            __m128i mask2 = _mm_andnot_si128(mask1, queue[index].mask);
            if (_mm_all_zeroes(mask2)) {
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
};*/
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
            simd_fvec16 temp1 = dd[0] * side[0];
            temp1 = temp1 + dd[1] * up[0];
            temp1 = temp1 + dd[2] * fwd[0];

            simd_fvec16 temp2 = dd[0] * side[1];
            temp2 = temp2 + dd[1] * up[1];
            temp2 = temp2 + dd[2] * fwd[1];

            simd_fvec16 temp3 = dd[0] * side[2];
            temp3 = temp3 + dd[1] * up[2];
            temp3 = temp3 + dd[2] * fwd[2];

            dd[0] = temp1;
            dd[1] = temp2;
            dd[2] = temp3;

            simd_fvec16 inv_l = dd[0] * dd[0];
            inv_l = inv_l + dd[1] * dd[1];
            inv_l = inv_l + dd[2] * dd[2];
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

/*bool ray::sse::IntersectTris(const ray_packet_t &r, const __m128i ray_mask, const tri_accel_t *tris, uint32_t num_tris, uint32_t obj_index, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.obj_index = _mm_set1_epi32(obj_index);
    inter.t = out_inter.t;

    for (uint32_t i = 0; i < num_tris; i++) {
        _IntersectTri(r, ray_mask, tris[i], i, inter);
    }

    out_inter.mask = _mm_or_si128(out_inter.mask, inter.mask);
    out_inter.obj_index = _mm_blendv_si128(out_inter.obj_index, inter.obj_index, inter.mask);
    out_inter.prim_index = _mm_blendv_si128(out_inter.prim_index, inter.prim_index, inter.mask);
    out_inter.t = inter.t; // already contains min value

    __m128 mask_ps = _mm_castsi128_ps(inter.mask);
    out_inter.u = _mm_blendv_ps(out_inter.u, inter.u, mask_ps);
    out_inter.v = _mm_blendv_ps(out_inter.v, inter.v, mask_ps);

    return _mm_not_all_zeroes(inter.mask);
}

bool ray::sse::IntersectTris(const ray_packet_t &r, const __m128i ray_mask, const tri_accel_t *tris, const uint32_t *indices, uint32_t num_tris, uint32_t obj_index, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.obj_index = _mm_set1_epi32(obj_index);
    inter.t = out_inter.t;

    for (uint32_t i = 0; i < num_tris; i++) {
        uint32_t index = indices[i];
        _IntersectTri(r, ray_mask, tris[index], index, inter);
    }

    out_inter.mask = _mm_or_si128(out_inter.mask, inter.mask);
    out_inter.obj_index = _mm_blendv_si128(out_inter.obj_index, inter.obj_index, inter.mask);
    out_inter.prim_index = _mm_blendv_si128(out_inter.prim_index, inter.prim_index, inter.mask);
    out_inter.t = inter.t; // already contains min value

    __m128 mask_ps = _mm_castsi128_ps(inter.mask);
    out_inter.u = _mm_blendv_ps(out_inter.u, inter.u, mask_ps);
    out_inter.v = _mm_blendv_ps(out_inter.v, inter.v, mask_ps);

    return _mm_not_all_zeroes(inter.mask);
}

bool ray::sse::IntersectCones(const ray_packet_t &r, const cone_accel_t *cones, uint32_t num_cones, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.t = out_inter.t;

    for (uint32_t i = 0; i < num_cones; i++) {
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

        if (_mm_all_zeroes(m)) continue;

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

        if (_mm_all_zeroes(mask)) continue;

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

        if (_mm_all_zeroes(mask)) continue;

        inter.mask = _mm_or_si128(inter.mask, mask);
        inter.prim_index = _mm_blendv_si128(inter.prim_index, _mm_set1_epi32(i), mask);
        //intersections.t = _mm_or_ps(_mm_andnot_ps(mask_ps, intersections.t1), _mm_and_ps(mask_ps, t1));
    }

    out_inter.mask = _mm_or_si128(out_inter.mask, inter.mask);
    out_inter.prim_index = _mm_blendv_si128(out_inter.prim_index, inter.prim_index, inter.mask);
    //out_intersections.t = intersections.t; // already contains min value

    return _mm_not_all_zeroes(inter.mask);
}

bool ray::sse::IntersectBoxes(const ray_packet_t &r, const aabox_t *boxes, uint32_t num_boxes, hit_data_t &out_inter) {

    hit_data_t inter;
    inter.t = out_inter.t;

    __m128 inv_d[3] = { _mm_rcp_ps(r.d[0]), _mm_rcp_ps(r.d[1]), _mm_rcp_ps(r.d[2]) };

    for (uint32_t i = 0; i < num_boxes; i++) {
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
        if (_mm_all_zeroes(imask)) continue;

        inter.mask = _mm_or_si128(inter.mask, imask);
        inter.prim_index = _mm_blendv_si128(inter.prim_index, _mm_set1_epi32(i), imask);
        inter.t = _mm_or_ps(_mm_andnot_ps(mask, inter.t), _mm_and_ps(mask, tmin));
    }

    out_inter.mask = _mm_or_si128(out_inter.mask, inter.mask);
    out_inter.prim_index = _mm_blendv_si128(out_inter.prim_index, inter.prim_index, inter.mask);
    //out_intersections.t = intersections.t; // already contains min value

    return _mm_not_all_zeroes(inter.mask);
}

bool ray::sse::Traverse_MacroTree_CPU(const ray_packet_t &r, const __m128i ray_mask, const __m128 inv_d[3], const bvh_node_t *nodes, uint32_t root_index,
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
            __m128i mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]);
            mask1 = _mm_and_si128(mask1, st.queue[st.index].mask);
            if (_mm_all_zeroes(mask1)) {
                cur = nodes[cur].parent;
                src = FromChild;
            } else {
                __m128i mask2 = _mm_andnot_si128(mask1, st.queue[st.index].mask);
                if (_mm_not_all_zeroes(mask2)) {
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

                        __m128i bbox_mask = bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max);
                        bbox_mask = _mm_and_si128(st.queue[st.index].mask, bbox_mask);
                        if (_mm_all_zeroes(bbox_mask)) continue;

                        ray_packet_t _r = TransformRay(r, tr.inv_xform);

                        __m128 _inv_d[3] = { _mm_div_ps(ONE, _r.d[0]), _mm_div_ps(ONE, _r.d[1]), _mm_div_ps(ONE, _r.d[2]) };

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
            __m128i mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]);
            mask1 = _mm_and_si128(mask1, st.queue[st.index].mask);
            if (_mm_all_zeroes(mask1)) {
                cur = nodes[cur].sibling;
                src = FromSibling;
            } else {
                __m128i mask2 = _mm_andnot_si128(mask1, st.queue[st.index].mask);
                if (_mm_not_all_zeroes(mask2)) {
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

                        __m128i bbox_mask = bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max);
                        bbox_mask = _mm_and_si128(st.queue[st.index].mask, bbox_mask);
                        if (_mm_all_zeroes(bbox_mask)) continue;

                        ray_packet_t _r = TransformRay(r, tr.inv_xform);

                        __m128 _inv_d[3] = { _mm_div_ps(ONE, _r.d[0]), _mm_div_ps(ONE, _r.d[1]), _mm_div_ps(ONE, _r.d[2]) };

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

bool ray::sse::Traverse_MicroTree_CPU(const ray_packet_t &r, const __m128i ray_mask, const __m128 inv_d[3], const bvh_node_t *nodes, uint32_t root_index,
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
            __m128i mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]);
            mask1 = _mm_and_si128(mask1, st.queue[st.index].mask);
            if (_mm_all_zeroes(mask1)) {
                cur = nodes[cur].parent;
                src = FromChild;
            } else {
                __m128i mask2 = _mm_andnot_si128(mask1, st.queue[st.index].mask);
                if (_mm_not_all_zeroes(mask2)) {
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
            __m128i mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]);
            mask1 = _mm_and_si128(mask1, st.queue[st.index].mask);
            if (_mm_all_zeroes(mask1)) {
                cur = nodes[cur].sibling;
                src = FromSibling;
            } else {
                __m128i mask2 = _mm_andnot_si128(mask1, st.queue[st.index].mask);
                if (_mm_not_all_zeroes(mask2)) {
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

ray::sse::ray_packet_t ray::sse::TransformRay(const ray_packet_t &r, const float *xform) {
    ray_packet_t _r = r;

    _r.o[0] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[0]), r.o[0]), _mm_mul_ps(_mm_set1_ps(xform[4]), r.o[1])), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[8]), r.o[2]), _mm_set1_ps(xform[12])));
    _r.o[1] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[1]), r.o[0]), _mm_mul_ps(_mm_set1_ps(xform[5]), r.o[1])), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[9]), r.o[2]), _mm_set1_ps(xform[13])));
    _r.o[2] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[2]), r.o[0]), _mm_mul_ps(_mm_set1_ps(xform[6]), r.o[1])), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[10]), r.o[2]), _mm_set1_ps(xform[14])));

    _r.d[0] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[0]), r.d[0]), _mm_mul_ps(_mm_set1_ps(xform[4]), r.d[1])), _mm_mul_ps(_mm_set1_ps(xform[8]), r.d[2]));
    _r.d[1] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[1]), r.d[0]), _mm_mul_ps(_mm_set1_ps(xform[5]), r.d[1])), _mm_mul_ps(_mm_set1_ps(xform[9]), r.d[2]));
    _r.d[2] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(xform[2]), r.d[0]), _mm_mul_ps(_mm_set1_ps(xform[6]), r.d[1])), _mm_mul_ps(_mm_set1_ps(xform[10]), r.d[2]));

    return _r;
}*/
