#include "CoreAVX.h"

#include <cassert>

#include "../Types.h"

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target ("avx")
#endif

#if defined(__GNUC__)
#define _mm256_test_all_zeros(mask, val) \
              _mm256_testz_si256((mask), (val))
#endif

namespace ray {
namespace avx {
const __m256 _0_5 = _mm256_set1_ps(0.5f);

const __m256 ZERO = _mm256_set1_ps(0);
const __m256 MINUS_ZERO = _mm256_set1_ps(-0.0f);

const __m256 ONE = _mm256_set1_ps(1);
const __m256 TWO = _mm256_set1_ps(2);
const __m256 FOUR = _mm256_set1_ps(4);

const __m256 HIT_EPS = _mm256_set1_ps(ray::HIT_EPS);
const __m256 M_HIT_EPS = _mm256_set1_ps(-ray::HIT_EPS);

const __m256i FF_MASK = _mm256_set1_epi32(0xffffffff);

force_inline __m256i _mm256_blendv_si128(__m256i x, __m256i y, __m256i mask) {
    return _mm256_blendv_epi8(x, y, mask);
}

force_inline bool _mm256_all_zeroes(__m256i x) {
    return _mm256_movemask_epi8(x) == 0;
}

force_inline bool _mm256_not_all_zeroes(__m256i x) {
    return _mm256_movemask_epi8(x) != 0;
}

force_inline void _IntersectTri(const ray_packet_t &r, const __m256i ray_mask, const tri_accel_t &tri, uint32_t prim_index, hit_data_t &inter) {
    __m256 nu = _mm256_set1_ps(tri.nu), nv = _mm256_set1_ps(tri.nv), np = _mm256_set1_ps(tri.np);
    __m256 pu = _mm256_set1_ps(tri.pu), pv = _mm256_set1_ps(tri.pv);

    __m256 e0u = _mm256_set1_ps(tri.e0u), e0v = _mm256_set1_ps(tri.e0v);
    __m256 e1u = _mm256_set1_ps(tri.e1u), e1v = _mm256_set1_ps(tri.e1v);

    const int _next_u[] = { 1, 0, 0 },
                          _next_v[] = { 2, 2, 1 };

    int w = (tri.ci & TRI_W_BITS),
        u = _next_u[w],
        v = _next_v[w];

    // from "Ray-Triangle Intersection Algorithm for Modern CPU Architectures" [2007]

    //temporary variables
    __m256 det, dett, detu, detv, nrv, nru, du, dv, ou, ov, tmpdet0;//, tmpdet1;

    // ----ray-packet/triangle hit test----
    //dett = np -(ou*nu+ov*nv+ow)
    dett = np;
    dett = _mm256_sub_ps(dett, r.o[w]);

    du = nu;
    dv = nv;

    ou = r.o[u];
    ov = r.o[v];

    du = _mm256_mul_ps(du, ou);
    dv = _mm256_mul_ps(dv, ov);
    dett = _mm256_sub_ps(dett, du);
    dett = _mm256_sub_ps(dett, dv);
    //det = du * nu + dv * nv + dw

    du = r.d[u];
    dv = r.d[v];

    ou = _mm256_sub_ps(pu, ou);
    ov = _mm256_sub_ps(pv, ov);

    det = nu;
    det = _mm256_mul_ps(det, du);
    nrv = nv;
    nrv = _mm256_mul_ps(nrv, dv);

    det = _mm256_add_ps(det, r.d[w]);

    det = _mm256_add_ps(det, nrv);
    //Du = du*dett - (pu-ou)*det
    nru = _mm256_mul_ps(ou, det);

    du = _mm256_mul_ps(du, dett);
    du = _mm256_sub_ps(du, nru);
    //Dv = dv*dett - (pv-ov)*det
    nrv = _mm256_mul_ps(ov, det);

    dv = _mm256_mul_ps(dv, dett);
    dv = _mm256_sub_ps(dv, nrv);
    //detu = (e1vDu ? e1u*Dv)
    nru = e1v;
    nrv = e1u;
    nru = _mm256_mul_ps(nru, du);
    nrv = _mm256_mul_ps(nrv, dv);
    detu = _mm256_sub_ps(nru, nrv);
    //detv = (e0u*Dv ? e0v*Du)
    nrv = e0u;
    nrv = _mm256_mul_ps(nrv, dv);
    dv = e0v;
    dv = _mm256_mul_ps(dv, du);
    detv = _mm256_sub_ps(nrv, dv);

    // same sign of 'det - detu - detv', 'detu', 'detv' indicates intersection

    tmpdet0 = _mm256_sub_ps(_mm256_sub_ps(det, detu), detv);

    //////////////////////////////////////////////////////////////////////////

    __m256 mm1 = _mm256_cmp_ps(tmpdet0, M_HIT_EPS, _CMP_GT_OS);
    mm1 = _mm256_and_ps(mm1, _mm256_cmp_ps(detu, M_HIT_EPS, _CMP_GT_OS));
    mm1 = _mm256_and_ps(mm1, _mm256_cmp_ps(detv, M_HIT_EPS, _CMP_GT_OS));

    __m256 mm2 = _mm256_cmp_ps(tmpdet0, HIT_EPS, _CMP_LT_OS);
    mm2 = _mm256_and_ps(mm2, _mm256_cmp_ps(detu, HIT_EPS, _CMP_LT_OS));
    mm2 = _mm256_and_ps(mm2, _mm256_cmp_ps(detv, HIT_EPS, _CMP_LT_OS));

    __m256 mm = _mm256_or_ps(mm1, mm2);

    __m256i mask = _mm256_castps_si256(mm);
    mask = _mm256_and_si256(mask, ray_mask);

    //////////////////////////////////////////////////////////////////////////

    if (_mm256_all_zeroes(mask)) return; // no intersection found

    __m256 rdet = _mm256_div_ps(ONE, det);
    __m256 t = _mm256_mul_ps(dett, rdet);

    __m256i t_valid = _mm256_castps_si256(_mm256_cmp_ps(t, inter.t, _CMP_LT_OS));
    mask = _mm256_and_si256(mask, t_valid);
    if (_mm256_all_zeroes(mask)) return; // all intersections further than needed

    __m256 bar_u = _mm256_mul_ps(detu, rdet);
    __m256 bar_v = _mm256_mul_ps(detv, rdet);

    mask = _mm256_and_si256(mask, t_valid);
    __m256 mask_ps = _mm256_castsi256_ps(mask);

    inter.mask = _mm256_or_si256(inter.mask, mask);
    inter.prim_index = _mm256_blendv_epi8(inter.prim_index, _mm256_set1_epi32(prim_index), mask);
    inter.t = _mm256_blendv_ps(inter.t, t, mask_ps);
    inter.u = _mm256_blendv_ps(inter.u, bar_u, mask_ps);
    inter.v = _mm256_blendv_ps(inter.v, bar_v, mask_ps);
}

__m256i bbox_test(const __m256 o[3], const __m256 inv_d[3], const __m256 t, const float _bbox_min[3], const float _bbox_max[3]) {
    __m256 box_min[3] = { _mm256_set1_ps(_bbox_min[0]), _mm256_set1_ps(_bbox_min[1]), _mm256_set1_ps(_bbox_min[2]) },
                        box_max[3] = { _mm256_set1_ps(_bbox_max[0]), _mm256_set1_ps(_bbox_max[1]), _mm256_set1_ps(_bbox_max[2]) };

    __m256 low, high, tmin, tmax;

    low = _mm256_mul_ps(inv_d[0], _mm256_sub_ps(box_min[0], o[0]));
    high = _mm256_mul_ps(inv_d[0], _mm256_sub_ps(box_max[0], o[0]));
    tmin = _mm256_min_ps(low, high);
    tmax = _mm256_max_ps(low, high);

    low = _mm256_mul_ps(inv_d[1], _mm256_sub_ps(box_min[1], o[1]));
    high = _mm256_mul_ps(inv_d[1], _mm256_sub_ps(box_max[1], o[1]));
    tmin = _mm256_max_ps(tmin, _mm256_min_ps(low, high));
    tmax = _mm256_min_ps(tmax, _mm256_max_ps(low, high));

    low = _mm256_mul_ps(inv_d[2], _mm256_sub_ps(box_min[2], o[2]));
    high = _mm256_mul_ps(inv_d[2], _mm256_sub_ps(box_max[2], o[2]));
    tmin = _mm256_max_ps(tmin, _mm256_min_ps(low, high));
    tmax = _mm256_min_ps(tmax, _mm256_max_ps(low, high));

    __m256 mask = _mm256_cmp_ps(tmin, tmax, _CMP_LE_OS);
    mask = _mm256_and_ps(mask, _mm256_cmp_ps(tmin, t, _CMP_LE_OS));
    mask = _mm256_and_ps(mask, _mm256_cmp_ps(tmax, ZERO, _CMP_GT_OS));

    return _mm256_castps_si256(mask);
}

force_inline __m256i bbox_test(const __m256 o[3], const __m256 inv_d[3], const __m256 t, const bvh_node_t &node) {
    return bbox_test(o, inv_d, t, node.bbox[0], node.bbox[1]);
}

force_inline uint32_t near_child(const ray_packet_t &r, const __m256i ray_mask, const bvh_node_t &node) {
    __m256i mask = _mm256_castps_si256(_mm256_cmp_ps(r.d[node.space_axis], ZERO, _CMP_LT_OS));
    if (_mm256_test_all_zeros(mask, ray_mask)) {
        return node.left_child;
    } else {
        assert(_mm256_test_all_zeros(_mm256_andnot_si256(mask, FF_MASK), ray_mask));
        return node.right_child;
    }
}

force_inline bool is_leaf_node(const bvh_node_t &node) {
    return node.prim_count != 0;
}

enum eTraversalSource { FromParent, FromChild, FromSibling };

struct TraversalState {
    struct {
        __m256i mask;
        uint32_t cur;
        eTraversalSource src;
    } queue[8];

    int index = 0, num = 1;

    force_inline void select_near_child(const ray_packet_t &r, const bvh_node_t &node) {
        __m256i mask1 = _mm256_castps_si256(_mm256_cmp_ps(r.d[node.space_axis], ZERO, _CMP_LT_OS));
        if (_mm256_test_all_zeros(mask1, queue[index].mask)) {
            queue[index].cur = node.left_child;
        } else {
            __m256i mask2 = _mm256_andnot_si256(mask1, FF_MASK);
            if (_mm256_test_all_zeros(mask2, queue[index].mask)) {
                queue[index].cur = node.right_child;
            } else {
                queue[num].cur = node.left_child;
                queue[num].mask = _mm256_and_si256(queue[index].mask, mask2);
                queue[num].src = queue[index].src;
                num++;
                queue[index].cur = node.right_child;
                queue[index].mask = _mm256_and_si256(queue[index].mask, mask1);
            }
        }
    }
};
}
}

ray::avx::hit_data_t::hit_data_t() {
    mask = _mm256_setzero_si256();
    obj_index = _mm256_set1_epi32(-1);
    prim_index = _mm256_set1_epi32(-1);
    t = _mm256_set1_ps(std::numeric_limits<float>::max());
}

void ray::avx::ConstructRayPacket(const float *o, const float *d, int size, ray_packet_t &out_r) {
    assert(size <= RayPacketSize);

    // ray_packet_t
    // x0 x1 x2 x3 x4 x5 x6 x7
    // y0 y1 y2 y3 y4 y5 y6 y7
    // z0 z1 z2 z3 z4 z5 z6 z7

    if (size == 8) {
        out_r.o[0] = _mm256_set_ps(o[0], o[3], o[6], o[9], o[12], o[15], o[18], o[21]);
        out_r.o[1] = _mm256_set_ps(o[1], o[4], o[7], o[10], o[13], o[16], o[19], o[22]);
        out_r.o[2] = _mm256_set_ps(o[2], o[5], o[8], o[11], o[14], o[17], o[20], o[23]);
        out_r.d[0] = _mm256_set_ps(d[0], d[3], d[6], d[9], d[12], d[15], d[18], d[21]);
        out_r.d[1] = _mm256_set_ps(d[1], d[4], d[7], d[10], d[13], d[16], d[19], d[22]);
        out_r.d[2] = _mm256_set_ps(d[2], d[5], d[8], d[11], d[14], d[17], d[20], d[23]);
    } else if (size == 7) {
        out_r.o[0] = _mm256_set_ps(o[0], o[3], o[6], o[9], o[12], o[15], o[18], 0);
        out_r.o[1] = _mm256_set_ps(o[1], o[4], o[7], o[10], o[13], o[16], o[19], 0);
        out_r.o[2] = _mm256_set_ps(o[2], o[5], o[8], o[11], o[14], o[17], o[20], 0);
        out_r.d[0] = _mm256_set_ps(d[0], d[3], d[6], d[9], d[12], d[15], d[18], 0);
        out_r.d[1] = _mm256_set_ps(d[1], d[4], d[7], d[10], d[13], d[16], d[19], 0);
        out_r.d[2] = _mm256_set_ps(d[2], d[5], d[8], d[11], d[14], d[17], d[20], 0);
    } else if (size == 6) {
        out_r.o[0] = _mm256_set_ps(o[0], o[3], o[6], o[9], o[12], o[15], 0, 0);
        out_r.o[1] = _mm256_set_ps(o[1], o[4], o[7], o[10], o[13], o[16], 0, 0);
        out_r.o[2] = _mm256_set_ps(o[2], o[5], o[8], o[11], o[14], o[17], 0, 0);
        out_r.d[0] = _mm256_set_ps(d[0], d[3], d[6], d[9], d[12], d[15], 0, 0);
        out_r.d[1] = _mm256_set_ps(d[1], d[4], d[7], d[10], d[13], d[16], 0, 0);
        out_r.d[2] = _mm256_set_ps(d[2], d[5], d[8], d[11], d[14], d[17], 0, 0);
    } else if (size == 5) {
        out_r.o[0] = _mm256_set_ps(o[0], o[3], o[6], o[9], o[12], 0, 0, 0);
        out_r.o[1] = _mm256_set_ps(o[1], o[4], o[7], o[10], o[13], 0, 0, 0);
        out_r.o[2] = _mm256_set_ps(o[2], o[5], o[8], o[11], o[14], 0, 0, 0);
        out_r.d[0] = _mm256_set_ps(d[0], d[3], d[6], d[9], d[12], 0, 0, 0);
        out_r.d[1] = _mm256_set_ps(d[1], d[4], d[7], d[10], d[13], 0, 0, 0);
        out_r.d[2] = _mm256_set_ps(d[2], d[5], d[8], d[11], d[14], 0, 0, 0);
    } else if (size == 4) {
        out_r.o[0] = _mm256_set_ps(o[0], o[3], o[6], o[9], 0, 0, 0, 0);
        out_r.o[1] = _mm256_set_ps(o[1], o[4], o[7], o[10], 0, 0, 0, 0);
        out_r.o[2] = _mm256_set_ps(o[2], o[5], o[8], o[11], 0, 0, 0, 0);
        out_r.d[0] = _mm256_set_ps(d[0], d[3], d[6], d[9], 0, 0, 0, 0);
        out_r.d[1] = _mm256_set_ps(d[1], d[4], d[7], d[10], 0, 0, 0, 0);
        out_r.d[2] = _mm256_set_ps(d[2], d[5], d[8], d[11], 0, 0, 0, 0);
    } else if (size == 3) {
        out_r.o[0] = _mm256_set_ps(o[0], o[3], o[6], 0, 0, 0, 0, 0);
        out_r.o[1] = _mm256_set_ps(o[1], o[4], o[7], 0, 0, 0, 0, 0);
        out_r.o[2] = _mm256_set_ps(o[2], o[5], o[8], 0, 0, 0, 0, 0);
        out_r.d[0] = _mm256_set_ps(d[0], d[3], d[6], 0, 0, 0, 0, 0);
        out_r.d[1] = _mm256_set_ps(d[1], d[4], d[7], 0, 0, 0, 0, 0);
        out_r.d[2] = _mm256_set_ps(d[2], d[5], d[8], 0, 0, 0, 0, 0);
    } else if (size == 2) {
        out_r.o[0] = _mm256_set_ps(o[0], o[3], 0, 0, 0, 0, 0, 0);
        out_r.o[1] = _mm256_set_ps(o[1], o[4], 0, 0, 0, 0, 0, 0);
        out_r.o[2] = _mm256_set_ps(o[2], o[5], 0, 0, 0, 0, 0, 0);
        out_r.d[0] = _mm256_set_ps(d[0], d[3], 0, 0, 0, 0, 0, 0);
        out_r.d[1] = _mm256_set_ps(d[1], d[4], 0, 0, 0, 0, 0, 0);
        out_r.d[2] = _mm256_set_ps(d[2], d[5], 0, 0, 0, 0, 0, 0);
    } else if (size == 1) {
        out_r.o[0] = _mm256_set_ps(o[0], 0, 0, 0, 0, 0, 0, 0);
        out_r.o[1] = _mm256_set_ps(o[1], 0, 0, 0, 0, 0, 0, 0);
        out_r.o[2] = _mm256_set_ps(o[2], 0, 0, 0, 0, 0, 0, 0);
        out_r.d[0] = _mm256_set_ps(d[0], 0, 0, 0, 0, 0, 0, 0);
        out_r.d[1] = _mm256_set_ps(d[1], 0, 0, 0, 0, 0, 0, 0);
        out_r.d[2] = _mm256_set_ps(d[2], 0, 0, 0, 0, 0, 0, 0);
    }
}

void ray::avx::GeneratePrimaryRays(const camera_t &cam, const rect_t &r, int w, int h, math::aligned_vector<ray_packet_t> &out_rays) {
    size_t i = 0;
    out_rays.resize((size_t)(r.w * r.h / 8 + ((r.w * r.h) % 4 != 0)));

    __m256 ww = _mm256_set1_ps((float)w), hh = _mm256_set1_ps((float)h);

    float k = float(h) / w;

    __m256 fwd[3] = { _mm256_set1_ps(cam.fwd[0]), _mm256_set1_ps(cam.fwd[1]), _mm256_set1_ps(cam.fwd[2]) },
                    side[3] = { _mm256_set1_ps(cam.side[0]), _mm256_set1_ps(cam.side[1]), _mm256_set1_ps(cam.side[2]) },
                              up[3] = { _mm256_set1_ps(cam.up[0] * k), _mm256_set1_ps(cam.up[1] * k), _mm256_set1_ps(cam.up[2] * k) };

    for (int y = r.y; y < r.y + r.h - (r.h & (RayPacketDimY - 1)); y += RayPacketDimY) {
        __m256 xx = _mm256_setr_ps(float(r.x + 0), float(r.x + 1), float(r.x + 0), float(r.x + 1), 
                                   float(r.x + 2), float(r.x + 3), float(r.x + 2), float(r.x + 3));
        float fy = float(y);
        __m256 yy = _mm256_setr_ps(-fy, -fy, -fy - 1, -fy - 1, -fy, -fy, -fy - 1, -fy - 1);
        yy = _mm256_div_ps(yy, hh);
        yy = _mm256_add_ps(yy, _0_5);

        for (int x = r.x; x < r.x + r.w - (r.w & (RayPacketDimX - 1)); x += RayPacketDimX) {
            __m256 dd[3];

            // x / w - 0.5
            dd[0] = _mm256_div_ps(xx, ww);
            dd[0] = _mm256_sub_ps(dd[0], _0_5);

            // -y / h + 0.5
            dd[1] = yy;

            dd[2] = ONE;

            // d = d.x * side + d.y * up + d.z * fwd
            __m256 temp1 = _mm256_mul_ps(dd[0], side[0]);
            temp1 = _mm256_add_ps(temp1, _mm256_mul_ps(dd[1], up[0]));
            temp1 = _mm256_add_ps(temp1, _mm256_mul_ps(dd[2], fwd[0]));

            __m256 temp2 = _mm256_mul_ps(dd[0], side[1]);
            temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(dd[1], up[1]));
            temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(dd[2], fwd[1]));

            __m256 temp3 = _mm256_mul_ps(dd[0], side[2]);
            temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(dd[1], up[2]));
            temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(dd[2], fwd[2]));

            dd[0] = temp1;
            dd[1] = temp2;
            dd[2] = temp3;

            __m256 inv_l = _mm256_mul_ps(dd[0], dd[0]);
            inv_l = _mm256_add_ps(inv_l, _mm256_mul_ps(dd[1], dd[1]));
            inv_l = _mm256_add_ps(inv_l, _mm256_mul_ps(dd[2], dd[2]));
            //inv_l = _mm_rsqrt_ps(inv_l);
            inv_l = _mm256_sqrt_ps(inv_l);
            inv_l = _mm256_div_ps(ONE, inv_l);

            assert(i < out_rays.size());
            auto &r = out_rays[i++];

            r.d[0] = _mm256_mul_ps(dd[0], inv_l);
            r.d[1] = _mm256_mul_ps(dd[1], inv_l);
            r.d[2] = _mm256_mul_ps(dd[2], inv_l);

            r.o[0] = _mm256_set1_ps(cam.origin[0]);
            r.o[1] = _mm256_set1_ps(cam.origin[1]);
            r.o[2] = _mm256_set1_ps(cam.origin[2]);

            r.id.x = x;
            r.id.y = y;

            xx = _mm256_add_ps(xx, FOUR);
        }
    }
}

bool ray::avx::IntersectTris(const ray_packet_t &r, const __m256i ray_mask, const tri_accel_t *tris, uint32_t num_tris, uint32_t obj_index, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.obj_index = _mm256_set1_epi32(obj_index);
    inter.t = out_inter.t;

    for (uint32_t i = 0; i < num_tris; i++) {
        _IntersectTri(r, ray_mask, tris[i], i, inter);
    }

    out_inter.mask = _mm256_or_si256(out_inter.mask, inter.mask);
    out_inter.obj_index = _mm256_blendv_epi8(out_inter.obj_index, inter.obj_index, inter.mask);
    out_inter.prim_index = _mm256_blendv_epi8(out_inter.prim_index, inter.prim_index, inter.mask);
    out_inter.t = inter.t; // already contains min value

    __m256 mask_ps = _mm256_castsi256_ps(inter.mask);
    out_inter.u = _mm256_blendv_ps(out_inter.u, inter.u, mask_ps);
    out_inter.v = _mm256_blendv_ps(out_inter.v, inter.v, mask_ps);

    return _mm256_movemask_epi8(inter.mask) != 0;
}

bool ray::avx::IntersectTris(const ray_packet_t &r, const __m256i ray_mask, const tri_accel_t *tris, const uint32_t *indices, uint32_t num_tris, uint32_t obj_index, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.obj_index = _mm256_set1_epi32(obj_index);
    inter.t = out_inter.t;

    for (uint32_t i = 0; i < num_tris; i++) {
        uint32_t index = indices[i];
        _IntersectTri(r, ray_mask, tris[index], index, inter);
    }

    out_inter.mask = _mm256_or_si256(out_inter.mask, inter.mask);
    out_inter.obj_index = _mm256_blendv_epi8(out_inter.obj_index, inter.obj_index, inter.mask);
    out_inter.prim_index = _mm256_blendv_epi8(out_inter.prim_index, inter.prim_index, inter.mask);
    out_inter.t = inter.t; // already contains min value

    __m256 mask_ps = _mm256_castsi256_ps(inter.mask);
    out_inter.u = _mm256_blendv_ps(out_inter.u, inter.u, mask_ps);
    out_inter.v = _mm256_blendv_ps(out_inter.v, inter.v, mask_ps);

    return _mm256_movemask_epi8(inter.mask) != 0;
}

bool ray::avx::IntersectCones(const ray_packet_t &r, const cone_accel_t *cones, uint32_t num_cones, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.t = out_inter.t;

    for (uint32_t i = 0; i < num_cones; i++) {
        const cone_accel_t &cone = cones[i];

        __m256 cone_o[3], cone_v[3], cone_cos_phi_sqr;
        cone_o[0] = _mm256_set1_ps(cone.o[0]);
        cone_o[1] = _mm256_set1_ps(cone.o[1]);
        cone_o[2] = _mm256_set1_ps(cone.o[2]);

        cone_v[0] = _mm256_set1_ps(cone.v[0]);
        cone_v[1] = _mm256_set1_ps(cone.v[1]);
        cone_v[2] = _mm256_set1_ps(cone.v[2]);

        cone_cos_phi_sqr = _mm256_set1_ps(cone.cos_phi_sqr);

        __m256 co[3];
        co[0] = _mm256_sub_ps(r.o[0], cone_o[0]);
        co[1] = _mm256_sub_ps(r.o[1], cone_o[1]);
        co[2] = _mm256_sub_ps(r.o[2], cone_o[2]);

        // a = dot(d, cone_v)
        __m256 a = _mm256_mul_ps(r.d[0], cone_v[0]);
        a = _mm256_add_ps(a, _mm256_mul_ps(r.d[1], cone_v[1]));
        a = _mm256_add_ps(a, _mm256_mul_ps(r.d[2], cone_v[2]));

        // c = dot(co, cone_v)
        __m256 c = _mm256_mul_ps(co[0], cone_v[0]);
        c = _mm256_add_ps(c, _mm256_mul_ps(co[1], cone_v[1]));
        c = _mm256_add_ps(c, _mm256_mul_ps(co[2], cone_v[2]));

        // b = 2 * (a * c - dot(d, co) * cone.cos_phi_sqr)
        __m256 b = _mm256_mul_ps(r.d[0], co[0]);
        b = _mm256_add_ps(b, _mm256_mul_ps(r.d[1], co[1]));
        b = _mm256_add_ps(b, _mm256_mul_ps(r.d[2], co[2]));
        b = _mm256_mul_ps(b, cone_cos_phi_sqr);
        b = _mm256_sub_ps(_mm256_mul_ps(a, c), b);
        b = _mm256_mul_ps(b, TWO);

        // a = a * a - cone.cos_phi_sqr
        a = _mm256_mul_ps(a, a);
        a = _mm256_sub_ps(a, cone_cos_phi_sqr);

        // c = c * c - dot(co, co) * cone.cos_phi_sqr
        c = _mm256_mul_ps(c, c);

        __m256 temp = _mm256_mul_ps(co[0], co[0]);
        temp = _mm256_add_ps(temp, _mm256_mul_ps(co[1], co[1]));
        temp = _mm256_add_ps(temp, _mm256_mul_ps(co[2], co[2]));

        temp = _mm256_mul_ps(temp, cone_cos_phi_sqr);

        c = _mm256_sub_ps(c, temp);

        // D = b * b - 4 * a * c
        __m256 D = _mm256_mul_ps(b, b);

        temp = _mm256_mul_ps(FOUR, a);
        temp = _mm256_mul_ps(temp, c);

        D = _mm256_sub_ps(D, temp);

        __m256i m = _mm256_castps_si256(_mm256_cmp_ps(D, ZERO, _CMP_GE_OS));

        if (_mm256_test_all_zeros(m, FF_MASK)) continue;

        D = _mm256_sqrt_ps(D);

        a = _mm256_mul_ps(a, TWO);
        b = _mm256_xor_ps(b, MINUS_ZERO); // swap sign

        __m256 t1, t2;
        t1 = _mm256_sub_ps(b, D);
        t1 = _mm256_div_ps(t1, a);
        t2 = _mm256_add_ps(b, D);
        t2 = _mm256_div_ps(t2, a);

        __m256 mask1 = _mm256_cmp_ps(t1, ZERO, _CMP_GT_OS);
        mask1 = _mm256_and_ps(mask1, _mm256_cmp_ps(t1, inter.t, _CMP_LT_OS));
        __m256 mask2 = _mm256_cmp_ps(t2, ZERO, _CMP_GT_OS);
        mask2 = _mm256_and_ps(mask2, _mm256_cmp_ps(t2, inter.t, _CMP_LT_OS));
        __m256i mask = _mm256_castps_si256(_mm256_or_ps(mask1, mask2));

        if (_mm256_test_all_zeros(mask, FF_MASK)) continue;

        __m256 p1c[3], p2c[3];
        p1c[0] = _mm256_add_ps(r.o[0], _mm256_mul_ps(t1, r.d[0]));
        p1c[1] = _mm256_add_ps(r.o[1], _mm256_mul_ps(t1, r.d[1]));
        p1c[2] = _mm256_add_ps(r.o[2], _mm256_mul_ps(t1, r.d[2]));

        p2c[0] = _mm256_add_ps(r.o[0], _mm256_mul_ps(t2, r.d[0]));
        p2c[1] = _mm256_add_ps(r.o[1], _mm256_mul_ps(t2, r.d[1]));
        p2c[2] = _mm256_add_ps(r.o[2], _mm256_mul_ps(t2, r.d[2]));

        p1c[0] = _mm256_sub_ps(cone_o[0], p1c[0]);
        p1c[1] = _mm256_sub_ps(cone_o[1], p1c[1]);
        p1c[2] = _mm256_sub_ps(cone_o[2], p1c[2]);

        p2c[0] = _mm256_sub_ps(cone_o[0], p2c[0]);
        p2c[1] = _mm256_sub_ps(cone_o[1], p2c[1]);
        p2c[2] = _mm256_sub_ps(cone_o[2], p2c[2]);

        __m256 dot1 = _mm256_mul_ps(p1c[0], cone_v[0]);
        dot1 = _mm256_add_ps(dot1, _mm256_mul_ps(p1c[1], cone_v[1]));
        dot1 = _mm256_add_ps(dot1, _mm256_mul_ps(p1c[2], cone_v[2]));

        __m256 dot2 = _mm256_mul_ps(p2c[0], cone_v[0]);
        dot2 = _mm256_add_ps(dot2, _mm256_mul_ps(p2c[1], cone_v[1]));
        dot2 = _mm256_add_ps(dot2, _mm256_mul_ps(p2c[2], cone_v[2]));

        __m256 cone_start = _mm256_set1_ps(cone.cone_start),
               cone_end = _mm256_set1_ps(cone.cone_end);

        mask1 = _mm256_cmp_ps(dot1, cone_start, _CMP_GE_OS);
        mask2 = _mm256_cmp_ps(dot2, cone_start, _CMP_GE_OS);
        mask1 = _mm256_and_ps(mask1, _mm256_cmp_ps(dot1, cone_end, _CMP_LE_OS));
        mask2 = _mm256_and_ps(mask2, _mm256_cmp_ps(dot2, cone_end, _CMP_LE_OS));
        mask = _mm256_castps_si256(_mm256_or_ps(mask1, mask2));

        if (_mm256_test_all_zeros(mask, FF_MASK)) continue;

        inter.mask = _mm256_or_si256(inter.mask, mask);
        inter.obj_index = _mm256_blendv_epi8(inter.obj_index, _mm256_set1_epi32(i), mask);
    }

    out_inter.mask = _mm256_or_si256(out_inter.mask, inter.mask);
    out_inter.obj_index = _mm256_blendv_epi8(out_inter.obj_index, inter.obj_index, inter.mask);

    return !_mm256_test_all_zeros(inter.mask, FF_MASK);
}

bool ray::avx::IntersectBoxes(const ray_packet_t &r, const aabox_t *boxes, uint32_t num_boxes, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.t = out_inter.t;

    __m256 inv_d[3] = { _mm256_rcp_ps(r.d[0]), _mm256_rcp_ps(r.d[1]), _mm256_rcp_ps(r.d[2]) };

    for (uint32_t i = 0; i < num_boxes; i++) {
        const aabox_t &box = boxes[i];

        __m256 box_min[3] = { _mm256_set1_ps(box.min[0]), _mm256_set1_ps(box.min[1]), _mm256_set1_ps(box.min[2]) },
                            box_max[3] = { _mm256_set1_ps(box.max[0]), _mm256_set1_ps(box.max[1]), _mm256_set1_ps(box.max[2]) };

        __m256 low, high, tmin, tmax;

        low = _mm256_mul_ps(inv_d[0], _mm256_sub_ps(box_min[0], r.o[0]));
        high = _mm256_mul_ps(inv_d[0], _mm256_sub_ps(box_max[0], r.o[0]));
        tmin = _mm256_min_ps(low, high);
        tmax = _mm256_max_ps(low, high);

        low = _mm256_mul_ps(inv_d[1], _mm256_sub_ps(box_min[1], r.o[1]));
        high = _mm256_mul_ps(inv_d[1], _mm256_sub_ps(box_max[1], r.o[1]));
        tmin = _mm256_max_ps(tmin, _mm256_min_ps(low, high));
        tmax = _mm256_min_ps(tmax, _mm256_max_ps(low, high));

        low = _mm256_mul_ps(inv_d[2], _mm256_sub_ps(box_min[2], r.o[2]));
        high = _mm256_mul_ps(inv_d[2], _mm256_sub_ps(box_max[2], r.o[2]));
        tmin = _mm256_max_ps(tmin, _mm256_min_ps(low, high));
        tmax = _mm256_min_ps(tmax, _mm256_max_ps(low, high));

        __m256 mask = _mm256_cmp_ps(tmin, tmax, _CMP_LE_OS);
        mask = _mm256_and_ps(mask, _mm256_cmp_ps(tmax, ZERO, _CMP_GT_OS));
        mask = _mm256_and_ps(mask, _mm256_cmp_ps(tmin, inter.t, _CMP_LT_OS));

        __m256i imask = _mm256_castps_si256(mask);
        if (_mm256_test_all_zeros(imask, FF_MASK)) continue;

        inter.mask = _mm256_or_si256(inter.mask, imask);
        inter.obj_index = _mm256_blendv_epi8(inter.obj_index, _mm256_set1_epi32(i), imask);
        inter.t = _mm256_blendv_ps(inter.t, tmin, mask);
    }

    out_inter.mask = _mm256_or_si256(out_inter.mask, inter.mask);
    out_inter.obj_index = _mm256_blendv_epi8(out_inter.obj_index, inter.obj_index, inter.mask);
    out_inter.t = inter.t; // already contains min value

    return !_mm256_test_all_zeros(inter.mask, FF_MASK);
}

bool ray::avx::Traverse_MacroTree_CPU(const ray_packet_t &r, const __m256i ray_mask, const __m256 inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                                      const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                      const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter) {
    bool res = false;

    TraversalState st;

    st.queue[0].mask = ray_mask;

    st.queue[0].src = FromSibling;
    st.queue[0].cur = node_index;

    if (!is_leaf_node(nodes[node_index])) {
        st.queue[0].src = FromParent;
        st.select_near_child(r, nodes[node_index]);
    }

    while (st.index < st.num) {
        uint32_t &cur = st.queue[st.index].cur;
        eTraversalSource &src = st.queue[st.index].src;

        switch (src) {
        case FromChild:
            if (cur == node_index || cur == 0xffffffff) {
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
            __m256i mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]);
            mask1 = _mm256_and_si256(mask1, st.queue[st.index].mask);
            if (_mm256_all_zeroes(mask1)) {
                cur = nodes[cur].parent;
                src = FromChild;
            } else {
                __m256i mask2 = _mm256_andnot_si256(mask1, st.queue[st.index].mask);
                if (_mm256_not_all_zeroes(mask2)) {
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

                        __m256i bbox_mask = bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max);
                        bbox_mask = _mm256_and_si256(st.queue[st.index].mask, bbox_mask);
                        if (_mm256_all_zeroes(bbox_mask)) continue;

                        ray_packet_t _r = TransformRay(r, tr.inv_xform);

                        __m256 _inv_d[3] = { _mm256_div_ps(ONE, _r.d[0]), _mm256_div_ps(ONE, _r.d[1]), _mm256_div_ps(ONE, _r.d[2]) };

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
            __m256i mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]);
            mask1 = _mm256_and_si256(mask1, st.queue[st.index].mask);
            if (_mm256_all_zeroes(mask1)) {
                cur = nodes[cur].sibling;
                src = FromSibling;
            } else {
                __m256i mask2 = _mm256_andnot_si256(mask1, st.queue[st.index].mask);
                if (!_mm256_not_all_zeroes(mask2)) {
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

                        __m256i bbox_mask = bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max);
                        bbox_mask = _mm256_and_si256(st.queue[st.index].mask, bbox_mask);
                        if (_mm256_all_zeroes(bbox_mask)) continue;

                        ray_packet_t _r = TransformRay(r, tr.inv_xform);

                        __m256 _inv_d[3] = { _mm256_div_ps(ONE, _r.d[0]), _mm256_div_ps(ONE, _r.d[1]), _mm256_div_ps(ONE, _r.d[2]) };

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

bool ray::avx::Traverse_MicroTree_CPU(const ray_packet_t &r, const __m256i ray_mask, const __m256 inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                                      const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t &inter) {
    bool ret = false;

    TraversalState st;

    st.queue[0].mask = ray_mask;

    st.queue[0].src = FromSibling;
    st.queue[0].cur = node_index;

    if (!is_leaf_node(nodes[node_index])) {
        st.queue[0].src = FromParent;
        st.select_near_child(r, nodes[node_index]);
    }

    while (st.index < st.num) {
        uint32_t &cur = st.queue[st.index].cur;
        eTraversalSource &src = st.queue[st.index].src;

        switch (src) {
        case FromChild:
            if (cur == node_index || cur == 0xffffffff) {
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
            __m256i mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]);
            mask1 = _mm256_and_si256(mask1, st.queue[st.index].mask);
            if (_mm256_all_zeroes(mask1)) {
                cur = nodes[cur].parent;
                src = FromChild;
            } else {
                __m256i mask2 = _mm256_andnot_si256(mask1, st.queue[st.index].mask);
                if (_mm256_not_all_zeroes(mask2)) {
                    st.queue[st.num].cur = nodes[cur].parent;
                    st.queue[st.num].mask = mask2;
                    st.queue[st.num].src = FromChild;
                    st.num++;
                    st.queue[st.index].mask = mask1;
                }

                if (is_leaf_node(nodes[cur])) {
                    // process leaf
                    ret |= IntersectTris(r, st.queue[st.index].mask, tris, &tri_indices[nodes[cur].prim_index], nodes[cur].prim_count, obj_index, inter);

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
            __m256i mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]);
            mask1 = _mm256_and_si256(mask1, st.queue[st.index].mask);
            if (_mm256_all_zeroes(mask1)) {
                cur = nodes[cur].sibling;
                src = FromSibling;
            } else {
                __m256i mask2 = _mm256_andnot_si256(mask1, st.queue[st.index].mask);
                if (_mm256_not_all_zeroes(mask2)) {
                    st.queue[st.num].cur = nodes[cur].sibling;
                    st.queue[st.num].mask = mask2;
                    st.queue[st.num].src = FromSibling;
                    st.num++;
                    st.queue[st.index].mask = mask1;
                }

                if (is_leaf_node(nodes[cur])) {
                    // process leaf
                    ret |= IntersectTris(r, st.queue[st.index].mask, tris, &tri_indices[nodes[cur].prim_index], nodes[cur].prim_count, obj_index, inter);

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
    return ret;
}

ray::avx::ray_packet_t ray::avx::TransformRay(const ray_packet_t &r, const float *xform) {
    ray_packet_t _r = r;

    _r.o[0] = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(xform[0]), r.o[0]), _mm256_mul_ps(_mm256_set1_ps(xform[4]), r.o[1])), _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(xform[8]), r.o[2]), _mm256_set1_ps(xform[12])));
    _r.o[1] = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(xform[1]), r.o[0]), _mm256_mul_ps(_mm256_set1_ps(xform[5]), r.o[1])), _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(xform[9]), r.o[2]), _mm256_set1_ps(xform[13])));
    _r.o[2] = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(xform[2]), r.o[0]), _mm256_mul_ps(_mm256_set1_ps(xform[6]), r.o[1])), _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(xform[10]), r.o[2]), _mm256_set1_ps(xform[14])));

    _r.d[0] = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(xform[0]), r.d[0]), _mm256_mul_ps(_mm256_set1_ps(xform[4]), r.d[1])), _mm256_mul_ps(_mm256_set1_ps(xform[8]), r.d[2]));
    _r.d[1] = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(xform[1]), r.d[0]), _mm256_mul_ps(_mm256_set1_ps(xform[5]), r.d[1])), _mm256_mul_ps(_mm256_set1_ps(xform[9]), r.d[2]));
    _r.d[2] = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(xform[2]), r.d[0]), _mm256_mul_ps(_mm256_set1_ps(xform[6]), r.d[1])), _mm256_mul_ps(_mm256_set1_ps(xform[10]), r.d[2]));

    return _r;
}

#ifdef __GNUC__
#pragma GCC pop_options
#endif
