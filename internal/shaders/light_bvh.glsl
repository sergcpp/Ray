#ifndef LIGHT_BVH_GLSL
#define LIGHT_BVH_GLSL

#include "types.h"
#include "common.glsl"

#define near_child_lnode(rd, n)   \
    (rd)[n.right_child >> 30] < 0 ? (n.right_child & RIGHT_CHILD_BITS) : n.left_child

#define far_child_lnode(rd, n)    \
    (rd)[n.right_child >> 30] < 0 ? n.left_child : (n.right_child & RIGHT_CHILD_BITS)

float approx_atan2(const float y, const float x) { // max error is 0.000004
    float t0, t1, t3, t4;

    t3 = abs(x);
    t1 = abs(y);
    t0 = max(t3, t1);
    t1 = min(t3, t1);
    t3 = 1.0 / t0;
    t3 = t1 * t3;

    t4 = t3 * t3;
    t0 = -0.013480470;
    t0 = t0 * t4 + 0.057477314;
    t0 = t0 * t4 - 0.121239071;
    t0 = t0 * t4 + 0.195635925;
    t0 = t0 * t4 - 0.332994597;
    t0 = t0 * t4 + 0.999995630;
    t3 = t0 * t3;

    t3 = (abs(y) > abs(x)) ? 1.570796327 - t3 : t3;
    t3 = (x < 0) ? 3.141592654 - t3 : t3;
    t3 = (y < 0) ? -t3 : t3;

    return t3;
}

float approx_acos(float x) { // max error is 0.000068f
    float negate = float(x < 0);
    x = abs(x);
    float ret = -0.0187293;
    ret = ret * x;
    ret = ret + 0.0742610;
    ret = ret * x;
    ret = ret - 0.2121144;
    ret = ret * x;
    ret = ret + 1.5707288;
    ret = ret * sqrt(1.0 - saturate(x));
    ret = ret - 2.0 * negate * ret;
    return negate * PI + ret;
}

float approx_cos(float x) { // max error is 0.056010
    const float tp = 1.0 / (2.0 * PI);
    x *= tp;
    x -= 0.25 + floor(x + 0.25);
    x *= 16.0 * (abs(x) - 0.5);
    return x;
}

float calc_lnode_importance(const light_bvh_node_t n, const vec3 P) {
    float mul = 1.0, v_len2 = 1.0;
    if (n.bbox_min[0] > -MAX_DIST) { // check if this is a local light
        vec3 v = P - 0.5 * (vec3(n.bbox_min[0], n.bbox_min[1], n.bbox_min[2]) +
                            vec3(n.bbox_max[0], n.bbox_max[1], n.bbox_max[2]));
        vec3 ext = vec3(n.bbox_max[0], n.bbox_max[1], n.bbox_max[2]) -
                   vec3(n.bbox_min[0], n.bbox_min[1], n.bbox_min[2]);

        const float extent = 0.5 * length(ext);
        v_len2 = dot(v, v);
        const float v_len = sqrt(v_len2);
        const float omega_u = approx_atan2(extent, v_len) + 0.000005;

        vec3 axis = vec3(n.axis[0], n.axis[1], n.axis[2]);

        const float omega = approx_acos(min(dot(axis, v / v_len), 1.0)) - 0.00007;
        const float omega_ = max(0.0, omega - n.omega_n - omega_u);
        mul = omega_ < n.omega_e ? cos(omega_) : 0.0;
    }

    // TODO: account for normal dot product here
    return n.flux * mul / v_len2;
}

float cos_sub_clamped(const float sin_omega_a, const float cos_omega_a, const float sin_omega_b, const float cos_omega_b) {
    if (cos_omega_a > cos_omega_b) {
        return 1.0;
    }
    return cos_omega_a * cos_omega_b + sin_omega_a * sin_omega_b;
}

float sin_sub_clamped(const float sin_omega_a, const float cos_omega_a, const float sin_omega_b, const float cos_omega_b) {
    if (cos_omega_a > cos_omega_b) {
        return 0.0;
    }
    return sin_omega_a * cos_omega_b - cos_omega_a * sin_omega_b;
}

vec3 decode_oct_dir(const uint oct) {
    vec3 ret;
    ret.x = -1.0 + 2.0 * float((oct >> 16) & 0x0000ffff) / 65535.0;
    ret.y = -1.0 + 2.0 * float(oct & 0x0000ffff) / 65535.0;
    ret.z = 1.0 - abs(ret.x) - abs(ret.y);
    if (ret.z < 0.0) {
        const float temp = ret.x;
        ret.x = (1.0 - abs(ret.y)) * _copysign(1.0, temp);
        ret.y = (1.0 - abs(temp)) * _copysign(1.0, ret.y);
    }
    return normalize(ret);
}

vec2 decode_cosines(const uint val) {
    const uvec2 ab = uvec2((val >> 16) & 0x0000ffff, (val & 0x0000ffff));
    return 2.0 * (vec2(ab) / 65534.0) - 1.0;
}

// 8-bit packed variants (compact light tree: axis is an oct pair, cos_omega_ne a cos pair)
vec3 decode_oct_dir8(const uint oct) {
    vec3 ret;
    ret.x = -1.0 + 2.0 * float((oct >> 8) & 0xff) / 255.0;
    ret.y = -1.0 + 2.0 * float(oct & 0xff) / 255.0;
    ret.z = 1.0 - abs(ret.x) - abs(ret.y);
    if (ret.z < 0.0) {
        const float temp = ret.x;
        ret.x = (1.0 - abs(ret.y)) * _copysign(1.0, temp);
        ret.y = (1.0 - abs(temp)) * _copysign(1.0, ret.y);
    }
    return normalize(ret);
}

vec2 decode_cosines8(const uint val) {
    const uvec2 ab = uvec2((val >> 8) & 0xff, (val & 0xff));
    return 2.0 * (vec2(ab) / 254.0) - 1.0;
}

// Packed per-child accessors of the compact light node (see Types::light_cwbvh_node_t)
uint light_flux_at(const light_cwbvh_node_t n, const uint i) {
    return (n.flux[i >> 2u] >> (8u * (i & 3u))) & 0xffu;
}

// 8-bit fraction of the parent's flux, cube-coded. The per-node scale cancels: every consumer
// normalizes importance inside the node.
float light_flux_weight(const light_cwbvh_node_t n, const uint i) {
    const float v = float(light_flux_at(n, i)) * (1.0 / 255.0);
    return v * v * v;
}

uint light_axis_at(const light_cwbvh_node_t n, const uint i) {
    return (n.axis[i >> 1u] >> (16u * (i & 1u))) & 0xffffu;
}

uint light_cos_at(const light_cwbvh_node_t n, const uint i) {
    return (n.cos_omega_ne[i >> 1u] >> (16u * (i & 1u))) & 0xffffu;
}

float calc_lnode_importance(const light_wbvh_node_t n, const vec3 P, out float importance[8]) {
    float total_importance = 0.0;
    [[unroll]] for (int i = 0; i < 8; ++i) {
        float mul = 1.0, v_len2 = 1.0;
        if (n.bbox_min[0][i] > -MAX_DIST) {
            const vec3 axis = decode_oct_dir(n.axis[i]);
            const vec3 ext = vec3(n.bbox_max[0][i] - n.bbox_min[0][i],
                                  n.bbox_max[1][i] - n.bbox_min[1][i],
                                  n.bbox_max[2][i] - n.bbox_min[2][i]);
            const float extent = 0.5 * length(ext);

            const vec3 pc = 0.5 * vec3(n.bbox_min[0][i] + n.bbox_max[0][i],
                                       n.bbox_min[1][i] + n.bbox_max[1][i],
                                       n.bbox_min[2][i] + n.bbox_max[2][i]);
            vec3 wi = P - pc;
            const float dist2 = dot(wi, wi);
            const float dist = sqrt(dist2);
            wi /= dist;

            v_len2 = max(dist2, extent);

            const float cos_omega_w = dot(axis, wi);
            const float sin_omega_w = sqrt(1.0 - cos_omega_w * cos_omega_w);

            float cos_omega_b = -1.0;
            if (dist2 >= extent * extent) {
                cos_omega_b = sqrt(1.0 - (extent * extent) / dist2);
            }
            const float sin_omega_b = sqrt(1.0 - cos_omega_b * cos_omega_b);

            const vec2 cos_omega_ne = decode_cosines(n.cos_omega_ne[i]);
            const float sin_omega_n = sqrt(1.0 - cos_omega_ne[0] * cos_omega_ne[0]);

            const float cos_omega_x = cos_sub_clamped(sin_omega_w, cos_omega_w, sin_omega_n, cos_omega_ne[0]);
            const float sin_omega_x = sin_sub_clamped(sin_omega_w, cos_omega_w, sin_omega_n, cos_omega_ne[0]);
            const float cos_omega = cos_sub_clamped(sin_omega_x, cos_omega_x, sin_omega_b, cos_omega_b);

            mul = cos_omega > cos_omega_ne[1] ? cos_omega : 0.0;
        }
        importance[i] = n.flux[i] * mul / v_len2;
        total_importance += importance[i];
    }
    return total_importance;
}

float calc_lnode_importance(const light_cwbvh_node_t n, const vec3 P, out float importance[8]) {
    // pow2 quantization step stored in ch_exp (see pow2Extents() on the CPU side)
    const vec3 step = vec3(cwbvh_step(n.ch_exp, 0u), cwbvh_step(n.ch_exp, 1u), cwbvh_step(n.ch_exp, 2u));

    float total_importance = 0.0;
    for (uint i = 0; i < 8; ++i) {
        importance[i] = light_flux_weight(n, i);
        if (((n.ch_bbox_min[0][i / 4] >> (8u * (i % 4u))) & 0xffu) != 0xff ||
            ((n.ch_bbox_max[0][i / 4] >> (8u * (i % 4u))) & 0xffu) != 0) {
            const vec3 bbox_min = vec3(n.bbox_min[0] + float((n.ch_bbox_min[0][i / 4] >> (8u * (i % 4u))) & 0xffu) * step[0],
                                       n.bbox_min[1] + float((n.ch_bbox_min[1][i / 4] >> (8u * (i % 4u))) & 0xffu) * step[1],
                                       n.bbox_min[2] + float((n.ch_bbox_min[2][i / 4] >> (8u * (i % 4u))) & 0xffu) * step[2]);
            const vec3 bbox_max = vec3(n.bbox_min[0] + float((n.ch_bbox_max[0][i / 4] >> (8u * (i % 4u))) & 0xffu) * step[0],
                                       n.bbox_min[1] + float((n.ch_bbox_max[1][i / 4] >> (8u * (i % 4u))) & 0xffu) * step[1],
                                       n.bbox_min[2] + float((n.ch_bbox_max[2][i / 4] >> (8u * (i % 4u))) & 0xffu) * step[2]);
            const float extent = 0.5 * length(bbox_max - bbox_min);

            const vec3 pc = 0.5 * (bbox_min + bbox_max);
            vec3 wi = P - pc;
            const float dist2 = dot(wi, wi);
            const float dist = sqrt(dist2);
            wi /= dist;

            const float v_len2 = max(dist2, extent);

            const vec3 axis = decode_oct_dir8(light_axis_at(n, i));
            const float cos_omega_w = dot(axis, wi);
            const float sin_omega_w = sqrt(1.0 - cos_omega_w * cos_omega_w);

            float cos_omega_b = -1.0;
            if (dist2 >= extent * extent) {
                cos_omega_b = sqrt(1.0 - (extent * extent) / dist2);
            }
            const float sin_omega_b = sqrt(1.0 - cos_omega_b * cos_omega_b);

            const vec2 cos_omega_ne = decode_cosines8(light_cos_at(n, i));
            const float sin_omega_n = sqrt(1.0 - cos_omega_ne[0] * cos_omega_ne[0]);

            const float cos_omega_x = cos_sub_clamped(sin_omega_w, cos_omega_w, sin_omega_n, cos_omega_ne[0]);
            const float sin_omega_x = sin_sub_clamped(sin_omega_w, cos_omega_w, sin_omega_n, cos_omega_ne[0]);
            const float cos_omega = cos_sub_clamped(sin_omega_x, cos_omega_x, sin_omega_b, cos_omega_b);

            importance[i] *= (cos_omega > cos_omega_ne[1]) ? (cos_omega / v_len2) : 0.0;
        }
        total_importance += importance[i];
    }
    return total_importance;
}

float calc_lnode_importance(const light_cwbvh_node_t n, const vec3 P, const vec3 P_test, out float importance[8]) {
    // pow2 quantization step stored in ch_exp (see pow2Extents() on the CPU side)
    const vec3 step = vec3(cwbvh_step(n.ch_exp, 0u), cwbvh_step(n.ch_exp, 1u), cwbvh_step(n.ch_exp, 2u));

    float total_importance = 0.0;
    for (uint i = 0; i < 8; ++i) {
        float imp = light_flux_weight(n, i);
        importance[i] = imp;
        if (((n.ch_bbox_min[0][i / 4] >> (8u * (i % 4u))) & 0xffu) != 0xff ||
            ((n.ch_bbox_max[0][i / 4] >> (8u * (i % 4u))) & 0xffu) != 0) {
            const vec3 bbox_min = vec3(n.bbox_min[0] + float((n.ch_bbox_min[0][i / 4] >> (8u * (i % 4u))) & 0xffu) * step[0],
                                       n.bbox_min[1] + float((n.ch_bbox_min[1][i / 4] >> (8u * (i % 4u))) & 0xffu) * step[1],
                                       n.bbox_min[2] + float((n.ch_bbox_min[2][i / 4] >> (8u * (i % 4u))) & 0xffu) * step[2]);
            const vec3 bbox_max = vec3(n.bbox_min[0] + float((n.ch_bbox_max[0][i / 4] >> (8u * (i % 4u))) & 0xffu) * step[0],
                                       n.bbox_min[1] + float((n.ch_bbox_max[1][i / 4] >> (8u * (i % 4u))) & 0xffu) * step[1],
                                       n.bbox_min[2] + float((n.ch_bbox_max[2][i / 4] >> (8u * (i % 4u))) & 0xffu) * step[2]);
            const float extent = 0.5 * length(bbox_max - bbox_min);

            const vec3 pc = 0.5 * (bbox_min + bbox_max);
            vec3 wi = P - pc;
            const float dist2 = dot(wi, wi);
            const float dist = sqrt(dist2);
            wi /= dist;

            const float v_len2 = max(dist2, extent);

            const vec3 axis = decode_oct_dir8(light_axis_at(n, i));
            const float cos_omega_w = dot(axis, wi);
            const float sin_omega_w = sqrt(1.0 - cos_omega_w * cos_omega_w);

            float cos_omega_b = -1.0;
            if (dist2 >= extent * extent) {
                cos_omega_b = sqrt(1.0 - (extent * extent) / dist2);
            }
            const float sin_omega_b = sqrt(1.0 - cos_omega_b * cos_omega_b);

            const vec2 cos_omega_ne = decode_cosines8(light_cos_at(n, i));
            const float sin_omega_n = sqrt(1.0 - cos_omega_ne[0] * cos_omega_ne[0]);

            const float cos_omega_x = cos_sub_clamped(sin_omega_w, cos_omega_w, sin_omega_n, cos_omega_ne[0]);
            const float sin_omega_x = sin_sub_clamped(sin_omega_w, cos_omega_w, sin_omega_n, cos_omega_ne[0]);
            const float cos_omega = cos_sub_clamped(sin_omega_x, cos_omega_x, sin_omega_b, cos_omega_b);

            const float mul = (cos_omega > cos_omega_ne[1]) ? (cos_omega / v_len2) : 0.0;
            importance[i] = imp = (imp * mul);
            if (!bbox_test(P_test, bbox_min, bbox_max)) {
                // NOTE: total_importance must not account for this!
                importance[i] = 0.0;
            }
        }
        total_importance += imp;
    }
    return total_importance;
}

float calc_lnode_importance(const light_cwbvh_node_t n, const vec3 P, const vec3 inv_d, const vec3 neg_inv_d_o, const float t, const uint accept_mask, out float importance[8]) {
    // pow2 quantization step stored in ch_exp (see pow2Extents() on the CPU side)
    const vec3 step = vec3(cwbvh_step(n.ch_exp, 0u), cwbvh_step(n.ch_exp, 1u), cwbvh_step(n.ch_exp, 2u));

    float total_importance = 0.0;
    for (uint i = 0; i < 4; ++i) {
        float imp = light_flux_weight(n, i);
        importance[i] = imp;
        if (((n.ch_bbox_min[0][0] >> (8u * i)) & 0xffu) != 0xff ||
            ((n.ch_bbox_max[0][0] >> (8u * i)) & 0xffu) != 0) {
            const vec3 bbox_min = vec3(n.bbox_min[0] + float((n.ch_bbox_min[0][0] >> (8u * i)) & 0xffu) * step[0],
                                       n.bbox_min[1] + float((n.ch_bbox_min[1][0] >> (8u * i)) & 0xffu) * step[1],
                                       n.bbox_min[2] + float((n.ch_bbox_min[2][0] >> (8u * i)) & 0xffu) * step[2]);
            const vec3 bbox_max = vec3(n.bbox_min[0] + float((n.ch_bbox_max[0][0] >> (8u * i)) & 0xffu) * step[0],
                                       n.bbox_min[1] + float((n.ch_bbox_max[1][0] >> (8u * i)) & 0xffu) * step[1],
                                       n.bbox_min[2] + float((n.ch_bbox_max[2][0] >> (8u * i)) & 0xffu) * step[2]);
            const float extent = 0.5 * length(bbox_max - bbox_min);

            const vec3 pc = 0.5 * (bbox_min + bbox_max);
            vec3 wi = P - pc;
            const float dist2 = dot(wi, wi);
            const float dist = sqrt(dist2);
            wi /= dist;

            const float v_len2 = max(dist2, extent);

            const vec3 axis = decode_oct_dir8(light_axis_at(n, i));
            const float cos_omega_w = dot(axis, wi);
            const float sin_omega_w = sqrt(1.0 - cos_omega_w * cos_omega_w);

            float cos_omega_b = -1.0;
            if (dist2 >= extent * extent) {
                cos_omega_b = sqrt(1.0 - (extent * extent) / dist2);
            }
            const float sin_omega_b = sqrt(1.0 - cos_omega_b * cos_omega_b);

            const vec2 cos_omega_ne = decode_cosines8(light_cos_at(n, i));
            const float sin_omega_n = sqrt(1.0 - cos_omega_ne[0] * cos_omega_ne[0]);

            const float cos_omega_x = cos_sub_clamped(sin_omega_w, cos_omega_w, sin_omega_n, cos_omega_ne[0]);
            const float sin_omega_x = sin_sub_clamped(sin_omega_w, cos_omega_w, sin_omega_n, cos_omega_ne[0]);
            const float cos_omega = cos_sub_clamped(sin_omega_x, cos_omega_x, sin_omega_b, cos_omega_b);

            const float mul = (cos_omega > cos_omega_ne[1]) ? (cos_omega / v_len2) : 0.0;
            importance[i] = imp = (imp * mul);
            if (!bbox_test(inv_d, neg_inv_d_o, t, bbox_min, bbox_max)) {
                // NOTE: total_importance must not account for this!
                importance[i] = 0.0;
            }
            importance[i] *= float((n.ch_bitmask[0] & (accept_mask << (8u * i))) != 0);
        }
        total_importance += imp;
    }
    for (uint i = 4; i < 8; ++i) {
        float imp = light_flux_weight(n, i);
        importance[i] = imp;
        if (((n.ch_bbox_min[0][1] >> (8u * (i - 4u))) & 0xffu) != 0xff ||
            ((n.ch_bbox_max[0][1] >> (8u * (i - 4u))) & 0xffu) != 0) {
            const vec3 bbox_min = vec3(n.bbox_min[0] + float((n.ch_bbox_min[0][1] >> (8u * (i - 4u))) & 0xffu) * step[0],
                                       n.bbox_min[1] + float((n.ch_bbox_min[1][1] >> (8u * (i - 4u))) & 0xffu) * step[1],
                                       n.bbox_min[2] + float((n.ch_bbox_min[2][1] >> (8u * (i - 4u))) & 0xffu) * step[2]);
            const vec3 bbox_max = vec3(n.bbox_min[0] + float((n.ch_bbox_max[0][1] >> (8u * (i - 4u))) & 0xffu) * step[0],
                                       n.bbox_min[1] + float((n.ch_bbox_max[1][1] >> (8u * (i - 4u))) & 0xffu) * step[1],
                                       n.bbox_min[2] + float((n.ch_bbox_max[2][1] >> (8u * (i - 4u))) & 0xffu) * step[2]);
            const float extent = 0.5 * length(bbox_max - bbox_min);

            const vec3 pc = 0.5 * (bbox_min + bbox_max);
            vec3 wi = P - pc;
            const float dist2 = dot(wi, wi);
            const float dist = sqrt(dist2);
            wi /= dist;

            const float v_len2 = max(dist2, extent);

            const vec3 axis = decode_oct_dir8(light_axis_at(n, i));
            const float cos_omega_w = dot(axis, wi);
            const float sin_omega_w = sqrt(1.0 - cos_omega_w * cos_omega_w);

            float cos_omega_b = -1.0;
            if (dist2 >= extent * extent) {
                cos_omega_b = sqrt(1.0 - (extent * extent) / dist2);
            }
            const float sin_omega_b = sqrt(1.0 - cos_omega_b * cos_omega_b);

            const vec2 cos_omega_ne = decode_cosines8(light_cos_at(n, i));
            const float sin_omega_n = sqrt(1.0 - cos_omega_ne[0] * cos_omega_ne[0]);

            const float cos_omega_x = cos_sub_clamped(sin_omega_w, cos_omega_w, sin_omega_n, cos_omega_ne[0]);
            const float sin_omega_x = sin_sub_clamped(sin_omega_w, cos_omega_w, sin_omega_n, cos_omega_ne[0]);
            const float cos_omega = cos_sub_clamped(sin_omega_x, cos_omega_x, sin_omega_b, cos_omega_b);

            const float mul = (cos_omega > cos_omega_ne[1]) ? (cos_omega / v_len2) : 0.0;
            importance[i] = imp = (imp * mul);
            if (!bbox_test(inv_d, neg_inv_d_o, t, bbox_min, bbox_max)) {
                // NOTE: total_importance must not account for this!
                importance[i] = 0.0;
            }
            importance[i] *= float((n.ch_bitmask[1] & (accept_mask << (8u * (i - 4u)))) != 0);
        }
        total_importance += imp;
    }
    return total_importance;
}

#endif // LIGHT_BVH_GLSL