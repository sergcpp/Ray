#include "CoreRef.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <limits>

#include "TextureAtlasRef.h"

//
// Useful macros for debugging
//
#define USE_VNDF_GGX_SAMPLING 1
#define USE_NEE 1

namespace Ray {
namespace Ref {
force_inline void _IntersectTri(const ray_packet_t &r, const tri_accel2_t &tri, const uint32_t i, hit_data_t &inter) {
#define _sign_of(f) (((f) >= 0) ? 1 : -1)
#define _dot(x, y) ((x)[0] * (y)[0] + (x)[1] * (y)[1] + (x)[2] * (y)[2])

    const float det = _dot(r.d, tri.n_plane);
    const float dett = tri.n_plane[3] - _dot(r.o, tri.n_plane);
    if (_sign_of(dett) != _sign_of(det * inter.t - dett)) {
        return;
    }

    const float p[3] = {det * r.o[0] + dett * r.d[0], det * r.o[1] + dett * r.d[1], det * r.o[2] + dett * r.d[2]};

    const float detu = _dot(p, tri.u_plane) + det * tri.u_plane[3];
    if (_sign_of(detu) != _sign_of(det - detu)) {
        return;
    }

    const float detv = _dot(p, tri.v_plane) + det * tri.v_plane[3];
    if (_sign_of(detv) != _sign_of(det - detu - detv)) {
        return;
    }

    const float rdet = (1.0f / det);

    inter.mask_values[0] = 0xffffffff;
    inter.prim_indices[0] = (det < 0.0f) ? int(i) : -int(i);
    inter.t = dett * rdet;
    inter.u = detu * rdet;
    inter.v = detv * rdet;

#undef _dot
#undef _sign_of
}

force_inline uint32_t near_child(const ray_packet_t &r, const bvh_node_t &node) {
    return r.d[node.prim_count >> 30] < 0 ? (node.right_child & RIGHT_CHILD_BITS) : node.left_child;
}

force_inline uint32_t far_child(const ray_packet_t &r, const bvh_node_t &node) {
    return r.d[node.prim_count >> 30] < 0 ? node.left_child : (node.right_child & RIGHT_CHILD_BITS);
}

force_inline uint32_t other_child(const bvh_node_t &node, const uint32_t cur_child) {
    return (node.left_child == cur_child) ? (node.right_child & RIGHT_CHILD_BITS) : node.left_child;
}

force_inline bool is_leaf_node(const bvh_node_t &node) { return (node.prim_index & LEAF_NODE_BIT) != 0; }

force_inline bool is_leaf_node(const mbvh_node_t &node) { return (node.child[0] & LEAF_NODE_BIT) != 0; }

force_inline bool bbox_test(const float o[3], const float inv_d[3], const float t, const float bbox_min[3],
                            const float bbox_max[3]) {
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

    return tmin <= tmax && tmin <= t && tmax > 0;
}

force_inline bool bbox_test(const float p[3], const float bbox_min[3], const float bbox_max[3]) {
    return p[0] > bbox_min[0] && p[0] < bbox_max[0] && p[1] > bbox_min[1] && p[1] < bbox_max[1] && p[2] > bbox_min[2] &&
           p[2] < bbox_max[2];
}

force_inline bool bbox_test(const float o[3], const float inv_d[3], const float t, const bvh_node_t &node) {
    return bbox_test(o, inv_d, t, node.bbox_min, node.bbox_max);
}

force_inline bool bbox_test(const float p[3], const bvh_node_t &node) {
    return bbox_test(p, node.bbox_min, node.bbox_max);
}

force_inline bool bbox_test_oct(const float p[3], const mbvh_node_t &node, const int i) {
    return p[0] > node.bbox_min[0][i] && p[0] < node.bbox_max[0][i] && p[1] > node.bbox_min[1][i] &&
           p[1] < node.bbox_max[1][i] && p[2] > node.bbox_min[2][i] && p[2] < node.bbox_max[2][i];
}

force_inline void bbox_test_oct(const float o[3], const float inv_d[3], const mbvh_node_t &node, int res[8],
                                float dist[8]){ITERATE_8({
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
})}

force_inline
    int bbox_test_oct(const float o[3], const float inv_d[3], const float t, const mbvh_node_t &node, float dist[8]) {
    int mask = 0;

    ITERATE_8({
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
        mask |= ((tmin <= tmax && tmin <= t && tmax > 0) ? 1 : 0) << i;
    })

    return mask;
}

enum eTraversalSource { FromParent, FromChild, FromSibling };

struct stack_entry_t {
    uint32_t index;
    float dist;
};

template <int StackSize> class TraversalStack {
  public:
    stack_entry_t stack[StackSize];
    uint32_t stack_size = 0;

    force_inline void push(const uint32_t index, const float dist) {
        stack[stack_size++] = {index, dist};
        assert(stack_size < StackSize && "Traversal stack overflow!");
    }

    force_inline stack_entry_t pop() { return stack[--stack_size]; }

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
                stack_entry_t tmp = stack[i];
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
                const stack_entry_t tmp = stack[i];
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

    void sort_topN(int count) {
        assert(stack_size >= uint32_t(count));
        const int start = int(stack_size - count);

        for (int i = start + 1; i < int(stack_size); ++i) {
            const stack_entry_t key = stack[i];

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

force_inline int hash(int x) {
    unsigned ret = reinterpret_cast<const unsigned &>(x);
    ret = ((ret >> 16) ^ ret) * 0x45d9f3b;
    ret = ((ret >> 16) ^ ret) * 0x45d9f3b;
    ret = (ret >> 16) ^ ret;
    return reinterpret_cast<const int &>(ret);
}

force_inline void safe_invert(const float v[3], float out_v[3]) {
    if (v[0] <= FLT_EPS && v[0] >= 0) {
        out_v[0] = std::numeric_limits<float>::max();
    } else if (v[0] >= -FLT_EPS && v[0] < 0) {
        out_v[0] = -std::numeric_limits<float>::max();
    } else {
        out_v[0] = 1.0f / v[0];
    }

    if (v[1] <= FLT_EPS && v[1] >= 0) {
        out_v[1] = std::numeric_limits<float>::max();
    } else if (v[1] >= -FLT_EPS && v[1] < 0) {
        out_v[1] = -std::numeric_limits<float>::max();
    } else {
        out_v[1] = 1.0f / v[1];
    }

    if (v[2] <= FLT_EPS && v[2] >= 0) {
        out_v[2] = std::numeric_limits<float>::max();
    } else if (v[2] >= -FLT_EPS && v[2] < 0) {
        out_v[2] = -std::numeric_limits<float>::max();
    } else {
        out_v[2] = 1.0f / v[2];
    }
}

force_inline float clamp(const float val, const float min, const float max) {
    return val < min ? min : (val > max ? max : val);
}

force_inline int clamp(const int val, const int min, const int max) {
    return val < min ? min : (val > max ? max : val);
}

force_inline simd_fvec3 cross(const simd_fvec3 &v1, const simd_fvec3 &v2) {
    return simd_fvec3{v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]};
}

force_inline simd_fvec3 reflect(const simd_fvec3 &I, const simd_fvec3 &N) { return I - 2 * dot(N, I) * N; }

force_inline float pow5(const float v) { return (v * v) * (v * v) * v; }

force_inline float mix(const float v1, const float v2, const float k) { return (1.0f - k) * v1 + k * v2; }

force_inline uint32_t get_ray_hash(const ray_packet_t &r, const float root_min[3], const float cell_size[3]) {
    int x = clamp(int((r.o[0] - root_min[0]) / cell_size[0]), 0, 255),
        y = clamp(int((r.o[1] - root_min[1]) / cell_size[1]), 0, 255),
        z = clamp(int((r.o[2] - root_min[2]) / cell_size[2]), 0, 255);

    // float omega = omega_table[int(r.d[2] / 0.0625f)];
    // float std::atan2(r.d[1], r.d[0]);
    // int o = (int)(16 * omega / (PI)), p = (int)(16 * (phi + PI) / (2 * PI));

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

force_inline float construct_float(uint32_t m) {
    const uint32_t ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint32_t ieeeOne = 0x3F800000u;      // 1.0 in IEEE binary32

    m &= ieeeMantissa; // Keep only mantissa bits (fractional part)
    m |= ieeeOne;      // Add fractional part to 1.0

    const float f = reinterpret_cast<float &>(m); // Range [1:2]
    return f - 1.0f;                              // Range [0:1]
}

force_inline simd_fvec4 rgbe_to_rgb(const pixel_color8_t &rgbe) {
    const float f = std::exp2(float(rgbe.a) - 128.0f);
    return simd_fvec4{to_norm_float(rgbe.r) * f, to_norm_float(rgbe.g) * f, to_norm_float(rgbe.b) * f, 1.0f};
}

force_inline simd_fvec4 srgb_to_rgb(const simd_fvec4 &col) {
    simd_fvec4 ret;
    ITERATE_3({
        if (col[i] > 0.04045f) {
            ret[i] = std::pow((col[i] + 0.055f) / 1.055f, 2.4f);
        } else {
            ret[i] = col[i] / 12.92f;
        }
    })
    ret[3] = col[3];

    return ret;
}

force_inline float fast_log2(float val) {
    // From https://stackoverflow.com/questions/9411823/fast-log2float-x-implementation-c
    union {
        float val;
        int32_t x;
    } u = {val};
    float log_2 = float(((u.x >> 23) & 255) - 128);
    u.x &= ~(255 << 23);
    u.x += 127 << 23;
    log_2 += ((-0.34484843f) * u.val + 2.02466578f) * u.val - 0.67487759f;
    return (log_2);
}

force_inline float lum(const simd_fvec3 &color) {
    return 0.212671f * color[0] + 0.715160f * color[1] + 0.072169f * color[2];
}

force_inline float lum(const simd_fvec4 &color) {
    return 0.212671f * color[0] + 0.715160f * color[1] + 0.072169f * color[2];
}

float get_texture_lod(const texture_t &t, const simd_fvec2 &duv_dx, const simd_fvec2 &duv_dy) {
    const simd_fvec2 sz = {float(t.width & TEXTURE_WIDTH_BITS), float(t.height & TEXTURE_HEIGHT_BITS)};
    const simd_fvec2 _duv_dx = duv_dx * sz, _duv_dy = duv_dy * sz;
    const simd_fvec2 _diagonal = _duv_dx + _duv_dy;

    // Find minimal dimention of parallelogram
    const float min_length2 = std::min(std::min(_duv_dx.length2(), _duv_dy.length2()), _diagonal.length2());
    // Find lod
    float lod = fast_log2(min_length2);
    // Substruct 1 from lod to always have 4 texels for interpolation
    lod = clamp(0.5f * lod - 1.0f, 0.0f, float(MAX_MIP_LEVEL));

    return lod;
}

void get_lobe_weights(const float base_color_lum, const float spec_color_lum, const float specular,
                      const float metallic, const float transmission, const float clearcoat, float *out_diffuse_weight,
                      float *out_specular_weight, float *out_clearcoat_weight, float *out_refraction_weight) {
    // taken from Cycles
    (*out_diffuse_weight) = base_color_lum * (1.0f - metallic) * (1.0f - transmission);
    const float final_transmission = transmission * (1.0f - metallic);
    (*out_specular_weight) =
        (specular != 0.0f || metallic != 0.0f) ? spec_color_lum * (1.0f - final_transmission) : 0.0f;
    (*out_clearcoat_weight) = 0.25f * clearcoat * (1.0f - metallic);
    (*out_refraction_weight) = final_transmission * base_color_lum;

    const float total_weight =
        (*out_diffuse_weight) + (*out_specular_weight) + (*out_clearcoat_weight) + (*out_refraction_weight);
    if (total_weight != 0.0f) {
        (*out_diffuse_weight) /= total_weight;
        (*out_specular_weight) /= total_weight;
        (*out_clearcoat_weight) /= total_weight;
        (*out_refraction_weight) /= total_weight;
    }
}

force_inline float power_heuristic(const float a, const float b) {
    const float t = a * a;
    return t / (b * b + t);
}

simd_fvec3 shlick_fresnel(const simd_fvec3 &f0, float u) {
    const float f = pow5(1.0f - u);
    return f + f0 * (1.0f - f);
}

force_inline float schlick_weight(const float u) {
    const float m = clamp(1.0f - u, 0.0f, 1.0f);
    return pow5(m);
}

float fresnel_dielectric_cos(float cosi, float eta) {
    // compute fresnel reflectance without explicitly computing the refracted direction
    float c = std::abs(cosi);
    float g = eta * eta - 1 + c * c;
    float result;

    if (g > 0) {
        g = std::sqrt(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1) / (c * (g - c) + 1);
        result = 0.5f * A * A * (1 + B * B);
    } else {
        result = 1.0f; /* TIR (no refracted component) */
    }

    return result;
}

//
// From "A Fast and Robust Method for Avoiding Self-Intersection"
//

force_inline int32_t float_as_int(const float v) { return reinterpret_cast<const int32_t &>(v); }
force_inline float int_as_float(const int32_t v) { return reinterpret_cast<const float &>(v); }

simd_fvec3 offset_ray(const simd_fvec3 &p, const simd_fvec3 &n) {
    const float Origin = 1.0f / 32.0f;
    const float FloatScale = 1.0f / 65536.0f;
    const float IntScale = 256.0f;

    const simd_ivec3 of_i(IntScale * n);

    const simd_fvec3 p_i(int_as_float(float_as_int(p[0]) + ((p[0] < 0.0f) ? -of_i[0] : of_i[0])),
                         int_as_float(float_as_int(p[1]) + ((p[1] < 0.0f) ? -of_i[1] : of_i[1])),
                         int_as_float(float_as_int(p[2]) + ((p[2] < 0.0f) ? -of_i[2] : of_i[2])));

    return simd_fvec3(std::abs(p[0]) < Origin ? (p[0] + FloatScale * n[0]) : p_i[0],
                      std::abs(p[1]) < Origin ? (p[1] + FloatScale * n[1]) : p_i[1],
                      std::abs(p[2]) < Origin ? (p[2] + FloatScale * n[2]) : p_i[2]);
}

simd_fvec3 sample_GTR1(const float rgh, const float r1, const float r2) {
    const float a = std::max(0.001f, rgh);
    const float a2 = a * a;

    const float phi = r1 * (2.0f * PI);

    const float cosTheta = std::sqrt(std::max(0.0f, 1.0f - std::pow(a2, 1.0f - r2)) / (1.0f - a2));
    const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - (cosTheta * cosTheta)));
    const float sinPhi = std::sin(phi);
    const float cosPhi = std::cos(phi);

    return simd_fvec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}

simd_fvec3 SampleGGX_NDF(const float rgh, const float r1, const float r2) {
    const float a = std::max(0.001f, rgh);

    const float phi = r1 * (2.0f * PI);

    const float cosTheta = std::sqrt((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
    const float sinTheta = clamp(std::sqrt(1.0f - (cosTheta * cosTheta)), 0.0f, 1.0f);
    const float sinPhi = std::sin(phi);
    const float cosPhi = std::cos(phi);

    return simd_fvec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}

// http://jcgt.org/published/0007/04/01/paper.pdf by Eric Heitz
// Input Ve: view direction
// Input alpha_x, alpha_y: roughness parameters
// Input U1, U2: uniform random numbers
// Output Ne: normal sampled with PDF D_Ve(Ne) = G1(Ve) * max(0, dot(Ve, Ne)) * D(Ne) / Ve.z
simd_fvec3 SampleGGX_VNDF(const simd_fvec3 &Ve, float alpha_x, float alpha_y, float U1, float U2) {
    // Section 3.2: transforming the view direction to the hemisphere configuration
    const simd_fvec3 Vh = normalize(simd_fvec3(alpha_x * Ve[0], alpha_y * Ve[1], Ve[2]));
    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    const float lensq = Vh[0] * Vh[0] + Vh[1] * Vh[1];
    const simd_fvec3 T1 =
        lensq > 0.0f ? simd_fvec3(-Vh[1], Vh[0], 0.0f) / std::sqrt(lensq) : simd_fvec3(1.0f, 0.0f, 0.0f);
    const simd_fvec3 T2 = cross(Vh, T1);
    // Section 4.2: parameterization of the projected area
    const float r = std::sqrt(U1);
    const float phi = 2.0f * PI * U2;
    const float t1 = r * std::cos(phi);
    float t2 = r * std::sin(phi);
    const float s = 0.5f * (1.0f + Vh[2]);
    t2 = (1.0f - s) * std::sqrt(1.0f - t1 * t1) + s * t2;
    // Section 4.3: reprojection onto hemisphere
    const simd_fvec3 Nh = t1 * T1 + t2 * T2 + std::sqrt(std::max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
    // Section 3.4: transforming the normal back to the ellipsoid configuration
    const simd_fvec3 Ne = normalize(simd_fvec3(alpha_x * Nh[0], alpha_y * Nh[1], std::max(0.0f, Nh[2])));
    return Ne;
}

// Smith shadowing function
force_inline float G1(const simd_fvec3 &Ve, float alpha_x, float alpha_y) {
    alpha_x *= alpha_x;
    alpha_y *= alpha_y;
    const float delta =
        (-1.0f + std::sqrt(1.0f + (alpha_x * Ve[0] * Ve[0] + alpha_y * Ve[1] * Ve[1]) / (Ve[2] * Ve[2]))) / 2.0f;
    return 1.0f / (1.0f + delta);
}

float SmithG_GGX(const float N_dot_V, const float alpha_g) {
    const float a = alpha_g * alpha_g;
    const float b = N_dot_V * N_dot_V;
    return 1.0f / (N_dot_V + std::sqrt(a + b - a * b));
}

float GGX_pdf(const simd_fvec3 &N, float alpha_x, float alpha_y) {
    const float a = ((N[0] * N[0] / (alpha_x * alpha_x)) + (N[1] * N[1] / (alpha_y * alpha_y)) + N[2] * N[2]);
    return 1.0f / (PI * alpha_x * alpha_y * a * a);
}

float D_GTR1(float NDotH, float a) {
    if (a >= 1.0f) {
        return 1.0f / PI;
    }
    const float a2 = a * a;
    const float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
    return (a2 - 1.0f) / (PI * std::log(a2) * t);
}

float D_GTR2(const float N_dot_H, const float a) {
    const float a2 = a * a;
    const float t = 1.0f + (a2 - 1.0f) * N_dot_H * N_dot_H;
    return a2 / (PI * t * t);
}

float D_GGX(const simd_fvec3 &H, const float alpha_x, const float alpha_y) {
    if (H[2] == 0.0f) {
        return 0.0f;
    }
    const float sx = -H[0] / (H[2] * alpha_x);
    const float sy = -H[1] / (H[2] * alpha_y);
    const float s1 = 1.0f + sx * sx + sy * sy;
    const float cos_theta_h4 = H[2] * H[2] * H[2] * H[2];
    return 1.0f / ((s1 * s1) * PI * alpha_x * alpha_y * cos_theta_h4);
}

void create_tbn_matrix(const simd_fvec3 &N, simd_fvec3 out_TBN[3]) {
    simd_fvec3 U;
    if (std::abs(N[1]) < 0.999f) {
        U = {0.0f, 1.0f, 0.0f};
    } else {
        U = {1.0f, 0.0f, 0.0f};
    }

    simd_fvec3 T = normalize(cross(U, N));
    U = cross(N, T);

    out_TBN[0][0] = T[0];
    out_TBN[1][0] = T[1];
    out_TBN[2][0] = T[2];

    out_TBN[0][1] = U[0];
    out_TBN[1][1] = U[1];
    out_TBN[2][1] = U[2];

    out_TBN[0][2] = N[0];
    out_TBN[1][2] = N[1];
    out_TBN[2][2] = N[2];
}

void create_tbn_matrix(const simd_fvec3 &N, simd_fvec3 &T, simd_fvec3 out_TBN[3]) {
    simd_fvec3 U = normalize(cross(T, N));
    T = cross(N, U);

    out_TBN[0][0] = T[0];
    out_TBN[1][0] = T[1];
    out_TBN[2][0] = T[2];

    out_TBN[0][1] = U[0];
    out_TBN[1][1] = U[1];
    out_TBN[2][1] = U[2];

    out_TBN[0][2] = N[0];
    out_TBN[1][2] = N[1];
    out_TBN[2][2] = N[2];
}

simd_fvec3 rotate_around_axis(const simd_fvec3 &p, const simd_fvec3 &axis, const float angle) {
    const float costheta = std::cos(angle);
    const float sintheta = std::sin(angle);
    simd_fvec3 r;

    r[0] = ((costheta + (1.0f - costheta) * axis[0] * axis[0]) * p[0]) +
           (((1.0f - costheta) * axis[0] * axis[1] - axis[2] * sintheta) * p[1]) +
           (((1.0f - costheta) * axis[0] * axis[2] + axis[1] * sintheta) * p[2]);

    r[1] = (((1.0f - costheta) * axis[0] * axis[1] + axis[2] * sintheta) * p[0]) +
           ((costheta + (1.0f - costheta) * axis[1] * axis[1]) * p[1]) +
           (((1.0f - costheta) * axis[1] * axis[2] - axis[0] * sintheta) * p[2]);

    r[2] = (((1.0f - costheta) * axis[0] * axis[2] - axis[1] * sintheta) * p[0]) +
           (((1.0f - costheta) * axis[1] * axis[2] + axis[0] * sintheta) * p[1]) +
           ((costheta + (1.0f - costheta) * axis[2] * axis[2]) * p[2]);

    return r;
}

void transpose(const simd_fvec3 in_3x3[3], simd_fvec3 out_3x3[3]) {
    out_3x3[0][0] = in_3x3[0][0];
    out_3x3[0][1] = in_3x3[1][0];
    out_3x3[0][2] = in_3x3[2][0];

    out_3x3[1][0] = in_3x3[0][1];
    out_3x3[1][1] = in_3x3[1][1];
    out_3x3[1][2] = in_3x3[2][1];

    out_3x3[2][0] = in_3x3[0][2];
    out_3x3[2][1] = in_3x3[1][2];
    out_3x3[2][2] = in_3x3[2][2];
}

simd_fvec3 mul(const simd_fvec3 in_mat[3], const simd_fvec3 &in_vec) {
    simd_fvec3 out_vec;
    out_vec[0] = in_mat[0][0] * in_vec[0] + in_mat[1][0] * in_vec[1] + in_mat[2][0] * in_vec[2];
    out_vec[1] = in_mat[0][1] * in_vec[0] + in_mat[1][1] * in_vec[1] + in_mat[2][1] * in_vec[2];
    out_vec[2] = in_mat[0][2] * in_vec[0] + in_mat[1][2] * in_vec[1] + in_mat[2][2] * in_vec[2];
    return out_vec;
}

} // namespace Ref
} // namespace Ray

Ray::Ref::hit_data_t::hit_data_t() {
    mask_values[0] = 0;
    obj_indices[0] = -1;
    prim_indices[0] = -1;
    t = MAX_DIST;
}

void Ray::Ref::GeneratePrimaryRays(int iteration, const camera_t &cam, const rect_t &r, const int w, const int h,
                                   const float *halton, aligned_vector<ray_packet_t> &out_rays) {
    const simd_fvec3 cam_origin = {cam.origin}, fwd = {cam.fwd}, side = {cam.side}, up = {cam.up};
    const float focus_distance = cam.focus_distance;

    const float k = float(w) / h;
    const float fov_k = std::tan(0.5f * cam.fov * PI / 180.0f) * focus_distance;

    auto get_pix_dir = [k, fov_k, focus_distance, cam_origin, fwd, side, up, w, h](const float x, const float y,
                                                                                   const simd_fvec3 &origin) {
        simd_fvec3 p(2 * fov_k * float(x) / w - fov_k, 2 * fov_k * float(-y) / h + fov_k, focus_distance);
        p = cam_origin + k * p[0] * side + p[1] * up + p[2] * fwd;
        return normalize(p - origin);
    };

    size_t i = 0;
    out_rays.resize(size_t(r.w) * r.h);

    for (int y = r.y; y < r.y + r.h; y += RayPacketDimY) {
        for (int x = r.x; x < r.x + r.w; x += RayPacketDimX) {
            ray_packet_t &out_r = out_rays[i++];

            float _x = float(x);
            float _y = float(y);

            const int index = y * w + x;
            const int hash_val = hash(index);

            const float sample_off[2] = {construct_float(hash_val), construct_float(hash(hash_val))};

            float _unused;
            if (cam.filter == Tent) {
                float rx = std::modf(halton[RAND_DIM_FILTER_U] + sample_off[0], &_unused);
                if (rx < 0.5f) {
                    rx = std::sqrt(2.0f * rx) - 1.0f;
                } else {
                    rx = 1.0f - std::sqrt(2.0f - 2 * rx);
                }

                float ry = std::modf(halton[RAND_DIM_FILTER_V] + sample_off[1], &_unused);
                if (ry < 0.5f) {
                    ry = std::sqrt(2.0f * ry) - 1.0f;
                } else {
                    ry = 1.0f - std::sqrt(2.0f - 2.0f * ry);
                }

                _x += 0.5f + rx;
                _y += 0.5f + ry;
            } else {
                _x += std::modf(halton[RAND_DIM_FILTER_U] + sample_off[0], &_unused);
                _y += std::modf(halton[RAND_DIM_FILTER_V] + sample_off[1], &_unused);
            }

            const float ff1 = cam.focus_factor * (-0.5f + std::modf(halton[RAND_DIM_LENS_U] + sample_off[0], &_unused));
            const float ff2 = cam.focus_factor * (-0.5f + std::modf(halton[RAND_DIM_LENS_V] + sample_off[1], &_unused));

            const simd_fvec3 _origin = cam_origin + side * ff1 + up * ff2;

            const simd_fvec3 _d = get_pix_dir(_x, _y, _origin);

            const simd_fvec3 _dx = get_pix_dir(_x + 1, _y, _origin), _dy = get_pix_dir(_x, _y + 1, _origin);

            for (int j = 0; j < 3; j++) {
                out_r.o[j] = _origin[j];
                out_r.d[j] = _d[j];
                out_r.c[j] = 1.0f;

                out_r.do_dx[j] = 0;
                out_r.dd_dx[j] = _dx[j] - _d[j];
                out_r.do_dy[j] = 0;
                out_r.dd_dy[j] = _dy[j] - _d[j];
            }

            out_r.xy = (x << 16) | y;
            out_r.ray_depth = 0;
        }
    }
}

void Ray::Ref::SampleMeshInTextureSpace(const int iteration, const int obj_index, const int uv_layer,
                                        const mesh_t &mesh, const transform_t &tr, const uint32_t *vtx_indices,
                                        const vertex_t *vertices, const rect_t &r, const int width, const int height,
                                        const float *halton, aligned_vector<ray_packet_t> &out_rays,
                                        aligned_vector<hit_data_t> &out_inters) {
    out_rays.resize(size_t(r.w) * r.h);
    out_inters.resize(out_rays.size());

    for (int y = r.y; y < r.y + r.h; y += RayPacketDimY) {
        for (int x = r.x; x < r.x + r.w; x += RayPacketDimX) {
            const int i = (y - r.y) * r.w + (x - r.x);

            ray_packet_t &out_ray = out_rays[i];
            hit_data_t &out_inter = out_inters[i];

            out_ray.xy = (x << 16) | y;
            out_ray.c[0] = out_ray.c[1] = out_ray.c[2] = 1.0f;
            out_inter.mask_values[0] = 0;
            out_inter.xy = out_ray.xy;
        }
    }

    const simd_ivec2 irect_min = {r.x, r.y}, irect_max = {r.x + r.w - 1, r.y + r.h - 1};
    const simd_fvec2 size = {float(width), float(height)};

    for (uint32_t tri = mesh.tris_index; tri < mesh.tris_index + mesh.tris_count; tri++) {
        const vertex_t &v0 = vertices[vtx_indices[tri * 3 + 0]];
        const vertex_t &v1 = vertices[vtx_indices[tri * 3 + 1]];
        const vertex_t &v2 = vertices[vtx_indices[tri * 3 + 2]];

        const auto t0 = simd_fvec2{v0.t[uv_layer][0], 1.0f - v0.t[uv_layer][1]} * size;
        const auto t1 = simd_fvec2{v1.t[uv_layer][0], 1.0f - v1.t[uv_layer][1]} * size;
        const auto t2 = simd_fvec2{v2.t[uv_layer][0], 1.0f - v2.t[uv_layer][1]} * size;

        simd_fvec2 bbox_min = t0, bbox_max = t0;

        bbox_min = min(bbox_min, t1);
        bbox_min = min(bbox_min, t2);

        bbox_max = max(bbox_max, t1);
        bbox_max = max(bbox_max, t2);

        simd_ivec2 ibbox_min = simd_ivec2{bbox_min},
                   ibbox_max = simd_ivec2{int(std::round(bbox_max[0])), int(std::round(bbox_max[1]))};

        if (ibbox_max[0] < irect_min[0] || ibbox_max[1] < irect_min[1] || ibbox_min[0] > irect_max[0] ||
            ibbox_min[1] > irect_max[1]) {
            continue;
        }

        ibbox_min = max(ibbox_min, irect_min);
        ibbox_max = min(ibbox_max, irect_max);

        const simd_fvec2 d01 = t0 - t1, d12 = t1 - t2, d20 = t2 - t0;

        const float area = d01[0] * d20[1] - d20[0] * d01[1];
        if (area < FLT_EPS) {
            continue;
        }

        const float inv_area = 1.0f / area;

        for (int y = ibbox_min[1]; y <= ibbox_max[1]; ++y) {
            for (int x = ibbox_min[0]; x <= ibbox_max[0]; ++x) {
                const int i = (y - r.y) * r.w + (x - r.x);
                ray_packet_t &out_ray = out_rays[i];
                hit_data_t &out_inter = out_inters[i];

                if (out_inter.mask_values[0]) {
                    continue;
                }

                const int index = y * width + x;
                const int hi = (iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT;

                const int hash_val = hash(index);

                float _unused;
                const float _x = float(x) + std::modf(halton[hi + 0] + construct_float(hash_val), &_unused);
                const float _y = float(y) + std::modf(halton[hi + 1] + construct_float(hash(hash_val)), &_unused);

                float u = d01[0] * (_y - t0[1]) - d01[1] * (_x - t0[0]),
                      v = d12[0] * (_y - t1[1]) - d12[1] * (_x - t1[0]),
                      w = d20[0] * (_y - t2[1]) - d20[1] * (_x - t2[0]);

                if (u >= -FLT_EPS && v >= -FLT_EPS && w >= -FLT_EPS) {
                    const simd_fvec3 p0 = {v0.p}, p1 = {v1.p}, p2 = {v2.p};
                    const simd_fvec3 n0 = {v0.n}, n1 = {v1.n}, n2 = {v2.n};

                    u *= inv_area;
                    v *= inv_area;
                    w *= inv_area;

                    const simd_fvec3 p = TransformPoint(p0 * v + p1 * w + p2 * u, tr.xform),
                                     n = TransformNormal(n0 * v + n1 * w + n2 * u, tr.inv_xform);

                    const simd_fvec3 o = p + n, d = -n;

                    memcpy(&out_ray.o[0], value_ptr(o), 3 * sizeof(float));
                    memcpy(&out_ray.d[0], value_ptr(d), 3 * sizeof(float));
                    out_ray.do_dx[0] = out_ray.do_dx[1] = out_ray.do_dx[2] = 0.0f;
                    out_ray.dd_dx[0] = out_ray.dd_dx[1] = out_ray.dd_dx[2] = 0.0f;
                    out_ray.do_dy[0] = out_ray.do_dy[1] = out_ray.do_dy[2] = 0.0f;
                    out_ray.dd_dy[0] = out_ray.dd_dy[1] = out_ray.dd_dy[2] = 0.0f;
                    out_ray.ray_depth = 0;

                    out_inter.mask_values[0] = 0xffffffff;
                    out_inter.prim_indices[0] = tri;
                    out_inter.obj_indices[0] = obj_index;
                    out_inter.t = 1.0f;
                    out_inter.u = w;
                    out_inter.v = u;
                }
            }
        }
    }
}

void Ray::Ref::SortRays_CPU(ray_packet_t *rays, const size_t rays_count, const float root_min[3],
                            const float cell_size[3], uint32_t *hash_values, uint32_t *scan_values, ray_chunk_t *chunks,
                            ray_chunk_t *chunks_temp) {
    // From "Fast Ray Sorting and Breadth-First Packet Traversal for GPU Ray Tracing" [2010]

    // compute ray hash values
    for (size_t i = 0; i < rays_count; ++i) {
        hash_values[i] = get_ray_hash(rays[i], root_min, cell_size);
    }

    size_t chunks_count = 0;

    // compress codes into spans of indentical values (makes sorting stage faster)
    for (uint32_t start = 0, end = 1; end <= (uint32_t)rays_count; end++) {
        if (end == (uint32_t)rays_count || (hash_values[start] != hash_values[end])) {
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
    for (uint32_t i = 0; i < uint32_t(rays_count); ++i) {
        uint32_t j;
        while (i != (j = scan_values[i])) {
            const int k = scan_values[j];
            std::swap(rays[j], rays[k]);
            std::swap(scan_values[i], scan_values[j]);
        }
    }
}

void Ray::Ref::SortRays_GPU(ray_packet_t *rays, const size_t rays_count, const float root_min[3],
                            const float cell_size[3], uint32_t *hash_values, int *head_flags, uint32_t *scan_values,
                            ray_chunk_t *chunks, ray_chunk_t *chunks_temp, uint32_t *skeleton) {
    // From "Fast Ray Sorting and Breadth-First Packet Traversal for GPU Ray Tracing" [2010]

    // compute ray hash values
    for (size_t i = 0; i < rays_count; ++i) {
        hash_values[i] = get_ray_hash(rays[i], root_min, cell_size);
    }

    // set head flags
    head_flags[0] = 1;
    for (size_t i = 1; i < rays_count; ++i) {
        head_flags[i] = hash_values[i] != hash_values[i - 1];
    }

    size_t chunks_count = 0;

    { // perform exclusive scan on head flags
        uint32_t cur_sum = 0;
        for (size_t i = 0; i < rays_count; ++i) {
            scan_values[i] = cur_sum;
            cur_sum += head_flags[i];
        }
        chunks_count = cur_sum;
    }

    // init Ray chunks hash and base index
    for (size_t i = 0; i < rays_count; ++i) {
        if (head_flags[i]) {
            chunks[scan_values[i]].hash = hash_values[i];
            chunks[scan_values[i]].base = (uint32_t)i;
        }
    }

    // init ray chunks size
    if (chunks_count) {
        for (size_t i = 0; i < chunks_count - 1; ++i) {
            chunks[i].size = chunks[i + 1].base - chunks[i].base;
        }
        chunks[chunks_count - 1].size = (uint32_t)rays_count - chunks[chunks_count - 1].base;
    }

    radix_sort(&chunks[0], &chunks[0] + chunks_count, &chunks_temp[0]);

    { // perform exclusive scan on chunks size
        uint32_t cur_sum = 0;
        for (size_t i = 0; i < chunks_count; ++i) {
            scan_values[i] = cur_sum;
            cur_sum += chunks[i].size;
        }
    }

    std::fill(skeleton, skeleton + rays_count, 1);
    std::fill(head_flags, head_flags + rays_count, 0);

    // init skeleton and head flags array
    for (size_t i = 0; i < chunks_count; ++i) {
        skeleton[scan_values[i]] = chunks[i].base;
        head_flags[scan_values[i]] = 1;
    }

    { // perform a segmented scan on skeleton array
        uint32_t cur_sum = 0;
        for (size_t i = 0; i < rays_count; ++i) {
            if (head_flags[i]) {
                cur_sum = 0;
            }
            cur_sum += skeleton[i];
            scan_values[i] = cur_sum;
        }
    }

    // reorder rays
    for (uint32_t i = 0; i < uint32_t(rays_count); ++i) {
        uint32_t j;
        while (i != (j = scan_values[i])) {
            const int k = scan_values[j];
            std::swap(rays[j], rays[k]);
            std::swap(scan_values[i], scan_values[j]);
        }
    }
}

bool Ray::Ref::IntersectTris_ClosestHit(const ray_packet_t &r, const tri_accel2_t *tris, const int tri_start,
                                        const int tri_end, const int obj_index, hit_data_t &out_inter) {
    hit_data_t inter{Uninitialize};
    inter.mask_values[0] = 0;
    inter.obj_indices[0] = obj_index;
    inter.t = out_inter.t;

    for (int i = tri_start; i < tri_end; ++i) {
        _IntersectTri(r, tris[i], i, inter);
    }

    out_inter.mask_values[0] |= inter.mask_values[0];
    out_inter.obj_indices[0] = inter.mask_values[0] ? inter.obj_indices[0] : out_inter.obj_indices[0];
    out_inter.prim_indices[0] = inter.mask_values[0] ? inter.prim_indices[0] : out_inter.prim_indices[0];
    out_inter.t = inter.t; // already contains min value
    out_inter.u = inter.mask_values[0] ? inter.u : out_inter.u;
    out_inter.v = inter.mask_values[0] ? inter.v : out_inter.v;

    return inter.mask_values[0] != 0;
}

bool Ray::Ref::IntersectTris_AnyHit(const ray_packet_t &r, const tri_accel2_t *tris, const tri_mat_data_t *materials,
                                    const uint32_t *indices, const int tri_start, const int tri_end,
                                    const int obj_index, hit_data_t &out_inter) {
    hit_data_t inter{Uninitialize};
    inter.mask_values[0] = 0;
    inter.obj_indices[0] = obj_index;
    inter.t = out_inter.t;

    for (int i = tri_start; i < tri_end; ++i) {
        _IntersectTri(r, tris[i], i, inter);
        if (inter.mask_values[0] &&
            ((inter.prim_indices[0] > 0 && (materials[indices[i]].front_mi & MATERIAL_SOLID_BIT)) ||
             (inter.prim_indices[0] < 0 && (materials[indices[i]].back_mi & MATERIAL_SOLID_BIT)))) {
            break;
        }
    }

    out_inter.mask_values[0] |= inter.mask_values[0];
    out_inter.obj_indices[0] = inter.mask_values[0] ? inter.obj_indices[0] : out_inter.obj_indices[0];
    out_inter.prim_indices[0] = inter.mask_values[0] ? inter.prim_indices[0] : out_inter.prim_indices[0];
    out_inter.t = inter.t; // already contains min value
    out_inter.u = inter.mask_values[0] ? inter.u : out_inter.u;
    out_inter.v = inter.mask_values[0] ? inter.v : out_inter.v;

    return inter.mask_values[0] != 0;
}

#ifdef USE_STACKLESS_BVH_TRAVERSAL
bool Ray::Ref::Traverse_MacroTree_Stackless_CPU(const ray_packet_t &r, const bvh_node_t *nodes, uint32_t root_index,
                                                const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                                const mesh_t *meshes, const transform_t *transforms,
                                                const tri_accel2_t *tris, const uint32_t *tri_indices,
                                                hit_data_t &inter) {
    bool res = false;

    float inv_d[3];
    safe_invert(r.d, inv_d);

    uint32_t cur = root_index;
    eTraversalSource src = FromSibling;

    if (!is_leaf_node(nodes[root_index])) {
        cur = near_child(r, nodes[root_index]);
        src = FromParent;
    }

    while (true) {
        switch (src) {
        case FromChild:
            if (cur == root_index || cur == 0xffffffff)
                return res;
            if (cur == near_child(r, nodes[nodes[cur].parent])) {
                cur = other_child(nodes[nodes[cur].parent], cur);
                src = FromSibling;
            } else {
                cur = nodes[cur].parent;
                src = FromChild;
            }
            break;
        case FromSibling:
            if (!bbox_test(r.o, inv_d, inter.t, nodes[cur])) {
                cur = nodes[cur].parent;
                src = FromChild;
            } else if (is_leaf_node(nodes[cur])) {
                // process leaf
                for (uint32_t i = nodes[cur].prim_index; i < nodes[cur].prim_index + nodes[cur].prim_count; ++i) {
                    const mesh_instance_t &mi = mesh_instances[mi_indices[i]];
                    const mesh_t &m = meshes[mi.mesh_index];
                    const transform_t &tr = transforms[mi.tr_index];

                    if (!bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max))
                        continue;

                    ray_packet_t _r = TransformRay(r, tr.inv_xform);

                    float _inv_d[3];
                    safe_invert(_r.d, _inv_d);

                    res |= Traverse_MicroTree_Stackless_CPU(_r, _inv_d, nodes, m.node_index, tris, tri_indices,
                                                            (int)mi_indices[i], inter);
                }

                cur = nodes[cur].parent;
                src = FromChild;
            } else {
                cur = near_child(r, nodes[cur]);
                src = FromParent;
            }
            break;
        case FromParent:
            if (!bbox_test(r.o, inv_d, inter.t, nodes[cur])) {
                cur = other_child(nodes[nodes[cur].parent], cur);
                src = FromSibling;
            } else if (is_leaf_node(nodes[cur])) {
                // process leaf
                for (uint32_t i = nodes[cur].prim_index; i < nodes[cur].prim_index + nodes[cur].prim_count; ++i) {
                    const mesh_instance_t &mi = mesh_instances[mi_indices[i]];
                    const mesh_t &m = meshes[mi.mesh_index];
                    const transform_t &tr = transforms[mi.tr_index];

                    if (!bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max))
                        continue;

                    ray_packet_t _r = TransformRay(r, tr.inv_xform);

                    float _inv_d[3];
                    safe_invert(_r.d, _inv_d);

                    res |= Traverse_MicroTree_Stackless_CPU(_r, _inv_d, nodes, m.node_index, tris, tri_indices,
                                                            (int)mi_indices[i], inter);
                }

                cur = other_child(nodes[nodes[cur].parent], cur);
                src = FromSibling;
            } else {
                cur = near_child(r, nodes[cur]);
                src = FromParent;
            }
            break;
        }
    }

    return res;
}

bool Ray::Ref::Traverse_MacroTree_Stackless_GPU(const ray_packet_t &r, const bvh_node_t *nodes, uint32_t root_index,
                                                const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                                const mesh_t *meshes, const transform_t *transforms,
                                                const tri_accel2_t *tris, const uint32_t *tri_indices,
                                                hit_data_t &inter) {
    bool res = false;

    float inv_d[3];
    safe_invert(r.d, inv_d);

    uint32_t cur = root_index;
    uint32_t last = root_index;

    if (!is_leaf_node(nodes[cur])) {
        cur = near_child(r, nodes[cur]);
    }

    while (true) {
        if (cur == 0xffffffff)
            return res;

        if (is_leaf_node(nodes[cur])) {
            for (uint32_t i = nodes[cur].prim_index; i < nodes[cur].prim_index + nodes[cur].prim_count; ++i) {
                const mesh_instance_t &mi = mesh_instances[mi_indices[i]];
                const mesh_t &m = meshes[mi.mesh_index];
                const transform_t &tr = transforms[mi.tr_index];

                if (!bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max))
                    continue;

                ray_packet_t _r = TransformRay(r, tr.inv_xform);

                float _inv_d[3];
                safe_invert(_r.d, _inv_d);
                res |= Traverse_MicroTree_Stackless_GPU(_r, _inv_d, nodes, m.node_index, tris, tri_indices,
                                                        (int)mi_indices[i], inter);
            }
            last = cur;
            cur = nodes[cur].parent;
            continue;
        }

        uint32_t near = near_child(r, nodes[cur]);
        uint32_t far = far_child(r, nodes[cur]);

        if (last == far) {
            last = cur;
            cur = nodes[cur].parent;
            continue;
        }

        uint32_t try_child = (last == nodes[cur].parent) ? near : far;
        if (bbox_test(r.o, inv_d, inter.t, nodes[try_child])) {
            last = cur;
            cur = try_child;
        } else {
            if (try_child == near) {
                last = near;
            } else {
                last = cur;
                cur = nodes[cur].parent;
            }
        }
    }

    return res;
}

bool Ray::Ref::Traverse_MicroTree_Stackless_CPU(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes,
                                                uint32_t root_index, const tri_accel2_t *tris,
                                                const uint32_t *tri_indices, int obj_index, hit_data_t &inter) {
    bool res = false;

    uint32_t cur = root_index;
    eTraversalSource src = FromSibling;

    if (!is_leaf_node(nodes[root_index])) {
        cur = near_child(r, nodes[root_index]);
        src = FromParent;
    }

    while (true) {
        switch (src) {
        case FromChild:
            if (cur == root_index || cur == 0xffffffff)
                return res;
            if (cur == near_child(r, nodes[nodes[cur].parent])) {
                cur = other_child(nodes[nodes[cur].parent], cur);
                src = FromSibling;
            } else {
                cur = nodes[cur].parent;
                src = FromChild;
            }
            break;
        case FromSibling:
            if (!bbox_test(r.o, inv_d, inter.t, nodes[cur])) {
                cur = nodes[cur].parent;
                src = FromChild;
            } else if (is_leaf_node(nodes[cur])) {
                // process leaf
                res |= IntersectTris_ClosestHit(r, tris, &tri_indices[nodes[cur].prim_index], nodes[cur].prim_count,
                                                obj_index, inter);

                cur = nodes[cur].parent;
                src = FromChild;
            } else {
                cur = near_child(r, nodes[cur]);
                src = FromParent;
            }
            break;
        case FromParent:
            if (!bbox_test(r.o, inv_d, inter.t, nodes[cur])) {
                cur = other_child(nodes[nodes[cur].parent], cur);
                src = FromSibling;
            } else if (is_leaf_node(nodes[cur])) {
                // process leaf
                res |= IntersectTris_ClosestHit(r, tris, &tri_indices[nodes[cur].prim_index], nodes[cur].prim_count,
                                                obj_index, inter);

                cur = other_child(nodes[nodes[cur].parent], cur);
                src = FromSibling;
            } else {
                cur = near_child(r, nodes[cur]);
                src = FromParent;
            }
            break;
        }
    }

    return res;
}

bool Ray::Ref::Traverse_MicroTree_Stackless_GPU(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes,
                                                uint32_t root_index, const tri_accel2_t *tris, const uint32_t *indices,
                                                int obj_index, hit_data_t &inter) {
    bool res = false;

    uint32_t cur = root_index;
    uint32_t last = root_index;

    if (!is_leaf_node(nodes[root_index])) {
        cur = near_child(r, nodes[root_index]);
        // last = cur;
    }

    while (true) {
        if (cur == 0xffffffff)
            return res;

        if (is_leaf_node(nodes[cur])) {
            res |= IntersectTris_ClosestHit(r, tris, &indices[nodes[cur].prim_index], nodes[cur].prim_count, obj_index,
                                            inter);

            last = cur;
            cur = nodes[cur].parent;
            continue;
        }

        uint32_t near = near_child(r, nodes[cur]);
        uint32_t far = far_child(r, nodes[cur]);

        if (last == far) {
            last = cur;
            cur = nodes[cur].parent;
            continue;
        }

        uint32_t try_child = (last == nodes[cur].parent) ? near : far;
        if (bbox_test(r.o, inv_d, inter.t, nodes[try_child])) {
            last = cur;
            cur = try_child;
        } else {
            if (try_child == near) {
                last = near;
            } else {
                last = cur;
                cur = nodes[cur].parent;
            }
        }
    }

    return res;
}
#endif

bool Ray::Ref::Traverse_MacroTree_WithStack_ClosestHit(const ray_packet_t &r, const bvh_node_t *nodes,
                                                       uint32_t root_index, const mesh_instance_t *mesh_instances,
                                                       const uint32_t *mi_indices, const mesh_t *meshes,
                                                       const transform_t *transforms, const tri_accel2_t *tris,
                                                       const uint32_t *tri_indices, hit_data_t &inter) {
    bool res = false;

    float inv_d[3];
    safe_invert(r.d, inv_d);

    uint32_t stack[MAX_STACK_SIZE];
    uint32_t stack_size = 0;

    stack[stack_size++] = root_index;

    while (stack_size) {
        uint32_t cur = stack[--stack_size];

        if (!bbox_test(r.o, inv_d, inter.t, nodes[cur])) {
            continue;
        }

        if (!is_leaf_node(nodes[cur])) {
            stack[stack_size++] = far_child(r, nodes[cur]);
            stack[stack_size++] = near_child(r, nodes[cur]);
        } else {
            const uint32_t prim_index = (nodes[cur].prim_index & PRIM_INDEX_BITS);
            for (uint32_t i = prim_index; i < prim_index + nodes[cur].prim_count; ++i) {
                const mesh_instance_t &mi = mesh_instances[mi_indices[i]];
                const mesh_t &m = meshes[mi.mesh_index];
                const transform_t &tr = transforms[mi.tr_index];

                if (!bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max)) {
                    continue;
                }

                const ray_packet_t _r = TransformRay(r, tr.inv_xform);

                float _inv_d[3];
                safe_invert(_r.d, _inv_d);
                res |= Traverse_MicroTree_WithStack_ClosestHit(_r, _inv_d, nodes, m.node_index, tris, tri_indices,
                                                               int(mi_indices[i]), inter);
            }
        }
    }

    return res;
}

bool Ray::Ref::Traverse_MacroTree_WithStack_ClosestHit(const ray_packet_t &r, const mbvh_node_t *nodes,
                                                       uint32_t root_index, const mesh_instance_t *mesh_instances,
                                                       const uint32_t *mi_indices, const mesh_t *meshes,
                                                       const transform_t *transforms, const tri_accel2_t *tris,
                                                       const uint32_t *tri_indices, hit_data_t &inter) {
    bool res = false;

    float inv_d[3];
    safe_invert(r.d, inv_d);

    TraversalStack<MAX_STACK_SIZE> st;
    st.push(root_index, 0.0f);

    while (!st.empty()) {
        stack_entry_t cur = st.pop();

        if (cur.dist > inter.t) {
            continue;
        }

    TRAVERSE:
        if (!is_leaf_node(nodes[cur.index])) {
            float dist[8];
            long mask = bbox_test_oct(r.o, inv_d, inter.t, nodes[cur.index], dist);
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
            const uint32_t prim_index = (nodes[cur.index].child[0] & PRIM_INDEX_BITS);
            for (uint32_t i = prim_index; i < prim_index + nodes[cur.index].child[1]; ++i) {
                const mesh_instance_t &mi = mesh_instances[mi_indices[i]];
                const mesh_t &m = meshes[mi.mesh_index];
                const transform_t &tr = transforms[mi.tr_index];

                if (!bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max)) {
                    continue;
                }

                const ray_packet_t _r = TransformRay(r, tr.inv_xform);

                float _inv_d[3];
                safe_invert(_r.d, _inv_d);
                res |= Traverse_MicroTree_WithStack_ClosestHit(_r, _inv_d, nodes, m.node_index, tris, tri_indices,
                                                               int(mi_indices[i]), inter);
            }
        }
    }

    return res;
}

bool Ray::Ref::Traverse_MacroTree_WithStack_AnyHit(const ray_packet_t &r, const bvh_node_t *nodes,
                                                   const uint32_t root_index, const mesh_instance_t *mesh_instances,
                                                   const uint32_t *mi_indices, const mesh_t *meshes,
                                                   const transform_t *transforms, const tri_accel2_t *tris,
                                                   const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                                   hit_data_t &inter) {
    bool res = false;

    float inv_d[3];
    safe_invert(r.d, inv_d);

    uint32_t stack[MAX_STACK_SIZE];
    uint32_t stack_size = 0;

    stack[stack_size++] = root_index;

    while (stack_size) {
        const uint32_t cur = stack[--stack_size];

        if (!bbox_test(r.o, inv_d, inter.t, nodes[cur])) {
            continue;
        }

        if (!is_leaf_node(nodes[cur])) {
            stack[stack_size++] = far_child(r, nodes[cur]);
            stack[stack_size++] = near_child(r, nodes[cur]);
        } else {
            const uint32_t prim_index = (nodes[cur].prim_index & PRIM_INDEX_BITS);
            for (uint32_t i = prim_index; i < prim_index + nodes[cur].prim_count; ++i) {
                const mesh_instance_t &mi = mesh_instances[mi_indices[i]];
                const mesh_t &m = meshes[mi.mesh_index];
                const transform_t &tr = transforms[mi.tr_index];

                if (!bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max)) {
                    continue;
                }

                const ray_packet_t _r = TransformRay(r, tr.inv_xform);

                float _inv_d[3];
                safe_invert(_r.d, _inv_d);

                const bool hit_found = Traverse_MicroTree_WithStack_AnyHit(
                    _r, _inv_d, nodes, m.node_index, tris, materials, tri_indices, int(mi_indices[i]), inter);
                if (hit_found) {
                    const bool is_backfacing = inter.prim_indices[0] < 0;
                    const uint32_t prim_index = is_backfacing ? -inter.prim_indices[0] : inter.prim_indices[0];

                    if ((!is_backfacing && (materials[tri_indices[prim_index]].front_mi & MATERIAL_SOLID_BIT)) ||
                        (is_backfacing && (materials[tri_indices[prim_index]].back_mi & MATERIAL_SOLID_BIT))) {
                        return true;
                    }
                }
                res |= hit_found;
            }
        }
    }

    return res;
}

bool Ray::Ref::Traverse_MacroTree_WithStack_AnyHit(const ray_packet_t &r, const mbvh_node_t *nodes,
                                                   const uint32_t root_index, const mesh_instance_t *mesh_instances,
                                                   const uint32_t *mi_indices, const mesh_t *meshes,
                                                   const transform_t *transforms, const tri_accel2_t *tris,
                                                   const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                                   hit_data_t &inter) {
    bool res = false;

    const int ray_dir_oct = ((r.d[2] > 0.0f) << 2) | ((r.d[1] > 0.0f) << 1) | (r.d[0] > 0.0f);

    int child_order[8];
    ITERATE_8({ child_order[i] = i ^ ray_dir_oct; })

    float inv_d[3];
    safe_invert(r.d, inv_d);

    TraversalStack<MAX_STACK_SIZE> st;
    st.push(root_index, 0.0f);

    while (!st.empty()) {
        stack_entry_t cur = st.pop();

        if (cur.dist > inter.t) {
            continue;
        }

    TRAVERSE:
        if (!is_leaf_node(nodes[cur.index])) {
            float dist[8];
            long mask = bbox_test_oct(r.o, inv_d, inter.t, nodes[cur.index], dist);
            if (mask) {
                long i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                if (mask == 0) { // only one box was hit
                    cur.index = nodes[cur.index].child[i];
                    goto TRAVERSE;
                }

                int i2 = GetFirstBit(mask);
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
            const uint32_t prim_index = (nodes[cur.index].child[0] & PRIM_INDEX_BITS);
            for (uint32_t i = prim_index; i < prim_index + nodes[cur.index].child[1]; ++i) {
                const mesh_instance_t &mi = mesh_instances[mi_indices[i]];
                const mesh_t &m = meshes[mi.mesh_index];
                const transform_t &tr = transforms[mi.tr_index];

                if (!bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max)) {
                    continue;
                }

                ray_packet_t _r = TransformRay(r, tr.inv_xform);

                float _inv_d[3];
                safe_invert(_r.d, _inv_d);
                bool hit_found = Traverse_MicroTree_WithStack_AnyHit(_r, _inv_d, nodes, m.node_index, tris, materials,
                                                                     tri_indices, (int)mi_indices[i], inter);
                if (hit_found) {
                    const bool is_backfacing = inter.prim_indices[0] < 0;
                    const uint32_t prim_index = is_backfacing ? -inter.prim_indices[0] : inter.prim_indices[0];

                    if ((!is_backfacing && (materials[tri_indices[prim_index]].front_mi & MATERIAL_SOLID_BIT)) ||
                        (is_backfacing && (materials[tri_indices[prim_index]].back_mi & MATERIAL_SOLID_BIT))) {
                        return true;
                    }
                }
                res |= hit_found;
            }
        }
    }

    return res;
}

bool Ray::Ref::Traverse_MicroTree_WithStack_ClosestHit(const ray_packet_t &r, const float inv_d[3],
                                                       const bvh_node_t *nodes, const uint32_t root_index,
                                                       const tri_accel2_t *tris, const uint32_t *tri_indices,
                                                       int obj_index, hit_data_t &inter) {
    bool res = false;

    uint32_t stack[MAX_STACK_SIZE];
    uint32_t stack_size = 0;

    stack[stack_size++] = root_index;

    while (stack_size) {
        const uint32_t cur = stack[--stack_size];

        if (!bbox_test(r.o, inv_d, inter.t, nodes[cur]))
            continue;

        if (!is_leaf_node(nodes[cur])) {
            stack[stack_size++] = far_child(r, nodes[cur]);
            stack[stack_size++] = near_child(r, nodes[cur]);
        } else {
            const int tri_start = nodes[cur].prim_index & PRIM_INDEX_BITS, tri_end = tri_start + nodes[cur].prim_count;
            res |= IntersectTris_ClosestHit(r, tris, tri_start, tri_end, obj_index, inter);
        }
    }

    return res;
}

bool Ray::Ref::Traverse_MicroTree_WithStack_ClosestHit(const ray_packet_t &r, const float inv_d[3],
                                                       const mbvh_node_t *nodes, const uint32_t root_index,
                                                       const tri_accel2_t *tris, const uint32_t *tri_indices,
                                                       int obj_index, hit_data_t &inter) {
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
            float dist[8];
            long mask = bbox_test_oct(r.o, inv_d, inter.t, nodes[cur.index], dist);
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
            const int tri_start = nodes[cur.index].child[0] & PRIM_INDEX_BITS,
                      tri_end = tri_start + nodes[cur.index].child[1];
            res |= IntersectTris_ClosestHit(r, tris, tri_start, tri_end, obj_index, inter);
        }
    }

    return res;
}

bool Ray::Ref::Traverse_MicroTree_WithStack_AnyHit(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes,
                                                   uint32_t root_index, const tri_accel2_t *tris,
                                                   const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                                   int obj_index, hit_data_t &inter) {
    bool res = false;

    uint32_t stack[MAX_STACK_SIZE];
    uint32_t stack_size = 0;

    stack[stack_size++] = root_index;

    while (stack_size) {
        const uint32_t cur = stack[--stack_size];

        if (!bbox_test(r.o, inv_d, inter.t, nodes[cur])) {
            continue;
        }

        if (!is_leaf_node(nodes[cur])) {
            stack[stack_size++] = far_child(r, nodes[cur]);
            stack[stack_size++] = near_child(r, nodes[cur]);
        } else {
            const int tri_start = nodes[cur].prim_index & PRIM_INDEX_BITS, tri_end = tri_start + nodes[cur].prim_count;
            const bool hit_found =
                IntersectTris_AnyHit(r, tris, materials, tri_indices, tri_start, tri_end, obj_index, inter);
            if (hit_found) {
                const bool is_backfacing = inter.prim_indices[0] < 0;
                const uint32_t prim_index = is_backfacing ? -inter.prim_indices[0] : inter.prim_indices[0];

                if ((!is_backfacing && (materials[tri_indices[prim_index]].front_mi & MATERIAL_SOLID_BIT)) ||
                    (is_backfacing && (materials[tri_indices[prim_index]].back_mi & MATERIAL_SOLID_BIT))) {
                    return true;
                }
            }
            res |= hit_found;
        }
    }

    return res;
}

bool Ray::Ref::Traverse_MicroTree_WithStack_AnyHit(const ray_packet_t &r, const float inv_d[3],
                                                   const mbvh_node_t *nodes, const uint32_t root_index,
                                                   const tri_accel2_t *tris, const tri_mat_data_t *materials,
                                                   const uint32_t *tri_indices, int obj_index, hit_data_t &inter) {
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
            float dist[8];
            long mask = bbox_test_oct(r.o, inv_d, inter.t, nodes[cur.index], dist);
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
            const int tri_start = nodes[cur.index].child[0] & PRIM_INDEX_BITS,
                      tri_end = tri_start + nodes[cur.index].child[1];
            const bool hit_found =
                IntersectTris_AnyHit(r, tris, materials, tri_indices, tri_start, tri_end, obj_index, inter);
            if (hit_found) {
                const bool is_backfacing = inter.prim_indices[0] < 0;
                const uint32_t prim_index = is_backfacing ? -inter.prim_indices[0] : inter.prim_indices[0];

                if ((!is_backfacing && (materials[tri_indices[prim_index]].front_mi & MATERIAL_SOLID_BIT)) ||
                    (is_backfacing && (materials[tri_indices[prim_index]].back_mi & MATERIAL_SOLID_BIT))) {
                    return true;
                }
            }
            res |= hit_found;
        }
    }

    return res;
}

float Ray::Ref::BRDF_PrincipledDiffuse(const simd_fvec3 &V, const simd_fvec3 &N, const simd_fvec3 &L,
                                       const simd_fvec3 &H, const float roughness) {
    const float N_dot_L = dot(N, L);
    const float N_dot_V = dot(N, V);
    if (N_dot_L <= 0.0f /*|| N_dot_V <= 0.0f*/) {
        return 0.0f;
    }

    const float FL = schlick_weight(N_dot_L);
    const float FV = schlick_weight(N_dot_V);

    const float L_dot_H = dot(L, H);
    const float Fd90 = 0.5f + 2.0f * L_dot_H * L_dot_H * roughness;
    const float Fd = mix(1.0f, Fd90, FL) * mix(1.0f, Fd90, FV);

    return Fd;
}

Ray::Ref::simd_fvec4 Ray::Ref::Evaluate_OrenDiffuse_BSDF(const simd_fvec3 &V, const simd_fvec3 &N, const simd_fvec3 &L,
                                                         const float roughness, const simd_fvec4 &base_color) {
    const float sigma = roughness;
    const float div = 1.0f / (PI + ((3.0f * PI - 4.0f) / 6.0f) * sigma);

    const float a = 1.0f * div;
    const float b = sigma * div;

    ////

    float nl = std::max(dot(N, L), 0.0f);
    float nv = std::max(dot(N, V), 0.0f);
    float t = dot(L, V) - nl * nv;

    if (t > 0.0f) {
        t /= std::max(nl, nv) + FLT_MIN;
    }
    float is = nl * (a + b * t);

    simd_fvec4 diff_col = is * base_color;

    diff_col[3] = 0.5f / PI;
    return diff_col;
}

Ray::Ref::simd_fvec4 Ray::Ref::Sample_OrenDiffuse_BSDF(const simd_fvec3 world_from_tangent[3], const simd_fvec3 &N,
                                                       const simd_fvec3 &I, const float roughness,
                                                       const simd_fvec4 &base_color, const float rand_u,
                                                       const float rand_v, simd_fvec3 &out_V) {

    const float phi = 2 * PI * rand_v;

    const float cos_phi = std::cos(phi);
    const float sin_phi = std::sin(phi);

    simd_fvec3 V;
    {
        const float dir = std::sqrt(1.0f - rand_u * rand_u);
        V = simd_fvec3{dir * cos_phi, dir * sin_phi, rand_u}; // in tangent-space
    }

    out_V = mul(world_from_tangent, V);
    return Evaluate_OrenDiffuse_BSDF(-I, N, out_V, roughness, base_color);
}

Ray::Ref::simd_fvec4 Ray::Ref::Evaluate_PrincipledDiffuse_BSDF(const simd_fvec3 &V, const simd_fvec3 &N,
                                                               const simd_fvec3 &L, const float roughness,
                                                               const simd_fvec4 &base_color,
                                                               const simd_fvec4 &sheen_color,
                                                               const bool uniform_sampling) {
    float weight, pdf;
    if (uniform_sampling) {
        weight = 2 * dot(N, L);
        pdf = 0.5f / PI;
    } else {
        weight = 1.0f;
        pdf = dot(N, L) / PI;
    }

    simd_fvec3 H = normalize(L + V);
    if (dot(V, H) < 0.0f) {
        H = -H;
    }

    simd_fvec4 diff_col = base_color * (weight * BRDF_PrincipledDiffuse(V, N, L, H, roughness));

    const float FH = PI * schlick_weight(dot(L, H));
    diff_col += FH * sheen_color;

    diff_col[3] = pdf;
    return diff_col;
}

Ray::Ref::simd_fvec4 Ray::Ref::Sample_PrincipledDiffuse_BSDF(const simd_fvec3 world_from_tangent[3],
                                                             const simd_fvec3 &N, const simd_fvec3 &I,
                                                             const float roughness, const simd_fvec4 &base_color,
                                                             const simd_fvec4 &sheen_color, const bool uniform_sampling,
                                                             const float rand_u, const float rand_v,
                                                             simd_fvec3 &out_V) {
    const float phi = 2 * PI * rand_v;

    const float cos_phi = std::cos(phi);
    const float sin_phi = std::sin(phi);

    simd_fvec3 V;
    if (uniform_sampling) {
        const float dir = std::sqrt(1.0f - rand_u * rand_u);
        V = simd_fvec3{dir * cos_phi, dir * sin_phi, rand_u}; // in tangent-space
    } else {
        const float dir = std::sqrt(rand_u);
        const float k = std::sqrt(1.0f - rand_u);
        V = simd_fvec3{dir * cos_phi, dir * sin_phi, k}; // in tangent-space
    }

    out_V = mul(world_from_tangent, V);
    return Evaluate_PrincipledDiffuse_BSDF(-I, N, out_V, roughness, base_color, sheen_color, uniform_sampling);
}

Ray::Ref::simd_fvec4 Ray::Ref::Evaluate_GGXSpecular_BSDF(const simd_fvec3 &view_dir_ts,
                                                         const simd_fvec3 &sampled_normal_ts,
                                                         const simd_fvec3 &reflected_dir_ts, const float alpha_x,
                                                         const float alpha_y, const float spec_ior, const float spec_F0,
                                                         const simd_fvec4 &spec_col) {
#if USE_VNDF_GGX_SAMPLING == 1
    const float D = D_GGX(sampled_normal_ts, alpha_x, alpha_y);
#else
    const float D = D_GTR2(sampled_normal_ts[2], alpha_x);
#endif

    const float G = G1(view_dir_ts, alpha_x, alpha_y) * G1(reflected_dir_ts, alpha_x, alpha_y);

    const float FH =
        (fresnel_dielectric_cos(dot(view_dir_ts, sampled_normal_ts), spec_ior) - spec_F0) / (1.0f - spec_F0);
    simd_fvec4 F = mix(spec_col, simd_fvec4(1.0f), FH);

    const float denom = 4.0f * std::abs(view_dir_ts[2] * reflected_dir_ts[2]);
    F *= (denom != 0.0f) ? (D * G / denom) : 0.0f;

#if USE_VNDF_GGX_SAMPLING == 1
    float pdf = D * G1(view_dir_ts, alpha_x, alpha_y) * std::max(dot(view_dir_ts, sampled_normal_ts), 0.0f) /
                std::abs(view_dir_ts[2]);
    const float div = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (div != 0.0f) {
        pdf /= div;
    }
#else
    const float pdf = D * sampled_normal_ts[2] / (4.0f * dot(view_dir_ts, sampled_normal_ts));
#endif

    F *= std::max(reflected_dir_ts[2], 0.0f);
    F[3] = pdf;
    return F;
}

Ray::Ref::simd_fvec4 Ray::Ref::Sample_GGXSpecular_BSDF(const simd_fvec3 world_from_tangent[3],
                                                       const simd_fvec3 tangent_from_world[3], const simd_fvec3 &N,
                                                       const simd_fvec3 &I, const float roughness,
                                                       const float anisotropic, const float spec_ior,
                                                       const float spec_F0, const simd_fvec4 &spec_col,
                                                       const float rand_u, const float rand_v, simd_fvec3 &out_V) {
    const float roughness2 = roughness * roughness;
    const float aspect = std::sqrt(1.0f - 0.9f * anisotropic);

    const float alpha_x = roughness2 / aspect;
    const float alpha_y = roughness2 * aspect;

    if (alpha_x * alpha_y < 1e-7f) {
        const simd_fvec3 V = reflect(I, N);
        const float FH = (fresnel_dielectric_cos(dot(V, N), spec_ior) - spec_F0) / (1.0f - spec_F0);
        simd_fvec4 F = mix(spec_col, simd_fvec4(1.0f), FH);
        out_V = V;
        return simd_fvec4{F[0] * 1e6f, F[1] * 1e6f, F[2] * 1e6f, 1e6f};
    }

    const simd_fvec3 view_dir_ts = normalize(mul(tangent_from_world, -I));
#if USE_VNDF_GGX_SAMPLING == 1
    const simd_fvec3 sampled_normal_ts = SampleGGX_VNDF(view_dir_ts, alpha_x, alpha_y, rand_u, rand_v);
#else
    const simd_fvec3 sampled_normal_ts = sample_GGX_NDF(alpha_x, rand_u, rand_v);
#endif
    const simd_fvec3 reflected_dir_ts = normalize(reflect(-view_dir_ts, sampled_normal_ts));

    out_V = mul(world_from_tangent, reflected_dir_ts);
    return Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, reflected_dir_ts, alpha_x, alpha_y, spec_ior,
                                     spec_F0, spec_col);
};

Ray::Ref::simd_fvec4 Ray::Ref::Evaluate_GGXRefraction_BSDF(const simd_fvec3 &view_dir_ts,
                                                           const simd_fvec3 &sampled_normal_ts,
                                                           const simd_fvec3 &refr_dir_ts, float roughness2, float eta,
                                                           const simd_fvec4 &refr_col) {

    if (refr_dir_ts[2] >= 0.0f || view_dir_ts[2] <= 0.0f) {
        return simd_fvec4{0.0f};
    }

#if USE_VNDF_GGX_SAMPLING == 1
    const float D = D_GGX(sampled_normal_ts, roughness2, roughness2);
#else
    const float D = D_GTR2(sampled_normal_ts[2], roughness2);
#endif

    const float G1o = G1(refr_dir_ts, roughness2, roughness2);
    const float G1i = G1(view_dir_ts, roughness2, roughness2);

    const float denom = dot(refr_dir_ts, sampled_normal_ts) + dot(view_dir_ts, sampled_normal_ts) * eta;
    const float jacobian = std::max(-dot(refr_dir_ts, sampled_normal_ts), 0.0f) / (denom * denom);

    float F = D * G1i * G1o * std::max(dot(view_dir_ts, sampled_normal_ts), 0.0f) * jacobian /
              (/*-refr_dir_ts[2] */ view_dir_ts[2]);

#if USE_VNDF_GGX_SAMPLING == 1
    float pdf = D * G1o * std::max(dot(view_dir_ts, sampled_normal_ts), 0.0f) * jacobian / view_dir_ts[2];
#else
    // const float pdf = D * std::max(sampled_normal_ts[2], 0.0f) * jacobian;
    const float pdf = D * sampled_normal_ts[2] * std::max(-dot(refr_dir_ts, sampled_normal_ts), 0.0f) / denom;
#endif

    simd_fvec4 ret = F * refr_col;
    // ret *= (-refr_dir_ts[2]);
    ret[3] = pdf;
    return ret;
}

Ray::Ref::simd_fvec4 Ray::Ref::Sample_GGXRefraction_BSDF(const simd_fvec3 world_from_tangent[3],
                                                         const simd_fvec3 tangent_from_world[3], const simd_fvec3 &N,
                                                         const simd_fvec3 &I, float roughness, const float eta,
                                                         const simd_fvec4 &refr_col, const float rand_u,
                                                         const float rand_v, simd_fvec4 &out_V) {
    const float roughness2 = (roughness * roughness);
    if (roughness2 * roughness2 < 1e-7f) {
        const float cosi = -dot(I, N);
        const float cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);
        if (cost2 < 0) {
            return simd_fvec4{0.0f};
        }
        const float m = eta * cosi - std::sqrt(cost2);
        const simd_fvec3 V = normalize(eta * I + m * N);

        out_V = simd_fvec4{V[0], V[1], V[2], m};
        return simd_fvec4{1e6f, 1e6f, 1e6f, 1e6f};
    }

    const simd_fvec3 view_dir_ts = normalize(mul(tangent_from_world, -I));
#if USE_VNDF_GGX_SAMPLING == 1
    const simd_fvec3 sampled_normal_ts = SampleGGX_VNDF(view_dir_ts, roughness2, roughness2, rand_u, rand_v);
#else
    const simd_fvec3 sampled_normal_ts = sample_GGX_NDF(roughness2, rand_u, rand_v);
#endif

    const float cosi = dot(view_dir_ts, sampled_normal_ts);
    const float cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);
    if (cost2 < 0) {
        return simd_fvec4{0.0f};
    }
    const float m = eta * cosi - std::sqrt(cost2);
    const simd_fvec3 refr_dir_ts = normalize(-eta * view_dir_ts + m * sampled_normal_ts);

    const simd_fvec4 F =
        Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, refr_dir_ts, roughness2, eta, refr_col);

    const simd_fvec3 V = mul(world_from_tangent, refr_dir_ts);
    out_V = simd_fvec4{V[0], V[1], V[2], m};
    return F;
}

Ray::Ref::simd_fvec4 Ray::Ref::Evaluate_PrincipledClearcoat_BSDF(const simd_fvec3 &view_dir_ts,
                                                                 const simd_fvec3 &sampled_normal_ts,
                                                                 const simd_fvec3 &reflected_dir_ts,
                                                                 const float clearcoat_roughness2,
                                                                 const float clearcoat_ior, const float clearcoat_F0) {
    const float D = D_GTR1(sampled_normal_ts[2], clearcoat_roughness2);
    // Always assume roughness of 0.25 for clearcoat
    const float clearcoat_alpha = (0.25f * 0.25f);
    const float G =
        G1(view_dir_ts, clearcoat_alpha, clearcoat_alpha) * G1(reflected_dir_ts, clearcoat_alpha, clearcoat_alpha);

    const float FH = (fresnel_dielectric_cos(dot(reflected_dir_ts, sampled_normal_ts), clearcoat_ior) - clearcoat_F0) /
                     (1.0f - clearcoat_F0);
    float F = mix(0.04f, 1.0f, FH);

    const float denom = 4.0f * std::abs(view_dir_ts[2]) * std::abs(reflected_dir_ts[2]);
    F *= (denom != 0.0f) ? D * G / denom : 0.0f;

#if USE_VNDF_GGX_SAMPLING == 1
    float pdf = D * G1(view_dir_ts, clearcoat_alpha, clearcoat_alpha) *
                std::max(dot(view_dir_ts, sampled_normal_ts), 0.0f) / std::abs(view_dir_ts[2]);
    const float div = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (div != 0.0f) {
        pdf /= div;
    }
#else
    float pdf = D * sampled_normal_ts[2] / (4.0f * dot(view_dir_ts, sampled_normal_ts));
#endif

    F *= std::max(reflected_dir_ts[2], 0.0f);
    return simd_fvec4{0.25f * F, 0.25f * F, 0.25f * F, pdf};
}

Ray::Ref::simd_fvec4 Ray::Ref::Sample_PrincipledClearcoat_BSDF(
    const simd_fvec3 world_from_tangent[3], const simd_fvec3 tangent_from_world[3], const simd_fvec3 &N,
    const simd_fvec3 &I, const float clearcoat_roughness2, const float clearcoat_ior, const float clearcoat_F0,
    const float rand_u, const float rand_v, simd_fvec3 &out_V) {
    if (clearcoat_roughness2 * clearcoat_roughness2 < 1e-7f) {
        const simd_fvec3 V = reflect(I, N);

        const float FH = (fresnel_dielectric_cos(dot(V, N), clearcoat_ior) - clearcoat_F0) / (1.0f - clearcoat_F0);
        const float F = mix(0.04f, 1.0f, FH);

        out_V = V;
        return simd_fvec4{0.25f * F, 0.25f * F, 0.25f * F, -1.0f};
    }

    const simd_fvec3 view_dir_ts = normalize(mul(tangent_from_world, -I));
    // NOTE: GTR1 distribution is not used for sampling because Cycles does it this way (???!)
#if USE_VNDF_GGX_SAMPLING == 1
    const simd_fvec3 sampled_normal_ts =
        SampleGGX_VNDF(view_dir_ts, clearcoat_roughness2, clearcoat_roughness2, rand_u, rand_v);
#else
    const simd_fvec3 sampled_normal_ts = sample_GGX_NDF(clearcoat_roughness2, rand_u, rand_v);
#endif
    const simd_fvec3 reflected_dir_ts = normalize(reflect(-view_dir_ts, sampled_normal_ts));

    out_V = mul(world_from_tangent, reflected_dir_ts);

    simd_fvec4 F = Evaluate_PrincipledClearcoat_BSDF(view_dir_ts, sampled_normal_ts, reflected_dir_ts,
                                                     clearcoat_roughness2, clearcoat_ior, clearcoat_F0);
    if (F[3] != 0.0f) {
        F[0] /= F[3];
        F[1] /= F[3];
        F[2] /= F[3];
    }
    return F;
}

Ray::Ref::ray_packet_t Ray::Ref::TransformRay(const ray_packet_t &r, const float *xform) {
    ray_packet_t _r = r;

    _r.o[0] = xform[0] * r.o[0] + xform[4] * r.o[1] + xform[8] * r.o[2] + xform[12];
    _r.o[1] = xform[1] * r.o[0] + xform[5] * r.o[1] + xform[9] * r.o[2] + xform[13];
    _r.o[2] = xform[2] * r.o[0] + xform[6] * r.o[1] + xform[10] * r.o[2] + xform[14];

    _r.d[0] = xform[0] * r.d[0] + xform[4] * r.d[1] + xform[8] * r.d[2];
    _r.d[1] = xform[1] * r.d[0] + xform[5] * r.d[1] + xform[9] * r.d[2];
    _r.d[2] = xform[2] * r.d[0] + xform[6] * r.d[1] + xform[10] * r.d[2];

    return _r;
}

Ray::Ref::simd_fvec3 Ray::Ref::TransformPoint(const simd_fvec3 &p, const float *xform) {
    return simd_fvec3{xform[0] * p[0] + xform[4] * p[1] + xform[8] * p[2] + xform[12],
                      xform[1] * p[0] + xform[5] * p[1] + xform[9] * p[2] + xform[13],
                      xform[2] * p[0] + xform[6] * p[1] + xform[10] * p[2] + xform[14]};
}

Ray::Ref::simd_fvec3 Ray::Ref::TransformDirection(const simd_fvec3 &p, const float *xform) {
    return simd_fvec3{xform[0] * p[0] + xform[4] * p[1] + xform[8] * p[2],
                      xform[1] * p[0] + xform[5] * p[1] + xform[9] * p[2],
                      xform[2] * p[0] + xform[6] * p[1] + xform[10] * p[2]};
}

Ray::Ref::simd_fvec3 Ray::Ref::TransformNormal(const simd_fvec3 &n, const float *inv_xform) {
    return simd_fvec3{inv_xform[0] * n[0] + inv_xform[1] * n[1] + inv_xform[2] * n[2],
                      inv_xform[4] * n[0] + inv_xform[5] * n[1] + inv_xform[6] * n[2],
                      inv_xform[8] * n[0] + inv_xform[9] * n[1] + inv_xform[10] * n[2]};
}

Ray::Ref::simd_fvec2 Ray::Ref::TransformUV(const simd_fvec2 &_uv, const simd_fvec2 &tex_atlas_size, const texture_t &t,
                                           const int mip_level) {
    const simd_fvec2 pos = {float(t.pos[mip_level][0]), float(t.pos[mip_level][1])};
    simd_fvec2 size = {float(t.width & TEXTURE_WIDTH_BITS), float(t.height & TEXTURE_HEIGHT_BITS)};
    if (t.height & TEXTURE_MIPS_BIT) {
        size = {float((t.width & TEXTURE_WIDTH_BITS) >> mip_level),
                float((t.height & TEXTURE_HEIGHT_BITS) >> mip_level)};
    }
    const simd_fvec2 uv = _uv - floor(_uv);
    simd_fvec2 res = pos + uv * size + 1.0f;
    res /= tex_atlas_size;
    return res;
}

Ray::Ref::simd_fvec4 Ray::Ref::SampleNearest(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs,
                                             const int lod) {
    const simd_fvec2 atlas_size = {atlas.size_x(), atlas.size_y()};
    simd_fvec2 _uvs = TransformUV(uvs, atlas_size, t, lod);
    _uvs = _uvs * atlas_size - 0.5f;

    const int page = t.page[lod];

    const pixel_color8_t &pix = atlas.Get(page, int(_uvs[0]), int(_uvs[1]));

    return simd_fvec4{to_norm_float(pix.r), to_norm_float(pix.g), to_norm_float(pix.b), to_norm_float(pix.a)};
}

Ray::Ref::simd_fvec4 Ray::Ref::SampleBilinear(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs,
                                              const int lod) {
    const simd_fvec2 atlas_size = {atlas.size_x(), atlas.size_y()};
    simd_fvec2 _uvs = TransformUV(uvs, atlas_size, t, lod);
    _uvs = _uvs * atlas_size - 0.5f;

    const int page = t.page[lod];

    const pixel_color8_t &p00 = atlas.Get(page, int(_uvs[0]) + 0, int(_uvs[1]) + 0);
    const pixel_color8_t &p01 = atlas.Get(page, int(_uvs[0]) + 1, int(_uvs[1]) + 0);
    const pixel_color8_t &p10 = atlas.Get(page, int(_uvs[0]) + 0, int(_uvs[1]) + 1);
    const pixel_color8_t &p11 = atlas.Get(page, int(_uvs[0]) + 1, int(_uvs[1]) + 1);

    const float kx = _uvs[0] - std::floor(_uvs[0]), ky = _uvs[1] - std::floor(_uvs[1]);

    const auto p0 = simd_fvec4{p01.r * kx + p00.r * (1 - kx), p01.g * kx + p00.g * (1 - kx),
                               p01.b * kx + p00.b * (1 - kx), p01.a * kx + p00.a * (1 - kx)};

    const auto p1 = simd_fvec4{p11.r * kx + p10.r * (1 - kx), p11.g * kx + p10.g * (1 - kx),
                               p11.b * kx + p10.b * (1 - kx), p11.a * kx + p10.a * (1 - kx)};

    const float k = 1.0f / 255.0f;
    return (p1 * ky + p0 * (1.0f - ky)) * k;
}

Ray::Ref::simd_fvec4 Ray::Ref::SampleBilinear(const TextureAtlas &atlas, const simd_fvec2 &uvs, const int page) {
    const pixel_color8_t &p00 = atlas.Get(page, int(uvs[0]) + 0, int(uvs[1]) + 0);
    const pixel_color8_t &p01 = atlas.Get(page, int(uvs[0]) + 1, int(uvs[1]) + 0);
    const pixel_color8_t &p10 = atlas.Get(page, int(uvs[0]) + 0, int(uvs[1]) + 1);
    const pixel_color8_t &p11 = atlas.Get(page, int(uvs[0]) + 1, int(uvs[1]) + 1);

    const simd_fvec2 k = uvs - floor(uvs);

    const auto _p00 =
        simd_fvec4{to_norm_float(p00.r), to_norm_float(p00.g), to_norm_float(p00.b), to_norm_float(p00.a)};
    const auto _p01 =
        simd_fvec4{to_norm_float(p01.r), to_norm_float(p01.g), to_norm_float(p01.b), to_norm_float(p01.a)};
    const auto _p10 =
        simd_fvec4{to_norm_float(p10.r), to_norm_float(p10.g), to_norm_float(p10.b), to_norm_float(p10.a)};
    const auto _p11 =
        simd_fvec4{to_norm_float(p11.r), to_norm_float(p11.g), to_norm_float(p11.b), to_norm_float(p11.a)};

    const simd_fvec4 p0X = _p01 * k[0] + _p00 * (1 - k[0]);
    const simd_fvec4 p1X = _p11 * k[0] + _p10 * (1 - k[0]);

    return (p1X * k[1] + p0X * (1 - k[1]));
}

Ray::Ref::simd_fvec4 Ray::Ref::SampleTrilinear(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs,
                                               const float lod) {
    const simd_fvec4 col1 = SampleBilinear(atlas, t, uvs, (int)std::floor(lod));
    const simd_fvec4 col2 = SampleBilinear(atlas, t, uvs, (int)std::ceil(lod));

    const float k = lod - std::floor(lod);
    return col1 * (1 - k) + col2 * k;
}

Ray::Ref::simd_fvec4 Ray::Ref::SampleAnisotropic(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs,
                                                 const simd_fvec2 &duv_dx, const simd_fvec2 &duv_dy) {
    const int width = int(t.width & TEXTURE_WIDTH_BITS);
    const int height = int(t.height & TEXTURE_HEIGHT_BITS);
    const simd_fvec2 sz = {float(width), float(height)};

    const simd_fvec2 _duv_dx = abs(duv_dx * sz);
    const simd_fvec2 _duv_dy = abs(duv_dy * sz);

    const float l1 = length(_duv_dx);
    const float l2 = length(_duv_dy);

    float lod, k;
    simd_fvec2 step;

    if (l1 <= l2) {
        lod = fast_log2(std::min(_duv_dx[0], _duv_dx[1]));
        k = l1 / l2;
        step = duv_dy;
    } else {
        lod = fast_log2(std::min(_duv_dy[0], _duv_dy[1]));
        k = l2 / l1;
        step = duv_dx;
    }

    lod = clamp(lod, 0.0f, float(MAX_MIP_LEVEL));

    simd_fvec2 _uvs = uvs - step * 0.5f;

    int num = int(2.0f / k);
    num = clamp(num, 1, 4);

    step = step / float(num);

    auto res = simd_fvec4{0.0f};

    const int lod1 = int(std::floor(lod));
    const int lod2 = int(std::ceil(lod));

    const int page1 = t.page[lod1];
    const int page2 = t.page[lod2];

    const simd_fvec2 pos1 = simd_fvec2{float(t.pos[lod1][0]), float(t.pos[lod1][1])} + 0.5f;
    const simd_fvec2 size1 = {float(width >> lod1), float(t.height >> lod1)};

    const simd_fvec2 pos2 = simd_fvec2{float(t.pos[lod2][0]), float(t.pos[lod2][1])} + 0.5f;
    const simd_fvec2 size2 = {float(width >> lod2), float(t.height >> lod2)};

    const float kz = lod - std::floor(lod);

    for (int i = 0; i < num; ++i) {
        _uvs = _uvs - floor(_uvs);

        const simd_fvec2 _uvs1 = pos1 + _uvs * size1;
        res += (1 - kz) * SampleBilinear(atlas, _uvs1, page1);

        if (kz > 0.0001f) {
            const simd_fvec2 _uvs2 = pos2 + _uvs * size2;
            res += kz * SampleBilinear(atlas, _uvs2, page2);
        }

        _uvs = _uvs + step;
    }

    return res / float(num);
}

Ray::Ref::simd_fvec4 Ray::Ref::SampleLatlong_RGBE(const TextureAtlas &atlas, const texture_t &t,
                                                  const simd_fvec3 &dir) {
    const float theta = std::acos(clamp(dir[1], -1.0f, 1.0f)) / PI;
    const float r = std::sqrt(dir[0] * dir[0] + dir[2] * dir[2]);
    float u = 0.5f * std::acos(r > FLT_EPS ? clamp(dir[0] / r, -1.0f, 1.0f) : 0.0f) / PI;
    if (dir[2] < 0.0f) {
        u = 1.0f - u;
    }

    const simd_fvec2 pos = {float(t.pos[0][0]), float(t.pos[0][1])},
                     size = {float((t.width & TEXTURE_WIDTH_BITS)), float((t.height & TEXTURE_HEIGHT_BITS))};

    const simd_fvec2 uvs = pos + simd_fvec2{u, theta} * size + simd_fvec2{1.0f, 1.0f};

    const pixel_color8_t &p00 = atlas.Get(t.page[0], int(uvs[0] + 0), int(uvs[1] + 0));
    const pixel_color8_t &p01 = atlas.Get(t.page[0], int(uvs[0] + 1), int(uvs[1] + 0));
    const pixel_color8_t &p10 = atlas.Get(t.page[0], int(uvs[0] + 0), int(uvs[1] + 1));
    const pixel_color8_t &p11 = atlas.Get(t.page[0], int(uvs[0] + 1), int(uvs[1] + 1));

    const simd_fvec2 k = uvs - floor(uvs);

    const simd_fvec4 _p00 = rgbe_to_rgb(p00), _p01 = rgbe_to_rgb(p01);
    const simd_fvec4 _p10 = rgbe_to_rgb(p10), _p11 = rgbe_to_rgb(p11);

    const simd_fvec4 p0X = _p01 * k[0] + _p00 * (1 - k[0]);
    const simd_fvec4 p1X = _p11 * k[0] + _p10 * (1 - k[0]);

    return (p1X * k[1] + p0X * (1 - k[1]));
}

float Ray::Ref::ComputeVisibility(const simd_fvec3 &p, const simd_fvec3 &d, float dist, const float rand_val,
                                  int rand_hash2, const scene_data_t &sc, const uint32_t node_index,
                                  const TextureAtlas &tex_atlas) {
    ray_packet_t r;

    memcpy(&r.o[0], value_ptr(p), 3 * sizeof(float));
    memcpy(&r.d[0], value_ptr(d), 3 * sizeof(float));

    float visibility = 1.0f;

    while (dist > HIT_BIAS) {
        hit_data_t sh_inter;
        sh_inter.t = dist;

        if (sc.mnodes) {
            Traverse_MacroTree_WithStack_AnyHit(r, sc.mnodes, node_index, sc.mesh_instances, sc.mi_indices, sc.meshes,
                                                sc.transforms, sc.tris2, sc.tri_materials, sc.tri_indices, sh_inter);
        } else {
            Traverse_MacroTree_WithStack_AnyHit(r, sc.nodes, node_index, sc.mesh_instances, sc.mi_indices, sc.meshes,
                                                sc.transforms, sc.tris2, sc.tri_materials, sc.tri_indices, sh_inter);
        }
        if (!sh_inter.mask_values[0]) {
            break;
        }

        const bool is_backfacing = (sh_inter.prim_indices[0] < 0);
        const uint32_t prim_index = is_backfacing ? -sh_inter.prim_indices[0] : sh_inter.prim_indices[0];

        const uint32_t tri_index = sc.tri_indices[prim_index];

        if ((!is_backfacing && (sc.tri_materials[tri_index].front_mi & MATERIAL_SOLID_BIT)) ||
            (is_backfacing && (sc.tri_materials[tri_index].back_mi & MATERIAL_SOLID_BIT))) {
            visibility = 0.0f;
            break;
        }

        const material_t *mat = is_backfacing
                                    ? &sc.materials[sc.tri_materials[tri_index].back_mi & MATERIAL_INDEX_BITS]
                                    : &sc.materials[sc.tri_materials[tri_index].front_mi & MATERIAL_INDEX_BITS];

        const transform_t *tr = &sc.transforms[sc.mesh_instances[sh_inter.obj_indices[0]].tr_index];

        const vertex_t &v1 = sc.vertices[sc.vtx_indices[tri_index * 3 + 0]];
        const vertex_t &v2 = sc.vertices[sc.vtx_indices[tri_index * 3 + 1]];
        const vertex_t &v3 = sc.vertices[sc.vtx_indices[tri_index * 3 + 2]];

        const auto I = simd_fvec3(r.d);

        float w = 1.0f - sh_inter.u - sh_inter.v;
        // simd_fvec3 sh_N = simd_fvec3(v1.n) * w + simd_fvec3(v2.n) * sh_inter.u + simd_fvec3(v3.n) * sh_inter.v;
        const simd_fvec2 sh_uvs =
            simd_fvec2(v1.t[0]) * w + simd_fvec2(v2.t[0]) * sh_inter.u + simd_fvec2(v3.t[0]) * sh_inter.v;

        const tri_accel2_t &tri = sc.tris2[prim_index];

        simd_fvec3 sh_plane_N = {tri.n_plane};
        sh_plane_N = TransformNormal(sh_plane_N, tr->inv_xform);

        if (is_backfacing) {
            sh_plane_N = -sh_plane_N;
        }

        {
            const int sh_rand_hash = hash(rand_hash2);
            const float sh_rand_offset = construct_float(sh_rand_hash);

            float _sh_unused;
            float sh_r = std::modf(rand_val + sh_rand_offset, &_sh_unused);

            // resolve mix material
            while (mat->type == MixNode) {
                float mix_val = mat->strength;
                if (mat->textures[BASE_TEXTURE] != 0xffffffff) {
                    mix_val *= SampleBilinear(tex_atlas, sc.textures[mat->textures[BASE_TEXTURE]], sh_uvs, 0)[0];
                }

                if (sh_r > mix_val) {
                    mat = &sc.materials[mat->textures[MIX_MAT1]];
                    sh_r = (sh_r - mix_val) / (1.0f - mix_val);
                } else {
                    mat = &sc.materials[mat->textures[MIX_MAT2]];
                    sh_r = sh_r / mix_val;
                }
            }

            if (mat->type != TransparentNode) {
                visibility = 0.0f;
                break;
            }
        }

        const float t = sh_inter.t + HIT_BIAS;
        r.o[0] += r.d[0] * t;
        r.o[1] += r.d[1] * t;
        r.o[2] += r.d[2] * t;
        dist -= t;
    }

    return visibility;
}

/*void Ray::Ref::AcumulateLightContribution(const light_t &l, const simd_fvec3 &I, const simd_fvec3 &P,
                                          const simd_fvec3 &N, const simd_fvec3 &B, const simd_fvec3 &plane_N,
                                          const scene_data_t &sc, uint32_t node_index, const TextureAtlas &tex_atlas,
                                          float sigma, const float *halton, const int hi, int rand_hash2,
                                          float rand_offset, float rand_offset2, simd_fvec3 &col) {
    simd_fvec3 L = P - simd_fvec3(l.pos);
    float distance = length(L);
    float d = std::max(distance - l.radius, 0.0f);
    L /= distance;

    float _unused;
    const float z = std::modf(halton[hi + 0] + rand_offset, &_unused);

    const float dir = std::sqrt(z);
    const float phi = 2 * PI * std::modf(halton[hi + 1] + rand_offset2, &_unused);

    const float cos_phi = std::cos(phi), sin_phi = std::sin(phi);

    simd_fvec3 TT = cross(L, B);
    simd_fvec3 BB = cross(L, TT);
    const simd_fvec3 V = dir * sin_phi * BB + std::sqrt(1.0f - dir) * L + dir * cos_phi * TT;

    L = normalize(simd_fvec3(l.pos) + V * l.radius - P);

    float denom = d / l.radius + 1.0f;
    float atten = 1.0f / (denom * denom);

    atten = (atten - LIGHT_ATTEN_CUTOFF / l.brightness) / (1.0f - LIGHT_ATTEN_CUTOFF);
    atten = std::max(atten, 0.0f);

    float _dot1 = std::max(dot(L, N), 0.0f);
    float _dot2 = dot(L, simd_fvec3{l.dir});

    if (_dot1 > FLT_EPS && _dot2 > l.spot && (l.brightness * atten) > FLT_EPS) {
        float visibility = ComputeVisibility(P + HIT_BIAS * plane_N, simd_fvec3(l.pos), halton, hi, rand_hash2, sc,
                                             node_index, tex_atlas);
        col += simd_fvec3(l.col) * _dot1 * visibility * atten * BRDF_OrenNayar(L, I, N, B, sigma);
    }
}

Ray::Ref::simd_fvec3 Ray::Ref::ComputeDirectLighting(const simd_fvec3 &I, const simd_fvec3 &P, const simd_fvec3 &N,
                                                     const simd_fvec3 &B, const simd_fvec3 &plane_N, float sigma,
                                                     const float *halton, const int hi, int rand_hash, int rand_hash2,
                                                     float rand_offset, float rand_offset2, const scene_data_t &sc,
                                                     uint32_t node_index, uint32_t light_node_index,
                                                     const TextureAtlas &tex_atlas) {
    unused(rand_hash);

    simd_fvec3 col = {0.0f};

    uint32_t stack[MAX_STACK_SIZE];
    uint32_t stack_size = 0;

    if (light_node_index != 0xffffffff) {
        stack[stack_size++] = light_node_index;
    }

    if (sc.mnodes) {
        while (stack_size) {
            const uint32_t cur = stack[--stack_size];

            if (!is_leaf_node(sc.mnodes[cur])) {
                bool res[8];
                for (int i = 0; i < 8; i++) {
                    res[i] = bbox_test_oct(value_ptr(P), sc.mnodes[cur], i);
                }

                for (int i = 0; i < 8; i++) {
                    if (!res[i]) {
                        continue;
                    }
                    stack[stack_size++] = sc.mnodes[cur].child[i];
                }
            } else {
                const uint32_t prim_index = (sc.mnodes[cur].child[0] & PRIM_INDEX_BITS);
                for (uint32_t i = prim_index; i < prim_index + sc.mnodes[cur].child[1]; i++) {
                    const light_t &l = sc.lights[sc.li_indices[i]];
                    AcumulateLightContribution(l, I, P, N, B, plane_N, sc, node_index, tex_atlas, sigma, halton, hi,
                                               rand_hash2, rand_offset, rand_offset2, col);
                }
            }
        }
    } else {
        while (stack_size) {
            const uint32_t cur = stack[--stack_size];

            if (!bbox_test(value_ptr(P), sc.nodes[cur])) {
                continue;
            }

            if (!is_leaf_node(sc.nodes[cur])) {
                stack[stack_size++] = sc.nodes[cur].left_child;
                stack[stack_size++] = (sc.nodes[cur].right_child & RIGHT_CHILD_BITS);
            } else {
                const uint32_t prim_index = (sc.nodes[cur].prim_index & PRIM_INDEX_BITS);
                for (uint32_t i = prim_index; i < prim_index + sc.nodes[cur].prim_count; i++) {
                    const light_t &l = sc.lights[sc.li_indices[i]];
                    AcumulateLightContribution(l, I, P, N, B, plane_N, sc, node_index, tex_atlas, sigma, halton, hi,
                                               rand_hash2, rand_offset, rand_offset2, col);
                }
            }
        }
    }

    return col;
}*/

void Ray::Ref::ComputeDerivatives(const simd_fvec3 &I, const float t, const simd_fvec3 &do_dx, const simd_fvec3 &do_dy,
                                  const simd_fvec3 &dd_dx, const simd_fvec3 &dd_dy, const vertex_t &v1,
                                  const vertex_t &v2, const vertex_t &v3, const transform_t &tr,
                                  const simd_fvec3 &plane_N, derivatives_t &out_der) {
    // From 'Tracing Ray Differentials' [1999]

    const float dot_I_N = -dot(I, plane_N);
    const float inv_dot = std::abs(dot_I_N) < FLT_EPS ? 0.0f : 1.0f / dot_I_N;
    const float dt_dx = dot(do_dx + t * dd_dx, plane_N) * inv_dot;
    const float dt_dy = dot(do_dy + t * dd_dy, plane_N) * inv_dot;

    out_der.do_dx = (do_dx + t * dd_dx) + dt_dx * I;
    out_der.do_dy = (do_dy + t * dd_dy) + dt_dy * I;
    out_der.dd_dx = dd_dx;
    out_der.dd_dy = dd_dy;

    // From 'Physically Based Rendering: ...' book

    const simd_fvec2 duv13 = simd_fvec2(v1.t[0]) - simd_fvec2(v3.t[0]),
                     duv23 = simd_fvec2(v2.t[0]) - simd_fvec2(v3.t[0]);
    simd_fvec3 dp13 = simd_fvec3(v1.p) - simd_fvec3(v3.p), dp23 = simd_fvec3(v2.p) - simd_fvec3(v3.p);

    dp13 = TransformDirection(dp13, tr.xform);
    dp23 = TransformDirection(dp23, tr.xform);

    const float det_uv = duv13[0] * duv23[1] - duv13[1] * duv23[0];
    const float inv_det_uv = std::abs(det_uv) < FLT_EPS ? 0 : 1.0f / det_uv;
    const simd_fvec3 dpdu = (duv23[1] * dp13 - duv13[1] * dp23) * inv_det_uv;
    const simd_fvec3 dpdv = (-duv23[0] * dp13 + duv13[0] * dp23) * inv_det_uv;

    // System of equations
    simd_fvec2 A[2] = {{dpdu[0], dpdv[0]}, {dpdu[1], dpdv[1]}};
    simd_fvec2 Bx = {out_der.do_dx[0], out_der.do_dx[1]};
    simd_fvec2 By = {out_der.do_dy[0], out_der.do_dy[1]};

    if (std::abs(plane_N[0]) > std::abs(plane_N[1]) && std::abs(plane_N[0]) > std::abs(plane_N[2])) {
        A[0] = {dpdu[1], dpdv[1]};
        A[1] = {dpdu[2], dpdv[2]};
        Bx = {out_der.do_dx[1], out_der.do_dx[2]};
        By = {out_der.do_dy[1], out_der.do_dy[2]};
    } else if (std::abs(plane_N[1]) > std::abs(plane_N[2])) {
        A[1] = {dpdu[2], dpdv[2]};
        Bx = {out_der.do_dx[0], out_der.do_dx[2]};
        By = {out_der.do_dy[0], out_der.do_dy[2]};
    }

    // Kramer's rule
    const float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    const float inv_det = std::abs(det) < FLT_EPS ? 0 : 1.0f / det;

    out_der.duv_dx = simd_fvec2{A[1][1] * Bx[0] - A[0][1] * Bx[1], A[0][0] * Bx[1] - A[1][0] * Bx[0]} * inv_det;
    out_der.duv_dy = simd_fvec2{A[1][1] * By[0] - A[0][1] * By[1], A[0][0] * By[1] - A[1][0] * By[0]} * inv_det;

    // Derivative for normal
    const simd_fvec3 dn1 = simd_fvec3(v1.n) - simd_fvec3(v3.n), dn2 = simd_fvec3(v2.n) - simd_fvec3(v3.n);
    const simd_fvec3 dndu = (duv23[1] * dn1 - duv13[1] * dn2) * inv_det_uv;
    const simd_fvec3 dndv = (-duv23[0] * dn1 + duv13[0] * dn2) * inv_det_uv;

    out_der.dndx = dndu * out_der.duv_dx[0] + dndv * out_der.duv_dx[1];
    out_der.dndy = dndu * out_der.duv_dy[0] + dndv * out_der.duv_dy[1];

    out_der.ddn_dx = dot(dd_dx, plane_N) + dot(I, out_der.dndx);
    out_der.ddn_dy = dot(dd_dy, plane_N) + dot(I, out_der.dndy);
}

Ray::pixel_color_t Ray::Ref::ShadeSurface(const pass_info_t &pi, const hit_data_t &inter, const ray_packet_t &ray,
                                          const float *halton, const scene_data_t &sc, const uint32_t node_index,
                                          const uint32_t light_node_index, const TextureAtlas &tex_atlas,
                                          ray_packet_t *out_secondary_rays, int *out_secondary_rays_count) {
    if (!inter.mask_values[0]) {
        simd_fvec4 env_col = {0.0f};
        if (pi.should_add_environment()) {
            if (sc.env->env_map != 0xffffffff) {
                env_col = SampleLatlong_RGBE(tex_atlas, sc.textures[sc.env->env_map], simd_fvec3{ray.d});
                if (sc.env->env_clamp > FLT_EPS) {
                    env_col = min(env_col, simd_fvec4{sc.env->env_clamp});
                }
            }
            env_col[3] = 1.0f;
        }
        return Ray::pixel_color_t{ray.c[0] * env_col[0] * sc.env->env_col[0],
                                  ray.c[1] * env_col[1] * sc.env->env_col[1],
                                  ray.c[2] * env_col[2] * sc.env->env_col[2], env_col[3]};
    }

    const auto I = simd_fvec3(ray.d);
    const auto P = simd_fvec3(ray.o) + inter.t * I;

    const bool is_backfacing = (inter.prim_indices[0] < 0);
    const uint32_t prim_index = is_backfacing ? -inter.prim_indices[0] : inter.prim_indices[0];

    const tri_accel2_t &tri = sc.tris2[prim_index];
    const uint32_t tri_index = sc.tri_indices[prim_index];

    const material_t *mat = &sc.materials[sc.tri_materials[tri_index].front_mi & MATERIAL_INDEX_BITS];

    const transform_t *tr = &sc.transforms[sc.mesh_instances[inter.obj_indices[0]].tr_index];

    const vertex_t &v1 = sc.vertices[sc.vtx_indices[tri_index * 3 + 0]];
    const vertex_t &v2 = sc.vertices[sc.vtx_indices[tri_index * 3 + 1]];
    const vertex_t &v3 = sc.vertices[sc.vtx_indices[tri_index * 3 + 2]];

    const float w = 1.0f - inter.u - inter.v;
    simd_fvec3 N = normalize(simd_fvec3(v1.n) * w + simd_fvec3(v2.n) * inter.u + simd_fvec3(v3.n) * inter.v);
    simd_fvec2 uvs = simd_fvec2(v1.t[0]) * w + simd_fvec2(v2.t[0]) * inter.u + simd_fvec2(v3.t[0]) * inter.v;

    // simd_fvec3 P = simd_fvec3(v1.p) * w + simd_fvec3(v2.p) * inter.u + simd_fvec3(v3.p) * inter.v;
    // P = TransformPoint(P, tr->xform);

    simd_fvec3 plane_N = {tri.n_plane};
    plane_N = TransformNormal(plane_N, tr->inv_xform);

    if (is_backfacing) {
        if (sc.tri_materials[tri_index].back_mi == 0xffff) {
            return pixel_color_t{0.0f, 0.0f, 0.0f, 0.0f};
        } else {
            mat = &sc.materials[sc.tri_materials[tri_index].back_mi & MATERIAL_INDEX_BITS];
            plane_N = -plane_N;
            N = -N;
        }
    }

    derivatives_t surf_der;
    ComputeDerivatives(I, inter.t, ray.do_dx, ray.do_dy, ray.dd_dx, ray.dd_dy, v1, v2, v3, *tr, plane_N, surf_der);

    // apply normal map
    simd_fvec3 B = simd_fvec3(v1.b) * w + simd_fvec3(v2.b) * inter.u + simd_fvec3(v3.b) * inter.v;
    simd_fvec3 T = cross(B, N);

    if (mat->textures[NORMALS_TEXTURE] != 0xffffffff) {
        simd_fvec4 normals = SampleBilinear(tex_atlas, sc.textures[mat->textures[NORMALS_TEXTURE]], uvs, 0);
        normals = normals * 2.0f - 1.0f;
        N = normalize(normals[0] * T + normals[2] * N + normals[1] * B);
    }

    N = TransformNormal(N, tr->inv_xform);
    B = TransformNormal(B, tr->inv_xform);
    T = TransformNormal(T, tr->inv_xform);

    // simd_fvec3 world_from_tangent[3] = {T, B, N}, tangent_from_world[3];
    // transpose(world_from_tangent, tangent_from_world);

    simd_fvec3 world_from_tangent[3], tangent_from_world[3];

#if 0
    create_tbn_matrix(N, tangent_from_world);
#else
    // Find radial tangent in local space
    const simd_fvec3 P_ls = simd_fvec3(v1.p) * w + simd_fvec3(v2.p) * inter.u + simd_fvec3(v3.p) * inter.v;
    // rotate around Y axis by 90 degrees in 2d
    simd_fvec3 tangent = {-P_ls[2], 0.0f, P_ls[0]};
    tangent = TransformNormal(tangent, tr->inv_xform);

    if (mat->tangent_rotation != 0.0f) {
        tangent = rotate_around_axis(tangent, N, mat->tangent_rotation);
    }

    create_tbn_matrix(N, tangent, tangent_from_world);
#endif
    transpose(tangent_from_world, world_from_tangent);

    // TODO: simplify this!!!
    T = world_from_tangent[0];
    B = world_from_tangent[1];

    // used to randomize halton sequence among pixels
    const float sample_off[2] = {construct_float(hash(pi.rand_index)), construct_float(hash(hash(pi.rand_index)))};

    simd_fvec3 col = {0.0f};

    const int diff_depth = ray.ray_depth & 0x000000ff;
    const int spec_depth = (ray.ray_depth >> 8) & 0x000000ff;
    const int refr_depth = (ray.ray_depth >> 16) & 0x000000ff;
    const int transp_depth = (ray.ray_depth >> 24) & 0x000000ff;
    const int total_depth = diff_depth + spec_depth + refr_depth + transp_depth;

#if USE_NEE == 1
    if (pi.should_add_direct_light() && sc.lights2_count && mat->type != EmissiveNode) {
        float _unused;
        const float u1 = std::modf(halton[RAND_DIM_LIGHT_PICK] + sample_off[0], &_unused);
        const auto light_index = std::min(uint32_t(u1 * sc.lights2_count), sc.lights2_count - 1);
        const uint32_t ltri_index = sc.lights2[light_index].tri.index;

        const vertex_t &v1 = sc.vertices[sc.vtx_indices[ltri_index * 3 + 0]];
        const vertex_t &v2 = sc.vertices[sc.vtx_indices[ltri_index * 3 + 1]];
        const vertex_t &v3 = sc.vertices[sc.vtx_indices[ltri_index * 3 + 2]];

        const transform_t &ltr = sc.transforms[sc.lights2[light_index].xform];

        const simd_fvec3 p1 = simd_fvec3(v1.p), p2 = simd_fvec3(v2.p), p3 = simd_fvec3(v3.p);
        const simd_fvec2 uv1 = simd_fvec2(v1.t[0]), uv2 = simd_fvec2(v2.t[0]), uv3 = simd_fvec2(v3.t[0]);

        const float r1 = std::sqrt(std::modf(halton[RAND_DIM_LIGHT_U] + sample_off[0], &_unused));
        const float r2 = std::modf(halton[RAND_DIM_LIGHT_V] + sample_off[1], &_unused);

        const simd_fvec2 luvs = uv1 * (1.0f - r1) + r1 * (uv2 * (1.0f - r2) + uv3 * r2);
        const simd_fvec3 lp = TransformPoint(p1 * (1.0f - r1) + r1 * (p2 * (1.0f - r2) + p3 * r2), ltr.xform);
        simd_fvec3 light_forward = TransformDirection(cross(p2 - p1, p3 - p1), ltr.xform);
        const float tri_area = 0.5f * length(light_forward);
        light_forward = normalize(light_forward);

        const simd_fvec3 to_light = lp - P;
        const float to_light_dist = length(to_light);
        const simd_fvec3 L = (to_light / to_light_dist);

        const float N_dot_L = dot(N, L);
        const bool _is_backfacing = (N_dot_L < 0.0f);

        const float cos_theta = std::abs(dot(-L, light_forward)); // abs for doublesided light
        if (/*N_dot_L > 0.0f &&*/ cos_theta > 0.0f) {
            const float visibility = ComputeVisibility(offset_ray(P, _is_backfacing ? -plane_N : plane_N), L,
                                                       to_light_dist - 10.0f * HIT_BIAS, halton[RAND_DIM_BSDF_PICK],
                                                       hash(pi.rand_index), sc, node_index, tex_atlas);
            if (visibility > 0.0f) {
                const material_t &lmat = sc.materials[sc.tri_materials[ltri_index].front_mi & MATERIAL_INDEX_BITS];
                simd_fvec4 lcol = simd_fvec4{sc.lights2[light_index].col[0], sc.lights2[light_index].col[1],
                                             sc.lights2[light_index].col[2], 0.0f};
                if ((lmat.flags & MAT_FLAG_SKY_PORTAL) == 0) {
                    if (lmat.textures[BASE_TEXTURE] != 0xffffffff) {
                        lcol *= SampleBilinear(tex_atlas, sc.textures[lmat.textures[BASE_TEXTURE]], luvs, 0 /* lod */);
                    }
                } else {
                    if (sc.env->env_map != 0xffffffff) {
                        lcol *= SampleLatlong_RGBE(tex_atlas, sc.textures[sc.env->env_map], L);
                        if (sc.env->env_clamp > FLT_EPS) {
                            lcol = min(lcol, simd_fvec4{sc.env->env_clamp});
                        }
                    }
                    lcol[0] *= sc.env->env_col[0];
                    lcol[1] *= sc.env->env_col[1];
                    lcol[2] *= sc.env->env_col[2];
                }

                const float light_pdf = (to_light_dist * to_light_dist) / (tri_area * cos_theta);

                struct {
                    const Ray::material_t *mat;
                    float weight;
                } materials_stack[16];

                uint32_t materials_count = 0;
                materials_stack[materials_count++] = {mat, 1.0f};

                while (materials_count) {
                    const auto &cur = materials_stack[--materials_count];
                    const Ray::material_t *cur_mat = cur.mat;
                    const float cur_weight = cur.weight;

                    const float eta =
                        is_backfacing ? (cur_mat->int_ior / cur_mat->ext_ior) : (cur_mat->ext_ior / cur_mat->int_ior);
                    simd_fvec3 H;
                    if (N_dot_L > 0.0f) {
                        H = normalize(L - I);
                    } else {
                        H = normalize(L - I * eta);
                    }

                    simd_fvec4 base_color =
                        simd_fvec4(cur_mat->base_color[0], cur_mat->base_color[1], cur_mat->base_color[2], 1.0f);
                    if (cur_mat->textures[BASE_TEXTURE] != 0xffffffff) {
                        const texture_t &base_texture = sc.textures[cur_mat->textures[BASE_TEXTURE]];
                        const float base_lod = get_texture_lod(base_texture, surf_der.duv_dx, surf_der.duv_dy);
                        simd_fvec4 tex_color = SampleBilinear(tex_atlas, base_texture, uvs, int(base_lod));
                        if (base_texture.width & TEXTURE_SRGB_BIT) {
                            tex_color = srgb_to_rgb(tex_color);
                        }
                        base_color *= tex_color;
                    }

                    simd_fvec4 tint_color = {0.0f};

                    const float base_color_lum = lum(base_color);
                    if (base_color_lum > 0.0f) {
                        tint_color = base_color / base_color_lum;
                    }

                    // sample roughness texture
                    float roughness = unpack_unorm_16(cur_mat->roughness_unorm);
                    if (cur_mat->textures[ROUGH_TEXTURE] != 0xffffffff) {
                        const texture_t &roughness_tex = sc.textures[cur_mat->textures[ROUGH_TEXTURE]];
                        const float roughness_lod = get_texture_lod(roughness_tex, surf_der.duv_dx, surf_der.duv_dy);
                        simd_fvec4 roughness_color =
                            SampleBilinear(tex_atlas, roughness_tex, uvs, int(roughness_lod))[0];
                        if (roughness_tex.width & TEXTURE_SRGB_BIT) {
                            roughness_color = srgb_to_rgb(roughness_color);
                        }
                        roughness *= roughness_color[0];
                    }

                    const float roughness2 = roughness * roughness;

                    if (cur_mat->type == DiffuseNode) {
                        if (N_dot_L > 0.0f) {
                            const simd_fvec4 _base_color = pi.should_consider_albedo() ? base_color : simd_fvec4(1.0f);
                            simd_fvec4 diff_col = Evaluate_OrenDiffuse_BSDF(-I, N, L, roughness, _base_color);
                            float bsdf_pdf = diff_col[3];

                            const float mis_weight = power_heuristic(light_pdf, bsdf_pdf);
                            col += simd_fvec3(&lcol[0]) * simd_fvec3(&diff_col[0]) *
                                   (cur_weight * mis_weight * float(sc.lights2_count) / light_pdf);
                        }
                    } else if (cur_mat->type == GlossyNode) {
                        if (roughness2 * roughness2 >= 1e-7f && N_dot_L > 0.0f) {
                            const simd_fvec3 view_dir_ts = mul(tangent_from_world, -I);
                            const simd_fvec3 light_dir_ts = mul(tangent_from_world, L);
                            const simd_fvec3 sampled_normal_ts = mul(tangent_from_world, H);

                            const float specular = 0.5f;
                            const float spec_ior = (2.0f / (1.0f - std::sqrt(0.08f * specular))) - 1.0f;
                            const float F0 = fresnel_dielectric_cos(1.0f, spec_ior);

                            const simd_fvec4 spec_col =
                                Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, roughness2,
                                                          roughness2, spec_ior, F0, base_color);
                            const float bsdf_pdf = spec_col[3];

                            const float mis_weight = power_heuristic(light_pdf, bsdf_pdf);
                            col += simd_fvec3(&lcol[0]) * simd_fvec3(&spec_col[0]) *
                                   simd_fvec3(cur_weight * mis_weight * float(sc.lights2_count) / light_pdf);
                        }
                    } else if (cur_mat->type == RefractiveNode) {
                        if (roughness2 * roughness2 >= 1e-7f && N_dot_L < 0.0f) {
                            const simd_fvec3 view_dir_ts = mul(tangent_from_world, -I);
                            const simd_fvec3 light_dir_ts = mul(tangent_from_world, L);
                            const simd_fvec3 sampled_normal_ts = mul(tangent_from_world, H);

                            const simd_fvec4 refr_col = Evaluate_GGXRefraction_BSDF(
                                view_dir_ts, sampled_normal_ts, light_dir_ts, roughness2, eta, base_color);
                            const float bsdf_pdf = refr_col[3];

                            const float mis_weight = power_heuristic(light_pdf, bsdf_pdf);
                            col += simd_fvec3(&lcol[0]) * simd_fvec3(&refr_col[0]) *
                                   simd_fvec3(cur_weight * mis_weight * float(sc.lights2_count) / light_pdf);
                        }
                    } else if (cur_mat->type == PrincipledNode) {
                        float metallic = unpack_unorm_16(cur_mat->metallic_unorm);
                        if (cur_mat->textures[METALLIC_TEXTURE] != 0xffffffff) {
                            const texture_t &metallic_tex = sc.textures[cur_mat->textures[METALLIC_TEXTURE]];
                            const float metallic_lod = get_texture_lod(metallic_tex, surf_der.duv_dx, surf_der.duv_dy);
                            metallic *= SampleBilinear(tex_atlas, metallic_tex, uvs, int(metallic_lod))[0];
                        }

                        const float specular = unpack_unorm_16(cur_mat->specular_unorm);
                        const float transmission = unpack_unorm_16(cur_mat->transmission_unorm);
                        const float clearcoat = cur_mat->clearcoat;
                        const float clearcoat_roughness = cur_mat->clearcoat_roughness;
                        const float sheen = unpack_unorm_16(cur_mat->sheen_unorm);
                        const float sheen_tint = unpack_unorm_16(cur_mat->sheen_tint_unorm);

                        simd_fvec4 spec_tmp_col =
                            mix(simd_fvec4{1.0f}, tint_color, unpack_unorm_16(cur_mat->specular_tint_unorm));
                        spec_tmp_col = mix(specular * 0.08f * spec_tmp_col, base_color, metallic);

                        const float spec_ior = (2.0f / (1.0f - std::sqrt(0.08f * specular))) - 1.0f;
                        const float F0 = fresnel_dielectric_cos(1.0f, spec_ior);

                        // Approximation of FH (using shading normal)
                        const float FN = (fresnel_dielectric_cos(dot(I, N), spec_ior) - F0) / (1.0f - F0);

                        const simd_fvec3 approx_spec_col = mix(simd_fvec3(&spec_tmp_col[0]), simd_fvec3(1.0f), FN);
                        const float spec_color_lum = lum(approx_spec_col);

                        float diffuse_weight, specular_weight, clearcoat_weight, refraction_weight;
                        get_lobe_weights(mix(base_color_lum, 1.0f, sheen), spec_color_lum, specular, metallic,
                                         transmission, clearcoat, &diffuse_weight, &specular_weight, &clearcoat_weight,
                                         &refraction_weight);

                        float bsdf_pdf = 0.0f;

                        if (diffuse_weight > 0.0f && N_dot_L > 0.0f) {
                            const simd_fvec4 _base_color = pi.should_consider_albedo() ? base_color : simd_fvec4(1.0f);
                            const simd_fvec4 sheen_color = sheen * mix(simd_fvec4{1.0f}, tint_color, sheen_tint);

                            simd_fvec4 diff_col = Evaluate_PrincipledDiffuse_BSDF(
                                -I, N, L, roughness, _base_color, sheen_color, pi.use_uniform_sampling());
                            bsdf_pdf += diffuse_weight * diff_col[3];
                            diff_col *= (1.0f - metallic);

                            col += simd_fvec3(&lcol[0]) * N_dot_L * simd_fvec3(&diff_col[0]) * cur_weight *
                                   float(sc.lights2_count) / (PI * light_pdf);
                        }

                        const float roughness2 = roughness * roughness;
                        const float aspect = std::sqrt(1.0f - 0.9f * unpack_unorm_16(cur_mat->anisotropic_unorm));

                        const float alpha_x = roughness2 / aspect;
                        const float alpha_y = roughness2 * aspect;

                        const simd_fvec3 view_dir_ts = mul(tangent_from_world, -I);
                        const simd_fvec3 light_dir_ts = mul(tangent_from_world, L);
                        const simd_fvec3 sampled_normal_ts = mul(tangent_from_world, H);

                        if (specular_weight > 0.0f && alpha_x * alpha_y >= 1e-7f && N_dot_L > 0.0f) {
                            const simd_fvec4 spec_col =
                                Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, alpha_x,
                                                          alpha_y, spec_ior, F0, spec_tmp_col);
                            bsdf_pdf += specular_weight * spec_col[3];

                            col += simd_fvec3(&lcol[0]) * simd_fvec3(&spec_col[0]) *
                                   simd_fvec3(cur_weight * float(sc.lights2_count) / light_pdf);
                        }

                        const float clearcoat_ior = (2.0f / (1.0f - std::sqrt(0.08f * clearcoat))) - 1.0f;
                        const float clearcoat_F0 = fresnel_dielectric_cos(1.0f, clearcoat_ior);
                        const float clearcoat_roughness2 = clearcoat_roughness * clearcoat_roughness;
                        if (clearcoat_weight > 0.0f && clearcoat_roughness2 * clearcoat_roughness2 >= 1e-7f &&
                            N_dot_L > 0.0f) {
                            const simd_fvec4 clearcoat_col =
                                Evaluate_PrincipledClearcoat_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts,
                                                                  clearcoat_roughness2, clearcoat_ior, clearcoat_F0);
                            bsdf_pdf += clearcoat_weight * clearcoat_col[3];

                            col += simd_fvec3(&lcol[0]) * simd_fvec3(&clearcoat_col[0]) *
                                   simd_fvec3(cur_weight * float(sc.lights2_count) / light_pdf);
                        }

                        if (refraction_weight > 0.0f) {
                            const float fresnel = fresnel_dielectric_cos(dot(I, N), 1.0f / eta);

                            if (fresnel != 0.0f && roughness2 * roughness2 >= 1e-7f && N_dot_L > 0.0f) {
                                const simd_fvec4 spec_col = Evaluate_GGXSpecular_BSDF(
                                    view_dir_ts, sampled_normal_ts, light_dir_ts, roughness2, roughness2,
                                    1.0f /* ior */, 0.0f /* F0 */, simd_fvec4{1.0f});
                                bsdf_pdf += refraction_weight * fresnel * spec_col[3];

                                col += simd_fvec3(&lcol[0]) * simd_fvec3(&spec_col[0]) *
                                       simd_fvec3(fresnel * cur_weight * float(sc.lights2_count) / light_pdf);
                            }

                            const float transmission_roughness =
                                1.0f -
                                (1.0f - roughness) * (1.0f - unpack_unorm_16(cur_mat->transmission_roughness_unorm));
                            const float transmission_roughness2 = transmission_roughness * transmission_roughness;
                            if (fresnel != 1.0f && transmission_roughness2 * transmission_roughness2 >= 1e-7f &&
                                N_dot_L < 0.0f) {
                                const simd_fvec4 refr_col =
                                    Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts,
                                                                transmission_roughness2, eta, base_color);
                                bsdf_pdf += refraction_weight * (1.0f - fresnel) * refr_col[3];

                                col += simd_fvec3(&lcol[0]) * simd_fvec3(&refr_col[0]) *
                                       simd_fvec3((1.0f - fresnel) * cur_weight * float(sc.lights2_count) / light_pdf);
                            }
                        }

                        const float mis_weight = power_heuristic(light_pdf, bsdf_pdf);
                        col *= mis_weight;
                    } else if (cur_mat->type == MixNode) {
                        if (cur_mat->flags & MAT_FLAG_MIX_ADD) {
                            materials_stack[materials_count++] = {&sc.materials[cur_mat->textures[MIX_MAT1]],
                                                                  cur_weight};
                            materials_stack[materials_count++] = {&sc.materials[cur_mat->textures[MIX_MAT2]],
                                                                  cur_weight};
                        } else {
                            float mix_val = cur_mat->strength;
                            if (cur_mat->textures[BASE_TEXTURE] != 0xffffffff) {
                                mix_val *=
                                    SampleBilinear(tex_atlas, sc.textures[cur_mat->textures[BASE_TEXTURE]], uvs, 0)[0];
                            }

                            const float eta = is_backfacing ? (cur_mat->ext_ior / cur_mat->int_ior)
                                                            : (cur_mat->int_ior / cur_mat->ext_ior);
                            const float RR = cur_mat->int_ior != 0.0f ? fresnel_dielectric_cos(dot(I, N), eta) : 1.0f;

                            mix_val *= clamp(RR, 0.0f, 1.0f);

                            materials_stack[materials_count++] = {&sc.materials[cur_mat->textures[MIX_MAT1]],
                                                                  (1.0f - mix_val) * cur_weight};
                            materials_stack[materials_count++] = {&sc.materials[cur_mat->textures[MIX_MAT2]],
                                                                  mix_val * cur_weight};
                        }
                    }
                }
            }
        }
    }
#endif

    float _unused;
    float mix_rand = std::modf(halton[RAND_DIM_BSDF_PICK] + sample_off[0], &_unused);

    float mix_weight = 1.0f;

    // resolve mix material
    while (mat->type == MixNode) {
        float mix_val = mat->strength;
        if (mat->textures[BASE_TEXTURE] != 0xffffffff) {
            mix_val *= SampleBilinear(tex_atlas, sc.textures[mat->textures[BASE_TEXTURE]], uvs, 0)[0];
        }

        // const float eta = is_backfacing ? (mat->int_ior / mat->ext_ior) : (mat->ext_ior / mat->int_ior);
        const float RR = mat->int_ior != 0.0f
                             ? fresnel_dielectric_cos(dot(I, N), is_backfacing ? (mat->ext_ior / mat->int_ior)
                                                                               : (mat->int_ior / mat->ext_ior))
                             : 1.0f;

        mix_val *= clamp(RR, 0.0f, 1.0f);

        if (mix_rand > mix_val) {
            mix_weight *= (mat->flags & MAT_FLAG_MIX_ADD) ? 1.0f / (1.0f - mix_val) : 1.0f;

            mat = &sc.materials[mat->textures[MIX_MAT1]];
            mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
        } else {
            mix_weight *= (mat->flags & MAT_FLAG_MIX_ADD) ? 1.0f / mix_val : 1.0f;

            mat = &sc.materials[mat->textures[MIX_MAT2]];
            mix_rand = mix_rand / mix_val;
        }
    }

    const float mat_ior = is_backfacing ? mat->ext_ior : mat->int_ior;

    // sample base texture
    simd_fvec4 base_color = simd_fvec4{mat->base_color[0], mat->base_color[1], mat->base_color[2], 1.0f};
    if (mat->textures[BASE_TEXTURE] != 0xffffffff) {
        const texture_t &base_texture = sc.textures[mat->textures[BASE_TEXTURE]];
        const float base_lod = get_texture_lod(base_texture, surf_der.duv_dx, surf_der.duv_dy);
        simd_fvec4 tex_color = SampleBilinear(tex_atlas, base_texture, uvs, int(base_lod));
        if (base_texture.width & TEXTURE_SRGB_BIT) {
            tex_color = srgb_to_rgb(tex_color);
        }
        base_color *= tex_color;
    }

    simd_fvec4 tint_color = {0.0f};

    const float base_color_lum = lum(base_color);
    if (base_color_lum > 0.0f) {
        tint_color = base_color / base_color_lum;
    }

    float roughness = unpack_unorm_16(mat->roughness_unorm);
    if (mat->textures[ROUGH_TEXTURE] != 0xffffffff) {
        const texture_t &roughness_tex = sc.textures[mat->textures[ROUGH_TEXTURE]];
        const float roughness_lod = get_texture_lod(roughness_tex, surf_der.duv_dx, surf_der.duv_dy);
        simd_fvec4 roughness_color = SampleBilinear(tex_atlas, roughness_tex, uvs, int(roughness_lod))[0];
        if (roughness_tex.width & TEXTURE_SRGB_BIT) {
            roughness_color = srgb_to_rgb(roughness_color);
        }
        roughness *= roughness_color[0];
    }

    const bool cant_terminate = true;
    // total_depth < pi.settings.termination_start_depth;

    const float rand_u = std::modf(halton[RAND_DIM_BSDF_U] + sample_off[0], &_unused);
    const float rand_v = std::modf(halton[RAND_DIM_BSDF_V] + sample_off[1], &_unused);

    // Sample materials
    if (mat->type == DiffuseNode) {
        if (diff_depth < pi.settings.max_diff_depth && total_depth < pi.settings.max_total_depth) {
            const simd_fvec4 _base_color = pi.should_consider_albedo() ? base_color : simd_fvec4(1.0f);

            simd_fvec3 V;
            const simd_fvec4 F =
                Sample_OrenDiffuse_BSDF(world_from_tangent, N, I, roughness, _base_color, rand_u, rand_v, V);
            ray_packet_t r;
            r.xy = ray.xy;
            r.ray_depth = ray.ray_depth + 0x00000001;

            memcpy(&r.o[0], value_ptr(offset_ray(P, plane_N)), 3 * sizeof(float));
            memcpy(&r.d[0], value_ptr(V), 3 * sizeof(float));

            r.c[0] = ray.c[0] * F[0] * mix_weight / F[3];
            r.c[1] = ray.c[1] * F[1] * mix_weight / F[3];
            r.c[2] = ray.c[2] * F[2] * mix_weight / F[3];
            r.pdf = F[3];

            memcpy(&r.do_dx[0], value_ptr(surf_der.do_dx), 3 * sizeof(float));
            memcpy(&r.do_dy[0], value_ptr(surf_der.do_dy), 3 * sizeof(float));

            memcpy(&r.dd_dx[0],
                   value_ptr(surf_der.dd_dx - 2 * (dot(I, plane_N) * surf_der.dndx + surf_der.ddn_dx * plane_N)),
                   3 * sizeof(float));
            memcpy(&r.dd_dy[0],
                   value_ptr(surf_der.dd_dy - 2 * (dot(I, plane_N) * surf_der.dndy + surf_der.ddn_dy * plane_N)),
                   3 * sizeof(float));

            const float lum = std::max(r.c[0], std::max(r.c[1], r.c[2]));
            const float p = std::modf(halton[RAND_DIM_TERMINATE] + sample_off[0], &_unused);
            const float q = cant_terminate ? 0.0f : std::max(0.05f, 1.0f - lum);
            if (p >= q && lum > 0.0f) {
                r.c[0] /= (1.0f - q);
                r.c[1] /= (1.0f - q);
                r.c[2] /= (1.0f - q);
                const int index = (*out_secondary_rays_count)++;
                out_secondary_rays[index] = r;
            }
        }
    } else if (mat->type == GlossyNode) {
        if (spec_depth < pi.settings.max_spec_depth && total_depth < pi.settings.max_total_depth) {
            const float specular = 0.5f;
            const float spec_ior = (2.0f / (1.0f - std::sqrt(0.08f * specular))) - 1.0f;
            const float spec_F0 = fresnel_dielectric_cos(1.0f, spec_ior);

            simd_fvec3 V;
            const simd_fvec4 F = Sample_GGXSpecular_BSDF(world_from_tangent, tangent_from_world, N, I, roughness, 0.0f,
                                                         spec_ior, spec_F0, base_color, rand_u, rand_v, V);

            ray_packet_t r;
            r.xy = ray.xy;
            r.ray_depth = ray.ray_depth + 0x00000100;

            r.c[0] = ray.c[0] * F[0] * mix_weight / F[3];
            r.c[1] = ray.c[1] * F[1] * mix_weight / F[3];
            r.c[2] = ray.c[2] * F[2] * mix_weight / F[3];
            r.pdf = F[3];

            memcpy(&r.o[0], value_ptr(offset_ray(P, plane_N)), 3 * sizeof(float));
            memcpy(&r.d[0], value_ptr(V), 3 * sizeof(float));

            memcpy(&r.do_dx[0], value_ptr(surf_der.do_dx), 3 * sizeof(float));
            memcpy(&r.do_dy[0], value_ptr(surf_der.do_dy), 3 * sizeof(float));

            memcpy(&r.dd_dx[0],
                   value_ptr(surf_der.dd_dx - 2 * (dot(I, plane_N) * surf_der.dndx + surf_der.ddn_dx * plane_N)),
                   3 * sizeof(float));
            memcpy(&r.dd_dy[0],
                   value_ptr(surf_der.dd_dy - 2 * (dot(I, plane_N) * surf_der.dndy + surf_der.ddn_dy * plane_N)),
                   3 * sizeof(float));

            const float lum = std::max(r.c[0], std::max(r.c[1], r.c[2]));
            const float p = std::modf(halton[RAND_DIM_TERMINATE] + sample_off[0], &_unused);
            const float q = cant_terminate ? 0.0f : std::max(0.05f, 1.0f - lum);
            if (p >= q && r.pdf != 0.0f && lum > 0.0f) {
                r.c[0] /= 1.0f - q;
                r.c[1] /= 1.0f - q;
                r.c[2] /= 1.0f - q;
                const int index = (*out_secondary_rays_count)++;
                out_secondary_rays[index] = r;
            }
        }
    } else if (mat->type == RefractiveNode) {
        if (refr_depth < pi.settings.max_refr_depth && total_depth < pi.settings.max_total_depth) {
            const float eta = is_backfacing ? (mat->int_ior / mat->ext_ior) : (mat->ext_ior / mat->int_ior);

            simd_fvec4 _V;
            const simd_fvec4 F = Sample_GGXRefraction_BSDF(world_from_tangent, tangent_from_world, N, I, roughness, eta,
                                                           base_color, rand_u, rand_v, _V);

            const simd_fvec3 V = {_V[0], _V[1], _V[2]};
            const float m = _V[3];

            ray_packet_t r;

            r.xy = ray.xy;
            r.ray_depth = ray.ray_depth + 0x00010000;

            r.c[0] = ray.c[0] * F[0] * mix_weight / F[3];
            r.c[1] = ray.c[1] * F[1] * mix_weight / F[3];
            r.c[2] = ray.c[2] * F[2] * mix_weight / F[3];
            r.pdf = F[3];

            const float k = (eta - eta * eta * dot(I, plane_N) / dot(V, plane_N));
            const float dmdx = k * surf_der.ddn_dx;
            const float dmdy = k * surf_der.ddn_dy;

            memcpy(&r.o[0], value_ptr(offset_ray(P, -plane_N)), 3 * sizeof(float));
            memcpy(&r.d[0], value_ptr(V), 3 * sizeof(float));

            memcpy(&r.do_dx[0], value_ptr(surf_der.do_dx), 3 * sizeof(float));
            memcpy(&r.do_dy[0], value_ptr(surf_der.do_dy), 3 * sizeof(float));

            memcpy(&r.dd_dx[0], value_ptr(eta * surf_der.dd_dx - (m * surf_der.dndx + dmdx * plane_N)),
                   3 * sizeof(float));
            memcpy(&r.dd_dy[0], value_ptr(eta * surf_der.dd_dy - (m * surf_der.dndy + dmdy * plane_N)),
                   3 * sizeof(float));

            const float lum = std::max(r.c[0], std::max(r.c[1], r.c[2]));
            const float p = std::modf(halton[RAND_DIM_TERMINATE] + sample_off[0], &_unused);
            const float q = cant_terminate ? 0.0f : std::max(0.05f, 1.0f - lum);
            if (p >= q && lum > 0.0f && r.pdf != 0.0f) {
                r.c[0] /= 1.0f - q;
                r.c[1] /= 1.0f - q;
                r.c[2] /= 1.0f - q;
                const int index = (*out_secondary_rays_count)++;
                out_secondary_rays[index] = r;
            }
        }
    } else if (mat->type == EmissiveNode) {
        float mis_weight = 1.0f;
#if USE_NEE == 1
        // TODO: refactor this!
        if (mat->flags & MAT_FLAG_SKY_PORTAL) {
            base_color = simd_fvec4{1.0f};
            if (sc.env->env_map != 0xffffffff) {
                base_color = SampleLatlong_RGBE(tex_atlas, sc.textures[sc.env->env_map], simd_fvec3{ray.d});
                if (sc.env->env_clamp > FLT_EPS) {
                    base_color = min(base_color, simd_fvec4{sc.env->env_clamp});
                }
            }
            base_color[0] *= sc.env->env_col[0];
            base_color[1] *= sc.env->env_col[1];
            base_color[2] *= sc.env->env_col[2];
        }

        if (total_depth > 0 && (mat->flags & (MAT_FLAG_MULT_IMPORTANCE | MAT_FLAG_SKY_PORTAL))) {
            const simd_fvec3 p1 = simd_fvec3(v1.p), p2 = simd_fvec3(v2.p), p3 = simd_fvec3(v3.p);

            simd_fvec3 light_forward = TransformDirection(cross(p2 - p1, p3 - p1), tr->xform);
            const float light_forward_len = length(light_forward);
            light_forward /= light_forward_len;
            const float tri_area = 0.5f * light_forward_len;

            const float cos_theta = std::abs(dot(-I, light_forward)); // abs for doublesided light
            if (cos_theta > 0.0f) {
                const float light_pdf = (inter.t * inter.t) / (tri_area * cos_theta);
                const float bsdf_pdf = ray.pdf;

                mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            }
        }
#endif
        col += mis_weight * mat->strength * mix_weight * simd_fvec3(&base_color[0]);
    } else if (mat->type == TransparentNode) {
        if (transp_depth < pi.settings.max_transp_depth && total_depth < pi.settings.max_total_depth) {
            ray_packet_t r;

            r.xy = ray.xy;
            r.ray_depth = ray.ray_depth + 0x01000000;
            r.pdf = ray.pdf;

            memcpy(&r.o[0], value_ptr(offset_ray(P, -plane_N)), 3 * sizeof(float));
            memcpy(&r.d[0], &ray.d[0], 3 * sizeof(float));
            memcpy(&r.c[0], &ray.c[0], 3 * sizeof(float));

            memcpy(&r.do_dx[0], &ray.do_dx[0], 3 * sizeof(float));
            memcpy(&r.do_dy[0], &ray.do_dy[0], 3 * sizeof(float));

            memcpy(&r.dd_dx[0], &ray.dd_dx[0], 3 * sizeof(float));
            memcpy(&r.dd_dy[0], &ray.dd_dy[0], 3 * sizeof(float));

            const float lum = std::max(r.c[0], std::max(r.c[1], r.c[2]));
            const float p = std::modf(halton[RAND_DIM_TERMINATE] + sample_off[0], &_unused);
            const float q = cant_terminate ? 0.0f : std::max(0.05f, 1.0f - lum);
            if (p >= q && lum > 0.0f) {
                r.c[0] /= 1.0f - q;
                r.c[1] /= 1.0f - q;
                r.c[2] /= 1.0f - q;
                const int index = (*out_secondary_rays_count)++;
                out_secondary_rays[index] = r;
            }
        }
    } else if (mat->type == PrincipledNode) {
        float metallic = unpack_unorm_16(mat->metallic_unorm);
        if (mat->textures[METALLIC_TEXTURE] != 0xffffffff) {
            const texture_t &metallic_tex = sc.textures[mat->textures[METALLIC_TEXTURE]];
            const float metallic_lod = get_texture_lod(metallic_tex, surf_der.duv_dx, surf_der.duv_dy);
            metallic *= SampleBilinear(tex_atlas, metallic_tex, uvs, int(metallic_lod))[0];
        }

        const float specular = unpack_unorm_16(mat->specular_unorm);
        const float transmission = unpack_unorm_16(mat->transmission_unorm);
        const float clearcoat = mat->clearcoat;
        const float clearcoat_roughness = mat->clearcoat_roughness;
        const float sheen = unpack_unorm_16(mat->sheen_unorm);
        const float sheen_tint = unpack_unorm_16(mat->sheen_tint_unorm);

        simd_fvec4 spec_tmp_col = mix(simd_fvec4{1.0f}, tint_color, unpack_unorm_16(mat->specular_tint_unorm));
        spec_tmp_col = mix(specular * 0.08f * spec_tmp_col, base_color, metallic);

        const float spec_ior = (2.0f / (1.0f - std::sqrt(0.08f * specular))) - 1.0f;
        const float spec_F0 = fresnel_dielectric_cos(1.0f, spec_ior);

        // Approximation of FH (using shading normal)
        const float FN = (fresnel_dielectric_cos(dot(I, N), spec_ior) - spec_F0) / (1.0f - spec_F0);

        const simd_fvec4 approx_spec_col = mix(spec_tmp_col, simd_fvec4(1.0f), FN);
        const float spec_color_lum = lum(approx_spec_col);

        float diffuse_weight, specular_weight, clearcoat_weight, refraction_weight;
        get_lobe_weights(mix(base_color_lum, 1.0f, sheen), spec_color_lum, specular, metallic, transmission, clearcoat,
                         &diffuse_weight, &specular_weight, &clearcoat_weight, &refraction_weight);

        if (mix_rand < diffuse_weight) {
            //
            // Diffuse lobe
            //
            if (diff_depth < pi.settings.max_diff_depth && total_depth < pi.settings.max_total_depth) {
                const simd_fvec4 _base_color = pi.should_consider_albedo() ? base_color : simd_fvec4(1.0f);
                const simd_fvec4 sheen_color = sheen * mix(simd_fvec4{1.0f}, tint_color, sheen_tint);

                simd_fvec3 V;
                simd_fvec4 diff_col =
                    Sample_PrincipledDiffuse_BSDF(world_from_tangent, N, I, roughness, _base_color, sheen_color,
                                                  pi.use_uniform_sampling(), rand_u, rand_v, V);
                const float pdf = diff_col[3];

                diff_col *= (1.0f - metallic);

                ray_packet_t r;

                r.xy = ray.xy;
                r.ray_depth = ray.ray_depth + 0x00000001;

                memcpy(&r.o[0], value_ptr(offset_ray(P, plane_N)), 3 * sizeof(float));
                memcpy(&r.d[0], value_ptr(V), 3 * sizeof(float));

                r.c[0] = ray.c[0] * diff_col[0] * mix_weight / diffuse_weight;
                r.c[1] = ray.c[1] * diff_col[1] * mix_weight / diffuse_weight;
                r.c[2] = ray.c[2] * diff_col[2] * mix_weight / diffuse_weight;
                r.pdf = pdf;

                memcpy(&r.do_dx[0], value_ptr(surf_der.do_dx), 3 * sizeof(float));
                memcpy(&r.do_dy[0], value_ptr(surf_der.do_dy), 3 * sizeof(float));

                memcpy(&r.dd_dx[0],
                       value_ptr(surf_der.dd_dx - 2 * (dot(I, plane_N) * surf_der.dndx + surf_der.ddn_dx * plane_N)),
                       3 * sizeof(float));
                memcpy(&r.dd_dy[0],
                       value_ptr(surf_der.dd_dy - 2 * (dot(I, plane_N) * surf_der.dndy + surf_der.ddn_dy * plane_N)),
                       3 * sizeof(float));

                const float lum = std::max(r.c[0], std::max(r.c[1], r.c[2]));
                const float p = std::modf(halton[RAND_DIM_TERMINATE] + sample_off[0], &_unused);
                const float q = cant_terminate ? 0.0f : std::max(0.05f, 1.0f - lum);
                if (p >= q && lum > 0.0f && r.pdf != 0.0f) {
                    r.c[0] /= (1.0f - q);
                    r.c[1] /= (1.0f - q);
                    r.c[2] /= (1.0f - q);
                    const int index = (*out_secondary_rays_count)++;
                    out_secondary_rays[index] = r;
                }
            }
        } else if (mix_rand < diffuse_weight + specular_weight) {
            //
            // Main specular lobe
            //
            if (spec_depth < pi.settings.max_spec_depth && total_depth < pi.settings.max_total_depth) {
                simd_fvec3 V;
                simd_fvec4 F = Sample_GGXSpecular_BSDF(world_from_tangent, tangent_from_world, N, I, roughness,
                                                       unpack_unorm_16(mat->anisotropic_unorm), spec_ior, spec_F0,
                                                       spec_tmp_col, rand_u, rand_v, V);
                F[3] *= specular_weight;

                ray_packet_t r;

                r.xy = ray.xy;
                r.ray_depth = ray.ray_depth + 0x00000100;

                r.c[0] = ray.c[0] * F[0] * mix_weight / F[3];
                r.c[1] = ray.c[1] * F[1] * mix_weight / F[3];
                r.c[2] = ray.c[2] * F[2] * mix_weight / F[3];
                r.pdf = F[3];

                memcpy(&r.o[0], value_ptr(offset_ray(P, plane_N)), 3 * sizeof(float));
                memcpy(&r.d[0], value_ptr(V), 3 * sizeof(float));

                memcpy(&r.do_dx[0], value_ptr(surf_der.do_dx), 3 * sizeof(float));
                memcpy(&r.do_dy[0], value_ptr(surf_der.do_dy), 3 * sizeof(float));

                memcpy(&r.dd_dx[0],
                       value_ptr(surf_der.dd_dx - 2 * (dot(I, plane_N) * surf_der.dndx + surf_der.ddn_dx * plane_N)),
                       3 * sizeof(float));
                memcpy(&r.dd_dy[0],
                       value_ptr(surf_der.dd_dy - 2 * (dot(I, plane_N) * surf_der.dndy + surf_der.ddn_dy * plane_N)),
                       3 * sizeof(float));

                const float lum = std::max(r.c[0], std::max(r.c[1], r.c[2]));
                const float p = std::modf(halton[RAND_DIM_TERMINATE] + sample_off[0], &_unused);
                const float q = cant_terminate ? 0.0f : std::max(0.05f, 1.0f - lum);
                if (p >= q && r.pdf != 0.0f && lum > 0.0f) {
                    r.c[0] /= 1.0f - q;
                    r.c[1] /= 1.0f - q;
                    r.c[2] /= 1.0f - q;
                    const int index = (*out_secondary_rays_count)++;
                    out_secondary_rays[index] = r;
                }
            }
        } else if (mix_rand < diffuse_weight + specular_weight + clearcoat_weight) {
            //
            // Clearcoat lobe (secondary specular)
            //
            if (spec_depth < pi.settings.max_spec_depth && total_depth < pi.settings.max_total_depth) {
                const float clearcoat_ior = (2.0f / (1.0f - std::sqrt(0.08f * clearcoat))) - 1.0f;
                const float clearcoat_F0 = fresnel_dielectric_cos(1.0f, clearcoat_ior);
                const float clearcoat_roughness2 = clearcoat_roughness * clearcoat_roughness;

                simd_fvec3 V;
                const simd_fvec4 F =
                    Sample_PrincipledClearcoat_BSDF(world_from_tangent, tangent_from_world, N, I, clearcoat_roughness2,
                                                    clearcoat_ior, clearcoat_F0, rand_u, rand_v, V);

                ray_packet_t r;

                r.xy = ray.xy;
                r.ray_depth = ray.ray_depth + 0x00000100;

                const float weight = (mix_weight / clearcoat_weight);
                r.c[0] = ray.c[0] * F[0] * weight;
                r.c[1] = ray.c[1] * F[1] * weight;
                r.c[2] = ray.c[2] * F[2] * weight;
                r.pdf = F[3];

                memcpy(&r.o[0], value_ptr(offset_ray(P, plane_N)), 3 * sizeof(float));
                memcpy(&r.d[0], value_ptr(V), 3 * sizeof(float));

                memcpy(&r.do_dx[0], value_ptr(surf_der.do_dx), 3 * sizeof(float));
                memcpy(&r.do_dy[0], value_ptr(surf_der.do_dy), 3 * sizeof(float));

                memcpy(&r.dd_dx[0],
                       value_ptr(surf_der.dd_dx - 2 * (dot(I, plane_N) * surf_der.dndx + surf_der.ddn_dx * plane_N)),
                       3 * sizeof(float));
                memcpy(&r.dd_dy[0],
                       value_ptr(surf_der.dd_dy - 2 * (dot(I, plane_N) * surf_der.dndy + surf_der.ddn_dy * plane_N)),
                       3 * sizeof(float));

                const float lum = std::max(r.c[0], std::max(r.c[1], r.c[2]));
                const float p = std::modf(halton[RAND_DIM_TERMINATE] + sample_off[0], &_unused);
                const float q = cant_terminate ? 0.0f : std::max(0.05f, 1.0f - lum);
                if (p >= q && r.pdf != 0.0f && lum > 0.0f) {
                    r.c[0] /= 1.0f - q;
                    r.c[1] /= 1.0f - q;
                    r.c[2] /= 1.0f - q;
                    const int index = (*out_secondary_rays_count)++;
                    out_secondary_rays[index] = r;
                }
            }
        } else /*if (mix_rand < diffuse_weight + specular_weight + clearcoat_weight + refraction_weight)*/ {
            //
            // Refraction/reflection lobes
            //
            const float eta = is_backfacing ? (mat->int_ior / mat->ext_ior) : (mat->ext_ior / mat->int_ior);
            const float fresnel = fresnel_dielectric_cos(dot(I, N), 1.0f / eta);

            if (((mix_rand >= fresnel && refr_depth < pi.settings.max_refr_depth) ||
                 (mix_rand < fresnel && spec_depth < pi.settings.max_spec_depth)) &&
                total_depth < pi.settings.max_total_depth) {
                mix_rand -= diffuse_weight + specular_weight + clearcoat_weight;
                mix_rand /= refraction_weight;

                //////////////////

                ray_packet_t r;

                r.xy = ray.xy;

                simd_fvec4 F;
                simd_fvec3 V;
                if (mix_rand < fresnel) {
                    F = Sample_GGXSpecular_BSDF(world_from_tangent, tangent_from_world, N, I, roughness,
                                                0.0f /* anisotropic */, 1.0f /* ior */, 0.0f /* F0 */, simd_fvec4{1.0f},
                                                rand_u, rand_v, V);

                    r.ray_depth = ray.ray_depth + 0x00000100;
                    memcpy(&r.o[0], value_ptr(offset_ray(P, plane_N)), 3 * sizeof(float));
                    memcpy(
                        &r.dd_dx[0],
                        value_ptr(surf_der.dd_dx - 2 * (dot(I, plane_N) * surf_der.dndx + surf_der.ddn_dx * plane_N)),
                        3 * sizeof(float));
                    memcpy(
                        &r.dd_dy[0],
                        value_ptr(surf_der.dd_dy - 2 * (dot(I, plane_N) * surf_der.dndy + surf_der.ddn_dy * plane_N)),
                        3 * sizeof(float));
                } else {
                    const float transmission_roughness =
                        1.0f - (1.0f - roughness) * (1.0f - unpack_unorm_16(mat->transmission_roughness_unorm));

                    simd_fvec4 _V;
                    F = Sample_GGXRefraction_BSDF(world_from_tangent, tangent_from_world, N, I, transmission_roughness,
                                                  eta, base_color, rand_u, rand_v, _V);

                    V = {_V[0], _V[1], _V[2]};
                    const float m = _V[3];

                    const float k = (eta - eta * eta * dot(I, plane_N) / dot(V, plane_N));
                    const float dmdx = k * surf_der.ddn_dx;
                    const float dmdy = k * surf_der.ddn_dy;

                    r.ray_depth = ray.ray_depth + 0x00010000;
                    memcpy(&r.o[0], value_ptr(offset_ray(P, -plane_N)), 3 * sizeof(float));
                    memcpy(&r.dd_dx[0], value_ptr(eta * surf_der.dd_dx - (m * surf_der.dndx + dmdx * plane_N)),
                           3 * sizeof(float));
                    memcpy(&r.dd_dy[0], value_ptr(eta * surf_der.dd_dy - (m * surf_der.dndy + dmdy * plane_N)),
                           3 * sizeof(float));
                }

                F[3] *= refraction_weight;

                r.c[0] = ray.c[0] * F[0] * mix_weight / F[3];
                r.c[1] = ray.c[1] * F[1] * mix_weight / F[3];
                r.c[2] = ray.c[2] * F[2] * mix_weight / F[3];
                r.pdf = F[3];

                //////////////////

                memcpy(&r.d[0], value_ptr(V), 3 * sizeof(float));

                memcpy(&r.do_dx[0], value_ptr(surf_der.do_dx), 3 * sizeof(float));
                memcpy(&r.do_dy[0], value_ptr(surf_der.do_dy), 3 * sizeof(float));

                const float lum = std::max(r.c[0], std::max(r.c[1], r.c[2]));
                const float p = std::modf(halton[RAND_DIM_TERMINATE] + sample_off[0], &_unused);
                const float q = cant_terminate ? 0.0f : std::max(0.05f, 1.0f - lum);
                if (p >= q && lum > 0.0f && r.pdf > 0.0f) {
                    r.c[0] /= 1.0f - q;
                    r.c[1] /= 1.0f - q;
                    r.c[2] /= 1.0f - q;
                    const int index = (*out_secondary_rays_count)++;
                    out_secondary_rays[index] = r;
                }
            }
        }
    }

    return pixel_color_t{ray.c[0] * col[0], ray.c[1] * col[1], ray.c[2] * col[2], 1.0f};
}

#undef USE_NEE
#undef USE_VNDF_GGX_SAMPLING
