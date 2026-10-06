#ifndef COMMON_GLSL
#define COMMON_GLSL

#extension GL_EXT_control_flow_attributes :  require

#include "types.h"

#include "trig.glsl"

//
// Useful macros for debugging
//
#define USE_NEE 1
#define USE_HIERARCHICAL_NEE 1
#define USE_PATH_TERMINATION 1
#define USE_SPHERICAL_AREA_LIGHT_SAMPLING 1
// #define FORCE_TEXTURE_LOD 0

uint hash(uint x) {
    // finalizer from murmurhash3
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

float saturate(float val) {
    return clamp(val, 0.0, 1.0);
}

vec2 saturate(vec2 val) {
    return clamp(val, vec2(0.0), vec2(1.0));
}

vec3 saturate(vec3 val) {
    return clamp(val, vec3(0.0), vec3(1.0));
}

float construct_float(uint m) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    const float  f = uintBitsToFloat(m);   // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

uint hash_combine(uint seed, uint v) { return seed ^ (v + (seed << 6) + (seed >> 2)); }

uint laine_karras_permutation(uint x, uint seed) {
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}

uint nested_uniform_scramble_base2(uint x, uint seed) {
    x = bitfieldReverse(x);
    x = laine_karras_permutation(x, seed);
    x = bitfieldReverse(x);
    return x;
}

float scramble_unorm(const uint seed, uint val) {
    val = nested_uniform_scramble_base2(val, seed);
    return float(val >> 8) / 16777216.0;
}

float lum(const vec3 color) {
    return 0.212671 * color[0] + 0.715160 * color[1] + 0.072169 * color[2];
}

vec3 normalize_len(const vec3 v, out float len) {
    return v / (len = length(v));
}

float _copysign(const float val, const float sign) {
    return sign < 0.0 ? -abs(val) : abs(val);
}

float from_unit_to_sub_uvs(const float u, const float resolution) {
    return (u + 0.5 / resolution) * (resolution / (resolution + 1.0));
}
float from_sub_uvs_to_unit(const float u, const float resolution) {
    return (u - 0.5 / resolution) * (resolution / (resolution - 1.0));
}

float power_heuristic(const float a, const float b) {
    const float t = a * a;
    return t / (b * b + t);
}

float linstep(const float smin, const float smax, const float x) {
    return saturate((x - smin) / (smax - smin));
}

vec3 safe_invert(vec3 v) {
    vec3 ret;
    [[unroll]] for (int i = 0; i < 3; ++i) {
        ret[i] = 1.0 / ((abs(v[i]) > FLT_EPS) ? v[i] : _copysign(FLT_EPS, v[i]));
    }
    return ret;
}

vec3 srgb_to_linear(vec3 col) {
    vec3 ret;
    [[unroll]] for (int i = 0; i < 3; ++i) {
        [[flatten]] if (col[i] > 0.04045) {
            ret[i] = pow((col[i] + 0.055) / 1.055, 2.4);
        } else {
            ret[i] = col[i] / 12.92;
        }
    }
    return ret;
}

float _srgb_to_linear(float col) {
    float ret;
    [[flatten]] if (col > 0.04045) {
        ret = pow((col + 0.055) / 1.055, 2.4);
    } else {
        ret = col / 12.92;
    }
    return ret;
}

vec3 YCoCg_to_RGB(vec4 col) {
    float scale = (col.b * (255.0 / 8.0)) + 1.0;
    float Y = col.a;
    float Co = (col.r - (0.5 * 256.0 / 255.0)) / scale;
    float Cg = (col.g - (0.5 * 256.0 / 255.0)) / scale;

    vec3 col_rgb;
    col_rgb.r = Y + Co - Cg;
    col_rgb.g = Y + Cg;
    col_rgb.b = Y - Co - Cg;

    return saturate(col_rgb);
}

float get_texture_lod(const ivec2 res, const float lambda) {
#ifdef FORCE_TEXTURE_LOD
    const float lod = float(FORCE_TEXTURE_LOD);
#else
    const float w = float(res.x);
    const float h = float(res.y);
    // Find lod
    float lod = lambda + 0.5 * log2(w * h);
    // Substruct 1 from lod to always have 4 texels for interpolation
    lod = clamp(lod - 1.0, 0.0, float(MAX_MIP_LEVEL));
#endif
    return lod;
}

vec3 TransformNormal(vec3 n, mat4 inv_xform) {
    return (transpose(inv_xform) * vec4(n, 0.0)).xyz;
}

bool bbox_test(const vec3 p, const vec3 bbox_min, const vec3 bbox_max) {
    return p.x >= bbox_min.x && p.x <= bbox_max.x &&
           p.y >= bbox_min.y && p.y <= bbox_max.y &&
           p.z >= bbox_min.z && p.z <= bbox_max.z;
}

bool bbox_test(vec3 inv_d, vec3 neg_inv_d_o, float t, vec3 bbox_min, vec3 bbox_max) {
    float low = fma(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float high = fma(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float tmin = min(low, high);
    float tmax = max(low, high);

    low = fma(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    high = fma(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));

    low = fma(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    high = fma(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));
    tmax *= 1.00000024;

    return tmin <= tmax && tmin <= t && tmax > 0.0;
}

bool bbox_test(vec3 inv_d, vec3 neg_inv_d_o, float t, vec3 bbox_min, vec3 bbox_max, out float dist) {
    float low = fma(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float high = fma(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float tmin = min(low, high);
    float tmax = max(low, high);

    low = fma(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    high = fma(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));

    low = fma(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    high = fma(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));
    tmax *= 1.00000024;

    dist = tmin;

    return tmin <= tmax && tmin <= t && tmax > 0.0;
}

//
// Compact (CWBVH-style) node helpers - mirror the CPU side (Core.h / CoreSIMD.h)
//

// Per-axis quantization step 2^ch_exp[axis] (CWBVH paper trick): sign-extend the stored int8
// exponent and place it in the float exponent field. ch_exp is always in [-126, 127].
float cwbvh_step(const uint ch_exp, const uint axis) {
    const uint b = (ch_exp >> (8u * axis)) & 0xffu;
    return uintBitsToFloat((((b ^ 0x80u) - 1u) & 0xffu) << 23u);
}

// meta4 byte of child slot i: 0 = empty slot, 0x40|rank = leaf child, 0x80|i = internal child
uint cwbvh_meta(const cwbvh_node_t n, const uint i) {
    return (i < 4u ? n.meta4_0 : n.meta4_1) >> (8u * (i & 3u)) & 0xffu;
}

uint cwbvh_meta(const light_cwbvh_node_t n, const uint i) {
    return (i < 4u ? n.meta4_0 : n.meta4_1) >> (8u * (i & 3u)) & 0xffu;
}

// Internal children are stored contiguously - slot i resolves to child_base + popc(internal slots below i)
uint cwbvh_internal_child(const cwbvh_node_t n, const uint i) {
    const uint imask = n.child_base_imask >> 24u;
    return (n.child_base_imask & 0x00ffffffu) + uint(bitCount(imask & ~(0xffu << i)));
}

uint cwbvh_internal_child(const light_cwbvh_node_t n, const uint i) {
    const uint imask = n.child_base_imask >> 24u;
    return (n.child_base_imask & 0x00ffffffu) + uint(bitCount(imask & ~(0xffu << i)));
}

// The hit mask is in *direction key space* (bit i ^ flip_bits), so draining it with findMSB()
// visits the octant closest to the ray origin first.
uint cwbvh_child_hit_mask(const cwbvh_node_t n, vec3 inv_d, vec3 neg_inv_do, float t, const uint flip_bits) {
    const vec3 _step = vec3(cwbvh_step(n.ch_exp, 0u), cwbvh_step(n.ch_exp, 1u), cwbvh_step(n.ch_exp, 2u));

    uint mask = 0u;
    for (uint i = 0; i < 8; ++i) {
        // empty slots are marked by a zero meta4 byte; scene children always have finite boxes
        if (cwbvh_meta(n, i) == 0u) {
            continue;
        }
        const vec3 bbox_min = vec3(n.bbox_min[0] + float((n.ch_bbox_min[0][i / 4u] >> (8u * (i % 4u))) & 0xffu) * _step[0],
                                   n.bbox_min[1] + float((n.ch_bbox_min[1][i / 4u] >> (8u * (i % 4u))) & 0xffu) * _step[1],
                                   n.bbox_min[2] + float((n.ch_bbox_min[2][i / 4u] >> (8u * (i % 4u))) & 0xffu) * _step[2]);
        const vec3 bbox_max = vec3(n.bbox_min[0] + float((n.ch_bbox_max[0][i / 4u] >> (8u * (i % 4u))) & 0xffu) * _step[0],
                                   n.bbox_min[1] + float((n.ch_bbox_max[1][i / 4u] >> (8u * (i % 4u))) & 0xffu) * _step[1],
                                   n.bbox_min[2] + float((n.ch_bbox_max[2][i / 4u] >> (8u * (i % 4u))) & 0xffu) * _step[2]);

        if (bbox_test(inv_d, neg_inv_do, t, bbox_min, bbox_max)) {
            mask |= (1u << (i ^ flip_bits));
        }
    }
    return mask;
}

#define STACK_PUSH(V)                                                                      \
    {                                                                                      \
        if (stack_size < SM_STACK_SIZE) {                                                  \
            g_stack[gl_LocalInvocationIndex][stack_size] = (V);                            \
        } else {                                                                           \
            g_stack_spill[stack_size - SM_STACK_SIZE] = (V);                               \
        }                                                                                  \
        ++stack_size;                                                                      \
    }

#define STACK_POP(V)                                                                       \
    {                                                                                      \
        --stack_size;                                                                      \
        if (stack_size < SM_STACK_SIZE) {                                                  \
            (V) = g_stack[gl_LocalInvocationIndex][stack_size];                            \
        } else {                                                                           \
            (V) = g_stack_spill[stack_size - SM_STACK_SIZE];                               \
        }                                                                                  \
    }

#define STACK_PUSH2(V, F)                                                                  \
    {                                                                                      \
        if (stack_size < SM_STACK_SIZE) {                                                  \
            g_stack[gl_LocalInvocationIndex][stack_size] = (V);                            \
            g_stack_factors[gl_LocalInvocationIndex][stack_size] = (F);                    \
        } else {                                                                           \
            g_stack_spill[stack_size - SM_STACK_SIZE] = (V);                               \
            g_stack_factors_spill[stack_size - SM_STACK_SIZE] = (F);                       \
        }                                                                                  \
        ++stack_size;                                                                      \
    }

#define STACK_POP2(V, F)                                                                   \
    {                                                                                      \
        --stack_size;                                                                      \
        if (stack_size < SM_STACK_SIZE) {                                                  \
            (V) = g_stack[gl_LocalInvocationIndex][stack_size];                            \
            (F) = g_stack_factors[gl_LocalInvocationIndex][stack_size];                    \
        } else {                                                                           \
            (V) = g_stack_spill[stack_size - SM_STACK_SIZE];                               \
            (F) = g_stack_factors_spill[stack_size - SM_STACK_SIZE];                       \
        }                                                                                  \
    }

// Equivalent to acosf(dot(a, b)), but more numerically stable
// Taken from PBRT source code
float angle_between(const vec3 v1, const vec3 v2) {
    if (dot(v1, v2) < 0) {
        return PI - 2 * portable_asinf(length(v1 + v2) / 2);
    } else {
        return 2 * portable_asinf(length(v2 - v1) / 2);
    }
}

float tan_angle(const vec3 a, const vec3 b) { return length(cross(a, b)) / dot(a, b); }

// Gram-Schmidt method
vec3 orthogonalize(const vec3 a, const vec3 b) {
    // we assume that a is normalized
    return normalize(b - dot(a, b) * a);
}

vec3 slerp(const vec3 start, const vec3 end, const float percent) {
    // Dot product - the cosine of the angle between 2 vectors.
    float cos_theta = dot(start, end);
    // Clamp it to be in the range of Acos()
    // This may be unnecessary, but floating point
    // precision can be a fickle mistress.
    cos_theta = clamp(cos_theta, -1.0, 1.0);
    // Acos(dot) returns the angle between start and end,
    // And multiplying that by percent returns the angle between
    // start and the final result.
    const float theta = portable_acosf(cos_theta) * percent;
    vec3 relative_vec = normalize(end - start * cos_theta);
    // Orthonormal basis
    // The final result.
    return start * cos(theta) + relative_vec * sin(theta);
}

uint mask_ray_depth(const uint depth) { return depth & 0x0fffffff; }
uint pack_ray_type(const int ray_type) { return uint(ray_type << 28); }
uint pack_ray_depth(const int diff_depth, const int spec_depth, const int refr_depth, const int transp_depth) {
    uint ret = 0;
    ret |= (diff_depth << 0);
    ret |= (spec_depth << 7);
    ret |= (refr_depth << 14);
    ret |= (transp_depth << 21);
    return ret;
}
int get_diff_depth(const uint depth) { return int(depth & 0x7f); }
int get_spec_depth(const uint depth) { return int(depth >> 7) & 0x7f; }
int get_refr_depth(const uint depth) { return int(depth >> 14) & 0x7f; }
int get_transp_depth(const uint depth) { return int(depth >> 21) & 0x7f; }
int get_total_depth(const uint depth) {
    return get_diff_depth(depth) + get_spec_depth(depth) + get_refr_depth(depth) + get_transp_depth(depth);
}
int get_ray_type(const uint depth) { return int(depth >> 28) & 0xf; }

bool is_indirect(const uint depth) {
    // not only transparency ray
    return (depth & 0x001fffff) != 0;
}

vec3 TonemapStandard(float inv_gamma, vec3 col) {
    [[unroll]] for (int i = 0; i < 3; ++i) {
        if (col[i] < 0.0031308) {
            col[i] = 12.92 * col[i];
        } else {
            col[i] = 1.055 * pow(col[i], (1.0 / 2.4)) - 0.055;
        }
    }

    if (inv_gamma != 1.0) {
        col = pow(col, vec3(inv_gamma));
    }

    return saturate(col);
}

vec4 TonemapStandard(float inv_gamma, vec4 col) {
    return vec4(TonemapStandard(inv_gamma, col.xyz), col.w);
}

vec3 TonemapLUT(sampler3D lut, float inv_gamma, vec3 col) {
    const vec3 encoded = col / (col + 1.0);

    const float LUT_DIMS = 48.0;
    const vec3 uv = encoded * ((LUT_DIMS - 1.0) / LUT_DIMS) + (0.5 / LUT_DIMS);

    vec3 ret = textureLod(lut, uv, 0.0).xyz;
    if (inv_gamma != 1.0) {
        ret = pow(ret, vec3(inv_gamma));
    }

    return ret;
}

vec4 TonemapLUT(sampler3D lut, float inv_gamma, vec4 col) {
    return vec4(TonemapLUT(lut, inv_gamma, col.xyz), col.w);
}

// https://gpuopen.com/learn/optimized-reversible-tonemapper-for-resolve/
vec3 reversible_tonemap(vec3 c) { return c / (max(c.x, max(c.y, c.z)) + 1.0); }
vec3 reversible_tonemap_invert(vec3 c) { return c / (1.0 - max(c.x, max(c.y, c.z))); }

vec4 reversible_tonemap(vec4 c) { return vec4(reversible_tonemap(c.xyz), c.w); }
vec4 reversible_tonemap_invert(vec4 c) { return vec4(reversible_tonemap_invert(c.xyz), c.w); }

#define pack_unorm_16(x) uint(x * 65535.0)
#define unpack_unorm_16(x) saturate(float(x) / 65535.0)

#define length2(x) dot(x, x)
#define sqr(x) ((x) * (x))

#endif // COMMON_GLSL