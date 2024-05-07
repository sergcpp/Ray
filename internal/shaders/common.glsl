#ifndef COMMON_GLSL
#define COMMON_GLSL

#extension GL_EXT_control_flow_attributes :  require

#include "types.h"

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
        ret[i] = (abs(v[i]) > FLT_EPS) ? (1.0 / v[i]) : _copysign(FLT_MAX, v[i]);
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

//
// asinf/acosf implemantation. Taken from apple libm source code
//

// Return arcsine(x) given that .57 < x
float asin_tail(const float x) {
    return (PI / 2) - ((x + 2.71745038) * x + 14.0375338) * (0.00440413551 * ((x - 8.31223679) * x + 25.3978882)) *
                          sqrt(1 - x);
}

// Taken from apple libm source code
float portable_asinf(float x) {
    const bool negate = (x < 0.0);
    if (abs(x) > 0.57) {
        const float ret = asin_tail(abs(x));
        return negate ? -ret : ret;
    } else {
        const float x2 = x * x;
        return float(x + (0.0517513789 * ((x2 + 1.83372748) * x2 + 1.56678128)) * x *
                             (x2 * ((x2 - 1.48268414) * x2 + 2.05554748)));
    }
}

float acos_positive_tail(const float x) {
    return (((x + 2.71850395) * x + 14.7303705)) * (0.00393401226 * ((x - 8.60734272) * x + 27.0927486)) *
           sqrt(1 - x);
}

float acos_negative_tail(const float x) {
    return PI - (((x - 2.71850395) * x + 14.7303705)) * (0.00393401226 * ((x + 8.60734272) * x + 27.0927486)) *
                    sqrt(1 + x);
}

float portable_acosf(float x) {
    if (x < -0.62) {
        return acos_negative_tail(x);
    } else if (x <= 0.62) {
        const float x2 = x * x;
        return (PI / 2) - x -
               (0.0700945929 * x * ((x2 + 1.57144082) * x2 + 1.25210774)) *
                   (x2 * ((x2 - 1.53757966) * x2 + 1.89929986));
    } else {
        return acos_positive_tail(x);
    }
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

    // Align the encoded range to texel centers
    const float LUT_DIMS = 48.0;
    const vec3 uv = (encoded + (0.5 / LUT_DIMS)) * ((LUT_DIMS - 1.0) / LUT_DIMS);

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