#ifndef COMMON_GLSL
#define COMMON_GLSL

#extension GL_EXT_control_flow_attributes :  require

#include "types.h"

//
// Useful macros for debugging
//
#define USE_VNDF_GGX_SAMPLING 1
#define USE_NEE 1
#define USE_PATH_TERMINATION 1
// #define FORCE_TEXTURE_LOD 0

int hash(const int x) {
    uint ret = uint(x);
    ret = ((ret >> 16) ^ ret) * 0x45d9f3b;
    ret = ((ret >> 16) ^ ret) * 0x45d9f3b;
    ret = (ret >> 16) ^ ret;
    return int(ret);
}

float construct_float(uint m) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    const float  f = uintBitsToFloat(m);   // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

float lum(const vec3 color) {
    return 0.212671 * color[0] + 0.715160 * color[1] + 0.072169 * color[2];
}

vec3 safe_invert(vec3 v) {
    vec3 inv_v = 1.0f / v;

    if (v.x <= FLT_EPS && v.x >= 0) {
        inv_v.x = FLT_MAX;
    } else if (v.x >= -FLT_EPS && v.x < 0) {
        inv_v.x = -FLT_MAX;
    }

    if (v.y <= FLT_EPS && v.y >= 0) {
        inv_v.y = FLT_MAX;
    } else if (v.y >= -FLT_EPS && v.y < 0) {
        inv_v.y = -FLT_MAX;
    }

    if (v.z <= FLT_EPS && v.z >= 0) {
        inv_v.z = FLT_MAX;
    } else if (v.z >= -FLT_EPS && v.z < 0) {
        inv_v.z = -FLT_MAX;
    }

    return inv_v;
}

vec3 srgb_to_rgb(vec3 col) {
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

vec3 YCoCg_to_RGB(vec4 col) {
    float scale = (col.b * (255.0 / 8.0)) + 1.0;
    float Y = col.a;
    float Co = (col.r - (0.5 * 256.0 / 255.0)) / scale;
    float Cg = (col.g - (0.5 * 256.0 / 255.0)) / scale;

    vec3 col_rgb;
    col_rgb.r = Y + Co - Cg;
    col_rgb.g = Y + Cg;
    col_rgb.b = Y - Co - Cg;

    return col_rgb;
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

int total_depth(const ray_data_t r) {
    const int diff_depth = r.depth & 0x000000ff;
    const int spec_depth = (r.depth >> 8) & 0x000000ff;
    const int refr_depth = (r.depth >> 16) & 0x000000ff;
    const int transp_depth = (r.depth >> 24) & 0x000000ff;
    return diff_depth + spec_depth + refr_depth + transp_depth;
}

int total_depth(const shadow_ray_t r) {
    const int diff_depth = r.depth & 0x000000ff;
    const int spec_depth = (r.depth >> 8) & 0x000000ff;
    const int refr_depth = (r.depth >> 16) & 0x000000ff;
    const int transp_depth = (r.depth >> 24) & 0x000000ff;
    return diff_depth + spec_depth + refr_depth + transp_depth;
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

    return clamp(col, vec3(0.0), vec3(1.0));
}

vec4 TonemapStandard(float inv_gamma, vec4 col) {
    return vec4(TonemapStandard(inv_gamma, col.xyz), col.w);
}

vec3 TonemapLUT(sampler3D lut, float inv_gamma, vec3 col) {
    const vec3 encoded = col / (col + 1.0);

    // Align the encoded range to texel centers
    const float LUT_DIMS = 48.0;
    const vec3 uv = encoded * ((LUT_DIMS - 1.0) / LUT_DIMS) + 0.5 / LUT_DIMS;

    vec3 ret = textureLod(lut, uv, 0.0).xyz;
    if (inv_gamma != 1.0) {
        ret = pow(ret, vec3(inv_gamma));
    }

    return ret;
}

vec4 TonemapLUT(sampler3D lut, float inv_gamma, vec4 col) {
    return vec4(TonemapLUT(lut, inv_gamma, col.xyz), col.w);
}

// Manual interpolation gives better result for some reason
vec3 TonemapLUT_manual(sampler3D lut, float inv_gamma, vec3 col) {
    const vec3 encoded = col / (col + 1.0);

    // Align the encoded range to texel centers
    const float LUT_DIMS = 48;
    const vec3 uv = encoded * (LUT_DIMS - 1.0) + 0.5;
    const ivec3 xyz = ivec3(uv);
    const ivec3 xyz_next = min(xyz + 1, ivec3(LUT_DIMS - 1));
    const vec3 f = fract(uv);

    const int ix = xyz.x, iy = xyz.y, iz = xyz.z;
    const int jx = xyz_next.x, jy = xyz_next.y, jz = xyz_next.z;
    const float fx = f.x, fy = f.y, fz = f.z;

    const vec3 c000 = texelFetch(lut, ivec3(ix, iy, iz), 0).xyz;
    const vec3 c001 = texelFetch(lut, ivec3(jx, iy, iz), 0).xyz;
    const vec3 c010 = texelFetch(lut, ivec3(ix, jy, iz), 0).xyz;
    const vec3 c011 = texelFetch(lut, ivec3(jx, jy, iz), 0).xyz;
    const vec3 c100 = texelFetch(lut, ivec3(ix, iy, jz), 0).xyz;
    const vec3 c101 = texelFetch(lut, ivec3(jx, iy, jz), 0).xyz;
    const vec3 c110 = texelFetch(lut, ivec3(ix, jy, jz), 0).xyz;
    const vec3 c111 = texelFetch(lut, ivec3(jx, jy, jz), 0).xyz;

    const vec3 c00x = (1.0 - fx) * c000 + fx * c001;
    const vec3 c01x = (1.0 - fx) * c010 + fx * c011;
    const vec3 c10x = (1.0 - fx) * c100 + fx * c101;
    const vec3 c11x = (1.0 - fx) * c110 + fx * c111;

    const vec3 c0xx = (1.0 - fy) * c00x + fy * c01x;
    const vec3 c1xx = (1.0 - fy) * c10x + fy * c11x;

    vec3 cxxx = (1.0 - fz) * c0xx + fz * c1xx;

    vec3 ret = cxxx;
    if (inv_gamma != 1.0) {
        ret = pow(ret, vec3(inv_gamma));
    }

    return ret;
}

vec4 TonemapLUT_manual(sampler3D lut, float inv_gamma, vec4 col) {
    return vec4(TonemapLUT_manual(lut, inv_gamma, col.xyz), col.w);
}

// https://gpuopen.com/learn/optimized-reversible-tonemapper-for-resolve/
vec3 reversible_tonemap(vec3 c) { return c / (max(c.x, max(c.y, c.z)) + 1.0); }
vec3 reversible_tonemap_invert(vec3 c) { return c / (1.0 - max(c.x, max(c.y, c.z))); }

vec4 reversible_tonemap(vec4 c) { return vec4(reversible_tonemap(c.xyz), c.w); }
vec4 reversible_tonemap_invert(vec4 c) { return vec4(reversible_tonemap_invert(c.xyz), c.w); }

#define pack_unorm_16(x) uint(x * 65535.0)
#define unpack_unorm_16(x) clamp(float(x) / 65535.0, 0.0, 1.0)

#define length2(x) dot(x, x)
#define sqr(x) ((x) * (x))

#endif // COMMON_GLSL