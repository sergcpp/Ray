#ifndef COMMON_GLSL
#define COMMON_GLSL

#extension GL_EXT_control_flow_attributes :  require

#include "types.glsl"

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

float get_texture_lod(const atlas_texture_t t, const float lambda) {
    const int w = int(t.size & ATLAS_TEX_WIDTH_BITS);
    const int h = int((t.size >> 16) & ATLAS_TEX_HEIGHT_BITS);
    return get_texture_lod(ivec2(w, h), lambda);
}

vec3 TransformNormal(vec3 n, mat4 inv_xform) {
    return (transpose(inv_xform) * vec4(n, 0.0)).xyz;
}

#define pack_unorm_16(x) uint(x * 65535.0)
#define unpack_unorm_16(x) (float(x) / 65535.0)

#endif // COMMON_GLSL