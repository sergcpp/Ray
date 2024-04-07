#ifndef TEXTURE_GLSL
#define TEXTURE_GLSL

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

#define USE_STOCH_TEXTURE_FILTERING 1

#include "types.h"

vec3 rgbe_to_rgb(vec4 rgbe) {
    const float f = exp2(255.0 * rgbe.w - 128.0);
    return rgbe.xyz * f;
}

#if BINDLESS

layout(binding = TEXTURES_SAMPLER_SLOT) uniform sampler g_sampler;
layout(binding = TEXTURES_SIZE_SLOT) readonly buffer TexSizes {
    uint g_tex_sizes[];
};
layout(set = 1, binding = 0) uniform texture2D g_textures[];

ivec2 texSize(uint index) {
    // NOTE: Disabled due to artifacts on AMD cards
    // return textureSize(g_textures[nonuniformEXT(index & 0x00ffffff)], 0);

    const uint packed_size = g_tex_sizes[index & 0x00ffffff];
    return ivec2(int(packed_size >> 16), int(packed_size & 0xffff));
}

vec4 SampleBilinear(const uint index, const vec2 uvs, int lod, const vec2 rand, const bool maybe_YCoCg, const bool maybe_SRGB) {
#if USE_STOCH_TEXTURE_FILTERING
    //const ivec2 size = textureSize(sampler2D(g_textures[nonuniformEXT(index & 0x00ffffff)], g_sampler), lod);
    const ivec2 size = max(texSize(index & 0x00ffffff) >> lod, ivec2(1, 1));
    vec4 res = textureLod(sampler2D(g_textures[nonuniformEXT(index & 0x00ffffff)], g_sampler), uvs + (rand - 0.5) / vec2(size), float(lod));
#else // USE_STOCH_TEXTURE_FILTERING
    vec4 res = textureLod(sampler2D(g_textures[nonuniformEXT(index & 0x00ffffff)], g_sampler), uvs, float(lod));
#endif // USE_STOCH_TEXTURE_FILTERING
    if (maybe_YCoCg && (index & TEX_YCOCG_BIT) != 0) {
        res.rgb = YCoCg_to_RGB(res);
        res.a = 1.0;
    }
    if (maybe_SRGB && (index & TEX_SRGB_BIT) != 0) {
        res.rgb = srgb_to_linear(res.rgb);
    }
    return res;
}

vec4 SampleBilinear(const uint index, const vec2 uvs, const int lod, const vec2 rand) {
    return SampleBilinear(index, uvs, lod, rand, false, false);
}

vec3 SampleLatlong_RGBE(const uint index, ivec2 tex_size, const vec3 dir, const float y_rotation, const vec2 rand) {
    const float theta = acos(clamp(dir[1], -1.0, 1.0)) / PI;
    const float r = sqrt(dir[0] * dir[0] + dir[2] * dir[2]);

    float phi = atan(dir[2], dir[0]) + y_rotation;
    if (phi < 0) {
        phi += 2 * PI;
    }
    if (phi > 2 * PI) {
        phi -= 2 * PI;
    }

    const float u = fract(0.5 * phi / PI);

    const uint tex = (index & 0x00ffffff);
    const vec2 uvs = vec2(u, theta) * vec2(tex_size);

#if USE_STOCH_TEXTURE_FILTERING
    const vec4 p00 = texelFetch(g_textures[nonuniformEXT(tex)], min(ivec2(uvs + rand - 0.5), tex_size - 1), 0);
    return rgbe_to_rgb(p00);
#else // USE_STOCH_TEXTURE_FILTERING
    const vec4 p00 = texelFetchOffset(g_textures[nonuniformEXT(tex)], ivec2(uvs), 0, ivec2(0, 0));
    const vec4 p01 = texelFetchOffset(g_textures[nonuniformEXT(tex)], ivec2(uvs), 0, ivec2(1, 0));
    const vec4 p10 = texelFetchOffset(g_textures[nonuniformEXT(tex)], ivec2(uvs), 0, ivec2(0, 1));
    const vec4 p11 = texelFetchOffset(g_textures[nonuniformEXT(tex)], ivec2(uvs), 0, ivec2(1, 1));

    const vec2 k = fract(uvs);

    const vec3 _p00 = rgbe_to_rgb(p00), _p01 = rgbe_to_rgb(p01);
    const vec3 _p10 = rgbe_to_rgb(p10), _p11 = rgbe_to_rgb(p11);

    const vec3 p0X = _p01 * k[0] + _p00 * (1 - k[0]);
    const vec3 p1X = _p11 * k[0] + _p10 * (1 - k[0]);

    return (p1X * k[1] + p0X * (1 - k[1]));
#endif // USE_STOCH_TEXTURE_FILTERING
}

vec3 SampleLatlong_RGBE(const uint index, const vec3 dir, const float y_rotation, const vec2 rand) {
    ivec2 tex_size = textureSize(g_textures[nonuniformEXT(index & 0x00ffffff)], 0);
    return SampleLatlong_RGBE(index, tex_size, dir, y_rotation, rand);
}

#else // BINDLESS

layout(std430, binding = TEXTURES_BUF_SLOT) readonly buffer Textures {
    atlas_texture_t g_textures[];
};

layout(binding = TEXTURE_ATLASES_SLOT) uniform sampler2DArray g_atlases[8];

ivec2 texSize(const uint index) {
    const atlas_texture_t t = g_textures[index];
    const int w = int(t.size & ATLAS_TEX_WIDTH_BITS);
    const int h = int((t.size >> 16) & ATLAS_TEX_HEIGHT_BITS);
    return ivec2(w, h);
}

vec2 TransformUV(const vec2 _uv, const vec2 px_off, const atlas_texture_t t, const int mip_level) {
    const vec2 pos = vec2(float(t.pos[mip_level] & 0xffff), float((t.pos[mip_level] >> 16) & 0xffff));
    vec2 size = {float(t.size & ATLAS_TEX_WIDTH_BITS), float((t.size >> 16) & ATLAS_TEX_HEIGHT_BITS)};
    if (((t.size >> 16) & ATLAS_TEX_MIPS_BIT) != 0) {
        size = vec2(float((t.size & ATLAS_TEX_WIDTH_BITS) >> mip_level),
                    float(((t.size >> 16) & ATLAS_TEX_HEIGHT_BITS) >> mip_level));
        size = max(size, vec2(MIN_ATLAS_TEXTURE_SIZE));
    }
    const vec2 uv = fract(_uv + (px_off / size));
    return pos + uv * size;
}

vec4 SampleBilinear(const uint index, const vec2 uvs, const int lod, const vec2 rand, const bool maybe_YCoCg, const bool maybe_SRGB) {
    const atlas_texture_t t = g_textures[index];

#if USE_STOCH_TEXTURE_FILTERING
    vec2 _uvs = TransformUV(uvs, rand - 0.5, t, lod) / vec2(TEXTURE_ATLAS_SIZE);
#else // USE_STOCH_TEXTURE_FILTERING
    // NOTE: This branch is incorrect now!
    vec2 _uvs = TransformUV(uvs, vec2(0.0), t, lod) / vec2(TEXTURE_ATLAS_SIZE);
#endif // USE_STOCH_TEXTURE_FILTERING

    const float page = float((t.page[lod / 4] >> (lod % 4) * 8) & 0xff);
    vec4 res = textureLod(g_atlases[nonuniformEXT(t.atlas)], vec3(_uvs, page), 0.0);
    if (maybe_YCoCg && ((t.size >> 16) & ATLAS_TEX_YCOCG_BIT) != 0) {
        res.rgb = YCoCg_to_RGB(res);
        res.a = 1.0;
    }
    if (maybe_SRGB && (t.size & ATLAS_TEX_SRGB_BIT) != 0) {
        res.rgb = srgb_to_linear(res.rgb);
    }
    return res;
}

vec4 SampleBilinear(const uint index, const vec2 uvs, const int lod, const vec2 rand) {
    return SampleBilinear(index, uvs, lod, rand, false, false);
}

vec3 SampleLatlong_RGBE(const atlas_texture_t t, const vec3 dir, float y_rotation, const vec2 rand) {
    const float theta = acos(clamp(dir[1], -1.0, 1.0)) / PI;
    float phi = atan(dir[2], dir[0]) + y_rotation;
    if (phi < 0) {
        phi += 2 * PI;
    }
    if (phi > 2 * PI) {
        phi -= 2 * PI;
    }

    const float u = fract(0.5 * phi / PI);
    vec2 uvs = TransformUV(vec2(u, theta), rand - 0.5, t, 0);
    const int page = int(t.page[0] & 0xff);

#if USE_STOCH_TEXTURE_FILTERING
    const vec4 p00 = texelFetch(g_atlases[nonuniformEXT(t.atlas)], ivec3(uvs, page), 0);
    return rgbe_to_rgb(p00);
#else // USE_STOCH_TEXTURE_FILTERING
    const vec4 p00 = texelFetchOffset(g_atlases[nonuniformEXT(t.atlas)], ivec3(uvs, page), 0, ivec2(0, 0));
    const vec4 p01 = texelFetchOffset(g_atlases[nonuniformEXT(t.atlas)], ivec3(uvs, page), 0, ivec2(1, 0));
    const vec4 p10 = texelFetchOffset(g_atlases[nonuniformEXT(t.atlas)], ivec3(uvs, page), 0, ivec2(0, 1));
    const vec4 p11 = texelFetchOffset(g_atlases[nonuniformEXT(t.atlas)], ivec3(uvs, page), 0, ivec2(1, 1));

    const vec2 k = fract(uvs);

    const vec3 _p00 = rgbe_to_rgb(p00), _p01 = rgbe_to_rgb(p01);
    const vec3 _p10 = rgbe_to_rgb(p10), _p11 = rgbe_to_rgb(p11);

    const vec3 p0X = _p01 * k[0] + _p00 * (1 - k[0]);
    const vec3 p1X = _p11 * k[0] + _p10 * (1 - k[0]);

    return (p1X * k[1] + p0X * (1 - k[1]));
#endif // USE_STOCH_TEXTURE_FILTERING
}

#endif // BINDLESS

#endif // TEXTURE_GLSL
