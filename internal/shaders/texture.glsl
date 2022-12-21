#ifndef TEXTURE_GLSL
#define TEXTURE_GLSL

#extension GL_EXT_nonuniform_qualifier : require

#include "types.h"

vec3 rgbe_to_rgb(vec4 rgbe) {
    const float f = exp2(255.0 * rgbe.w - 128.0);
    return rgbe.xyz * f;
}

#if BINDLESS

layout(set = 1, binding = 0) uniform sampler2D g_textures[];

ivec2 texSize(const uint index) {
    return textureSize(g_textures[nonuniformEXT(index & 0x00ffffff)], 0);
}

vec4 SampleBilinear(const uint index, const vec2 uvs, const int lod, const bool maybe_YCoCg, const bool maybe_SRGB) {
    vec4 res = textureLod(g_textures[nonuniformEXT(index & 0x00ffffff)], uvs, float(lod));
    if (maybe_YCoCg && (index & TEX_YCOCG_BIT) != 0) {
        res.rgb = YCoCg_to_RGB(res);
        res.a = 1.0;
    }
    if (maybe_SRGB && (index & TEX_SRGB_BIT) != 0) {
        res.rgb = srgb_to_rgb(res.rgb);
    }
    return res;
}

vec4 SampleBilinear(const uint index, const vec2 uvs, const int lod) {
    return SampleBilinear(index, uvs, lod, false, false);
}

vec3 SampleLatlong_RGBE(const uint index, const vec3 dir, const float y_rotation) {
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
    ivec2 size = textureSize(g_textures[tex], 0);
    const vec2 uvs = vec2(u, theta) * vec2(size);

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
}

#else // BINDLESS

layout(std430, binding = TEXTURES_BUF_SLOT) readonly buffer Textures {
    atlas_texture_t g_textures[];
};

layout(binding = TEXTURE_ATLASES_SLOT) uniform sampler2DArray g_atlases[7];

ivec2 texSize(const uint index) {
    const atlas_texture_t t = g_textures[index];
    const int w = int(t.size & ATLAS_TEX_WIDTH_BITS);
    const int h = int((t.size >> 16) & ATLAS_TEX_HEIGHT_BITS);
    return ivec2(w, h);
}

vec2 TransformUV(const vec2 _uv, const atlas_texture_t t, const int mip_level) {
    const vec2 pos = vec2(float(t.pos[mip_level] & 0xffff), float((t.pos[mip_level] >> 16) & 0xffff));
    vec2 size = {float(t.size & ATLAS_TEX_WIDTH_BITS), float((t.size >> 16) & ATLAS_TEX_HEIGHT_BITS)};
    if (((t.size >> 16) & ATLAS_TEX_MIPS_BIT) != 0) {
        size = vec2(float((t.size & ATLAS_TEX_WIDTH_BITS) >> mip_level),
                    float(((t.size >> 16) & ATLAS_TEX_HEIGHT_BITS) >> mip_level));
    }
    const vec2 uv = fract(_uv);
    return pos + uv * size + 1.0;
}

vec4 SampleBilinear(const uint index, const vec2 uvs, const int lod, const bool maybe_YCoCg, const bool maybe_SRGB) {
    const atlas_texture_t t = g_textures[index];
    vec2 _uvs = TransformUV(uvs, t, lod) / vec2(TEXTURE_ATLAS_SIZE);

    const float page = float((t.page[lod / 4] >> (lod % 4) * 8) & 0xff);
    vec4 res = textureLod(g_atlases[nonuniformEXT(t.atlas)], vec3(_uvs, page), 0.0);
    if (maybe_YCoCg && t.atlas == 4) {
        res.rgb = YCoCg_to_RGB(res);
        res.a = 1.0;
    }
    if (maybe_SRGB && (t.size & ATLAS_TEX_SRGB_BIT) != 0) {
        res.rgb = srgb_to_rgb(res.rgb);
    }
    return res;
}

vec4 SampleBilinear(const uint index, const vec2 uvs, const int lod) {
    return SampleBilinear(index, uvs, lod, false, false);
}

vec3 SampleLatlong_RGBE(const atlas_texture_t t, const vec3 dir, float y_rotation) {
    const float theta = acos(clamp(dir[1], -1.0, 1.0)) / PI;
    float phi = atan(dir[2], dir[0]) + y_rotation;
    if (phi < 0) {
        phi += 2 * PI;
    }
    if (phi > 2 * PI) {
        phi -= 2 * PI;
    }

    const float u = fract(0.5 * phi / PI);
    const vec2 uvs = TransformUV(vec2(u, theta), t, 0);

    const int page = int(t.page[0] & 0xff);
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
}

#endif // BINDLESS

#endif // TEXTURE_GLSL
