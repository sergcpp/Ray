#ifndef TEXTURE_GLSL
#define TEXTURE_GLSL

#extension GL_EXT_nonuniform_qualifier : require

#include "types.glsl"

#if BINDLESS

layout(set = 1, binding = 0) uniform sampler2D g_textures[];

ivec2 texSize(const uint index) {
    return textureSize(g_textures[nonuniformEXT(index & 0x00ffffff)], 0);
}

vec4 SampleBilinear(const uint index, const vec2 uvs, const int lod) {
    vec4 res = textureLod(g_textures[nonuniformEXT(index & 0x00ffffff)], uvs, float(lod));
    if ((index & TEX_YCOCG_BIT) != 0) {
        res.rgb = YCoCg_to_RGB(res);
        res.a = 1.0;
    }
    return res;
}

#else // BINDLESS

vec2 TransformUV(const vec2 _uv, const vec2 tex_atlas_size, const atlas_texture_t t, const int mip_level) {
    const vec2 pos = vec2(float(t.pos[mip_level] & 0xffff), float((t.pos[mip_level] >> 16) & 0xffff));
    vec2 size = {float(t.size & ATLAS_TEX_WIDTH_BITS), float((t.size >> 16) & ATLAS_TEX_HEIGHT_BITS)};
    if (((t.size >> 16) & ATLAS_TEX_MIPS_BIT) != 0) {
        size = vec2(float((t.size & ATLAS_TEX_WIDTH_BITS) >> mip_level),
                    float(((t.size >> 16) & ATLAS_TEX_HEIGHT_BITS) >> mip_level));
    }
    const vec2 uv = _uv - floor(_uv);
    vec2 res = pos + uv * size + 1.0;
    res /= tex_atlas_size;
    return res;
}

vec4 SampleBilinear(sampler2DArray atlases[7], const atlas_texture_t t, const vec2 uvs, const int lod) {
    const vec2 atlas_size = vec2(TEXTURE_ATLAS_SIZE);
    vec2 _uvs = TransformUV(uvs, atlas_size, t, lod);
    //_uvs = _uvs * atlas_size - 0.5;

    const float page = float((t.page[lod / 4] >> (lod % 4) * 8) & 0xff);
    vec4 res = textureLod(atlases[nonuniformEXT(t.atlas)], vec3(_uvs, page), 0.0);
    if (t.atlas == 4) {
        res.rgb = YCoCg_to_RGB(res);
        res.a = 1.0;
    }
    return res;
}

#endif // BINDLESS

#endif // TEXTURE_GLSL