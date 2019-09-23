R"(

__constant sampler_t LINEAR_SAMPLER = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
__constant sampler_t NEAREST_SAMPLER = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__constant float2 tex_atlas_size = (float2)(TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE);

float4 rgbe_to_rgb(float4 rgbe) {
    float f = native_exp2(rgbe.w * 255.0f - 128.0f);
    return (float4)(rgbe.xyz * f, 1.0f);
}

float4 srgb_to_rgb(float4 col) {
    return (float4)(
        (col.x > 0.04045f) ? (native_powr((col.x + 0.055f) / 1.055f, 2.4f)) : (col.x / 12.92f),
        (col.y > 0.04045f) ? (native_powr((col.y + 0.055f) / 1.055f, 2.4f)) : (col.y / 12.92f),
        (col.z > 0.04045f) ? (native_powr((col.z + 0.055f) / 1.055f, 2.4f)) : (col.z / 12.92f),
        col.w);
}

float4 rgb_to_srgb(float4 col) {
    return (float4)(
            (col.x > 0.0031308f) ? (native_powr(1.055f * col.x, (1.0f / 2.4f)) - 0.055f) : (12.92f * col.x),
            (col.y > 0.0031308f) ? (native_powr(1.055f * col.y, (1.0f / 2.4f)) - 0.055f) : (12.92f * col.y),
            (col.z > 0.0031308f) ? (native_powr(1.055f * col.z, (1.0f / 2.4f)) - 0.055f) : (12.92f * col.z),
            col.w);
}

float get_texture_lod(__global const texture_t *texture, const float2 duv_dx, const float2 duv_dy) {
    const float2 _duv_dx = duv_dx * (float2)(texture->width & TEXTURE_WIDTH_BITS, texture->height),
                 _duv_dy = duv_dy * (float2)(texture->width & TEXTURE_WIDTH_BITS, texture->height);

    const float2 _diagonal = _duv_dx + _duv_dy;

    const float dim = fmin(fmin(dot(_duv_dx, _duv_dx), dot(_duv_dy, _duv_dy)), dot(_diagonal, _diagonal));

    float lod = native_log2(dim);
    lod = clamp(0.5f * lod - 1.0f, 0.0f, (float)MAX_MIP_LEVEL);

    return lod;
}

float4 SampleTextureBilinear(__read_only image2d_array_t texture_atlas, __global const texture_t *texture,
                              const float2 uvs, int lod) {
    const float2 uvs1 = TransformUVs(uvs, tex_atlas_size, texture, lod);

    float4 coord1 = (float4)(uvs1, (float)texture->page[lod], 0);

    return read_imagef(texture_atlas, LINEAR_SAMPLER, coord1);
}

float4 SampleTextureTrilinear(__read_only image2d_array_t texture_atlas, __global const texture_t *texture,
                              const float2 uvs, float lod) {
    const float2 uvs1 = TransformUVs(uvs, tex_atlas_size, texture, floor(lod));
    const float2 uvs2 = TransformUVs(uvs, tex_atlas_size, texture, ceil(lod));

    int page1 = (int)min(floor(lod), (float)MAX_MIP_LEVEL);
    int page2 = (int)min(ceil(lod), (float)MAX_MIP_LEVEL);

    float4 coord1 = (float4)(uvs1, (float)texture->page[page1], 0);
    float4 coord2 = (float4)(uvs2, (float)texture->page[page2], 0);

    float4 tex_col1 = read_imagef(texture_atlas, LINEAR_SAMPLER, coord1);
    float4 tex_col2 = read_imagef(texture_atlas, LINEAR_SAMPLER, coord2);

    return mix(tex_col1, tex_col2, lod - floor(lod));
}

float4 SampleTextureAnisotropic(__read_only image2d_array_t texture_atlas, __global const texture_t *texture,
                                const float2 uvs, const float2 duv_dx, const float2 duv_dy) {
    float2 _duv_dx = fabs(duv_dx * (float2)(texture->width & TEXTURE_WIDTH_BITS, texture->height));
    float2 _duv_dy = fabs(duv_dy * (float2)(texture->width & TEXTURE_WIDTH_BITS, texture->height));

    float l1 = fast_length(_duv_dx);
    float l2 = fast_length(_duv_dy);

    float lod;
    float k;
    float2 step;

    if (l1 <= l2) {
        lod = native_log2(fmin(_duv_dx.x, _duv_dx.y));
        k = l1 / l2;
        step = duv_dy;
    } else {
        lod = native_log2(fmin(_duv_dy.x, _duv_dy.y));
        k = l2 / l1;
        step = duv_dx;
    }

    lod = clamp(lod, 0.0f, (float)MAX_MIP_LEVEL);

    float2 _uvs = uvs - step * 0.5f;

    int num = clamp((int)(2.0f / k), 1, 4);
    step = step / num;

    float4 res = 0;
    
    int lod1 = (int)floor(lod);
    int lod2 = (int)ceil(lod);

    int page1 = texture->page[lod1];
    int page2 = texture->page[lod2];

    float2 pos1 = (float2)((float)texture->pos[lod1][0] + 0.5f, (float)texture->pos[lod1][1] + 0.5f);
    float2 size1 = (float2)((float)((texture->width & TEXTURE_WIDTH_BITS) >> lod1), (float)(texture->height >> lod1));
    float4 coord1 = (float4)(0.0f, 0.0f, (float)page1, 0);

    float2 pos2 = (float2)((float)texture->pos[lod2][0] + 0.5f, (float)texture->pos[lod2][1] + 0.5f);
    float2 size2 = (float2)((float)((texture->width & TEXTURE_WIDTH_BITS) >> lod2), (float)(texture->height >> lod2));
    float4 coord2 = (float4)(0.0f, 0.0f, (float)page2, 0);

    const float kz = lod - floor(lod);

    for (int i = 0; i < num; i++) {
        _uvs = _uvs - floor(_uvs);

        coord1.xy = (pos1 + _uvs * size1) / tex_atlas_size;
        res += (1 - kz) * read_imagef(texture_atlas, LINEAR_SAMPLER, coord1);

        if (kz > 0.0001f) {
            coord2.xy = (pos2 + _uvs * size2) / tex_atlas_size;
            res += kz * read_imagef(texture_atlas, LINEAR_SAMPLER, coord2);
        }

        _uvs = _uvs + step;
    }

    return res / num;
}

float4 SampleTextureLatlong_RGBE(__read_only image2d_array_t texture_atlas, __global const texture_t *t, const float3 dir) {
    float2 kk = 1.0f / tex_atlas_size;
    
    float theta = acospi(clamp(dir.y, -1.0f, 1.0f));
    float r = length(dir.xz);
    float u = 0.5f * acospi(r > FLT_EPS ? clamp(dir.x/r, -1.0f, 1.0f) : 0.0f);
    if (dir.z < 0) u = 1.0f - u;

    float2 pos = (float2)((float)t->pos[0][0], (float)t->pos[0][1]);
    float2 size = (float2)((float)(t->width & TEXTURE_WIDTH_BITS), (float)t->height);

    float2 uvs = pos + (float2)(u, theta) * size + (float2)(1.0f, 1.0f);
    const float kx = uvs.x - floor(uvs.x), ky = uvs.y - floor(uvs.y);
    uvs /= tex_atlas_size;

    const float4 coord00 = (float4)(uvs + (float2)(0.0f, 0.0f), (float)t->page[0], 0);
    const float4 coord01 = (float4)(uvs + (float2)(kk.x, 0.0f), (float)t->page[0], 0);
    const float4 coord10 = (float4)(uvs + (float2)(0.0f, kk.y), (float)t->page[0], 0);
    const float4 coord11 = (float4)(uvs + (float2)(kk.x, kk.y), (float)t->page[0], 0);

    const float4 c00 = rgbe_to_rgb(read_imagef(texture_atlas, NEAREST_SAMPLER, coord00));
    const float4 c01 = rgbe_to_rgb(read_imagef(texture_atlas, NEAREST_SAMPLER, coord01));
    const float4 c10 = rgbe_to_rgb(read_imagef(texture_atlas, NEAREST_SAMPLER, coord10));
    const float4 c11 = rgbe_to_rgb(read_imagef(texture_atlas, NEAREST_SAMPLER, coord11));

    const float4 c0X = c01 * kx + c00 * (1.0f - kx),
                 c1X = c11 * kx + c10 * (1.0f - kx);

    return c1X * ky + c0X * (1.0f - ky);
}

__kernel
void TextureDebugPage(__read_only image2d_array_t texture_atlas, int page, __write_only image2d_t frame_buf) {
    int i = get_global_id(0),
        j = get_global_id(1);

    float x = 1.0f * ((float)i) / get_global_size(0);
    float y = 1.0f * ((float)j) / get_global_size(1);

    float4 coord = (float4)(x, y, (float)page, 0);
    float4 col = read_imagef(texture_atlas, LINEAR_SAMPLER, coord);

    write_imagef(frame_buf, (int2)(i, j), col);
}

)"