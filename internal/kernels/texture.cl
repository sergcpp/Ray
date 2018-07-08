R"(

__constant sampler_t TEX_SAMPLER = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

float4 SampleTextureBilinear(__read_only image2d_array_t texture_atlas, __global const texture_t *texture,
                              const float2 uvs, int lod) {
    const float2 tex_atlas_size = (float2)(get_image_width(texture_atlas), get_image_height(texture_atlas));
    
    const float2 uvs1 = TransformUVs(uvs, tex_atlas_size, texture, lod);

    float4 coord1 = (float4)(uvs1, (float)texture->page[lod], 0);

    return read_imagef(texture_atlas, TEX_SAMPLER, coord1);
}

float4 SampleTextureTrilinear(__read_only image2d_array_t texture_atlas, __global const texture_t *texture,
                              const float2 uvs, float lod) {
    const float2 tex_atlas_size = (float2)(get_image_width(texture_atlas), get_image_height(texture_atlas));
    
    const float2 uvs1 = TransformUVs(uvs, tex_atlas_size, texture, floor(lod));
    const float2 uvs2 = TransformUVs(uvs, tex_atlas_size, texture, ceil(lod));

    int page1 = (int)min(floor(lod), (float)MAX_MIP_LEVEL);
    int page2 = (int)min(ceil(lod), (float)MAX_MIP_LEVEL);

    float4 coord1 = (float4)(uvs1, (float)texture->page[page1], 0);
    float4 coord2 = (float4)(uvs2, (float)texture->page[page2], 0);

    float4 tex_col1 = read_imagef(texture_atlas, TEX_SAMPLER, coord1);
    float4 tex_col2 = read_imagef(texture_atlas, TEX_SAMPLER, coord2);

    return mix(tex_col1, tex_col2, lod - floor(lod));
}

float4 SampleTextureAnisotropic(__read_only image2d_array_t texture_atlas, __global const texture_t *texture,
                                const float2 uvs, const float2 duv_dx, const float2 duv_dy) {
    const float2 tex_atlas_size = (float2)(get_image_width(texture_atlas), get_image_height(texture_atlas));

    float2 _duv_dx = fabs(duv_dx * (float2)(texture->size[0], texture->size[1]));
    float2 _duv_dy = fabs(duv_dy * (float2)(texture->size[0], texture->size[1]));

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
    float2 size1 = (float2)((float)(texture->size[0] >> lod1), (float)(texture->size[1] >> lod1));
    float4 coord1 = (float4)(0.0f, 0.0f, (float)page1, 0);

    float2 pos2 = (float2)((float)texture->pos[lod2][0] + 0.5f, (float)texture->pos[lod2][1] + 0.5f);
    float2 size2 = (float2)((float)(texture->size[0] >> lod2), (float)(texture->size[1] >> lod2));
    float4 coord2 = (float4)(0.0f, 0.0f, (float)page2, 0);

    const float kz = lod - floor(lod);

    for (int i = 0; i < num; i++) {
        _uvs = _uvs - floor(_uvs);

        coord1.xy = (pos1 + _uvs * size1) / tex_atlas_size;
        res += (1 - kz) * read_imagef(texture_atlas, TEX_SAMPLER, coord1);

        if (kz > 0.0001f) {
            coord2.xy = (pos2 + _uvs * size2) / tex_atlas_size;
            res += kz * read_imagef(texture_atlas, TEX_SAMPLER, coord2);
        }

        _uvs = _uvs + step;
    }

    return res / num;
}

__kernel
void TextureDebugPage(__read_only image2d_array_t texture_atlas, int page, __write_only image2d_t frame_buf) {
    int i = get_global_id(0),
        j = get_global_id(1);

    float x = 1.0f * ((float)i) / get_global_size(0);
    float y = 1.0f * ((float)j) / get_global_size(1);

    float4 coord = (float4)(x, y, (float)page, 0);
    float4 col = read_imagef(texture_atlas, TEX_SAMPLER, coord);

    write_imagef(frame_buf, (int2)(i, j), col);
}

)"