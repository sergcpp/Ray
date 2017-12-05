R"(

__constant sampler_t TEX_SAMPLER = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

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
    float l1 = fast_length(duv_dx * (float2)(texture->size[0], texture->size[1]));
    float l2 = fast_length(duv_dy * (float2)(texture->size[0], texture->size[1]));

    float lod;
    float k;
    float2 step;

    if (l1 <= l2) {
        lod = native_log2(l1);
        k = l1 / l2;
        step = duv_dx / (float2)(texture->size[0], texture->size[1]);
    } else {
        lod = native_log2(l2);
        k = l2 / l1;
        step = duv_dy / (float2)(texture->size[0], texture->size[1]);
    }

    lod = clamp(lod, 0.0f, (float)MAX_MIP_LEVEL);

    float2 _uvs = uvs - step * 0.5f;

    step = step * k;
    int num = clamp((int)(1.0f / k), 1, 32);

    float4 res = 0;
    
    for (int i = 0; i < num; i++) {
        res += SampleTextureTrilinear(texture_atlas, texture, _uvs, lod);
        _uvs += step;
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