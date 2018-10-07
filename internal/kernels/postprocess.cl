R"(

__constant float FXAA_SPAN_MAX = 2.0f;
__constant float FXAA_REDUCE_MUL = 1.0f / 8.0f;
__constant float FXAA_REDUCE_MIN = 1.0f / 128.0f;

__constant float3 luma = (float3)(0.299f, 0.587f, 0.114f);

__constant sampler_t fsampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
__constant sampler_t isampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST;

float3 Fxaa(__read_only image2d_t frame_buf, int x, int y, int2 size) {
    float2 rcp_size = (float2)(1.0f / size.x, 1.0f / size.y);
    float2 texCoords = (float2)(x + 0.5f, y + 0.5f) * rcp_size;

    float3 rgbNW = read_imagef(frame_buf, fsampler, texCoords + (float2)(-1, -1) * rcp_size).xyz;
    float3 rgbNE = read_imagef(frame_buf, fsampler, texCoords + (float2)(1, -1) * rcp_size).xyz;
    float3 rgbSW = read_imagef(frame_buf, fsampler, texCoords + (float2)(-1, 1) * rcp_size).xyz;
    float3 rgbSE = read_imagef(frame_buf, fsampler, texCoords + (float2)(1, 1) * rcp_size).xyz;
    float3 rgbM = read_imagef(frame_buf, fsampler, texCoords).xyz;

    //return rgbM;

    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM  = dot(rgbM,  luma);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
        
    float2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25f * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);
    float rcpDirMin = 1.0f/(min(fabs(dir.x), fabs(dir.y)) + dirReduce);

    dir = min((float2)(FXAA_SPAN_MAX, FXAA_SPAN_MAX), max((float2)(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX), dir * rcpDirMin)) * rcp_size;

    float3 rgbA = (1.0f/2.0f) * (read_imagef(frame_buf, fsampler, texCoords + dir * (1.0f/3.0f - 0.5f)).xyz + read_imagef(frame_buf, fsampler, texCoords + dir * (2.0f/3.0f - 0.5f)).xyz);
    float3 rgbB = rgbA * (1.0f/2.0f) + (1.0f/4.0f) * (read_imagef(frame_buf, fsampler, texCoords + dir * (0.0f/3.0f - 0.5f)).xyz + read_imagef(frame_buf, fsampler, texCoords + dir * (3.0f/3.0f - 0.5f)).xyz);
    float lumaB = dot(rgbB, luma);

    if ((lumaB < lumaMin) || (lumaB > lumaMax)) {
        return rgbA;
    } else {
        return rgbB;
    }
}

__kernel
void PostProcess(__read_only image2d_t frame_buf, int w, int h, float inv_gamma, int _clamp,
                 __write_only image2d_t pixels) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    float4 col = read_imagef(frame_buf, isampler, (int2)(i, j));

    col = native_powr(col, inv_gamma);
    if (_clamp) {
        col = clamp(col, 0.0f, 1.0f);
    }

    write_imagef(pixels, (int2)(i, j), col);
}

__kernel void MixIncremental(__read_only image2d_t fbuf1, __read_only image2d_t fbuf2, float k, __write_only image2d_t res) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    float4 col1 = read_imagef(fbuf1, isampler, (int2)(i, j));
    float4 col2 = read_imagef(fbuf2, isampler, (int2)(i, j));

    float4 diff = col2 - col1;
    write_imagef(res, (int2)(i, j), col1 + diff * k);

    //write_imagef(res, (int2)(i, j), col2);
}

)"