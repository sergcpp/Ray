R"(

__constant sampler_t isampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST;

__kernel
void PostProcess(__read_only image2d_t frame_buf, int w, int h, float inv_gamma, int _clamp, int _srgb,
                 __write_only image2d_t pixels) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    float4 col = read_imagef(frame_buf, isampler, (int2)(i, j));

    if (_srgb) {
        col = rgb_to_srgb(col);
    }

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