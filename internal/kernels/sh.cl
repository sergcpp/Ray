R"(

__constant float SH_Y0 = 0.282094806f; // sqrt(1.0f / (4.0f * PI))
__constant float SH_Y1 = 0.488602519f; // sqrt(3.0f / (4.0f * PI))

__constant float SH_A0 = 0.886226952f; // PI / sqrt(4.0f * Pi)
__constant float SH_A1 = 1.02332675f;  // sqrt(PI / 3.0f)

__constant float SH_AY0 = 0.25f; // SH_A0 * SH_Y0
__constant float SH_AY1 = 0.5f;  // SH_A1 * SH_Y1

__constant sampler_t i_fbuf_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST;

float4 SH_EvaluateL1(const float3 v) {
    return (float4)(SH_Y0, SH_Y1 * v.y, SH_Y1 * v.z, SH_Y1 * v.x);
}

float4 SH_ApplyDiffuseConvolutionL1(const float4 coeff) {
    return coeff * (float4)(SH_A0, SH_A1, SH_A1, SH_A1);
}

float4 SH_EvaluateDiffuseL1(const float3 v) {
    return (float4)(SH_AY0, SH_AY1 * v.y, SH_AY1 * v.z, SH_AY1 * v.x);
}

__kernel
void StoreSHCoeffs(const __global ray_packet_t *rays, int w, __global shl1_data_t *out_sh_data) {
    const int i = get_global_id(0);

    const int2 px = (int2)(rays[i].o.w, rays[i].d.w);
    const int index = px.y * w + px.x;

    out_sh_data[index].coeff_r = SH_EvaluateL1(rays[i].d.xyz);
    out_sh_data[index].coeff_g.x = rays[i].c.x;
}

__kernel
void ComputeSHData(__read_only image2d_t clean_buf, int w, __global shl1_data_t *in_out_sh_data) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    __global shl1_data_t *sh_data = &in_out_sh_data[j * w + i];

    float4 sh_coeff = sh_data->coeff_r;
    const float inv_weight = 1.0f / sh_data->coeff_g[0];

    float4 col = read_imagef(clean_buf, i_fbuf_sampler, (int2)(i, j));
    col *= inv_weight;

    sh_data->coeff_r = sh_coeff * col.x;
    sh_data->coeff_g = sh_coeff * col.y;
    sh_data->coeff_b = sh_coeff * col.z;
}

__kernel
void MixSHData(__global shl1_data_t *in_sh_data, float k, __global shl1_data_t *out_sh_data) {
    const int i = get_global_id(0);

    __global shl1_data_t *in_sh = &in_sh_data[i];
    __global shl1_data_t *out_sh = &out_sh_data[i];

    out_sh->coeff_r += (in_sh->coeff_r - out_sh->coeff_r) * k;
    out_sh->coeff_g += (in_sh->coeff_g - out_sh->coeff_g) * k;
    out_sh->coeff_b += (in_sh->coeff_b - out_sh->coeff_b) * k;
}

)"