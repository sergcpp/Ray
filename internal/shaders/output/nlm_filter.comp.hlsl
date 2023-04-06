struct Params
{
    uint2 img_size;
    float alpha;
    float damping;
    int srgb;
    int _clamp;
    float exposure;
    float inv_gamma;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

cbuffer UniformParams
{
    Params _213_g_params : packoffset(c0);
};

Texture2D<float4> g_in_img : register(t2, space0);
SamplerState _g_in_img_sampler : register(s2, space0);
Texture2D<float4> g_variance_img : register(t3, space0);
SamplerState _g_variance_img_sampler : register(s3, space0);
RWTexture2D<float4> g_out_raw_img : register(u1, space0);
RWTexture2D<float4> g_out_img : register(u0, space0);

static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared uint g_temp_color0[16][16];
groupshared uint g_temp_color1[16][16];
groupshared uint g_temp_variance0[16][16];
groupshared uint g_temp_variance1[16][16];

uint spvPackHalf2x16(float2 value)
{
    uint2 Packed = f32tof16(value);
    return Packed.x | (Packed.y << 16);
}

float2 spvUnpackHalf2x16(uint value)
{
    return f16tof32(uint2(value & 0xffff, value >> 16));
}

float3 reversible_tonemap(float3 c)
{
    return c / (max(c.x, max(c.y, c.z)) + 1.0f).xxx;
}

float4 reversible_tonemap(float4 c)
{
    float3 param = c.xyz;
    return float4(reversible_tonemap(param), c.w);
}

float3 reversible_tonemap_invert(float3 c)
{
    return c / (1.0f - max(c.x, max(c.y, c.z))).xxx;
}

float4 reversible_tonemap_invert(float4 c)
{
    float3 param = c.xyz;
    return float4(reversible_tonemap_invert(param), c.w);
}

float3 clamp_and_gamma_correct(bool srgb, float exposure, bool _clamp, float inv_gamma, inout float3 col)
{
    col *= exposure;
    [unroll]
    for (int i = 0; (i < 3) && srgb; i++)
    {
        if (col[i] < 0.003130800090730190277099609375f)
        {
            col[i] = 12.9200000762939453125f * col[i];
        }
        else
        {
            col[i] = mad(1.05499994754791259765625f, pow(col[i], 0.4166666567325592041015625f), -0.054999999701976776123046875f);
        }
    }
    if (inv_gamma != 1.0f)
    {
        col = pow(col, inv_gamma.xxx);
    }
    if (_clamp)
    {
        col = clamp(col, 0.0f.xxx, 1.0f.xxx);
    }
    return col;
}

float4 clamp_and_gamma_correct(bool srgb, float exposure, bool _clamp, float inv_gamma, float4 col)
{
    bool param = srgb;
    float param_1 = exposure;
    bool param_2 = _clamp;
    float param_3 = inv_gamma;
    float3 param_4 = col.xyz;
    float3 _121 = clamp_and_gamma_correct(param, param_1, param_2, param_3, param_4);
    return float4(_121, col.w);
}

void comp_main()
{
    do
    {
        int2 _196 = int2(gl_GlobalInvocationID.xy);
        int2 _201 = int2(gl_LocalInvocationID.xy);
        float2 _218 = (float2(_196) + 0.5f.xx) / float2(_213_g_params.img_size);
        float4 param = g_in_img.SampleLevel(_g_in_img_sampler, _218, 0.0f, int2(-4, -4));
        float4 _230 = reversible_tonemap(param);
        int _237 = _201.y;
        int _240 = _201.x;
        g_temp_color0[_237][_240] = spvPackHalf2x16(_230.xy);
        g_temp_color1[_237][_240] = spvPackHalf2x16(_230.zw);
        float4 param_1 = g_in_img.SampleLevel(_g_in_img_sampler, _218, 0.0f, int2(4, -4));
        float4 _265 = reversible_tonemap(param_1);
        int _272 = 8 + _240;
        g_temp_color0[_237][_272] = spvPackHalf2x16(_265.xy);
        g_temp_color1[_237][_272] = spvPackHalf2x16(_265.zw);
        float4 param_2 = g_in_img.SampleLevel(_g_in_img_sampler, _218, 0.0f, int2(-4, 4));
        float4 _293 = reversible_tonemap(param_2);
        int _296 = 8 + _237;
        g_temp_color0[_296][_240] = spvPackHalf2x16(_293.xy);
        g_temp_color1[_296][_240] = spvPackHalf2x16(_293.zw);
        float4 param_3 = g_in_img.SampleLevel(_g_in_img_sampler, _218, 0.0f, int2(4, 4));
        float4 _320 = reversible_tonemap(param_3);
        g_temp_color0[_296][_272] = spvPackHalf2x16(_320.xy);
        g_temp_color1[_296][_272] = spvPackHalf2x16(_320.zw);
        float4 _345 = g_variance_img.SampleLevel(_g_variance_img_sampler, _218, 0.0f, int2(-4, -4));
        g_temp_variance0[_237][_240] = spvPackHalf2x16(_345.xy);
        g_temp_variance1[_237][_240] = spvPackHalf2x16(_345.zw);
        float4 _371 = g_variance_img.SampleLevel(_g_variance_img_sampler, _218, 0.0f, int2(4, -4));
        g_temp_variance0[_237][_272] = spvPackHalf2x16(_371.xy);
        g_temp_variance1[_237][_272] = spvPackHalf2x16(_371.zw);
        float4 _395 = g_variance_img.SampleLevel(_g_variance_img_sampler, _218, 0.0f, int2(-4, 4));
        g_temp_variance0[_296][_240] = spvPackHalf2x16(_395.xy);
        g_temp_variance1[_296][_240] = spvPackHalf2x16(_395.zw);
        float4 _419 = g_variance_img.SampleLevel(_g_variance_img_sampler, _218, 0.0f, int2(4, 4));
        g_temp_variance0[_296][_272] = spvPackHalf2x16(_419.xy);
        g_temp_variance1[_296][_272] = spvPackHalf2x16(_419.zw);
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        bool _448 = gl_GlobalInvocationID.x >= _213_g_params.img_size.x;
        bool _457;
        if (!_448)
        {
            _457 = gl_GlobalInvocationID.y >= _213_g_params.img_size.y;
        }
        else
        {
            _457 = _448;
        }
        if (_457)
        {
            break;
        }
        float4 sum_output = 0.0f.xxxx;
        float sum_weight = 0.0f;
        int k = -3;
        for (; k <= 3; k++)
        {
            int l = -3;
            for (; l <= 3; l++)
            {
                float4 _distance = 0.0f.xxxx;
                int q = -1;
                [unroll]
                for (; q <= 1; q++)
                {
                    [unroll]
                    for (int p = -1; p <= 1; )
                    {
                        int _502 = _237 + 4;
                        int _504 = _502 + q;
                        int _507 = _240 + 4;
                        int _509 = _507 + p;
                        int _538 = (_502 + k) + q;
                        int _545 = (_507 + l) + p;
                        float4 _602 = float4(spvUnpackHalf2x16(g_temp_variance0[_504][_509]), spvUnpackHalf2x16(g_temp_variance1[_504][_509]));
                        float4 _642 = float4(spvUnpackHalf2x16(g_temp_variance0[_538][_545]), spvUnpackHalf2x16(g_temp_variance1[_538][_545]));
                        float4 _649 = float4(spvUnpackHalf2x16(g_temp_color0[_504][_509]), spvUnpackHalf2x16(g_temp_color1[_504][_509])) - float4(spvUnpackHalf2x16(g_temp_color0[_538][_545]), spvUnpackHalf2x16(g_temp_color1[_538][_545]));
                        _distance += (mad(_649, _649, -((_602 + min(_602, _642)) * _213_g_params.alpha)) / (9.9999997473787516355514526367188e-05f.xxxx + ((_602 + _642) * (_213_g_params.damping * _213_g_params.damping))));
                        p++;
                        continue;
                    }
                }
                float _700 = exp(-max(0.0f, 2.25f * (((_distance.x + _distance.y) + _distance.z) + _distance.w)));
                int _705 = (_237 + 4) + k;
                int _710 = (_240 + 4) + l;
                sum_output += (float4(spvUnpackHalf2x16(g_temp_color0[_705][_710]), spvUnpackHalf2x16(g_temp_color1[_705][_710])) * _700);
                sum_weight += _700;
            }
        }
        [flatten]
        if (sum_weight != 0.0f)
        {
            sum_output /= sum_weight.xxxx;
        }
        float4 param_4 = sum_output;
        float4 _753 = reversible_tonemap_invert(param_4);
        sum_output = _753;
        g_out_raw_img[_196] = _753;
        bool param_5 = _213_g_params.srgb != 0;
        float param_6 = _213_g_params.exposure;
        bool param_7 = _213_g_params._clamp != 0;
        float param_8 = _213_g_params.inv_gamma;
        float4 param_9 = _753;
        g_out_img[_196] = clamp_and_gamma_correct(param_5, param_6, param_7, param_8, param_9);
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
