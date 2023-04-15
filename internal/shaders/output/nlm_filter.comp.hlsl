struct Params
{
    uint4 rect;
    float2 inv_img_size;
    float alpha;
    float damping;
    int srgb;
    float inv_gamma;
    float _pad0;
    float _pad1;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

cbuffer UniformParams
{
    Params _180_g_params : packoffset(c0);
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

float3 clamp_and_gamma_correct(bool srgb, float inv_gamma, inout float3 col)
{
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
    return clamp(col, 0.0f.xxx, 1.0f.xxx);
}

float4 clamp_and_gamma_correct(bool srgb, float inv_gamma, float4 col)
{
    bool param = srgb;
    float param_1 = inv_gamma;
    float3 param_2 = col.xyz;
    float3 _106 = clamp_and_gamma_correct(param, param_1, param_2);
    return float4(_106, col.w);
}

void comp_main()
{
    do
    {
        int2 _192 = int2(_180_g_params.rect.xy + gl_GlobalInvocationID.xy);
        int2 _197 = int2(gl_LocalInvocationID.xy);
        float2 _208 = (float2(_192) + 0.5f.xx) * _180_g_params.inv_img_size;
        float4 param = g_in_img.SampleLevel(_g_in_img_sampler, _208, 0.0f, int2(-4, -4));
        float4 _220 = reversible_tonemap(param);
        int _227 = _197.y;
        int _230 = _197.x;
        g_temp_color0[_227][_230] = spvPackHalf2x16(_220.xy);
        g_temp_color1[_227][_230] = spvPackHalf2x16(_220.zw);
        float4 param_1 = g_in_img.SampleLevel(_g_in_img_sampler, _208, 0.0f, int2(4, -4));
        float4 _255 = reversible_tonemap(param_1);
        int _262 = 8 + _230;
        g_temp_color0[_227][_262] = spvPackHalf2x16(_255.xy);
        g_temp_color1[_227][_262] = spvPackHalf2x16(_255.zw);
        float4 param_2 = g_in_img.SampleLevel(_g_in_img_sampler, _208, 0.0f, int2(-4, 4));
        float4 _283 = reversible_tonemap(param_2);
        int _286 = 8 + _227;
        g_temp_color0[_286][_230] = spvPackHalf2x16(_283.xy);
        g_temp_color1[_286][_230] = spvPackHalf2x16(_283.zw);
        float4 param_3 = g_in_img.SampleLevel(_g_in_img_sampler, _208, 0.0f, int2(4, 4));
        float4 _310 = reversible_tonemap(param_3);
        g_temp_color0[_286][_262] = spvPackHalf2x16(_310.xy);
        g_temp_color1[_286][_262] = spvPackHalf2x16(_310.zw);
        float4 _335 = g_variance_img.SampleLevel(_g_variance_img_sampler, _208, 0.0f, int2(-4, -4));
        g_temp_variance0[_227][_230] = spvPackHalf2x16(_335.xy);
        g_temp_variance1[_227][_230] = spvPackHalf2x16(_335.zw);
        float4 _361 = g_variance_img.SampleLevel(_g_variance_img_sampler, _208, 0.0f, int2(4, -4));
        g_temp_variance0[_227][_262] = spvPackHalf2x16(_361.xy);
        g_temp_variance1[_227][_262] = spvPackHalf2x16(_361.zw);
        float4 _385 = g_variance_img.SampleLevel(_g_variance_img_sampler, _208, 0.0f, int2(-4, 4));
        g_temp_variance0[_286][_230] = spvPackHalf2x16(_385.xy);
        g_temp_variance1[_286][_230] = spvPackHalf2x16(_385.zw);
        float4 _409 = g_variance_img.SampleLevel(_g_variance_img_sampler, _208, 0.0f, int2(4, 4));
        g_temp_variance0[_286][_262] = spvPackHalf2x16(_409.xy);
        g_temp_variance1[_286][_262] = spvPackHalf2x16(_409.zw);
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        bool _438 = gl_GlobalInvocationID.x >= _180_g_params.rect.z;
        bool _447;
        if (!_438)
        {
            _447 = gl_GlobalInvocationID.y >= _180_g_params.rect.w;
        }
        else
        {
            _447 = _438;
        }
        if (_447)
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
                        int _492 = _227 + 4;
                        int _494 = _492 + q;
                        int _497 = _230 + 4;
                        int _499 = _497 + p;
                        int _528 = (_492 + k) + q;
                        int _535 = (_497 + l) + p;
                        float4 _592 = float4(spvUnpackHalf2x16(g_temp_variance0[_494][_499]), spvUnpackHalf2x16(g_temp_variance1[_494][_499]));
                        float4 _632 = float4(spvUnpackHalf2x16(g_temp_variance0[_528][_535]), spvUnpackHalf2x16(g_temp_variance1[_528][_535]));
                        float4 _639 = float4(spvUnpackHalf2x16(g_temp_color0[_494][_499]), spvUnpackHalf2x16(g_temp_color1[_494][_499])) - float4(spvUnpackHalf2x16(g_temp_color0[_528][_535]), spvUnpackHalf2x16(g_temp_color1[_528][_535]));
                        _distance += (mad(_639, _639, -((_592 + min(_592, _632)) * _180_g_params.alpha)) / (9.9999997473787516355514526367188e-05f.xxxx + ((_592 + _632) * (_180_g_params.damping * _180_g_params.damping))));
                        p++;
                        continue;
                    }
                }
                float _690 = exp(-max(0.0f, 2.25f * (((_distance.x + _distance.y) + _distance.z) + _distance.w)));
                int _695 = (_227 + 4) + k;
                int _700 = (_230 + 4) + l;
                sum_output += (float4(spvUnpackHalf2x16(g_temp_color0[_695][_700]), spvUnpackHalf2x16(g_temp_color1[_695][_700])) * _690);
                sum_weight += _690;
            }
        }
        [flatten]
        if (sum_weight != 0.0f)
        {
            sum_output /= sum_weight.xxxx;
        }
        float4 param_4 = sum_output;
        float4 _743 = reversible_tonemap_invert(param_4);
        sum_output = _743;
        g_out_raw_img[_192] = _743;
        bool param_5 = _180_g_params.srgb != 0;
        float param_6 = _180_g_params.inv_gamma;
        float4 param_7 = _743;
        g_out_img[_192] = clamp_and_gamma_correct(param_5, param_6, param_7);
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
