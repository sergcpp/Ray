struct Params
{
    uint4 rect;
    float2 inv_img_size;
    float alpha;
    float damping;
    float inv_gamma;
    int tonemap_mode;
    float _pad1;
    float _pad2;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

cbuffer UniformParams
{
    Params _231_g_params : packoffset(c0);
};

Texture2D<float4> g_in_img : register(t2, space0);
SamplerState _g_in_img_sampler : register(s2, space0);
Texture2D<float4> g_variance_img : register(t3, space0);
SamplerState _g_variance_img_sampler : register(s3, space0);
RWTexture2D<float4> g_out_raw_img : register(u1, space0);
Texture3D<float4> g_tonemap_lut : register(t4, space0);
SamplerState _g_tonemap_lut_sampler : register(s4, space0);
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

float3 TonemapStandard(float inv_gamma, inout float3 col)
{
    [unroll]
    for (int i = 0; i < 3; i++)
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

float4 TonemapStandard(float inv_gamma, float4 col)
{
    float param = inv_gamma;
    float3 param_1 = col.xyz;
    float3 _114 = TonemapStandard(param, param_1);
    return float4(_114, col.w);
}

float3 TonemapLUT(Texture3D<float4> lut, SamplerState _lut_sampler, float inv_gamma, float3 col)
{
    float3 ret = lut.SampleLevel(_lut_sampler, ((col / (col + 1.0f.xxx)) * 0.979166686534881591796875f) + 0.010416666977107524871826171875f.xxx, 0.0f).xyz;
    if (inv_gamma != 1.0f)
    {
        ret = pow(ret, inv_gamma.xxx);
    }
    return ret;
}

float4 TonemapLUT(Texture3D<float4> lut, SamplerState _lut_sampler, float inv_gamma, float4 col)
{
    float param = inv_gamma;
    float3 param_1 = col.xyz;
    return float4(TonemapLUT(lut, _lut_sampler, param, param_1), col.w);
}

void comp_main()
{
    do
    {
        int2 _243 = int2(_231_g_params.rect.xy + gl_GlobalInvocationID.xy);
        int2 _248 = int2(gl_LocalInvocationID.xy);
        float2 _259 = (float2(_243) + 0.5f.xx) * _231_g_params.inv_img_size;
        float4 param = g_in_img.SampleLevel(_g_in_img_sampler, _259, 0.0f, int2(-4, -4));
        float4 _271 = reversible_tonemap(param);
        int _278 = _248.y;
        int _281 = _248.x;
        g_temp_color0[_278][_281] = spvPackHalf2x16(_271.xy);
        g_temp_color1[_278][_281] = spvPackHalf2x16(_271.zw);
        float4 param_1 = g_in_img.SampleLevel(_g_in_img_sampler, _259, 0.0f, int2(4, -4));
        float4 _306 = reversible_tonemap(param_1);
        int _313 = 8 + _281;
        g_temp_color0[_278][_313] = spvPackHalf2x16(_306.xy);
        g_temp_color1[_278][_313] = spvPackHalf2x16(_306.zw);
        float4 param_2 = g_in_img.SampleLevel(_g_in_img_sampler, _259, 0.0f, int2(-4, 4));
        float4 _334 = reversible_tonemap(param_2);
        int _337 = 8 + _278;
        g_temp_color0[_337][_281] = spvPackHalf2x16(_334.xy);
        g_temp_color1[_337][_281] = spvPackHalf2x16(_334.zw);
        float4 param_3 = g_in_img.SampleLevel(_g_in_img_sampler, _259, 0.0f, int2(4, 4));
        float4 _361 = reversible_tonemap(param_3);
        g_temp_color0[_337][_313] = spvPackHalf2x16(_361.xy);
        g_temp_color1[_337][_313] = spvPackHalf2x16(_361.zw);
        float4 _386 = g_variance_img.SampleLevel(_g_variance_img_sampler, _259, 0.0f, int2(-4, -4));
        g_temp_variance0[_278][_281] = spvPackHalf2x16(_386.xy);
        g_temp_variance1[_278][_281] = spvPackHalf2x16(_386.zw);
        float4 _412 = g_variance_img.SampleLevel(_g_variance_img_sampler, _259, 0.0f, int2(4, -4));
        g_temp_variance0[_278][_313] = spvPackHalf2x16(_412.xy);
        g_temp_variance1[_278][_313] = spvPackHalf2x16(_412.zw);
        float4 _436 = g_variance_img.SampleLevel(_g_variance_img_sampler, _259, 0.0f, int2(-4, 4));
        g_temp_variance0[_337][_281] = spvPackHalf2x16(_436.xy);
        g_temp_variance1[_337][_281] = spvPackHalf2x16(_436.zw);
        float4 _460 = g_variance_img.SampleLevel(_g_variance_img_sampler, _259, 0.0f, int2(4, 4));
        g_temp_variance0[_337][_313] = spvPackHalf2x16(_460.xy);
        g_temp_variance1[_337][_313] = spvPackHalf2x16(_460.zw);
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        bool _489 = gl_GlobalInvocationID.x >= _231_g_params.rect.z;
        bool _498;
        if (!_489)
        {
            _498 = gl_GlobalInvocationID.y >= _231_g_params.rect.w;
        }
        else
        {
            _498 = _489;
        }
        if (_498)
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
                        int _543 = _278 + 4;
                        int _545 = _543 + q;
                        int _548 = _281 + 4;
                        int _550 = _548 + p;
                        int _579 = (_543 + k) + q;
                        int _586 = (_548 + l) + p;
                        float4 _643 = float4(spvUnpackHalf2x16(g_temp_variance0[_545][_550]), spvUnpackHalf2x16(g_temp_variance1[_545][_550]));
                        float4 _683 = float4(spvUnpackHalf2x16(g_temp_variance0[_579][_586]), spvUnpackHalf2x16(g_temp_variance1[_579][_586]));
                        float4 _690 = float4(spvUnpackHalf2x16(g_temp_color0[_545][_550]), spvUnpackHalf2x16(g_temp_color1[_545][_550])) - float4(spvUnpackHalf2x16(g_temp_color0[_579][_586]), spvUnpackHalf2x16(g_temp_color1[_579][_586]));
                        _distance += (mad(_690, _690, -((_643 + min(_643, _683)) * _231_g_params.alpha)) / (9.9999997473787516355514526367188e-05f.xxxx + ((_643 + _683) * (_231_g_params.damping * _231_g_params.damping))));
                        p++;
                        continue;
                    }
                }
                float _741 = exp(-max(0.0f, 2.25f * (((_distance.x + _distance.y) + _distance.z) + _distance.w)));
                int _746 = (_278 + 4) + k;
                int _751 = (_281 + 4) + l;
                sum_output += (float4(spvUnpackHalf2x16(g_temp_color0[_746][_751]), spvUnpackHalf2x16(g_temp_color1[_746][_751])) * _741);
                sum_weight += _741;
            }
        }
        [flatten]
        if (sum_weight != 0.0f)
        {
            sum_output /= sum_weight.xxxx;
        }
        float4 param_4 = sum_output;
        float4 _794 = reversible_tonemap_invert(param_4);
        sum_output = _794;
        g_out_raw_img[_243] = _794;
        [branch]
        if (_231_g_params.tonemap_mode == 0)
        {
            float param_5 = _231_g_params.inv_gamma;
            float4 param_6 = sum_output;
            sum_output = TonemapStandard(param_5, param_6);
        }
        else
        {
            float param_7 = _231_g_params.inv_gamma;
            float4 param_8 = sum_output;
            sum_output = TonemapLUT(g_tonemap_lut, _g_tonemap_lut_sampler, param_7, param_8);
        }
        g_out_img[_243] = sum_output;
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
