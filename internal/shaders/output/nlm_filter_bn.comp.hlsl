struct Params
{
    uint4 rect;
    float2 inv_img_size;
    float alpha;
    float damping;
    float inv_gamma;
    int tonemap_mode;
    float base_color_weight;
    float depth_normal_weight;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

cbuffer UniformParams
{
    Params _364_g_params : packoffset(c0);
};

Texture2D<float4> g_in_img : register(t2, space0);
SamplerState _g_in_img_sampler : register(s2, space0);
Texture2D<float4> g_variance_img : register(t3, space0);
SamplerState _g_variance_img_sampler : register(s3, space0);
Texture2D<float4> g_base_color_img : register(t5, space0);
SamplerState _g_base_color_img_sampler : register(s5, space0);
Texture2D<float4> g_depth_normal_img : register(t6, space0);
SamplerState _g_depth_normal_img_sampler : register(s6, space0);
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
groupshared uint g_temp_base_color[16][16];
groupshared uint g_temp_depth_normal[16][16];

uint spvPackHalf2x16(float2 value)
{
    uint2 Packed = f32tof16(value);
    return Packed.x | (Packed.y << 16);
}

float2 spvUnpackHalf2x16(uint value)
{
    return f16tof32(uint2(value & 0xffff, value >> 16));
}

uint spvPackUnorm4x8(float4 value)
{
    uint4 Packed = uint4(round(saturate(value) * 255.0));
    return Packed.x | (Packed.y << 8) | (Packed.z << 16) | (Packed.w << 24);
}

float4 spvUnpackUnorm4x8(uint value)
{
    uint4 Packed = uint4(value & 0xff, (value >> 8) & 0xff, (value >> 16) & 0xff, value >> 24);
    return float4(Packed) / 255.0;
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

float3 TonemapLUT_manual(Texture3D<float4> lut, SamplerState _lut_sampler, float inv_gamma, float3 col)
{
    float3 _137 = ((col / (col + 1.0f.xxx)) * 47.0f) + 0.5f.xxx;
    int3 _142 = int3(_137);
    float3 _145 = frac(_137);
    float _149 = _145.x;
    float _153 = _145.y;
    float _157 = _145.z;
    float _216 = 1.0f - _149;
    float _252 = 1.0f - _153;
    float3 ret = (((((lut.Load(int4(_142, 0), int3(0, 0, 0)).xyz * _216) + (lut.Load(int4(_142, 0), int3(1, 0, 0)).xyz * _149)) * _252) + (((lut.Load(int4(_142, 0), int3(0, 1, 0)).xyz * _216) + (lut.Load(int4(_142, 0), int3(1, 1, 0)).xyz * _149)) * _153)) * (1.0f - _157)) + (((((lut.Load(int4(_142, 0), int3(0, 0, 1)).xyz * _216) + (lut.Load(int4(_142, 0), int3(1, 0, 1)).xyz * _149)) * _252) + (((lut.Load(int4(_142, 0), int3(0, 1, 1)).xyz * _216) + (lut.Load(int4(_142, 0), int3(1, 1, 1)).xyz * _149)) * _153)) * _157);
    if (inv_gamma != 1.0f)
    {
        ret = pow(ret, inv_gamma.xxx);
    }
    return ret;
}

float4 TonemapLUT_manual(Texture3D<float4> lut, SamplerState _lut_sampler, float inv_gamma, float4 col)
{
    float param = inv_gamma;
    float3 param_1 = col.xyz;
    return float4(TonemapLUT_manual(lut, _lut_sampler, param, param_1), col.w);
}

void comp_main()
{
    do
    {
        int2 _376 = int2(_364_g_params.rect.xy + gl_GlobalInvocationID.xy);
        int2 _381 = int2(gl_LocalInvocationID.xy);
        float2 _391 = (float2(_376) + 0.5f.xx) * _364_g_params.inv_img_size;
        float4 param = g_in_img.SampleLevel(_g_in_img_sampler, _391, 0.0f, int2(-4, -4));
        float4 _403 = reversible_tonemap(param);
        int _410 = _381.y;
        int _413 = _381.x;
        g_temp_color0[_410][_413] = spvPackHalf2x16(_403.xy);
        g_temp_color1[_410][_413] = spvPackHalf2x16(_403.zw);
        float4 param_1 = g_in_img.SampleLevel(_g_in_img_sampler, _391, 0.0f, int2(4, -4));
        float4 _438 = reversible_tonemap(param_1);
        int _445 = 8 + _413;
        g_temp_color0[_410][_445] = spvPackHalf2x16(_438.xy);
        g_temp_color1[_410][_445] = spvPackHalf2x16(_438.zw);
        float4 param_2 = g_in_img.SampleLevel(_g_in_img_sampler, _391, 0.0f, int2(-4, 4));
        float4 _466 = reversible_tonemap(param_2);
        int _469 = 8 + _410;
        g_temp_color0[_469][_413] = spvPackHalf2x16(_466.xy);
        g_temp_color1[_469][_413] = spvPackHalf2x16(_466.zw);
        float4 param_3 = g_in_img.SampleLevel(_g_in_img_sampler, _391, 0.0f, int2(4, 4));
        float4 _493 = reversible_tonemap(param_3);
        g_temp_color0[_469][_445] = spvPackHalf2x16(_493.xy);
        g_temp_color1[_469][_445] = spvPackHalf2x16(_493.zw);
        float4 _518 = g_variance_img.SampleLevel(_g_variance_img_sampler, _391, 0.0f, int2(-4, -4));
        g_temp_variance0[_410][_413] = spvPackHalf2x16(_518.xy);
        g_temp_variance1[_410][_413] = spvPackHalf2x16(_518.zw);
        float4 _544 = g_variance_img.SampleLevel(_g_variance_img_sampler, _391, 0.0f, int2(4, -4));
        g_temp_variance0[_410][_445] = spvPackHalf2x16(_544.xy);
        g_temp_variance1[_410][_445] = spvPackHalf2x16(_544.zw);
        float4 _568 = g_variance_img.SampleLevel(_g_variance_img_sampler, _391, 0.0f, int2(-4, 4));
        g_temp_variance0[_469][_413] = spvPackHalf2x16(_568.xy);
        g_temp_variance1[_469][_413] = spvPackHalf2x16(_568.zw);
        float4 _592 = g_variance_img.SampleLevel(_g_variance_img_sampler, _391, 0.0f, int2(4, 4));
        g_temp_variance0[_469][_445] = spvPackHalf2x16(_592.xy);
        g_temp_variance1[_469][_445] = spvPackHalf2x16(_592.zw);
        g_temp_base_color[_410][_413] = spvPackUnorm4x8(g_base_color_img.SampleLevel(_g_base_color_img_sampler, _391, 0.0f, int2(-4, -4)));
        g_temp_base_color[_410][_445] = spvPackUnorm4x8(g_base_color_img.SampleLevel(_g_base_color_img_sampler, _391, 0.0f, int2(4, -4)));
        g_temp_base_color[_469][_413] = spvPackUnorm4x8(g_base_color_img.SampleLevel(_g_base_color_img_sampler, _391, 0.0f, int2(-4, 4)));
        g_temp_base_color[_469][_445] = spvPackUnorm4x8(g_base_color_img.SampleLevel(_g_base_color_img_sampler, _391, 0.0f, int2(4, 4)));
        g_temp_depth_normal[_410][_413] = spvPackUnorm4x8(mad(g_depth_normal_img.SampleLevel(_g_depth_normal_img_sampler, _391, 0.0f, int2(-4, -4)), float4(0.5f, 0.5f, 0.5f, 0.0625f), float4(0.5f, 0.5f, 0.5f, 0.0f)));
        g_temp_depth_normal[_410][_445] = spvPackUnorm4x8(mad(g_depth_normal_img.SampleLevel(_g_depth_normal_img_sampler, _391, 0.0f, int2(4, -4)), float4(0.5f, 0.5f, 0.5f, 0.0625f), float4(0.5f, 0.5f, 0.5f, 0.0f)));
        g_temp_depth_normal[_469][_413] = spvPackUnorm4x8(mad(g_depth_normal_img.SampleLevel(_g_depth_normal_img_sampler, _391, 0.0f, int2(-4, 4)), float4(0.5f, 0.5f, 0.5f, 0.0625f), float4(0.5f, 0.5f, 0.5f, 0.0f)));
        g_temp_depth_normal[_469][_445] = spvPackUnorm4x8(mad(g_depth_normal_img.SampleLevel(_g_depth_normal_img_sampler, _391, 0.0f, int2(4, 4)), float4(0.5f, 0.5f, 0.5f, 0.0625f), float4(0.5f, 0.5f, 0.5f, 0.0f)));
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        bool _740 = gl_GlobalInvocationID.x >= _364_g_params.rect.z;
        bool _749;
        if (!_740)
        {
            _749 = gl_GlobalInvocationID.y >= _364_g_params.rect.w;
        }
        else
        {
            _749 = _740;
        }
        if (_749)
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
                float4 color_distance = 0.0f.xxxx;
                int q = -1;
                [unroll]
                for (; q <= 1; q++)
                {
                    [unroll]
                    for (int p = -1; p <= 1; )
                    {
                        int _794 = _410 + 4;
                        int _796 = _794 + q;
                        int _799 = _413 + 4;
                        int _801 = _799 + p;
                        int _830 = (_794 + k) + q;
                        int _837 = (_799 + l) + p;
                        float4 _894 = float4(spvUnpackHalf2x16(g_temp_variance0[_796][_801]), spvUnpackHalf2x16(g_temp_variance1[_796][_801]));
                        float4 _934 = float4(spvUnpackHalf2x16(g_temp_variance0[_830][_837]), spvUnpackHalf2x16(g_temp_variance1[_830][_837]));
                        float4 _941 = float4(spvUnpackHalf2x16(g_temp_color0[_796][_801]), spvUnpackHalf2x16(g_temp_color1[_796][_801])) - float4(spvUnpackHalf2x16(g_temp_color0[_830][_837]), spvUnpackHalf2x16(g_temp_color1[_830][_837]));
                        color_distance += (mad(_941, _941, -((_894 + min(_894, _934)) * _364_g_params.alpha)) / (9.9999997473787516355514526367188e-05f.xxxx + ((_894 + _934) * (_364_g_params.damping * _364_g_params.damping))));
                        p++;
                        continue;
                    }
                }
                int _997 = _410 + 4;
                int _1000 = _413 + 4;
                int _1009 = _997 + k;
                int _1014 = _1000 + l;
                float4 _1023 = spvUnpackUnorm4x8(g_temp_base_color[_997][_1000]) - spvUnpackUnorm4x8(g_temp_base_color[_1009][_1014]);
                float4 _1067 = mad(spvUnpackUnorm4x8(g_temp_depth_normal[_997][_1000]), float4(2.0f, 2.0f, 2.0f, 16.0f), float4(-1.0f, -1.0f, -1.0f, -0.0f)) - mad(spvUnpackUnorm4x8(g_temp_depth_normal[_1009][_1014]), float4(2.0f, 2.0f, 2.0f, 16.0f), float4(-1.0f, -1.0f, -1.0f, -0.0f));
                float4 _1073 = max((_1023 * _364_g_params.base_color_weight) * _1023, (_1067 * _364_g_params.depth_normal_weight) * _1067);
                float _1095 = min(exp(-max(0.0f, 2.25f * (((color_distance.x + color_distance.y) + color_distance.z) + color_distance.w))), exp(-max(0.0f, 0.25f * (((_1073.x + _1073.y) + _1073.z) + _1073.w))));
                sum_output += (float4(spvUnpackHalf2x16(g_temp_color0[_1009][_1014]), spvUnpackHalf2x16(g_temp_color1[_1009][_1014])) * _1095);
                sum_weight += _1095;
            }
        }
        [flatten]
        if (sum_weight != 0.0f)
        {
            sum_output /= sum_weight.xxxx;
        }
        float4 param_4 = sum_output;
        float4 _1148 = reversible_tonemap_invert(param_4);
        sum_output = _1148;
        g_out_raw_img[_376] = _1148;
        [branch]
        if (_364_g_params.tonemap_mode == 0)
        {
            float param_5 = _364_g_params.inv_gamma;
            float4 param_6 = sum_output;
            sum_output = TonemapStandard(param_5, param_6);
        }
        else
        {
            float param_7 = _364_g_params.inv_gamma;
            float4 param_8 = sum_output;
            sum_output = TonemapLUT_manual(g_tonemap_lut, _g_tonemap_lut_sampler, param_7, param_8);
        }
        g_out_img[_376] = sum_output;
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
