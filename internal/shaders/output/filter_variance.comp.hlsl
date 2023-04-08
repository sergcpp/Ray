struct Params
{
    uint4 rect;
    float2 inv_img_size;
    float _pad0;
    float _pad1;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

static const float _321[5] = { 0.2270270287990570068359375f, 0.1945945918560028076171875f, 0.12162162363529205322265625f, 0.0540540553629398345947265625f, 0.01621621660888195037841796875f };

cbuffer UniformParams
{
    Params _61_g_params : packoffset(c0);
};

Texture2D<float4> g_in_img : register(t1, space0);
SamplerState _g_in_img_sampler : register(s1, space0);
RWTexture2D<float4> g_out_img : register(u0, space0);

static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared uint g_temp_variance0_0[16][16];
groupshared uint g_temp_variance0_1[16][16];
groupshared uint g_temp_variance1_0[16][8];
groupshared uint g_temp_variance1_1[16][8];

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

void comp_main()
{
    do
    {
        int2 _74 = int2(_61_g_params.rect.xy + gl_GlobalInvocationID.xy);
        int2 _79 = int2(gl_LocalInvocationID.xy);
        float2 _91 = (float2(_74) + 0.5f.xx) * _61_g_params.inv_img_size;
        float4 param = g_in_img.SampleLevel(_g_in_img_sampler, _91, 0.0f, int2(-4, -4));
        float4 _104 = reversible_tonemap(param);
        int _112 = _79.y;
        int _115 = _79.x;
        g_temp_variance0_0[_112][_115] = spvPackHalf2x16(_104.xy);
        g_temp_variance0_1[_112][_115] = spvPackHalf2x16(_104.zw);
        float4 param_1 = g_in_img.SampleLevel(_g_in_img_sampler, _91, 0.0f, int2(4, -4));
        float4 _140 = reversible_tonemap(param_1);
        int _147 = 8 + _115;
        g_temp_variance0_0[_112][_147] = spvPackHalf2x16(_140.xy);
        g_temp_variance0_1[_112][_147] = spvPackHalf2x16(_140.zw);
        float4 param_2 = g_in_img.SampleLevel(_g_in_img_sampler, _91, 0.0f, int2(-4, 4));
        float4 _168 = reversible_tonemap(param_2);
        int _171 = 8 + _112;
        g_temp_variance0_0[_171][_115] = spvPackHalf2x16(_168.xy);
        g_temp_variance0_1[_171][_115] = spvPackHalf2x16(_168.zw);
        float4 param_3 = g_in_img.SampleLevel(_g_in_img_sampler, _91, 0.0f, int2(4, 4));
        float4 _195 = reversible_tonemap(param_3);
        g_temp_variance0_0[_171][_147] = spvPackHalf2x16(_195.xy);
        g_temp_variance0_1[_171][_147] = spvPackHalf2x16(_195.zw);
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        bool _225 = gl_GlobalInvocationID.x >= _61_g_params.rect.z;
        bool _234;
        if (!_225)
        {
            _234 = gl_GlobalInvocationID.y >= _61_g_params.rect.w;
        }
        else
        {
            _234 = _225;
        }
        if (_234)
        {
            break;
        }
        int j = 0;
        [unroll]
        for (; j < 16; j += 8)
        {
            int _251 = _112 + j;
            int _254 = _115 + 4;
            float4 _272 = float4(spvUnpackHalf2x16(g_temp_variance0_0[_251][_254]), spvUnpackHalf2x16(g_temp_variance0_1[_251][_254]));
            float4 res = _272 * 0.2270270287990570068359375f;
            [unroll]
            for (int i = 1; i < 5; )
            {
                int _289 = _112 + j;
                int _294 = _254 + i;
                int _338 = _254 - i;
                res = (res + (float4(spvUnpackHalf2x16(g_temp_variance0_0[_289][_294]), spvUnpackHalf2x16(g_temp_variance0_1[_289][_294])) * _321[i])) + (float4(spvUnpackHalf2x16(g_temp_variance0_0[_289][_338]), spvUnpackHalf2x16(g_temp_variance0_1[_289][_338])) * _321[i]);
                i++;
                continue;
            }
            float4 _368 = res;
            float4 _370 = max(_368, _272);
            res = _370;
            int _379 = _112 + j;
            g_temp_variance1_0[_379][_115] = spvPackHalf2x16(_370.xy);
            g_temp_variance1_1[_379][_115] = spvPackHalf2x16(_370.zw);
        }
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        int _402 = _112 + 4;
        float4 _420 = float4(spvUnpackHalf2x16(g_temp_variance1_0[_402][_115]), spvUnpackHalf2x16(g_temp_variance1_1[_402][_115]));
        float4 res_variance = _420 * 0.2270270287990570068359375f;
        [unroll]
        for (int i_1 = 1; i_1 < 5; )
        {
            int _436 = _402 + i_1;
            int _468 = _402 - i_1;
            res_variance = (res_variance + (float4(spvUnpackHalf2x16(g_temp_variance1_0[_436][_115]), spvUnpackHalf2x16(g_temp_variance1_1[_436][_115])) * _321[i_1])) + (float4(spvUnpackHalf2x16(g_temp_variance1_0[_468][_115]), spvUnpackHalf2x16(g_temp_variance1_1[_468][_115])) * _321[i_1]);
            i_1++;
            continue;
        }
        float4 _498 = res_variance;
        float4 _500 = max(_498, _420);
        res_variance = _500;
        g_out_img[_74] = _500;
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
