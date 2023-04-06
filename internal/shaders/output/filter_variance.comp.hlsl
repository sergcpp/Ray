struct Params
{
    uint2 img_size;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

static const float _316[5] = { 0.2270270287990570068359375f, 0.1945945918560028076171875f, 0.12162162363529205322265625f, 0.0540540553629398345947265625f, 0.01621621660888195037841796875f };

cbuffer UniformParams
{
    Params _79_g_params : packoffset(c0);
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
        int2 _62 = int2(gl_GlobalInvocationID.xy);
        int2 _67 = int2(gl_LocalInvocationID.xy);
        float2 _85 = (float2(_62) + 0.5f.xx) / float2(_79_g_params.img_size);
        float4 param = g_in_img.SampleLevel(_g_in_img_sampler, _85, 0.0f, int2(-4, -4));
        float4 _98 = reversible_tonemap(param);
        int _106 = _67.y;
        int _109 = _67.x;
        g_temp_variance0_0[_106][_109] = spvPackHalf2x16(_98.xy);
        g_temp_variance0_1[_106][_109] = spvPackHalf2x16(_98.zw);
        float4 param_1 = g_in_img.SampleLevel(_g_in_img_sampler, _85, 0.0f, int2(4, -4));
        float4 _134 = reversible_tonemap(param_1);
        int _141 = 8 + _109;
        g_temp_variance0_0[_106][_141] = spvPackHalf2x16(_134.xy);
        g_temp_variance0_1[_106][_141] = spvPackHalf2x16(_134.zw);
        float4 param_2 = g_in_img.SampleLevel(_g_in_img_sampler, _85, 0.0f, int2(-4, 4));
        float4 _162 = reversible_tonemap(param_2);
        int _165 = 8 + _106;
        g_temp_variance0_0[_165][_109] = spvPackHalf2x16(_162.xy);
        g_temp_variance0_1[_165][_109] = spvPackHalf2x16(_162.zw);
        float4 param_3 = g_in_img.SampleLevel(_g_in_img_sampler, _85, 0.0f, int2(4, 4));
        float4 _189 = reversible_tonemap(param_3);
        g_temp_variance0_0[_165][_141] = spvPackHalf2x16(_189.xy);
        g_temp_variance0_1[_165][_141] = spvPackHalf2x16(_189.zw);
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        bool _219 = gl_GlobalInvocationID.x >= _79_g_params.img_size.x;
        bool _228;
        if (!_219)
        {
            _228 = gl_GlobalInvocationID.y >= _79_g_params.img_size.y;
        }
        else
        {
            _228 = _219;
        }
        if (_228)
        {
            break;
        }
        int j = 0;
        [unroll]
        for (; j < 16; j += 8)
        {
            int _245 = _106 + j;
            int _248 = _109 + 4;
            float4 _266 = float4(spvUnpackHalf2x16(g_temp_variance0_0[_245][_248]), spvUnpackHalf2x16(g_temp_variance0_1[_245][_248]));
            float4 res = _266 * 0.2270270287990570068359375f;
            [unroll]
            for (int i = 1; i < 5; )
            {
                int _284 = _106 + j;
                int _289 = _248 + i;
                int _333 = _248 - i;
                res = (res + (float4(spvUnpackHalf2x16(g_temp_variance0_0[_284][_289]), spvUnpackHalf2x16(g_temp_variance0_1[_284][_289])) * _316[i])) + (float4(spvUnpackHalf2x16(g_temp_variance0_0[_284][_333]), spvUnpackHalf2x16(g_temp_variance0_1[_284][_333])) * _316[i]);
                i++;
                continue;
            }
            float4 _363 = res;
            float4 _365 = max(_363, _266);
            res = _365;
            int _374 = _106 + j;
            g_temp_variance1_0[_374][_109] = spvPackHalf2x16(_365.xy);
            g_temp_variance1_1[_374][_109] = spvPackHalf2x16(_365.zw);
        }
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        int _397 = _106 + 4;
        float4 _415 = float4(spvUnpackHalf2x16(g_temp_variance1_0[_397][_109]), spvUnpackHalf2x16(g_temp_variance1_1[_397][_109]));
        float4 res_variance = _415 * 0.2270270287990570068359375f;
        [unroll]
        for (int i_1 = 1; i_1 < 5; )
        {
            int _431 = _397 + i_1;
            int _463 = _397 - i_1;
            res_variance = (res_variance + (float4(spvUnpackHalf2x16(g_temp_variance1_0[_431][_109]), spvUnpackHalf2x16(g_temp_variance1_1[_431][_109])) * _316[i_1])) + (float4(spvUnpackHalf2x16(g_temp_variance1_0[_463][_109]), spvUnpackHalf2x16(g_temp_variance1_1[_463][_109])) * _316[i_1]);
            i_1++;
            continue;
        }
        float4 _493 = res_variance;
        float4 _495 = max(_493, _415);
        res_variance = _495;
        g_out_img[_62] = _495;
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
