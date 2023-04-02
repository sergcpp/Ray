struct Params
{
    uint2 img_size;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

static const float _269[5] = { 0.2270270287990570068359375f, 0.1945945918560028076171875f, 0.12162162363529205322265625f, 0.0540540553629398345947265625f, 0.01621621660888195037841796875f };

cbuffer UniformParams
{
    Params _35_g_params : packoffset(c0);
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

void comp_main()
{
    do
    {
        int2 _17 = int2(gl_GlobalInvocationID.xy);
        int2 _22 = int2(gl_LocalInvocationID.xy);
        float2 _41 = (float2(_17) + 0.5f.xx) / float2(_35_g_params.img_size);
        float4 _54 = g_in_img.SampleLevel(_g_in_img_sampler, _41, 0.0f, int2(-4, -4));
        int _63 = _22.y;
        int _67 = _22.x;
        g_temp_variance0_0[_63][_67] = spvPackHalf2x16(_54.xy);
        g_temp_variance0_1[_63][_67] = spvPackHalf2x16(_54.zw);
        float4 _90 = g_in_img.SampleLevel(_g_in_img_sampler, _41, 0.0f, int2(4, -4));
        int _97 = 8 + _67;
        g_temp_variance0_0[_63][_97] = spvPackHalf2x16(_90.xy);
        g_temp_variance0_1[_63][_97] = spvPackHalf2x16(_90.zw);
        float4 _116 = g_in_img.SampleLevel(_g_in_img_sampler, _41, 0.0f, int2(-4, 4));
        int _119 = 8 + _63;
        g_temp_variance0_0[_119][_67] = spvPackHalf2x16(_116.xy);
        g_temp_variance0_1[_119][_67] = spvPackHalf2x16(_116.zw);
        float4 _141 = g_in_img.SampleLevel(_g_in_img_sampler, _41, 0.0f, int2(4, 4));
        g_temp_variance0_0[_119][_97] = spvPackHalf2x16(_141.xy);
        g_temp_variance0_1[_119][_97] = spvPackHalf2x16(_141.zw);
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        bool _172 = gl_GlobalInvocationID.x >= _35_g_params.img_size.x;
        bool _181;
        if (!_172)
        {
            _181 = gl_GlobalInvocationID.y >= _35_g_params.img_size.y;
        }
        else
        {
            _181 = _172;
        }
        if (_181)
        {
            break;
        }
        int j = 0;
        [unroll]
        for (; j < 16; j += 8)
        {
            int _198 = _63 + j;
            int _201 = _67 + 4;
            float4 _219 = float4(spvUnpackHalf2x16(g_temp_variance0_0[_198][_201]), spvUnpackHalf2x16(g_temp_variance0_1[_198][_201]));
            float4 res = _219 * 0.2270270287990570068359375f;
            [unroll]
            for (int i = 1; i < 5; )
            {
                int _237 = _63 + j;
                int _242 = _201 + i;
                int _287 = _201 - i;
                res = (res + (float4(spvUnpackHalf2x16(g_temp_variance0_0[_237][_242]), spvUnpackHalf2x16(g_temp_variance0_1[_237][_242])) * _269[i])) + (float4(spvUnpackHalf2x16(g_temp_variance0_0[_237][_287]), spvUnpackHalf2x16(g_temp_variance0_1[_237][_287])) * _269[i]);
                i++;
                continue;
            }
            float4 _317 = res;
            float4 _319 = max(_317, _219);
            res = _319;
            int _328 = _63 + j;
            g_temp_variance1_0[_328][_67] = spvPackHalf2x16(_319.xy);
            g_temp_variance1_1[_328][_67] = spvPackHalf2x16(_319.zw);
        }
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        int _351 = _63 + 4;
        float4 _369 = float4(spvUnpackHalf2x16(g_temp_variance1_0[_351][_67]), spvUnpackHalf2x16(g_temp_variance1_1[_351][_67]));
        float4 res_variance = _369 * 0.2270270287990570068359375f;
        [unroll]
        for (int i_1 = 1; i_1 < 5; )
        {
            int _385 = _351 + i_1;
            int _417 = _351 - i_1;
            res_variance = (res_variance + (float4(spvUnpackHalf2x16(g_temp_variance1_0[_385][_67]), spvUnpackHalf2x16(g_temp_variance1_1[_385][_67])) * _269[i_1])) + (float4(spvUnpackHalf2x16(g_temp_variance1_0[_417][_67]), spvUnpackHalf2x16(g_temp_variance1_1[_417][_67])) * _269[i_1]);
            i_1++;
            continue;
        }
        float4 _447 = res_variance;
        float4 _449 = max(_447, _369);
        res_variance = _449;
        g_out_img[_17] = _449;
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
