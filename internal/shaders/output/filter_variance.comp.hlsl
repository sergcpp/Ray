struct Params
{
    uint2 img_size;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

static const float _178[5] = { 0.2270270287990570068359375f, 0.1945945918560028076171875f, 0.12162162363529205322265625f, 0.0540540553629398345947265625f, 0.01621621660888195037841796875f };

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

groupshared float4 g_temp_variance0[16][16];
groupshared float4 g_temp_variance1[16][8];

void comp_main()
{
    do
    {
        int2 _17 = int2(gl_GlobalInvocationID.xy);
        int2 _22 = int2(gl_LocalInvocationID.xy);
        float2 _41 = (float2(_17) + 0.5f.xx) / float2(_35_g_params.img_size);
        int _51 = _22.y;
        int _55 = _22.x;
        g_temp_variance0[_51][_55] = g_in_img.SampleLevel(_g_in_img_sampler, _41, 0.0f, int2(-4, -4));
        int _75 = 8 + _55;
        g_temp_variance0[_51][_75] = g_in_img.SampleLevel(_g_in_img_sampler, _41, 0.0f, int2(4, -4));
        int _84 = 8 + _51;
        g_temp_variance0[_84][_55] = g_in_img.SampleLevel(_g_in_img_sampler, _41, 0.0f, int2(-4, 4));
        g_temp_variance0[_84][_75] = g_in_img.SampleLevel(_g_in_img_sampler, _41, 0.0f, int2(4, 4));
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        bool _114 = gl_GlobalInvocationID.x >= _35_g_params.img_size.x;
        bool _123;
        if (!_114)
        {
            _123 = gl_GlobalInvocationID.y >= _35_g_params.img_size.y;
        }
        else
        {
            _123 = _114;
        }
        if (_123)
        {
            break;
        }
        int j = 0;
        [unroll]
        for (; j < 16; j += 8)
        {
            int _141 = _51 + j;
            int _144 = _55 + 4;
            float4 res = g_temp_variance0[_141][_144] * 0.2270270287990570068359375f;
            [unroll]
            for (int i = 1; i < 5; )
            {
                int _164 = _51 + j;
                res = (res + (g_temp_variance0[_164][_144 + i] * _178[i])) + (g_temp_variance0[_164][_144 - i] * _178[i]);
                i++;
                continue;
            }
            float4 _208 = res;
            float4 _210 = max(_208, g_temp_variance0[_141][_144]);
            res = _210;
            g_temp_variance1[_51 + j][_55] = _210;
        }
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        int _229 = _51 + 4;
        float4 res_variance = g_temp_variance1[_229][_55] * 0.2270270287990570068359375f;
        [unroll]
        for (int i_1 = 1; i_1 < 5; )
        {
            res_variance = (res_variance + (g_temp_variance1[_229 + i_1][_55] * _178[i_1])) + (g_temp_variance1[_229 - i_1][_55] * _178[i_1]);
            i_1++;
            continue;
        }
        float4 _279 = res_variance;
        float4 _281 = max(_279, g_temp_variance1[_229][_55]);
        res_variance = _281;
        g_out_img[_17] = _281;
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
