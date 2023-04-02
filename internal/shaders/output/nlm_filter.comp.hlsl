struct Params
{
    uint2 img_size;
    float alpha;
    float damping;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

cbuffer UniformParams
{
    Params _35_g_params : packoffset(c0);
};

Texture2D<float4> g_in_img : register(t1, space0);
SamplerState _g_in_img_sampler : register(s1, space0);
Texture2D<float4> g_variance_img : register(t2, space0);
SamplerState _g_variance_img_sampler : register(s2, space0);
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
        g_temp_color0[_63][_67] = spvPackHalf2x16(_54.xy);
        g_temp_color1[_63][_67] = spvPackHalf2x16(_54.zw);
        float4 _90 = g_in_img.SampleLevel(_g_in_img_sampler, _41, 0.0f, int2(4, -4));
        int _97 = 8 + _67;
        g_temp_color0[_63][_97] = spvPackHalf2x16(_90.xy);
        g_temp_color1[_63][_97] = spvPackHalf2x16(_90.zw);
        float4 _116 = g_in_img.SampleLevel(_g_in_img_sampler, _41, 0.0f, int2(-4, 4));
        int _119 = 8 + _63;
        g_temp_color0[_119][_67] = spvPackHalf2x16(_116.xy);
        g_temp_color1[_119][_67] = spvPackHalf2x16(_116.zw);
        float4 _141 = g_in_img.SampleLevel(_g_in_img_sampler, _41, 0.0f, int2(4, 4));
        g_temp_color0[_119][_97] = spvPackHalf2x16(_141.xy);
        g_temp_color1[_119][_97] = spvPackHalf2x16(_141.zw);
        float4 _166 = g_variance_img.SampleLevel(_g_variance_img_sampler, _41, 0.0f, int2(-4, -4));
        g_temp_variance0[_63][_67] = spvPackHalf2x16(_166.xy);
        g_temp_variance1[_63][_67] = spvPackHalf2x16(_166.zw);
        float4 _192 = g_variance_img.SampleLevel(_g_variance_img_sampler, _41, 0.0f, int2(4, -4));
        g_temp_variance0[_63][_97] = spvPackHalf2x16(_192.xy);
        g_temp_variance1[_63][_97] = spvPackHalf2x16(_192.zw);
        float4 _216 = g_variance_img.SampleLevel(_g_variance_img_sampler, _41, 0.0f, int2(-4, 4));
        g_temp_variance0[_119][_67] = spvPackHalf2x16(_216.xy);
        g_temp_variance1[_119][_67] = spvPackHalf2x16(_216.zw);
        float4 _240 = g_variance_img.SampleLevel(_g_variance_img_sampler, _41, 0.0f, int2(4, 4));
        g_temp_variance0[_119][_97] = spvPackHalf2x16(_240.xy);
        g_temp_variance1[_119][_97] = spvPackHalf2x16(_240.zw);
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        bool _271 = gl_GlobalInvocationID.x >= _35_g_params.img_size.x;
        bool _280;
        if (!_271)
        {
            _280 = gl_GlobalInvocationID.y >= _35_g_params.img_size.y;
        }
        else
        {
            _280 = _271;
        }
        if (_280)
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
                        int _328 = _63 + 4;
                        int _330 = _328 + q;
                        int _333 = _67 + 4;
                        int _335 = _333 + p;
                        int _364 = (_328 + k) + q;
                        int _371 = (_333 + l) + p;
                        float4 _428 = float4(spvUnpackHalf2x16(g_temp_variance0[_330][_335]), spvUnpackHalf2x16(g_temp_variance1[_330][_335]));
                        float4 _468 = float4(spvUnpackHalf2x16(g_temp_variance0[_364][_371]), spvUnpackHalf2x16(g_temp_variance1[_364][_371]));
                        float4 _475 = float4(spvUnpackHalf2x16(g_temp_color0[_330][_335]), spvUnpackHalf2x16(g_temp_color1[_330][_335])) - float4(spvUnpackHalf2x16(g_temp_color0[_364][_371]), spvUnpackHalf2x16(g_temp_color1[_364][_371]));
                        _distance += (mad(_475, _475, -((_428 + min(_428, _468)) * _35_g_params.alpha)) / (9.9999997473787516355514526367188e-05f.xxxx + ((_428 + _468) * (_35_g_params.damping * _35_g_params.damping))));
                        p++;
                        continue;
                    }
                }
                float _527 = exp(-max(0.0f, 2.25f * (((_distance.x + _distance.y) + _distance.z) + _distance.w)));
                int _532 = (_63 + 4) + k;
                int _537 = (_67 + 4) + l;
                sum_output += (float4(spvUnpackHalf2x16(g_temp_color0[_532][_537]), spvUnpackHalf2x16(g_temp_color1[_532][_537])) * _527);
                sum_weight += _527;
            }
        }
        [flatten]
        if (sum_weight != 0.0f)
        {
            sum_output /= sum_weight.xxxx;
        }
        g_out_img[_17] = sum_output;
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
