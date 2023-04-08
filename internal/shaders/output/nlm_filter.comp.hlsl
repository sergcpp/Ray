struct Params
{
    uint4 rect;
    float2 inv_img_size;
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
    Params _195_g_params : packoffset(c0);
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
        int2 _207 = int2(_195_g_params.rect.xy + gl_GlobalInvocationID.xy);
        int2 _212 = int2(gl_LocalInvocationID.xy);
        float2 _223 = (float2(_207) + 0.5f.xx) * _195_g_params.inv_img_size;
        float4 param = g_in_img.SampleLevel(_g_in_img_sampler, _223, 0.0f, int2(-4, -4));
        float4 _235 = reversible_tonemap(param);
        int _242 = _212.y;
        int _245 = _212.x;
        g_temp_color0[_242][_245] = spvPackHalf2x16(_235.xy);
        g_temp_color1[_242][_245] = spvPackHalf2x16(_235.zw);
        float4 param_1 = g_in_img.SampleLevel(_g_in_img_sampler, _223, 0.0f, int2(4, -4));
        float4 _270 = reversible_tonemap(param_1);
        int _277 = 8 + _245;
        g_temp_color0[_242][_277] = spvPackHalf2x16(_270.xy);
        g_temp_color1[_242][_277] = spvPackHalf2x16(_270.zw);
        float4 param_2 = g_in_img.SampleLevel(_g_in_img_sampler, _223, 0.0f, int2(-4, 4));
        float4 _298 = reversible_tonemap(param_2);
        int _301 = 8 + _242;
        g_temp_color0[_301][_245] = spvPackHalf2x16(_298.xy);
        g_temp_color1[_301][_245] = spvPackHalf2x16(_298.zw);
        float4 param_3 = g_in_img.SampleLevel(_g_in_img_sampler, _223, 0.0f, int2(4, 4));
        float4 _325 = reversible_tonemap(param_3);
        g_temp_color0[_301][_277] = spvPackHalf2x16(_325.xy);
        g_temp_color1[_301][_277] = spvPackHalf2x16(_325.zw);
        float4 _350 = g_variance_img.SampleLevel(_g_variance_img_sampler, _223, 0.0f, int2(-4, -4));
        g_temp_variance0[_242][_245] = spvPackHalf2x16(_350.xy);
        g_temp_variance1[_242][_245] = spvPackHalf2x16(_350.zw);
        float4 _376 = g_variance_img.SampleLevel(_g_variance_img_sampler, _223, 0.0f, int2(4, -4));
        g_temp_variance0[_242][_277] = spvPackHalf2x16(_376.xy);
        g_temp_variance1[_242][_277] = spvPackHalf2x16(_376.zw);
        float4 _400 = g_variance_img.SampleLevel(_g_variance_img_sampler, _223, 0.0f, int2(-4, 4));
        g_temp_variance0[_301][_245] = spvPackHalf2x16(_400.xy);
        g_temp_variance1[_301][_245] = spvPackHalf2x16(_400.zw);
        float4 _424 = g_variance_img.SampleLevel(_g_variance_img_sampler, _223, 0.0f, int2(4, 4));
        g_temp_variance0[_301][_277] = spvPackHalf2x16(_424.xy);
        g_temp_variance1[_301][_277] = spvPackHalf2x16(_424.zw);
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        bool _453 = gl_GlobalInvocationID.x >= _195_g_params.rect.z;
        bool _462;
        if (!_453)
        {
            _462 = gl_GlobalInvocationID.y >= _195_g_params.rect.w;
        }
        else
        {
            _462 = _453;
        }
        if (_462)
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
                        int _507 = _242 + 4;
                        int _509 = _507 + q;
                        int _512 = _245 + 4;
                        int _514 = _512 + p;
                        int _543 = (_507 + k) + q;
                        int _550 = (_512 + l) + p;
                        float4 _607 = float4(spvUnpackHalf2x16(g_temp_variance0[_509][_514]), spvUnpackHalf2x16(g_temp_variance1[_509][_514]));
                        float4 _647 = float4(spvUnpackHalf2x16(g_temp_variance0[_543][_550]), spvUnpackHalf2x16(g_temp_variance1[_543][_550]));
                        float4 _654 = float4(spvUnpackHalf2x16(g_temp_color0[_509][_514]), spvUnpackHalf2x16(g_temp_color1[_509][_514])) - float4(spvUnpackHalf2x16(g_temp_color0[_543][_550]), spvUnpackHalf2x16(g_temp_color1[_543][_550]));
                        _distance += (mad(_654, _654, -((_607 + min(_607, _647)) * _195_g_params.alpha)) / (9.9999997473787516355514526367188e-05f.xxxx + ((_607 + _647) * (_195_g_params.damping * _195_g_params.damping))));
                        p++;
                        continue;
                    }
                }
                float _705 = exp(-max(0.0f, 2.25f * (((_distance.x + _distance.y) + _distance.z) + _distance.w)));
                int _710 = (_242 + 4) + k;
                int _715 = (_245 + 4) + l;
                sum_output += (float4(spvUnpackHalf2x16(g_temp_color0[_710][_715]), spvUnpackHalf2x16(g_temp_color1[_710][_715])) * _705);
                sum_weight += _705;
            }
        }
        [flatten]
        if (sum_weight != 0.0f)
        {
            sum_output /= sum_weight.xxxx;
        }
        float4 param_4 = sum_output;
        float4 _758 = reversible_tonemap_invert(param_4);
        sum_output = _758;
        g_out_raw_img[_207] = _758;
        bool param_5 = _195_g_params.srgb != 0;
        float param_6 = _195_g_params.exposure;
        bool param_7 = _195_g_params._clamp != 0;
        float param_8 = _195_g_params.inv_gamma;
        float4 param_9 = _758;
        g_out_img[_207] = clamp_and_gamma_correct(param_5, param_6, param_7, param_8, param_9);
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
