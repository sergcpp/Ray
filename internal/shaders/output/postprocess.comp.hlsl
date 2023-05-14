struct Params
{
    uint4 rect;
    float exposure;
    float inv_gamma;
    float img0_weight;
    float img1_weight;
    int tonemap_mode;
    float variance_threshold;
    int iteration;
    float _pad2;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

cbuffer UniformParams
{
    Params _334_g_params : packoffset(c0);
};

RWTexture2D<float4> g_in_img0 : register(u4, space0);
RWTexture2D<float4> g_in_img1 : register(u5, space0);
RWTexture2D<float4> g_out_raw_img : register(u1, space0);
Texture3D<float4> g_tonemap_lut : register(t6, space0);
SamplerState _g_tonemap_lut_sampler : register(s6, space0);
RWTexture2D<float4> g_out_img : register(u0, space0);
RWTexture2D<float4> g_out_variance_img : register(u2, space0);
RWTexture2D<uint> g_out_req_samples_img : register(u3, space0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

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
    float3 _108 = TonemapStandard(param, param_1);
    return float4(_108, col.w);
}

float3 TonemapLUT_manual(Texture3D<float4> lut, SamplerState _lut_sampler, float inv_gamma, float3 col)
{
    float3 _131 = ((col / (col + 1.0f.xxx)) * 47.0f) + 0.5f.xxx;
    int3 _136 = int3(_131);
    float3 _139 = frac(_131);
    float _143 = _139.x;
    float _147 = _139.y;
    float _151 = _139.z;
    float _210 = 1.0f - _143;
    float _246 = 1.0f - _147;
    float3 ret = (((((lut.Load(int4(_136, 0), int3(0, 0, 0)).xyz * _210) + (lut.Load(int4(_136, 0), int3(1, 0, 0)).xyz * _143)) * _246) + (((lut.Load(int4(_136, 0), int3(0, 1, 0)).xyz * _210) + (lut.Load(int4(_136, 0), int3(1, 1, 0)).xyz * _143)) * _147)) * (1.0f - _151)) + (((((lut.Load(int4(_136, 0), int3(0, 0, 1)).xyz * _210) + (lut.Load(int4(_136, 0), int3(1, 0, 1)).xyz * _143)) * _246) + (((lut.Load(int4(_136, 0), int3(0, 1, 1)).xyz * _210) + (lut.Load(int4(_136, 0), int3(1, 1, 1)).xyz * _143)) * _147)) * _151);
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
        bool _338 = gl_GlobalInvocationID.x >= _334_g_params.rect.z;
        bool _347;
        if (!_338)
        {
            _347 = gl_GlobalInvocationID.y >= _334_g_params.rect.w;
        }
        else
        {
            _347 = _338;
        }
        if (_347)
        {
            break;
        }
        int2 _362 = int2(_334_g_params.rect.xy + gl_GlobalInvocationID.xy);
        float4 _369 = g_in_img0[_362];
        float4 _374 = g_in_img1[_362];
        float3 _380 = _369.xyz * _334_g_params.exposure;
        float4 _508 = _369;
        _508.x = _380.x;
        float4 _510 = _508;
        _510.y = _380.y;
        float4 _512 = _510;
        _512.z = _380.z;
        float4 img0 = _512;
        float3 _391 = _374.xyz * _334_g_params.exposure;
        float4 _514 = _374;
        _514.x = _391.x;
        float4 _516 = _514;
        _516.y = _391.y;
        float4 _518 = _516;
        _518.z = _391.z;
        float4 img1 = _518;
        float4 _408 = (_512 * _334_g_params.img0_weight) + (_518 * _334_g_params.img1_weight);
        g_out_raw_img[_362] = _408;
        float4 tonemapped_res;
        [branch]
        if (_334_g_params.tonemap_mode == 0)
        {
            float param = _334_g_params.inv_gamma;
            float4 param_1 = _408;
            tonemapped_res = TonemapStandard(param, param_1);
        }
        else
        {
            float param_2 = _334_g_params.inv_gamma;
            float4 param_3 = _408;
            tonemapped_res = TonemapLUT_manual(g_tonemap_lut, _g_tonemap_lut_sampler, param_2, param_3);
        }
        g_out_img[_362] = tonemapped_res;
        float4 param_4 = img0;
        img0 = reversible_tonemap(param_4);
        float4 param_5 = img1;
        float4 _445 = reversible_tonemap(param_5);
        img1 = _445;
        float4 _449 = img0 - _445;
        float4 _454 = (_449 * 0.5f) * _449;
        g_out_variance_img[_362] = _454;
        float4 _463 = _334_g_params.variance_threshold.xxxx;
        if (any(bool4(_454.x >= _463.x, _454.y >= _463.y, _454.z >= _463.z, _454.w >= _463.w)))
        {
            g_out_req_samples_img[_362] = uint(_334_g_params.iteration + 1).x;
        }
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
