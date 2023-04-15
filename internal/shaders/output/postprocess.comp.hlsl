struct Params
{
    uint4 rect;
    int srgb;
    int _clamp;
    float exposure;
    float inv_gamma;
    float img0_weight;
    float img1_weight;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

cbuffer UniformParams
{
    Params _158_g_params : packoffset(c0);
};

RWTexture2D<float4> g_in_img0 : register(u3, space0);
RWTexture2D<float4> g_in_img1 : register(u4, space0);
RWTexture2D<float4> g_out_raw_img : register(u1, space0);
RWTexture2D<float4> g_out_img : register(u0, space0);
RWTexture2D<float4> g_out_variance_img : register(u2, space0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

float3 clamp_and_gamma_correct(bool srgb, bool _clamp, float inv_gamma, inout float3 col)
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
    if (_clamp)
    {
        col = clamp(col, 0.0f.xxx, 1.0f.xxx);
    }
    return col;
}

float4 clamp_and_gamma_correct(bool srgb, bool _clamp, float inv_gamma, float4 col)
{
    bool param = srgb;
    bool param_1 = _clamp;
    float param_2 = inv_gamma;
    float3 param_3 = col.xyz;
    float3 _108 = clamp_and_gamma_correct(param, param_1, param_2, param_3);
    return float4(_108, col.w);
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
        bool _162 = gl_GlobalInvocationID.x >= _158_g_params.rect.z;
        bool _171;
        if (!_162)
        {
            _171 = gl_GlobalInvocationID.y >= _158_g_params.rect.w;
        }
        else
        {
            _171 = _162;
        }
        if (_171)
        {
            break;
        }
        int2 _186 = int2(_158_g_params.rect.xy + gl_GlobalInvocationID.xy);
        float4 _193 = g_in_img0[_186];
        float4 _198 = g_in_img1[_186];
        float3 _204 = _193.xyz * _158_g_params.exposure;
        float4 _307 = _193;
        _307.x = _204.x;
        float4 _309 = _307;
        _309.y = _204.y;
        float4 _311 = _309;
        _311.z = _204.z;
        float4 img0 = _311;
        float3 _215 = _198.xyz * _158_g_params.exposure;
        float4 _313 = _198;
        _313.x = _215.x;
        float4 _315 = _313;
        _315.y = _215.y;
        float4 _317 = _315;
        _317.z = _215.z;
        float4 img1 = _317;
        float4 _233 = (_311 * _158_g_params.img0_weight) + (_317 * _158_g_params.img1_weight);
        g_out_raw_img[_186] = _233;
        bool param = _158_g_params.srgb != 0;
        bool param_1 = _158_g_params._clamp != 0;
        float param_2 = _158_g_params.inv_gamma;
        float4 param_3 = _233;
        g_out_img[_186] = clamp_and_gamma_correct(param, param_1, param_2, param_3);
        float4 param_4 = img0;
        img0 = reversible_tonemap(param_4);
        float4 param_5 = img1;
        float4 _265 = reversible_tonemap(param_5);
        img1 = _265;
        float4 _270 = img0 - _265;
        g_out_variance_img[_186] = (_270 * 0.5f) * _270;
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
