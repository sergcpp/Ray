struct Params
{
    uint2 img_size;
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
    Params _165_g_params : packoffset(c0);
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
    float3 _115 = clamp_and_gamma_correct(param, param_1, param_2, param_3, param_4);
    return float4(_115, col.w);
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
        bool _169 = gl_GlobalInvocationID.x >= _165_g_params.img_size.x;
        bool _178;
        if (!_169)
        {
            _178 = gl_GlobalInvocationID.y >= _165_g_params.img_size.y;
        }
        else
        {
            _178 = _169;
        }
        if (_178)
        {
            break;
        }
        int2 _187 = int2(gl_GlobalInvocationID.xy);
        float4 _194 = g_in_img0[_187];
        float4 img0 = _194;
        float4 _199 = g_in_img1[_187];
        float4 img1 = _199;
        float4 _212 = (_194 * _165_g_params.img0_weight) + (_199 * _165_g_params.img1_weight);
        g_out_raw_img[_187] = _212;
        bool param = _165_g_params.srgb != 0;
        float param_1 = _165_g_params.exposure;
        bool param_2 = _165_g_params._clamp != 0;
        float param_3 = _165_g_params.inv_gamma;
        float4 param_4 = _212;
        g_out_img[_187] = clamp_and_gamma_correct(param, param_1, param_2, param_3, param_4);
        float4 param_5 = img0;
        img0 = reversible_tonemap(param_5);
        float4 param_6 = img1;
        float4 _247 = reversible_tonemap(param_6);
        img1 = _247;
        float4 _252 = img0 - _247;
        g_out_variance_img[_187] = (_252 * 0.5f) * _252;
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
