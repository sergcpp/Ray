struct Params
{
    uint4 rect;
    int srgb;
    float exposure;
    float inv_gamma;
    float img0_weight;
    float img1_weight;
    float _pad0;
    float _pad1;
    float _pad2;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

cbuffer UniformParams
{
    Params _150_g_params : packoffset(c0);
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

float3 clamp_and_gamma_correct(bool srgb, float inv_gamma, inout float3 col)
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
    return clamp(col, 0.0f.xxx, 1.0f.xxx);
}

float4 clamp_and_gamma_correct(bool srgb, float inv_gamma, float4 col)
{
    bool param = srgb;
    float param_1 = inv_gamma;
    float3 param_2 = col.xyz;
    float3 _100 = clamp_and_gamma_correct(param, param_1, param_2);
    return float4(_100, col.w);
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
        bool _154 = gl_GlobalInvocationID.x >= _150_g_params.rect.z;
        bool _163;
        if (!_154)
        {
            _163 = gl_GlobalInvocationID.y >= _150_g_params.rect.w;
        }
        else
        {
            _163 = _154;
        }
        if (_163)
        {
            break;
        }
        int2 _178 = int2(_150_g_params.rect.xy + gl_GlobalInvocationID.xy);
        float4 _185 = g_in_img0[_178];
        float4 _190 = g_in_img1[_178];
        float3 _197 = _185.xyz * _150_g_params.exposure;
        float4 _295 = _185;
        _295.x = _197.x;
        float4 _297 = _295;
        _297.y = _197.y;
        float4 _299 = _297;
        _299.z = _197.z;
        float4 img0 = _299;
        float3 _208 = _190.xyz * _150_g_params.exposure;
        float4 _301 = _190;
        _301.x = _208.x;
        float4 _303 = _301;
        _303.y = _208.y;
        float4 _305 = _303;
        _305.z = _208.z;
        float4 img1 = _305;
        float4 _226 = (_299 * _150_g_params.img0_weight) + (_305 * _150_g_params.img1_weight);
        g_out_raw_img[_178] = _226;
        bool param = _150_g_params.srgb != 0;
        float param_1 = _150_g_params.inv_gamma;
        float4 param_2 = _226;
        g_out_img[_178] = clamp_and_gamma_correct(param, param_1, param_2);
        float4 param_3 = img0;
        img0 = reversible_tonemap(param_3);
        float4 param_4 = img1;
        float4 _252 = reversible_tonemap(param_4);
        img1 = _252;
        float4 _257 = img0 - _252;
        g_out_variance_img[_178] = (_257 * 0.5f) * _257;
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
