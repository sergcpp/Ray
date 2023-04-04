struct Params
{
    uint2 img_size;
    int srgb;
    int _clamp;
    float exposure;
    float img0_weight;
    float img1_weight;
    float _pad2;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

cbuffer UniformParams
{
    Params _19_g_params : packoffset(c0);
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

float4 clamp_and_gamma_correct(inout float4 col)
{
    col *= _19_g_params.exposure;
    int i = 0;
    [unroll]
    for (;;)
    {
        bool _36 = i < 3;
        bool _44;
        if (_36)
        {
            _44 = _19_g_params.srgb != 0;
        }
        else
        {
            _44 = _36;
        }
        if (_44)
        {
            if (col[i] < 0.003130800090730190277099609375f)
            {
                col[i] = 12.9200000762939453125f * col[i];
            }
            else
            {
                col[i] = mad(1.05499994754791259765625f, pow(col[i], 0.4166666567325592041015625f), -0.054999999701976776123046875f);
            }
            i++;
            continue;
        }
        else
        {
            break;
        }
    }
    if (_19_g_params._clamp != 0)
    {
        float4 _81 = col;
        float3 _87 = clamp(_81.xyz, 0.0f.xxx, 1.0f.xxx);
        col.x = _87.x;
        col.y = _87.y;
        col.z = _87.z;
    }
    return col;
}

void comp_main()
{
    do
    {
        bool _109 = gl_GlobalInvocationID.x >= _19_g_params.img_size.x;
        bool _118;
        if (!_109)
        {
            _118 = gl_GlobalInvocationID.y >= _19_g_params.img_size.y;
        }
        else
        {
            _118 = _109;
        }
        if (_118)
        {
            break;
        }
        int2 _127 = int2(gl_GlobalInvocationID.xy);
        float4 _134 = g_in_img0[_127];
        float4 img0 = _134;
        float4 _139 = g_in_img1[_127];
        float4 img1 = _139;
        float4 _151 = (_134 * _19_g_params.img0_weight) + (_139 * _19_g_params.img1_weight);
        g_out_raw_img[_127] = _151;
        float4 param = _151;
        float4 _159 = clamp_and_gamma_correct(param);
        g_out_img[_127] = _159;
        float4 param_1 = img0;
        float4 _166 = clamp_and_gamma_correct(param_1);
        img0 = _166;
        float4 param_2 = img1;
        float4 _169 = clamp_and_gamma_correct(param_2);
        img1 = _169;
        float4 _174 = img0 - _169;
        g_out_variance_img[_127] = (_174 * 0.5f) * _174;
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
