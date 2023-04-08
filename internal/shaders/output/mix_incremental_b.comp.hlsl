struct Params
{
    uint4 rect;
    float main_mix_factor;
    float aux_mix_factor;
    float _pad0;
    float _pad1;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

cbuffer UniformParams
{
    Params _20_g_params : packoffset(c0);
};

RWTexture2D<float4> g_out_img : register(u0, space0);
RWTexture2D<float4> g_temp_img : register(u3, space0);
RWTexture2D<float4> g_out_base_color_img : register(u1, space0);
RWTexture2D<float4> g_temp_base_color : register(u5, space0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

void comp_main()
{
    do
    {
        bool _27 = gl_GlobalInvocationID.x >= _20_g_params.rect.z;
        bool _38;
        if (!_27)
        {
            _38 = gl_GlobalInvocationID.y >= _20_g_params.rect.w;
        }
        else
        {
            _38 = _27;
        }
        if (_38)
        {
            break;
        }
        int2 _53 = int2(_20_g_params.rect.xy + gl_GlobalInvocationID.xy);
        float3 _64 = g_out_img[_53].xyz;
        g_out_img[_53] = float4(_64 + ((g_temp_img[_53].xyz - _64) * _20_g_params.main_mix_factor), 1.0f);
        float3 _95 = g_out_base_color_img[_53].xyz;
        g_out_base_color_img[_53] = float4(_95 + ((g_temp_base_color[_53].xyz - _95) * _20_g_params.aux_mix_factor), 1.0f);
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
