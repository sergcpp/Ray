struct Params
{
    uint4 rect;
    float main_mix_factor;
    float aux_mix_factor;
    int iteration;
    float _pad1;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

cbuffer UniformParams
{
    Params _21_g_params : packoffset(c0);
};

RWTexture2D<uint> g_req_samples_img : register(u7, space0);
RWTexture2D<float4> g_out_img : register(u0, space0);
RWTexture2D<float4> g_temp_img : register(u3, space0);
RWTexture2D<float4> g_out_depth_normals_img : register(u2, space0);
RWTexture2D<float4> g_temp_depth_normals_img : register(u6, space0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

void comp_main()
{
    do
    {
        bool _27 = gl_GlobalInvocationID.x >= _21_g_params.rect.z;
        bool _38;
        if (!_27)
        {
            _38 = gl_GlobalInvocationID.y >= _21_g_params.rect.w;
        }
        else
        {
            _38 = _27;
        }
        if (_38)
        {
            break;
        }
        int2 _53 = int2(_21_g_params.rect.xy + gl_GlobalInvocationID.xy);
        [branch]
        if (g_req_samples_img[_53].xxxx.x < uint(_21_g_params.iteration))
        {
            break;
        }
        float3 _83 = g_out_img[_53].xyz;
        g_out_img[_53] = float4(_83 + ((g_temp_img[_53].xyz - _83) * _21_g_params.main_mix_factor), 1.0f);
        float4 _114 = g_out_depth_normals_img[_53];
        float4 _119 = g_temp_depth_normals_img[_53];
        float3 _125 = clamp(_119.xyz, (-1.0f).xxx, 1.0f.xxx);
        float4 _183 = _119;
        _183.x = _125.x;
        float4 _185 = _183;
        _185.y = _125.y;
        float4 _187 = _185;
        _187.z = _125.z;
        g_out_depth_normals_img[_53] = _114 + ((_187 - _114) * _21_g_params.aux_mix_factor);
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}