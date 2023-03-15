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

groupshared float4 g_temp_color[16][16];
groupshared float4 g_temp_variance[16][16];

void comp_main()
{
    do
    {
        int2 _17 = int2(gl_GlobalInvocationID.xy);
        int2 _22 = int2(gl_LocalInvocationID.xy);
        float2 _41 = (float2(_17) + 0.5f.xx) / float2(_35_g_params.img_size);
        int _51 = _22.y;
        int _55 = _22.x;
        g_temp_color[_51][_55] = g_in_img.SampleLevel(_g_in_img_sampler, _41, 0.0f, int2(-4, -4));
        int _75 = 8 + _55;
        g_temp_color[_51][_75] = g_in_img.SampleLevel(_g_in_img_sampler, _41, 0.0f, int2(4, -4));
        int _84 = 8 + _51;
        g_temp_color[_84][_55] = g_in_img.SampleLevel(_g_in_img_sampler, _41, 0.0f, int2(-4, 4));
        g_temp_color[_84][_75] = g_in_img.SampleLevel(_g_in_img_sampler, _41, 0.0f, int2(4, 4));
        g_temp_variance[_51][_55] = g_variance_img.SampleLevel(_g_variance_img_sampler, _41, 0.0f, int2(-4, -4));
        g_temp_variance[_51][_75] = g_variance_img.SampleLevel(_g_variance_img_sampler, _41, 0.0f, int2(4, -4));
        g_temp_variance[_84][_55] = g_variance_img.SampleLevel(_g_variance_img_sampler, _41, 0.0f, int2(-4, 4));
        g_temp_variance[_84][_75] = g_variance_img.SampleLevel(_g_variance_img_sampler, _41, 0.0f, int2(4, 4));
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        bool _156 = gl_GlobalInvocationID.x >= _35_g_params.img_size.x;
        bool _165;
        if (!_156)
        {
            _165 = gl_GlobalInvocationID.y >= _35_g_params.img_size.y;
        }
        else
        {
            _165 = _156;
        }
        if (_165)
        {
            break;
        }
        float4 sum_output = 0.0f.xxxx;
        float sum_weight = 0.0f;
        int k = -3;
        [unroll]
        for (; k <= 3; k++)
        {
            int l = -3;
            [unroll]
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
                        int _214 = _51 + 4;
                        int _216 = _214 + q;
                        int _219 = _55 + 4;
                        int _221 = _219 + p;
                        int _231 = (_214 + k) + q;
                        int _238 = (_219 + l) + p;
                        float4 _277 = g_temp_color[_216][_221] - g_temp_color[_231][_238];
                        _distance += (mad(_277, _277, -((g_temp_variance[_216][_221] + min(g_temp_variance[_216][_221], g_temp_variance[_231][_238])) * _35_g_params.alpha)) / (9.9999997473787516355514526367188e-05f.xxxx + ((g_temp_variance[_216][_221] + g_temp_variance[_231][_238]) * (_35_g_params.damping * _35_g_params.damping))));
                        p++;
                        continue;
                    }
                }
                float _329 = exp(-max(0.0f, 2.25f * (((_distance.x + _distance.y) + _distance.z) + _distance.w)));
                sum_output += (g_temp_color[(_51 + 4) + k][(_55 + 4) + l] * _329);
                sum_weight += _329;
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
