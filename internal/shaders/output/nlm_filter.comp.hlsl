struct Params
{
    uint4 rect;
    float2 inv_img_size;
    float alpha;
    float damping;
    int srgb;
    int _clamp;
    float inv_gamma;
    float _pad0;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

cbuffer UniformParams
{
    Params _188_g_params : packoffset(c0);
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
    float3 _114 = clamp_and_gamma_correct(param, param_1, param_2, param_3);
    return float4(_114, col.w);
}

void comp_main()
{
    do
    {
        int2 _200 = int2(_188_g_params.rect.xy + gl_GlobalInvocationID.xy);
        int2 _205 = int2(gl_LocalInvocationID.xy);
        float2 _216 = (float2(_200) + 0.5f.xx) * _188_g_params.inv_img_size;
        float4 param = g_in_img.SampleLevel(_g_in_img_sampler, _216, 0.0f, int2(-4, -4));
        float4 _228 = reversible_tonemap(param);
        int _235 = _205.y;
        int _238 = _205.x;
        g_temp_color0[_235][_238] = spvPackHalf2x16(_228.xy);
        g_temp_color1[_235][_238] = spvPackHalf2x16(_228.zw);
        float4 param_1 = g_in_img.SampleLevel(_g_in_img_sampler, _216, 0.0f, int2(4, -4));
        float4 _263 = reversible_tonemap(param_1);
        int _270 = 8 + _238;
        g_temp_color0[_235][_270] = spvPackHalf2x16(_263.xy);
        g_temp_color1[_235][_270] = spvPackHalf2x16(_263.zw);
        float4 param_2 = g_in_img.SampleLevel(_g_in_img_sampler, _216, 0.0f, int2(-4, 4));
        float4 _291 = reversible_tonemap(param_2);
        int _294 = 8 + _235;
        g_temp_color0[_294][_238] = spvPackHalf2x16(_291.xy);
        g_temp_color1[_294][_238] = spvPackHalf2x16(_291.zw);
        float4 param_3 = g_in_img.SampleLevel(_g_in_img_sampler, _216, 0.0f, int2(4, 4));
        float4 _318 = reversible_tonemap(param_3);
        g_temp_color0[_294][_270] = spvPackHalf2x16(_318.xy);
        g_temp_color1[_294][_270] = spvPackHalf2x16(_318.zw);
        float4 _343 = g_variance_img.SampleLevel(_g_variance_img_sampler, _216, 0.0f, int2(-4, -4));
        g_temp_variance0[_235][_238] = spvPackHalf2x16(_343.xy);
        g_temp_variance1[_235][_238] = spvPackHalf2x16(_343.zw);
        float4 _369 = g_variance_img.SampleLevel(_g_variance_img_sampler, _216, 0.0f, int2(4, -4));
        g_temp_variance0[_235][_270] = spvPackHalf2x16(_369.xy);
        g_temp_variance1[_235][_270] = spvPackHalf2x16(_369.zw);
        float4 _393 = g_variance_img.SampleLevel(_g_variance_img_sampler, _216, 0.0f, int2(-4, 4));
        g_temp_variance0[_294][_238] = spvPackHalf2x16(_393.xy);
        g_temp_variance1[_294][_238] = spvPackHalf2x16(_393.zw);
        float4 _417 = g_variance_img.SampleLevel(_g_variance_img_sampler, _216, 0.0f, int2(4, 4));
        g_temp_variance0[_294][_270] = spvPackHalf2x16(_417.xy);
        g_temp_variance1[_294][_270] = spvPackHalf2x16(_417.zw);
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        bool _446 = gl_GlobalInvocationID.x >= _188_g_params.rect.z;
        bool _455;
        if (!_446)
        {
            _455 = gl_GlobalInvocationID.y >= _188_g_params.rect.w;
        }
        else
        {
            _455 = _446;
        }
        if (_455)
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
                        int _500 = _235 + 4;
                        int _502 = _500 + q;
                        int _505 = _238 + 4;
                        int _507 = _505 + p;
                        int _536 = (_500 + k) + q;
                        int _543 = (_505 + l) + p;
                        float4 _600 = float4(spvUnpackHalf2x16(g_temp_variance0[_502][_507]), spvUnpackHalf2x16(g_temp_variance1[_502][_507]));
                        float4 _640 = float4(spvUnpackHalf2x16(g_temp_variance0[_536][_543]), spvUnpackHalf2x16(g_temp_variance1[_536][_543]));
                        float4 _647 = float4(spvUnpackHalf2x16(g_temp_color0[_502][_507]), spvUnpackHalf2x16(g_temp_color1[_502][_507])) - float4(spvUnpackHalf2x16(g_temp_color0[_536][_543]), spvUnpackHalf2x16(g_temp_color1[_536][_543]));
                        _distance += (mad(_647, _647, -((_600 + min(_600, _640)) * _188_g_params.alpha)) / (9.9999997473787516355514526367188e-05f.xxxx + ((_600 + _640) * (_188_g_params.damping * _188_g_params.damping))));
                        p++;
                        continue;
                    }
                }
                float _698 = exp(-max(0.0f, 2.25f * (((_distance.x + _distance.y) + _distance.z) + _distance.w)));
                int _703 = (_235 + 4) + k;
                int _708 = (_238 + 4) + l;
                sum_output += (float4(spvUnpackHalf2x16(g_temp_color0[_703][_708]), spvUnpackHalf2x16(g_temp_color1[_703][_708])) * _698);
                sum_weight += _698;
            }
        }
        [flatten]
        if (sum_weight != 0.0f)
        {
            sum_output /= sum_weight.xxxx;
        }
        float4 param_4 = sum_output;
        float4 _751 = reversible_tonemap_invert(param_4);
        sum_output = _751;
        g_out_raw_img[_200] = _751;
        bool param_5 = _188_g_params.srgb != 0;
        bool param_6 = _188_g_params._clamp != 0;
        float param_7 = _188_g_params.inv_gamma;
        float4 param_8 = _751;
        g_out_img[_200] = clamp_and_gamma_correct(param_5, param_6, param_7, param_8);
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
