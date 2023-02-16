struct Params
{
    uint2 img_size;
    int srgb;
    int _clamp;
    float exposure;
    float _pad0;
    float _pad1;
    float _pad2;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

cbuffer UniformParams
{
    Params _21_g_params : packoffset(c0);
};

RWTexture2D<float4> g_in_img : register(u1, space0);
RWTexture2D<float4> g_out_img : register(u0, space0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

void comp_main()
{
    do
    {
        bool _26 = gl_GlobalInvocationID.x >= _21_g_params.img_size.x;
        bool _36;
        if (!_26)
        {
            _36 = gl_GlobalInvocationID.y >= _21_g_params.img_size.y;
        }
        else
        {
            _36 = _26;
        }
        if (_36)
        {
            break;
        }
        int2 _45 = int2(gl_GlobalInvocationID.xy);
        float4 col = g_in_img[_45];
        if (_21_g_params.exposure != 0.0f)
        {
            col *= _21_g_params.exposure;
        }
        int i = 0;
        [unroll]
        for (;;)
        {
            bool _75 = i < 3;
            bool _83;
            if (_75)
            {
                _83 = _21_g_params.srgb != 0;
            }
            else
            {
                _83 = _75;
            }
            if (_83)
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
        if (_21_g_params._clamp != 0)
        {
            float4 _120 = col;
            float3 _125 = clamp(_120.xyz, 0.0f.xxx, 1.0f.xxx);
            col.x = _125.x;
            col.y = _125.y;
            col.z = _125.z;
        }
        g_out_img[_45] = col;
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
