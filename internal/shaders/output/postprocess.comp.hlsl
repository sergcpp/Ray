struct Params
{
    uint2 img_size;
    int srgb;
    int _clamp;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

cbuffer UniformParams
{
    Params _20_g_params : packoffset(c0);
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
        bool _25 = gl_GlobalInvocationID.x >= _20_g_params.img_size.x;
        bool _35;
        if (!_25)
        {
            _35 = gl_GlobalInvocationID.y >= _20_g_params.img_size.y;
        }
        else
        {
            _35 = _25;
        }
        if (_35)
        {
            break;
        }
        int2 _44 = int2(gl_GlobalInvocationID.xy);
        float4 col = g_in_img[_44];
        int i = 0;
        [unroll]
        for (;;)
        {
            bool _64 = i < 3;
            bool _72;
            if (_64)
            {
                _72 = _20_g_params.srgb != 0;
            }
            else
            {
                _72 = _64;
            }
            if (_72)
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
        if (_20_g_params._clamp != 0)
        {
            float4 _109 = col;
            float3 _115 = clamp(_109.xyz, 0.0f.xxx, 1.0f.xxx);
            col.x = _115.x;
            col.y = _115.y;
            col.z = _115.z;
        }
        g_out_img[_44] = col;
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
