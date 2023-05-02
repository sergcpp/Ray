struct Params
{
    int offset;
    int stride;
    int _pad0;
    int _pad1;
};

static const uint3 gl_WorkGroupSize = uint3(64u, 1u, 1u);

ByteAddressBuffer _38 : register(t2, space0);
RWByteAddressBuffer _108 : register(u0, space0);
RWByteAddressBuffer _123 : register(u1, space0);
cbuffer UniformParams
{
    Params _42_g_params : packoffset(c0);
};


static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared uint g_temp[2][64];

void comp_main()
{
    int _17 = int(gl_GlobalInvocationID.x);
    int _22 = int(gl_LocalInvocationID.x);
    int pout = 0;
    int pin = 1;
    g_temp[0][_22] = _38.Load(((_42_g_params.stride * _17) + _42_g_params.offset) * 4 + 0);
    g_temp[1][_22] = 0u;
    AllMemoryBarrier();
    GroupMemoryBarrierWithGroupSync();
    for (int offset = 1; offset < 64; offset *= 2)
    {
        int _71 = pout;
        pout = 1 - _71;
        pin = _71;
        if (_22 >= offset)
        {
            g_temp[pout][_22] = g_temp[pin][_22] + g_temp[pin][_22 - offset];
        }
        else
        {
            g_temp[pout][_22] = g_temp[pin][_22];
        }
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
    }
    _108.Store(_17 * 4 + 0, g_temp[pout][_22]);
    if (_22 == 63)
    {
        _123.Store(gl_WorkGroupID.x * 4 + 0, g_temp[pout][_22]);
    }
}

[numthreads(64, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
