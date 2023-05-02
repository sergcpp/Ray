static const uint3 gl_WorkGroupSize = uint3(64u, 1u, 1u);

ByteAddressBuffer _38 : register(t3, space0);
ByteAddressBuffer _51 : register(t4, space0);
RWByteAddressBuffer _147 : register(u0, space0);
RWByteAddressBuffer _162 : register(u1, space0);
RWByteAddressBuffer _174 : register(u2, space0);

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
groupshared uint g_temp_flags[2][64];

void comp_main()
{
    int _17 = int(gl_GlobalInvocationID.x);
    int _22 = int(gl_LocalInvocationID.x);
    int pout = 0;
    int pin = 1;
    g_temp[0][_22] = _38.Load(_17 * 4 + 0);
    g_temp_flags[0][_22] = _51.Load(_17 * 4 + 0);
    g_temp[1][_22] = 0u;
    g_temp_flags[1][_22] = 0u;
    AllMemoryBarrier();
    GroupMemoryBarrierWithGroupSync();
    uint _105;
    for (int offset = 1; offset < 64; offset *= 2)
    {
        int _74 = pout;
        pout = 1 - _74;
        pin = _74;
        if (_22 >= offset)
        {
            g_temp_flags[pout][_22] = g_temp_flags[pin][_22] | g_temp_flags[pin][_22 - offset];
            if (g_temp_flags[pin][_22] != 0u)
            {
                _105 = g_temp[pin][_22];
            }
            else
            {
                _105 = g_temp[pin][_22] + g_temp[pin][_22 - offset];
            }
            g_temp[pout][_22] = _105;
        }
        else
        {
            g_temp_flags[pout][_22] = g_temp_flags[pin][_22];
            g_temp[pout][_22] = g_temp[pin][_22];
        }
        AllMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
    }
    _147.Store(_17 * 4 + 0, g_temp[pout][_22]);
    if (_22 == 63)
    {
        _162.Store(gl_WorkGroupID.x * 4 + 0, g_temp[pout][_22]);
        _174.Store(gl_WorkGroupID.x * 4 + 0, g_temp_flags[pout][_22]);
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
