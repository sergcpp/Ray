struct Params
{
    int shift;
    int counter;
    int _pad0;
    int _pad1;
};

struct ray_chunk_t
{
    uint hash;
    uint base;
    uint size;
};

static const uint3 gl_WorkGroupSize = uint3(64u, 1u, 1u);

ByteAddressBuffer _52 : register(t2, space0);
ByteAddressBuffer _72 : register(t1, space0);
RWByteAddressBuffer _96 : register(u0, space0);
cbuffer SPIRV_Cross_NumWorkgroups : register(b0, space0)
{
    uint3 SPIRV_Cross_NumWorkgroups_1_count : packoffset(c0);
};

cbuffer UniformParams
{
    Params _57_g_params : packoffset(c0);
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

groupshared uint g_shared_counters[16];

void comp_main()
{
    int _17 = int(gl_GlobalInvocationID.x);
    int _22 = int(gl_LocalInvocationID.x);
    for (int i = _22; i < 16; )
    {
        g_shared_counters[i] = 0u;
        i += 64;
        continue;
    }
    AllMemoryBarrier();
    GroupMemoryBarrierWithGroupSync();
    if (uint(_17) < _52.Load(_57_g_params.counter * 4 + 0))
    {
        uint _83;
        InterlockedAdd(g_shared_counters[(_72.Load(_17 * 12 + 0) >> uint(_57_g_params.shift)) & 15u], 1u, _83);
    }
    AllMemoryBarrier();
    GroupMemoryBarrierWithGroupSync();
    for (int i_1 = _22; i_1 < 16; )
    {
        _96.Store(((uint(i_1) * SPIRV_Cross_NumWorkgroups_1_count.x) + gl_WorkGroupID.x) * 4 + 0, g_shared_counters[i_1]);
        i_1 += 64;
        continue;
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
