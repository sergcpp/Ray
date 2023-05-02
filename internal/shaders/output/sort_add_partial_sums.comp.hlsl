static const uint3 gl_WorkGroupSize = uint3(64u, 1u, 1u);

RWByteAddressBuffer _32 : register(u0, space0);
ByteAddressBuffer _37 : register(t1, space0);

static uint3 gl_WorkGroupID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

void comp_main()
{
    int _17 = int(gl_GlobalInvocationID.x);
    int _22 = int(gl_WorkGroupID.x);
    if (_22 != 0)
    {
        _32.Store(_17 * 4 + 0, _32.Load(_17 * 4 + 0) + _37.Load((_22 - 1) * 4 + 0));
    }
}

[numthreads(64, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
