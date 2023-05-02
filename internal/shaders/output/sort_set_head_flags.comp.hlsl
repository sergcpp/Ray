static const uint3 gl_WorkGroupSize = uint3(64u, 1u, 1u);

ByteAddressBuffer _23 : register(t2, space0);
RWByteAddressBuffer _41 : register(u0, space0);
ByteAddressBuffer _50 : register(t1, space0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

void comp_main()
{
    do
    {
        int _17 = int(gl_GlobalInvocationID.x);
        if (uint(_17) >= _23.Load(4))
        {
            break;
        }
        if (_17 == 0)
        {
            _41.Store(_17 * 4 + 0, 1u);
        }
        else
        {
            _41.Store(_17 * 4 + 0, uint(int(_50.Load(_17 * 4 + 0) != _50.Load((_17 - 1) * 4 + 0))));
        }
        break;
    } while(false);
}

[numthreads(64, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
