struct Params
{
    int counter;
    int _pad0;
    int _pad1;
    int _pad2;
};

static const uint3 gl_WorkGroupSize = uint3(64u, 1u, 1u);

ByteAddressBuffer _23 : register(t3, space0);
ByteAddressBuffer _57 : register(t2, space0);
RWByteAddressBuffer _69 : register(u0, space0);
ByteAddressBuffer _95 : register(t1, space0);
cbuffer UniformParams
{
    Params _28_g_params : packoffset(c0);
};


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
        if (uint(_17) >= _23.Load(_28_g_params.counter * 4 + 0))
        {
            break;
        }
        uint flag = 0u;
        uint _76;
        for (int i = 0; (i < 64) && (_17 != 0); i++)
        {
            int _59 = _17 * 64;
            int _61 = _59 + i;
            uint _64 = flag;
            uint _65 = _64 | _57.Load(_61 * 4 + 0);
            flag = _65;
            if (_65 != 0u)
            {
                _76 = _69.Load((_59 + i) * 4 + 0);
            }
            else
            {
                _76 = _69.Load((_59 + i) * 4 + 0) + _95.Load((_17 - 1) * 4 + 0);
            }
            _69.Store(_61 * 4 + 0, _76);
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
