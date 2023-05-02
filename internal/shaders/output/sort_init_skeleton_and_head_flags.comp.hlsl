struct Params
{
    int counter;
    int _pad0;
    int _pad1;
    int _pad2;
};

struct ray_chunk_t
{
    uint hash;
    uint base;
    uint size;
};

static const uint3 gl_WorkGroupSize = uint3(64u, 1u, 1u);

ByteAddressBuffer _23 : register(t2, space0);
ByteAddressBuffer _45 : register(t3, space0);
RWByteAddressBuffer _52 : register(u0, space0);
ByteAddressBuffer _58 : register(t4, space0);
RWByteAddressBuffer _67 : register(u1, space0);
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
        uint _48 = _45.Load(_17 * 4 + 0);
        _52.Store(_48 * 4 + 0, _58.Load(_17 * 12 + 4));
        _67.Store(_48 * 4 + 0, 1u);
        break;
    } while(false);
}

[numthreads(64, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
