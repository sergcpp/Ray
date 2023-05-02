struct Params
{
    int chunks_counter;
    int rays_counter;
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

ByteAddressBuffer _20 : register(t4, space0);
RWByteAddressBuffer _52 : register(u0, space0);
cbuffer UniformParams
{
    Params _26_g_params : packoffset(c0);
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
        if (gl_GlobalInvocationID.x >= _20.Load(_26_g_params.chunks_counter * 4 + 0))
        {
            break;
        }
        if (gl_GlobalInvocationID.x == (_20.Load(_26_g_params.chunks_counter * 4 + 0) - 1u))
        {
            _52.Store(gl_GlobalInvocationID.x * 12 + 8, _20.Load(_26_g_params.rays_counter * 4 + 0) - _52.Load(gl_GlobalInvocationID.x * 12 + 4));
        }
        else
        {
            _52.Store(gl_GlobalInvocationID.x * 12 + 8, _52.Load((gl_GlobalInvocationID.x + 1u) * 12 + 4) - _52.Load(gl_GlobalInvocationID.x * 12 + 4));
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
