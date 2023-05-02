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
ByteAddressBuffer _41 : register(t2, space0);
RWByteAddressBuffer _52 : register(u0, space0);
ByteAddressBuffer _56 : register(t3, space0);
ByteAddressBuffer _63 : register(t1, space0);
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
        if (_41.Load(gl_GlobalInvocationID.x * 4 + 0) != 0u)
        {
            _52.Store(_56.Load(gl_GlobalInvocationID.x * 4 + 0) * 12 + 0, _63.Load(gl_GlobalInvocationID.x * 4 + 0));
            _52.Store(_56.Load(gl_GlobalInvocationID.x * 4 + 0) * 12 + 4, gl_GlobalInvocationID.x);
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
