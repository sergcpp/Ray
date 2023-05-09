struct Params
{
    int counter;
    int _pad0;
    int _pad1;
    int _pad2;
};

struct ray_data_t
{
    float o[3];
    float d[3];
    float pdf;
    float c[3];
    float ior[4];
    float cone_width;
    float cone_spread;
    int xy;
    int depth;
};

struct ray_hash_t
{
    uint hash;
    uint index;
};

static const uint3 gl_WorkGroupSize = uint3(64u, 1u, 1u);

ByteAddressBuffer _23 : register(t3, space0);
RWByteAddressBuffer _51 : register(u0, space0);
ByteAddressBuffer _56 : register(t1, space0);
ByteAddressBuffer _61 : register(t2, space0);
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
        ray_data_t _68;
        [unroll]
        for (int _8ident = 0; _8ident < 3; _8ident++)
        {
            _68.o[_8ident] = asfloat(_56.Load(_8ident * 4 + _61.Load(_17 * 8 + 4) * 72 + 0));
        }
        [unroll]
        for (int _9ident = 0; _9ident < 3; _9ident++)
        {
            _68.d[_9ident] = asfloat(_56.Load(_9ident * 4 + _61.Load(_17 * 8 + 4) * 72 + 12));
        }
        _68.pdf = asfloat(_56.Load(_61.Load(_17 * 8 + 4) * 72 + 24));
        [unroll]
        for (int _10ident = 0; _10ident < 3; _10ident++)
        {
            _68.c[_10ident] = asfloat(_56.Load(_10ident * 4 + _61.Load(_17 * 8 + 4) * 72 + 28));
        }
        [unroll]
        for (int _11ident = 0; _11ident < 4; _11ident++)
        {
            _68.ior[_11ident] = asfloat(_56.Load(_11ident * 4 + _61.Load(_17 * 8 + 4) * 72 + 40));
        }
        _68.cone_width = asfloat(_56.Load(_61.Load(_17 * 8 + 4) * 72 + 56));
        _68.cone_spread = asfloat(_56.Load(_61.Load(_17 * 8 + 4) * 72 + 60));
        _68.xy = int(_56.Load(_61.Load(_17 * 8 + 4) * 72 + 64));
        _68.depth = int(_56.Load(_61.Load(_17 * 8 + 4) * 72 + 68));
        [unroll]
        for (int _12ident = 0; _12ident < 3; _12ident++)
        {
            _51.Store(_12ident * 4 + _17 * 72 + 0, asuint(_68.o[_12ident]));
        }
        [unroll]
        for (int _13ident = 0; _13ident < 3; _13ident++)
        {
            _51.Store(_13ident * 4 + _17 * 72 + 12, asuint(_68.d[_13ident]));
        }
        _51.Store(_17 * 72 + 24, asuint(_68.pdf));
        [unroll]
        for (int _14ident = 0; _14ident < 3; _14ident++)
        {
            _51.Store(_14ident * 4 + _17 * 72 + 28, asuint(_68.c[_14ident]));
        }
        [unroll]
        for (int _15ident = 0; _15ident < 4; _15ident++)
        {
            _51.Store(_15ident * 4 + _17 * 72 + 40, asuint(_68.ior[_15ident]));
        }
        _51.Store(_17 * 72 + 56, asuint(_68.cone_width));
        _51.Store(_17 * 72 + 60, asuint(_68.cone_spread));
        _51.Store(_17 * 72 + 64, uint(_68.xy));
        _51.Store(_17 * 72 + 68, uint(_68.depth));
        break;
    } while(false);
}

[numthreads(64, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
