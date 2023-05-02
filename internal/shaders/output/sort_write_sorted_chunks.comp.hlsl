struct Params
{
    int shift;
    int counter;
    int chunks_counter;
    int _pad0;
};

struct ray_chunk_t
{
    uint hash;
    uint base;
    uint size;
};

static const uint3 gl_WorkGroupSize = uint3(64u, 1u, 1u);

ByteAddressBuffer _23 : register(t3, space0);
ByteAddressBuffer _58 : register(t2, space0);
ByteAddressBuffer _102 : register(t1, space0);
RWByteAddressBuffer _122 : register(u0, space0);
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
        uint _19 = uint(_17);
        if (_19 >= _23.Load(_28_g_params.counter * 4 + 0))
        {
            break;
        }
        uint local_offsets[16];
        for (int i = 0; i < 16; )
        {
            local_offsets[i] = _58.Load(((uint(i) * _23.Load(_28_g_params.counter * 4 + 0)) + _19) * 4 + 0);
            i++;
            continue;
        }
        for (int i_1 = 0; i_1 < 64; i_1++)
        {
            int _85 = _17 * 64;
            if (uint(_85 + i_1) < _23.Load(_28_g_params.chunks_counter * 4 + 0))
            {
                int _106 = _85 + i_1;
                uint _113 = (_102.Load(_106 * 12 + 0) >> uint(_28_g_params.shift)) & 15u;
                uint _117 = local_offsets[_113];
                local_offsets[_113] = _117 + uint(1);
                ray_chunk_t _130;
                _130.hash = _102.Load(_106 * 12 + 0);
                _130.base = _102.Load(_106 * 12 + 4);
                _130.size = _102.Load(_106 * 12 + 8);
                _122.Store(_117 * 12 + 0, _130.hash);
                _122.Store(_117 * 12 + 4, _130.base);
                _122.Store(_117 * 12 + 8, _130.size);
            }
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
