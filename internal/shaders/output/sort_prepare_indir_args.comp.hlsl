struct Params
{
    int in_counter;
    int out_counter;
    int indir_args_index;
    int _pad0;
};

static const uint3 gl_WorkGroupSize = uint3(1u, 1u, 1u);

RWByteAddressBuffer _13 : register(u0, space0);
ByteAddressBuffer _33 : register(t3, space0);
ByteAddressBuffer _45 : register(t2, space0);
RWByteAddressBuffer _63 : register(u1, space0);
cbuffer UniformParams
{
    Params _19_g_params : packoffset(c0);
};


void comp_main()
{
    uint chunks_count = 0u;
    if (_13.Load(_19_g_params.in_counter * 4 + 0) != 0u)
    {
        chunks_count = _33.Load((_13.Load(_19_g_params.in_counter * 4 + 0) - 1u) * 4 + 0) + _45.Load((_13.Load(_19_g_params.in_counter * 4 + 0) - 1u) * 4 + 0);
    }
    uint _59 = (chunks_count + 63u) / 64u;
    int _69 = 3 * _19_g_params.indir_args_index;
    _63.Store(_69 * 4 + 0, _59);
    _63.Store((_69 + 1) * 4 + 0, 1u);
    _63.Store((_69 + 2) * 4 + 0, 1u);
    _13.Store(_19_g_params.out_counter * 4 + 0, chunks_count);
    uint chunks_scan_count = _59;
    int i = 0;
    for (; i < 3; )
    {
        _13.Store(((_19_g_params.out_counter + 1) + i) * 4 + 0, chunks_scan_count);
        int _113 = 3 * ((_19_g_params.indir_args_index + 1) + i);
        uint _117 = (chunks_scan_count + 63u) / 64u;
        _63.Store(_113 * 4 + 0, _117);
        _63.Store((_113 + 1) * 4 + 0, 1u);
        _63.Store((_113 + 2) * 4 + 0, 1u);
        chunks_scan_count = _117;
        i++;
        continue;
    }
    uint _143 = _59 * 16u;
    int _148 = 3 * (_19_g_params.indir_args_index + 4);
    uint _152 = (_143 + 63u) / 64u;
    _63.Store(_148 * 4 + 0, _152);
    _63.Store((_148 + 1) * 4 + 0, 1u);
    _63.Store((_148 + 2) * 4 + 0, 1u);
    _13.Store((_19_g_params.out_counter + 4) * 4 + 0, _143);
    uint scan_count = _152;
    int i_1 = 0;
    for (; i_1 < 4; )
    {
        _13.Store(((_19_g_params.out_counter + 5) + i_1) * 4 + 0, scan_count);
        int _196 = 3 * ((_19_g_params.indir_args_index + 5) + i_1);
        uint _200 = (scan_count + 63u) / 64u;
        _63.Store(_196 * 4 + 0, _200);
        _63.Store((_196 + 1) * 4 + 0, 1u);
        _63.Store((_196 + 2) * 4 + 0, 1u);
        scan_count = _200;
        i_1++;
        continue;
    }
}

[numthreads(1, 1, 1)]
void main()
{
    comp_main();
}
