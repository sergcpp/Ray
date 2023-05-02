struct Params
{
    int in_counter;
    int out_counter;
    int indir_args_index;
    int _pad0;
};

static const uint3 gl_WorkGroupSize = uint3(1u, 1u, 1u);

ByteAddressBuffer _12 : register(t3, space0);
RWByteAddressBuffer _18 : register(u0, space0);
ByteAddressBuffer _36 : register(t2, space0);
RWByteAddressBuffer _54 : register(u1, space0);
cbuffer UniformParams
{
    Params _22_g_params : packoffset(c0);
};


void comp_main()
{
    uint _32 = _12.Load((_18.Load(_22_g_params.in_counter * 4 + 0) - 1u) * 4 + 0);
    uint _43 = _36.Load((_18.Load(_22_g_params.in_counter * 4 + 0) - 1u) * 4 + 0);
    uint _44 = _32 + _43;
    uint _50 = (_44 + 63u) / 64u;
    int _60 = 3 * _22_g_params.indir_args_index;
    _54.Store(_60 * 4 + 0, _50);
    _54.Store((_60 + 1) * 4 + 0, 1u);
    _54.Store((_60 + 2) * 4 + 0, 1u);
    _18.Store(_22_g_params.out_counter * 4 + 0, _44);
    uint chunks_scan_count = _50;
    int i = 0;
    for (; i < 3; )
    {
        _18.Store(((_22_g_params.out_counter + 1) + i) * 4 + 0, chunks_scan_count);
        int _105 = 3 * ((_22_g_params.indir_args_index + 1) + i);
        uint _109 = (chunks_scan_count + 63u) / 64u;
        _54.Store(_105 * 4 + 0, _109);
        _54.Store((_105 + 1) * 4 + 0, 1u);
        _54.Store((_105 + 2) * 4 + 0, 1u);
        chunks_scan_count = _109;
        i++;
        continue;
    }
    uint _135 = _50 * 16u;
    int _140 = 3 * (_22_g_params.indir_args_index + 4);
    uint _144 = (_135 + 63u) / 64u;
    _54.Store(_140 * 4 + 0, _144);
    _54.Store((_140 + 1) * 4 + 0, 1u);
    _54.Store((_140 + 2) * 4 + 0, 1u);
    _18.Store((_22_g_params.out_counter + 4) * 4 + 0, _135);
    uint scan_count = _144;
    int i_1 = 0;
    for (; i_1 < 4; )
    {
        _18.Store(((_22_g_params.out_counter + 5) + i_1) * 4 + 0, scan_count);
        int _188 = 3 * ((_22_g_params.indir_args_index + 5) + i_1);
        uint _192 = (scan_count + 63u) / 64u;
        _54.Store(_188 * 4 + 0, _192);
        _54.Store((_188 + 1) * 4 + 0, 1u);
        _54.Store((_188 + 2) * 4 + 0, 1u);
        scan_count = _192;
        i_1++;
        continue;
    }
}

[numthreads(1, 1, 1)]
void main()
{
    comp_main();
}
