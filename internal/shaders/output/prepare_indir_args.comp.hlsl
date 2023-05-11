static const uint3 gl_WorkGroupSize = uint3(1u, 1u, 1u);

RWByteAddressBuffer _12 : register(u0, space0);
RWByteAddressBuffer _21 : register(u1, space0);

void comp_main()
{
    uint _17 = _12.Load(0);
    _21.Store(0, (_17 + 63u) / 64u);
    _21.Store(4, 1u);
    _21.Store(8, 1u);
    _21.Store(12, _17);
    _21.Store(16, 1u);
    _21.Store(20, 1u);
    uint _45 = (_17 + 255u) / 256u;
    _12.Store(16, _17);
    _21.Store(48, _45);
    _21.Store(52, 1u);
    _21.Store(56, 1u);
    _12.Store(20, _45);
    _21.Store(60, (_45 + 255u) / 256u);
    _21.Store(64, 1u);
    _21.Store(68, 1u);
    uint _69 = _45 * 16u;
    uint counters_count = _69;
    int i = 0;
    for (; i < 4; )
    {
        _12.Store((6 + i) * 4 + 0, counters_count);
        int _87 = 3 * i;
        uint _92 = (counters_count + 255u) / 256u;
        _21.Store((18 + _87) * 4 + 0, _92);
        _21.Store((_87 + 19) * 4 + 0, 1u);
        _21.Store((_87 + 20) * 4 + 0, 1u);
        counters_count = _92;
        i++;
        continue;
    }
    _12.Store(0, 0u);
    _12.Store(4, _17);
    uint _115 = _12.Load(8);
    _21.Store(24, (_115 + 63u) / 64u);
    _21.Store(28, 1u);
    _21.Store(32, 1u);
    _21.Store(36, _115);
    _21.Store(40, 1u);
    _21.Store(44, 1u);
    _12.Store(8, 0u);
    _12.Store(12, _115);
}

[numthreads(1, 1, 1)]
void main()
{
    comp_main();
}
