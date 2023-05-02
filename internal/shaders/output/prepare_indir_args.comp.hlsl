static const uint3 gl_WorkGroupSize = uint3(1u, 1u, 1u);

RWByteAddressBuffer _12 : register(u0, space0);
RWByteAddressBuffer _21 : register(u1, space0);

void comp_main()
{
    uint _17 = _12.Load(0);
    uint _26 = (_17 + 63u) / 64u;
    _21.Store(0, _26);
    _21.Store(4, 1u);
    _21.Store(8, 1u);
    uint scan_count = _26;
    int i = 0;
    for (; i < 7; )
    {
        _12.Store((4 + i) * 4 + 0, scan_count);
        int _56 = 3 * i;
        uint _61 = (scan_count + 63u) / 64u;
        _21.Store((6 + _56) * 4 + 0, _61);
        _21.Store((_56 + 7) * 4 + 0, 1u);
        _21.Store((_56 + 8) * 4 + 0, 1u);
        scan_count = _61;
        i++;
        continue;
    }
    _12.Store(0, 0u);
    _12.Store(4, _17);
    uint _84 = _12.Load(8);
    _21.Store(12, (_84 + 63u) / 64u);
    _21.Store(16, 1u);
    _21.Store(20, 1u);
    _12.Store(8, 0u);
    _12.Store(12, _84);
}

[numthreads(1, 1, 1)]
void main()
{
    comp_main();
}
