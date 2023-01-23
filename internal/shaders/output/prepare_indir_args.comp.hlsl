static const uint3 gl_WorkGroupSize = uint3(1u, 1u, 1u);

RWByteAddressBuffer _12 : register(u0, space0);
RWByteAddressBuffer _21 : register(u1, space0);

void comp_main()
{
    uint _17 = _12.Load(0);
    _21.Store(0, (_17 + 63u) / 64u);
    _21.Store(4, 1u);
    _21.Store(8, 1u);
    _12.Store(0, 0u);
    _12.Store(4, _17);
    uint _39 = _12.Load(8);
    _21.Store(12, (_39 + 63u) / 64u);
    _21.Store(16, 1u);
    _21.Store(20, 1u);
    _12.Store(8, 0u);
    _12.Store(12, _39);
}

[numthreads(1, 1, 1)]
void main()
{
    comp_main();
}
