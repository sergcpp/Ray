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

struct hit_data_t
{
    int mask;
    int obj_index;
    int prim_index;
    float t;
    float u;
    float v;
};

struct Params
{
    uint2 img_size;
    uint visible_lights_count;
};

struct light_t
{
    uint4 type_and_param0;
    float4 param1;
    float4 param2;
    float4 param3;
};

struct transform_t
{
    row_major float4x4 xform;
    row_major float4x4 inv_xform;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

ByteAddressBuffer _83 : register(t5, space0);
ByteAddressBuffer _104 : register(t4, space0);
RWByteAddressBuffer _197 : register(u0, space0);
ByteAddressBuffer _235 : register(t2, space0);
ByteAddressBuffer _248 : register(t1, space0);
ByteAddressBuffer _812 : register(t3, space0);
cbuffer UniformParams
{
    Params _226_g_params : packoffset(c0);
};


static uint3 gl_WorkGroupID;
static uint gl_LocalInvocationIndex;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint gl_LocalInvocationIndex : SV_GroupIndex;
};

bool quadratic(float a, float b, float c, inout float t0, inout float t1)
{
    bool _819;
    do
    {
        float _26 = mad(b, b, -((4.0f * a) * c));
        if (_26 < 0.0f)
        {
            _819 = false;
            break;
        }
        float _36 = sqrt(_26);
        float q;
        if (b < 0.0f)
        {
            q = (-0.5f) * (b - _36);
        }
        else
        {
            q = (-0.5f) * (b + _36);
        }
        t0 = q / a;
        t1 = c / q;
        _819 = true;
        break;
    } while(false);
    return _819;
}

void comp_main()
{
    do
    {
        int _77 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_77) >= _83.Load(4))
        {
            break;
        }
        ray_data_t _125;
        [unroll]
        for (int _4ident = 0; _4ident < 3; _4ident++)
        {
            _125.o[_4ident] = asfloat(_104.Load(_4ident * 4 + _77 * 72 + 0));
        }
        [unroll]
        for (int _5ident = 0; _5ident < 3; _5ident++)
        {
            _125.d[_5ident] = asfloat(_104.Load(_5ident * 4 + _77 * 72 + 12));
        }
        _125.pdf = asfloat(_104.Load(_77 * 72 + 24));
        [unroll]
        for (int _6ident = 0; _6ident < 3; _6ident++)
        {
            _125.c[_6ident] = asfloat(_104.Load(_6ident * 4 + _77 * 72 + 28));
        }
        [unroll]
        for (int _7ident = 0; _7ident < 4; _7ident++)
        {
            _125.ior[_7ident] = asfloat(_104.Load(_7ident * 4 + _77 * 72 + 40));
        }
        _125.cone_width = asfloat(_104.Load(_77 * 72 + 56));
        _125.cone_spread = asfloat(_104.Load(_77 * 72 + 60));
        _125.xy = int(_104.Load(_77 * 72 + 64));
        _125.depth = int(_104.Load(_77 * 72 + 68));
        if ((_125.depth & 16777215) == 0)
        {
            break;
        }
        float3 _181 = float3(_125.o[0], _125.o[1], _125.o[2]);
        float3 _189 = float3(_125.d[0], _125.d[1], _125.d[2]);
        hit_data_t _201;
        _201.mask = int(_197.Load(_77 * 24 + 0));
        _201.obj_index = int(_197.Load(_77 * 24 + 4));
        _201.prim_index = int(_197.Load(_77 * 24 + 8));
        _201.t = asfloat(_197.Load(_77 * 24 + 12));
        _201.u = asfloat(_197.Load(_77 * 24 + 16));
        _201.v = asfloat(_197.Load(_77 * 24 + 20));
        int _840 = _201.mask;
        int _841 = _201.obj_index;
        float _843 = _201.t;
        float param_3;
        float param_4;
        for (uint li = 0u; li < _226_g_params.visible_lights_count; li++)
        {
            light_t _252;
            _252.type_and_param0 = _248.Load4(_235.Load(li * 4 + 0) * 64 + 0);
            _252.param1 = asfloat(_248.Load4(_235.Load(li * 4 + 0) * 64 + 16));
            _252.param2 = asfloat(_248.Load4(_235.Load(li * 4 + 0) * 64 + 32));
            _252.param3 = asfloat(_248.Load4(_235.Load(li * 4 + 0) * 64 + 48));
            bool _265 = _840 != 0;
            bool _273;
            if (_265)
            {
                _273 = (_252.type_and_param0.x & 128u) != 0u;
            }
            else
            {
                _273 = _265;
            }
            [branch]
            if (_273)
            {
                continue;
            }
            bool _283 = (_252.type_and_param0.x & 32u) == 0u;
            uint _288 = _252.type_and_param0.x & 31u;
            if (_288 == 0u)
            {
                float3 _300 = _252.param1.xyz - _181;
                float _304 = dot(_300, _189);
                float _318 = mad(_252.param2.w, _252.param2.w, mad(_304, _304, -dot(_300, _300)));
                float det = _318;
                if (_318 >= 0.0f)
                {
                    float _323 = det;
                    float _324 = sqrt(_323);
                    det = _324;
                    float _328 = _304 - _324;
                    float _332 = _304 + _324;
                    bool _335 = _328 > 9.9999999747524270787835121154785e-07f;
                    bool _344;
                    if (_335)
                    {
                        _344 = (_328 < _843) || _283;
                    }
                    else
                    {
                        _344 = _335;
                    }
                    if (_344)
                    {
                        bool accept = true;
                        if (_252.param3.x > 0.0f)
                        {
                            float _359 = -dot(_189, _252.param2.xyz);
                            if (_359 > 0.0f)
                            {
                                bool _376;
                                if (accept)
                                {
                                    _376 = acos(clamp(_359, 0.0f, 1.0f)) <= _252.param3.x;
                                }
                                else
                                {
                                    _376 = accept;
                                }
                                accept = _376;
                            }
                            else
                            {
                                accept = false;
                            }
                        }
                        if (accept)
                        {
                            _840 = -1;
                            _841 = (-1) - int(_235.Load(li * 4 + 0));
                            _843 = _328;
                        }
                    }
                    else
                    {
                        bool _392 = _332 > 9.9999999747524270787835121154785e-07f;
                        bool _401;
                        if (_392)
                        {
                            _401 = (_332 < _843) || _283;
                        }
                        else
                        {
                            _401 = _392;
                        }
                        if (_401)
                        {
                            _840 = -1;
                            _841 = (-1) - int(_235.Load(li * 4 + 0));
                            _843 = _332;
                        }
                    }
                }
            }
            else
            {
                if (_288 == 4u)
                {
                    float3 light_u = _252.param2.xyz;
                    float3 light_v = _252.param3.xyz;
                    float3 _433 = normalize(cross(_252.param2.xyz, _252.param3.xyz));
                    float _441 = dot(_189, _433);
                    float _449 = (dot(_433, _252.param1.xyz) - dot(_433, _181)) / _441;
                    bool _454 = (_441 < 0.0f) && (_449 > 9.9999999747524270787835121154785e-07f);
                    bool _463;
                    if (_454)
                    {
                        _463 = (_449 < _843) || _283;
                    }
                    else
                    {
                        _463 = _454;
                    }
                    if (_463)
                    {
                        float3 _466 = light_u;
                        float3 _471 = _466 / dot(_466, _466).xxx;
                        light_u = _471;
                        light_v /= dot(light_v, light_v).xxx;
                        float3 _487 = (_181 + (_189 * _449)) - _252.param1.xyz;
                        float _491 = dot(_471, _487);
                        if ((_491 >= (-0.5f)) && (_491 <= 0.5f))
                        {
                            float _503 = dot(light_v, _487);
                            if ((_503 >= (-0.5f)) && (_503 <= 0.5f))
                            {
                                _840 = -1;
                                _841 = (-1) - int(_235.Load(li * 4 + 0));
                                _843 = _449;
                            }
                        }
                    }
                }
                else
                {
                    if (_288 == 5u)
                    {
                        float3 light_u_1 = _252.param2.xyz;
                        float3 light_v_1 = _252.param3.xyz;
                        float3 _541 = normalize(cross(_252.param2.xyz, _252.param3.xyz));
                        float _549 = dot(_189, _541);
                        float _557 = (dot(_541, _252.param1.xyz) - dot(_541, _181)) / _549;
                        bool _562 = (_549 < 0.0f) && (_557 > 9.9999999747524270787835121154785e-07f);
                        bool _571;
                        if (_562)
                        {
                            _571 = (_557 < _843) || _283;
                        }
                        else
                        {
                            _571 = _562;
                        }
                        if (_571)
                        {
                            float3 _574 = light_u_1;
                            float3 _579 = _574 / dot(_574, _574).xxx;
                            light_u_1 = _579;
                            float3 _580 = light_v_1;
                            float3 _585 = _580 / dot(_580, _580).xxx;
                            light_v_1 = _585;
                            float3 _595 = (_181 + (_189 * _557)) - _252.param1.xyz;
                            float _599 = dot(_579, _595);
                            float _603 = dot(_585, _595);
                            if (sqrt(mad(_599, _599, _603 * _603)) <= 0.5f)
                            {
                                _840 = -1;
                                _841 = (-1) - int(_235.Load(li * 4 + 0));
                                _843 = _557;
                            }
                        }
                    }
                    else
                    {
                        if (_288 == 3u)
                        {
                            float3 _643 = cross(_252.param2.xyz, _252.param3.xyz);
                            float3 _647 = _181 - _252.param1.xyz;
                            float _653 = dot(_647, _252.param2.xyz);
                            float _656 = dot(_647, _643);
                            float _664 = dot(_189, _252.param2.xyz);
                            float _667 = dot(_189, _643);
                            float param = mad(_667, _667, _664 * _664);
                            float param_1 = 2.0f * mad(_667, _656, _664 * _653);
                            float param_2 = mad(-_252.param2.w, _252.param2.w, mad(_656, _656, _653 * _653));
                            bool _725 = quadratic(param, param_1, param_2, param_3, param_4);
                            if ((_725 && (param_3 > 9.9999999747524270787835121154785e-07f)) && (param_4 > 9.9999999747524270787835121154785e-07f))
                            {
                                float _739 = min(param_3, param_4);
                                bool _752 = abs((float3(dot(_647, _252.param3.xyz), _653, _656) + (float3(dot(_189, _252.param3.xyz), _664, _667) * _739)).x) < (0.5f * _252.param3.w);
                                bool _761;
                                if (_752)
                                {
                                    _761 = (_739 < _843) || _283;
                                }
                                else
                                {
                                    _761 = _752;
                                }
                                if (_761)
                                {
                                    _840 = -1;
                                    _841 = (-1) - int(_235.Load(li * 4 + 0));
                                    _843 = _739;
                                }
                            }
                        }
                    }
                }
            }
        }
        _197.Store(_77 * 24 + 0, uint(_840));
        _197.Store(_77 * 24 + 4, uint(_841));
        _197.Store(_77 * 24 + 8, uint(_201.prim_index));
        _197.Store(_77 * 24 + 12, asuint(_843));
        _197.Store(_77 * 24 + 16, asuint(_201.u));
        _197.Store(_77 * 24 + 20, asuint(_201.v));
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationIndex = stage_input.gl_LocalInvocationIndex;
    comp_main();
}
