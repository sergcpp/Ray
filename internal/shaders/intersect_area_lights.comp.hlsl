struct ray_data_t
{
    float o[3];
    float d[3];
    float pdf;
    float c[3];
    float cone_width;
    float cone_spread;
    int xy;
    int ray_depth;
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
ByteAddressBuffer _102 : register(t4, space0);
RWByteAddressBuffer _191 : register(u0, space0);
ByteAddressBuffer _229 : register(t2, space0);
ByteAddressBuffer _242 : register(t1, space0);
ByteAddressBuffer _794 : register(t3, space0);
cbuffer UniformParams
{
    Params _220_g_params : packoffset(c0);
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
    bool _801;
    do
    {
        float _26 = mad(b, b, -((4.0f * a) * c));
        if (_26 < 0.0f)
        {
            _801 = false;
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
        _801 = true;
        break;
    } while(false);
    return _801;
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
        ray_data_t _123;
        [unroll]
        for (int _3ident = 0; _3ident < 3; _3ident++)
        {
            _123.o[_3ident] = asfloat(_102.Load(_3ident * 4 + _77 * 56 + 0));
        }
        [unroll]
        for (int _4ident = 0; _4ident < 3; _4ident++)
        {
            _123.d[_4ident] = asfloat(_102.Load(_4ident * 4 + _77 * 56 + 12));
        }
        _123.pdf = asfloat(_102.Load(_77 * 56 + 24));
        [unroll]
        for (int _5ident = 0; _5ident < 3; _5ident++)
        {
            _123.c[_5ident] = asfloat(_102.Load(_5ident * 4 + _77 * 56 + 28));
        }
        _123.cone_width = asfloat(_102.Load(_77 * 56 + 40));
        _123.cone_spread = asfloat(_102.Load(_77 * 56 + 44));
        _123.xy = int(_102.Load(_77 * 56 + 48));
        _123.ray_depth = int(_102.Load(_77 * 56 + 52));
        if ((_123.ray_depth & 16777215) == 0)
        {
            break;
        }
        float3 _175 = float3(_123.o[0], _123.o[1], _123.o[2]);
        float3 _183 = float3(_123.d[0], _123.d[1], _123.d[2]);
        hit_data_t _195;
        _195.mask = int(_191.Load(_77 * 24 + 0));
        _195.obj_index = int(_191.Load(_77 * 24 + 4));
        _195.prim_index = int(_191.Load(_77 * 24 + 8));
        _195.t = asfloat(_191.Load(_77 * 24 + 12));
        _195.u = asfloat(_191.Load(_77 * 24 + 16));
        _195.v = asfloat(_191.Load(_77 * 24 + 20));
        int _820 = _195.mask;
        int _821 = _195.obj_index;
        float _823 = _195.t;
        float param_3;
        float param_4;
        for (uint li = 0u; li < _220_g_params.visible_lights_count; li++)
        {
            light_t _246;
            _246.type_and_param0 = _242.Load4(_229.Load(li * 4 + 0) * 64 + 0);
            _246.param1 = asfloat(_242.Load4(_229.Load(li * 4 + 0) * 64 + 16));
            _246.param2 = asfloat(_242.Load4(_229.Load(li * 4 + 0) * 64 + 32));
            _246.param3 = asfloat(_242.Load4(_229.Load(li * 4 + 0) * 64 + 48));
            bool _263 = (_246.type_and_param0.x & 32u) == 0u;
            uint _268 = _246.type_and_param0.x & 31u;
            if (_268 == 0u)
            {
                float3 _280 = _246.param1.xyz - _175;
                float _284 = dot(_280, _183);
                float _298 = mad(_246.param2.w, _246.param2.w, mad(_284, _284, -dot(_280, _280)));
                float det = _298;
                if (_298 >= 0.0f)
                {
                    float _303 = det;
                    float _304 = sqrt(_303);
                    det = _304;
                    float _308 = _284 - _304;
                    float _312 = _284 + _304;
                    bool _315 = _308 > 9.9999999747524270787835121154785e-07f;
                    bool _324;
                    if (_315)
                    {
                        _324 = (_308 < _823) || _263;
                    }
                    else
                    {
                        _324 = _315;
                    }
                    if (_324)
                    {
                        bool accept = true;
                        if (_246.param3.x > 0.0f)
                        {
                            float _339 = -dot(_183, _246.param2.xyz);
                            if (_339 > 0.0f)
                            {
                                bool _356;
                                if (accept)
                                {
                                    _356 = acos(clamp(_339, 0.0f, 1.0f)) <= _246.param3.x;
                                }
                                else
                                {
                                    _356 = accept;
                                }
                                accept = _356;
                            }
                            else
                            {
                                accept = false;
                            }
                        }
                        if (accept)
                        {
                            _820 = -1;
                            _821 = (-1) - int(_229.Load(li * 4 + 0));
                            _823 = _308;
                        }
                    }
                    else
                    {
                        bool _372 = _312 > 9.9999999747524270787835121154785e-07f;
                        bool _381;
                        if (_372)
                        {
                            _381 = (_312 < _823) || _263;
                        }
                        else
                        {
                            _381 = _372;
                        }
                        if (_381)
                        {
                            _820 = -1;
                            _821 = (-1) - int(_229.Load(li * 4 + 0));
                            _823 = _312;
                        }
                    }
                }
            }
            else
            {
                if (_268 == 4u)
                {
                    float3 light_u = _246.param2.xyz;
                    float3 light_v = _246.param3.xyz;
                    float3 _414 = normalize(cross(_246.param2.xyz, _246.param3.xyz));
                    float _422 = dot(_183, _414);
                    float _430 = (dot(_414, _246.param1.xyz) - dot(_414, _175)) / _422;
                    bool _435 = (_422 < 0.0f) && (_430 > 9.9999999747524270787835121154785e-07f);
                    bool _444;
                    if (_435)
                    {
                        _444 = (_430 < _823) || _263;
                    }
                    else
                    {
                        _444 = _435;
                    }
                    if (_444)
                    {
                        float3 _447 = light_u;
                        float3 _452 = _447 / dot(_447, _447).xxx;
                        light_u = _452;
                        light_v /= dot(light_v, light_v).xxx;
                        float3 _468 = (_175 + (_183 * _430)) - _246.param1.xyz;
                        float _472 = dot(_452, _468);
                        if ((_472 >= (-0.5f)) && (_472 <= 0.5f))
                        {
                            float _484 = dot(light_v, _468);
                            if ((_484 >= (-0.5f)) && (_484 <= 0.5f))
                            {
                                _820 = -1;
                                _821 = (-1) - int(_229.Load(li * 4 + 0));
                                _823 = _430;
                            }
                        }
                    }
                }
                else
                {
                    if (_268 == 5u)
                    {
                        float3 light_u_1 = _246.param2.xyz;
                        float3 light_v_1 = _246.param3.xyz;
                        float3 _522 = normalize(cross(_246.param2.xyz, _246.param3.xyz));
                        float _530 = dot(_183, _522);
                        float _538 = (dot(_522, _246.param1.xyz) - dot(_522, _175)) / _530;
                        bool _543 = (_530 < 0.0f) && (_538 > 9.9999999747524270787835121154785e-07f);
                        bool _552;
                        if (_543)
                        {
                            _552 = (_538 < _823) || _263;
                        }
                        else
                        {
                            _552 = _543;
                        }
                        if (_552)
                        {
                            float3 _555 = light_u_1;
                            float3 _560 = _555 / dot(_555, _555).xxx;
                            light_u_1 = _560;
                            float3 _561 = light_v_1;
                            float3 _566 = _561 / dot(_561, _561).xxx;
                            light_v_1 = _566;
                            float3 _576 = (_175 + (_183 * _538)) - _246.param1.xyz;
                            float _580 = dot(_560, _576);
                            float _584 = dot(_566, _576);
                            if (sqrt(mad(_580, _580, _584 * _584)) <= 0.5f)
                            {
                                _820 = -1;
                                _821 = (-1) - int(_229.Load(li * 4 + 0));
                                _823 = _538;
                            }
                        }
                    }
                    else
                    {
                        if (_268 == 3u)
                        {
                            float3 _624 = cross(_246.param2.xyz, _246.param3.xyz);
                            float3 _628 = _175 - _246.param1.xyz;
                            float _634 = dot(_628, _246.param2.xyz);
                            float _637 = dot(_628, _624);
                            float _645 = dot(_183, _246.param2.xyz);
                            float _648 = dot(_183, _624);
                            float param = mad(_648, _648, _645 * _645);
                            float param_1 = 2.0f * mad(_648, _637, _645 * _634);
                            float param_2 = mad(-_246.param2.w, _246.param2.w, mad(_637, _637, _634 * _634));
                            bool _706 = quadratic(param, param_1, param_2, param_3, param_4);
                            if ((_706 && (param_3 > 9.9999999747524270787835121154785e-07f)) && (param_4 > 9.9999999747524270787835121154785e-07f))
                            {
                                float _720 = min(param_3, param_4);
                                bool _733 = abs((float3(dot(_628, _246.param3.xyz), _634, _637) + (float3(dot(_183, _246.param3.xyz), _645, _648) * _720)).x) < (0.5f * _246.param3.w);
                                bool _742;
                                if (_733)
                                {
                                    _742 = (_720 < _823) || _263;
                                }
                                else
                                {
                                    _742 = _733;
                                }
                                if (_742)
                                {
                                    _820 = -1;
                                    _821 = (-1) - int(_229.Load(li * 4 + 0));
                                    _823 = _720;
                                }
                            }
                        }
                    }
                }
            }
        }
        _191.Store(_77 * 24 + 0, uint(_820));
        _191.Store(_77 * 24 + 4, uint(_821));
        _191.Store(_77 * 24 + 8, uint(_195.prim_index));
        _191.Store(_77 * 24 + 12, asuint(_823));
        _191.Store(_77 * 24 + 16, asuint(_195.u));
        _191.Store(_77 * 24 + 20, asuint(_195.v));
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
