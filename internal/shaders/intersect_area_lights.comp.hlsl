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
ByteAddressBuffer _760 : register(t3, space0);
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
    bool _767;
    do
    {
        float _26 = mad(b, b, -((4.0f * a) * c));
        if (_26 < 0.0f)
        {
            _767 = false;
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
        _767 = true;
        break;
    } while(false);
    return _767;
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
        int _786 = _195.mask;
        int _787 = _195.obj_index;
        float _789 = _195.t;
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
                float _298 = mad(_246.param2.x, _246.param2.x, mad(_284, _284, -dot(_280, _280)));
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
                        _324 = (_308 < _789) || _263;
                    }
                    else
                    {
                        _324 = _315;
                    }
                    if (_324)
                    {
                        _786 = -1;
                        _787 = (-1) - int(_229.Load(li * 4 + 0));
                        _789 = _308;
                    }
                    else
                    {
                        bool _338 = _312 > 9.9999999747524270787835121154785e-07f;
                        bool _347;
                        if (_338)
                        {
                            _347 = (_312 < _789) || _263;
                        }
                        else
                        {
                            _347 = _338;
                        }
                        if (_347)
                        {
                            _786 = -1;
                            _787 = (-1) - int(_229.Load(li * 4 + 0));
                            _789 = _312;
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
                    float3 _380 = normalize(cross(_246.param2.xyz, _246.param3.xyz));
                    float _388 = dot(_183, _380);
                    float _396 = (dot(_380, _246.param1.xyz) - dot(_380, _175)) / _388;
                    bool _401 = (_388 < 0.0f) && (_396 > 9.9999999747524270787835121154785e-07f);
                    bool _410;
                    if (_401)
                    {
                        _410 = (_396 < _789) || _263;
                    }
                    else
                    {
                        _410 = _401;
                    }
                    if (_410)
                    {
                        float3 _413 = light_u;
                        float3 _418 = _413 / dot(_413, _413).xxx;
                        light_u = _418;
                        light_v /= dot(light_v, light_v).xxx;
                        float3 _434 = (_175 + (_183 * _396)) - _246.param1.xyz;
                        float _438 = dot(_418, _434);
                        if ((_438 >= (-0.5f)) && (_438 <= 0.5f))
                        {
                            float _450 = dot(light_v, _434);
                            if ((_450 >= (-0.5f)) && (_450 <= 0.5f))
                            {
                                _786 = -1;
                                _787 = (-1) - int(_229.Load(li * 4 + 0));
                                _789 = _396;
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
                        float3 _488 = normalize(cross(_246.param2.xyz, _246.param3.xyz));
                        float _496 = dot(_183, _488);
                        float _504 = (dot(_488, _246.param1.xyz) - dot(_488, _175)) / _496;
                        bool _509 = (_496 < 0.0f) && (_504 > 9.9999999747524270787835121154785e-07f);
                        bool _518;
                        if (_509)
                        {
                            _518 = (_504 < _789) || _263;
                        }
                        else
                        {
                            _518 = _509;
                        }
                        if (_518)
                        {
                            float3 _521 = light_u_1;
                            float3 _526 = _521 / dot(_521, _521).xxx;
                            light_u_1 = _526;
                            float3 _527 = light_v_1;
                            float3 _532 = _527 / dot(_527, _527).xxx;
                            light_v_1 = _532;
                            float3 _542 = (_175 + (_183 * _504)) - _246.param1.xyz;
                            float _546 = dot(_526, _542);
                            float _550 = dot(_532, _542);
                            if (sqrt(mad(_546, _546, _550 * _550)) <= 0.5f)
                            {
                                _786 = -1;
                                _787 = (-1) - int(_229.Load(li * 4 + 0));
                                _789 = _504;
                            }
                        }
                    }
                    else
                    {
                        if (_268 == 3u)
                        {
                            float3 _590 = cross(_246.param2.xyz, _246.param3.xyz);
                            float3 _594 = _175 - _246.param1.xyz;
                            float _600 = dot(_594, _246.param2.xyz);
                            float _603 = dot(_594, _590);
                            float _611 = dot(_183, _246.param2.xyz);
                            float _614 = dot(_183, _590);
                            float param = mad(_614, _614, _611 * _611);
                            float param_1 = 2.0f * mad(_614, _603, _611 * _600);
                            float param_2 = mad(-_246.param2.w, _246.param2.w, mad(_603, _603, _600 * _600));
                            bool _672 = quadratic(param, param_1, param_2, param_3, param_4);
                            if ((_672 && (param_3 > 9.9999999747524270787835121154785e-07f)) && (param_4 > 9.9999999747524270787835121154785e-07f))
                            {
                                float _686 = min(param_3, param_4);
                                bool _699 = abs((float3(dot(_594, _246.param3.xyz), _600, _603) + (float3(dot(_183, _246.param3.xyz), _611, _614) * _686)).x) < (0.5f * _246.param3.w);
                                bool _708;
                                if (_699)
                                {
                                    _708 = (_686 < _789) || _263;
                                }
                                else
                                {
                                    _708 = _699;
                                }
                                if (_708)
                                {
                                    _786 = -1;
                                    _787 = (-1) - int(_229.Load(li * 4 + 0));
                                    _789 = _686;
                                }
                            }
                        }
                    }
                }
            }
        }
        _191.Store(_77 * 24 + 0, uint(_786));
        _191.Store(_77 * 24 + 4, uint(_787));
        _191.Store(_77 * 24 + 8, uint(_195.prim_index));
        _191.Store(_77 * 24 + 12, asuint(_789));
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
