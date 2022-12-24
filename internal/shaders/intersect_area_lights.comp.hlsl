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
ByteAddressBuffer _808 : register(t3, space0);
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
    bool _815;
    do
    {
        float _26 = mad(b, b, -((4.0f * a) * c));
        if (_26 < 0.0f)
        {
            _815 = false;
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
        _815 = true;
        break;
    } while(false);
    return _815;
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
        int _834 = _195.mask;
        int _835 = _195.obj_index;
        float _837 = _195.t;
        float param_3;
        float param_4;
        for (uint li = 0u; li < _220_g_params.visible_lights_count; li++)
        {
            light_t _246;
            _246.type_and_param0 = _242.Load4(_229.Load(li * 4 + 0) * 64 + 0);
            _246.param1 = asfloat(_242.Load4(_229.Load(li * 4 + 0) * 64 + 16));
            _246.param2 = asfloat(_242.Load4(_229.Load(li * 4 + 0) * 64 + 32));
            _246.param3 = asfloat(_242.Load4(_229.Load(li * 4 + 0) * 64 + 48));
            bool _259 = _834 != 0;
            bool _267;
            if (_259)
            {
                _267 = (_246.type_and_param0.x & 128u) != 0u;
            }
            else
            {
                _267 = _259;
            }
            [branch]
            if (_267)
            {
                continue;
            }
            bool _277 = (_246.type_and_param0.x & 32u) == 0u;
            uint _282 = _246.type_and_param0.x & 31u;
            if (_282 == 0u)
            {
                float3 _294 = _246.param1.xyz - _175;
                float _298 = dot(_294, _183);
                float _312 = mad(_246.param2.w, _246.param2.w, mad(_298, _298, -dot(_294, _294)));
                float det = _312;
                if (_312 >= 0.0f)
                {
                    float _317 = det;
                    float _318 = sqrt(_317);
                    det = _318;
                    float _322 = _298 - _318;
                    float _326 = _298 + _318;
                    bool _329 = _322 > 9.9999999747524270787835121154785e-07f;
                    bool _338;
                    if (_329)
                    {
                        _338 = (_322 < _837) || _277;
                    }
                    else
                    {
                        _338 = _329;
                    }
                    if (_338)
                    {
                        bool accept = true;
                        if (_246.param3.x > 0.0f)
                        {
                            float _353 = -dot(_183, _246.param2.xyz);
                            if (_353 > 0.0f)
                            {
                                bool _370;
                                if (accept)
                                {
                                    _370 = acos(clamp(_353, 0.0f, 1.0f)) <= _246.param3.x;
                                }
                                else
                                {
                                    _370 = accept;
                                }
                                accept = _370;
                            }
                            else
                            {
                                accept = false;
                            }
                        }
                        if (accept)
                        {
                            _834 = -1;
                            _835 = (-1) - int(_229.Load(li * 4 + 0));
                            _837 = _322;
                        }
                    }
                    else
                    {
                        bool _386 = _326 > 9.9999999747524270787835121154785e-07f;
                        bool _395;
                        if (_386)
                        {
                            _395 = (_326 < _837) || _277;
                        }
                        else
                        {
                            _395 = _386;
                        }
                        if (_395)
                        {
                            _834 = -1;
                            _835 = (-1) - int(_229.Load(li * 4 + 0));
                            _837 = _326;
                        }
                    }
                }
            }
            else
            {
                if (_282 == 4u)
                {
                    float3 light_u = _246.param2.xyz;
                    float3 light_v = _246.param3.xyz;
                    float3 _428 = normalize(cross(_246.param2.xyz, _246.param3.xyz));
                    float _436 = dot(_183, _428);
                    float _444 = (dot(_428, _246.param1.xyz) - dot(_428, _175)) / _436;
                    bool _449 = (_436 < 0.0f) && (_444 > 9.9999999747524270787835121154785e-07f);
                    bool _458;
                    if (_449)
                    {
                        _458 = (_444 < _837) || _277;
                    }
                    else
                    {
                        _458 = _449;
                    }
                    if (_458)
                    {
                        float3 _461 = light_u;
                        float3 _466 = _461 / dot(_461, _461).xxx;
                        light_u = _466;
                        light_v /= dot(light_v, light_v).xxx;
                        float3 _482 = (_175 + (_183 * _444)) - _246.param1.xyz;
                        float _486 = dot(_466, _482);
                        if ((_486 >= (-0.5f)) && (_486 <= 0.5f))
                        {
                            float _498 = dot(light_v, _482);
                            if ((_498 >= (-0.5f)) && (_498 <= 0.5f))
                            {
                                _834 = -1;
                                _835 = (-1) - int(_229.Load(li * 4 + 0));
                                _837 = _444;
                            }
                        }
                    }
                }
                else
                {
                    if (_282 == 5u)
                    {
                        float3 light_u_1 = _246.param2.xyz;
                        float3 light_v_1 = _246.param3.xyz;
                        float3 _536 = normalize(cross(_246.param2.xyz, _246.param3.xyz));
                        float _544 = dot(_183, _536);
                        float _552 = (dot(_536, _246.param1.xyz) - dot(_536, _175)) / _544;
                        bool _557 = (_544 < 0.0f) && (_552 > 9.9999999747524270787835121154785e-07f);
                        bool _566;
                        if (_557)
                        {
                            _566 = (_552 < _837) || _277;
                        }
                        else
                        {
                            _566 = _557;
                        }
                        if (_566)
                        {
                            float3 _569 = light_u_1;
                            float3 _574 = _569 / dot(_569, _569).xxx;
                            light_u_1 = _574;
                            float3 _575 = light_v_1;
                            float3 _580 = _575 / dot(_575, _575).xxx;
                            light_v_1 = _580;
                            float3 _590 = (_175 + (_183 * _552)) - _246.param1.xyz;
                            float _594 = dot(_574, _590);
                            float _598 = dot(_580, _590);
                            if (sqrt(mad(_594, _594, _598 * _598)) <= 0.5f)
                            {
                                _834 = -1;
                                _835 = (-1) - int(_229.Load(li * 4 + 0));
                                _837 = _552;
                            }
                        }
                    }
                    else
                    {
                        if (_282 == 3u)
                        {
                            float3 _638 = cross(_246.param2.xyz, _246.param3.xyz);
                            float3 _642 = _175 - _246.param1.xyz;
                            float _648 = dot(_642, _246.param2.xyz);
                            float _651 = dot(_642, _638);
                            float _659 = dot(_183, _246.param2.xyz);
                            float _662 = dot(_183, _638);
                            float param = mad(_662, _662, _659 * _659);
                            float param_1 = 2.0f * mad(_662, _651, _659 * _648);
                            float param_2 = mad(-_246.param2.w, _246.param2.w, mad(_651, _651, _648 * _648));
                            bool _720 = quadratic(param, param_1, param_2, param_3, param_4);
                            if ((_720 && (param_3 > 9.9999999747524270787835121154785e-07f)) && (param_4 > 9.9999999747524270787835121154785e-07f))
                            {
                                float _734 = min(param_3, param_4);
                                bool _747 = abs((float3(dot(_642, _246.param3.xyz), _648, _651) + (float3(dot(_183, _246.param3.xyz), _659, _662) * _734)).x) < (0.5f * _246.param3.w);
                                bool _756;
                                if (_747)
                                {
                                    _756 = (_734 < _837) || _277;
                                }
                                else
                                {
                                    _756 = _747;
                                }
                                if (_756)
                                {
                                    _834 = -1;
                                    _835 = (-1) - int(_229.Load(li * 4 + 0));
                                    _837 = _734;
                                }
                            }
                        }
                    }
                }
            }
        }
        _191.Store(_77 * 24 + 0, uint(_834));
        _191.Store(_77 * 24 + 4, uint(_835));
        _191.Store(_77 * 24 + 8, uint(_195.prim_index));
        _191.Store(_77 * 24 + 12, asuint(_837));
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
