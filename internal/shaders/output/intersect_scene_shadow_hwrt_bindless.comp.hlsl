struct shadow_ray_t
{
    float o[3];
    int depth;
    float d[3];
    float dist;
    float c[3];
    int xy;
};

struct Params
{
    uint node_index;
    int max_transp_depth;
    int blocker_lights_count;
    float clamp_val;
    int _pad0;
};

struct vertex_t
{
    float p[3];
    float n[3];
    float b[3];
    float t[2][2];
};

struct material_t
{
    uint textures[5];
    float base_color[3];
    uint flags;
    uint type;
    float tangent_rotation_or_strength;
    uint roughness_and_anisotropic;
    float ior;
    uint sheen_and_sheen_tint;
    uint tint_and_metallic;
    uint transmission_and_transmission_roughness;
    uint specular_and_specular_tint;
    uint clearcoat_and_clearcoat_roughness;
    uint normal_map_strength_unorm;
};

struct light_t
{
    uint4 type_and_param0;
    float4 param1;
    float4 param2;
    float4 param3;
};

struct tri_accel_t
{
    float4 n_plane;
    float4 u_plane;
    float4 v_plane;
};

struct bvh_node_t
{
    float4 bbox_min;
    float4 bbox_max;
};

struct mesh_t
{
    float bbox_min[3];
    float bbox_max[3];
    uint node_index;
    uint node_count;
    uint tris_index;
    uint tris_count;
    uint vert_index;
    uint vert_count;
};

struct mesh_instance_t
{
    float4 bbox_min;
    float4 bbox_max;
};

struct transform_t
{
    row_major float4x4 xform;
    row_major float4x4 inv_xform;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

ByteAddressBuffer _305 : register(t3, space0);
ByteAddressBuffer _383 : register(t10, space0);
ByteAddressBuffer _387 : register(t11, space0);
ByteAddressBuffer _487 : register(t4, space0);
ByteAddressBuffer _643 : register(t15, space0);
ByteAddressBuffer _655 : register(t14, space0);
ByteAddressBuffer _886 : register(t13, space0);
ByteAddressBuffer _901 : register(t12, space0);
ByteAddressBuffer _989 : register(t1, space0);
ByteAddressBuffer _993 : register(t2, space0);
ByteAddressBuffer _998 : register(t5, space0);
ByteAddressBuffer _1005 : register(t6, space0);
ByteAddressBuffer _1010 : register(t7, space0);
ByteAddressBuffer _1014 : register(t8, space0);
ByteAddressBuffer _1020 : register(t9, space0);
cbuffer UniformParams
{
    Params _352_g_params : packoffset(c0);
};

Texture2D<float4> g_textures[] : register(t0, space1);
SamplerState _g_textures_sampler[] : register(s0, space1);
uniform ??? g_tlas;
RWTexture2D<float4> g_inout_img : register(u0, space0);

static uint3 gl_WorkGroupID;
static uint gl_LocalInvocationIndex;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint gl_LocalInvocationIndex : SV_GroupIndex;
};

static ??? rq;
groupshared uint g_stack[64][48];

float3 YCoCg_to_RGB(float4 col)
{
    float _120 = mad(col.z, 31.875f, 1.0f);
    float _130 = (col.x - 0.501960813999176025390625f) / _120;
    float _136 = (col.y - 0.501960813999176025390625f) / _120;
    return float3((col.w + _130) - _136, col.w + _136, (col.w - _130) - _136);
}

float3 srgb_to_rgb(float3 col)
{
    float3 ret;
    [unroll]
    for (int i = 0; i < 3; i++)
    {
        [flatten]
        if (col[i] > 0.040449999272823333740234375f)
        {
            ret[i] = pow((col[i] + 0.054999999701976776123046875f) * 0.947867333889007568359375f, 2.400000095367431640625f);
        }
        else
        {
            ret[i] = col[i] * 0.077399380505084991455078125f;
        }
    }
    return ret;
}

float4 SampleBilinear(uint index, float2 uvs, int lod, bool maybe_YCoCg, bool maybe_SRGB)
{
    uint _165 = index & 16777215u;
    float4 res = g_textures[NonUniformResourceIndex(_165)].SampleLevel(_g_textures_sampler[NonUniformResourceIndex(_165)], uvs, float(lod));
    bool _176;
    if (maybe_YCoCg)
    {
        _176 = (index & 67108864u) != 0u;
    }
    else
    {
        _176 = maybe_YCoCg;
    }
    if (_176)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _194;
    if (maybe_SRGB)
    {
        _194 = (index & 16777216u) != 0u;
    }
    else
    {
        _194 = maybe_SRGB;
    }
    if (_194)
    {
        float3 param_1 = res.xyz;
        float3 _200 = srgb_to_rgb(param_1);
        float4 _1146 = res;
        _1146.x = _200.x;
        float4 _1148 = _1146;
        _1148.y = _200.y;
        float4 _1150 = _1148;
        _1150.z = _200.z;
        res = _1150;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod)
{
    return SampleBilinear(index, uvs, lod, false, false);
}

float lum(float3 color)
{
    return mad(0.072168998420238494873046875f, color.z, mad(0.21267099678516387939453125f, color.x, 0.71516001224517822265625f * color.y));
}

float3 IntersectSceneShadow(shadow_ray_t r)
{
    bool _1030 = false;
    float3 _1027;
    do
    {
        float3 ro = float3(r.o[0], r.o[1], r.o[2]);
        float3 _230 = float3(r.d[0], r.d[1], r.d[2]);
        float3 rc = float3(r.c[0], r.c[1], r.c[2]);
        int depth = r.depth >> 24;
        float _250;
        if (r.dist > 0.0f)
        {
            _250 = r.dist;
        }
        else
        {
            _250 = 3402823346297367662189621542912.0f;
        }
        float dist = _250;
        while (dist > 9.9999997473787516355514526367188e-06f)
        {
            rayQueryInitializeEXT(rq, g_tlas, 0u, 255u, ro, 0.0f, _230, dist);
            for (;;)
            {
                bool _282 = rayQueryProceedEXT(rq);
                if (_282)
                {
                    uint _283 = rayQueryGetIntersectionTypeEXT(rq, bool(0));
                    if (_283 == 0u)
                    {
                        rayQueryConfirmIntersectionEXT(rq);
                    }
                    continue;
                }
                else
                {
                    break;
                }
            }
            uint _288 = rayQueryGetIntersectionTypeEXT(rq, bool(1));
            if (_288 != 0u)
            {
                int _293 = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, bool(1));
                int _295 = rayQueryGetIntersectionInstanceIdEXT(rq, bool(1));
                int _298 = rayQueryGetIntersectionPrimitiveIndexEXT(rq, bool(1));
                int _299 = _293 + _298;
                uint _313 = (_305.Load(_299 * 4 + 0) >> 16u) & 65535u;
                uint _318 = _305.Load(_299 * 4 + 0) & 65535u;
                bool _321 = rayQueryGetIntersectionFrontFaceEXT(rq, bool(1));
                bool _322 = !_321;
                bool _325 = !_322;
                bool _332;
                if (_325)
                {
                    _332 = (_313 & 32768u) != 0u;
                }
                else
                {
                    _332 = _325;
                }
                bool _343;
                if (!_332)
                {
                    bool _342;
                    if (_322)
                    {
                        _342 = (_318 & 32768u) != 0u;
                    }
                    else
                    {
                        _342 = _322;
                    }
                    _343 = _342;
                }
                else
                {
                    _343 = _332;
                }
                bool _357;
                if (!_343)
                {
                    _357 = depth > _352_g_params.max_transp_depth;
                }
                else
                {
                    _357 = _343;
                }
                if (_357)
                {
                    _1030 = true;
                    _1027 = 0.0f.xxx;
                    break;
                }
                int _389 = _299 * 3;
                vertex_t _395;
                [unroll]
                for (int _20ident = 0; _20ident < 3; _20ident++)
                {
                    _395.p[_20ident] = asfloat(_383.Load(_20ident * 4 + _387.Load(_389 * 4 + 0) * 52 + 0));
                }
                [unroll]
                for (int _21ident = 0; _21ident < 3; _21ident++)
                {
                    _395.n[_21ident] = asfloat(_383.Load(_21ident * 4 + _387.Load(_389 * 4 + 0) * 52 + 12));
                }
                [unroll]
                for (int _22ident = 0; _22ident < 3; _22ident++)
                {
                    _395.b[_22ident] = asfloat(_383.Load(_22ident * 4 + _387.Load(_389 * 4 + 0) * 52 + 24));
                }
                [unroll]
                for (int _23ident = 0; _23ident < 2; _23ident++)
                {
                    [unroll]
                    for (int _24ident = 0; _24ident < 2; _24ident++)
                    {
                        _395.t[_23ident][_24ident] = asfloat(_383.Load(_24ident * 4 + _23ident * 8 + _387.Load(_389 * 4 + 0) * 52 + 36));
                    }
                }
                vertex_t _396;
                _396.p[0] = _395.p[0];
                _396.p[1] = _395.p[1];
                _396.p[2] = _395.p[2];
                _396.n[0] = _395.n[0];
                _396.n[1] = _395.n[1];
                _396.n[2] = _395.n[2];
                _396.b[0] = _395.b[0];
                _396.b[1] = _395.b[1];
                _396.b[2] = _395.b[2];
                _396.t[0][0] = _395.t[0][0];
                _396.t[0][1] = _395.t[0][1];
                _396.t[1][0] = _395.t[1][0];
                _396.t[1][1] = _395.t[1][1];
                vertex_t _404;
                [unroll]
                for (int _25ident = 0; _25ident < 3; _25ident++)
                {
                    _404.p[_25ident] = asfloat(_383.Load(_25ident * 4 + _387.Load((_389 + 1) * 4 + 0) * 52 + 0));
                }
                [unroll]
                for (int _26ident = 0; _26ident < 3; _26ident++)
                {
                    _404.n[_26ident] = asfloat(_383.Load(_26ident * 4 + _387.Load((_389 + 1) * 4 + 0) * 52 + 12));
                }
                [unroll]
                for (int _27ident = 0; _27ident < 3; _27ident++)
                {
                    _404.b[_27ident] = asfloat(_383.Load(_27ident * 4 + _387.Load((_389 + 1) * 4 + 0) * 52 + 24));
                }
                [unroll]
                for (int _28ident = 0; _28ident < 2; _28ident++)
                {
                    [unroll]
                    for (int _29ident = 0; _29ident < 2; _29ident++)
                    {
                        _404.t[_28ident][_29ident] = asfloat(_383.Load(_29ident * 4 + _28ident * 8 + _387.Load((_389 + 1) * 4 + 0) * 52 + 36));
                    }
                }
                vertex_t _405;
                _405.p[0] = _404.p[0];
                _405.p[1] = _404.p[1];
                _405.p[2] = _404.p[2];
                _405.n[0] = _404.n[0];
                _405.n[1] = _404.n[1];
                _405.n[2] = _404.n[2];
                _405.b[0] = _404.b[0];
                _405.b[1] = _404.b[1];
                _405.b[2] = _404.b[2];
                _405.t[0][0] = _404.t[0][0];
                _405.t[0][1] = _404.t[0][1];
                _405.t[1][0] = _404.t[1][0];
                _405.t[1][1] = _404.t[1][1];
                vertex_t _413;
                [unroll]
                for (int _30ident = 0; _30ident < 3; _30ident++)
                {
                    _413.p[_30ident] = asfloat(_383.Load(_30ident * 4 + _387.Load((_389 + 2) * 4 + 0) * 52 + 0));
                }
                [unroll]
                for (int _31ident = 0; _31ident < 3; _31ident++)
                {
                    _413.n[_31ident] = asfloat(_383.Load(_31ident * 4 + _387.Load((_389 + 2) * 4 + 0) * 52 + 12));
                }
                [unroll]
                for (int _32ident = 0; _32ident < 3; _32ident++)
                {
                    _413.b[_32ident] = asfloat(_383.Load(_32ident * 4 + _387.Load((_389 + 2) * 4 + 0) * 52 + 24));
                }
                [unroll]
                for (int _33ident = 0; _33ident < 2; _33ident++)
                {
                    [unroll]
                    for (int _34ident = 0; _34ident < 2; _34ident++)
                    {
                        _413.t[_33ident][_34ident] = asfloat(_383.Load(_34ident * 4 + _33ident * 8 + _387.Load((_389 + 2) * 4 + 0) * 52 + 36));
                    }
                }
                vertex_t _414;
                _414.p[0] = _413.p[0];
                _414.p[1] = _413.p[1];
                _414.p[2] = _413.p[2];
                _414.n[0] = _413.n[0];
                _414.n[1] = _413.n[1];
                _414.n[2] = _413.n[2];
                _414.b[0] = _413.b[0];
                _414.b[1] = _413.b[1];
                _414.b[2] = _413.b[2];
                _414.t[0][0] = _413.t[0][0];
                _414.t[0][1] = _413.t[0][1];
                _414.t[1][0] = _413.t[1][0];
                _414.t[1][1] = _413.t[1][1];
                float2 _417 = rayQueryGetIntersectionBarycentricsEXT(rq, bool(1));
                float2 _450 = ((float2(_396.t[0][0], _396.t[0][1]) * ((1.0f - _417.x) - _417.y)) + (float2(_405.t[0][0], _405.t[0][1]) * _417.x)) + (float2(_414.t[0][0], _414.t[0][1]) * _417.y);
                g_stack[gl_LocalInvocationIndex][0] = (_322 ? _318 : _313) & 16383u;
                g_stack[gl_LocalInvocationIndex][1] = 1065353216u;
                int stack_size = 1;
                float3 throughput = 0.0f.xxx;
                for (;;)
                {
                    int _473 = stack_size;
                    stack_size = _473 - 1;
                    if (_473 != 0)
                    {
                        int _489 = stack_size;
                        int _490 = 2 * _489;
                        material_t _496;
                        [unroll]
                        for (int _35ident = 0; _35ident < 5; _35ident++)
                        {
                            _496.textures[_35ident] = _487.Load(_35ident * 4 + g_stack[gl_LocalInvocationIndex][_490] * 76 + 0);
                        }
                        [unroll]
                        for (int _36ident = 0; _36ident < 3; _36ident++)
                        {
                            _496.base_color[_36ident] = asfloat(_487.Load(_36ident * 4 + g_stack[gl_LocalInvocationIndex][_490] * 76 + 20));
                        }
                        _496.flags = _487.Load(g_stack[gl_LocalInvocationIndex][_490] * 76 + 32);
                        _496.type = _487.Load(g_stack[gl_LocalInvocationIndex][_490] * 76 + 36);
                        _496.tangent_rotation_or_strength = asfloat(_487.Load(g_stack[gl_LocalInvocationIndex][_490] * 76 + 40));
                        _496.roughness_and_anisotropic = _487.Load(g_stack[gl_LocalInvocationIndex][_490] * 76 + 44);
                        _496.ior = asfloat(_487.Load(g_stack[gl_LocalInvocationIndex][_490] * 76 + 48));
                        _496.sheen_and_sheen_tint = _487.Load(g_stack[gl_LocalInvocationIndex][_490] * 76 + 52);
                        _496.tint_and_metallic = _487.Load(g_stack[gl_LocalInvocationIndex][_490] * 76 + 56);
                        _496.transmission_and_transmission_roughness = _487.Load(g_stack[gl_LocalInvocationIndex][_490] * 76 + 60);
                        _496.specular_and_specular_tint = _487.Load(g_stack[gl_LocalInvocationIndex][_490] * 76 + 64);
                        _496.clearcoat_and_clearcoat_roughness = _487.Load(g_stack[gl_LocalInvocationIndex][_490] * 76 + 68);
                        _496.normal_map_strength_unorm = _487.Load(g_stack[gl_LocalInvocationIndex][_490] * 76 + 72);
                        material_t _497;
                        _497.textures[0] = _496.textures[0];
                        _497.textures[1] = _496.textures[1];
                        _497.textures[2] = _496.textures[2];
                        _497.textures[3] = _496.textures[3];
                        _497.textures[4] = _496.textures[4];
                        _497.base_color[0] = _496.base_color[0];
                        _497.base_color[1] = _496.base_color[1];
                        _497.base_color[2] = _496.base_color[2];
                        _497.flags = _496.flags;
                        _497.type = _496.type;
                        _497.tangent_rotation_or_strength = _496.tangent_rotation_or_strength;
                        _497.roughness_and_anisotropic = _496.roughness_and_anisotropic;
                        _497.ior = _496.ior;
                        _497.sheen_and_sheen_tint = _496.sheen_and_sheen_tint;
                        _497.tint_and_metallic = _496.tint_and_metallic;
                        _497.transmission_and_transmission_roughness = _496.transmission_and_transmission_roughness;
                        _497.specular_and_specular_tint = _496.specular_and_specular_tint;
                        _497.clearcoat_and_clearcoat_roughness = _496.clearcoat_and_clearcoat_roughness;
                        _497.normal_map_strength_unorm = _496.normal_map_strength_unorm;
                        uint _504 = g_stack[gl_LocalInvocationIndex][_490 + 1];
                        float _505 = asfloat(_504);
                        if (_497.type == 4u)
                        {
                            float mix_val = _497.tangent_rotation_or_strength;
                            if (_497.textures[1] != 4294967295u)
                            {
                                mix_val *= SampleBilinear(_497.textures[1], _450, 0).x;
                            }
                            int _530 = 2 * stack_size;
                            g_stack[gl_LocalInvocationIndex][_530] = _497.textures[3];
                            g_stack[gl_LocalInvocationIndex][_530 + 1] = asuint(_505 * (1.0f - mix_val));
                            int _549 = 2 * (stack_size + 1);
                            g_stack[gl_LocalInvocationIndex][_549] = _497.textures[4];
                            g_stack[gl_LocalInvocationIndex][_549 + 1] = asuint(_505 * mix_val);
                            stack_size += 2;
                        }
                        else
                        {
                            if (_497.type == 5u)
                            {
                                throughput += (float3(_497.base_color[0], _497.base_color[1], _497.base_color[2]) * _505);
                            }
                        }
                        continue;
                    }
                    else
                    {
                        break;
                    }
                }
                float3 _583 = rc;
                float3 _584 = _583 * throughput;
                rc = _584;
                if (lum(_584) < 1.0000000116860974230803549289703e-07f)
                {
                    break;
                }
            }
            float _593 = rayQueryGetIntersectionTEXT(rq, bool(1));
            float _594 = _593 + 9.9999997473787516355514526367188e-06f;
            ro += (_230 * _594);
            dist -= _594;
            depth++;
        }
        if (_1030)
        {
            break;
        }
        _1030 = true;
        _1027 = rc;
        break;
    } while(false);
    return _1027;
}

float IntersectAreaLightsShadow(shadow_ray_t r)
{
    bool _1037 = false;
    float _1034;
    do
    {
        float3 _615 = float3(r.o[0], r.o[1], r.o[2]);
        float3 _623 = float3(r.d[0], r.d[1], r.d[2]);
        float _627 = abs(r.dist);
        for (uint li = 0u; li < uint(_352_g_params.blocker_lights_count); li++)
        {
            light_t _659;
            _659.type_and_param0 = _655.Load4(_643.Load(li * 4 + 0) * 64 + 0);
            _659.param1 = asfloat(_655.Load4(_643.Load(li * 4 + 0) * 64 + 16));
            _659.param2 = asfloat(_655.Load4(_643.Load(li * 4 + 0) * 64 + 32));
            _659.param3 = asfloat(_655.Load4(_643.Load(li * 4 + 0) * 64 + 48));
            light_t _660;
            _660.type_and_param0 = _659.type_and_param0;
            _660.param1 = _659.param1;
            _660.param2 = _659.param2;
            _660.param3 = _659.param3;
            bool _665 = (_660.type_and_param0.x & 128u) != 0u;
            bool _671;
            if (_665)
            {
                _671 = r.dist >= 0.0f;
            }
            else
            {
                _671 = _665;
            }
            [branch]
            if (_671)
            {
                continue;
            }
            uint _679 = _660.type_and_param0.x & 31u;
            if (_679 == 4u)
            {
                float3 light_u = _660.param2.xyz;
                float3 light_v = _660.param3.xyz;
                float3 _700 = normalize(cross(_660.param2.xyz, _660.param3.xyz));
                float _708 = dot(_623, _700);
                float _716 = (dot(_700, _660.param1.xyz) - dot(_700, _615)) / _708;
                if (((_708 < 0.0f) && (_716 > 9.9999999747524270787835121154785e-07f)) && (_716 < _627))
                {
                    float3 _729 = light_u;
                    float3 _734 = _729 / dot(_729, _729).xxx;
                    light_u = _734;
                    light_v /= dot(light_v, light_v).xxx;
                    float3 _750 = (_615 + (_623 * _716)) - _660.param1.xyz;
                    float _754 = dot(_734, _750);
                    if ((_754 >= (-0.5f)) && (_754 <= 0.5f))
                    {
                        float _767 = dot(light_v, _750);
                        if ((_767 >= (-0.5f)) && (_767 <= 0.5f))
                        {
                            _1037 = true;
                            _1034 = 0.0f;
                            break;
                        }
                    }
                }
            }
            else
            {
                if (_679 == 5u)
                {
                    float3 light_u_1 = _660.param2.xyz;
                    float3 light_v_1 = _660.param3.xyz;
                    float3 _797 = normalize(cross(_660.param2.xyz, _660.param3.xyz));
                    float _805 = dot(_623, _797);
                    float _813 = (dot(_797, _660.param1.xyz) - dot(_797, _615)) / _805;
                    if (((_805 < 0.0f) && (_813 > 9.9999999747524270787835121154785e-07f)) && (_813 < _627))
                    {
                        float3 _825 = light_u_1;
                        float3 _830 = _825 / dot(_825, _825).xxx;
                        light_u_1 = _830;
                        float3 _831 = light_v_1;
                        float3 _836 = _831 / dot(_831, _831).xxx;
                        light_v_1 = _836;
                        float3 _846 = (_615 + (_623 * _813)) - _660.param1.xyz;
                        float _850 = dot(_830, _846);
                        float _854 = dot(_836, _846);
                        if (sqrt(mad(_850, _850, _854 * _854)) <= 0.5f)
                        {
                            _1037 = true;
                            _1034 = 0.0f;
                            break;
                        }
                    }
                }
            }
        }
        if (_1037)
        {
            break;
        }
        _1037 = true;
        _1034 = 1.0f;
        break;
    } while(false);
    return _1034;
}

void comp_main()
{
    do
    {
        int _880 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_880) >= _886.Load(12))
        {
            break;
        }
        shadow_ray_t _905;
        [unroll]
        for (int _37ident = 0; _37ident < 3; _37ident++)
        {
            _905.o[_37ident] = asfloat(_901.Load(_37ident * 4 + _880 * 48 + 0));
        }
        _905.depth = int(_901.Load(_880 * 48 + 12));
        [unroll]
        for (int _38ident = 0; _38ident < 3; _38ident++)
        {
            _905.d[_38ident] = asfloat(_901.Load(_38ident * 4 + _880 * 48 + 16));
        }
        _905.dist = asfloat(_901.Load(_880 * 48 + 28));
        [unroll]
        for (int _39ident = 0; _39ident < 3; _39ident++)
        {
            _905.c[_39ident] = asfloat(_901.Load(_39ident * 4 + _880 * 48 + 32));
        }
        _905.xy = int(_901.Load(_880 * 48 + 44));
        shadow_ray_t _906;
        _906.o[0] = _905.o[0];
        _906.o[1] = _905.o[1];
        _906.o[2] = _905.o[2];
        _906.depth = _905.depth;
        _906.d[0] = _905.d[0];
        _906.d[1] = _905.d[1];
        _906.d[2] = _905.d[2];
        _906.dist = _905.dist;
        _906.c[0] = _905.c[0];
        _906.c[1] = _905.c[1];
        _906.c[2] = _905.c[2];
        _906.xy = _905.xy;
        shadow_ray_t param = _906;
        float3 _910 = IntersectSceneShadow(param);
        shadow_ray_t param_1 = _906;
        float3 _914 = _910 * IntersectAreaLightsShadow(param_1);
        if (lum(_914) > 0.0f)
        {
            int2 _940 = int2((_906.xy >> 16) & 65535, _906.xy & 65535);
            float4 _941 = g_inout_img[_940];
            float3 _950 = _941.xyz + min(_914, _352_g_params.clamp_val.xxx);
            float4 _1126 = _941;
            _1126.x = _950.x;
            float4 _1128 = _1126;
            _1128.y = _950.y;
            float4 _1130 = _1128;
            _1130.z = _950.z;
            g_inout_img[_940] = _1130;
        }
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
