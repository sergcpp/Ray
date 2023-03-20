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
    uint2 img_size;
    uint node_index;
    int max_transp_depth;
    int blocker_lights_count;
    int _pad0;
    int _pad1;
    int _pad2;
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
ByteAddressBuffer _384 : register(t10, space0);
ByteAddressBuffer _388 : register(t11, space0);
ByteAddressBuffer _488 : register(t4, space0);
ByteAddressBuffer _644 : register(t15, space0);
ByteAddressBuffer _656 : register(t14, space0);
ByteAddressBuffer _887 : register(t13, space0);
ByteAddressBuffer _902 : register(t12, space0);
ByteAddressBuffer _983 : register(t1, space0);
ByteAddressBuffer _987 : register(t2, space0);
ByteAddressBuffer _992 : register(t5, space0);
ByteAddressBuffer _999 : register(t6, space0);
ByteAddressBuffer _1004 : register(t7, space0);
ByteAddressBuffer _1008 : register(t8, space0);
ByteAddressBuffer _1014 : register(t9, space0);
cbuffer UniformParams
{
    Params _353_g_params : packoffset(c0);
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
        float4 _1134 = res;
        _1134.x = _200.x;
        float4 _1136 = _1134;
        _1136.y = _200.y;
        float4 _1138 = _1136;
        _1138.z = _200.z;
        res = _1138;
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
    bool _1024 = false;
    float3 _1021;
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
                bool _358;
                if (!_343)
                {
                    _358 = depth > _353_g_params.max_transp_depth;
                }
                else
                {
                    _358 = _343;
                }
                if (_358)
                {
                    _1024 = true;
                    _1021 = 0.0f.xxx;
                    break;
                }
                int _390 = _299 * 3;
                vertex_t _396;
                [unroll]
                for (int _20ident = 0; _20ident < 3; _20ident++)
                {
                    _396.p[_20ident] = asfloat(_384.Load(_20ident * 4 + _388.Load(_390 * 4 + 0) * 52 + 0));
                }
                [unroll]
                for (int _21ident = 0; _21ident < 3; _21ident++)
                {
                    _396.n[_21ident] = asfloat(_384.Load(_21ident * 4 + _388.Load(_390 * 4 + 0) * 52 + 12));
                }
                [unroll]
                for (int _22ident = 0; _22ident < 3; _22ident++)
                {
                    _396.b[_22ident] = asfloat(_384.Load(_22ident * 4 + _388.Load(_390 * 4 + 0) * 52 + 24));
                }
                [unroll]
                for (int _23ident = 0; _23ident < 2; _23ident++)
                {
                    [unroll]
                    for (int _24ident = 0; _24ident < 2; _24ident++)
                    {
                        _396.t[_23ident][_24ident] = asfloat(_384.Load(_24ident * 4 + _23ident * 8 + _388.Load(_390 * 4 + 0) * 52 + 36));
                    }
                }
                vertex_t _397;
                _397.p[0] = _396.p[0];
                _397.p[1] = _396.p[1];
                _397.p[2] = _396.p[2];
                _397.n[0] = _396.n[0];
                _397.n[1] = _396.n[1];
                _397.n[2] = _396.n[2];
                _397.b[0] = _396.b[0];
                _397.b[1] = _396.b[1];
                _397.b[2] = _396.b[2];
                _397.t[0][0] = _396.t[0][0];
                _397.t[0][1] = _396.t[0][1];
                _397.t[1][0] = _396.t[1][0];
                _397.t[1][1] = _396.t[1][1];
                vertex_t _405;
                [unroll]
                for (int _25ident = 0; _25ident < 3; _25ident++)
                {
                    _405.p[_25ident] = asfloat(_384.Load(_25ident * 4 + _388.Load((_390 + 1) * 4 + 0) * 52 + 0));
                }
                [unroll]
                for (int _26ident = 0; _26ident < 3; _26ident++)
                {
                    _405.n[_26ident] = asfloat(_384.Load(_26ident * 4 + _388.Load((_390 + 1) * 4 + 0) * 52 + 12));
                }
                [unroll]
                for (int _27ident = 0; _27ident < 3; _27ident++)
                {
                    _405.b[_27ident] = asfloat(_384.Load(_27ident * 4 + _388.Load((_390 + 1) * 4 + 0) * 52 + 24));
                }
                [unroll]
                for (int _28ident = 0; _28ident < 2; _28ident++)
                {
                    [unroll]
                    for (int _29ident = 0; _29ident < 2; _29ident++)
                    {
                        _405.t[_28ident][_29ident] = asfloat(_384.Load(_29ident * 4 + _28ident * 8 + _388.Load((_390 + 1) * 4 + 0) * 52 + 36));
                    }
                }
                vertex_t _406;
                _406.p[0] = _405.p[0];
                _406.p[1] = _405.p[1];
                _406.p[2] = _405.p[2];
                _406.n[0] = _405.n[0];
                _406.n[1] = _405.n[1];
                _406.n[2] = _405.n[2];
                _406.b[0] = _405.b[0];
                _406.b[1] = _405.b[1];
                _406.b[2] = _405.b[2];
                _406.t[0][0] = _405.t[0][0];
                _406.t[0][1] = _405.t[0][1];
                _406.t[1][0] = _405.t[1][0];
                _406.t[1][1] = _405.t[1][1];
                vertex_t _414;
                [unroll]
                for (int _30ident = 0; _30ident < 3; _30ident++)
                {
                    _414.p[_30ident] = asfloat(_384.Load(_30ident * 4 + _388.Load((_390 + 2) * 4 + 0) * 52 + 0));
                }
                [unroll]
                for (int _31ident = 0; _31ident < 3; _31ident++)
                {
                    _414.n[_31ident] = asfloat(_384.Load(_31ident * 4 + _388.Load((_390 + 2) * 4 + 0) * 52 + 12));
                }
                [unroll]
                for (int _32ident = 0; _32ident < 3; _32ident++)
                {
                    _414.b[_32ident] = asfloat(_384.Load(_32ident * 4 + _388.Load((_390 + 2) * 4 + 0) * 52 + 24));
                }
                [unroll]
                for (int _33ident = 0; _33ident < 2; _33ident++)
                {
                    [unroll]
                    for (int _34ident = 0; _34ident < 2; _34ident++)
                    {
                        _414.t[_33ident][_34ident] = asfloat(_384.Load(_34ident * 4 + _33ident * 8 + _388.Load((_390 + 2) * 4 + 0) * 52 + 36));
                    }
                }
                vertex_t _415;
                _415.p[0] = _414.p[0];
                _415.p[1] = _414.p[1];
                _415.p[2] = _414.p[2];
                _415.n[0] = _414.n[0];
                _415.n[1] = _414.n[1];
                _415.n[2] = _414.n[2];
                _415.b[0] = _414.b[0];
                _415.b[1] = _414.b[1];
                _415.b[2] = _414.b[2];
                _415.t[0][0] = _414.t[0][0];
                _415.t[0][1] = _414.t[0][1];
                _415.t[1][0] = _414.t[1][0];
                _415.t[1][1] = _414.t[1][1];
                float2 _418 = rayQueryGetIntersectionBarycentricsEXT(rq, bool(1));
                float2 _451 = ((float2(_397.t[0][0], _397.t[0][1]) * ((1.0f - _418.x) - _418.y)) + (float2(_406.t[0][0], _406.t[0][1]) * _418.x)) + (float2(_415.t[0][0], _415.t[0][1]) * _418.y);
                g_stack[gl_LocalInvocationIndex][0] = (_322 ? _318 : _313) & 16383u;
                g_stack[gl_LocalInvocationIndex][1] = 1065353216u;
                int stack_size = 1;
                float3 throughput = 0.0f.xxx;
                for (;;)
                {
                    int _474 = stack_size;
                    stack_size = _474 - 1;
                    if (_474 != 0)
                    {
                        int _490 = stack_size;
                        int _491 = 2 * _490;
                        material_t _497;
                        [unroll]
                        for (int _35ident = 0; _35ident < 5; _35ident++)
                        {
                            _497.textures[_35ident] = _488.Load(_35ident * 4 + g_stack[gl_LocalInvocationIndex][_491] * 76 + 0);
                        }
                        [unroll]
                        for (int _36ident = 0; _36ident < 3; _36ident++)
                        {
                            _497.base_color[_36ident] = asfloat(_488.Load(_36ident * 4 + g_stack[gl_LocalInvocationIndex][_491] * 76 + 20));
                        }
                        _497.flags = _488.Load(g_stack[gl_LocalInvocationIndex][_491] * 76 + 32);
                        _497.type = _488.Load(g_stack[gl_LocalInvocationIndex][_491] * 76 + 36);
                        _497.tangent_rotation_or_strength = asfloat(_488.Load(g_stack[gl_LocalInvocationIndex][_491] * 76 + 40));
                        _497.roughness_and_anisotropic = _488.Load(g_stack[gl_LocalInvocationIndex][_491] * 76 + 44);
                        _497.ior = asfloat(_488.Load(g_stack[gl_LocalInvocationIndex][_491] * 76 + 48));
                        _497.sheen_and_sheen_tint = _488.Load(g_stack[gl_LocalInvocationIndex][_491] * 76 + 52);
                        _497.tint_and_metallic = _488.Load(g_stack[gl_LocalInvocationIndex][_491] * 76 + 56);
                        _497.transmission_and_transmission_roughness = _488.Load(g_stack[gl_LocalInvocationIndex][_491] * 76 + 60);
                        _497.specular_and_specular_tint = _488.Load(g_stack[gl_LocalInvocationIndex][_491] * 76 + 64);
                        _497.clearcoat_and_clearcoat_roughness = _488.Load(g_stack[gl_LocalInvocationIndex][_491] * 76 + 68);
                        _497.normal_map_strength_unorm = _488.Load(g_stack[gl_LocalInvocationIndex][_491] * 76 + 72);
                        material_t _498;
                        _498.textures[0] = _497.textures[0];
                        _498.textures[1] = _497.textures[1];
                        _498.textures[2] = _497.textures[2];
                        _498.textures[3] = _497.textures[3];
                        _498.textures[4] = _497.textures[4];
                        _498.base_color[0] = _497.base_color[0];
                        _498.base_color[1] = _497.base_color[1];
                        _498.base_color[2] = _497.base_color[2];
                        _498.flags = _497.flags;
                        _498.type = _497.type;
                        _498.tangent_rotation_or_strength = _497.tangent_rotation_or_strength;
                        _498.roughness_and_anisotropic = _497.roughness_and_anisotropic;
                        _498.ior = _497.ior;
                        _498.sheen_and_sheen_tint = _497.sheen_and_sheen_tint;
                        _498.tint_and_metallic = _497.tint_and_metallic;
                        _498.transmission_and_transmission_roughness = _497.transmission_and_transmission_roughness;
                        _498.specular_and_specular_tint = _497.specular_and_specular_tint;
                        _498.clearcoat_and_clearcoat_roughness = _497.clearcoat_and_clearcoat_roughness;
                        _498.normal_map_strength_unorm = _497.normal_map_strength_unorm;
                        uint _505 = g_stack[gl_LocalInvocationIndex][_491 + 1];
                        float _506 = asfloat(_505);
                        if (_498.type == 4u)
                        {
                            float mix_val = _498.tangent_rotation_or_strength;
                            if (_498.textures[1] != 4294967295u)
                            {
                                mix_val *= SampleBilinear(_498.textures[1], _451, 0).x;
                            }
                            int _531 = 2 * stack_size;
                            g_stack[gl_LocalInvocationIndex][_531] = _498.textures[3];
                            g_stack[gl_LocalInvocationIndex][_531 + 1] = asuint(_506 * (1.0f - mix_val));
                            int _550 = 2 * (stack_size + 1);
                            g_stack[gl_LocalInvocationIndex][_550] = _498.textures[4];
                            g_stack[gl_LocalInvocationIndex][_550 + 1] = asuint(_506 * mix_val);
                            stack_size += 2;
                        }
                        else
                        {
                            if (_498.type == 5u)
                            {
                                throughput += (float3(_498.base_color[0], _498.base_color[1], _498.base_color[2]) * _506);
                            }
                        }
                        continue;
                    }
                    else
                    {
                        break;
                    }
                }
                float3 _584 = rc;
                float3 _585 = _584 * throughput;
                rc = _585;
                if (lum(_585) < 1.0000000116860974230803549289703e-07f)
                {
                    break;
                }
            }
            float _594 = rayQueryGetIntersectionTEXT(rq, bool(1));
            float _595 = _594 + 9.9999997473787516355514526367188e-06f;
            ro += (_230 * _595);
            dist -= _595;
            depth++;
        }
        if (_1024)
        {
            break;
        }
        _1024 = true;
        _1021 = rc;
        break;
    } while(false);
    return _1021;
}

float IntersectAreaLightsShadow(shadow_ray_t r)
{
    bool _1031 = false;
    float _1028;
    do
    {
        float3 _616 = float3(r.o[0], r.o[1], r.o[2]);
        float3 _624 = float3(r.d[0], r.d[1], r.d[2]);
        float _628 = abs(r.dist);
        for (uint li = 0u; li < uint(_353_g_params.blocker_lights_count); li++)
        {
            light_t _660;
            _660.type_and_param0 = _656.Load4(_644.Load(li * 4 + 0) * 64 + 0);
            _660.param1 = asfloat(_656.Load4(_644.Load(li * 4 + 0) * 64 + 16));
            _660.param2 = asfloat(_656.Load4(_644.Load(li * 4 + 0) * 64 + 32));
            _660.param3 = asfloat(_656.Load4(_644.Load(li * 4 + 0) * 64 + 48));
            light_t _661;
            _661.type_and_param0 = _660.type_and_param0;
            _661.param1 = _660.param1;
            _661.param2 = _660.param2;
            _661.param3 = _660.param3;
            bool _666 = (_661.type_and_param0.x & 128u) != 0u;
            bool _672;
            if (_666)
            {
                _672 = r.dist >= 0.0f;
            }
            else
            {
                _672 = _666;
            }
            [branch]
            if (_672)
            {
                continue;
            }
            uint _680 = _661.type_and_param0.x & 31u;
            if (_680 == 4u)
            {
                float3 light_u = _661.param2.xyz;
                float3 light_v = _661.param3.xyz;
                float3 _701 = normalize(cross(_661.param2.xyz, _661.param3.xyz));
                float _709 = dot(_624, _701);
                float _717 = (dot(_701, _661.param1.xyz) - dot(_701, _616)) / _709;
                if (((_709 < 0.0f) && (_717 > 9.9999999747524270787835121154785e-07f)) && (_717 < _628))
                {
                    float3 _730 = light_u;
                    float3 _735 = _730 / dot(_730, _730).xxx;
                    light_u = _735;
                    light_v /= dot(light_v, light_v).xxx;
                    float3 _751 = (_616 + (_624 * _717)) - _661.param1.xyz;
                    float _755 = dot(_735, _751);
                    if ((_755 >= (-0.5f)) && (_755 <= 0.5f))
                    {
                        float _768 = dot(light_v, _751);
                        if ((_768 >= (-0.5f)) && (_768 <= 0.5f))
                        {
                            _1031 = true;
                            _1028 = 0.0f;
                            break;
                        }
                    }
                }
            }
            else
            {
                if (_680 == 5u)
                {
                    float3 light_u_1 = _661.param2.xyz;
                    float3 light_v_1 = _661.param3.xyz;
                    float3 _798 = normalize(cross(_661.param2.xyz, _661.param3.xyz));
                    float _806 = dot(_624, _798);
                    float _814 = (dot(_798, _661.param1.xyz) - dot(_798, _616)) / _806;
                    if (((_806 < 0.0f) && (_814 > 9.9999999747524270787835121154785e-07f)) && (_814 < _628))
                    {
                        float3 _826 = light_u_1;
                        float3 _831 = _826 / dot(_826, _826).xxx;
                        light_u_1 = _831;
                        float3 _832 = light_v_1;
                        float3 _837 = _832 / dot(_832, _832).xxx;
                        light_v_1 = _837;
                        float3 _847 = (_616 + (_624 * _814)) - _661.param1.xyz;
                        float _851 = dot(_831, _847);
                        float _855 = dot(_837, _847);
                        if (sqrt(mad(_851, _851, _855 * _855)) <= 0.5f)
                        {
                            _1031 = true;
                            _1028 = 0.0f;
                            break;
                        }
                    }
                }
            }
        }
        if (_1031)
        {
            break;
        }
        _1031 = true;
        _1028 = 1.0f;
        break;
    } while(false);
    return _1028;
}

void comp_main()
{
    do
    {
        int _881 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_881) >= _887.Load(12))
        {
            break;
        }
        shadow_ray_t _906;
        [unroll]
        for (int _37ident = 0; _37ident < 3; _37ident++)
        {
            _906.o[_37ident] = asfloat(_902.Load(_37ident * 4 + _881 * 48 + 0));
        }
        _906.depth = int(_902.Load(_881 * 48 + 12));
        [unroll]
        for (int _38ident = 0; _38ident < 3; _38ident++)
        {
            _906.d[_38ident] = asfloat(_902.Load(_38ident * 4 + _881 * 48 + 16));
        }
        _906.dist = asfloat(_902.Load(_881 * 48 + 28));
        [unroll]
        for (int _39ident = 0; _39ident < 3; _39ident++)
        {
            _906.c[_39ident] = asfloat(_902.Load(_39ident * 4 + _881 * 48 + 32));
        }
        _906.xy = int(_902.Load(_881 * 48 + 44));
        shadow_ray_t _907;
        _907.o[0] = _906.o[0];
        _907.o[1] = _906.o[1];
        _907.o[2] = _906.o[2];
        _907.depth = _906.depth;
        _907.d[0] = _906.d[0];
        _907.d[1] = _906.d[1];
        _907.d[2] = _906.d[2];
        _907.dist = _906.dist;
        _907.c[0] = _906.c[0];
        _907.c[1] = _906.c[1];
        _907.c[2] = _906.c[2];
        _907.xy = _906.xy;
        shadow_ray_t param = _907;
        float3 _911 = IntersectSceneShadow(param);
        shadow_ray_t param_1 = _907;
        float3 _915 = _911 * IntersectAreaLightsShadow(param_1);
        if (lum(_915) > 0.0f)
        {
            int2 _941 = int2((_907.xy >> 16) & 65535, _907.xy & 65535);
            g_inout_img[_941] = float4(g_inout_img[_941].xyz + _915, 1.0f);
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
