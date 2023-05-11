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

struct Params
{
    uint4 rect;
    uint node_index;
    float inter_t;
    int min_transp_depth;
    int max_transp_depth;
    int hi;
    int _pad0;
    int _pad1;
    int _pad2;
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

struct vertex_t
{
    float p[3];
    float n[3];
    float b[3];
    float t[2][2];
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

RWByteAddressBuffer _415 : register(u12, space0);
ByteAddressBuffer _575 : register(t5, space0);
ByteAddressBuffer _613 : register(t6, space0);
ByteAddressBuffer _648 : register(t1, space0);
ByteAddressBuffer _652 : register(t2, space0);
ByteAddressBuffer _717 : register(t15, space0);
RWByteAddressBuffer _921 : register(u0, space0);
cbuffer UniformParams
{
    Params _365_g_params : packoffset(c0);
};

Texture2D<float4> g_textures[] : register(t0, space1);
SamplerState _g_textures_sampler[] : register(s0, space1);
uniform ??? g_tlas;

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

static ??? rq;

float3 safe_invert(float3 v)
{
    float3 inv_v = 1.0f.xxx / v;
    bool _103 = v.x <= 1.0000000116860974230803549289703e-07f;
    bool _110;
    if (_103)
    {
        _110 = v.x >= 0.0f;
    }
    else
    {
        _110 = _103;
    }
    if (_110)
    {
        float3 _1071 = inv_v;
        _1071.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _1071;
    }
    else
    {
        bool _119 = v.x >= (-1.0000000116860974230803549289703e-07f);
        bool _125;
        if (_119)
        {
            _125 = v.x < 0.0f;
        }
        else
        {
            _125 = _119;
        }
        if (_125)
        {
            float3 _1069 = inv_v;
            _1069.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _1069;
        }
    }
    bool _133 = v.y <= 1.0000000116860974230803549289703e-07f;
    bool _139;
    if (_133)
    {
        _139 = v.y >= 0.0f;
    }
    else
    {
        _139 = _133;
    }
    if (_139)
    {
        float3 _1075 = inv_v;
        _1075.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _1075;
    }
    else
    {
        bool _146 = v.y >= (-1.0000000116860974230803549289703e-07f);
        bool _152;
        if (_146)
        {
            _152 = v.y < 0.0f;
        }
        else
        {
            _152 = _146;
        }
        if (_152)
        {
            float3 _1073 = inv_v;
            _1073.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _1073;
        }
    }
    bool _159 = v.z <= 1.0000000116860974230803549289703e-07f;
    bool _165;
    if (_159)
    {
        _165 = v.z >= 0.0f;
    }
    else
    {
        _165 = _159;
    }
    if (_165)
    {
        float3 _1079 = inv_v;
        _1079.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _1079;
    }
    else
    {
        bool _172 = v.z >= (-1.0000000116860974230803549289703e-07f);
        bool _178;
        if (_172)
        {
            _178 = v.z < 0.0f;
        }
        else
        {
            _178 = _172;
        }
        if (_178)
        {
            float3 _1077 = inv_v;
            _1077.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _1077;
        }
    }
    return inv_v;
}

int hash(int x)
{
    uint _59 = uint(x);
    uint _66 = ((_59 >> uint(16)) ^ _59) * 73244475u;
    uint _71 = ((_66 >> uint(16)) ^ _66) * 73244475u;
    return int((_71 >> uint(16)) ^ _71);
}

float construct_float(inout uint m)
{
    m &= 8388607u;
    m |= 1065353216u;
    return asfloat(m) - 1.0f;
}

int total_depth(ray_data_t r)
{
    return (((r.depth & 255) + ((r.depth >> 8) & 255)) + ((r.depth >> 16) & 255)) + ((r.depth >> 24) & 255);
}

float3 YCoCg_to_RGB(float4 col)
{
    float _234 = mad(col.z, 31.875f, 1.0f);
    float _244 = (col.x - 0.501960813999176025390625f) / _234;
    float _250 = (col.y - 0.501960813999176025390625f) / _234;
    return float3((col.w + _244) - _250, col.w + _250, (col.w - _244) - _250);
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
    uint _306 = index & 16777215u;
    float4 res = g_textures[NonUniformResourceIndex(_306)].SampleLevel(_g_textures_sampler[NonUniformResourceIndex(_306)], uvs, float(lod));
    bool _317;
    if (maybe_YCoCg)
    {
        _317 = (index & 67108864u) != 0u;
    }
    else
    {
        _317 = maybe_YCoCg;
    }
    if (_317)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _335;
    if (maybe_SRGB)
    {
        _335 = (index & 16777216u) != 0u;
    }
    else
    {
        _335 = maybe_SRGB;
    }
    if (_335)
    {
        float3 param_1 = res.xyz;
        float3 _341 = srgb_to_rgb(param_1);
        float4 _1095 = res;
        _1095.x = _341.x;
        float4 _1097 = _1095;
        _1097.y = _341.y;
        float4 _1099 = _1097;
        _1099.z = _341.z;
        res = _1099;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod)
{
    return SampleBilinear(index, uvs, lod, false, false);
}

void comp_main()
{
    do
    {
        bool _369 = gl_GlobalInvocationID.x >= _365_g_params.rect.z;
        bool _378;
        if (!_369)
        {
            _378 = gl_GlobalInvocationID.y >= _365_g_params.rect.w;
        }
        else
        {
            _378 = _369;
        }
        if (_378)
        {
            break;
        }
        int _405 = int((gl_GlobalInvocationID.y * _365_g_params.rect.z) + gl_GlobalInvocationID.x);
        float3 ro = float3(asfloat(_415.Load(_405 * 72 + 0)), asfloat(_415.Load(_405 * 72 + 4)), asfloat(_415.Load(_405 * 72 + 8)));
        float _431 = asfloat(_415.Load(_405 * 72 + 12));
        float _434 = asfloat(_415.Load(_405 * 72 + 16));
        float _437 = asfloat(_415.Load(_405 * 72 + 20));
        float3 _438 = float3(_431, _434, _437);
        float3 param = _438;
        int _952 = 0;
        int _954 = 0;
        int _953 = 0;
        float _955 = _365_g_params.inter_t;
        float _957 = 0.0f;
        float _956 = 0.0f;
        uint param_1 = uint(hash(int(_415.Load(_405 * 72 + 64))));
        float _466 = construct_float(param_1);
        ray_data_t _474;
        [unroll]
        for (int _27ident = 0; _27ident < 3; _27ident++)
        {
            _474.o[_27ident] = asfloat(_415.Load(_27ident * 4 + _405 * 72 + 0));
        }
        [unroll]
        for (int _28ident = 0; _28ident < 3; _28ident++)
        {
            _474.d[_28ident] = asfloat(_415.Load(_28ident * 4 + _405 * 72 + 12));
        }
        _474.pdf = asfloat(_415.Load(_405 * 72 + 24));
        [unroll]
        for (int _29ident = 0; _29ident < 3; _29ident++)
        {
            _474.c[_29ident] = asfloat(_415.Load(_29ident * 4 + _405 * 72 + 28));
        }
        [unroll]
        for (int _30ident = 0; _30ident < 4; _30ident++)
        {
            _474.ior[_30ident] = asfloat(_415.Load(_30ident * 4 + _405 * 72 + 40));
        }
        _474.cone_width = asfloat(_415.Load(_405 * 72 + 56));
        _474.cone_spread = asfloat(_415.Load(_405 * 72 + 60));
        _474.xy = int(_415.Load(_405 * 72 + 64));
        _474.depth = int(_415.Load(_405 * 72 + 68));
        ray_data_t _477;
        _477.o[0] = _474.o[0];
        _477.o[1] = _474.o[1];
        _477.o[2] = _474.o[2];
        _477.d[0] = _474.d[0];
        _477.d[1] = _474.d[1];
        _477.d[2] = _474.d[2];
        _477.pdf = _474.pdf;
        _477.c[0] = _474.c[0];
        _477.c[1] = _474.c[1];
        _477.c[2] = _474.c[2];
        _477.ior[0] = _474.ior[0];
        _477.ior[1] = _474.ior[1];
        _477.ior[2] = _474.ior[2];
        _477.ior[3] = _474.ior[3];
        _477.cone_width = _474.cone_width;
        _477.cone_spread = _474.cone_spread;
        _477.xy = _474.xy;
        _477.depth = _474.depth;
        int rand_index = _365_g_params.hi + (total_depth(_477) * 7);
        int _556;
        float _811;
        float _490;
        for (;;)
        {
            _490 = _955;
            rayQueryInitializeEXT(rq, g_tlas, 0u, 255u, ro, 0.0f, _438, _490);
            for (;;)
            {
                bool _508 = rayQueryProceedEXT(rq);
                if (_508)
                {
                    rayQueryConfirmIntersectionEXT(rq);
                    continue;
                }
                else
                {
                    break;
                }
            }
            uint _509 = rayQueryGetIntersectionTypeEXT(rq, bool(1));
            if (_509 != 0u)
            {
                int _514 = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, bool(1));
                _952 = -1;
                int _517 = rayQueryGetIntersectionInstanceIdEXT(rq, bool(1));
                _953 = _517;
                int _520 = rayQueryGetIntersectionPrimitiveIndexEXT(rq, bool(1));
                _954 = _514 + _520;
                bool _523 = rayQueryGetIntersectionFrontFaceEXT(rq, bool(1));
                [flatten]
                if (_523 == false)
                {
                    _954 = (-1) - _954;
                }
                float2 _534 = rayQueryGetIntersectionBarycentricsEXT(rq, bool(1));
                _956 = _534.x;
                _957 = _534.y;
                float _541 = rayQueryGetIntersectionTEXT(rq, bool(1));
                _955 = _541;
            }
            if (_952 == 0)
            {
                break;
            }
            bool _553 = _954 < 0;
            if (_553)
            {
                _556 = (-1) - _954;
            }
            else
            {
                _556 = _954;
            }
            uint _567 = uint(_556);
            bool _569 = !_553;
            bool _585;
            if (_569)
            {
                _585 = ((_575.Load(_567 * 4 + 0) >> 16u) & 32768u) != 0u;
            }
            else
            {
                _585 = _569;
            }
            bool _598;
            if (!_585)
            {
                bool _597;
                if (_553)
                {
                    _597 = (_575.Load(_567 * 4 + 0) & 32768u) != 0u;
                }
                else
                {
                    _597 = _553;
                }
                _598 = _597;
            }
            else
            {
                _598 = _585;
            }
            if (_598)
            {
                break;
            }
            material_t _622;
            [unroll]
            for (int _31ident = 0; _31ident < 5; _31ident++)
            {
                _622.textures[_31ident] = _613.Load(_31ident * 4 + ((_575.Load(_567 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _32ident = 0; _32ident < 3; _32ident++)
            {
                _622.base_color[_32ident] = asfloat(_613.Load(_32ident * 4 + ((_575.Load(_567 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
            }
            _622.flags = _613.Load(((_575.Load(_567 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
            _622.type = _613.Load(((_575.Load(_567 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
            _622.tangent_rotation_or_strength = asfloat(_613.Load(((_575.Load(_567 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
            _622.roughness_and_anisotropic = _613.Load(((_575.Load(_567 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
            _622.ior = asfloat(_613.Load(((_575.Load(_567 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
            _622.sheen_and_sheen_tint = _613.Load(((_575.Load(_567 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
            _622.tint_and_metallic = _613.Load(((_575.Load(_567 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
            _622.transmission_and_transmission_roughness = _613.Load(((_575.Load(_567 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
            _622.specular_and_specular_tint = _613.Load(((_575.Load(_567 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
            _622.clearcoat_and_clearcoat_roughness = _613.Load(((_575.Load(_567 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
            _622.normal_map_strength_unorm = _613.Load(((_575.Load(_567 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
            material_t _623;
            _623.textures[0] = _622.textures[0];
            _623.textures[1] = _622.textures[1];
            _623.textures[2] = _622.textures[2];
            _623.textures[3] = _622.textures[3];
            _623.textures[4] = _622.textures[4];
            _623.base_color[0] = _622.base_color[0];
            _623.base_color[1] = _622.base_color[1];
            _623.base_color[2] = _622.base_color[2];
            _623.flags = _622.flags;
            _623.type = _622.type;
            _623.tangent_rotation_or_strength = _622.tangent_rotation_or_strength;
            _623.roughness_and_anisotropic = _622.roughness_and_anisotropic;
            _623.ior = _622.ior;
            _623.sheen_and_sheen_tint = _622.sheen_and_sheen_tint;
            _623.tint_and_metallic = _622.tint_and_metallic;
            _623.transmission_and_transmission_roughness = _622.transmission_and_transmission_roughness;
            _623.specular_and_specular_tint = _622.specular_and_specular_tint;
            _623.clearcoat_and_clearcoat_roughness = _622.clearcoat_and_clearcoat_roughness;
            _623.normal_map_strength_unorm = _622.normal_map_strength_unorm;
            uint _1010 = _623.textures[1];
            uint _1011 = _623.textures[3];
            uint _1012 = _623.textures[4];
            float _1025 = _623.base_color[0];
            float _1026 = _623.base_color[1];
            float _1027 = _623.base_color[2];
            uint _970 = _623.type;
            float _971 = _623.tangent_rotation_or_strength;
            if (_553)
            {
                material_t _632;
                [unroll]
                for (int _33ident = 0; _33ident < 5; _33ident++)
                {
                    _632.textures[_33ident] = _613.Load(_33ident * 4 + (_575.Load(_567 * 4 + 0) & 16383u) * 76 + 0);
                }
                [unroll]
                for (int _34ident = 0; _34ident < 3; _34ident++)
                {
                    _632.base_color[_34ident] = asfloat(_613.Load(_34ident * 4 + (_575.Load(_567 * 4 + 0) & 16383u) * 76 + 20));
                }
                _632.flags = _613.Load((_575.Load(_567 * 4 + 0) & 16383u) * 76 + 32);
                _632.type = _613.Load((_575.Load(_567 * 4 + 0) & 16383u) * 76 + 36);
                _632.tangent_rotation_or_strength = asfloat(_613.Load((_575.Load(_567 * 4 + 0) & 16383u) * 76 + 40));
                _632.roughness_and_anisotropic = _613.Load((_575.Load(_567 * 4 + 0) & 16383u) * 76 + 44);
                _632.ior = asfloat(_613.Load((_575.Load(_567 * 4 + 0) & 16383u) * 76 + 48));
                _632.sheen_and_sheen_tint = _613.Load((_575.Load(_567 * 4 + 0) & 16383u) * 76 + 52);
                _632.tint_and_metallic = _613.Load((_575.Load(_567 * 4 + 0) & 16383u) * 76 + 56);
                _632.transmission_and_transmission_roughness = _613.Load((_575.Load(_567 * 4 + 0) & 16383u) * 76 + 60);
                _632.specular_and_specular_tint = _613.Load((_575.Load(_567 * 4 + 0) & 16383u) * 76 + 64);
                _632.clearcoat_and_clearcoat_roughness = _613.Load((_575.Load(_567 * 4 + 0) & 16383u) * 76 + 68);
                _632.normal_map_strength_unorm = _613.Load((_575.Load(_567 * 4 + 0) & 16383u) * 76 + 72);
                material_t _633;
                _633.textures[0] = _632.textures[0];
                _633.textures[1] = _632.textures[1];
                _633.textures[2] = _632.textures[2];
                _633.textures[3] = _632.textures[3];
                _633.textures[4] = _632.textures[4];
                _633.base_color[0] = _632.base_color[0];
                _633.base_color[1] = _632.base_color[1];
                _633.base_color[2] = _632.base_color[2];
                _633.flags = _632.flags;
                _633.type = _632.type;
                _633.tangent_rotation_or_strength = _632.tangent_rotation_or_strength;
                _633.roughness_and_anisotropic = _632.roughness_and_anisotropic;
                _633.ior = _632.ior;
                _633.sheen_and_sheen_tint = _632.sheen_and_sheen_tint;
                _633.tint_and_metallic = _632.tint_and_metallic;
                _633.transmission_and_transmission_roughness = _632.transmission_and_transmission_roughness;
                _633.specular_and_specular_tint = _632.specular_and_specular_tint;
                _633.clearcoat_and_clearcoat_roughness = _632.clearcoat_and_clearcoat_roughness;
                _633.normal_map_strength_unorm = _632.normal_map_strength_unorm;
                _1010 = _633.textures[1];
                _1011 = _633.textures[3];
                _1012 = _633.textures[4];
                _1025 = _633.base_color[0];
                _1026 = _633.base_color[1];
                _1027 = _633.base_color[2];
                _970 = _633.type;
                _971 = _633.tangent_rotation_or_strength;
            }
            uint _654 = _567 * 3u;
            vertex_t _660;
            [unroll]
            for (int _35ident = 0; _35ident < 3; _35ident++)
            {
                _660.p[_35ident] = asfloat(_648.Load(_35ident * 4 + _652.Load(_654 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _660.n[_36ident] = asfloat(_648.Load(_36ident * 4 + _652.Load(_654 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _37ident = 0; _37ident < 3; _37ident++)
            {
                _660.b[_37ident] = asfloat(_648.Load(_37ident * 4 + _652.Load(_654 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _38ident = 0; _38ident < 2; _38ident++)
            {
                [unroll]
                for (int _39ident = 0; _39ident < 2; _39ident++)
                {
                    _660.t[_38ident][_39ident] = asfloat(_648.Load(_39ident * 4 + _38ident * 8 + _652.Load(_654 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _661;
            _661.p[0] = _660.p[0];
            _661.p[1] = _660.p[1];
            _661.p[2] = _660.p[2];
            _661.n[0] = _660.n[0];
            _661.n[1] = _660.n[1];
            _661.n[2] = _660.n[2];
            _661.b[0] = _660.b[0];
            _661.b[1] = _660.b[1];
            _661.b[2] = _660.b[2];
            _661.t[0][0] = _660.t[0][0];
            _661.t[0][1] = _660.t[0][1];
            _661.t[1][0] = _660.t[1][0];
            _661.t[1][1] = _660.t[1][1];
            vertex_t _669;
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _669.p[_40ident] = asfloat(_648.Load(_40ident * 4 + _652.Load((_654 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 3; _41ident++)
            {
                _669.n[_41ident] = asfloat(_648.Load(_41ident * 4 + _652.Load((_654 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _42ident = 0; _42ident < 3; _42ident++)
            {
                _669.b[_42ident] = asfloat(_648.Load(_42ident * 4 + _652.Load((_654 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _43ident = 0; _43ident < 2; _43ident++)
            {
                [unroll]
                for (int _44ident = 0; _44ident < 2; _44ident++)
                {
                    _669.t[_43ident][_44ident] = asfloat(_648.Load(_44ident * 4 + _43ident * 8 + _652.Load((_654 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _670;
            _670.p[0] = _669.p[0];
            _670.p[1] = _669.p[1];
            _670.p[2] = _669.p[2];
            _670.n[0] = _669.n[0];
            _670.n[1] = _669.n[1];
            _670.n[2] = _669.n[2];
            _670.b[0] = _669.b[0];
            _670.b[1] = _669.b[1];
            _670.b[2] = _669.b[2];
            _670.t[0][0] = _669.t[0][0];
            _670.t[0][1] = _669.t[0][1];
            _670.t[1][0] = _669.t[1][0];
            _670.t[1][1] = _669.t[1][1];
            vertex_t _678;
            [unroll]
            for (int _45ident = 0; _45ident < 3; _45ident++)
            {
                _678.p[_45ident] = asfloat(_648.Load(_45ident * 4 + _652.Load((_654 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _46ident = 0; _46ident < 3; _46ident++)
            {
                _678.n[_46ident] = asfloat(_648.Load(_46ident * 4 + _652.Load((_654 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _47ident = 0; _47ident < 3; _47ident++)
            {
                _678.b[_47ident] = asfloat(_648.Load(_47ident * 4 + _652.Load((_654 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _48ident = 0; _48ident < 2; _48ident++)
            {
                [unroll]
                for (int _49ident = 0; _49ident < 2; _49ident++)
                {
                    _678.t[_48ident][_49ident] = asfloat(_648.Load(_49ident * 4 + _48ident * 8 + _652.Load((_654 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _679;
            _679.p[0] = _678.p[0];
            _679.p[1] = _678.p[1];
            _679.p[2] = _678.p[2];
            _679.n[0] = _678.n[0];
            _679.n[1] = _678.n[1];
            _679.n[2] = _678.n[2];
            _679.b[0] = _678.b[0];
            _679.b[1] = _678.b[1];
            _679.b[2] = _678.b[2];
            _679.t[0][0] = _678.t[0][0];
            _679.t[0][1] = _678.t[0][1];
            _679.t[1][0] = _678.t[1][0];
            _679.t[1][1] = _678.t[1][1];
            float2 _712 = ((float2(_661.t[0][0], _661.t[0][1]) * ((1.0f - _956) - _957)) + (float2(_670.t[0][0], _670.t[0][1]) * _956)) + (float2(_679.t[0][0], _679.t[0][1]) * _957);
            float trans_r = frac(asfloat(_717.Load(rand_index * 4 + 0)) + _466);
            while (_970 == 4u)
            {
                float mix_val = _971;
                if (_1010 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_1010, _712, 0).x;
                }
                if (trans_r > mix_val)
                {
                    material_t _757;
                    [unroll]
                    for (int _50ident = 0; _50ident < 5; _50ident++)
                    {
                        _757.textures[_50ident] = _613.Load(_50ident * 4 + _1011 * 76 + 0);
                    }
                    [unroll]
                    for (int _51ident = 0; _51ident < 3; _51ident++)
                    {
                        _757.base_color[_51ident] = asfloat(_613.Load(_51ident * 4 + _1011 * 76 + 20));
                    }
                    _757.flags = _613.Load(_1011 * 76 + 32);
                    _757.type = _613.Load(_1011 * 76 + 36);
                    _757.tangent_rotation_or_strength = asfloat(_613.Load(_1011 * 76 + 40));
                    _757.roughness_and_anisotropic = _613.Load(_1011 * 76 + 44);
                    _757.ior = asfloat(_613.Load(_1011 * 76 + 48));
                    _757.sheen_and_sheen_tint = _613.Load(_1011 * 76 + 52);
                    _757.tint_and_metallic = _613.Load(_1011 * 76 + 56);
                    _757.transmission_and_transmission_roughness = _613.Load(_1011 * 76 + 60);
                    _757.specular_and_specular_tint = _613.Load(_1011 * 76 + 64);
                    _757.clearcoat_and_clearcoat_roughness = _613.Load(_1011 * 76 + 68);
                    _757.normal_map_strength_unorm = _613.Load(_1011 * 76 + 72);
                    material_t _758;
                    _758.textures[0] = _757.textures[0];
                    _758.textures[1] = _757.textures[1];
                    _758.textures[2] = _757.textures[2];
                    _758.textures[3] = _757.textures[3];
                    _758.textures[4] = _757.textures[4];
                    _758.base_color[0] = _757.base_color[0];
                    _758.base_color[1] = _757.base_color[1];
                    _758.base_color[2] = _757.base_color[2];
                    _758.flags = _757.flags;
                    _758.type = _757.type;
                    _758.tangent_rotation_or_strength = _757.tangent_rotation_or_strength;
                    _758.roughness_and_anisotropic = _757.roughness_and_anisotropic;
                    _758.ior = _757.ior;
                    _758.sheen_and_sheen_tint = _757.sheen_and_sheen_tint;
                    _758.tint_and_metallic = _757.tint_and_metallic;
                    _758.transmission_and_transmission_roughness = _757.transmission_and_transmission_roughness;
                    _758.specular_and_specular_tint = _757.specular_and_specular_tint;
                    _758.clearcoat_and_clearcoat_roughness = _757.clearcoat_and_clearcoat_roughness;
                    _758.normal_map_strength_unorm = _757.normal_map_strength_unorm;
                    _1010 = _758.textures[1];
                    _1011 = _758.textures[3];
                    _1012 = _758.textures[4];
                    _1025 = _758.base_color[0];
                    _1026 = _758.base_color[1];
                    _1027 = _758.base_color[2];
                    _970 = _758.type;
                    _971 = _758.tangent_rotation_or_strength;
                    trans_r = (trans_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _769;
                    [unroll]
                    for (int _52ident = 0; _52ident < 5; _52ident++)
                    {
                        _769.textures[_52ident] = _613.Load(_52ident * 4 + _1012 * 76 + 0);
                    }
                    [unroll]
                    for (int _53ident = 0; _53ident < 3; _53ident++)
                    {
                        _769.base_color[_53ident] = asfloat(_613.Load(_53ident * 4 + _1012 * 76 + 20));
                    }
                    _769.flags = _613.Load(_1012 * 76 + 32);
                    _769.type = _613.Load(_1012 * 76 + 36);
                    _769.tangent_rotation_or_strength = asfloat(_613.Load(_1012 * 76 + 40));
                    _769.roughness_and_anisotropic = _613.Load(_1012 * 76 + 44);
                    _769.ior = asfloat(_613.Load(_1012 * 76 + 48));
                    _769.sheen_and_sheen_tint = _613.Load(_1012 * 76 + 52);
                    _769.tint_and_metallic = _613.Load(_1012 * 76 + 56);
                    _769.transmission_and_transmission_roughness = _613.Load(_1012 * 76 + 60);
                    _769.specular_and_specular_tint = _613.Load(_1012 * 76 + 64);
                    _769.clearcoat_and_clearcoat_roughness = _613.Load(_1012 * 76 + 68);
                    _769.normal_map_strength_unorm = _613.Load(_1012 * 76 + 72);
                    material_t _770;
                    _770.textures[0] = _769.textures[0];
                    _770.textures[1] = _769.textures[1];
                    _770.textures[2] = _769.textures[2];
                    _770.textures[3] = _769.textures[3];
                    _770.textures[4] = _769.textures[4];
                    _770.base_color[0] = _769.base_color[0];
                    _770.base_color[1] = _769.base_color[1];
                    _770.base_color[2] = _769.base_color[2];
                    _770.flags = _769.flags;
                    _770.type = _769.type;
                    _770.tangent_rotation_or_strength = _769.tangent_rotation_or_strength;
                    _770.roughness_and_anisotropic = _769.roughness_and_anisotropic;
                    _770.ior = _769.ior;
                    _770.sheen_and_sheen_tint = _769.sheen_and_sheen_tint;
                    _770.tint_and_metallic = _769.tint_and_metallic;
                    _770.transmission_and_transmission_roughness = _769.transmission_and_transmission_roughness;
                    _770.specular_and_specular_tint = _769.specular_and_specular_tint;
                    _770.clearcoat_and_clearcoat_roughness = _769.clearcoat_and_clearcoat_roughness;
                    _770.normal_map_strength_unorm = _769.normal_map_strength_unorm;
                    _1010 = _770.textures[1];
                    _1011 = _770.textures[3];
                    _1012 = _770.textures[4];
                    _1025 = _770.base_color[0];
                    _1026 = _770.base_color[1];
                    _1027 = _770.base_color[2];
                    _970 = _770.type;
                    _971 = _770.tangent_rotation_or_strength;
                    trans_r /= mix_val;
                }
            }
            if (_970 != 5u)
            {
                break;
            }
            float _799 = max(asfloat(_415.Load(_405 * 72 + 28)), max(asfloat(_415.Load(_405 * 72 + 32)), asfloat(_415.Load(_405 * 72 + 36))));
            if ((int(_415.Load(_405 * 72 + 68)) >> 24) > _365_g_params.min_transp_depth)
            {
                _811 = max(0.0500000007450580596923828125f, 1.0f - _799);
            }
            else
            {
                _811 = 0.0f;
            }
            bool _825 = (frac(asfloat(_717.Load((rand_index + 6) * 4 + 0)) + _466) < _811) || (_799 == 0.0f);
            bool _837;
            if (!_825)
            {
                _837 = ((int(_415.Load(_405 * 72 + 68)) >> 24) + 1) >= _365_g_params.max_transp_depth;
            }
            else
            {
                _837 = _825;
            }
            if (_837)
            {
                _415.Store(_405 * 72 + 36, asuint(0.0f));
                _415.Store(_405 * 72 + 32, asuint(0.0f));
                _415.Store(_405 * 72 + 28, asuint(0.0f));
                break;
            }
            float _851 = 1.0f - _811;
            _415.Store(_405 * 72 + 28, asuint(asfloat(_415.Load(_405 * 72 + 28)) * (_1025 / _851)));
            _415.Store(_405 * 72 + 32, asuint(asfloat(_415.Load(_405 * 72 + 32)) * (_1026 / _851)));
            _415.Store(_405 * 72 + 36, asuint(asfloat(_415.Load(_405 * 72 + 36)) * (_1027 / _851)));
            ro += (_438 * (_955 + 9.9999997473787516355514526367188e-06f));
            _952 = 0;
            _955 = _490 - _955;
            _415.Store(_405 * 72 + 68, uint(int(_415.Load(_405 * 72 + 68)) + 16777216));
            rand_index += 7;
            continue;
        }
        float _914 = _955;
        float _915 = _914 + distance(float3(asfloat(_415.Load(_405 * 72 + 0)), asfloat(_415.Load(_405 * 72 + 4)), asfloat(_415.Load(_405 * 72 + 8))), ro);
        _955 = _915;
        hit_data_t _964 = { _952, _953, _954, _915, _956, _957 };
        hit_data_t _926;
        _926.mask = _964.mask;
        _926.obj_index = _964.obj_index;
        _926.prim_index = _964.prim_index;
        _926.t = _964.t;
        _926.u = _964.u;
        _926.v = _964.v;
        _921.Store(_405 * 24 + 0, uint(_926.mask));
        _921.Store(_405 * 24 + 4, uint(_926.obj_index));
        _921.Store(_405 * 24 + 8, uint(_926.prim_index));
        _921.Store(_405 * 24 + 12, asuint(_926.t));
        _921.Store(_405 * 24 + 16, asuint(_926.u));
        _921.Store(_405 * 24 + 20, asuint(_926.v));
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
