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

ByteAddressBuffer _373 : register(t14, space0);
RWByteAddressBuffer _390 : register(u12, space0);
ByteAddressBuffer _567 : register(t5, space0);
ByteAddressBuffer _604 : register(t6, space0);
ByteAddressBuffer _639 : register(t1, space0);
ByteAddressBuffer _643 : register(t2, space0);
ByteAddressBuffer _708 : register(t15, space0);
RWByteAddressBuffer _913 : register(u0, space0);
cbuffer UniformParams
{
    Params _442_g_params : packoffset(c0);
};

Texture2D<float4> g_textures[] : register(t0, space1);
SamplerState _g_textures_sampler[] : register(s0, space1);
uniform ??? g_tlas;

static uint3 gl_WorkGroupID;
static uint gl_LocalInvocationIndex;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint gl_LocalInvocationIndex : SV_GroupIndex;
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
        float3 _1063 = inv_v;
        _1063.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _1063;
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
            float3 _1061 = inv_v;
            _1061.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _1061;
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
        float3 _1067 = inv_v;
        _1067.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _1067;
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
            float3 _1065 = inv_v;
            _1065.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _1065;
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
        float3 _1071 = inv_v;
        _1071.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _1071;
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
            float3 _1069 = inv_v;
            _1069.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _1069;
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
        float4 _1087 = res;
        _1087.x = _341.x;
        float4 _1089 = _1087;
        _1089.y = _341.y;
        float4 _1091 = _1089;
        _1091.z = _341.z;
        res = _1091;
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
        int _367 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_367) >= _373.Load(4))
        {
            break;
        }
        float3 ro = float3(asfloat(_390.Load(_367 * 72 + 0)), asfloat(_390.Load(_367 * 72 + 4)), asfloat(_390.Load(_367 * 72 + 8)));
        float _420 = asfloat(_390.Load(_367 * 72 + 12));
        float _423 = asfloat(_390.Load(_367 * 72 + 16));
        float _426 = asfloat(_390.Load(_367 * 72 + 20));
        float3 _427 = float3(_420, _423, _426);
        float3 param = _427;
        int _944 = 0;
        int _946 = 0;
        int _945 = 0;
        float _947 = _442_g_params.inter_t;
        float _949 = 0.0f;
        float _948 = 0.0f;
        uint param_1 = uint(hash(int(_390.Load(_367 * 72 + 64))));
        float _458 = construct_float(param_1);
        ray_data_t _466;
        [unroll]
        for (int _27ident = 0; _27ident < 3; _27ident++)
        {
            _466.o[_27ident] = asfloat(_390.Load(_27ident * 4 + _367 * 72 + 0));
        }
        [unroll]
        for (int _28ident = 0; _28ident < 3; _28ident++)
        {
            _466.d[_28ident] = asfloat(_390.Load(_28ident * 4 + _367 * 72 + 12));
        }
        _466.pdf = asfloat(_390.Load(_367 * 72 + 24));
        [unroll]
        for (int _29ident = 0; _29ident < 3; _29ident++)
        {
            _466.c[_29ident] = asfloat(_390.Load(_29ident * 4 + _367 * 72 + 28));
        }
        [unroll]
        for (int _30ident = 0; _30ident < 4; _30ident++)
        {
            _466.ior[_30ident] = asfloat(_390.Load(_30ident * 4 + _367 * 72 + 40));
        }
        _466.cone_width = asfloat(_390.Load(_367 * 72 + 56));
        _466.cone_spread = asfloat(_390.Load(_367 * 72 + 60));
        _466.xy = int(_390.Load(_367 * 72 + 64));
        _466.depth = int(_390.Load(_367 * 72 + 68));
        ray_data_t _469;
        _469.o[0] = _466.o[0];
        _469.o[1] = _466.o[1];
        _469.o[2] = _466.o[2];
        _469.d[0] = _466.d[0];
        _469.d[1] = _466.d[1];
        _469.d[2] = _466.d[2];
        _469.pdf = _466.pdf;
        _469.c[0] = _466.c[0];
        _469.c[1] = _466.c[1];
        _469.c[2] = _466.c[2];
        _469.ior[0] = _466.ior[0];
        _469.ior[1] = _466.ior[1];
        _469.ior[2] = _466.ior[2];
        _469.ior[3] = _466.ior[3];
        _469.cone_width = _466.cone_width;
        _469.cone_spread = _466.cone_spread;
        _469.xy = _466.xy;
        _469.depth = _466.depth;
        int rand_index = _442_g_params.hi + (total_depth(_469) * 7);
        int _548;
        float _802;
        float _482;
        for (;;)
        {
            _482 = _947;
            rayQueryInitializeEXT(rq, g_tlas, 0u, 255u, ro, 0.0f, _427, _482);
            for (;;)
            {
                bool _500 = rayQueryProceedEXT(rq);
                if (_500)
                {
                    rayQueryConfirmIntersectionEXT(rq);
                    continue;
                }
                else
                {
                    break;
                }
            }
            uint _501 = rayQueryGetIntersectionTypeEXT(rq, bool(1));
            if (_501 != 0u)
            {
                int _506 = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, bool(1));
                _944 = -1;
                int _509 = rayQueryGetIntersectionInstanceIdEXT(rq, bool(1));
                _945 = _509;
                int _512 = rayQueryGetIntersectionPrimitiveIndexEXT(rq, bool(1));
                _946 = _506 + _512;
                bool _515 = rayQueryGetIntersectionFrontFaceEXT(rq, bool(1));
                [flatten]
                if (_515 == false)
                {
                    _946 = (-1) - _946;
                }
                float2 _526 = rayQueryGetIntersectionBarycentricsEXT(rq, bool(1));
                _948 = _526.x;
                _949 = _526.y;
                float _533 = rayQueryGetIntersectionTEXT(rq, bool(1));
                _947 = _533;
            }
            if (_944 == 0)
            {
                break;
            }
            bool _545 = _946 < 0;
            if (_545)
            {
                _548 = (-1) - _946;
            }
            else
            {
                _548 = _946;
            }
            uint _559 = uint(_548);
            bool _561 = !_545;
            bool _576;
            if (_561)
            {
                _576 = ((_567.Load(_559 * 4 + 0) >> 16u) & 32768u) != 0u;
            }
            else
            {
                _576 = _561;
            }
            bool _589;
            if (!_576)
            {
                bool _588;
                if (_545)
                {
                    _588 = (_567.Load(_559 * 4 + 0) & 32768u) != 0u;
                }
                else
                {
                    _588 = _545;
                }
                _589 = _588;
            }
            else
            {
                _589 = _576;
            }
            if (_589)
            {
                break;
            }
            material_t _613;
            [unroll]
            for (int _31ident = 0; _31ident < 5; _31ident++)
            {
                _613.textures[_31ident] = _604.Load(_31ident * 4 + ((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _32ident = 0; _32ident < 3; _32ident++)
            {
                _613.base_color[_32ident] = asfloat(_604.Load(_32ident * 4 + ((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
            }
            _613.flags = _604.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
            _613.type = _604.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
            _613.tangent_rotation_or_strength = asfloat(_604.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
            _613.roughness_and_anisotropic = _604.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
            _613.ior = asfloat(_604.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
            _613.sheen_and_sheen_tint = _604.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
            _613.tint_and_metallic = _604.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
            _613.transmission_and_transmission_roughness = _604.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
            _613.specular_and_specular_tint = _604.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
            _613.clearcoat_and_clearcoat_roughness = _604.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
            _613.normal_map_strength_unorm = _604.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
            material_t _614;
            _614.textures[0] = _613.textures[0];
            _614.textures[1] = _613.textures[1];
            _614.textures[2] = _613.textures[2];
            _614.textures[3] = _613.textures[3];
            _614.textures[4] = _613.textures[4];
            _614.base_color[0] = _613.base_color[0];
            _614.base_color[1] = _613.base_color[1];
            _614.base_color[2] = _613.base_color[2];
            _614.flags = _613.flags;
            _614.type = _613.type;
            _614.tangent_rotation_or_strength = _613.tangent_rotation_or_strength;
            _614.roughness_and_anisotropic = _613.roughness_and_anisotropic;
            _614.ior = _613.ior;
            _614.sheen_and_sheen_tint = _613.sheen_and_sheen_tint;
            _614.tint_and_metallic = _613.tint_and_metallic;
            _614.transmission_and_transmission_roughness = _613.transmission_and_transmission_roughness;
            _614.specular_and_specular_tint = _613.specular_and_specular_tint;
            _614.clearcoat_and_clearcoat_roughness = _613.clearcoat_and_clearcoat_roughness;
            _614.normal_map_strength_unorm = _613.normal_map_strength_unorm;
            uint _1002 = _614.textures[1];
            uint _1003 = _614.textures[3];
            uint _1004 = _614.textures[4];
            float _1017 = _614.base_color[0];
            float _1018 = _614.base_color[1];
            float _1019 = _614.base_color[2];
            uint _962 = _614.type;
            float _963 = _614.tangent_rotation_or_strength;
            if (_545)
            {
                material_t _623;
                [unroll]
                for (int _33ident = 0; _33ident < 5; _33ident++)
                {
                    _623.textures[_33ident] = _604.Load(_33ident * 4 + (_567.Load(_559 * 4 + 0) & 16383u) * 76 + 0);
                }
                [unroll]
                for (int _34ident = 0; _34ident < 3; _34ident++)
                {
                    _623.base_color[_34ident] = asfloat(_604.Load(_34ident * 4 + (_567.Load(_559 * 4 + 0) & 16383u) * 76 + 20));
                }
                _623.flags = _604.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 32);
                _623.type = _604.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 36);
                _623.tangent_rotation_or_strength = asfloat(_604.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 40));
                _623.roughness_and_anisotropic = _604.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 44);
                _623.ior = asfloat(_604.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 48));
                _623.sheen_and_sheen_tint = _604.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 52);
                _623.tint_and_metallic = _604.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 56);
                _623.transmission_and_transmission_roughness = _604.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 60);
                _623.specular_and_specular_tint = _604.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 64);
                _623.clearcoat_and_clearcoat_roughness = _604.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 68);
                _623.normal_map_strength_unorm = _604.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 72);
                material_t _624;
                _624.textures[0] = _623.textures[0];
                _624.textures[1] = _623.textures[1];
                _624.textures[2] = _623.textures[2];
                _624.textures[3] = _623.textures[3];
                _624.textures[4] = _623.textures[4];
                _624.base_color[0] = _623.base_color[0];
                _624.base_color[1] = _623.base_color[1];
                _624.base_color[2] = _623.base_color[2];
                _624.flags = _623.flags;
                _624.type = _623.type;
                _624.tangent_rotation_or_strength = _623.tangent_rotation_or_strength;
                _624.roughness_and_anisotropic = _623.roughness_and_anisotropic;
                _624.ior = _623.ior;
                _624.sheen_and_sheen_tint = _623.sheen_and_sheen_tint;
                _624.tint_and_metallic = _623.tint_and_metallic;
                _624.transmission_and_transmission_roughness = _623.transmission_and_transmission_roughness;
                _624.specular_and_specular_tint = _623.specular_and_specular_tint;
                _624.clearcoat_and_clearcoat_roughness = _623.clearcoat_and_clearcoat_roughness;
                _624.normal_map_strength_unorm = _623.normal_map_strength_unorm;
                _1002 = _624.textures[1];
                _1003 = _624.textures[3];
                _1004 = _624.textures[4];
                _1017 = _624.base_color[0];
                _1018 = _624.base_color[1];
                _1019 = _624.base_color[2];
                _962 = _624.type;
                _963 = _624.tangent_rotation_or_strength;
            }
            uint _645 = _559 * 3u;
            vertex_t _651;
            [unroll]
            for (int _35ident = 0; _35ident < 3; _35ident++)
            {
                _651.p[_35ident] = asfloat(_639.Load(_35ident * 4 + _643.Load(_645 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _651.n[_36ident] = asfloat(_639.Load(_36ident * 4 + _643.Load(_645 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _37ident = 0; _37ident < 3; _37ident++)
            {
                _651.b[_37ident] = asfloat(_639.Load(_37ident * 4 + _643.Load(_645 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _38ident = 0; _38ident < 2; _38ident++)
            {
                [unroll]
                for (int _39ident = 0; _39ident < 2; _39ident++)
                {
                    _651.t[_38ident][_39ident] = asfloat(_639.Load(_39ident * 4 + _38ident * 8 + _643.Load(_645 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _652;
            _652.p[0] = _651.p[0];
            _652.p[1] = _651.p[1];
            _652.p[2] = _651.p[2];
            _652.n[0] = _651.n[0];
            _652.n[1] = _651.n[1];
            _652.n[2] = _651.n[2];
            _652.b[0] = _651.b[0];
            _652.b[1] = _651.b[1];
            _652.b[2] = _651.b[2];
            _652.t[0][0] = _651.t[0][0];
            _652.t[0][1] = _651.t[0][1];
            _652.t[1][0] = _651.t[1][0];
            _652.t[1][1] = _651.t[1][1];
            vertex_t _660;
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _660.p[_40ident] = asfloat(_639.Load(_40ident * 4 + _643.Load((_645 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 3; _41ident++)
            {
                _660.n[_41ident] = asfloat(_639.Load(_41ident * 4 + _643.Load((_645 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _42ident = 0; _42ident < 3; _42ident++)
            {
                _660.b[_42ident] = asfloat(_639.Load(_42ident * 4 + _643.Load((_645 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _43ident = 0; _43ident < 2; _43ident++)
            {
                [unroll]
                for (int _44ident = 0; _44ident < 2; _44ident++)
                {
                    _660.t[_43ident][_44ident] = asfloat(_639.Load(_44ident * 4 + _43ident * 8 + _643.Load((_645 + 1u) * 4 + 0) * 52 + 36));
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
            for (int _45ident = 0; _45ident < 3; _45ident++)
            {
                _669.p[_45ident] = asfloat(_639.Load(_45ident * 4 + _643.Load((_645 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _46ident = 0; _46ident < 3; _46ident++)
            {
                _669.n[_46ident] = asfloat(_639.Load(_46ident * 4 + _643.Load((_645 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _47ident = 0; _47ident < 3; _47ident++)
            {
                _669.b[_47ident] = asfloat(_639.Load(_47ident * 4 + _643.Load((_645 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _48ident = 0; _48ident < 2; _48ident++)
            {
                [unroll]
                for (int _49ident = 0; _49ident < 2; _49ident++)
                {
                    _669.t[_48ident][_49ident] = asfloat(_639.Load(_49ident * 4 + _48ident * 8 + _643.Load((_645 + 2u) * 4 + 0) * 52 + 36));
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
            float2 _703 = ((float2(_652.t[0][0], _652.t[0][1]) * ((1.0f - _948) - _949)) + (float2(_661.t[0][0], _661.t[0][1]) * _948)) + (float2(_670.t[0][0], _670.t[0][1]) * _949);
            float trans_r = frac(asfloat(_708.Load(rand_index * 4 + 0)) + _458);
            while (_962 == 4u)
            {
                float mix_val = _963;
                if (_1002 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_1002, _703, 0).x;
                }
                if (trans_r > mix_val)
                {
                    material_t _748;
                    [unroll]
                    for (int _50ident = 0; _50ident < 5; _50ident++)
                    {
                        _748.textures[_50ident] = _604.Load(_50ident * 4 + _1003 * 76 + 0);
                    }
                    [unroll]
                    for (int _51ident = 0; _51ident < 3; _51ident++)
                    {
                        _748.base_color[_51ident] = asfloat(_604.Load(_51ident * 4 + _1003 * 76 + 20));
                    }
                    _748.flags = _604.Load(_1003 * 76 + 32);
                    _748.type = _604.Load(_1003 * 76 + 36);
                    _748.tangent_rotation_or_strength = asfloat(_604.Load(_1003 * 76 + 40));
                    _748.roughness_and_anisotropic = _604.Load(_1003 * 76 + 44);
                    _748.ior = asfloat(_604.Load(_1003 * 76 + 48));
                    _748.sheen_and_sheen_tint = _604.Load(_1003 * 76 + 52);
                    _748.tint_and_metallic = _604.Load(_1003 * 76 + 56);
                    _748.transmission_and_transmission_roughness = _604.Load(_1003 * 76 + 60);
                    _748.specular_and_specular_tint = _604.Load(_1003 * 76 + 64);
                    _748.clearcoat_and_clearcoat_roughness = _604.Load(_1003 * 76 + 68);
                    _748.normal_map_strength_unorm = _604.Load(_1003 * 76 + 72);
                    material_t _749;
                    _749.textures[0] = _748.textures[0];
                    _749.textures[1] = _748.textures[1];
                    _749.textures[2] = _748.textures[2];
                    _749.textures[3] = _748.textures[3];
                    _749.textures[4] = _748.textures[4];
                    _749.base_color[0] = _748.base_color[0];
                    _749.base_color[1] = _748.base_color[1];
                    _749.base_color[2] = _748.base_color[2];
                    _749.flags = _748.flags;
                    _749.type = _748.type;
                    _749.tangent_rotation_or_strength = _748.tangent_rotation_or_strength;
                    _749.roughness_and_anisotropic = _748.roughness_and_anisotropic;
                    _749.ior = _748.ior;
                    _749.sheen_and_sheen_tint = _748.sheen_and_sheen_tint;
                    _749.tint_and_metallic = _748.tint_and_metallic;
                    _749.transmission_and_transmission_roughness = _748.transmission_and_transmission_roughness;
                    _749.specular_and_specular_tint = _748.specular_and_specular_tint;
                    _749.clearcoat_and_clearcoat_roughness = _748.clearcoat_and_clearcoat_roughness;
                    _749.normal_map_strength_unorm = _748.normal_map_strength_unorm;
                    _1002 = _749.textures[1];
                    _1003 = _749.textures[3];
                    _1004 = _749.textures[4];
                    _1017 = _749.base_color[0];
                    _1018 = _749.base_color[1];
                    _1019 = _749.base_color[2];
                    _962 = _749.type;
                    _963 = _749.tangent_rotation_or_strength;
                    trans_r = (trans_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _760;
                    [unroll]
                    for (int _52ident = 0; _52ident < 5; _52ident++)
                    {
                        _760.textures[_52ident] = _604.Load(_52ident * 4 + _1004 * 76 + 0);
                    }
                    [unroll]
                    for (int _53ident = 0; _53ident < 3; _53ident++)
                    {
                        _760.base_color[_53ident] = asfloat(_604.Load(_53ident * 4 + _1004 * 76 + 20));
                    }
                    _760.flags = _604.Load(_1004 * 76 + 32);
                    _760.type = _604.Load(_1004 * 76 + 36);
                    _760.tangent_rotation_or_strength = asfloat(_604.Load(_1004 * 76 + 40));
                    _760.roughness_and_anisotropic = _604.Load(_1004 * 76 + 44);
                    _760.ior = asfloat(_604.Load(_1004 * 76 + 48));
                    _760.sheen_and_sheen_tint = _604.Load(_1004 * 76 + 52);
                    _760.tint_and_metallic = _604.Load(_1004 * 76 + 56);
                    _760.transmission_and_transmission_roughness = _604.Load(_1004 * 76 + 60);
                    _760.specular_and_specular_tint = _604.Load(_1004 * 76 + 64);
                    _760.clearcoat_and_clearcoat_roughness = _604.Load(_1004 * 76 + 68);
                    _760.normal_map_strength_unorm = _604.Load(_1004 * 76 + 72);
                    material_t _761;
                    _761.textures[0] = _760.textures[0];
                    _761.textures[1] = _760.textures[1];
                    _761.textures[2] = _760.textures[2];
                    _761.textures[3] = _760.textures[3];
                    _761.textures[4] = _760.textures[4];
                    _761.base_color[0] = _760.base_color[0];
                    _761.base_color[1] = _760.base_color[1];
                    _761.base_color[2] = _760.base_color[2];
                    _761.flags = _760.flags;
                    _761.type = _760.type;
                    _761.tangent_rotation_or_strength = _760.tangent_rotation_or_strength;
                    _761.roughness_and_anisotropic = _760.roughness_and_anisotropic;
                    _761.ior = _760.ior;
                    _761.sheen_and_sheen_tint = _760.sheen_and_sheen_tint;
                    _761.tint_and_metallic = _760.tint_and_metallic;
                    _761.transmission_and_transmission_roughness = _760.transmission_and_transmission_roughness;
                    _761.specular_and_specular_tint = _760.specular_and_specular_tint;
                    _761.clearcoat_and_clearcoat_roughness = _760.clearcoat_and_clearcoat_roughness;
                    _761.normal_map_strength_unorm = _760.normal_map_strength_unorm;
                    _1002 = _761.textures[1];
                    _1003 = _761.textures[3];
                    _1004 = _761.textures[4];
                    _1017 = _761.base_color[0];
                    _1018 = _761.base_color[1];
                    _1019 = _761.base_color[2];
                    _962 = _761.type;
                    _963 = _761.tangent_rotation_or_strength;
                    trans_r /= mix_val;
                }
            }
            if (_962 != 5u)
            {
                break;
            }
            float _790 = max(asfloat(_390.Load(_367 * 72 + 28)), max(asfloat(_390.Load(_367 * 72 + 32)), asfloat(_390.Load(_367 * 72 + 36))));
            if ((int(_390.Load(_367 * 72 + 68)) >> 24) > _442_g_params.min_transp_depth)
            {
                _802 = max(0.0500000007450580596923828125f, 1.0f - _790);
            }
            else
            {
                _802 = 0.0f;
            }
            bool _816 = (frac(asfloat(_708.Load((rand_index + 6) * 4 + 0)) + _458) < _802) || (_790 == 0.0f);
            bool _828;
            if (!_816)
            {
                _828 = ((int(_390.Load(_367 * 72 + 68)) >> 24) + 1) >= _442_g_params.max_transp_depth;
            }
            else
            {
                _828 = _816;
            }
            if (_828)
            {
                _390.Store(_367 * 72 + 36, asuint(0.0f));
                _390.Store(_367 * 72 + 32, asuint(0.0f));
                _390.Store(_367 * 72 + 28, asuint(0.0f));
                break;
            }
            float _842 = 1.0f - _802;
            _390.Store(_367 * 72 + 28, asuint(asfloat(_390.Load(_367 * 72 + 28)) * (_1017 / _842)));
            _390.Store(_367 * 72 + 32, asuint(asfloat(_390.Load(_367 * 72 + 32)) * (_1018 / _842)));
            _390.Store(_367 * 72 + 36, asuint(asfloat(_390.Load(_367 * 72 + 36)) * (_1019 / _842)));
            ro += (_427 * (_947 + 9.9999997473787516355514526367188e-06f));
            _944 = 0;
            _947 = _482 - _947;
            _390.Store(_367 * 72 + 68, uint(int(_390.Load(_367 * 72 + 68)) + 16777216));
            rand_index += 7;
            continue;
        }
        float _906 = _947;
        float _907 = _906 + length(float3(asfloat(_390.Load(_367 * 72 + 0)), asfloat(_390.Load(_367 * 72 + 4)), asfloat(_390.Load(_367 * 72 + 8))) - ro);
        _947 = _907;
        hit_data_t _956 = { _944, _945, _946, _907, _948, _949 };
        hit_data_t _918;
        _918.mask = _956.mask;
        _918.obj_index = _956.obj_index;
        _918.prim_index = _956.prim_index;
        _918.t = _956.t;
        _918.u = _956.u;
        _918.v = _956.v;
        _913.Store(_367 * 24 + 0, uint(_918.mask));
        _913.Store(_367 * 24 + 4, uint(_918.obj_index));
        _913.Store(_367 * 24 + 8, uint(_918.prim_index));
        _913.Store(_367 * 24 + 12, asuint(_918.t));
        _913.Store(_367 * 24 + 16, asuint(_918.u));
        _913.Store(_367 * 24 + 20, asuint(_918.v));
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
