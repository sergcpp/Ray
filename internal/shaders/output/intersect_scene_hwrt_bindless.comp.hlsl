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

RWByteAddressBuffer _414 : register(u12, space0);
ByteAddressBuffer _574 : register(t5, space0);
ByteAddressBuffer _612 : register(t6, space0);
ByteAddressBuffer _647 : register(t1, space0);
ByteAddressBuffer _651 : register(t2, space0);
ByteAddressBuffer _716 : register(t15, space0);
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
        int _404 = int(gl_GlobalInvocationID.y * _365_g_params.rect.z) + int(_365_g_params.rect.x + gl_GlobalInvocationID.x);
        float3 ro = float3(asfloat(_414.Load(_404 * 72 + 0)), asfloat(_414.Load(_404 * 72 + 4)), asfloat(_414.Load(_404 * 72 + 8)));
        float _430 = asfloat(_414.Load(_404 * 72 + 12));
        float _433 = asfloat(_414.Load(_404 * 72 + 16));
        float _436 = asfloat(_414.Load(_404 * 72 + 20));
        float3 _437 = float3(_430, _433, _436);
        float3 param = _437;
        int _952 = 0;
        int _954 = 0;
        int _953 = 0;
        float _955 = _365_g_params.inter_t;
        float _957 = 0.0f;
        float _956 = 0.0f;
        uint param_1 = uint(hash(int(_414.Load(_404 * 72 + 64))));
        float _465 = construct_float(param_1);
        ray_data_t _473;
        [unroll]
        for (int _27ident = 0; _27ident < 3; _27ident++)
        {
            _473.o[_27ident] = asfloat(_414.Load(_27ident * 4 + _404 * 72 + 0));
        }
        [unroll]
        for (int _28ident = 0; _28ident < 3; _28ident++)
        {
            _473.d[_28ident] = asfloat(_414.Load(_28ident * 4 + _404 * 72 + 12));
        }
        _473.pdf = asfloat(_414.Load(_404 * 72 + 24));
        [unroll]
        for (int _29ident = 0; _29ident < 3; _29ident++)
        {
            _473.c[_29ident] = asfloat(_414.Load(_29ident * 4 + _404 * 72 + 28));
        }
        [unroll]
        for (int _30ident = 0; _30ident < 4; _30ident++)
        {
            _473.ior[_30ident] = asfloat(_414.Load(_30ident * 4 + _404 * 72 + 40));
        }
        _473.cone_width = asfloat(_414.Load(_404 * 72 + 56));
        _473.cone_spread = asfloat(_414.Load(_404 * 72 + 60));
        _473.xy = int(_414.Load(_404 * 72 + 64));
        _473.depth = int(_414.Load(_404 * 72 + 68));
        ray_data_t _476;
        _476.o[0] = _473.o[0];
        _476.o[1] = _473.o[1];
        _476.o[2] = _473.o[2];
        _476.d[0] = _473.d[0];
        _476.d[1] = _473.d[1];
        _476.d[2] = _473.d[2];
        _476.pdf = _473.pdf;
        _476.c[0] = _473.c[0];
        _476.c[1] = _473.c[1];
        _476.c[2] = _473.c[2];
        _476.ior[0] = _473.ior[0];
        _476.ior[1] = _473.ior[1];
        _476.ior[2] = _473.ior[2];
        _476.ior[3] = _473.ior[3];
        _476.cone_width = _473.cone_width;
        _476.cone_spread = _473.cone_spread;
        _476.xy = _473.xy;
        _476.depth = _473.depth;
        int rand_index = _365_g_params.hi + (total_depth(_476) * 7);
        int _555;
        float _810;
        float _489;
        for (;;)
        {
            _489 = _955;
            rayQueryInitializeEXT(rq, g_tlas, 0u, 255u, ro, 0.0f, _437, _489);
            for (;;)
            {
                bool _507 = rayQueryProceedEXT(rq);
                if (_507)
                {
                    rayQueryConfirmIntersectionEXT(rq);
                    continue;
                }
                else
                {
                    break;
                }
            }
            uint _508 = rayQueryGetIntersectionTypeEXT(rq, bool(1));
            if (_508 != 0u)
            {
                int _513 = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, bool(1));
                _952 = -1;
                int _516 = rayQueryGetIntersectionInstanceIdEXT(rq, bool(1));
                _953 = _516;
                int _519 = rayQueryGetIntersectionPrimitiveIndexEXT(rq, bool(1));
                _954 = _513 + _519;
                bool _522 = rayQueryGetIntersectionFrontFaceEXT(rq, bool(1));
                [flatten]
                if (_522 == false)
                {
                    _954 = (-1) - _954;
                }
                float2 _533 = rayQueryGetIntersectionBarycentricsEXT(rq, bool(1));
                _956 = _533.x;
                _957 = _533.y;
                float _540 = rayQueryGetIntersectionTEXT(rq, bool(1));
                _955 = _540;
            }
            if (_952 == 0)
            {
                break;
            }
            bool _552 = _954 < 0;
            if (_552)
            {
                _555 = (-1) - _954;
            }
            else
            {
                _555 = _954;
            }
            uint _566 = uint(_555);
            bool _568 = !_552;
            bool _584;
            if (_568)
            {
                _584 = ((_574.Load(_566 * 4 + 0) >> 16u) & 32768u) != 0u;
            }
            else
            {
                _584 = _568;
            }
            bool _597;
            if (!_584)
            {
                bool _596;
                if (_552)
                {
                    _596 = (_574.Load(_566 * 4 + 0) & 32768u) != 0u;
                }
                else
                {
                    _596 = _552;
                }
                _597 = _596;
            }
            else
            {
                _597 = _584;
            }
            if (_597)
            {
                break;
            }
            material_t _621;
            [unroll]
            for (int _31ident = 0; _31ident < 5; _31ident++)
            {
                _621.textures[_31ident] = _612.Load(_31ident * 4 + ((_574.Load(_566 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _32ident = 0; _32ident < 3; _32ident++)
            {
                _621.base_color[_32ident] = asfloat(_612.Load(_32ident * 4 + ((_574.Load(_566 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
            }
            _621.flags = _612.Load(((_574.Load(_566 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
            _621.type = _612.Load(((_574.Load(_566 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
            _621.tangent_rotation_or_strength = asfloat(_612.Load(((_574.Load(_566 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
            _621.roughness_and_anisotropic = _612.Load(((_574.Load(_566 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
            _621.ior = asfloat(_612.Load(((_574.Load(_566 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
            _621.sheen_and_sheen_tint = _612.Load(((_574.Load(_566 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
            _621.tint_and_metallic = _612.Load(((_574.Load(_566 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
            _621.transmission_and_transmission_roughness = _612.Load(((_574.Load(_566 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
            _621.specular_and_specular_tint = _612.Load(((_574.Load(_566 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
            _621.clearcoat_and_clearcoat_roughness = _612.Load(((_574.Load(_566 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
            _621.normal_map_strength_unorm = _612.Load(((_574.Load(_566 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
            material_t _622;
            _622.textures[0] = _621.textures[0];
            _622.textures[1] = _621.textures[1];
            _622.textures[2] = _621.textures[2];
            _622.textures[3] = _621.textures[3];
            _622.textures[4] = _621.textures[4];
            _622.base_color[0] = _621.base_color[0];
            _622.base_color[1] = _621.base_color[1];
            _622.base_color[2] = _621.base_color[2];
            _622.flags = _621.flags;
            _622.type = _621.type;
            _622.tangent_rotation_or_strength = _621.tangent_rotation_or_strength;
            _622.roughness_and_anisotropic = _621.roughness_and_anisotropic;
            _622.ior = _621.ior;
            _622.sheen_and_sheen_tint = _621.sheen_and_sheen_tint;
            _622.tint_and_metallic = _621.tint_and_metallic;
            _622.transmission_and_transmission_roughness = _621.transmission_and_transmission_roughness;
            _622.specular_and_specular_tint = _621.specular_and_specular_tint;
            _622.clearcoat_and_clearcoat_roughness = _621.clearcoat_and_clearcoat_roughness;
            _622.normal_map_strength_unorm = _621.normal_map_strength_unorm;
            uint _1010 = _622.textures[1];
            uint _1011 = _622.textures[3];
            uint _1012 = _622.textures[4];
            float _1025 = _622.base_color[0];
            float _1026 = _622.base_color[1];
            float _1027 = _622.base_color[2];
            uint _970 = _622.type;
            float _971 = _622.tangent_rotation_or_strength;
            if (_552)
            {
                material_t _631;
                [unroll]
                for (int _33ident = 0; _33ident < 5; _33ident++)
                {
                    _631.textures[_33ident] = _612.Load(_33ident * 4 + (_574.Load(_566 * 4 + 0) & 16383u) * 76 + 0);
                }
                [unroll]
                for (int _34ident = 0; _34ident < 3; _34ident++)
                {
                    _631.base_color[_34ident] = asfloat(_612.Load(_34ident * 4 + (_574.Load(_566 * 4 + 0) & 16383u) * 76 + 20));
                }
                _631.flags = _612.Load((_574.Load(_566 * 4 + 0) & 16383u) * 76 + 32);
                _631.type = _612.Load((_574.Load(_566 * 4 + 0) & 16383u) * 76 + 36);
                _631.tangent_rotation_or_strength = asfloat(_612.Load((_574.Load(_566 * 4 + 0) & 16383u) * 76 + 40));
                _631.roughness_and_anisotropic = _612.Load((_574.Load(_566 * 4 + 0) & 16383u) * 76 + 44);
                _631.ior = asfloat(_612.Load((_574.Load(_566 * 4 + 0) & 16383u) * 76 + 48));
                _631.sheen_and_sheen_tint = _612.Load((_574.Load(_566 * 4 + 0) & 16383u) * 76 + 52);
                _631.tint_and_metallic = _612.Load((_574.Load(_566 * 4 + 0) & 16383u) * 76 + 56);
                _631.transmission_and_transmission_roughness = _612.Load((_574.Load(_566 * 4 + 0) & 16383u) * 76 + 60);
                _631.specular_and_specular_tint = _612.Load((_574.Load(_566 * 4 + 0) & 16383u) * 76 + 64);
                _631.clearcoat_and_clearcoat_roughness = _612.Load((_574.Load(_566 * 4 + 0) & 16383u) * 76 + 68);
                _631.normal_map_strength_unorm = _612.Load((_574.Load(_566 * 4 + 0) & 16383u) * 76 + 72);
                material_t _632;
                _632.textures[0] = _631.textures[0];
                _632.textures[1] = _631.textures[1];
                _632.textures[2] = _631.textures[2];
                _632.textures[3] = _631.textures[3];
                _632.textures[4] = _631.textures[4];
                _632.base_color[0] = _631.base_color[0];
                _632.base_color[1] = _631.base_color[1];
                _632.base_color[2] = _631.base_color[2];
                _632.flags = _631.flags;
                _632.type = _631.type;
                _632.tangent_rotation_or_strength = _631.tangent_rotation_or_strength;
                _632.roughness_and_anisotropic = _631.roughness_and_anisotropic;
                _632.ior = _631.ior;
                _632.sheen_and_sheen_tint = _631.sheen_and_sheen_tint;
                _632.tint_and_metallic = _631.tint_and_metallic;
                _632.transmission_and_transmission_roughness = _631.transmission_and_transmission_roughness;
                _632.specular_and_specular_tint = _631.specular_and_specular_tint;
                _632.clearcoat_and_clearcoat_roughness = _631.clearcoat_and_clearcoat_roughness;
                _632.normal_map_strength_unorm = _631.normal_map_strength_unorm;
                _1010 = _632.textures[1];
                _1011 = _632.textures[3];
                _1012 = _632.textures[4];
                _1025 = _632.base_color[0];
                _1026 = _632.base_color[1];
                _1027 = _632.base_color[2];
                _970 = _632.type;
                _971 = _632.tangent_rotation_or_strength;
            }
            uint _653 = _566 * 3u;
            vertex_t _659;
            [unroll]
            for (int _35ident = 0; _35ident < 3; _35ident++)
            {
                _659.p[_35ident] = asfloat(_647.Load(_35ident * 4 + _651.Load(_653 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _659.n[_36ident] = asfloat(_647.Load(_36ident * 4 + _651.Load(_653 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _37ident = 0; _37ident < 3; _37ident++)
            {
                _659.b[_37ident] = asfloat(_647.Load(_37ident * 4 + _651.Load(_653 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _38ident = 0; _38ident < 2; _38ident++)
            {
                [unroll]
                for (int _39ident = 0; _39ident < 2; _39ident++)
                {
                    _659.t[_38ident][_39ident] = asfloat(_647.Load(_39ident * 4 + _38ident * 8 + _651.Load(_653 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _660;
            _660.p[0] = _659.p[0];
            _660.p[1] = _659.p[1];
            _660.p[2] = _659.p[2];
            _660.n[0] = _659.n[0];
            _660.n[1] = _659.n[1];
            _660.n[2] = _659.n[2];
            _660.b[0] = _659.b[0];
            _660.b[1] = _659.b[1];
            _660.b[2] = _659.b[2];
            _660.t[0][0] = _659.t[0][0];
            _660.t[0][1] = _659.t[0][1];
            _660.t[1][0] = _659.t[1][0];
            _660.t[1][1] = _659.t[1][1];
            vertex_t _668;
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _668.p[_40ident] = asfloat(_647.Load(_40ident * 4 + _651.Load((_653 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 3; _41ident++)
            {
                _668.n[_41ident] = asfloat(_647.Load(_41ident * 4 + _651.Load((_653 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _42ident = 0; _42ident < 3; _42ident++)
            {
                _668.b[_42ident] = asfloat(_647.Load(_42ident * 4 + _651.Load((_653 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _43ident = 0; _43ident < 2; _43ident++)
            {
                [unroll]
                for (int _44ident = 0; _44ident < 2; _44ident++)
                {
                    _668.t[_43ident][_44ident] = asfloat(_647.Load(_44ident * 4 + _43ident * 8 + _651.Load((_653 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _669;
            _669.p[0] = _668.p[0];
            _669.p[1] = _668.p[1];
            _669.p[2] = _668.p[2];
            _669.n[0] = _668.n[0];
            _669.n[1] = _668.n[1];
            _669.n[2] = _668.n[2];
            _669.b[0] = _668.b[0];
            _669.b[1] = _668.b[1];
            _669.b[2] = _668.b[2];
            _669.t[0][0] = _668.t[0][0];
            _669.t[0][1] = _668.t[0][1];
            _669.t[1][0] = _668.t[1][0];
            _669.t[1][1] = _668.t[1][1];
            vertex_t _677;
            [unroll]
            for (int _45ident = 0; _45ident < 3; _45ident++)
            {
                _677.p[_45ident] = asfloat(_647.Load(_45ident * 4 + _651.Load((_653 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _46ident = 0; _46ident < 3; _46ident++)
            {
                _677.n[_46ident] = asfloat(_647.Load(_46ident * 4 + _651.Load((_653 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _47ident = 0; _47ident < 3; _47ident++)
            {
                _677.b[_47ident] = asfloat(_647.Load(_47ident * 4 + _651.Load((_653 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _48ident = 0; _48ident < 2; _48ident++)
            {
                [unroll]
                for (int _49ident = 0; _49ident < 2; _49ident++)
                {
                    _677.t[_48ident][_49ident] = asfloat(_647.Load(_49ident * 4 + _48ident * 8 + _651.Load((_653 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _678;
            _678.p[0] = _677.p[0];
            _678.p[1] = _677.p[1];
            _678.p[2] = _677.p[2];
            _678.n[0] = _677.n[0];
            _678.n[1] = _677.n[1];
            _678.n[2] = _677.n[2];
            _678.b[0] = _677.b[0];
            _678.b[1] = _677.b[1];
            _678.b[2] = _677.b[2];
            _678.t[0][0] = _677.t[0][0];
            _678.t[0][1] = _677.t[0][1];
            _678.t[1][0] = _677.t[1][0];
            _678.t[1][1] = _677.t[1][1];
            float2 _711 = ((float2(_660.t[0][0], _660.t[0][1]) * ((1.0f - _956) - _957)) + (float2(_669.t[0][0], _669.t[0][1]) * _956)) + (float2(_678.t[0][0], _678.t[0][1]) * _957);
            float trans_r = frac(asfloat(_716.Load(rand_index * 4 + 0)) + _465);
            while (_970 == 4u)
            {
                float mix_val = _971;
                if (_1010 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_1010, _711, 0).x;
                }
                if (trans_r > mix_val)
                {
                    material_t _756;
                    [unroll]
                    for (int _50ident = 0; _50ident < 5; _50ident++)
                    {
                        _756.textures[_50ident] = _612.Load(_50ident * 4 + _1011 * 76 + 0);
                    }
                    [unroll]
                    for (int _51ident = 0; _51ident < 3; _51ident++)
                    {
                        _756.base_color[_51ident] = asfloat(_612.Load(_51ident * 4 + _1011 * 76 + 20));
                    }
                    _756.flags = _612.Load(_1011 * 76 + 32);
                    _756.type = _612.Load(_1011 * 76 + 36);
                    _756.tangent_rotation_or_strength = asfloat(_612.Load(_1011 * 76 + 40));
                    _756.roughness_and_anisotropic = _612.Load(_1011 * 76 + 44);
                    _756.ior = asfloat(_612.Load(_1011 * 76 + 48));
                    _756.sheen_and_sheen_tint = _612.Load(_1011 * 76 + 52);
                    _756.tint_and_metallic = _612.Load(_1011 * 76 + 56);
                    _756.transmission_and_transmission_roughness = _612.Load(_1011 * 76 + 60);
                    _756.specular_and_specular_tint = _612.Load(_1011 * 76 + 64);
                    _756.clearcoat_and_clearcoat_roughness = _612.Load(_1011 * 76 + 68);
                    _756.normal_map_strength_unorm = _612.Load(_1011 * 76 + 72);
                    material_t _757;
                    _757.textures[0] = _756.textures[0];
                    _757.textures[1] = _756.textures[1];
                    _757.textures[2] = _756.textures[2];
                    _757.textures[3] = _756.textures[3];
                    _757.textures[4] = _756.textures[4];
                    _757.base_color[0] = _756.base_color[0];
                    _757.base_color[1] = _756.base_color[1];
                    _757.base_color[2] = _756.base_color[2];
                    _757.flags = _756.flags;
                    _757.type = _756.type;
                    _757.tangent_rotation_or_strength = _756.tangent_rotation_or_strength;
                    _757.roughness_and_anisotropic = _756.roughness_and_anisotropic;
                    _757.ior = _756.ior;
                    _757.sheen_and_sheen_tint = _756.sheen_and_sheen_tint;
                    _757.tint_and_metallic = _756.tint_and_metallic;
                    _757.transmission_and_transmission_roughness = _756.transmission_and_transmission_roughness;
                    _757.specular_and_specular_tint = _756.specular_and_specular_tint;
                    _757.clearcoat_and_clearcoat_roughness = _756.clearcoat_and_clearcoat_roughness;
                    _757.normal_map_strength_unorm = _756.normal_map_strength_unorm;
                    _1010 = _757.textures[1];
                    _1011 = _757.textures[3];
                    _1012 = _757.textures[4];
                    _1025 = _757.base_color[0];
                    _1026 = _757.base_color[1];
                    _1027 = _757.base_color[2];
                    _970 = _757.type;
                    _971 = _757.tangent_rotation_or_strength;
                    trans_r = (trans_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _768;
                    [unroll]
                    for (int _52ident = 0; _52ident < 5; _52ident++)
                    {
                        _768.textures[_52ident] = _612.Load(_52ident * 4 + _1012 * 76 + 0);
                    }
                    [unroll]
                    for (int _53ident = 0; _53ident < 3; _53ident++)
                    {
                        _768.base_color[_53ident] = asfloat(_612.Load(_53ident * 4 + _1012 * 76 + 20));
                    }
                    _768.flags = _612.Load(_1012 * 76 + 32);
                    _768.type = _612.Load(_1012 * 76 + 36);
                    _768.tangent_rotation_or_strength = asfloat(_612.Load(_1012 * 76 + 40));
                    _768.roughness_and_anisotropic = _612.Load(_1012 * 76 + 44);
                    _768.ior = asfloat(_612.Load(_1012 * 76 + 48));
                    _768.sheen_and_sheen_tint = _612.Load(_1012 * 76 + 52);
                    _768.tint_and_metallic = _612.Load(_1012 * 76 + 56);
                    _768.transmission_and_transmission_roughness = _612.Load(_1012 * 76 + 60);
                    _768.specular_and_specular_tint = _612.Load(_1012 * 76 + 64);
                    _768.clearcoat_and_clearcoat_roughness = _612.Load(_1012 * 76 + 68);
                    _768.normal_map_strength_unorm = _612.Load(_1012 * 76 + 72);
                    material_t _769;
                    _769.textures[0] = _768.textures[0];
                    _769.textures[1] = _768.textures[1];
                    _769.textures[2] = _768.textures[2];
                    _769.textures[3] = _768.textures[3];
                    _769.textures[4] = _768.textures[4];
                    _769.base_color[0] = _768.base_color[0];
                    _769.base_color[1] = _768.base_color[1];
                    _769.base_color[2] = _768.base_color[2];
                    _769.flags = _768.flags;
                    _769.type = _768.type;
                    _769.tangent_rotation_or_strength = _768.tangent_rotation_or_strength;
                    _769.roughness_and_anisotropic = _768.roughness_and_anisotropic;
                    _769.ior = _768.ior;
                    _769.sheen_and_sheen_tint = _768.sheen_and_sheen_tint;
                    _769.tint_and_metallic = _768.tint_and_metallic;
                    _769.transmission_and_transmission_roughness = _768.transmission_and_transmission_roughness;
                    _769.specular_and_specular_tint = _768.specular_and_specular_tint;
                    _769.clearcoat_and_clearcoat_roughness = _768.clearcoat_and_clearcoat_roughness;
                    _769.normal_map_strength_unorm = _768.normal_map_strength_unorm;
                    _1010 = _769.textures[1];
                    _1011 = _769.textures[3];
                    _1012 = _769.textures[4];
                    _1025 = _769.base_color[0];
                    _1026 = _769.base_color[1];
                    _1027 = _769.base_color[2];
                    _970 = _769.type;
                    _971 = _769.tangent_rotation_or_strength;
                    trans_r /= mix_val;
                }
            }
            if (_970 != 5u)
            {
                break;
            }
            float _798 = max(asfloat(_414.Load(_404 * 72 + 28)), max(asfloat(_414.Load(_404 * 72 + 32)), asfloat(_414.Load(_404 * 72 + 36))));
            if ((int(_414.Load(_404 * 72 + 68)) >> 24) > _365_g_params.min_transp_depth)
            {
                _810 = max(0.0500000007450580596923828125f, 1.0f - _798);
            }
            else
            {
                _810 = 0.0f;
            }
            bool _824 = (frac(asfloat(_716.Load((rand_index + 6) * 4 + 0)) + _465) < _810) || (_798 == 0.0f);
            bool _836;
            if (!_824)
            {
                _836 = ((int(_414.Load(_404 * 72 + 68)) >> 24) + 1) >= _365_g_params.max_transp_depth;
            }
            else
            {
                _836 = _824;
            }
            if (_836)
            {
                _414.Store(_404 * 72 + 36, asuint(0.0f));
                _414.Store(_404 * 72 + 32, asuint(0.0f));
                _414.Store(_404 * 72 + 28, asuint(0.0f));
                break;
            }
            float _850 = 1.0f - _810;
            _414.Store(_404 * 72 + 28, asuint(asfloat(_414.Load(_404 * 72 + 28)) * (_1025 / _850)));
            _414.Store(_404 * 72 + 32, asuint(asfloat(_414.Load(_404 * 72 + 32)) * (_1026 / _850)));
            _414.Store(_404 * 72 + 36, asuint(asfloat(_414.Load(_404 * 72 + 36)) * (_1027 / _850)));
            ro += (_437 * (_955 + 9.9999997473787516355514526367188e-06f));
            _952 = 0;
            _955 = _489 - _955;
            _414.Store(_404 * 72 + 68, uint(int(_414.Load(_404 * 72 + 68)) + 16777216));
            rand_index += 7;
            continue;
        }
        float _914 = _955;
        float _915 = _914 + length(float3(asfloat(_414.Load(_404 * 72 + 0)), asfloat(_414.Load(_404 * 72 + 4)), asfloat(_414.Load(_404 * 72 + 8))) - ro);
        _955 = _915;
        hit_data_t _964 = { _952, _953, _954, _915, _956, _957 };
        hit_data_t _926;
        _926.mask = _964.mask;
        _926.obj_index = _964.obj_index;
        _926.prim_index = _964.prim_index;
        _926.t = _964.t;
        _926.u = _964.u;
        _926.v = _964.v;
        _921.Store(_404 * 24 + 0, uint(_926.mask));
        _921.Store(_404 * 24 + 4, uint(_926.obj_index));
        _921.Store(_404 * 24 + 8, uint(_926.prim_index));
        _921.Store(_404 * 24 + 12, asuint(_926.t));
        _921.Store(_404 * 24 + 16, asuint(_926.u));
        _921.Store(_404 * 24 + 20, asuint(_926.v));
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
