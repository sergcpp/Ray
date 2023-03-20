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
    uint2 img_size;
    uint node_index;
    float cam_clip_end;
    int min_transp_depth;
    int max_transp_depth;
    int hi;
    int pad2;
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

RWByteAddressBuffer _407 : register(u12, space0);
ByteAddressBuffer _567 : register(t5, space0);
ByteAddressBuffer _605 : register(t6, space0);
ByteAddressBuffer _640 : register(t1, space0);
ByteAddressBuffer _644 : register(t2, space0);
ByteAddressBuffer _709 : register(t15, space0);
RWByteAddressBuffer _914 : register(u0, space0);
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
        float3 _1064 = inv_v;
        _1064.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _1064;
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
            float3 _1062 = inv_v;
            _1062.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _1062;
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
        float3 _1068 = inv_v;
        _1068.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _1068;
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
            float3 _1066 = inv_v;
            _1066.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _1066;
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
        float3 _1072 = inv_v;
        _1072.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _1072;
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
            float3 _1070 = inv_v;
            _1070.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _1070;
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
        float4 _1088 = res;
        _1088.x = _341.x;
        float4 _1090 = _1088;
        _1090.y = _341.y;
        float4 _1092 = _1090;
        _1092.z = _341.z;
        res = _1092;
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
        bool _369 = gl_GlobalInvocationID.x >= _365_g_params.img_size.x;
        bool _378;
        if (!_369)
        {
            _378 = gl_GlobalInvocationID.y >= _365_g_params.img_size.y;
        }
        else
        {
            _378 = _369;
        }
        if (_378)
        {
            break;
        }
        int _397 = (int(gl_GlobalInvocationID.y) * int(_365_g_params.img_size.x)) + int(gl_GlobalInvocationID.x);
        float3 ro = float3(asfloat(_407.Load(_397 * 72 + 0)), asfloat(_407.Load(_397 * 72 + 4)), asfloat(_407.Load(_397 * 72 + 8)));
        float _423 = asfloat(_407.Load(_397 * 72 + 12));
        float _426 = asfloat(_407.Load(_397 * 72 + 16));
        float _429 = asfloat(_407.Load(_397 * 72 + 20));
        float3 _430 = float3(_423, _426, _429);
        float3 param = _430;
        int _945 = 0;
        int _947 = 0;
        int _946 = 0;
        float _948 = _365_g_params.cam_clip_end;
        float _950 = 0.0f;
        float _949 = 0.0f;
        uint param_1 = uint(hash(int(_407.Load(_397 * 72 + 64))));
        float _458 = construct_float(param_1);
        ray_data_t _466;
        [unroll]
        for (int _27ident = 0; _27ident < 3; _27ident++)
        {
            _466.o[_27ident] = asfloat(_407.Load(_27ident * 4 + _397 * 72 + 0));
        }
        [unroll]
        for (int _28ident = 0; _28ident < 3; _28ident++)
        {
            _466.d[_28ident] = asfloat(_407.Load(_28ident * 4 + _397 * 72 + 12));
        }
        _466.pdf = asfloat(_407.Load(_397 * 72 + 24));
        [unroll]
        for (int _29ident = 0; _29ident < 3; _29ident++)
        {
            _466.c[_29ident] = asfloat(_407.Load(_29ident * 4 + _397 * 72 + 28));
        }
        [unroll]
        for (int _30ident = 0; _30ident < 4; _30ident++)
        {
            _466.ior[_30ident] = asfloat(_407.Load(_30ident * 4 + _397 * 72 + 40));
        }
        _466.cone_width = asfloat(_407.Load(_397 * 72 + 56));
        _466.cone_spread = asfloat(_407.Load(_397 * 72 + 60));
        _466.xy = int(_407.Load(_397 * 72 + 64));
        _466.depth = int(_407.Load(_397 * 72 + 68));
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
        int rand_index = _365_g_params.hi + (total_depth(_469) * 7);
        int _548;
        float _803;
        float _482;
        for (;;)
        {
            _482 = _948;
            rayQueryInitializeEXT(rq, g_tlas, 0u, 255u, ro, 0.0f, _430, _482);
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
                _945 = -1;
                int _509 = rayQueryGetIntersectionInstanceIdEXT(rq, bool(1));
                _946 = _509;
                int _512 = rayQueryGetIntersectionPrimitiveIndexEXT(rq, bool(1));
                _947 = _506 + _512;
                bool _515 = rayQueryGetIntersectionFrontFaceEXT(rq, bool(1));
                [flatten]
                if (_515 == false)
                {
                    _947 = (-1) - _947;
                }
                float2 _526 = rayQueryGetIntersectionBarycentricsEXT(rq, bool(1));
                _949 = _526.x;
                _950 = _526.y;
                float _533 = rayQueryGetIntersectionTEXT(rq, bool(1));
                _948 = _533;
            }
            if (_945 == 0)
            {
                break;
            }
            bool _545 = _947 < 0;
            if (_545)
            {
                _548 = (-1) - _947;
            }
            else
            {
                _548 = _947;
            }
            uint _559 = uint(_548);
            bool _561 = !_545;
            bool _577;
            if (_561)
            {
                _577 = ((_567.Load(_559 * 4 + 0) >> 16u) & 32768u) != 0u;
            }
            else
            {
                _577 = _561;
            }
            bool _590;
            if (!_577)
            {
                bool _589;
                if (_545)
                {
                    _589 = (_567.Load(_559 * 4 + 0) & 32768u) != 0u;
                }
                else
                {
                    _589 = _545;
                }
                _590 = _589;
            }
            else
            {
                _590 = _577;
            }
            if (_590)
            {
                break;
            }
            material_t _614;
            [unroll]
            for (int _31ident = 0; _31ident < 5; _31ident++)
            {
                _614.textures[_31ident] = _605.Load(_31ident * 4 + ((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _32ident = 0; _32ident < 3; _32ident++)
            {
                _614.base_color[_32ident] = asfloat(_605.Load(_32ident * 4 + ((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
            }
            _614.flags = _605.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
            _614.type = _605.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
            _614.tangent_rotation_or_strength = asfloat(_605.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
            _614.roughness_and_anisotropic = _605.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
            _614.ior = asfloat(_605.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
            _614.sheen_and_sheen_tint = _605.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
            _614.tint_and_metallic = _605.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
            _614.transmission_and_transmission_roughness = _605.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
            _614.specular_and_specular_tint = _605.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
            _614.clearcoat_and_clearcoat_roughness = _605.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
            _614.normal_map_strength_unorm = _605.Load(((_567.Load(_559 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
            material_t _615;
            _615.textures[0] = _614.textures[0];
            _615.textures[1] = _614.textures[1];
            _615.textures[2] = _614.textures[2];
            _615.textures[3] = _614.textures[3];
            _615.textures[4] = _614.textures[4];
            _615.base_color[0] = _614.base_color[0];
            _615.base_color[1] = _614.base_color[1];
            _615.base_color[2] = _614.base_color[2];
            _615.flags = _614.flags;
            _615.type = _614.type;
            _615.tangent_rotation_or_strength = _614.tangent_rotation_or_strength;
            _615.roughness_and_anisotropic = _614.roughness_and_anisotropic;
            _615.ior = _614.ior;
            _615.sheen_and_sheen_tint = _614.sheen_and_sheen_tint;
            _615.tint_and_metallic = _614.tint_and_metallic;
            _615.transmission_and_transmission_roughness = _614.transmission_and_transmission_roughness;
            _615.specular_and_specular_tint = _614.specular_and_specular_tint;
            _615.clearcoat_and_clearcoat_roughness = _614.clearcoat_and_clearcoat_roughness;
            _615.normal_map_strength_unorm = _614.normal_map_strength_unorm;
            uint _1003 = _615.textures[1];
            uint _1004 = _615.textures[3];
            uint _1005 = _615.textures[4];
            float _1018 = _615.base_color[0];
            float _1019 = _615.base_color[1];
            float _1020 = _615.base_color[2];
            uint _963 = _615.type;
            float _964 = _615.tangent_rotation_or_strength;
            if (_545)
            {
                material_t _624;
                [unroll]
                for (int _33ident = 0; _33ident < 5; _33ident++)
                {
                    _624.textures[_33ident] = _605.Load(_33ident * 4 + (_567.Load(_559 * 4 + 0) & 16383u) * 76 + 0);
                }
                [unroll]
                for (int _34ident = 0; _34ident < 3; _34ident++)
                {
                    _624.base_color[_34ident] = asfloat(_605.Load(_34ident * 4 + (_567.Load(_559 * 4 + 0) & 16383u) * 76 + 20));
                }
                _624.flags = _605.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 32);
                _624.type = _605.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 36);
                _624.tangent_rotation_or_strength = asfloat(_605.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 40));
                _624.roughness_and_anisotropic = _605.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 44);
                _624.ior = asfloat(_605.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 48));
                _624.sheen_and_sheen_tint = _605.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 52);
                _624.tint_and_metallic = _605.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 56);
                _624.transmission_and_transmission_roughness = _605.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 60);
                _624.specular_and_specular_tint = _605.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 64);
                _624.clearcoat_and_clearcoat_roughness = _605.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 68);
                _624.normal_map_strength_unorm = _605.Load((_567.Load(_559 * 4 + 0) & 16383u) * 76 + 72);
                material_t _625;
                _625.textures[0] = _624.textures[0];
                _625.textures[1] = _624.textures[1];
                _625.textures[2] = _624.textures[2];
                _625.textures[3] = _624.textures[3];
                _625.textures[4] = _624.textures[4];
                _625.base_color[0] = _624.base_color[0];
                _625.base_color[1] = _624.base_color[1];
                _625.base_color[2] = _624.base_color[2];
                _625.flags = _624.flags;
                _625.type = _624.type;
                _625.tangent_rotation_or_strength = _624.tangent_rotation_or_strength;
                _625.roughness_and_anisotropic = _624.roughness_and_anisotropic;
                _625.ior = _624.ior;
                _625.sheen_and_sheen_tint = _624.sheen_and_sheen_tint;
                _625.tint_and_metallic = _624.tint_and_metallic;
                _625.transmission_and_transmission_roughness = _624.transmission_and_transmission_roughness;
                _625.specular_and_specular_tint = _624.specular_and_specular_tint;
                _625.clearcoat_and_clearcoat_roughness = _624.clearcoat_and_clearcoat_roughness;
                _625.normal_map_strength_unorm = _624.normal_map_strength_unorm;
                _1003 = _625.textures[1];
                _1004 = _625.textures[3];
                _1005 = _625.textures[4];
                _1018 = _625.base_color[0];
                _1019 = _625.base_color[1];
                _1020 = _625.base_color[2];
                _963 = _625.type;
                _964 = _625.tangent_rotation_or_strength;
            }
            uint _646 = _559 * 3u;
            vertex_t _652;
            [unroll]
            for (int _35ident = 0; _35ident < 3; _35ident++)
            {
                _652.p[_35ident] = asfloat(_640.Load(_35ident * 4 + _644.Load(_646 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _652.n[_36ident] = asfloat(_640.Load(_36ident * 4 + _644.Load(_646 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _37ident = 0; _37ident < 3; _37ident++)
            {
                _652.b[_37ident] = asfloat(_640.Load(_37ident * 4 + _644.Load(_646 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _38ident = 0; _38ident < 2; _38ident++)
            {
                [unroll]
                for (int _39ident = 0; _39ident < 2; _39ident++)
                {
                    _652.t[_38ident][_39ident] = asfloat(_640.Load(_39ident * 4 + _38ident * 8 + _644.Load(_646 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _653;
            _653.p[0] = _652.p[0];
            _653.p[1] = _652.p[1];
            _653.p[2] = _652.p[2];
            _653.n[0] = _652.n[0];
            _653.n[1] = _652.n[1];
            _653.n[2] = _652.n[2];
            _653.b[0] = _652.b[0];
            _653.b[1] = _652.b[1];
            _653.b[2] = _652.b[2];
            _653.t[0][0] = _652.t[0][0];
            _653.t[0][1] = _652.t[0][1];
            _653.t[1][0] = _652.t[1][0];
            _653.t[1][1] = _652.t[1][1];
            vertex_t _661;
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _661.p[_40ident] = asfloat(_640.Load(_40ident * 4 + _644.Load((_646 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 3; _41ident++)
            {
                _661.n[_41ident] = asfloat(_640.Load(_41ident * 4 + _644.Load((_646 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _42ident = 0; _42ident < 3; _42ident++)
            {
                _661.b[_42ident] = asfloat(_640.Load(_42ident * 4 + _644.Load((_646 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _43ident = 0; _43ident < 2; _43ident++)
            {
                [unroll]
                for (int _44ident = 0; _44ident < 2; _44ident++)
                {
                    _661.t[_43ident][_44ident] = asfloat(_640.Load(_44ident * 4 + _43ident * 8 + _644.Load((_646 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _662;
            _662.p[0] = _661.p[0];
            _662.p[1] = _661.p[1];
            _662.p[2] = _661.p[2];
            _662.n[0] = _661.n[0];
            _662.n[1] = _661.n[1];
            _662.n[2] = _661.n[2];
            _662.b[0] = _661.b[0];
            _662.b[1] = _661.b[1];
            _662.b[2] = _661.b[2];
            _662.t[0][0] = _661.t[0][0];
            _662.t[0][1] = _661.t[0][1];
            _662.t[1][0] = _661.t[1][0];
            _662.t[1][1] = _661.t[1][1];
            vertex_t _670;
            [unroll]
            for (int _45ident = 0; _45ident < 3; _45ident++)
            {
                _670.p[_45ident] = asfloat(_640.Load(_45ident * 4 + _644.Load((_646 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _46ident = 0; _46ident < 3; _46ident++)
            {
                _670.n[_46ident] = asfloat(_640.Load(_46ident * 4 + _644.Load((_646 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _47ident = 0; _47ident < 3; _47ident++)
            {
                _670.b[_47ident] = asfloat(_640.Load(_47ident * 4 + _644.Load((_646 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _48ident = 0; _48ident < 2; _48ident++)
            {
                [unroll]
                for (int _49ident = 0; _49ident < 2; _49ident++)
                {
                    _670.t[_48ident][_49ident] = asfloat(_640.Load(_49ident * 4 + _48ident * 8 + _644.Load((_646 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _671;
            _671.p[0] = _670.p[0];
            _671.p[1] = _670.p[1];
            _671.p[2] = _670.p[2];
            _671.n[0] = _670.n[0];
            _671.n[1] = _670.n[1];
            _671.n[2] = _670.n[2];
            _671.b[0] = _670.b[0];
            _671.b[1] = _670.b[1];
            _671.b[2] = _670.b[2];
            _671.t[0][0] = _670.t[0][0];
            _671.t[0][1] = _670.t[0][1];
            _671.t[1][0] = _670.t[1][0];
            _671.t[1][1] = _670.t[1][1];
            float2 _704 = ((float2(_653.t[0][0], _653.t[0][1]) * ((1.0f - _949) - _950)) + (float2(_662.t[0][0], _662.t[0][1]) * _949)) + (float2(_671.t[0][0], _671.t[0][1]) * _950);
            float trans_r = frac(asfloat(_709.Load(rand_index * 4 + 0)) + _458);
            while (_963 == 4u)
            {
                float mix_val = _964;
                if (_1003 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_1003, _704, 0).x;
                }
                if (trans_r > mix_val)
                {
                    material_t _749;
                    [unroll]
                    for (int _50ident = 0; _50ident < 5; _50ident++)
                    {
                        _749.textures[_50ident] = _605.Load(_50ident * 4 + _1004 * 76 + 0);
                    }
                    [unroll]
                    for (int _51ident = 0; _51ident < 3; _51ident++)
                    {
                        _749.base_color[_51ident] = asfloat(_605.Load(_51ident * 4 + _1004 * 76 + 20));
                    }
                    _749.flags = _605.Load(_1004 * 76 + 32);
                    _749.type = _605.Load(_1004 * 76 + 36);
                    _749.tangent_rotation_or_strength = asfloat(_605.Load(_1004 * 76 + 40));
                    _749.roughness_and_anisotropic = _605.Load(_1004 * 76 + 44);
                    _749.ior = asfloat(_605.Load(_1004 * 76 + 48));
                    _749.sheen_and_sheen_tint = _605.Load(_1004 * 76 + 52);
                    _749.tint_and_metallic = _605.Load(_1004 * 76 + 56);
                    _749.transmission_and_transmission_roughness = _605.Load(_1004 * 76 + 60);
                    _749.specular_and_specular_tint = _605.Load(_1004 * 76 + 64);
                    _749.clearcoat_and_clearcoat_roughness = _605.Load(_1004 * 76 + 68);
                    _749.normal_map_strength_unorm = _605.Load(_1004 * 76 + 72);
                    material_t _750;
                    _750.textures[0] = _749.textures[0];
                    _750.textures[1] = _749.textures[1];
                    _750.textures[2] = _749.textures[2];
                    _750.textures[3] = _749.textures[3];
                    _750.textures[4] = _749.textures[4];
                    _750.base_color[0] = _749.base_color[0];
                    _750.base_color[1] = _749.base_color[1];
                    _750.base_color[2] = _749.base_color[2];
                    _750.flags = _749.flags;
                    _750.type = _749.type;
                    _750.tangent_rotation_or_strength = _749.tangent_rotation_or_strength;
                    _750.roughness_and_anisotropic = _749.roughness_and_anisotropic;
                    _750.ior = _749.ior;
                    _750.sheen_and_sheen_tint = _749.sheen_and_sheen_tint;
                    _750.tint_and_metallic = _749.tint_and_metallic;
                    _750.transmission_and_transmission_roughness = _749.transmission_and_transmission_roughness;
                    _750.specular_and_specular_tint = _749.specular_and_specular_tint;
                    _750.clearcoat_and_clearcoat_roughness = _749.clearcoat_and_clearcoat_roughness;
                    _750.normal_map_strength_unorm = _749.normal_map_strength_unorm;
                    _1003 = _750.textures[1];
                    _1004 = _750.textures[3];
                    _1005 = _750.textures[4];
                    _1018 = _750.base_color[0];
                    _1019 = _750.base_color[1];
                    _1020 = _750.base_color[2];
                    _963 = _750.type;
                    _964 = _750.tangent_rotation_or_strength;
                    trans_r = (trans_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _761;
                    [unroll]
                    for (int _52ident = 0; _52ident < 5; _52ident++)
                    {
                        _761.textures[_52ident] = _605.Load(_52ident * 4 + _1005 * 76 + 0);
                    }
                    [unroll]
                    for (int _53ident = 0; _53ident < 3; _53ident++)
                    {
                        _761.base_color[_53ident] = asfloat(_605.Load(_53ident * 4 + _1005 * 76 + 20));
                    }
                    _761.flags = _605.Load(_1005 * 76 + 32);
                    _761.type = _605.Load(_1005 * 76 + 36);
                    _761.tangent_rotation_or_strength = asfloat(_605.Load(_1005 * 76 + 40));
                    _761.roughness_and_anisotropic = _605.Load(_1005 * 76 + 44);
                    _761.ior = asfloat(_605.Load(_1005 * 76 + 48));
                    _761.sheen_and_sheen_tint = _605.Load(_1005 * 76 + 52);
                    _761.tint_and_metallic = _605.Load(_1005 * 76 + 56);
                    _761.transmission_and_transmission_roughness = _605.Load(_1005 * 76 + 60);
                    _761.specular_and_specular_tint = _605.Load(_1005 * 76 + 64);
                    _761.clearcoat_and_clearcoat_roughness = _605.Load(_1005 * 76 + 68);
                    _761.normal_map_strength_unorm = _605.Load(_1005 * 76 + 72);
                    material_t _762;
                    _762.textures[0] = _761.textures[0];
                    _762.textures[1] = _761.textures[1];
                    _762.textures[2] = _761.textures[2];
                    _762.textures[3] = _761.textures[3];
                    _762.textures[4] = _761.textures[4];
                    _762.base_color[0] = _761.base_color[0];
                    _762.base_color[1] = _761.base_color[1];
                    _762.base_color[2] = _761.base_color[2];
                    _762.flags = _761.flags;
                    _762.type = _761.type;
                    _762.tangent_rotation_or_strength = _761.tangent_rotation_or_strength;
                    _762.roughness_and_anisotropic = _761.roughness_and_anisotropic;
                    _762.ior = _761.ior;
                    _762.sheen_and_sheen_tint = _761.sheen_and_sheen_tint;
                    _762.tint_and_metallic = _761.tint_and_metallic;
                    _762.transmission_and_transmission_roughness = _761.transmission_and_transmission_roughness;
                    _762.specular_and_specular_tint = _761.specular_and_specular_tint;
                    _762.clearcoat_and_clearcoat_roughness = _761.clearcoat_and_clearcoat_roughness;
                    _762.normal_map_strength_unorm = _761.normal_map_strength_unorm;
                    _1003 = _762.textures[1];
                    _1004 = _762.textures[3];
                    _1005 = _762.textures[4];
                    _1018 = _762.base_color[0];
                    _1019 = _762.base_color[1];
                    _1020 = _762.base_color[2];
                    _963 = _762.type;
                    _964 = _762.tangent_rotation_or_strength;
                    trans_r /= mix_val;
                }
            }
            if (_963 != 5u)
            {
                break;
            }
            float _791 = max(asfloat(_407.Load(_397 * 72 + 28)), max(asfloat(_407.Load(_397 * 72 + 32)), asfloat(_407.Load(_397 * 72 + 36))));
            if ((int(_407.Load(_397 * 72 + 68)) >> 24) > _365_g_params.min_transp_depth)
            {
                _803 = max(0.0500000007450580596923828125f, 1.0f - _791);
            }
            else
            {
                _803 = 0.0f;
            }
            bool _817 = (frac(asfloat(_709.Load((rand_index + 6) * 4 + 0)) + _458) < _803) || (_791 == 0.0f);
            bool _829;
            if (!_817)
            {
                _829 = ((int(_407.Load(_397 * 72 + 68)) >> 24) + 1) >= _365_g_params.max_transp_depth;
            }
            else
            {
                _829 = _817;
            }
            if (_829)
            {
                _407.Store(_397 * 72 + 36, asuint(0.0f));
                _407.Store(_397 * 72 + 32, asuint(0.0f));
                _407.Store(_397 * 72 + 28, asuint(0.0f));
                break;
            }
            float _843 = 1.0f - _803;
            _407.Store(_397 * 72 + 28, asuint(asfloat(_407.Load(_397 * 72 + 28)) * (_1018 / _843)));
            _407.Store(_397 * 72 + 32, asuint(asfloat(_407.Load(_397 * 72 + 32)) * (_1019 / _843)));
            _407.Store(_397 * 72 + 36, asuint(asfloat(_407.Load(_397 * 72 + 36)) * (_1020 / _843)));
            ro += (_430 * (_948 + 9.9999997473787516355514526367188e-06f));
            _945 = 0;
            _948 = _482 - _948;
            _407.Store(_397 * 72 + 68, uint(int(_407.Load(_397 * 72 + 68)) + 16777216));
            rand_index += 7;
            continue;
        }
        float _907 = _948;
        float _908 = _907 + length(float3(asfloat(_407.Load(_397 * 72 + 0)), asfloat(_407.Load(_397 * 72 + 4)), asfloat(_407.Load(_397 * 72 + 8))) - ro);
        _948 = _908;
        hit_data_t _957 = { _945, _946, _947, _908, _949, _950 };
        hit_data_t _919;
        _919.mask = _957.mask;
        _919.obj_index = _957.obj_index;
        _919.prim_index = _957.prim_index;
        _919.t = _957.t;
        _919.u = _957.u;
        _919.v = _957.v;
        _914.Store(_397 * 24 + 0, uint(_919.mask));
        _914.Store(_397 * 24 + 4, uint(_919.obj_index));
        _914.Store(_397 * 24 + 8, uint(_919.prim_index));
        _914.Store(_397 * 24 + 12, asuint(_919.t));
        _914.Store(_397 * 24 + 16, asuint(_919.u));
        _914.Store(_397 * 24 + 20, asuint(_919.v));
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
