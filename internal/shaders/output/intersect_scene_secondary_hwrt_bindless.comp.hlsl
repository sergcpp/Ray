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
    uint node_index;
    float cam_clip_end;
    int min_transp_depth;
    int max_transp_depth;
    int hi;
    int pad2;
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
ByteAddressBuffer _565 : register(t5, space0);
ByteAddressBuffer _602 : register(t6, space0);
ByteAddressBuffer _637 : register(t1, space0);
ByteAddressBuffer _641 : register(t2, space0);
ByteAddressBuffer _706 : register(t15, space0);
RWByteAddressBuffer _911 : register(u0, space0);
cbuffer UniformParams
{
    Params _457_g_params : packoffset(c0);
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
        float3 _1061 = inv_v;
        _1061.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _1061;
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
            float3 _1059 = inv_v;
            _1059.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _1059;
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
        float3 _1065 = inv_v;
        _1065.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _1065;
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
            float3 _1063 = inv_v;
            _1063.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _1063;
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
        float3 _1069 = inv_v;
        _1069.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _1069;
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
            float3 _1067 = inv_v;
            _1067.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _1067;
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
        float4 _1085 = res;
        _1085.x = _341.x;
        float4 _1087 = _1085;
        _1087.y = _341.y;
        float4 _1089 = _1087;
        _1089.z = _341.z;
        res = _1089;
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
        int _942 = 0;
        int _944 = 0;
        int _943 = 0;
        float _945 = 3402823346297367662189621542912.0f;
        float _947 = 0.0f;
        float _946 = 0.0f;
        uint param_1 = uint(hash(int(_390.Load(_367 * 72 + 64))));
        float _451 = construct_float(param_1);
        ray_data_t _464;
        [unroll]
        for (int _27ident = 0; _27ident < 3; _27ident++)
        {
            _464.o[_27ident] = asfloat(_390.Load(_27ident * 4 + _367 * 72 + 0));
        }
        [unroll]
        for (int _28ident = 0; _28ident < 3; _28ident++)
        {
            _464.d[_28ident] = asfloat(_390.Load(_28ident * 4 + _367 * 72 + 12));
        }
        _464.pdf = asfloat(_390.Load(_367 * 72 + 24));
        [unroll]
        for (int _29ident = 0; _29ident < 3; _29ident++)
        {
            _464.c[_29ident] = asfloat(_390.Load(_29ident * 4 + _367 * 72 + 28));
        }
        [unroll]
        for (int _30ident = 0; _30ident < 4; _30ident++)
        {
            _464.ior[_30ident] = asfloat(_390.Load(_30ident * 4 + _367 * 72 + 40));
        }
        _464.cone_width = asfloat(_390.Load(_367 * 72 + 56));
        _464.cone_spread = asfloat(_390.Load(_367 * 72 + 60));
        _464.xy = int(_390.Load(_367 * 72 + 64));
        _464.depth = int(_390.Load(_367 * 72 + 68));
        ray_data_t _467;
        _467.o[0] = _464.o[0];
        _467.o[1] = _464.o[1];
        _467.o[2] = _464.o[2];
        _467.d[0] = _464.d[0];
        _467.d[1] = _464.d[1];
        _467.d[2] = _464.d[2];
        _467.pdf = _464.pdf;
        _467.c[0] = _464.c[0];
        _467.c[1] = _464.c[1];
        _467.c[2] = _464.c[2];
        _467.ior[0] = _464.ior[0];
        _467.ior[1] = _464.ior[1];
        _467.ior[2] = _464.ior[2];
        _467.ior[3] = _464.ior[3];
        _467.cone_width = _464.cone_width;
        _467.cone_spread = _464.cone_spread;
        _467.xy = _464.xy;
        _467.depth = _464.depth;
        int rand_index = _457_g_params.hi + (total_depth(_467) * 7);
        int _546;
        float _800;
        float _480;
        for (;;)
        {
            _480 = _945;
            rayQueryInitializeEXT(rq, g_tlas, 0u, 255u, ro, 0.0f, _427, _480);
            for (;;)
            {
                bool _498 = rayQueryProceedEXT(rq);
                if (_498)
                {
                    rayQueryConfirmIntersectionEXT(rq);
                    continue;
                }
                else
                {
                    break;
                }
            }
            uint _499 = rayQueryGetIntersectionTypeEXT(rq, bool(1));
            if (_499 != 0u)
            {
                int _504 = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, bool(1));
                _942 = -1;
                int _507 = rayQueryGetIntersectionInstanceIdEXT(rq, bool(1));
                _943 = _507;
                int _510 = rayQueryGetIntersectionPrimitiveIndexEXT(rq, bool(1));
                _944 = _504 + _510;
                bool _513 = rayQueryGetIntersectionFrontFaceEXT(rq, bool(1));
                [flatten]
                if (_513 == false)
                {
                    _944 = (-1) - _944;
                }
                float2 _524 = rayQueryGetIntersectionBarycentricsEXT(rq, bool(1));
                _946 = _524.x;
                _947 = _524.y;
                float _531 = rayQueryGetIntersectionTEXT(rq, bool(1));
                _945 = _531;
            }
            if (_942 == 0)
            {
                break;
            }
            bool _543 = _944 < 0;
            if (_543)
            {
                _546 = (-1) - _944;
            }
            else
            {
                _546 = _944;
            }
            uint _557 = uint(_546);
            bool _559 = !_543;
            bool _574;
            if (_559)
            {
                _574 = ((_565.Load(_557 * 4 + 0) >> 16u) & 32768u) != 0u;
            }
            else
            {
                _574 = _559;
            }
            bool _587;
            if (!_574)
            {
                bool _586;
                if (_543)
                {
                    _586 = (_565.Load(_557 * 4 + 0) & 32768u) != 0u;
                }
                else
                {
                    _586 = _543;
                }
                _587 = _586;
            }
            else
            {
                _587 = _574;
            }
            if (_587)
            {
                break;
            }
            material_t _611;
            [unroll]
            for (int _31ident = 0; _31ident < 5; _31ident++)
            {
                _611.textures[_31ident] = _602.Load(_31ident * 4 + ((_565.Load(_557 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _32ident = 0; _32ident < 3; _32ident++)
            {
                _611.base_color[_32ident] = asfloat(_602.Load(_32ident * 4 + ((_565.Load(_557 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
            }
            _611.flags = _602.Load(((_565.Load(_557 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
            _611.type = _602.Load(((_565.Load(_557 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
            _611.tangent_rotation_or_strength = asfloat(_602.Load(((_565.Load(_557 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
            _611.roughness_and_anisotropic = _602.Load(((_565.Load(_557 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
            _611.ior = asfloat(_602.Load(((_565.Load(_557 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
            _611.sheen_and_sheen_tint = _602.Load(((_565.Load(_557 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
            _611.tint_and_metallic = _602.Load(((_565.Load(_557 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
            _611.transmission_and_transmission_roughness = _602.Load(((_565.Load(_557 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
            _611.specular_and_specular_tint = _602.Load(((_565.Load(_557 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
            _611.clearcoat_and_clearcoat_roughness = _602.Load(((_565.Load(_557 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
            _611.normal_map_strength_unorm = _602.Load(((_565.Load(_557 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
            material_t _612;
            _612.textures[0] = _611.textures[0];
            _612.textures[1] = _611.textures[1];
            _612.textures[2] = _611.textures[2];
            _612.textures[3] = _611.textures[3];
            _612.textures[4] = _611.textures[4];
            _612.base_color[0] = _611.base_color[0];
            _612.base_color[1] = _611.base_color[1];
            _612.base_color[2] = _611.base_color[2];
            _612.flags = _611.flags;
            _612.type = _611.type;
            _612.tangent_rotation_or_strength = _611.tangent_rotation_or_strength;
            _612.roughness_and_anisotropic = _611.roughness_and_anisotropic;
            _612.ior = _611.ior;
            _612.sheen_and_sheen_tint = _611.sheen_and_sheen_tint;
            _612.tint_and_metallic = _611.tint_and_metallic;
            _612.transmission_and_transmission_roughness = _611.transmission_and_transmission_roughness;
            _612.specular_and_specular_tint = _611.specular_and_specular_tint;
            _612.clearcoat_and_clearcoat_roughness = _611.clearcoat_and_clearcoat_roughness;
            _612.normal_map_strength_unorm = _611.normal_map_strength_unorm;
            uint _1000 = _612.textures[1];
            uint _1001 = _612.textures[3];
            uint _1002 = _612.textures[4];
            float _1015 = _612.base_color[0];
            float _1016 = _612.base_color[1];
            float _1017 = _612.base_color[2];
            uint _960 = _612.type;
            float _961 = _612.tangent_rotation_or_strength;
            if (_543)
            {
                material_t _621;
                [unroll]
                for (int _33ident = 0; _33ident < 5; _33ident++)
                {
                    _621.textures[_33ident] = _602.Load(_33ident * 4 + (_565.Load(_557 * 4 + 0) & 16383u) * 76 + 0);
                }
                [unroll]
                for (int _34ident = 0; _34ident < 3; _34ident++)
                {
                    _621.base_color[_34ident] = asfloat(_602.Load(_34ident * 4 + (_565.Load(_557 * 4 + 0) & 16383u) * 76 + 20));
                }
                _621.flags = _602.Load((_565.Load(_557 * 4 + 0) & 16383u) * 76 + 32);
                _621.type = _602.Load((_565.Load(_557 * 4 + 0) & 16383u) * 76 + 36);
                _621.tangent_rotation_or_strength = asfloat(_602.Load((_565.Load(_557 * 4 + 0) & 16383u) * 76 + 40));
                _621.roughness_and_anisotropic = _602.Load((_565.Load(_557 * 4 + 0) & 16383u) * 76 + 44);
                _621.ior = asfloat(_602.Load((_565.Load(_557 * 4 + 0) & 16383u) * 76 + 48));
                _621.sheen_and_sheen_tint = _602.Load((_565.Load(_557 * 4 + 0) & 16383u) * 76 + 52);
                _621.tint_and_metallic = _602.Load((_565.Load(_557 * 4 + 0) & 16383u) * 76 + 56);
                _621.transmission_and_transmission_roughness = _602.Load((_565.Load(_557 * 4 + 0) & 16383u) * 76 + 60);
                _621.specular_and_specular_tint = _602.Load((_565.Load(_557 * 4 + 0) & 16383u) * 76 + 64);
                _621.clearcoat_and_clearcoat_roughness = _602.Load((_565.Load(_557 * 4 + 0) & 16383u) * 76 + 68);
                _621.normal_map_strength_unorm = _602.Load((_565.Load(_557 * 4 + 0) & 16383u) * 76 + 72);
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
                _1000 = _622.textures[1];
                _1001 = _622.textures[3];
                _1002 = _622.textures[4];
                _1015 = _622.base_color[0];
                _1016 = _622.base_color[1];
                _1017 = _622.base_color[2];
                _960 = _622.type;
                _961 = _622.tangent_rotation_or_strength;
            }
            uint _643 = _557 * 3u;
            vertex_t _649;
            [unroll]
            for (int _35ident = 0; _35ident < 3; _35ident++)
            {
                _649.p[_35ident] = asfloat(_637.Load(_35ident * 4 + _641.Load(_643 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _649.n[_36ident] = asfloat(_637.Load(_36ident * 4 + _641.Load(_643 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _37ident = 0; _37ident < 3; _37ident++)
            {
                _649.b[_37ident] = asfloat(_637.Load(_37ident * 4 + _641.Load(_643 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _38ident = 0; _38ident < 2; _38ident++)
            {
                [unroll]
                for (int _39ident = 0; _39ident < 2; _39ident++)
                {
                    _649.t[_38ident][_39ident] = asfloat(_637.Load(_39ident * 4 + _38ident * 8 + _641.Load(_643 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _650;
            _650.p[0] = _649.p[0];
            _650.p[1] = _649.p[1];
            _650.p[2] = _649.p[2];
            _650.n[0] = _649.n[0];
            _650.n[1] = _649.n[1];
            _650.n[2] = _649.n[2];
            _650.b[0] = _649.b[0];
            _650.b[1] = _649.b[1];
            _650.b[2] = _649.b[2];
            _650.t[0][0] = _649.t[0][0];
            _650.t[0][1] = _649.t[0][1];
            _650.t[1][0] = _649.t[1][0];
            _650.t[1][1] = _649.t[1][1];
            vertex_t _658;
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _658.p[_40ident] = asfloat(_637.Load(_40ident * 4 + _641.Load((_643 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 3; _41ident++)
            {
                _658.n[_41ident] = asfloat(_637.Load(_41ident * 4 + _641.Load((_643 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _42ident = 0; _42ident < 3; _42ident++)
            {
                _658.b[_42ident] = asfloat(_637.Load(_42ident * 4 + _641.Load((_643 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _43ident = 0; _43ident < 2; _43ident++)
            {
                [unroll]
                for (int _44ident = 0; _44ident < 2; _44ident++)
                {
                    _658.t[_43ident][_44ident] = asfloat(_637.Load(_44ident * 4 + _43ident * 8 + _641.Load((_643 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _659;
            _659.p[0] = _658.p[0];
            _659.p[1] = _658.p[1];
            _659.p[2] = _658.p[2];
            _659.n[0] = _658.n[0];
            _659.n[1] = _658.n[1];
            _659.n[2] = _658.n[2];
            _659.b[0] = _658.b[0];
            _659.b[1] = _658.b[1];
            _659.b[2] = _658.b[2];
            _659.t[0][0] = _658.t[0][0];
            _659.t[0][1] = _658.t[0][1];
            _659.t[1][0] = _658.t[1][0];
            _659.t[1][1] = _658.t[1][1];
            vertex_t _667;
            [unroll]
            for (int _45ident = 0; _45ident < 3; _45ident++)
            {
                _667.p[_45ident] = asfloat(_637.Load(_45ident * 4 + _641.Load((_643 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _46ident = 0; _46ident < 3; _46ident++)
            {
                _667.n[_46ident] = asfloat(_637.Load(_46ident * 4 + _641.Load((_643 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _47ident = 0; _47ident < 3; _47ident++)
            {
                _667.b[_47ident] = asfloat(_637.Load(_47ident * 4 + _641.Load((_643 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _48ident = 0; _48ident < 2; _48ident++)
            {
                [unroll]
                for (int _49ident = 0; _49ident < 2; _49ident++)
                {
                    _667.t[_48ident][_49ident] = asfloat(_637.Load(_49ident * 4 + _48ident * 8 + _641.Load((_643 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _668;
            _668.p[0] = _667.p[0];
            _668.p[1] = _667.p[1];
            _668.p[2] = _667.p[2];
            _668.n[0] = _667.n[0];
            _668.n[1] = _667.n[1];
            _668.n[2] = _667.n[2];
            _668.b[0] = _667.b[0];
            _668.b[1] = _667.b[1];
            _668.b[2] = _667.b[2];
            _668.t[0][0] = _667.t[0][0];
            _668.t[0][1] = _667.t[0][1];
            _668.t[1][0] = _667.t[1][0];
            _668.t[1][1] = _667.t[1][1];
            float2 _701 = ((float2(_650.t[0][0], _650.t[0][1]) * ((1.0f - _946) - _947)) + (float2(_659.t[0][0], _659.t[0][1]) * _946)) + (float2(_668.t[0][0], _668.t[0][1]) * _947);
            float trans_r = frac(asfloat(_706.Load(rand_index * 4 + 0)) + _451);
            while (_960 == 4u)
            {
                float mix_val = _961;
                if (_1000 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_1000, _701, 0).x;
                }
                if (trans_r > mix_val)
                {
                    material_t _746;
                    [unroll]
                    for (int _50ident = 0; _50ident < 5; _50ident++)
                    {
                        _746.textures[_50ident] = _602.Load(_50ident * 4 + _1001 * 76 + 0);
                    }
                    [unroll]
                    for (int _51ident = 0; _51ident < 3; _51ident++)
                    {
                        _746.base_color[_51ident] = asfloat(_602.Load(_51ident * 4 + _1001 * 76 + 20));
                    }
                    _746.flags = _602.Load(_1001 * 76 + 32);
                    _746.type = _602.Load(_1001 * 76 + 36);
                    _746.tangent_rotation_or_strength = asfloat(_602.Load(_1001 * 76 + 40));
                    _746.roughness_and_anisotropic = _602.Load(_1001 * 76 + 44);
                    _746.ior = asfloat(_602.Load(_1001 * 76 + 48));
                    _746.sheen_and_sheen_tint = _602.Load(_1001 * 76 + 52);
                    _746.tint_and_metallic = _602.Load(_1001 * 76 + 56);
                    _746.transmission_and_transmission_roughness = _602.Load(_1001 * 76 + 60);
                    _746.specular_and_specular_tint = _602.Load(_1001 * 76 + 64);
                    _746.clearcoat_and_clearcoat_roughness = _602.Load(_1001 * 76 + 68);
                    _746.normal_map_strength_unorm = _602.Load(_1001 * 76 + 72);
                    material_t _747;
                    _747.textures[0] = _746.textures[0];
                    _747.textures[1] = _746.textures[1];
                    _747.textures[2] = _746.textures[2];
                    _747.textures[3] = _746.textures[3];
                    _747.textures[4] = _746.textures[4];
                    _747.base_color[0] = _746.base_color[0];
                    _747.base_color[1] = _746.base_color[1];
                    _747.base_color[2] = _746.base_color[2];
                    _747.flags = _746.flags;
                    _747.type = _746.type;
                    _747.tangent_rotation_or_strength = _746.tangent_rotation_or_strength;
                    _747.roughness_and_anisotropic = _746.roughness_and_anisotropic;
                    _747.ior = _746.ior;
                    _747.sheen_and_sheen_tint = _746.sheen_and_sheen_tint;
                    _747.tint_and_metallic = _746.tint_and_metallic;
                    _747.transmission_and_transmission_roughness = _746.transmission_and_transmission_roughness;
                    _747.specular_and_specular_tint = _746.specular_and_specular_tint;
                    _747.clearcoat_and_clearcoat_roughness = _746.clearcoat_and_clearcoat_roughness;
                    _747.normal_map_strength_unorm = _746.normal_map_strength_unorm;
                    _1000 = _747.textures[1];
                    _1001 = _747.textures[3];
                    _1002 = _747.textures[4];
                    _1015 = _747.base_color[0];
                    _1016 = _747.base_color[1];
                    _1017 = _747.base_color[2];
                    _960 = _747.type;
                    _961 = _747.tangent_rotation_or_strength;
                    trans_r = (trans_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _758;
                    [unroll]
                    for (int _52ident = 0; _52ident < 5; _52ident++)
                    {
                        _758.textures[_52ident] = _602.Load(_52ident * 4 + _1002 * 76 + 0);
                    }
                    [unroll]
                    for (int _53ident = 0; _53ident < 3; _53ident++)
                    {
                        _758.base_color[_53ident] = asfloat(_602.Load(_53ident * 4 + _1002 * 76 + 20));
                    }
                    _758.flags = _602.Load(_1002 * 76 + 32);
                    _758.type = _602.Load(_1002 * 76 + 36);
                    _758.tangent_rotation_or_strength = asfloat(_602.Load(_1002 * 76 + 40));
                    _758.roughness_and_anisotropic = _602.Load(_1002 * 76 + 44);
                    _758.ior = asfloat(_602.Load(_1002 * 76 + 48));
                    _758.sheen_and_sheen_tint = _602.Load(_1002 * 76 + 52);
                    _758.tint_and_metallic = _602.Load(_1002 * 76 + 56);
                    _758.transmission_and_transmission_roughness = _602.Load(_1002 * 76 + 60);
                    _758.specular_and_specular_tint = _602.Load(_1002 * 76 + 64);
                    _758.clearcoat_and_clearcoat_roughness = _602.Load(_1002 * 76 + 68);
                    _758.normal_map_strength_unorm = _602.Load(_1002 * 76 + 72);
                    material_t _759;
                    _759.textures[0] = _758.textures[0];
                    _759.textures[1] = _758.textures[1];
                    _759.textures[2] = _758.textures[2];
                    _759.textures[3] = _758.textures[3];
                    _759.textures[4] = _758.textures[4];
                    _759.base_color[0] = _758.base_color[0];
                    _759.base_color[1] = _758.base_color[1];
                    _759.base_color[2] = _758.base_color[2];
                    _759.flags = _758.flags;
                    _759.type = _758.type;
                    _759.tangent_rotation_or_strength = _758.tangent_rotation_or_strength;
                    _759.roughness_and_anisotropic = _758.roughness_and_anisotropic;
                    _759.ior = _758.ior;
                    _759.sheen_and_sheen_tint = _758.sheen_and_sheen_tint;
                    _759.tint_and_metallic = _758.tint_and_metallic;
                    _759.transmission_and_transmission_roughness = _758.transmission_and_transmission_roughness;
                    _759.specular_and_specular_tint = _758.specular_and_specular_tint;
                    _759.clearcoat_and_clearcoat_roughness = _758.clearcoat_and_clearcoat_roughness;
                    _759.normal_map_strength_unorm = _758.normal_map_strength_unorm;
                    _1000 = _759.textures[1];
                    _1001 = _759.textures[3];
                    _1002 = _759.textures[4];
                    _1015 = _759.base_color[0];
                    _1016 = _759.base_color[1];
                    _1017 = _759.base_color[2];
                    _960 = _759.type;
                    _961 = _759.tangent_rotation_or_strength;
                    trans_r /= mix_val;
                }
            }
            if (_960 != 5u)
            {
                break;
            }
            float _788 = max(asfloat(_390.Load(_367 * 72 + 28)), max(asfloat(_390.Load(_367 * 72 + 32)), asfloat(_390.Load(_367 * 72 + 36))));
            if ((int(_390.Load(_367 * 72 + 68)) >> 24) > _457_g_params.min_transp_depth)
            {
                _800 = max(0.0500000007450580596923828125f, 1.0f - _788);
            }
            else
            {
                _800 = 0.0f;
            }
            bool _814 = (frac(asfloat(_706.Load((rand_index + 6) * 4 + 0)) + _451) < _800) || (_788 == 0.0f);
            bool _826;
            if (!_814)
            {
                _826 = ((int(_390.Load(_367 * 72 + 68)) >> 24) + 1) >= _457_g_params.max_transp_depth;
            }
            else
            {
                _826 = _814;
            }
            if (_826)
            {
                _390.Store(_367 * 72 + 36, asuint(0.0f));
                _390.Store(_367 * 72 + 32, asuint(0.0f));
                _390.Store(_367 * 72 + 28, asuint(0.0f));
                break;
            }
            float _840 = 1.0f - _800;
            _390.Store(_367 * 72 + 28, asuint(asfloat(_390.Load(_367 * 72 + 28)) * (_1015 / _840)));
            _390.Store(_367 * 72 + 32, asuint(asfloat(_390.Load(_367 * 72 + 32)) * (_1016 / _840)));
            _390.Store(_367 * 72 + 36, asuint(asfloat(_390.Load(_367 * 72 + 36)) * (_1017 / _840)));
            ro += (_427 * (_945 + 9.9999997473787516355514526367188e-06f));
            _942 = 0;
            _945 = _480 - _945;
            _390.Store(_367 * 72 + 68, uint(int(_390.Load(_367 * 72 + 68)) + 16777216));
            rand_index += 7;
            continue;
        }
        float _904 = _945;
        float _905 = _904 + length(float3(asfloat(_390.Load(_367 * 72 + 0)), asfloat(_390.Load(_367 * 72 + 4)), asfloat(_390.Load(_367 * 72 + 8))) - ro);
        _945 = _905;
        hit_data_t _954 = { _942, _943, _944, _905, _946, _947 };
        hit_data_t _916;
        _916.mask = _954.mask;
        _916.obj_index = _954.obj_index;
        _916.prim_index = _954.prim_index;
        _916.t = _954.t;
        _916.u = _954.u;
        _916.v = _954.v;
        _911.Store(_367 * 24 + 0, uint(_916.mask));
        _911.Store(_367 * 24 + 4, uint(_916.obj_index));
        _911.Store(_367 * 24 + 8, uint(_916.prim_index));
        _911.Store(_367 * 24 + 12, asuint(_916.t));
        _911.Store(_367 * 24 + 16, asuint(_916.u));
        _911.Store(_367 * 24 + 20, asuint(_916.v));
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
