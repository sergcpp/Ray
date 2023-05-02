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

struct atlas_texture_t
{
    uint size;
    uint atlas;
    uint page[4];
    uint pos[14];
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

ByteAddressBuffer _369 : register(t20, space0);
RWByteAddressBuffer _514 : register(u12, space0);
ByteAddressBuffer _670 : register(t5, space0);
ByteAddressBuffer _707 : register(t6, space0);
ByteAddressBuffer _741 : register(t1, space0);
ByteAddressBuffer _745 : register(t2, space0);
ByteAddressBuffer _810 : register(t15, space0);
RWByteAddressBuffer _1015 : register(u0, space0);
cbuffer UniformParams
{
    Params _465_g_params : packoffset(c0);
};

Texture2DArray<float4> g_atlases[7] : register(t21, space0);
SamplerState _g_atlases_sampler[7] : register(s21, space0);
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
    bool _113 = v.x <= 1.0000000116860974230803549289703e-07f;
    bool _120;
    if (_113)
    {
        _120 = v.x >= 0.0f;
    }
    else
    {
        _120 = _113;
    }
    if (_120)
    {
        float3 _1184 = inv_v;
        _1184.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _1184;
    }
    else
    {
        bool _129 = v.x >= (-1.0000000116860974230803549289703e-07f);
        bool _135;
        if (_129)
        {
            _135 = v.x < 0.0f;
        }
        else
        {
            _135 = _129;
        }
        if (_135)
        {
            float3 _1182 = inv_v;
            _1182.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _1182;
        }
    }
    bool _143 = v.y <= 1.0000000116860974230803549289703e-07f;
    bool _149;
    if (_143)
    {
        _149 = v.y >= 0.0f;
    }
    else
    {
        _149 = _143;
    }
    if (_149)
    {
        float3 _1188 = inv_v;
        _1188.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _1188;
    }
    else
    {
        bool _156 = v.y >= (-1.0000000116860974230803549289703e-07f);
        bool _162;
        if (_156)
        {
            _162 = v.y < 0.0f;
        }
        else
        {
            _162 = _156;
        }
        if (_162)
        {
            float3 _1186 = inv_v;
            _1186.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _1186;
        }
    }
    bool _169 = v.z <= 1.0000000116860974230803549289703e-07f;
    bool _175;
    if (_169)
    {
        _175 = v.z >= 0.0f;
    }
    else
    {
        _175 = _169;
    }
    if (_175)
    {
        float3 _1192 = inv_v;
        _1192.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _1192;
    }
    else
    {
        bool _182 = v.z >= (-1.0000000116860974230803549289703e-07f);
        bool _188;
        if (_182)
        {
            _188 = v.z < 0.0f;
        }
        else
        {
            _188 = _182;
        }
        if (_188)
        {
            float3 _1190 = inv_v;
            _1190.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _1190;
        }
    }
    return inv_v;
}

int hash(int x)
{
    uint _69 = uint(x);
    uint _76 = ((_69 >> uint(16)) ^ _69) * 73244475u;
    uint _81 = ((_76 >> uint(16)) ^ _76) * 73244475u;
    return int((_81 >> uint(16)) ^ _81);
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

float2 TransformUV(float2 _uv, atlas_texture_t t, int mip_level)
{
    uint _1164[14] = t.pos;
    uint _1167[14] = t.pos;
    uint _327 = t.size & 16383u;
    uint _330 = t.size >> uint(16);
    uint _331 = _330 & 16383u;
    float2 size = float2(float(_327), float(_331));
    if ((_330 & 32768u) != 0u)
    {
        size = float2(float(_327 >> uint(mip_level)), float(_331 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_1164[mip_level] & 65535u), float((_1167[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _244 = mad(col.z, 31.875f, 1.0f);
    float _254 = (col.x - 0.501960813999176025390625f) / _244;
    float _260 = (col.y - 0.501960813999176025390625f) / _244;
    return float3((col.w + _254) - _260, col.w + _260, (col.w - _254) - _260);
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
    atlas_texture_t _372;
    _372.size = _369.Load(index * 80 + 0);
    _372.atlas = _369.Load(index * 80 + 4);
    [unroll]
    for (int _29ident = 0; _29ident < 4; _29ident++)
    {
        _372.page[_29ident] = _369.Load(_29ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _30ident = 0; _30ident < 14; _30ident++)
    {
        _372.pos[_30ident] = _369.Load(_30ident * 4 + index * 80 + 24);
    }
    atlas_texture_t _373;
    _373.size = _372.size;
    _373.atlas = _372.atlas;
    _373.page[0] = _372.page[0];
    _373.page[1] = _372.page[1];
    _373.page[2] = _372.page[2];
    _373.page[3] = _372.page[3];
    _373.pos[0] = _372.pos[0];
    _373.pos[1] = _372.pos[1];
    _373.pos[2] = _372.pos[2];
    _373.pos[3] = _372.pos[3];
    _373.pos[4] = _372.pos[4];
    _373.pos[5] = _372.pos[5];
    _373.pos[6] = _372.pos[6];
    _373.pos[7] = _372.pos[7];
    _373.pos[8] = _372.pos[8];
    _373.pos[9] = _372.pos[9];
    _373.pos[10] = _372.pos[10];
    _373.pos[11] = _372.pos[11];
    _373.pos[12] = _372.pos[12];
    _373.pos[13] = _372.pos[13];
    uint _1173[4] = _373.page;
    uint _401 = _373.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_401)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_401)], float3(TransformUV(uvs, _373, lod) * 0.000118371215648949146270751953125f.xx, float((_1173[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
    bool _416;
    if (maybe_YCoCg)
    {
        _416 = _373.atlas == 4u;
    }
    else
    {
        _416 = maybe_YCoCg;
    }
    if (_416)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _435;
    if (maybe_SRGB)
    {
        _435 = (_373.size & 32768u) != 0u;
    }
    else
    {
        _435 = maybe_SRGB;
    }
    if (_435)
    {
        float3 param_1 = res.xyz;
        float3 _441 = srgb_to_rgb(param_1);
        float4 _1208 = res;
        _1208.x = _441.x;
        float4 _1210 = _1208;
        _1210.y = _441.y;
        float4 _1212 = _1210;
        _1212.z = _441.z;
        res = _1212;
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
        bool _469 = gl_GlobalInvocationID.x >= _465_g_params.rect.z;
        bool _478;
        if (!_469)
        {
            _478 = gl_GlobalInvocationID.y >= _465_g_params.rect.w;
        }
        else
        {
            _478 = _469;
        }
        if (_478)
        {
            break;
        }
        int _504 = int(gl_GlobalInvocationID.y * _465_g_params.rect.z) + int(_465_g_params.rect.x + gl_GlobalInvocationID.x);
        float3 ro = float3(asfloat(_514.Load(_504 * 72 + 0)), asfloat(_514.Load(_504 * 72 + 4)), asfloat(_514.Load(_504 * 72 + 8)));
        float _529 = asfloat(_514.Load(_504 * 72 + 12));
        float _532 = asfloat(_514.Load(_504 * 72 + 16));
        float _535 = asfloat(_514.Load(_504 * 72 + 20));
        float3 _536 = float3(_529, _532, _535);
        float3 param = _536;
        int _1048 = 0;
        int _1050 = 0;
        int _1049 = 0;
        float _1051 = _465_g_params.inter_t;
        float _1053 = 0.0f;
        float _1052 = 0.0f;
        uint param_1 = uint(hash(int(_514.Load(_504 * 72 + 64))));
        float _563 = construct_float(param_1);
        ray_data_t _571;
        [unroll]
        for (int _31ident = 0; _31ident < 3; _31ident++)
        {
            _571.o[_31ident] = asfloat(_514.Load(_31ident * 4 + _504 * 72 + 0));
        }
        [unroll]
        for (int _32ident = 0; _32ident < 3; _32ident++)
        {
            _571.d[_32ident] = asfloat(_514.Load(_32ident * 4 + _504 * 72 + 12));
        }
        _571.pdf = asfloat(_514.Load(_504 * 72 + 24));
        [unroll]
        for (int _33ident = 0; _33ident < 3; _33ident++)
        {
            _571.c[_33ident] = asfloat(_514.Load(_33ident * 4 + _504 * 72 + 28));
        }
        [unroll]
        for (int _34ident = 0; _34ident < 4; _34ident++)
        {
            _571.ior[_34ident] = asfloat(_514.Load(_34ident * 4 + _504 * 72 + 40));
        }
        _571.cone_width = asfloat(_514.Load(_504 * 72 + 56));
        _571.cone_spread = asfloat(_514.Load(_504 * 72 + 60));
        _571.xy = int(_514.Load(_504 * 72 + 64));
        _571.depth = int(_514.Load(_504 * 72 + 68));
        ray_data_t _574;
        _574.o[0] = _571.o[0];
        _574.o[1] = _571.o[1];
        _574.o[2] = _571.o[2];
        _574.d[0] = _571.d[0];
        _574.d[1] = _571.d[1];
        _574.d[2] = _571.d[2];
        _574.pdf = _571.pdf;
        _574.c[0] = _571.c[0];
        _574.c[1] = _571.c[1];
        _574.c[2] = _571.c[2];
        _574.ior[0] = _571.ior[0];
        _574.ior[1] = _571.ior[1];
        _574.ior[2] = _571.ior[2];
        _574.ior[3] = _571.ior[3];
        _574.cone_width = _571.cone_width;
        _574.cone_spread = _571.cone_spread;
        _574.xy = _571.xy;
        _574.depth = _571.depth;
        int rand_index = _465_g_params.hi + (total_depth(_574) * 7);
        int _651;
        float _904;
        float _587;
        for (;;)
        {
            _587 = _1051;
            rayQueryInitializeEXT(rq, g_tlas, 0u, 255u, ro, 0.0f, _536, _587);
            for (;;)
            {
                bool _604 = rayQueryProceedEXT(rq);
                if (_604)
                {
                    rayQueryConfirmIntersectionEXT(rq);
                    continue;
                }
                else
                {
                    break;
                }
            }
            uint _605 = rayQueryGetIntersectionTypeEXT(rq, bool(1));
            if (_605 != 0u)
            {
                int _610 = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, bool(1));
                _1048 = -1;
                int _613 = rayQueryGetIntersectionInstanceIdEXT(rq, bool(1));
                _1049 = _613;
                int _616 = rayQueryGetIntersectionPrimitiveIndexEXT(rq, bool(1));
                _1050 = _610 + _616;
                bool _619 = rayQueryGetIntersectionFrontFaceEXT(rq, bool(1));
                [flatten]
                if (_619 == false)
                {
                    _1050 = (-1) - _1050;
                }
                float2 _629 = rayQueryGetIntersectionBarycentricsEXT(rq, bool(1));
                _1052 = _629.x;
                _1053 = _629.y;
                float _636 = rayQueryGetIntersectionTEXT(rq, bool(1));
                _1051 = _636;
            }
            if (_1048 == 0)
            {
                break;
            }
            bool _648 = _1050 < 0;
            if (_648)
            {
                _651 = (-1) - _1050;
            }
            else
            {
                _651 = _1050;
            }
            uint _662 = uint(_651);
            bool _664 = !_648;
            bool _679;
            if (_664)
            {
                _679 = ((_670.Load(_662 * 4 + 0) >> 16u) & 32768u) != 0u;
            }
            else
            {
                _679 = _664;
            }
            bool _692;
            if (!_679)
            {
                bool _691;
                if (_648)
                {
                    _691 = (_670.Load(_662 * 4 + 0) & 32768u) != 0u;
                }
                else
                {
                    _691 = _648;
                }
                _692 = _691;
            }
            else
            {
                _692 = _679;
            }
            if (_692)
            {
                break;
            }
            material_t _715;
            [unroll]
            for (int _35ident = 0; _35ident < 5; _35ident++)
            {
                _715.textures[_35ident] = _707.Load(_35ident * 4 + ((_670.Load(_662 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _715.base_color[_36ident] = asfloat(_707.Load(_36ident * 4 + ((_670.Load(_662 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
            }
            _715.flags = _707.Load(((_670.Load(_662 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
            _715.type = _707.Load(((_670.Load(_662 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
            _715.tangent_rotation_or_strength = asfloat(_707.Load(((_670.Load(_662 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
            _715.roughness_and_anisotropic = _707.Load(((_670.Load(_662 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
            _715.ior = asfloat(_707.Load(((_670.Load(_662 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
            _715.sheen_and_sheen_tint = _707.Load(((_670.Load(_662 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
            _715.tint_and_metallic = _707.Load(((_670.Load(_662 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
            _715.transmission_and_transmission_roughness = _707.Load(((_670.Load(_662 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
            _715.specular_and_specular_tint = _707.Load(((_670.Load(_662 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
            _715.clearcoat_and_clearcoat_roughness = _707.Load(((_670.Load(_662 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
            _715.normal_map_strength_unorm = _707.Load(((_670.Load(_662 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
            material_t _716;
            _716.textures[0] = _715.textures[0];
            _716.textures[1] = _715.textures[1];
            _716.textures[2] = _715.textures[2];
            _716.textures[3] = _715.textures[3];
            _716.textures[4] = _715.textures[4];
            _716.base_color[0] = _715.base_color[0];
            _716.base_color[1] = _715.base_color[1];
            _716.base_color[2] = _715.base_color[2];
            _716.flags = _715.flags;
            _716.type = _715.type;
            _716.tangent_rotation_or_strength = _715.tangent_rotation_or_strength;
            _716.roughness_and_anisotropic = _715.roughness_and_anisotropic;
            _716.ior = _715.ior;
            _716.sheen_and_sheen_tint = _715.sheen_and_sheen_tint;
            _716.tint_and_metallic = _715.tint_and_metallic;
            _716.transmission_and_transmission_roughness = _715.transmission_and_transmission_roughness;
            _716.specular_and_specular_tint = _715.specular_and_specular_tint;
            _716.clearcoat_and_clearcoat_roughness = _715.clearcoat_and_clearcoat_roughness;
            _716.normal_map_strength_unorm = _715.normal_map_strength_unorm;
            uint _1106 = _716.textures[1];
            uint _1107 = _716.textures[3];
            uint _1108 = _716.textures[4];
            float _1121 = _716.base_color[0];
            float _1122 = _716.base_color[1];
            float _1123 = _716.base_color[2];
            uint _1066 = _716.type;
            float _1067 = _716.tangent_rotation_or_strength;
            if (_648)
            {
                material_t _725;
                [unroll]
                for (int _37ident = 0; _37ident < 5; _37ident++)
                {
                    _725.textures[_37ident] = _707.Load(_37ident * 4 + (_670.Load(_662 * 4 + 0) & 16383u) * 76 + 0);
                }
                [unroll]
                for (int _38ident = 0; _38ident < 3; _38ident++)
                {
                    _725.base_color[_38ident] = asfloat(_707.Load(_38ident * 4 + (_670.Load(_662 * 4 + 0) & 16383u) * 76 + 20));
                }
                _725.flags = _707.Load((_670.Load(_662 * 4 + 0) & 16383u) * 76 + 32);
                _725.type = _707.Load((_670.Load(_662 * 4 + 0) & 16383u) * 76 + 36);
                _725.tangent_rotation_or_strength = asfloat(_707.Load((_670.Load(_662 * 4 + 0) & 16383u) * 76 + 40));
                _725.roughness_and_anisotropic = _707.Load((_670.Load(_662 * 4 + 0) & 16383u) * 76 + 44);
                _725.ior = asfloat(_707.Load((_670.Load(_662 * 4 + 0) & 16383u) * 76 + 48));
                _725.sheen_and_sheen_tint = _707.Load((_670.Load(_662 * 4 + 0) & 16383u) * 76 + 52);
                _725.tint_and_metallic = _707.Load((_670.Load(_662 * 4 + 0) & 16383u) * 76 + 56);
                _725.transmission_and_transmission_roughness = _707.Load((_670.Load(_662 * 4 + 0) & 16383u) * 76 + 60);
                _725.specular_and_specular_tint = _707.Load((_670.Load(_662 * 4 + 0) & 16383u) * 76 + 64);
                _725.clearcoat_and_clearcoat_roughness = _707.Load((_670.Load(_662 * 4 + 0) & 16383u) * 76 + 68);
                _725.normal_map_strength_unorm = _707.Load((_670.Load(_662 * 4 + 0) & 16383u) * 76 + 72);
                material_t _726;
                _726.textures[0] = _725.textures[0];
                _726.textures[1] = _725.textures[1];
                _726.textures[2] = _725.textures[2];
                _726.textures[3] = _725.textures[3];
                _726.textures[4] = _725.textures[4];
                _726.base_color[0] = _725.base_color[0];
                _726.base_color[1] = _725.base_color[1];
                _726.base_color[2] = _725.base_color[2];
                _726.flags = _725.flags;
                _726.type = _725.type;
                _726.tangent_rotation_or_strength = _725.tangent_rotation_or_strength;
                _726.roughness_and_anisotropic = _725.roughness_and_anisotropic;
                _726.ior = _725.ior;
                _726.sheen_and_sheen_tint = _725.sheen_and_sheen_tint;
                _726.tint_and_metallic = _725.tint_and_metallic;
                _726.transmission_and_transmission_roughness = _725.transmission_and_transmission_roughness;
                _726.specular_and_specular_tint = _725.specular_and_specular_tint;
                _726.clearcoat_and_clearcoat_roughness = _725.clearcoat_and_clearcoat_roughness;
                _726.normal_map_strength_unorm = _725.normal_map_strength_unorm;
                _1106 = _726.textures[1];
                _1107 = _726.textures[3];
                _1108 = _726.textures[4];
                _1121 = _726.base_color[0];
                _1122 = _726.base_color[1];
                _1123 = _726.base_color[2];
                _1066 = _726.type;
                _1067 = _726.tangent_rotation_or_strength;
            }
            uint _747 = _662 * 3u;
            vertex_t _753;
            [unroll]
            for (int _39ident = 0; _39ident < 3; _39ident++)
            {
                _753.p[_39ident] = asfloat(_741.Load(_39ident * 4 + _745.Load(_747 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _753.n[_40ident] = asfloat(_741.Load(_40ident * 4 + _745.Load(_747 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 3; _41ident++)
            {
                _753.b[_41ident] = asfloat(_741.Load(_41ident * 4 + _745.Load(_747 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _42ident = 0; _42ident < 2; _42ident++)
            {
                [unroll]
                for (int _43ident = 0; _43ident < 2; _43ident++)
                {
                    _753.t[_42ident][_43ident] = asfloat(_741.Load(_43ident * 4 + _42ident * 8 + _745.Load(_747 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _754;
            _754.p[0] = _753.p[0];
            _754.p[1] = _753.p[1];
            _754.p[2] = _753.p[2];
            _754.n[0] = _753.n[0];
            _754.n[1] = _753.n[1];
            _754.n[2] = _753.n[2];
            _754.b[0] = _753.b[0];
            _754.b[1] = _753.b[1];
            _754.b[2] = _753.b[2];
            _754.t[0][0] = _753.t[0][0];
            _754.t[0][1] = _753.t[0][1];
            _754.t[1][0] = _753.t[1][0];
            _754.t[1][1] = _753.t[1][1];
            vertex_t _762;
            [unroll]
            for (int _44ident = 0; _44ident < 3; _44ident++)
            {
                _762.p[_44ident] = asfloat(_741.Load(_44ident * 4 + _745.Load((_747 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _45ident = 0; _45ident < 3; _45ident++)
            {
                _762.n[_45ident] = asfloat(_741.Load(_45ident * 4 + _745.Load((_747 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _46ident = 0; _46ident < 3; _46ident++)
            {
                _762.b[_46ident] = asfloat(_741.Load(_46ident * 4 + _745.Load((_747 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _47ident = 0; _47ident < 2; _47ident++)
            {
                [unroll]
                for (int _48ident = 0; _48ident < 2; _48ident++)
                {
                    _762.t[_47ident][_48ident] = asfloat(_741.Load(_48ident * 4 + _47ident * 8 + _745.Load((_747 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _763;
            _763.p[0] = _762.p[0];
            _763.p[1] = _762.p[1];
            _763.p[2] = _762.p[2];
            _763.n[0] = _762.n[0];
            _763.n[1] = _762.n[1];
            _763.n[2] = _762.n[2];
            _763.b[0] = _762.b[0];
            _763.b[1] = _762.b[1];
            _763.b[2] = _762.b[2];
            _763.t[0][0] = _762.t[0][0];
            _763.t[0][1] = _762.t[0][1];
            _763.t[1][0] = _762.t[1][0];
            _763.t[1][1] = _762.t[1][1];
            vertex_t _771;
            [unroll]
            for (int _49ident = 0; _49ident < 3; _49ident++)
            {
                _771.p[_49ident] = asfloat(_741.Load(_49ident * 4 + _745.Load((_747 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _50ident = 0; _50ident < 3; _50ident++)
            {
                _771.n[_50ident] = asfloat(_741.Load(_50ident * 4 + _745.Load((_747 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _51ident = 0; _51ident < 3; _51ident++)
            {
                _771.b[_51ident] = asfloat(_741.Load(_51ident * 4 + _745.Load((_747 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _52ident = 0; _52ident < 2; _52ident++)
            {
                [unroll]
                for (int _53ident = 0; _53ident < 2; _53ident++)
                {
                    _771.t[_52ident][_53ident] = asfloat(_741.Load(_53ident * 4 + _52ident * 8 + _745.Load((_747 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _772;
            _772.p[0] = _771.p[0];
            _772.p[1] = _771.p[1];
            _772.p[2] = _771.p[2];
            _772.n[0] = _771.n[0];
            _772.n[1] = _771.n[1];
            _772.n[2] = _771.n[2];
            _772.b[0] = _771.b[0];
            _772.b[1] = _771.b[1];
            _772.b[2] = _771.b[2];
            _772.t[0][0] = _771.t[0][0];
            _772.t[0][1] = _771.t[0][1];
            _772.t[1][0] = _771.t[1][0];
            _772.t[1][1] = _771.t[1][1];
            float2 _805 = ((float2(_754.t[0][0], _754.t[0][1]) * ((1.0f - _1052) - _1053)) + (float2(_763.t[0][0], _763.t[0][1]) * _1052)) + (float2(_772.t[0][0], _772.t[0][1]) * _1053);
            float trans_r = frac(asfloat(_810.Load(rand_index * 4 + 0)) + _563);
            while (_1066 == 4u)
            {
                float mix_val = _1067;
                if (_1106 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_1106, _805, 0).x;
                }
                if (trans_r > mix_val)
                {
                    material_t _850;
                    [unroll]
                    for (int _54ident = 0; _54ident < 5; _54ident++)
                    {
                        _850.textures[_54ident] = _707.Load(_54ident * 4 + _1107 * 76 + 0);
                    }
                    [unroll]
                    for (int _55ident = 0; _55ident < 3; _55ident++)
                    {
                        _850.base_color[_55ident] = asfloat(_707.Load(_55ident * 4 + _1107 * 76 + 20));
                    }
                    _850.flags = _707.Load(_1107 * 76 + 32);
                    _850.type = _707.Load(_1107 * 76 + 36);
                    _850.tangent_rotation_or_strength = asfloat(_707.Load(_1107 * 76 + 40));
                    _850.roughness_and_anisotropic = _707.Load(_1107 * 76 + 44);
                    _850.ior = asfloat(_707.Load(_1107 * 76 + 48));
                    _850.sheen_and_sheen_tint = _707.Load(_1107 * 76 + 52);
                    _850.tint_and_metallic = _707.Load(_1107 * 76 + 56);
                    _850.transmission_and_transmission_roughness = _707.Load(_1107 * 76 + 60);
                    _850.specular_and_specular_tint = _707.Load(_1107 * 76 + 64);
                    _850.clearcoat_and_clearcoat_roughness = _707.Load(_1107 * 76 + 68);
                    _850.normal_map_strength_unorm = _707.Load(_1107 * 76 + 72);
                    material_t _851;
                    _851.textures[0] = _850.textures[0];
                    _851.textures[1] = _850.textures[1];
                    _851.textures[2] = _850.textures[2];
                    _851.textures[3] = _850.textures[3];
                    _851.textures[4] = _850.textures[4];
                    _851.base_color[0] = _850.base_color[0];
                    _851.base_color[1] = _850.base_color[1];
                    _851.base_color[2] = _850.base_color[2];
                    _851.flags = _850.flags;
                    _851.type = _850.type;
                    _851.tangent_rotation_or_strength = _850.tangent_rotation_or_strength;
                    _851.roughness_and_anisotropic = _850.roughness_and_anisotropic;
                    _851.ior = _850.ior;
                    _851.sheen_and_sheen_tint = _850.sheen_and_sheen_tint;
                    _851.tint_and_metallic = _850.tint_and_metallic;
                    _851.transmission_and_transmission_roughness = _850.transmission_and_transmission_roughness;
                    _851.specular_and_specular_tint = _850.specular_and_specular_tint;
                    _851.clearcoat_and_clearcoat_roughness = _850.clearcoat_and_clearcoat_roughness;
                    _851.normal_map_strength_unorm = _850.normal_map_strength_unorm;
                    _1106 = _851.textures[1];
                    _1107 = _851.textures[3];
                    _1108 = _851.textures[4];
                    _1121 = _851.base_color[0];
                    _1122 = _851.base_color[1];
                    _1123 = _851.base_color[2];
                    _1066 = _851.type;
                    _1067 = _851.tangent_rotation_or_strength;
                    trans_r = (trans_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _862;
                    [unroll]
                    for (int _56ident = 0; _56ident < 5; _56ident++)
                    {
                        _862.textures[_56ident] = _707.Load(_56ident * 4 + _1108 * 76 + 0);
                    }
                    [unroll]
                    for (int _57ident = 0; _57ident < 3; _57ident++)
                    {
                        _862.base_color[_57ident] = asfloat(_707.Load(_57ident * 4 + _1108 * 76 + 20));
                    }
                    _862.flags = _707.Load(_1108 * 76 + 32);
                    _862.type = _707.Load(_1108 * 76 + 36);
                    _862.tangent_rotation_or_strength = asfloat(_707.Load(_1108 * 76 + 40));
                    _862.roughness_and_anisotropic = _707.Load(_1108 * 76 + 44);
                    _862.ior = asfloat(_707.Load(_1108 * 76 + 48));
                    _862.sheen_and_sheen_tint = _707.Load(_1108 * 76 + 52);
                    _862.tint_and_metallic = _707.Load(_1108 * 76 + 56);
                    _862.transmission_and_transmission_roughness = _707.Load(_1108 * 76 + 60);
                    _862.specular_and_specular_tint = _707.Load(_1108 * 76 + 64);
                    _862.clearcoat_and_clearcoat_roughness = _707.Load(_1108 * 76 + 68);
                    _862.normal_map_strength_unorm = _707.Load(_1108 * 76 + 72);
                    material_t _863;
                    _863.textures[0] = _862.textures[0];
                    _863.textures[1] = _862.textures[1];
                    _863.textures[2] = _862.textures[2];
                    _863.textures[3] = _862.textures[3];
                    _863.textures[4] = _862.textures[4];
                    _863.base_color[0] = _862.base_color[0];
                    _863.base_color[1] = _862.base_color[1];
                    _863.base_color[2] = _862.base_color[2];
                    _863.flags = _862.flags;
                    _863.type = _862.type;
                    _863.tangent_rotation_or_strength = _862.tangent_rotation_or_strength;
                    _863.roughness_and_anisotropic = _862.roughness_and_anisotropic;
                    _863.ior = _862.ior;
                    _863.sheen_and_sheen_tint = _862.sheen_and_sheen_tint;
                    _863.tint_and_metallic = _862.tint_and_metallic;
                    _863.transmission_and_transmission_roughness = _862.transmission_and_transmission_roughness;
                    _863.specular_and_specular_tint = _862.specular_and_specular_tint;
                    _863.clearcoat_and_clearcoat_roughness = _862.clearcoat_and_clearcoat_roughness;
                    _863.normal_map_strength_unorm = _862.normal_map_strength_unorm;
                    _1106 = _863.textures[1];
                    _1107 = _863.textures[3];
                    _1108 = _863.textures[4];
                    _1121 = _863.base_color[0];
                    _1122 = _863.base_color[1];
                    _1123 = _863.base_color[2];
                    _1066 = _863.type;
                    _1067 = _863.tangent_rotation_or_strength;
                    trans_r /= mix_val;
                }
            }
            if (_1066 != 5u)
            {
                break;
            }
            float _892 = max(asfloat(_514.Load(_504 * 72 + 28)), max(asfloat(_514.Load(_504 * 72 + 32)), asfloat(_514.Load(_504 * 72 + 36))));
            if ((int(_514.Load(_504 * 72 + 68)) >> 24) > _465_g_params.min_transp_depth)
            {
                _904 = max(0.0500000007450580596923828125f, 1.0f - _892);
            }
            else
            {
                _904 = 0.0f;
            }
            bool _918 = (frac(asfloat(_810.Load((rand_index + 6) * 4 + 0)) + _563) < _904) || (_892 == 0.0f);
            bool _930;
            if (!_918)
            {
                _930 = ((int(_514.Load(_504 * 72 + 68)) >> 24) + 1) >= _465_g_params.max_transp_depth;
            }
            else
            {
                _930 = _918;
            }
            if (_930)
            {
                _514.Store(_504 * 72 + 36, asuint(0.0f));
                _514.Store(_504 * 72 + 32, asuint(0.0f));
                _514.Store(_504 * 72 + 28, asuint(0.0f));
                break;
            }
            float _944 = 1.0f - _904;
            _514.Store(_504 * 72 + 28, asuint(asfloat(_514.Load(_504 * 72 + 28)) * (_1121 / _944)));
            _514.Store(_504 * 72 + 32, asuint(asfloat(_514.Load(_504 * 72 + 32)) * (_1122 / _944)));
            _514.Store(_504 * 72 + 36, asuint(asfloat(_514.Load(_504 * 72 + 36)) * (_1123 / _944)));
            ro += (_536 * (_1051 + 9.9999997473787516355514526367188e-06f));
            _1048 = 0;
            _1051 = _587 - _1051;
            _514.Store(_504 * 72 + 68, uint(int(_514.Load(_504 * 72 + 68)) + 16777216));
            rand_index += 7;
            continue;
        }
        float _1008 = _1051;
        float _1009 = _1008 + length(float3(asfloat(_514.Load(_504 * 72 + 0)), asfloat(_514.Load(_504 * 72 + 4)), asfloat(_514.Load(_504 * 72 + 8))) - ro);
        _1051 = _1009;
        hit_data_t _1060 = { _1048, _1049, _1050, _1009, _1052, _1053 };
        hit_data_t _1020;
        _1020.mask = _1060.mask;
        _1020.obj_index = _1060.obj_index;
        _1020.prim_index = _1060.prim_index;
        _1020.t = _1060.t;
        _1020.u = _1060.u;
        _1020.v = _1060.v;
        _1015.Store(_504 * 24 + 0, uint(_1020.mask));
        _1015.Store(_504 * 24 + 4, uint(_1020.obj_index));
        _1015.Store(_504 * 24 + 8, uint(_1020.prim_index));
        _1015.Store(_504 * 24 + 12, asuint(_1020.t));
        _1015.Store(_504 * 24 + 16, asuint(_1020.u));
        _1015.Store(_504 * 24 + 20, asuint(_1020.v));
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
