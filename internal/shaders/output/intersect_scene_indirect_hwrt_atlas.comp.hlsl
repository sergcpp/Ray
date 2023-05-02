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

ByteAddressBuffer _369 : register(t20, space0);
ByteAddressBuffer _473 : register(t14, space0);
RWByteAddressBuffer _490 : register(u12, space0);
ByteAddressBuffer _663 : register(t5, space0);
ByteAddressBuffer _699 : register(t6, space0);
ByteAddressBuffer _733 : register(t1, space0);
ByteAddressBuffer _737 : register(t2, space0);
ByteAddressBuffer _802 : register(t15, space0);
RWByteAddressBuffer _1007 : register(u0, space0);
cbuffer UniformParams
{
    Params _541_g_params : packoffset(c0);
};

Texture2DArray<float4> g_atlases[7] : register(t21, space0);
SamplerState _g_atlases_sampler[7] : register(s21, space0);
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
        float3 _1176 = inv_v;
        _1176.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _1176;
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
            float3 _1174 = inv_v;
            _1174.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _1174;
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
        float3 _1180 = inv_v;
        _1180.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _1180;
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
            float3 _1178 = inv_v;
            _1178.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _1178;
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
        float3 _1184 = inv_v;
        _1184.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _1184;
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
            float3 _1182 = inv_v;
            _1182.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _1182;
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
    uint _1156[14] = t.pos;
    uint _1159[14] = t.pos;
    uint _327 = t.size & 16383u;
    uint _330 = t.size >> uint(16);
    uint _331 = _330 & 16383u;
    float2 size = float2(float(_327), float(_331));
    if ((_330 & 32768u) != 0u)
    {
        size = float2(float(_327 >> uint(mip_level)), float(_331 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_1156[mip_level] & 65535u), float((_1159[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
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
    uint _1165[4] = _373.page;
    uint _401 = _373.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_401)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_401)], float3(TransformUV(uvs, _373, lod) * 0.000118371215648949146270751953125f.xx, float((_1165[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
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
        float4 _1200 = res;
        _1200.x = _441.x;
        float4 _1202 = _1200;
        _1202.y = _441.y;
        float4 _1204 = _1202;
        _1204.z = _441.z;
        res = _1204;
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
        int _467 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_467) >= _473.Load(4))
        {
            break;
        }
        float3 ro = float3(asfloat(_490.Load(_467 * 72 + 0)), asfloat(_490.Load(_467 * 72 + 4)), asfloat(_490.Load(_467 * 72 + 8)));
        float _519 = asfloat(_490.Load(_467 * 72 + 12));
        float _522 = asfloat(_490.Load(_467 * 72 + 16));
        float _525 = asfloat(_490.Load(_467 * 72 + 20));
        float3 _526 = float3(_519, _522, _525);
        float3 param = _526;
        int _1040 = 0;
        int _1042 = 0;
        int _1041 = 0;
        float _1043 = _541_g_params.inter_t;
        float _1045 = 0.0f;
        float _1044 = 0.0f;
        uint param_1 = uint(hash(int(_490.Load(_467 * 72 + 64))));
        float _556 = construct_float(param_1);
        ray_data_t _564;
        [unroll]
        for (int _31ident = 0; _31ident < 3; _31ident++)
        {
            _564.o[_31ident] = asfloat(_490.Load(_31ident * 4 + _467 * 72 + 0));
        }
        [unroll]
        for (int _32ident = 0; _32ident < 3; _32ident++)
        {
            _564.d[_32ident] = asfloat(_490.Load(_32ident * 4 + _467 * 72 + 12));
        }
        _564.pdf = asfloat(_490.Load(_467 * 72 + 24));
        [unroll]
        for (int _33ident = 0; _33ident < 3; _33ident++)
        {
            _564.c[_33ident] = asfloat(_490.Load(_33ident * 4 + _467 * 72 + 28));
        }
        [unroll]
        for (int _34ident = 0; _34ident < 4; _34ident++)
        {
            _564.ior[_34ident] = asfloat(_490.Load(_34ident * 4 + _467 * 72 + 40));
        }
        _564.cone_width = asfloat(_490.Load(_467 * 72 + 56));
        _564.cone_spread = asfloat(_490.Load(_467 * 72 + 60));
        _564.xy = int(_490.Load(_467 * 72 + 64));
        _564.depth = int(_490.Load(_467 * 72 + 68));
        ray_data_t _567;
        _567.o[0] = _564.o[0];
        _567.o[1] = _564.o[1];
        _567.o[2] = _564.o[2];
        _567.d[0] = _564.d[0];
        _567.d[1] = _564.d[1];
        _567.d[2] = _564.d[2];
        _567.pdf = _564.pdf;
        _567.c[0] = _564.c[0];
        _567.c[1] = _564.c[1];
        _567.c[2] = _564.c[2];
        _567.ior[0] = _564.ior[0];
        _567.ior[1] = _564.ior[1];
        _567.ior[2] = _564.ior[2];
        _567.ior[3] = _564.ior[3];
        _567.cone_width = _564.cone_width;
        _567.cone_spread = _564.cone_spread;
        _567.xy = _564.xy;
        _567.depth = _564.depth;
        int rand_index = _541_g_params.hi + (total_depth(_567) * 7);
        int _644;
        float _896;
        float _580;
        for (;;)
        {
            _580 = _1043;
            rayQueryInitializeEXT(rq, g_tlas, 0u, 255u, ro, 0.0f, _526, _580);
            for (;;)
            {
                bool _597 = rayQueryProceedEXT(rq);
                if (_597)
                {
                    rayQueryConfirmIntersectionEXT(rq);
                    continue;
                }
                else
                {
                    break;
                }
            }
            uint _598 = rayQueryGetIntersectionTypeEXT(rq, bool(1));
            if (_598 != 0u)
            {
                int _603 = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, bool(1));
                _1040 = -1;
                int _606 = rayQueryGetIntersectionInstanceIdEXT(rq, bool(1));
                _1041 = _606;
                int _609 = rayQueryGetIntersectionPrimitiveIndexEXT(rq, bool(1));
                _1042 = _603 + _609;
                bool _612 = rayQueryGetIntersectionFrontFaceEXT(rq, bool(1));
                [flatten]
                if (_612 == false)
                {
                    _1042 = (-1) - _1042;
                }
                float2 _622 = rayQueryGetIntersectionBarycentricsEXT(rq, bool(1));
                _1044 = _622.x;
                _1045 = _622.y;
                float _629 = rayQueryGetIntersectionTEXT(rq, bool(1));
                _1043 = _629;
            }
            if (_1040 == 0)
            {
                break;
            }
            bool _641 = _1042 < 0;
            if (_641)
            {
                _644 = (-1) - _1042;
            }
            else
            {
                _644 = _1042;
            }
            uint _655 = uint(_644);
            bool _657 = !_641;
            bool _671;
            if (_657)
            {
                _671 = ((_663.Load(_655 * 4 + 0) >> 16u) & 32768u) != 0u;
            }
            else
            {
                _671 = _657;
            }
            bool _684;
            if (!_671)
            {
                bool _683;
                if (_641)
                {
                    _683 = (_663.Load(_655 * 4 + 0) & 32768u) != 0u;
                }
                else
                {
                    _683 = _641;
                }
                _684 = _683;
            }
            else
            {
                _684 = _671;
            }
            if (_684)
            {
                break;
            }
            material_t _707;
            [unroll]
            for (int _35ident = 0; _35ident < 5; _35ident++)
            {
                _707.textures[_35ident] = _699.Load(_35ident * 4 + ((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _707.base_color[_36ident] = asfloat(_699.Load(_36ident * 4 + ((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
            }
            _707.flags = _699.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
            _707.type = _699.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
            _707.tangent_rotation_or_strength = asfloat(_699.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
            _707.roughness_and_anisotropic = _699.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
            _707.ior = asfloat(_699.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
            _707.sheen_and_sheen_tint = _699.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
            _707.tint_and_metallic = _699.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
            _707.transmission_and_transmission_roughness = _699.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
            _707.specular_and_specular_tint = _699.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
            _707.clearcoat_and_clearcoat_roughness = _699.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
            _707.normal_map_strength_unorm = _699.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
            material_t _708;
            _708.textures[0] = _707.textures[0];
            _708.textures[1] = _707.textures[1];
            _708.textures[2] = _707.textures[2];
            _708.textures[3] = _707.textures[3];
            _708.textures[4] = _707.textures[4];
            _708.base_color[0] = _707.base_color[0];
            _708.base_color[1] = _707.base_color[1];
            _708.base_color[2] = _707.base_color[2];
            _708.flags = _707.flags;
            _708.type = _707.type;
            _708.tangent_rotation_or_strength = _707.tangent_rotation_or_strength;
            _708.roughness_and_anisotropic = _707.roughness_and_anisotropic;
            _708.ior = _707.ior;
            _708.sheen_and_sheen_tint = _707.sheen_and_sheen_tint;
            _708.tint_and_metallic = _707.tint_and_metallic;
            _708.transmission_and_transmission_roughness = _707.transmission_and_transmission_roughness;
            _708.specular_and_specular_tint = _707.specular_and_specular_tint;
            _708.clearcoat_and_clearcoat_roughness = _707.clearcoat_and_clearcoat_roughness;
            _708.normal_map_strength_unorm = _707.normal_map_strength_unorm;
            uint _1098 = _708.textures[1];
            uint _1099 = _708.textures[3];
            uint _1100 = _708.textures[4];
            float _1113 = _708.base_color[0];
            float _1114 = _708.base_color[1];
            float _1115 = _708.base_color[2];
            uint _1058 = _708.type;
            float _1059 = _708.tangent_rotation_or_strength;
            if (_641)
            {
                material_t _717;
                [unroll]
                for (int _37ident = 0; _37ident < 5; _37ident++)
                {
                    _717.textures[_37ident] = _699.Load(_37ident * 4 + (_663.Load(_655 * 4 + 0) & 16383u) * 76 + 0);
                }
                [unroll]
                for (int _38ident = 0; _38ident < 3; _38ident++)
                {
                    _717.base_color[_38ident] = asfloat(_699.Load(_38ident * 4 + (_663.Load(_655 * 4 + 0) & 16383u) * 76 + 20));
                }
                _717.flags = _699.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 32);
                _717.type = _699.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 36);
                _717.tangent_rotation_or_strength = asfloat(_699.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 40));
                _717.roughness_and_anisotropic = _699.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 44);
                _717.ior = asfloat(_699.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 48));
                _717.sheen_and_sheen_tint = _699.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 52);
                _717.tint_and_metallic = _699.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 56);
                _717.transmission_and_transmission_roughness = _699.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 60);
                _717.specular_and_specular_tint = _699.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 64);
                _717.clearcoat_and_clearcoat_roughness = _699.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 68);
                _717.normal_map_strength_unorm = _699.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 72);
                material_t _718;
                _718.textures[0] = _717.textures[0];
                _718.textures[1] = _717.textures[1];
                _718.textures[2] = _717.textures[2];
                _718.textures[3] = _717.textures[3];
                _718.textures[4] = _717.textures[4];
                _718.base_color[0] = _717.base_color[0];
                _718.base_color[1] = _717.base_color[1];
                _718.base_color[2] = _717.base_color[2];
                _718.flags = _717.flags;
                _718.type = _717.type;
                _718.tangent_rotation_or_strength = _717.tangent_rotation_or_strength;
                _718.roughness_and_anisotropic = _717.roughness_and_anisotropic;
                _718.ior = _717.ior;
                _718.sheen_and_sheen_tint = _717.sheen_and_sheen_tint;
                _718.tint_and_metallic = _717.tint_and_metallic;
                _718.transmission_and_transmission_roughness = _717.transmission_and_transmission_roughness;
                _718.specular_and_specular_tint = _717.specular_and_specular_tint;
                _718.clearcoat_and_clearcoat_roughness = _717.clearcoat_and_clearcoat_roughness;
                _718.normal_map_strength_unorm = _717.normal_map_strength_unorm;
                _1098 = _718.textures[1];
                _1099 = _718.textures[3];
                _1100 = _718.textures[4];
                _1113 = _718.base_color[0];
                _1114 = _718.base_color[1];
                _1115 = _718.base_color[2];
                _1058 = _718.type;
                _1059 = _718.tangent_rotation_or_strength;
            }
            uint _739 = _655 * 3u;
            vertex_t _745;
            [unroll]
            for (int _39ident = 0; _39ident < 3; _39ident++)
            {
                _745.p[_39ident] = asfloat(_733.Load(_39ident * 4 + _737.Load(_739 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _745.n[_40ident] = asfloat(_733.Load(_40ident * 4 + _737.Load(_739 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 3; _41ident++)
            {
                _745.b[_41ident] = asfloat(_733.Load(_41ident * 4 + _737.Load(_739 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _42ident = 0; _42ident < 2; _42ident++)
            {
                [unroll]
                for (int _43ident = 0; _43ident < 2; _43ident++)
                {
                    _745.t[_42ident][_43ident] = asfloat(_733.Load(_43ident * 4 + _42ident * 8 + _737.Load(_739 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _746;
            _746.p[0] = _745.p[0];
            _746.p[1] = _745.p[1];
            _746.p[2] = _745.p[2];
            _746.n[0] = _745.n[0];
            _746.n[1] = _745.n[1];
            _746.n[2] = _745.n[2];
            _746.b[0] = _745.b[0];
            _746.b[1] = _745.b[1];
            _746.b[2] = _745.b[2];
            _746.t[0][0] = _745.t[0][0];
            _746.t[0][1] = _745.t[0][1];
            _746.t[1][0] = _745.t[1][0];
            _746.t[1][1] = _745.t[1][1];
            vertex_t _754;
            [unroll]
            for (int _44ident = 0; _44ident < 3; _44ident++)
            {
                _754.p[_44ident] = asfloat(_733.Load(_44ident * 4 + _737.Load((_739 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _45ident = 0; _45ident < 3; _45ident++)
            {
                _754.n[_45ident] = asfloat(_733.Load(_45ident * 4 + _737.Load((_739 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _46ident = 0; _46ident < 3; _46ident++)
            {
                _754.b[_46ident] = asfloat(_733.Load(_46ident * 4 + _737.Load((_739 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _47ident = 0; _47ident < 2; _47ident++)
            {
                [unroll]
                for (int _48ident = 0; _48ident < 2; _48ident++)
                {
                    _754.t[_47ident][_48ident] = asfloat(_733.Load(_48ident * 4 + _47ident * 8 + _737.Load((_739 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _755;
            _755.p[0] = _754.p[0];
            _755.p[1] = _754.p[1];
            _755.p[2] = _754.p[2];
            _755.n[0] = _754.n[0];
            _755.n[1] = _754.n[1];
            _755.n[2] = _754.n[2];
            _755.b[0] = _754.b[0];
            _755.b[1] = _754.b[1];
            _755.b[2] = _754.b[2];
            _755.t[0][0] = _754.t[0][0];
            _755.t[0][1] = _754.t[0][1];
            _755.t[1][0] = _754.t[1][0];
            _755.t[1][1] = _754.t[1][1];
            vertex_t _763;
            [unroll]
            for (int _49ident = 0; _49ident < 3; _49ident++)
            {
                _763.p[_49ident] = asfloat(_733.Load(_49ident * 4 + _737.Load((_739 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _50ident = 0; _50ident < 3; _50ident++)
            {
                _763.n[_50ident] = asfloat(_733.Load(_50ident * 4 + _737.Load((_739 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _51ident = 0; _51ident < 3; _51ident++)
            {
                _763.b[_51ident] = asfloat(_733.Load(_51ident * 4 + _737.Load((_739 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _52ident = 0; _52ident < 2; _52ident++)
            {
                [unroll]
                for (int _53ident = 0; _53ident < 2; _53ident++)
                {
                    _763.t[_52ident][_53ident] = asfloat(_733.Load(_53ident * 4 + _52ident * 8 + _737.Load((_739 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _764;
            _764.p[0] = _763.p[0];
            _764.p[1] = _763.p[1];
            _764.p[2] = _763.p[2];
            _764.n[0] = _763.n[0];
            _764.n[1] = _763.n[1];
            _764.n[2] = _763.n[2];
            _764.b[0] = _763.b[0];
            _764.b[1] = _763.b[1];
            _764.b[2] = _763.b[2];
            _764.t[0][0] = _763.t[0][0];
            _764.t[0][1] = _763.t[0][1];
            _764.t[1][0] = _763.t[1][0];
            _764.t[1][1] = _763.t[1][1];
            float2 _797 = ((float2(_746.t[0][0], _746.t[0][1]) * ((1.0f - _1044) - _1045)) + (float2(_755.t[0][0], _755.t[0][1]) * _1044)) + (float2(_764.t[0][0], _764.t[0][1]) * _1045);
            float trans_r = frac(asfloat(_802.Load(rand_index * 4 + 0)) + _556);
            while (_1058 == 4u)
            {
                float mix_val = _1059;
                if (_1098 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_1098, _797, 0).x;
                }
                if (trans_r > mix_val)
                {
                    material_t _842;
                    [unroll]
                    for (int _54ident = 0; _54ident < 5; _54ident++)
                    {
                        _842.textures[_54ident] = _699.Load(_54ident * 4 + _1099 * 76 + 0);
                    }
                    [unroll]
                    for (int _55ident = 0; _55ident < 3; _55ident++)
                    {
                        _842.base_color[_55ident] = asfloat(_699.Load(_55ident * 4 + _1099 * 76 + 20));
                    }
                    _842.flags = _699.Load(_1099 * 76 + 32);
                    _842.type = _699.Load(_1099 * 76 + 36);
                    _842.tangent_rotation_or_strength = asfloat(_699.Load(_1099 * 76 + 40));
                    _842.roughness_and_anisotropic = _699.Load(_1099 * 76 + 44);
                    _842.ior = asfloat(_699.Load(_1099 * 76 + 48));
                    _842.sheen_and_sheen_tint = _699.Load(_1099 * 76 + 52);
                    _842.tint_and_metallic = _699.Load(_1099 * 76 + 56);
                    _842.transmission_and_transmission_roughness = _699.Load(_1099 * 76 + 60);
                    _842.specular_and_specular_tint = _699.Load(_1099 * 76 + 64);
                    _842.clearcoat_and_clearcoat_roughness = _699.Load(_1099 * 76 + 68);
                    _842.normal_map_strength_unorm = _699.Load(_1099 * 76 + 72);
                    material_t _843;
                    _843.textures[0] = _842.textures[0];
                    _843.textures[1] = _842.textures[1];
                    _843.textures[2] = _842.textures[2];
                    _843.textures[3] = _842.textures[3];
                    _843.textures[4] = _842.textures[4];
                    _843.base_color[0] = _842.base_color[0];
                    _843.base_color[1] = _842.base_color[1];
                    _843.base_color[2] = _842.base_color[2];
                    _843.flags = _842.flags;
                    _843.type = _842.type;
                    _843.tangent_rotation_or_strength = _842.tangent_rotation_or_strength;
                    _843.roughness_and_anisotropic = _842.roughness_and_anisotropic;
                    _843.ior = _842.ior;
                    _843.sheen_and_sheen_tint = _842.sheen_and_sheen_tint;
                    _843.tint_and_metallic = _842.tint_and_metallic;
                    _843.transmission_and_transmission_roughness = _842.transmission_and_transmission_roughness;
                    _843.specular_and_specular_tint = _842.specular_and_specular_tint;
                    _843.clearcoat_and_clearcoat_roughness = _842.clearcoat_and_clearcoat_roughness;
                    _843.normal_map_strength_unorm = _842.normal_map_strength_unorm;
                    _1098 = _843.textures[1];
                    _1099 = _843.textures[3];
                    _1100 = _843.textures[4];
                    _1113 = _843.base_color[0];
                    _1114 = _843.base_color[1];
                    _1115 = _843.base_color[2];
                    _1058 = _843.type;
                    _1059 = _843.tangent_rotation_or_strength;
                    trans_r = (trans_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _854;
                    [unroll]
                    for (int _56ident = 0; _56ident < 5; _56ident++)
                    {
                        _854.textures[_56ident] = _699.Load(_56ident * 4 + _1100 * 76 + 0);
                    }
                    [unroll]
                    for (int _57ident = 0; _57ident < 3; _57ident++)
                    {
                        _854.base_color[_57ident] = asfloat(_699.Load(_57ident * 4 + _1100 * 76 + 20));
                    }
                    _854.flags = _699.Load(_1100 * 76 + 32);
                    _854.type = _699.Load(_1100 * 76 + 36);
                    _854.tangent_rotation_or_strength = asfloat(_699.Load(_1100 * 76 + 40));
                    _854.roughness_and_anisotropic = _699.Load(_1100 * 76 + 44);
                    _854.ior = asfloat(_699.Load(_1100 * 76 + 48));
                    _854.sheen_and_sheen_tint = _699.Load(_1100 * 76 + 52);
                    _854.tint_and_metallic = _699.Load(_1100 * 76 + 56);
                    _854.transmission_and_transmission_roughness = _699.Load(_1100 * 76 + 60);
                    _854.specular_and_specular_tint = _699.Load(_1100 * 76 + 64);
                    _854.clearcoat_and_clearcoat_roughness = _699.Load(_1100 * 76 + 68);
                    _854.normal_map_strength_unorm = _699.Load(_1100 * 76 + 72);
                    material_t _855;
                    _855.textures[0] = _854.textures[0];
                    _855.textures[1] = _854.textures[1];
                    _855.textures[2] = _854.textures[2];
                    _855.textures[3] = _854.textures[3];
                    _855.textures[4] = _854.textures[4];
                    _855.base_color[0] = _854.base_color[0];
                    _855.base_color[1] = _854.base_color[1];
                    _855.base_color[2] = _854.base_color[2];
                    _855.flags = _854.flags;
                    _855.type = _854.type;
                    _855.tangent_rotation_or_strength = _854.tangent_rotation_or_strength;
                    _855.roughness_and_anisotropic = _854.roughness_and_anisotropic;
                    _855.ior = _854.ior;
                    _855.sheen_and_sheen_tint = _854.sheen_and_sheen_tint;
                    _855.tint_and_metallic = _854.tint_and_metallic;
                    _855.transmission_and_transmission_roughness = _854.transmission_and_transmission_roughness;
                    _855.specular_and_specular_tint = _854.specular_and_specular_tint;
                    _855.clearcoat_and_clearcoat_roughness = _854.clearcoat_and_clearcoat_roughness;
                    _855.normal_map_strength_unorm = _854.normal_map_strength_unorm;
                    _1098 = _855.textures[1];
                    _1099 = _855.textures[3];
                    _1100 = _855.textures[4];
                    _1113 = _855.base_color[0];
                    _1114 = _855.base_color[1];
                    _1115 = _855.base_color[2];
                    _1058 = _855.type;
                    _1059 = _855.tangent_rotation_or_strength;
                    trans_r /= mix_val;
                }
            }
            if (_1058 != 5u)
            {
                break;
            }
            float _884 = max(asfloat(_490.Load(_467 * 72 + 28)), max(asfloat(_490.Load(_467 * 72 + 32)), asfloat(_490.Load(_467 * 72 + 36))));
            if ((int(_490.Load(_467 * 72 + 68)) >> 24) > _541_g_params.min_transp_depth)
            {
                _896 = max(0.0500000007450580596923828125f, 1.0f - _884);
            }
            else
            {
                _896 = 0.0f;
            }
            bool _910 = (frac(asfloat(_802.Load((rand_index + 6) * 4 + 0)) + _556) < _896) || (_884 == 0.0f);
            bool _922;
            if (!_910)
            {
                _922 = ((int(_490.Load(_467 * 72 + 68)) >> 24) + 1) >= _541_g_params.max_transp_depth;
            }
            else
            {
                _922 = _910;
            }
            if (_922)
            {
                _490.Store(_467 * 72 + 36, asuint(0.0f));
                _490.Store(_467 * 72 + 32, asuint(0.0f));
                _490.Store(_467 * 72 + 28, asuint(0.0f));
                break;
            }
            float _936 = 1.0f - _896;
            _490.Store(_467 * 72 + 28, asuint(asfloat(_490.Load(_467 * 72 + 28)) * (_1113 / _936)));
            _490.Store(_467 * 72 + 32, asuint(asfloat(_490.Load(_467 * 72 + 32)) * (_1114 / _936)));
            _490.Store(_467 * 72 + 36, asuint(asfloat(_490.Load(_467 * 72 + 36)) * (_1115 / _936)));
            ro += (_526 * (_1043 + 9.9999997473787516355514526367188e-06f));
            _1040 = 0;
            _1043 = _580 - _1043;
            _490.Store(_467 * 72 + 68, uint(int(_490.Load(_467 * 72 + 68)) + 16777216));
            rand_index += 7;
            continue;
        }
        float _1000 = _1043;
        float _1001 = _1000 + length(float3(asfloat(_490.Load(_467 * 72 + 0)), asfloat(_490.Load(_467 * 72 + 4)), asfloat(_490.Load(_467 * 72 + 8))) - ro);
        _1043 = _1001;
        hit_data_t _1052 = { _1040, _1041, _1042, _1001, _1044, _1045 };
        hit_data_t _1012;
        _1012.mask = _1052.mask;
        _1012.obj_index = _1052.obj_index;
        _1012.prim_index = _1052.prim_index;
        _1012.t = _1052.t;
        _1012.u = _1052.u;
        _1012.v = _1052.v;
        _1007.Store(_467 * 24 + 0, uint(_1012.mask));
        _1007.Store(_467 * 24 + 4, uint(_1012.obj_index));
        _1007.Store(_467 * 24 + 8, uint(_1012.prim_index));
        _1007.Store(_467 * 24 + 12, asuint(_1012.t));
        _1007.Store(_467 * 24 + 16, asuint(_1012.u));
        _1007.Store(_467 * 24 + 20, asuint(_1012.v));
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
