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
    float cam_clip_end;
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
ByteAddressBuffer _661 : register(t5, space0);
ByteAddressBuffer _697 : register(t6, space0);
ByteAddressBuffer _731 : register(t1, space0);
ByteAddressBuffer _735 : register(t2, space0);
ByteAddressBuffer _800 : register(t15, space0);
RWByteAddressBuffer _1005 : register(u0, space0);
cbuffer UniformParams
{
    Params _555_g_params : packoffset(c0);
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
        float3 _1174 = inv_v;
        _1174.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _1174;
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
            float3 _1172 = inv_v;
            _1172.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _1172;
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
        float3 _1178 = inv_v;
        _1178.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _1178;
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
            float3 _1176 = inv_v;
            _1176.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _1176;
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
        float3 _1182 = inv_v;
        _1182.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _1182;
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
            float3 _1180 = inv_v;
            _1180.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _1180;
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
    uint _1154[14] = t.pos;
    uint _1157[14] = t.pos;
    uint _327 = t.size & 16383u;
    uint _330 = t.size >> uint(16);
    uint _331 = _330 & 16383u;
    float2 size = float2(float(_327), float(_331));
    if ((_330 & 32768u) != 0u)
    {
        size = float2(float(_327 >> uint(mip_level)), float(_331 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_1154[mip_level] & 65535u), float((_1157[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
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
    uint _1163[4] = _373.page;
    uint _401 = _373.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_401)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_401)], float3(TransformUV(uvs, _373, lod) * 0.000118371215648949146270751953125f.xx, float((_1163[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
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
        float4 _1198 = res;
        _1198.x = _441.x;
        float4 _1200 = _1198;
        _1200.y = _441.y;
        float4 _1202 = _1200;
        _1202.z = _441.z;
        res = _1202;
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
        int _1038 = 0;
        int _1040 = 0;
        int _1039 = 0;
        float _1041 = 3402823346297367662189621542912.0f;
        float _1043 = 0.0f;
        float _1042 = 0.0f;
        uint param_1 = uint(hash(int(_490.Load(_467 * 72 + 64))));
        float _549 = construct_float(param_1);
        ray_data_t _562;
        [unroll]
        for (int _31ident = 0; _31ident < 3; _31ident++)
        {
            _562.o[_31ident] = asfloat(_490.Load(_31ident * 4 + _467 * 72 + 0));
        }
        [unroll]
        for (int _32ident = 0; _32ident < 3; _32ident++)
        {
            _562.d[_32ident] = asfloat(_490.Load(_32ident * 4 + _467 * 72 + 12));
        }
        _562.pdf = asfloat(_490.Load(_467 * 72 + 24));
        [unroll]
        for (int _33ident = 0; _33ident < 3; _33ident++)
        {
            _562.c[_33ident] = asfloat(_490.Load(_33ident * 4 + _467 * 72 + 28));
        }
        [unroll]
        for (int _34ident = 0; _34ident < 4; _34ident++)
        {
            _562.ior[_34ident] = asfloat(_490.Load(_34ident * 4 + _467 * 72 + 40));
        }
        _562.cone_width = asfloat(_490.Load(_467 * 72 + 56));
        _562.cone_spread = asfloat(_490.Load(_467 * 72 + 60));
        _562.xy = int(_490.Load(_467 * 72 + 64));
        _562.depth = int(_490.Load(_467 * 72 + 68));
        ray_data_t _565;
        _565.o[0] = _562.o[0];
        _565.o[1] = _562.o[1];
        _565.o[2] = _562.o[2];
        _565.d[0] = _562.d[0];
        _565.d[1] = _562.d[1];
        _565.d[2] = _562.d[2];
        _565.pdf = _562.pdf;
        _565.c[0] = _562.c[0];
        _565.c[1] = _562.c[1];
        _565.c[2] = _562.c[2];
        _565.ior[0] = _562.ior[0];
        _565.ior[1] = _562.ior[1];
        _565.ior[2] = _562.ior[2];
        _565.ior[3] = _562.ior[3];
        _565.cone_width = _562.cone_width;
        _565.cone_spread = _562.cone_spread;
        _565.xy = _562.xy;
        _565.depth = _562.depth;
        int rand_index = _555_g_params.hi + (total_depth(_565) * 7);
        int _642;
        float _894;
        float _578;
        for (;;)
        {
            _578 = _1041;
            rayQueryInitializeEXT(rq, g_tlas, 0u, 255u, ro, 0.0f, _526, _578);
            for (;;)
            {
                bool _595 = rayQueryProceedEXT(rq);
                if (_595)
                {
                    rayQueryConfirmIntersectionEXT(rq);
                    continue;
                }
                else
                {
                    break;
                }
            }
            uint _596 = rayQueryGetIntersectionTypeEXT(rq, bool(1));
            if (_596 != 0u)
            {
                int _601 = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, bool(1));
                _1038 = -1;
                int _604 = rayQueryGetIntersectionInstanceIdEXT(rq, bool(1));
                _1039 = _604;
                int _607 = rayQueryGetIntersectionPrimitiveIndexEXT(rq, bool(1));
                _1040 = _601 + _607;
                bool _610 = rayQueryGetIntersectionFrontFaceEXT(rq, bool(1));
                [flatten]
                if (_610 == false)
                {
                    _1040 = (-1) - _1040;
                }
                float2 _620 = rayQueryGetIntersectionBarycentricsEXT(rq, bool(1));
                _1042 = _620.x;
                _1043 = _620.y;
                float _627 = rayQueryGetIntersectionTEXT(rq, bool(1));
                _1041 = _627;
            }
            if (_1038 == 0)
            {
                break;
            }
            bool _639 = _1040 < 0;
            if (_639)
            {
                _642 = (-1) - _1040;
            }
            else
            {
                _642 = _1040;
            }
            uint _653 = uint(_642);
            bool _655 = !_639;
            bool _669;
            if (_655)
            {
                _669 = ((_661.Load(_653 * 4 + 0) >> 16u) & 32768u) != 0u;
            }
            else
            {
                _669 = _655;
            }
            bool _682;
            if (!_669)
            {
                bool _681;
                if (_639)
                {
                    _681 = (_661.Load(_653 * 4 + 0) & 32768u) != 0u;
                }
                else
                {
                    _681 = _639;
                }
                _682 = _681;
            }
            else
            {
                _682 = _669;
            }
            if (_682)
            {
                break;
            }
            material_t _705;
            [unroll]
            for (int _35ident = 0; _35ident < 5; _35ident++)
            {
                _705.textures[_35ident] = _697.Load(_35ident * 4 + ((_661.Load(_653 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _705.base_color[_36ident] = asfloat(_697.Load(_36ident * 4 + ((_661.Load(_653 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
            }
            _705.flags = _697.Load(((_661.Load(_653 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
            _705.type = _697.Load(((_661.Load(_653 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
            _705.tangent_rotation_or_strength = asfloat(_697.Load(((_661.Load(_653 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
            _705.roughness_and_anisotropic = _697.Load(((_661.Load(_653 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
            _705.ior = asfloat(_697.Load(((_661.Load(_653 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
            _705.sheen_and_sheen_tint = _697.Load(((_661.Load(_653 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
            _705.tint_and_metallic = _697.Load(((_661.Load(_653 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
            _705.transmission_and_transmission_roughness = _697.Load(((_661.Load(_653 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
            _705.specular_and_specular_tint = _697.Load(((_661.Load(_653 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
            _705.clearcoat_and_clearcoat_roughness = _697.Load(((_661.Load(_653 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
            _705.normal_map_strength_unorm = _697.Load(((_661.Load(_653 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
            material_t _706;
            _706.textures[0] = _705.textures[0];
            _706.textures[1] = _705.textures[1];
            _706.textures[2] = _705.textures[2];
            _706.textures[3] = _705.textures[3];
            _706.textures[4] = _705.textures[4];
            _706.base_color[0] = _705.base_color[0];
            _706.base_color[1] = _705.base_color[1];
            _706.base_color[2] = _705.base_color[2];
            _706.flags = _705.flags;
            _706.type = _705.type;
            _706.tangent_rotation_or_strength = _705.tangent_rotation_or_strength;
            _706.roughness_and_anisotropic = _705.roughness_and_anisotropic;
            _706.ior = _705.ior;
            _706.sheen_and_sheen_tint = _705.sheen_and_sheen_tint;
            _706.tint_and_metallic = _705.tint_and_metallic;
            _706.transmission_and_transmission_roughness = _705.transmission_and_transmission_roughness;
            _706.specular_and_specular_tint = _705.specular_and_specular_tint;
            _706.clearcoat_and_clearcoat_roughness = _705.clearcoat_and_clearcoat_roughness;
            _706.normal_map_strength_unorm = _705.normal_map_strength_unorm;
            uint _1096 = _706.textures[1];
            uint _1097 = _706.textures[3];
            uint _1098 = _706.textures[4];
            float _1111 = _706.base_color[0];
            float _1112 = _706.base_color[1];
            float _1113 = _706.base_color[2];
            uint _1056 = _706.type;
            float _1057 = _706.tangent_rotation_or_strength;
            if (_639)
            {
                material_t _715;
                [unroll]
                for (int _37ident = 0; _37ident < 5; _37ident++)
                {
                    _715.textures[_37ident] = _697.Load(_37ident * 4 + (_661.Load(_653 * 4 + 0) & 16383u) * 76 + 0);
                }
                [unroll]
                for (int _38ident = 0; _38ident < 3; _38ident++)
                {
                    _715.base_color[_38ident] = asfloat(_697.Load(_38ident * 4 + (_661.Load(_653 * 4 + 0) & 16383u) * 76 + 20));
                }
                _715.flags = _697.Load((_661.Load(_653 * 4 + 0) & 16383u) * 76 + 32);
                _715.type = _697.Load((_661.Load(_653 * 4 + 0) & 16383u) * 76 + 36);
                _715.tangent_rotation_or_strength = asfloat(_697.Load((_661.Load(_653 * 4 + 0) & 16383u) * 76 + 40));
                _715.roughness_and_anisotropic = _697.Load((_661.Load(_653 * 4 + 0) & 16383u) * 76 + 44);
                _715.ior = asfloat(_697.Load((_661.Load(_653 * 4 + 0) & 16383u) * 76 + 48));
                _715.sheen_and_sheen_tint = _697.Load((_661.Load(_653 * 4 + 0) & 16383u) * 76 + 52);
                _715.tint_and_metallic = _697.Load((_661.Load(_653 * 4 + 0) & 16383u) * 76 + 56);
                _715.transmission_and_transmission_roughness = _697.Load((_661.Load(_653 * 4 + 0) & 16383u) * 76 + 60);
                _715.specular_and_specular_tint = _697.Load((_661.Load(_653 * 4 + 0) & 16383u) * 76 + 64);
                _715.clearcoat_and_clearcoat_roughness = _697.Load((_661.Load(_653 * 4 + 0) & 16383u) * 76 + 68);
                _715.normal_map_strength_unorm = _697.Load((_661.Load(_653 * 4 + 0) & 16383u) * 76 + 72);
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
                _1096 = _716.textures[1];
                _1097 = _716.textures[3];
                _1098 = _716.textures[4];
                _1111 = _716.base_color[0];
                _1112 = _716.base_color[1];
                _1113 = _716.base_color[2];
                _1056 = _716.type;
                _1057 = _716.tangent_rotation_or_strength;
            }
            uint _737 = _653 * 3u;
            vertex_t _743;
            [unroll]
            for (int _39ident = 0; _39ident < 3; _39ident++)
            {
                _743.p[_39ident] = asfloat(_731.Load(_39ident * 4 + _735.Load(_737 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _743.n[_40ident] = asfloat(_731.Load(_40ident * 4 + _735.Load(_737 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 3; _41ident++)
            {
                _743.b[_41ident] = asfloat(_731.Load(_41ident * 4 + _735.Load(_737 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _42ident = 0; _42ident < 2; _42ident++)
            {
                [unroll]
                for (int _43ident = 0; _43ident < 2; _43ident++)
                {
                    _743.t[_42ident][_43ident] = asfloat(_731.Load(_43ident * 4 + _42ident * 8 + _735.Load(_737 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _744;
            _744.p[0] = _743.p[0];
            _744.p[1] = _743.p[1];
            _744.p[2] = _743.p[2];
            _744.n[0] = _743.n[0];
            _744.n[1] = _743.n[1];
            _744.n[2] = _743.n[2];
            _744.b[0] = _743.b[0];
            _744.b[1] = _743.b[1];
            _744.b[2] = _743.b[2];
            _744.t[0][0] = _743.t[0][0];
            _744.t[0][1] = _743.t[0][1];
            _744.t[1][0] = _743.t[1][0];
            _744.t[1][1] = _743.t[1][1];
            vertex_t _752;
            [unroll]
            for (int _44ident = 0; _44ident < 3; _44ident++)
            {
                _752.p[_44ident] = asfloat(_731.Load(_44ident * 4 + _735.Load((_737 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _45ident = 0; _45ident < 3; _45ident++)
            {
                _752.n[_45ident] = asfloat(_731.Load(_45ident * 4 + _735.Load((_737 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _46ident = 0; _46ident < 3; _46ident++)
            {
                _752.b[_46ident] = asfloat(_731.Load(_46ident * 4 + _735.Load((_737 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _47ident = 0; _47ident < 2; _47ident++)
            {
                [unroll]
                for (int _48ident = 0; _48ident < 2; _48ident++)
                {
                    _752.t[_47ident][_48ident] = asfloat(_731.Load(_48ident * 4 + _47ident * 8 + _735.Load((_737 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _753;
            _753.p[0] = _752.p[0];
            _753.p[1] = _752.p[1];
            _753.p[2] = _752.p[2];
            _753.n[0] = _752.n[0];
            _753.n[1] = _752.n[1];
            _753.n[2] = _752.n[2];
            _753.b[0] = _752.b[0];
            _753.b[1] = _752.b[1];
            _753.b[2] = _752.b[2];
            _753.t[0][0] = _752.t[0][0];
            _753.t[0][1] = _752.t[0][1];
            _753.t[1][0] = _752.t[1][0];
            _753.t[1][1] = _752.t[1][1];
            vertex_t _761;
            [unroll]
            for (int _49ident = 0; _49ident < 3; _49ident++)
            {
                _761.p[_49ident] = asfloat(_731.Load(_49ident * 4 + _735.Load((_737 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _50ident = 0; _50ident < 3; _50ident++)
            {
                _761.n[_50ident] = asfloat(_731.Load(_50ident * 4 + _735.Load((_737 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _51ident = 0; _51ident < 3; _51ident++)
            {
                _761.b[_51ident] = asfloat(_731.Load(_51ident * 4 + _735.Load((_737 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _52ident = 0; _52ident < 2; _52ident++)
            {
                [unroll]
                for (int _53ident = 0; _53ident < 2; _53ident++)
                {
                    _761.t[_52ident][_53ident] = asfloat(_731.Load(_53ident * 4 + _52ident * 8 + _735.Load((_737 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _762;
            _762.p[0] = _761.p[0];
            _762.p[1] = _761.p[1];
            _762.p[2] = _761.p[2];
            _762.n[0] = _761.n[0];
            _762.n[1] = _761.n[1];
            _762.n[2] = _761.n[2];
            _762.b[0] = _761.b[0];
            _762.b[1] = _761.b[1];
            _762.b[2] = _761.b[2];
            _762.t[0][0] = _761.t[0][0];
            _762.t[0][1] = _761.t[0][1];
            _762.t[1][0] = _761.t[1][0];
            _762.t[1][1] = _761.t[1][1];
            float2 _795 = ((float2(_744.t[0][0], _744.t[0][1]) * ((1.0f - _1042) - _1043)) + (float2(_753.t[0][0], _753.t[0][1]) * _1042)) + (float2(_762.t[0][0], _762.t[0][1]) * _1043);
            float trans_r = frac(asfloat(_800.Load(rand_index * 4 + 0)) + _549);
            while (_1056 == 4u)
            {
                float mix_val = _1057;
                if (_1096 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_1096, _795, 0).x;
                }
                if (trans_r > mix_val)
                {
                    material_t _840;
                    [unroll]
                    for (int _54ident = 0; _54ident < 5; _54ident++)
                    {
                        _840.textures[_54ident] = _697.Load(_54ident * 4 + _1097 * 76 + 0);
                    }
                    [unroll]
                    for (int _55ident = 0; _55ident < 3; _55ident++)
                    {
                        _840.base_color[_55ident] = asfloat(_697.Load(_55ident * 4 + _1097 * 76 + 20));
                    }
                    _840.flags = _697.Load(_1097 * 76 + 32);
                    _840.type = _697.Load(_1097 * 76 + 36);
                    _840.tangent_rotation_or_strength = asfloat(_697.Load(_1097 * 76 + 40));
                    _840.roughness_and_anisotropic = _697.Load(_1097 * 76 + 44);
                    _840.ior = asfloat(_697.Load(_1097 * 76 + 48));
                    _840.sheen_and_sheen_tint = _697.Load(_1097 * 76 + 52);
                    _840.tint_and_metallic = _697.Load(_1097 * 76 + 56);
                    _840.transmission_and_transmission_roughness = _697.Load(_1097 * 76 + 60);
                    _840.specular_and_specular_tint = _697.Load(_1097 * 76 + 64);
                    _840.clearcoat_and_clearcoat_roughness = _697.Load(_1097 * 76 + 68);
                    _840.normal_map_strength_unorm = _697.Load(_1097 * 76 + 72);
                    material_t _841;
                    _841.textures[0] = _840.textures[0];
                    _841.textures[1] = _840.textures[1];
                    _841.textures[2] = _840.textures[2];
                    _841.textures[3] = _840.textures[3];
                    _841.textures[4] = _840.textures[4];
                    _841.base_color[0] = _840.base_color[0];
                    _841.base_color[1] = _840.base_color[1];
                    _841.base_color[2] = _840.base_color[2];
                    _841.flags = _840.flags;
                    _841.type = _840.type;
                    _841.tangent_rotation_or_strength = _840.tangent_rotation_or_strength;
                    _841.roughness_and_anisotropic = _840.roughness_and_anisotropic;
                    _841.ior = _840.ior;
                    _841.sheen_and_sheen_tint = _840.sheen_and_sheen_tint;
                    _841.tint_and_metallic = _840.tint_and_metallic;
                    _841.transmission_and_transmission_roughness = _840.transmission_and_transmission_roughness;
                    _841.specular_and_specular_tint = _840.specular_and_specular_tint;
                    _841.clearcoat_and_clearcoat_roughness = _840.clearcoat_and_clearcoat_roughness;
                    _841.normal_map_strength_unorm = _840.normal_map_strength_unorm;
                    _1096 = _841.textures[1];
                    _1097 = _841.textures[3];
                    _1098 = _841.textures[4];
                    _1111 = _841.base_color[0];
                    _1112 = _841.base_color[1];
                    _1113 = _841.base_color[2];
                    _1056 = _841.type;
                    _1057 = _841.tangent_rotation_or_strength;
                    trans_r = (trans_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _852;
                    [unroll]
                    for (int _56ident = 0; _56ident < 5; _56ident++)
                    {
                        _852.textures[_56ident] = _697.Load(_56ident * 4 + _1098 * 76 + 0);
                    }
                    [unroll]
                    for (int _57ident = 0; _57ident < 3; _57ident++)
                    {
                        _852.base_color[_57ident] = asfloat(_697.Load(_57ident * 4 + _1098 * 76 + 20));
                    }
                    _852.flags = _697.Load(_1098 * 76 + 32);
                    _852.type = _697.Load(_1098 * 76 + 36);
                    _852.tangent_rotation_or_strength = asfloat(_697.Load(_1098 * 76 + 40));
                    _852.roughness_and_anisotropic = _697.Load(_1098 * 76 + 44);
                    _852.ior = asfloat(_697.Load(_1098 * 76 + 48));
                    _852.sheen_and_sheen_tint = _697.Load(_1098 * 76 + 52);
                    _852.tint_and_metallic = _697.Load(_1098 * 76 + 56);
                    _852.transmission_and_transmission_roughness = _697.Load(_1098 * 76 + 60);
                    _852.specular_and_specular_tint = _697.Load(_1098 * 76 + 64);
                    _852.clearcoat_and_clearcoat_roughness = _697.Load(_1098 * 76 + 68);
                    _852.normal_map_strength_unorm = _697.Load(_1098 * 76 + 72);
                    material_t _853;
                    _853.textures[0] = _852.textures[0];
                    _853.textures[1] = _852.textures[1];
                    _853.textures[2] = _852.textures[2];
                    _853.textures[3] = _852.textures[3];
                    _853.textures[4] = _852.textures[4];
                    _853.base_color[0] = _852.base_color[0];
                    _853.base_color[1] = _852.base_color[1];
                    _853.base_color[2] = _852.base_color[2];
                    _853.flags = _852.flags;
                    _853.type = _852.type;
                    _853.tangent_rotation_or_strength = _852.tangent_rotation_or_strength;
                    _853.roughness_and_anisotropic = _852.roughness_and_anisotropic;
                    _853.ior = _852.ior;
                    _853.sheen_and_sheen_tint = _852.sheen_and_sheen_tint;
                    _853.tint_and_metallic = _852.tint_and_metallic;
                    _853.transmission_and_transmission_roughness = _852.transmission_and_transmission_roughness;
                    _853.specular_and_specular_tint = _852.specular_and_specular_tint;
                    _853.clearcoat_and_clearcoat_roughness = _852.clearcoat_and_clearcoat_roughness;
                    _853.normal_map_strength_unorm = _852.normal_map_strength_unorm;
                    _1096 = _853.textures[1];
                    _1097 = _853.textures[3];
                    _1098 = _853.textures[4];
                    _1111 = _853.base_color[0];
                    _1112 = _853.base_color[1];
                    _1113 = _853.base_color[2];
                    _1056 = _853.type;
                    _1057 = _853.tangent_rotation_or_strength;
                    trans_r /= mix_val;
                }
            }
            if (_1056 != 5u)
            {
                break;
            }
            float _882 = max(asfloat(_490.Load(_467 * 72 + 28)), max(asfloat(_490.Load(_467 * 72 + 32)), asfloat(_490.Load(_467 * 72 + 36))));
            if ((int(_490.Load(_467 * 72 + 68)) >> 24) > _555_g_params.min_transp_depth)
            {
                _894 = max(0.0500000007450580596923828125f, 1.0f - _882);
            }
            else
            {
                _894 = 0.0f;
            }
            bool _908 = (frac(asfloat(_800.Load((rand_index + 6) * 4 + 0)) + _549) < _894) || (_882 == 0.0f);
            bool _920;
            if (!_908)
            {
                _920 = ((int(_490.Load(_467 * 72 + 68)) >> 24) + 1) >= _555_g_params.max_transp_depth;
            }
            else
            {
                _920 = _908;
            }
            if (_920)
            {
                _490.Store(_467 * 72 + 36, asuint(0.0f));
                _490.Store(_467 * 72 + 32, asuint(0.0f));
                _490.Store(_467 * 72 + 28, asuint(0.0f));
                break;
            }
            float _934 = 1.0f - _894;
            _490.Store(_467 * 72 + 28, asuint(asfloat(_490.Load(_467 * 72 + 28)) * (_1111 / _934)));
            _490.Store(_467 * 72 + 32, asuint(asfloat(_490.Load(_467 * 72 + 32)) * (_1112 / _934)));
            _490.Store(_467 * 72 + 36, asuint(asfloat(_490.Load(_467 * 72 + 36)) * (_1113 / _934)));
            ro += (_526 * (_1041 + 9.9999997473787516355514526367188e-06f));
            _1038 = 0;
            _1041 = _578 - _1041;
            _490.Store(_467 * 72 + 68, uint(int(_490.Load(_467 * 72 + 68)) + 16777216));
            rand_index += 7;
            continue;
        }
        float _998 = _1041;
        float _999 = _998 + length(float3(asfloat(_490.Load(_467 * 72 + 0)), asfloat(_490.Load(_467 * 72 + 4)), asfloat(_490.Load(_467 * 72 + 8))) - ro);
        _1041 = _999;
        hit_data_t _1050 = { _1038, _1039, _1040, _999, _1042, _1043 };
        hit_data_t _1010;
        _1010.mask = _1050.mask;
        _1010.obj_index = _1050.obj_index;
        _1010.prim_index = _1050.prim_index;
        _1010.t = _1050.t;
        _1010.u = _1050.u;
        _1010.v = _1050.v;
        _1005.Store(_467 * 24 + 0, uint(_1010.mask));
        _1005.Store(_467 * 24 + 4, uint(_1010.obj_index));
        _1005.Store(_467 * 24 + 8, uint(_1010.prim_index));
        _1005.Store(_467 * 24 + 12, asuint(_1010.t));
        _1005.Store(_467 * 24 + 16, asuint(_1010.u));
        _1005.Store(_467 * 24 + 20, asuint(_1010.v));
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
