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

ByteAddressBuffer _369 : register(t20, space0);
RWByteAddressBuffer _507 : register(u12, space0);
ByteAddressBuffer _663 : register(t5, space0);
ByteAddressBuffer _700 : register(t6, space0);
ByteAddressBuffer _734 : register(t1, space0);
ByteAddressBuffer _738 : register(t2, space0);
ByteAddressBuffer _803 : register(t15, space0);
RWByteAddressBuffer _1008 : register(u0, space0);
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
        float3 _1177 = inv_v;
        _1177.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _1177;
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
            float3 _1175 = inv_v;
            _1175.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _1175;
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
        float3 _1181 = inv_v;
        _1181.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _1181;
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
            float3 _1179 = inv_v;
            _1179.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _1179;
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
        float3 _1185 = inv_v;
        _1185.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _1185;
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
            float3 _1183 = inv_v;
            _1183.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _1183;
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
    uint _1157[14] = t.pos;
    uint _1160[14] = t.pos;
    uint _327 = t.size & 16383u;
    uint _330 = t.size >> uint(16);
    uint _331 = _330 & 16383u;
    float2 size = float2(float(_327), float(_331));
    if ((_330 & 32768u) != 0u)
    {
        size = float2(float(_327 >> uint(mip_level)), float(_331 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_1157[mip_level] & 65535u), float((_1160[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
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
    uint _1166[4] = _373.page;
    uint _401 = _373.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_401)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_401)], float3(TransformUV(uvs, _373, lod) * 0.000118371215648949146270751953125f.xx, float((_1166[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
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
        float4 _1201 = res;
        _1201.x = _441.x;
        float4 _1203 = _1201;
        _1203.y = _441.y;
        float4 _1205 = _1203;
        _1205.z = _441.z;
        res = _1205;
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
        bool _469 = gl_GlobalInvocationID.x >= _465_g_params.img_size.x;
        bool _478;
        if (!_469)
        {
            _478 = gl_GlobalInvocationID.y >= _465_g_params.img_size.y;
        }
        else
        {
            _478 = _469;
        }
        if (_478)
        {
            break;
        }
        int _497 = (int(gl_GlobalInvocationID.y) * int(_465_g_params.img_size.x)) + int(gl_GlobalInvocationID.x);
        float3 ro = float3(asfloat(_507.Load(_497 * 72 + 0)), asfloat(_507.Load(_497 * 72 + 4)), asfloat(_507.Load(_497 * 72 + 8)));
        float _522 = asfloat(_507.Load(_497 * 72 + 12));
        float _525 = asfloat(_507.Load(_497 * 72 + 16));
        float _528 = asfloat(_507.Load(_497 * 72 + 20));
        float3 _529 = float3(_522, _525, _528);
        float3 param = _529;
        int _1041 = 0;
        int _1043 = 0;
        int _1042 = 0;
        float _1044 = _465_g_params.cam_clip_end;
        float _1046 = 0.0f;
        float _1045 = 0.0f;
        uint param_1 = uint(hash(int(_507.Load(_497 * 72 + 64))));
        float _556 = construct_float(param_1);
        ray_data_t _564;
        [unroll]
        for (int _31ident = 0; _31ident < 3; _31ident++)
        {
            _564.o[_31ident] = asfloat(_507.Load(_31ident * 4 + _497 * 72 + 0));
        }
        [unroll]
        for (int _32ident = 0; _32ident < 3; _32ident++)
        {
            _564.d[_32ident] = asfloat(_507.Load(_32ident * 4 + _497 * 72 + 12));
        }
        _564.pdf = asfloat(_507.Load(_497 * 72 + 24));
        [unroll]
        for (int _33ident = 0; _33ident < 3; _33ident++)
        {
            _564.c[_33ident] = asfloat(_507.Load(_33ident * 4 + _497 * 72 + 28));
        }
        [unroll]
        for (int _34ident = 0; _34ident < 4; _34ident++)
        {
            _564.ior[_34ident] = asfloat(_507.Load(_34ident * 4 + _497 * 72 + 40));
        }
        _564.cone_width = asfloat(_507.Load(_497 * 72 + 56));
        _564.cone_spread = asfloat(_507.Load(_497 * 72 + 60));
        _564.xy = int(_507.Load(_497 * 72 + 64));
        _564.depth = int(_507.Load(_497 * 72 + 68));
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
        int rand_index = _465_g_params.hi + (total_depth(_567) * 7);
        int _644;
        float _897;
        float _580;
        for (;;)
        {
            _580 = _1044;
            rayQueryInitializeEXT(rq, g_tlas, 0u, 255u, ro, 0.0f, _529, _580);
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
                _1041 = -1;
                int _606 = rayQueryGetIntersectionInstanceIdEXT(rq, bool(1));
                _1042 = _606;
                int _609 = rayQueryGetIntersectionPrimitiveIndexEXT(rq, bool(1));
                _1043 = _603 + _609;
                bool _612 = rayQueryGetIntersectionFrontFaceEXT(rq, bool(1));
                [flatten]
                if (_612 == false)
                {
                    _1043 = (-1) - _1043;
                }
                float2 _622 = rayQueryGetIntersectionBarycentricsEXT(rq, bool(1));
                _1045 = _622.x;
                _1046 = _622.y;
                float _629 = rayQueryGetIntersectionTEXT(rq, bool(1));
                _1044 = _629;
            }
            if (_1041 == 0)
            {
                break;
            }
            bool _641 = _1043 < 0;
            if (_641)
            {
                _644 = (-1) - _1043;
            }
            else
            {
                _644 = _1043;
            }
            uint _655 = uint(_644);
            bool _657 = !_641;
            bool _672;
            if (_657)
            {
                _672 = ((_663.Load(_655 * 4 + 0) >> 16u) & 32768u) != 0u;
            }
            else
            {
                _672 = _657;
            }
            bool _685;
            if (!_672)
            {
                bool _684;
                if (_641)
                {
                    _684 = (_663.Load(_655 * 4 + 0) & 32768u) != 0u;
                }
                else
                {
                    _684 = _641;
                }
                _685 = _684;
            }
            else
            {
                _685 = _672;
            }
            if (_685)
            {
                break;
            }
            material_t _708;
            [unroll]
            for (int _35ident = 0; _35ident < 5; _35ident++)
            {
                _708.textures[_35ident] = _700.Load(_35ident * 4 + ((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _708.base_color[_36ident] = asfloat(_700.Load(_36ident * 4 + ((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
            }
            _708.flags = _700.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
            _708.type = _700.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
            _708.tangent_rotation_or_strength = asfloat(_700.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
            _708.roughness_and_anisotropic = _700.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
            _708.ior = asfloat(_700.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
            _708.sheen_and_sheen_tint = _700.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
            _708.tint_and_metallic = _700.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
            _708.transmission_and_transmission_roughness = _700.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
            _708.specular_and_specular_tint = _700.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
            _708.clearcoat_and_clearcoat_roughness = _700.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
            _708.normal_map_strength_unorm = _700.Load(((_663.Load(_655 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
            material_t _709;
            _709.textures[0] = _708.textures[0];
            _709.textures[1] = _708.textures[1];
            _709.textures[2] = _708.textures[2];
            _709.textures[3] = _708.textures[3];
            _709.textures[4] = _708.textures[4];
            _709.base_color[0] = _708.base_color[0];
            _709.base_color[1] = _708.base_color[1];
            _709.base_color[2] = _708.base_color[2];
            _709.flags = _708.flags;
            _709.type = _708.type;
            _709.tangent_rotation_or_strength = _708.tangent_rotation_or_strength;
            _709.roughness_and_anisotropic = _708.roughness_and_anisotropic;
            _709.ior = _708.ior;
            _709.sheen_and_sheen_tint = _708.sheen_and_sheen_tint;
            _709.tint_and_metallic = _708.tint_and_metallic;
            _709.transmission_and_transmission_roughness = _708.transmission_and_transmission_roughness;
            _709.specular_and_specular_tint = _708.specular_and_specular_tint;
            _709.clearcoat_and_clearcoat_roughness = _708.clearcoat_and_clearcoat_roughness;
            _709.normal_map_strength_unorm = _708.normal_map_strength_unorm;
            uint _1099 = _709.textures[1];
            uint _1100 = _709.textures[3];
            uint _1101 = _709.textures[4];
            float _1114 = _709.base_color[0];
            float _1115 = _709.base_color[1];
            float _1116 = _709.base_color[2];
            uint _1059 = _709.type;
            float _1060 = _709.tangent_rotation_or_strength;
            if (_641)
            {
                material_t _718;
                [unroll]
                for (int _37ident = 0; _37ident < 5; _37ident++)
                {
                    _718.textures[_37ident] = _700.Load(_37ident * 4 + (_663.Load(_655 * 4 + 0) & 16383u) * 76 + 0);
                }
                [unroll]
                for (int _38ident = 0; _38ident < 3; _38ident++)
                {
                    _718.base_color[_38ident] = asfloat(_700.Load(_38ident * 4 + (_663.Load(_655 * 4 + 0) & 16383u) * 76 + 20));
                }
                _718.flags = _700.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 32);
                _718.type = _700.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 36);
                _718.tangent_rotation_or_strength = asfloat(_700.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 40));
                _718.roughness_and_anisotropic = _700.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 44);
                _718.ior = asfloat(_700.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 48));
                _718.sheen_and_sheen_tint = _700.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 52);
                _718.tint_and_metallic = _700.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 56);
                _718.transmission_and_transmission_roughness = _700.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 60);
                _718.specular_and_specular_tint = _700.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 64);
                _718.clearcoat_and_clearcoat_roughness = _700.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 68);
                _718.normal_map_strength_unorm = _700.Load((_663.Load(_655 * 4 + 0) & 16383u) * 76 + 72);
                material_t _719;
                _719.textures[0] = _718.textures[0];
                _719.textures[1] = _718.textures[1];
                _719.textures[2] = _718.textures[2];
                _719.textures[3] = _718.textures[3];
                _719.textures[4] = _718.textures[4];
                _719.base_color[0] = _718.base_color[0];
                _719.base_color[1] = _718.base_color[1];
                _719.base_color[2] = _718.base_color[2];
                _719.flags = _718.flags;
                _719.type = _718.type;
                _719.tangent_rotation_or_strength = _718.tangent_rotation_or_strength;
                _719.roughness_and_anisotropic = _718.roughness_and_anisotropic;
                _719.ior = _718.ior;
                _719.sheen_and_sheen_tint = _718.sheen_and_sheen_tint;
                _719.tint_and_metallic = _718.tint_and_metallic;
                _719.transmission_and_transmission_roughness = _718.transmission_and_transmission_roughness;
                _719.specular_and_specular_tint = _718.specular_and_specular_tint;
                _719.clearcoat_and_clearcoat_roughness = _718.clearcoat_and_clearcoat_roughness;
                _719.normal_map_strength_unorm = _718.normal_map_strength_unorm;
                _1099 = _719.textures[1];
                _1100 = _719.textures[3];
                _1101 = _719.textures[4];
                _1114 = _719.base_color[0];
                _1115 = _719.base_color[1];
                _1116 = _719.base_color[2];
                _1059 = _719.type;
                _1060 = _719.tangent_rotation_or_strength;
            }
            uint _740 = _655 * 3u;
            vertex_t _746;
            [unroll]
            for (int _39ident = 0; _39ident < 3; _39ident++)
            {
                _746.p[_39ident] = asfloat(_734.Load(_39ident * 4 + _738.Load(_740 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _746.n[_40ident] = asfloat(_734.Load(_40ident * 4 + _738.Load(_740 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 3; _41ident++)
            {
                _746.b[_41ident] = asfloat(_734.Load(_41ident * 4 + _738.Load(_740 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _42ident = 0; _42ident < 2; _42ident++)
            {
                [unroll]
                for (int _43ident = 0; _43ident < 2; _43ident++)
                {
                    _746.t[_42ident][_43ident] = asfloat(_734.Load(_43ident * 4 + _42ident * 8 + _738.Load(_740 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _747;
            _747.p[0] = _746.p[0];
            _747.p[1] = _746.p[1];
            _747.p[2] = _746.p[2];
            _747.n[0] = _746.n[0];
            _747.n[1] = _746.n[1];
            _747.n[2] = _746.n[2];
            _747.b[0] = _746.b[0];
            _747.b[1] = _746.b[1];
            _747.b[2] = _746.b[2];
            _747.t[0][0] = _746.t[0][0];
            _747.t[0][1] = _746.t[0][1];
            _747.t[1][0] = _746.t[1][0];
            _747.t[1][1] = _746.t[1][1];
            vertex_t _755;
            [unroll]
            for (int _44ident = 0; _44ident < 3; _44ident++)
            {
                _755.p[_44ident] = asfloat(_734.Load(_44ident * 4 + _738.Load((_740 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _45ident = 0; _45ident < 3; _45ident++)
            {
                _755.n[_45ident] = asfloat(_734.Load(_45ident * 4 + _738.Load((_740 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _46ident = 0; _46ident < 3; _46ident++)
            {
                _755.b[_46ident] = asfloat(_734.Load(_46ident * 4 + _738.Load((_740 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _47ident = 0; _47ident < 2; _47ident++)
            {
                [unroll]
                for (int _48ident = 0; _48ident < 2; _48ident++)
                {
                    _755.t[_47ident][_48ident] = asfloat(_734.Load(_48ident * 4 + _47ident * 8 + _738.Load((_740 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _756;
            _756.p[0] = _755.p[0];
            _756.p[1] = _755.p[1];
            _756.p[2] = _755.p[2];
            _756.n[0] = _755.n[0];
            _756.n[1] = _755.n[1];
            _756.n[2] = _755.n[2];
            _756.b[0] = _755.b[0];
            _756.b[1] = _755.b[1];
            _756.b[2] = _755.b[2];
            _756.t[0][0] = _755.t[0][0];
            _756.t[0][1] = _755.t[0][1];
            _756.t[1][0] = _755.t[1][0];
            _756.t[1][1] = _755.t[1][1];
            vertex_t _764;
            [unroll]
            for (int _49ident = 0; _49ident < 3; _49ident++)
            {
                _764.p[_49ident] = asfloat(_734.Load(_49ident * 4 + _738.Load((_740 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _50ident = 0; _50ident < 3; _50ident++)
            {
                _764.n[_50ident] = asfloat(_734.Load(_50ident * 4 + _738.Load((_740 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _51ident = 0; _51ident < 3; _51ident++)
            {
                _764.b[_51ident] = asfloat(_734.Load(_51ident * 4 + _738.Load((_740 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _52ident = 0; _52ident < 2; _52ident++)
            {
                [unroll]
                for (int _53ident = 0; _53ident < 2; _53ident++)
                {
                    _764.t[_52ident][_53ident] = asfloat(_734.Load(_53ident * 4 + _52ident * 8 + _738.Load((_740 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _765;
            _765.p[0] = _764.p[0];
            _765.p[1] = _764.p[1];
            _765.p[2] = _764.p[2];
            _765.n[0] = _764.n[0];
            _765.n[1] = _764.n[1];
            _765.n[2] = _764.n[2];
            _765.b[0] = _764.b[0];
            _765.b[1] = _764.b[1];
            _765.b[2] = _764.b[2];
            _765.t[0][0] = _764.t[0][0];
            _765.t[0][1] = _764.t[0][1];
            _765.t[1][0] = _764.t[1][0];
            _765.t[1][1] = _764.t[1][1];
            float2 _798 = ((float2(_747.t[0][0], _747.t[0][1]) * ((1.0f - _1045) - _1046)) + (float2(_756.t[0][0], _756.t[0][1]) * _1045)) + (float2(_765.t[0][0], _765.t[0][1]) * _1046);
            float trans_r = frac(asfloat(_803.Load(rand_index * 4 + 0)) + _556);
            while (_1059 == 4u)
            {
                float mix_val = _1060;
                if (_1099 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_1099, _798, 0).x;
                }
                if (trans_r > mix_val)
                {
                    material_t _843;
                    [unroll]
                    for (int _54ident = 0; _54ident < 5; _54ident++)
                    {
                        _843.textures[_54ident] = _700.Load(_54ident * 4 + _1100 * 76 + 0);
                    }
                    [unroll]
                    for (int _55ident = 0; _55ident < 3; _55ident++)
                    {
                        _843.base_color[_55ident] = asfloat(_700.Load(_55ident * 4 + _1100 * 76 + 20));
                    }
                    _843.flags = _700.Load(_1100 * 76 + 32);
                    _843.type = _700.Load(_1100 * 76 + 36);
                    _843.tangent_rotation_or_strength = asfloat(_700.Load(_1100 * 76 + 40));
                    _843.roughness_and_anisotropic = _700.Load(_1100 * 76 + 44);
                    _843.ior = asfloat(_700.Load(_1100 * 76 + 48));
                    _843.sheen_and_sheen_tint = _700.Load(_1100 * 76 + 52);
                    _843.tint_and_metallic = _700.Load(_1100 * 76 + 56);
                    _843.transmission_and_transmission_roughness = _700.Load(_1100 * 76 + 60);
                    _843.specular_and_specular_tint = _700.Load(_1100 * 76 + 64);
                    _843.clearcoat_and_clearcoat_roughness = _700.Load(_1100 * 76 + 68);
                    _843.normal_map_strength_unorm = _700.Load(_1100 * 76 + 72);
                    material_t _844;
                    _844.textures[0] = _843.textures[0];
                    _844.textures[1] = _843.textures[1];
                    _844.textures[2] = _843.textures[2];
                    _844.textures[3] = _843.textures[3];
                    _844.textures[4] = _843.textures[4];
                    _844.base_color[0] = _843.base_color[0];
                    _844.base_color[1] = _843.base_color[1];
                    _844.base_color[2] = _843.base_color[2];
                    _844.flags = _843.flags;
                    _844.type = _843.type;
                    _844.tangent_rotation_or_strength = _843.tangent_rotation_or_strength;
                    _844.roughness_and_anisotropic = _843.roughness_and_anisotropic;
                    _844.ior = _843.ior;
                    _844.sheen_and_sheen_tint = _843.sheen_and_sheen_tint;
                    _844.tint_and_metallic = _843.tint_and_metallic;
                    _844.transmission_and_transmission_roughness = _843.transmission_and_transmission_roughness;
                    _844.specular_and_specular_tint = _843.specular_and_specular_tint;
                    _844.clearcoat_and_clearcoat_roughness = _843.clearcoat_and_clearcoat_roughness;
                    _844.normal_map_strength_unorm = _843.normal_map_strength_unorm;
                    _1099 = _844.textures[1];
                    _1100 = _844.textures[3];
                    _1101 = _844.textures[4];
                    _1114 = _844.base_color[0];
                    _1115 = _844.base_color[1];
                    _1116 = _844.base_color[2];
                    _1059 = _844.type;
                    _1060 = _844.tangent_rotation_or_strength;
                    trans_r = (trans_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _855;
                    [unroll]
                    for (int _56ident = 0; _56ident < 5; _56ident++)
                    {
                        _855.textures[_56ident] = _700.Load(_56ident * 4 + _1101 * 76 + 0);
                    }
                    [unroll]
                    for (int _57ident = 0; _57ident < 3; _57ident++)
                    {
                        _855.base_color[_57ident] = asfloat(_700.Load(_57ident * 4 + _1101 * 76 + 20));
                    }
                    _855.flags = _700.Load(_1101 * 76 + 32);
                    _855.type = _700.Load(_1101 * 76 + 36);
                    _855.tangent_rotation_or_strength = asfloat(_700.Load(_1101 * 76 + 40));
                    _855.roughness_and_anisotropic = _700.Load(_1101 * 76 + 44);
                    _855.ior = asfloat(_700.Load(_1101 * 76 + 48));
                    _855.sheen_and_sheen_tint = _700.Load(_1101 * 76 + 52);
                    _855.tint_and_metallic = _700.Load(_1101 * 76 + 56);
                    _855.transmission_and_transmission_roughness = _700.Load(_1101 * 76 + 60);
                    _855.specular_and_specular_tint = _700.Load(_1101 * 76 + 64);
                    _855.clearcoat_and_clearcoat_roughness = _700.Load(_1101 * 76 + 68);
                    _855.normal_map_strength_unorm = _700.Load(_1101 * 76 + 72);
                    material_t _856;
                    _856.textures[0] = _855.textures[0];
                    _856.textures[1] = _855.textures[1];
                    _856.textures[2] = _855.textures[2];
                    _856.textures[3] = _855.textures[3];
                    _856.textures[4] = _855.textures[4];
                    _856.base_color[0] = _855.base_color[0];
                    _856.base_color[1] = _855.base_color[1];
                    _856.base_color[2] = _855.base_color[2];
                    _856.flags = _855.flags;
                    _856.type = _855.type;
                    _856.tangent_rotation_or_strength = _855.tangent_rotation_or_strength;
                    _856.roughness_and_anisotropic = _855.roughness_and_anisotropic;
                    _856.ior = _855.ior;
                    _856.sheen_and_sheen_tint = _855.sheen_and_sheen_tint;
                    _856.tint_and_metallic = _855.tint_and_metallic;
                    _856.transmission_and_transmission_roughness = _855.transmission_and_transmission_roughness;
                    _856.specular_and_specular_tint = _855.specular_and_specular_tint;
                    _856.clearcoat_and_clearcoat_roughness = _855.clearcoat_and_clearcoat_roughness;
                    _856.normal_map_strength_unorm = _855.normal_map_strength_unorm;
                    _1099 = _856.textures[1];
                    _1100 = _856.textures[3];
                    _1101 = _856.textures[4];
                    _1114 = _856.base_color[0];
                    _1115 = _856.base_color[1];
                    _1116 = _856.base_color[2];
                    _1059 = _856.type;
                    _1060 = _856.tangent_rotation_or_strength;
                    trans_r /= mix_val;
                }
            }
            if (_1059 != 5u)
            {
                break;
            }
            float _885 = max(asfloat(_507.Load(_497 * 72 + 28)), max(asfloat(_507.Load(_497 * 72 + 32)), asfloat(_507.Load(_497 * 72 + 36))));
            if ((int(_507.Load(_497 * 72 + 68)) >> 24) > _465_g_params.min_transp_depth)
            {
                _897 = max(0.0500000007450580596923828125f, 1.0f - _885);
            }
            else
            {
                _897 = 0.0f;
            }
            bool _911 = (frac(asfloat(_803.Load((rand_index + 6) * 4 + 0)) + _556) < _897) || (_885 == 0.0f);
            bool _923;
            if (!_911)
            {
                _923 = ((int(_507.Load(_497 * 72 + 68)) >> 24) + 1) >= _465_g_params.max_transp_depth;
            }
            else
            {
                _923 = _911;
            }
            if (_923)
            {
                _507.Store(_497 * 72 + 36, asuint(0.0f));
                _507.Store(_497 * 72 + 32, asuint(0.0f));
                _507.Store(_497 * 72 + 28, asuint(0.0f));
                break;
            }
            float _937 = 1.0f - _897;
            _507.Store(_497 * 72 + 28, asuint(asfloat(_507.Load(_497 * 72 + 28)) * (_1114 / _937)));
            _507.Store(_497 * 72 + 32, asuint(asfloat(_507.Load(_497 * 72 + 32)) * (_1115 / _937)));
            _507.Store(_497 * 72 + 36, asuint(asfloat(_507.Load(_497 * 72 + 36)) * (_1116 / _937)));
            ro += (_529 * (_1044 + 9.9999997473787516355514526367188e-06f));
            _1041 = 0;
            _1044 = _580 - _1044;
            _507.Store(_497 * 72 + 68, uint(int(_507.Load(_497 * 72 + 68)) + 16777216));
            rand_index += 7;
            continue;
        }
        float _1001 = _1044;
        float _1002 = _1001 + length(float3(asfloat(_507.Load(_497 * 72 + 0)), asfloat(_507.Load(_497 * 72 + 4)), asfloat(_507.Load(_497 * 72 + 8))) - ro);
        _1044 = _1002;
        hit_data_t _1053 = { _1041, _1042, _1043, _1002, _1045, _1046 };
        hit_data_t _1013;
        _1013.mask = _1053.mask;
        _1013.obj_index = _1053.obj_index;
        _1013.prim_index = _1053.prim_index;
        _1013.t = _1053.t;
        _1013.u = _1053.u;
        _1013.v = _1053.v;
        _1008.Store(_497 * 24 + 0, uint(_1013.mask));
        _1008.Store(_497 * 24 + 4, uint(_1013.obj_index));
        _1008.Store(_497 * 24 + 8, uint(_1013.prim_index));
        _1008.Store(_497 * 24 + 12, asuint(_1013.t));
        _1008.Store(_497 * 24 + 16, asuint(_1013.u));
        _1008.Store(_497 * 24 + 20, asuint(_1013.v));
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
