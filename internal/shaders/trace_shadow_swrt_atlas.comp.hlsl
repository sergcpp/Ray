struct atlas_texture_t
{
    uint size;
    uint atlas;
    uint page[4];
    uint pos[14];
};

struct tri_accel_t
{
    float4 n_plane;
    float4 u_plane;
    float4 v_plane;
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

struct bvh_node_t
{
    float4 bbox_min;
    float4 bbox_max;
};

struct mesh_instance_t
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

struct transform_t
{
    row_major float4x4 xform;
    row_major float4x4 inv_xform;
};

struct Params
{
    uint2 img_size;
    uint node_index;
    float random_val;
};

struct material_t
{
    uint textures[5];
    float base_color[3];
    uint flags;
    uint type;
    float tangent_rotation_or_strength;
    uint roughness_and_anisotropic;
    float int_ior;
    float ext_ior;
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

struct shadow_ray_t
{
    float o[3];
    float d[3];
    float dist;
    float c[3];
    int xy;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

ByteAddressBuffer _390 : register(t20, space0);
ByteAddressBuffer _662 : register(t1, space0);
ByteAddressBuffer _882 : register(t5, space0);
ByteAddressBuffer _1022 : register(t2, space0);
ByteAddressBuffer _1031 : register(t3, space0);
ByteAddressBuffer _1202 : register(t7, space0);
ByteAddressBuffer _1206 : register(t8, space0);
ByteAddressBuffer _1227 : register(t6, space0);
ByteAddressBuffer _1271 : register(t9, space0);
ByteAddressBuffer _1452 : register(t4, space0);
ByteAddressBuffer _1529 : register(t10, space0);
ByteAddressBuffer _1533 : register(t11, space0);
ByteAddressBuffer _1890 : register(t13, space0);
ByteAddressBuffer _1907 : register(t12, space0);
cbuffer UniformParams
{
    Params _1377_g_params : packoffset(c0);
};

Texture2DArray<float4> g_atlases[7] : register(t21, space0);
SamplerState _g_atlases_sampler[7] : register(s21, space0);
RWTexture2D<float4> g_out_img : register(u0, space0);

static uint3 gl_WorkGroupID;
static uint gl_LocalInvocationIndex;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint gl_LocalInvocationIndex : SV_GroupIndex;
};

groupshared uint g_stack[64][48];

int hash(int x)
{
    uint _118 = uint(x);
    uint _125 = ((_118 >> uint(16)) ^ _118) * 73244475u;
    uint _130 = ((_125 >> uint(16)) ^ _125) * 73244475u;
    return int((_130 >> uint(16)) ^ _130);
}

float3 safe_invert(float3 v)
{
    float3 inv_v = 1.0f.xxx / v;
    bool _161 = v.x <= 1.0000000116860974230803549289703e-07f;
    bool _168;
    if (_161)
    {
        _168 = v.x >= 0.0f;
    }
    else
    {
        _168 = _161;
    }
    if (_168)
    {
        float3 _2306 = inv_v;
        _2306.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _2306;
    }
    else
    {
        bool _177 = v.x >= (-1.0000000116860974230803549289703e-07f);
        bool _183;
        if (_177)
        {
            _183 = v.x < 0.0f;
        }
        else
        {
            _183 = _177;
        }
        if (_183)
        {
            float3 _2308 = inv_v;
            _2308.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _2308;
        }
    }
    bool _191 = v.y <= 1.0000000116860974230803549289703e-07f;
    bool _197;
    if (_191)
    {
        _197 = v.y >= 0.0f;
    }
    else
    {
        _197 = _191;
    }
    if (_197)
    {
        float3 _2310 = inv_v;
        _2310.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _2310;
    }
    else
    {
        bool _204 = v.y >= (-1.0000000116860974230803549289703e-07f);
        bool _210;
        if (_204)
        {
            _210 = v.y < 0.0f;
        }
        else
        {
            _210 = _204;
        }
        if (_210)
        {
            float3 _2312 = inv_v;
            _2312.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _2312;
        }
    }
    bool _217 = v.z <= 1.0000000116860974230803549289703e-07f;
    bool _223;
    if (_217)
    {
        _223 = v.z >= 0.0f;
    }
    else
    {
        _223 = _217;
    }
    if (_223)
    {
        float3 _2314 = inv_v;
        _2314.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _2314;
    }
    else
    {
        bool _230 = v.z >= (-1.0000000116860974230803549289703e-07f);
        bool _236;
        if (_230)
        {
            _236 = v.z < 0.0f;
        }
        else
        {
            _236 = _230;
        }
        if (_236)
        {
            float3 _2316 = inv_v;
            _2316.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _2316;
        }
    }
    return inv_v;
}

bool _bbox_test_fma(float3 inv_d, float3 neg_inv_d_o, float t, float3 bbox_min, float3 bbox_max)
{
    float _760 = mad(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float _768 = mad(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float _783 = mad(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    float _790 = mad(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    float _807 = mad(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    float _814 = mad(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    float _819 = max(max(min(_760, _768), min(_783, _790)), min(_807, _814));
    float _827 = min(min(max(_760, _768), max(_783, _790)), max(_807, _814)) * 1.0000002384185791015625f;
    return ((_819 <= _827) && (_819 <= t)) && (_827 > 0.0f);
}

void IntersectTri(float3 ro, float3 rd, tri_accel_t tri, uint prim_index, inout hit_data_t inter)
{
    do
    {
        float _535 = dot(rd, tri.n_plane.xyz);
        float _544 = tri.n_plane.w - dot(ro, tri.n_plane.xyz);
        if (sign(_544) != sign(mad(_535, inter.t, -_544)))
        {
            break;
        }
        float3 _565 = (ro * _535) + (rd * _544);
        float _576 = mad(_535, tri.u_plane.w, dot(_565, tri.u_plane.xyz));
        float _581 = _535 - _576;
        if (sign(_576) != sign(_581))
        {
            break;
        }
        float _597 = mad(_535, tri.v_plane.w, dot(_565, tri.v_plane.xyz));
        if (sign(_597) != sign(_581 - _597))
        {
            break;
        }
        float _612 = 1.0f / _535;
        inter.mask = -1;
        int _617;
        if (_535 < 0.0f)
        {
            _617 = int(prim_index);
        }
        else
        {
            _617 = (-1) - int(prim_index);
        }
        inter.prim_index = _617;
        inter.t = _544 * _612;
        inter.u = _576 * _612;
        inter.v = _597 * _612;
        break;
    } while(false);
}

bool IntersectTris_AnyHit(float3 ro, float3 rd, int tri_start, int tri_end, int obj_index, inout hit_data_t out_inter)
{
    int _2130 = 0;
    int _2131 = obj_index;
    float _2133 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _2132;
    float _2134;
    float _2135;
    for (int i = tri_start; i < tri_end; )
    {
        param = ro;
        param_1 = rd;
        tri_accel_t _673;
        _673.n_plane = asfloat(_662.Load4(i * 48 + 0));
        _673.u_plane = asfloat(_662.Load4(i * 48 + 16));
        _673.v_plane = asfloat(_662.Load4(i * 48 + 32));
        param_2.n_plane = _673.n_plane;
        param_2.u_plane = _673.u_plane;
        param_2.v_plane = _673.v_plane;
        param_3 = uint(i);
        hit_data_t _2142 = { _2130, _2131, _2132, _2133, _2134, _2135 };
        param_4 = _2142;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _2130 = param_4.mask;
        _2131 = param_4.obj_index;
        _2132 = param_4.prim_index;
        _2133 = param_4.t;
        _2134 = param_4.u;
        _2135 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _2130;
    int _696;
    if (_2130 != 0)
    {
        _696 = _2131;
    }
    else
    {
        _696 = out_inter.obj_index;
    }
    out_inter.obj_index = _696;
    int _709;
    if (_2130 != 0)
    {
        _709 = _2132;
    }
    else
    {
        _709 = out_inter.prim_index;
    }
    out_inter.prim_index = _709;
    out_inter.t = _2133;
    float _725;
    if (_2130 != 0)
    {
        _725 = _2134;
    }
    else
    {
        _725 = out_inter.u;
    }
    out_inter.u = _725;
    float _738;
    if (_2130 != 0)
    {
        _738 = _2135;
    }
    else
    {
        _738 = out_inter.v;
    }
    out_inter.v = _738;
    return _2130 != 0;
}

bool Traverse_MicroTree_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    bool _2060 = false;
    bool _2057;
    do
    {
        float3 _844 = (-inv_d) * ro;
        uint _846 = stack_size;
        uint _856 = stack_size;
        stack_size = _856 + uint(1);
        g_stack[gl_LocalInvocationIndex][_856] = node_index;
        uint _930;
        uint _954;
        int _1006;
        while (stack_size != _846)
        {
            uint _871 = stack_size;
            uint _872 = _871 - uint(1);
            stack_size = _872;
            bvh_node_t _886;
            _886.bbox_min = asfloat(_882.Load4(g_stack[gl_LocalInvocationIndex][_872] * 32 + 0));
            _886.bbox_max = asfloat(_882.Load4(g_stack[gl_LocalInvocationIndex][_872] * 32 + 16));
            float3 param = inv_d;
            float3 param_1 = _844;
            float param_2 = inter.t;
            float3 param_3 = _886.bbox_min.xyz;
            float3 param_4 = _886.bbox_max.xyz;
            if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
            {
                continue;
            }
            uint _913 = asuint(_886.bbox_min.w);
            if ((_913 & 2147483648u) == 0u)
            {
                uint _920 = stack_size;
                stack_size = _920 + uint(1);
                uint _924 = asuint(_886.bbox_max.w);
                uint _926 = _924 >> uint(30);
                if (rd[_926] < 0.0f)
                {
                    _930 = _913;
                }
                else
                {
                    _930 = _924 & 1073741823u;
                }
                g_stack[gl_LocalInvocationIndex][_920] = _930;
                uint _945 = stack_size;
                stack_size = _945 + uint(1);
                if (rd[_926] < 0.0f)
                {
                    _954 = _924 & 1073741823u;
                }
                else
                {
                    _954 = _913;
                }
                g_stack[gl_LocalInvocationIndex][_945] = _954;
            }
            else
            {
                int _974 = int(_913 & 2147483647u);
                float3 param_5 = ro;
                float3 param_6 = rd;
                int param_7 = _974;
                int param_8 = _974 + asint(_886.bbox_max.w);
                int param_9 = obj_index;
                hit_data_t param_10 = inter;
                bool _995 = IntersectTris_AnyHit(param_5, param_6, param_7, param_8, param_9, param_10);
                inter = param_10;
                if (_995)
                {
                    bool _1003 = inter.prim_index < 0;
                    if (_1003)
                    {
                        _1006 = (-1) - inter.prim_index;
                    }
                    else
                    {
                        _1006 = inter.prim_index;
                    }
                    uint _1017 = uint(_1006);
                    bool _1044 = !_1003;
                    bool _1050;
                    if (_1044)
                    {
                        _1050 = (((_1031.Load(_1022.Load(_1017 * 4 + 0) * 4 + 0) >> 16u) & 65535u) & 32768u) != 0u;
                    }
                    else
                    {
                        _1050 = _1044;
                    }
                    bool _1061;
                    if (!_1050)
                    {
                        bool _1060;
                        if (_1003)
                        {
                            _1060 = ((_1031.Load(_1022.Load(_1017 * 4 + 0) * 4 + 0) & 65535u) & 32768u) != 0u;
                        }
                        else
                        {
                            _1060 = _1003;
                        }
                        _1061 = _1060;
                    }
                    else
                    {
                        _1061 = _1050;
                    }
                    if (_1061)
                    {
                        _2060 = true;
                        _2057 = true;
                        break;
                    }
                }
            }
        }
        if (_2060)
        {
            break;
        }
        _2060 = true;
        _2057 = false;
        break;
    } while(false);
    return _2057;
}

bool Traverse_MacroTree_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    bool _2051 = false;
    bool _2048;
    do
    {
        float3 _1072 = (-orig_inv_rd) * orig_ro;
        uint stack_size = 1u;
        g_stack[gl_LocalInvocationIndex][0u] = node_index;
        uint _1137;
        uint _1160;
        while (stack_size != 0u)
        {
            uint _1088 = stack_size;
            uint _1089 = _1088 - uint(1);
            stack_size = _1089;
            bvh_node_t _1095;
            _1095.bbox_min = asfloat(_882.Load4(g_stack[gl_LocalInvocationIndex][_1089] * 32 + 0));
            _1095.bbox_max = asfloat(_882.Load4(g_stack[gl_LocalInvocationIndex][_1089] * 32 + 16));
            float3 param = orig_inv_rd;
            float3 param_1 = _1072;
            float param_2 = inter.t;
            float3 param_3 = _1095.bbox_min.xyz;
            float3 param_4 = _1095.bbox_max.xyz;
            if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
            {
                continue;
            }
            uint _1122 = asuint(_1095.bbox_min.w);
            if ((_1122 & 2147483648u) == 0u)
            {
                uint _1128 = stack_size;
                stack_size = _1128 + uint(1);
                uint _1132 = asuint(_1095.bbox_max.w);
                uint _1133 = _1132 >> uint(30);
                if (orig_rd[_1133] < 0.0f)
                {
                    _1137 = _1122;
                }
                else
                {
                    _1137 = _1132 & 1073741823u;
                }
                g_stack[gl_LocalInvocationIndex][_1128] = _1137;
                uint _1151 = stack_size;
                stack_size = _1151 + uint(1);
                if (orig_rd[_1133] < 0.0f)
                {
                    _1160 = _1132 & 1073741823u;
                }
                else
                {
                    _1160 = _1122;
                }
                g_stack[gl_LocalInvocationIndex][_1151] = _1160;
            }
            else
            {
                uint _1178 = _1122 & 2147483647u;
                uint _1182 = asuint(_1095.bbox_max.w);
                for (uint i = _1178; i < (_1178 + _1182); i++)
                {
                    mesh_instance_t _1212;
                    _1212.bbox_min = asfloat(_1202.Load4(_1206.Load(i * 4 + 0) * 32 + 0));
                    _1212.bbox_max = asfloat(_1202.Load4(_1206.Load(i * 4 + 0) * 32 + 16));
                    mesh_t _1233;
                    [unroll]
                    for (int _28ident = 0; _28ident < 3; _28ident++)
                    {
                        _1233.bbox_min[_28ident] = asfloat(_1227.Load(_28ident * 4 + asuint(_1212.bbox_max.w) * 48 + 0));
                    }
                    [unroll]
                    for (int _29ident = 0; _29ident < 3; _29ident++)
                    {
                        _1233.bbox_max[_29ident] = asfloat(_1227.Load(_29ident * 4 + asuint(_1212.bbox_max.w) * 48 + 12));
                    }
                    _1233.node_index = _1227.Load(asuint(_1212.bbox_max.w) * 48 + 24);
                    _1233.node_count = _1227.Load(asuint(_1212.bbox_max.w) * 48 + 28);
                    _1233.tris_index = _1227.Load(asuint(_1212.bbox_max.w) * 48 + 32);
                    _1233.tris_count = _1227.Load(asuint(_1212.bbox_max.w) * 48 + 36);
                    _1233.vert_index = _1227.Load(asuint(_1212.bbox_max.w) * 48 + 40);
                    _1233.vert_count = _1227.Load(asuint(_1212.bbox_max.w) * 48 + 44);
                    transform_t _1277;
                    _1277.xform = asfloat(uint4x4(_1271.Load4(asuint(_1212.bbox_min.w) * 128 + 0), _1271.Load4(asuint(_1212.bbox_min.w) * 128 + 16), _1271.Load4(asuint(_1212.bbox_min.w) * 128 + 32), _1271.Load4(asuint(_1212.bbox_min.w) * 128 + 48)));
                    _1277.inv_xform = asfloat(uint4x4(_1271.Load4(asuint(_1212.bbox_min.w) * 128 + 64), _1271.Load4(asuint(_1212.bbox_min.w) * 128 + 80), _1271.Load4(asuint(_1212.bbox_min.w) * 128 + 96), _1271.Load4(asuint(_1212.bbox_min.w) * 128 + 112)));
                    float3 param_5 = orig_inv_rd;
                    float3 param_6 = _1072;
                    float param_7 = inter.t;
                    float3 param_8 = _1212.bbox_min.xyz;
                    float3 param_9 = _1212.bbox_max.xyz;
                    if (!_bbox_test_fma(param_5, param_6, param_7, param_8, param_9))
                    {
                        continue;
                    }
                    float3 _1322 = mul(float4(orig_rd, 0.0f), _1277.inv_xform).xyz;
                    float3 param_10 = _1322;
                    float3 param_11 = mul(float4(orig_ro, 1.0f), _1277.inv_xform).xyz;
                    float3 param_12 = _1322;
                    float3 param_13 = safe_invert(param_10);
                    int param_14 = int(_1206.Load(i * 4 + 0));
                    uint param_15 = _1233.node_index;
                    uint param_16 = stack_size;
                    hit_data_t param_17 = inter;
                    bool _1346 = Traverse_MicroTree_WithStack(param_11, param_12, param_13, param_14, param_15, param_16, param_17);
                    inter = param_17;
                    if (_1346)
                    {
                        _2051 = true;
                        _2048 = true;
                        break;
                    }
                }
                if (_2051)
                {
                    break;
                }
            }
        }
        if (_2051)
        {
            break;
        }
        _2051 = true;
        _2048 = false;
        break;
    } while(false);
    return _2048;
}

float construct_float(inout uint m)
{
    m &= 8388607u;
    m |= 1065353216u;
    return asfloat(m) - 1.0f;
}

float2 TransformUV(float2 _uv, atlas_texture_t t, int mip_level)
{
    uint _2085[14] = t.pos;
    uint _2088[14] = t.pos;
    uint _348 = t.size & 16383u;
    uint _351 = t.size >> uint(16);
    uint _352 = _351 & 16383u;
    float2 size = float2(float(_348), float(_352));
    if ((_351 & 32768u) != 0u)
    {
        size = float2(float(_348 >> uint(mip_level)), float(_352 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_2085[mip_level] & 65535u), float((_2088[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _291 = mad(col.z, 31.875f, 1.0f);
    float _302 = (col.x - 0.501960813999176025390625f) / _291;
    float _308 = (col.y - 0.501960813999176025390625f) / _291;
    return float3((col.w + _302) - _308, col.w + _308, (col.w - _302) - _308);
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
    atlas_texture_t _393;
    _393.size = _390.Load(index * 80 + 0);
    _393.atlas = _390.Load(index * 80 + 4);
    [unroll]
    for (int _30ident = 0; _30ident < 4; _30ident++)
    {
        _393.page[_30ident] = _390.Load(_30ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _31ident = 0; _31ident < 14; _31ident++)
    {
        _393.pos[_31ident] = _390.Load(_31ident * 4 + index * 80 + 24);
    }
    uint _2093[4];
    _2093[0] = _393.page[0];
    _2093[1] = _393.page[1];
    _2093[2] = _393.page[2];
    _2093[3] = _393.page[3];
    uint _2129[14] = { _393.pos[0], _393.pos[1], _393.pos[2], _393.pos[3], _393.pos[4], _393.pos[5], _393.pos[6], _393.pos[7], _393.pos[8], _393.pos[9], _393.pos[10], _393.pos[11], _393.pos[12], _393.pos[13] };
    atlas_texture_t _2099 = { _393.size, _393.atlas, _2093, _2129 };
    uint _476 = _393.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_476)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_476)], float3(TransformUV(uvs, _2099, lod) * 0.000118371215648949146270751953125f.xx, float((_2093[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
    bool _491;
    if (maybe_YCoCg)
    {
        _491 = _393.atlas == 4u;
    }
    else
    {
        _491 = maybe_YCoCg;
    }
    if (_491)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _510;
    if (maybe_SRGB)
    {
        _510 = (_393.size & 32768u) != 0u;
    }
    else
    {
        _510 = maybe_SRGB;
    }
    if (_510)
    {
        float3 param_1 = res.xyz;
        float3 _516 = srgb_to_rgb(param_1);
        float4 _2332 = res;
        _2332.x = _516.x;
        float4 _2334 = _2332;
        _2334.y = _516.y;
        float4 _2336 = _2334;
        _2336.z = _516.z;
        res = _2336;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod)
{
    return SampleBilinear(index, uvs, lod, false, false);
}

bool ComputeVisibility(inout float3 p, float3 d, inout float dist, float rand_val, int rand_hash2)
{
    bool _2044 = false;
    bool _2041;
    do
    {
        float3 param = d;
        float3 _1359 = safe_invert(param);
        int _1408;
        int _2192;
        int _2193;
        float _2195;
        float _2196;
        while (dist > 9.9999997473787516355514526367188e-06f)
        {
            int _2191 = 0;
            float _2194 = dist;
            float3 param_1 = p;
            float3 param_2 = d;
            float3 param_3 = _1359;
            uint param_4 = _1377_g_params.node_index;
            hit_data_t _2203 = { 0, _2192, _2193, dist, _2195, _2196 };
            hit_data_t param_5 = _2203;
            bool _1390 = Traverse_MacroTree_WithStack(param_1, param_2, param_3, param_4, param_5);
            _2191 = param_5.mask;
            _2192 = param_5.obj_index;
            _2193 = param_5.prim_index;
            _2194 = param_5.t;
            _2195 = param_5.u;
            _2196 = param_5.v;
            if (_1390)
            {
                _2044 = true;
                _2041 = false;
                break;
            }
            if (_2191 == 0)
            {
                _2044 = true;
                _2041 = true;
                break;
            }
            bool _1405 = param_5.prim_index < 0;
            if (_1405)
            {
                _1408 = (-1) - param_5.prim_index;
            }
            else
            {
                _1408 = param_5.prim_index;
            }
            uint _1419 = uint(_1408);
            material_t _1456;
            [unroll]
            for (int _32ident = 0; _32ident < 5; _32ident++)
            {
                _1456.textures[_32ident] = _1452.Load(_32ident * 4 + ((_1405 ? (_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) & 65535u) : ((_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 0);
            }
            [unroll]
            for (int _33ident = 0; _33ident < 3; _33ident++)
            {
                _1456.base_color[_33ident] = asfloat(_1452.Load(_33ident * 4 + ((_1405 ? (_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) & 65535u) : ((_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 20));
            }
            _1456.flags = _1452.Load(((_1405 ? (_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) & 65535u) : ((_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 32);
            _1456.type = _1452.Load(((_1405 ? (_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) & 65535u) : ((_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 36);
            _1456.tangent_rotation_or_strength = asfloat(_1452.Load(((_1405 ? (_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) & 65535u) : ((_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 40));
            _1456.roughness_and_anisotropic = _1452.Load(((_1405 ? (_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) & 65535u) : ((_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 44);
            _1456.int_ior = asfloat(_1452.Load(((_1405 ? (_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) & 65535u) : ((_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 48));
            _1456.ext_ior = asfloat(_1452.Load(((_1405 ? (_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) & 65535u) : ((_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 52));
            _1456.sheen_and_sheen_tint = _1452.Load(((_1405 ? (_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) & 65535u) : ((_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 56);
            _1456.tint_and_metallic = _1452.Load(((_1405 ? (_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) & 65535u) : ((_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 60);
            _1456.transmission_and_transmission_roughness = _1452.Load(((_1405 ? (_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) & 65535u) : ((_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 64);
            _1456.specular_and_specular_tint = _1452.Load(((_1405 ? (_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) & 65535u) : ((_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 68);
            _1456.clearcoat_and_clearcoat_roughness = _1452.Load(((_1405 ? (_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) & 65535u) : ((_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 72);
            _1456.normal_map_strength_unorm = _1452.Load(((_1405 ? (_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) & 65535u) : ((_1031.Load(_1022.Load(_1419 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 76);
            uint _2247 = _1456.textures[1];
            uint _2249 = _1456.textures[3];
            uint _2250 = _1456.textures[4];
            uint _2213 = _1456.type;
            float _2214 = _1456.tangent_rotation_or_strength;
            uint _1535 = _1022.Load(_1419 * 4 + 0) * 3u;
            vertex_t _1541;
            [unroll]
            for (int _34ident = 0; _34ident < 3; _34ident++)
            {
                _1541.p[_34ident] = asfloat(_1529.Load(_34ident * 4 + _1533.Load(_1535 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _35ident = 0; _35ident < 3; _35ident++)
            {
                _1541.n[_35ident] = asfloat(_1529.Load(_35ident * 4 + _1533.Load(_1535 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _1541.b[_36ident] = asfloat(_1529.Load(_36ident * 4 + _1533.Load(_1535 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _37ident = 0; _37ident < 2; _37ident++)
            {
                [unroll]
                for (int _38ident = 0; _38ident < 2; _38ident++)
                {
                    _1541.t[_37ident][_38ident] = asfloat(_1529.Load(_38ident * 4 + _37ident * 8 + _1533.Load(_1535 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1589;
            [unroll]
            for (int _39ident = 0; _39ident < 3; _39ident++)
            {
                _1589.p[_39ident] = asfloat(_1529.Load(_39ident * 4 + _1533.Load((_1535 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _1589.n[_40ident] = asfloat(_1529.Load(_40ident * 4 + _1533.Load((_1535 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 3; _41ident++)
            {
                _1589.b[_41ident] = asfloat(_1529.Load(_41ident * 4 + _1533.Load((_1535 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _42ident = 0; _42ident < 2; _42ident++)
            {
                [unroll]
                for (int _43ident = 0; _43ident < 2; _43ident++)
                {
                    _1589.t[_42ident][_43ident] = asfloat(_1529.Load(_43ident * 4 + _42ident * 8 + _1533.Load((_1535 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1635;
            [unroll]
            for (int _44ident = 0; _44ident < 3; _44ident++)
            {
                _1635.p[_44ident] = asfloat(_1529.Load(_44ident * 4 + _1533.Load((_1535 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _45ident = 0; _45ident < 3; _45ident++)
            {
                _1635.n[_45ident] = asfloat(_1529.Load(_45ident * 4 + _1533.Load((_1535 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _46ident = 0; _46ident < 3; _46ident++)
            {
                _1635.b[_46ident] = asfloat(_1529.Load(_46ident * 4 + _1533.Load((_1535 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _47ident = 0; _47ident < 2; _47ident++)
            {
                [unroll]
                for (int _48ident = 0; _48ident < 2; _48ident++)
                {
                    _1635.t[_47ident][_48ident] = asfloat(_1529.Load(_48ident * 4 + _47ident * 8 + _1533.Load((_1535 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            float2 _1706 = ((float2(_1541.t[0][0], _1541.t[0][1]) * ((1.0f - param_5.u) - param_5.v)) + (float2(_1589.t[0][0], _1589.t[0][1]) * param_5.u)) + (float2(_1635.t[0][0], _1635.t[0][1]) * param_5.v);
            uint param_6 = uint(hash(rand_hash2));
            float _1714 = construct_float(param_6);
            float sh_r = frac(rand_val + _1714);
            while (_2213 == 4u)
            {
                float mix_val = _2214;
                if (_2247 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_2247, _1706, 0).x;
                }
                if (sh_r > mix_val)
                {
                    material_t _1752;
                    [unroll]
                    for (int _49ident = 0; _49ident < 5; _49ident++)
                    {
                        _1752.textures[_49ident] = _1452.Load(_49ident * 4 + _2249 * 80 + 0);
                    }
                    [unroll]
                    for (int _50ident = 0; _50ident < 3; _50ident++)
                    {
                        _1752.base_color[_50ident] = asfloat(_1452.Load(_50ident * 4 + _2249 * 80 + 20));
                    }
                    _1752.flags = _1452.Load(_2249 * 80 + 32);
                    _1752.type = _1452.Load(_2249 * 80 + 36);
                    _1752.tangent_rotation_or_strength = asfloat(_1452.Load(_2249 * 80 + 40));
                    _1752.roughness_and_anisotropic = _1452.Load(_2249 * 80 + 44);
                    _1752.int_ior = asfloat(_1452.Load(_2249 * 80 + 48));
                    _1752.ext_ior = asfloat(_1452.Load(_2249 * 80 + 52));
                    _1752.sheen_and_sheen_tint = _1452.Load(_2249 * 80 + 56);
                    _1752.tint_and_metallic = _1452.Load(_2249 * 80 + 60);
                    _1752.transmission_and_transmission_roughness = _1452.Load(_2249 * 80 + 64);
                    _1752.specular_and_specular_tint = _1452.Load(_2249 * 80 + 68);
                    _1752.clearcoat_and_clearcoat_roughness = _1452.Load(_2249 * 80 + 72);
                    _1752.normal_map_strength_unorm = _1452.Load(_2249 * 80 + 76);
                    _2247 = _1752.textures[1];
                    _2249 = _1752.textures[3];
                    _2250 = _1752.textures[4];
                    _2213 = _1752.type;
                    _2214 = _1752.tangent_rotation_or_strength;
                    sh_r = (sh_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _1807;
                    [unroll]
                    for (int _51ident = 0; _51ident < 5; _51ident++)
                    {
                        _1807.textures[_51ident] = _1452.Load(_51ident * 4 + _2250 * 80 + 0);
                    }
                    [unroll]
                    for (int _52ident = 0; _52ident < 3; _52ident++)
                    {
                        _1807.base_color[_52ident] = asfloat(_1452.Load(_52ident * 4 + _2250 * 80 + 20));
                    }
                    _1807.flags = _1452.Load(_2250 * 80 + 32);
                    _1807.type = _1452.Load(_2250 * 80 + 36);
                    _1807.tangent_rotation_or_strength = asfloat(_1452.Load(_2250 * 80 + 40));
                    _1807.roughness_and_anisotropic = _1452.Load(_2250 * 80 + 44);
                    _1807.int_ior = asfloat(_1452.Load(_2250 * 80 + 48));
                    _1807.ext_ior = asfloat(_1452.Load(_2250 * 80 + 52));
                    _1807.sheen_and_sheen_tint = _1452.Load(_2250 * 80 + 56);
                    _1807.tint_and_metallic = _1452.Load(_2250 * 80 + 60);
                    _1807.transmission_and_transmission_roughness = _1452.Load(_2250 * 80 + 64);
                    _1807.specular_and_specular_tint = _1452.Load(_2250 * 80 + 68);
                    _1807.clearcoat_and_clearcoat_roughness = _1452.Load(_2250 * 80 + 72);
                    _1807.normal_map_strength_unorm = _1452.Load(_2250 * 80 + 76);
                    _2247 = _1807.textures[1];
                    _2249 = _1807.textures[3];
                    _2250 = _1807.textures[4];
                    _2213 = _1807.type;
                    _2214 = _1807.tangent_rotation_or_strength;
                    sh_r /= mix_val;
                }
            }
            if (_2213 != 5u)
            {
                _2044 = true;
                _2041 = false;
                break;
            }
            float _1864 = _2194 + 9.9999997473787516355514526367188e-06f;
            p += (d * _1864);
            dist -= _1864;
        }
        if (_2044)
        {
            break;
        }
        _2044 = true;
        _2041 = true;
        break;
    } while(false);
    return _2041;
}

void comp_main()
{
    do
    {
        int _1884 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_1884) >= _1890.Load(12))
        {
            break;
        }
        shadow_ray_t _1911;
        [unroll]
        for (int _53ident = 0; _53ident < 3; _53ident++)
        {
            _1911.o[_53ident] = asfloat(_1907.Load(_53ident * 4 + _1884 * 44 + 0));
        }
        [unroll]
        for (int _54ident = 0; _54ident < 3; _54ident++)
        {
            _1911.d[_54ident] = asfloat(_1907.Load(_54ident * 4 + _1884 * 44 + 12));
        }
        _1911.dist = asfloat(_1907.Load(_1884 * 44 + 24));
        [unroll]
        for (int _55ident = 0; _55ident < 3; _55ident++)
        {
            _1911.c[_55ident] = asfloat(_1907.Load(_55ident * 4 + _1884 * 44 + 28));
        }
        _1911.xy = int(_1907.Load(_1884 * 44 + 40));
        int _1945 = (_1911.xy >> 16) & 65535;
        int _1949 = _1911.xy & 65535;
        float3 param = float3(asfloat(_1907.Load(_1884 * 44 + 0)), asfloat(_1907.Load(_1884 * 44 + 4)), asfloat(_1907.Load(_1884 * 44 + 8)));
        float3 param_1 = float3(asfloat(_1907.Load(_1884 * 44 + 12)), asfloat(_1907.Load(_1884 * 44 + 16)), asfloat(_1907.Load(_1884 * 44 + 20)));
        float param_2 = _1911.dist;
        float param_3 = _1377_g_params.random_val;
        int param_4 = hash((_1945 << 16) | _1949);
        bool _1989 = ComputeVisibility(param, param_1, param_2, param_3, param_4);
        if (_1989)
        {
            int2 _2000 = int2(_1945, _1949);
            g_out_img[_2000] = float4(g_out_img[_2000].xyz + float3(_1911.c[0], _1911.c[1], _1911.c[2]), 1.0f);
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
