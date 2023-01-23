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

struct shadow_ray_t
{
    float o[3];
    int depth;
    float d[3];
    float dist;
    float c[3];
    int xy;
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
    int max_transp_depth;
};

struct vertex_t
{
    float p[3];
    float n[3];
    float b[3];
    float t[2][2];
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

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

ByteAddressBuffer _365 : register(t20, space0);
ByteAddressBuffer _637 : register(t1, space0);
ByteAddressBuffer _857 : register(t5, space0);
ByteAddressBuffer _997 : register(t2, space0);
ByteAddressBuffer _1006 : register(t3, space0);
ByteAddressBuffer _1177 : register(t7, space0);
ByteAddressBuffer _1181 : register(t8, space0);
ByteAddressBuffer _1201 : register(t6, space0);
ByteAddressBuffer _1245 : register(t9, space0);
ByteAddressBuffer _1469 : register(t10, space0);
ByteAddressBuffer _1473 : register(t11, space0);
ByteAddressBuffer _1674 : register(t4, space0);
ByteAddressBuffer _1853 : register(t13, space0);
ByteAddressBuffer _1868 : register(t12, space0);
cbuffer UniformParams
{
    Params _1380_g_params : packoffset(c0);
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

float3 safe_invert(float3 v)
{
    float3 inv_v = 1.0f.xxx / v;
    bool _138 = v.x <= 1.0000000116860974230803549289703e-07f;
    bool _145;
    if (_138)
    {
        _145 = v.x >= 0.0f;
    }
    else
    {
        _145 = _138;
    }
    if (_145)
    {
        float3 _2253 = inv_v;
        _2253.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _2253;
    }
    else
    {
        bool _154 = v.x >= (-1.0000000116860974230803549289703e-07f);
        bool _160;
        if (_154)
        {
            _160 = v.x < 0.0f;
        }
        else
        {
            _160 = _154;
        }
        if (_160)
        {
            float3 _2255 = inv_v;
            _2255.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _2255;
        }
    }
    bool _167 = v.y <= 1.0000000116860974230803549289703e-07f;
    bool _173;
    if (_167)
    {
        _173 = v.y >= 0.0f;
    }
    else
    {
        _173 = _167;
    }
    if (_173)
    {
        float3 _2257 = inv_v;
        _2257.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _2257;
    }
    else
    {
        bool _180 = v.y >= (-1.0000000116860974230803549289703e-07f);
        bool _186;
        if (_180)
        {
            _186 = v.y < 0.0f;
        }
        else
        {
            _186 = _180;
        }
        if (_186)
        {
            float3 _2259 = inv_v;
            _2259.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _2259;
        }
    }
    bool _192 = v.z <= 1.0000000116860974230803549289703e-07f;
    bool _198;
    if (_192)
    {
        _198 = v.z >= 0.0f;
    }
    else
    {
        _198 = _192;
    }
    if (_198)
    {
        float3 _2261 = inv_v;
        _2261.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _2261;
    }
    else
    {
        bool _205 = v.z >= (-1.0000000116860974230803549289703e-07f);
        bool _211;
        if (_205)
        {
            _211 = v.z < 0.0f;
        }
        else
        {
            _211 = _205;
        }
        if (_211)
        {
            float3 _2263 = inv_v;
            _2263.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _2263;
        }
    }
    return inv_v;
}

bool _bbox_test_fma(float3 inv_d, float3 neg_inv_d_o, float t, float3 bbox_min, float3 bbox_max)
{
    float _735 = mad(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float _743 = mad(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float _758 = mad(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    float _765 = mad(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    float _782 = mad(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    float _789 = mad(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    float _794 = max(max(min(_735, _743), min(_758, _765)), min(_782, _789));
    float _802 = min(min(max(_735, _743), max(_758, _765)), max(_782, _789)) * 1.0000002384185791015625f;
    return ((_794 <= _802) && (_794 <= t)) && (_802 > 0.0f);
}

void IntersectTri(float3 ro, float3 rd, tri_accel_t tri, uint prim_index, inout hit_data_t inter)
{
    do
    {
        float _510 = dot(rd, tri.n_plane.xyz);
        float _519 = tri.n_plane.w - dot(ro, tri.n_plane.xyz);
        if (sign(_519) != sign(mad(_510, inter.t, -_519)))
        {
            break;
        }
        float3 _540 = (ro * _510) + (rd * _519);
        float _551 = mad(_510, tri.u_plane.w, dot(_540, tri.u_plane.xyz));
        float _556 = _510 - _551;
        if (sign(_551) != sign(_556))
        {
            break;
        }
        float _572 = mad(_510, tri.v_plane.w, dot(_540, tri.v_plane.xyz));
        if (sign(_572) != sign(_556 - _572))
        {
            break;
        }
        float _587 = 1.0f / _510;
        inter.mask = -1;
        int _592;
        if (_510 < 0.0f)
        {
            _592 = int(prim_index);
        }
        else
        {
            _592 = (-1) - int(prim_index);
        }
        inter.prim_index = _592;
        inter.t = _519 * _587;
        inter.u = _551 * _587;
        inter.v = _572 * _587;
        break;
    } while(false);
}

bool IntersectTris_AnyHit(float3 ro, float3 rd, int tri_start, int tri_end, int obj_index, inout hit_data_t out_inter)
{
    int _2074 = 0;
    int _2075 = obj_index;
    float _2077 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _2076;
    float _2078;
    float _2079;
    for (int i = tri_start; i < tri_end; )
    {
        param = ro;
        param_1 = rd;
        tri_accel_t _648;
        _648.n_plane = asfloat(_637.Load4(i * 48 + 0));
        _648.u_plane = asfloat(_637.Load4(i * 48 + 16));
        _648.v_plane = asfloat(_637.Load4(i * 48 + 32));
        param_2.n_plane = _648.n_plane;
        param_2.u_plane = _648.u_plane;
        param_2.v_plane = _648.v_plane;
        param_3 = uint(i);
        hit_data_t _2086 = { _2074, _2075, _2076, _2077, _2078, _2079 };
        param_4 = _2086;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _2074 = param_4.mask;
        _2075 = param_4.obj_index;
        _2076 = param_4.prim_index;
        _2077 = param_4.t;
        _2078 = param_4.u;
        _2079 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _2074;
    int _671;
    if (_2074 != 0)
    {
        _671 = _2075;
    }
    else
    {
        _671 = out_inter.obj_index;
    }
    out_inter.obj_index = _671;
    int _684;
    if (_2074 != 0)
    {
        _684 = _2076;
    }
    else
    {
        _684 = out_inter.prim_index;
    }
    out_inter.prim_index = _684;
    out_inter.t = _2077;
    float _700;
    if (_2074 != 0)
    {
        _700 = _2078;
    }
    else
    {
        _700 = out_inter.u;
    }
    out_inter.u = _700;
    float _713;
    if (_2074 != 0)
    {
        _713 = _2079;
    }
    else
    {
        _713 = out_inter.v;
    }
    out_inter.v = _713;
    return _2074 != 0;
}

bool Traverse_MicroTree_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    bool _1987 = false;
    bool _1984;
    do
    {
        float3 _819 = (-inv_d) * ro;
        uint _821 = stack_size;
        uint _831 = stack_size;
        stack_size = _831 + uint(1);
        g_stack[gl_LocalInvocationIndex][_831] = node_index;
        uint _905;
        uint _929;
        int _981;
        while (stack_size != _821)
        {
            uint _846 = stack_size;
            uint _847 = _846 - uint(1);
            stack_size = _847;
            bvh_node_t _861;
            _861.bbox_min = asfloat(_857.Load4(g_stack[gl_LocalInvocationIndex][_847] * 32 + 0));
            _861.bbox_max = asfloat(_857.Load4(g_stack[gl_LocalInvocationIndex][_847] * 32 + 16));
            float3 param = inv_d;
            float3 param_1 = _819;
            float param_2 = inter.t;
            float3 param_3 = _861.bbox_min.xyz;
            float3 param_4 = _861.bbox_max.xyz;
            if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
            {
                continue;
            }
            uint _888 = asuint(_861.bbox_min.w);
            if ((_888 & 2147483648u) == 0u)
            {
                uint _895 = stack_size;
                stack_size = _895 + uint(1);
                uint _899 = asuint(_861.bbox_max.w);
                uint _901 = _899 >> uint(30);
                if (rd[_901] < 0.0f)
                {
                    _905 = _888;
                }
                else
                {
                    _905 = _899 & 1073741823u;
                }
                g_stack[gl_LocalInvocationIndex][_895] = _905;
                uint _920 = stack_size;
                stack_size = _920 + uint(1);
                if (rd[_901] < 0.0f)
                {
                    _929 = _899 & 1073741823u;
                }
                else
                {
                    _929 = _888;
                }
                g_stack[gl_LocalInvocationIndex][_920] = _929;
            }
            else
            {
                int _949 = int(_888 & 2147483647u);
                float3 param_5 = ro;
                float3 param_6 = rd;
                int param_7 = _949;
                int param_8 = _949 + asint(_861.bbox_max.w);
                int param_9 = obj_index;
                hit_data_t param_10 = inter;
                bool _970 = IntersectTris_AnyHit(param_5, param_6, param_7, param_8, param_9, param_10);
                inter = param_10;
                if (_970)
                {
                    bool _978 = inter.prim_index < 0;
                    if (_978)
                    {
                        _981 = (-1) - inter.prim_index;
                    }
                    else
                    {
                        _981 = inter.prim_index;
                    }
                    uint _992 = uint(_981);
                    bool _1019 = !_978;
                    bool _1025;
                    if (_1019)
                    {
                        _1025 = (((_1006.Load(_997.Load(_992 * 4 + 0) * 4 + 0) >> 16u) & 65535u) & 32768u) != 0u;
                    }
                    else
                    {
                        _1025 = _1019;
                    }
                    bool _1036;
                    if (!_1025)
                    {
                        bool _1035;
                        if (_978)
                        {
                            _1035 = ((_1006.Load(_997.Load(_992 * 4 + 0) * 4 + 0) & 65535u) & 32768u) != 0u;
                        }
                        else
                        {
                            _1035 = _978;
                        }
                        _1036 = _1035;
                    }
                    else
                    {
                        _1036 = _1025;
                    }
                    if (_1036)
                    {
                        _1987 = true;
                        _1984 = true;
                        break;
                    }
                }
            }
        }
        if (_1987)
        {
            break;
        }
        _1987 = true;
        _1984 = false;
        break;
    } while(false);
    return _1984;
}

bool Traverse_MacroTree_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    bool _1978 = false;
    bool _1975;
    do
    {
        float3 _1047 = (-orig_inv_rd) * orig_ro;
        uint stack_size = 1u;
        g_stack[gl_LocalInvocationIndex][0u] = node_index;
        uint _1112;
        uint _1135;
        while (stack_size != 0u)
        {
            uint _1063 = stack_size;
            uint _1064 = _1063 - uint(1);
            stack_size = _1064;
            bvh_node_t _1070;
            _1070.bbox_min = asfloat(_857.Load4(g_stack[gl_LocalInvocationIndex][_1064] * 32 + 0));
            _1070.bbox_max = asfloat(_857.Load4(g_stack[gl_LocalInvocationIndex][_1064] * 32 + 16));
            float3 param = orig_inv_rd;
            float3 param_1 = _1047;
            float param_2 = inter.t;
            float3 param_3 = _1070.bbox_min.xyz;
            float3 param_4 = _1070.bbox_max.xyz;
            if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
            {
                continue;
            }
            uint _1097 = asuint(_1070.bbox_min.w);
            if ((_1097 & 2147483648u) == 0u)
            {
                uint _1103 = stack_size;
                stack_size = _1103 + uint(1);
                uint _1107 = asuint(_1070.bbox_max.w);
                uint _1108 = _1107 >> uint(30);
                if (orig_rd[_1108] < 0.0f)
                {
                    _1112 = _1097;
                }
                else
                {
                    _1112 = _1107 & 1073741823u;
                }
                g_stack[gl_LocalInvocationIndex][_1103] = _1112;
                uint _1126 = stack_size;
                stack_size = _1126 + uint(1);
                if (orig_rd[_1108] < 0.0f)
                {
                    _1135 = _1107 & 1073741823u;
                }
                else
                {
                    _1135 = _1097;
                }
                g_stack[gl_LocalInvocationIndex][_1126] = _1135;
            }
            else
            {
                uint _1153 = _1097 & 2147483647u;
                uint _1157 = asuint(_1070.bbox_max.w);
                for (uint i = _1153; i < (_1153 + _1157); i++)
                {
                    mesh_instance_t _1187;
                    _1187.bbox_min = asfloat(_1177.Load4(_1181.Load(i * 4 + 0) * 32 + 0));
                    _1187.bbox_max = asfloat(_1177.Load4(_1181.Load(i * 4 + 0) * 32 + 16));
                    mesh_t _1207;
                    [unroll]
                    for (int _24ident = 0; _24ident < 3; _24ident++)
                    {
                        _1207.bbox_min[_24ident] = asfloat(_1201.Load(_24ident * 4 + asuint(_1187.bbox_max.w) * 48 + 0));
                    }
                    [unroll]
                    for (int _25ident = 0; _25ident < 3; _25ident++)
                    {
                        _1207.bbox_max[_25ident] = asfloat(_1201.Load(_25ident * 4 + asuint(_1187.bbox_max.w) * 48 + 12));
                    }
                    _1207.node_index = _1201.Load(asuint(_1187.bbox_max.w) * 48 + 24);
                    _1207.node_count = _1201.Load(asuint(_1187.bbox_max.w) * 48 + 28);
                    _1207.tris_index = _1201.Load(asuint(_1187.bbox_max.w) * 48 + 32);
                    _1207.tris_count = _1201.Load(asuint(_1187.bbox_max.w) * 48 + 36);
                    _1207.vert_index = _1201.Load(asuint(_1187.bbox_max.w) * 48 + 40);
                    _1207.vert_count = _1201.Load(asuint(_1187.bbox_max.w) * 48 + 44);
                    transform_t _1251;
                    _1251.xform = asfloat(uint4x4(_1245.Load4(asuint(_1187.bbox_min.w) * 128 + 0), _1245.Load4(asuint(_1187.bbox_min.w) * 128 + 16), _1245.Load4(asuint(_1187.bbox_min.w) * 128 + 32), _1245.Load4(asuint(_1187.bbox_min.w) * 128 + 48)));
                    _1251.inv_xform = asfloat(uint4x4(_1245.Load4(asuint(_1187.bbox_min.w) * 128 + 64), _1245.Load4(asuint(_1187.bbox_min.w) * 128 + 80), _1245.Load4(asuint(_1187.bbox_min.w) * 128 + 96), _1245.Load4(asuint(_1187.bbox_min.w) * 128 + 112)));
                    float3 param_5 = orig_inv_rd;
                    float3 param_6 = _1047;
                    float param_7 = inter.t;
                    float3 param_8 = _1187.bbox_min.xyz;
                    float3 param_9 = _1187.bbox_max.xyz;
                    if (!_bbox_test_fma(param_5, param_6, param_7, param_8, param_9))
                    {
                        continue;
                    }
                    float3 _1296 = mul(float4(orig_rd, 0.0f), _1251.inv_xform).xyz;
                    float3 param_10 = _1296;
                    float3 param_11 = mul(float4(orig_ro, 1.0f), _1251.inv_xform).xyz;
                    float3 param_12 = _1296;
                    float3 param_13 = safe_invert(param_10);
                    int param_14 = int(_1181.Load(i * 4 + 0));
                    uint param_15 = _1207.node_index;
                    uint param_16 = stack_size;
                    hit_data_t param_17 = inter;
                    bool _1320 = Traverse_MicroTree_WithStack(param_11, param_12, param_13, param_14, param_15, param_16, param_17);
                    inter = param_17;
                    if (_1320)
                    {
                        _1978 = true;
                        _1975 = true;
                        break;
                    }
                }
                if (_1978)
                {
                    break;
                }
            }
        }
        if (_1978)
        {
            break;
        }
        _1978 = true;
        _1975 = false;
        break;
    } while(false);
    return _1975;
}

float2 TransformUV(float2 _uv, atlas_texture_t t, int mip_level)
{
    uint _2029[14] = t.pos;
    uint _2032[14] = t.pos;
    uint _323 = t.size & 16383u;
    uint _326 = t.size >> uint(16);
    uint _327 = _326 & 16383u;
    float2 size = float2(float(_323), float(_327));
    if ((_326 & 32768u) != 0u)
    {
        size = float2(float(_323 >> uint(mip_level)), float(_327 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_2029[mip_level] & 65535u), float((_2032[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _266 = mad(col.z, 31.875f, 1.0f);
    float _276 = (col.x - 0.501960813999176025390625f) / _266;
    float _282 = (col.y - 0.501960813999176025390625f) / _266;
    return float3((col.w + _276) - _282, col.w + _282, (col.w - _276) - _282);
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
    atlas_texture_t _368;
    _368.size = _365.Load(index * 80 + 0);
    _368.atlas = _365.Load(index * 80 + 4);
    [unroll]
    for (int _26ident = 0; _26ident < 4; _26ident++)
    {
        _368.page[_26ident] = _365.Load(_26ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _27ident = 0; _27ident < 14; _27ident++)
    {
        _368.pos[_27ident] = _365.Load(_27ident * 4 + index * 80 + 24);
    }
    uint _2037[4];
    _2037[0] = _368.page[0];
    _2037[1] = _368.page[1];
    _2037[2] = _368.page[2];
    _2037[3] = _368.page[3];
    uint _2073[14] = { _368.pos[0], _368.pos[1], _368.pos[2], _368.pos[3], _368.pos[4], _368.pos[5], _368.pos[6], _368.pos[7], _368.pos[8], _368.pos[9], _368.pos[10], _368.pos[11], _368.pos[12], _368.pos[13] };
    atlas_texture_t _2043 = { _368.size, _368.atlas, _2037, _2073 };
    uint _451 = _368.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_451)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_451)], float3(TransformUV(uvs, _2043, lod) * 0.000118371215648949146270751953125f.xx, float((_2037[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
    bool _466;
    if (maybe_YCoCg)
    {
        _466 = _368.atlas == 4u;
    }
    else
    {
        _466 = maybe_YCoCg;
    }
    if (_466)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _485;
    if (maybe_SRGB)
    {
        _485 = (_368.size & 32768u) != 0u;
    }
    else
    {
        _485 = maybe_SRGB;
    }
    if (_485)
    {
        float3 param_1 = res.xyz;
        float3 _491 = srgb_to_rgb(param_1);
        float4 _2279 = res;
        _2279.x = _491.x;
        float4 _2281 = _2279;
        _2281.y = _491.y;
        float4 _2283 = _2281;
        _2283.z = _491.z;
        res = _2283;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod)
{
    return SampleBilinear(index, uvs, lod, false, false);
}

float lum(float3 color)
{
    return mad(0.072168998420238494873046875f, color.z, mad(0.21267099678516387939453125f, color.x, 0.71516001224517822265625f * color.y));
}

float3 IntersectSceneShadow(shadow_ray_t r, inout float dist)
{
    bool _1971 = false;
    float3 _1968;
    do
    {
        float3 ro = float3(r.o[0], r.o[1], r.o[2]);
        float3 _1345 = float3(r.d[0], r.d[1], r.d[2]);
        float3 rc = float3(r.c[0], r.c[1], r.c[2]);
        int depth = r.depth >> 24;
        float3 param = _1345;
        float3 _1362 = safe_invert(param);
        int _1422;
        int _2136;
        int _2137;
        float _2139;
        float _2140;
        while (dist > 9.9999997473787516355514526367188e-06f)
        {
            int _2135 = 0;
            float _2138 = dist;
            float3 param_1 = ro;
            float3 param_2 = _1345;
            float3 param_3 = _1362;
            uint param_4 = _1380_g_params.node_index;
            hit_data_t _2147 = { 0, _2136, _2137, dist, _2139, _2140 };
            hit_data_t param_5 = _2147;
            bool _1393 = Traverse_MacroTree_WithStack(param_1, param_2, param_3, param_4, param_5);
            _2135 = param_5.mask;
            _2136 = param_5.obj_index;
            _2137 = param_5.prim_index;
            _2138 = param_5.t;
            _2139 = param_5.u;
            _2140 = param_5.v;
            bool _1404;
            if (!_1393)
            {
                _1404 = depth > _1380_g_params.max_transp_depth;
            }
            else
            {
                _1404 = _1393;
            }
            if (_1404)
            {
                _1971 = true;
                _1968 = 0.0f.xxx;
                break;
            }
            if (_2135 == 0)
            {
                _1971 = true;
                _1968 = rc;
                break;
            }
            bool _1419 = param_5.prim_index < 0;
            if (_1419)
            {
                _1422 = (-1) - param_5.prim_index;
            }
            else
            {
                _1422 = param_5.prim_index;
            }
            uint _1433 = uint(_1422);
            uint _1475 = _997.Load(_1433 * 4 + 0) * 3u;
            vertex_t _1481;
            [unroll]
            for (int _28ident = 0; _28ident < 3; _28ident++)
            {
                _1481.p[_28ident] = asfloat(_1469.Load(_28ident * 4 + _1473.Load(_1475 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _29ident = 0; _29ident < 3; _29ident++)
            {
                _1481.n[_29ident] = asfloat(_1469.Load(_29ident * 4 + _1473.Load(_1475 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _30ident = 0; _30ident < 3; _30ident++)
            {
                _1481.b[_30ident] = asfloat(_1469.Load(_30ident * 4 + _1473.Load(_1475 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _31ident = 0; _31ident < 2; _31ident++)
            {
                [unroll]
                for (int _32ident = 0; _32ident < 2; _32ident++)
                {
                    _1481.t[_31ident][_32ident] = asfloat(_1469.Load(_32ident * 4 + _31ident * 8 + _1473.Load(_1475 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1529;
            [unroll]
            for (int _33ident = 0; _33ident < 3; _33ident++)
            {
                _1529.p[_33ident] = asfloat(_1469.Load(_33ident * 4 + _1473.Load((_1475 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _34ident = 0; _34ident < 3; _34ident++)
            {
                _1529.n[_34ident] = asfloat(_1469.Load(_34ident * 4 + _1473.Load((_1475 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _35ident = 0; _35ident < 3; _35ident++)
            {
                _1529.b[_35ident] = asfloat(_1469.Load(_35ident * 4 + _1473.Load((_1475 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _36ident = 0; _36ident < 2; _36ident++)
            {
                [unroll]
                for (int _37ident = 0; _37ident < 2; _37ident++)
                {
                    _1529.t[_36ident][_37ident] = asfloat(_1469.Load(_37ident * 4 + _36ident * 8 + _1473.Load((_1475 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1575;
            [unroll]
            for (int _38ident = 0; _38ident < 3; _38ident++)
            {
                _1575.p[_38ident] = asfloat(_1469.Load(_38ident * 4 + _1473.Load((_1475 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _39ident = 0; _39ident < 3; _39ident++)
            {
                _1575.n[_39ident] = asfloat(_1469.Load(_39ident * 4 + _1473.Load((_1475 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _1575.b[_40ident] = asfloat(_1469.Load(_40ident * 4 + _1473.Load((_1475 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 2; _41ident++)
            {
                [unroll]
                for (int _42ident = 0; _42ident < 2; _42ident++)
                {
                    _1575.t[_41ident][_42ident] = asfloat(_1469.Load(_42ident * 4 + _41ident * 8 + _1473.Load((_1475 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            float2 _1646 = ((float2(_1481.t[0][0], _1481.t[0][1]) * ((1.0f - param_5.u) - param_5.v)) + (float2(_1529.t[0][0], _1529.t[0][1]) * param_5.u)) + (float2(_1575.t[0][0], _1575.t[0][1]) * param_5.v);
            g_stack[gl_LocalInvocationIndex][0] = (_1419 ? (_1006.Load(_997.Load(_1433 * 4 + 0) * 4 + 0) & 65535u) : ((_1006.Load(_997.Load(_1433 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u;
            g_stack[gl_LocalInvocationIndex][1] = 1065353216u;
            int stack_size = 1;
            float3 throughput = 0.0f.xxx;
            for (;;)
            {
                int _1660 = stack_size;
                stack_size = _1660 - 1;
                if (_1660 != 0)
                {
                    int _1676 = stack_size;
                    int _1677 = 2 * _1676;
                    material_t _1683;
                    [unroll]
                    for (int _43ident = 0; _43ident < 5; _43ident++)
                    {
                        _1683.textures[_43ident] = _1674.Load(_43ident * 4 + g_stack[gl_LocalInvocationIndex][_1677] * 80 + 0);
                    }
                    [unroll]
                    for (int _44ident = 0; _44ident < 3; _44ident++)
                    {
                        _1683.base_color[_44ident] = asfloat(_1674.Load(_44ident * 4 + g_stack[gl_LocalInvocationIndex][_1677] * 80 + 20));
                    }
                    _1683.flags = _1674.Load(g_stack[gl_LocalInvocationIndex][_1677] * 80 + 32);
                    _1683.type = _1674.Load(g_stack[gl_LocalInvocationIndex][_1677] * 80 + 36);
                    _1683.tangent_rotation_or_strength = asfloat(_1674.Load(g_stack[gl_LocalInvocationIndex][_1677] * 80 + 40));
                    _1683.roughness_and_anisotropic = _1674.Load(g_stack[gl_LocalInvocationIndex][_1677] * 80 + 44);
                    _1683.int_ior = asfloat(_1674.Load(g_stack[gl_LocalInvocationIndex][_1677] * 80 + 48));
                    _1683.ext_ior = asfloat(_1674.Load(g_stack[gl_LocalInvocationIndex][_1677] * 80 + 52));
                    _1683.sheen_and_sheen_tint = _1674.Load(g_stack[gl_LocalInvocationIndex][_1677] * 80 + 56);
                    _1683.tint_and_metallic = _1674.Load(g_stack[gl_LocalInvocationIndex][_1677] * 80 + 60);
                    _1683.transmission_and_transmission_roughness = _1674.Load(g_stack[gl_LocalInvocationIndex][_1677] * 80 + 64);
                    _1683.specular_and_specular_tint = _1674.Load(g_stack[gl_LocalInvocationIndex][_1677] * 80 + 68);
                    _1683.clearcoat_and_clearcoat_roughness = _1674.Load(g_stack[gl_LocalInvocationIndex][_1677] * 80 + 72);
                    _1683.normal_map_strength_unorm = _1674.Load(g_stack[gl_LocalInvocationIndex][_1677] * 80 + 76);
                    uint _1735 = g_stack[gl_LocalInvocationIndex][_1677 + 1];
                    float _1736 = asfloat(_1735);
                    if (_1683.type == 4u)
                    {
                        float mix_val = _1683.tangent_rotation_or_strength;
                        if (_1683.textures[1] != 4294967295u)
                        {
                            mix_val *= SampleBilinear(_1683.textures[1], _1646, 0).x;
                        }
                        int _1760 = 2 * stack_size;
                        g_stack[gl_LocalInvocationIndex][_1760] = _1683.textures[3];
                        g_stack[gl_LocalInvocationIndex][_1760 + 1] = asuint(_1736 * (1.0f - mix_val));
                        int _1779 = 2 * (stack_size + 1);
                        g_stack[gl_LocalInvocationIndex][_1779] = _1683.textures[4];
                        g_stack[gl_LocalInvocationIndex][_1779 + 1] = asuint(_1736 * mix_val);
                        stack_size += 2;
                    }
                    else
                    {
                        if (_1683.type == 5u)
                        {
                            throughput += (float3(_1683.base_color[0], _1683.base_color[1], _1683.base_color[2]) * _1736);
                        }
                    }
                    continue;
                }
                else
                {
                    break;
                }
            }
            float3 _1813 = rc;
            float3 _1814 = _1813 * throughput;
            rc = _1814;
            if (lum(_1814) < 1.0000000116860974230803549289703e-07f)
            {
                break;
            }
            float _1824 = _2138 + 9.9999997473787516355514526367188e-06f;
            ro += (_1345 * _1824);
            dist -= _1824;
            depth++;
        }
        if (_1971)
        {
            break;
        }
        _1971 = true;
        _1968 = rc;
        break;
    } while(false);
    return _1968;
}

void comp_main()
{
    do
    {
        int _1847 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_1847) >= _1853.Load(12))
        {
            break;
        }
        shadow_ray_t _1872;
        [unroll]
        for (int _45ident = 0; _45ident < 3; _45ident++)
        {
            _1872.o[_45ident] = asfloat(_1868.Load(_45ident * 4 + _1847 * 48 + 0));
        }
        _1872.depth = int(_1868.Load(_1847 * 48 + 12));
        [unroll]
        for (int _46ident = 0; _46ident < 3; _46ident++)
        {
            _1872.d[_46ident] = asfloat(_1868.Load(_46ident * 4 + _1847 * 48 + 16));
        }
        _1872.dist = asfloat(_1868.Load(_1847 * 48 + 28));
        [unroll]
        for (int _47ident = 0; _47ident < 3; _47ident++)
        {
            _1872.c[_47ident] = asfloat(_1868.Load(_47ident * 4 + _1847 * 48 + 32));
        }
        _1872.xy = int(_1868.Load(_1847 * 48 + 44));
        float _2026[3] = { _1872.c[0], _1872.c[1], _1872.c[2] };
        float _2019[3] = { _1872.d[0], _1872.d[1], _1872.d[2] };
        float _2012[3] = { _1872.o[0], _1872.o[1], _1872.o[2] };
        shadow_ray_t _2005 = { _2012, _1872.depth, _2019, _1872.dist, _2026, _1872.xy };
        shadow_ray_t param = _2005;
        float param_1 = _1872.dist;
        float3 _1909 = IntersectSceneShadow(param, param_1);
        if (lum(_1909) > 0.0f)
        {
            int2 _1933 = int2((_1872.xy >> 16) & 65535, _1872.xy & 65535);
            g_out_img[_1933] = float4(g_out_img[_1933].xyz + _1909, 1.0f);
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
