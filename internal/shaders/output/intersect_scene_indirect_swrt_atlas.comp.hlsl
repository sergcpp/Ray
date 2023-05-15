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

ByteAddressBuffer _418 : register(t20, space0);
ByteAddressBuffer _693 : register(t3, space0);
ByteAddressBuffer _908 : register(t7, space0);
ByteAddressBuffer _1155 : register(t9, space0);
ByteAddressBuffer _1159 : register(t10, space0);
ByteAddressBuffer _1180 : register(t8, space0);
ByteAddressBuffer _1224 : register(t11, space0);
ByteAddressBuffer _1317 : register(t14, space0);
RWByteAddressBuffer _1333 : register(u12, space0);
ByteAddressBuffer _1495 : register(t4, space0);
ByteAddressBuffer _1545 : register(t5, space0);
ByteAddressBuffer _1581 : register(t6, space0);
ByteAddressBuffer _1698 : register(t1, space0);
ByteAddressBuffer _1702 : register(t2, space0);
ByteAddressBuffer _1880 : register(t15, space0);
RWByteAddressBuffer _2186 : register(u0, space0);
cbuffer UniformParams
{
    Params _1381_g_params : packoffset(c0);
};

Texture2DArray<float4> g_atlases[7] : register(t21, space0);
SamplerState _g_atlases_sampler[7] : register(s21, space0);

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
    bool _163 = v.x <= 1.0000000116860974230803549289703e-07f;
    bool _170;
    if (_163)
    {
        _170 = v.x >= 0.0f;
    }
    else
    {
        _170 = _163;
    }
    if (_170)
    {
        float3 _2512 = inv_v;
        _2512.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _2512;
    }
    else
    {
        bool _179 = v.x >= (-1.0000000116860974230803549289703e-07f);
        bool _185;
        if (_179)
        {
            _185 = v.x < 0.0f;
        }
        else
        {
            _185 = _179;
        }
        if (_185)
        {
            float3 _2510 = inv_v;
            _2510.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _2510;
        }
    }
    bool _193 = v.y <= 1.0000000116860974230803549289703e-07f;
    bool _199;
    if (_193)
    {
        _199 = v.y >= 0.0f;
    }
    else
    {
        _199 = _193;
    }
    if (_199)
    {
        float3 _2516 = inv_v;
        _2516.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _2516;
    }
    else
    {
        bool _206 = v.y >= (-1.0000000116860974230803549289703e-07f);
        bool _212;
        if (_206)
        {
            _212 = v.y < 0.0f;
        }
        else
        {
            _212 = _206;
        }
        if (_212)
        {
            float3 _2514 = inv_v;
            _2514.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _2514;
        }
    }
    bool _219 = v.z <= 1.0000000116860974230803549289703e-07f;
    bool _225;
    if (_219)
    {
        _225 = v.z >= 0.0f;
    }
    else
    {
        _225 = _219;
    }
    if (_225)
    {
        float3 _2520 = inv_v;
        _2520.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _2520;
    }
    else
    {
        bool _232 = v.z >= (-1.0000000116860974230803549289703e-07f);
        bool _238;
        if (_232)
        {
            _238 = v.z < 0.0f;
        }
        else
        {
            _238 = _232;
        }
        if (_238)
        {
            float3 _2518 = inv_v;
            _2518.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _2518;
        }
    }
    return inv_v;
}

int hash(int x)
{
    uint _120 = uint(x);
    uint _127 = ((_120 >> uint(16)) ^ _120) * 73244475u;
    uint _132 = ((_127 >> uint(16)) ^ _127) * 73244475u;
    return int((_132 >> uint(16)) ^ _132);
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

bool _bbox_test_fma(float3 inv_d, float3 neg_inv_d_o, float t, float3 bbox_min, float3 bbox_max)
{
    float _786 = mad(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float _794 = mad(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float _809 = mad(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    float _816 = mad(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    float _833 = mad(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    float _840 = mad(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    float _845 = max(max(min(_786, _794), min(_809, _816)), min(_833, _840));
    float _853 = min(min(max(_786, _794), max(_809, _816)), max(_833, _840)) * 1.0000002384185791015625f;
    return ((_845 <= _853) && (_845 <= t)) && (_853 > 0.0f);
}

void IntersectTri(float3 ro, float3 rd, tri_accel_t tri, uint prim_index, inout hit_data_t inter)
{
    do
    {
        float _566 = dot(rd, tri.n_plane.xyz);
        float _575 = tri.n_plane.w - dot(ro, tri.n_plane.xyz);
        if (sign(_575) != sign(mad(_566, inter.t, -_575)))
        {
            break;
        }
        float3 _596 = (ro * _566) + (rd * _575);
        float _607 = mad(_566, tri.u_plane.w, dot(_596, tri.u_plane.xyz));
        float _612 = _566 - _607;
        if (sign(_607) != sign(_612))
        {
            break;
        }
        float _628 = mad(_566, tri.v_plane.w, dot(_596, tri.v_plane.xyz));
        if (sign(_628) != sign(_612 - _628))
        {
            break;
        }
        float _643 = 1.0f / _566;
        inter.mask = -1;
        int _648;
        if (_566 < 0.0f)
        {
            _648 = int(prim_index);
        }
        else
        {
            _648 = (-1) - int(prim_index);
        }
        inter.prim_index = _648;
        inter.t = _575 * _643;
        inter.u = _607 * _643;
        inter.v = _628 * _643;
        break;
    } while(false);
}

void IntersectTris_ClosestHit(float3 ro, float3 rd, int tri_start, int tri_end, int obj_index, inout hit_data_t out_inter)
{
    int _2444 = 0;
    int _2445 = obj_index;
    float _2447 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _2446;
    float _2448;
    float _2449;
    for (int i = tri_start; i < tri_end; )
    {
        param = ro;
        param_1 = rd;
        tri_accel_t _704;
        _704.n_plane = asfloat(_693.Load4(i * 48 + 0));
        _704.u_plane = asfloat(_693.Load4(i * 48 + 16));
        _704.v_plane = asfloat(_693.Load4(i * 48 + 32));
        param_2.n_plane = _704.n_plane;
        param_2.u_plane = _704.u_plane;
        param_2.v_plane = _704.v_plane;
        param_3 = uint(i);
        hit_data_t _2456 = { _2444, _2445, _2446, _2447, _2448, _2449 };
        param_4 = _2456;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _2444 = param_4.mask;
        _2445 = param_4.obj_index;
        _2446 = param_4.prim_index;
        _2447 = param_4.t;
        _2448 = param_4.u;
        _2449 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _2444;
    int _727;
    if (_2444 != 0)
    {
        _727 = _2445;
    }
    else
    {
        _727 = out_inter.obj_index;
    }
    out_inter.obj_index = _727;
    int _740;
    if (_2444 != 0)
    {
        _740 = _2446;
    }
    else
    {
        _740 = out_inter.prim_index;
    }
    out_inter.prim_index = _740;
    out_inter.t = _2447;
    float _756;
    if (_2444 != 0)
    {
        _756 = _2448;
    }
    else
    {
        _756 = out_inter.u;
    }
    out_inter.u = _756;
    float _769;
    if (_2444 != 0)
    {
        _769 = _2449;
    }
    else
    {
        _769 = out_inter.v;
    }
    out_inter.v = _769;
}

void Traverse_BLAS_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    float3 _870 = (-inv_d) * ro;
    uint _872 = stack_size;
    uint _882 = stack_size;
    stack_size = _882 + uint(1);
    g_stack[gl_LocalInvocationIndex][_882] = node_index;
    uint _956;
    uint _980;
    while (stack_size != _872)
    {
        uint _897 = stack_size;
        uint _898 = _897 - uint(1);
        stack_size = _898;
        bvh_node_t _912;
        _912.bbox_min = asfloat(_908.Load4(g_stack[gl_LocalInvocationIndex][_898] * 32 + 0));
        _912.bbox_max = asfloat(_908.Load4(g_stack[gl_LocalInvocationIndex][_898] * 32 + 16));
        float3 param = inv_d;
        float3 param_1 = _870;
        float param_2 = inter.t;
        float3 param_3 = _912.bbox_min.xyz;
        float3 param_4 = _912.bbox_max.xyz;
        if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
        {
            continue;
        }
        uint _939 = asuint(_912.bbox_min.w);
        if ((_939 & 2147483648u) == 0u)
        {
            uint _946 = stack_size;
            stack_size = _946 + uint(1);
            uint _950 = asuint(_912.bbox_max.w);
            uint _952 = _950 >> uint(30);
            if (rd[_952] < 0.0f)
            {
                _956 = _939;
            }
            else
            {
                _956 = _950 & 1073741823u;
            }
            g_stack[gl_LocalInvocationIndex][_946] = _956;
            uint _971 = stack_size;
            stack_size = _971 + uint(1);
            if (rd[_952] < 0.0f)
            {
                _980 = _950 & 1073741823u;
            }
            else
            {
                _980 = _939;
            }
            g_stack[gl_LocalInvocationIndex][_971] = _980;
        }
        else
        {
            int _1000 = int(_939 & 2147483647u);
            float3 param_5 = ro;
            float3 param_6 = rd;
            int param_7 = _1000;
            int param_8 = _1000 + asint(_912.bbox_max.w);
            int param_9 = obj_index;
            hit_data_t param_10 = inter;
            IntersectTris_ClosestHit(param_5, param_6, param_7, param_8, param_9, param_10);
            inter = param_10;
        }
    }
}

void Traverse_TLAS_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    float3 _1025 = (-orig_inv_rd) * orig_ro;
    uint stack_size = 1u;
    g_stack[gl_LocalInvocationIndex][0u] = node_index;
    uint _1090;
    uint _1113;
    while (stack_size != 0u)
    {
        uint _1041 = stack_size;
        uint _1042 = _1041 - uint(1);
        stack_size = _1042;
        bvh_node_t _1048;
        _1048.bbox_min = asfloat(_908.Load4(g_stack[gl_LocalInvocationIndex][_1042] * 32 + 0));
        _1048.bbox_max = asfloat(_908.Load4(g_stack[gl_LocalInvocationIndex][_1042] * 32 + 16));
        float3 param = orig_inv_rd;
        float3 param_1 = _1025;
        float param_2 = inter.t;
        float3 param_3 = _1048.bbox_min.xyz;
        float3 param_4 = _1048.bbox_max.xyz;
        if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
        {
            continue;
        }
        uint _1075 = asuint(_1048.bbox_min.w);
        if ((_1075 & 2147483648u) == 0u)
        {
            uint _1081 = stack_size;
            stack_size = _1081 + uint(1);
            uint _1085 = asuint(_1048.bbox_max.w);
            uint _1086 = _1085 >> uint(30);
            if (orig_rd[_1086] < 0.0f)
            {
                _1090 = _1075;
            }
            else
            {
                _1090 = _1085 & 1073741823u;
            }
            g_stack[gl_LocalInvocationIndex][_1081] = _1090;
            uint _1104 = stack_size;
            stack_size = _1104 + uint(1);
            if (orig_rd[_1086] < 0.0f)
            {
                _1113 = _1085 & 1073741823u;
            }
            else
            {
                _1113 = _1075;
            }
            g_stack[gl_LocalInvocationIndex][_1104] = _1113;
        }
        else
        {
            uint _1131 = _1075 & 2147483647u;
            uint _1135 = asuint(_1048.bbox_max.w);
            for (uint i = _1131; i < (_1131 + _1135); i++)
            {
                mesh_instance_t _1166;
                _1166.bbox_min = asfloat(_1155.Load4(_1159.Load(i * 4 + 0) * 32 + 0));
                _1166.bbox_max = asfloat(_1155.Load4(_1159.Load(i * 4 + 0) * 32 + 16));
                mesh_t _1186;
                [unroll]
                for (int _31ident = 0; _31ident < 3; _31ident++)
                {
                    _1186.bbox_min[_31ident] = asfloat(_1180.Load(_31ident * 4 + asuint(_1166.bbox_max.w) * 48 + 0));
                }
                [unroll]
                for (int _32ident = 0; _32ident < 3; _32ident++)
                {
                    _1186.bbox_max[_32ident] = asfloat(_1180.Load(_32ident * 4 + asuint(_1166.bbox_max.w) * 48 + 12));
                }
                _1186.node_index = _1180.Load(asuint(_1166.bbox_max.w) * 48 + 24);
                _1186.node_count = _1180.Load(asuint(_1166.bbox_max.w) * 48 + 28);
                _1186.tris_index = _1180.Load(asuint(_1166.bbox_max.w) * 48 + 32);
                _1186.tris_count = _1180.Load(asuint(_1166.bbox_max.w) * 48 + 36);
                _1186.vert_index = _1180.Load(asuint(_1166.bbox_max.w) * 48 + 40);
                _1186.vert_count = _1180.Load(asuint(_1166.bbox_max.w) * 48 + 44);
                transform_t _1230;
                _1230.xform = asfloat(uint4x4(_1224.Load4(asuint(_1166.bbox_min.w) * 128 + 0), _1224.Load4(asuint(_1166.bbox_min.w) * 128 + 16), _1224.Load4(asuint(_1166.bbox_min.w) * 128 + 32), _1224.Load4(asuint(_1166.bbox_min.w) * 128 + 48)));
                _1230.inv_xform = asfloat(uint4x4(_1224.Load4(asuint(_1166.bbox_min.w) * 128 + 64), _1224.Load4(asuint(_1166.bbox_min.w) * 128 + 80), _1224.Load4(asuint(_1166.bbox_min.w) * 128 + 96), _1224.Load4(asuint(_1166.bbox_min.w) * 128 + 112)));
                float3 param_5 = orig_inv_rd;
                float3 param_6 = _1025;
                float param_7 = inter.t;
                float3 param_8 = _1166.bbox_min.xyz;
                float3 param_9 = _1166.bbox_max.xyz;
                if (!_bbox_test_fma(param_5, param_6, param_7, param_8, param_9))
                {
                    continue;
                }
                float3 _1275 = mul(float4(orig_rd, 0.0f), _1230.inv_xform).xyz;
                float3 param_10 = _1275;
                float3 param_11 = mul(float4(orig_ro, 1.0f), _1230.inv_xform).xyz;
                float3 param_12 = _1275;
                float3 param_13 = safe_invert(param_10);
                int param_14 = int(_1159.Load(i * 4 + 0));
                uint param_15 = _1186.node_index;
                uint param_16 = stack_size;
                hit_data_t param_17 = inter;
                Traverse_BLAS_WithStack(param_11, param_12, param_13, param_14, param_15, param_16, param_17);
                inter = param_17;
            }
        }
    }
}

float2 TransformUV(float2 _uv, atlas_texture_t t, int mip_level)
{
    uint _2399[14] = t.pos;
    uint _2402[14] = t.pos;
    uint _376 = t.size & 16383u;
    uint _379 = t.size >> uint(16);
    uint _380 = _379 & 16383u;
    float2 size = float2(float(_376), float(_380));
    if ((_379 & 32768u) != 0u)
    {
        size = float2(float(_376 >> uint(mip_level)), float(_380 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_2399[mip_level] & 65535u), float((_2402[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _293 = mad(col.z, 31.875f, 1.0f);
    float _303 = (col.x - 0.501960813999176025390625f) / _293;
    float _309 = (col.y - 0.501960813999176025390625f) / _293;
    return float3((col.w + _303) - _309, col.w + _309, (col.w - _303) - _309);
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

float4 SampleBilinear(uint index, float2 uvs, int lod, float2 rand, bool maybe_YCoCg, bool maybe_SRGB)
{
    atlas_texture_t _421;
    _421.size = _418.Load(index * 80 + 0);
    _421.atlas = _418.Load(index * 80 + 4);
    [unroll]
    for (int _33ident = 0; _33ident < 4; _33ident++)
    {
        _421.page[_33ident] = _418.Load(_33ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _34ident = 0; _34ident < 14; _34ident++)
    {
        _421.pos[_34ident] = _418.Load(_34ident * 4 + index * 80 + 24);
    }
    uint _2407[4];
    _2407[0] = _421.page[0];
    _2407[1] = _421.page[1];
    _2407[2] = _421.page[2];
    _2407[3] = _421.page[3];
    uint _2443[14] = { _421.pos[0], _421.pos[1], _421.pos[2], _421.pos[3], _421.pos[4], _421.pos[5], _421.pos[6], _421.pos[7], _421.pos[8], _421.pos[9], _421.pos[10], _421.pos[11], _421.pos[12], _421.pos[13] };
    atlas_texture_t _2413 = { _421.size, _421.atlas, _2407, _2443 };
    uint _507 = _421.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_507)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_507)], float3(((TransformUV(uvs, _2413, lod) + rand) - 0.5f.xx) * 0.000118371215648949146270751953125f.xx, float((_2407[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
    bool _522;
    if (maybe_YCoCg)
    {
        _522 = _421.atlas == 4u;
    }
    else
    {
        _522 = maybe_YCoCg;
    }
    if (_522)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _541;
    if (maybe_SRGB)
    {
        _541 = (_421.size & 32768u) != 0u;
    }
    else
    {
        _541 = maybe_SRGB;
    }
    if (_541)
    {
        float3 param_1 = res.xyz;
        float3 _547 = srgb_to_rgb(param_1);
        float4 _2536 = res;
        _2536.x = _547.x;
        float4 _2538 = _2536;
        _2538.y = _547.y;
        float4 _2540 = _2538;
        _2540.z = _547.z;
        res = _2540;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod, float2 rand)
{
    return SampleBilinear(index, uvs, lod, rand, false, false);
}

void comp_main()
{
    do
    {
        int _1311 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_1311) >= _1317.Load(4))
        {
            break;
        }
        float3 ro = float3(asfloat(_1333.Load(_1311 * 72 + 0)), asfloat(_1333.Load(_1311 * 72 + 4)), asfloat(_1333.Load(_1311 * 72 + 8)));
        float _1361 = asfloat(_1333.Load(_1311 * 72 + 12));
        float _1364 = asfloat(_1333.Load(_1311 * 72 + 16));
        float _1367 = asfloat(_1333.Load(_1311 * 72 + 20));
        float3 _1368 = float3(_1361, _1364, _1367);
        float3 param = _1368;
        float3 _1372 = safe_invert(param);
        int _2225 = 0;
        int _2227 = 0;
        int _2226 = 0;
        float _2228 = _1381_g_params.inter_t;
        float _2230 = 0.0f;
        float _2229 = 0.0f;
        uint param_1 = uint(hash(int(_1333.Load(_1311 * 72 + 64))));
        float _1395 = construct_float(param_1);
        uint param_2 = uint(hash(hash(int(_1333.Load(_1311 * 72 + 64)))));
        float _1403 = construct_float(param_2);
        ray_data_t _1412;
        [unroll]
        for (int _35ident = 0; _35ident < 3; _35ident++)
        {
            _1412.o[_35ident] = asfloat(_1333.Load(_35ident * 4 + _1311 * 72 + 0));
        }
        [unroll]
        for (int _36ident = 0; _36ident < 3; _36ident++)
        {
            _1412.d[_36ident] = asfloat(_1333.Load(_36ident * 4 + _1311 * 72 + 12));
        }
        _1412.pdf = asfloat(_1333.Load(_1311 * 72 + 24));
        [unroll]
        for (int _37ident = 0; _37ident < 3; _37ident++)
        {
            _1412.c[_37ident] = asfloat(_1333.Load(_37ident * 4 + _1311 * 72 + 28));
        }
        [unroll]
        for (int _38ident = 0; _38ident < 4; _38ident++)
        {
            _1412.ior[_38ident] = asfloat(_1333.Load(_38ident * 4 + _1311 * 72 + 40));
        }
        _1412.cone_width = asfloat(_1333.Load(_1311 * 72 + 56));
        _1412.cone_spread = asfloat(_1333.Load(_1311 * 72 + 60));
        _1412.xy = int(_1333.Load(_1311 * 72 + 64));
        _1412.depth = int(_1333.Load(_1311 * 72 + 68));
        float _2337[4] = { _1412.ior[0], _1412.ior[1], _1412.ior[2], _1412.ior[3] };
        float _2328[3] = { _1412.c[0], _1412.c[1], _1412.c[2] };
        float _2321[3] = { _1412.d[0], _1412.d[1], _1412.d[2] };
        float _2314[3] = { _1412.o[0], _1412.o[1], _1412.o[2] };
        ray_data_t _2269 = { _2314, _2321, _1412.pdf, _2328, _2337, _1412.cone_width, _1412.cone_spread, _1412.xy, _1412.depth };
        int rand_index = _1381_g_params.hi + (total_depth(_2269) * 9);
        int _1526;
        float _2076;
        for (;;)
        {
            float _1472 = _2228;
            float3 param_3 = ro;
            float3 param_4 = _1368;
            float3 param_5 = _1372;
            uint param_6 = _1381_g_params.node_index;
            hit_data_t _2237 = { _2225, _2226, _2227, _1472, _2229, _2230 };
            hit_data_t param_7 = _2237;
            Traverse_TLAS_WithStack(param_3, param_4, param_5, param_6, param_7);
            _2225 = param_7.mask;
            _2226 = param_7.obj_index;
            _2227 = param_7.prim_index;
            _2228 = param_7.t;
            _2229 = param_7.u;
            _2230 = param_7.v;
            if (param_7.prim_index < 0)
            {
                _2227 = (-1) - int(_1495.Load(((-1) - _2227) * 4 + 0));
            }
            else
            {
                _2227 = int(_1495.Load(_2227 * 4 + 0));
            }
            if (_2225 == 0)
            {
                break;
            }
            bool _1523 = _2227 < 0;
            if (_1523)
            {
                _1526 = (-1) - _2227;
            }
            else
            {
                _1526 = _2227;
            }
            uint _1537 = uint(_1526);
            bool _1539 = !_1523;
            bool _1553;
            if (_1539)
            {
                _1553 = ((_1545.Load(_1537 * 4 + 0) >> 16u) & 32768u) != 0u;
            }
            else
            {
                _1553 = _1539;
            }
            bool _1566;
            if (!_1553)
            {
                bool _1565;
                if (_1523)
                {
                    _1565 = (_1545.Load(_1537 * 4 + 0) & 32768u) != 0u;
                }
                else
                {
                    _1565 = _1523;
                }
                _1566 = _1565;
            }
            else
            {
                _1566 = _1553;
            }
            if (_1566)
            {
                break;
            }
            material_t _1589;
            [unroll]
            for (int _39ident = 0; _39ident < 5; _39ident++)
            {
                _1589.textures[_39ident] = _1581.Load(_39ident * 4 + ((_1545.Load(_1537 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _1589.base_color[_40ident] = asfloat(_1581.Load(_40ident * 4 + ((_1545.Load(_1537 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
            }
            _1589.flags = _1581.Load(((_1545.Load(_1537 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
            _1589.type = _1581.Load(((_1545.Load(_1537 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
            _1589.tangent_rotation_or_strength = asfloat(_1581.Load(((_1545.Load(_1537 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
            _1589.roughness_and_anisotropic = _1581.Load(((_1545.Load(_1537 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
            _1589.ior = asfloat(_1581.Load(((_1545.Load(_1537 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
            _1589.sheen_and_sheen_tint = _1581.Load(((_1545.Load(_1537 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
            _1589.tint_and_metallic = _1581.Load(((_1545.Load(_1537 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
            _1589.transmission_and_transmission_roughness = _1581.Load(((_1545.Load(_1537 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
            _1589.specular_and_specular_tint = _1581.Load(((_1545.Load(_1537 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
            _1589.clearcoat_and_clearcoat_roughness = _1581.Load(((_1545.Load(_1537 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
            _1589.normal_map_strength_unorm = _1581.Load(((_1545.Load(_1537 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
            uint _2339 = _1589.textures[1];
            uint _2341 = _1589.textures[3];
            uint _2342 = _1589.textures[4];
            float _2343 = _1589.base_color[0];
            float _2344 = _1589.base_color[1];
            float _2345 = _1589.base_color[2];
            uint _2273 = _1589.type;
            float _2274 = _1589.tangent_rotation_or_strength;
            if (_1523)
            {
                material_t _1641;
                [unroll]
                for (int _41ident = 0; _41ident < 5; _41ident++)
                {
                    _1641.textures[_41ident] = _1581.Load(_41ident * 4 + (_1545.Load(_1537 * 4 + 0) & 16383u) * 76 + 0);
                }
                [unroll]
                for (int _42ident = 0; _42ident < 3; _42ident++)
                {
                    _1641.base_color[_42ident] = asfloat(_1581.Load(_42ident * 4 + (_1545.Load(_1537 * 4 + 0) & 16383u) * 76 + 20));
                }
                _1641.flags = _1581.Load((_1545.Load(_1537 * 4 + 0) & 16383u) * 76 + 32);
                _1641.type = _1581.Load((_1545.Load(_1537 * 4 + 0) & 16383u) * 76 + 36);
                _1641.tangent_rotation_or_strength = asfloat(_1581.Load((_1545.Load(_1537 * 4 + 0) & 16383u) * 76 + 40));
                _1641.roughness_and_anisotropic = _1581.Load((_1545.Load(_1537 * 4 + 0) & 16383u) * 76 + 44);
                _1641.ior = asfloat(_1581.Load((_1545.Load(_1537 * 4 + 0) & 16383u) * 76 + 48));
                _1641.sheen_and_sheen_tint = _1581.Load((_1545.Load(_1537 * 4 + 0) & 16383u) * 76 + 52);
                _1641.tint_and_metallic = _1581.Load((_1545.Load(_1537 * 4 + 0) & 16383u) * 76 + 56);
                _1641.transmission_and_transmission_roughness = _1581.Load((_1545.Load(_1537 * 4 + 0) & 16383u) * 76 + 60);
                _1641.specular_and_specular_tint = _1581.Load((_1545.Load(_1537 * 4 + 0) & 16383u) * 76 + 64);
                _1641.clearcoat_and_clearcoat_roughness = _1581.Load((_1545.Load(_1537 * 4 + 0) & 16383u) * 76 + 68);
                _1641.normal_map_strength_unorm = _1581.Load((_1545.Load(_1537 * 4 + 0) & 16383u) * 76 + 72);
                _2339 = _1641.textures[1];
                _2341 = _1641.textures[3];
                _2342 = _1641.textures[4];
                _2343 = _1641.base_color[0];
                _2344 = _1641.base_color[1];
                _2345 = _1641.base_color[2];
                _2273 = _1641.type;
                _2274 = _1641.tangent_rotation_or_strength;
            }
            uint _1704 = _1537 * 3u;
            vertex_t _1710;
            [unroll]
            for (int _43ident = 0; _43ident < 3; _43ident++)
            {
                _1710.p[_43ident] = asfloat(_1698.Load(_43ident * 4 + _1702.Load(_1704 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _44ident = 0; _44ident < 3; _44ident++)
            {
                _1710.n[_44ident] = asfloat(_1698.Load(_44ident * 4 + _1702.Load(_1704 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _45ident = 0; _45ident < 3; _45ident++)
            {
                _1710.b[_45ident] = asfloat(_1698.Load(_45ident * 4 + _1702.Load(_1704 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _46ident = 0; _46ident < 2; _46ident++)
            {
                [unroll]
                for (int _47ident = 0; _47ident < 2; _47ident++)
                {
                    _1710.t[_46ident][_47ident] = asfloat(_1698.Load(_47ident * 4 + _46ident * 8 + _1702.Load(_1704 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1758;
            [unroll]
            for (int _48ident = 0; _48ident < 3; _48ident++)
            {
                _1758.p[_48ident] = asfloat(_1698.Load(_48ident * 4 + _1702.Load((_1704 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _49ident = 0; _49ident < 3; _49ident++)
            {
                _1758.n[_49ident] = asfloat(_1698.Load(_49ident * 4 + _1702.Load((_1704 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _50ident = 0; _50ident < 3; _50ident++)
            {
                _1758.b[_50ident] = asfloat(_1698.Load(_50ident * 4 + _1702.Load((_1704 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _51ident = 0; _51ident < 2; _51ident++)
            {
                [unroll]
                for (int _52ident = 0; _52ident < 2; _52ident++)
                {
                    _1758.t[_51ident][_52ident] = asfloat(_1698.Load(_52ident * 4 + _51ident * 8 + _1702.Load((_1704 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1804;
            [unroll]
            for (int _53ident = 0; _53ident < 3; _53ident++)
            {
                _1804.p[_53ident] = asfloat(_1698.Load(_53ident * 4 + _1702.Load((_1704 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _54ident = 0; _54ident < 3; _54ident++)
            {
                _1804.n[_54ident] = asfloat(_1698.Load(_54ident * 4 + _1702.Load((_1704 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _55ident = 0; _55ident < 3; _55ident++)
            {
                _1804.b[_55ident] = asfloat(_1698.Load(_55ident * 4 + _1702.Load((_1704 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _56ident = 0; _56ident < 2; _56ident++)
            {
                [unroll]
                for (int _57ident = 0; _57ident < 2; _57ident++)
                {
                    _1804.t[_56ident][_57ident] = asfloat(_1698.Load(_57ident * 4 + _56ident * 8 + _1702.Load((_1704 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            float2 _1875 = ((float2(_1710.t[0][0], _1710.t[0][1]) * ((1.0f - _2229) - _2230)) + (float2(_1758.t[0][0], _1758.t[0][1]) * _2229)) + (float2(_1804.t[0][0], _1804.t[0][1]) * _2230);
            float trans_r = frac(asfloat(_1880.Load(rand_index * 4 + 0)) + _1395);
            float2 _1906 = float2(frac(asfloat(_1880.Load((rand_index + 7) * 4 + 0)) + _1395), frac(asfloat(_1880.Load((rand_index + 8) * 4 + 0)) + _1403));
            while (_2273 == 4u)
            {
                float mix_val = _2274;
                if (_2339 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_2339, _1875, 0, _1906).x;
                }
                if (trans_r > mix_val)
                {
                    material_t _1940;
                    [unroll]
                    for (int _58ident = 0; _58ident < 5; _58ident++)
                    {
                        _1940.textures[_58ident] = _1581.Load(_58ident * 4 + _2341 * 76 + 0);
                    }
                    [unroll]
                    for (int _59ident = 0; _59ident < 3; _59ident++)
                    {
                        _1940.base_color[_59ident] = asfloat(_1581.Load(_59ident * 4 + _2341 * 76 + 20));
                    }
                    _1940.flags = _1581.Load(_2341 * 76 + 32);
                    _1940.type = _1581.Load(_2341 * 76 + 36);
                    _1940.tangent_rotation_or_strength = asfloat(_1581.Load(_2341 * 76 + 40));
                    _1940.roughness_and_anisotropic = _1581.Load(_2341 * 76 + 44);
                    _1940.ior = asfloat(_1581.Load(_2341 * 76 + 48));
                    _1940.sheen_and_sheen_tint = _1581.Load(_2341 * 76 + 52);
                    _1940.tint_and_metallic = _1581.Load(_2341 * 76 + 56);
                    _1940.transmission_and_transmission_roughness = _1581.Load(_2341 * 76 + 60);
                    _1940.specular_and_specular_tint = _1581.Load(_2341 * 76 + 64);
                    _1940.clearcoat_and_clearcoat_roughness = _1581.Load(_2341 * 76 + 68);
                    _1940.normal_map_strength_unorm = _1581.Load(_2341 * 76 + 72);
                    _2339 = _1940.textures[1];
                    _2341 = _1940.textures[3];
                    _2342 = _1940.textures[4];
                    _2343 = _1940.base_color[0];
                    _2344 = _1940.base_color[1];
                    _2345 = _1940.base_color[2];
                    _2273 = _1940.type;
                    _2274 = _1940.tangent_rotation_or_strength;
                    trans_r = (trans_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _1993;
                    [unroll]
                    for (int _60ident = 0; _60ident < 5; _60ident++)
                    {
                        _1993.textures[_60ident] = _1581.Load(_60ident * 4 + _2342 * 76 + 0);
                    }
                    [unroll]
                    for (int _61ident = 0; _61ident < 3; _61ident++)
                    {
                        _1993.base_color[_61ident] = asfloat(_1581.Load(_61ident * 4 + _2342 * 76 + 20));
                    }
                    _1993.flags = _1581.Load(_2342 * 76 + 32);
                    _1993.type = _1581.Load(_2342 * 76 + 36);
                    _1993.tangent_rotation_or_strength = asfloat(_1581.Load(_2342 * 76 + 40));
                    _1993.roughness_and_anisotropic = _1581.Load(_2342 * 76 + 44);
                    _1993.ior = asfloat(_1581.Load(_2342 * 76 + 48));
                    _1993.sheen_and_sheen_tint = _1581.Load(_2342 * 76 + 52);
                    _1993.tint_and_metallic = _1581.Load(_2342 * 76 + 56);
                    _1993.transmission_and_transmission_roughness = _1581.Load(_2342 * 76 + 60);
                    _1993.specular_and_specular_tint = _1581.Load(_2342 * 76 + 64);
                    _1993.clearcoat_and_clearcoat_roughness = _1581.Load(_2342 * 76 + 68);
                    _1993.normal_map_strength_unorm = _1581.Load(_2342 * 76 + 72);
                    _2339 = _1993.textures[1];
                    _2341 = _1993.textures[3];
                    _2342 = _1993.textures[4];
                    _2343 = _1993.base_color[0];
                    _2344 = _1993.base_color[1];
                    _2345 = _1993.base_color[2];
                    _2273 = _1993.type;
                    _2274 = _1993.tangent_rotation_or_strength;
                    trans_r /= mix_val;
                }
            }
            if (_2273 != 5u)
            {
                break;
            }
            float _2064 = max(asfloat(_1333.Load(_1311 * 72 + 28)), max(asfloat(_1333.Load(_1311 * 72 + 32)), asfloat(_1333.Load(_1311 * 72 + 36))));
            if ((int(_1333.Load(_1311 * 72 + 68)) >> 24) > _1381_g_params.min_transp_depth)
            {
                _2076 = max(0.0500000007450580596923828125f, 1.0f - _2064);
            }
            else
            {
                _2076 = 0.0f;
            }
            bool _2090 = (frac(asfloat(_1880.Load((rand_index + 6) * 4 + 0)) + _1395) < _2076) || (_2064 == 0.0f);
            bool _2102;
            if (!_2090)
            {
                _2102 = ((int(_1333.Load(_1311 * 72 + 68)) >> 24) + 1) >= _1381_g_params.max_transp_depth;
            }
            else
            {
                _2102 = _2090;
            }
            if (_2102)
            {
                _1333.Store(_1311 * 72 + 36, asuint(0.0f));
                _1333.Store(_1311 * 72 + 32, asuint(0.0f));
                _1333.Store(_1311 * 72 + 28, asuint(0.0f));
                break;
            }
            float _2116 = 1.0f - _2076;
            _1333.Store(_1311 * 72 + 28, asuint(asfloat(_1333.Load(_1311 * 72 + 28)) * (_2343 / _2116)));
            _1333.Store(_1311 * 72 + 32, asuint(asfloat(_1333.Load(_1311 * 72 + 32)) * (_2344 / _2116)));
            _1333.Store(_1311 * 72 + 36, asuint(asfloat(_1333.Load(_1311 * 72 + 36)) * (_2345 / _2116)));
            ro += (_1368 * (_2228 + 9.9999997473787516355514526367188e-06f));
            _2225 = 0;
            _2228 = _1472 - _2228;
            _1333.Store(_1311 * 72 + 68, uint(int(_1333.Load(_1311 * 72 + 68)) + 16777216));
            rand_index += 9;
            continue;
        }
        float _2168 = asfloat(_1333.Load(_1311 * 72 + 0));
        float _2171 = asfloat(_1333.Load(_1311 * 72 + 4));
        float _2174 = asfloat(_1333.Load(_1311 * 72 + 8));
        float _2179 = _2228;
        float _2180 = _2179 + distance(float3(_2168, _2171, _2174), ro);
        _2228 = _2180;
        _2186.Store(_1311 * 24 + 0, uint(_2225));
        _2186.Store(_1311 * 24 + 4, uint(_2226));
        _2186.Store(_1311 * 24 + 8, uint(_2227));
        _2186.Store(_1311 * 24 + 12, asuint(_2180));
        _2186.Store(_1311 * 24 + 16, asuint(_2229));
        _2186.Store(_1311 * 24 + 20, asuint(_2230));
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
