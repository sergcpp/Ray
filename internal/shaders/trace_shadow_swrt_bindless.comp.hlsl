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
    float halton;
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

ByteAddressBuffer _510 : register(t1, space0);
ByteAddressBuffer _730 : register(t5, space0);
ByteAddressBuffer _870 : register(t2, space0);
ByteAddressBuffer _879 : register(t3, space0);
ByteAddressBuffer _1052 : register(t7, space0);
ByteAddressBuffer _1056 : register(t8, space0);
ByteAddressBuffer _1077 : register(t6, space0);
ByteAddressBuffer _1123 : register(t9, space0);
ByteAddressBuffer _1305 : register(t4, space0);
ByteAddressBuffer _1388 : register(t10, space0);
ByteAddressBuffer _1392 : register(t11, space0);
ByteAddressBuffer _1751 : register(t13, space0);
ByteAddressBuffer _1768 : register(t12, space0);
cbuffer UniformParams
{
    Params _1229_g_params : packoffset(c0);
};

Texture2D<float4> g_textures[] : register(t0, space1);
SamplerState _g_textures_sampler[] : register(s0, space1);
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
    uint _107 = uint(x);
    uint _114 = ((_107 >> uint(16)) ^ _107) * 73244475u;
    uint _119 = ((_114 >> uint(16)) ^ _114) * 73244475u;
    return int((_119 >> uint(16)) ^ _119);
}

float3 safe_invert(float3 v)
{
    float3 inv_v = 1.0f.xxx / v;
    bool _150 = v.x <= 1.0000000116860974230803549289703e-07f;
    bool _157;
    if (_150)
    {
        _157 = v.x >= 0.0f;
    }
    else
    {
        _157 = _150;
    }
    if (_157)
    {
        float3 _2123 = inv_v;
        _2123.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _2123;
    }
    else
    {
        bool _166 = v.x >= (-1.0000000116860974230803549289703e-07f);
        bool _172;
        if (_166)
        {
            _172 = v.x < 0.0f;
        }
        else
        {
            _172 = _166;
        }
        if (_172)
        {
            float3 _2125 = inv_v;
            _2125.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _2125;
        }
    }
    bool _180 = v.y <= 1.0000000116860974230803549289703e-07f;
    bool _186;
    if (_180)
    {
        _186 = v.y >= 0.0f;
    }
    else
    {
        _186 = _180;
    }
    if (_186)
    {
        float3 _2127 = inv_v;
        _2127.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _2127;
    }
    else
    {
        bool _193 = v.y >= (-1.0000000116860974230803549289703e-07f);
        bool _199;
        if (_193)
        {
            _199 = v.y < 0.0f;
        }
        else
        {
            _199 = _193;
        }
        if (_199)
        {
            float3 _2129 = inv_v;
            _2129.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _2129;
        }
    }
    bool _206 = v.z <= 1.0000000116860974230803549289703e-07f;
    bool _212;
    if (_206)
    {
        _212 = v.z >= 0.0f;
    }
    else
    {
        _212 = _206;
    }
    if (_212)
    {
        float3 _2131 = inv_v;
        _2131.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _2131;
    }
    else
    {
        bool _219 = v.z >= (-1.0000000116860974230803549289703e-07f);
        bool _225;
        if (_219)
        {
            _225 = v.z < 0.0f;
        }
        else
        {
            _225 = _219;
        }
        if (_225)
        {
            float3 _2133 = inv_v;
            _2133.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _2133;
        }
    }
    return inv_v;
}

bool _bbox_test_fma(float3 inv_d, float3 neg_inv_d_o, float t, float3 bbox_min, float3 bbox_max)
{
    float _608 = mad(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float _616 = mad(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float _631 = mad(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    float _638 = mad(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    float _655 = mad(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    float _662 = mad(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    float _667 = max(max(min(_608, _616), min(_631, _638)), min(_655, _662));
    float _675 = min(min(max(_608, _616), max(_631, _638)), max(_655, _662)) * 1.0000002384185791015625f;
    return ((_667 <= _675) && (_667 <= t)) && (_675 > 0.0f);
}

void IntersectTri(float3 ro, float3 rd, tri_accel_t tri, uint prim_index, inout hit_data_t inter)
{
    do
    {
        float _380 = dot(rd, tri.n_plane.xyz);
        float _389 = tri.n_plane.w - dot(ro, tri.n_plane.xyz);
        if (sign(_389) != sign(mad(_380, inter.t, -_389)))
        {
            break;
        }
        float3 _410 = (ro * _380) + (rd * _389);
        float _421 = mad(_380, tri.u_plane.w, dot(_410, tri.u_plane.xyz));
        float _426 = _380 - _421;
        if (sign(_421) != sign(_426))
        {
            break;
        }
        float _443 = mad(_380, tri.v_plane.w, dot(_410, tri.v_plane.xyz));
        if (sign(_443) != sign(_426 - _443))
        {
            break;
        }
        float _458 = 1.0f / _380;
        inter.mask = -1;
        int _463;
        if (_380 < 0.0f)
        {
            _463 = int(prim_index);
        }
        else
        {
            _463 = (-1) - int(prim_index);
        }
        inter.prim_index = _463;
        inter.t = _389 * _458;
        inter.u = _421 * _458;
        inter.v = _443 * _458;
        break;
    } while(false);
}

bool IntersectTris_AnyHit(float3 ro, float3 rd, int tri_start, int tri_end, int obj_index, inout hit_data_t out_inter)
{
    int _1947 = 0;
    int _1948 = obj_index;
    float _1950 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _1949;
    float _1951;
    float _1952;
    for (int i = tri_start; i < tri_end; )
    {
        param = ro;
        param_1 = rd;
        tri_accel_t _521;
        _521.n_plane = asfloat(_510.Load4(i * 48 + 0));
        _521.u_plane = asfloat(_510.Load4(i * 48 + 16));
        _521.v_plane = asfloat(_510.Load4(i * 48 + 32));
        param_2.n_plane = _521.n_plane;
        param_2.u_plane = _521.u_plane;
        param_2.v_plane = _521.v_plane;
        param_3 = uint(i);
        hit_data_t _1959 = { _1947, _1948, _1949, _1950, _1951, _1952 };
        param_4 = _1959;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _1947 = param_4.mask;
        _1948 = param_4.obj_index;
        _1949 = param_4.prim_index;
        _1950 = param_4.t;
        _1951 = param_4.u;
        _1952 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _1947;
    int _544;
    if (_1947 != 0)
    {
        _544 = _1948;
    }
    else
    {
        _544 = out_inter.obj_index;
    }
    out_inter.obj_index = _544;
    int _557;
    if (_1947 != 0)
    {
        _557 = _1949;
    }
    else
    {
        _557 = out_inter.prim_index;
    }
    out_inter.prim_index = _557;
    out_inter.t = _1950;
    float _573;
    if (_1947 != 0)
    {
        _573 = _1951;
    }
    else
    {
        _573 = out_inter.u;
    }
    out_inter.u = _573;
    float _586;
    if (_1947 != 0)
    {
        _586 = _1952;
    }
    else
    {
        _586 = out_inter.v;
    }
    out_inter.v = _586;
    return _1947 != 0;
}

bool Traverse_MicroTree_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    bool _1924 = false;
    bool _1921;
    do
    {
        float3 _692 = (-inv_d) * ro;
        uint _694 = stack_size;
        uint _704 = stack_size;
        stack_size = _704 + uint(1);
        g_stack[gl_LocalInvocationIndex][_704] = node_index;
        uint _778;
        uint _802;
        int _854;
        while (stack_size != _694)
        {
            uint _719 = stack_size;
            uint _720 = _719 - uint(1);
            stack_size = _720;
            bvh_node_t _734;
            _734.bbox_min = asfloat(_730.Load4(g_stack[gl_LocalInvocationIndex][_720] * 32 + 0));
            _734.bbox_max = asfloat(_730.Load4(g_stack[gl_LocalInvocationIndex][_720] * 32 + 16));
            float3 param = inv_d;
            float3 param_1 = _692;
            float param_2 = inter.t;
            float3 param_3 = _734.bbox_min.xyz;
            float3 param_4 = _734.bbox_max.xyz;
            if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
            {
                continue;
            }
            uint _761 = asuint(_734.bbox_min.w);
            if ((_761 & 2147483648u) == 0u)
            {
                uint _768 = stack_size;
                stack_size = _768 + uint(1);
                uint _772 = asuint(_734.bbox_max.w);
                uint _774 = _772 >> uint(30);
                if (rd[_774] < 0.0f)
                {
                    _778 = _761;
                }
                else
                {
                    _778 = _772 & 1073741823u;
                }
                g_stack[gl_LocalInvocationIndex][_768] = _778;
                uint _793 = stack_size;
                stack_size = _793 + uint(1);
                if (rd[_774] < 0.0f)
                {
                    _802 = _772 & 1073741823u;
                }
                else
                {
                    _802 = _761;
                }
                g_stack[gl_LocalInvocationIndex][_793] = _802;
            }
            else
            {
                int _822 = int(_761 & 2147483647u);
                float3 param_5 = ro;
                float3 param_6 = rd;
                int param_7 = _822;
                int param_8 = _822 + asint(_734.bbox_max.w);
                int param_9 = obj_index;
                hit_data_t param_10 = inter;
                bool _843 = IntersectTris_AnyHit(param_5, param_6, param_7, param_8, param_9, param_10);
                inter = param_10;
                if (_843)
                {
                    bool _851 = inter.prim_index < 0;
                    if (_851)
                    {
                        _854 = (-1) - inter.prim_index;
                    }
                    else
                    {
                        _854 = inter.prim_index;
                    }
                    uint _865 = uint(_854);
                    bool _893 = !_851;
                    bool _900;
                    if (_893)
                    {
                        _900 = (((_879.Load(_870.Load(_865 * 4 + 0) * 4 + 0) >> 16u) & 65535u) & 32768u) != 0u;
                    }
                    else
                    {
                        _900 = _893;
                    }
                    bool _911;
                    if (!_900)
                    {
                        bool _910;
                        if (_851)
                        {
                            _910 = ((_879.Load(_870.Load(_865 * 4 + 0) * 4 + 0) & 65535u) & 32768u) != 0u;
                        }
                        else
                        {
                            _910 = _851;
                        }
                        _911 = _910;
                    }
                    else
                    {
                        _911 = _900;
                    }
                    if (_911)
                    {
                        _1924 = true;
                        _1921 = true;
                        break;
                    }
                }
            }
        }
        if (_1924)
        {
            break;
        }
        _1924 = true;
        _1921 = false;
        break;
    } while(false);
    return _1921;
}

bool Traverse_MacroTree_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    bool _1915 = false;
    bool _1912;
    do
    {
        float3 _922 = (-orig_inv_rd) * orig_ro;
        uint stack_size = 1u;
        g_stack[gl_LocalInvocationIndex][0u] = node_index;
        uint _987;
        uint _1010;
        while (stack_size != 0u)
        {
            uint _938 = stack_size;
            uint _939 = _938 - uint(1);
            stack_size = _939;
            bvh_node_t _945;
            _945.bbox_min = asfloat(_730.Load4(g_stack[gl_LocalInvocationIndex][_939] * 32 + 0));
            _945.bbox_max = asfloat(_730.Load4(g_stack[gl_LocalInvocationIndex][_939] * 32 + 16));
            float3 param = orig_inv_rd;
            float3 param_1 = _922;
            float param_2 = inter.t;
            float3 param_3 = _945.bbox_min.xyz;
            float3 param_4 = _945.bbox_max.xyz;
            if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
            {
                continue;
            }
            uint _972 = asuint(_945.bbox_min.w);
            if ((_972 & 2147483648u) == 0u)
            {
                uint _978 = stack_size;
                stack_size = _978 + uint(1);
                uint _982 = asuint(_945.bbox_max.w);
                uint _983 = _982 >> uint(30);
                if (orig_rd[_983] < 0.0f)
                {
                    _987 = _972;
                }
                else
                {
                    _987 = _982 & 1073741823u;
                }
                g_stack[gl_LocalInvocationIndex][_978] = _987;
                uint _1001 = stack_size;
                stack_size = _1001 + uint(1);
                if (orig_rd[_983] < 0.0f)
                {
                    _1010 = _982 & 1073741823u;
                }
                else
                {
                    _1010 = _972;
                }
                g_stack[gl_LocalInvocationIndex][_1001] = _1010;
            }
            else
            {
                uint _1028 = _972 & 2147483647u;
                uint _1032 = asuint(_945.bbox_max.w);
                for (uint i = _1028; i < (_1028 + _1032); i++)
                {
                    mesh_instance_t _1062;
                    _1062.bbox_min = asfloat(_1052.Load4(_1056.Load(i * 4 + 0) * 32 + 0));
                    _1062.bbox_max = asfloat(_1052.Load4(_1056.Load(i * 4 + 0) * 32 + 16));
                    mesh_t _1083;
                    [unroll]
                    for (int _26ident = 0; _26ident < 3; _26ident++)
                    {
                        _1083.bbox_min[_26ident] = asfloat(_1077.Load(_26ident * 4 + asuint(_1062.bbox_max.w) * 48 + 0));
                    }
                    [unroll]
                    for (int _27ident = 0; _27ident < 3; _27ident++)
                    {
                        _1083.bbox_max[_27ident] = asfloat(_1077.Load(_27ident * 4 + asuint(_1062.bbox_max.w) * 48 + 12));
                    }
                    _1083.node_index = _1077.Load(asuint(_1062.bbox_max.w) * 48 + 24);
                    _1083.node_count = _1077.Load(asuint(_1062.bbox_max.w) * 48 + 28);
                    _1083.tris_index = _1077.Load(asuint(_1062.bbox_max.w) * 48 + 32);
                    _1083.tris_count = _1077.Load(asuint(_1062.bbox_max.w) * 48 + 36);
                    _1083.vert_index = _1077.Load(asuint(_1062.bbox_max.w) * 48 + 40);
                    _1083.vert_count = _1077.Load(asuint(_1062.bbox_max.w) * 48 + 44);
                    transform_t _1129;
                    _1129.xform = asfloat(uint4x4(_1123.Load4(asuint(_1062.bbox_min.w) * 128 + 0), _1123.Load4(asuint(_1062.bbox_min.w) * 128 + 16), _1123.Load4(asuint(_1062.bbox_min.w) * 128 + 32), _1123.Load4(asuint(_1062.bbox_min.w) * 128 + 48)));
                    _1129.inv_xform = asfloat(uint4x4(_1123.Load4(asuint(_1062.bbox_min.w) * 128 + 64), _1123.Load4(asuint(_1062.bbox_min.w) * 128 + 80), _1123.Load4(asuint(_1062.bbox_min.w) * 128 + 96), _1123.Load4(asuint(_1062.bbox_min.w) * 128 + 112)));
                    float3 param_5 = orig_inv_rd;
                    float3 param_6 = _922;
                    float param_7 = inter.t;
                    float3 param_8 = _1062.bbox_min.xyz;
                    float3 param_9 = _1062.bbox_max.xyz;
                    if (!_bbox_test_fma(param_5, param_6, param_7, param_8, param_9))
                    {
                        continue;
                    }
                    float3 _1174 = mul(float4(orig_rd, 0.0f), _1129.inv_xform).xyz;
                    float3 param_10 = _1174;
                    float3 param_11 = mul(float4(orig_ro, 1.0f), _1129.inv_xform).xyz;
                    float3 param_12 = _1174;
                    float3 param_13 = safe_invert(param_10);
                    int param_14 = int(_1056.Load(i * 4 + 0));
                    uint param_15 = _1083.node_index;
                    uint param_16 = stack_size;
                    hit_data_t param_17 = inter;
                    bool _1198 = Traverse_MicroTree_WithStack(param_11, param_12, param_13, param_14, param_15, param_16, param_17);
                    inter = param_17;
                    if (_1198)
                    {
                        _1915 = true;
                        _1912 = true;
                        break;
                    }
                }
                if (_1915)
                {
                    break;
                }
            }
        }
        if (_1915)
        {
            break;
        }
        _1915 = true;
        _1912 = false;
        break;
    } while(false);
    return _1912;
}

float construct_float(inout uint m)
{
    m &= 8388607u;
    m |= 1065353216u;
    return asfloat(m) - 1.0f;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _280 = mad(col.z, 31.875f, 1.0f);
    float _291 = (col.x - 0.501960813999176025390625f) / _280;
    float _297 = (col.y - 0.501960813999176025390625f) / _280;
    return float3((col.w + _291) - _297, col.w + _297, (col.w - _291) - _297);
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
    uint _326 = index & 16777215u;
    float4 res = g_textures[NonUniformResourceIndex(_326)].SampleLevel(_g_textures_sampler[NonUniformResourceIndex(_326)], uvs, float(lod));
    bool _337;
    if (maybe_YCoCg)
    {
        _337 = (index & 67108864u) != 0u;
    }
    else
    {
        _337 = maybe_YCoCg;
    }
    if (_337)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _355;
    if (maybe_SRGB)
    {
        _355 = (index & 16777216u) != 0u;
    }
    else
    {
        _355 = maybe_SRGB;
    }
    if (_355)
    {
        float3 param_1 = res.xyz;
        float3 _361 = srgb_to_rgb(param_1);
        float4 _2149 = res;
        _2149.x = _361.x;
        float4 _2151 = _2149;
        _2151.y = _361.y;
        float4 _2153 = _2151;
        _2153.z = _361.z;
        res = _2153;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod)
{
    return SampleBilinear(index, uvs, lod, false, false);
}

bool ComputeVisibility(inout float3 p, float3 d, inout float dist, float rand_val, int rand_hash2)
{
    bool _1908 = false;
    bool _1905;
    do
    {
        float3 param = d;
        float3 _1211 = safe_invert(param);
        int _1260;
        int _2009;
        int _2010;
        float _2012;
        float _2013;
        while (dist > 9.9999997473787516355514526367188e-06f)
        {
            int _2008 = 0;
            float _2011 = dist;
            float3 param_1 = p;
            float3 param_2 = d;
            float3 param_3 = _1211;
            uint param_4 = _1229_g_params.node_index;
            hit_data_t _2020 = { 0, _2009, _2010, dist, _2012, _2013 };
            hit_data_t param_5 = _2020;
            bool _1242 = Traverse_MacroTree_WithStack(param_1, param_2, param_3, param_4, param_5);
            _2008 = param_5.mask;
            _2009 = param_5.obj_index;
            _2010 = param_5.prim_index;
            _2011 = param_5.t;
            _2012 = param_5.u;
            _2013 = param_5.v;
            if (_1242)
            {
                _1908 = true;
                _1905 = false;
                break;
            }
            if (_2008 == 0)
            {
                _1908 = true;
                _1905 = true;
                break;
            }
            bool _1257 = param_5.prim_index < 0;
            if (_1257)
            {
                _1260 = (-1) - param_5.prim_index;
            }
            else
            {
                _1260 = param_5.prim_index;
            }
            uint _1271 = uint(_1260);
            material_t _1309;
            [unroll]
            for (int _28ident = 0; _28ident < 5; _28ident++)
            {
                _1309.textures[_28ident] = _1305.Load(_28ident * 4 + ((_1257 ? (_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) & 65535u) : ((_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 0);
            }
            [unroll]
            for (int _29ident = 0; _29ident < 3; _29ident++)
            {
                _1309.base_color[_29ident] = asfloat(_1305.Load(_29ident * 4 + ((_1257 ? (_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) & 65535u) : ((_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 20));
            }
            _1309.flags = _1305.Load(((_1257 ? (_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) & 65535u) : ((_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 32);
            _1309.type = _1305.Load(((_1257 ? (_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) & 65535u) : ((_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 36);
            _1309.tangent_rotation_or_strength = asfloat(_1305.Load(((_1257 ? (_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) & 65535u) : ((_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 40));
            _1309.roughness_and_anisotropic = _1305.Load(((_1257 ? (_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) & 65535u) : ((_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 44);
            _1309.int_ior = asfloat(_1305.Load(((_1257 ? (_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) & 65535u) : ((_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 48));
            _1309.ext_ior = asfloat(_1305.Load(((_1257 ? (_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) & 65535u) : ((_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 52));
            _1309.sheen_and_sheen_tint = _1305.Load(((_1257 ? (_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) & 65535u) : ((_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 56);
            _1309.tint_and_metallic = _1305.Load(((_1257 ? (_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) & 65535u) : ((_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 60);
            _1309.transmission_and_transmission_roughness = _1305.Load(((_1257 ? (_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) & 65535u) : ((_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 64);
            _1309.specular_and_specular_tint = _1305.Load(((_1257 ? (_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) & 65535u) : ((_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 68);
            _1309.clearcoat_and_clearcoat_roughness = _1305.Load(((_1257 ? (_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) & 65535u) : ((_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 72);
            _1309.normal_map_strength_unorm = _1305.Load(((_1257 ? (_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) & 65535u) : ((_879.Load(_870.Load(_1271 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u) * 80 + 76);
            uint _2064 = _1309.textures[1];
            uint _2066 = _1309.textures[3];
            uint _2067 = _1309.textures[4];
            uint _2030 = _1309.type;
            float _2031 = _1309.tangent_rotation_or_strength;
            uint _1394 = _870.Load(_1271 * 4 + 0) * 3u;
            vertex_t _1400;
            [unroll]
            for (int _30ident = 0; _30ident < 3; _30ident++)
            {
                _1400.p[_30ident] = asfloat(_1388.Load(_30ident * 4 + _1392.Load(_1394 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _31ident = 0; _31ident < 3; _31ident++)
            {
                _1400.n[_31ident] = asfloat(_1388.Load(_31ident * 4 + _1392.Load(_1394 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _32ident = 0; _32ident < 3; _32ident++)
            {
                _1400.b[_32ident] = asfloat(_1388.Load(_32ident * 4 + _1392.Load(_1394 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _33ident = 0; _33ident < 2; _33ident++)
            {
                [unroll]
                for (int _34ident = 0; _34ident < 2; _34ident++)
                {
                    _1400.t[_33ident][_34ident] = asfloat(_1388.Load(_34ident * 4 + _33ident * 8 + _1392.Load(_1394 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1448;
            [unroll]
            for (int _35ident = 0; _35ident < 3; _35ident++)
            {
                _1448.p[_35ident] = asfloat(_1388.Load(_35ident * 4 + _1392.Load((_1394 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _1448.n[_36ident] = asfloat(_1388.Load(_36ident * 4 + _1392.Load((_1394 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _37ident = 0; _37ident < 3; _37ident++)
            {
                _1448.b[_37ident] = asfloat(_1388.Load(_37ident * 4 + _1392.Load((_1394 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _38ident = 0; _38ident < 2; _38ident++)
            {
                [unroll]
                for (int _39ident = 0; _39ident < 2; _39ident++)
                {
                    _1448.t[_38ident][_39ident] = asfloat(_1388.Load(_39ident * 4 + _38ident * 8 + _1392.Load((_1394 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1494;
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _1494.p[_40ident] = asfloat(_1388.Load(_40ident * 4 + _1392.Load((_1394 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 3; _41ident++)
            {
                _1494.n[_41ident] = asfloat(_1388.Load(_41ident * 4 + _1392.Load((_1394 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _42ident = 0; _42ident < 3; _42ident++)
            {
                _1494.b[_42ident] = asfloat(_1388.Load(_42ident * 4 + _1392.Load((_1394 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _43ident = 0; _43ident < 2; _43ident++)
            {
                [unroll]
                for (int _44ident = 0; _44ident < 2; _44ident++)
                {
                    _1494.t[_43ident][_44ident] = asfloat(_1388.Load(_44ident * 4 + _43ident * 8 + _1392.Load((_1394 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            float2 _1566 = ((float2(_1400.t[0][0], _1400.t[0][1]) * ((1.0f - param_5.u) - param_5.v)) + (float2(_1448.t[0][0], _1448.t[0][1]) * param_5.u)) + (float2(_1494.t[0][0], _1494.t[0][1]) * param_5.v);
            uint param_6 = uint(hash(rand_hash2));
            float _1574 = construct_float(param_6);
            float sh_r = frac(rand_val + _1574);
            while (_2030 == 4u)
            {
                float mix_val = _2031;
                if (_2064 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_2064, _1566, 0).x;
                }
                if (sh_r > mix_val)
                {
                    material_t _1613;
                    [unroll]
                    for (int _45ident = 0; _45ident < 5; _45ident++)
                    {
                        _1613.textures[_45ident] = _1305.Load(_45ident * 4 + _2066 * 80 + 0);
                    }
                    [unroll]
                    for (int _46ident = 0; _46ident < 3; _46ident++)
                    {
                        _1613.base_color[_46ident] = asfloat(_1305.Load(_46ident * 4 + _2066 * 80 + 20));
                    }
                    _1613.flags = _1305.Load(_2066 * 80 + 32);
                    _1613.type = _1305.Load(_2066 * 80 + 36);
                    _1613.tangent_rotation_or_strength = asfloat(_1305.Load(_2066 * 80 + 40));
                    _1613.roughness_and_anisotropic = _1305.Load(_2066 * 80 + 44);
                    _1613.int_ior = asfloat(_1305.Load(_2066 * 80 + 48));
                    _1613.ext_ior = asfloat(_1305.Load(_2066 * 80 + 52));
                    _1613.sheen_and_sheen_tint = _1305.Load(_2066 * 80 + 56);
                    _1613.tint_and_metallic = _1305.Load(_2066 * 80 + 60);
                    _1613.transmission_and_transmission_roughness = _1305.Load(_2066 * 80 + 64);
                    _1613.specular_and_specular_tint = _1305.Load(_2066 * 80 + 68);
                    _1613.clearcoat_and_clearcoat_roughness = _1305.Load(_2066 * 80 + 72);
                    _1613.normal_map_strength_unorm = _1305.Load(_2066 * 80 + 76);
                    _2064 = _1613.textures[1];
                    _2066 = _1613.textures[3];
                    _2067 = _1613.textures[4];
                    _2030 = _1613.type;
                    _2031 = _1613.tangent_rotation_or_strength;
                    sh_r = (sh_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _1668;
                    [unroll]
                    for (int _47ident = 0; _47ident < 5; _47ident++)
                    {
                        _1668.textures[_47ident] = _1305.Load(_47ident * 4 + _2067 * 80 + 0);
                    }
                    [unroll]
                    for (int _48ident = 0; _48ident < 3; _48ident++)
                    {
                        _1668.base_color[_48ident] = asfloat(_1305.Load(_48ident * 4 + _2067 * 80 + 20));
                    }
                    _1668.flags = _1305.Load(_2067 * 80 + 32);
                    _1668.type = _1305.Load(_2067 * 80 + 36);
                    _1668.tangent_rotation_or_strength = asfloat(_1305.Load(_2067 * 80 + 40));
                    _1668.roughness_and_anisotropic = _1305.Load(_2067 * 80 + 44);
                    _1668.int_ior = asfloat(_1305.Load(_2067 * 80 + 48));
                    _1668.ext_ior = asfloat(_1305.Load(_2067 * 80 + 52));
                    _1668.sheen_and_sheen_tint = _1305.Load(_2067 * 80 + 56);
                    _1668.tint_and_metallic = _1305.Load(_2067 * 80 + 60);
                    _1668.transmission_and_transmission_roughness = _1305.Load(_2067 * 80 + 64);
                    _1668.specular_and_specular_tint = _1305.Load(_2067 * 80 + 68);
                    _1668.clearcoat_and_clearcoat_roughness = _1305.Load(_2067 * 80 + 72);
                    _1668.normal_map_strength_unorm = _1305.Load(_2067 * 80 + 76);
                    _2064 = _1668.textures[1];
                    _2066 = _1668.textures[3];
                    _2067 = _1668.textures[4];
                    _2030 = _1668.type;
                    _2031 = _1668.tangent_rotation_or_strength;
                    sh_r /= mix_val;
                }
            }
            if (_2030 != 5u)
            {
                _1908 = true;
                _1905 = false;
                break;
            }
            float _1725 = _2011 + 9.9999997473787516355514526367188e-06f;
            p += (d * _1725);
            dist -= _1725;
        }
        if (_1908)
        {
            break;
        }
        _1908 = true;
        _1905 = true;
        break;
    } while(false);
    return _1905;
}

void comp_main()
{
    do
    {
        int _1745 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_1745) >= _1751.Load(12))
        {
            break;
        }
        shadow_ray_t _1772;
        [unroll]
        for (int _49ident = 0; _49ident < 3; _49ident++)
        {
            _1772.o[_49ident] = asfloat(_1768.Load(_49ident * 4 + _1745 * 44 + 0));
        }
        [unroll]
        for (int _50ident = 0; _50ident < 3; _50ident++)
        {
            _1772.d[_50ident] = asfloat(_1768.Load(_50ident * 4 + _1745 * 44 + 12));
        }
        _1772.dist = asfloat(_1768.Load(_1745 * 44 + 24));
        [unroll]
        for (int _51ident = 0; _51ident < 3; _51ident++)
        {
            _1772.c[_51ident] = asfloat(_1768.Load(_51ident * 4 + _1745 * 44 + 28));
        }
        _1772.xy = int(_1768.Load(_1745 * 44 + 40));
        int _1806 = (_1772.xy >> 16) & 65535;
        int _1810 = _1772.xy & 65535;
        float3 param = float3(asfloat(_1768.Load(_1745 * 44 + 0)), asfloat(_1768.Load(_1745 * 44 + 4)), asfloat(_1768.Load(_1745 * 44 + 8)));
        float3 param_1 = float3(asfloat(_1768.Load(_1745 * 44 + 12)), asfloat(_1768.Load(_1745 * 44 + 16)), asfloat(_1768.Load(_1745 * 44 + 20)));
        float param_2 = _1772.dist;
        float param_3 = _1229_g_params.halton;
        int param_4 = hash((_1810 * int(_1229_g_params.img_size.x)) + _1806);
        bool _1855 = ComputeVisibility(param, param_1, param_2, param_3, param_4);
        if (_1855)
        {
            int2 _1866 = int2(_1806, _1810);
            g_out_img[_1866] = float4(g_out_img[_1866].xyz + float3(_1772.c[0], _1772.c[1], _1772.c[2]), 1.0f);
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
