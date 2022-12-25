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

ByteAddressBuffer _484 : register(t1, space0);
ByteAddressBuffer _704 : register(t5, space0);
ByteAddressBuffer _844 : register(t2, space0);
ByteAddressBuffer _853 : register(t3, space0);
ByteAddressBuffer _1026 : register(t7, space0);
ByteAddressBuffer _1030 : register(t8, space0);
ByteAddressBuffer _1050 : register(t6, space0);
ByteAddressBuffer _1096 : register(t9, space0);
ByteAddressBuffer _1321 : register(t10, space0);
ByteAddressBuffer _1325 : register(t11, space0);
ByteAddressBuffer _1527 : register(t4, space0);
ByteAddressBuffer _1713 : register(t13, space0);
ByteAddressBuffer _1728 : register(t12, space0);
cbuffer UniformParams
{
    Params _1231_g_params : packoffset(c0);
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

float3 safe_invert(float3 v)
{
    float3 inv_v = 1.0f.xxx / v;
    bool _127 = v.x <= 1.0000000116860974230803549289703e-07f;
    bool _134;
    if (_127)
    {
        _134 = v.x >= 0.0f;
    }
    else
    {
        _134 = _127;
    }
    if (_134)
    {
        float3 _2065 = inv_v;
        _2065.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _2065;
    }
    else
    {
        bool _143 = v.x >= (-1.0000000116860974230803549289703e-07f);
        bool _149;
        if (_143)
        {
            _149 = v.x < 0.0f;
        }
        else
        {
            _149 = _143;
        }
        if (_149)
        {
            float3 _2067 = inv_v;
            _2067.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _2067;
        }
    }
    bool _156 = v.y <= 1.0000000116860974230803549289703e-07f;
    bool _162;
    if (_156)
    {
        _162 = v.y >= 0.0f;
    }
    else
    {
        _162 = _156;
    }
    if (_162)
    {
        float3 _2069 = inv_v;
        _2069.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _2069;
    }
    else
    {
        bool _169 = v.y >= (-1.0000000116860974230803549289703e-07f);
        bool _175;
        if (_169)
        {
            _175 = v.y < 0.0f;
        }
        else
        {
            _175 = _169;
        }
        if (_175)
        {
            float3 _2071 = inv_v;
            _2071.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _2071;
        }
    }
    bool _181 = v.z <= 1.0000000116860974230803549289703e-07f;
    bool _187;
    if (_181)
    {
        _187 = v.z >= 0.0f;
    }
    else
    {
        _187 = _181;
    }
    if (_187)
    {
        float3 _2073 = inv_v;
        _2073.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _2073;
    }
    else
    {
        bool _194 = v.z >= (-1.0000000116860974230803549289703e-07f);
        bool _200;
        if (_194)
        {
            _200 = v.z < 0.0f;
        }
        else
        {
            _200 = _194;
        }
        if (_200)
        {
            float3 _2075 = inv_v;
            _2075.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _2075;
        }
    }
    return inv_v;
}

bool _bbox_test_fma(float3 inv_d, float3 neg_inv_d_o, float t, float3 bbox_min, float3 bbox_max)
{
    float _582 = mad(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float _590 = mad(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float _605 = mad(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    float _612 = mad(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    float _629 = mad(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    float _636 = mad(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    float _641 = max(max(min(_582, _590), min(_605, _612)), min(_629, _636));
    float _649 = min(min(max(_582, _590), max(_605, _612)), max(_629, _636)) * 1.0000002384185791015625f;
    return ((_641 <= _649) && (_641 <= t)) && (_649 > 0.0f);
}

void IntersectTri(float3 ro, float3 rd, tri_accel_t tri, uint prim_index, inout hit_data_t inter)
{
    do
    {
        float _354 = dot(rd, tri.n_plane.xyz);
        float _363 = tri.n_plane.w - dot(ro, tri.n_plane.xyz);
        if (sign(_363) != sign(mad(_354, inter.t, -_363)))
        {
            break;
        }
        float3 _384 = (ro * _354) + (rd * _363);
        float _395 = mad(_354, tri.u_plane.w, dot(_384, tri.u_plane.xyz));
        float _400 = _354 - _395;
        if (sign(_395) != sign(_400))
        {
            break;
        }
        float _417 = mad(_354, tri.v_plane.w, dot(_384, tri.v_plane.xyz));
        if (sign(_417) != sign(_400 - _417))
        {
            break;
        }
        float _432 = 1.0f / _354;
        inter.mask = -1;
        int _437;
        if (_354 < 0.0f)
        {
            _437 = int(prim_index);
        }
        else
        {
            _437 = (-1) - int(prim_index);
        }
        inter.prim_index = _437;
        inter.t = _363 * _432;
        inter.u = _395 * _432;
        inter.v = _417 * _432;
        break;
    } while(false);
}

bool IntersectTris_AnyHit(float3 ro, float3 rd, int tri_start, int tri_end, int obj_index, inout hit_data_t out_inter)
{
    int _1886 = 0;
    int _1887 = obj_index;
    float _1889 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _1888;
    float _1890;
    float _1891;
    for (int i = tri_start; i < tri_end; )
    {
        param = ro;
        param_1 = rd;
        tri_accel_t _495;
        _495.n_plane = asfloat(_484.Load4(i * 48 + 0));
        _495.u_plane = asfloat(_484.Load4(i * 48 + 16));
        _495.v_plane = asfloat(_484.Load4(i * 48 + 32));
        param_2.n_plane = _495.n_plane;
        param_2.u_plane = _495.u_plane;
        param_2.v_plane = _495.v_plane;
        param_3 = uint(i);
        hit_data_t _1898 = { _1886, _1887, _1888, _1889, _1890, _1891 };
        param_4 = _1898;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _1886 = param_4.mask;
        _1887 = param_4.obj_index;
        _1888 = param_4.prim_index;
        _1889 = param_4.t;
        _1890 = param_4.u;
        _1891 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _1886;
    int _518;
    if (_1886 != 0)
    {
        _518 = _1887;
    }
    else
    {
        _518 = out_inter.obj_index;
    }
    out_inter.obj_index = _518;
    int _531;
    if (_1886 != 0)
    {
        _531 = _1888;
    }
    else
    {
        _531 = out_inter.prim_index;
    }
    out_inter.prim_index = _531;
    out_inter.t = _1889;
    float _547;
    if (_1886 != 0)
    {
        _547 = _1890;
    }
    else
    {
        _547 = out_inter.u;
    }
    out_inter.u = _547;
    float _560;
    if (_1886 != 0)
    {
        _560 = _1891;
    }
    else
    {
        _560 = out_inter.v;
    }
    out_inter.v = _560;
    return _1886 != 0;
}

bool Traverse_MicroTree_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    bool _1846 = false;
    bool _1843;
    do
    {
        float3 _666 = (-inv_d) * ro;
        uint _668 = stack_size;
        uint _678 = stack_size;
        stack_size = _678 + uint(1);
        g_stack[gl_LocalInvocationIndex][_678] = node_index;
        uint _752;
        uint _776;
        int _828;
        while (stack_size != _668)
        {
            uint _693 = stack_size;
            uint _694 = _693 - uint(1);
            stack_size = _694;
            bvh_node_t _708;
            _708.bbox_min = asfloat(_704.Load4(g_stack[gl_LocalInvocationIndex][_694] * 32 + 0));
            _708.bbox_max = asfloat(_704.Load4(g_stack[gl_LocalInvocationIndex][_694] * 32 + 16));
            float3 param = inv_d;
            float3 param_1 = _666;
            float param_2 = inter.t;
            float3 param_3 = _708.bbox_min.xyz;
            float3 param_4 = _708.bbox_max.xyz;
            if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
            {
                continue;
            }
            uint _735 = asuint(_708.bbox_min.w);
            if ((_735 & 2147483648u) == 0u)
            {
                uint _742 = stack_size;
                stack_size = _742 + uint(1);
                uint _746 = asuint(_708.bbox_max.w);
                uint _748 = _746 >> uint(30);
                if (rd[_748] < 0.0f)
                {
                    _752 = _735;
                }
                else
                {
                    _752 = _746 & 1073741823u;
                }
                g_stack[gl_LocalInvocationIndex][_742] = _752;
                uint _767 = stack_size;
                stack_size = _767 + uint(1);
                if (rd[_748] < 0.0f)
                {
                    _776 = _746 & 1073741823u;
                }
                else
                {
                    _776 = _735;
                }
                g_stack[gl_LocalInvocationIndex][_767] = _776;
            }
            else
            {
                int _796 = int(_735 & 2147483647u);
                float3 param_5 = ro;
                float3 param_6 = rd;
                int param_7 = _796;
                int param_8 = _796 + asint(_708.bbox_max.w);
                int param_9 = obj_index;
                hit_data_t param_10 = inter;
                bool _817 = IntersectTris_AnyHit(param_5, param_6, param_7, param_8, param_9, param_10);
                inter = param_10;
                if (_817)
                {
                    bool _825 = inter.prim_index < 0;
                    if (_825)
                    {
                        _828 = (-1) - inter.prim_index;
                    }
                    else
                    {
                        _828 = inter.prim_index;
                    }
                    uint _839 = uint(_828);
                    bool _867 = !_825;
                    bool _874;
                    if (_867)
                    {
                        _874 = (((_853.Load(_844.Load(_839 * 4 + 0) * 4 + 0) >> 16u) & 65535u) & 32768u) != 0u;
                    }
                    else
                    {
                        _874 = _867;
                    }
                    bool _885;
                    if (!_874)
                    {
                        bool _884;
                        if (_825)
                        {
                            _884 = ((_853.Load(_844.Load(_839 * 4 + 0) * 4 + 0) & 65535u) & 32768u) != 0u;
                        }
                        else
                        {
                            _884 = _825;
                        }
                        _885 = _884;
                    }
                    else
                    {
                        _885 = _874;
                    }
                    if (_885)
                    {
                        _1846 = true;
                        _1843 = true;
                        break;
                    }
                }
            }
        }
        if (_1846)
        {
            break;
        }
        _1846 = true;
        _1843 = false;
        break;
    } while(false);
    return _1843;
}

bool Traverse_MacroTree_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    bool _1837 = false;
    bool _1834;
    do
    {
        float3 _896 = (-orig_inv_rd) * orig_ro;
        uint stack_size = 1u;
        g_stack[gl_LocalInvocationIndex][0u] = node_index;
        uint _961;
        uint _984;
        while (stack_size != 0u)
        {
            uint _912 = stack_size;
            uint _913 = _912 - uint(1);
            stack_size = _913;
            bvh_node_t _919;
            _919.bbox_min = asfloat(_704.Load4(g_stack[gl_LocalInvocationIndex][_913] * 32 + 0));
            _919.bbox_max = asfloat(_704.Load4(g_stack[gl_LocalInvocationIndex][_913] * 32 + 16));
            float3 param = orig_inv_rd;
            float3 param_1 = _896;
            float param_2 = inter.t;
            float3 param_3 = _919.bbox_min.xyz;
            float3 param_4 = _919.bbox_max.xyz;
            if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
            {
                continue;
            }
            uint _946 = asuint(_919.bbox_min.w);
            if ((_946 & 2147483648u) == 0u)
            {
                uint _952 = stack_size;
                stack_size = _952 + uint(1);
                uint _956 = asuint(_919.bbox_max.w);
                uint _957 = _956 >> uint(30);
                if (orig_rd[_957] < 0.0f)
                {
                    _961 = _946;
                }
                else
                {
                    _961 = _956 & 1073741823u;
                }
                g_stack[gl_LocalInvocationIndex][_952] = _961;
                uint _975 = stack_size;
                stack_size = _975 + uint(1);
                if (orig_rd[_957] < 0.0f)
                {
                    _984 = _956 & 1073741823u;
                }
                else
                {
                    _984 = _946;
                }
                g_stack[gl_LocalInvocationIndex][_975] = _984;
            }
            else
            {
                uint _1002 = _946 & 2147483647u;
                uint _1006 = asuint(_919.bbox_max.w);
                for (uint i = _1002; i < (_1002 + _1006); i++)
                {
                    mesh_instance_t _1036;
                    _1036.bbox_min = asfloat(_1026.Load4(_1030.Load(i * 4 + 0) * 32 + 0));
                    _1036.bbox_max = asfloat(_1026.Load4(_1030.Load(i * 4 + 0) * 32 + 16));
                    mesh_t _1056;
                    [unroll]
                    for (int _22ident = 0; _22ident < 3; _22ident++)
                    {
                        _1056.bbox_min[_22ident] = asfloat(_1050.Load(_22ident * 4 + asuint(_1036.bbox_max.w) * 48 + 0));
                    }
                    [unroll]
                    for (int _23ident = 0; _23ident < 3; _23ident++)
                    {
                        _1056.bbox_max[_23ident] = asfloat(_1050.Load(_23ident * 4 + asuint(_1036.bbox_max.w) * 48 + 12));
                    }
                    _1056.node_index = _1050.Load(asuint(_1036.bbox_max.w) * 48 + 24);
                    _1056.node_count = _1050.Load(asuint(_1036.bbox_max.w) * 48 + 28);
                    _1056.tris_index = _1050.Load(asuint(_1036.bbox_max.w) * 48 + 32);
                    _1056.tris_count = _1050.Load(asuint(_1036.bbox_max.w) * 48 + 36);
                    _1056.vert_index = _1050.Load(asuint(_1036.bbox_max.w) * 48 + 40);
                    _1056.vert_count = _1050.Load(asuint(_1036.bbox_max.w) * 48 + 44);
                    transform_t _1102;
                    _1102.xform = asfloat(uint4x4(_1096.Load4(asuint(_1036.bbox_min.w) * 128 + 0), _1096.Load4(asuint(_1036.bbox_min.w) * 128 + 16), _1096.Load4(asuint(_1036.bbox_min.w) * 128 + 32), _1096.Load4(asuint(_1036.bbox_min.w) * 128 + 48)));
                    _1102.inv_xform = asfloat(uint4x4(_1096.Load4(asuint(_1036.bbox_min.w) * 128 + 64), _1096.Load4(asuint(_1036.bbox_min.w) * 128 + 80), _1096.Load4(asuint(_1036.bbox_min.w) * 128 + 96), _1096.Load4(asuint(_1036.bbox_min.w) * 128 + 112)));
                    float3 param_5 = orig_inv_rd;
                    float3 param_6 = _896;
                    float param_7 = inter.t;
                    float3 param_8 = _1036.bbox_min.xyz;
                    float3 param_9 = _1036.bbox_max.xyz;
                    if (!_bbox_test_fma(param_5, param_6, param_7, param_8, param_9))
                    {
                        continue;
                    }
                    float3 _1147 = mul(float4(orig_rd, 0.0f), _1102.inv_xform).xyz;
                    float3 param_10 = _1147;
                    float3 param_11 = mul(float4(orig_ro, 1.0f), _1102.inv_xform).xyz;
                    float3 param_12 = _1147;
                    float3 param_13 = safe_invert(param_10);
                    int param_14 = int(_1030.Load(i * 4 + 0));
                    uint param_15 = _1056.node_index;
                    uint param_16 = stack_size;
                    hit_data_t param_17 = inter;
                    bool _1171 = Traverse_MicroTree_WithStack(param_11, param_12, param_13, param_14, param_15, param_16, param_17);
                    inter = param_17;
                    if (_1171)
                    {
                        _1837 = true;
                        _1834 = true;
                        break;
                    }
                }
                if (_1837)
                {
                    break;
                }
            }
        }
        if (_1837)
        {
            break;
        }
        _1837 = true;
        _1834 = false;
        break;
    } while(false);
    return _1834;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _255 = mad(col.z, 31.875f, 1.0f);
    float _265 = (col.x - 0.501960813999176025390625f) / _255;
    float _271 = (col.y - 0.501960813999176025390625f) / _255;
    return float3((col.w + _265) - _271, col.w + _271, (col.w - _265) - _271);
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
    uint _300 = index & 16777215u;
    float4 res = g_textures[NonUniformResourceIndex(_300)].SampleLevel(_g_textures_sampler[NonUniformResourceIndex(_300)], uvs, float(lod));
    bool _311;
    if (maybe_YCoCg)
    {
        _311 = (index & 67108864u) != 0u;
    }
    else
    {
        _311 = maybe_YCoCg;
    }
    if (_311)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _329;
    if (maybe_SRGB)
    {
        _329 = (index & 16777216u) != 0u;
    }
    else
    {
        _329 = maybe_SRGB;
    }
    if (_329)
    {
        float3 param_1 = res.xyz;
        float3 _335 = srgb_to_rgb(param_1);
        float4 _2091 = res;
        _2091.x = _335.x;
        float4 _2093 = _2091;
        _2093.y = _335.y;
        float4 _2095 = _2093;
        _2095.z = _335.z;
        res = _2095;
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
    bool _1830 = false;
    float3 _1827;
    do
    {
        float3 ro = float3(r.o[0], r.o[1], r.o[2]);
        float3 _1196 = float3(r.d[0], r.d[1], r.d[2]);
        float3 rc = float3(r.c[0], r.c[1], r.c[2]);
        int depth = r.depth >> 24;
        float3 param = _1196;
        float3 _1213 = safe_invert(param);
        int _1273;
        int _1948;
        int _1949;
        float _1951;
        float _1952;
        while (dist > 9.9999997473787516355514526367188e-06f)
        {
            int _1947 = 0;
            float _1950 = dist;
            float3 param_1 = ro;
            float3 param_2 = _1196;
            float3 param_3 = _1213;
            uint param_4 = _1231_g_params.node_index;
            hit_data_t _1959 = { 0, _1948, _1949, dist, _1951, _1952 };
            hit_data_t param_5 = _1959;
            bool _1244 = Traverse_MacroTree_WithStack(param_1, param_2, param_3, param_4, param_5);
            _1947 = param_5.mask;
            _1948 = param_5.obj_index;
            _1949 = param_5.prim_index;
            _1950 = param_5.t;
            _1951 = param_5.u;
            _1952 = param_5.v;
            bool _1255;
            if (!_1244)
            {
                _1255 = depth > _1231_g_params.max_transp_depth;
            }
            else
            {
                _1255 = _1244;
            }
            if (_1255)
            {
                _1830 = true;
                _1827 = 0.0f.xxx;
                break;
            }
            if (_1947 == 0)
            {
                _1830 = true;
                _1827 = rc;
                break;
            }
            bool _1270 = param_5.prim_index < 0;
            if (_1270)
            {
                _1273 = (-1) - param_5.prim_index;
            }
            else
            {
                _1273 = param_5.prim_index;
            }
            uint _1284 = uint(_1273);
            uint _1327 = _844.Load(_1284 * 4 + 0) * 3u;
            vertex_t _1333;
            [unroll]
            for (int _24ident = 0; _24ident < 3; _24ident++)
            {
                _1333.p[_24ident] = asfloat(_1321.Load(_24ident * 4 + _1325.Load(_1327 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _25ident = 0; _25ident < 3; _25ident++)
            {
                _1333.n[_25ident] = asfloat(_1321.Load(_25ident * 4 + _1325.Load(_1327 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _26ident = 0; _26ident < 3; _26ident++)
            {
                _1333.b[_26ident] = asfloat(_1321.Load(_26ident * 4 + _1325.Load(_1327 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _27ident = 0; _27ident < 2; _27ident++)
            {
                [unroll]
                for (int _28ident = 0; _28ident < 2; _28ident++)
                {
                    _1333.t[_27ident][_28ident] = asfloat(_1321.Load(_28ident * 4 + _27ident * 8 + _1325.Load(_1327 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1381;
            [unroll]
            for (int _29ident = 0; _29ident < 3; _29ident++)
            {
                _1381.p[_29ident] = asfloat(_1321.Load(_29ident * 4 + _1325.Load((_1327 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _30ident = 0; _30ident < 3; _30ident++)
            {
                _1381.n[_30ident] = asfloat(_1321.Load(_30ident * 4 + _1325.Load((_1327 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _31ident = 0; _31ident < 3; _31ident++)
            {
                _1381.b[_31ident] = asfloat(_1321.Load(_31ident * 4 + _1325.Load((_1327 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _32ident = 0; _32ident < 2; _32ident++)
            {
                [unroll]
                for (int _33ident = 0; _33ident < 2; _33ident++)
                {
                    _1381.t[_32ident][_33ident] = asfloat(_1321.Load(_33ident * 4 + _32ident * 8 + _1325.Load((_1327 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1427;
            [unroll]
            for (int _34ident = 0; _34ident < 3; _34ident++)
            {
                _1427.p[_34ident] = asfloat(_1321.Load(_34ident * 4 + _1325.Load((_1327 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _35ident = 0; _35ident < 3; _35ident++)
            {
                _1427.n[_35ident] = asfloat(_1321.Load(_35ident * 4 + _1325.Load((_1327 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _1427.b[_36ident] = asfloat(_1321.Load(_36ident * 4 + _1325.Load((_1327 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _37ident = 0; _37ident < 2; _37ident++)
            {
                [unroll]
                for (int _38ident = 0; _38ident < 2; _38ident++)
                {
                    _1427.t[_37ident][_38ident] = asfloat(_1321.Load(_38ident * 4 + _37ident * 8 + _1325.Load((_1327 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            float2 _1499 = ((float2(_1333.t[0][0], _1333.t[0][1]) * ((1.0f - param_5.u) - param_5.v)) + (float2(_1381.t[0][0], _1381.t[0][1]) * param_5.u)) + (float2(_1427.t[0][0], _1427.t[0][1]) * param_5.v);
            g_stack[gl_LocalInvocationIndex][0] = (_1270 ? (_853.Load(_844.Load(_1284 * 4 + 0) * 4 + 0) & 65535u) : ((_853.Load(_844.Load(_1284 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u;
            g_stack[gl_LocalInvocationIndex][1] = 1065353216u;
            int stack_size = 1;
            float3 throughput = 0.0f.xxx;
            for (;;)
            {
                int _1513 = stack_size;
                stack_size = _1513 - 1;
                if (_1513 != 0)
                {
                    int _1529 = stack_size;
                    int _1530 = 2 * _1529;
                    material_t _1536;
                    [unroll]
                    for (int _39ident = 0; _39ident < 5; _39ident++)
                    {
                        _1536.textures[_39ident] = _1527.Load(_39ident * 4 + g_stack[gl_LocalInvocationIndex][_1530] * 80 + 0);
                    }
                    [unroll]
                    for (int _40ident = 0; _40ident < 3; _40ident++)
                    {
                        _1536.base_color[_40ident] = asfloat(_1527.Load(_40ident * 4 + g_stack[gl_LocalInvocationIndex][_1530] * 80 + 20));
                    }
                    _1536.flags = _1527.Load(g_stack[gl_LocalInvocationIndex][_1530] * 80 + 32);
                    _1536.type = _1527.Load(g_stack[gl_LocalInvocationIndex][_1530] * 80 + 36);
                    _1536.tangent_rotation_or_strength = asfloat(_1527.Load(g_stack[gl_LocalInvocationIndex][_1530] * 80 + 40));
                    _1536.roughness_and_anisotropic = _1527.Load(g_stack[gl_LocalInvocationIndex][_1530] * 80 + 44);
                    _1536.int_ior = asfloat(_1527.Load(g_stack[gl_LocalInvocationIndex][_1530] * 80 + 48));
                    _1536.ext_ior = asfloat(_1527.Load(g_stack[gl_LocalInvocationIndex][_1530] * 80 + 52));
                    _1536.sheen_and_sheen_tint = _1527.Load(g_stack[gl_LocalInvocationIndex][_1530] * 80 + 56);
                    _1536.tint_and_metallic = _1527.Load(g_stack[gl_LocalInvocationIndex][_1530] * 80 + 60);
                    _1536.transmission_and_transmission_roughness = _1527.Load(g_stack[gl_LocalInvocationIndex][_1530] * 80 + 64);
                    _1536.specular_and_specular_tint = _1527.Load(g_stack[gl_LocalInvocationIndex][_1530] * 80 + 68);
                    _1536.clearcoat_and_clearcoat_roughness = _1527.Load(g_stack[gl_LocalInvocationIndex][_1530] * 80 + 72);
                    _1536.normal_map_strength_unorm = _1527.Load(g_stack[gl_LocalInvocationIndex][_1530] * 80 + 76);
                    uint _1594 = g_stack[gl_LocalInvocationIndex][_1530 + 1];
                    float _1595 = asfloat(_1594);
                    if (_1536.type == 4u)
                    {
                        float mix_val = _1536.tangent_rotation_or_strength;
                        if (_1536.textures[1] != 4294967295u)
                        {
                            mix_val *= SampleBilinear(_1536.textures[1], _1499, 0).x;
                        }
                        int _1620 = 2 * stack_size;
                        g_stack[gl_LocalInvocationIndex][_1620] = _1536.textures[3];
                        g_stack[gl_LocalInvocationIndex][_1620 + 1] = asuint(_1595 * (1.0f - mix_val));
                        int _1639 = 2 * (stack_size + 1);
                        g_stack[gl_LocalInvocationIndex][_1639] = _1536.textures[4];
                        g_stack[gl_LocalInvocationIndex][_1639 + 1] = asuint(_1595 * mix_val);
                        stack_size += 2;
                    }
                    else
                    {
                        if (_1536.type == 5u)
                        {
                            throughput += (float3(_1536.base_color[0], _1536.base_color[1], _1536.base_color[2]) * _1595);
                        }
                    }
                    continue;
                }
                else
                {
                    break;
                }
            }
            float3 _1673 = rc;
            float3 _1674 = _1673 * throughput;
            rc = _1674;
            if (lum(_1674) < 1.0000000116860974230803549289703e-07f)
            {
                break;
            }
            float _1684 = _1950 + 9.9999997473787516355514526367188e-06f;
            ro += (_1196 * _1684);
            dist -= _1684;
            depth++;
        }
        if (_1830)
        {
            break;
        }
        _1830 = true;
        _1827 = rc;
        break;
    } while(false);
    return _1827;
}

void comp_main()
{
    do
    {
        int _1707 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_1707) >= _1713.Load(12))
        {
            break;
        }
        shadow_ray_t _1732;
        [unroll]
        for (int _41ident = 0; _41ident < 3; _41ident++)
        {
            _1732.o[_41ident] = asfloat(_1728.Load(_41ident * 4 + _1707 * 48 + 0));
        }
        _1732.depth = int(_1728.Load(_1707 * 48 + 12));
        [unroll]
        for (int _42ident = 0; _42ident < 3; _42ident++)
        {
            _1732.d[_42ident] = asfloat(_1728.Load(_42ident * 4 + _1707 * 48 + 16));
        }
        _1732.dist = asfloat(_1728.Load(_1707 * 48 + 28));
        [unroll]
        for (int _43ident = 0; _43ident < 3; _43ident++)
        {
            _1732.c[_43ident] = asfloat(_1728.Load(_43ident * 4 + _1707 * 48 + 32));
        }
        _1732.xy = int(_1728.Load(_1707 * 48 + 44));
        float _1885[3] = { _1732.c[0], _1732.c[1], _1732.c[2] };
        float _1878[3] = { _1732.d[0], _1732.d[1], _1732.d[2] };
        float _1871[3] = { _1732.o[0], _1732.o[1], _1732.o[2] };
        shadow_ray_t _1864 = { _1871, _1732.depth, _1878, _1732.dist, _1885, _1732.xy };
        shadow_ray_t param = _1864;
        float param_1 = _1732.dist;
        float3 _1769 = IntersectSceneShadow(param, param_1);
        if (lum(_1769) > 0.0f)
        {
            int2 _1794 = int2((_1732.xy >> 16) & 65535, _1732.xy & 65535);
            g_out_img[_1794] = float4(g_out_img[_1794].xyz + _1769, 1.0f);
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
