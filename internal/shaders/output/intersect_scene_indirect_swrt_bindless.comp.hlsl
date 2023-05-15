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

ByteAddressBuffer _555 : register(t3, space0);
ByteAddressBuffer _770 : register(t7, space0);
ByteAddressBuffer _1017 : register(t9, space0);
ByteAddressBuffer _1021 : register(t10, space0);
ByteAddressBuffer _1042 : register(t8, space0);
ByteAddressBuffer _1088 : register(t11, space0);
ByteAddressBuffer _1181 : register(t14, space0);
RWByteAddressBuffer _1197 : register(u12, space0);
ByteAddressBuffer _1361 : register(t4, space0);
ByteAddressBuffer _1411 : register(t5, space0);
ByteAddressBuffer _1448 : register(t6, space0);
ByteAddressBuffer _1569 : register(t1, space0);
ByteAddressBuffer _1573 : register(t2, space0);
ByteAddressBuffer _1751 : register(t15, space0);
RWByteAddressBuffer _2057 : register(u0, space0);
cbuffer UniformParams
{
    Params _1245_g_params : packoffset(c0);
};

Texture2D<float4> g_textures[] : register(t0, space1);
SamplerState _g_textures_sampler[] : register(s0, space1);

static uint3 gl_WorkGroupID;
static uint gl_LocalInvocationIndex;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint gl_LocalInvocationIndex : SV_GroupIndex;
};

groupshared uint g_stack[64][48];

uint2 spvTextureSize(Texture2D<float4> Tex, uint Level, out uint Param)
{
    uint2 ret;
    Tex.GetDimensions(Level, ret.x, ret.y, Param);
    return ret;
}

float3 safe_invert(float3 v)
{
    float3 inv_v = 1.0f.xxx / v;
    bool _153 = v.x <= 1.0000000116860974230803549289703e-07f;
    bool _160;
    if (_153)
    {
        _160 = v.x >= 0.0f;
    }
    else
    {
        _160 = _153;
    }
    if (_160)
    {
        float3 _2335 = inv_v;
        _2335.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _2335;
    }
    else
    {
        bool _169 = v.x >= (-1.0000000116860974230803549289703e-07f);
        bool _175;
        if (_169)
        {
            _175 = v.x < 0.0f;
        }
        else
        {
            _175 = _169;
        }
        if (_175)
        {
            float3 _2333 = inv_v;
            _2333.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _2333;
        }
    }
    bool _183 = v.y <= 1.0000000116860974230803549289703e-07f;
    bool _189;
    if (_183)
    {
        _189 = v.y >= 0.0f;
    }
    else
    {
        _189 = _183;
    }
    if (_189)
    {
        float3 _2339 = inv_v;
        _2339.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _2339;
    }
    else
    {
        bool _196 = v.y >= (-1.0000000116860974230803549289703e-07f);
        bool _202;
        if (_196)
        {
            _202 = v.y < 0.0f;
        }
        else
        {
            _202 = _196;
        }
        if (_202)
        {
            float3 _2337 = inv_v;
            _2337.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _2337;
        }
    }
    bool _209 = v.z <= 1.0000000116860974230803549289703e-07f;
    bool _215;
    if (_209)
    {
        _215 = v.z >= 0.0f;
    }
    else
    {
        _215 = _209;
    }
    if (_215)
    {
        float3 _2343 = inv_v;
        _2343.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _2343;
    }
    else
    {
        bool _222 = v.z >= (-1.0000000116860974230803549289703e-07f);
        bool _228;
        if (_222)
        {
            _228 = v.z < 0.0f;
        }
        else
        {
            _228 = _222;
        }
        if (_228)
        {
            float3 _2341 = inv_v;
            _2341.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _2341;
        }
    }
    return inv_v;
}

int hash(int x)
{
    uint _110 = uint(x);
    uint _117 = ((_110 >> uint(16)) ^ _110) * 73244475u;
    uint _122 = ((_117 >> uint(16)) ^ _117) * 73244475u;
    return int((_122 >> uint(16)) ^ _122);
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
    float _648 = mad(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float _656 = mad(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float _671 = mad(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    float _678 = mad(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    float _695 = mad(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    float _702 = mad(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    float _707 = max(max(min(_648, _656), min(_671, _678)), min(_695, _702));
    float _715 = min(min(max(_648, _656), max(_671, _678)), max(_695, _702)) * 1.0000002384185791015625f;
    return ((_707 <= _715) && (_707 <= t)) && (_715 > 0.0f);
}

void IntersectTri(float3 ro, float3 rd, tri_accel_t tri, uint prim_index, inout hit_data_t inter)
{
    do
    {
        float _425 = dot(rd, tri.n_plane.xyz);
        float _434 = tri.n_plane.w - dot(ro, tri.n_plane.xyz);
        if (sign(_434) != sign(mad(_425, inter.t, -_434)))
        {
            break;
        }
        float3 _455 = (ro * _425) + (rd * _434);
        float _466 = mad(_425, tri.u_plane.w, dot(_455, tri.u_plane.xyz));
        float _471 = _425 - _466;
        if (sign(_466) != sign(_471))
        {
            break;
        }
        float _488 = mad(_425, tri.v_plane.w, dot(_455, tri.v_plane.xyz));
        if (sign(_488) != sign(_471 - _488))
        {
            break;
        }
        float _503 = 1.0f / _425;
        inter.mask = -1;
        int _508;
        if (_425 < 0.0f)
        {
            _508 = int(prim_index);
        }
        else
        {
            _508 = (-1) - int(prim_index);
        }
        inter.prim_index = _508;
        inter.t = _434 * _503;
        inter.u = _466 * _503;
        inter.v = _488 * _503;
        break;
    } while(false);
}

void IntersectTris_ClosestHit(float3 ro, float3 rd, int tri_start, int tri_end, int obj_index, inout hit_data_t out_inter)
{
    int _2267 = 0;
    int _2268 = obj_index;
    float _2270 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _2269;
    float _2271;
    float _2272;
    for (int i = tri_start; i < tri_end; )
    {
        param = ro;
        param_1 = rd;
        tri_accel_t _566;
        _566.n_plane = asfloat(_555.Load4(i * 48 + 0));
        _566.u_plane = asfloat(_555.Load4(i * 48 + 16));
        _566.v_plane = asfloat(_555.Load4(i * 48 + 32));
        param_2.n_plane = _566.n_plane;
        param_2.u_plane = _566.u_plane;
        param_2.v_plane = _566.v_plane;
        param_3 = uint(i);
        hit_data_t _2279 = { _2267, _2268, _2269, _2270, _2271, _2272 };
        param_4 = _2279;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _2267 = param_4.mask;
        _2268 = param_4.obj_index;
        _2269 = param_4.prim_index;
        _2270 = param_4.t;
        _2271 = param_4.u;
        _2272 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _2267;
    int _589;
    if (_2267 != 0)
    {
        _589 = _2268;
    }
    else
    {
        _589 = out_inter.obj_index;
    }
    out_inter.obj_index = _589;
    int _602;
    if (_2267 != 0)
    {
        _602 = _2269;
    }
    else
    {
        _602 = out_inter.prim_index;
    }
    out_inter.prim_index = _602;
    out_inter.t = _2270;
    float _618;
    if (_2267 != 0)
    {
        _618 = _2271;
    }
    else
    {
        _618 = out_inter.u;
    }
    out_inter.u = _618;
    float _631;
    if (_2267 != 0)
    {
        _631 = _2272;
    }
    else
    {
        _631 = out_inter.v;
    }
    out_inter.v = _631;
}

void Traverse_BLAS_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    float3 _732 = (-inv_d) * ro;
    uint _734 = stack_size;
    uint _744 = stack_size;
    stack_size = _744 + uint(1);
    g_stack[gl_LocalInvocationIndex][_744] = node_index;
    uint _818;
    uint _842;
    while (stack_size != _734)
    {
        uint _759 = stack_size;
        uint _760 = _759 - uint(1);
        stack_size = _760;
        bvh_node_t _774;
        _774.bbox_min = asfloat(_770.Load4(g_stack[gl_LocalInvocationIndex][_760] * 32 + 0));
        _774.bbox_max = asfloat(_770.Load4(g_stack[gl_LocalInvocationIndex][_760] * 32 + 16));
        float3 param = inv_d;
        float3 param_1 = _732;
        float param_2 = inter.t;
        float3 param_3 = _774.bbox_min.xyz;
        float3 param_4 = _774.bbox_max.xyz;
        if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
        {
            continue;
        }
        uint _801 = asuint(_774.bbox_min.w);
        if ((_801 & 2147483648u) == 0u)
        {
            uint _808 = stack_size;
            stack_size = _808 + uint(1);
            uint _812 = asuint(_774.bbox_max.w);
            uint _814 = _812 >> uint(30);
            if (rd[_814] < 0.0f)
            {
                _818 = _801;
            }
            else
            {
                _818 = _812 & 1073741823u;
            }
            g_stack[gl_LocalInvocationIndex][_808] = _818;
            uint _833 = stack_size;
            stack_size = _833 + uint(1);
            if (rd[_814] < 0.0f)
            {
                _842 = _812 & 1073741823u;
            }
            else
            {
                _842 = _801;
            }
            g_stack[gl_LocalInvocationIndex][_833] = _842;
        }
        else
        {
            int _862 = int(_801 & 2147483647u);
            float3 param_5 = ro;
            float3 param_6 = rd;
            int param_7 = _862;
            int param_8 = _862 + asint(_774.bbox_max.w);
            int param_9 = obj_index;
            hit_data_t param_10 = inter;
            IntersectTris_ClosestHit(param_5, param_6, param_7, param_8, param_9, param_10);
            inter = param_10;
        }
    }
}

void Traverse_TLAS_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    float3 _887 = (-orig_inv_rd) * orig_ro;
    uint stack_size = 1u;
    g_stack[gl_LocalInvocationIndex][0u] = node_index;
    uint _952;
    uint _975;
    while (stack_size != 0u)
    {
        uint _903 = stack_size;
        uint _904 = _903 - uint(1);
        stack_size = _904;
        bvh_node_t _910;
        _910.bbox_min = asfloat(_770.Load4(g_stack[gl_LocalInvocationIndex][_904] * 32 + 0));
        _910.bbox_max = asfloat(_770.Load4(g_stack[gl_LocalInvocationIndex][_904] * 32 + 16));
        float3 param = orig_inv_rd;
        float3 param_1 = _887;
        float param_2 = inter.t;
        float3 param_3 = _910.bbox_min.xyz;
        float3 param_4 = _910.bbox_max.xyz;
        if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
        {
            continue;
        }
        uint _937 = asuint(_910.bbox_min.w);
        if ((_937 & 2147483648u) == 0u)
        {
            uint _943 = stack_size;
            stack_size = _943 + uint(1);
            uint _947 = asuint(_910.bbox_max.w);
            uint _948 = _947 >> uint(30);
            if (orig_rd[_948] < 0.0f)
            {
                _952 = _937;
            }
            else
            {
                _952 = _947 & 1073741823u;
            }
            g_stack[gl_LocalInvocationIndex][_943] = _952;
            uint _966 = stack_size;
            stack_size = _966 + uint(1);
            if (orig_rd[_948] < 0.0f)
            {
                _975 = _947 & 1073741823u;
            }
            else
            {
                _975 = _937;
            }
            g_stack[gl_LocalInvocationIndex][_966] = _975;
        }
        else
        {
            uint _993 = _937 & 2147483647u;
            uint _997 = asuint(_910.bbox_max.w);
            for (uint i = _993; i < (_993 + _997); i++)
            {
                mesh_instance_t _1028;
                _1028.bbox_min = asfloat(_1017.Load4(_1021.Load(i * 4 + 0) * 32 + 0));
                _1028.bbox_max = asfloat(_1017.Load4(_1021.Load(i * 4 + 0) * 32 + 16));
                mesh_t _1048;
                [unroll]
                for (int _29ident = 0; _29ident < 3; _29ident++)
                {
                    _1048.bbox_min[_29ident] = asfloat(_1042.Load(_29ident * 4 + asuint(_1028.bbox_max.w) * 48 + 0));
                }
                [unroll]
                for (int _30ident = 0; _30ident < 3; _30ident++)
                {
                    _1048.bbox_max[_30ident] = asfloat(_1042.Load(_30ident * 4 + asuint(_1028.bbox_max.w) * 48 + 12));
                }
                _1048.node_index = _1042.Load(asuint(_1028.bbox_max.w) * 48 + 24);
                _1048.node_count = _1042.Load(asuint(_1028.bbox_max.w) * 48 + 28);
                _1048.tris_index = _1042.Load(asuint(_1028.bbox_max.w) * 48 + 32);
                _1048.tris_count = _1042.Load(asuint(_1028.bbox_max.w) * 48 + 36);
                _1048.vert_index = _1042.Load(asuint(_1028.bbox_max.w) * 48 + 40);
                _1048.vert_count = _1042.Load(asuint(_1028.bbox_max.w) * 48 + 44);
                transform_t _1094;
                _1094.xform = asfloat(uint4x4(_1088.Load4(asuint(_1028.bbox_min.w) * 128 + 0), _1088.Load4(asuint(_1028.bbox_min.w) * 128 + 16), _1088.Load4(asuint(_1028.bbox_min.w) * 128 + 32), _1088.Load4(asuint(_1028.bbox_min.w) * 128 + 48)));
                _1094.inv_xform = asfloat(uint4x4(_1088.Load4(asuint(_1028.bbox_min.w) * 128 + 64), _1088.Load4(asuint(_1028.bbox_min.w) * 128 + 80), _1088.Load4(asuint(_1028.bbox_min.w) * 128 + 96), _1088.Load4(asuint(_1028.bbox_min.w) * 128 + 112)));
                float3 param_5 = orig_inv_rd;
                float3 param_6 = _887;
                float param_7 = inter.t;
                float3 param_8 = _1028.bbox_min.xyz;
                float3 param_9 = _1028.bbox_max.xyz;
                if (!_bbox_test_fma(param_5, param_6, param_7, param_8, param_9))
                {
                    continue;
                }
                float3 _1139 = mul(float4(orig_rd, 0.0f), _1094.inv_xform).xyz;
                float3 param_10 = _1139;
                float3 param_11 = mul(float4(orig_ro, 1.0f), _1094.inv_xform).xyz;
                float3 param_12 = _1139;
                float3 param_13 = safe_invert(param_10);
                int param_14 = int(_1021.Load(i * 4 + 0));
                uint param_15 = _1048.node_index;
                uint param_16 = stack_size;
                hit_data_t param_17 = inter;
                Traverse_BLAS_WithStack(param_11, param_12, param_13, param_14, param_15, param_16, param_17);
                inter = param_17;
            }
        }
    }
}

float3 YCoCg_to_RGB(float4 col)
{
    float _283 = mad(col.z, 31.875f, 1.0f);
    float _293 = (col.x - 0.501960813999176025390625f) / _283;
    float _299 = (col.y - 0.501960813999176025390625f) / _283;
    return float3((col.w + _293) - _299, col.w + _299, (col.w - _293) - _299);
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
    uint _357 = index & 16777215u;
    uint _362_dummy_parameter;
    float4 res = g_textures[NonUniformResourceIndex(_357)].SampleLevel(_g_textures_sampler[NonUniformResourceIndex(_357)], uvs + ((rand - 0.5f.xx) / float2(int2(spvTextureSize(g_textures[NonUniformResourceIndex(_357)], uint(lod), _362_dummy_parameter)))), float(lod));
    bool _382;
    if (maybe_YCoCg)
    {
        _382 = (index & 67108864u) != 0u;
    }
    else
    {
        _382 = maybe_YCoCg;
    }
    if (_382)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _400;
    if (maybe_SRGB)
    {
        _400 = (index & 16777216u) != 0u;
    }
    else
    {
        _400 = maybe_SRGB;
    }
    if (_400)
    {
        float3 param_1 = res.xyz;
        float3 _406 = srgb_to_rgb(param_1);
        float4 _2359 = res;
        _2359.x = _406.x;
        float4 _2361 = _2359;
        _2361.y = _406.y;
        float4 _2363 = _2361;
        _2363.z = _406.z;
        res = _2363;
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
        int _1175 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_1175) >= _1181.Load(4))
        {
            break;
        }
        float3 ro = float3(asfloat(_1197.Load(_1175 * 72 + 0)), asfloat(_1197.Load(_1175 * 72 + 4)), asfloat(_1197.Load(_1175 * 72 + 8)));
        float _1225 = asfloat(_1197.Load(_1175 * 72 + 12));
        float _1228 = asfloat(_1197.Load(_1175 * 72 + 16));
        float _1231 = asfloat(_1197.Load(_1175 * 72 + 20));
        float3 _1232 = float3(_1225, _1228, _1231);
        float3 param = _1232;
        float3 _1236 = safe_invert(param);
        int _2095 = 0;
        int _2097 = 0;
        int _2096 = 0;
        float _2098 = _1245_g_params.inter_t;
        float _2100 = 0.0f;
        float _2099 = 0.0f;
        uint param_1 = uint(hash(int(_1197.Load(_1175 * 72 + 64))));
        float _1260 = construct_float(param_1);
        uint param_2 = uint(hash(hash(int(_1197.Load(_1175 * 72 + 64)))));
        float _1268 = construct_float(param_2);
        ray_data_t _1277;
        [unroll]
        for (int _31ident = 0; _31ident < 3; _31ident++)
        {
            _1277.o[_31ident] = asfloat(_1197.Load(_31ident * 4 + _1175 * 72 + 0));
        }
        [unroll]
        for (int _32ident = 0; _32ident < 3; _32ident++)
        {
            _1277.d[_32ident] = asfloat(_1197.Load(_32ident * 4 + _1175 * 72 + 12));
        }
        _1277.pdf = asfloat(_1197.Load(_1175 * 72 + 24));
        [unroll]
        for (int _33ident = 0; _33ident < 3; _33ident++)
        {
            _1277.c[_33ident] = asfloat(_1197.Load(_33ident * 4 + _1175 * 72 + 28));
        }
        [unroll]
        for (int _34ident = 0; _34ident < 4; _34ident++)
        {
            _1277.ior[_34ident] = asfloat(_1197.Load(_34ident * 4 + _1175 * 72 + 40));
        }
        _1277.cone_width = asfloat(_1197.Load(_1175 * 72 + 56));
        _1277.cone_spread = asfloat(_1197.Load(_1175 * 72 + 60));
        _1277.xy = int(_1197.Load(_1175 * 72 + 64));
        _1277.depth = int(_1197.Load(_1175 * 72 + 68));
        float _2207[4] = { _1277.ior[0], _1277.ior[1], _1277.ior[2], _1277.ior[3] };
        float _2198[3] = { _1277.c[0], _1277.c[1], _1277.c[2] };
        float _2191[3] = { _1277.d[0], _1277.d[1], _1277.d[2] };
        float _2184[3] = { _1277.o[0], _1277.o[1], _1277.o[2] };
        ray_data_t _2139 = { _2184, _2191, _1277.pdf, _2198, _2207, _1277.cone_width, _1277.cone_spread, _1277.xy, _1277.depth };
        int rand_index = _1245_g_params.hi + (total_depth(_2139) * 9);
        int _1392;
        float _1947;
        for (;;)
        {
            float _1338 = _2098;
            float3 param_3 = ro;
            float3 param_4 = _1232;
            float3 param_5 = _1236;
            uint param_6 = _1245_g_params.node_index;
            hit_data_t _2107 = { _2095, _2096, _2097, _1338, _2099, _2100 };
            hit_data_t param_7 = _2107;
            Traverse_TLAS_WithStack(param_3, param_4, param_5, param_6, param_7);
            _2095 = param_7.mask;
            _2096 = param_7.obj_index;
            _2097 = param_7.prim_index;
            _2098 = param_7.t;
            _2099 = param_7.u;
            _2100 = param_7.v;
            if (param_7.prim_index < 0)
            {
                _2097 = (-1) - int(_1361.Load(((-1) - _2097) * 4 + 0));
            }
            else
            {
                _2097 = int(_1361.Load(_2097 * 4 + 0));
            }
            if (_2095 == 0)
            {
                break;
            }
            bool _1389 = _2097 < 0;
            if (_1389)
            {
                _1392 = (-1) - _2097;
            }
            else
            {
                _1392 = _2097;
            }
            uint _1403 = uint(_1392);
            bool _1405 = !_1389;
            bool _1420;
            if (_1405)
            {
                _1420 = ((_1411.Load(_1403 * 4 + 0) >> 16u) & 32768u) != 0u;
            }
            else
            {
                _1420 = _1405;
            }
            bool _1433;
            if (!_1420)
            {
                bool _1432;
                if (_1389)
                {
                    _1432 = (_1411.Load(_1403 * 4 + 0) & 32768u) != 0u;
                }
                else
                {
                    _1432 = _1389;
                }
                _1433 = _1432;
            }
            else
            {
                _1433 = _1420;
            }
            if (_1433)
            {
                break;
            }
            material_t _1457;
            [unroll]
            for (int _35ident = 0; _35ident < 5; _35ident++)
            {
                _1457.textures[_35ident] = _1448.Load(_35ident * 4 + ((_1411.Load(_1403 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _1457.base_color[_36ident] = asfloat(_1448.Load(_36ident * 4 + ((_1411.Load(_1403 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
            }
            _1457.flags = _1448.Load(((_1411.Load(_1403 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
            _1457.type = _1448.Load(((_1411.Load(_1403 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
            _1457.tangent_rotation_or_strength = asfloat(_1448.Load(((_1411.Load(_1403 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
            _1457.roughness_and_anisotropic = _1448.Load(((_1411.Load(_1403 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
            _1457.ior = asfloat(_1448.Load(((_1411.Load(_1403 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
            _1457.sheen_and_sheen_tint = _1448.Load(((_1411.Load(_1403 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
            _1457.tint_and_metallic = _1448.Load(((_1411.Load(_1403 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
            _1457.transmission_and_transmission_roughness = _1448.Load(((_1411.Load(_1403 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
            _1457.specular_and_specular_tint = _1448.Load(((_1411.Load(_1403 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
            _1457.clearcoat_and_clearcoat_roughness = _1448.Load(((_1411.Load(_1403 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
            _1457.normal_map_strength_unorm = _1448.Load(((_1411.Load(_1403 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
            uint _2209 = _1457.textures[1];
            uint _2211 = _1457.textures[3];
            uint _2212 = _1457.textures[4];
            float _2213 = _1457.base_color[0];
            float _2214 = _1457.base_color[1];
            float _2215 = _1457.base_color[2];
            uint _2143 = _1457.type;
            float _2144 = _1457.tangent_rotation_or_strength;
            if (_1389)
            {
                material_t _1512;
                [unroll]
                for (int _37ident = 0; _37ident < 5; _37ident++)
                {
                    _1512.textures[_37ident] = _1448.Load(_37ident * 4 + (_1411.Load(_1403 * 4 + 0) & 16383u) * 76 + 0);
                }
                [unroll]
                for (int _38ident = 0; _38ident < 3; _38ident++)
                {
                    _1512.base_color[_38ident] = asfloat(_1448.Load(_38ident * 4 + (_1411.Load(_1403 * 4 + 0) & 16383u) * 76 + 20));
                }
                _1512.flags = _1448.Load((_1411.Load(_1403 * 4 + 0) & 16383u) * 76 + 32);
                _1512.type = _1448.Load((_1411.Load(_1403 * 4 + 0) & 16383u) * 76 + 36);
                _1512.tangent_rotation_or_strength = asfloat(_1448.Load((_1411.Load(_1403 * 4 + 0) & 16383u) * 76 + 40));
                _1512.roughness_and_anisotropic = _1448.Load((_1411.Load(_1403 * 4 + 0) & 16383u) * 76 + 44);
                _1512.ior = asfloat(_1448.Load((_1411.Load(_1403 * 4 + 0) & 16383u) * 76 + 48));
                _1512.sheen_and_sheen_tint = _1448.Load((_1411.Load(_1403 * 4 + 0) & 16383u) * 76 + 52);
                _1512.tint_and_metallic = _1448.Load((_1411.Load(_1403 * 4 + 0) & 16383u) * 76 + 56);
                _1512.transmission_and_transmission_roughness = _1448.Load((_1411.Load(_1403 * 4 + 0) & 16383u) * 76 + 60);
                _1512.specular_and_specular_tint = _1448.Load((_1411.Load(_1403 * 4 + 0) & 16383u) * 76 + 64);
                _1512.clearcoat_and_clearcoat_roughness = _1448.Load((_1411.Load(_1403 * 4 + 0) & 16383u) * 76 + 68);
                _1512.normal_map_strength_unorm = _1448.Load((_1411.Load(_1403 * 4 + 0) & 16383u) * 76 + 72);
                _2209 = _1512.textures[1];
                _2211 = _1512.textures[3];
                _2212 = _1512.textures[4];
                _2213 = _1512.base_color[0];
                _2214 = _1512.base_color[1];
                _2215 = _1512.base_color[2];
                _2143 = _1512.type;
                _2144 = _1512.tangent_rotation_or_strength;
            }
            uint _1575 = _1403 * 3u;
            vertex_t _1581;
            [unroll]
            for (int _39ident = 0; _39ident < 3; _39ident++)
            {
                _1581.p[_39ident] = asfloat(_1569.Load(_39ident * 4 + _1573.Load(_1575 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _1581.n[_40ident] = asfloat(_1569.Load(_40ident * 4 + _1573.Load(_1575 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 3; _41ident++)
            {
                _1581.b[_41ident] = asfloat(_1569.Load(_41ident * 4 + _1573.Load(_1575 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _42ident = 0; _42ident < 2; _42ident++)
            {
                [unroll]
                for (int _43ident = 0; _43ident < 2; _43ident++)
                {
                    _1581.t[_42ident][_43ident] = asfloat(_1569.Load(_43ident * 4 + _42ident * 8 + _1573.Load(_1575 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1629;
            [unroll]
            for (int _44ident = 0; _44ident < 3; _44ident++)
            {
                _1629.p[_44ident] = asfloat(_1569.Load(_44ident * 4 + _1573.Load((_1575 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _45ident = 0; _45ident < 3; _45ident++)
            {
                _1629.n[_45ident] = asfloat(_1569.Load(_45ident * 4 + _1573.Load((_1575 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _46ident = 0; _46ident < 3; _46ident++)
            {
                _1629.b[_46ident] = asfloat(_1569.Load(_46ident * 4 + _1573.Load((_1575 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _47ident = 0; _47ident < 2; _47ident++)
            {
                [unroll]
                for (int _48ident = 0; _48ident < 2; _48ident++)
                {
                    _1629.t[_47ident][_48ident] = asfloat(_1569.Load(_48ident * 4 + _47ident * 8 + _1573.Load((_1575 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1675;
            [unroll]
            for (int _49ident = 0; _49ident < 3; _49ident++)
            {
                _1675.p[_49ident] = asfloat(_1569.Load(_49ident * 4 + _1573.Load((_1575 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _50ident = 0; _50ident < 3; _50ident++)
            {
                _1675.n[_50ident] = asfloat(_1569.Load(_50ident * 4 + _1573.Load((_1575 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _51ident = 0; _51ident < 3; _51ident++)
            {
                _1675.b[_51ident] = asfloat(_1569.Load(_51ident * 4 + _1573.Load((_1575 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _52ident = 0; _52ident < 2; _52ident++)
            {
                [unroll]
                for (int _53ident = 0; _53ident < 2; _53ident++)
                {
                    _1675.t[_52ident][_53ident] = asfloat(_1569.Load(_53ident * 4 + _52ident * 8 + _1573.Load((_1575 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            float2 _1746 = ((float2(_1581.t[0][0], _1581.t[0][1]) * ((1.0f - _2099) - _2100)) + (float2(_1629.t[0][0], _1629.t[0][1]) * _2099)) + (float2(_1675.t[0][0], _1675.t[0][1]) * _2100);
            float trans_r = frac(asfloat(_1751.Load(rand_index * 4 + 0)) + _1260);
            float2 _1777 = float2(frac(asfloat(_1751.Load((rand_index + 7) * 4 + 0)) + _1260), frac(asfloat(_1751.Load((rand_index + 8) * 4 + 0)) + _1268));
            while (_2143 == 4u)
            {
                float mix_val = _2144;
                if (_2209 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_2209, _1746, 0, _1777).x;
                }
                if (trans_r > mix_val)
                {
                    material_t _1811;
                    [unroll]
                    for (int _54ident = 0; _54ident < 5; _54ident++)
                    {
                        _1811.textures[_54ident] = _1448.Load(_54ident * 4 + _2211 * 76 + 0);
                    }
                    [unroll]
                    for (int _55ident = 0; _55ident < 3; _55ident++)
                    {
                        _1811.base_color[_55ident] = asfloat(_1448.Load(_55ident * 4 + _2211 * 76 + 20));
                    }
                    _1811.flags = _1448.Load(_2211 * 76 + 32);
                    _1811.type = _1448.Load(_2211 * 76 + 36);
                    _1811.tangent_rotation_or_strength = asfloat(_1448.Load(_2211 * 76 + 40));
                    _1811.roughness_and_anisotropic = _1448.Load(_2211 * 76 + 44);
                    _1811.ior = asfloat(_1448.Load(_2211 * 76 + 48));
                    _1811.sheen_and_sheen_tint = _1448.Load(_2211 * 76 + 52);
                    _1811.tint_and_metallic = _1448.Load(_2211 * 76 + 56);
                    _1811.transmission_and_transmission_roughness = _1448.Load(_2211 * 76 + 60);
                    _1811.specular_and_specular_tint = _1448.Load(_2211 * 76 + 64);
                    _1811.clearcoat_and_clearcoat_roughness = _1448.Load(_2211 * 76 + 68);
                    _1811.normal_map_strength_unorm = _1448.Load(_2211 * 76 + 72);
                    _2209 = _1811.textures[1];
                    _2211 = _1811.textures[3];
                    _2212 = _1811.textures[4];
                    _2213 = _1811.base_color[0];
                    _2214 = _1811.base_color[1];
                    _2215 = _1811.base_color[2];
                    _2143 = _1811.type;
                    _2144 = _1811.tangent_rotation_or_strength;
                    trans_r = (trans_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _1864;
                    [unroll]
                    for (int _56ident = 0; _56ident < 5; _56ident++)
                    {
                        _1864.textures[_56ident] = _1448.Load(_56ident * 4 + _2212 * 76 + 0);
                    }
                    [unroll]
                    for (int _57ident = 0; _57ident < 3; _57ident++)
                    {
                        _1864.base_color[_57ident] = asfloat(_1448.Load(_57ident * 4 + _2212 * 76 + 20));
                    }
                    _1864.flags = _1448.Load(_2212 * 76 + 32);
                    _1864.type = _1448.Load(_2212 * 76 + 36);
                    _1864.tangent_rotation_or_strength = asfloat(_1448.Load(_2212 * 76 + 40));
                    _1864.roughness_and_anisotropic = _1448.Load(_2212 * 76 + 44);
                    _1864.ior = asfloat(_1448.Load(_2212 * 76 + 48));
                    _1864.sheen_and_sheen_tint = _1448.Load(_2212 * 76 + 52);
                    _1864.tint_and_metallic = _1448.Load(_2212 * 76 + 56);
                    _1864.transmission_and_transmission_roughness = _1448.Load(_2212 * 76 + 60);
                    _1864.specular_and_specular_tint = _1448.Load(_2212 * 76 + 64);
                    _1864.clearcoat_and_clearcoat_roughness = _1448.Load(_2212 * 76 + 68);
                    _1864.normal_map_strength_unorm = _1448.Load(_2212 * 76 + 72);
                    _2209 = _1864.textures[1];
                    _2211 = _1864.textures[3];
                    _2212 = _1864.textures[4];
                    _2213 = _1864.base_color[0];
                    _2214 = _1864.base_color[1];
                    _2215 = _1864.base_color[2];
                    _2143 = _1864.type;
                    _2144 = _1864.tangent_rotation_or_strength;
                    trans_r /= mix_val;
                }
            }
            if (_2143 != 5u)
            {
                break;
            }
            float _1935 = max(asfloat(_1197.Load(_1175 * 72 + 28)), max(asfloat(_1197.Load(_1175 * 72 + 32)), asfloat(_1197.Load(_1175 * 72 + 36))));
            if ((int(_1197.Load(_1175 * 72 + 68)) >> 24) > _1245_g_params.min_transp_depth)
            {
                _1947 = max(0.0500000007450580596923828125f, 1.0f - _1935);
            }
            else
            {
                _1947 = 0.0f;
            }
            bool _1961 = (frac(asfloat(_1751.Load((rand_index + 6) * 4 + 0)) + _1260) < _1947) || (_1935 == 0.0f);
            bool _1973;
            if (!_1961)
            {
                _1973 = ((int(_1197.Load(_1175 * 72 + 68)) >> 24) + 1) >= _1245_g_params.max_transp_depth;
            }
            else
            {
                _1973 = _1961;
            }
            if (_1973)
            {
                _1197.Store(_1175 * 72 + 36, asuint(0.0f));
                _1197.Store(_1175 * 72 + 32, asuint(0.0f));
                _1197.Store(_1175 * 72 + 28, asuint(0.0f));
                break;
            }
            float _1987 = 1.0f - _1947;
            _1197.Store(_1175 * 72 + 28, asuint(asfloat(_1197.Load(_1175 * 72 + 28)) * (_2213 / _1987)));
            _1197.Store(_1175 * 72 + 32, asuint(asfloat(_1197.Load(_1175 * 72 + 32)) * (_2214 / _1987)));
            _1197.Store(_1175 * 72 + 36, asuint(asfloat(_1197.Load(_1175 * 72 + 36)) * (_2215 / _1987)));
            ro += (_1232 * (_2098 + 9.9999997473787516355514526367188e-06f));
            _2095 = 0;
            _2098 = _1338 - _2098;
            _1197.Store(_1175 * 72 + 68, uint(int(_1197.Load(_1175 * 72 + 68)) + 16777216));
            rand_index += 9;
            continue;
        }
        float _2039 = asfloat(_1197.Load(_1175 * 72 + 0));
        float _2042 = asfloat(_1197.Load(_1175 * 72 + 4));
        float _2045 = asfloat(_1197.Load(_1175 * 72 + 8));
        float _2050 = _2098;
        float _2051 = _2050 + distance(float3(_2039, _2042, _2045), ro);
        _2098 = _2051;
        _2057.Store(_1175 * 24 + 0, uint(_2095));
        _2057.Store(_1175 * 24 + 4, uint(_2096));
        _2057.Store(_1175 * 24 + 8, uint(_2097));
        _2057.Store(_1175 * 24 + 12, asuint(_2051));
        _2057.Store(_1175 * 24 + 16, asuint(_2099));
        _2057.Store(_1175 * 24 + 20, asuint(_2100));
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
