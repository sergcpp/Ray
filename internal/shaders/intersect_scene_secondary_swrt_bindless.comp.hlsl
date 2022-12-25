struct ray_data_t
{
    float o[3];
    float d[3];
    float pdf;
    float c[3];
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
    uint2 img_size;
    uint node_index;
    float cam_clip_end;
    int min_transp_depth;
    int max_transp_depth;
    int hi;
    int pad2;
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

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

ByteAddressBuffer _536 : register(t3, space0);
ByteAddressBuffer _751 : register(t7, space0);
ByteAddressBuffer _998 : register(t9, space0);
ByteAddressBuffer _1002 : register(t10, space0);
ByteAddressBuffer _1023 : register(t8, space0);
ByteAddressBuffer _1068 : register(t11, space0);
ByteAddressBuffer _1161 : register(t14, space0);
RWByteAddressBuffer _1176 : register(u12, space0);
ByteAddressBuffer _1316 : register(t4, space0);
ByteAddressBuffer _1366 : register(t5, space0);
ByteAddressBuffer _1403 : register(t6, space0);
ByteAddressBuffer _1530 : register(t1, space0);
ByteAddressBuffer _1534 : register(t2, space0);
ByteAddressBuffer _1713 : register(t15, space0);
RWByteAddressBuffer _2004 : register(u0, space0);
cbuffer UniformParams
{
    Params _1237_g_params : packoffset(c0);
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

float3 safe_invert(float3 v)
{
    float3 inv_v = 1.0f.xxx / v;
    bool _149 = v.x <= 1.0000000116860974230803549289703e-07f;
    bool _156;
    if (_149)
    {
        _156 = v.x >= 0.0f;
    }
    else
    {
        _156 = _149;
    }
    if (_156)
    {
        float3 _2267 = inv_v;
        _2267.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _2267;
    }
    else
    {
        bool _165 = v.x >= (-1.0000000116860974230803549289703e-07f);
        bool _171;
        if (_165)
        {
            _171 = v.x < 0.0f;
        }
        else
        {
            _171 = _165;
        }
        if (_171)
        {
            float3 _2265 = inv_v;
            _2265.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _2265;
        }
    }
    bool _179 = v.y <= 1.0000000116860974230803549289703e-07f;
    bool _185;
    if (_179)
    {
        _185 = v.y >= 0.0f;
    }
    else
    {
        _185 = _179;
    }
    if (_185)
    {
        float3 _2271 = inv_v;
        _2271.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _2271;
    }
    else
    {
        bool _192 = v.y >= (-1.0000000116860974230803549289703e-07f);
        bool _198;
        if (_192)
        {
            _198 = v.y < 0.0f;
        }
        else
        {
            _198 = _192;
        }
        if (_198)
        {
            float3 _2269 = inv_v;
            _2269.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _2269;
        }
    }
    bool _205 = v.z <= 1.0000000116860974230803549289703e-07f;
    bool _211;
    if (_205)
    {
        _211 = v.z >= 0.0f;
    }
    else
    {
        _211 = _205;
    }
    if (_211)
    {
        float3 _2275 = inv_v;
        _2275.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _2275;
    }
    else
    {
        bool _218 = v.z >= (-1.0000000116860974230803549289703e-07f);
        bool _224;
        if (_218)
        {
            _224 = v.z < 0.0f;
        }
        else
        {
            _224 = _218;
        }
        if (_224)
        {
            float3 _2273 = inv_v;
            _2273.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _2273;
        }
    }
    return inv_v;
}

int hash(int x)
{
    uint _106 = uint(x);
    uint _113 = ((_106 >> uint(16)) ^ _106) * 73244475u;
    uint _118 = ((_113 >> uint(16)) ^ _113) * 73244475u;
    return int((_118 >> uint(16)) ^ _118);
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
    float _629 = mad(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float _637 = mad(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float _652 = mad(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    float _659 = mad(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    float _676 = mad(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    float _683 = mad(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    float _688 = max(max(min(_629, _637), min(_652, _659)), min(_676, _683));
    float _696 = min(min(max(_629, _637), max(_652, _659)), max(_676, _683)) * 1.0000002384185791015625f;
    return ((_688 <= _696) && (_688 <= t)) && (_696 > 0.0f);
}

void IntersectTri(float3 ro, float3 rd, tri_accel_t tri, uint prim_index, inout hit_data_t inter)
{
    do
    {
        float _406 = dot(rd, tri.n_plane.xyz);
        float _415 = tri.n_plane.w - dot(ro, tri.n_plane.xyz);
        if (sign(_415) != sign(mad(_406, inter.t, -_415)))
        {
            break;
        }
        float3 _436 = (ro * _406) + (rd * _415);
        float _447 = mad(_406, tri.u_plane.w, dot(_436, tri.u_plane.xyz));
        float _452 = _406 - _447;
        if (sign(_447) != sign(_452))
        {
            break;
        }
        float _469 = mad(_406, tri.v_plane.w, dot(_436, tri.v_plane.xyz));
        if (sign(_469) != sign(_452 - _469))
        {
            break;
        }
        float _484 = 1.0f / _406;
        inter.mask = -1;
        int _489;
        if (_406 < 0.0f)
        {
            _489 = int(prim_index);
        }
        else
        {
            _489 = (-1) - int(prim_index);
        }
        inter.prim_index = _489;
        inter.t = _415 * _484;
        inter.u = _447 * _484;
        inter.v = _469 * _484;
        break;
    } while(false);
}

void IntersectTris_ClosestHit(float3 ro, float3 rd, int tri_start, int tri_end, int obj_index, inout hit_data_t out_inter)
{
    int _2203 = 0;
    int _2204 = obj_index;
    float _2206 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _2205;
    float _2207;
    float _2208;
    for (int i = tri_start; i < tri_end; )
    {
        param = ro;
        param_1 = rd;
        tri_accel_t _547;
        _547.n_plane = asfloat(_536.Load4(i * 48 + 0));
        _547.u_plane = asfloat(_536.Load4(i * 48 + 16));
        _547.v_plane = asfloat(_536.Load4(i * 48 + 32));
        param_2.n_plane = _547.n_plane;
        param_2.u_plane = _547.u_plane;
        param_2.v_plane = _547.v_plane;
        param_3 = uint(i);
        hit_data_t _2215 = { _2203, _2204, _2205, _2206, _2207, _2208 };
        param_4 = _2215;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _2203 = param_4.mask;
        _2204 = param_4.obj_index;
        _2205 = param_4.prim_index;
        _2206 = param_4.t;
        _2207 = param_4.u;
        _2208 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _2203;
    int _570;
    if (_2203 != 0)
    {
        _570 = _2204;
    }
    else
    {
        _570 = out_inter.obj_index;
    }
    out_inter.obj_index = _570;
    int _583;
    if (_2203 != 0)
    {
        _583 = _2205;
    }
    else
    {
        _583 = out_inter.prim_index;
    }
    out_inter.prim_index = _583;
    out_inter.t = _2206;
    float _599;
    if (_2203 != 0)
    {
        _599 = _2207;
    }
    else
    {
        _599 = out_inter.u;
    }
    out_inter.u = _599;
    float _612;
    if (_2203 != 0)
    {
        _612 = _2208;
    }
    else
    {
        _612 = out_inter.v;
    }
    out_inter.v = _612;
}

void Traverse_MicroTree_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    float3 _713 = (-inv_d) * ro;
    uint _715 = stack_size;
    uint _725 = stack_size;
    stack_size = _725 + uint(1);
    g_stack[gl_LocalInvocationIndex][_725] = node_index;
    uint _799;
    uint _823;
    while (stack_size != _715)
    {
        uint _740 = stack_size;
        uint _741 = _740 - uint(1);
        stack_size = _741;
        bvh_node_t _755;
        _755.bbox_min = asfloat(_751.Load4(g_stack[gl_LocalInvocationIndex][_741] * 32 + 0));
        _755.bbox_max = asfloat(_751.Load4(g_stack[gl_LocalInvocationIndex][_741] * 32 + 16));
        float3 param = inv_d;
        float3 param_1 = _713;
        float param_2 = inter.t;
        float3 param_3 = _755.bbox_min.xyz;
        float3 param_4 = _755.bbox_max.xyz;
        if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
        {
            continue;
        }
        uint _782 = asuint(_755.bbox_min.w);
        if ((_782 & 2147483648u) == 0u)
        {
            uint _789 = stack_size;
            stack_size = _789 + uint(1);
            uint _793 = asuint(_755.bbox_max.w);
            uint _795 = _793 >> uint(30);
            if (rd[_795] < 0.0f)
            {
                _799 = _782;
            }
            else
            {
                _799 = _793 & 1073741823u;
            }
            g_stack[gl_LocalInvocationIndex][_789] = _799;
            uint _814 = stack_size;
            stack_size = _814 + uint(1);
            if (rd[_795] < 0.0f)
            {
                _823 = _793 & 1073741823u;
            }
            else
            {
                _823 = _782;
            }
            g_stack[gl_LocalInvocationIndex][_814] = _823;
        }
        else
        {
            int _843 = int(_782 & 2147483647u);
            float3 param_5 = ro;
            float3 param_6 = rd;
            int param_7 = _843;
            int param_8 = _843 + asint(_755.bbox_max.w);
            int param_9 = obj_index;
            hit_data_t param_10 = inter;
            IntersectTris_ClosestHit(param_5, param_6, param_7, param_8, param_9, param_10);
            inter = param_10;
        }
    }
}

void Traverse_MacroTree_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    float3 _868 = (-orig_inv_rd) * orig_ro;
    uint stack_size = 1u;
    g_stack[gl_LocalInvocationIndex][0u] = node_index;
    uint _933;
    uint _956;
    while (stack_size != 0u)
    {
        uint _884 = stack_size;
        uint _885 = _884 - uint(1);
        stack_size = _885;
        bvh_node_t _891;
        _891.bbox_min = asfloat(_751.Load4(g_stack[gl_LocalInvocationIndex][_885] * 32 + 0));
        _891.bbox_max = asfloat(_751.Load4(g_stack[gl_LocalInvocationIndex][_885] * 32 + 16));
        float3 param = orig_inv_rd;
        float3 param_1 = _868;
        float param_2 = inter.t;
        float3 param_3 = _891.bbox_min.xyz;
        float3 param_4 = _891.bbox_max.xyz;
        if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
        {
            continue;
        }
        uint _918 = asuint(_891.bbox_min.w);
        if ((_918 & 2147483648u) == 0u)
        {
            uint _924 = stack_size;
            stack_size = _924 + uint(1);
            uint _928 = asuint(_891.bbox_max.w);
            uint _929 = _928 >> uint(30);
            if (orig_rd[_929] < 0.0f)
            {
                _933 = _918;
            }
            else
            {
                _933 = _928 & 1073741823u;
            }
            g_stack[gl_LocalInvocationIndex][_924] = _933;
            uint _947 = stack_size;
            stack_size = _947 + uint(1);
            if (orig_rd[_929] < 0.0f)
            {
                _956 = _928 & 1073741823u;
            }
            else
            {
                _956 = _918;
            }
            g_stack[gl_LocalInvocationIndex][_947] = _956;
        }
        else
        {
            uint _974 = _918 & 2147483647u;
            uint _978 = asuint(_891.bbox_max.w);
            for (uint i = _974; i < (_974 + _978); i++)
            {
                mesh_instance_t _1009;
                _1009.bbox_min = asfloat(_998.Load4(_1002.Load(i * 4 + 0) * 32 + 0));
                _1009.bbox_max = asfloat(_998.Load4(_1002.Load(i * 4 + 0) * 32 + 16));
                mesh_t _1029;
                [unroll]
                for (int _28ident = 0; _28ident < 3; _28ident++)
                {
                    _1029.bbox_min[_28ident] = asfloat(_1023.Load(_28ident * 4 + asuint(_1009.bbox_max.w) * 48 + 0));
                }
                [unroll]
                for (int _29ident = 0; _29ident < 3; _29ident++)
                {
                    _1029.bbox_max[_29ident] = asfloat(_1023.Load(_29ident * 4 + asuint(_1009.bbox_max.w) * 48 + 12));
                }
                _1029.node_index = _1023.Load(asuint(_1009.bbox_max.w) * 48 + 24);
                _1029.node_count = _1023.Load(asuint(_1009.bbox_max.w) * 48 + 28);
                _1029.tris_index = _1023.Load(asuint(_1009.bbox_max.w) * 48 + 32);
                _1029.tris_count = _1023.Load(asuint(_1009.bbox_max.w) * 48 + 36);
                _1029.vert_index = _1023.Load(asuint(_1009.bbox_max.w) * 48 + 40);
                _1029.vert_count = _1023.Load(asuint(_1009.bbox_max.w) * 48 + 44);
                transform_t _1074;
                _1074.xform = asfloat(uint4x4(_1068.Load4(asuint(_1009.bbox_min.w) * 128 + 0), _1068.Load4(asuint(_1009.bbox_min.w) * 128 + 16), _1068.Load4(asuint(_1009.bbox_min.w) * 128 + 32), _1068.Load4(asuint(_1009.bbox_min.w) * 128 + 48)));
                _1074.inv_xform = asfloat(uint4x4(_1068.Load4(asuint(_1009.bbox_min.w) * 128 + 64), _1068.Load4(asuint(_1009.bbox_min.w) * 128 + 80), _1068.Load4(asuint(_1009.bbox_min.w) * 128 + 96), _1068.Load4(asuint(_1009.bbox_min.w) * 128 + 112)));
                float3 param_5 = orig_inv_rd;
                float3 param_6 = _868;
                float param_7 = inter.t;
                float3 param_8 = _1009.bbox_min.xyz;
                float3 param_9 = _1009.bbox_max.xyz;
                if (!_bbox_test_fma(param_5, param_6, param_7, param_8, param_9))
                {
                    continue;
                }
                float3 _1119 = mul(float4(orig_rd, 0.0f), _1074.inv_xform).xyz;
                float3 param_10 = _1119;
                float3 param_11 = mul(float4(orig_ro, 1.0f), _1074.inv_xform).xyz;
                float3 param_12 = _1119;
                float3 param_13 = safe_invert(param_10);
                int param_14 = int(_1002.Load(i * 4 + 0));
                uint param_15 = _1029.node_index;
                uint param_16 = stack_size;
                hit_data_t param_17 = inter;
                Traverse_MicroTree_WithStack(param_11, param_12, param_13, param_14, param_15, param_16, param_17);
                inter = param_17;
            }
        }
    }
}

float3 YCoCg_to_RGB(float4 col)
{
    float _279 = mad(col.z, 31.875f, 1.0f);
    float _289 = (col.x - 0.501960813999176025390625f) / _279;
    float _295 = (col.y - 0.501960813999176025390625f) / _279;
    return float3((col.w + _289) - _295, col.w + _295, (col.w - _289) - _295);
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
    uint _352 = index & 16777215u;
    float4 res = g_textures[NonUniformResourceIndex(_352)].SampleLevel(_g_textures_sampler[NonUniformResourceIndex(_352)], uvs, float(lod));
    bool _363;
    if (maybe_YCoCg)
    {
        _363 = (index & 67108864u) != 0u;
    }
    else
    {
        _363 = maybe_YCoCg;
    }
    if (_363)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _381;
    if (maybe_SRGB)
    {
        _381 = (index & 16777216u) != 0u;
    }
    else
    {
        _381 = maybe_SRGB;
    }
    if (_381)
    {
        float3 param_1 = res.xyz;
        float3 _387 = srgb_to_rgb(param_1);
        float4 _2291 = res;
        _2291.x = _387.x;
        float4 _2293 = _2291;
        _2293.y = _387.y;
        float4 _2295 = _2293;
        _2295.z = _387.z;
        res = _2295;
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
        int _1155 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_1155) >= _1161.Load(4))
        {
            break;
        }
        float3 ro = float3(asfloat(_1176.Load(_1155 * 56 + 0)), asfloat(_1176.Load(_1155 * 56 + 4)), asfloat(_1176.Load(_1155 * 56 + 8)));
        float _1204 = asfloat(_1176.Load(_1155 * 56 + 12));
        float _1207 = asfloat(_1176.Load(_1155 * 56 + 16));
        float _1210 = asfloat(_1176.Load(_1155 * 56 + 20));
        float3 _1211 = float3(_1204, _1207, _1210);
        float3 param = _1211;
        float3 _1215 = safe_invert(param);
        int _2041 = 0;
        int _2043 = 0;
        int _2042 = 0;
        float _2044 = 3402823346297367662189621542912.0f;
        float _2046 = 0.0f;
        float _2045 = 0.0f;
        uint param_1 = uint(hash(int(_1176.Load(_1155 * 56 + 48))));
        float _1231 = construct_float(param_1);
        ray_data_t _1244;
        [unroll]
        for (int _30ident = 0; _30ident < 3; _30ident++)
        {
            _1244.o[_30ident] = asfloat(_1176.Load(_30ident * 4 + _1155 * 56 + 0));
        }
        [unroll]
        for (int _31ident = 0; _31ident < 3; _31ident++)
        {
            _1244.d[_31ident] = asfloat(_1176.Load(_31ident * 4 + _1155 * 56 + 12));
        }
        _1244.pdf = asfloat(_1176.Load(_1155 * 56 + 24));
        [unroll]
        for (int _32ident = 0; _32ident < 3; _32ident++)
        {
            _1244.c[_32ident] = asfloat(_1176.Load(_32ident * 4 + _1155 * 56 + 28));
        }
        _1244.cone_width = asfloat(_1176.Load(_1155 * 56 + 40));
        _1244.cone_spread = asfloat(_1176.Load(_1155 * 56 + 44));
        _1244.xy = int(_1176.Load(_1155 * 56 + 48));
        _1244.depth = int(_1176.Load(_1155 * 56 + 52));
        float _2143[3] = { _1244.c[0], _1244.c[1], _1244.c[2] };
        float _2136[3] = { _1244.d[0], _1244.d[1], _1244.d[2] };
        float _2129[3] = { _1244.o[0], _1244.o[1], _1244.o[2] };
        ray_data_t _2083 = { _2129, _2136, _1244.pdf, _2143, _1244.cone_width, _1244.cone_spread, _1244.xy, _1244.depth };
        int rand_index = _1237_g_params.hi + (total_depth(_2083) * 7);
        int _1347;
        float _1893;
        for (;;)
        {
            float _1293 = _2044;
            float3 param_2 = ro;
            float3 param_3 = _1211;
            float3 param_4 = _1215;
            uint param_5 = _1237_g_params.node_index;
            hit_data_t _2053 = { _2041, _2042, _2043, _1293, _2045, _2046 };
            hit_data_t param_6 = _2053;
            Traverse_MacroTree_WithStack(param_2, param_3, param_4, param_5, param_6);
            _2041 = param_6.mask;
            _2042 = param_6.obj_index;
            _2043 = param_6.prim_index;
            _2044 = param_6.t;
            _2045 = param_6.u;
            _2046 = param_6.v;
            if (param_6.prim_index < 0)
            {
                _2043 = (-1) - int(_1316.Load(((-1) - _2043) * 4 + 0));
            }
            else
            {
                _2043 = int(_1316.Load(_2043 * 4 + 0));
            }
            if (_2041 == 0)
            {
                break;
            }
            bool _1344 = _2043 < 0;
            if (_1344)
            {
                _1347 = (-1) - _2043;
            }
            else
            {
                _1347 = _2043;
            }
            uint _1358 = uint(_1347);
            bool _1360 = !_1344;
            bool _1375;
            if (_1360)
            {
                _1375 = ((_1366.Load(_1358 * 4 + 0) >> 16u) & 32768u) != 0u;
            }
            else
            {
                _1375 = _1360;
            }
            bool _1388;
            if (!_1375)
            {
                bool _1387;
                if (_1344)
                {
                    _1387 = (_1366.Load(_1358 * 4 + 0) & 32768u) != 0u;
                }
                else
                {
                    _1387 = _1344;
                }
                _1388 = _1387;
            }
            else
            {
                _1388 = _1375;
            }
            if (_1388)
            {
                break;
            }
            material_t _1412;
            [unroll]
            for (int _33ident = 0; _33ident < 5; _33ident++)
            {
                _1412.textures[_33ident] = _1403.Load(_33ident * 4 + ((_1366.Load(_1358 * 4 + 0) >> 16u) & 16383u) * 80 + 0);
            }
            [unroll]
            for (int _34ident = 0; _34ident < 3; _34ident++)
            {
                _1412.base_color[_34ident] = asfloat(_1403.Load(_34ident * 4 + ((_1366.Load(_1358 * 4 + 0) >> 16u) & 16383u) * 80 + 20));
            }
            _1412.flags = _1403.Load(((_1366.Load(_1358 * 4 + 0) >> 16u) & 16383u) * 80 + 32);
            _1412.type = _1403.Load(((_1366.Load(_1358 * 4 + 0) >> 16u) & 16383u) * 80 + 36);
            _1412.tangent_rotation_or_strength = asfloat(_1403.Load(((_1366.Load(_1358 * 4 + 0) >> 16u) & 16383u) * 80 + 40));
            _1412.roughness_and_anisotropic = _1403.Load(((_1366.Load(_1358 * 4 + 0) >> 16u) & 16383u) * 80 + 44);
            _1412.int_ior = asfloat(_1403.Load(((_1366.Load(_1358 * 4 + 0) >> 16u) & 16383u) * 80 + 48));
            _1412.ext_ior = asfloat(_1403.Load(((_1366.Load(_1358 * 4 + 0) >> 16u) & 16383u) * 80 + 52));
            _1412.sheen_and_sheen_tint = _1403.Load(((_1366.Load(_1358 * 4 + 0) >> 16u) & 16383u) * 80 + 56);
            _1412.tint_and_metallic = _1403.Load(((_1366.Load(_1358 * 4 + 0) >> 16u) & 16383u) * 80 + 60);
            _1412.transmission_and_transmission_roughness = _1403.Load(((_1366.Load(_1358 * 4 + 0) >> 16u) & 16383u) * 80 + 64);
            _1412.specular_and_specular_tint = _1403.Load(((_1366.Load(_1358 * 4 + 0) >> 16u) & 16383u) * 80 + 68);
            _1412.clearcoat_and_clearcoat_roughness = _1403.Load(((_1366.Load(_1358 * 4 + 0) >> 16u) & 16383u) * 80 + 72);
            _1412.normal_map_strength_unorm = _1403.Load(((_1366.Load(_1358 * 4 + 0) >> 16u) & 16383u) * 80 + 76);
            uint _2145 = _1412.textures[1];
            uint _2147 = _1412.textures[3];
            uint _2148 = _1412.textures[4];
            float _2149 = _1412.base_color[0];
            float _2150 = _1412.base_color[1];
            float _2151 = _1412.base_color[2];
            uint _2087 = _1412.type;
            float _2088 = _1412.tangent_rotation_or_strength;
            if (_1344)
            {
                material_t _1471;
                [unroll]
                for (int _35ident = 0; _35ident < 5; _35ident++)
                {
                    _1471.textures[_35ident] = _1403.Load(_35ident * 4 + (_1366.Load(_1358 * 4 + 0) & 16383u) * 80 + 0);
                }
                [unroll]
                for (int _36ident = 0; _36ident < 3; _36ident++)
                {
                    _1471.base_color[_36ident] = asfloat(_1403.Load(_36ident * 4 + (_1366.Load(_1358 * 4 + 0) & 16383u) * 80 + 20));
                }
                _1471.flags = _1403.Load((_1366.Load(_1358 * 4 + 0) & 16383u) * 80 + 32);
                _1471.type = _1403.Load((_1366.Load(_1358 * 4 + 0) & 16383u) * 80 + 36);
                _1471.tangent_rotation_or_strength = asfloat(_1403.Load((_1366.Load(_1358 * 4 + 0) & 16383u) * 80 + 40));
                _1471.roughness_and_anisotropic = _1403.Load((_1366.Load(_1358 * 4 + 0) & 16383u) * 80 + 44);
                _1471.int_ior = asfloat(_1403.Load((_1366.Load(_1358 * 4 + 0) & 16383u) * 80 + 48));
                _1471.ext_ior = asfloat(_1403.Load((_1366.Load(_1358 * 4 + 0) & 16383u) * 80 + 52));
                _1471.sheen_and_sheen_tint = _1403.Load((_1366.Load(_1358 * 4 + 0) & 16383u) * 80 + 56);
                _1471.tint_and_metallic = _1403.Load((_1366.Load(_1358 * 4 + 0) & 16383u) * 80 + 60);
                _1471.transmission_and_transmission_roughness = _1403.Load((_1366.Load(_1358 * 4 + 0) & 16383u) * 80 + 64);
                _1471.specular_and_specular_tint = _1403.Load((_1366.Load(_1358 * 4 + 0) & 16383u) * 80 + 68);
                _1471.clearcoat_and_clearcoat_roughness = _1403.Load((_1366.Load(_1358 * 4 + 0) & 16383u) * 80 + 72);
                _1471.normal_map_strength_unorm = _1403.Load((_1366.Load(_1358 * 4 + 0) & 16383u) * 80 + 76);
                _2145 = _1471.textures[1];
                _2147 = _1471.textures[3];
                _2148 = _1471.textures[4];
                _2149 = _1471.base_color[0];
                _2150 = _1471.base_color[1];
                _2151 = _1471.base_color[2];
                _2087 = _1471.type;
                _2088 = _1471.tangent_rotation_or_strength;
            }
            uint _1536 = _1358 * 3u;
            vertex_t _1542;
            [unroll]
            for (int _37ident = 0; _37ident < 3; _37ident++)
            {
                _1542.p[_37ident] = asfloat(_1530.Load(_37ident * 4 + _1534.Load(_1536 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _38ident = 0; _38ident < 3; _38ident++)
            {
                _1542.n[_38ident] = asfloat(_1530.Load(_38ident * 4 + _1534.Load(_1536 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _39ident = 0; _39ident < 3; _39ident++)
            {
                _1542.b[_39ident] = asfloat(_1530.Load(_39ident * 4 + _1534.Load(_1536 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _40ident = 0; _40ident < 2; _40ident++)
            {
                [unroll]
                for (int _41ident = 0; _41ident < 2; _41ident++)
                {
                    _1542.t[_40ident][_41ident] = asfloat(_1530.Load(_41ident * 4 + _40ident * 8 + _1534.Load(_1536 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1590;
            [unroll]
            for (int _42ident = 0; _42ident < 3; _42ident++)
            {
                _1590.p[_42ident] = asfloat(_1530.Load(_42ident * 4 + _1534.Load((_1536 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _43ident = 0; _43ident < 3; _43ident++)
            {
                _1590.n[_43ident] = asfloat(_1530.Load(_43ident * 4 + _1534.Load((_1536 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _44ident = 0; _44ident < 3; _44ident++)
            {
                _1590.b[_44ident] = asfloat(_1530.Load(_44ident * 4 + _1534.Load((_1536 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _45ident = 0; _45ident < 2; _45ident++)
            {
                [unroll]
                for (int _46ident = 0; _46ident < 2; _46ident++)
                {
                    _1590.t[_45ident][_46ident] = asfloat(_1530.Load(_46ident * 4 + _45ident * 8 + _1534.Load((_1536 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1636;
            [unroll]
            for (int _47ident = 0; _47ident < 3; _47ident++)
            {
                _1636.p[_47ident] = asfloat(_1530.Load(_47ident * 4 + _1534.Load((_1536 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _48ident = 0; _48ident < 3; _48ident++)
            {
                _1636.n[_48ident] = asfloat(_1530.Load(_48ident * 4 + _1534.Load((_1536 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _49ident = 0; _49ident < 3; _49ident++)
            {
                _1636.b[_49ident] = asfloat(_1530.Load(_49ident * 4 + _1534.Load((_1536 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _50ident = 0; _50ident < 2; _50ident++)
            {
                [unroll]
                for (int _51ident = 0; _51ident < 2; _51ident++)
                {
                    _1636.t[_50ident][_51ident] = asfloat(_1530.Load(_51ident * 4 + _50ident * 8 + _1534.Load((_1536 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            float2 _1708 = ((float2(_1542.t[0][0], _1542.t[0][1]) * ((1.0f - _2045) - _2046)) + (float2(_1590.t[0][0], _1590.t[0][1]) * _2045)) + (float2(_1636.t[0][0], _1636.t[0][1]) * _2046);
            float trans_r = frac(asfloat(_1713.Load(rand_index * 4 + 0)) + _1231);
            while (_2087 == 4u)
            {
                float mix_val = _2088;
                if (_2145 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_2145, _1708, 0).x;
                }
                if (trans_r > mix_val)
                {
                    material_t _1754;
                    [unroll]
                    for (int _52ident = 0; _52ident < 5; _52ident++)
                    {
                        _1754.textures[_52ident] = _1403.Load(_52ident * 4 + _2147 * 80 + 0);
                    }
                    [unroll]
                    for (int _53ident = 0; _53ident < 3; _53ident++)
                    {
                        _1754.base_color[_53ident] = asfloat(_1403.Load(_53ident * 4 + _2147 * 80 + 20));
                    }
                    _1754.flags = _1403.Load(_2147 * 80 + 32);
                    _1754.type = _1403.Load(_2147 * 80 + 36);
                    _1754.tangent_rotation_or_strength = asfloat(_1403.Load(_2147 * 80 + 40));
                    _1754.roughness_and_anisotropic = _1403.Load(_2147 * 80 + 44);
                    _1754.int_ior = asfloat(_1403.Load(_2147 * 80 + 48));
                    _1754.ext_ior = asfloat(_1403.Load(_2147 * 80 + 52));
                    _1754.sheen_and_sheen_tint = _1403.Load(_2147 * 80 + 56);
                    _1754.tint_and_metallic = _1403.Load(_2147 * 80 + 60);
                    _1754.transmission_and_transmission_roughness = _1403.Load(_2147 * 80 + 64);
                    _1754.specular_and_specular_tint = _1403.Load(_2147 * 80 + 68);
                    _1754.clearcoat_and_clearcoat_roughness = _1403.Load(_2147 * 80 + 72);
                    _1754.normal_map_strength_unorm = _1403.Load(_2147 * 80 + 76);
                    _2145 = _1754.textures[1];
                    _2147 = _1754.textures[3];
                    _2148 = _1754.textures[4];
                    _2149 = _1754.base_color[0];
                    _2150 = _1754.base_color[1];
                    _2151 = _1754.base_color[2];
                    _2087 = _1754.type;
                    _2088 = _1754.tangent_rotation_or_strength;
                    trans_r = (trans_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _1809;
                    [unroll]
                    for (int _54ident = 0; _54ident < 5; _54ident++)
                    {
                        _1809.textures[_54ident] = _1403.Load(_54ident * 4 + _2148 * 80 + 0);
                    }
                    [unroll]
                    for (int _55ident = 0; _55ident < 3; _55ident++)
                    {
                        _1809.base_color[_55ident] = asfloat(_1403.Load(_55ident * 4 + _2148 * 80 + 20));
                    }
                    _1809.flags = _1403.Load(_2148 * 80 + 32);
                    _1809.type = _1403.Load(_2148 * 80 + 36);
                    _1809.tangent_rotation_or_strength = asfloat(_1403.Load(_2148 * 80 + 40));
                    _1809.roughness_and_anisotropic = _1403.Load(_2148 * 80 + 44);
                    _1809.int_ior = asfloat(_1403.Load(_2148 * 80 + 48));
                    _1809.ext_ior = asfloat(_1403.Load(_2148 * 80 + 52));
                    _1809.sheen_and_sheen_tint = _1403.Load(_2148 * 80 + 56);
                    _1809.tint_and_metallic = _1403.Load(_2148 * 80 + 60);
                    _1809.transmission_and_transmission_roughness = _1403.Load(_2148 * 80 + 64);
                    _1809.specular_and_specular_tint = _1403.Load(_2148 * 80 + 68);
                    _1809.clearcoat_and_clearcoat_roughness = _1403.Load(_2148 * 80 + 72);
                    _1809.normal_map_strength_unorm = _1403.Load(_2148 * 80 + 76);
                    _2145 = _1809.textures[1];
                    _2147 = _1809.textures[3];
                    _2148 = _1809.textures[4];
                    _2149 = _1809.base_color[0];
                    _2150 = _1809.base_color[1];
                    _2151 = _1809.base_color[2];
                    _2087 = _1809.type;
                    _2088 = _1809.tangent_rotation_or_strength;
                    trans_r /= mix_val;
                }
            }
            if (_2087 != 5u)
            {
                break;
            }
            float _1882 = max(asfloat(_1176.Load(_1155 * 56 + 28)), max(asfloat(_1176.Load(_1155 * 56 + 32)), asfloat(_1176.Load(_1155 * 56 + 36))));
            if ((int(_1176.Load(_1155 * 56 + 52)) >> 24) > _1237_g_params.min_transp_depth)
            {
                _1893 = max(0.0500000007450580596923828125f, 1.0f - _1882);
            }
            else
            {
                _1893 = 0.0f;
            }
            bool _1907 = (frac(asfloat(_1713.Load((rand_index + 6) * 4 + 0)) + _1231) < _1893) || (_1882 == 0.0f);
            bool _1919;
            if (!_1907)
            {
                _1919 = ((int(_1176.Load(_1155 * 56 + 52)) >> 24) + 1) >= _1237_g_params.max_transp_depth;
            }
            else
            {
                _1919 = _1907;
            }
            if (_1919)
            {
                _1176.Store(_1155 * 56 + 36, asuint(0.0f));
                _1176.Store(_1155 * 56 + 32, asuint(0.0f));
                _1176.Store(_1155 * 56 + 28, asuint(0.0f));
                break;
            }
            float _1933 = 1.0f - _1893;
            _1176.Store(_1155 * 56 + 28, asuint(asfloat(_1176.Load(_1155 * 56 + 28)) * (_2149 / _1933)));
            _1176.Store(_1155 * 56 + 32, asuint(asfloat(_1176.Load(_1155 * 56 + 32)) * (_2150 / _1933)));
            _1176.Store(_1155 * 56 + 36, asuint(asfloat(_1176.Load(_1155 * 56 + 36)) * (_2151 / _1933)));
            ro += (_1211 * (_2044 + 9.9999997473787516355514526367188e-06f));
            _2041 = 0;
            _2044 = _1293 - _2044;
            _1176.Store(_1155 * 56 + 52, uint(int(_1176.Load(_1155 * 56 + 52)) + 16777216));
            rand_index += 7;
            continue;
        }
        float _1985 = asfloat(_1176.Load(_1155 * 56 + 0));
        float _1988 = asfloat(_1176.Load(_1155 * 56 + 4));
        float _1991 = asfloat(_1176.Load(_1155 * 56 + 8));
        float _1997 = _2044;
        float _1998 = _1997 + length(float3(_1985, _1988, _1991) - ro);
        _2044 = _1998;
        _2004.Store(_1155 * 24 + 0, uint(_2041));
        _2004.Store(_1155 * 24 + 4, uint(_2042));
        _2004.Store(_1155 * 24 + 8, uint(_2043));
        _2004.Store(_1155 * 24 + 12, asuint(_1998));
        _2004.Store(_1155 * 24 + 16, asuint(_2045));
        _2004.Store(_1155 * 24 + 20, asuint(_2046));
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
