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

ByteAddressBuffer _416 : register(t20, space0);
ByteAddressBuffer _687 : register(t3, space0);
ByteAddressBuffer _902 : register(t7, space0);
ByteAddressBuffer _1149 : register(t9, space0);
ByteAddressBuffer _1153 : register(t10, space0);
ByteAddressBuffer _1174 : register(t8, space0);
ByteAddressBuffer _1218 : register(t11, space0);
ByteAddressBuffer _1311 : register(t14, space0);
RWByteAddressBuffer _1327 : register(u12, space0);
ByteAddressBuffer _1478 : register(t4, space0);
ByteAddressBuffer _1528 : register(t5, space0);
ByteAddressBuffer _1564 : register(t6, space0);
ByteAddressBuffer _1681 : register(t1, space0);
ByteAddressBuffer _1685 : register(t2, space0);
ByteAddressBuffer _1863 : register(t15, space0);
RWByteAddressBuffer _2149 : register(u0, space0);
cbuffer UniformParams
{
    Params _1388_g_params : packoffset(c0);
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
        float3 _2471 = inv_v;
        _2471.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _2471;
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
            float3 _2469 = inv_v;
            _2469.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _2469;
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
        float3 _2475 = inv_v;
        _2475.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _2475;
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
            float3 _2473 = inv_v;
            _2473.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _2473;
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
        float3 _2479 = inv_v;
        _2479.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _2479;
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
            float3 _2477 = inv_v;
            _2477.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _2477;
        }
    }
    return inv_v;
}

int hash(int x)
{
    uint _118 = uint(x);
    uint _125 = ((_118 >> uint(16)) ^ _118) * 73244475u;
    uint _130 = ((_125 >> uint(16)) ^ _125) * 73244475u;
    return int((_130 >> uint(16)) ^ _130);
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
    float _780 = mad(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float _788 = mad(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float _803 = mad(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    float _810 = mad(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    float _827 = mad(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    float _834 = mad(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    float _839 = max(max(min(_780, _788), min(_803, _810)), min(_827, _834));
    float _847 = min(min(max(_780, _788), max(_803, _810)), max(_827, _834)) * 1.0000002384185791015625f;
    return ((_839 <= _847) && (_839 <= t)) && (_847 > 0.0f);
}

void IntersectTri(float3 ro, float3 rd, tri_accel_t tri, uint prim_index, inout hit_data_t inter)
{
    do
    {
        float _560 = dot(rd, tri.n_plane.xyz);
        float _569 = tri.n_plane.w - dot(ro, tri.n_plane.xyz);
        if (sign(_569) != sign(mad(_560, inter.t, -_569)))
        {
            break;
        }
        float3 _590 = (ro * _560) + (rd * _569);
        float _601 = mad(_560, tri.u_plane.w, dot(_590, tri.u_plane.xyz));
        float _606 = _560 - _601;
        if (sign(_601) != sign(_606))
        {
            break;
        }
        float _622 = mad(_560, tri.v_plane.w, dot(_590, tri.v_plane.xyz));
        if (sign(_622) != sign(_606 - _622))
        {
            break;
        }
        float _637 = 1.0f / _560;
        inter.mask = -1;
        int _642;
        if (_560 < 0.0f)
        {
            _642 = int(prim_index);
        }
        else
        {
            _642 = (-1) - int(prim_index);
        }
        inter.prim_index = _642;
        inter.t = _569 * _637;
        inter.u = _601 * _637;
        inter.v = _622 * _637;
        break;
    } while(false);
}

void IntersectTris_ClosestHit(float3 ro, float3 rd, int tri_start, int tri_end, int obj_index, inout hit_data_t out_inter)
{
    int _2407 = 0;
    int _2408 = obj_index;
    float _2410 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _2409;
    float _2411;
    float _2412;
    for (int i = tri_start; i < tri_end; )
    {
        param = ro;
        param_1 = rd;
        tri_accel_t _698;
        _698.n_plane = asfloat(_687.Load4(i * 48 + 0));
        _698.u_plane = asfloat(_687.Load4(i * 48 + 16));
        _698.v_plane = asfloat(_687.Load4(i * 48 + 32));
        param_2.n_plane = _698.n_plane;
        param_2.u_plane = _698.u_plane;
        param_2.v_plane = _698.v_plane;
        param_3 = uint(i);
        hit_data_t _2419 = { _2407, _2408, _2409, _2410, _2411, _2412 };
        param_4 = _2419;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _2407 = param_4.mask;
        _2408 = param_4.obj_index;
        _2409 = param_4.prim_index;
        _2410 = param_4.t;
        _2411 = param_4.u;
        _2412 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _2407;
    int _721;
    if (_2407 != 0)
    {
        _721 = _2408;
    }
    else
    {
        _721 = out_inter.obj_index;
    }
    out_inter.obj_index = _721;
    int _734;
    if (_2407 != 0)
    {
        _734 = _2409;
    }
    else
    {
        _734 = out_inter.prim_index;
    }
    out_inter.prim_index = _734;
    out_inter.t = _2410;
    float _750;
    if (_2407 != 0)
    {
        _750 = _2411;
    }
    else
    {
        _750 = out_inter.u;
    }
    out_inter.u = _750;
    float _763;
    if (_2407 != 0)
    {
        _763 = _2412;
    }
    else
    {
        _763 = out_inter.v;
    }
    out_inter.v = _763;
}

void Traverse_MicroTree_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    float3 _864 = (-inv_d) * ro;
    uint _866 = stack_size;
    uint _876 = stack_size;
    stack_size = _876 + uint(1);
    g_stack[gl_LocalInvocationIndex][_876] = node_index;
    uint _950;
    uint _974;
    while (stack_size != _866)
    {
        uint _891 = stack_size;
        uint _892 = _891 - uint(1);
        stack_size = _892;
        bvh_node_t _906;
        _906.bbox_min = asfloat(_902.Load4(g_stack[gl_LocalInvocationIndex][_892] * 32 + 0));
        _906.bbox_max = asfloat(_902.Load4(g_stack[gl_LocalInvocationIndex][_892] * 32 + 16));
        float3 param = inv_d;
        float3 param_1 = _864;
        float param_2 = inter.t;
        float3 param_3 = _906.bbox_min.xyz;
        float3 param_4 = _906.bbox_max.xyz;
        if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
        {
            continue;
        }
        uint _933 = asuint(_906.bbox_min.w);
        if ((_933 & 2147483648u) == 0u)
        {
            uint _940 = stack_size;
            stack_size = _940 + uint(1);
            uint _944 = asuint(_906.bbox_max.w);
            uint _946 = _944 >> uint(30);
            if (rd[_946] < 0.0f)
            {
                _950 = _933;
            }
            else
            {
                _950 = _944 & 1073741823u;
            }
            g_stack[gl_LocalInvocationIndex][_940] = _950;
            uint _965 = stack_size;
            stack_size = _965 + uint(1);
            if (rd[_946] < 0.0f)
            {
                _974 = _944 & 1073741823u;
            }
            else
            {
                _974 = _933;
            }
            g_stack[gl_LocalInvocationIndex][_965] = _974;
        }
        else
        {
            int _994 = int(_933 & 2147483647u);
            float3 param_5 = ro;
            float3 param_6 = rd;
            int param_7 = _994;
            int param_8 = _994 + asint(_906.bbox_max.w);
            int param_9 = obj_index;
            hit_data_t param_10 = inter;
            IntersectTris_ClosestHit(param_5, param_6, param_7, param_8, param_9, param_10);
            inter = param_10;
        }
    }
}

void Traverse_MacroTree_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    float3 _1019 = (-orig_inv_rd) * orig_ro;
    uint stack_size = 1u;
    g_stack[gl_LocalInvocationIndex][0u] = node_index;
    uint _1084;
    uint _1107;
    while (stack_size != 0u)
    {
        uint _1035 = stack_size;
        uint _1036 = _1035 - uint(1);
        stack_size = _1036;
        bvh_node_t _1042;
        _1042.bbox_min = asfloat(_902.Load4(g_stack[gl_LocalInvocationIndex][_1036] * 32 + 0));
        _1042.bbox_max = asfloat(_902.Load4(g_stack[gl_LocalInvocationIndex][_1036] * 32 + 16));
        float3 param = orig_inv_rd;
        float3 param_1 = _1019;
        float param_2 = inter.t;
        float3 param_3 = _1042.bbox_min.xyz;
        float3 param_4 = _1042.bbox_max.xyz;
        if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
        {
            continue;
        }
        uint _1069 = asuint(_1042.bbox_min.w);
        if ((_1069 & 2147483648u) == 0u)
        {
            uint _1075 = stack_size;
            stack_size = _1075 + uint(1);
            uint _1079 = asuint(_1042.bbox_max.w);
            uint _1080 = _1079 >> uint(30);
            if (orig_rd[_1080] < 0.0f)
            {
                _1084 = _1069;
            }
            else
            {
                _1084 = _1079 & 1073741823u;
            }
            g_stack[gl_LocalInvocationIndex][_1075] = _1084;
            uint _1098 = stack_size;
            stack_size = _1098 + uint(1);
            if (orig_rd[_1080] < 0.0f)
            {
                _1107 = _1079 & 1073741823u;
            }
            else
            {
                _1107 = _1069;
            }
            g_stack[gl_LocalInvocationIndex][_1098] = _1107;
        }
        else
        {
            uint _1125 = _1069 & 2147483647u;
            uint _1129 = asuint(_1042.bbox_max.w);
            for (uint i = _1125; i < (_1125 + _1129); i++)
            {
                mesh_instance_t _1160;
                _1160.bbox_min = asfloat(_1149.Load4(_1153.Load(i * 4 + 0) * 32 + 0));
                _1160.bbox_max = asfloat(_1149.Load4(_1153.Load(i * 4 + 0) * 32 + 16));
                mesh_t _1180;
                [unroll]
                for (int _31ident = 0; _31ident < 3; _31ident++)
                {
                    _1180.bbox_min[_31ident] = asfloat(_1174.Load(_31ident * 4 + asuint(_1160.bbox_max.w) * 48 + 0));
                }
                [unroll]
                for (int _32ident = 0; _32ident < 3; _32ident++)
                {
                    _1180.bbox_max[_32ident] = asfloat(_1174.Load(_32ident * 4 + asuint(_1160.bbox_max.w) * 48 + 12));
                }
                _1180.node_index = _1174.Load(asuint(_1160.bbox_max.w) * 48 + 24);
                _1180.node_count = _1174.Load(asuint(_1160.bbox_max.w) * 48 + 28);
                _1180.tris_index = _1174.Load(asuint(_1160.bbox_max.w) * 48 + 32);
                _1180.tris_count = _1174.Load(asuint(_1160.bbox_max.w) * 48 + 36);
                _1180.vert_index = _1174.Load(asuint(_1160.bbox_max.w) * 48 + 40);
                _1180.vert_count = _1174.Load(asuint(_1160.bbox_max.w) * 48 + 44);
                transform_t _1224;
                _1224.xform = asfloat(uint4x4(_1218.Load4(asuint(_1160.bbox_min.w) * 128 + 0), _1218.Load4(asuint(_1160.bbox_min.w) * 128 + 16), _1218.Load4(asuint(_1160.bbox_min.w) * 128 + 32), _1218.Load4(asuint(_1160.bbox_min.w) * 128 + 48)));
                _1224.inv_xform = asfloat(uint4x4(_1218.Load4(asuint(_1160.bbox_min.w) * 128 + 64), _1218.Load4(asuint(_1160.bbox_min.w) * 128 + 80), _1218.Load4(asuint(_1160.bbox_min.w) * 128 + 96), _1218.Load4(asuint(_1160.bbox_min.w) * 128 + 112)));
                float3 param_5 = orig_inv_rd;
                float3 param_6 = _1019;
                float param_7 = inter.t;
                float3 param_8 = _1160.bbox_min.xyz;
                float3 param_9 = _1160.bbox_max.xyz;
                if (!_bbox_test_fma(param_5, param_6, param_7, param_8, param_9))
                {
                    continue;
                }
                float3 _1269 = mul(float4(orig_rd, 0.0f), _1224.inv_xform).xyz;
                float3 param_10 = _1269;
                float3 param_11 = mul(float4(orig_ro, 1.0f), _1224.inv_xform).xyz;
                float3 param_12 = _1269;
                float3 param_13 = safe_invert(param_10);
                int param_14 = int(_1153.Load(i * 4 + 0));
                uint param_15 = _1180.node_index;
                uint param_16 = stack_size;
                hit_data_t param_17 = inter;
                Traverse_MicroTree_WithStack(param_11, param_12, param_13, param_14, param_15, param_16, param_17);
                inter = param_17;
            }
        }
    }
}

float2 TransformUV(float2 _uv, atlas_texture_t t, int mip_level)
{
    uint _2362[14] = t.pos;
    uint _2365[14] = t.pos;
    uint _374 = t.size & 16383u;
    uint _377 = t.size >> uint(16);
    uint _378 = _377 & 16383u;
    float2 size = float2(float(_374), float(_378));
    if ((_377 & 32768u) != 0u)
    {
        size = float2(float(_374 >> uint(mip_level)), float(_378 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_2362[mip_level] & 65535u), float((_2365[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _291 = mad(col.z, 31.875f, 1.0f);
    float _301 = (col.x - 0.501960813999176025390625f) / _291;
    float _307 = (col.y - 0.501960813999176025390625f) / _291;
    return float3((col.w + _301) - _307, col.w + _307, (col.w - _301) - _307);
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
    atlas_texture_t _419;
    _419.size = _416.Load(index * 80 + 0);
    _419.atlas = _416.Load(index * 80 + 4);
    [unroll]
    for (int _33ident = 0; _33ident < 4; _33ident++)
    {
        _419.page[_33ident] = _416.Load(_33ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _34ident = 0; _34ident < 14; _34ident++)
    {
        _419.pos[_34ident] = _416.Load(_34ident * 4 + index * 80 + 24);
    }
    uint _2370[4];
    _2370[0] = _419.page[0];
    _2370[1] = _419.page[1];
    _2370[2] = _419.page[2];
    _2370[3] = _419.page[3];
    uint _2406[14] = { _419.pos[0], _419.pos[1], _419.pos[2], _419.pos[3], _419.pos[4], _419.pos[5], _419.pos[6], _419.pos[7], _419.pos[8], _419.pos[9], _419.pos[10], _419.pos[11], _419.pos[12], _419.pos[13] };
    atlas_texture_t _2376 = { _419.size, _419.atlas, _2370, _2406 };
    uint _501 = _419.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_501)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_501)], float3(TransformUV(uvs, _2376, lod) * 0.000118371215648949146270751953125f.xx, float((_2370[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
    bool _516;
    if (maybe_YCoCg)
    {
        _516 = _419.atlas == 4u;
    }
    else
    {
        _516 = maybe_YCoCg;
    }
    if (_516)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _535;
    if (maybe_SRGB)
    {
        _535 = (_419.size & 32768u) != 0u;
    }
    else
    {
        _535 = maybe_SRGB;
    }
    if (_535)
    {
        float3 param_1 = res.xyz;
        float3 _541 = srgb_to_rgb(param_1);
        float4 _2495 = res;
        _2495.x = _541.x;
        float4 _2497 = _2495;
        _2497.y = _541.y;
        float4 _2499 = _2497;
        _2499.z = _541.z;
        res = _2499;
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
        int _1305 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_1305) >= _1311.Load(4))
        {
            break;
        }
        float3 ro = float3(asfloat(_1327.Load(_1305 * 72 + 0)), asfloat(_1327.Load(_1305 * 72 + 4)), asfloat(_1327.Load(_1305 * 72 + 8)));
        float _1355 = asfloat(_1327.Load(_1305 * 72 + 12));
        float _1358 = asfloat(_1327.Load(_1305 * 72 + 16));
        float _1361 = asfloat(_1327.Load(_1305 * 72 + 20));
        float3 _1362 = float3(_1355, _1358, _1361);
        float3 param = _1362;
        float3 _1366 = safe_invert(param);
        int _2188 = 0;
        int _2190 = 0;
        int _2189 = 0;
        float _2191 = 3402823346297367662189621542912.0f;
        float _2193 = 0.0f;
        float _2192 = 0.0f;
        uint param_1 = uint(hash(int(_1327.Load(_1305 * 72 + 64))));
        float _1382 = construct_float(param_1);
        ray_data_t _1395;
        [unroll]
        for (int _35ident = 0; _35ident < 3; _35ident++)
        {
            _1395.o[_35ident] = asfloat(_1327.Load(_35ident * 4 + _1305 * 72 + 0));
        }
        [unroll]
        for (int _36ident = 0; _36ident < 3; _36ident++)
        {
            _1395.d[_36ident] = asfloat(_1327.Load(_36ident * 4 + _1305 * 72 + 12));
        }
        _1395.pdf = asfloat(_1327.Load(_1305 * 72 + 24));
        [unroll]
        for (int _37ident = 0; _37ident < 3; _37ident++)
        {
            _1395.c[_37ident] = asfloat(_1327.Load(_37ident * 4 + _1305 * 72 + 28));
        }
        [unroll]
        for (int _38ident = 0; _38ident < 4; _38ident++)
        {
            _1395.ior[_38ident] = asfloat(_1327.Load(_38ident * 4 + _1305 * 72 + 40));
        }
        _1395.cone_width = asfloat(_1327.Load(_1305 * 72 + 56));
        _1395.cone_spread = asfloat(_1327.Load(_1305 * 72 + 60));
        _1395.xy = int(_1327.Load(_1305 * 72 + 64));
        _1395.depth = int(_1327.Load(_1305 * 72 + 68));
        float _2300[4] = { _1395.ior[0], _1395.ior[1], _1395.ior[2], _1395.ior[3] };
        float _2291[3] = { _1395.c[0], _1395.c[1], _1395.c[2] };
        float _2284[3] = { _1395.d[0], _1395.d[1], _1395.d[2] };
        float _2277[3] = { _1395.o[0], _1395.o[1], _1395.o[2] };
        ray_data_t _2232 = { _2277, _2284, _1395.pdf, _2291, _2300, _1395.cone_width, _1395.cone_spread, _1395.xy, _1395.depth };
        int rand_index = _1388_g_params.hi + (total_depth(_2232) * 7);
        int _1509;
        float _2038;
        for (;;)
        {
            float _1455 = _2191;
            float3 param_2 = ro;
            float3 param_3 = _1362;
            float3 param_4 = _1366;
            uint param_5 = _1388_g_params.node_index;
            hit_data_t _2200 = { _2188, _2189, _2190, _1455, _2192, _2193 };
            hit_data_t param_6 = _2200;
            Traverse_MacroTree_WithStack(param_2, param_3, param_4, param_5, param_6);
            _2188 = param_6.mask;
            _2189 = param_6.obj_index;
            _2190 = param_6.prim_index;
            _2191 = param_6.t;
            _2192 = param_6.u;
            _2193 = param_6.v;
            if (param_6.prim_index < 0)
            {
                _2190 = (-1) - int(_1478.Load(((-1) - _2190) * 4 + 0));
            }
            else
            {
                _2190 = int(_1478.Load(_2190 * 4 + 0));
            }
            if (_2188 == 0)
            {
                break;
            }
            bool _1506 = _2190 < 0;
            if (_1506)
            {
                _1509 = (-1) - _2190;
            }
            else
            {
                _1509 = _2190;
            }
            uint _1520 = uint(_1509);
            bool _1522 = !_1506;
            bool _1536;
            if (_1522)
            {
                _1536 = ((_1528.Load(_1520 * 4 + 0) >> 16u) & 32768u) != 0u;
            }
            else
            {
                _1536 = _1522;
            }
            bool _1549;
            if (!_1536)
            {
                bool _1548;
                if (_1506)
                {
                    _1548 = (_1528.Load(_1520 * 4 + 0) & 32768u) != 0u;
                }
                else
                {
                    _1548 = _1506;
                }
                _1549 = _1548;
            }
            else
            {
                _1549 = _1536;
            }
            if (_1549)
            {
                break;
            }
            material_t _1572;
            [unroll]
            for (int _39ident = 0; _39ident < 5; _39ident++)
            {
                _1572.textures[_39ident] = _1564.Load(_39ident * 4 + ((_1528.Load(_1520 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _1572.base_color[_40ident] = asfloat(_1564.Load(_40ident * 4 + ((_1528.Load(_1520 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
            }
            _1572.flags = _1564.Load(((_1528.Load(_1520 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
            _1572.type = _1564.Load(((_1528.Load(_1520 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
            _1572.tangent_rotation_or_strength = asfloat(_1564.Load(((_1528.Load(_1520 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
            _1572.roughness_and_anisotropic = _1564.Load(((_1528.Load(_1520 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
            _1572.ior = asfloat(_1564.Load(((_1528.Load(_1520 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
            _1572.sheen_and_sheen_tint = _1564.Load(((_1528.Load(_1520 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
            _1572.tint_and_metallic = _1564.Load(((_1528.Load(_1520 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
            _1572.transmission_and_transmission_roughness = _1564.Load(((_1528.Load(_1520 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
            _1572.specular_and_specular_tint = _1564.Load(((_1528.Load(_1520 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
            _1572.clearcoat_and_clearcoat_roughness = _1564.Load(((_1528.Load(_1520 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
            _1572.normal_map_strength_unorm = _1564.Load(((_1528.Load(_1520 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
            uint _2302 = _1572.textures[1];
            uint _2304 = _1572.textures[3];
            uint _2305 = _1572.textures[4];
            float _2306 = _1572.base_color[0];
            float _2307 = _1572.base_color[1];
            float _2308 = _1572.base_color[2];
            uint _2236 = _1572.type;
            float _2237 = _1572.tangent_rotation_or_strength;
            if (_1506)
            {
                material_t _1624;
                [unroll]
                for (int _41ident = 0; _41ident < 5; _41ident++)
                {
                    _1624.textures[_41ident] = _1564.Load(_41ident * 4 + (_1528.Load(_1520 * 4 + 0) & 16383u) * 76 + 0);
                }
                [unroll]
                for (int _42ident = 0; _42ident < 3; _42ident++)
                {
                    _1624.base_color[_42ident] = asfloat(_1564.Load(_42ident * 4 + (_1528.Load(_1520 * 4 + 0) & 16383u) * 76 + 20));
                }
                _1624.flags = _1564.Load((_1528.Load(_1520 * 4 + 0) & 16383u) * 76 + 32);
                _1624.type = _1564.Load((_1528.Load(_1520 * 4 + 0) & 16383u) * 76 + 36);
                _1624.tangent_rotation_or_strength = asfloat(_1564.Load((_1528.Load(_1520 * 4 + 0) & 16383u) * 76 + 40));
                _1624.roughness_and_anisotropic = _1564.Load((_1528.Load(_1520 * 4 + 0) & 16383u) * 76 + 44);
                _1624.ior = asfloat(_1564.Load((_1528.Load(_1520 * 4 + 0) & 16383u) * 76 + 48));
                _1624.sheen_and_sheen_tint = _1564.Load((_1528.Load(_1520 * 4 + 0) & 16383u) * 76 + 52);
                _1624.tint_and_metallic = _1564.Load((_1528.Load(_1520 * 4 + 0) & 16383u) * 76 + 56);
                _1624.transmission_and_transmission_roughness = _1564.Load((_1528.Load(_1520 * 4 + 0) & 16383u) * 76 + 60);
                _1624.specular_and_specular_tint = _1564.Load((_1528.Load(_1520 * 4 + 0) & 16383u) * 76 + 64);
                _1624.clearcoat_and_clearcoat_roughness = _1564.Load((_1528.Load(_1520 * 4 + 0) & 16383u) * 76 + 68);
                _1624.normal_map_strength_unorm = _1564.Load((_1528.Load(_1520 * 4 + 0) & 16383u) * 76 + 72);
                _2302 = _1624.textures[1];
                _2304 = _1624.textures[3];
                _2305 = _1624.textures[4];
                _2306 = _1624.base_color[0];
                _2307 = _1624.base_color[1];
                _2308 = _1624.base_color[2];
                _2236 = _1624.type;
                _2237 = _1624.tangent_rotation_or_strength;
            }
            uint _1687 = _1520 * 3u;
            vertex_t _1693;
            [unroll]
            for (int _43ident = 0; _43ident < 3; _43ident++)
            {
                _1693.p[_43ident] = asfloat(_1681.Load(_43ident * 4 + _1685.Load(_1687 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _44ident = 0; _44ident < 3; _44ident++)
            {
                _1693.n[_44ident] = asfloat(_1681.Load(_44ident * 4 + _1685.Load(_1687 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _45ident = 0; _45ident < 3; _45ident++)
            {
                _1693.b[_45ident] = asfloat(_1681.Load(_45ident * 4 + _1685.Load(_1687 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _46ident = 0; _46ident < 2; _46ident++)
            {
                [unroll]
                for (int _47ident = 0; _47ident < 2; _47ident++)
                {
                    _1693.t[_46ident][_47ident] = asfloat(_1681.Load(_47ident * 4 + _46ident * 8 + _1685.Load(_1687 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1741;
            [unroll]
            for (int _48ident = 0; _48ident < 3; _48ident++)
            {
                _1741.p[_48ident] = asfloat(_1681.Load(_48ident * 4 + _1685.Load((_1687 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _49ident = 0; _49ident < 3; _49ident++)
            {
                _1741.n[_49ident] = asfloat(_1681.Load(_49ident * 4 + _1685.Load((_1687 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _50ident = 0; _50ident < 3; _50ident++)
            {
                _1741.b[_50ident] = asfloat(_1681.Load(_50ident * 4 + _1685.Load((_1687 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _51ident = 0; _51ident < 2; _51ident++)
            {
                [unroll]
                for (int _52ident = 0; _52ident < 2; _52ident++)
                {
                    _1741.t[_51ident][_52ident] = asfloat(_1681.Load(_52ident * 4 + _51ident * 8 + _1685.Load((_1687 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1787;
            [unroll]
            for (int _53ident = 0; _53ident < 3; _53ident++)
            {
                _1787.p[_53ident] = asfloat(_1681.Load(_53ident * 4 + _1685.Load((_1687 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _54ident = 0; _54ident < 3; _54ident++)
            {
                _1787.n[_54ident] = asfloat(_1681.Load(_54ident * 4 + _1685.Load((_1687 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _55ident = 0; _55ident < 3; _55ident++)
            {
                _1787.b[_55ident] = asfloat(_1681.Load(_55ident * 4 + _1685.Load((_1687 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _56ident = 0; _56ident < 2; _56ident++)
            {
                [unroll]
                for (int _57ident = 0; _57ident < 2; _57ident++)
                {
                    _1787.t[_56ident][_57ident] = asfloat(_1681.Load(_57ident * 4 + _56ident * 8 + _1685.Load((_1687 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            float2 _1858 = ((float2(_1693.t[0][0], _1693.t[0][1]) * ((1.0f - _2192) - _2193)) + (float2(_1741.t[0][0], _1741.t[0][1]) * _2192)) + (float2(_1787.t[0][0], _1787.t[0][1]) * _2193);
            float trans_r = frac(asfloat(_1863.Load(rand_index * 4 + 0)) + _1382);
            while (_2236 == 4u)
            {
                float mix_val = _2237;
                if (_2302 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_2302, _1858, 0).x;
                }
                if (trans_r > mix_val)
                {
                    material_t _1903;
                    [unroll]
                    for (int _58ident = 0; _58ident < 5; _58ident++)
                    {
                        _1903.textures[_58ident] = _1564.Load(_58ident * 4 + _2304 * 76 + 0);
                    }
                    [unroll]
                    for (int _59ident = 0; _59ident < 3; _59ident++)
                    {
                        _1903.base_color[_59ident] = asfloat(_1564.Load(_59ident * 4 + _2304 * 76 + 20));
                    }
                    _1903.flags = _1564.Load(_2304 * 76 + 32);
                    _1903.type = _1564.Load(_2304 * 76 + 36);
                    _1903.tangent_rotation_or_strength = asfloat(_1564.Load(_2304 * 76 + 40));
                    _1903.roughness_and_anisotropic = _1564.Load(_2304 * 76 + 44);
                    _1903.ior = asfloat(_1564.Load(_2304 * 76 + 48));
                    _1903.sheen_and_sheen_tint = _1564.Load(_2304 * 76 + 52);
                    _1903.tint_and_metallic = _1564.Load(_2304 * 76 + 56);
                    _1903.transmission_and_transmission_roughness = _1564.Load(_2304 * 76 + 60);
                    _1903.specular_and_specular_tint = _1564.Load(_2304 * 76 + 64);
                    _1903.clearcoat_and_clearcoat_roughness = _1564.Load(_2304 * 76 + 68);
                    _1903.normal_map_strength_unorm = _1564.Load(_2304 * 76 + 72);
                    _2302 = _1903.textures[1];
                    _2304 = _1903.textures[3];
                    _2305 = _1903.textures[4];
                    _2306 = _1903.base_color[0];
                    _2307 = _1903.base_color[1];
                    _2308 = _1903.base_color[2];
                    _2236 = _1903.type;
                    _2237 = _1903.tangent_rotation_or_strength;
                    trans_r = (trans_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _1956;
                    [unroll]
                    for (int _60ident = 0; _60ident < 5; _60ident++)
                    {
                        _1956.textures[_60ident] = _1564.Load(_60ident * 4 + _2305 * 76 + 0);
                    }
                    [unroll]
                    for (int _61ident = 0; _61ident < 3; _61ident++)
                    {
                        _1956.base_color[_61ident] = asfloat(_1564.Load(_61ident * 4 + _2305 * 76 + 20));
                    }
                    _1956.flags = _1564.Load(_2305 * 76 + 32);
                    _1956.type = _1564.Load(_2305 * 76 + 36);
                    _1956.tangent_rotation_or_strength = asfloat(_1564.Load(_2305 * 76 + 40));
                    _1956.roughness_and_anisotropic = _1564.Load(_2305 * 76 + 44);
                    _1956.ior = asfloat(_1564.Load(_2305 * 76 + 48));
                    _1956.sheen_and_sheen_tint = _1564.Load(_2305 * 76 + 52);
                    _1956.tint_and_metallic = _1564.Load(_2305 * 76 + 56);
                    _1956.transmission_and_transmission_roughness = _1564.Load(_2305 * 76 + 60);
                    _1956.specular_and_specular_tint = _1564.Load(_2305 * 76 + 64);
                    _1956.clearcoat_and_clearcoat_roughness = _1564.Load(_2305 * 76 + 68);
                    _1956.normal_map_strength_unorm = _1564.Load(_2305 * 76 + 72);
                    _2302 = _1956.textures[1];
                    _2304 = _1956.textures[3];
                    _2305 = _1956.textures[4];
                    _2306 = _1956.base_color[0];
                    _2307 = _1956.base_color[1];
                    _2308 = _1956.base_color[2];
                    _2236 = _1956.type;
                    _2237 = _1956.tangent_rotation_or_strength;
                    trans_r /= mix_val;
                }
            }
            if (_2236 != 5u)
            {
                break;
            }
            float _2027 = max(asfloat(_1327.Load(_1305 * 72 + 28)), max(asfloat(_1327.Load(_1305 * 72 + 32)), asfloat(_1327.Load(_1305 * 72 + 36))));
            if ((int(_1327.Load(_1305 * 72 + 68)) >> 24) > _1388_g_params.min_transp_depth)
            {
                _2038 = max(0.0500000007450580596923828125f, 1.0f - _2027);
            }
            else
            {
                _2038 = 0.0f;
            }
            bool _2052 = (frac(asfloat(_1863.Load((rand_index + 6) * 4 + 0)) + _1382) < _2038) || (_2027 == 0.0f);
            bool _2064;
            if (!_2052)
            {
                _2064 = ((int(_1327.Load(_1305 * 72 + 68)) >> 24) + 1) >= _1388_g_params.max_transp_depth;
            }
            else
            {
                _2064 = _2052;
            }
            if (_2064)
            {
                _1327.Store(_1305 * 72 + 36, asuint(0.0f));
                _1327.Store(_1305 * 72 + 32, asuint(0.0f));
                _1327.Store(_1305 * 72 + 28, asuint(0.0f));
                break;
            }
            float _2078 = 1.0f - _2038;
            _1327.Store(_1305 * 72 + 28, asuint(asfloat(_1327.Load(_1305 * 72 + 28)) * (_2306 / _2078)));
            _1327.Store(_1305 * 72 + 32, asuint(asfloat(_1327.Load(_1305 * 72 + 32)) * (_2307 / _2078)));
            _1327.Store(_1305 * 72 + 36, asuint(asfloat(_1327.Load(_1305 * 72 + 36)) * (_2308 / _2078)));
            ro += (_1362 * (_2191 + 9.9999997473787516355514526367188e-06f));
            _2188 = 0;
            _2191 = _1455 - _2191;
            _1327.Store(_1305 * 72 + 68, uint(int(_1327.Load(_1305 * 72 + 68)) + 16777216));
            rand_index += 7;
            continue;
        }
        float _2130 = asfloat(_1327.Load(_1305 * 72 + 0));
        float _2133 = asfloat(_1327.Load(_1305 * 72 + 4));
        float _2136 = asfloat(_1327.Load(_1305 * 72 + 8));
        float _2142 = _2191;
        float _2143 = _2142 + length(float3(_2130, _2133, _2136) - ro);
        _2191 = _2143;
        _2149.Store(_1305 * 24 + 0, uint(_2188));
        _2149.Store(_1305 * 24 + 4, uint(_2189));
        _2149.Store(_1305 * 24 + 8, uint(_2190));
        _2149.Store(_1305 * 24 + 12, asuint(_2143));
        _2149.Store(_1305 * 24 + 16, asuint(_2192));
        _2149.Store(_1305 * 24 + 20, asuint(_2193));
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
