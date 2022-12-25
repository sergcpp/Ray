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

ByteAddressBuffer _416 : register(t20, space0);
ByteAddressBuffer _686 : register(t3, space0);
ByteAddressBuffer _901 : register(t7, space0);
ByteAddressBuffer _1148 : register(t9, space0);
ByteAddressBuffer _1152 : register(t10, space0);
ByteAddressBuffer _1173 : register(t8, space0);
ByteAddressBuffer _1217 : register(t11, space0);
ByteAddressBuffer _1310 : register(t14, space0);
RWByteAddressBuffer _1325 : register(u12, space0);
ByteAddressBuffer _1465 : register(t4, space0);
ByteAddressBuffer _1515 : register(t5, space0);
ByteAddressBuffer _1551 : register(t6, space0);
ByteAddressBuffer _1672 : register(t1, space0);
ByteAddressBuffer _1676 : register(t2, space0);
ByteAddressBuffer _1854 : register(t15, space0);
RWByteAddressBuffer _2144 : register(u0, space0);
cbuffer UniformParams
{
    Params _1386_g_params : packoffset(c0);
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
    bool _160 = v.x <= 1.0000000116860974230803549289703e-07f;
    bool _167;
    if (_160)
    {
        _167 = v.x >= 0.0f;
    }
    else
    {
        _167 = _160;
    }
    if (_167)
    {
        float3 _2456 = inv_v;
        _2456.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _2456;
    }
    else
    {
        bool _176 = v.x >= (-1.0000000116860974230803549289703e-07f);
        bool _182;
        if (_176)
        {
            _182 = v.x < 0.0f;
        }
        else
        {
            _182 = _176;
        }
        if (_182)
        {
            float3 _2454 = inv_v;
            _2454.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _2454;
        }
    }
    bool _190 = v.y <= 1.0000000116860974230803549289703e-07f;
    bool _196;
    if (_190)
    {
        _196 = v.y >= 0.0f;
    }
    else
    {
        _196 = _190;
    }
    if (_196)
    {
        float3 _2460 = inv_v;
        _2460.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _2460;
    }
    else
    {
        bool _203 = v.y >= (-1.0000000116860974230803549289703e-07f);
        bool _209;
        if (_203)
        {
            _209 = v.y < 0.0f;
        }
        else
        {
            _209 = _203;
        }
        if (_209)
        {
            float3 _2458 = inv_v;
            _2458.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _2458;
        }
    }
    bool _216 = v.z <= 1.0000000116860974230803549289703e-07f;
    bool _222;
    if (_216)
    {
        _222 = v.z >= 0.0f;
    }
    else
    {
        _222 = _216;
    }
    if (_222)
    {
        float3 _2464 = inv_v;
        _2464.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _2464;
    }
    else
    {
        bool _229 = v.z >= (-1.0000000116860974230803549289703e-07f);
        bool _235;
        if (_229)
        {
            _235 = v.z < 0.0f;
        }
        else
        {
            _235 = _229;
        }
        if (_235)
        {
            float3 _2462 = inv_v;
            _2462.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _2462;
        }
    }
    return inv_v;
}

int hash(int x)
{
    uint _117 = uint(x);
    uint _124 = ((_117 >> uint(16)) ^ _117) * 73244475u;
    uint _129 = ((_124 >> uint(16)) ^ _124) * 73244475u;
    return int((_129 >> uint(16)) ^ _129);
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
    float _779 = mad(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float _787 = mad(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float _802 = mad(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    float _809 = mad(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    float _826 = mad(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    float _833 = mad(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    float _838 = max(max(min(_779, _787), min(_802, _809)), min(_826, _833));
    float _846 = min(min(max(_779, _787), max(_802, _809)), max(_826, _833)) * 1.0000002384185791015625f;
    return ((_838 <= _846) && (_838 <= t)) && (_846 > 0.0f);
}

void IntersectTri(float3 ro, float3 rd, tri_accel_t tri, uint prim_index, inout hit_data_t inter)
{
    do
    {
        float _559 = dot(rd, tri.n_plane.xyz);
        float _568 = tri.n_plane.w - dot(ro, tri.n_plane.xyz);
        if (sign(_568) != sign(mad(_559, inter.t, -_568)))
        {
            break;
        }
        float3 _589 = (ro * _559) + (rd * _568);
        float _600 = mad(_559, tri.u_plane.w, dot(_589, tri.u_plane.xyz));
        float _605 = _559 - _600;
        if (sign(_600) != sign(_605))
        {
            break;
        }
        float _621 = mad(_559, tri.v_plane.w, dot(_589, tri.v_plane.xyz));
        if (sign(_621) != sign(_605 - _621))
        {
            break;
        }
        float _636 = 1.0f / _559;
        inter.mask = -1;
        int _641;
        if (_559 < 0.0f)
        {
            _641 = int(prim_index);
        }
        else
        {
            _641 = (-1) - int(prim_index);
        }
        inter.prim_index = _641;
        inter.t = _568 * _636;
        inter.u = _600 * _636;
        inter.v = _621 * _636;
        break;
    } while(false);
}

void IntersectTris_ClosestHit(float3 ro, float3 rd, int tri_start, int tri_end, int obj_index, inout hit_data_t out_inter)
{
    int _2392 = 0;
    int _2393 = obj_index;
    float _2395 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _2394;
    float _2396;
    float _2397;
    for (int i = tri_start; i < tri_end; )
    {
        param = ro;
        param_1 = rd;
        tri_accel_t _697;
        _697.n_plane = asfloat(_686.Load4(i * 48 + 0));
        _697.u_plane = asfloat(_686.Load4(i * 48 + 16));
        _697.v_plane = asfloat(_686.Load4(i * 48 + 32));
        param_2.n_plane = _697.n_plane;
        param_2.u_plane = _697.u_plane;
        param_2.v_plane = _697.v_plane;
        param_3 = uint(i);
        hit_data_t _2404 = { _2392, _2393, _2394, _2395, _2396, _2397 };
        param_4 = _2404;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _2392 = param_4.mask;
        _2393 = param_4.obj_index;
        _2394 = param_4.prim_index;
        _2395 = param_4.t;
        _2396 = param_4.u;
        _2397 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _2392;
    int _720;
    if (_2392 != 0)
    {
        _720 = _2393;
    }
    else
    {
        _720 = out_inter.obj_index;
    }
    out_inter.obj_index = _720;
    int _733;
    if (_2392 != 0)
    {
        _733 = _2394;
    }
    else
    {
        _733 = out_inter.prim_index;
    }
    out_inter.prim_index = _733;
    out_inter.t = _2395;
    float _749;
    if (_2392 != 0)
    {
        _749 = _2396;
    }
    else
    {
        _749 = out_inter.u;
    }
    out_inter.u = _749;
    float _762;
    if (_2392 != 0)
    {
        _762 = _2397;
    }
    else
    {
        _762 = out_inter.v;
    }
    out_inter.v = _762;
}

void Traverse_MicroTree_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    float3 _863 = (-inv_d) * ro;
    uint _865 = stack_size;
    uint _875 = stack_size;
    stack_size = _875 + uint(1);
    g_stack[gl_LocalInvocationIndex][_875] = node_index;
    uint _949;
    uint _973;
    while (stack_size != _865)
    {
        uint _890 = stack_size;
        uint _891 = _890 - uint(1);
        stack_size = _891;
        bvh_node_t _905;
        _905.bbox_min = asfloat(_901.Load4(g_stack[gl_LocalInvocationIndex][_891] * 32 + 0));
        _905.bbox_max = asfloat(_901.Load4(g_stack[gl_LocalInvocationIndex][_891] * 32 + 16));
        float3 param = inv_d;
        float3 param_1 = _863;
        float param_2 = inter.t;
        float3 param_3 = _905.bbox_min.xyz;
        float3 param_4 = _905.bbox_max.xyz;
        if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
        {
            continue;
        }
        uint _932 = asuint(_905.bbox_min.w);
        if ((_932 & 2147483648u) == 0u)
        {
            uint _939 = stack_size;
            stack_size = _939 + uint(1);
            uint _943 = asuint(_905.bbox_max.w);
            uint _945 = _943 >> uint(30);
            if (rd[_945] < 0.0f)
            {
                _949 = _932;
            }
            else
            {
                _949 = _943 & 1073741823u;
            }
            g_stack[gl_LocalInvocationIndex][_939] = _949;
            uint _964 = stack_size;
            stack_size = _964 + uint(1);
            if (rd[_945] < 0.0f)
            {
                _973 = _943 & 1073741823u;
            }
            else
            {
                _973 = _932;
            }
            g_stack[gl_LocalInvocationIndex][_964] = _973;
        }
        else
        {
            int _993 = int(_932 & 2147483647u);
            float3 param_5 = ro;
            float3 param_6 = rd;
            int param_7 = _993;
            int param_8 = _993 + asint(_905.bbox_max.w);
            int param_9 = obj_index;
            hit_data_t param_10 = inter;
            IntersectTris_ClosestHit(param_5, param_6, param_7, param_8, param_9, param_10);
            inter = param_10;
        }
    }
}

void Traverse_MacroTree_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    float3 _1018 = (-orig_inv_rd) * orig_ro;
    uint stack_size = 1u;
    g_stack[gl_LocalInvocationIndex][0u] = node_index;
    uint _1083;
    uint _1106;
    while (stack_size != 0u)
    {
        uint _1034 = stack_size;
        uint _1035 = _1034 - uint(1);
        stack_size = _1035;
        bvh_node_t _1041;
        _1041.bbox_min = asfloat(_901.Load4(g_stack[gl_LocalInvocationIndex][_1035] * 32 + 0));
        _1041.bbox_max = asfloat(_901.Load4(g_stack[gl_LocalInvocationIndex][_1035] * 32 + 16));
        float3 param = orig_inv_rd;
        float3 param_1 = _1018;
        float param_2 = inter.t;
        float3 param_3 = _1041.bbox_min.xyz;
        float3 param_4 = _1041.bbox_max.xyz;
        if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
        {
            continue;
        }
        uint _1068 = asuint(_1041.bbox_min.w);
        if ((_1068 & 2147483648u) == 0u)
        {
            uint _1074 = stack_size;
            stack_size = _1074 + uint(1);
            uint _1078 = asuint(_1041.bbox_max.w);
            uint _1079 = _1078 >> uint(30);
            if (orig_rd[_1079] < 0.0f)
            {
                _1083 = _1068;
            }
            else
            {
                _1083 = _1078 & 1073741823u;
            }
            g_stack[gl_LocalInvocationIndex][_1074] = _1083;
            uint _1097 = stack_size;
            stack_size = _1097 + uint(1);
            if (orig_rd[_1079] < 0.0f)
            {
                _1106 = _1078 & 1073741823u;
            }
            else
            {
                _1106 = _1068;
            }
            g_stack[gl_LocalInvocationIndex][_1097] = _1106;
        }
        else
        {
            uint _1124 = _1068 & 2147483647u;
            uint _1128 = asuint(_1041.bbox_max.w);
            for (uint i = _1124; i < (_1124 + _1128); i++)
            {
                mesh_instance_t _1159;
                _1159.bbox_min = asfloat(_1148.Load4(_1152.Load(i * 4 + 0) * 32 + 0));
                _1159.bbox_max = asfloat(_1148.Load4(_1152.Load(i * 4 + 0) * 32 + 16));
                mesh_t _1179;
                [unroll]
                for (int _30ident = 0; _30ident < 3; _30ident++)
                {
                    _1179.bbox_min[_30ident] = asfloat(_1173.Load(_30ident * 4 + asuint(_1159.bbox_max.w) * 48 + 0));
                }
                [unroll]
                for (int _31ident = 0; _31ident < 3; _31ident++)
                {
                    _1179.bbox_max[_31ident] = asfloat(_1173.Load(_31ident * 4 + asuint(_1159.bbox_max.w) * 48 + 12));
                }
                _1179.node_index = _1173.Load(asuint(_1159.bbox_max.w) * 48 + 24);
                _1179.node_count = _1173.Load(asuint(_1159.bbox_max.w) * 48 + 28);
                _1179.tris_index = _1173.Load(asuint(_1159.bbox_max.w) * 48 + 32);
                _1179.tris_count = _1173.Load(asuint(_1159.bbox_max.w) * 48 + 36);
                _1179.vert_index = _1173.Load(asuint(_1159.bbox_max.w) * 48 + 40);
                _1179.vert_count = _1173.Load(asuint(_1159.bbox_max.w) * 48 + 44);
                transform_t _1223;
                _1223.xform = asfloat(uint4x4(_1217.Load4(asuint(_1159.bbox_min.w) * 128 + 0), _1217.Load4(asuint(_1159.bbox_min.w) * 128 + 16), _1217.Load4(asuint(_1159.bbox_min.w) * 128 + 32), _1217.Load4(asuint(_1159.bbox_min.w) * 128 + 48)));
                _1223.inv_xform = asfloat(uint4x4(_1217.Load4(asuint(_1159.bbox_min.w) * 128 + 64), _1217.Load4(asuint(_1159.bbox_min.w) * 128 + 80), _1217.Load4(asuint(_1159.bbox_min.w) * 128 + 96), _1217.Load4(asuint(_1159.bbox_min.w) * 128 + 112)));
                float3 param_5 = orig_inv_rd;
                float3 param_6 = _1018;
                float param_7 = inter.t;
                float3 param_8 = _1159.bbox_min.xyz;
                float3 param_9 = _1159.bbox_max.xyz;
                if (!_bbox_test_fma(param_5, param_6, param_7, param_8, param_9))
                {
                    continue;
                }
                float3 _1268 = mul(float4(orig_rd, 0.0f), _1223.inv_xform).xyz;
                float3 param_10 = _1268;
                float3 param_11 = mul(float4(orig_ro, 1.0f), _1223.inv_xform).xyz;
                float3 param_12 = _1268;
                float3 param_13 = safe_invert(param_10);
                int param_14 = int(_1152.Load(i * 4 + 0));
                uint param_15 = _1179.node_index;
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
    uint _2347[14] = t.pos;
    uint _2350[14] = t.pos;
    uint _374 = t.size & 16383u;
    uint _377 = t.size >> uint(16);
    uint _378 = _377 & 16383u;
    float2 size = float2(float(_374), float(_378));
    if ((_377 & 32768u) != 0u)
    {
        size = float2(float(_374 >> uint(mip_level)), float(_378 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_2347[mip_level] & 65535u), float((_2350[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _290 = mad(col.z, 31.875f, 1.0f);
    float _300 = (col.x - 0.501960813999176025390625f) / _290;
    float _306 = (col.y - 0.501960813999176025390625f) / _290;
    return float3((col.w + _300) - _306, col.w + _306, (col.w - _300) - _306);
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
    for (int _32ident = 0; _32ident < 4; _32ident++)
    {
        _419.page[_32ident] = _416.Load(_32ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _33ident = 0; _33ident < 14; _33ident++)
    {
        _419.pos[_33ident] = _416.Load(_33ident * 4 + index * 80 + 24);
    }
    uint _2355[4];
    _2355[0] = _419.page[0];
    _2355[1] = _419.page[1];
    _2355[2] = _419.page[2];
    _2355[3] = _419.page[3];
    uint _2391[14] = { _419.pos[0], _419.pos[1], _419.pos[2], _419.pos[3], _419.pos[4], _419.pos[5], _419.pos[6], _419.pos[7], _419.pos[8], _419.pos[9], _419.pos[10], _419.pos[11], _419.pos[12], _419.pos[13] };
    atlas_texture_t _2361 = { _419.size, _419.atlas, _2355, _2391 };
    uint _500 = _419.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_500)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_500)], float3(TransformUV(uvs, _2361, lod) * 0.000118371215648949146270751953125f.xx, float((_2355[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
    bool _515;
    if (maybe_YCoCg)
    {
        _515 = _419.atlas == 4u;
    }
    else
    {
        _515 = maybe_YCoCg;
    }
    if (_515)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _534;
    if (maybe_SRGB)
    {
        _534 = (_419.size & 32768u) != 0u;
    }
    else
    {
        _534 = maybe_SRGB;
    }
    if (_534)
    {
        float3 param_1 = res.xyz;
        float3 _540 = srgb_to_rgb(param_1);
        float4 _2480 = res;
        _2480.x = _540.x;
        float4 _2482 = _2480;
        _2482.y = _540.y;
        float4 _2484 = _2482;
        _2484.z = _540.z;
        res = _2484;
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
        int _1304 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_1304) >= _1310.Load(4))
        {
            break;
        }
        float3 ro = float3(asfloat(_1325.Load(_1304 * 56 + 0)), asfloat(_1325.Load(_1304 * 56 + 4)), asfloat(_1325.Load(_1304 * 56 + 8)));
        float _1353 = asfloat(_1325.Load(_1304 * 56 + 12));
        float _1356 = asfloat(_1325.Load(_1304 * 56 + 16));
        float _1359 = asfloat(_1325.Load(_1304 * 56 + 20));
        float3 _1360 = float3(_1353, _1356, _1359);
        float3 param = _1360;
        float3 _1364 = safe_invert(param);
        int _2183 = 0;
        int _2185 = 0;
        int _2184 = 0;
        float _2186 = 3402823346297367662189621542912.0f;
        float _2188 = 0.0f;
        float _2187 = 0.0f;
        uint param_1 = uint(hash(int(_1325.Load(_1304 * 56 + 48))));
        float _1380 = construct_float(param_1);
        ray_data_t _1393;
        [unroll]
        for (int _34ident = 0; _34ident < 3; _34ident++)
        {
            _1393.o[_34ident] = asfloat(_1325.Load(_34ident * 4 + _1304 * 56 + 0));
        }
        [unroll]
        for (int _35ident = 0; _35ident < 3; _35ident++)
        {
            _1393.d[_35ident] = asfloat(_1325.Load(_35ident * 4 + _1304 * 56 + 12));
        }
        _1393.pdf = asfloat(_1325.Load(_1304 * 56 + 24));
        [unroll]
        for (int _36ident = 0; _36ident < 3; _36ident++)
        {
            _1393.c[_36ident] = asfloat(_1325.Load(_36ident * 4 + _1304 * 56 + 28));
        }
        _1393.cone_width = asfloat(_1325.Load(_1304 * 56 + 40));
        _1393.cone_spread = asfloat(_1325.Load(_1304 * 56 + 44));
        _1393.xy = int(_1325.Load(_1304 * 56 + 48));
        _1393.depth = int(_1325.Load(_1304 * 56 + 52));
        float _2285[3] = { _1393.c[0], _1393.c[1], _1393.c[2] };
        float _2278[3] = { _1393.d[0], _1393.d[1], _1393.d[2] };
        float _2271[3] = { _1393.o[0], _1393.o[1], _1393.o[2] };
        ray_data_t _2225 = { _2271, _2278, _1393.pdf, _2285, _1393.cone_width, _1393.cone_spread, _1393.xy, _1393.depth };
        int rand_index = _1386_g_params.hi + (total_depth(_2225) * 7);
        int _1496;
        float _2033;
        for (;;)
        {
            float _1442 = _2186;
            float3 param_2 = ro;
            float3 param_3 = _1360;
            float3 param_4 = _1364;
            uint param_5 = _1386_g_params.node_index;
            hit_data_t _2195 = { _2183, _2184, _2185, _1442, _2187, _2188 };
            hit_data_t param_6 = _2195;
            Traverse_MacroTree_WithStack(param_2, param_3, param_4, param_5, param_6);
            _2183 = param_6.mask;
            _2184 = param_6.obj_index;
            _2185 = param_6.prim_index;
            _2186 = param_6.t;
            _2187 = param_6.u;
            _2188 = param_6.v;
            if (param_6.prim_index < 0)
            {
                _2185 = (-1) - int(_1465.Load(((-1) - _2185) * 4 + 0));
            }
            else
            {
                _2185 = int(_1465.Load(_2185 * 4 + 0));
            }
            if (_2183 == 0)
            {
                break;
            }
            bool _1493 = _2185 < 0;
            if (_1493)
            {
                _1496 = (-1) - _2185;
            }
            else
            {
                _1496 = _2185;
            }
            uint _1507 = uint(_1496);
            bool _1509 = !_1493;
            bool _1523;
            if (_1509)
            {
                _1523 = ((_1515.Load(_1507 * 4 + 0) >> 16u) & 32768u) != 0u;
            }
            else
            {
                _1523 = _1509;
            }
            bool _1536;
            if (!_1523)
            {
                bool _1535;
                if (_1493)
                {
                    _1535 = (_1515.Load(_1507 * 4 + 0) & 32768u) != 0u;
                }
                else
                {
                    _1535 = _1493;
                }
                _1536 = _1535;
            }
            else
            {
                _1536 = _1523;
            }
            if (_1536)
            {
                break;
            }
            material_t _1559;
            [unroll]
            for (int _37ident = 0; _37ident < 5; _37ident++)
            {
                _1559.textures[_37ident] = _1551.Load(_37ident * 4 + ((_1515.Load(_1507 * 4 + 0) >> 16u) & 16383u) * 80 + 0);
            }
            [unroll]
            for (int _38ident = 0; _38ident < 3; _38ident++)
            {
                _1559.base_color[_38ident] = asfloat(_1551.Load(_38ident * 4 + ((_1515.Load(_1507 * 4 + 0) >> 16u) & 16383u) * 80 + 20));
            }
            _1559.flags = _1551.Load(((_1515.Load(_1507 * 4 + 0) >> 16u) & 16383u) * 80 + 32);
            _1559.type = _1551.Load(((_1515.Load(_1507 * 4 + 0) >> 16u) & 16383u) * 80 + 36);
            _1559.tangent_rotation_or_strength = asfloat(_1551.Load(((_1515.Load(_1507 * 4 + 0) >> 16u) & 16383u) * 80 + 40));
            _1559.roughness_and_anisotropic = _1551.Load(((_1515.Load(_1507 * 4 + 0) >> 16u) & 16383u) * 80 + 44);
            _1559.int_ior = asfloat(_1551.Load(((_1515.Load(_1507 * 4 + 0) >> 16u) & 16383u) * 80 + 48));
            _1559.ext_ior = asfloat(_1551.Load(((_1515.Load(_1507 * 4 + 0) >> 16u) & 16383u) * 80 + 52));
            _1559.sheen_and_sheen_tint = _1551.Load(((_1515.Load(_1507 * 4 + 0) >> 16u) & 16383u) * 80 + 56);
            _1559.tint_and_metallic = _1551.Load(((_1515.Load(_1507 * 4 + 0) >> 16u) & 16383u) * 80 + 60);
            _1559.transmission_and_transmission_roughness = _1551.Load(((_1515.Load(_1507 * 4 + 0) >> 16u) & 16383u) * 80 + 64);
            _1559.specular_and_specular_tint = _1551.Load(((_1515.Load(_1507 * 4 + 0) >> 16u) & 16383u) * 80 + 68);
            _1559.clearcoat_and_clearcoat_roughness = _1551.Load(((_1515.Load(_1507 * 4 + 0) >> 16u) & 16383u) * 80 + 72);
            _1559.normal_map_strength_unorm = _1551.Load(((_1515.Load(_1507 * 4 + 0) >> 16u) & 16383u) * 80 + 76);
            uint _2287 = _1559.textures[1];
            uint _2289 = _1559.textures[3];
            uint _2290 = _1559.textures[4];
            float _2291 = _1559.base_color[0];
            float _2292 = _1559.base_color[1];
            float _2293 = _1559.base_color[2];
            uint _2229 = _1559.type;
            float _2230 = _1559.tangent_rotation_or_strength;
            if (_1493)
            {
                material_t _1613;
                [unroll]
                for (int _39ident = 0; _39ident < 5; _39ident++)
                {
                    _1613.textures[_39ident] = _1551.Load(_39ident * 4 + (_1515.Load(_1507 * 4 + 0) & 16383u) * 80 + 0);
                }
                [unroll]
                for (int _40ident = 0; _40ident < 3; _40ident++)
                {
                    _1613.base_color[_40ident] = asfloat(_1551.Load(_40ident * 4 + (_1515.Load(_1507 * 4 + 0) & 16383u) * 80 + 20));
                }
                _1613.flags = _1551.Load((_1515.Load(_1507 * 4 + 0) & 16383u) * 80 + 32);
                _1613.type = _1551.Load((_1515.Load(_1507 * 4 + 0) & 16383u) * 80 + 36);
                _1613.tangent_rotation_or_strength = asfloat(_1551.Load((_1515.Load(_1507 * 4 + 0) & 16383u) * 80 + 40));
                _1613.roughness_and_anisotropic = _1551.Load((_1515.Load(_1507 * 4 + 0) & 16383u) * 80 + 44);
                _1613.int_ior = asfloat(_1551.Load((_1515.Load(_1507 * 4 + 0) & 16383u) * 80 + 48));
                _1613.ext_ior = asfloat(_1551.Load((_1515.Load(_1507 * 4 + 0) & 16383u) * 80 + 52));
                _1613.sheen_and_sheen_tint = _1551.Load((_1515.Load(_1507 * 4 + 0) & 16383u) * 80 + 56);
                _1613.tint_and_metallic = _1551.Load((_1515.Load(_1507 * 4 + 0) & 16383u) * 80 + 60);
                _1613.transmission_and_transmission_roughness = _1551.Load((_1515.Load(_1507 * 4 + 0) & 16383u) * 80 + 64);
                _1613.specular_and_specular_tint = _1551.Load((_1515.Load(_1507 * 4 + 0) & 16383u) * 80 + 68);
                _1613.clearcoat_and_clearcoat_roughness = _1551.Load((_1515.Load(_1507 * 4 + 0) & 16383u) * 80 + 72);
                _1613.normal_map_strength_unorm = _1551.Load((_1515.Load(_1507 * 4 + 0) & 16383u) * 80 + 76);
                _2287 = _1613.textures[1];
                _2289 = _1613.textures[3];
                _2290 = _1613.textures[4];
                _2291 = _1613.base_color[0];
                _2292 = _1613.base_color[1];
                _2293 = _1613.base_color[2];
                _2229 = _1613.type;
                _2230 = _1613.tangent_rotation_or_strength;
            }
            uint _1678 = _1507 * 3u;
            vertex_t _1684;
            [unroll]
            for (int _41ident = 0; _41ident < 3; _41ident++)
            {
                _1684.p[_41ident] = asfloat(_1672.Load(_41ident * 4 + _1676.Load(_1678 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _42ident = 0; _42ident < 3; _42ident++)
            {
                _1684.n[_42ident] = asfloat(_1672.Load(_42ident * 4 + _1676.Load(_1678 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _43ident = 0; _43ident < 3; _43ident++)
            {
                _1684.b[_43ident] = asfloat(_1672.Load(_43ident * 4 + _1676.Load(_1678 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _44ident = 0; _44ident < 2; _44ident++)
            {
                [unroll]
                for (int _45ident = 0; _45ident < 2; _45ident++)
                {
                    _1684.t[_44ident][_45ident] = asfloat(_1672.Load(_45ident * 4 + _44ident * 8 + _1676.Load(_1678 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1732;
            [unroll]
            for (int _46ident = 0; _46ident < 3; _46ident++)
            {
                _1732.p[_46ident] = asfloat(_1672.Load(_46ident * 4 + _1676.Load((_1678 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _47ident = 0; _47ident < 3; _47ident++)
            {
                _1732.n[_47ident] = asfloat(_1672.Load(_47ident * 4 + _1676.Load((_1678 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _48ident = 0; _48ident < 3; _48ident++)
            {
                _1732.b[_48ident] = asfloat(_1672.Load(_48ident * 4 + _1676.Load((_1678 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _49ident = 0; _49ident < 2; _49ident++)
            {
                [unroll]
                for (int _50ident = 0; _50ident < 2; _50ident++)
                {
                    _1732.t[_49ident][_50ident] = asfloat(_1672.Load(_50ident * 4 + _49ident * 8 + _1676.Load((_1678 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1778;
            [unroll]
            for (int _51ident = 0; _51ident < 3; _51ident++)
            {
                _1778.p[_51ident] = asfloat(_1672.Load(_51ident * 4 + _1676.Load((_1678 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _52ident = 0; _52ident < 3; _52ident++)
            {
                _1778.n[_52ident] = asfloat(_1672.Load(_52ident * 4 + _1676.Load((_1678 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _53ident = 0; _53ident < 3; _53ident++)
            {
                _1778.b[_53ident] = asfloat(_1672.Load(_53ident * 4 + _1676.Load((_1678 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _54ident = 0; _54ident < 2; _54ident++)
            {
                [unroll]
                for (int _55ident = 0; _55ident < 2; _55ident++)
                {
                    _1778.t[_54ident][_55ident] = asfloat(_1672.Load(_55ident * 4 + _54ident * 8 + _1676.Load((_1678 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            float2 _1849 = ((float2(_1684.t[0][0], _1684.t[0][1]) * ((1.0f - _2187) - _2188)) + (float2(_1732.t[0][0], _1732.t[0][1]) * _2187)) + (float2(_1778.t[0][0], _1778.t[0][1]) * _2188);
            float trans_r = frac(asfloat(_1854.Load(rand_index * 4 + 0)) + _1380);
            while (_2229 == 4u)
            {
                float mix_val = _2230;
                if (_2287 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_2287, _1849, 0).x;
                }
                if (trans_r > mix_val)
                {
                    material_t _1894;
                    [unroll]
                    for (int _56ident = 0; _56ident < 5; _56ident++)
                    {
                        _1894.textures[_56ident] = _1551.Load(_56ident * 4 + _2289 * 80 + 0);
                    }
                    [unroll]
                    for (int _57ident = 0; _57ident < 3; _57ident++)
                    {
                        _1894.base_color[_57ident] = asfloat(_1551.Load(_57ident * 4 + _2289 * 80 + 20));
                    }
                    _1894.flags = _1551.Load(_2289 * 80 + 32);
                    _1894.type = _1551.Load(_2289 * 80 + 36);
                    _1894.tangent_rotation_or_strength = asfloat(_1551.Load(_2289 * 80 + 40));
                    _1894.roughness_and_anisotropic = _1551.Load(_2289 * 80 + 44);
                    _1894.int_ior = asfloat(_1551.Load(_2289 * 80 + 48));
                    _1894.ext_ior = asfloat(_1551.Load(_2289 * 80 + 52));
                    _1894.sheen_and_sheen_tint = _1551.Load(_2289 * 80 + 56);
                    _1894.tint_and_metallic = _1551.Load(_2289 * 80 + 60);
                    _1894.transmission_and_transmission_roughness = _1551.Load(_2289 * 80 + 64);
                    _1894.specular_and_specular_tint = _1551.Load(_2289 * 80 + 68);
                    _1894.clearcoat_and_clearcoat_roughness = _1551.Load(_2289 * 80 + 72);
                    _1894.normal_map_strength_unorm = _1551.Load(_2289 * 80 + 76);
                    _2287 = _1894.textures[1];
                    _2289 = _1894.textures[3];
                    _2290 = _1894.textures[4];
                    _2291 = _1894.base_color[0];
                    _2292 = _1894.base_color[1];
                    _2293 = _1894.base_color[2];
                    _2229 = _1894.type;
                    _2230 = _1894.tangent_rotation_or_strength;
                    trans_r = (trans_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _1949;
                    [unroll]
                    for (int _58ident = 0; _58ident < 5; _58ident++)
                    {
                        _1949.textures[_58ident] = _1551.Load(_58ident * 4 + _2290 * 80 + 0);
                    }
                    [unroll]
                    for (int _59ident = 0; _59ident < 3; _59ident++)
                    {
                        _1949.base_color[_59ident] = asfloat(_1551.Load(_59ident * 4 + _2290 * 80 + 20));
                    }
                    _1949.flags = _1551.Load(_2290 * 80 + 32);
                    _1949.type = _1551.Load(_2290 * 80 + 36);
                    _1949.tangent_rotation_or_strength = asfloat(_1551.Load(_2290 * 80 + 40));
                    _1949.roughness_and_anisotropic = _1551.Load(_2290 * 80 + 44);
                    _1949.int_ior = asfloat(_1551.Load(_2290 * 80 + 48));
                    _1949.ext_ior = asfloat(_1551.Load(_2290 * 80 + 52));
                    _1949.sheen_and_sheen_tint = _1551.Load(_2290 * 80 + 56);
                    _1949.tint_and_metallic = _1551.Load(_2290 * 80 + 60);
                    _1949.transmission_and_transmission_roughness = _1551.Load(_2290 * 80 + 64);
                    _1949.specular_and_specular_tint = _1551.Load(_2290 * 80 + 68);
                    _1949.clearcoat_and_clearcoat_roughness = _1551.Load(_2290 * 80 + 72);
                    _1949.normal_map_strength_unorm = _1551.Load(_2290 * 80 + 76);
                    _2287 = _1949.textures[1];
                    _2289 = _1949.textures[3];
                    _2290 = _1949.textures[4];
                    _2291 = _1949.base_color[0];
                    _2292 = _1949.base_color[1];
                    _2293 = _1949.base_color[2];
                    _2229 = _1949.type;
                    _2230 = _1949.tangent_rotation_or_strength;
                    trans_r /= mix_val;
                }
            }
            if (_2229 != 5u)
            {
                break;
            }
            float _2022 = max(asfloat(_1325.Load(_1304 * 56 + 28)), max(asfloat(_1325.Load(_1304 * 56 + 32)), asfloat(_1325.Load(_1304 * 56 + 36))));
            if ((int(_1325.Load(_1304 * 56 + 52)) >> 24) > _1386_g_params.min_transp_depth)
            {
                _2033 = max(0.0500000007450580596923828125f, 1.0f - _2022);
            }
            else
            {
                _2033 = 0.0f;
            }
            bool _2047 = (frac(asfloat(_1854.Load((rand_index + 6) * 4 + 0)) + _1380) < _2033) || (_2022 == 0.0f);
            bool _2059;
            if (!_2047)
            {
                _2059 = ((int(_1325.Load(_1304 * 56 + 52)) >> 24) + 1) >= _1386_g_params.max_transp_depth;
            }
            else
            {
                _2059 = _2047;
            }
            if (_2059)
            {
                _1325.Store(_1304 * 56 + 36, asuint(0.0f));
                _1325.Store(_1304 * 56 + 32, asuint(0.0f));
                _1325.Store(_1304 * 56 + 28, asuint(0.0f));
                break;
            }
            float _2073 = 1.0f - _2033;
            _1325.Store(_1304 * 56 + 28, asuint(asfloat(_1325.Load(_1304 * 56 + 28)) * (_2291 / _2073)));
            _1325.Store(_1304 * 56 + 32, asuint(asfloat(_1325.Load(_1304 * 56 + 32)) * (_2292 / _2073)));
            _1325.Store(_1304 * 56 + 36, asuint(asfloat(_1325.Load(_1304 * 56 + 36)) * (_2293 / _2073)));
            ro += (_1360 * (_2186 + 9.9999997473787516355514526367188e-06f));
            _2183 = 0;
            _2186 = _1442 - _2186;
            _1325.Store(_1304 * 56 + 52, uint(int(_1325.Load(_1304 * 56 + 52)) + 16777216));
            rand_index += 7;
            continue;
        }
        float _2125 = asfloat(_1325.Load(_1304 * 56 + 0));
        float _2128 = asfloat(_1325.Load(_1304 * 56 + 4));
        float _2131 = asfloat(_1325.Load(_1304 * 56 + 8));
        float _2137 = _2186;
        float _2138 = _2137 + length(float3(_2125, _2128, _2131) - ro);
        _2186 = _2138;
        _2144.Store(_1304 * 24 + 0, uint(_2183));
        _2144.Store(_1304 * 24 + 4, uint(_2184));
        _2144.Store(_1304 * 24 + 8, uint(_2185));
        _2144.Store(_1304 * 24 + 12, asuint(_2138));
        _2144.Store(_1304 * 24 + 16, asuint(_2187));
        _2144.Store(_1304 * 24 + 20, asuint(_2188));
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
