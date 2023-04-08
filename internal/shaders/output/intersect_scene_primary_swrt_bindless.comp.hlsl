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
    float cam_clip_end;
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

ByteAddressBuffer _537 : register(t3, space0);
ByteAddressBuffer _752 : register(t7, space0);
ByteAddressBuffer _999 : register(t9, space0);
ByteAddressBuffer _1003 : register(t10, space0);
ByteAddressBuffer _1024 : register(t8, space0);
ByteAddressBuffer _1070 : register(t11, space0);
RWByteAddressBuffer _1206 : register(u12, space0);
ByteAddressBuffer _1341 : register(t4, space0);
ByteAddressBuffer _1391 : register(t5, space0);
ByteAddressBuffer _1428 : register(t6, space0);
ByteAddressBuffer _1550 : register(t1, space0);
ByteAddressBuffer _1554 : register(t2, space0);
ByteAddressBuffer _1733 : register(t15, space0);
RWByteAddressBuffer _2019 : register(u0, space0);
cbuffer UniformParams
{
    Params _1157_g_params : packoffset(c0);
};

Texture2D<float4> g_textures[] : register(t0, space1);
SamplerState _g_textures_sampler[] : register(s0, space1);

static uint3 gl_GlobalInvocationID;
static uint gl_LocalInvocationIndex;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
    uint gl_LocalInvocationIndex : SV_GroupIndex;
};

groupshared uint g_stack[64][48];

float3 safe_invert(float3 v)
{
    float3 inv_v = 1.0f.xxx / v;
    bool _151 = v.x <= 1.0000000116860974230803549289703e-07f;
    bool _158;
    if (_151)
    {
        _158 = v.x >= 0.0f;
    }
    else
    {
        _158 = _151;
    }
    if (_158)
    {
        float3 _2293 = inv_v;
        _2293.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _2293;
    }
    else
    {
        bool _167 = v.x >= (-1.0000000116860974230803549289703e-07f);
        bool _173;
        if (_167)
        {
            _173 = v.x < 0.0f;
        }
        else
        {
            _173 = _167;
        }
        if (_173)
        {
            float3 _2291 = inv_v;
            _2291.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _2291;
        }
    }
    bool _181 = v.y <= 1.0000000116860974230803549289703e-07f;
    bool _187;
    if (_181)
    {
        _187 = v.y >= 0.0f;
    }
    else
    {
        _187 = _181;
    }
    if (_187)
    {
        float3 _2297 = inv_v;
        _2297.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _2297;
    }
    else
    {
        bool _194 = v.y >= (-1.0000000116860974230803549289703e-07f);
        bool _200;
        if (_194)
        {
            _200 = v.y < 0.0f;
        }
        else
        {
            _200 = _194;
        }
        if (_200)
        {
            float3 _2295 = inv_v;
            _2295.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _2295;
        }
    }
    bool _207 = v.z <= 1.0000000116860974230803549289703e-07f;
    bool _213;
    if (_207)
    {
        _213 = v.z >= 0.0f;
    }
    else
    {
        _213 = _207;
    }
    if (_213)
    {
        float3 _2301 = inv_v;
        _2301.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _2301;
    }
    else
    {
        bool _220 = v.z >= (-1.0000000116860974230803549289703e-07f);
        bool _226;
        if (_220)
        {
            _226 = v.z < 0.0f;
        }
        else
        {
            _226 = _220;
        }
        if (_226)
        {
            float3 _2299 = inv_v;
            _2299.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _2299;
        }
    }
    return inv_v;
}

int hash(int x)
{
    uint _108 = uint(x);
    uint _115 = ((_108 >> uint(16)) ^ _108) * 73244475u;
    uint _120 = ((_115 >> uint(16)) ^ _115) * 73244475u;
    return int((_120 >> uint(16)) ^ _120);
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
    float _630 = mad(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float _638 = mad(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float _653 = mad(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    float _660 = mad(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    float _677 = mad(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    float _684 = mad(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    float _689 = max(max(min(_630, _638), min(_653, _660)), min(_677, _684));
    float _697 = min(min(max(_630, _638), max(_653, _660)), max(_677, _684)) * 1.0000002384185791015625f;
    return ((_689 <= _697) && (_689 <= t)) && (_697 > 0.0f);
}

void IntersectTri(float3 ro, float3 rd, tri_accel_t tri, uint prim_index, inout hit_data_t inter)
{
    do
    {
        float _407 = dot(rd, tri.n_plane.xyz);
        float _416 = tri.n_plane.w - dot(ro, tri.n_plane.xyz);
        if (sign(_416) != sign(mad(_407, inter.t, -_416)))
        {
            break;
        }
        float3 _437 = (ro * _407) + (rd * _416);
        float _448 = mad(_407, tri.u_plane.w, dot(_437, tri.u_plane.xyz));
        float _453 = _407 - _448;
        if (sign(_448) != sign(_453))
        {
            break;
        }
        float _470 = mad(_407, tri.v_plane.w, dot(_437, tri.v_plane.xyz));
        if (sign(_470) != sign(_453 - _470))
        {
            break;
        }
        float _485 = 1.0f / _407;
        inter.mask = -1;
        int _490;
        if (_407 < 0.0f)
        {
            _490 = int(prim_index);
        }
        else
        {
            _490 = (-1) - int(prim_index);
        }
        inter.prim_index = _490;
        inter.t = _416 * _485;
        inter.u = _448 * _485;
        inter.v = _470 * _485;
        break;
    } while(false);
}

void IntersectTris_ClosestHit(float3 ro, float3 rd, int tri_start, int tri_end, int obj_index, inout hit_data_t out_inter)
{
    int _2229 = 0;
    int _2230 = obj_index;
    float _2232 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _2231;
    float _2233;
    float _2234;
    for (int i = tri_start; i < tri_end; )
    {
        param = ro;
        param_1 = rd;
        tri_accel_t _548;
        _548.n_plane = asfloat(_537.Load4(i * 48 + 0));
        _548.u_plane = asfloat(_537.Load4(i * 48 + 16));
        _548.v_plane = asfloat(_537.Load4(i * 48 + 32));
        param_2.n_plane = _548.n_plane;
        param_2.u_plane = _548.u_plane;
        param_2.v_plane = _548.v_plane;
        param_3 = uint(i);
        hit_data_t _2241 = { _2229, _2230, _2231, _2232, _2233, _2234 };
        param_4 = _2241;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _2229 = param_4.mask;
        _2230 = param_4.obj_index;
        _2231 = param_4.prim_index;
        _2232 = param_4.t;
        _2233 = param_4.u;
        _2234 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _2229;
    int _571;
    if (_2229 != 0)
    {
        _571 = _2230;
    }
    else
    {
        _571 = out_inter.obj_index;
    }
    out_inter.obj_index = _571;
    int _584;
    if (_2229 != 0)
    {
        _584 = _2231;
    }
    else
    {
        _584 = out_inter.prim_index;
    }
    out_inter.prim_index = _584;
    out_inter.t = _2232;
    float _600;
    if (_2229 != 0)
    {
        _600 = _2233;
    }
    else
    {
        _600 = out_inter.u;
    }
    out_inter.u = _600;
    float _613;
    if (_2229 != 0)
    {
        _613 = _2234;
    }
    else
    {
        _613 = out_inter.v;
    }
    out_inter.v = _613;
}

void Traverse_MicroTree_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    float3 _714 = (-inv_d) * ro;
    uint _716 = stack_size;
    uint _726 = stack_size;
    stack_size = _726 + uint(1);
    g_stack[gl_LocalInvocationIndex][_726] = node_index;
    uint _800;
    uint _824;
    while (stack_size != _716)
    {
        uint _741 = stack_size;
        uint _742 = _741 - uint(1);
        stack_size = _742;
        bvh_node_t _756;
        _756.bbox_min = asfloat(_752.Load4(g_stack[gl_LocalInvocationIndex][_742] * 32 + 0));
        _756.bbox_max = asfloat(_752.Load4(g_stack[gl_LocalInvocationIndex][_742] * 32 + 16));
        float3 param = inv_d;
        float3 param_1 = _714;
        float param_2 = inter.t;
        float3 param_3 = _756.bbox_min.xyz;
        float3 param_4 = _756.bbox_max.xyz;
        if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
        {
            continue;
        }
        uint _783 = asuint(_756.bbox_min.w);
        if ((_783 & 2147483648u) == 0u)
        {
            uint _790 = stack_size;
            stack_size = _790 + uint(1);
            uint _794 = asuint(_756.bbox_max.w);
            uint _796 = _794 >> uint(30);
            if (rd[_796] < 0.0f)
            {
                _800 = _783;
            }
            else
            {
                _800 = _794 & 1073741823u;
            }
            g_stack[gl_LocalInvocationIndex][_790] = _800;
            uint _815 = stack_size;
            stack_size = _815 + uint(1);
            if (rd[_796] < 0.0f)
            {
                _824 = _794 & 1073741823u;
            }
            else
            {
                _824 = _783;
            }
            g_stack[gl_LocalInvocationIndex][_815] = _824;
        }
        else
        {
            int _844 = int(_783 & 2147483647u);
            float3 param_5 = ro;
            float3 param_6 = rd;
            int param_7 = _844;
            int param_8 = _844 + asint(_756.bbox_max.w);
            int param_9 = obj_index;
            hit_data_t param_10 = inter;
            IntersectTris_ClosestHit(param_5, param_6, param_7, param_8, param_9, param_10);
            inter = param_10;
        }
    }
}

void Traverse_MacroTree_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    float3 _869 = (-orig_inv_rd) * orig_ro;
    uint stack_size = 1u;
    g_stack[gl_LocalInvocationIndex][0u] = node_index;
    uint _934;
    uint _957;
    while (stack_size != 0u)
    {
        uint _885 = stack_size;
        uint _886 = _885 - uint(1);
        stack_size = _886;
        bvh_node_t _892;
        _892.bbox_min = asfloat(_752.Load4(g_stack[gl_LocalInvocationIndex][_886] * 32 + 0));
        _892.bbox_max = asfloat(_752.Load4(g_stack[gl_LocalInvocationIndex][_886] * 32 + 16));
        float3 param = orig_inv_rd;
        float3 param_1 = _869;
        float param_2 = inter.t;
        float3 param_3 = _892.bbox_min.xyz;
        float3 param_4 = _892.bbox_max.xyz;
        if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
        {
            continue;
        }
        uint _919 = asuint(_892.bbox_min.w);
        if ((_919 & 2147483648u) == 0u)
        {
            uint _925 = stack_size;
            stack_size = _925 + uint(1);
            uint _929 = asuint(_892.bbox_max.w);
            uint _930 = _929 >> uint(30);
            if (orig_rd[_930] < 0.0f)
            {
                _934 = _919;
            }
            else
            {
                _934 = _929 & 1073741823u;
            }
            g_stack[gl_LocalInvocationIndex][_925] = _934;
            uint _948 = stack_size;
            stack_size = _948 + uint(1);
            if (orig_rd[_930] < 0.0f)
            {
                _957 = _929 & 1073741823u;
            }
            else
            {
                _957 = _919;
            }
            g_stack[gl_LocalInvocationIndex][_948] = _957;
        }
        else
        {
            uint _975 = _919 & 2147483647u;
            uint _979 = asuint(_892.bbox_max.w);
            for (uint i = _975; i < (_975 + _979); i++)
            {
                mesh_instance_t _1010;
                _1010.bbox_min = asfloat(_999.Load4(_1003.Load(i * 4 + 0) * 32 + 0));
                _1010.bbox_max = asfloat(_999.Load4(_1003.Load(i * 4 + 0) * 32 + 16));
                mesh_t _1030;
                [unroll]
                for (int _29ident = 0; _29ident < 3; _29ident++)
                {
                    _1030.bbox_min[_29ident] = asfloat(_1024.Load(_29ident * 4 + asuint(_1010.bbox_max.w) * 48 + 0));
                }
                [unroll]
                for (int _30ident = 0; _30ident < 3; _30ident++)
                {
                    _1030.bbox_max[_30ident] = asfloat(_1024.Load(_30ident * 4 + asuint(_1010.bbox_max.w) * 48 + 12));
                }
                _1030.node_index = _1024.Load(asuint(_1010.bbox_max.w) * 48 + 24);
                _1030.node_count = _1024.Load(asuint(_1010.bbox_max.w) * 48 + 28);
                _1030.tris_index = _1024.Load(asuint(_1010.bbox_max.w) * 48 + 32);
                _1030.tris_count = _1024.Load(asuint(_1010.bbox_max.w) * 48 + 36);
                _1030.vert_index = _1024.Load(asuint(_1010.bbox_max.w) * 48 + 40);
                _1030.vert_count = _1024.Load(asuint(_1010.bbox_max.w) * 48 + 44);
                transform_t _1076;
                _1076.xform = asfloat(uint4x4(_1070.Load4(asuint(_1010.bbox_min.w) * 128 + 0), _1070.Load4(asuint(_1010.bbox_min.w) * 128 + 16), _1070.Load4(asuint(_1010.bbox_min.w) * 128 + 32), _1070.Load4(asuint(_1010.bbox_min.w) * 128 + 48)));
                _1076.inv_xform = asfloat(uint4x4(_1070.Load4(asuint(_1010.bbox_min.w) * 128 + 64), _1070.Load4(asuint(_1010.bbox_min.w) * 128 + 80), _1070.Load4(asuint(_1010.bbox_min.w) * 128 + 96), _1070.Load4(asuint(_1010.bbox_min.w) * 128 + 112)));
                float3 param_5 = orig_inv_rd;
                float3 param_6 = _869;
                float param_7 = inter.t;
                float3 param_8 = _1010.bbox_min.xyz;
                float3 param_9 = _1010.bbox_max.xyz;
                if (!_bbox_test_fma(param_5, param_6, param_7, param_8, param_9))
                {
                    continue;
                }
                float3 _1121 = mul(float4(orig_rd, 0.0f), _1076.inv_xform).xyz;
                float3 param_10 = _1121;
                float3 param_11 = mul(float4(orig_ro, 1.0f), _1076.inv_xform).xyz;
                float3 param_12 = _1121;
                float3 param_13 = safe_invert(param_10);
                int param_14 = int(_1003.Load(i * 4 + 0));
                uint param_15 = _1030.node_index;
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
    float _281 = mad(col.z, 31.875f, 1.0f);
    float _291 = (col.x - 0.501960813999176025390625f) / _281;
    float _297 = (col.y - 0.501960813999176025390625f) / _281;
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
    uint _353 = index & 16777215u;
    float4 res = g_textures[NonUniformResourceIndex(_353)].SampleLevel(_g_textures_sampler[NonUniformResourceIndex(_353)], uvs, float(lod));
    bool _364;
    if (maybe_YCoCg)
    {
        _364 = (index & 67108864u) != 0u;
    }
    else
    {
        _364 = maybe_YCoCg;
    }
    if (_364)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _382;
    if (maybe_SRGB)
    {
        _382 = (index & 16777216u) != 0u;
    }
    else
    {
        _382 = maybe_SRGB;
    }
    if (_382)
    {
        float3 param_1 = res.xyz;
        float3 _388 = srgb_to_rgb(param_1);
        float4 _2317 = res;
        _2317.x = _388.x;
        float4 _2319 = _2317;
        _2319.y = _388.y;
        float4 _2321 = _2319;
        _2321.z = _388.z;
        res = _2321;
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
        bool _1161 = gl_GlobalInvocationID.x >= _1157_g_params.rect.z;
        bool _1170;
        if (!_1161)
        {
            _1170 = gl_GlobalInvocationID.y >= _1157_g_params.rect.w;
        }
        else
        {
            _1170 = _1161;
        }
        if (_1170)
        {
            break;
        }
        int _1196 = int(gl_GlobalInvocationID.y * _1157_g_params.rect.z) + int(_1157_g_params.rect.x + gl_GlobalInvocationID.x);
        float3 ro = float3(asfloat(_1206.Load(_1196 * 72 + 0)), asfloat(_1206.Load(_1196 * 72 + 4)), asfloat(_1206.Load(_1196 * 72 + 8)));
        float _1221 = asfloat(_1206.Load(_1196 * 72 + 12));
        float _1224 = asfloat(_1206.Load(_1196 * 72 + 16));
        float _1227 = asfloat(_1206.Load(_1196 * 72 + 20));
        float3 _1228 = float3(_1221, _1224, _1227);
        float3 param = _1228;
        float3 _1232 = safe_invert(param);
        int _2057 = 0;
        int _2059 = 0;
        int _2058 = 0;
        float _2060 = _1157_g_params.cam_clip_end;
        float _2062 = 0.0f;
        float _2061 = 0.0f;
        uint param_1 = uint(hash(int(_1206.Load(_1196 * 72 + 64))));
        float _1251 = construct_float(param_1);
        ray_data_t _1259;
        [unroll]
        for (int _31ident = 0; _31ident < 3; _31ident++)
        {
            _1259.o[_31ident] = asfloat(_1206.Load(_31ident * 4 + _1196 * 72 + 0));
        }
        [unroll]
        for (int _32ident = 0; _32ident < 3; _32ident++)
        {
            _1259.d[_32ident] = asfloat(_1206.Load(_32ident * 4 + _1196 * 72 + 12));
        }
        _1259.pdf = asfloat(_1206.Load(_1196 * 72 + 24));
        [unroll]
        for (int _33ident = 0; _33ident < 3; _33ident++)
        {
            _1259.c[_33ident] = asfloat(_1206.Load(_33ident * 4 + _1196 * 72 + 28));
        }
        [unroll]
        for (int _34ident = 0; _34ident < 4; _34ident++)
        {
            _1259.ior[_34ident] = asfloat(_1206.Load(_34ident * 4 + _1196 * 72 + 40));
        }
        _1259.cone_width = asfloat(_1206.Load(_1196 * 72 + 56));
        _1259.cone_spread = asfloat(_1206.Load(_1196 * 72 + 60));
        _1259.xy = int(_1206.Load(_1196 * 72 + 64));
        _1259.depth = int(_1206.Load(_1196 * 72 + 68));
        float _2169[4] = { _1259.ior[0], _1259.ior[1], _1259.ior[2], _1259.ior[3] };
        float _2160[3] = { _1259.c[0], _1259.c[1], _1259.c[2] };
        float _2153[3] = { _1259.d[0], _1259.d[1], _1259.d[2] };
        float _2146[3] = { _1259.o[0], _1259.o[1], _1259.o[2] };
        ray_data_t _2101 = { _2146, _2153, _1259.pdf, _2160, _2169, _1259.cone_width, _1259.cone_spread, _1259.xy, _1259.depth };
        int rand_index = _1157_g_params.hi + (total_depth(_2101) * 7);
        int _1372;
        float _1908;
        for (;;)
        {
            float _1319 = _2060;
            float3 param_2 = ro;
            float3 param_3 = _1228;
            float3 param_4 = _1232;
            uint param_5 = _1157_g_params.node_index;
            hit_data_t _2069 = { _2057, _2058, _2059, _1319, _2061, _2062 };
            hit_data_t param_6 = _2069;
            Traverse_MacroTree_WithStack(param_2, param_3, param_4, param_5, param_6);
            _2057 = param_6.mask;
            _2058 = param_6.obj_index;
            _2059 = param_6.prim_index;
            _2060 = param_6.t;
            _2061 = param_6.u;
            _2062 = param_6.v;
            if (param_6.prim_index < 0)
            {
                _2059 = (-1) - int(_1341.Load(((-1) - _2059) * 4 + 0));
            }
            else
            {
                _2059 = int(_1341.Load(_2059 * 4 + 0));
            }
            if (_2057 == 0)
            {
                break;
            }
            bool _1369 = _2059 < 0;
            if (_1369)
            {
                _1372 = (-1) - _2059;
            }
            else
            {
                _1372 = _2059;
            }
            uint _1383 = uint(_1372);
            bool _1385 = !_1369;
            bool _1400;
            if (_1385)
            {
                _1400 = ((_1391.Load(_1383 * 4 + 0) >> 16u) & 32768u) != 0u;
            }
            else
            {
                _1400 = _1385;
            }
            bool _1413;
            if (!_1400)
            {
                bool _1412;
                if (_1369)
                {
                    _1412 = (_1391.Load(_1383 * 4 + 0) & 32768u) != 0u;
                }
                else
                {
                    _1412 = _1369;
                }
                _1413 = _1412;
            }
            else
            {
                _1413 = _1400;
            }
            if (_1413)
            {
                break;
            }
            material_t _1437;
            [unroll]
            for (int _35ident = 0; _35ident < 5; _35ident++)
            {
                _1437.textures[_35ident] = _1428.Load(_35ident * 4 + ((_1391.Load(_1383 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _1437.base_color[_36ident] = asfloat(_1428.Load(_36ident * 4 + ((_1391.Load(_1383 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
            }
            _1437.flags = _1428.Load(((_1391.Load(_1383 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
            _1437.type = _1428.Load(((_1391.Load(_1383 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
            _1437.tangent_rotation_or_strength = asfloat(_1428.Load(((_1391.Load(_1383 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
            _1437.roughness_and_anisotropic = _1428.Load(((_1391.Load(_1383 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
            _1437.ior = asfloat(_1428.Load(((_1391.Load(_1383 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
            _1437.sheen_and_sheen_tint = _1428.Load(((_1391.Load(_1383 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
            _1437.tint_and_metallic = _1428.Load(((_1391.Load(_1383 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
            _1437.transmission_and_transmission_roughness = _1428.Load(((_1391.Load(_1383 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
            _1437.specular_and_specular_tint = _1428.Load(((_1391.Load(_1383 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
            _1437.clearcoat_and_clearcoat_roughness = _1428.Load(((_1391.Load(_1383 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
            _1437.normal_map_strength_unorm = _1428.Load(((_1391.Load(_1383 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
            uint _2171 = _1437.textures[1];
            uint _2173 = _1437.textures[3];
            uint _2174 = _1437.textures[4];
            float _2175 = _1437.base_color[0];
            float _2176 = _1437.base_color[1];
            float _2177 = _1437.base_color[2];
            uint _2105 = _1437.type;
            float _2106 = _1437.tangent_rotation_or_strength;
            if (_1369)
            {
                material_t _1493;
                [unroll]
                for (int _37ident = 0; _37ident < 5; _37ident++)
                {
                    _1493.textures[_37ident] = _1428.Load(_37ident * 4 + (_1391.Load(_1383 * 4 + 0) & 16383u) * 76 + 0);
                }
                [unroll]
                for (int _38ident = 0; _38ident < 3; _38ident++)
                {
                    _1493.base_color[_38ident] = asfloat(_1428.Load(_38ident * 4 + (_1391.Load(_1383 * 4 + 0) & 16383u) * 76 + 20));
                }
                _1493.flags = _1428.Load((_1391.Load(_1383 * 4 + 0) & 16383u) * 76 + 32);
                _1493.type = _1428.Load((_1391.Load(_1383 * 4 + 0) & 16383u) * 76 + 36);
                _1493.tangent_rotation_or_strength = asfloat(_1428.Load((_1391.Load(_1383 * 4 + 0) & 16383u) * 76 + 40));
                _1493.roughness_and_anisotropic = _1428.Load((_1391.Load(_1383 * 4 + 0) & 16383u) * 76 + 44);
                _1493.ior = asfloat(_1428.Load((_1391.Load(_1383 * 4 + 0) & 16383u) * 76 + 48));
                _1493.sheen_and_sheen_tint = _1428.Load((_1391.Load(_1383 * 4 + 0) & 16383u) * 76 + 52);
                _1493.tint_and_metallic = _1428.Load((_1391.Load(_1383 * 4 + 0) & 16383u) * 76 + 56);
                _1493.transmission_and_transmission_roughness = _1428.Load((_1391.Load(_1383 * 4 + 0) & 16383u) * 76 + 60);
                _1493.specular_and_specular_tint = _1428.Load((_1391.Load(_1383 * 4 + 0) & 16383u) * 76 + 64);
                _1493.clearcoat_and_clearcoat_roughness = _1428.Load((_1391.Load(_1383 * 4 + 0) & 16383u) * 76 + 68);
                _1493.normal_map_strength_unorm = _1428.Load((_1391.Load(_1383 * 4 + 0) & 16383u) * 76 + 72);
                _2171 = _1493.textures[1];
                _2173 = _1493.textures[3];
                _2174 = _1493.textures[4];
                _2175 = _1493.base_color[0];
                _2176 = _1493.base_color[1];
                _2177 = _1493.base_color[2];
                _2105 = _1493.type;
                _2106 = _1493.tangent_rotation_or_strength;
            }
            uint _1556 = _1383 * 3u;
            vertex_t _1562;
            [unroll]
            for (int _39ident = 0; _39ident < 3; _39ident++)
            {
                _1562.p[_39ident] = asfloat(_1550.Load(_39ident * 4 + _1554.Load(_1556 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _1562.n[_40ident] = asfloat(_1550.Load(_40ident * 4 + _1554.Load(_1556 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 3; _41ident++)
            {
                _1562.b[_41ident] = asfloat(_1550.Load(_41ident * 4 + _1554.Load(_1556 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _42ident = 0; _42ident < 2; _42ident++)
            {
                [unroll]
                for (int _43ident = 0; _43ident < 2; _43ident++)
                {
                    _1562.t[_42ident][_43ident] = asfloat(_1550.Load(_43ident * 4 + _42ident * 8 + _1554.Load(_1556 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1610;
            [unroll]
            for (int _44ident = 0; _44ident < 3; _44ident++)
            {
                _1610.p[_44ident] = asfloat(_1550.Load(_44ident * 4 + _1554.Load((_1556 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _45ident = 0; _45ident < 3; _45ident++)
            {
                _1610.n[_45ident] = asfloat(_1550.Load(_45ident * 4 + _1554.Load((_1556 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _46ident = 0; _46ident < 3; _46ident++)
            {
                _1610.b[_46ident] = asfloat(_1550.Load(_46ident * 4 + _1554.Load((_1556 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _47ident = 0; _47ident < 2; _47ident++)
            {
                [unroll]
                for (int _48ident = 0; _48ident < 2; _48ident++)
                {
                    _1610.t[_47ident][_48ident] = asfloat(_1550.Load(_48ident * 4 + _47ident * 8 + _1554.Load((_1556 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1656;
            [unroll]
            for (int _49ident = 0; _49ident < 3; _49ident++)
            {
                _1656.p[_49ident] = asfloat(_1550.Load(_49ident * 4 + _1554.Load((_1556 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _50ident = 0; _50ident < 3; _50ident++)
            {
                _1656.n[_50ident] = asfloat(_1550.Load(_50ident * 4 + _1554.Load((_1556 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _51ident = 0; _51ident < 3; _51ident++)
            {
                _1656.b[_51ident] = asfloat(_1550.Load(_51ident * 4 + _1554.Load((_1556 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _52ident = 0; _52ident < 2; _52ident++)
            {
                [unroll]
                for (int _53ident = 0; _53ident < 2; _53ident++)
                {
                    _1656.t[_52ident][_53ident] = asfloat(_1550.Load(_53ident * 4 + _52ident * 8 + _1554.Load((_1556 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            float2 _1728 = ((float2(_1562.t[0][0], _1562.t[0][1]) * ((1.0f - _2061) - _2062)) + (float2(_1610.t[0][0], _1610.t[0][1]) * _2061)) + (float2(_1656.t[0][0], _1656.t[0][1]) * _2062);
            float trans_r = frac(asfloat(_1733.Load(rand_index * 4 + 0)) + _1251);
            while (_2105 == 4u)
            {
                float mix_val = _2106;
                if (_2171 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_2171, _1728, 0).x;
                }
                if (trans_r > mix_val)
                {
                    material_t _1773;
                    [unroll]
                    for (int _54ident = 0; _54ident < 5; _54ident++)
                    {
                        _1773.textures[_54ident] = _1428.Load(_54ident * 4 + _2173 * 76 + 0);
                    }
                    [unroll]
                    for (int _55ident = 0; _55ident < 3; _55ident++)
                    {
                        _1773.base_color[_55ident] = asfloat(_1428.Load(_55ident * 4 + _2173 * 76 + 20));
                    }
                    _1773.flags = _1428.Load(_2173 * 76 + 32);
                    _1773.type = _1428.Load(_2173 * 76 + 36);
                    _1773.tangent_rotation_or_strength = asfloat(_1428.Load(_2173 * 76 + 40));
                    _1773.roughness_and_anisotropic = _1428.Load(_2173 * 76 + 44);
                    _1773.ior = asfloat(_1428.Load(_2173 * 76 + 48));
                    _1773.sheen_and_sheen_tint = _1428.Load(_2173 * 76 + 52);
                    _1773.tint_and_metallic = _1428.Load(_2173 * 76 + 56);
                    _1773.transmission_and_transmission_roughness = _1428.Load(_2173 * 76 + 60);
                    _1773.specular_and_specular_tint = _1428.Load(_2173 * 76 + 64);
                    _1773.clearcoat_and_clearcoat_roughness = _1428.Load(_2173 * 76 + 68);
                    _1773.normal_map_strength_unorm = _1428.Load(_2173 * 76 + 72);
                    _2171 = _1773.textures[1];
                    _2173 = _1773.textures[3];
                    _2174 = _1773.textures[4];
                    _2175 = _1773.base_color[0];
                    _2176 = _1773.base_color[1];
                    _2177 = _1773.base_color[2];
                    _2105 = _1773.type;
                    _2106 = _1773.tangent_rotation_or_strength;
                    trans_r = (trans_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _1826;
                    [unroll]
                    for (int _56ident = 0; _56ident < 5; _56ident++)
                    {
                        _1826.textures[_56ident] = _1428.Load(_56ident * 4 + _2174 * 76 + 0);
                    }
                    [unroll]
                    for (int _57ident = 0; _57ident < 3; _57ident++)
                    {
                        _1826.base_color[_57ident] = asfloat(_1428.Load(_57ident * 4 + _2174 * 76 + 20));
                    }
                    _1826.flags = _1428.Load(_2174 * 76 + 32);
                    _1826.type = _1428.Load(_2174 * 76 + 36);
                    _1826.tangent_rotation_or_strength = asfloat(_1428.Load(_2174 * 76 + 40));
                    _1826.roughness_and_anisotropic = _1428.Load(_2174 * 76 + 44);
                    _1826.ior = asfloat(_1428.Load(_2174 * 76 + 48));
                    _1826.sheen_and_sheen_tint = _1428.Load(_2174 * 76 + 52);
                    _1826.tint_and_metallic = _1428.Load(_2174 * 76 + 56);
                    _1826.transmission_and_transmission_roughness = _1428.Load(_2174 * 76 + 60);
                    _1826.specular_and_specular_tint = _1428.Load(_2174 * 76 + 64);
                    _1826.clearcoat_and_clearcoat_roughness = _1428.Load(_2174 * 76 + 68);
                    _1826.normal_map_strength_unorm = _1428.Load(_2174 * 76 + 72);
                    _2171 = _1826.textures[1];
                    _2173 = _1826.textures[3];
                    _2174 = _1826.textures[4];
                    _2175 = _1826.base_color[0];
                    _2176 = _1826.base_color[1];
                    _2177 = _1826.base_color[2];
                    _2105 = _1826.type;
                    _2106 = _1826.tangent_rotation_or_strength;
                    trans_r /= mix_val;
                }
            }
            if (_2105 != 5u)
            {
                break;
            }
            float _1897 = max(asfloat(_1206.Load(_1196 * 72 + 28)), max(asfloat(_1206.Load(_1196 * 72 + 32)), asfloat(_1206.Load(_1196 * 72 + 36))));
            if ((int(_1206.Load(_1196 * 72 + 68)) >> 24) > _1157_g_params.min_transp_depth)
            {
                _1908 = max(0.0500000007450580596923828125f, 1.0f - _1897);
            }
            else
            {
                _1908 = 0.0f;
            }
            bool _1922 = (frac(asfloat(_1733.Load((rand_index + 6) * 4 + 0)) + _1251) < _1908) || (_1897 == 0.0f);
            bool _1934;
            if (!_1922)
            {
                _1934 = ((int(_1206.Load(_1196 * 72 + 68)) >> 24) + 1) >= _1157_g_params.max_transp_depth;
            }
            else
            {
                _1934 = _1922;
            }
            if (_1934)
            {
                _1206.Store(_1196 * 72 + 36, asuint(0.0f));
                _1206.Store(_1196 * 72 + 32, asuint(0.0f));
                _1206.Store(_1196 * 72 + 28, asuint(0.0f));
                break;
            }
            float _1948 = 1.0f - _1908;
            _1206.Store(_1196 * 72 + 28, asuint(asfloat(_1206.Load(_1196 * 72 + 28)) * (_2175 / _1948)));
            _1206.Store(_1196 * 72 + 32, asuint(asfloat(_1206.Load(_1196 * 72 + 32)) * (_2176 / _1948)));
            _1206.Store(_1196 * 72 + 36, asuint(asfloat(_1206.Load(_1196 * 72 + 36)) * (_2177 / _1948)));
            ro += (_1228 * (_2060 + 9.9999997473787516355514526367188e-06f));
            _2057 = 0;
            _2060 = _1319 - _2060;
            _1206.Store(_1196 * 72 + 68, uint(int(_1206.Load(_1196 * 72 + 68)) + 16777216));
            rand_index += 7;
            continue;
        }
        float _2000 = asfloat(_1206.Load(_1196 * 72 + 0));
        float _2003 = asfloat(_1206.Load(_1196 * 72 + 4));
        float _2006 = asfloat(_1206.Load(_1196 * 72 + 8));
        float _2012 = _2060;
        float _2013 = _2012 + length(float3(_2000, _2003, _2006) - ro);
        _2060 = _2013;
        _2019.Store(_1196 * 24 + 0, uint(_2057));
        _2019.Store(_1196 * 24 + 4, uint(_2058));
        _2019.Store(_1196 * 24 + 8, uint(_2059));
        _2019.Store(_1196 * 24 + 12, asuint(_2013));
        _2019.Store(_1196 * 24 + 16, asuint(_2061));
        _2019.Store(_1196 * 24 + 20, asuint(_2062));
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    gl_LocalInvocationIndex = stage_input.gl_LocalInvocationIndex;
    comp_main();
}
