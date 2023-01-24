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

ByteAddressBuffer _537 : register(t3, space0);
ByteAddressBuffer _752 : register(t7, space0);
ByteAddressBuffer _999 : register(t9, space0);
ByteAddressBuffer _1003 : register(t10, space0);
ByteAddressBuffer _1024 : register(t8, space0);
ByteAddressBuffer _1070 : register(t11, space0);
RWByteAddressBuffer _1199 : register(u12, space0);
ByteAddressBuffer _1334 : register(t4, space0);
ByteAddressBuffer _1384 : register(t5, space0);
ByteAddressBuffer _1421 : register(t6, space0);
ByteAddressBuffer _1543 : register(t1, space0);
ByteAddressBuffer _1547 : register(t2, space0);
ByteAddressBuffer _1726 : register(t15, space0);
RWByteAddressBuffer _2012 : register(u0, space0);
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
        float3 _2286 = inv_v;
        _2286.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _2286;
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
            float3 _2284 = inv_v;
            _2284.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _2284;
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
        float3 _2290 = inv_v;
        _2290.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _2290;
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
            float3 _2288 = inv_v;
            _2288.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _2288;
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
        float3 _2294 = inv_v;
        _2294.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _2294;
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
            float3 _2292 = inv_v;
            _2292.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _2292;
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
    int _2222 = 0;
    int _2223 = obj_index;
    float _2225 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _2224;
    float _2226;
    float _2227;
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
        hit_data_t _2234 = { _2222, _2223, _2224, _2225, _2226, _2227 };
        param_4 = _2234;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _2222 = param_4.mask;
        _2223 = param_4.obj_index;
        _2224 = param_4.prim_index;
        _2225 = param_4.t;
        _2226 = param_4.u;
        _2227 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _2222;
    int _571;
    if (_2222 != 0)
    {
        _571 = _2223;
    }
    else
    {
        _571 = out_inter.obj_index;
    }
    out_inter.obj_index = _571;
    int _584;
    if (_2222 != 0)
    {
        _584 = _2224;
    }
    else
    {
        _584 = out_inter.prim_index;
    }
    out_inter.prim_index = _584;
    out_inter.t = _2225;
    float _600;
    if (_2222 != 0)
    {
        _600 = _2226;
    }
    else
    {
        _600 = out_inter.u;
    }
    out_inter.u = _600;
    float _613;
    if (_2222 != 0)
    {
        _613 = _2227;
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
        float4 _2310 = res;
        _2310.x = _388.x;
        float4 _2312 = _2310;
        _2312.y = _388.y;
        float4 _2314 = _2312;
        _2314.z = _388.z;
        res = _2314;
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
        bool _1161 = gl_GlobalInvocationID.x >= _1157_g_params.img_size.x;
        bool _1170;
        if (!_1161)
        {
            _1170 = gl_GlobalInvocationID.y >= _1157_g_params.img_size.y;
        }
        else
        {
            _1170 = _1161;
        }
        if (_1170)
        {
            break;
        }
        int _1189 = (int(gl_GlobalInvocationID.y) * int(_1157_g_params.img_size.x)) + int(gl_GlobalInvocationID.x);
        float3 ro = float3(asfloat(_1199.Load(_1189 * 72 + 0)), asfloat(_1199.Load(_1189 * 72 + 4)), asfloat(_1199.Load(_1189 * 72 + 8)));
        float _1214 = asfloat(_1199.Load(_1189 * 72 + 12));
        float _1217 = asfloat(_1199.Load(_1189 * 72 + 16));
        float _1220 = asfloat(_1199.Load(_1189 * 72 + 20));
        float3 _1221 = float3(_1214, _1217, _1220);
        float3 param = _1221;
        float3 _1225 = safe_invert(param);
        int _2050 = 0;
        int _2052 = 0;
        int _2051 = 0;
        float _2053 = _1157_g_params.cam_clip_end;
        float _2055 = 0.0f;
        float _2054 = 0.0f;
        uint param_1 = uint(hash(int(_1199.Load(_1189 * 72 + 64))));
        float _1244 = construct_float(param_1);
        ray_data_t _1252;
        [unroll]
        for (int _31ident = 0; _31ident < 3; _31ident++)
        {
            _1252.o[_31ident] = asfloat(_1199.Load(_31ident * 4 + _1189 * 72 + 0));
        }
        [unroll]
        for (int _32ident = 0; _32ident < 3; _32ident++)
        {
            _1252.d[_32ident] = asfloat(_1199.Load(_32ident * 4 + _1189 * 72 + 12));
        }
        _1252.pdf = asfloat(_1199.Load(_1189 * 72 + 24));
        [unroll]
        for (int _33ident = 0; _33ident < 3; _33ident++)
        {
            _1252.c[_33ident] = asfloat(_1199.Load(_33ident * 4 + _1189 * 72 + 28));
        }
        [unroll]
        for (int _34ident = 0; _34ident < 4; _34ident++)
        {
            _1252.ior[_34ident] = asfloat(_1199.Load(_34ident * 4 + _1189 * 72 + 40));
        }
        _1252.cone_width = asfloat(_1199.Load(_1189 * 72 + 56));
        _1252.cone_spread = asfloat(_1199.Load(_1189 * 72 + 60));
        _1252.xy = int(_1199.Load(_1189 * 72 + 64));
        _1252.depth = int(_1199.Load(_1189 * 72 + 68));
        float _2162[4] = { _1252.ior[0], _1252.ior[1], _1252.ior[2], _1252.ior[3] };
        float _2153[3] = { _1252.c[0], _1252.c[1], _1252.c[2] };
        float _2146[3] = { _1252.d[0], _1252.d[1], _1252.d[2] };
        float _2139[3] = { _1252.o[0], _1252.o[1], _1252.o[2] };
        ray_data_t _2094 = { _2139, _2146, _1252.pdf, _2153, _2162, _1252.cone_width, _1252.cone_spread, _1252.xy, _1252.depth };
        int rand_index = _1157_g_params.hi + (total_depth(_2094) * 7);
        int _1365;
        float _1901;
        for (;;)
        {
            float _1312 = _2053;
            float3 param_2 = ro;
            float3 param_3 = _1221;
            float3 param_4 = _1225;
            uint param_5 = _1157_g_params.node_index;
            hit_data_t _2062 = { _2050, _2051, _2052, _1312, _2054, _2055 };
            hit_data_t param_6 = _2062;
            Traverse_MacroTree_WithStack(param_2, param_3, param_4, param_5, param_6);
            _2050 = param_6.mask;
            _2051 = param_6.obj_index;
            _2052 = param_6.prim_index;
            _2053 = param_6.t;
            _2054 = param_6.u;
            _2055 = param_6.v;
            if (param_6.prim_index < 0)
            {
                _2052 = (-1) - int(_1334.Load(((-1) - _2052) * 4 + 0));
            }
            else
            {
                _2052 = int(_1334.Load(_2052 * 4 + 0));
            }
            if (_2050 == 0)
            {
                break;
            }
            bool _1362 = _2052 < 0;
            if (_1362)
            {
                _1365 = (-1) - _2052;
            }
            else
            {
                _1365 = _2052;
            }
            uint _1376 = uint(_1365);
            bool _1378 = !_1362;
            bool _1393;
            if (_1378)
            {
                _1393 = ((_1384.Load(_1376 * 4 + 0) >> 16u) & 32768u) != 0u;
            }
            else
            {
                _1393 = _1378;
            }
            bool _1406;
            if (!_1393)
            {
                bool _1405;
                if (_1362)
                {
                    _1405 = (_1384.Load(_1376 * 4 + 0) & 32768u) != 0u;
                }
                else
                {
                    _1405 = _1362;
                }
                _1406 = _1405;
            }
            else
            {
                _1406 = _1393;
            }
            if (_1406)
            {
                break;
            }
            material_t _1430;
            [unroll]
            for (int _35ident = 0; _35ident < 5; _35ident++)
            {
                _1430.textures[_35ident] = _1421.Load(_35ident * 4 + ((_1384.Load(_1376 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _1430.base_color[_36ident] = asfloat(_1421.Load(_36ident * 4 + ((_1384.Load(_1376 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
            }
            _1430.flags = _1421.Load(((_1384.Load(_1376 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
            _1430.type = _1421.Load(((_1384.Load(_1376 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
            _1430.tangent_rotation_or_strength = asfloat(_1421.Load(((_1384.Load(_1376 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
            _1430.roughness_and_anisotropic = _1421.Load(((_1384.Load(_1376 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
            _1430.ior = asfloat(_1421.Load(((_1384.Load(_1376 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
            _1430.sheen_and_sheen_tint = _1421.Load(((_1384.Load(_1376 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
            _1430.tint_and_metallic = _1421.Load(((_1384.Load(_1376 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
            _1430.transmission_and_transmission_roughness = _1421.Load(((_1384.Load(_1376 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
            _1430.specular_and_specular_tint = _1421.Load(((_1384.Load(_1376 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
            _1430.clearcoat_and_clearcoat_roughness = _1421.Load(((_1384.Load(_1376 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
            _1430.normal_map_strength_unorm = _1421.Load(((_1384.Load(_1376 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
            uint _2164 = _1430.textures[1];
            uint _2166 = _1430.textures[3];
            uint _2167 = _1430.textures[4];
            float _2168 = _1430.base_color[0];
            float _2169 = _1430.base_color[1];
            float _2170 = _1430.base_color[2];
            uint _2098 = _1430.type;
            float _2099 = _1430.tangent_rotation_or_strength;
            if (_1362)
            {
                material_t _1486;
                [unroll]
                for (int _37ident = 0; _37ident < 5; _37ident++)
                {
                    _1486.textures[_37ident] = _1421.Load(_37ident * 4 + (_1384.Load(_1376 * 4 + 0) & 16383u) * 76 + 0);
                }
                [unroll]
                for (int _38ident = 0; _38ident < 3; _38ident++)
                {
                    _1486.base_color[_38ident] = asfloat(_1421.Load(_38ident * 4 + (_1384.Load(_1376 * 4 + 0) & 16383u) * 76 + 20));
                }
                _1486.flags = _1421.Load((_1384.Load(_1376 * 4 + 0) & 16383u) * 76 + 32);
                _1486.type = _1421.Load((_1384.Load(_1376 * 4 + 0) & 16383u) * 76 + 36);
                _1486.tangent_rotation_or_strength = asfloat(_1421.Load((_1384.Load(_1376 * 4 + 0) & 16383u) * 76 + 40));
                _1486.roughness_and_anisotropic = _1421.Load((_1384.Load(_1376 * 4 + 0) & 16383u) * 76 + 44);
                _1486.ior = asfloat(_1421.Load((_1384.Load(_1376 * 4 + 0) & 16383u) * 76 + 48));
                _1486.sheen_and_sheen_tint = _1421.Load((_1384.Load(_1376 * 4 + 0) & 16383u) * 76 + 52);
                _1486.tint_and_metallic = _1421.Load((_1384.Load(_1376 * 4 + 0) & 16383u) * 76 + 56);
                _1486.transmission_and_transmission_roughness = _1421.Load((_1384.Load(_1376 * 4 + 0) & 16383u) * 76 + 60);
                _1486.specular_and_specular_tint = _1421.Load((_1384.Load(_1376 * 4 + 0) & 16383u) * 76 + 64);
                _1486.clearcoat_and_clearcoat_roughness = _1421.Load((_1384.Load(_1376 * 4 + 0) & 16383u) * 76 + 68);
                _1486.normal_map_strength_unorm = _1421.Load((_1384.Load(_1376 * 4 + 0) & 16383u) * 76 + 72);
                _2164 = _1486.textures[1];
                _2166 = _1486.textures[3];
                _2167 = _1486.textures[4];
                _2168 = _1486.base_color[0];
                _2169 = _1486.base_color[1];
                _2170 = _1486.base_color[2];
                _2098 = _1486.type;
                _2099 = _1486.tangent_rotation_or_strength;
            }
            uint _1549 = _1376 * 3u;
            vertex_t _1555;
            [unroll]
            for (int _39ident = 0; _39ident < 3; _39ident++)
            {
                _1555.p[_39ident] = asfloat(_1543.Load(_39ident * 4 + _1547.Load(_1549 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _1555.n[_40ident] = asfloat(_1543.Load(_40ident * 4 + _1547.Load(_1549 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 3; _41ident++)
            {
                _1555.b[_41ident] = asfloat(_1543.Load(_41ident * 4 + _1547.Load(_1549 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _42ident = 0; _42ident < 2; _42ident++)
            {
                [unroll]
                for (int _43ident = 0; _43ident < 2; _43ident++)
                {
                    _1555.t[_42ident][_43ident] = asfloat(_1543.Load(_43ident * 4 + _42ident * 8 + _1547.Load(_1549 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1603;
            [unroll]
            for (int _44ident = 0; _44ident < 3; _44ident++)
            {
                _1603.p[_44ident] = asfloat(_1543.Load(_44ident * 4 + _1547.Load((_1549 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _45ident = 0; _45ident < 3; _45ident++)
            {
                _1603.n[_45ident] = asfloat(_1543.Load(_45ident * 4 + _1547.Load((_1549 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _46ident = 0; _46ident < 3; _46ident++)
            {
                _1603.b[_46ident] = asfloat(_1543.Load(_46ident * 4 + _1547.Load((_1549 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _47ident = 0; _47ident < 2; _47ident++)
            {
                [unroll]
                for (int _48ident = 0; _48ident < 2; _48ident++)
                {
                    _1603.t[_47ident][_48ident] = asfloat(_1543.Load(_48ident * 4 + _47ident * 8 + _1547.Load((_1549 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1649;
            [unroll]
            for (int _49ident = 0; _49ident < 3; _49ident++)
            {
                _1649.p[_49ident] = asfloat(_1543.Load(_49ident * 4 + _1547.Load((_1549 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _50ident = 0; _50ident < 3; _50ident++)
            {
                _1649.n[_50ident] = asfloat(_1543.Load(_50ident * 4 + _1547.Load((_1549 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _51ident = 0; _51ident < 3; _51ident++)
            {
                _1649.b[_51ident] = asfloat(_1543.Load(_51ident * 4 + _1547.Load((_1549 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _52ident = 0; _52ident < 2; _52ident++)
            {
                [unroll]
                for (int _53ident = 0; _53ident < 2; _53ident++)
                {
                    _1649.t[_52ident][_53ident] = asfloat(_1543.Load(_53ident * 4 + _52ident * 8 + _1547.Load((_1549 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            float2 _1721 = ((float2(_1555.t[0][0], _1555.t[0][1]) * ((1.0f - _2054) - _2055)) + (float2(_1603.t[0][0], _1603.t[0][1]) * _2054)) + (float2(_1649.t[0][0], _1649.t[0][1]) * _2055);
            float trans_r = frac(asfloat(_1726.Load(rand_index * 4 + 0)) + _1244);
            while (_2098 == 4u)
            {
                float mix_val = _2099;
                if (_2164 != 4294967295u)
                {
                    mix_val *= SampleBilinear(_2164, _1721, 0).x;
                }
                if (trans_r > mix_val)
                {
                    material_t _1766;
                    [unroll]
                    for (int _54ident = 0; _54ident < 5; _54ident++)
                    {
                        _1766.textures[_54ident] = _1421.Load(_54ident * 4 + _2166 * 76 + 0);
                    }
                    [unroll]
                    for (int _55ident = 0; _55ident < 3; _55ident++)
                    {
                        _1766.base_color[_55ident] = asfloat(_1421.Load(_55ident * 4 + _2166 * 76 + 20));
                    }
                    _1766.flags = _1421.Load(_2166 * 76 + 32);
                    _1766.type = _1421.Load(_2166 * 76 + 36);
                    _1766.tangent_rotation_or_strength = asfloat(_1421.Load(_2166 * 76 + 40));
                    _1766.roughness_and_anisotropic = _1421.Load(_2166 * 76 + 44);
                    _1766.ior = asfloat(_1421.Load(_2166 * 76 + 48));
                    _1766.sheen_and_sheen_tint = _1421.Load(_2166 * 76 + 52);
                    _1766.tint_and_metallic = _1421.Load(_2166 * 76 + 56);
                    _1766.transmission_and_transmission_roughness = _1421.Load(_2166 * 76 + 60);
                    _1766.specular_and_specular_tint = _1421.Load(_2166 * 76 + 64);
                    _1766.clearcoat_and_clearcoat_roughness = _1421.Load(_2166 * 76 + 68);
                    _1766.normal_map_strength_unorm = _1421.Load(_2166 * 76 + 72);
                    _2164 = _1766.textures[1];
                    _2166 = _1766.textures[3];
                    _2167 = _1766.textures[4];
                    _2168 = _1766.base_color[0];
                    _2169 = _1766.base_color[1];
                    _2170 = _1766.base_color[2];
                    _2098 = _1766.type;
                    _2099 = _1766.tangent_rotation_or_strength;
                    trans_r = (trans_r - mix_val) / (1.0f - mix_val);
                }
                else
                {
                    material_t _1819;
                    [unroll]
                    for (int _56ident = 0; _56ident < 5; _56ident++)
                    {
                        _1819.textures[_56ident] = _1421.Load(_56ident * 4 + _2167 * 76 + 0);
                    }
                    [unroll]
                    for (int _57ident = 0; _57ident < 3; _57ident++)
                    {
                        _1819.base_color[_57ident] = asfloat(_1421.Load(_57ident * 4 + _2167 * 76 + 20));
                    }
                    _1819.flags = _1421.Load(_2167 * 76 + 32);
                    _1819.type = _1421.Load(_2167 * 76 + 36);
                    _1819.tangent_rotation_or_strength = asfloat(_1421.Load(_2167 * 76 + 40));
                    _1819.roughness_and_anisotropic = _1421.Load(_2167 * 76 + 44);
                    _1819.ior = asfloat(_1421.Load(_2167 * 76 + 48));
                    _1819.sheen_and_sheen_tint = _1421.Load(_2167 * 76 + 52);
                    _1819.tint_and_metallic = _1421.Load(_2167 * 76 + 56);
                    _1819.transmission_and_transmission_roughness = _1421.Load(_2167 * 76 + 60);
                    _1819.specular_and_specular_tint = _1421.Load(_2167 * 76 + 64);
                    _1819.clearcoat_and_clearcoat_roughness = _1421.Load(_2167 * 76 + 68);
                    _1819.normal_map_strength_unorm = _1421.Load(_2167 * 76 + 72);
                    _2164 = _1819.textures[1];
                    _2166 = _1819.textures[3];
                    _2167 = _1819.textures[4];
                    _2168 = _1819.base_color[0];
                    _2169 = _1819.base_color[1];
                    _2170 = _1819.base_color[2];
                    _2098 = _1819.type;
                    _2099 = _1819.tangent_rotation_or_strength;
                    trans_r /= mix_val;
                }
            }
            if (_2098 != 5u)
            {
                break;
            }
            float _1890 = max(asfloat(_1199.Load(_1189 * 72 + 28)), max(asfloat(_1199.Load(_1189 * 72 + 32)), asfloat(_1199.Load(_1189 * 72 + 36))));
            if ((int(_1199.Load(_1189 * 72 + 68)) >> 24) > _1157_g_params.min_transp_depth)
            {
                _1901 = max(0.0500000007450580596923828125f, 1.0f - _1890);
            }
            else
            {
                _1901 = 0.0f;
            }
            bool _1915 = (frac(asfloat(_1726.Load((rand_index + 6) * 4 + 0)) + _1244) < _1901) || (_1890 == 0.0f);
            bool _1927;
            if (!_1915)
            {
                _1927 = ((int(_1199.Load(_1189 * 72 + 68)) >> 24) + 1) >= _1157_g_params.max_transp_depth;
            }
            else
            {
                _1927 = _1915;
            }
            if (_1927)
            {
                _1199.Store(_1189 * 72 + 36, asuint(0.0f));
                _1199.Store(_1189 * 72 + 32, asuint(0.0f));
                _1199.Store(_1189 * 72 + 28, asuint(0.0f));
                break;
            }
            float _1941 = 1.0f - _1901;
            _1199.Store(_1189 * 72 + 28, asuint(asfloat(_1199.Load(_1189 * 72 + 28)) * (_2168 / _1941)));
            _1199.Store(_1189 * 72 + 32, asuint(asfloat(_1199.Load(_1189 * 72 + 32)) * (_2169 / _1941)));
            _1199.Store(_1189 * 72 + 36, asuint(asfloat(_1199.Load(_1189 * 72 + 36)) * (_2170 / _1941)));
            ro += (_1221 * (_2053 + 9.9999997473787516355514526367188e-06f));
            _2050 = 0;
            _2053 = _1312 - _2053;
            _1199.Store(_1189 * 72 + 68, uint(int(_1199.Load(_1189 * 72 + 68)) + 16777216));
            rand_index += 7;
            continue;
        }
        float _1993 = asfloat(_1199.Load(_1189 * 72 + 0));
        float _1996 = asfloat(_1199.Load(_1189 * 72 + 4));
        float _1999 = asfloat(_1199.Load(_1189 * 72 + 8));
        float _2005 = _2053;
        float _2006 = _2005 + length(float3(_1993, _1996, _1999) - ro);
        _2053 = _2006;
        _2012.Store(_1189 * 24 + 0, uint(_2050));
        _2012.Store(_1189 * 24 + 4, uint(_2051));
        _2012.Store(_1189 * 24 + 8, uint(_2052));
        _2012.Store(_1189 * 24 + 12, asuint(_2006));
        _2012.Store(_1189 * 24 + 16, asuint(_2054));
        _2012.Store(_1189 * 24 + 20, asuint(_2055));
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
