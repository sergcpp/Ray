struct shadow_ray_t
{
    float o[3];
    int depth;
    float d[3];
    float dist;
    float c[3];
    int xy;
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
    uint node_index;
    int max_transp_depth;
    int blocker_lights_count;
    float clamp_val;
    int hi;
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
    float ior;
    uint sheen_and_sheen_tint;
    uint tint_and_metallic;
    uint transmission_and_transmission_roughness;
    uint specular_and_specular_tint;
    uint clearcoat_and_clearcoat_roughness;
    uint normal_map_strength_unorm;
};

struct light_t
{
    uint4 type_and_param0;
    float4 param1;
    float4 param2;
    float4 param3;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

ByteAddressBuffer _579 : register(t1, space0);
ByteAddressBuffer _799 : register(t5, space0);
ByteAddressBuffer _939 : register(t2, space0);
ByteAddressBuffer _948 : register(t3, space0);
ByteAddressBuffer _1121 : register(t7, space0);
ByteAddressBuffer _1125 : register(t8, space0);
ByteAddressBuffer _1145 : register(t6, space0);
ByteAddressBuffer _1191 : register(t9, space0);
ByteAddressBuffer _1450 : register(t10, space0);
ByteAddressBuffer _1454 : register(t11, space0);
ByteAddressBuffer _1640 : register(t16, space0);
ByteAddressBuffer _1678 : register(t4, space0);
ByteAddressBuffer _1882 : register(t15, space0);
ByteAddressBuffer _1894 : register(t14, space0);
ByteAddressBuffer _2132 : register(t13, space0);
ByteAddressBuffer _2147 : register(t12, space0);
cbuffer UniformParams
{
    Params _1324_g_params : packoffset(c0);
};

Texture2D<float4> g_textures[] : register(t0, space1);
SamplerState _g_textures_sampler[] : register(s0, space1);
RWTexture2D<float4> g_inout_img : register(u0, space0);

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

int hash(int x)
{
    uint _121 = uint(x);
    uint _128 = ((_121 >> uint(16)) ^ _121) * 73244475u;
    uint _133 = ((_128 >> uint(16)) ^ _128) * 73244475u;
    return int((_133 >> uint(16)) ^ _133);
}

float construct_float(inout uint m)
{
    m &= 8388607u;
    m |= 1065353216u;
    return asfloat(m) - 1.0f;
}

int total_depth(shadow_ray_t r)
{
    return (((r.depth & 255) + ((r.depth >> 8) & 255)) + ((r.depth >> 16) & 255)) + ((r.depth >> 24) & 255);
}

float3 safe_invert(float3 v)
{
    float3 inv_v = 1.0f.xxx / v;
    bool _179 = v.x <= 1.0000000116860974230803549289703e-07f;
    bool _186;
    if (_179)
    {
        _186 = v.x >= 0.0f;
    }
    else
    {
        _186 = _179;
    }
    if (_186)
    {
        float3 _2529 = inv_v;
        _2529.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _2529;
    }
    else
    {
        bool _195 = v.x >= (-1.0000000116860974230803549289703e-07f);
        bool _201;
        if (_195)
        {
            _201 = v.x < 0.0f;
        }
        else
        {
            _201 = _195;
        }
        if (_201)
        {
            float3 _2531 = inv_v;
            _2531.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _2531;
        }
    }
    bool _208 = v.y <= 1.0000000116860974230803549289703e-07f;
    bool _214;
    if (_208)
    {
        _214 = v.y >= 0.0f;
    }
    else
    {
        _214 = _208;
    }
    if (_214)
    {
        float3 _2533 = inv_v;
        _2533.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _2533;
    }
    else
    {
        bool _221 = v.y >= (-1.0000000116860974230803549289703e-07f);
        bool _227;
        if (_221)
        {
            _227 = v.y < 0.0f;
        }
        else
        {
            _227 = _221;
        }
        if (_227)
        {
            float3 _2535 = inv_v;
            _2535.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _2535;
        }
    }
    bool _233 = v.z <= 1.0000000116860974230803549289703e-07f;
    bool _239;
    if (_233)
    {
        _239 = v.z >= 0.0f;
    }
    else
    {
        _239 = _233;
    }
    if (_239)
    {
        float3 _2537 = inv_v;
        _2537.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _2537;
    }
    else
    {
        bool _246 = v.z >= (-1.0000000116860974230803549289703e-07f);
        bool _252;
        if (_246)
        {
            _252 = v.z < 0.0f;
        }
        else
        {
            _252 = _246;
        }
        if (_252)
        {
            float3 _2539 = inv_v;
            _2539.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _2539;
        }
    }
    return inv_v;
}

bool _bbox_test_fma(float3 inv_d, float3 neg_inv_d_o, float t, float3 bbox_min, float3 bbox_max)
{
    float _677 = mad(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float _685 = mad(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float _700 = mad(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    float _707 = mad(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    float _724 = mad(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    float _731 = mad(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    float _736 = max(max(min(_677, _685), min(_700, _707)), min(_724, _731));
    float _744 = min(min(max(_677, _685), max(_700, _707)), max(_724, _731)) * 1.0000002384185791015625f;
    return ((_736 <= _744) && (_736 <= t)) && (_744 > 0.0f);
}

void IntersectTri(float3 ro, float3 rd, tri_accel_t tri, uint prim_index, inout hit_data_t inter)
{
    do
    {
        float _449 = dot(rd, tri.n_plane.xyz);
        float _458 = tri.n_plane.w - dot(ro, tri.n_plane.xyz);
        if (sign(_458) != sign(mad(_449, inter.t, -_458)))
        {
            break;
        }
        float3 _479 = (ro * _449) + (rd * _458);
        float _490 = mad(_449, tri.u_plane.w, dot(_479, tri.u_plane.xyz));
        float _495 = _449 - _490;
        if (sign(_490) != sign(_495))
        {
            break;
        }
        float _512 = mad(_449, tri.v_plane.w, dot(_479, tri.v_plane.xyz));
        if (sign(_512) != sign(_495 - _512))
        {
            break;
        }
        float _527 = 1.0f / _449;
        inter.mask = -1;
        int _532;
        if (_449 < 0.0f)
        {
            _532 = int(prim_index);
        }
        else
        {
            _532 = (-1) - int(prim_index);
        }
        inter.prim_index = _532;
        inter.t = _458 * _527;
        inter.u = _490 * _527;
        inter.v = _512 * _527;
        break;
    } while(false);
}

bool IntersectTris_AnyHit(float3 ro, float3 rd, int tri_start, int tri_end, int obj_index, inout hit_data_t out_inter)
{
    int _2339 = 0;
    int _2340 = obj_index;
    float _2342 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _2341;
    float _2343;
    float _2344;
    for (int i = tri_start; i < tri_end; )
    {
        param = ro;
        param_1 = rd;
        tri_accel_t _590;
        _590.n_plane = asfloat(_579.Load4(i * 48 + 0));
        _590.u_plane = asfloat(_579.Load4(i * 48 + 16));
        _590.v_plane = asfloat(_579.Load4(i * 48 + 32));
        param_2.n_plane = _590.n_plane;
        param_2.u_plane = _590.u_plane;
        param_2.v_plane = _590.v_plane;
        param_3 = uint(i);
        hit_data_t _2351 = { _2339, _2340, _2341, _2342, _2343, _2344 };
        param_4 = _2351;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _2339 = param_4.mask;
        _2340 = param_4.obj_index;
        _2341 = param_4.prim_index;
        _2342 = param_4.t;
        _2343 = param_4.u;
        _2344 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _2339;
    int _613;
    if (_2339 != 0)
    {
        _613 = _2340;
    }
    else
    {
        _613 = out_inter.obj_index;
    }
    out_inter.obj_index = _613;
    int _626;
    if (_2339 != 0)
    {
        _626 = _2341;
    }
    else
    {
        _626 = out_inter.prim_index;
    }
    out_inter.prim_index = _626;
    out_inter.t = _2342;
    float _642;
    if (_2339 != 0)
    {
        _642 = _2343;
    }
    else
    {
        _642 = out_inter.u;
    }
    out_inter.u = _642;
    float _655;
    if (_2339 != 0)
    {
        _655 = _2344;
    }
    else
    {
        _655 = out_inter.v;
    }
    out_inter.v = _655;
    return _2339 != 0;
}

bool Traverse_BLAS_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    bool _2280 = false;
    bool _2277;
    do
    {
        float3 _761 = (-inv_d) * ro;
        uint _763 = stack_size;
        uint _773 = stack_size;
        stack_size = _773 + uint(1);
        g_stack[gl_LocalInvocationIndex][_773] = node_index;
        uint _847;
        uint _871;
        int _923;
        while (stack_size != _763)
        {
            uint _788 = stack_size;
            uint _789 = _788 - uint(1);
            stack_size = _789;
            bvh_node_t _803;
            _803.bbox_min = asfloat(_799.Load4(g_stack[gl_LocalInvocationIndex][_789] * 32 + 0));
            _803.bbox_max = asfloat(_799.Load4(g_stack[gl_LocalInvocationIndex][_789] * 32 + 16));
            float3 param = inv_d;
            float3 param_1 = _761;
            float param_2 = inter.t;
            float3 param_3 = _803.bbox_min.xyz;
            float3 param_4 = _803.bbox_max.xyz;
            if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
            {
                continue;
            }
            uint _830 = asuint(_803.bbox_min.w);
            if ((_830 & 2147483648u) == 0u)
            {
                uint _837 = stack_size;
                stack_size = _837 + uint(1);
                uint _841 = asuint(_803.bbox_max.w);
                uint _843 = _841 >> uint(30);
                if (rd[_843] < 0.0f)
                {
                    _847 = _830;
                }
                else
                {
                    _847 = _841 & 1073741823u;
                }
                g_stack[gl_LocalInvocationIndex][_837] = _847;
                uint _862 = stack_size;
                stack_size = _862 + uint(1);
                if (rd[_843] < 0.0f)
                {
                    _871 = _841 & 1073741823u;
                }
                else
                {
                    _871 = _830;
                }
                g_stack[gl_LocalInvocationIndex][_862] = _871;
            }
            else
            {
                int _891 = int(_830 & 2147483647u);
                float3 param_5 = ro;
                float3 param_6 = rd;
                int param_7 = _891;
                int param_8 = _891 + asint(_803.bbox_max.w);
                int param_9 = obj_index;
                hit_data_t param_10 = inter;
                bool _912 = IntersectTris_AnyHit(param_5, param_6, param_7, param_8, param_9, param_10);
                inter = param_10;
                if (_912)
                {
                    bool _920 = inter.prim_index < 0;
                    if (_920)
                    {
                        _923 = (-1) - inter.prim_index;
                    }
                    else
                    {
                        _923 = inter.prim_index;
                    }
                    uint _934 = uint(_923);
                    bool _962 = !_920;
                    bool _969;
                    if (_962)
                    {
                        _969 = (((_948.Load(_939.Load(_934 * 4 + 0) * 4 + 0) >> 16u) & 65535u) & 32768u) != 0u;
                    }
                    else
                    {
                        _969 = _962;
                    }
                    bool _980;
                    if (!_969)
                    {
                        bool _979;
                        if (_920)
                        {
                            _979 = ((_948.Load(_939.Load(_934 * 4 + 0) * 4 + 0) & 65535u) & 32768u) != 0u;
                        }
                        else
                        {
                            _979 = _920;
                        }
                        _980 = _979;
                    }
                    else
                    {
                        _980 = _969;
                    }
                    if (_980)
                    {
                        _2280 = true;
                        _2277 = true;
                        break;
                    }
                }
            }
        }
        if (_2280)
        {
            break;
        }
        _2280 = true;
        _2277 = false;
        break;
    } while(false);
    return _2277;
}

bool Traverse_TLAS_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    bool _2271 = false;
    bool _2268;
    do
    {
        float3 _991 = (-orig_inv_rd) * orig_ro;
        uint stack_size = 1u;
        g_stack[gl_LocalInvocationIndex][0u] = node_index;
        uint _1056;
        uint _1079;
        while (stack_size != 0u)
        {
            uint _1007 = stack_size;
            uint _1008 = _1007 - uint(1);
            stack_size = _1008;
            bvh_node_t _1014;
            _1014.bbox_min = asfloat(_799.Load4(g_stack[gl_LocalInvocationIndex][_1008] * 32 + 0));
            _1014.bbox_max = asfloat(_799.Load4(g_stack[gl_LocalInvocationIndex][_1008] * 32 + 16));
            float3 param = orig_inv_rd;
            float3 param_1 = _991;
            float param_2 = inter.t;
            float3 param_3 = _1014.bbox_min.xyz;
            float3 param_4 = _1014.bbox_max.xyz;
            if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
            {
                continue;
            }
            uint _1041 = asuint(_1014.bbox_min.w);
            if ((_1041 & 2147483648u) == 0u)
            {
                uint _1047 = stack_size;
                stack_size = _1047 + uint(1);
                uint _1051 = asuint(_1014.bbox_max.w);
                uint _1052 = _1051 >> uint(30);
                if (orig_rd[_1052] < 0.0f)
                {
                    _1056 = _1041;
                }
                else
                {
                    _1056 = _1051 & 1073741823u;
                }
                g_stack[gl_LocalInvocationIndex][_1047] = _1056;
                uint _1070 = stack_size;
                stack_size = _1070 + uint(1);
                if (orig_rd[_1052] < 0.0f)
                {
                    _1079 = _1051 & 1073741823u;
                }
                else
                {
                    _1079 = _1041;
                }
                g_stack[gl_LocalInvocationIndex][_1070] = _1079;
            }
            else
            {
                uint _1097 = _1041 & 2147483647u;
                uint _1101 = asuint(_1014.bbox_max.w);
                for (uint i = _1097; i < (_1097 + _1101); i++)
                {
                    mesh_instance_t _1131;
                    _1131.bbox_min = asfloat(_1121.Load4(_1125.Load(i * 4 + 0) * 32 + 0));
                    _1131.bbox_max = asfloat(_1121.Load4(_1125.Load(i * 4 + 0) * 32 + 16));
                    mesh_t _1151;
                    [unroll]
                    for (int _22ident = 0; _22ident < 3; _22ident++)
                    {
                        _1151.bbox_min[_22ident] = asfloat(_1145.Load(_22ident * 4 + asuint(_1131.bbox_max.w) * 48 + 0));
                    }
                    [unroll]
                    for (int _23ident = 0; _23ident < 3; _23ident++)
                    {
                        _1151.bbox_max[_23ident] = asfloat(_1145.Load(_23ident * 4 + asuint(_1131.bbox_max.w) * 48 + 12));
                    }
                    _1151.node_index = _1145.Load(asuint(_1131.bbox_max.w) * 48 + 24);
                    _1151.node_count = _1145.Load(asuint(_1131.bbox_max.w) * 48 + 28);
                    _1151.tris_index = _1145.Load(asuint(_1131.bbox_max.w) * 48 + 32);
                    _1151.tris_count = _1145.Load(asuint(_1131.bbox_max.w) * 48 + 36);
                    _1151.vert_index = _1145.Load(asuint(_1131.bbox_max.w) * 48 + 40);
                    _1151.vert_count = _1145.Load(asuint(_1131.bbox_max.w) * 48 + 44);
                    transform_t _1197;
                    _1197.xform = asfloat(uint4x4(_1191.Load4(asuint(_1131.bbox_min.w) * 128 + 0), _1191.Load4(asuint(_1131.bbox_min.w) * 128 + 16), _1191.Load4(asuint(_1131.bbox_min.w) * 128 + 32), _1191.Load4(asuint(_1131.bbox_min.w) * 128 + 48)));
                    _1197.inv_xform = asfloat(uint4x4(_1191.Load4(asuint(_1131.bbox_min.w) * 128 + 64), _1191.Load4(asuint(_1131.bbox_min.w) * 128 + 80), _1191.Load4(asuint(_1131.bbox_min.w) * 128 + 96), _1191.Load4(asuint(_1131.bbox_min.w) * 128 + 112)));
                    float3 param_5 = orig_inv_rd;
                    float3 param_6 = _991;
                    float param_7 = inter.t;
                    float3 param_8 = _1131.bbox_min.xyz;
                    float3 param_9 = _1131.bbox_max.xyz;
                    if (!_bbox_test_fma(param_5, param_6, param_7, param_8, param_9))
                    {
                        continue;
                    }
                    float3 _1242 = mul(float4(orig_rd, 0.0f), _1197.inv_xform).xyz;
                    float3 param_10 = _1242;
                    float3 param_11 = mul(float4(orig_ro, 1.0f), _1197.inv_xform).xyz;
                    float3 param_12 = _1242;
                    float3 param_13 = safe_invert(param_10);
                    int param_14 = int(_1125.Load(i * 4 + 0));
                    uint param_15 = _1151.node_index;
                    uint param_16 = stack_size;
                    hit_data_t param_17 = inter;
                    bool _1266 = Traverse_BLAS_WithStack(param_11, param_12, param_13, param_14, param_15, param_16, param_17);
                    inter = param_17;
                    if (_1266)
                    {
                        _2271 = true;
                        _2268 = true;
                        break;
                    }
                }
                if (_2271)
                {
                    break;
                }
            }
        }
        if (_2271)
        {
            break;
        }
        _2271 = true;
        _2268 = false;
        break;
    } while(false);
    return _2268;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _307 = mad(col.z, 31.875f, 1.0f);
    float _317 = (col.x - 0.501960813999176025390625f) / _307;
    float _323 = (col.y - 0.501960813999176025390625f) / _307;
    return float3((col.w + _317) - _323, col.w + _323, (col.w - _317) - _323);
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
    uint _381 = index & 16777215u;
    uint _386_dummy_parameter;
    float4 res = g_textures[NonUniformResourceIndex(_381)].SampleLevel(_g_textures_sampler[NonUniformResourceIndex(_381)], uvs + ((rand - 0.5f.xx) / float2(int2(spvTextureSize(g_textures[NonUniformResourceIndex(_381)], uint(lod), _386_dummy_parameter)))), float(lod));
    bool _406;
    if (maybe_YCoCg)
    {
        _406 = (index & 67108864u) != 0u;
    }
    else
    {
        _406 = maybe_YCoCg;
    }
    if (_406)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _424;
    if (maybe_SRGB)
    {
        _424 = (index & 16777216u) != 0u;
    }
    else
    {
        _424 = maybe_SRGB;
    }
    if (_424)
    {
        float3 param_1 = res.xyz;
        float3 _430 = srgb_to_rgb(param_1);
        float4 _2555 = res;
        _2555.x = _430.x;
        float4 _2557 = _2555;
        _2557.y = _430.y;
        float4 _2559 = _2557;
        _2559.z = _430.z;
        res = _2559;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod, float2 rand)
{
    return SampleBilinear(index, uvs, lod, rand, false, false);
}

float lum(float3 color)
{
    return mad(0.072168998420238494873046875f, color.z, mad(0.21267099678516387939453125f, color.x, 0.71516001224517822265625f * color.y));
}

float3 IntersectSceneShadow(shadow_ray_t r)
{
    bool _2257 = false;
    float3 _2254;
    do
    {
        float3 ro = float3(r.o[0], r.o[1], r.o[2]);
        float3 _1291 = float3(r.d[0], r.d[1], r.d[2]);
        float3 rc = float3(r.c[0], r.c[1], r.c[2]);
        int depth = r.depth >> 24;
        uint param = uint(hash(r.xy));
        float _1311 = construct_float(param);
        uint param_1 = uint(hash(hash(r.xy)));
        float _1318 = construct_float(param_1);
        int rand_index = _1324_g_params.hi + (total_depth(r) * 9);
        float _1337;
        if (r.dist > 0.0f)
        {
            _1337 = r.dist;
        }
        else
        {
            _1337 = 3402823346297367662189621542912.0f;
        }
        float dist = _1337;
        float3 param_2 = _1291;
        float3 _1348 = safe_invert(param_2);
        int _1402;
        int _2401;
        int _2402;
        float _2404;
        float _2405;
        while (dist > 9.9999997473787516355514526367188e-06f)
        {
            int _2400 = 0;
            float _2403 = dist;
            float3 param_3 = ro;
            float3 param_4 = _1291;
            float3 param_5 = _1348;
            uint param_6 = _1324_g_params.node_index;
            hit_data_t _2412 = { 0, _2401, _2402, dist, _2404, _2405 };
            hit_data_t param_7 = _2412;
            bool _1374 = Traverse_TLAS_WithStack(param_3, param_4, param_5, param_6, param_7);
            _2400 = param_7.mask;
            _2401 = param_7.obj_index;
            _2402 = param_7.prim_index;
            _2403 = param_7.t;
            _2404 = param_7.u;
            _2405 = param_7.v;
            bool _1384;
            if (!_1374)
            {
                _1384 = depth > _1324_g_params.max_transp_depth;
            }
            else
            {
                _1384 = _1374;
            }
            if (_1384)
            {
                _2257 = true;
                _2254 = 0.0f.xxx;
                break;
            }
            if (_2400 == 0)
            {
                _2257 = true;
                _2254 = rc;
                break;
            }
            bool _1399 = param_7.prim_index < 0;
            if (_1399)
            {
                _1402 = (-1) - param_7.prim_index;
            }
            else
            {
                _1402 = param_7.prim_index;
            }
            uint _1413 = uint(_1402);
            uint _1456 = _939.Load(_1413 * 4 + 0) * 3u;
            vertex_t _1462;
            [unroll]
            for (int _24ident = 0; _24ident < 3; _24ident++)
            {
                _1462.p[_24ident] = asfloat(_1450.Load(_24ident * 4 + _1454.Load(_1456 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _25ident = 0; _25ident < 3; _25ident++)
            {
                _1462.n[_25ident] = asfloat(_1450.Load(_25ident * 4 + _1454.Load(_1456 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _26ident = 0; _26ident < 3; _26ident++)
            {
                _1462.b[_26ident] = asfloat(_1450.Load(_26ident * 4 + _1454.Load(_1456 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _27ident = 0; _27ident < 2; _27ident++)
            {
                [unroll]
                for (int _28ident = 0; _28ident < 2; _28ident++)
                {
                    _1462.t[_27ident][_28ident] = asfloat(_1450.Load(_28ident * 4 + _27ident * 8 + _1454.Load(_1456 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1510;
            [unroll]
            for (int _29ident = 0; _29ident < 3; _29ident++)
            {
                _1510.p[_29ident] = asfloat(_1450.Load(_29ident * 4 + _1454.Load((_1456 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _30ident = 0; _30ident < 3; _30ident++)
            {
                _1510.n[_30ident] = asfloat(_1450.Load(_30ident * 4 + _1454.Load((_1456 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _31ident = 0; _31ident < 3; _31ident++)
            {
                _1510.b[_31ident] = asfloat(_1450.Load(_31ident * 4 + _1454.Load((_1456 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _32ident = 0; _32ident < 2; _32ident++)
            {
                [unroll]
                for (int _33ident = 0; _33ident < 2; _33ident++)
                {
                    _1510.t[_32ident][_33ident] = asfloat(_1450.Load(_33ident * 4 + _32ident * 8 + _1454.Load((_1456 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1556;
            [unroll]
            for (int _34ident = 0; _34ident < 3; _34ident++)
            {
                _1556.p[_34ident] = asfloat(_1450.Load(_34ident * 4 + _1454.Load((_1456 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _35ident = 0; _35ident < 3; _35ident++)
            {
                _1556.n[_35ident] = asfloat(_1450.Load(_35ident * 4 + _1454.Load((_1456 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _1556.b[_36ident] = asfloat(_1450.Load(_36ident * 4 + _1454.Load((_1456 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _37ident = 0; _37ident < 2; _37ident++)
            {
                [unroll]
                for (int _38ident = 0; _38ident < 2; _38ident++)
                {
                    _1556.t[_37ident][_38ident] = asfloat(_1450.Load(_38ident * 4 + _37ident * 8 + _1454.Load((_1456 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            float2 _1627 = ((float2(_1462.t[0][0], _1462.t[0][1]) * ((1.0f - param_7.u) - param_7.v)) + (float2(_1510.t[0][0], _1510.t[0][1]) * param_7.u)) + (float2(_1556.t[0][0], _1556.t[0][1]) * param_7.v);
            g_stack[gl_LocalInvocationIndex][0] = (_1399 ? (_948.Load(_939.Load(_1413 * 4 + 0) * 4 + 0) & 65535u) : ((_948.Load(_939.Load(_1413 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u;
            g_stack[gl_LocalInvocationIndex][1] = 1065353216u;
            int stack_size = 1;
            float3 throughput = 0.0f.xxx;
            float2 _1658 = float2(frac(asfloat(_1640.Load((rand_index + 7) * 4 + 0)) + _1311), frac(asfloat(_1640.Load((rand_index + 8) * 4 + 0)) + _1318));
            for (;;)
            {
                int _1664 = stack_size;
                stack_size = _1664 - 1;
                if (_1664 != 0)
                {
                    int _1680 = stack_size;
                    int _1681 = 2 * _1680;
                    material_t _1687;
                    [unroll]
                    for (int _39ident = 0; _39ident < 5; _39ident++)
                    {
                        _1687.textures[_39ident] = _1678.Load(_39ident * 4 + g_stack[gl_LocalInvocationIndex][_1681] * 76 + 0);
                    }
                    [unroll]
                    for (int _40ident = 0; _40ident < 3; _40ident++)
                    {
                        _1687.base_color[_40ident] = asfloat(_1678.Load(_40ident * 4 + g_stack[gl_LocalInvocationIndex][_1681] * 76 + 20));
                    }
                    _1687.flags = _1678.Load(g_stack[gl_LocalInvocationIndex][_1681] * 76 + 32);
                    _1687.type = _1678.Load(g_stack[gl_LocalInvocationIndex][_1681] * 76 + 36);
                    _1687.tangent_rotation_or_strength = asfloat(_1678.Load(g_stack[gl_LocalInvocationIndex][_1681] * 76 + 40));
                    _1687.roughness_and_anisotropic = _1678.Load(g_stack[gl_LocalInvocationIndex][_1681] * 76 + 44);
                    _1687.ior = asfloat(_1678.Load(g_stack[gl_LocalInvocationIndex][_1681] * 76 + 48));
                    _1687.sheen_and_sheen_tint = _1678.Load(g_stack[gl_LocalInvocationIndex][_1681] * 76 + 52);
                    _1687.tint_and_metallic = _1678.Load(g_stack[gl_LocalInvocationIndex][_1681] * 76 + 56);
                    _1687.transmission_and_transmission_roughness = _1678.Load(g_stack[gl_LocalInvocationIndex][_1681] * 76 + 60);
                    _1687.specular_and_specular_tint = _1678.Load(g_stack[gl_LocalInvocationIndex][_1681] * 76 + 64);
                    _1687.clearcoat_and_clearcoat_roughness = _1678.Load(g_stack[gl_LocalInvocationIndex][_1681] * 76 + 68);
                    _1687.normal_map_strength_unorm = _1678.Load(g_stack[gl_LocalInvocationIndex][_1681] * 76 + 72);
                    uint _1740 = g_stack[gl_LocalInvocationIndex][_1681 + 1];
                    float _1741 = asfloat(_1740);
                    if (_1687.type == 4u)
                    {
                        float mix_val = _1687.tangent_rotation_or_strength;
                        if (_1687.textures[1] != 4294967295u)
                        {
                            mix_val *= SampleBilinear(_1687.textures[1], _1627, 0, _1658).x;
                        }
                        int _1767 = 2 * stack_size;
                        g_stack[gl_LocalInvocationIndex][_1767] = _1687.textures[3];
                        g_stack[gl_LocalInvocationIndex][_1767 + 1] = asuint(_1741 * (1.0f - mix_val));
                        int _1786 = 2 * (stack_size + 1);
                        g_stack[gl_LocalInvocationIndex][_1786] = _1687.textures[4];
                        g_stack[gl_LocalInvocationIndex][_1786 + 1] = asuint(_1741 * mix_val);
                        stack_size += 2;
                    }
                    else
                    {
                        if (_1687.type == 5u)
                        {
                            throughput += (float3(_1687.base_color[0], _1687.base_color[1], _1687.base_color[2]) * _1741);
                        }
                    }
                    continue;
                }
                else
                {
                    break;
                }
            }
            float3 _1820 = rc;
            float3 _1821 = _1820 * throughput;
            rc = _1821;
            if (lum(_1821) < 1.0000000116860974230803549289703e-07f)
            {
                break;
            }
            float _1831 = _2403 + 9.9999997473787516355514526367188e-06f;
            ro += (_1291 * _1831);
            dist -= _1831;
            depth++;
            rand_index += 9;
        }
        if (_2257)
        {
            break;
        }
        _2257 = true;
        _2254 = rc;
        break;
    } while(false);
    return _2254;
}

float IntersectAreaLightsShadow(shadow_ray_t r)
{
    bool _2264 = false;
    float _2261;
    do
    {
        float3 _1854 = float3(r.o[0], r.o[1], r.o[2]);
        float3 _1862 = float3(r.d[0], r.d[1], r.d[2]);
        float _1866 = abs(r.dist);
        for (uint li = 0u; li < uint(_1324_g_params.blocker_lights_count); li++)
        {
            light_t _1898;
            _1898.type_and_param0 = _1894.Load4(_1882.Load(li * 4 + 0) * 64 + 0);
            _1898.param1 = asfloat(_1894.Load4(_1882.Load(li * 4 + 0) * 64 + 16));
            _1898.param2 = asfloat(_1894.Load4(_1882.Load(li * 4 + 0) * 64 + 32));
            _1898.param3 = asfloat(_1894.Load4(_1882.Load(li * 4 + 0) * 64 + 48));
            bool _1912 = (_1898.type_and_param0.x & 128u) != 0u;
            bool _1918;
            if (_1912)
            {
                _1918 = r.dist >= 0.0f;
            }
            else
            {
                _1918 = _1912;
            }
            [branch]
            if (_1918)
            {
                continue;
            }
            uint _1926 = _1898.type_and_param0.x & 31u;
            if (_1926 == 4u)
            {
                float3 light_u = _1898.param2.xyz;
                float3 light_v = _1898.param3.xyz;
                float3 _1947 = normalize(cross(_1898.param2.xyz, _1898.param3.xyz));
                float _1955 = dot(_1862, _1947);
                float _1963 = (dot(_1947, _1898.param1.xyz) - dot(_1947, _1854)) / _1955;
                if (((_1955 < 0.0f) && (_1963 > 9.9999999747524270787835121154785e-07f)) && (_1963 < _1866))
                {
                    float3 _1976 = light_u;
                    float3 _1981 = _1976 / dot(_1976, _1976).xxx;
                    light_u = _1981;
                    light_v /= dot(light_v, light_v).xxx;
                    float3 _1997 = (_1854 + (_1862 * _1963)) - _1898.param1.xyz;
                    float _2001 = dot(_1981, _1997);
                    if ((_2001 >= (-0.5f)) && (_2001 <= 0.5f))
                    {
                        float _2013 = dot(light_v, _1997);
                        if ((_2013 >= (-0.5f)) && (_2013 <= 0.5f))
                        {
                            _2264 = true;
                            _2261 = 0.0f;
                            break;
                        }
                    }
                }
            }
            else
            {
                if (_1926 == 5u)
                {
                    float3 light_u_1 = _1898.param2.xyz;
                    float3 light_v_1 = _1898.param3.xyz;
                    float3 _2043 = normalize(cross(_1898.param2.xyz, _1898.param3.xyz));
                    float _2051 = dot(_1862, _2043);
                    float _2059 = (dot(_2043, _1898.param1.xyz) - dot(_2043, _1854)) / _2051;
                    if (((_2051 < 0.0f) && (_2059 > 9.9999999747524270787835121154785e-07f)) && (_2059 < _1866))
                    {
                        float3 _2071 = light_u_1;
                        float3 _2076 = _2071 / dot(_2071, _2071).xxx;
                        light_u_1 = _2076;
                        float3 _2077 = light_v_1;
                        float3 _2082 = _2077 / dot(_2077, _2077).xxx;
                        light_v_1 = _2082;
                        float3 _2092 = (_1854 + (_1862 * _2059)) - _1898.param1.xyz;
                        float _2096 = dot(_2076, _2092);
                        float _2100 = dot(_2082, _2092);
                        if (sqrt(mad(_2096, _2096, _2100 * _2100)) <= 0.5f)
                        {
                            _2264 = true;
                            _2261 = 0.0f;
                            break;
                        }
                    }
                }
            }
        }
        if (_2264)
        {
            break;
        }
        _2264 = true;
        _2261 = 1.0f;
        break;
    } while(false);
    return _2261;
}

void comp_main()
{
    do
    {
        int _2126 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_2126) >= _2132.Load(12))
        {
            break;
        }
        shadow_ray_t _2151;
        [unroll]
        for (int _41ident = 0; _41ident < 3; _41ident++)
        {
            _2151.o[_41ident] = asfloat(_2147.Load(_41ident * 4 + _2126 * 48 + 0));
        }
        _2151.depth = int(_2147.Load(_2126 * 48 + 12));
        [unroll]
        for (int _42ident = 0; _42ident < 3; _42ident++)
        {
            _2151.d[_42ident] = asfloat(_2147.Load(_42ident * 4 + _2126 * 48 + 16));
        }
        _2151.dist = asfloat(_2147.Load(_2126 * 48 + 28));
        [unroll]
        for (int _43ident = 0; _43ident < 3; _43ident++)
        {
            _2151.c[_43ident] = asfloat(_2147.Load(_43ident * 4 + _2126 * 48 + 32));
        }
        _2151.xy = int(_2147.Load(_2126 * 48 + 44));
        float _2334[3] = { _2151.c[0], _2151.c[1], _2151.c[2] };
        float _2323[3] = { _2151.d[0], _2151.d[1], _2151.d[2] };
        float _2312[3] = { _2151.o[0], _2151.o[1], _2151.o[2] };
        shadow_ray_t _2298 = { _2312, _2151.depth, _2323, _2151.dist, _2334, _2151.xy };
        shadow_ray_t param = _2298;
        float3 _2185 = IntersectSceneShadow(param);
        shadow_ray_t param_1 = _2298;
        float3 _2189 = _2185 * IntersectAreaLightsShadow(param_1);
        if (lum(_2189) > 0.0f)
        {
            int2 _2212 = int2((_2151.xy >> 16) & 65535, _2151.xy & 65535);
            float4 _2213 = g_inout_img[_2212];
            float3 _2222 = _2213.xyz + min(_2189, _1324_g_params.clamp_val.xxx);
            float4 _2523 = _2213;
            _2523.x = _2222.x;
            float4 _2525 = _2523;
            _2525.y = _2222.y;
            float4 _2527 = _2525;
            _2527.z = _2222.z;
            g_inout_img[_2212] = _2527;
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
