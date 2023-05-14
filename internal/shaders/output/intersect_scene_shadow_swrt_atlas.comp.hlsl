struct shadow_ray_t
{
    float o[3];
    int depth;
    float d[3];
    float dist;
    float c[3];
    int xy;
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

ByteAddressBuffer _443 : register(t20, space0);
ByteAddressBuffer _718 : register(t1, space0);
ByteAddressBuffer _938 : register(t5, space0);
ByteAddressBuffer _1078 : register(t2, space0);
ByteAddressBuffer _1087 : register(t3, space0);
ByteAddressBuffer _1258 : register(t7, space0);
ByteAddressBuffer _1262 : register(t8, space0);
ByteAddressBuffer _1282 : register(t6, space0);
ByteAddressBuffer _1326 : register(t9, space0);
ByteAddressBuffer _1582 : register(t10, space0);
ByteAddressBuffer _1586 : register(t11, space0);
ByteAddressBuffer _1772 : register(t16, space0);
ByteAddressBuffer _1810 : register(t4, space0);
ByteAddressBuffer _2010 : register(t15, space0);
ByteAddressBuffer _2022 : register(t14, space0);
ByteAddressBuffer _2260 : register(t13, space0);
ByteAddressBuffer _2275 : register(t12, space0);
cbuffer UniformParams
{
    Params _1458_g_params : packoffset(c0);
};

Texture2DArray<float4> g_atlases[7] : register(t21, space0);
SamplerState _g_atlases_sampler[7] : register(s21, space0);
RWTexture2D<float4> g_inout_img : register(u0, space0);

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
    uint _132 = uint(x);
    uint _139 = ((_132 >> uint(16)) ^ _132) * 73244475u;
    uint _144 = ((_139 >> uint(16)) ^ _139) * 73244475u;
    return int((_144 >> uint(16)) ^ _144);
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
    bool _190 = v.x <= 1.0000000116860974230803549289703e-07f;
    bool _197;
    if (_190)
    {
        _197 = v.x >= 0.0f;
    }
    else
    {
        _197 = _190;
    }
    if (_197)
    {
        float3 _2706 = inv_v;
        _2706.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _2706;
    }
    else
    {
        bool _206 = v.x >= (-1.0000000116860974230803549289703e-07f);
        bool _212;
        if (_206)
        {
            _212 = v.x < 0.0f;
        }
        else
        {
            _212 = _206;
        }
        if (_212)
        {
            float3 _2708 = inv_v;
            _2708.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _2708;
        }
    }
    bool _219 = v.y <= 1.0000000116860974230803549289703e-07f;
    bool _225;
    if (_219)
    {
        _225 = v.y >= 0.0f;
    }
    else
    {
        _225 = _219;
    }
    if (_225)
    {
        float3 _2710 = inv_v;
        _2710.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _2710;
    }
    else
    {
        bool _232 = v.y >= (-1.0000000116860974230803549289703e-07f);
        bool _238;
        if (_232)
        {
            _238 = v.y < 0.0f;
        }
        else
        {
            _238 = _232;
        }
        if (_238)
        {
            float3 _2712 = inv_v;
            _2712.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _2712;
        }
    }
    bool _244 = v.z <= 1.0000000116860974230803549289703e-07f;
    bool _250;
    if (_244)
    {
        _250 = v.z >= 0.0f;
    }
    else
    {
        _250 = _244;
    }
    if (_250)
    {
        float3 _2714 = inv_v;
        _2714.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _2714;
    }
    else
    {
        bool _257 = v.z >= (-1.0000000116860974230803549289703e-07f);
        bool _263;
        if (_257)
        {
            _263 = v.z < 0.0f;
        }
        else
        {
            _263 = _257;
        }
        if (_263)
        {
            float3 _2716 = inv_v;
            _2716.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _2716;
        }
    }
    return inv_v;
}

bool _bbox_test_fma(float3 inv_d, float3 neg_inv_d_o, float t, float3 bbox_min, float3 bbox_max)
{
    float _816 = mad(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float _824 = mad(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float _839 = mad(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    float _846 = mad(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    float _863 = mad(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    float _870 = mad(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    float _875 = max(max(min(_816, _824), min(_839, _846)), min(_863, _870));
    float _883 = min(min(max(_816, _824), max(_839, _846)), max(_863, _870)) * 1.0000002384185791015625f;
    return ((_875 <= _883) && (_875 <= t)) && (_883 > 0.0f);
}

void IntersectTri(float3 ro, float3 rd, tri_accel_t tri, uint prim_index, inout hit_data_t inter)
{
    do
    {
        float _591 = dot(rd, tri.n_plane.xyz);
        float _600 = tri.n_plane.w - dot(ro, tri.n_plane.xyz);
        if (sign(_600) != sign(mad(_591, inter.t, -_600)))
        {
            break;
        }
        float3 _621 = (ro * _591) + (rd * _600);
        float _632 = mad(_591, tri.u_plane.w, dot(_621, tri.u_plane.xyz));
        float _637 = _591 - _632;
        if (sign(_632) != sign(_637))
        {
            break;
        }
        float _653 = mad(_591, tri.v_plane.w, dot(_621, tri.v_plane.xyz));
        if (sign(_653) != sign(_637 - _653))
        {
            break;
        }
        float _668 = 1.0f / _591;
        inter.mask = -1;
        int _673;
        if (_591 < 0.0f)
        {
            _673 = int(prim_index);
        }
        else
        {
            _673 = (-1) - int(prim_index);
        }
        inter.prim_index = _673;
        inter.t = _600 * _668;
        inter.u = _632 * _668;
        inter.v = _653 * _668;
        break;
    } while(false);
}

bool IntersectTris_AnyHit(float3 ro, float3 rd, int tri_start, int tri_end, int obj_index, inout hit_data_t out_inter)
{
    int _2516 = 0;
    int _2517 = obj_index;
    float _2519 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _2518;
    float _2520;
    float _2521;
    for (int i = tri_start; i < tri_end; )
    {
        param = ro;
        param_1 = rd;
        tri_accel_t _729;
        _729.n_plane = asfloat(_718.Load4(i * 48 + 0));
        _729.u_plane = asfloat(_718.Load4(i * 48 + 16));
        _729.v_plane = asfloat(_718.Load4(i * 48 + 32));
        param_2.n_plane = _729.n_plane;
        param_2.u_plane = _729.u_plane;
        param_2.v_plane = _729.v_plane;
        param_3 = uint(i);
        hit_data_t _2528 = { _2516, _2517, _2518, _2519, _2520, _2521 };
        param_4 = _2528;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _2516 = param_4.mask;
        _2517 = param_4.obj_index;
        _2518 = param_4.prim_index;
        _2519 = param_4.t;
        _2520 = param_4.u;
        _2521 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _2516;
    int _752;
    if (_2516 != 0)
    {
        _752 = _2517;
    }
    else
    {
        _752 = out_inter.obj_index;
    }
    out_inter.obj_index = _752;
    int _765;
    if (_2516 != 0)
    {
        _765 = _2518;
    }
    else
    {
        _765 = out_inter.prim_index;
    }
    out_inter.prim_index = _765;
    out_inter.t = _2519;
    float _781;
    if (_2516 != 0)
    {
        _781 = _2520;
    }
    else
    {
        _781 = out_inter.u;
    }
    out_inter.u = _781;
    float _794;
    if (_2516 != 0)
    {
        _794 = _2521;
    }
    else
    {
        _794 = out_inter.v;
    }
    out_inter.v = _794;
    return _2516 != 0;
}

bool Traverse_MicroTree_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    bool _2410 = false;
    bool _2407;
    do
    {
        float3 _900 = (-inv_d) * ro;
        uint _902 = stack_size;
        uint _912 = stack_size;
        stack_size = _912 + uint(1);
        g_stack[gl_LocalInvocationIndex][_912] = node_index;
        uint _986;
        uint _1010;
        int _1062;
        while (stack_size != _902)
        {
            uint _927 = stack_size;
            uint _928 = _927 - uint(1);
            stack_size = _928;
            bvh_node_t _942;
            _942.bbox_min = asfloat(_938.Load4(g_stack[gl_LocalInvocationIndex][_928] * 32 + 0));
            _942.bbox_max = asfloat(_938.Load4(g_stack[gl_LocalInvocationIndex][_928] * 32 + 16));
            float3 param = inv_d;
            float3 param_1 = _900;
            float param_2 = inter.t;
            float3 param_3 = _942.bbox_min.xyz;
            float3 param_4 = _942.bbox_max.xyz;
            if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
            {
                continue;
            }
            uint _969 = asuint(_942.bbox_min.w);
            if ((_969 & 2147483648u) == 0u)
            {
                uint _976 = stack_size;
                stack_size = _976 + uint(1);
                uint _980 = asuint(_942.bbox_max.w);
                uint _982 = _980 >> uint(30);
                if (rd[_982] < 0.0f)
                {
                    _986 = _969;
                }
                else
                {
                    _986 = _980 & 1073741823u;
                }
                g_stack[gl_LocalInvocationIndex][_976] = _986;
                uint _1001 = stack_size;
                stack_size = _1001 + uint(1);
                if (rd[_982] < 0.0f)
                {
                    _1010 = _980 & 1073741823u;
                }
                else
                {
                    _1010 = _969;
                }
                g_stack[gl_LocalInvocationIndex][_1001] = _1010;
            }
            else
            {
                int _1030 = int(_969 & 2147483647u);
                float3 param_5 = ro;
                float3 param_6 = rd;
                int param_7 = _1030;
                int param_8 = _1030 + asint(_942.bbox_max.w);
                int param_9 = obj_index;
                hit_data_t param_10 = inter;
                bool _1051 = IntersectTris_AnyHit(param_5, param_6, param_7, param_8, param_9, param_10);
                inter = param_10;
                if (_1051)
                {
                    bool _1059 = inter.prim_index < 0;
                    if (_1059)
                    {
                        _1062 = (-1) - inter.prim_index;
                    }
                    else
                    {
                        _1062 = inter.prim_index;
                    }
                    uint _1073 = uint(_1062);
                    bool _1100 = !_1059;
                    bool _1106;
                    if (_1100)
                    {
                        _1106 = (((_1087.Load(_1078.Load(_1073 * 4 + 0) * 4 + 0) >> 16u) & 65535u) & 32768u) != 0u;
                    }
                    else
                    {
                        _1106 = _1100;
                    }
                    bool _1117;
                    if (!_1106)
                    {
                        bool _1116;
                        if (_1059)
                        {
                            _1116 = ((_1087.Load(_1078.Load(_1073 * 4 + 0) * 4 + 0) & 65535u) & 32768u) != 0u;
                        }
                        else
                        {
                            _1116 = _1059;
                        }
                        _1117 = _1116;
                    }
                    else
                    {
                        _1117 = _1106;
                    }
                    if (_1117)
                    {
                        _2410 = true;
                        _2407 = true;
                        break;
                    }
                }
            }
        }
        if (_2410)
        {
            break;
        }
        _2410 = true;
        _2407 = false;
        break;
    } while(false);
    return _2407;
}

bool Traverse_MacroTree_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    bool _2401 = false;
    bool _2398;
    do
    {
        float3 _1128 = (-orig_inv_rd) * orig_ro;
        uint stack_size = 1u;
        g_stack[gl_LocalInvocationIndex][0u] = node_index;
        uint _1193;
        uint _1216;
        while (stack_size != 0u)
        {
            uint _1144 = stack_size;
            uint _1145 = _1144 - uint(1);
            stack_size = _1145;
            bvh_node_t _1151;
            _1151.bbox_min = asfloat(_938.Load4(g_stack[gl_LocalInvocationIndex][_1145] * 32 + 0));
            _1151.bbox_max = asfloat(_938.Load4(g_stack[gl_LocalInvocationIndex][_1145] * 32 + 16));
            float3 param = orig_inv_rd;
            float3 param_1 = _1128;
            float param_2 = inter.t;
            float3 param_3 = _1151.bbox_min.xyz;
            float3 param_4 = _1151.bbox_max.xyz;
            if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
            {
                continue;
            }
            uint _1178 = asuint(_1151.bbox_min.w);
            if ((_1178 & 2147483648u) == 0u)
            {
                uint _1184 = stack_size;
                stack_size = _1184 + uint(1);
                uint _1188 = asuint(_1151.bbox_max.w);
                uint _1189 = _1188 >> uint(30);
                if (orig_rd[_1189] < 0.0f)
                {
                    _1193 = _1178;
                }
                else
                {
                    _1193 = _1188 & 1073741823u;
                }
                g_stack[gl_LocalInvocationIndex][_1184] = _1193;
                uint _1207 = stack_size;
                stack_size = _1207 + uint(1);
                if (orig_rd[_1189] < 0.0f)
                {
                    _1216 = _1188 & 1073741823u;
                }
                else
                {
                    _1216 = _1178;
                }
                g_stack[gl_LocalInvocationIndex][_1207] = _1216;
            }
            else
            {
                uint _1234 = _1178 & 2147483647u;
                uint _1238 = asuint(_1151.bbox_max.w);
                for (uint i = _1234; i < (_1234 + _1238); i++)
                {
                    mesh_instance_t _1268;
                    _1268.bbox_min = asfloat(_1258.Load4(_1262.Load(i * 4 + 0) * 32 + 0));
                    _1268.bbox_max = asfloat(_1258.Load4(_1262.Load(i * 4 + 0) * 32 + 16));
                    mesh_t _1288;
                    [unroll]
                    for (int _24ident = 0; _24ident < 3; _24ident++)
                    {
                        _1288.bbox_min[_24ident] = asfloat(_1282.Load(_24ident * 4 + asuint(_1268.bbox_max.w) * 48 + 0));
                    }
                    [unroll]
                    for (int _25ident = 0; _25ident < 3; _25ident++)
                    {
                        _1288.bbox_max[_25ident] = asfloat(_1282.Load(_25ident * 4 + asuint(_1268.bbox_max.w) * 48 + 12));
                    }
                    _1288.node_index = _1282.Load(asuint(_1268.bbox_max.w) * 48 + 24);
                    _1288.node_count = _1282.Load(asuint(_1268.bbox_max.w) * 48 + 28);
                    _1288.tris_index = _1282.Load(asuint(_1268.bbox_max.w) * 48 + 32);
                    _1288.tris_count = _1282.Load(asuint(_1268.bbox_max.w) * 48 + 36);
                    _1288.vert_index = _1282.Load(asuint(_1268.bbox_max.w) * 48 + 40);
                    _1288.vert_count = _1282.Load(asuint(_1268.bbox_max.w) * 48 + 44);
                    transform_t _1332;
                    _1332.xform = asfloat(uint4x4(_1326.Load4(asuint(_1268.bbox_min.w) * 128 + 0), _1326.Load4(asuint(_1268.bbox_min.w) * 128 + 16), _1326.Load4(asuint(_1268.bbox_min.w) * 128 + 32), _1326.Load4(asuint(_1268.bbox_min.w) * 128 + 48)));
                    _1332.inv_xform = asfloat(uint4x4(_1326.Load4(asuint(_1268.bbox_min.w) * 128 + 64), _1326.Load4(asuint(_1268.bbox_min.w) * 128 + 80), _1326.Load4(asuint(_1268.bbox_min.w) * 128 + 96), _1326.Load4(asuint(_1268.bbox_min.w) * 128 + 112)));
                    float3 param_5 = orig_inv_rd;
                    float3 param_6 = _1128;
                    float param_7 = inter.t;
                    float3 param_8 = _1268.bbox_min.xyz;
                    float3 param_9 = _1268.bbox_max.xyz;
                    if (!_bbox_test_fma(param_5, param_6, param_7, param_8, param_9))
                    {
                        continue;
                    }
                    float3 _1377 = mul(float4(orig_rd, 0.0f), _1332.inv_xform).xyz;
                    float3 param_10 = _1377;
                    float3 param_11 = mul(float4(orig_ro, 1.0f), _1332.inv_xform).xyz;
                    float3 param_12 = _1377;
                    float3 param_13 = safe_invert(param_10);
                    int param_14 = int(_1262.Load(i * 4 + 0));
                    uint param_15 = _1288.node_index;
                    uint param_16 = stack_size;
                    hit_data_t param_17 = inter;
                    bool _1401 = Traverse_MicroTree_WithStack(param_11, param_12, param_13, param_14, param_15, param_16, param_17);
                    inter = param_17;
                    if (_1401)
                    {
                        _2401 = true;
                        _2398 = true;
                        break;
                    }
                }
                if (_2401)
                {
                    break;
                }
            }
        }
        if (_2401)
        {
            break;
        }
        _2401 = true;
        _2398 = false;
        break;
    } while(false);
    return _2398;
}

float2 TransformUV(float2 _uv, atlas_texture_t t, int mip_level)
{
    uint _2471[14] = t.pos;
    uint _2474[14] = t.pos;
    uint _401 = t.size & 16383u;
    uint _404 = t.size >> uint(16);
    uint _405 = _404 & 16383u;
    float2 size = float2(float(_401), float(_405));
    if ((_404 & 32768u) != 0u)
    {
        size = float2(float(_401 >> uint(mip_level)), float(_405 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_2471[mip_level] & 65535u), float((_2474[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _318 = mad(col.z, 31.875f, 1.0f);
    float _328 = (col.x - 0.501960813999176025390625f) / _318;
    float _334 = (col.y - 0.501960813999176025390625f) / _318;
    return float3((col.w + _328) - _334, col.w + _334, (col.w - _328) - _334);
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
    atlas_texture_t _446;
    _446.size = _443.Load(index * 80 + 0);
    _446.atlas = _443.Load(index * 80 + 4);
    [unroll]
    for (int _26ident = 0; _26ident < 4; _26ident++)
    {
        _446.page[_26ident] = _443.Load(_26ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _27ident = 0; _27ident < 14; _27ident++)
    {
        _446.pos[_27ident] = _443.Load(_27ident * 4 + index * 80 + 24);
    }
    uint _2479[4];
    _2479[0] = _446.page[0];
    _2479[1] = _446.page[1];
    _2479[2] = _446.page[2];
    _2479[3] = _446.page[3];
    uint _2515[14] = { _446.pos[0], _446.pos[1], _446.pos[2], _446.pos[3], _446.pos[4], _446.pos[5], _446.pos[6], _446.pos[7], _446.pos[8], _446.pos[9], _446.pos[10], _446.pos[11], _446.pos[12], _446.pos[13] };
    atlas_texture_t _2485 = { _446.size, _446.atlas, _2479, _2515 };
    uint _532 = _446.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_532)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_532)], float3(((TransformUV(uvs, _2485, lod) + rand) - 0.5f.xx) * 0.000118371215648949146270751953125f.xx, float((_2479[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
    bool _547;
    if (maybe_YCoCg)
    {
        _547 = _446.atlas == 4u;
    }
    else
    {
        _547 = maybe_YCoCg;
    }
    if (_547)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _566;
    if (maybe_SRGB)
    {
        _566 = (_446.size & 32768u) != 0u;
    }
    else
    {
        _566 = maybe_SRGB;
    }
    if (_566)
    {
        float3 param_1 = res.xyz;
        float3 _572 = srgb_to_rgb(param_1);
        float4 _2732 = res;
        _2732.x = _572.x;
        float4 _2734 = _2732;
        _2734.y = _572.y;
        float4 _2736 = _2734;
        _2736.z = _572.z;
        res = _2736;
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
    bool _2387 = false;
    float3 _2384;
    do
    {
        float3 ro = float3(r.o[0], r.o[1], r.o[2]);
        float3 _1426 = float3(r.d[0], r.d[1], r.d[2]);
        float3 rc = float3(r.c[0], r.c[1], r.c[2]);
        int depth = r.depth >> 24;
        uint param = uint(hash(r.xy));
        float _1445 = construct_float(param);
        uint param_1 = uint(hash(hash(r.xy)));
        float _1452 = construct_float(param_1);
        int rand_index = _1458_g_params.hi + (total_depth(r) * 9);
        float _1470;
        if (r.dist > 0.0f)
        {
            _1470 = r.dist;
        }
        else
        {
            _1470 = 3402823346297367662189621542912.0f;
        }
        float dist = _1470;
        float3 param_2 = _1426;
        float3 _1481 = safe_invert(param_2);
        int _1535;
        int _2578;
        int _2579;
        float _2581;
        float _2582;
        while (dist > 9.9999997473787516355514526367188e-06f)
        {
            int _2577 = 0;
            float _2580 = dist;
            float3 param_3 = ro;
            float3 param_4 = _1426;
            float3 param_5 = _1481;
            uint param_6 = _1458_g_params.node_index;
            hit_data_t _2589 = { 0, _2578, _2579, dist, _2581, _2582 };
            hit_data_t param_7 = _2589;
            bool _1507 = Traverse_MacroTree_WithStack(param_3, param_4, param_5, param_6, param_7);
            _2577 = param_7.mask;
            _2578 = param_7.obj_index;
            _2579 = param_7.prim_index;
            _2580 = param_7.t;
            _2581 = param_7.u;
            _2582 = param_7.v;
            bool _1517;
            if (!_1507)
            {
                _1517 = depth > _1458_g_params.max_transp_depth;
            }
            else
            {
                _1517 = _1507;
            }
            if (_1517)
            {
                _2387 = true;
                _2384 = 0.0f.xxx;
                break;
            }
            if (_2577 == 0)
            {
                _2387 = true;
                _2384 = rc;
                break;
            }
            bool _1532 = param_7.prim_index < 0;
            if (_1532)
            {
                _1535 = (-1) - param_7.prim_index;
            }
            else
            {
                _1535 = param_7.prim_index;
            }
            uint _1546 = uint(_1535);
            uint _1588 = _1078.Load(_1546 * 4 + 0) * 3u;
            vertex_t _1594;
            [unroll]
            for (int _28ident = 0; _28ident < 3; _28ident++)
            {
                _1594.p[_28ident] = asfloat(_1582.Load(_28ident * 4 + _1586.Load(_1588 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _29ident = 0; _29ident < 3; _29ident++)
            {
                _1594.n[_29ident] = asfloat(_1582.Load(_29ident * 4 + _1586.Load(_1588 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _30ident = 0; _30ident < 3; _30ident++)
            {
                _1594.b[_30ident] = asfloat(_1582.Load(_30ident * 4 + _1586.Load(_1588 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _31ident = 0; _31ident < 2; _31ident++)
            {
                [unroll]
                for (int _32ident = 0; _32ident < 2; _32ident++)
                {
                    _1594.t[_31ident][_32ident] = asfloat(_1582.Load(_32ident * 4 + _31ident * 8 + _1586.Load(_1588 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1642;
            [unroll]
            for (int _33ident = 0; _33ident < 3; _33ident++)
            {
                _1642.p[_33ident] = asfloat(_1582.Load(_33ident * 4 + _1586.Load((_1588 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _34ident = 0; _34ident < 3; _34ident++)
            {
                _1642.n[_34ident] = asfloat(_1582.Load(_34ident * 4 + _1586.Load((_1588 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _35ident = 0; _35ident < 3; _35ident++)
            {
                _1642.b[_35ident] = asfloat(_1582.Load(_35ident * 4 + _1586.Load((_1588 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _36ident = 0; _36ident < 2; _36ident++)
            {
                [unroll]
                for (int _37ident = 0; _37ident < 2; _37ident++)
                {
                    _1642.t[_36ident][_37ident] = asfloat(_1582.Load(_37ident * 4 + _36ident * 8 + _1586.Load((_1588 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1688;
            [unroll]
            for (int _38ident = 0; _38ident < 3; _38ident++)
            {
                _1688.p[_38ident] = asfloat(_1582.Load(_38ident * 4 + _1586.Load((_1588 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _39ident = 0; _39ident < 3; _39ident++)
            {
                _1688.n[_39ident] = asfloat(_1582.Load(_39ident * 4 + _1586.Load((_1588 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _1688.b[_40ident] = asfloat(_1582.Load(_40ident * 4 + _1586.Load((_1588 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 2; _41ident++)
            {
                [unroll]
                for (int _42ident = 0; _42ident < 2; _42ident++)
                {
                    _1688.t[_41ident][_42ident] = asfloat(_1582.Load(_42ident * 4 + _41ident * 8 + _1586.Load((_1588 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            float2 _1759 = ((float2(_1594.t[0][0], _1594.t[0][1]) * ((1.0f - param_7.u) - param_7.v)) + (float2(_1642.t[0][0], _1642.t[0][1]) * param_7.u)) + (float2(_1688.t[0][0], _1688.t[0][1]) * param_7.v);
            g_stack[gl_LocalInvocationIndex][0] = (_1532 ? (_1087.Load(_1078.Load(_1546 * 4 + 0) * 4 + 0) & 65535u) : ((_1087.Load(_1078.Load(_1546 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u;
            g_stack[gl_LocalInvocationIndex][1] = 1065353216u;
            int stack_size = 1;
            float3 throughput = 0.0f.xxx;
            float2 _1790 = float2(frac(asfloat(_1772.Load((rand_index + 7) * 4 + 0)) + _1445), frac(asfloat(_1772.Load((rand_index + 8) * 4 + 0)) + _1452));
            for (;;)
            {
                int _1796 = stack_size;
                stack_size = _1796 - 1;
                if (_1796 != 0)
                {
                    int _1812 = stack_size;
                    int _1813 = 2 * _1812;
                    material_t _1819;
                    [unroll]
                    for (int _43ident = 0; _43ident < 5; _43ident++)
                    {
                        _1819.textures[_43ident] = _1810.Load(_43ident * 4 + g_stack[gl_LocalInvocationIndex][_1813] * 76 + 0);
                    }
                    [unroll]
                    for (int _44ident = 0; _44ident < 3; _44ident++)
                    {
                        _1819.base_color[_44ident] = asfloat(_1810.Load(_44ident * 4 + g_stack[gl_LocalInvocationIndex][_1813] * 76 + 20));
                    }
                    _1819.flags = _1810.Load(g_stack[gl_LocalInvocationIndex][_1813] * 76 + 32);
                    _1819.type = _1810.Load(g_stack[gl_LocalInvocationIndex][_1813] * 76 + 36);
                    _1819.tangent_rotation_or_strength = asfloat(_1810.Load(g_stack[gl_LocalInvocationIndex][_1813] * 76 + 40));
                    _1819.roughness_and_anisotropic = _1810.Load(g_stack[gl_LocalInvocationIndex][_1813] * 76 + 44);
                    _1819.ior = asfloat(_1810.Load(g_stack[gl_LocalInvocationIndex][_1813] * 76 + 48));
                    _1819.sheen_and_sheen_tint = _1810.Load(g_stack[gl_LocalInvocationIndex][_1813] * 76 + 52);
                    _1819.tint_and_metallic = _1810.Load(g_stack[gl_LocalInvocationIndex][_1813] * 76 + 56);
                    _1819.transmission_and_transmission_roughness = _1810.Load(g_stack[gl_LocalInvocationIndex][_1813] * 76 + 60);
                    _1819.specular_and_specular_tint = _1810.Load(g_stack[gl_LocalInvocationIndex][_1813] * 76 + 64);
                    _1819.clearcoat_and_clearcoat_roughness = _1810.Load(g_stack[gl_LocalInvocationIndex][_1813] * 76 + 68);
                    _1819.normal_map_strength_unorm = _1810.Load(g_stack[gl_LocalInvocationIndex][_1813] * 76 + 72);
                    uint _1869 = g_stack[gl_LocalInvocationIndex][_1813 + 1];
                    float _1870 = asfloat(_1869);
                    if (_1819.type == 4u)
                    {
                        float mix_val = _1819.tangent_rotation_or_strength;
                        if (_1819.textures[1] != 4294967295u)
                        {
                            mix_val *= SampleBilinear(_1819.textures[1], _1759, 0, _1790).x;
                        }
                        int _1895 = 2 * stack_size;
                        g_stack[gl_LocalInvocationIndex][_1895] = _1819.textures[3];
                        g_stack[gl_LocalInvocationIndex][_1895 + 1] = asuint(_1870 * (1.0f - mix_val));
                        int _1914 = 2 * (stack_size + 1);
                        g_stack[gl_LocalInvocationIndex][_1914] = _1819.textures[4];
                        g_stack[gl_LocalInvocationIndex][_1914 + 1] = asuint(_1870 * mix_val);
                        stack_size += 2;
                    }
                    else
                    {
                        if (_1819.type == 5u)
                        {
                            throughput += (float3(_1819.base_color[0], _1819.base_color[1], _1819.base_color[2]) * _1870);
                        }
                    }
                    continue;
                }
                else
                {
                    break;
                }
            }
            float3 _1948 = rc;
            float3 _1949 = _1948 * throughput;
            rc = _1949;
            if (lum(_1949) < 1.0000000116860974230803549289703e-07f)
            {
                break;
            }
            float _1959 = _2580 + 9.9999997473787516355514526367188e-06f;
            ro += (_1426 * _1959);
            dist -= _1959;
            depth++;
            rand_index += 9;
        }
        if (_2387)
        {
            break;
        }
        _2387 = true;
        _2384 = rc;
        break;
    } while(false);
    return _2384;
}

float IntersectAreaLightsShadow(shadow_ray_t r)
{
    bool _2394 = false;
    float _2391;
    do
    {
        float3 _1982 = float3(r.o[0], r.o[1], r.o[2]);
        float3 _1990 = float3(r.d[0], r.d[1], r.d[2]);
        float _1994 = abs(r.dist);
        for (uint li = 0u; li < uint(_1458_g_params.blocker_lights_count); li++)
        {
            light_t _2026;
            _2026.type_and_param0 = _2022.Load4(_2010.Load(li * 4 + 0) * 64 + 0);
            _2026.param1 = asfloat(_2022.Load4(_2010.Load(li * 4 + 0) * 64 + 16));
            _2026.param2 = asfloat(_2022.Load4(_2010.Load(li * 4 + 0) * 64 + 32));
            _2026.param3 = asfloat(_2022.Load4(_2010.Load(li * 4 + 0) * 64 + 48));
            bool _2040 = (_2026.type_and_param0.x & 128u) != 0u;
            bool _2046;
            if (_2040)
            {
                _2046 = r.dist >= 0.0f;
            }
            else
            {
                _2046 = _2040;
            }
            [branch]
            if (_2046)
            {
                continue;
            }
            uint _2054 = _2026.type_and_param0.x & 31u;
            if (_2054 == 4u)
            {
                float3 light_u = _2026.param2.xyz;
                float3 light_v = _2026.param3.xyz;
                float3 _2075 = normalize(cross(_2026.param2.xyz, _2026.param3.xyz));
                float _2083 = dot(_1990, _2075);
                float _2091 = (dot(_2075, _2026.param1.xyz) - dot(_2075, _1982)) / _2083;
                if (((_2083 < 0.0f) && (_2091 > 9.9999999747524270787835121154785e-07f)) && (_2091 < _1994))
                {
                    float3 _2104 = light_u;
                    float3 _2109 = _2104 / dot(_2104, _2104).xxx;
                    light_u = _2109;
                    light_v /= dot(light_v, light_v).xxx;
                    float3 _2125 = (_1982 + (_1990 * _2091)) - _2026.param1.xyz;
                    float _2129 = dot(_2109, _2125);
                    if ((_2129 >= (-0.5f)) && (_2129 <= 0.5f))
                    {
                        float _2141 = dot(light_v, _2125);
                        if ((_2141 >= (-0.5f)) && (_2141 <= 0.5f))
                        {
                            _2394 = true;
                            _2391 = 0.0f;
                            break;
                        }
                    }
                }
            }
            else
            {
                if (_2054 == 5u)
                {
                    float3 light_u_1 = _2026.param2.xyz;
                    float3 light_v_1 = _2026.param3.xyz;
                    float3 _2171 = normalize(cross(_2026.param2.xyz, _2026.param3.xyz));
                    float _2179 = dot(_1990, _2171);
                    float _2187 = (dot(_2171, _2026.param1.xyz) - dot(_2171, _1982)) / _2179;
                    if (((_2179 < 0.0f) && (_2187 > 9.9999999747524270787835121154785e-07f)) && (_2187 < _1994))
                    {
                        float3 _2199 = light_u_1;
                        float3 _2204 = _2199 / dot(_2199, _2199).xxx;
                        light_u_1 = _2204;
                        float3 _2205 = light_v_1;
                        float3 _2210 = _2205 / dot(_2205, _2205).xxx;
                        light_v_1 = _2210;
                        float3 _2220 = (_1982 + (_1990 * _2187)) - _2026.param1.xyz;
                        float _2224 = dot(_2204, _2220);
                        float _2228 = dot(_2210, _2220);
                        if (sqrt(mad(_2224, _2224, _2228 * _2228)) <= 0.5f)
                        {
                            _2394 = true;
                            _2391 = 0.0f;
                            break;
                        }
                    }
                }
            }
        }
        if (_2394)
        {
            break;
        }
        _2394 = true;
        _2391 = 1.0f;
        break;
    } while(false);
    return _2391;
}

void comp_main()
{
    do
    {
        int _2254 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_2254) >= _2260.Load(12))
        {
            break;
        }
        shadow_ray_t _2279;
        [unroll]
        for (int _45ident = 0; _45ident < 3; _45ident++)
        {
            _2279.o[_45ident] = asfloat(_2275.Load(_45ident * 4 + _2254 * 48 + 0));
        }
        _2279.depth = int(_2275.Load(_2254 * 48 + 12));
        [unroll]
        for (int _46ident = 0; _46ident < 3; _46ident++)
        {
            _2279.d[_46ident] = asfloat(_2275.Load(_46ident * 4 + _2254 * 48 + 16));
        }
        _2279.dist = asfloat(_2275.Load(_2254 * 48 + 28));
        [unroll]
        for (int _47ident = 0; _47ident < 3; _47ident++)
        {
            _2279.c[_47ident] = asfloat(_2275.Load(_47ident * 4 + _2254 * 48 + 32));
        }
        _2279.xy = int(_2275.Load(_2254 * 48 + 44));
        float _2464[3] = { _2279.c[0], _2279.c[1], _2279.c[2] };
        float _2453[3] = { _2279.d[0], _2279.d[1], _2279.d[2] };
        float _2442[3] = { _2279.o[0], _2279.o[1], _2279.o[2] };
        shadow_ray_t _2428 = { _2442, _2279.depth, _2453, _2279.dist, _2464, _2279.xy };
        shadow_ray_t param = _2428;
        float3 _2313 = IntersectSceneShadow(param);
        shadow_ray_t param_1 = _2428;
        float3 _2317 = _2313 * IntersectAreaLightsShadow(param_1);
        if (lum(_2317) > 0.0f)
        {
            int2 _2341 = int2((_2279.xy >> 16) & 65535, _2279.xy & 65535);
            float4 _2342 = g_inout_img[_2341];
            float3 _2351 = _2342.xyz + min(_2317, _1458_g_params.clamp_val.xxx);
            float4 _2700 = _2342;
            _2700.x = _2351.x;
            float4 _2702 = _2700;
            _2702.y = _2351.y;
            float4 _2704 = _2702;
            _2704.z = _2351.z;
            g_inout_img[_2341] = _2704;
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
