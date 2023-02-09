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
    int blocker_lights_count;
    int _pad0;
    int _pad1;
    int _pad2;
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

ByteAddressBuffer _368 : register(t20, space0);
ByteAddressBuffer _640 : register(t1, space0);
ByteAddressBuffer _860 : register(t5, space0);
ByteAddressBuffer _1000 : register(t2, space0);
ByteAddressBuffer _1009 : register(t3, space0);
ByteAddressBuffer _1180 : register(t7, space0);
ByteAddressBuffer _1184 : register(t8, space0);
ByteAddressBuffer _1204 : register(t6, space0);
ByteAddressBuffer _1248 : register(t9, space0);
ByteAddressBuffer _1484 : register(t10, space0);
ByteAddressBuffer _1488 : register(t11, space0);
ByteAddressBuffer _1689 : register(t4, space0);
ByteAddressBuffer _1886 : register(t15, space0);
ByteAddressBuffer _1898 : register(t14, space0);
ByteAddressBuffer _2137 : register(t13, space0);
ByteAddressBuffer _2152 : register(t12, space0);
cbuffer UniformParams
{
    Params _1395_g_params : packoffset(c0);
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

float3 safe_invert(float3 v)
{
    float3 inv_v = 1.0f.xxx / v;
    bool _141 = v.x <= 1.0000000116860974230803549289703e-07f;
    bool _148;
    if (_141)
    {
        _148 = v.x >= 0.0f;
    }
    else
    {
        _148 = _141;
    }
    if (_148)
    {
        float3 _2569 = inv_v;
        _2569.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _2569;
    }
    else
    {
        bool _157 = v.x >= (-1.0000000116860974230803549289703e-07f);
        bool _163;
        if (_157)
        {
            _163 = v.x < 0.0f;
        }
        else
        {
            _163 = _157;
        }
        if (_163)
        {
            float3 _2571 = inv_v;
            _2571.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _2571;
        }
    }
    bool _170 = v.y <= 1.0000000116860974230803549289703e-07f;
    bool _176;
    if (_170)
    {
        _176 = v.y >= 0.0f;
    }
    else
    {
        _176 = _170;
    }
    if (_176)
    {
        float3 _2573 = inv_v;
        _2573.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _2573;
    }
    else
    {
        bool _183 = v.y >= (-1.0000000116860974230803549289703e-07f);
        bool _189;
        if (_183)
        {
            _189 = v.y < 0.0f;
        }
        else
        {
            _189 = _183;
        }
        if (_189)
        {
            float3 _2575 = inv_v;
            _2575.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _2575;
        }
    }
    bool _195 = v.z <= 1.0000000116860974230803549289703e-07f;
    bool _201;
    if (_195)
    {
        _201 = v.z >= 0.0f;
    }
    else
    {
        _201 = _195;
    }
    if (_201)
    {
        float3 _2577 = inv_v;
        _2577.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _2577;
    }
    else
    {
        bool _208 = v.z >= (-1.0000000116860974230803549289703e-07f);
        bool _214;
        if (_208)
        {
            _214 = v.z < 0.0f;
        }
        else
        {
            _214 = _208;
        }
        if (_214)
        {
            float3 _2579 = inv_v;
            _2579.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _2579;
        }
    }
    return inv_v;
}

bool _bbox_test_fma(float3 inv_d, float3 neg_inv_d_o, float t, float3 bbox_min, float3 bbox_max)
{
    float _738 = mad(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float _746 = mad(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float _761 = mad(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    float _768 = mad(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    float _785 = mad(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    float _792 = mad(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    float _797 = max(max(min(_738, _746), min(_761, _768)), min(_785, _792));
    float _805 = min(min(max(_738, _746), max(_761, _768)), max(_785, _792)) * 1.0000002384185791015625f;
    return ((_797 <= _805) && (_797 <= t)) && (_805 > 0.0f);
}

void IntersectTri(float3 ro, float3 rd, tri_accel_t tri, uint prim_index, inout hit_data_t inter)
{
    do
    {
        float _513 = dot(rd, tri.n_plane.xyz);
        float _522 = tri.n_plane.w - dot(ro, tri.n_plane.xyz);
        if (sign(_522) != sign(mad(_513, inter.t, -_522)))
        {
            break;
        }
        float3 _543 = (ro * _513) + (rd * _522);
        float _554 = mad(_513, tri.u_plane.w, dot(_543, tri.u_plane.xyz));
        float _559 = _513 - _554;
        if (sign(_554) != sign(_559))
        {
            break;
        }
        float _575 = mad(_513, tri.v_plane.w, dot(_543, tri.v_plane.xyz));
        if (sign(_575) != sign(_559 - _575))
        {
            break;
        }
        float _590 = 1.0f / _513;
        inter.mask = -1;
        int _595;
        if (_513 < 0.0f)
        {
            _595 = int(prim_index);
        }
        else
        {
            _595 = (-1) - int(prim_index);
        }
        inter.prim_index = _595;
        inter.t = _522 * _590;
        inter.u = _554 * _590;
        inter.v = _575 * _590;
        break;
    } while(false);
}

bool IntersectTris_AnyHit(float3 ro, float3 rd, int tri_start, int tri_end, int obj_index, inout hit_data_t out_inter)
{
    int _2385 = 0;
    int _2386 = obj_index;
    float _2388 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _2387;
    float _2389;
    float _2390;
    for (int i = tri_start; i < tri_end; )
    {
        param = ro;
        param_1 = rd;
        tri_accel_t _651;
        _651.n_plane = asfloat(_640.Load4(i * 48 + 0));
        _651.u_plane = asfloat(_640.Load4(i * 48 + 16));
        _651.v_plane = asfloat(_640.Load4(i * 48 + 32));
        param_2.n_plane = _651.n_plane;
        param_2.u_plane = _651.u_plane;
        param_2.v_plane = _651.v_plane;
        param_3 = uint(i);
        hit_data_t _2397 = { _2385, _2386, _2387, _2388, _2389, _2390 };
        param_4 = _2397;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _2385 = param_4.mask;
        _2386 = param_4.obj_index;
        _2387 = param_4.prim_index;
        _2388 = param_4.t;
        _2389 = param_4.u;
        _2390 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _2385;
    int _674;
    if (_2385 != 0)
    {
        _674 = _2386;
    }
    else
    {
        _674 = out_inter.obj_index;
    }
    out_inter.obj_index = _674;
    int _687;
    if (_2385 != 0)
    {
        _687 = _2387;
    }
    else
    {
        _687 = out_inter.prim_index;
    }
    out_inter.prim_index = _687;
    out_inter.t = _2388;
    float _703;
    if (_2385 != 0)
    {
        _703 = _2389;
    }
    else
    {
        _703 = out_inter.u;
    }
    out_inter.u = _703;
    float _716;
    if (_2385 != 0)
    {
        _716 = _2390;
    }
    else
    {
        _716 = out_inter.v;
    }
    out_inter.v = _716;
    return _2385 != 0;
}

bool Traverse_MicroTree_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    bool _2279 = false;
    bool _2276;
    do
    {
        float3 _822 = (-inv_d) * ro;
        uint _824 = stack_size;
        uint _834 = stack_size;
        stack_size = _834 + uint(1);
        g_stack[gl_LocalInvocationIndex][_834] = node_index;
        uint _908;
        uint _932;
        int _984;
        while (stack_size != _824)
        {
            uint _849 = stack_size;
            uint _850 = _849 - uint(1);
            stack_size = _850;
            bvh_node_t _864;
            _864.bbox_min = asfloat(_860.Load4(g_stack[gl_LocalInvocationIndex][_850] * 32 + 0));
            _864.bbox_max = asfloat(_860.Load4(g_stack[gl_LocalInvocationIndex][_850] * 32 + 16));
            float3 param = inv_d;
            float3 param_1 = _822;
            float param_2 = inter.t;
            float3 param_3 = _864.bbox_min.xyz;
            float3 param_4 = _864.bbox_max.xyz;
            if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
            {
                continue;
            }
            uint _891 = asuint(_864.bbox_min.w);
            if ((_891 & 2147483648u) == 0u)
            {
                uint _898 = stack_size;
                stack_size = _898 + uint(1);
                uint _902 = asuint(_864.bbox_max.w);
                uint _904 = _902 >> uint(30);
                if (rd[_904] < 0.0f)
                {
                    _908 = _891;
                }
                else
                {
                    _908 = _902 & 1073741823u;
                }
                g_stack[gl_LocalInvocationIndex][_898] = _908;
                uint _923 = stack_size;
                stack_size = _923 + uint(1);
                if (rd[_904] < 0.0f)
                {
                    _932 = _902 & 1073741823u;
                }
                else
                {
                    _932 = _891;
                }
                g_stack[gl_LocalInvocationIndex][_923] = _932;
            }
            else
            {
                int _952 = int(_891 & 2147483647u);
                float3 param_5 = ro;
                float3 param_6 = rd;
                int param_7 = _952;
                int param_8 = _952 + asint(_864.bbox_max.w);
                int param_9 = obj_index;
                hit_data_t param_10 = inter;
                bool _973 = IntersectTris_AnyHit(param_5, param_6, param_7, param_8, param_9, param_10);
                inter = param_10;
                if (_973)
                {
                    bool _981 = inter.prim_index < 0;
                    if (_981)
                    {
                        _984 = (-1) - inter.prim_index;
                    }
                    else
                    {
                        _984 = inter.prim_index;
                    }
                    uint _995 = uint(_984);
                    bool _1022 = !_981;
                    bool _1028;
                    if (_1022)
                    {
                        _1028 = (((_1009.Load(_1000.Load(_995 * 4 + 0) * 4 + 0) >> 16u) & 65535u) & 32768u) != 0u;
                    }
                    else
                    {
                        _1028 = _1022;
                    }
                    bool _1039;
                    if (!_1028)
                    {
                        bool _1038;
                        if (_981)
                        {
                            _1038 = ((_1009.Load(_1000.Load(_995 * 4 + 0) * 4 + 0) & 65535u) & 32768u) != 0u;
                        }
                        else
                        {
                            _1038 = _981;
                        }
                        _1039 = _1038;
                    }
                    else
                    {
                        _1039 = _1028;
                    }
                    if (_1039)
                    {
                        _2279 = true;
                        _2276 = true;
                        break;
                    }
                }
            }
        }
        if (_2279)
        {
            break;
        }
        _2279 = true;
        _2276 = false;
        break;
    } while(false);
    return _2276;
}

bool Traverse_MacroTree_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    bool _2270 = false;
    bool _2267;
    do
    {
        float3 _1050 = (-orig_inv_rd) * orig_ro;
        uint stack_size = 1u;
        g_stack[gl_LocalInvocationIndex][0u] = node_index;
        uint _1115;
        uint _1138;
        while (stack_size != 0u)
        {
            uint _1066 = stack_size;
            uint _1067 = _1066 - uint(1);
            stack_size = _1067;
            bvh_node_t _1073;
            _1073.bbox_min = asfloat(_860.Load4(g_stack[gl_LocalInvocationIndex][_1067] * 32 + 0));
            _1073.bbox_max = asfloat(_860.Load4(g_stack[gl_LocalInvocationIndex][_1067] * 32 + 16));
            float3 param = orig_inv_rd;
            float3 param_1 = _1050;
            float param_2 = inter.t;
            float3 param_3 = _1073.bbox_min.xyz;
            float3 param_4 = _1073.bbox_max.xyz;
            if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
            {
                continue;
            }
            uint _1100 = asuint(_1073.bbox_min.w);
            if ((_1100 & 2147483648u) == 0u)
            {
                uint _1106 = stack_size;
                stack_size = _1106 + uint(1);
                uint _1110 = asuint(_1073.bbox_max.w);
                uint _1111 = _1110 >> uint(30);
                if (orig_rd[_1111] < 0.0f)
                {
                    _1115 = _1100;
                }
                else
                {
                    _1115 = _1110 & 1073741823u;
                }
                g_stack[gl_LocalInvocationIndex][_1106] = _1115;
                uint _1129 = stack_size;
                stack_size = _1129 + uint(1);
                if (orig_rd[_1111] < 0.0f)
                {
                    _1138 = _1110 & 1073741823u;
                }
                else
                {
                    _1138 = _1100;
                }
                g_stack[gl_LocalInvocationIndex][_1129] = _1138;
            }
            else
            {
                uint _1156 = _1100 & 2147483647u;
                uint _1160 = asuint(_1073.bbox_max.w);
                for (uint i = _1156; i < (_1156 + _1160); i++)
                {
                    mesh_instance_t _1190;
                    _1190.bbox_min = asfloat(_1180.Load4(_1184.Load(i * 4 + 0) * 32 + 0));
                    _1190.bbox_max = asfloat(_1180.Load4(_1184.Load(i * 4 + 0) * 32 + 16));
                    mesh_t _1210;
                    [unroll]
                    for (int _24ident = 0; _24ident < 3; _24ident++)
                    {
                        _1210.bbox_min[_24ident] = asfloat(_1204.Load(_24ident * 4 + asuint(_1190.bbox_max.w) * 48 + 0));
                    }
                    [unroll]
                    for (int _25ident = 0; _25ident < 3; _25ident++)
                    {
                        _1210.bbox_max[_25ident] = asfloat(_1204.Load(_25ident * 4 + asuint(_1190.bbox_max.w) * 48 + 12));
                    }
                    _1210.node_index = _1204.Load(asuint(_1190.bbox_max.w) * 48 + 24);
                    _1210.node_count = _1204.Load(asuint(_1190.bbox_max.w) * 48 + 28);
                    _1210.tris_index = _1204.Load(asuint(_1190.bbox_max.w) * 48 + 32);
                    _1210.tris_count = _1204.Load(asuint(_1190.bbox_max.w) * 48 + 36);
                    _1210.vert_index = _1204.Load(asuint(_1190.bbox_max.w) * 48 + 40);
                    _1210.vert_count = _1204.Load(asuint(_1190.bbox_max.w) * 48 + 44);
                    transform_t _1254;
                    _1254.xform = asfloat(uint4x4(_1248.Load4(asuint(_1190.bbox_min.w) * 128 + 0), _1248.Load4(asuint(_1190.bbox_min.w) * 128 + 16), _1248.Load4(asuint(_1190.bbox_min.w) * 128 + 32), _1248.Load4(asuint(_1190.bbox_min.w) * 128 + 48)));
                    _1254.inv_xform = asfloat(uint4x4(_1248.Load4(asuint(_1190.bbox_min.w) * 128 + 64), _1248.Load4(asuint(_1190.bbox_min.w) * 128 + 80), _1248.Load4(asuint(_1190.bbox_min.w) * 128 + 96), _1248.Load4(asuint(_1190.bbox_min.w) * 128 + 112)));
                    float3 param_5 = orig_inv_rd;
                    float3 param_6 = _1050;
                    float param_7 = inter.t;
                    float3 param_8 = _1190.bbox_min.xyz;
                    float3 param_9 = _1190.bbox_max.xyz;
                    if (!_bbox_test_fma(param_5, param_6, param_7, param_8, param_9))
                    {
                        continue;
                    }
                    float3 _1299 = mul(float4(orig_rd, 0.0f), _1254.inv_xform).xyz;
                    float3 param_10 = _1299;
                    float3 param_11 = mul(float4(orig_ro, 1.0f), _1254.inv_xform).xyz;
                    float3 param_12 = _1299;
                    float3 param_13 = safe_invert(param_10);
                    int param_14 = int(_1184.Load(i * 4 + 0));
                    uint param_15 = _1210.node_index;
                    uint param_16 = stack_size;
                    hit_data_t param_17 = inter;
                    bool _1323 = Traverse_MicroTree_WithStack(param_11, param_12, param_13, param_14, param_15, param_16, param_17);
                    inter = param_17;
                    if (_1323)
                    {
                        _2270 = true;
                        _2267 = true;
                        break;
                    }
                }
                if (_2270)
                {
                    break;
                }
            }
        }
        if (_2270)
        {
            break;
        }
        _2270 = true;
        _2267 = false;
        break;
    } while(false);
    return _2267;
}

float2 TransformUV(float2 _uv, atlas_texture_t t, int mip_level)
{
    uint _2340[14] = t.pos;
    uint _2343[14] = t.pos;
    uint _326 = t.size & 16383u;
    uint _329 = t.size >> uint(16);
    uint _330 = _329 & 16383u;
    float2 size = float2(float(_326), float(_330));
    if ((_329 & 32768u) != 0u)
    {
        size = float2(float(_326 >> uint(mip_level)), float(_330 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_2340[mip_level] & 65535u), float((_2343[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _269 = mad(col.z, 31.875f, 1.0f);
    float _279 = (col.x - 0.501960813999176025390625f) / _269;
    float _285 = (col.y - 0.501960813999176025390625f) / _269;
    return float3((col.w + _279) - _285, col.w + _285, (col.w - _279) - _285);
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
    atlas_texture_t _371;
    _371.size = _368.Load(index * 80 + 0);
    _371.atlas = _368.Load(index * 80 + 4);
    [unroll]
    for (int _26ident = 0; _26ident < 4; _26ident++)
    {
        _371.page[_26ident] = _368.Load(_26ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _27ident = 0; _27ident < 14; _27ident++)
    {
        _371.pos[_27ident] = _368.Load(_27ident * 4 + index * 80 + 24);
    }
    uint _2348[4];
    _2348[0] = _371.page[0];
    _2348[1] = _371.page[1];
    _2348[2] = _371.page[2];
    _2348[3] = _371.page[3];
    uint _2384[14] = { _371.pos[0], _371.pos[1], _371.pos[2], _371.pos[3], _371.pos[4], _371.pos[5], _371.pos[6], _371.pos[7], _371.pos[8], _371.pos[9], _371.pos[10], _371.pos[11], _371.pos[12], _371.pos[13] };
    atlas_texture_t _2354 = { _371.size, _371.atlas, _2348, _2384 };
    uint _454 = _371.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_454)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_454)], float3(TransformUV(uvs, _2354, lod) * 0.000118371215648949146270751953125f.xx, float((_2348[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
    bool _469;
    if (maybe_YCoCg)
    {
        _469 = _371.atlas == 4u;
    }
    else
    {
        _469 = maybe_YCoCg;
    }
    if (_469)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _488;
    if (maybe_SRGB)
    {
        _488 = (_371.size & 32768u) != 0u;
    }
    else
    {
        _488 = maybe_SRGB;
    }
    if (_488)
    {
        float3 param_1 = res.xyz;
        float3 _494 = srgb_to_rgb(param_1);
        float4 _2595 = res;
        _2595.x = _494.x;
        float4 _2597 = _2595;
        _2597.y = _494.y;
        float4 _2599 = _2597;
        _2599.z = _494.z;
        res = _2599;
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

float3 IntersectSceneShadow(shadow_ray_t r)
{
    bool _2256 = false;
    float3 _2253;
    do
    {
        float3 ro = float3(r.o[0], r.o[1], r.o[2]);
        float3 _1348 = float3(r.d[0], r.d[1], r.d[2]);
        float3 rc = float3(r.c[0], r.c[1], r.c[2]);
        int depth = r.depth >> 24;
        float _1366;
        if (r.dist > 0.0f)
        {
            _1366 = r.dist;
        }
        else
        {
            _1366 = 3402823346297367662189621542912.0f;
        }
        float dist = _1366;
        float3 param = _1348;
        float3 _1377 = safe_invert(param);
        int _1437;
        int _2447;
        int _2448;
        float _2450;
        float _2451;
        while (dist > 9.9999997473787516355514526367188e-06f)
        {
            int _2446 = 0;
            float _2449 = dist;
            float3 param_1 = ro;
            float3 param_2 = _1348;
            float3 param_3 = _1377;
            uint param_4 = _1395_g_params.node_index;
            hit_data_t _2458 = { 0, _2447, _2448, dist, _2450, _2451 };
            hit_data_t param_5 = _2458;
            bool _1408 = Traverse_MacroTree_WithStack(param_1, param_2, param_3, param_4, param_5);
            _2446 = param_5.mask;
            _2447 = param_5.obj_index;
            _2448 = param_5.prim_index;
            _2449 = param_5.t;
            _2450 = param_5.u;
            _2451 = param_5.v;
            bool _1419;
            if (!_1408)
            {
                _1419 = depth > _1395_g_params.max_transp_depth;
            }
            else
            {
                _1419 = _1408;
            }
            if (_1419)
            {
                _2256 = true;
                _2253 = 0.0f.xxx;
                break;
            }
            if (_2446 == 0)
            {
                _2256 = true;
                _2253 = rc;
                break;
            }
            bool _1434 = param_5.prim_index < 0;
            if (_1434)
            {
                _1437 = (-1) - param_5.prim_index;
            }
            else
            {
                _1437 = param_5.prim_index;
            }
            uint _1448 = uint(_1437);
            uint _1490 = _1000.Load(_1448 * 4 + 0) * 3u;
            vertex_t _1496;
            [unroll]
            for (int _28ident = 0; _28ident < 3; _28ident++)
            {
                _1496.p[_28ident] = asfloat(_1484.Load(_28ident * 4 + _1488.Load(_1490 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _29ident = 0; _29ident < 3; _29ident++)
            {
                _1496.n[_29ident] = asfloat(_1484.Load(_29ident * 4 + _1488.Load(_1490 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _30ident = 0; _30ident < 3; _30ident++)
            {
                _1496.b[_30ident] = asfloat(_1484.Load(_30ident * 4 + _1488.Load(_1490 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _31ident = 0; _31ident < 2; _31ident++)
            {
                [unroll]
                for (int _32ident = 0; _32ident < 2; _32ident++)
                {
                    _1496.t[_31ident][_32ident] = asfloat(_1484.Load(_32ident * 4 + _31ident * 8 + _1488.Load(_1490 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1544;
            [unroll]
            for (int _33ident = 0; _33ident < 3; _33ident++)
            {
                _1544.p[_33ident] = asfloat(_1484.Load(_33ident * 4 + _1488.Load((_1490 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _34ident = 0; _34ident < 3; _34ident++)
            {
                _1544.n[_34ident] = asfloat(_1484.Load(_34ident * 4 + _1488.Load((_1490 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _35ident = 0; _35ident < 3; _35ident++)
            {
                _1544.b[_35ident] = asfloat(_1484.Load(_35ident * 4 + _1488.Load((_1490 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _36ident = 0; _36ident < 2; _36ident++)
            {
                [unroll]
                for (int _37ident = 0; _37ident < 2; _37ident++)
                {
                    _1544.t[_36ident][_37ident] = asfloat(_1484.Load(_37ident * 4 + _36ident * 8 + _1488.Load((_1490 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1590;
            [unroll]
            for (int _38ident = 0; _38ident < 3; _38ident++)
            {
                _1590.p[_38ident] = asfloat(_1484.Load(_38ident * 4 + _1488.Load((_1490 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _39ident = 0; _39ident < 3; _39ident++)
            {
                _1590.n[_39ident] = asfloat(_1484.Load(_39ident * 4 + _1488.Load((_1490 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _1590.b[_40ident] = asfloat(_1484.Load(_40ident * 4 + _1488.Load((_1490 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 2; _41ident++)
            {
                [unroll]
                for (int _42ident = 0; _42ident < 2; _42ident++)
                {
                    _1590.t[_41ident][_42ident] = asfloat(_1484.Load(_42ident * 4 + _41ident * 8 + _1488.Load((_1490 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            float2 _1661 = ((float2(_1496.t[0][0], _1496.t[0][1]) * ((1.0f - param_5.u) - param_5.v)) + (float2(_1544.t[0][0], _1544.t[0][1]) * param_5.u)) + (float2(_1590.t[0][0], _1590.t[0][1]) * param_5.v);
            g_stack[gl_LocalInvocationIndex][0] = (_1434 ? (_1009.Load(_1000.Load(_1448 * 4 + 0) * 4 + 0) & 65535u) : ((_1009.Load(_1000.Load(_1448 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u;
            g_stack[gl_LocalInvocationIndex][1] = 1065353216u;
            int stack_size = 1;
            float3 throughput = 0.0f.xxx;
            for (;;)
            {
                int _1675 = stack_size;
                stack_size = _1675 - 1;
                if (_1675 != 0)
                {
                    int _1691 = stack_size;
                    int _1692 = 2 * _1691;
                    material_t _1698;
                    [unroll]
                    for (int _43ident = 0; _43ident < 5; _43ident++)
                    {
                        _1698.textures[_43ident] = _1689.Load(_43ident * 4 + g_stack[gl_LocalInvocationIndex][_1692] * 76 + 0);
                    }
                    [unroll]
                    for (int _44ident = 0; _44ident < 3; _44ident++)
                    {
                        _1698.base_color[_44ident] = asfloat(_1689.Load(_44ident * 4 + g_stack[gl_LocalInvocationIndex][_1692] * 76 + 20));
                    }
                    _1698.flags = _1689.Load(g_stack[gl_LocalInvocationIndex][_1692] * 76 + 32);
                    _1698.type = _1689.Load(g_stack[gl_LocalInvocationIndex][_1692] * 76 + 36);
                    _1698.tangent_rotation_or_strength = asfloat(_1689.Load(g_stack[gl_LocalInvocationIndex][_1692] * 76 + 40));
                    _1698.roughness_and_anisotropic = _1689.Load(g_stack[gl_LocalInvocationIndex][_1692] * 76 + 44);
                    _1698.ior = asfloat(_1689.Load(g_stack[gl_LocalInvocationIndex][_1692] * 76 + 48));
                    _1698.sheen_and_sheen_tint = _1689.Load(g_stack[gl_LocalInvocationIndex][_1692] * 76 + 52);
                    _1698.tint_and_metallic = _1689.Load(g_stack[gl_LocalInvocationIndex][_1692] * 76 + 56);
                    _1698.transmission_and_transmission_roughness = _1689.Load(g_stack[gl_LocalInvocationIndex][_1692] * 76 + 60);
                    _1698.specular_and_specular_tint = _1689.Load(g_stack[gl_LocalInvocationIndex][_1692] * 76 + 64);
                    _1698.clearcoat_and_clearcoat_roughness = _1689.Load(g_stack[gl_LocalInvocationIndex][_1692] * 76 + 68);
                    _1698.normal_map_strength_unorm = _1689.Load(g_stack[gl_LocalInvocationIndex][_1692] * 76 + 72);
                    uint _1748 = g_stack[gl_LocalInvocationIndex][_1692 + 1];
                    float _1749 = asfloat(_1748);
                    if (_1698.type == 4u)
                    {
                        float mix_val = _1698.tangent_rotation_or_strength;
                        if (_1698.textures[1] != 4294967295u)
                        {
                            mix_val *= SampleBilinear(_1698.textures[1], _1661, 0).x;
                        }
                        int _1773 = 2 * stack_size;
                        g_stack[gl_LocalInvocationIndex][_1773] = _1698.textures[3];
                        g_stack[gl_LocalInvocationIndex][_1773 + 1] = asuint(_1749 * (1.0f - mix_val));
                        int _1792 = 2 * (stack_size + 1);
                        g_stack[gl_LocalInvocationIndex][_1792] = _1698.textures[4];
                        g_stack[gl_LocalInvocationIndex][_1792 + 1] = asuint(_1749 * mix_val);
                        stack_size += 2;
                    }
                    else
                    {
                        if (_1698.type == 5u)
                        {
                            throughput += (float3(_1698.base_color[0], _1698.base_color[1], _1698.base_color[2]) * _1749);
                        }
                    }
                    continue;
                }
                else
                {
                    break;
                }
            }
            float3 _1826 = rc;
            float3 _1827 = _1826 * throughput;
            rc = _1827;
            if (lum(_1827) < 1.0000000116860974230803549289703e-07f)
            {
                break;
            }
            float _1837 = _2449 + 9.9999997473787516355514526367188e-06f;
            ro += (_1348 * _1837);
            dist -= _1837;
            depth++;
        }
        if (_2256)
        {
            break;
        }
        _2256 = true;
        _2253 = rc;
        break;
    } while(false);
    return _2253;
}

float IntersectAreaLightsShadow(shadow_ray_t r)
{
    bool _2263 = false;
    float _2260;
    do
    {
        float3 _1858 = float3(r.o[0], r.o[1], r.o[2]);
        float3 _1866 = float3(r.d[0], r.d[1], r.d[2]);
        float _1870 = abs(r.dist);
        for (uint li = 0u; li < uint(_1395_g_params.blocker_lights_count); li++)
        {
            light_t _1902;
            _1902.type_and_param0 = _1898.Load4(_1886.Load(li * 4 + 0) * 64 + 0);
            _1902.param1 = asfloat(_1898.Load4(_1886.Load(li * 4 + 0) * 64 + 16));
            _1902.param2 = asfloat(_1898.Load4(_1886.Load(li * 4 + 0) * 64 + 32));
            _1902.param3 = asfloat(_1898.Load4(_1886.Load(li * 4 + 0) * 64 + 48));
            bool _1916 = (_1902.type_and_param0.x & 128u) != 0u;
            bool _1922;
            if (_1916)
            {
                _1922 = r.dist >= 0.0f;
            }
            else
            {
                _1922 = _1916;
            }
            [branch]
            if (_1922)
            {
                continue;
            }
            uint _1930 = _1902.type_and_param0.x & 31u;
            if (_1930 == 4u)
            {
                float3 light_u = _1902.param2.xyz;
                float3 light_v = _1902.param3.xyz;
                float3 _1951 = normalize(cross(_1902.param2.xyz, _1902.param3.xyz));
                float _1959 = dot(_1866, _1951);
                float _1967 = (dot(_1951, _1902.param1.xyz) - dot(_1951, _1858)) / _1959;
                if (((_1959 < 0.0f) && (_1967 > 9.9999999747524270787835121154785e-07f)) && (_1967 < _1870))
                {
                    float3 _1980 = light_u;
                    float3 _1985 = _1980 / dot(_1980, _1980).xxx;
                    light_u = _1985;
                    light_v /= dot(light_v, light_v).xxx;
                    float3 _2001 = (_1858 + (_1866 * _1967)) - _1902.param1.xyz;
                    float _2005 = dot(_1985, _2001);
                    if ((_2005 >= (-0.5f)) && (_2005 <= 0.5f))
                    {
                        float _2018 = dot(light_v, _2001);
                        if ((_2018 >= (-0.5f)) && (_2018 <= 0.5f))
                        {
                            _2263 = true;
                            _2260 = 0.0f;
                            break;
                        }
                    }
                }
            }
            else
            {
                if (_1930 == 5u)
                {
                    float3 light_u_1 = _1902.param2.xyz;
                    float3 light_v_1 = _1902.param3.xyz;
                    float3 _2048 = normalize(cross(_1902.param2.xyz, _1902.param3.xyz));
                    float _2056 = dot(_1866, _2048);
                    float _2064 = (dot(_2048, _1902.param1.xyz) - dot(_2048, _1858)) / _2056;
                    if (((_2056 < 0.0f) && (_2064 > 9.9999999747524270787835121154785e-07f)) && (_2064 < _1870))
                    {
                        float3 _2076 = light_u_1;
                        float3 _2081 = _2076 / dot(_2076, _2076).xxx;
                        light_u_1 = _2081;
                        float3 _2082 = light_v_1;
                        float3 _2087 = _2082 / dot(_2082, _2082).xxx;
                        light_v_1 = _2087;
                        float3 _2097 = (_1858 + (_1866 * _2064)) - _1902.param1.xyz;
                        float _2101 = dot(_2081, _2097);
                        float _2105 = dot(_2087, _2097);
                        if (sqrt(mad(_2101, _2101, _2105 * _2105)) <= 0.5f)
                        {
                            _2263 = true;
                            _2260 = 0.0f;
                            break;
                        }
                    }
                }
            }
        }
        if (_2263)
        {
            break;
        }
        _2263 = true;
        _2260 = 1.0f;
        break;
    } while(false);
    return _2260;
}

void comp_main()
{
    do
    {
        int _2131 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_2131) >= _2137.Load(12))
        {
            break;
        }
        shadow_ray_t _2156;
        [unroll]
        for (int _45ident = 0; _45ident < 3; _45ident++)
        {
            _2156.o[_45ident] = asfloat(_2152.Load(_45ident * 4 + _2131 * 48 + 0));
        }
        _2156.depth = int(_2152.Load(_2131 * 48 + 12));
        [unroll]
        for (int _46ident = 0; _46ident < 3; _46ident++)
        {
            _2156.d[_46ident] = asfloat(_2152.Load(_46ident * 4 + _2131 * 48 + 16));
        }
        _2156.dist = asfloat(_2152.Load(_2131 * 48 + 28));
        [unroll]
        for (int _47ident = 0; _47ident < 3; _47ident++)
        {
            _2156.c[_47ident] = asfloat(_2152.Load(_47ident * 4 + _2131 * 48 + 32));
        }
        _2156.xy = int(_2152.Load(_2131 * 48 + 44));
        float _2333[3] = { _2156.c[0], _2156.c[1], _2156.c[2] };
        float _2322[3] = { _2156.d[0], _2156.d[1], _2156.d[2] };
        float _2311[3] = { _2156.o[0], _2156.o[1], _2156.o[2] };
        shadow_ray_t _2297 = { _2311, _2156.depth, _2322, _2156.dist, _2333, _2156.xy };
        shadow_ray_t param = _2297;
        float3 _2190 = IntersectSceneShadow(param);
        shadow_ray_t param_1 = _2297;
        float3 _2194 = _2190 * IntersectAreaLightsShadow(param_1);
        if (lum(_2194) > 0.0f)
        {
            int2 _2218 = int2((_2156.xy >> 16) & 65535, _2156.xy & 65535);
            g_inout_img[_2218] = float4(g_inout_img[_2218].xyz + _2194, 1.0f);
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
