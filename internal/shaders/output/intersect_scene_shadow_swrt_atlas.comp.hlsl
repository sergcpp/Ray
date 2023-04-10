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
    uint node_index;
    int max_transp_depth;
    int blocker_lights_count;
    float clamp_val;
    int _pad0;
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
ByteAddressBuffer _1483 : register(t10, space0);
ByteAddressBuffer _1487 : register(t11, space0);
ByteAddressBuffer _1688 : register(t4, space0);
ByteAddressBuffer _1885 : register(t15, space0);
ByteAddressBuffer _1897 : register(t14, space0);
ByteAddressBuffer _2136 : register(t13, space0);
ByteAddressBuffer _2151 : register(t12, space0);
cbuffer UniformParams
{
    Params _1394_g_params : packoffset(c0);
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
        float3 _2581 = inv_v;
        _2581.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _2581;
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
            float3 _2583 = inv_v;
            _2583.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _2583;
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
        float3 _2585 = inv_v;
        _2585.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _2585;
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
            float3 _2587 = inv_v;
            _2587.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _2587;
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
        float3 _2589 = inv_v;
        _2589.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _2589;
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
            float3 _2591 = inv_v;
            _2591.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _2591;
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
    int _2391 = 0;
    int _2392 = obj_index;
    float _2394 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _2393;
    float _2395;
    float _2396;
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
        hit_data_t _2403 = { _2391, _2392, _2393, _2394, _2395, _2396 };
        param_4 = _2403;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _2391 = param_4.mask;
        _2392 = param_4.obj_index;
        _2393 = param_4.prim_index;
        _2394 = param_4.t;
        _2395 = param_4.u;
        _2396 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _2391;
    int _674;
    if (_2391 != 0)
    {
        _674 = _2392;
    }
    else
    {
        _674 = out_inter.obj_index;
    }
    out_inter.obj_index = _674;
    int _687;
    if (_2391 != 0)
    {
        _687 = _2393;
    }
    else
    {
        _687 = out_inter.prim_index;
    }
    out_inter.prim_index = _687;
    out_inter.t = _2394;
    float _703;
    if (_2391 != 0)
    {
        _703 = _2395;
    }
    else
    {
        _703 = out_inter.u;
    }
    out_inter.u = _703;
    float _716;
    if (_2391 != 0)
    {
        _716 = _2396;
    }
    else
    {
        _716 = out_inter.v;
    }
    out_inter.v = _716;
    return _2391 != 0;
}

bool Traverse_MicroTree_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    bool _2285 = false;
    bool _2282;
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
                        _2285 = true;
                        _2282 = true;
                        break;
                    }
                }
            }
        }
        if (_2285)
        {
            break;
        }
        _2285 = true;
        _2282 = false;
        break;
    } while(false);
    return _2282;
}

bool Traverse_MacroTree_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    bool _2276 = false;
    bool _2273;
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
                        _2276 = true;
                        _2273 = true;
                        break;
                    }
                }
                if (_2276)
                {
                    break;
                }
            }
        }
        if (_2276)
        {
            break;
        }
        _2276 = true;
        _2273 = false;
        break;
    } while(false);
    return _2273;
}

float2 TransformUV(float2 _uv, atlas_texture_t t, int mip_level)
{
    uint _2346[14] = t.pos;
    uint _2349[14] = t.pos;
    uint _326 = t.size & 16383u;
    uint _329 = t.size >> uint(16);
    uint _330 = _329 & 16383u;
    float2 size = float2(float(_326), float(_330));
    if ((_329 & 32768u) != 0u)
    {
        size = float2(float(_326 >> uint(mip_level)), float(_330 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_2346[mip_level] & 65535u), float((_2349[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
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
    uint _2354[4];
    _2354[0] = _371.page[0];
    _2354[1] = _371.page[1];
    _2354[2] = _371.page[2];
    _2354[3] = _371.page[3];
    uint _2390[14] = { _371.pos[0], _371.pos[1], _371.pos[2], _371.pos[3], _371.pos[4], _371.pos[5], _371.pos[6], _371.pos[7], _371.pos[8], _371.pos[9], _371.pos[10], _371.pos[11], _371.pos[12], _371.pos[13] };
    atlas_texture_t _2360 = { _371.size, _371.atlas, _2354, _2390 };
    uint _454 = _371.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_454)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_454)], float3(TransformUV(uvs, _2360, lod) * 0.000118371215648949146270751953125f.xx, float((_2354[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
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
        float4 _2607 = res;
        _2607.x = _494.x;
        float4 _2609 = _2607;
        _2609.y = _494.y;
        float4 _2611 = _2609;
        _2611.z = _494.z;
        res = _2611;
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
    bool _2262 = false;
    float3 _2259;
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
        int _1436;
        int _2453;
        int _2454;
        float _2456;
        float _2457;
        while (dist > 9.9999997473787516355514526367188e-06f)
        {
            int _2452 = 0;
            float _2455 = dist;
            float3 param_1 = ro;
            float3 param_2 = _1348;
            float3 param_3 = _1377;
            uint param_4 = _1394_g_params.node_index;
            hit_data_t _2464 = { 0, _2453, _2454, dist, _2456, _2457 };
            hit_data_t param_5 = _2464;
            bool _1407 = Traverse_MacroTree_WithStack(param_1, param_2, param_3, param_4, param_5);
            _2452 = param_5.mask;
            _2453 = param_5.obj_index;
            _2454 = param_5.prim_index;
            _2455 = param_5.t;
            _2456 = param_5.u;
            _2457 = param_5.v;
            bool _1418;
            if (!_1407)
            {
                _1418 = depth > _1394_g_params.max_transp_depth;
            }
            else
            {
                _1418 = _1407;
            }
            if (_1418)
            {
                _2262 = true;
                _2259 = 0.0f.xxx;
                break;
            }
            if (_2452 == 0)
            {
                _2262 = true;
                _2259 = rc;
                break;
            }
            bool _1433 = param_5.prim_index < 0;
            if (_1433)
            {
                _1436 = (-1) - param_5.prim_index;
            }
            else
            {
                _1436 = param_5.prim_index;
            }
            uint _1447 = uint(_1436);
            uint _1489 = _1000.Load(_1447 * 4 + 0) * 3u;
            vertex_t _1495;
            [unroll]
            for (int _28ident = 0; _28ident < 3; _28ident++)
            {
                _1495.p[_28ident] = asfloat(_1483.Load(_28ident * 4 + _1487.Load(_1489 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _29ident = 0; _29ident < 3; _29ident++)
            {
                _1495.n[_29ident] = asfloat(_1483.Load(_29ident * 4 + _1487.Load(_1489 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _30ident = 0; _30ident < 3; _30ident++)
            {
                _1495.b[_30ident] = asfloat(_1483.Load(_30ident * 4 + _1487.Load(_1489 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _31ident = 0; _31ident < 2; _31ident++)
            {
                [unroll]
                for (int _32ident = 0; _32ident < 2; _32ident++)
                {
                    _1495.t[_31ident][_32ident] = asfloat(_1483.Load(_32ident * 4 + _31ident * 8 + _1487.Load(_1489 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1543;
            [unroll]
            for (int _33ident = 0; _33ident < 3; _33ident++)
            {
                _1543.p[_33ident] = asfloat(_1483.Load(_33ident * 4 + _1487.Load((_1489 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _34ident = 0; _34ident < 3; _34ident++)
            {
                _1543.n[_34ident] = asfloat(_1483.Load(_34ident * 4 + _1487.Load((_1489 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _35ident = 0; _35ident < 3; _35ident++)
            {
                _1543.b[_35ident] = asfloat(_1483.Load(_35ident * 4 + _1487.Load((_1489 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _36ident = 0; _36ident < 2; _36ident++)
            {
                [unroll]
                for (int _37ident = 0; _37ident < 2; _37ident++)
                {
                    _1543.t[_36ident][_37ident] = asfloat(_1483.Load(_37ident * 4 + _36ident * 8 + _1487.Load((_1489 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1589;
            [unroll]
            for (int _38ident = 0; _38ident < 3; _38ident++)
            {
                _1589.p[_38ident] = asfloat(_1483.Load(_38ident * 4 + _1487.Load((_1489 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _39ident = 0; _39ident < 3; _39ident++)
            {
                _1589.n[_39ident] = asfloat(_1483.Load(_39ident * 4 + _1487.Load((_1489 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _40ident = 0; _40ident < 3; _40ident++)
            {
                _1589.b[_40ident] = asfloat(_1483.Load(_40ident * 4 + _1487.Load((_1489 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _41ident = 0; _41ident < 2; _41ident++)
            {
                [unroll]
                for (int _42ident = 0; _42ident < 2; _42ident++)
                {
                    _1589.t[_41ident][_42ident] = asfloat(_1483.Load(_42ident * 4 + _41ident * 8 + _1487.Load((_1489 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            float2 _1660 = ((float2(_1495.t[0][0], _1495.t[0][1]) * ((1.0f - param_5.u) - param_5.v)) + (float2(_1543.t[0][0], _1543.t[0][1]) * param_5.u)) + (float2(_1589.t[0][0], _1589.t[0][1]) * param_5.v);
            g_stack[gl_LocalInvocationIndex][0] = (_1433 ? (_1009.Load(_1000.Load(_1447 * 4 + 0) * 4 + 0) & 65535u) : ((_1009.Load(_1000.Load(_1447 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u;
            g_stack[gl_LocalInvocationIndex][1] = 1065353216u;
            int stack_size = 1;
            float3 throughput = 0.0f.xxx;
            for (;;)
            {
                int _1674 = stack_size;
                stack_size = _1674 - 1;
                if (_1674 != 0)
                {
                    int _1690 = stack_size;
                    int _1691 = 2 * _1690;
                    material_t _1697;
                    [unroll]
                    for (int _43ident = 0; _43ident < 5; _43ident++)
                    {
                        _1697.textures[_43ident] = _1688.Load(_43ident * 4 + g_stack[gl_LocalInvocationIndex][_1691] * 76 + 0);
                    }
                    [unroll]
                    for (int _44ident = 0; _44ident < 3; _44ident++)
                    {
                        _1697.base_color[_44ident] = asfloat(_1688.Load(_44ident * 4 + g_stack[gl_LocalInvocationIndex][_1691] * 76 + 20));
                    }
                    _1697.flags = _1688.Load(g_stack[gl_LocalInvocationIndex][_1691] * 76 + 32);
                    _1697.type = _1688.Load(g_stack[gl_LocalInvocationIndex][_1691] * 76 + 36);
                    _1697.tangent_rotation_or_strength = asfloat(_1688.Load(g_stack[gl_LocalInvocationIndex][_1691] * 76 + 40));
                    _1697.roughness_and_anisotropic = _1688.Load(g_stack[gl_LocalInvocationIndex][_1691] * 76 + 44);
                    _1697.ior = asfloat(_1688.Load(g_stack[gl_LocalInvocationIndex][_1691] * 76 + 48));
                    _1697.sheen_and_sheen_tint = _1688.Load(g_stack[gl_LocalInvocationIndex][_1691] * 76 + 52);
                    _1697.tint_and_metallic = _1688.Load(g_stack[gl_LocalInvocationIndex][_1691] * 76 + 56);
                    _1697.transmission_and_transmission_roughness = _1688.Load(g_stack[gl_LocalInvocationIndex][_1691] * 76 + 60);
                    _1697.specular_and_specular_tint = _1688.Load(g_stack[gl_LocalInvocationIndex][_1691] * 76 + 64);
                    _1697.clearcoat_and_clearcoat_roughness = _1688.Load(g_stack[gl_LocalInvocationIndex][_1691] * 76 + 68);
                    _1697.normal_map_strength_unorm = _1688.Load(g_stack[gl_LocalInvocationIndex][_1691] * 76 + 72);
                    uint _1747 = g_stack[gl_LocalInvocationIndex][_1691 + 1];
                    float _1748 = asfloat(_1747);
                    if (_1697.type == 4u)
                    {
                        float mix_val = _1697.tangent_rotation_or_strength;
                        if (_1697.textures[1] != 4294967295u)
                        {
                            mix_val *= SampleBilinear(_1697.textures[1], _1660, 0).x;
                        }
                        int _1772 = 2 * stack_size;
                        g_stack[gl_LocalInvocationIndex][_1772] = _1697.textures[3];
                        g_stack[gl_LocalInvocationIndex][_1772 + 1] = asuint(_1748 * (1.0f - mix_val));
                        int _1791 = 2 * (stack_size + 1);
                        g_stack[gl_LocalInvocationIndex][_1791] = _1697.textures[4];
                        g_stack[gl_LocalInvocationIndex][_1791 + 1] = asuint(_1748 * mix_val);
                        stack_size += 2;
                    }
                    else
                    {
                        if (_1697.type == 5u)
                        {
                            throughput += (float3(_1697.base_color[0], _1697.base_color[1], _1697.base_color[2]) * _1748);
                        }
                    }
                    continue;
                }
                else
                {
                    break;
                }
            }
            float3 _1825 = rc;
            float3 _1826 = _1825 * throughput;
            rc = _1826;
            if (lum(_1826) < 1.0000000116860974230803549289703e-07f)
            {
                break;
            }
            float _1836 = _2455 + 9.9999997473787516355514526367188e-06f;
            ro += (_1348 * _1836);
            dist -= _1836;
            depth++;
        }
        if (_2262)
        {
            break;
        }
        _2262 = true;
        _2259 = rc;
        break;
    } while(false);
    return _2259;
}

float IntersectAreaLightsShadow(shadow_ray_t r)
{
    bool _2269 = false;
    float _2266;
    do
    {
        float3 _1857 = float3(r.o[0], r.o[1], r.o[2]);
        float3 _1865 = float3(r.d[0], r.d[1], r.d[2]);
        float _1869 = abs(r.dist);
        for (uint li = 0u; li < uint(_1394_g_params.blocker_lights_count); li++)
        {
            light_t _1901;
            _1901.type_and_param0 = _1897.Load4(_1885.Load(li * 4 + 0) * 64 + 0);
            _1901.param1 = asfloat(_1897.Load4(_1885.Load(li * 4 + 0) * 64 + 16));
            _1901.param2 = asfloat(_1897.Load4(_1885.Load(li * 4 + 0) * 64 + 32));
            _1901.param3 = asfloat(_1897.Load4(_1885.Load(li * 4 + 0) * 64 + 48));
            bool _1915 = (_1901.type_and_param0.x & 128u) != 0u;
            bool _1921;
            if (_1915)
            {
                _1921 = r.dist >= 0.0f;
            }
            else
            {
                _1921 = _1915;
            }
            [branch]
            if (_1921)
            {
                continue;
            }
            uint _1929 = _1901.type_and_param0.x & 31u;
            if (_1929 == 4u)
            {
                float3 light_u = _1901.param2.xyz;
                float3 light_v = _1901.param3.xyz;
                float3 _1950 = normalize(cross(_1901.param2.xyz, _1901.param3.xyz));
                float _1958 = dot(_1865, _1950);
                float _1966 = (dot(_1950, _1901.param1.xyz) - dot(_1950, _1857)) / _1958;
                if (((_1958 < 0.0f) && (_1966 > 9.9999999747524270787835121154785e-07f)) && (_1966 < _1869))
                {
                    float3 _1979 = light_u;
                    float3 _1984 = _1979 / dot(_1979, _1979).xxx;
                    light_u = _1984;
                    light_v /= dot(light_v, light_v).xxx;
                    float3 _2000 = (_1857 + (_1865 * _1966)) - _1901.param1.xyz;
                    float _2004 = dot(_1984, _2000);
                    if ((_2004 >= (-0.5f)) && (_2004 <= 0.5f))
                    {
                        float _2017 = dot(light_v, _2000);
                        if ((_2017 >= (-0.5f)) && (_2017 <= 0.5f))
                        {
                            _2269 = true;
                            _2266 = 0.0f;
                            break;
                        }
                    }
                }
            }
            else
            {
                if (_1929 == 5u)
                {
                    float3 light_u_1 = _1901.param2.xyz;
                    float3 light_v_1 = _1901.param3.xyz;
                    float3 _2047 = normalize(cross(_1901.param2.xyz, _1901.param3.xyz));
                    float _2055 = dot(_1865, _2047);
                    float _2063 = (dot(_2047, _1901.param1.xyz) - dot(_2047, _1857)) / _2055;
                    if (((_2055 < 0.0f) && (_2063 > 9.9999999747524270787835121154785e-07f)) && (_2063 < _1869))
                    {
                        float3 _2075 = light_u_1;
                        float3 _2080 = _2075 / dot(_2075, _2075).xxx;
                        light_u_1 = _2080;
                        float3 _2081 = light_v_1;
                        float3 _2086 = _2081 / dot(_2081, _2081).xxx;
                        light_v_1 = _2086;
                        float3 _2096 = (_1857 + (_1865 * _2063)) - _1901.param1.xyz;
                        float _2100 = dot(_2080, _2096);
                        float _2104 = dot(_2086, _2096);
                        if (sqrt(mad(_2100, _2100, _2104 * _2104)) <= 0.5f)
                        {
                            _2269 = true;
                            _2266 = 0.0f;
                            break;
                        }
                    }
                }
            }
        }
        if (_2269)
        {
            break;
        }
        _2269 = true;
        _2266 = 1.0f;
        break;
    } while(false);
    return _2266;
}

void comp_main()
{
    do
    {
        int _2130 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_2130) >= _2136.Load(12))
        {
            break;
        }
        shadow_ray_t _2155;
        [unroll]
        for (int _45ident = 0; _45ident < 3; _45ident++)
        {
            _2155.o[_45ident] = asfloat(_2151.Load(_45ident * 4 + _2130 * 48 + 0));
        }
        _2155.depth = int(_2151.Load(_2130 * 48 + 12));
        [unroll]
        for (int _46ident = 0; _46ident < 3; _46ident++)
        {
            _2155.d[_46ident] = asfloat(_2151.Load(_46ident * 4 + _2130 * 48 + 16));
        }
        _2155.dist = asfloat(_2151.Load(_2130 * 48 + 28));
        [unroll]
        for (int _47ident = 0; _47ident < 3; _47ident++)
        {
            _2155.c[_47ident] = asfloat(_2151.Load(_47ident * 4 + _2130 * 48 + 32));
        }
        _2155.xy = int(_2151.Load(_2130 * 48 + 44));
        float _2339[3] = { _2155.c[0], _2155.c[1], _2155.c[2] };
        float _2328[3] = { _2155.d[0], _2155.d[1], _2155.d[2] };
        float _2317[3] = { _2155.o[0], _2155.o[1], _2155.o[2] };
        shadow_ray_t _2303 = { _2317, _2155.depth, _2328, _2155.dist, _2339, _2155.xy };
        shadow_ray_t param = _2303;
        float3 _2189 = IntersectSceneShadow(param);
        shadow_ray_t param_1 = _2303;
        float3 _2193 = _2189 * IntersectAreaLightsShadow(param_1);
        if (lum(_2193) > 0.0f)
        {
            int2 _2217 = int2((_2155.xy >> 16) & 65535, _2155.xy & 65535);
            float4 _2218 = g_inout_img[_2217];
            float3 _2227 = _2218.xyz + min(_2193, _1394_g_params.clamp_val.xxx);
            float4 _2575 = _2218;
            _2575.x = _2227.x;
            float4 _2577 = _2575;
            _2577.y = _2227.y;
            float4 _2579 = _2577;
            _2579.z = _2227.z;
            g_inout_img[_2217] = _2579;
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
