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

ByteAddressBuffer _487 : register(t1, space0);
ByteAddressBuffer _707 : register(t5, space0);
ByteAddressBuffer _847 : register(t2, space0);
ByteAddressBuffer _856 : register(t3, space0);
ByteAddressBuffer _1029 : register(t7, space0);
ByteAddressBuffer _1033 : register(t8, space0);
ByteAddressBuffer _1053 : register(t6, space0);
ByteAddressBuffer _1099 : register(t9, space0);
ByteAddressBuffer _1336 : register(t10, space0);
ByteAddressBuffer _1340 : register(t11, space0);
ByteAddressBuffer _1542 : register(t4, space0);
ByteAddressBuffer _1745 : register(t15, space0);
ByteAddressBuffer _1757 : register(t14, space0);
ByteAddressBuffer _1982 : register(t13, space0);
ByteAddressBuffer _1997 : register(t12, space0);
cbuffer UniformParams
{
    Params _1246_g_params : packoffset(c0);
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

float3 safe_invert(float3 v)
{
    float3 inv_v = 1.0f.xxx / v;
    bool _130 = v.x <= 1.0000000116860974230803549289703e-07f;
    bool _137;
    if (_130)
    {
        _137 = v.x >= 0.0f;
    }
    else
    {
        _137 = _130;
    }
    if (_137)
    {
        float3 _2366 = inv_v;
        _2366.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _2366;
    }
    else
    {
        bool _146 = v.x >= (-1.0000000116860974230803549289703e-07f);
        bool _152;
        if (_146)
        {
            _152 = v.x < 0.0f;
        }
        else
        {
            _152 = _146;
        }
        if (_152)
        {
            float3 _2368 = inv_v;
            _2368.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _2368;
        }
    }
    bool _159 = v.y <= 1.0000000116860974230803549289703e-07f;
    bool _165;
    if (_159)
    {
        _165 = v.y >= 0.0f;
    }
    else
    {
        _165 = _159;
    }
    if (_165)
    {
        float3 _2370 = inv_v;
        _2370.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _2370;
    }
    else
    {
        bool _172 = v.y >= (-1.0000000116860974230803549289703e-07f);
        bool _178;
        if (_172)
        {
            _178 = v.y < 0.0f;
        }
        else
        {
            _178 = _172;
        }
        if (_178)
        {
            float3 _2372 = inv_v;
            _2372.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _2372;
        }
    }
    bool _184 = v.z <= 1.0000000116860974230803549289703e-07f;
    bool _190;
    if (_184)
    {
        _190 = v.z >= 0.0f;
    }
    else
    {
        _190 = _184;
    }
    if (_190)
    {
        float3 _2374 = inv_v;
        _2374.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _2374;
    }
    else
    {
        bool _197 = v.z >= (-1.0000000116860974230803549289703e-07f);
        bool _203;
        if (_197)
        {
            _203 = v.z < 0.0f;
        }
        else
        {
            _203 = _197;
        }
        if (_203)
        {
            float3 _2376 = inv_v;
            _2376.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _2376;
        }
    }
    return inv_v;
}

bool _bbox_test_fma(float3 inv_d, float3 neg_inv_d_o, float t, float3 bbox_min, float3 bbox_max)
{
    float _585 = mad(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float _593 = mad(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float _608 = mad(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    float _615 = mad(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    float _632 = mad(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    float _639 = mad(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    float _644 = max(max(min(_585, _593), min(_608, _615)), min(_632, _639));
    float _652 = min(min(max(_585, _593), max(_608, _615)), max(_632, _639)) * 1.0000002384185791015625f;
    return ((_644 <= _652) && (_644 <= t)) && (_652 > 0.0f);
}

void IntersectTri(float3 ro, float3 rd, tri_accel_t tri, uint prim_index, inout hit_data_t inter)
{
    do
    {
        float _357 = dot(rd, tri.n_plane.xyz);
        float _366 = tri.n_plane.w - dot(ro, tri.n_plane.xyz);
        if (sign(_366) != sign(mad(_357, inter.t, -_366)))
        {
            break;
        }
        float3 _387 = (ro * _357) + (rd * _366);
        float _398 = mad(_357, tri.u_plane.w, dot(_387, tri.u_plane.xyz));
        float _403 = _357 - _398;
        if (sign(_398) != sign(_403))
        {
            break;
        }
        float _420 = mad(_357, tri.v_plane.w, dot(_387, tri.v_plane.xyz));
        if (sign(_420) != sign(_403 - _420))
        {
            break;
        }
        float _435 = 1.0f / _357;
        inter.mask = -1;
        int _440;
        if (_357 < 0.0f)
        {
            _440 = int(prim_index);
        }
        else
        {
            _440 = (-1) - int(prim_index);
        }
        inter.prim_index = _440;
        inter.t = _366 * _435;
        inter.u = _398 * _435;
        inter.v = _420 * _435;
        break;
    } while(false);
}

bool IntersectTris_AnyHit(float3 ro, float3 rd, int tri_start, int tri_end, int obj_index, inout hit_data_t out_inter)
{
    int _2183 = 0;
    int _2184 = obj_index;
    float _2186 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _2185;
    float _2187;
    float _2188;
    for (int i = tri_start; i < tri_end; )
    {
        param = ro;
        param_1 = rd;
        tri_accel_t _498;
        _498.n_plane = asfloat(_487.Load4(i * 48 + 0));
        _498.u_plane = asfloat(_487.Load4(i * 48 + 16));
        _498.v_plane = asfloat(_487.Load4(i * 48 + 32));
        param_2.n_plane = _498.n_plane;
        param_2.u_plane = _498.u_plane;
        param_2.v_plane = _498.v_plane;
        param_3 = uint(i);
        hit_data_t _2195 = { _2183, _2184, _2185, _2186, _2187, _2188 };
        param_4 = _2195;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _2183 = param_4.mask;
        _2184 = param_4.obj_index;
        _2185 = param_4.prim_index;
        _2186 = param_4.t;
        _2187 = param_4.u;
        _2188 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _2183;
    int _521;
    if (_2183 != 0)
    {
        _521 = _2184;
    }
    else
    {
        _521 = out_inter.obj_index;
    }
    out_inter.obj_index = _521;
    int _534;
    if (_2183 != 0)
    {
        _534 = _2185;
    }
    else
    {
        _534 = out_inter.prim_index;
    }
    out_inter.prim_index = _534;
    out_inter.t = _2186;
    float _550;
    if (_2183 != 0)
    {
        _550 = _2187;
    }
    else
    {
        _550 = out_inter.u;
    }
    out_inter.u = _550;
    float _563;
    if (_2183 != 0)
    {
        _563 = _2188;
    }
    else
    {
        _563 = out_inter.v;
    }
    out_inter.v = _563;
    return _2183 != 0;
}

bool Traverse_MicroTree_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    bool _2124 = false;
    bool _2121;
    do
    {
        float3 _669 = (-inv_d) * ro;
        uint _671 = stack_size;
        uint _681 = stack_size;
        stack_size = _681 + uint(1);
        g_stack[gl_LocalInvocationIndex][_681] = node_index;
        uint _755;
        uint _779;
        int _831;
        while (stack_size != _671)
        {
            uint _696 = stack_size;
            uint _697 = _696 - uint(1);
            stack_size = _697;
            bvh_node_t _711;
            _711.bbox_min = asfloat(_707.Load4(g_stack[gl_LocalInvocationIndex][_697] * 32 + 0));
            _711.bbox_max = asfloat(_707.Load4(g_stack[gl_LocalInvocationIndex][_697] * 32 + 16));
            float3 param = inv_d;
            float3 param_1 = _669;
            float param_2 = inter.t;
            float3 param_3 = _711.bbox_min.xyz;
            float3 param_4 = _711.bbox_max.xyz;
            if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
            {
                continue;
            }
            uint _738 = asuint(_711.bbox_min.w);
            if ((_738 & 2147483648u) == 0u)
            {
                uint _745 = stack_size;
                stack_size = _745 + uint(1);
                uint _749 = asuint(_711.bbox_max.w);
                uint _751 = _749 >> uint(30);
                if (rd[_751] < 0.0f)
                {
                    _755 = _738;
                }
                else
                {
                    _755 = _749 & 1073741823u;
                }
                g_stack[gl_LocalInvocationIndex][_745] = _755;
                uint _770 = stack_size;
                stack_size = _770 + uint(1);
                if (rd[_751] < 0.0f)
                {
                    _779 = _749 & 1073741823u;
                }
                else
                {
                    _779 = _738;
                }
                g_stack[gl_LocalInvocationIndex][_770] = _779;
            }
            else
            {
                int _799 = int(_738 & 2147483647u);
                float3 param_5 = ro;
                float3 param_6 = rd;
                int param_7 = _799;
                int param_8 = _799 + asint(_711.bbox_max.w);
                int param_9 = obj_index;
                hit_data_t param_10 = inter;
                bool _820 = IntersectTris_AnyHit(param_5, param_6, param_7, param_8, param_9, param_10);
                inter = param_10;
                if (_820)
                {
                    bool _828 = inter.prim_index < 0;
                    if (_828)
                    {
                        _831 = (-1) - inter.prim_index;
                    }
                    else
                    {
                        _831 = inter.prim_index;
                    }
                    uint _842 = uint(_831);
                    bool _870 = !_828;
                    bool _877;
                    if (_870)
                    {
                        _877 = (((_856.Load(_847.Load(_842 * 4 + 0) * 4 + 0) >> 16u) & 65535u) & 32768u) != 0u;
                    }
                    else
                    {
                        _877 = _870;
                    }
                    bool _888;
                    if (!_877)
                    {
                        bool _887;
                        if (_828)
                        {
                            _887 = ((_856.Load(_847.Load(_842 * 4 + 0) * 4 + 0) & 65535u) & 32768u) != 0u;
                        }
                        else
                        {
                            _887 = _828;
                        }
                        _888 = _887;
                    }
                    else
                    {
                        _888 = _877;
                    }
                    if (_888)
                    {
                        _2124 = true;
                        _2121 = true;
                        break;
                    }
                }
            }
        }
        if (_2124)
        {
            break;
        }
        _2124 = true;
        _2121 = false;
        break;
    } while(false);
    return _2121;
}

bool Traverse_MacroTree_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    bool _2115 = false;
    bool _2112;
    do
    {
        float3 _899 = (-orig_inv_rd) * orig_ro;
        uint stack_size = 1u;
        g_stack[gl_LocalInvocationIndex][0u] = node_index;
        uint _964;
        uint _987;
        while (stack_size != 0u)
        {
            uint _915 = stack_size;
            uint _916 = _915 - uint(1);
            stack_size = _916;
            bvh_node_t _922;
            _922.bbox_min = asfloat(_707.Load4(g_stack[gl_LocalInvocationIndex][_916] * 32 + 0));
            _922.bbox_max = asfloat(_707.Load4(g_stack[gl_LocalInvocationIndex][_916] * 32 + 16));
            float3 param = orig_inv_rd;
            float3 param_1 = _899;
            float param_2 = inter.t;
            float3 param_3 = _922.bbox_min.xyz;
            float3 param_4 = _922.bbox_max.xyz;
            if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
            {
                continue;
            }
            uint _949 = asuint(_922.bbox_min.w);
            if ((_949 & 2147483648u) == 0u)
            {
                uint _955 = stack_size;
                stack_size = _955 + uint(1);
                uint _959 = asuint(_922.bbox_max.w);
                uint _960 = _959 >> uint(30);
                if (orig_rd[_960] < 0.0f)
                {
                    _964 = _949;
                }
                else
                {
                    _964 = _959 & 1073741823u;
                }
                g_stack[gl_LocalInvocationIndex][_955] = _964;
                uint _978 = stack_size;
                stack_size = _978 + uint(1);
                if (orig_rd[_960] < 0.0f)
                {
                    _987 = _959 & 1073741823u;
                }
                else
                {
                    _987 = _949;
                }
                g_stack[gl_LocalInvocationIndex][_978] = _987;
            }
            else
            {
                uint _1005 = _949 & 2147483647u;
                uint _1009 = asuint(_922.bbox_max.w);
                for (uint i = _1005; i < (_1005 + _1009); i++)
                {
                    mesh_instance_t _1039;
                    _1039.bbox_min = asfloat(_1029.Load4(_1033.Load(i * 4 + 0) * 32 + 0));
                    _1039.bbox_max = asfloat(_1029.Load4(_1033.Load(i * 4 + 0) * 32 + 16));
                    mesh_t _1059;
                    [unroll]
                    for (int _22ident = 0; _22ident < 3; _22ident++)
                    {
                        _1059.bbox_min[_22ident] = asfloat(_1053.Load(_22ident * 4 + asuint(_1039.bbox_max.w) * 48 + 0));
                    }
                    [unroll]
                    for (int _23ident = 0; _23ident < 3; _23ident++)
                    {
                        _1059.bbox_max[_23ident] = asfloat(_1053.Load(_23ident * 4 + asuint(_1039.bbox_max.w) * 48 + 12));
                    }
                    _1059.node_index = _1053.Load(asuint(_1039.bbox_max.w) * 48 + 24);
                    _1059.node_count = _1053.Load(asuint(_1039.bbox_max.w) * 48 + 28);
                    _1059.tris_index = _1053.Load(asuint(_1039.bbox_max.w) * 48 + 32);
                    _1059.tris_count = _1053.Load(asuint(_1039.bbox_max.w) * 48 + 36);
                    _1059.vert_index = _1053.Load(asuint(_1039.bbox_max.w) * 48 + 40);
                    _1059.vert_count = _1053.Load(asuint(_1039.bbox_max.w) * 48 + 44);
                    transform_t _1105;
                    _1105.xform = asfloat(uint4x4(_1099.Load4(asuint(_1039.bbox_min.w) * 128 + 0), _1099.Load4(asuint(_1039.bbox_min.w) * 128 + 16), _1099.Load4(asuint(_1039.bbox_min.w) * 128 + 32), _1099.Load4(asuint(_1039.bbox_min.w) * 128 + 48)));
                    _1105.inv_xform = asfloat(uint4x4(_1099.Load4(asuint(_1039.bbox_min.w) * 128 + 64), _1099.Load4(asuint(_1039.bbox_min.w) * 128 + 80), _1099.Load4(asuint(_1039.bbox_min.w) * 128 + 96), _1099.Load4(asuint(_1039.bbox_min.w) * 128 + 112)));
                    float3 param_5 = orig_inv_rd;
                    float3 param_6 = _899;
                    float param_7 = inter.t;
                    float3 param_8 = _1039.bbox_min.xyz;
                    float3 param_9 = _1039.bbox_max.xyz;
                    if (!_bbox_test_fma(param_5, param_6, param_7, param_8, param_9))
                    {
                        continue;
                    }
                    float3 _1150 = mul(float4(orig_rd, 0.0f), _1105.inv_xform).xyz;
                    float3 param_10 = _1150;
                    float3 param_11 = mul(float4(orig_ro, 1.0f), _1105.inv_xform).xyz;
                    float3 param_12 = _1150;
                    float3 param_13 = safe_invert(param_10);
                    int param_14 = int(_1033.Load(i * 4 + 0));
                    uint param_15 = _1059.node_index;
                    uint param_16 = stack_size;
                    hit_data_t param_17 = inter;
                    bool _1174 = Traverse_MicroTree_WithStack(param_11, param_12, param_13, param_14, param_15, param_16, param_17);
                    inter = param_17;
                    if (_1174)
                    {
                        _2115 = true;
                        _2112 = true;
                        break;
                    }
                }
                if (_2115)
                {
                    break;
                }
            }
        }
        if (_2115)
        {
            break;
        }
        _2115 = true;
        _2112 = false;
        break;
    } while(false);
    return _2112;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _258 = mad(col.z, 31.875f, 1.0f);
    float _268 = (col.x - 0.501960813999176025390625f) / _258;
    float _274 = (col.y - 0.501960813999176025390625f) / _258;
    return float3((col.w + _268) - _274, col.w + _274, (col.w - _268) - _274);
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
    uint _303 = index & 16777215u;
    float4 res = g_textures[NonUniformResourceIndex(_303)].SampleLevel(_g_textures_sampler[NonUniformResourceIndex(_303)], uvs, float(lod));
    bool _314;
    if (maybe_YCoCg)
    {
        _314 = (index & 67108864u) != 0u;
    }
    else
    {
        _314 = maybe_YCoCg;
    }
    if (_314)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _332;
    if (maybe_SRGB)
    {
        _332 = (index & 16777216u) != 0u;
    }
    else
    {
        _332 = maybe_SRGB;
    }
    if (_332)
    {
        float3 param_1 = res.xyz;
        float3 _338 = srgb_to_rgb(param_1);
        float4 _2392 = res;
        _2392.x = _338.x;
        float4 _2394 = _2392;
        _2394.y = _338.y;
        float4 _2396 = _2394;
        _2396.z = _338.z;
        res = _2396;
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
    bool _2101 = false;
    float3 _2098;
    do
    {
        float3 ro = float3(r.o[0], r.o[1], r.o[2]);
        float3 _1199 = float3(r.d[0], r.d[1], r.d[2]);
        float3 rc = float3(r.c[0], r.c[1], r.c[2]);
        int depth = r.depth >> 24;
        float _1217;
        if (r.dist > 0.0f)
        {
            _1217 = r.dist;
        }
        else
        {
            _1217 = 3402823346297367662189621542912.0f;
        }
        float dist = _1217;
        float3 param = _1199;
        float3 _1228 = safe_invert(param);
        int _1288;
        int _2245;
        int _2246;
        float _2248;
        float _2249;
        while (dist > 9.9999997473787516355514526367188e-06f)
        {
            int _2244 = 0;
            float _2247 = dist;
            float3 param_1 = ro;
            float3 param_2 = _1199;
            float3 param_3 = _1228;
            uint param_4 = _1246_g_params.node_index;
            hit_data_t _2256 = { 0, _2245, _2246, dist, _2248, _2249 };
            hit_data_t param_5 = _2256;
            bool _1259 = Traverse_MacroTree_WithStack(param_1, param_2, param_3, param_4, param_5);
            _2244 = param_5.mask;
            _2245 = param_5.obj_index;
            _2246 = param_5.prim_index;
            _2247 = param_5.t;
            _2248 = param_5.u;
            _2249 = param_5.v;
            bool _1270;
            if (!_1259)
            {
                _1270 = depth > _1246_g_params.max_transp_depth;
            }
            else
            {
                _1270 = _1259;
            }
            if (_1270)
            {
                _2101 = true;
                _2098 = 0.0f.xxx;
                break;
            }
            if (_2244 == 0)
            {
                _2101 = true;
                _2098 = rc;
                break;
            }
            bool _1285 = param_5.prim_index < 0;
            if (_1285)
            {
                _1288 = (-1) - param_5.prim_index;
            }
            else
            {
                _1288 = param_5.prim_index;
            }
            uint _1299 = uint(_1288);
            uint _1342 = _847.Load(_1299 * 4 + 0) * 3u;
            vertex_t _1348;
            [unroll]
            for (int _24ident = 0; _24ident < 3; _24ident++)
            {
                _1348.p[_24ident] = asfloat(_1336.Load(_24ident * 4 + _1340.Load(_1342 * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _25ident = 0; _25ident < 3; _25ident++)
            {
                _1348.n[_25ident] = asfloat(_1336.Load(_25ident * 4 + _1340.Load(_1342 * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _26ident = 0; _26ident < 3; _26ident++)
            {
                _1348.b[_26ident] = asfloat(_1336.Load(_26ident * 4 + _1340.Load(_1342 * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _27ident = 0; _27ident < 2; _27ident++)
            {
                [unroll]
                for (int _28ident = 0; _28ident < 2; _28ident++)
                {
                    _1348.t[_27ident][_28ident] = asfloat(_1336.Load(_28ident * 4 + _27ident * 8 + _1340.Load(_1342 * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1396;
            [unroll]
            for (int _29ident = 0; _29ident < 3; _29ident++)
            {
                _1396.p[_29ident] = asfloat(_1336.Load(_29ident * 4 + _1340.Load((_1342 + 1u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _30ident = 0; _30ident < 3; _30ident++)
            {
                _1396.n[_30ident] = asfloat(_1336.Load(_30ident * 4 + _1340.Load((_1342 + 1u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _31ident = 0; _31ident < 3; _31ident++)
            {
                _1396.b[_31ident] = asfloat(_1336.Load(_31ident * 4 + _1340.Load((_1342 + 1u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _32ident = 0; _32ident < 2; _32ident++)
            {
                [unroll]
                for (int _33ident = 0; _33ident < 2; _33ident++)
                {
                    _1396.t[_32ident][_33ident] = asfloat(_1336.Load(_33ident * 4 + _32ident * 8 + _1340.Load((_1342 + 1u) * 4 + 0) * 52 + 36));
                }
            }
            vertex_t _1442;
            [unroll]
            for (int _34ident = 0; _34ident < 3; _34ident++)
            {
                _1442.p[_34ident] = asfloat(_1336.Load(_34ident * 4 + _1340.Load((_1342 + 2u) * 4 + 0) * 52 + 0));
            }
            [unroll]
            for (int _35ident = 0; _35ident < 3; _35ident++)
            {
                _1442.n[_35ident] = asfloat(_1336.Load(_35ident * 4 + _1340.Load((_1342 + 2u) * 4 + 0) * 52 + 12));
            }
            [unroll]
            for (int _36ident = 0; _36ident < 3; _36ident++)
            {
                _1442.b[_36ident] = asfloat(_1336.Load(_36ident * 4 + _1340.Load((_1342 + 2u) * 4 + 0) * 52 + 24));
            }
            [unroll]
            for (int _37ident = 0; _37ident < 2; _37ident++)
            {
                [unroll]
                for (int _38ident = 0; _38ident < 2; _38ident++)
                {
                    _1442.t[_37ident][_38ident] = asfloat(_1336.Load(_38ident * 4 + _37ident * 8 + _1340.Load((_1342 + 2u) * 4 + 0) * 52 + 36));
                }
            }
            float2 _1514 = ((float2(_1348.t[0][0], _1348.t[0][1]) * ((1.0f - param_5.u) - param_5.v)) + (float2(_1396.t[0][0], _1396.t[0][1]) * param_5.u)) + (float2(_1442.t[0][0], _1442.t[0][1]) * param_5.v);
            g_stack[gl_LocalInvocationIndex][0] = (_1285 ? (_856.Load(_847.Load(_1299 * 4 + 0) * 4 + 0) & 65535u) : ((_856.Load(_847.Load(_1299 * 4 + 0) * 4 + 0) >> 16u) & 65535u)) & 16383u;
            g_stack[gl_LocalInvocationIndex][1] = 1065353216u;
            int stack_size = 1;
            float3 throughput = 0.0f.xxx;
            for (;;)
            {
                int _1528 = stack_size;
                stack_size = _1528 - 1;
                if (_1528 != 0)
                {
                    int _1544 = stack_size;
                    int _1545 = 2 * _1544;
                    material_t _1551;
                    [unroll]
                    for (int _39ident = 0; _39ident < 5; _39ident++)
                    {
                        _1551.textures[_39ident] = _1542.Load(_39ident * 4 + g_stack[gl_LocalInvocationIndex][_1545] * 76 + 0);
                    }
                    [unroll]
                    for (int _40ident = 0; _40ident < 3; _40ident++)
                    {
                        _1551.base_color[_40ident] = asfloat(_1542.Load(_40ident * 4 + g_stack[gl_LocalInvocationIndex][_1545] * 76 + 20));
                    }
                    _1551.flags = _1542.Load(g_stack[gl_LocalInvocationIndex][_1545] * 76 + 32);
                    _1551.type = _1542.Load(g_stack[gl_LocalInvocationIndex][_1545] * 76 + 36);
                    _1551.tangent_rotation_or_strength = asfloat(_1542.Load(g_stack[gl_LocalInvocationIndex][_1545] * 76 + 40));
                    _1551.roughness_and_anisotropic = _1542.Load(g_stack[gl_LocalInvocationIndex][_1545] * 76 + 44);
                    _1551.ior = asfloat(_1542.Load(g_stack[gl_LocalInvocationIndex][_1545] * 76 + 48));
                    _1551.sheen_and_sheen_tint = _1542.Load(g_stack[gl_LocalInvocationIndex][_1545] * 76 + 52);
                    _1551.tint_and_metallic = _1542.Load(g_stack[gl_LocalInvocationIndex][_1545] * 76 + 56);
                    _1551.transmission_and_transmission_roughness = _1542.Load(g_stack[gl_LocalInvocationIndex][_1545] * 76 + 60);
                    _1551.specular_and_specular_tint = _1542.Load(g_stack[gl_LocalInvocationIndex][_1545] * 76 + 64);
                    _1551.clearcoat_and_clearcoat_roughness = _1542.Load(g_stack[gl_LocalInvocationIndex][_1545] * 76 + 68);
                    _1551.normal_map_strength_unorm = _1542.Load(g_stack[gl_LocalInvocationIndex][_1545] * 76 + 72);
                    uint _1606 = g_stack[gl_LocalInvocationIndex][_1545 + 1];
                    float _1607 = asfloat(_1606);
                    if (_1551.type == 4u)
                    {
                        float mix_val = _1551.tangent_rotation_or_strength;
                        if (_1551.textures[1] != 4294967295u)
                        {
                            mix_val *= SampleBilinear(_1551.textures[1], _1514, 0).x;
                        }
                        int _1632 = 2 * stack_size;
                        g_stack[gl_LocalInvocationIndex][_1632] = _1551.textures[3];
                        g_stack[gl_LocalInvocationIndex][_1632 + 1] = asuint(_1607 * (1.0f - mix_val));
                        int _1651 = 2 * (stack_size + 1);
                        g_stack[gl_LocalInvocationIndex][_1651] = _1551.textures[4];
                        g_stack[gl_LocalInvocationIndex][_1651 + 1] = asuint(_1607 * mix_val);
                        stack_size += 2;
                    }
                    else
                    {
                        if (_1551.type == 5u)
                        {
                            throughput += (float3(_1551.base_color[0], _1551.base_color[1], _1551.base_color[2]) * _1607);
                        }
                    }
                    continue;
                }
                else
                {
                    break;
                }
            }
            float3 _1685 = rc;
            float3 _1686 = _1685 * throughput;
            rc = _1686;
            if (lum(_1686) < 1.0000000116860974230803549289703e-07f)
            {
                break;
            }
            float _1696 = _2247 + 9.9999997473787516355514526367188e-06f;
            ro += (_1199 * _1696);
            dist -= _1696;
            depth++;
        }
        if (_2101)
        {
            break;
        }
        _2101 = true;
        _2098 = rc;
        break;
    } while(false);
    return _2098;
}

float IntersectAreaLightsShadow(shadow_ray_t r)
{
    bool _2108 = false;
    float _2105;
    do
    {
        float3 _1717 = float3(r.o[0], r.o[1], r.o[2]);
        float3 _1725 = float3(r.d[0], r.d[1], r.d[2]);
        float _1729 = abs(r.dist);
        for (uint li = 0u; li < uint(_1246_g_params.blocker_lights_count); li++)
        {
            light_t _1761;
            _1761.type_and_param0 = _1757.Load4(_1745.Load(li * 4 + 0) * 64 + 0);
            _1761.param1 = asfloat(_1757.Load4(_1745.Load(li * 4 + 0) * 64 + 16));
            _1761.param2 = asfloat(_1757.Load4(_1745.Load(li * 4 + 0) * 64 + 32));
            _1761.param3 = asfloat(_1757.Load4(_1745.Load(li * 4 + 0) * 64 + 48));
            uint _1775 = _1761.type_and_param0.x & 31u;
            if (_1775 == 4u)
            {
                float3 light_u = _1761.param2.xyz;
                float3 light_v = _1761.param3.xyz;
                float3 _1796 = normalize(cross(_1761.param2.xyz, _1761.param3.xyz));
                float _1804 = dot(_1725, _1796);
                float _1812 = (dot(_1796, _1761.param1.xyz) - dot(_1796, _1717)) / _1804;
                if (((_1804 < 0.0f) && (_1812 > 9.9999999747524270787835121154785e-07f)) && (_1812 < _1729))
                {
                    float3 _1825 = light_u;
                    float3 _1830 = _1825 / dot(_1825, _1825).xxx;
                    light_u = _1830;
                    light_v /= dot(light_v, light_v).xxx;
                    float3 _1846 = (_1717 + (_1725 * _1812)) - _1761.param1.xyz;
                    float _1850 = dot(_1830, _1846);
                    if ((_1850 >= (-0.5f)) && (_1850 <= 0.5f))
                    {
                        float _1863 = dot(light_v, _1846);
                        if ((_1863 >= (-0.5f)) && (_1863 <= 0.5f))
                        {
                            _2108 = true;
                            _2105 = 0.0f;
                            break;
                        }
                    }
                }
            }
            else
            {
                if (_1775 == 5u)
                {
                    float3 light_u_1 = _1761.param2.xyz;
                    float3 light_v_1 = _1761.param3.xyz;
                    float3 _1893 = normalize(cross(_1761.param2.xyz, _1761.param3.xyz));
                    float _1901 = dot(_1725, _1893);
                    float _1909 = (dot(_1893, _1761.param1.xyz) - dot(_1893, _1717)) / _1901;
                    if (((_1901 < 0.0f) && (_1909 > 9.9999999747524270787835121154785e-07f)) && (_1909 < _1729))
                    {
                        float3 _1921 = light_u_1;
                        float3 _1926 = _1921 / dot(_1921, _1921).xxx;
                        light_u_1 = _1926;
                        float3 _1927 = light_v_1;
                        float3 _1932 = _1927 / dot(_1927, _1927).xxx;
                        light_v_1 = _1932;
                        float3 _1942 = (_1717 + (_1725 * _1909)) - _1761.param1.xyz;
                        float _1946 = dot(_1926, _1942);
                        float _1950 = dot(_1932, _1942);
                        if (sqrt(mad(_1946, _1946, _1950 * _1950)) <= 0.5f)
                        {
                            _2108 = true;
                            _2105 = 0.0f;
                            break;
                        }
                    }
                }
            }
        }
        if (_2108)
        {
            break;
        }
        _2108 = true;
        _2105 = 1.0f;
        break;
    } while(false);
    return _2105;
}

void comp_main()
{
    do
    {
        int _1976 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_1976) >= _1982.Load(12))
        {
            break;
        }
        shadow_ray_t _2001;
        [unroll]
        for (int _41ident = 0; _41ident < 3; _41ident++)
        {
            _2001.o[_41ident] = asfloat(_1997.Load(_41ident * 4 + _1976 * 48 + 0));
        }
        _2001.depth = int(_1997.Load(_1976 * 48 + 12));
        [unroll]
        for (int _42ident = 0; _42ident < 3; _42ident++)
        {
            _2001.d[_42ident] = asfloat(_1997.Load(_42ident * 4 + _1976 * 48 + 16));
        }
        _2001.dist = asfloat(_1997.Load(_1976 * 48 + 28));
        [unroll]
        for (int _43ident = 0; _43ident < 3; _43ident++)
        {
            _2001.c[_43ident] = asfloat(_1997.Load(_43ident * 4 + _1976 * 48 + 32));
        }
        _2001.xy = int(_1997.Load(_1976 * 48 + 44));
        float _2178[3] = { _2001.c[0], _2001.c[1], _2001.c[2] };
        float _2167[3] = { _2001.d[0], _2001.d[1], _2001.d[2] };
        float _2156[3] = { _2001.o[0], _2001.o[1], _2001.o[2] };
        shadow_ray_t _2142 = { _2156, _2001.depth, _2167, _2001.dist, _2178, _2001.xy };
        shadow_ray_t param = _2142;
        float3 _2035 = IntersectSceneShadow(param);
        shadow_ray_t param_1 = _2142;
        float3 _2039 = _2035 * IntersectAreaLightsShadow(param_1);
        if (lum(_2039) > 0.0f)
        {
            int2 _2064 = int2((_2001.xy >> 16) & 65535, _2001.xy & 65535);
            g_inout_img[_2064] = float4(g_inout_img[_2064].xyz + _2039, 1.0f);
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
