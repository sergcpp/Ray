struct atlas_texture_t
{
    uint size;
    uint atlas;
    uint page[4];
    uint pos[14];
};

struct lobe_weights_t
{
    float diffuse;
    float specular;
    float clearcoat;
    float refraction;
};

struct light_sample_t
{
    float3 col;
    float3 L;
    float3 lp;
    float area;
    float dist_mul;
    float pdf;
    bool cast_shadow;
    bool from_env;
};

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

struct hit_data_t
{
    int mask;
    int obj_index;
    int prim_index;
    float t;
    float u;
    float v;
};

struct surface_t
{
    float3 P;
    float3 T;
    float3 B;
    float3 N;
    float3 plane_N;
    float2 uvs;
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

struct diff_params_t
{
    float3 base_color;
    float3 sheen_color;
    float roughness;
};

struct spec_params_t
{
    float3 tmp_col;
    float roughness;
    float ior;
    float F0;
    float anisotropy;
};

struct clearcoat_params_t
{
    float roughness;
    float ior;
    float F0;
};

struct transmission_params_t
{
    float roughness;
    float int_ior;
    float eta;
    float fresnel;
    bool backfacing;
};

struct Params
{
    uint4 rect;
    float4 env_col;
    float4 back_col;
    int hi;
    int li_count;
    int max_diff_depth;
    int max_spec_depth;
    int max_refr_depth;
    int max_transp_depth;
    int max_total_depth;
    int min_total_depth;
    int min_transp_depth;
    int env_qtree_levels;
    float env_rotation;
    float back_rotation;
    int env_mult_importance;
    float clamp_val;
    float _pad0;
    float _pad1;
};

struct light_t
{
    uint4 type_and_param0;
    float4 param1;
    float4 param2;
    float4 param3;
};

struct transform_t
{
    row_major float4x4 xform;
    row_major float4x4 inv_xform;
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

struct mesh_instance_t
{
    float4 bbox_min;
    float4 bbox_max;
};

struct tri_accel_t
{
    float4 n_plane;
    float4 u_plane;
    float4 v_plane;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

ByteAddressBuffer _1003 : register(t20, space0);
ByteAddressBuffer _3523 : register(t17, space0);
ByteAddressBuffer _3558 : register(t8, space0);
ByteAddressBuffer _3562 : register(t9, space0);
ByteAddressBuffer _4405 : register(t13, space0);
ByteAddressBuffer _4430 : register(t15, space0);
ByteAddressBuffer _4434 : register(t16, space0);
ByteAddressBuffer _4758 : register(t12, space0);
ByteAddressBuffer _4762 : register(t11, space0);
ByteAddressBuffer _7153 : register(t14, space0);
RWByteAddressBuffer _8809 : register(u3, space0);
RWByteAddressBuffer _8820 : register(u1, space0);
RWByteAddressBuffer _8936 : register(u2, space0);
ByteAddressBuffer _9015 : register(t7, space0);
ByteAddressBuffer _9032 : register(t6, space0);
ByteAddressBuffer _9171 : register(t10, space0);
cbuffer UniformParams
{
    Params _3539_g_params : packoffset(c0);
};

Texture2DArray<float4> g_atlases[7] : register(t21, space0);
SamplerState _g_atlases_sampler[7] : register(s21, space0);
Texture2D<float4> g_env_qtree : register(t18, space0);
SamplerState _g_env_qtree_sampler : register(s18, space0);
RWTexture2D<float4> g_out_img : register(u0, space0);
RWTexture2D<float4> g_out_base_color_img : register(u4, space0);
RWTexture2D<float4> g_out_depth_normals_img : register(u5, space0);

static uint3 gl_WorkGroupID;
static uint gl_LocalInvocationIndex;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint gl_LocalInvocationIndex : SV_GroupIndex;
};

float2 TransformUV(float2 _uv, atlas_texture_t t, int mip_level)
{
    uint _9346[14] = t.pos;
    uint _9349[14] = t.pos;
    uint _1096 = t.size & 16383u;
    uint _1099 = t.size >> uint(16);
    uint _1100 = _1099 & 16383u;
    float2 size = float2(float(_1096), float(_1100));
    if ((_1099 & 32768u) != 0u)
    {
        size = float2(float(_1096 >> uint(mip_level)), float(_1100 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_9346[mip_level] & 65535u), float((_9349[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
}

float3 rgbe_to_rgb(float4 rgbe)
{
    return rgbe.xyz * exp2(mad(255.0f, rgbe.w, -128.0f));
}

float3 SampleLatlong_RGBE(atlas_texture_t t, float3 dir, float y_rotation)
{
    float _1268 = atan2(dir.z, dir.x) + y_rotation;
    float phi = _1268;
    if (_1268 < 0.0f)
    {
        phi += 6.283185482025146484375f;
    }
    if (phi > 6.283185482025146484375f)
    {
        phi -= 6.283185482025146484375f;
    }
    float2 _1290 = TransformUV(float2(frac(phi * 0.15915493667125701904296875f), acos(clamp(dir.y, -1.0f, 1.0f)) * 0.3183098733425140380859375f), t, 0);
    uint _1297 = t.atlas;
    int3 _1306 = int3(int2(_1290), int(t.page[0] & 255u));
    float2 _1353 = frac(_1290);
    float4 param = g_atlases[NonUniformResourceIndex(_1297)].Load(int4(_1306, 0), int2(0, 0));
    float4 param_1 = g_atlases[NonUniformResourceIndex(_1297)].Load(int4(_1306, 0), int2(1, 0));
    float4 param_2 = g_atlases[NonUniformResourceIndex(_1297)].Load(int4(_1306, 0), int2(0, 1));
    float4 param_3 = g_atlases[NonUniformResourceIndex(_1297)].Load(int4(_1306, 0), int2(1, 1));
    float _1373 = _1353.x;
    float _1378 = 1.0f - _1373;
    float _1394 = _1353.y;
    return (((rgbe_to_rgb(param_3) * _1373) + (rgbe_to_rgb(param_2) * _1378)) * _1394) + (((rgbe_to_rgb(param_1) * _1373) + (rgbe_to_rgb(param) * _1378)) * (1.0f - _1394));
}

float2 DirToCanonical(float3 d, float y_rotation)
{
    float _732 = (-atan2(d.z, d.x)) + y_rotation;
    float phi = _732;
    if (_732 < 0.0f)
    {
        phi += 6.283185482025146484375f;
    }
    if (phi > 6.283185482025146484375f)
    {
        phi -= 6.283185482025146484375f;
    }
    return float2((clamp(d.y, -1.0f, 1.0f) + 1.0f) * 0.5f, phi * 0.15915493667125701904296875f);
}

float Evaluate_EnvQTree(float y_rotation, Texture2D<float4> qtree_tex, SamplerState _qtree_tex_sampler, int qtree_levels, float3 L)
{
    int res = 2;
    int lod = qtree_levels - 1;
    float2 _759 = DirToCanonical(L, -y_rotation);
    float factor = 1.0f;
    while (lod >= 0)
    {
        int2 _779 = clamp(int2(_759 * float(res)), int2(0, 0), (res - 1).xx);
        float4 quad = qtree_tex.Load(int3(_779 / int2(2, 2), lod));
        float _814 = ((quad.x + quad.y) + quad.z) + quad.w;
        if (_814 <= 0.0f)
        {
            break;
        }
        factor *= ((4.0f * quad[(0 | ((_779.x & 1) << 0)) | ((_779.y & 1) << 1)]) / _814);
        lod--;
        res *= 2;
    }
    return factor * 0.079577468335628509521484375f;
}

float power_heuristic(float a, float b)
{
    float _1407 = a * a;
    return _1407 / mad(b, b, _1407);
}

float3 Evaluate_EnvColor(ray_data_t ray)
{
    float3 _5012 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 env_col = _3539_g_params.back_col.xyz;
    uint _5020 = asuint(_3539_g_params.back_col.w);
    if (_5020 != 4294967295u)
    {
        atlas_texture_t _5031;
        _5031.size = _1003.Load(_5020 * 80 + 0);
        _5031.atlas = _1003.Load(_5020 * 80 + 4);
        [unroll]
        for (int _58ident = 0; _58ident < 4; _58ident++)
        {
            _5031.page[_58ident] = _1003.Load(_58ident * 4 + _5020 * 80 + 8);
        }
        [unroll]
        for (int _59ident = 0; _59ident < 14; _59ident++)
        {
            _5031.pos[_59ident] = _1003.Load(_59ident * 4 + _5020 * 80 + 24);
        }
        uint _9716[14] = { _5031.pos[0], _5031.pos[1], _5031.pos[2], _5031.pos[3], _5031.pos[4], _5031.pos[5], _5031.pos[6], _5031.pos[7], _5031.pos[8], _5031.pos[9], _5031.pos[10], _5031.pos[11], _5031.pos[12], _5031.pos[13] };
        uint _9687[4] = { _5031.page[0], _5031.page[1], _5031.page[2], _5031.page[3] };
        atlas_texture_t _9678 = { _5031.size, _5031.atlas, _9687, _9716 };
        float param = _3539_g_params.back_rotation;
        env_col *= SampleLatlong_RGBE(_9678, _5012, param);
    }
    if (_3539_g_params.env_qtree_levels > 0)
    {
        float param_1 = ray.pdf;
        float param_2 = Evaluate_EnvQTree(_3539_g_params.back_rotation, g_env_qtree, _g_env_qtree_sampler, _3539_g_params.env_qtree_levels, _5012);
        env_col *= power_heuristic(param_1, param_2);
    }
    else
    {
        if (_3539_g_params.env_mult_importance != 0)
        {
            float param_3 = ray.pdf;
            float param_4 = 0.15915493667125701904296875f;
            env_col *= power_heuristic(param_3, param_4);
        }
    }
    return env_col;
}

float3 Evaluate_LightColor(ray_data_t ray, hit_data_t inter)
{
    float3 _5143 = float3(ray.d[0], ray.d[1], ray.d[2]);
    light_t _5157;
    _5157.type_and_param0 = _3558.Load4(((-1) - inter.obj_index) * 64 + 0);
    _5157.param1 = asfloat(_3558.Load4(((-1) - inter.obj_index) * 64 + 16));
    _5157.param2 = asfloat(_3558.Load4(((-1) - inter.obj_index) * 64 + 32));
    _5157.param3 = asfloat(_3558.Load4(((-1) - inter.obj_index) * 64 + 48));
    float3 lcol = asfloat(_5157.type_and_param0.yzw);
    [branch]
    if ((_5157.type_and_param0.x & 128u) != 0u)
    {
        float3 env_col = _3539_g_params.env_col.xyz;
        uint _5184 = asuint(_3539_g_params.env_col.w);
        if (_5184 != 4294967295u)
        {
            atlas_texture_t _5191;
            _5191.size = _1003.Load(_5184 * 80 + 0);
            _5191.atlas = _1003.Load(_5184 * 80 + 4);
            [unroll]
            for (int _60ident = 0; _60ident < 4; _60ident++)
            {
                _5191.page[_60ident] = _1003.Load(_60ident * 4 + _5184 * 80 + 8);
            }
            [unroll]
            for (int _61ident = 0; _61ident < 14; _61ident++)
            {
                _5191.pos[_61ident] = _1003.Load(_61ident * 4 + _5184 * 80 + 24);
            }
            uint _9778[14] = { _5191.pos[0], _5191.pos[1], _5191.pos[2], _5191.pos[3], _5191.pos[4], _5191.pos[5], _5191.pos[6], _5191.pos[7], _5191.pos[8], _5191.pos[9], _5191.pos[10], _5191.pos[11], _5191.pos[12], _5191.pos[13] };
            uint _9749[4] = { _5191.page[0], _5191.page[1], _5191.page[2], _5191.page[3] };
            atlas_texture_t _9740 = { _5191.size, _5191.atlas, _9749, _9778 };
            float param = _3539_g_params.env_rotation;
            env_col *= SampleLatlong_RGBE(_9740, _5143, param);
        }
        lcol *= env_col;
    }
    uint _5251 = _5157.type_and_param0.x & 31u;
    if (_5251 == 0u)
    {
        float param_1 = ray.pdf;
        float param_2 = (inter.t * inter.t) / ((0.5f * _5157.param1.w) * dot(_5143, normalize(_5157.param1.xyz - (float3(ray.o[0], ray.o[1], ray.o[2]) + (_5143 * inter.t)))));
        lcol *= power_heuristic(param_1, param_2);
        bool _5318 = _5157.param3.x > 0.0f;
        bool _5324;
        if (_5318)
        {
            _5324 = _5157.param3.y > 0.0f;
        }
        else
        {
            _5324 = _5318;
        }
        [branch]
        if (_5324)
        {
            [flatten]
            if (_5157.param3.y > 0.0f)
            {
                lcol *= clamp((_5157.param3.x - acos(clamp(-dot(_5143, _5157.param2.xyz), 0.0f, 1.0f))) / _5157.param3.y, 0.0f, 1.0f);
            }
        }
    }
    else
    {
        if (_5251 == 4u)
        {
            float param_3 = ray.pdf;
            float param_4 = (inter.t * inter.t) / (_5157.param1.w * dot(_5143, normalize(cross(_5157.param2.xyz, _5157.param3.xyz))));
            lcol *= power_heuristic(param_3, param_4);
        }
        else
        {
            if (_5251 == 5u)
            {
                float param_5 = ray.pdf;
                float param_6 = (inter.t * inter.t) / (_5157.param1.w * dot(_5143, normalize(cross(_5157.param2.xyz, _5157.param3.xyz))));
                lcol *= power_heuristic(param_5, param_6);
            }
            else
            {
                if (_5251 == 3u)
                {
                    float param_7 = ray.pdf;
                    float param_8 = (inter.t * inter.t) / (_5157.param1.w * (1.0f - abs(dot(_5143, _5157.param3.xyz))));
                    lcol *= power_heuristic(param_7, param_8);
                }
            }
        }
    }
    return lcol;
}

float3 TransformNormal(float3 n, float4x4 inv_xform)
{
    return mul(float4(n, 0.0f), transpose(inv_xform)).xyz;
}

int hash(int x)
{
    uint _509 = uint(x);
    uint _516 = ((_509 >> uint(16)) ^ _509) * 73244475u;
    uint _521 = ((_516 >> uint(16)) ^ _516) * 73244475u;
    return int((_521 >> uint(16)) ^ _521);
}

float construct_float(inout uint m)
{
    m &= 8388607u;
    m |= 1065353216u;
    return asfloat(m) - 1.0f;
}

bool exchange(inout bool old_value, bool new_value)
{
    bool _2314 = old_value;
    old_value = new_value;
    return _2314;
}

float peek_ior_stack(float stack[4], inout bool skip_first, float default_value)
{
    float _9183;
    do
    {
        bool _2398 = stack[3] > 0.0f;
        bool _2407;
        if (_2398)
        {
            bool param = skip_first;
            bool param_1 = false;
            bool _2404 = exchange(param, param_1);
            skip_first = param;
            _2407 = !_2404;
        }
        else
        {
            _2407 = _2398;
        }
        if (_2407)
        {
            _9183 = stack[3];
            break;
        }
        bool _2415 = stack[2] > 0.0f;
        bool _2424;
        if (_2415)
        {
            bool param_2 = skip_first;
            bool param_3 = false;
            bool _2421 = exchange(param_2, param_3);
            skip_first = param_2;
            _2424 = !_2421;
        }
        else
        {
            _2424 = _2415;
        }
        if (_2424)
        {
            _9183 = stack[2];
            break;
        }
        bool _2432 = stack[1] > 0.0f;
        bool _2441;
        if (_2432)
        {
            bool param_4 = skip_first;
            bool param_5 = false;
            bool _2438 = exchange(param_4, param_5);
            skip_first = param_4;
            _2441 = !_2438;
        }
        else
        {
            _2441 = _2432;
        }
        if (_2441)
        {
            _9183 = stack[1];
            break;
        }
        bool _2449 = stack[0] > 0.0f;
        bool _2458;
        if (_2449)
        {
            bool param_6 = skip_first;
            bool param_7 = false;
            bool _2455 = exchange(param_6, param_7);
            skip_first = param_6;
            _2458 = !_2455;
        }
        else
        {
            _2458 = _2449;
        }
        if (_2458)
        {
            _9183 = stack[0];
            break;
        }
        _9183 = default_value;
        break;
    } while(false);
    return _9183;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _608 = mad(col.z, 31.875f, 1.0f);
    float _618 = (col.x - 0.501960813999176025390625f) / _608;
    float _624 = (col.y - 0.501960813999176025390625f) / _608;
    return float3((col.w + _618) - _624, col.w + _624, (col.w - _618) - _624);
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
    atlas_texture_t _1133;
    _1133.size = _1003.Load(index * 80 + 0);
    _1133.atlas = _1003.Load(index * 80 + 4);
    [unroll]
    for (int _62ident = 0; _62ident < 4; _62ident++)
    {
        _1133.page[_62ident] = _1003.Load(_62ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _63ident = 0; _63ident < 14; _63ident++)
    {
        _1133.pos[_63ident] = _1003.Load(_63ident * 4 + index * 80 + 24);
    }
    uint _9354[4];
    _9354[0] = _1133.page[0];
    _9354[1] = _1133.page[1];
    _9354[2] = _1133.page[2];
    _9354[3] = _1133.page[3];
    uint _9390[14] = { _1133.pos[0], _1133.pos[1], _1133.pos[2], _1133.pos[3], _1133.pos[4], _1133.pos[5], _1133.pos[6], _1133.pos[7], _1133.pos[8], _1133.pos[9], _1133.pos[10], _1133.pos[11], _1133.pos[12], _1133.pos[13] };
    atlas_texture_t _9360 = { _1133.size, _1133.atlas, _9354, _9390 };
    uint _1203 = _1133.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_1203)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_1203)], float3(TransformUV(uvs, _9360, lod) * 0.000118371215648949146270751953125f.xx, float((_9354[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
    bool _1218;
    if (maybe_YCoCg)
    {
        _1218 = _1133.atlas == 4u;
    }
    else
    {
        _1218 = maybe_YCoCg;
    }
    if (_1218)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _1237;
    if (maybe_SRGB)
    {
        _1237 = (_1133.size & 32768u) != 0u;
    }
    else
    {
        _1237 = maybe_SRGB;
    }
    if (_1237)
    {
        float3 param_1 = res.xyz;
        float3 _1243 = srgb_to_rgb(param_1);
        float4 _10533 = res;
        _10533.x = _1243.x;
        float4 _10535 = _10533;
        _10535.y = _1243.y;
        float4 _10537 = _10535;
        _10537.z = _1243.z;
        res = _10537;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod)
{
    return SampleBilinear(index, uvs, lod, false, false);
}

float fresnel_dielectric_cos(float cosi, float eta)
{
    float _1439 = abs(cosi);
    float _1448 = mad(_1439, _1439, mad(eta, eta, -1.0f));
    float g = _1448;
    float result;
    if (_1448 > 0.0f)
    {
        float _1453 = g;
        float _1454 = sqrt(_1453);
        g = _1454;
        float _1458 = _1454 - _1439;
        float _1461 = _1454 + _1439;
        float _1462 = _1458 / _1461;
        float _1476 = mad(_1439, _1461, -1.0f) / mad(_1439, _1458, 1.0f);
        result = ((0.5f * _1462) * _1462) * mad(_1476, _1476, 1.0f);
    }
    else
    {
        result = 1.0f;
    }
    return result;
}

float safe_sqrtf(float f)
{
    return sqrt(max(f, 0.0f));
}

float3 ensure_valid_reflection(float3 Ng, float3 I, float3 N)
{
    float3 _9188;
    do
    {
        float _1512 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1512)
        {
            _9188 = N;
            break;
        }
        float3 _1532 = normalize(N - (Ng * dot(N, Ng)));
        float _1536 = dot(I, _1532);
        float _1540 = dot(I, Ng);
        float _1552 = mad(_1536, _1536, _1540 * _1540);
        float param = (_1536 * _1536) * mad(-_1512, _1512, _1552);
        float _1562 = safe_sqrtf(param);
        float _1568 = mad(_1540, _1512, _1552);
        float _1571 = 0.5f / _1552;
        float _1576 = _1562 + _1568;
        float _1577 = _1571 * _1576;
        float _1583 = (-_1562) + _1568;
        float _1584 = _1571 * _1583;
        bool _1592 = (_1577 > 9.9999997473787516355514526367188e-06f) && (_1577 <= 1.000010013580322265625f);
        bool valid1 = _1592;
        bool _1598 = (_1584 > 9.9999997473787516355514526367188e-06f) && (_1584 <= 1.000010013580322265625f);
        bool valid2 = _1598;
        float2 N_new;
        if (_1592 && _1598)
        {
            float _10836 = (-0.5f) / _1552;
            float param_1 = mad(_10836, _1576, 1.0f);
            float _1608 = safe_sqrtf(param_1);
            float param_2 = _1577;
            float _1611 = safe_sqrtf(param_2);
            float2 _1612 = float2(_1608, _1611);
            float param_3 = mad(_10836, _1583, 1.0f);
            float _1617 = safe_sqrtf(param_3);
            float param_4 = _1584;
            float _1620 = safe_sqrtf(param_4);
            float2 _1621 = float2(_1617, _1620);
            float _10838 = -_1540;
            float _1637 = mad(2.0f * mad(_1608, _1536, _1611 * _1540), _1611, _10838);
            float _1653 = mad(2.0f * mad(_1617, _1536, _1620 * _1540), _1620, _10838);
            bool _1655 = _1637 >= 9.9999997473787516355514526367188e-06f;
            valid1 = _1655;
            bool _1657 = _1653 >= 9.9999997473787516355514526367188e-06f;
            valid2 = _1657;
            if (_1655 && _1657)
            {
                bool2 _1670 = (_1637 < _1653).xx;
                N_new = float2(_1670.x ? _1612.x : _1621.x, _1670.y ? _1612.y : _1621.y);
            }
            else
            {
                bool2 _1678 = (_1637 > _1653).xx;
                N_new = float2(_1678.x ? _1612.x : _1621.x, _1678.y ? _1612.y : _1621.y);
            }
        }
        else
        {
            if (!(valid1 || valid2))
            {
                _9188 = Ng;
                break;
            }
            float _1690 = valid1 ? _1577 : _1584;
            float param_5 = 1.0f - _1690;
            float param_6 = _1690;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _9188 = (_1532 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _9188;
}

float3 rotate_around_axis(float3 p, float3 axis, float angle)
{
    float _1784 = cos(angle);
    float _1787 = sin(angle);
    float _1791 = 1.0f - _1784;
    return float3(mad(mad(_1791 * axis.x, axis.z, axis.y * _1787), p.z, mad(mad(_1791 * axis.x, axis.x, _1784), p.x, mad(_1791 * axis.x, axis.y, -(axis.z * _1787)) * p.y)), mad(mad(_1791 * axis.y, axis.z, -(axis.x * _1787)), p.z, mad(mad(_1791 * axis.x, axis.y, axis.z * _1787), p.x, mad(_1791 * axis.y, axis.y, _1784) * p.y)), mad(mad(_1791 * axis.z, axis.z, _1784), p.z, mad(mad(_1791 * axis.x, axis.z, -(axis.y * _1787)), p.x, mad(_1791 * axis.y, axis.z, axis.x * _1787) * p.y)));
}

void create_tbn(float3 N, inout float3 out_T, out float3 out_B)
{
    float3 U;
    [flatten]
    if (abs(N.y) < 0.999000012874603271484375f)
    {
        U = float3(0.0f, 1.0f, 0.0f);
    }
    else
    {
        U = float3(1.0f, 0.0f, 0.0f);
    }
    out_T = normalize(cross(U, N));
    out_B = cross(N, out_T);
}

float3 offset_ray(float3 p, float3 n)
{
    int3 _1940 = int3(n * 128.0f);
    int _1948;
    if (p.x < 0.0f)
    {
        _1948 = -_1940.x;
    }
    else
    {
        _1948 = _1940.x;
    }
    int _1966;
    if (p.y < 0.0f)
    {
        _1966 = -_1940.y;
    }
    else
    {
        _1966 = _1940.y;
    }
    int _1984;
    if (p.z < 0.0f)
    {
        _1984 = -_1940.z;
    }
    else
    {
        _1984 = _1940.z;
    }
    float _2002;
    if (abs(p.x) < 0.03125f)
    {
        _2002 = mad(1.52587890625e-05f, n.x, p.x);
    }
    else
    {
        _2002 = asfloat(asint(p.x) + _1948);
    }
    float _2020;
    if (abs(p.y) < 0.03125f)
    {
        _2020 = mad(1.52587890625e-05f, n.y, p.y);
    }
    else
    {
        _2020 = asfloat(asint(p.y) + _1966);
    }
    float _2037;
    if (abs(p.z) < 0.03125f)
    {
        _2037 = mad(1.52587890625e-05f, n.z, p.z);
    }
    else
    {
        _2037 = asfloat(asint(p.z) + _1984);
    }
    return float3(_2002, _2020, _2037);
}

float3 MapToCone(float r1, float r2, float3 N, float radius)
{
    float3 _9213;
    do
    {
        float2 _3438 = (float2(r1, r2) * 2.0f) - 1.0f.xx;
        float _3440 = _3438.x;
        bool _3441 = _3440 == 0.0f;
        bool _3447;
        if (_3441)
        {
            _3447 = _3438.y == 0.0f;
        }
        else
        {
            _3447 = _3441;
        }
        if (_3447)
        {
            _9213 = N;
            break;
        }
        float _3456 = _3438.y;
        float r;
        float theta;
        if (abs(_3440) > abs(_3456))
        {
            r = _3440;
            theta = 0.785398185253143310546875f * (_3456 / _3440);
        }
        else
        {
            r = _3456;
            theta = 1.57079637050628662109375f * mad(-0.5f, _3440 / _3456, 1.0f);
        }
        float3 param;
        float3 param_1;
        create_tbn(N, param, param_1);
        _9213 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _9213;
}

float3 CanonicalToDir(float2 p, float y_rotation)
{
    float _682 = mad(2.0f, p.x, -1.0f);
    float _687 = mad(6.283185482025146484375f, p.y, y_rotation);
    float phi = _687;
    if (_687 < 0.0f)
    {
        phi += 6.283185482025146484375f;
    }
    if (phi > 6.283185482025146484375f)
    {
        phi -= 6.283185482025146484375f;
    }
    float _705 = sqrt(mad(-_682, _682, 1.0f));
    return float3(_705 * cos(phi), _682, (-_705) * sin(phi));
}

float4 Sample_EnvQTree(float y_rotation, Texture2D<float4> qtree_tex, SamplerState _qtree_tex_sampler, int qtree_levels, float rand, float rx, float ry)
{
    int res = 2;
    float _step = 0.5f;
    float _sample = rand;
    int lod = qtree_levels - 1;
    float2 origin = 0.0f.xx;
    float factor = 1.0f;
    while (lod >= 0)
    {
        float4 quad = qtree_tex.Load(int3(int2(origin * float(res)) / int2(2, 2), lod));
        float _880 = quad.x + quad.z;
        float partial = _880;
        float _887 = (_880 + quad.y) + quad.w;
        if (_887 <= 0.0f)
        {
            break;
        }
        float _896 = partial / _887;
        float boundary = _896;
        int index = 0;
        if (_sample < _896)
        {
            _sample /= boundary;
            boundary = quad.x / partial;
        }
        else
        {
            float _911 = partial;
            float _912 = _887 - _911;
            partial = _912;
            float2 _10520 = origin;
            _10520.x = origin.x + _step;
            origin = _10520;
            _sample = (_sample - boundary) / (1.0f - boundary);
            boundary = quad.y / _912;
            index |= 1;
        }
        if (_sample < boundary)
        {
            _sample /= boundary;
        }
        else
        {
            float2 _10523 = origin;
            _10523.y = origin.y + _step;
            origin = _10523;
            _sample = (_sample - boundary) / (1.0f - boundary);
            index |= 2;
        }
        factor *= ((4.0f * quad[index]) / _887);
        lod--;
        res *= 2;
        _step *= 0.5f;
    }
    float2 _969 = origin;
    float2 _970 = _969 + (float2(rx, ry) * (2.0f * _step));
    origin = _970;
    return float4(CanonicalToDir(_970, y_rotation), factor * 0.079577468335628509521484375f);
}

float3 world_from_tangent(float3 T, float3 B, float3 N, float3 V)
{
    return ((T * V.x) + (B * V.y)) + (N * V.z);
}

void SampleLightSource(float3 P, float3 T, float3 B, float3 N, int hi, float2 sample_off, inout light_sample_t ls)
{
    float _3532 = frac(asfloat(_3523.Load((hi + 3) * 4 + 0)) + sample_off.x);
    float _3543 = float(_3539_g_params.li_count);
    uint _3550 = min(uint(_3532 * _3543), uint(_3539_g_params.li_count - 1));
    light_t _3569;
    _3569.type_and_param0 = _3558.Load4(_3562.Load(_3550 * 4 + 0) * 64 + 0);
    _3569.param1 = asfloat(_3558.Load4(_3562.Load(_3550 * 4 + 0) * 64 + 16));
    _3569.param2 = asfloat(_3558.Load4(_3562.Load(_3550 * 4 + 0) * 64 + 32));
    _3569.param3 = asfloat(_3558.Load4(_3562.Load(_3550 * 4 + 0) * 64 + 48));
    ls.col = asfloat(_3569.type_and_param0.yzw);
    ls.col *= _3543;
    ls.cast_shadow = (_3569.type_and_param0.x & 32u) != 0u;
    ls.from_env = false;
    uint _3603 = _3569.type_and_param0.x & 31u;
    [branch]
    if (_3603 == 0u)
    {
        float _3616 = frac(asfloat(_3523.Load((hi + 4) * 4 + 0)) + sample_off.x);
        float3 _3631 = P - _3569.param1.xyz;
        float3 _3638 = _3631 / length(_3631).xxx;
        float _3645 = sqrt(clamp(mad(-_3616, _3616, 1.0f), 0.0f, 1.0f));
        float _3648 = 6.283185482025146484375f * frac(asfloat(_3523.Load((hi + 5) * 4 + 0)) + sample_off.y);
        float3 sampled_dir = float3(_3645 * cos(_3648), _3645 * sin(_3648), _3616);
        float3 param;
        float3 param_1;
        create_tbn(_3638, param, param_1);
        float3 _10600 = sampled_dir;
        float3 _3681 = ((param * _10600.x) + (param_1 * _10600.y)) + (_3638 * _10600.z);
        sampled_dir = _3681;
        float3 _3690 = _3569.param1.xyz + (_3681 * _3569.param2.w);
        float3 _3697 = normalize(_3690 - _3569.param1.xyz);
        float3 param_2 = _3690;
        float3 param_3 = _3697;
        ls.lp = offset_ray(param_2, param_3);
        ls.L = _3690 - P;
        float3 _3710 = ls.L;
        float _3711 = length(_3710);
        ls.L /= _3711.xxx;
        ls.area = _3569.param1.w;
        float _3726 = abs(dot(ls.L, _3697));
        [flatten]
        if (_3726 > 0.0f)
        {
            ls.pdf = (_3711 * _3711) / ((0.5f * ls.area) * _3726);
        }
        [branch]
        if (_3569.param3.x > 0.0f)
        {
            float _3753 = -dot(ls.L, _3569.param2.xyz);
            if (_3753 > 0.0f)
            {
                ls.col *= clamp((_3569.param3.x - acos(clamp(_3753, 0.0f, 1.0f))) / _3569.param3.y, 0.0f, 1.0f);
            }
            else
            {
                ls.col = 0.0f.xxx;
            }
        }
    }
    else
    {
        [branch]
        if (_3603 == 2u)
        {
            ls.L = _3569.param1.xyz;
            if (_3569.param1.w != 0.0f)
            {
                float param_4 = frac(asfloat(_3523.Load((hi + 4) * 4 + 0)) + sample_off.x);
                float param_5 = frac(asfloat(_3523.Load((hi + 5) * 4 + 0)) + sample_off.y);
                float3 param_6 = ls.L;
                float param_7 = tan(_3569.param1.w);
                ls.L = normalize(MapToCone(param_4, param_5, param_6, param_7));
            }
            ls.area = 0.0f;
            ls.lp = P + ls.L;
            ls.dist_mul = 3402823346297367662189621542912.0f;
            ls.pdf = 1.0f;
            if ((_3569.type_and_param0.x & 64u) == 0u)
            {
                ls.area = 0.0f;
            }
        }
        else
        {
            [branch]
            if (_3603 == 4u)
            {
                float3 _3890 = (_3569.param1.xyz + (_3569.param2.xyz * (frac(asfloat(_3523.Load((hi + 4) * 4 + 0)) + sample_off.x) - 0.5f))) + (_3569.param3.xyz * (frac(asfloat(_3523.Load((hi + 5) * 4 + 0)) + sample_off.y) - 0.5f));
                float3 _3895 = normalize(cross(_3569.param2.xyz, _3569.param3.xyz));
                float3 param_8 = _3890;
                float3 param_9 = _3895;
                ls.lp = offset_ray(param_8, param_9);
                ls.L = _3890 - P;
                float3 _3908 = ls.L;
                float _3909 = length(_3908);
                ls.L /= _3909.xxx;
                ls.area = _3569.param1.w;
                float _3924 = dot(-ls.L, _3895);
                if (_3924 > 0.0f)
                {
                    ls.pdf = (_3909 * _3909) / (ls.area * _3924);
                }
                if ((_3569.type_and_param0.x & 64u) == 0u)
                {
                    ls.area = 0.0f;
                }
                [branch]
                if ((_3569.type_and_param0.x & 128u) != 0u)
                {
                    float3 env_col = _3539_g_params.env_col.xyz;
                    uint _3961 = asuint(_3539_g_params.env_col.w);
                    if (_3961 != 4294967295u)
                    {
                        atlas_texture_t _3969;
                        _3969.size = _1003.Load(_3961 * 80 + 0);
                        _3969.atlas = _1003.Load(_3961 * 80 + 4);
                        [unroll]
                        for (int _64ident = 0; _64ident < 4; _64ident++)
                        {
                            _3969.page[_64ident] = _1003.Load(_64ident * 4 + _3961 * 80 + 8);
                        }
                        [unroll]
                        for (int _65ident = 0; _65ident < 14; _65ident++)
                        {
                            _3969.pos[_65ident] = _1003.Load(_65ident * 4 + _3961 * 80 + 24);
                        }
                        uint _9534[14] = { _3969.pos[0], _3969.pos[1], _3969.pos[2], _3969.pos[3], _3969.pos[4], _3969.pos[5], _3969.pos[6], _3969.pos[7], _3969.pos[8], _3969.pos[9], _3969.pos[10], _3969.pos[11], _3969.pos[12], _3969.pos[13] };
                        uint _9505[4] = { _3969.page[0], _3969.page[1], _3969.page[2], _3969.page[3] };
                        atlas_texture_t _9434 = { _3969.size, _3969.atlas, _9505, _9534 };
                        float param_10 = _3539_g_params.env_rotation;
                        env_col *= SampleLatlong_RGBE(_9434, ls.L, param_10);
                    }
                    ls.col *= env_col;
                    ls.from_env = true;
                }
            }
            else
            {
                [branch]
                if (_3603 == 5u)
                {
                    float2 _4072 = (float2(frac(asfloat(_3523.Load((hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_3523.Load((hi + 5) * 4 + 0)) + sample_off.y)) * 2.0f) - 1.0f.xx;
                    float2 offset = _4072;
                    bool _4075 = _4072.x != 0.0f;
                    bool _4081;
                    if (_4075)
                    {
                        _4081 = offset.y != 0.0f;
                    }
                    else
                    {
                        _4081 = _4075;
                    }
                    if (_4081)
                    {
                        float r;
                        float theta;
                        if (abs(offset.x) > abs(offset.y))
                        {
                            r = offset.x;
                            theta = 0.785398185253143310546875f * (offset.y / offset.x);
                        }
                        else
                        {
                            r = offset.y;
                            theta = mad(-0.785398185253143310546875f, offset.x / offset.y, 1.57079637050628662109375f);
                        }
                        float _4114 = 0.5f * r;
                        offset = float2(_4114 * cos(theta), _4114 * sin(theta));
                    }
                    float3 _4136 = (_3569.param1.xyz + (_3569.param2.xyz * offset.x)) + (_3569.param3.xyz * offset.y);
                    float3 _4141 = normalize(cross(_3569.param2.xyz, _3569.param3.xyz));
                    float3 param_11 = _4136;
                    float3 param_12 = _4141;
                    ls.lp = offset_ray(param_11, param_12);
                    ls.L = _4136 - P;
                    float3 _4154 = ls.L;
                    float _4155 = length(_4154);
                    ls.L /= _4155.xxx;
                    ls.area = _3569.param1.w;
                    float _4170 = dot(-ls.L, _4141);
                    [flatten]
                    if (_4170 > 0.0f)
                    {
                        ls.pdf = (_4155 * _4155) / (ls.area * _4170);
                    }
                    if ((_3569.type_and_param0.x & 64u) == 0u)
                    {
                        ls.area = 0.0f;
                    }
                    [branch]
                    if ((_3569.type_and_param0.x & 128u) != 0u)
                    {
                        float3 env_col_1 = _3539_g_params.env_col.xyz;
                        uint _4204 = asuint(_3539_g_params.env_col.w);
                        if (_4204 != 4294967295u)
                        {
                            atlas_texture_t _4211;
                            _4211.size = _1003.Load(_4204 * 80 + 0);
                            _4211.atlas = _1003.Load(_4204 * 80 + 4);
                            [unroll]
                            for (int _66ident = 0; _66ident < 4; _66ident++)
                            {
                                _4211.page[_66ident] = _1003.Load(_66ident * 4 + _4204 * 80 + 8);
                            }
                            [unroll]
                            for (int _67ident = 0; _67ident < 14; _67ident++)
                            {
                                _4211.pos[_67ident] = _1003.Load(_67ident * 4 + _4204 * 80 + 24);
                            }
                            uint _9572[14] = { _4211.pos[0], _4211.pos[1], _4211.pos[2], _4211.pos[3], _4211.pos[4], _4211.pos[5], _4211.pos[6], _4211.pos[7], _4211.pos[8], _4211.pos[9], _4211.pos[10], _4211.pos[11], _4211.pos[12], _4211.pos[13] };
                            uint _9543[4] = { _4211.page[0], _4211.page[1], _4211.page[2], _4211.page[3] };
                            atlas_texture_t _9443 = { _4211.size, _4211.atlas, _9543, _9572 };
                            float param_13 = _3539_g_params.env_rotation;
                            env_col_1 *= SampleLatlong_RGBE(_9443, ls.L, param_13);
                        }
                        ls.col *= env_col_1;
                        ls.from_env = true;
                    }
                }
                else
                {
                    [branch]
                    if (_3603 == 3u)
                    {
                        float3 _4311 = normalize(cross(P - _3569.param1.xyz, _3569.param3.xyz));
                        float _4318 = 3.1415927410125732421875f * frac(asfloat(_3523.Load((hi + 4) * 4 + 0)) + sample_off.x);
                        float3 _4343 = (_3569.param1.xyz + (((_4311 * cos(_4318)) + (cross(_4311, _3569.param3.xyz) * sin(_4318))) * _3569.param2.w)) + ((_3569.param3.xyz * (frac(asfloat(_3523.Load((hi + 5) * 4 + 0)) + sample_off.y) - 0.5f)) * _3569.param3.w);
                        ls.lp = _4343;
                        float3 _4349 = _4343 - P;
                        float _4352 = length(_4349);
                        ls.L = _4349 / _4352.xxx;
                        ls.area = _3569.param1.w;
                        float _4367 = 1.0f - abs(dot(ls.L, _3569.param3.xyz));
                        [flatten]
                        if (_4367 != 0.0f)
                        {
                            ls.pdf = (_4352 * _4352) / (ls.area * _4367);
                        }
                        if ((_3569.type_and_param0.x & 64u) == 0u)
                        {
                            ls.area = 0.0f;
                        }
                    }
                    else
                    {
                        [branch]
                        if (_3603 == 6u)
                        {
                            uint _4397 = asuint(_3569.param1.x);
                            transform_t _4411;
                            _4411.xform = asfloat(uint4x4(_4405.Load4(asuint(_3569.param1.y) * 128 + 0), _4405.Load4(asuint(_3569.param1.y) * 128 + 16), _4405.Load4(asuint(_3569.param1.y) * 128 + 32), _4405.Load4(asuint(_3569.param1.y) * 128 + 48)));
                            _4411.inv_xform = asfloat(uint4x4(_4405.Load4(asuint(_3569.param1.y) * 128 + 64), _4405.Load4(asuint(_3569.param1.y) * 128 + 80), _4405.Load4(asuint(_3569.param1.y) * 128 + 96), _4405.Load4(asuint(_3569.param1.y) * 128 + 112)));
                            uint _4436 = _4397 * 3u;
                            vertex_t _4442;
                            [unroll]
                            for (int _68ident = 0; _68ident < 3; _68ident++)
                            {
                                _4442.p[_68ident] = asfloat(_4430.Load(_68ident * 4 + _4434.Load(_4436 * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _69ident = 0; _69ident < 3; _69ident++)
                            {
                                _4442.n[_69ident] = asfloat(_4430.Load(_69ident * 4 + _4434.Load(_4436 * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _70ident = 0; _70ident < 3; _70ident++)
                            {
                                _4442.b[_70ident] = asfloat(_4430.Load(_70ident * 4 + _4434.Load(_4436 * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _71ident = 0; _71ident < 2; _71ident++)
                            {
                                [unroll]
                                for (int _72ident = 0; _72ident < 2; _72ident++)
                                {
                                    _4442.t[_71ident][_72ident] = asfloat(_4430.Load(_72ident * 4 + _71ident * 8 + _4434.Load(_4436 * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4491;
                            [unroll]
                            for (int _73ident = 0; _73ident < 3; _73ident++)
                            {
                                _4491.p[_73ident] = asfloat(_4430.Load(_73ident * 4 + _4434.Load((_4436 + 1u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _74ident = 0; _74ident < 3; _74ident++)
                            {
                                _4491.n[_74ident] = asfloat(_4430.Load(_74ident * 4 + _4434.Load((_4436 + 1u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _75ident = 0; _75ident < 3; _75ident++)
                            {
                                _4491.b[_75ident] = asfloat(_4430.Load(_75ident * 4 + _4434.Load((_4436 + 1u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _76ident = 0; _76ident < 2; _76ident++)
                            {
                                [unroll]
                                for (int _77ident = 0; _77ident < 2; _77ident++)
                                {
                                    _4491.t[_76ident][_77ident] = asfloat(_4430.Load(_77ident * 4 + _76ident * 8 + _4434.Load((_4436 + 1u) * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4537;
                            [unroll]
                            for (int _78ident = 0; _78ident < 3; _78ident++)
                            {
                                _4537.p[_78ident] = asfloat(_4430.Load(_78ident * 4 + _4434.Load((_4436 + 2u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _79ident = 0; _79ident < 3; _79ident++)
                            {
                                _4537.n[_79ident] = asfloat(_4430.Load(_79ident * 4 + _4434.Load((_4436 + 2u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _80ident = 0; _80ident < 3; _80ident++)
                            {
                                _4537.b[_80ident] = asfloat(_4430.Load(_80ident * 4 + _4434.Load((_4436 + 2u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _81ident = 0; _81ident < 2; _81ident++)
                            {
                                [unroll]
                                for (int _82ident = 0; _82ident < 2; _82ident++)
                                {
                                    _4537.t[_81ident][_82ident] = asfloat(_4430.Load(_82ident * 4 + _81ident * 8 + _4434.Load((_4436 + 2u) * 4 + 0) * 52 + 36));
                                }
                            }
                            float3 _4583 = float3(_4442.p[0], _4442.p[1], _4442.p[2]);
                            float3 _4591 = float3(_4491.p[0], _4491.p[1], _4491.p[2]);
                            float3 _4599 = float3(_4537.p[0], _4537.p[1], _4537.p[2]);
                            float _4627 = sqrt(frac(asfloat(_3523.Load((hi + 4) * 4 + 0)) + sample_off.x));
                            float _4636 = frac(asfloat(_3523.Load((hi + 5) * 4 + 0)) + sample_off.y);
                            float _4640 = 1.0f - _4627;
                            float _4645 = 1.0f - _4636;
                            float3 _4676 = mul(float4((_4583 * _4640) + (((_4591 * _4645) + (_4599 * _4636)) * _4627), 1.0f), _4411.xform).xyz;
                            float3 _4692 = mul(float4(cross(_4591 - _4583, _4599 - _4583), 0.0f), _4411.xform).xyz;
                            ls.area = 0.5f * length(_4692);
                            float3 _4698 = normalize(_4692);
                            ls.L = _4676 - P;
                            float3 _4705 = ls.L;
                            float _4706 = length(_4705);
                            ls.L /= _4706.xxx;
                            float _4717 = dot(ls.L, _4698);
                            float cos_theta = _4717;
                            float3 _4720;
                            if (_4717 >= 0.0f)
                            {
                                _4720 = -_4698;
                            }
                            else
                            {
                                _4720 = _4698;
                            }
                            float3 param_14 = _4676;
                            float3 param_15 = _4720;
                            ls.lp = offset_ray(param_14, param_15);
                            float _4733 = cos_theta;
                            float _4734 = abs(_4733);
                            cos_theta = _4734;
                            [flatten]
                            if (_4734 > 0.0f)
                            {
                                ls.pdf = (_4706 * _4706) / (ls.area * cos_theta);
                            }
                            material_t _4771;
                            [unroll]
                            for (int _83ident = 0; _83ident < 5; _83ident++)
                            {
                                _4771.textures[_83ident] = _4758.Load(_83ident * 4 + ((_4762.Load(_4397 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
                            }
                            [unroll]
                            for (int _84ident = 0; _84ident < 3; _84ident++)
                            {
                                _4771.base_color[_84ident] = asfloat(_4758.Load(_84ident * 4 + ((_4762.Load(_4397 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
                            }
                            _4771.flags = _4758.Load(((_4762.Load(_4397 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
                            _4771.type = _4758.Load(((_4762.Load(_4397 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
                            _4771.tangent_rotation_or_strength = asfloat(_4758.Load(((_4762.Load(_4397 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
                            _4771.roughness_and_anisotropic = _4758.Load(((_4762.Load(_4397 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
                            _4771.ior = asfloat(_4758.Load(((_4762.Load(_4397 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
                            _4771.sheen_and_sheen_tint = _4758.Load(((_4762.Load(_4397 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
                            _4771.tint_and_metallic = _4758.Load(((_4762.Load(_4397 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
                            _4771.transmission_and_transmission_roughness = _4758.Load(((_4762.Load(_4397 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
                            _4771.specular_and_specular_tint = _4758.Load(((_4762.Load(_4397 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
                            _4771.clearcoat_and_clearcoat_roughness = _4758.Load(((_4762.Load(_4397 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
                            _4771.normal_map_strength_unorm = _4758.Load(((_4762.Load(_4397 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
                            if (_4771.textures[1] != 4294967295u)
                            {
                                ls.col *= SampleBilinear(_4771.textures[1], (float2(_4442.t[0][0], _4442.t[0][1]) * _4640) + (((float2(_4491.t[0][0], _4491.t[0][1]) * _4645) + (float2(_4537.t[0][0], _4537.t[0][1]) * _4636)) * _4627), 0).xyz;
                            }
                        }
                        else
                        {
                            [branch]
                            if (_3603 == 7u)
                            {
                                float _4851 = frac(asfloat(_3523.Load((hi + 4) * 4 + 0)) + sample_off.x);
                                float _4860 = frac(asfloat(_3523.Load((hi + 5) * 4 + 0)) + sample_off.y);
                                float4 dir_and_pdf;
                                if (_3539_g_params.env_qtree_levels > 0)
                                {
                                    dir_and_pdf = Sample_EnvQTree(_3539_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3539_g_params.env_qtree_levels, mad(_3532, _3543, -float(_3550)), _4851, _4860);
                                }
                                else
                                {
                                    float _4879 = 6.283185482025146484375f * _4860;
                                    float _4891 = sqrt(mad(-_4851, _4851, 1.0f));
                                    float3 param_16 = T;
                                    float3 param_17 = B;
                                    float3 param_18 = N;
                                    float3 param_19 = float3(_4891 * cos(_4879), _4891 * sin(_4879), _4851);
                                    dir_and_pdf = float4(world_from_tangent(param_16, param_17, param_18, param_19), 0.15915493667125701904296875f);
                                }
                                ls.L = dir_and_pdf.xyz;
                                ls.col *= _3539_g_params.env_col.xyz;
                                uint _4930 = asuint(_3539_g_params.env_col.w);
                                if (_4930 != 4294967295u)
                                {
                                    atlas_texture_t _4937;
                                    _4937.size = _1003.Load(_4930 * 80 + 0);
                                    _4937.atlas = _1003.Load(_4930 * 80 + 4);
                                    [unroll]
                                    for (int _85ident = 0; _85ident < 4; _85ident++)
                                    {
                                        _4937.page[_85ident] = _1003.Load(_85ident * 4 + _4930 * 80 + 8);
                                    }
                                    [unroll]
                                    for (int _86ident = 0; _86ident < 14; _86ident++)
                                    {
                                        _4937.pos[_86ident] = _1003.Load(_86ident * 4 + _4930 * 80 + 24);
                                    }
                                    uint _9657[14] = { _4937.pos[0], _4937.pos[1], _4937.pos[2], _4937.pos[3], _4937.pos[4], _4937.pos[5], _4937.pos[6], _4937.pos[7], _4937.pos[8], _4937.pos[9], _4937.pos[10], _4937.pos[11], _4937.pos[12], _4937.pos[13] };
                                    uint _9628[4] = { _4937.page[0], _4937.page[1], _4937.page[2], _4937.page[3] };
                                    atlas_texture_t _9496 = { _4937.size, _4937.atlas, _9628, _9657 };
                                    float param_20 = _3539_g_params.env_rotation;
                                    ls.col *= SampleLatlong_RGBE(_9496, ls.L, param_20);
                                }
                                ls.area = 1.0f;
                                ls.lp = P + ls.L;
                                ls.dist_mul = 3402823346297367662189621542912.0f;
                                ls.pdf = dir_and_pdf.w;
                                ls.from_env = true;
                            }
                        }
                    }
                }
            }
        }
    }
}

int2 texSize(uint index)
{
    atlas_texture_t _1006;
    _1006.size = _1003.Load(index * 80 + 0);
    _1006.atlas = _1003.Load(index * 80 + 4);
    [unroll]
    for (int _87ident = 0; _87ident < 4; _87ident++)
    {
        _1006.page[_87ident] = _1003.Load(_87ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _88ident = 0; _88ident < 14; _88ident++)
    {
        _1006.pos[_88ident] = _1003.Load(_88ident * 4 + index * 80 + 24);
    }
    return int2(int(_1006.size & 16383u), int((_1006.size >> uint(16)) & 16383u));
}

float get_texture_lod(int2 res, float lambda)
{
    return clamp(mad(0.5f, log2(float(res.x) * float(res.y)), lambda) - 1.0f, 0.0f, 13.0f);
}

float lum(float3 color)
{
    return mad(0.072168998420238494873046875f, color.z, mad(0.21267099678516387939453125f, color.x, 0.71516001224517822265625f * color.y));
}

float4 Evaluate_OrenDiffuse_BSDF(float3 V, float3 N, float3 L, float roughness, float3 base_color)
{
    float _2518 = 1.0f / mad(0.904129683971405029296875f, roughness, 3.1415927410125732421875f);
    float _2530 = max(dot(N, L), 0.0f);
    float _2535 = max(dot(N, V), 0.0f);
    float _2543 = mad(-_2530, _2535, dot(L, V));
    float t = _2543;
    if (_2543 > 0.0f)
    {
        t /= (max(_2530, _2535) + 1.1754943508222875079687365372222e-38f);
    }
    return float4(base_color * (_2530 * mad(roughness * _2518, t, _2518)), 0.15915493667125701904296875f);
}

float3 Evaluate_DiffuseNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _9193;
    do
    {
        float3 param = -float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param_1 = surf.N;
        float3 param_2 = ls.L;
        float param_3 = roughness;
        float3 param_4 = base_color;
        float4 _5521 = Evaluate_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_5 = ls.pdf;
            float param_6 = _5521.w;
            mis_weight = power_heuristic(param_5, param_6);
        }
        float3 _5544 = (ls.col * _5521.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9193 = _5544;
            break;
        }
        float3 param_7 = surf.P;
        float3 param_8 = surf.plane_N;
        float3 _5556 = offset_ray(param_7, param_8);
        sh_r.o[0] = _5556.x;
        sh_r.o[1] = _5556.y;
        sh_r.o[2] = _5556.z;
        sh_r.c[0] = ray.c[0] * _5544.x;
        sh_r.c[1] = ray.c[1] * _5544.y;
        sh_r.c[2] = ray.c[2] * _5544.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9193 = 0.0f.xxx;
        break;
    } while(false);
    return _9193;
}

float4 Sample_OrenDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float rand_u, float rand_v, inout float3 out_V)
{
    float _2577 = 6.283185482025146484375f * rand_v;
    float _2589 = sqrt(mad(-rand_u, rand_u, 1.0f));
    float3 param = T;
    float3 param_1 = B;
    float3 param_2 = N;
    float3 param_3 = float3(_2589 * cos(_2577), _2589 * sin(_2577), rand_u);
    out_V = world_from_tangent(param, param_1, param_2, param_3);
    float3 param_4 = -I;
    float3 param_5 = N;
    float3 param_6 = out_V;
    float param_7 = roughness;
    float3 param_8 = base_color;
    return Evaluate_OrenDiffuse_BSDF(param_4, param_5, param_6, param_7, param_8);
}

void Sample_DiffuseNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float param_4 = roughness;
    float3 param_5 = base_color;
    float param_6 = rand_u;
    float param_7 = rand_v;
    float3 param_8;
    float4 _5807 = Sample_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8);
    new_ray.depth = ray.depth + 1;
    float3 param_9 = surf.P;
    float3 param_10 = surf.plane_N;
    float3 _5817 = offset_ray(param_9, param_10);
    new_ray.o[0] = _5817.x;
    new_ray.o[1] = _5817.y;
    new_ray.o[2] = _5817.z;
    new_ray.d[0] = param_8.x;
    new_ray.d[1] = param_8.y;
    new_ray.d[2] = param_8.z;
    new_ray.c[0] = ((ray.c[0] * _5807.x) * mix_weight) / _5807.w;
    new_ray.c[1] = ((ray.c[1] * _5807.y) * mix_weight) / _5807.w;
    new_ray.c[2] = ((ray.c[2] * _5807.z) * mix_weight) / _5807.w;
    new_ray.pdf = _5807.w;
}

float3 tangent_from_world(float3 T, float3 B, float3 N, float3 V)
{
    return float3(dot(V, T), dot(V, B), dot(V, N));
}

float D_GGX(float3 H, float alpha_x, float alpha_y)
{
    float _9246;
    do
    {
        if (H.z == 0.0f)
        {
            _9246 = 0.0f;
            break;
        }
        float _2244 = (-H.x) / (H.z * alpha_x);
        float _2250 = (-H.y) / (H.z * alpha_y);
        float _2259 = mad(_2250, _2250, mad(_2244, _2244, 1.0f));
        _9246 = 1.0f / (((((_2259 * _2259) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _9246;
}

float G1(float3 Ve, inout float alpha_x, inout float alpha_y)
{
    alpha_x *= alpha_x;
    alpha_y *= alpha_y;
    return 1.0f / mad((-1.0f) + sqrt(1.0f + (mad(alpha_x * Ve.x, Ve.x, (alpha_y * Ve.y) * Ve.y) / (Ve.z * Ve.z))), 0.5f, 1.0f);
}

float4 Evaluate_GGXSpecular_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float alpha_x, float alpha_y, float spec_ior, float spec_F0, float3 spec_col)
{
    float _2759 = D_GGX(sampled_normal_ts, alpha_x, alpha_y);
    float3 param = view_dir_ts;
    float param_1 = alpha_x;
    float param_2 = alpha_y;
    float _2767 = G1(param, param_1, param_2);
    float3 param_3 = reflected_dir_ts;
    float param_4 = alpha_x;
    float param_5 = alpha_y;
    float _2774 = G1(param_3, param_4, param_5);
    float param_6 = dot(view_dir_ts, sampled_normal_ts);
    float param_7 = spec_ior;
    float3 F = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param_6, param_7) - spec_F0) / (1.0f - spec_F0)).xxx);
    float _2802 = 4.0f * abs(view_dir_ts.z * reflected_dir_ts.z);
    float _2805;
    if (_2802 != 0.0f)
    {
        _2805 = (_2759 * (_2767 * _2774)) / _2802;
    }
    else
    {
        _2805 = 0.0f;
    }
    F *= _2805;
    float3 param_8 = view_dir_ts;
    float param_9 = alpha_x;
    float param_10 = alpha_y;
    float _2825 = G1(param_8, param_9, param_10);
    float pdf = ((_2759 * _2825) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2840 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2840 != 0.0f)
    {
        pdf /= _2840;
    }
    float3 _2851 = F;
    float3 _2852 = _2851 * max(reflected_dir_ts.z, 0.0f);
    F = _2852;
    return float4(_2852, pdf);
}

float3 Evaluate_GlossyNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _9198;
    do
    {
        float3 _5592 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5592;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - _5592);
        float _5630 = roughness * roughness;
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = _5630;
        float param_16 = _5630;
        float param_17 = spec_ior;
        float param_18 = spec_F0;
        float3 param_19 = base_color;
        float4 _5645 = Evaluate_GGXSpecular_BSDF(param_12, param_13, param_14, param_15, param_16, param_17, param_18, param_19);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_20 = ls.pdf;
            float param_21 = _5645.w;
            mis_weight = power_heuristic(param_20, param_21);
        }
        float3 _5668 = (ls.col * _5645.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9198 = _5668;
            break;
        }
        float3 param_22 = surf.P;
        float3 param_23 = surf.plane_N;
        float3 _5680 = offset_ray(param_22, param_23);
        sh_r.o[0] = _5680.x;
        sh_r.o[1] = _5680.y;
        sh_r.o[2] = _5680.z;
        sh_r.c[0] = ray.c[0] * _5668.x;
        sh_r.c[1] = ray.c[1] * _5668.y;
        sh_r.c[2] = ray.c[2] * _5668.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9198 = 0.0f.xxx;
        break;
    } while(false);
    return _9198;
}

float3 SampleGGX_VNDF(float3 Ve, float alpha_x, float alpha_y, float U1, float U2)
{
    float3 _2062 = normalize(float3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
    float _2065 = _2062.x;
    float _2070 = _2062.y;
    float _2074 = mad(_2065, _2065, _2070 * _2070);
    float3 _2078;
    if (_2074 > 0.0f)
    {
        _2078 = float3(-_2070, _2065, 0.0f) / sqrt(_2074).xxx;
    }
    else
    {
        _2078 = float3(1.0f, 0.0f, 0.0f);
    }
    float _2100 = sqrt(U1);
    float _2103 = 6.283185482025146484375f * U2;
    float _2108 = _2100 * cos(_2103);
    float _2117 = 1.0f + _2062.z;
    float _2124 = mad(-_2108, _2108, 1.0f);
    float _2130 = mad(mad(-0.5f, _2117, 1.0f), sqrt(_2124), (0.5f * _2117) * (_2100 * sin(_2103)));
    float3 _2151 = ((_2078 * _2108) + (cross(_2062, _2078) * _2130)) + (_2062 * sqrt(max(0.0f, mad(-_2130, _2130, _2124))));
    return normalize(float3(alpha_x * _2151.x, alpha_y * _2151.y, max(0.0f, _2151.z)));
}

float4 Sample_GGXSpecular_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float anisotropic, float spec_ior, float spec_F0, float3 spec_col, float rand_u, float rand_v, inout float3 out_V)
{
    float4 _9218;
    do
    {
        float _2862 = roughness * roughness;
        float _2866 = sqrt(mad(-0.89999997615814208984375f, anisotropic, 1.0f));
        float _2870 = _2862 / _2866;
        float _2874 = _2862 * _2866;
        [branch]
        if ((_2870 * _2874) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2885 = reflect(I, N);
            float param = dot(_2885, N);
            float param_1 = spec_ior;
            float3 _2899 = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param, param_1) - spec_F0) / (1.0f - spec_F0)).xxx);
            out_V = _2885;
            _9218 = float4(_2899.x * 1000000.0f, _2899.y * 1000000.0f, _2899.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2924 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = _2870;
        float param_7 = _2874;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2933 = SampleGGX_VNDF(_2924, param_6, param_7, param_8, param_9);
        float3 _2944 = normalize(reflect(-_2924, _2933));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2944;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2924;
        float3 param_15 = _2933;
        float3 param_16 = _2944;
        float param_17 = _2870;
        float param_18 = _2874;
        float param_19 = spec_ior;
        float param_20 = spec_F0;
        float3 param_21 = spec_col;
        _9218 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _9218;
}

void Sample_GlossyNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 param_4;
    float4 _5727 = Sample_GGXSpecular_BSDF(param, param_1, param_2, param_3, roughness, 0.0f, spec_ior, spec_F0, base_color, rand_u, rand_v, param_4);
    new_ray.depth = ray.depth + 256;
    float3 param_5 = surf.P;
    float3 param_6 = surf.plane_N;
    float3 _5738 = offset_ray(param_5, param_6);
    new_ray.o[0] = _5738.x;
    new_ray.o[1] = _5738.y;
    new_ray.o[2] = _5738.z;
    new_ray.d[0] = param_4.x;
    new_ray.d[1] = param_4.y;
    new_ray.d[2] = param_4.z;
    new_ray.c[0] = ((ray.c[0] * _5727.x) * mix_weight) / _5727.w;
    new_ray.c[1] = ((ray.c[1] * _5727.y) * mix_weight) / _5727.w;
    new_ray.c[2] = ((ray.c[2] * _5727.z) * mix_weight) / _5727.w;
    new_ray.pdf = _5727.w;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _9223;
    do
    {
        bool _3166 = refr_dir_ts.z >= 0.0f;
        bool _3173;
        if (!_3166)
        {
            _3173 = view_dir_ts.z <= 0.0f;
        }
        else
        {
            _3173 = _3166;
        }
        if (_3173)
        {
            _9223 = 0.0f.xxxx;
            break;
        }
        float _3182 = D_GGX(sampled_normal_ts, roughness2, roughness2);
        float3 param = refr_dir_ts;
        float param_1 = roughness2;
        float param_2 = roughness2;
        float _3190 = G1(param, param_1, param_2);
        float3 param_3 = view_dir_ts;
        float param_4 = roughness2;
        float param_5 = roughness2;
        float _3198 = G1(param_3, param_4, param_5);
        float _3208 = mad(dot(view_dir_ts, sampled_normal_ts), eta, dot(refr_dir_ts, sampled_normal_ts));
        float _3218 = clamp(-dot(refr_dir_ts, sampled_normal_ts), 0.0f, 1.0f) / (_3208 * _3208);
        _9223 = float4(refr_col * (((((_3182 * _3198) * _3190) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3218) / view_dir_ts.z), (((_3182 * _3190) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3218) / view_dir_ts.z);
        break;
    } while(false);
    return _9223;
}

float3 Evaluate_RefractiveNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness2, float eta, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _9203;
    do
    {
        float3 _5870 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5870;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - (_5870 * eta));
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = roughness2;
        float param_16 = eta;
        float3 param_17 = base_color;
        float4 _5918 = Evaluate_GGXRefraction_BSDF(param_12, param_13, param_14, param_15, param_16, param_17);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_18 = ls.pdf;
            float param_19 = _5918.w;
            mis_weight = power_heuristic(param_18, param_19);
        }
        float3 _5941 = (ls.col * _5918.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9203 = _5941;
            break;
        }
        float3 param_20 = surf.P;
        float3 param_21 = -surf.plane_N;
        float3 _5954 = offset_ray(param_20, param_21);
        sh_r.o[0] = _5954.x;
        sh_r.o[1] = _5954.y;
        sh_r.o[2] = _5954.z;
        sh_r.c[0] = ray.c[0] * _5941.x;
        sh_r.c[1] = ray.c[1] * _5941.y;
        sh_r.c[2] = ray.c[2] * _5941.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9203 = 0.0f.xxx;
        break;
    } while(false);
    return _9203;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _9228;
    do
    {
        float _3262 = roughness * roughness;
        [branch]
        if ((_3262 * _3262) < 1.0000000116860974230803549289703e-07f)
        {
            float _3272 = dot(I, N);
            float _3273 = -_3272;
            float _3283 = mad(-(eta * eta), mad(_3272, _3273, 1.0f), 1.0f);
            if (_3283 < 0.0f)
            {
                _9228 = 0.0f.xxxx;
                break;
            }
            float _3295 = mad(eta, _3273, -sqrt(_3283));
            out_V = float4(normalize((I * eta) + (N * _3295)), _3295);
            _9228 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param = T;
        float3 param_1 = B;
        float3 param_2 = N;
        float3 param_3 = -I;
        float3 _3335 = normalize(tangent_from_world(param, param_1, param_2, param_3));
        float param_4 = _3262;
        float param_5 = _3262;
        float param_6 = rand_u;
        float param_7 = rand_v;
        float3 _3346 = SampleGGX_VNDF(_3335, param_4, param_5, param_6, param_7);
        float _3350 = dot(_3335, _3346);
        float _3360 = mad(-(eta * eta), mad(-_3350, _3350, 1.0f), 1.0f);
        if (_3360 < 0.0f)
        {
            _9228 = 0.0f.xxxx;
            break;
        }
        float _3372 = mad(eta, _3350, -sqrt(_3360));
        float3 _3382 = normalize((_3335 * (-eta)) + (_3346 * _3372));
        float3 param_8 = _3335;
        float3 param_9 = _3346;
        float3 param_10 = _3382;
        float param_11 = _3262;
        float param_12 = eta;
        float3 param_13 = refr_col;
        float3 param_14 = T;
        float3 param_15 = B;
        float3 param_16 = N;
        float3 param_17 = _3382;
        out_V = float4(world_from_tangent(param_14, param_15, param_16, param_17), _3372);
        _9228 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _9228;
}

void push_ior_stack(inout float stack[4], float val)
{
    do
    {
        if (stack[0] < 0.0f)
        {
            stack[0] = val;
            break;
        }
        if (stack[1] < 0.0f)
        {
            stack[1] = val;
            break;
        }
        if (stack[2] < 0.0f)
        {
            stack[2] = val;
            break;
        }
        stack[3] = val;
        break;
    } while(false);
}

float exchange(inout float old_value, float new_value)
{
    float _2308 = old_value;
    old_value = new_value;
    return _2308;
}

float pop_ior_stack(inout float stack[4], float default_value)
{
    float _9236;
    do
    {
        if (stack[3] > 0.0f)
        {
            float param = stack[3];
            float param_1 = -1.0f;
            float _2350 = exchange(param, param_1);
            stack[3] = param;
            _9236 = _2350;
            break;
        }
        if (stack[2] > 0.0f)
        {
            float param_2 = stack[2];
            float param_3 = -1.0f;
            float _2363 = exchange(param_2, param_3);
            stack[2] = param_2;
            _9236 = _2363;
            break;
        }
        if (stack[1] > 0.0f)
        {
            float param_4 = stack[1];
            float param_5 = -1.0f;
            float _2376 = exchange(param_4, param_5);
            stack[1] = param_4;
            _9236 = _2376;
            break;
        }
        if (stack[0] > 0.0f)
        {
            float param_6 = stack[0];
            float param_7 = -1.0f;
            float _2389 = exchange(param_6, param_7);
            stack[0] = param_6;
            _9236 = _2389;
            break;
        }
        _9236 = default_value;
        break;
    } while(false);
    return _9236;
}

void Sample_RefractiveNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, bool is_backfacing, float int_ior, float ext_ior, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float _5991;
    if (is_backfacing)
    {
        _5991 = int_ior / ext_ior;
    }
    else
    {
        _5991 = ext_ior / int_ior;
    }
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float param_4 = roughness;
    float param_5 = _5991;
    float3 param_6 = base_color;
    float param_7 = rand_u;
    float param_8 = rand_v;
    float4 param_9;
    float4 _6015 = Sample_GGXRefraction_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9);
    new_ray.depth = ray.depth + 65536;
    new_ray.c[0] = ((ray.c[0] * _6015.x) * mix_weight) / _6015.w;
    new_ray.c[1] = ((ray.c[1] * _6015.y) * mix_weight) / _6015.w;
    new_ray.c[2] = ((ray.c[2] * _6015.z) * mix_weight) / _6015.w;
    new_ray.pdf = _6015.w;
    if (!is_backfacing)
    {
        float param_10[4] = new_ray.ior;
        push_ior_stack(param_10, int_ior);
        new_ray.ior = param_10;
    }
    else
    {
        float param_11[4] = new_ray.ior;
        float param_12 = 1.0f;
        float _6071 = pop_ior_stack(param_11, param_12);
        new_ray.ior = param_11;
    }
    float3 param_13 = surf.P;
    float3 param_14 = -surf.plane_N;
    float3 _6080 = offset_ray(param_13, param_14);
    new_ray.o[0] = _6080.x;
    new_ray.o[1] = _6080.y;
    new_ray.o[2] = _6080.z;
    new_ray.d[0] = param_9.x;
    new_ray.d[1] = param_9.y;
    new_ray.d[2] = param_9.z;
}

lobe_weights_t get_lobe_weights(float base_color_lum, float spec_color_lum, float specular, float metallic, float transmission, float clearcoat)
{
    float _1715 = 1.0f - metallic;
    float _9391 = (base_color_lum * _1715) * (1.0f - transmission);
    float _1722 = transmission * _1715;
    float _1726;
    if ((specular != 0.0f) || (metallic != 0.0f))
    {
        _1726 = spec_color_lum * mad(-transmission, _1715, 1.0f);
    }
    else
    {
        _1726 = 0.0f;
    }
    float _9392 = _1726;
    float _1736 = 0.25f * clearcoat;
    float _9393 = _1736 * _1715;
    float _9394 = _1722 * base_color_lum;
    float _1745 = _9391;
    float _1754 = mad(_1722, base_color_lum, mad(_1736, _1715, _1745 + _1726));
    if (_1754 != 0.0f)
    {
        _9391 /= _1754;
        _9392 /= _1754;
        _9393 /= _1754;
        _9394 /= _1754;
    }
    lobe_weights_t _9399 = { _9391, _9392, _9393, _9394 };
    return _9399;
}

float pow5(float v)
{
    return ((v * v) * (v * v)) * v;
}

float schlick_weight(float u)
{
    float param = clamp(1.0f - u, 0.0f, 1.0f);
    return pow5(param);
}

float BRDF_PrincipledDiffuse(float3 V, float3 N, float3 L, float3 H, float roughness)
{
    float _9251;
    do
    {
        float _2470 = dot(N, L);
        if (_2470 <= 0.0f)
        {
            _9251 = 0.0f;
            break;
        }
        float param = _2470;
        float param_1 = dot(N, V);
        float _2491 = dot(L, H);
        float _2499 = mad((2.0f * _2491) * _2491, roughness, 0.5f);
        _9251 = lerp(1.0f, _2499, schlick_weight(param)) * lerp(1.0f, _2499, schlick_weight(param_1));
        break;
    } while(false);
    return _9251;
}

float4 Evaluate_PrincipledDiffuse_BSDF(float3 V, float3 N, float3 L, float roughness, float3 base_color, float3 sheen_color, bool uniform_sampling)
{
    float weight;
    float pdf;
    if (uniform_sampling)
    {
        weight = 2.0f * dot(N, L);
        pdf = 0.15915493667125701904296875f;
    }
    else
    {
        weight = 1.0f;
        pdf = dot(N, L) * 0.3183098733425140380859375f;
    }
    float3 _2640 = normalize(L + V);
    float3 H = _2640;
    if (dot(V, _2640) < 0.0f)
    {
        H = -H;
    }
    float3 param = V;
    float3 param_1 = N;
    float3 param_2 = L;
    float3 param_3 = H;
    float param_4 = roughness;
    float3 diff_col = base_color * (weight * BRDF_PrincipledDiffuse(param, param_1, param_2, param_3, param_4));
    float param_5 = dot(L, H);
    float3 _2675 = diff_col;
    float3 _2676 = _2675 + (sheen_color * (3.1415927410125732421875f * schlick_weight(param_5)));
    diff_col = _2676;
    return float4(_2676, pdf);
}

float D_GTR1(float NDotH, float a)
{
    float _9256;
    do
    {
        if (a >= 1.0f)
        {
            _9256 = 0.3183098733425140380859375f;
            break;
        }
        float _2218 = mad(a, a, -1.0f);
        _9256 = _2218 / ((3.1415927410125732421875f * log(a * a)) * mad(_2218 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _9256;
}

float4 Evaluate_PrincipledClearcoat_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float clearcoat_roughness2, float clearcoat_ior, float clearcoat_F0)
{
    float param = sampled_normal_ts.z;
    float param_1 = clearcoat_roughness2;
    float _2976 = D_GTR1(param, param_1);
    float3 param_2 = view_dir_ts;
    float param_3 = 0.0625f;
    float param_4 = 0.0625f;
    float _2983 = G1(param_2, param_3, param_4);
    float3 param_5 = reflected_dir_ts;
    float param_6 = 0.0625f;
    float param_7 = 0.0625f;
    float _2988 = G1(param_5, param_6, param_7);
    float param_8 = dot(reflected_dir_ts, sampled_normal_ts);
    float param_9 = clearcoat_ior;
    float F = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param_8, param_9) - clearcoat_F0) / (1.0f - clearcoat_F0));
    float _3015 = (4.0f * abs(view_dir_ts.z)) * abs(reflected_dir_ts.z);
    float _3018;
    if (_3015 != 0.0f)
    {
        _3018 = (_2976 * (_2983 * _2988)) / _3015;
    }
    else
    {
        _3018 = 0.0f;
    }
    F *= _3018;
    float3 param_10 = view_dir_ts;
    float param_11 = 0.0625f;
    float param_12 = 0.0625f;
    float _3036 = G1(param_10, param_11, param_12);
    float pdf = ((_2976 * _3036) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _3051 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_3051 != 0.0f)
    {
        pdf /= _3051;
    }
    float _3062 = F;
    float _3063 = _3062 * clamp(reflected_dir_ts.z, 0.0f, 1.0f);
    F = _3063;
    return float4(_3063, _3063, _3063, pdf);
}

float3 Evaluate_PrincipledNode(light_sample_t ls, ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float N_dot_L, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _9208;
    do
    {
        float3 _6103 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 lcol = 0.0f.xxx;
        float bsdf_pdf = 0.0f;
        bool _6108 = N_dot_L > 0.0f;
        [branch]
        if ((lobe_weights.diffuse > 1.0000000116860974230803549289703e-07f) && _6108)
        {
            float3 param = -_6103;
            float3 param_1 = surf.N;
            float3 param_2 = ls.L;
            float param_3 = diff.roughness;
            float3 param_4 = diff.base_color;
            float3 param_5 = diff.sheen_color;
            bool param_6 = false;
            float4 _6127 = Evaluate_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6);
            bsdf_pdf = mad(lobe_weights.diffuse, _6127.w, bsdf_pdf);
            lcol += (((ls.col * N_dot_L) * (_6127 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * ls.pdf).xxx);
        }
        float3 H;
        [flatten]
        if (_6108)
        {
            H = normalize(ls.L - _6103);
        }
        else
        {
            H = normalize(ls.L - (_6103 * trans.eta));
        }
        float _6166 = spec.roughness * spec.roughness;
        float _6171 = sqrt(mad(-0.89999997615814208984375f, spec.anisotropy, 1.0f));
        float _6175 = _6166 / _6171;
        float _6179 = _6166 * _6171;
        float3 param_7 = surf.T;
        float3 param_8 = surf.B;
        float3 param_9 = surf.N;
        float3 param_10 = -_6103;
        float3 _6190 = tangent_from_world(param_7, param_8, param_9, param_10);
        float3 param_11 = surf.T;
        float3 param_12 = surf.B;
        float3 param_13 = surf.N;
        float3 param_14 = ls.L;
        float3 _6200 = tangent_from_world(param_11, param_12, param_13, param_14);
        float3 param_15 = surf.T;
        float3 param_16 = surf.B;
        float3 param_17 = surf.N;
        float3 param_18 = H;
        float3 _6210 = tangent_from_world(param_15, param_16, param_17, param_18);
        bool _6212 = lobe_weights.specular > 0.0f;
        bool _6219;
        if (_6212)
        {
            _6219 = (_6175 * _6179) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _6219 = _6212;
        }
        [branch]
        if (_6219 && _6108)
        {
            float3 param_19 = _6190;
            float3 param_20 = _6210;
            float3 param_21 = _6200;
            float param_22 = _6175;
            float param_23 = _6179;
            float param_24 = spec.ior;
            float param_25 = spec.F0;
            float3 param_26 = spec.tmp_col;
            float4 _6241 = Evaluate_GGXSpecular_BSDF(param_19, param_20, param_21, param_22, param_23, param_24, param_25, param_26);
            bsdf_pdf = mad(lobe_weights.specular, _6241.w, bsdf_pdf);
            lcol += ((ls.col * _6241.xyz) / ls.pdf.xxx);
        }
        float _6260 = coat.roughness * coat.roughness;
        bool _6262 = lobe_weights.clearcoat > 0.0f;
        bool _6269;
        if (_6262)
        {
            _6269 = (_6260 * _6260) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _6269 = _6262;
        }
        [branch]
        if (_6269 && _6108)
        {
            float3 param_27 = _6190;
            float3 param_28 = _6210;
            float3 param_29 = _6200;
            float param_30 = _6260;
            float param_31 = coat.ior;
            float param_32 = coat.F0;
            float4 _6287 = Evaluate_PrincipledClearcoat_BSDF(param_27, param_28, param_29, param_30, param_31, param_32);
            bsdf_pdf = mad(lobe_weights.clearcoat, _6287.w, bsdf_pdf);
            lcol += (((ls.col * 0.25f) * _6287.xyz) / ls.pdf.xxx);
        }
        [branch]
        if (lobe_weights.refraction > 0.0f)
        {
            bool _6309 = trans.fresnel != 0.0f;
            bool _6316;
            if (_6309)
            {
                _6316 = (_6166 * _6166) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _6316 = _6309;
            }
            [branch]
            if (_6316 && _6108)
            {
                float3 param_33 = _6190;
                float3 param_34 = _6210;
                float3 param_35 = _6200;
                float param_36 = _6166;
                float param_37 = _6166;
                float param_38 = 1.0f;
                float param_39 = 0.0f;
                float3 param_40 = 1.0f.xxx;
                float4 _6335 = Evaluate_GGXSpecular_BSDF(param_33, param_34, param_35, param_36, param_37, param_38, param_39, param_40);
                bsdf_pdf = mad(lobe_weights.refraction * trans.fresnel, _6335.w, bsdf_pdf);
                lcol += ((ls.col * _6335.xyz) * (trans.fresnel / ls.pdf));
            }
            float _6357 = trans.roughness * trans.roughness;
            bool _6359 = trans.fresnel != 1.0f;
            bool _6366;
            if (_6359)
            {
                _6366 = (_6357 * _6357) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _6366 = _6359;
            }
            [branch]
            if (_6366 && (N_dot_L < 0.0f))
            {
                float3 param_41 = _6190;
                float3 param_42 = _6210;
                float3 param_43 = _6200;
                float param_44 = _6357;
                float param_45 = trans.eta;
                float3 param_46 = diff.base_color;
                float4 _6384 = Evaluate_GGXRefraction_BSDF(param_41, param_42, param_43, param_44, param_45, param_46);
                float _6387 = 1.0f - trans.fresnel;
                bsdf_pdf = mad(lobe_weights.refraction * _6387, _6384.w, bsdf_pdf);
                lcol += ((ls.col * _6384.xyz) * (_6387 / ls.pdf));
            }
        }
        float mis_weight = 1.0f;
        [flatten]
        if (ls.area > 0.0f)
        {
            float param_47 = ls.pdf;
            float param_48 = bsdf_pdf;
            mis_weight = power_heuristic(param_47, param_48);
        }
        lcol *= (mix_weight * mis_weight);
        [branch]
        if (!ls.cast_shadow)
        {
            _9208 = lcol;
            break;
        }
        float3 _6427;
        if (N_dot_L < 0.0f)
        {
            _6427 = -surf.plane_N;
        }
        else
        {
            _6427 = surf.plane_N;
        }
        float3 param_49 = surf.P;
        float3 param_50 = _6427;
        float3 _6438 = offset_ray(param_49, param_50);
        sh_r.o[0] = _6438.x;
        sh_r.o[1] = _6438.y;
        sh_r.o[2] = _6438.z;
        sh_r.c[0] = ray.c[0] * lcol.x;
        sh_r.c[1] = ray.c[1] * lcol.y;
        sh_r.c[2] = ray.c[2] * lcol.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9208 = 0.0f.xxx;
        break;
    } while(false);
    return _9208;
}

float4 Sample_PrincipledDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float3 sheen_color, bool uniform_sampling, float rand_u, float rand_v, inout float3 out_V)
{
    float _2687 = 6.283185482025146484375f * rand_v;
    float _2690 = cos(_2687);
    float _2693 = sin(_2687);
    float3 V;
    if (uniform_sampling)
    {
        float _2702 = sqrt(mad(-rand_u, rand_u, 1.0f));
        V = float3(_2702 * _2690, _2702 * _2693, rand_u);
    }
    else
    {
        float _2715 = sqrt(rand_u);
        V = float3(_2715 * _2690, _2715 * _2693, sqrt(1.0f - rand_u));
    }
    float3 param = T;
    float3 param_1 = B;
    float3 param_2 = N;
    float3 param_3 = V;
    out_V = world_from_tangent(param, param_1, param_2, param_3);
    float3 param_4 = -I;
    float3 param_5 = N;
    float3 param_6 = out_V;
    float param_7 = roughness;
    float3 param_8 = base_color;
    float3 param_9 = sheen_color;
    bool param_10 = uniform_sampling;
    return Evaluate_PrincipledDiffuse_BSDF(param_4, param_5, param_6, param_7, param_8, param_9, param_10);
}

float4 Sample_PrincipledClearcoat_BSDF(float3 T, float3 B, float3 N, float3 I, float clearcoat_roughness2, float clearcoat_ior, float clearcoat_F0, float rand_u, float rand_v, inout float3 out_V)
{
    float4 _9241;
    do
    {
        [branch]
        if ((clearcoat_roughness2 * clearcoat_roughness2) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _3080 = reflect(I, N);
            float param = dot(_3080, N);
            float param_1 = clearcoat_ior;
            out_V = _3080;
            float _3099 = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param, param_1) - clearcoat_F0) / (1.0f - clearcoat_F0)) * 1000000.0f;
            _9241 = float4(_3099, _3099, _3099, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _3117 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = clearcoat_roughness2;
        float param_7 = clearcoat_roughness2;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _3128 = SampleGGX_VNDF(_3117, param_6, param_7, param_8, param_9);
        float3 _3139 = normalize(reflect(-_3117, _3128));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _3139;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _3117;
        float3 param_15 = _3128;
        float3 param_16 = _3139;
        float param_17 = clearcoat_roughness2;
        float param_18 = clearcoat_ior;
        float param_19 = clearcoat_F0;
        _9241 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _9241;
}

void Sample_PrincipledNode(ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float rand_u, float rand_v, inout float mix_rand, float mix_weight, inout ray_data_t new_ray)
{
    float3 _6473 = float3(ray.d[0], ray.d[1], ray.d[2]);
    int _6477 = ray.depth & 255;
    int _6481 = (ray.depth >> 8) & 255;
    int _6485 = (ray.depth >> 16) & 255;
    int _6496 = (_6477 + _6481) + _6485;
    [branch]
    if (mix_rand < lobe_weights.diffuse)
    {
        bool _6505 = _6477 < _3539_g_params.max_diff_depth;
        bool _6512;
        if (_6505)
        {
            _6512 = _6496 < _3539_g_params.max_total_depth;
        }
        else
        {
            _6512 = _6505;
        }
        if (_6512)
        {
            float3 param = surf.T;
            float3 param_1 = surf.B;
            float3 param_2 = surf.N;
            float3 param_3 = _6473;
            float param_4 = diff.roughness;
            float3 param_5 = diff.base_color;
            float3 param_6 = diff.sheen_color;
            bool param_7 = false;
            float param_8 = rand_u;
            float param_9 = rand_v;
            float3 param_10;
            float4 _6535 = Sample_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9, param_10);
            float3 _6540 = _6535.xyz * (1.0f - metallic);
            new_ray.depth = ray.depth + 1;
            float3 param_11 = surf.P;
            float3 param_12 = surf.plane_N;
            float3 _6555 = offset_ray(param_11, param_12);
            new_ray.o[0] = _6555.x;
            new_ray.o[1] = _6555.y;
            new_ray.o[2] = _6555.z;
            new_ray.d[0] = param_10.x;
            new_ray.d[1] = param_10.y;
            new_ray.d[2] = param_10.z;
            new_ray.c[0] = ((ray.c[0] * _6540.x) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[1] = ((ray.c[1] * _6540.y) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[2] = ((ray.c[2] * _6540.z) * mix_weight) / lobe_weights.diffuse;
            new_ray.pdf = _6535.w;
        }
    }
    else
    {
        float _6605 = lobe_weights.diffuse + lobe_weights.specular;
        [branch]
        if (mix_rand < _6605)
        {
            bool _6612 = _6481 < _3539_g_params.max_spec_depth;
            bool _6619;
            if (_6612)
            {
                _6619 = _6496 < _3539_g_params.max_total_depth;
            }
            else
            {
                _6619 = _6612;
            }
            if (_6619)
            {
                float3 param_13 = surf.T;
                float3 param_14 = surf.B;
                float3 param_15 = surf.N;
                float3 param_16 = _6473;
                float3 param_17;
                float4 _6638 = Sample_GGXSpecular_BSDF(param_13, param_14, param_15, param_16, spec.roughness, spec.anisotropy, spec.ior, spec.F0, spec.tmp_col, rand_u, rand_v, param_17);
                float _6643 = _6638.w * lobe_weights.specular;
                new_ray.depth = ray.depth + 256;
                new_ray.c[0] = ((ray.c[0] * _6638.x) * mix_weight) / _6643;
                new_ray.c[1] = ((ray.c[1] * _6638.y) * mix_weight) / _6643;
                new_ray.c[2] = ((ray.c[2] * _6638.z) * mix_weight) / _6643;
                new_ray.pdf = _6643;
                float3 param_18 = surf.P;
                float3 param_19 = surf.plane_N;
                float3 _6683 = offset_ray(param_18, param_19);
                new_ray.o[0] = _6683.x;
                new_ray.o[1] = _6683.y;
                new_ray.o[2] = _6683.z;
                new_ray.d[0] = param_17.x;
                new_ray.d[1] = param_17.y;
                new_ray.d[2] = param_17.z;
            }
        }
        else
        {
            float _6708 = _6605 + lobe_weights.clearcoat;
            [branch]
            if (mix_rand < _6708)
            {
                bool _6715 = _6481 < _3539_g_params.max_spec_depth;
                bool _6722;
                if (_6715)
                {
                    _6722 = _6496 < _3539_g_params.max_total_depth;
                }
                else
                {
                    _6722 = _6715;
                }
                if (_6722)
                {
                    float3 param_20 = surf.T;
                    float3 param_21 = surf.B;
                    float3 param_22 = surf.N;
                    float3 param_23 = _6473;
                    float param_24 = coat.roughness * coat.roughness;
                    float param_25 = coat.ior;
                    float param_26 = coat.F0;
                    float param_27 = rand_u;
                    float param_28 = rand_v;
                    float3 param_29;
                    float4 _6746 = Sample_PrincipledClearcoat_BSDF(param_20, param_21, param_22, param_23, param_24, param_25, param_26, param_27, param_28, param_29);
                    float _6751 = _6746.w * lobe_weights.clearcoat;
                    new_ray.depth = ray.depth + 256;
                    new_ray.c[0] = (((0.25f * ray.c[0]) * _6746.x) * mix_weight) / _6751;
                    new_ray.c[1] = (((0.25f * ray.c[1]) * _6746.y) * mix_weight) / _6751;
                    new_ray.c[2] = (((0.25f * ray.c[2]) * _6746.z) * mix_weight) / _6751;
                    new_ray.pdf = _6751;
                    float3 param_30 = surf.P;
                    float3 param_31 = surf.plane_N;
                    float3 _6794 = offset_ray(param_30, param_31);
                    new_ray.o[0] = _6794.x;
                    new_ray.o[1] = _6794.y;
                    new_ray.o[2] = _6794.z;
                    new_ray.d[0] = param_29.x;
                    new_ray.d[1] = param_29.y;
                    new_ray.d[2] = param_29.z;
                }
            }
            else
            {
                bool _6816 = mix_rand >= trans.fresnel;
                bool _6823;
                if (_6816)
                {
                    _6823 = _6485 < _3539_g_params.max_refr_depth;
                }
                else
                {
                    _6823 = _6816;
                }
                bool _6837;
                if (!_6823)
                {
                    bool _6829 = mix_rand < trans.fresnel;
                    bool _6836;
                    if (_6829)
                    {
                        _6836 = _6481 < _3539_g_params.max_spec_depth;
                    }
                    else
                    {
                        _6836 = _6829;
                    }
                    _6837 = _6836;
                }
                else
                {
                    _6837 = _6823;
                }
                bool _6844;
                if (_6837)
                {
                    _6844 = _6496 < _3539_g_params.max_total_depth;
                }
                else
                {
                    _6844 = _6837;
                }
                [branch]
                if (_6844)
                {
                    mix_rand -= _6708;
                    mix_rand /= lobe_weights.refraction;
                    float4 F;
                    float3 V;
                    [branch]
                    if (mix_rand < trans.fresnel)
                    {
                        float3 param_32 = surf.T;
                        float3 param_33 = surf.B;
                        float3 param_34 = surf.N;
                        float3 param_35 = _6473;
                        float3 param_36;
                        float4 _6874 = Sample_GGXSpecular_BSDF(param_32, param_33, param_34, param_35, spec.roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, rand_u, rand_v, param_36);
                        V = param_36;
                        F = _6874;
                        new_ray.depth = ray.depth + 256;
                        float3 param_37 = surf.P;
                        float3 param_38 = surf.plane_N;
                        float3 _6884 = offset_ray(param_37, param_38);
                        new_ray.o[0] = _6884.x;
                        new_ray.o[1] = _6884.y;
                        new_ray.o[2] = _6884.z;
                    }
                    else
                    {
                        float3 param_39 = surf.T;
                        float3 param_40 = surf.B;
                        float3 param_41 = surf.N;
                        float3 param_42 = _6473;
                        float param_43 = trans.roughness;
                        float param_44 = trans.eta;
                        float3 param_45 = diff.base_color;
                        float param_46 = rand_u;
                        float param_47 = rand_v;
                        float4 param_48;
                        float4 _6913 = Sample_GGXRefraction_BSDF(param_39, param_40, param_41, param_42, param_43, param_44, param_45, param_46, param_47, param_48);
                        F = _6913;
                        V = param_48.xyz;
                        new_ray.depth = ray.depth + 65536;
                        float3 param_49 = surf.P;
                        float3 param_50 = -surf.plane_N;
                        float3 _6926 = offset_ray(param_49, param_50);
                        new_ray.o[0] = _6926.x;
                        new_ray.o[1] = _6926.y;
                        new_ray.o[2] = _6926.z;
                        if (!trans.backfacing)
                        {
                            float param_51[4] = new_ray.ior;
                            push_ior_stack(param_51, trans.int_ior);
                            new_ray.ior = param_51;
                        }
                        else
                        {
                            float param_52[4] = new_ray.ior;
                            float param_53 = 1.0f;
                            float _6952 = pop_ior_stack(param_52, param_53);
                            new_ray.ior = param_52;
                        }
                    }
                    float4 _10784 = F;
                    float _6958 = _10784.w * lobe_weights.refraction;
                    float4 _10786 = _10784;
                    _10786.w = _6958;
                    F = _10786;
                    new_ray.c[0] = ((ray.c[0] * _10784.x) * mix_weight) / _6958;
                    new_ray.c[1] = ((ray.c[1] * _10784.y) * mix_weight) / _6958;
                    new_ray.c[2] = ((ray.c[2] * _10784.z) * mix_weight) / _6958;
                    new_ray.pdf = _6958;
                    new_ray.d[0] = V.x;
                    new_ray.d[1] = V.y;
                    new_ray.d[2] = V.z;
                }
            }
        }
    }
}

float3 ShadeSurface(hit_data_t inter, ray_data_t ray, inout float3 out_base_color, inout float3 out_normals)
{
    float3 _9178;
    do
    {
        float3 _7014 = float3(ray.d[0], ray.d[1], ray.d[2]);
        [branch]
        if (inter.mask == 0)
        {
            ray_data_t param = ray;
            float3 _7023 = Evaluate_EnvColor(param);
            _9178 = float3(ray.c[0] * _7023.x, ray.c[1] * _7023.y, ray.c[2] * _7023.z);
            break;
        }
        float3 _7050 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_7014 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            ray_data_t param_1 = ray;
            hit_data_t param_2 = inter;
            float3 _7062 = Evaluate_LightColor(param_1, param_2);
            _9178 = float3(ray.c[0] * _7062.x, ray.c[1] * _7062.y, ray.c[2] * _7062.z);
            break;
        }
        bool _7083 = inter.prim_index < 0;
        int _7086;
        if (_7083)
        {
            _7086 = (-1) - inter.prim_index;
        }
        else
        {
            _7086 = inter.prim_index;
        }
        uint _7097 = uint(_7086);
        material_t _7105;
        [unroll]
        for (int _89ident = 0; _89ident < 5; _89ident++)
        {
            _7105.textures[_89ident] = _4758.Load(_89ident * 4 + ((_4762.Load(_7097 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
        }
        [unroll]
        for (int _90ident = 0; _90ident < 3; _90ident++)
        {
            _7105.base_color[_90ident] = asfloat(_4758.Load(_90ident * 4 + ((_4762.Load(_7097 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
        }
        _7105.flags = _4758.Load(((_4762.Load(_7097 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
        _7105.type = _4758.Load(((_4762.Load(_7097 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
        _7105.tangent_rotation_or_strength = asfloat(_4758.Load(((_4762.Load(_7097 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
        _7105.roughness_and_anisotropic = _4758.Load(((_4762.Load(_7097 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
        _7105.ior = asfloat(_4758.Load(((_4762.Load(_7097 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
        _7105.sheen_and_sheen_tint = _4758.Load(((_4762.Load(_7097 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
        _7105.tint_and_metallic = _4758.Load(((_4762.Load(_7097 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
        _7105.transmission_and_transmission_roughness = _4758.Load(((_4762.Load(_7097 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
        _7105.specular_and_specular_tint = _4758.Load(((_4762.Load(_7097 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
        _7105.clearcoat_and_clearcoat_roughness = _4758.Load(((_4762.Load(_7097 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
        _7105.normal_map_strength_unorm = _4758.Load(((_4762.Load(_7097 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
        uint _10233 = _7105.textures[0];
        uint _10234 = _7105.textures[1];
        uint _10235 = _7105.textures[2];
        uint _10236 = _7105.textures[3];
        uint _10237 = _7105.textures[4];
        float _10238 = _7105.base_color[0];
        float _10239 = _7105.base_color[1];
        float _10240 = _7105.base_color[2];
        uint _9843 = _7105.flags;
        uint _9844 = _7105.type;
        float _9845 = _7105.tangent_rotation_or_strength;
        uint _9846 = _7105.roughness_and_anisotropic;
        float _9847 = _7105.ior;
        uint _9848 = _7105.sheen_and_sheen_tint;
        uint _9849 = _7105.tint_and_metallic;
        uint _9850 = _7105.transmission_and_transmission_roughness;
        uint _9851 = _7105.specular_and_specular_tint;
        uint _9852 = _7105.clearcoat_and_clearcoat_roughness;
        uint _9853 = _7105.normal_map_strength_unorm;
        transform_t _7160;
        _7160.xform = asfloat(uint4x4(_4405.Load4(asuint(asfloat(_7153.Load(inter.obj_index * 32 + 12))) * 128 + 0), _4405.Load4(asuint(asfloat(_7153.Load(inter.obj_index * 32 + 12))) * 128 + 16), _4405.Load4(asuint(asfloat(_7153.Load(inter.obj_index * 32 + 12))) * 128 + 32), _4405.Load4(asuint(asfloat(_7153.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _7160.inv_xform = asfloat(uint4x4(_4405.Load4(asuint(asfloat(_7153.Load(inter.obj_index * 32 + 12))) * 128 + 64), _4405.Load4(asuint(asfloat(_7153.Load(inter.obj_index * 32 + 12))) * 128 + 80), _4405.Load4(asuint(asfloat(_7153.Load(inter.obj_index * 32 + 12))) * 128 + 96), _4405.Load4(asuint(asfloat(_7153.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _7167 = _7097 * 3u;
        vertex_t _7172;
        [unroll]
        for (int _91ident = 0; _91ident < 3; _91ident++)
        {
            _7172.p[_91ident] = asfloat(_4430.Load(_91ident * 4 + _4434.Load(_7167 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _92ident = 0; _92ident < 3; _92ident++)
        {
            _7172.n[_92ident] = asfloat(_4430.Load(_92ident * 4 + _4434.Load(_7167 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _93ident = 0; _93ident < 3; _93ident++)
        {
            _7172.b[_93ident] = asfloat(_4430.Load(_93ident * 4 + _4434.Load(_7167 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _94ident = 0; _94ident < 2; _94ident++)
        {
            [unroll]
            for (int _95ident = 0; _95ident < 2; _95ident++)
            {
                _7172.t[_94ident][_95ident] = asfloat(_4430.Load(_95ident * 4 + _94ident * 8 + _4434.Load(_7167 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _7218;
        [unroll]
        for (int _96ident = 0; _96ident < 3; _96ident++)
        {
            _7218.p[_96ident] = asfloat(_4430.Load(_96ident * 4 + _4434.Load((_7167 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _97ident = 0; _97ident < 3; _97ident++)
        {
            _7218.n[_97ident] = asfloat(_4430.Load(_97ident * 4 + _4434.Load((_7167 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _98ident = 0; _98ident < 3; _98ident++)
        {
            _7218.b[_98ident] = asfloat(_4430.Load(_98ident * 4 + _4434.Load((_7167 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _99ident = 0; _99ident < 2; _99ident++)
        {
            [unroll]
            for (int _100ident = 0; _100ident < 2; _100ident++)
            {
                _7218.t[_99ident][_100ident] = asfloat(_4430.Load(_100ident * 4 + _99ident * 8 + _4434.Load((_7167 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _7264;
        [unroll]
        for (int _101ident = 0; _101ident < 3; _101ident++)
        {
            _7264.p[_101ident] = asfloat(_4430.Load(_101ident * 4 + _4434.Load((_7167 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _102ident = 0; _102ident < 3; _102ident++)
        {
            _7264.n[_102ident] = asfloat(_4430.Load(_102ident * 4 + _4434.Load((_7167 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _103ident = 0; _103ident < 3; _103ident++)
        {
            _7264.b[_103ident] = asfloat(_4430.Load(_103ident * 4 + _4434.Load((_7167 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _104ident = 0; _104ident < 2; _104ident++)
        {
            [unroll]
            for (int _105ident = 0; _105ident < 2; _105ident++)
            {
                _7264.t[_104ident][_105ident] = asfloat(_4430.Load(_105ident * 4 + _104ident * 8 + _4434.Load((_7167 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _7310 = float3(_7172.p[0], _7172.p[1], _7172.p[2]);
        float3 _7318 = float3(_7218.p[0], _7218.p[1], _7218.p[2]);
        float3 _7326 = float3(_7264.p[0], _7264.p[1], _7264.p[2]);
        float _7333 = (1.0f - inter.u) - inter.v;
        float3 _7365 = normalize(((float3(_7172.n[0], _7172.n[1], _7172.n[2]) * _7333) + (float3(_7218.n[0], _7218.n[1], _7218.n[2]) * inter.u)) + (float3(_7264.n[0], _7264.n[1], _7264.n[2]) * inter.v));
        float3 _9782 = _7365;
        float2 _7391 = ((float2(_7172.t[0][0], _7172.t[0][1]) * _7333) + (float2(_7218.t[0][0], _7218.t[0][1]) * inter.u)) + (float2(_7264.t[0][0], _7264.t[0][1]) * inter.v);
        float3 _7407 = cross(_7318 - _7310, _7326 - _7310);
        float _7412 = length(_7407);
        float3 _9783 = _7407 / _7412.xxx;
        float3 _7449 = ((float3(_7172.b[0], _7172.b[1], _7172.b[2]) * _7333) + (float3(_7218.b[0], _7218.b[1], _7218.b[2]) * inter.u)) + (float3(_7264.b[0], _7264.b[1], _7264.b[2]) * inter.v);
        float3 _9781 = _7449;
        float3 _9780 = cross(_7449, _7365);
        if (_7083)
        {
            if ((_4762.Load(_7097 * 4 + 0) & 65535u) == 65535u)
            {
                _9178 = 0.0f.xxx;
                break;
            }
            material_t _7474;
            [unroll]
            for (int _106ident = 0; _106ident < 5; _106ident++)
            {
                _7474.textures[_106ident] = _4758.Load(_106ident * 4 + (_4762.Load(_7097 * 4 + 0) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _107ident = 0; _107ident < 3; _107ident++)
            {
                _7474.base_color[_107ident] = asfloat(_4758.Load(_107ident * 4 + (_4762.Load(_7097 * 4 + 0) & 16383u) * 76 + 20));
            }
            _7474.flags = _4758.Load((_4762.Load(_7097 * 4 + 0) & 16383u) * 76 + 32);
            _7474.type = _4758.Load((_4762.Load(_7097 * 4 + 0) & 16383u) * 76 + 36);
            _7474.tangent_rotation_or_strength = asfloat(_4758.Load((_4762.Load(_7097 * 4 + 0) & 16383u) * 76 + 40));
            _7474.roughness_and_anisotropic = _4758.Load((_4762.Load(_7097 * 4 + 0) & 16383u) * 76 + 44);
            _7474.ior = asfloat(_4758.Load((_4762.Load(_7097 * 4 + 0) & 16383u) * 76 + 48));
            _7474.sheen_and_sheen_tint = _4758.Load((_4762.Load(_7097 * 4 + 0) & 16383u) * 76 + 52);
            _7474.tint_and_metallic = _4758.Load((_4762.Load(_7097 * 4 + 0) & 16383u) * 76 + 56);
            _7474.transmission_and_transmission_roughness = _4758.Load((_4762.Load(_7097 * 4 + 0) & 16383u) * 76 + 60);
            _7474.specular_and_specular_tint = _4758.Load((_4762.Load(_7097 * 4 + 0) & 16383u) * 76 + 64);
            _7474.clearcoat_and_clearcoat_roughness = _4758.Load((_4762.Load(_7097 * 4 + 0) & 16383u) * 76 + 68);
            _7474.normal_map_strength_unorm = _4758.Load((_4762.Load(_7097 * 4 + 0) & 16383u) * 76 + 72);
            _10233 = _7474.textures[0];
            _10234 = _7474.textures[1];
            _10235 = _7474.textures[2];
            _10236 = _7474.textures[3];
            _10237 = _7474.textures[4];
            _10238 = _7474.base_color[0];
            _10239 = _7474.base_color[1];
            _10240 = _7474.base_color[2];
            _9843 = _7474.flags;
            _9844 = _7474.type;
            _9845 = _7474.tangent_rotation_or_strength;
            _9846 = _7474.roughness_and_anisotropic;
            _9847 = _7474.ior;
            _9848 = _7474.sheen_and_sheen_tint;
            _9849 = _7474.tint_and_metallic;
            _9850 = _7474.transmission_and_transmission_roughness;
            _9851 = _7474.specular_and_specular_tint;
            _9852 = _7474.clearcoat_and_clearcoat_roughness;
            _9853 = _7474.normal_map_strength_unorm;
            _9783 = -_9783;
            _9782 = -_9782;
            _9781 = -_9781;
            _9780 = -_9780;
        }
        float3 param_3 = _9783;
        float4x4 param_4 = _7160.inv_xform;
        _9783 = TransformNormal(param_3, param_4);
        float3 param_5 = _9782;
        float4x4 param_6 = _7160.inv_xform;
        _9782 = TransformNormal(param_5, param_6);
        float3 param_7 = _9781;
        float4x4 param_8 = _7160.inv_xform;
        _9781 = TransformNormal(param_7, param_8);
        float3 param_9 = _9780;
        float4x4 param_10 = _7160.inv_xform;
        _9783 = normalize(_9783);
        _9782 = normalize(_9782);
        _9781 = normalize(_9781);
        _9780 = normalize(TransformNormal(param_9, param_10));
        float _7614 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _7624 = mad(0.5f, log2(abs(mad(_7218.t[0][0] - _7172.t[0][0], _7264.t[0][1] - _7172.t[0][1], -((_7264.t[0][0] - _7172.t[0][0]) * (_7218.t[0][1] - _7172.t[0][1])))) / _7412), log2(_7614));
        uint param_11 = uint(hash(ray.xy));
        float _7631 = construct_float(param_11);
        uint param_12 = uint(hash(hash(ray.xy)));
        float _7638 = construct_float(param_12);
        float param_13[4] = ray.ior;
        bool param_14 = _7083;
        float param_15 = 1.0f;
        float _7647 = peek_ior_stack(param_13, param_14, param_15);
        float3 col = 0.0f.xxx;
        int _7652 = ray.depth & 255;
        int _7657 = (ray.depth >> 8) & 255;
        int _7662 = (ray.depth >> 16) & 255;
        int _7673 = (_7652 + _7657) + _7662;
        int _7681 = _3539_g_params.hi + ((_7673 + ((ray.depth >> 24) & 255)) * 7);
        float mix_rand = frac(asfloat(_3523.Load(_7681 * 4 + 0)) + _7631);
        float mix_weight = 1.0f;
        float _7718;
        float _7735;
        float _7761;
        float _7828;
        while (_9844 == 4u)
        {
            float mix_val = _9845;
            if (_10234 != 4294967295u)
            {
                mix_val *= SampleBilinear(_10234, _7391, 0).x;
            }
            if (_7083)
            {
                _7718 = _7647 / _9847;
            }
            else
            {
                _7718 = _9847 / _7647;
            }
            if (_9847 != 0.0f)
            {
                float param_16 = dot(_7014, _9782);
                float param_17 = _7718;
                _7735 = fresnel_dielectric_cos(param_16, param_17);
            }
            else
            {
                _7735 = 1.0f;
            }
            float _7750 = mix_val;
            float _7751 = _7750 * clamp(_7735, 0.0f, 1.0f);
            mix_val = _7751;
            if (mix_rand > _7751)
            {
                if ((_9843 & 2u) != 0u)
                {
                    _7761 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _7761 = 1.0f;
                }
                mix_weight *= _7761;
                material_t _7774;
                [unroll]
                for (int _108ident = 0; _108ident < 5; _108ident++)
                {
                    _7774.textures[_108ident] = _4758.Load(_108ident * 4 + _10236 * 76 + 0);
                }
                [unroll]
                for (int _109ident = 0; _109ident < 3; _109ident++)
                {
                    _7774.base_color[_109ident] = asfloat(_4758.Load(_109ident * 4 + _10236 * 76 + 20));
                }
                _7774.flags = _4758.Load(_10236 * 76 + 32);
                _7774.type = _4758.Load(_10236 * 76 + 36);
                _7774.tangent_rotation_or_strength = asfloat(_4758.Load(_10236 * 76 + 40));
                _7774.roughness_and_anisotropic = _4758.Load(_10236 * 76 + 44);
                _7774.ior = asfloat(_4758.Load(_10236 * 76 + 48));
                _7774.sheen_and_sheen_tint = _4758.Load(_10236 * 76 + 52);
                _7774.tint_and_metallic = _4758.Load(_10236 * 76 + 56);
                _7774.transmission_and_transmission_roughness = _4758.Load(_10236 * 76 + 60);
                _7774.specular_and_specular_tint = _4758.Load(_10236 * 76 + 64);
                _7774.clearcoat_and_clearcoat_roughness = _4758.Load(_10236 * 76 + 68);
                _7774.normal_map_strength_unorm = _4758.Load(_10236 * 76 + 72);
                _10233 = _7774.textures[0];
                _10234 = _7774.textures[1];
                _10235 = _7774.textures[2];
                _10236 = _7774.textures[3];
                _10237 = _7774.textures[4];
                _10238 = _7774.base_color[0];
                _10239 = _7774.base_color[1];
                _10240 = _7774.base_color[2];
                _9843 = _7774.flags;
                _9844 = _7774.type;
                _9845 = _7774.tangent_rotation_or_strength;
                _9846 = _7774.roughness_and_anisotropic;
                _9847 = _7774.ior;
                _9848 = _7774.sheen_and_sheen_tint;
                _9849 = _7774.tint_and_metallic;
                _9850 = _7774.transmission_and_transmission_roughness;
                _9851 = _7774.specular_and_specular_tint;
                _9852 = _7774.clearcoat_and_clearcoat_roughness;
                _9853 = _7774.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_9843 & 2u) != 0u)
                {
                    _7828 = 1.0f / mix_val;
                }
                else
                {
                    _7828 = 1.0f;
                }
                mix_weight *= _7828;
                material_t _7840;
                [unroll]
                for (int _110ident = 0; _110ident < 5; _110ident++)
                {
                    _7840.textures[_110ident] = _4758.Load(_110ident * 4 + _10237 * 76 + 0);
                }
                [unroll]
                for (int _111ident = 0; _111ident < 3; _111ident++)
                {
                    _7840.base_color[_111ident] = asfloat(_4758.Load(_111ident * 4 + _10237 * 76 + 20));
                }
                _7840.flags = _4758.Load(_10237 * 76 + 32);
                _7840.type = _4758.Load(_10237 * 76 + 36);
                _7840.tangent_rotation_or_strength = asfloat(_4758.Load(_10237 * 76 + 40));
                _7840.roughness_and_anisotropic = _4758.Load(_10237 * 76 + 44);
                _7840.ior = asfloat(_4758.Load(_10237 * 76 + 48));
                _7840.sheen_and_sheen_tint = _4758.Load(_10237 * 76 + 52);
                _7840.tint_and_metallic = _4758.Load(_10237 * 76 + 56);
                _7840.transmission_and_transmission_roughness = _4758.Load(_10237 * 76 + 60);
                _7840.specular_and_specular_tint = _4758.Load(_10237 * 76 + 64);
                _7840.clearcoat_and_clearcoat_roughness = _4758.Load(_10237 * 76 + 68);
                _7840.normal_map_strength_unorm = _4758.Load(_10237 * 76 + 72);
                _10233 = _7840.textures[0];
                _10234 = _7840.textures[1];
                _10235 = _7840.textures[2];
                _10236 = _7840.textures[3];
                _10237 = _7840.textures[4];
                _10238 = _7840.base_color[0];
                _10239 = _7840.base_color[1];
                _10240 = _7840.base_color[2];
                _9843 = _7840.flags;
                _9844 = _7840.type;
                _9845 = _7840.tangent_rotation_or_strength;
                _9846 = _7840.roughness_and_anisotropic;
                _9847 = _7840.ior;
                _9848 = _7840.sheen_and_sheen_tint;
                _9849 = _7840.tint_and_metallic;
                _9850 = _7840.transmission_and_transmission_roughness;
                _9851 = _7840.specular_and_specular_tint;
                _9852 = _7840.clearcoat_and_clearcoat_roughness;
                _9853 = _7840.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_10233 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_10233, _7391, 0).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_1003.Load(_10233 * 80 + 0) & 16384u) != 0u)
            {
                float3 _10805 = normals;
                _10805.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _10805;
            }
            float3 _7924 = _9782;
            _9782 = normalize(((_9780 * normals.x) + (_7924 * normals.z)) + (_9781 * normals.y));
            if ((_9853 & 65535u) != 65535u)
            {
                _9782 = normalize(_7924 + ((_9782 - _7924) * clamp(float(_9853 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_18 = _9783;
            float3 param_19 = -_7014;
            float3 param_20 = _9782;
            _9782 = ensure_valid_reflection(param_18, param_19, param_20);
        }
        float3 _7990 = ((_7310 * _7333) + (_7318 * inter.u)) + (_7326 * inter.v);
        float3 _7997 = float3(-_7990.z, 0.0f, _7990.x);
        float3 tangent = _7997;
        float3 param_21 = _7997;
        float4x4 param_22 = _7160.inv_xform;
        float3 _8003 = TransformNormal(param_21, param_22);
        tangent = _8003;
        float3 _8007 = cross(_8003, _9782);
        if (dot(_8007, _8007) == 0.0f)
        {
            float3 param_23 = _7990;
            float4x4 param_24 = _7160.inv_xform;
            tangent = TransformNormal(param_23, param_24);
        }
        if (_9845 != 0.0f)
        {
            float3 param_25 = tangent;
            float3 param_26 = _9782;
            float param_27 = _9845;
            tangent = rotate_around_axis(param_25, param_26, param_27);
        }
        float3 _8040 = normalize(cross(tangent, _9782));
        _9781 = _8040;
        _9780 = cross(_9782, _8040);
        float3 _9932 = 0.0f.xxx;
        float3 _9931 = 0.0f.xxx;
        float _9936 = 0.0f;
        float _9934 = 0.0f;
        float _9935 = 1.0f;
        bool _8056 = _3539_g_params.li_count != 0;
        bool _8062;
        if (_8056)
        {
            _8062 = _9844 != 3u;
        }
        else
        {
            _8062 = _8056;
        }
        float3 _9933;
        bool _9937;
        bool _9938;
        if (_8062)
        {
            float3 param_28 = _7050;
            float3 param_29 = _9780;
            float3 param_30 = _9781;
            float3 param_31 = _9782;
            int param_32 = _7681;
            float2 param_33 = float2(_7631, _7638);
            light_sample_t _9947 = { _9931, _9932, _9933, _9934, _9935, _9936, _9937, _9938 };
            light_sample_t param_34 = _9947;
            SampleLightSource(param_28, param_29, param_30, param_31, param_32, param_33, param_34);
            _9931 = param_34.col;
            _9932 = param_34.L;
            _9933 = param_34.lp;
            _9934 = param_34.area;
            _9935 = param_34.dist_mul;
            _9936 = param_34.pdf;
            _9937 = param_34.cast_shadow;
            _9938 = param_34.from_env;
        }
        float _8090 = dot(_9782, _9932);
        float3 base_color = float3(_10238, _10239, _10240);
        [branch]
        if (_10234 != 4294967295u)
        {
            base_color *= SampleBilinear(_10234, _7391, int(get_texture_lod(texSize(_10234), _7624)), true, true).xyz;
        }
        out_base_color = base_color;
        out_normals = _9782;
        float3 tint_color = 0.0f.xxx;
        float _8126 = lum(base_color);
        [flatten]
        if (_8126 > 0.0f)
        {
            tint_color = base_color / _8126.xxx;
        }
        float roughness = clamp(float(_9846 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_10235 != 4294967295u)
        {
            roughness *= SampleBilinear(_10235, _7391, int(get_texture_lod(texSize(_10235), _7624)), false, true).x;
        }
        float _8171 = frac(asfloat(_3523.Load((_7681 + 1) * 4 + 0)) + _7631);
        float _8180 = frac(asfloat(_3523.Load((_7681 + 2) * 4 + 0)) + _7638);
        float _10360 = 0.0f;
        float _10359 = 0.0f;
        float _10358 = 0.0f;
        float _9996[4];
        [unroll]
        for (int i = 0; i < 4; )
        {
            _9996[i] = ray.ior[i];
            i++;
            continue;
        }
        float _9997 = _7614;
        float _9998 = ray.cone_spread;
        int _9999 = ray.xy;
        float _9994 = 0.0f;
        float _10465 = 0.0f;
        float _10464 = 0.0f;
        float _10463 = 0.0f;
        int _10101 = ray.depth;
        int _10105 = ray.xy;
        int _10000;
        float _10103;
        float _10288;
        float _10289;
        float _10290;
        float _10323;
        float _10324;
        float _10325;
        float _10393;
        float _10394;
        float _10395;
        float _10428;
        float _10429;
        float _10430;
        [branch]
        if (_9844 == 0u)
        {
            [branch]
            if ((_9936 > 0.0f) && (_8090 > 0.0f))
            {
                light_sample_t _9964 = { _9931, _9932, _9933, _9934, _9935, _9936, _9937, _9938 };
                surface_t _9791 = { _7050, _9780, _9781, _9782, _9783, _7391 };
                float _10469[3] = { _10463, _10464, _10465 };
                float _10434[3] = { _10428, _10429, _10430 };
                float _10399[3] = { _10393, _10394, _10395 };
                shadow_ray_t _10115 = { _10399, _10101, _10434, _10103, _10469, _10105 };
                shadow_ray_t param_35 = _10115;
                float3 _8240 = Evaluate_DiffuseNode(_9964, ray, _9791, base_color, roughness, mix_weight, param_35);
                _10393 = param_35.o[0];
                _10394 = param_35.o[1];
                _10395 = param_35.o[2];
                _10101 = param_35.depth;
                _10428 = param_35.d[0];
                _10429 = param_35.d[1];
                _10430 = param_35.d[2];
                _10103 = param_35.dist;
                _10463 = param_35.c[0];
                _10464 = param_35.c[1];
                _10465 = param_35.c[2];
                _10105 = param_35.xy;
                col += _8240;
            }
            bool _8247 = _7652 < _3539_g_params.max_diff_depth;
            bool _8254;
            if (_8247)
            {
                _8254 = _7673 < _3539_g_params.max_total_depth;
            }
            else
            {
                _8254 = _8247;
            }
            [branch]
            if (_8254)
            {
                surface_t _9798 = { _7050, _9780, _9781, _9782, _9783, _7391 };
                float _10364[3] = { _10358, _10359, _10360 };
                float _10329[3] = { _10323, _10324, _10325 };
                float _10294[3] = { _10288, _10289, _10290 };
                ray_data_t _10014 = { _10294, _10329, _9994, _10364, _9996, _9997, _9998, _9999, _10000 };
                ray_data_t param_36 = _10014;
                Sample_DiffuseNode(ray, _9798, base_color, roughness, _8171, _8180, mix_weight, param_36);
                _10288 = param_36.o[0];
                _10289 = param_36.o[1];
                _10290 = param_36.o[2];
                _10323 = param_36.d[0];
                _10324 = param_36.d[1];
                _10325 = param_36.d[2];
                _9994 = param_36.pdf;
                _10358 = param_36.c[0];
                _10359 = param_36.c[1];
                _10360 = param_36.c[2];
                _9996 = param_36.ior;
                _9997 = param_36.cone_width;
                _9998 = param_36.cone_spread;
                _9999 = param_36.xy;
                _10000 = param_36.depth;
            }
        }
        else
        {
            [branch]
            if (_9844 == 1u)
            {
                float param_37 = 1.0f;
                float param_38 = 1.5f;
                float _8278 = fresnel_dielectric_cos(param_37, param_38);
                float _8282 = roughness * roughness;
                bool _8285 = _9936 > 0.0f;
                bool _8292;
                if (_8285)
                {
                    _8292 = (_8282 * _8282) >= 1.0000000116860974230803549289703e-07f;
                }
                else
                {
                    _8292 = _8285;
                }
                [branch]
                if (_8292 && (_8090 > 0.0f))
                {
                    light_sample_t _9973 = { _9931, _9932, _9933, _9934, _9935, _9936, _9937, _9938 };
                    surface_t _9805 = { _7050, _9780, _9781, _9782, _9783, _7391 };
                    float _10476[3] = { _10463, _10464, _10465 };
                    float _10441[3] = { _10428, _10429, _10430 };
                    float _10406[3] = { _10393, _10394, _10395 };
                    shadow_ray_t _10128 = { _10406, _10101, _10441, _10103, _10476, _10105 };
                    shadow_ray_t param_39 = _10128;
                    float3 _8307 = Evaluate_GlossyNode(_9973, ray, _9805, base_color, roughness, 1.5f, _8278, mix_weight, param_39);
                    _10393 = param_39.o[0];
                    _10394 = param_39.o[1];
                    _10395 = param_39.o[2];
                    _10101 = param_39.depth;
                    _10428 = param_39.d[0];
                    _10429 = param_39.d[1];
                    _10430 = param_39.d[2];
                    _10103 = param_39.dist;
                    _10463 = param_39.c[0];
                    _10464 = param_39.c[1];
                    _10465 = param_39.c[2];
                    _10105 = param_39.xy;
                    col += _8307;
                }
                bool _8314 = _7657 < _3539_g_params.max_spec_depth;
                bool _8321;
                if (_8314)
                {
                    _8321 = _7673 < _3539_g_params.max_total_depth;
                }
                else
                {
                    _8321 = _8314;
                }
                [branch]
                if (_8321)
                {
                    surface_t _9812 = { _7050, _9780, _9781, _9782, _9783, _7391 };
                    float _10371[3] = { _10358, _10359, _10360 };
                    float _10336[3] = { _10323, _10324, _10325 };
                    float _10301[3] = { _10288, _10289, _10290 };
                    ray_data_t _10033 = { _10301, _10336, _9994, _10371, _9996, _9997, _9998, _9999, _10000 };
                    ray_data_t param_40 = _10033;
                    Sample_GlossyNode(ray, _9812, base_color, roughness, 1.5f, _8278, _8171, _8180, mix_weight, param_40);
                    _10288 = param_40.o[0];
                    _10289 = param_40.o[1];
                    _10290 = param_40.o[2];
                    _10323 = param_40.d[0];
                    _10324 = param_40.d[1];
                    _10325 = param_40.d[2];
                    _9994 = param_40.pdf;
                    _10358 = param_40.c[0];
                    _10359 = param_40.c[1];
                    _10360 = param_40.c[2];
                    _9996 = param_40.ior;
                    _9997 = param_40.cone_width;
                    _9998 = param_40.cone_spread;
                    _9999 = param_40.xy;
                    _10000 = param_40.depth;
                }
            }
            else
            {
                [branch]
                if (_9844 == 2u)
                {
                    float _8345 = roughness * roughness;
                    bool _8348 = _9936 > 0.0f;
                    bool _8355;
                    if (_8348)
                    {
                        _8355 = (_8345 * _8345) >= 1.0000000116860974230803549289703e-07f;
                    }
                    else
                    {
                        _8355 = _8348;
                    }
                    [branch]
                    if (_8355 && (_8090 < 0.0f))
                    {
                        float _8363;
                        if (_7083)
                        {
                            _8363 = _9847 / _7647;
                        }
                        else
                        {
                            _8363 = _7647 / _9847;
                        }
                        light_sample_t _9982 = { _9931, _9932, _9933, _9934, _9935, _9936, _9937, _9938 };
                        surface_t _9819 = { _7050, _9780, _9781, _9782, _9783, _7391 };
                        float _10483[3] = { _10463, _10464, _10465 };
                        float _10448[3] = { _10428, _10429, _10430 };
                        float _10413[3] = { _10393, _10394, _10395 };
                        shadow_ray_t _10141 = { _10413, _10101, _10448, _10103, _10483, _10105 };
                        shadow_ray_t param_41 = _10141;
                        float3 _8385 = Evaluate_RefractiveNode(_9982, ray, _9819, base_color, _8345, _8363, mix_weight, param_41);
                        _10393 = param_41.o[0];
                        _10394 = param_41.o[1];
                        _10395 = param_41.o[2];
                        _10101 = param_41.depth;
                        _10428 = param_41.d[0];
                        _10429 = param_41.d[1];
                        _10430 = param_41.d[2];
                        _10103 = param_41.dist;
                        _10463 = param_41.c[0];
                        _10464 = param_41.c[1];
                        _10465 = param_41.c[2];
                        _10105 = param_41.xy;
                        col += _8385;
                    }
                    bool _8392 = _7662 < _3539_g_params.max_refr_depth;
                    bool _8399;
                    if (_8392)
                    {
                        _8399 = _7673 < _3539_g_params.max_total_depth;
                    }
                    else
                    {
                        _8399 = _8392;
                    }
                    [branch]
                    if (_8399)
                    {
                        surface_t _9826 = { _7050, _9780, _9781, _9782, _9783, _7391 };
                        float _10378[3] = { _10358, _10359, _10360 };
                        float _10343[3] = { _10323, _10324, _10325 };
                        float _10308[3] = { _10288, _10289, _10290 };
                        ray_data_t _10052 = { _10308, _10343, _9994, _10378, _9996, _9997, _9998, _9999, _10000 };
                        ray_data_t param_42 = _10052;
                        Sample_RefractiveNode(ray, _9826, base_color, roughness, _7083, _9847, _7647, _8171, _8180, mix_weight, param_42);
                        _10288 = param_42.o[0];
                        _10289 = param_42.o[1];
                        _10290 = param_42.o[2];
                        _10323 = param_42.d[0];
                        _10324 = param_42.d[1];
                        _10325 = param_42.d[2];
                        _9994 = param_42.pdf;
                        _10358 = param_42.c[0];
                        _10359 = param_42.c[1];
                        _10360 = param_42.c[2];
                        _9996 = param_42.ior;
                        _9997 = param_42.cone_width;
                        _9998 = param_42.cone_spread;
                        _9999 = param_42.xy;
                        _10000 = param_42.depth;
                    }
                }
                else
                {
                    [branch]
                    if (_9844 == 3u)
                    {
                        col += (base_color * (mix_weight * _9845));
                    }
                    else
                    {
                        [branch]
                        if (_9844 == 6u)
                        {
                            float metallic = clamp(float((_9849 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_10236 != 4294967295u)
                            {
                                metallic *= SampleBilinear(_10236, _7391, int(get_texture_lod(texSize(_10236), _7624))).x;
                            }
                            float specular = clamp(float(_9851 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_10237 != 4294967295u)
                            {
                                specular *= SampleBilinear(_10237, _7391, int(get_texture_lod(texSize(_10237), _7624))).x;
                            }
                            float _8518 = clamp(float(_9852 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8526 = clamp(float((_9852 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8534 = 2.0f * clamp(float(_9848 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float3 _8552 = lerp(1.0f.xxx, tint_color, clamp(float((_9848 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _8534;
                            float3 _8572 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_9851 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                            float _8581 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                            float param_43 = 1.0f;
                            float param_44 = _8581;
                            float _8587 = fresnel_dielectric_cos(param_43, param_44);
                            float _8595 = clamp(float((_9846 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8606 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _8518))) - 1.0f;
                            float param_45 = 1.0f;
                            float param_46 = _8606;
                            float _8612 = fresnel_dielectric_cos(param_45, param_46);
                            float _8627 = mad(roughness - 1.0f, 1.0f - clamp(float((_9850 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                            float _8633;
                            if (_7083)
                            {
                                _8633 = _9847 / _7647;
                            }
                            else
                            {
                                _8633 = _7647 / _9847;
                            }
                            float param_47 = dot(_7014, _9782);
                            float param_48 = 1.0f / _8633;
                            float _8656 = fresnel_dielectric_cos(param_47, param_48);
                            float param_49 = dot(_7014, _9782);
                            float param_50 = _8581;
                            lobe_weights_t _8695 = get_lobe_weights(lerp(_8126, 1.0f, _8534), lum(lerp(_8572, 1.0f.xxx, ((fresnel_dielectric_cos(param_49, param_50) - _8587) / (1.0f - _8587)).xxx)), specular, metallic, clamp(float(_9850 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _8518);
                            [branch]
                            if (_9936 > 0.0f)
                            {
                                light_sample_t _9991 = { _9931, _9932, _9933, _9934, _9935, _9936, _9937, _9938 };
                                surface_t _9833 = { _7050, _9780, _9781, _9782, _9783, _7391 };
                                diff_params_t _10183 = { base_color, _8552, roughness };
                                spec_params_t _10198 = { _8572, roughness, _8581, _8587, _8595 };
                                clearcoat_params_t _10211 = { _8526, _8606, _8612 };
                                transmission_params_t _10226 = { _8627, _9847, _8633, _8656, _7083 };
                                float _10490[3] = { _10463, _10464, _10465 };
                                float _10455[3] = { _10428, _10429, _10430 };
                                float _10420[3] = { _10393, _10394, _10395 };
                                shadow_ray_t _10154 = { _10420, _10101, _10455, _10103, _10490, _10105 };
                                shadow_ray_t param_51 = _10154;
                                float3 _8714 = Evaluate_PrincipledNode(_9991, ray, _9833, _8695, _10183, _10198, _10211, _10226, metallic, _8090, mix_weight, param_51);
                                _10393 = param_51.o[0];
                                _10394 = param_51.o[1];
                                _10395 = param_51.o[2];
                                _10101 = param_51.depth;
                                _10428 = param_51.d[0];
                                _10429 = param_51.d[1];
                                _10430 = param_51.d[2];
                                _10103 = param_51.dist;
                                _10463 = param_51.c[0];
                                _10464 = param_51.c[1];
                                _10465 = param_51.c[2];
                                _10105 = param_51.xy;
                                col += _8714;
                            }
                            surface_t _9840 = { _7050, _9780, _9781, _9782, _9783, _7391 };
                            diff_params_t _10187 = { base_color, _8552, roughness };
                            spec_params_t _10204 = { _8572, roughness, _8581, _8587, _8595 };
                            clearcoat_params_t _10215 = { _8526, _8606, _8612 };
                            transmission_params_t _10232 = { _8627, _9847, _8633, _8656, _7083 };
                            float param_52 = mix_rand;
                            float _10385[3] = { _10358, _10359, _10360 };
                            float _10350[3] = { _10323, _10324, _10325 };
                            float _10315[3] = { _10288, _10289, _10290 };
                            ray_data_t _10071 = { _10315, _10350, _9994, _10385, _9996, _9997, _9998, _9999, _10000 };
                            ray_data_t param_53 = _10071;
                            Sample_PrincipledNode(ray, _9840, _8695, _10187, _10204, _10215, _10232, metallic, _8171, _8180, param_52, mix_weight, param_53);
                            _10288 = param_53.o[0];
                            _10289 = param_53.o[1];
                            _10290 = param_53.o[2];
                            _10323 = param_53.d[0];
                            _10324 = param_53.d[1];
                            _10325 = param_53.d[2];
                            _9994 = param_53.pdf;
                            _10358 = param_53.c[0];
                            _10359 = param_53.c[1];
                            _10360 = param_53.c[2];
                            _9996 = param_53.ior;
                            _9997 = param_53.cone_width;
                            _9998 = param_53.cone_spread;
                            _9999 = param_53.xy;
                            _10000 = param_53.depth;
                        }
                    }
                }
            }
        }
        float _8748 = max(_10358, max(_10359, _10360));
        float _8760;
        if (_7673 > _3539_g_params.min_total_depth)
        {
            _8760 = max(0.0500000007450580596923828125f, 1.0f - _8748);
        }
        else
        {
            _8760 = 0.0f;
        }
        bool _8774 = (frac(asfloat(_3523.Load((_7681 + 6) * 4 + 0)) + _7631) >= _8760) && (_8748 > 0.0f);
        bool _8780;
        if (_8774)
        {
            _8780 = _9994 > 0.0f;
        }
        else
        {
            _8780 = _8774;
        }
        [branch]
        if (_8780)
        {
            float _8784 = _9994;
            float _8785 = min(_8784, 1000000.0f);
            _9994 = _8785;
            float _8788 = 1.0f - _8760;
            float _8790 = _10358;
            float _8791 = _8790 / _8788;
            _10358 = _8791;
            float _8796 = _10359;
            float _8797 = _8796 / _8788;
            _10359 = _8797;
            float _8802 = _10360;
            float _8803 = _8802 / _8788;
            _10360 = _8803;
            uint _8811;
            _8809.InterlockedAdd(0, 1u, _8811);
            _8820.Store(_8811 * 72 + 0, asuint(_10288));
            _8820.Store(_8811 * 72 + 4, asuint(_10289));
            _8820.Store(_8811 * 72 + 8, asuint(_10290));
            _8820.Store(_8811 * 72 + 12, asuint(_10323));
            _8820.Store(_8811 * 72 + 16, asuint(_10324));
            _8820.Store(_8811 * 72 + 20, asuint(_10325));
            _8820.Store(_8811 * 72 + 24, asuint(_8785));
            _8820.Store(_8811 * 72 + 28, asuint(_8791));
            _8820.Store(_8811 * 72 + 32, asuint(_8797));
            _8820.Store(_8811 * 72 + 36, asuint(_8803));
            _8820.Store(_8811 * 72 + 40, asuint(_9996[0]));
            _8820.Store(_8811 * 72 + 44, asuint(_9996[1]));
            _8820.Store(_8811 * 72 + 48, asuint(_9996[2]));
            _8820.Store(_8811 * 72 + 52, asuint(_9996[3]));
            _8820.Store(_8811 * 72 + 56, asuint(_9997));
            _8820.Store(_8811 * 72 + 60, asuint(_9998));
            _8820.Store(_8811 * 72 + 64, uint(_9999));
            _8820.Store(_8811 * 72 + 68, uint(_10000));
        }
        [branch]
        if (max(_10463, max(_10464, _10465)) > 0.0f)
        {
            float3 _8897 = _9933 - float3(_10393, _10394, _10395);
            float _8900 = length(_8897);
            float3 _8904 = _8897 / _8900.xxx;
            float sh_dist = _8900 * _9935;
            if (_9938)
            {
                sh_dist = -sh_dist;
            }
            float _8916 = _8904.x;
            _10428 = _8916;
            float _8919 = _8904.y;
            _10429 = _8919;
            float _8922 = _8904.z;
            _10430 = _8922;
            _10103 = sh_dist;
            uint _8928;
            _8809.InterlockedAdd(8, 1u, _8928);
            _8936.Store(_8928 * 48 + 0, asuint(_10393));
            _8936.Store(_8928 * 48 + 4, asuint(_10394));
            _8936.Store(_8928 * 48 + 8, asuint(_10395));
            _8936.Store(_8928 * 48 + 12, uint(_10101));
            _8936.Store(_8928 * 48 + 16, asuint(_8916));
            _8936.Store(_8928 * 48 + 20, asuint(_8919));
            _8936.Store(_8928 * 48 + 24, asuint(_8922));
            _8936.Store(_8928 * 48 + 28, asuint(sh_dist));
            _8936.Store(_8928 * 48 + 32, asuint(_10463));
            _8936.Store(_8928 * 48 + 36, asuint(_10464));
            _8936.Store(_8928 * 48 + 40, asuint(_10465));
            _8936.Store(_8928 * 48 + 44, uint(_10105));
        }
        _9178 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _9178;
}

void comp_main()
{
    do
    {
        int _9002 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_9002) >= _8809.Load(4))
        {
            break;
        }
        int _9018 = int(_9015.Load(_9002 * 72 + 64));
        int _9025 = int(_9015.Load(_9002 * 72 + 64));
        hit_data_t _9036;
        _9036.mask = int(_9032.Load(_9002 * 24 + 0));
        _9036.obj_index = int(_9032.Load(_9002 * 24 + 4));
        _9036.prim_index = int(_9032.Load(_9002 * 24 + 8));
        _9036.t = asfloat(_9032.Load(_9002 * 24 + 12));
        _9036.u = asfloat(_9032.Load(_9002 * 24 + 16));
        _9036.v = asfloat(_9032.Load(_9002 * 24 + 20));
        ray_data_t _9052;
        [unroll]
        for (int _112ident = 0; _112ident < 3; _112ident++)
        {
            _9052.o[_112ident] = asfloat(_9015.Load(_112ident * 4 + _9002 * 72 + 0));
        }
        [unroll]
        for (int _113ident = 0; _113ident < 3; _113ident++)
        {
            _9052.d[_113ident] = asfloat(_9015.Load(_113ident * 4 + _9002 * 72 + 12));
        }
        _9052.pdf = asfloat(_9015.Load(_9002 * 72 + 24));
        [unroll]
        for (int _114ident = 0; _114ident < 3; _114ident++)
        {
            _9052.c[_114ident] = asfloat(_9015.Load(_114ident * 4 + _9002 * 72 + 28));
        }
        [unroll]
        for (int _115ident = 0; _115ident < 4; _115ident++)
        {
            _9052.ior[_115ident] = asfloat(_9015.Load(_115ident * 4 + _9002 * 72 + 40));
        }
        _9052.cone_width = asfloat(_9015.Load(_9002 * 72 + 56));
        _9052.cone_spread = asfloat(_9015.Load(_9002 * 72 + 60));
        _9052.xy = int(_9015.Load(_9002 * 72 + 64));
        _9052.depth = int(_9015.Load(_9002 * 72 + 68));
        hit_data_t _9272 = { _9036.mask, _9036.obj_index, _9036.prim_index, _9036.t, _9036.u, _9036.v };
        hit_data_t param = _9272;
        float _9321[4] = { _9052.ior[0], _9052.ior[1], _9052.ior[2], _9052.ior[3] };
        float _9312[3] = { _9052.c[0], _9052.c[1], _9052.c[2] };
        float _9305[3] = { _9052.d[0], _9052.d[1], _9052.d[2] };
        float _9298[3] = { _9052.o[0], _9052.o[1], _9052.o[2] };
        ray_data_t _9291 = { _9298, _9305, _9052.pdf, _9312, _9321, _9052.cone_width, _9052.cone_spread, _9052.xy, _9052.depth };
        ray_data_t param_1 = _9291;
        float3 param_2 = 0.0f.xxx;
        float3 param_3 = 0.0f.xxx;
        float3 _9108 = ShadeSurface(param, param_1, param_2, param_3);
        int2 _9122 = int2((_9018 >> 16) & 65535, _9025 & 65535);
        g_out_img[_9122] = float4(min(_9108, _3539_g_params.clamp_val.xxx), 1.0f);
        g_out_base_color_img[_9122] = float4(param_2, 0.0f);
        g_out_depth_normals_img[_9122] = float4(param_3, _9036.t);
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
