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
    uint2 img_size;
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
    float4 env_col;
    float4 back_col;
    float env_rotation;
    float back_rotation;
    int env_mult_importance;
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
ByteAddressBuffer _3559 : register(t8, space0);
ByteAddressBuffer _3563 : register(t9, space0);
ByteAddressBuffer _4406 : register(t13, space0);
ByteAddressBuffer _4431 : register(t15, space0);
ByteAddressBuffer _4435 : register(t16, space0);
ByteAddressBuffer _4759 : register(t12, space0);
ByteAddressBuffer _4763 : register(t11, space0);
ByteAddressBuffer _7154 : register(t14, space0);
RWByteAddressBuffer _8810 : register(u3, space0);
RWByteAddressBuffer _8821 : register(u1, space0);
RWByteAddressBuffer _8937 : register(u2, space0);
ByteAddressBuffer _9035 : register(t6, space0);
ByteAddressBuffer _9056 : register(t7, space0);
ByteAddressBuffer _9173 : register(t10, space0);
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

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

float2 TransformUV(float2 _uv, atlas_texture_t t, int mip_level)
{
    uint _9348[14] = t.pos;
    uint _9351[14] = t.pos;
    uint _1096 = t.size & 16383u;
    uint _1099 = t.size >> uint(16);
    uint _1100 = _1099 & 16383u;
    float2 size = float2(float(_1096), float(_1100));
    if ((_1099 & 32768u) != 0u)
    {
        size = float2(float(_1096 >> uint(mip_level)), float(_1100 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_9348[mip_level] & 65535u), float((_9351[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
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
    float3 _5013 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 env_col = _3539_g_params.back_col.xyz;
    uint _5021 = asuint(_3539_g_params.back_col.w);
    if (_5021 != 4294967295u)
    {
        atlas_texture_t _5032;
        _5032.size = _1003.Load(_5021 * 80 + 0);
        _5032.atlas = _1003.Load(_5021 * 80 + 4);
        [unroll]
        for (int _58ident = 0; _58ident < 4; _58ident++)
        {
            _5032.page[_58ident] = _1003.Load(_58ident * 4 + _5021 * 80 + 8);
        }
        [unroll]
        for (int _59ident = 0; _59ident < 14; _59ident++)
        {
            _5032.pos[_59ident] = _1003.Load(_59ident * 4 + _5021 * 80 + 24);
        }
        uint _9718[14] = { _5032.pos[0], _5032.pos[1], _5032.pos[2], _5032.pos[3], _5032.pos[4], _5032.pos[5], _5032.pos[6], _5032.pos[7], _5032.pos[8], _5032.pos[9], _5032.pos[10], _5032.pos[11], _5032.pos[12], _5032.pos[13] };
        uint _9689[4] = { _5032.page[0], _5032.page[1], _5032.page[2], _5032.page[3] };
        atlas_texture_t _9680 = { _5032.size, _5032.atlas, _9689, _9718 };
        float param = _3539_g_params.back_rotation;
        env_col *= SampleLatlong_RGBE(_9680, _5013, param);
    }
    if (_3539_g_params.env_qtree_levels > 0)
    {
        float param_1 = ray.pdf;
        float param_2 = Evaluate_EnvQTree(_3539_g_params.back_rotation, g_env_qtree, _g_env_qtree_sampler, _3539_g_params.env_qtree_levels, _5013);
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
    float3 _5144 = float3(ray.d[0], ray.d[1], ray.d[2]);
    light_t _5158;
    _5158.type_and_param0 = _3559.Load4(((-1) - inter.obj_index) * 64 + 0);
    _5158.param1 = asfloat(_3559.Load4(((-1) - inter.obj_index) * 64 + 16));
    _5158.param2 = asfloat(_3559.Load4(((-1) - inter.obj_index) * 64 + 32));
    _5158.param3 = asfloat(_3559.Load4(((-1) - inter.obj_index) * 64 + 48));
    float3 lcol = asfloat(_5158.type_and_param0.yzw);
    [branch]
    if ((_5158.type_and_param0.x & 128u) != 0u)
    {
        float3 env_col = _3539_g_params.env_col.xyz;
        uint _5185 = asuint(_3539_g_params.env_col.w);
        if (_5185 != 4294967295u)
        {
            atlas_texture_t _5192;
            _5192.size = _1003.Load(_5185 * 80 + 0);
            _5192.atlas = _1003.Load(_5185 * 80 + 4);
            [unroll]
            for (int _60ident = 0; _60ident < 4; _60ident++)
            {
                _5192.page[_60ident] = _1003.Load(_60ident * 4 + _5185 * 80 + 8);
            }
            [unroll]
            for (int _61ident = 0; _61ident < 14; _61ident++)
            {
                _5192.pos[_61ident] = _1003.Load(_61ident * 4 + _5185 * 80 + 24);
            }
            uint _9780[14] = { _5192.pos[0], _5192.pos[1], _5192.pos[2], _5192.pos[3], _5192.pos[4], _5192.pos[5], _5192.pos[6], _5192.pos[7], _5192.pos[8], _5192.pos[9], _5192.pos[10], _5192.pos[11], _5192.pos[12], _5192.pos[13] };
            uint _9751[4] = { _5192.page[0], _5192.page[1], _5192.page[2], _5192.page[3] };
            atlas_texture_t _9742 = { _5192.size, _5192.atlas, _9751, _9780 };
            float param = _3539_g_params.env_rotation;
            env_col *= SampleLatlong_RGBE(_9742, _5144, param);
        }
        lcol *= env_col;
    }
    uint _5252 = _5158.type_and_param0.x & 31u;
    if (_5252 == 0u)
    {
        float param_1 = ray.pdf;
        float param_2 = (inter.t * inter.t) / ((0.5f * _5158.param1.w) * dot(_5144, normalize(_5158.param1.xyz - (float3(ray.o[0], ray.o[1], ray.o[2]) + (_5144 * inter.t)))));
        lcol *= power_heuristic(param_1, param_2);
        bool _5319 = _5158.param3.x > 0.0f;
        bool _5325;
        if (_5319)
        {
            _5325 = _5158.param3.y > 0.0f;
        }
        else
        {
            _5325 = _5319;
        }
        [branch]
        if (_5325)
        {
            [flatten]
            if (_5158.param3.y > 0.0f)
            {
                lcol *= clamp((_5158.param3.x - acos(clamp(-dot(_5144, _5158.param2.xyz), 0.0f, 1.0f))) / _5158.param3.y, 0.0f, 1.0f);
            }
        }
    }
    else
    {
        if (_5252 == 4u)
        {
            float param_3 = ray.pdf;
            float param_4 = (inter.t * inter.t) / (_5158.param1.w * dot(_5144, normalize(cross(_5158.param2.xyz, _5158.param3.xyz))));
            lcol *= power_heuristic(param_3, param_4);
        }
        else
        {
            if (_5252 == 5u)
            {
                float param_5 = ray.pdf;
                float param_6 = (inter.t * inter.t) / (_5158.param1.w * dot(_5144, normalize(cross(_5158.param2.xyz, _5158.param3.xyz))));
                lcol *= power_heuristic(param_5, param_6);
            }
            else
            {
                if (_5252 == 3u)
                {
                    float param_7 = ray.pdf;
                    float param_8 = (inter.t * inter.t) / (_5158.param1.w * (1.0f - abs(dot(_5144, _5158.param3.xyz))));
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
    float _9185;
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
            _9185 = stack[3];
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
            _9185 = stack[2];
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
            _9185 = stack[1];
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
            _9185 = stack[0];
            break;
        }
        _9185 = default_value;
        break;
    } while(false);
    return _9185;
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
    uint _9356[4];
    _9356[0] = _1133.page[0];
    _9356[1] = _1133.page[1];
    _9356[2] = _1133.page[2];
    _9356[3] = _1133.page[3];
    uint _9392[14] = { _1133.pos[0], _1133.pos[1], _1133.pos[2], _1133.pos[3], _1133.pos[4], _1133.pos[5], _1133.pos[6], _1133.pos[7], _1133.pos[8], _1133.pos[9], _1133.pos[10], _1133.pos[11], _1133.pos[12], _1133.pos[13] };
    atlas_texture_t _9362 = { _1133.size, _1133.atlas, _9356, _9392 };
    uint _1203 = _1133.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_1203)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_1203)], float3(TransformUV(uvs, _9362, lod) * 0.000118371215648949146270751953125f.xx, float((_9356[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
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
        float4 _10535 = res;
        _10535.x = _1243.x;
        float4 _10537 = _10535;
        _10537.y = _1243.y;
        float4 _10539 = _10537;
        _10539.z = _1243.z;
        res = _10539;
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
    float3 _9190;
    do
    {
        float _1512 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1512)
        {
            _9190 = N;
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
            float _10838 = (-0.5f) / _1552;
            float param_1 = mad(_10838, _1576, 1.0f);
            float _1608 = safe_sqrtf(param_1);
            float param_2 = _1577;
            float _1611 = safe_sqrtf(param_2);
            float2 _1612 = float2(_1608, _1611);
            float param_3 = mad(_10838, _1583, 1.0f);
            float _1617 = safe_sqrtf(param_3);
            float param_4 = _1584;
            float _1620 = safe_sqrtf(param_4);
            float2 _1621 = float2(_1617, _1620);
            float _10840 = -_1540;
            float _1637 = mad(2.0f * mad(_1608, _1536, _1611 * _1540), _1611, _10840);
            float _1653 = mad(2.0f * mad(_1617, _1536, _1620 * _1540), _1620, _10840);
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
                _9190 = Ng;
                break;
            }
            float _1690 = valid1 ? _1577 : _1584;
            float param_5 = 1.0f - _1690;
            float param_6 = _1690;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _9190 = (_1532 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _9190;
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
    float3 _9215;
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
            _9215 = N;
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
        _9215 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _9215;
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
            float2 _10522 = origin;
            _10522.x = origin.x + _step;
            origin = _10522;
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
            float2 _10525 = origin;
            _10525.y = origin.y + _step;
            origin = _10525;
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
    light_t _3570;
    _3570.type_and_param0 = _3559.Load4(_3563.Load(_3550 * 4 + 0) * 64 + 0);
    _3570.param1 = asfloat(_3559.Load4(_3563.Load(_3550 * 4 + 0) * 64 + 16));
    _3570.param2 = asfloat(_3559.Load4(_3563.Load(_3550 * 4 + 0) * 64 + 32));
    _3570.param3 = asfloat(_3559.Load4(_3563.Load(_3550 * 4 + 0) * 64 + 48));
    ls.col = asfloat(_3570.type_and_param0.yzw);
    ls.col *= _3543;
    ls.cast_shadow = (_3570.type_and_param0.x & 32u) != 0u;
    ls.from_env = false;
    uint _3604 = _3570.type_and_param0.x & 31u;
    [branch]
    if (_3604 == 0u)
    {
        float _3617 = frac(asfloat(_3523.Load((hi + 4) * 4 + 0)) + sample_off.x);
        float3 _3632 = P - _3570.param1.xyz;
        float3 _3639 = _3632 / length(_3632).xxx;
        float _3646 = sqrt(clamp(mad(-_3617, _3617, 1.0f), 0.0f, 1.0f));
        float _3649 = 6.283185482025146484375f * frac(asfloat(_3523.Load((hi + 5) * 4 + 0)) + sample_off.y);
        float3 sampled_dir = float3(_3646 * cos(_3649), _3646 * sin(_3649), _3617);
        float3 param;
        float3 param_1;
        create_tbn(_3639, param, param_1);
        float3 _10602 = sampled_dir;
        float3 _3682 = ((param * _10602.x) + (param_1 * _10602.y)) + (_3639 * _10602.z);
        sampled_dir = _3682;
        float3 _3691 = _3570.param1.xyz + (_3682 * _3570.param2.w);
        float3 _3698 = normalize(_3691 - _3570.param1.xyz);
        float3 param_2 = _3691;
        float3 param_3 = _3698;
        ls.lp = offset_ray(param_2, param_3);
        ls.L = _3691 - P;
        float3 _3711 = ls.L;
        float _3712 = length(_3711);
        ls.L /= _3712.xxx;
        ls.area = _3570.param1.w;
        float _3727 = abs(dot(ls.L, _3698));
        [flatten]
        if (_3727 > 0.0f)
        {
            ls.pdf = (_3712 * _3712) / ((0.5f * ls.area) * _3727);
        }
        [branch]
        if (_3570.param3.x > 0.0f)
        {
            float _3754 = -dot(ls.L, _3570.param2.xyz);
            if (_3754 > 0.0f)
            {
                ls.col *= clamp((_3570.param3.x - acos(clamp(_3754, 0.0f, 1.0f))) / _3570.param3.y, 0.0f, 1.0f);
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
        if (_3604 == 2u)
        {
            ls.L = _3570.param1.xyz;
            if (_3570.param1.w != 0.0f)
            {
                float param_4 = frac(asfloat(_3523.Load((hi + 4) * 4 + 0)) + sample_off.x);
                float param_5 = frac(asfloat(_3523.Load((hi + 5) * 4 + 0)) + sample_off.y);
                float3 param_6 = ls.L;
                float param_7 = tan(_3570.param1.w);
                ls.L = normalize(MapToCone(param_4, param_5, param_6, param_7));
            }
            ls.area = 0.0f;
            ls.lp = P + ls.L;
            ls.dist_mul = 3402823346297367662189621542912.0f;
            ls.pdf = 1.0f;
            if ((_3570.type_and_param0.x & 64u) == 0u)
            {
                ls.area = 0.0f;
            }
        }
        else
        {
            [branch]
            if (_3604 == 4u)
            {
                float3 _3891 = (_3570.param1.xyz + (_3570.param2.xyz * (frac(asfloat(_3523.Load((hi + 4) * 4 + 0)) + sample_off.x) - 0.5f))) + (_3570.param3.xyz * (frac(asfloat(_3523.Load((hi + 5) * 4 + 0)) + sample_off.y) - 0.5f));
                float3 _3896 = normalize(cross(_3570.param2.xyz, _3570.param3.xyz));
                float3 param_8 = _3891;
                float3 param_9 = _3896;
                ls.lp = offset_ray(param_8, param_9);
                ls.L = _3891 - P;
                float3 _3909 = ls.L;
                float _3910 = length(_3909);
                ls.L /= _3910.xxx;
                ls.area = _3570.param1.w;
                float _3925 = dot(-ls.L, _3896);
                if (_3925 > 0.0f)
                {
                    ls.pdf = (_3910 * _3910) / (ls.area * _3925);
                }
                if ((_3570.type_and_param0.x & 64u) == 0u)
                {
                    ls.area = 0.0f;
                }
                [branch]
                if ((_3570.type_and_param0.x & 128u) != 0u)
                {
                    float3 env_col = _3539_g_params.env_col.xyz;
                    uint _3962 = asuint(_3539_g_params.env_col.w);
                    if (_3962 != 4294967295u)
                    {
                        atlas_texture_t _3970;
                        _3970.size = _1003.Load(_3962 * 80 + 0);
                        _3970.atlas = _1003.Load(_3962 * 80 + 4);
                        [unroll]
                        for (int _64ident = 0; _64ident < 4; _64ident++)
                        {
                            _3970.page[_64ident] = _1003.Load(_64ident * 4 + _3962 * 80 + 8);
                        }
                        [unroll]
                        for (int _65ident = 0; _65ident < 14; _65ident++)
                        {
                            _3970.pos[_65ident] = _1003.Load(_65ident * 4 + _3962 * 80 + 24);
                        }
                        uint _9536[14] = { _3970.pos[0], _3970.pos[1], _3970.pos[2], _3970.pos[3], _3970.pos[4], _3970.pos[5], _3970.pos[6], _3970.pos[7], _3970.pos[8], _3970.pos[9], _3970.pos[10], _3970.pos[11], _3970.pos[12], _3970.pos[13] };
                        uint _9507[4] = { _3970.page[0], _3970.page[1], _3970.page[2], _3970.page[3] };
                        atlas_texture_t _9436 = { _3970.size, _3970.atlas, _9507, _9536 };
                        float param_10 = _3539_g_params.env_rotation;
                        env_col *= SampleLatlong_RGBE(_9436, ls.L, param_10);
                    }
                    ls.col *= env_col;
                    ls.from_env = true;
                }
            }
            else
            {
                [branch]
                if (_3604 == 5u)
                {
                    float2 _4073 = (float2(frac(asfloat(_3523.Load((hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_3523.Load((hi + 5) * 4 + 0)) + sample_off.y)) * 2.0f) - 1.0f.xx;
                    float2 offset = _4073;
                    bool _4076 = _4073.x != 0.0f;
                    bool _4082;
                    if (_4076)
                    {
                        _4082 = offset.y != 0.0f;
                    }
                    else
                    {
                        _4082 = _4076;
                    }
                    if (_4082)
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
                        float _4115 = 0.5f * r;
                        offset = float2(_4115 * cos(theta), _4115 * sin(theta));
                    }
                    float3 _4137 = (_3570.param1.xyz + (_3570.param2.xyz * offset.x)) + (_3570.param3.xyz * offset.y);
                    float3 _4142 = normalize(cross(_3570.param2.xyz, _3570.param3.xyz));
                    float3 param_11 = _4137;
                    float3 param_12 = _4142;
                    ls.lp = offset_ray(param_11, param_12);
                    ls.L = _4137 - P;
                    float3 _4155 = ls.L;
                    float _4156 = length(_4155);
                    ls.L /= _4156.xxx;
                    ls.area = _3570.param1.w;
                    float _4171 = dot(-ls.L, _4142);
                    [flatten]
                    if (_4171 > 0.0f)
                    {
                        ls.pdf = (_4156 * _4156) / (ls.area * _4171);
                    }
                    if ((_3570.type_and_param0.x & 64u) == 0u)
                    {
                        ls.area = 0.0f;
                    }
                    [branch]
                    if ((_3570.type_and_param0.x & 128u) != 0u)
                    {
                        float3 env_col_1 = _3539_g_params.env_col.xyz;
                        uint _4205 = asuint(_3539_g_params.env_col.w);
                        if (_4205 != 4294967295u)
                        {
                            atlas_texture_t _4212;
                            _4212.size = _1003.Load(_4205 * 80 + 0);
                            _4212.atlas = _1003.Load(_4205 * 80 + 4);
                            [unroll]
                            for (int _66ident = 0; _66ident < 4; _66ident++)
                            {
                                _4212.page[_66ident] = _1003.Load(_66ident * 4 + _4205 * 80 + 8);
                            }
                            [unroll]
                            for (int _67ident = 0; _67ident < 14; _67ident++)
                            {
                                _4212.pos[_67ident] = _1003.Load(_67ident * 4 + _4205 * 80 + 24);
                            }
                            uint _9574[14] = { _4212.pos[0], _4212.pos[1], _4212.pos[2], _4212.pos[3], _4212.pos[4], _4212.pos[5], _4212.pos[6], _4212.pos[7], _4212.pos[8], _4212.pos[9], _4212.pos[10], _4212.pos[11], _4212.pos[12], _4212.pos[13] };
                            uint _9545[4] = { _4212.page[0], _4212.page[1], _4212.page[2], _4212.page[3] };
                            atlas_texture_t _9445 = { _4212.size, _4212.atlas, _9545, _9574 };
                            float param_13 = _3539_g_params.env_rotation;
                            env_col_1 *= SampleLatlong_RGBE(_9445, ls.L, param_13);
                        }
                        ls.col *= env_col_1;
                        ls.from_env = true;
                    }
                }
                else
                {
                    [branch]
                    if (_3604 == 3u)
                    {
                        float3 _4312 = normalize(cross(P - _3570.param1.xyz, _3570.param3.xyz));
                        float _4319 = 3.1415927410125732421875f * frac(asfloat(_3523.Load((hi + 4) * 4 + 0)) + sample_off.x);
                        float3 _4344 = (_3570.param1.xyz + (((_4312 * cos(_4319)) + (cross(_4312, _3570.param3.xyz) * sin(_4319))) * _3570.param2.w)) + ((_3570.param3.xyz * (frac(asfloat(_3523.Load((hi + 5) * 4 + 0)) + sample_off.y) - 0.5f)) * _3570.param3.w);
                        ls.lp = _4344;
                        float3 _4350 = _4344 - P;
                        float _4353 = length(_4350);
                        ls.L = _4350 / _4353.xxx;
                        ls.area = _3570.param1.w;
                        float _4368 = 1.0f - abs(dot(ls.L, _3570.param3.xyz));
                        [flatten]
                        if (_4368 != 0.0f)
                        {
                            ls.pdf = (_4353 * _4353) / (ls.area * _4368);
                        }
                        if ((_3570.type_and_param0.x & 64u) == 0u)
                        {
                            ls.area = 0.0f;
                        }
                    }
                    else
                    {
                        [branch]
                        if (_3604 == 6u)
                        {
                            uint _4398 = asuint(_3570.param1.x);
                            transform_t _4412;
                            _4412.xform = asfloat(uint4x4(_4406.Load4(asuint(_3570.param1.y) * 128 + 0), _4406.Load4(asuint(_3570.param1.y) * 128 + 16), _4406.Load4(asuint(_3570.param1.y) * 128 + 32), _4406.Load4(asuint(_3570.param1.y) * 128 + 48)));
                            _4412.inv_xform = asfloat(uint4x4(_4406.Load4(asuint(_3570.param1.y) * 128 + 64), _4406.Load4(asuint(_3570.param1.y) * 128 + 80), _4406.Load4(asuint(_3570.param1.y) * 128 + 96), _4406.Load4(asuint(_3570.param1.y) * 128 + 112)));
                            uint _4437 = _4398 * 3u;
                            vertex_t _4443;
                            [unroll]
                            for (int _68ident = 0; _68ident < 3; _68ident++)
                            {
                                _4443.p[_68ident] = asfloat(_4431.Load(_68ident * 4 + _4435.Load(_4437 * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _69ident = 0; _69ident < 3; _69ident++)
                            {
                                _4443.n[_69ident] = asfloat(_4431.Load(_69ident * 4 + _4435.Load(_4437 * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _70ident = 0; _70ident < 3; _70ident++)
                            {
                                _4443.b[_70ident] = asfloat(_4431.Load(_70ident * 4 + _4435.Load(_4437 * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _71ident = 0; _71ident < 2; _71ident++)
                            {
                                [unroll]
                                for (int _72ident = 0; _72ident < 2; _72ident++)
                                {
                                    _4443.t[_71ident][_72ident] = asfloat(_4431.Load(_72ident * 4 + _71ident * 8 + _4435.Load(_4437 * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4492;
                            [unroll]
                            for (int _73ident = 0; _73ident < 3; _73ident++)
                            {
                                _4492.p[_73ident] = asfloat(_4431.Load(_73ident * 4 + _4435.Load((_4437 + 1u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _74ident = 0; _74ident < 3; _74ident++)
                            {
                                _4492.n[_74ident] = asfloat(_4431.Load(_74ident * 4 + _4435.Load((_4437 + 1u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _75ident = 0; _75ident < 3; _75ident++)
                            {
                                _4492.b[_75ident] = asfloat(_4431.Load(_75ident * 4 + _4435.Load((_4437 + 1u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _76ident = 0; _76ident < 2; _76ident++)
                            {
                                [unroll]
                                for (int _77ident = 0; _77ident < 2; _77ident++)
                                {
                                    _4492.t[_76ident][_77ident] = asfloat(_4431.Load(_77ident * 4 + _76ident * 8 + _4435.Load((_4437 + 1u) * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4538;
                            [unroll]
                            for (int _78ident = 0; _78ident < 3; _78ident++)
                            {
                                _4538.p[_78ident] = asfloat(_4431.Load(_78ident * 4 + _4435.Load((_4437 + 2u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _79ident = 0; _79ident < 3; _79ident++)
                            {
                                _4538.n[_79ident] = asfloat(_4431.Load(_79ident * 4 + _4435.Load((_4437 + 2u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _80ident = 0; _80ident < 3; _80ident++)
                            {
                                _4538.b[_80ident] = asfloat(_4431.Load(_80ident * 4 + _4435.Load((_4437 + 2u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _81ident = 0; _81ident < 2; _81ident++)
                            {
                                [unroll]
                                for (int _82ident = 0; _82ident < 2; _82ident++)
                                {
                                    _4538.t[_81ident][_82ident] = asfloat(_4431.Load(_82ident * 4 + _81ident * 8 + _4435.Load((_4437 + 2u) * 4 + 0) * 52 + 36));
                                }
                            }
                            float3 _4584 = float3(_4443.p[0], _4443.p[1], _4443.p[2]);
                            float3 _4592 = float3(_4492.p[0], _4492.p[1], _4492.p[2]);
                            float3 _4600 = float3(_4538.p[0], _4538.p[1], _4538.p[2]);
                            float _4628 = sqrt(frac(asfloat(_3523.Load((hi + 4) * 4 + 0)) + sample_off.x));
                            float _4637 = frac(asfloat(_3523.Load((hi + 5) * 4 + 0)) + sample_off.y);
                            float _4641 = 1.0f - _4628;
                            float _4646 = 1.0f - _4637;
                            float3 _4677 = mul(float4((_4584 * _4641) + (((_4592 * _4646) + (_4600 * _4637)) * _4628), 1.0f), _4412.xform).xyz;
                            float3 _4693 = mul(float4(cross(_4592 - _4584, _4600 - _4584), 0.0f), _4412.xform).xyz;
                            ls.area = 0.5f * length(_4693);
                            float3 _4699 = normalize(_4693);
                            ls.L = _4677 - P;
                            float3 _4706 = ls.L;
                            float _4707 = length(_4706);
                            ls.L /= _4707.xxx;
                            float _4718 = dot(ls.L, _4699);
                            float cos_theta = _4718;
                            float3 _4721;
                            if (_4718 >= 0.0f)
                            {
                                _4721 = -_4699;
                            }
                            else
                            {
                                _4721 = _4699;
                            }
                            float3 param_14 = _4677;
                            float3 param_15 = _4721;
                            ls.lp = offset_ray(param_14, param_15);
                            float _4734 = cos_theta;
                            float _4735 = abs(_4734);
                            cos_theta = _4735;
                            [flatten]
                            if (_4735 > 0.0f)
                            {
                                ls.pdf = (_4707 * _4707) / (ls.area * cos_theta);
                            }
                            material_t _4772;
                            [unroll]
                            for (int _83ident = 0; _83ident < 5; _83ident++)
                            {
                                _4772.textures[_83ident] = _4759.Load(_83ident * 4 + ((_4763.Load(_4398 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
                            }
                            [unroll]
                            for (int _84ident = 0; _84ident < 3; _84ident++)
                            {
                                _4772.base_color[_84ident] = asfloat(_4759.Load(_84ident * 4 + ((_4763.Load(_4398 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
                            }
                            _4772.flags = _4759.Load(((_4763.Load(_4398 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
                            _4772.type = _4759.Load(((_4763.Load(_4398 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
                            _4772.tangent_rotation_or_strength = asfloat(_4759.Load(((_4763.Load(_4398 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
                            _4772.roughness_and_anisotropic = _4759.Load(((_4763.Load(_4398 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
                            _4772.ior = asfloat(_4759.Load(((_4763.Load(_4398 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
                            _4772.sheen_and_sheen_tint = _4759.Load(((_4763.Load(_4398 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
                            _4772.tint_and_metallic = _4759.Load(((_4763.Load(_4398 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
                            _4772.transmission_and_transmission_roughness = _4759.Load(((_4763.Load(_4398 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
                            _4772.specular_and_specular_tint = _4759.Load(((_4763.Load(_4398 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
                            _4772.clearcoat_and_clearcoat_roughness = _4759.Load(((_4763.Load(_4398 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
                            _4772.normal_map_strength_unorm = _4759.Load(((_4763.Load(_4398 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
                            if (_4772.textures[1] != 4294967295u)
                            {
                                ls.col *= SampleBilinear(_4772.textures[1], (float2(_4443.t[0][0], _4443.t[0][1]) * _4641) + (((float2(_4492.t[0][0], _4492.t[0][1]) * _4646) + (float2(_4538.t[0][0], _4538.t[0][1]) * _4637)) * _4628), 0).xyz;
                            }
                        }
                        else
                        {
                            [branch]
                            if (_3604 == 7u)
                            {
                                float _4852 = frac(asfloat(_3523.Load((hi + 4) * 4 + 0)) + sample_off.x);
                                float _4861 = frac(asfloat(_3523.Load((hi + 5) * 4 + 0)) + sample_off.y);
                                float4 dir_and_pdf;
                                if (_3539_g_params.env_qtree_levels > 0)
                                {
                                    dir_and_pdf = Sample_EnvQTree(_3539_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3539_g_params.env_qtree_levels, mad(_3532, _3543, -float(_3550)), _4852, _4861);
                                }
                                else
                                {
                                    float _4880 = 6.283185482025146484375f * _4861;
                                    float _4892 = sqrt(mad(-_4852, _4852, 1.0f));
                                    float3 param_16 = T;
                                    float3 param_17 = B;
                                    float3 param_18 = N;
                                    float3 param_19 = float3(_4892 * cos(_4880), _4892 * sin(_4880), _4852);
                                    dir_and_pdf = float4(world_from_tangent(param_16, param_17, param_18, param_19), 0.15915493667125701904296875f);
                                }
                                ls.L = dir_and_pdf.xyz;
                                ls.col *= _3539_g_params.env_col.xyz;
                                uint _4931 = asuint(_3539_g_params.env_col.w);
                                if (_4931 != 4294967295u)
                                {
                                    atlas_texture_t _4938;
                                    _4938.size = _1003.Load(_4931 * 80 + 0);
                                    _4938.atlas = _1003.Load(_4931 * 80 + 4);
                                    [unroll]
                                    for (int _85ident = 0; _85ident < 4; _85ident++)
                                    {
                                        _4938.page[_85ident] = _1003.Load(_85ident * 4 + _4931 * 80 + 8);
                                    }
                                    [unroll]
                                    for (int _86ident = 0; _86ident < 14; _86ident++)
                                    {
                                        _4938.pos[_86ident] = _1003.Load(_86ident * 4 + _4931 * 80 + 24);
                                    }
                                    uint _9659[14] = { _4938.pos[0], _4938.pos[1], _4938.pos[2], _4938.pos[3], _4938.pos[4], _4938.pos[5], _4938.pos[6], _4938.pos[7], _4938.pos[8], _4938.pos[9], _4938.pos[10], _4938.pos[11], _4938.pos[12], _4938.pos[13] };
                                    uint _9630[4] = { _4938.page[0], _4938.page[1], _4938.page[2], _4938.page[3] };
                                    atlas_texture_t _9498 = { _4938.size, _4938.atlas, _9630, _9659 };
                                    float param_20 = _3539_g_params.env_rotation;
                                    ls.col *= SampleLatlong_RGBE(_9498, ls.L, param_20);
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
    float3 _9195;
    do
    {
        float3 param = -float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param_1 = surf.N;
        float3 param_2 = ls.L;
        float param_3 = roughness;
        float3 param_4 = base_color;
        float4 _5522 = Evaluate_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_5 = ls.pdf;
            float param_6 = _5522.w;
            mis_weight = power_heuristic(param_5, param_6);
        }
        float3 _5545 = (ls.col * _5522.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9195 = _5545;
            break;
        }
        float3 param_7 = surf.P;
        float3 param_8 = surf.plane_N;
        float3 _5557 = offset_ray(param_7, param_8);
        sh_r.o[0] = _5557.x;
        sh_r.o[1] = _5557.y;
        sh_r.o[2] = _5557.z;
        sh_r.c[0] = ray.c[0] * _5545.x;
        sh_r.c[1] = ray.c[1] * _5545.y;
        sh_r.c[2] = ray.c[2] * _5545.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9195 = 0.0f.xxx;
        break;
    } while(false);
    return _9195;
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
    float4 _5808 = Sample_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8);
    new_ray.depth = ray.depth + 1;
    float3 param_9 = surf.P;
    float3 param_10 = surf.plane_N;
    float3 _5818 = offset_ray(param_9, param_10);
    new_ray.o[0] = _5818.x;
    new_ray.o[1] = _5818.y;
    new_ray.o[2] = _5818.z;
    new_ray.d[0] = param_8.x;
    new_ray.d[1] = param_8.y;
    new_ray.d[2] = param_8.z;
    new_ray.c[0] = ((ray.c[0] * _5808.x) * mix_weight) / _5808.w;
    new_ray.c[1] = ((ray.c[1] * _5808.y) * mix_weight) / _5808.w;
    new_ray.c[2] = ((ray.c[2] * _5808.z) * mix_weight) / _5808.w;
    new_ray.pdf = _5808.w;
}

float3 tangent_from_world(float3 T, float3 B, float3 N, float3 V)
{
    return float3(dot(V, T), dot(V, B), dot(V, N));
}

float D_GGX(float3 H, float alpha_x, float alpha_y)
{
    float _9248;
    do
    {
        if (H.z == 0.0f)
        {
            _9248 = 0.0f;
            break;
        }
        float _2244 = (-H.x) / (H.z * alpha_x);
        float _2250 = (-H.y) / (H.z * alpha_y);
        float _2259 = mad(_2250, _2250, mad(_2244, _2244, 1.0f));
        _9248 = 1.0f / (((((_2259 * _2259) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _9248;
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
    float3 _9200;
    do
    {
        float3 _5593 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5593;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - _5593);
        float _5631 = roughness * roughness;
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = _5631;
        float param_16 = _5631;
        float param_17 = spec_ior;
        float param_18 = spec_F0;
        float3 param_19 = base_color;
        float4 _5646 = Evaluate_GGXSpecular_BSDF(param_12, param_13, param_14, param_15, param_16, param_17, param_18, param_19);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_20 = ls.pdf;
            float param_21 = _5646.w;
            mis_weight = power_heuristic(param_20, param_21);
        }
        float3 _5669 = (ls.col * _5646.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9200 = _5669;
            break;
        }
        float3 param_22 = surf.P;
        float3 param_23 = surf.plane_N;
        float3 _5681 = offset_ray(param_22, param_23);
        sh_r.o[0] = _5681.x;
        sh_r.o[1] = _5681.y;
        sh_r.o[2] = _5681.z;
        sh_r.c[0] = ray.c[0] * _5669.x;
        sh_r.c[1] = ray.c[1] * _5669.y;
        sh_r.c[2] = ray.c[2] * _5669.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9200 = 0.0f.xxx;
        break;
    } while(false);
    return _9200;
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
    float4 _9220;
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
            _9220 = float4(_2899.x * 1000000.0f, _2899.y * 1000000.0f, _2899.z * 1000000.0f, 1000000.0f);
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
        _9220 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _9220;
}

void Sample_GlossyNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 param_4;
    float4 _5728 = Sample_GGXSpecular_BSDF(param, param_1, param_2, param_3, roughness, 0.0f, spec_ior, spec_F0, base_color, rand_u, rand_v, param_4);
    new_ray.depth = ray.depth + 256;
    float3 param_5 = surf.P;
    float3 param_6 = surf.plane_N;
    float3 _5739 = offset_ray(param_5, param_6);
    new_ray.o[0] = _5739.x;
    new_ray.o[1] = _5739.y;
    new_ray.o[2] = _5739.z;
    new_ray.d[0] = param_4.x;
    new_ray.d[1] = param_4.y;
    new_ray.d[2] = param_4.z;
    new_ray.c[0] = ((ray.c[0] * _5728.x) * mix_weight) / _5728.w;
    new_ray.c[1] = ((ray.c[1] * _5728.y) * mix_weight) / _5728.w;
    new_ray.c[2] = ((ray.c[2] * _5728.z) * mix_weight) / _5728.w;
    new_ray.pdf = _5728.w;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _9225;
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
            _9225 = 0.0f.xxxx;
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
        _9225 = float4(refr_col * (((((_3182 * _3198) * _3190) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3218) / view_dir_ts.z), (((_3182 * _3190) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3218) / view_dir_ts.z);
        break;
    } while(false);
    return _9225;
}

float3 Evaluate_RefractiveNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness2, float eta, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _9205;
    do
    {
        float3 _5871 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5871;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - (_5871 * eta));
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = roughness2;
        float param_16 = eta;
        float3 param_17 = base_color;
        float4 _5919 = Evaluate_GGXRefraction_BSDF(param_12, param_13, param_14, param_15, param_16, param_17);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_18 = ls.pdf;
            float param_19 = _5919.w;
            mis_weight = power_heuristic(param_18, param_19);
        }
        float3 _5942 = (ls.col * _5919.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9205 = _5942;
            break;
        }
        float3 param_20 = surf.P;
        float3 param_21 = -surf.plane_N;
        float3 _5955 = offset_ray(param_20, param_21);
        sh_r.o[0] = _5955.x;
        sh_r.o[1] = _5955.y;
        sh_r.o[2] = _5955.z;
        sh_r.c[0] = ray.c[0] * _5942.x;
        sh_r.c[1] = ray.c[1] * _5942.y;
        sh_r.c[2] = ray.c[2] * _5942.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9205 = 0.0f.xxx;
        break;
    } while(false);
    return _9205;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _9230;
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
                _9230 = 0.0f.xxxx;
                break;
            }
            float _3295 = mad(eta, _3273, -sqrt(_3283));
            out_V = float4(normalize((I * eta) + (N * _3295)), _3295);
            _9230 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
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
            _9230 = 0.0f.xxxx;
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
        _9230 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _9230;
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
    float _9238;
    do
    {
        if (stack[3] > 0.0f)
        {
            float param = stack[3];
            float param_1 = -1.0f;
            float _2350 = exchange(param, param_1);
            stack[3] = param;
            _9238 = _2350;
            break;
        }
        if (stack[2] > 0.0f)
        {
            float param_2 = stack[2];
            float param_3 = -1.0f;
            float _2363 = exchange(param_2, param_3);
            stack[2] = param_2;
            _9238 = _2363;
            break;
        }
        if (stack[1] > 0.0f)
        {
            float param_4 = stack[1];
            float param_5 = -1.0f;
            float _2376 = exchange(param_4, param_5);
            stack[1] = param_4;
            _9238 = _2376;
            break;
        }
        if (stack[0] > 0.0f)
        {
            float param_6 = stack[0];
            float param_7 = -1.0f;
            float _2389 = exchange(param_6, param_7);
            stack[0] = param_6;
            _9238 = _2389;
            break;
        }
        _9238 = default_value;
        break;
    } while(false);
    return _9238;
}

void Sample_RefractiveNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, bool is_backfacing, float int_ior, float ext_ior, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float _5992;
    if (is_backfacing)
    {
        _5992 = int_ior / ext_ior;
    }
    else
    {
        _5992 = ext_ior / int_ior;
    }
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float param_4 = roughness;
    float param_5 = _5992;
    float3 param_6 = base_color;
    float param_7 = rand_u;
    float param_8 = rand_v;
    float4 param_9;
    float4 _6016 = Sample_GGXRefraction_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9);
    new_ray.depth = ray.depth + 65536;
    new_ray.c[0] = ((ray.c[0] * _6016.x) * mix_weight) / _6016.w;
    new_ray.c[1] = ((ray.c[1] * _6016.y) * mix_weight) / _6016.w;
    new_ray.c[2] = ((ray.c[2] * _6016.z) * mix_weight) / _6016.w;
    new_ray.pdf = _6016.w;
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
        float _6072 = pop_ior_stack(param_11, param_12);
        new_ray.ior = param_11;
    }
    float3 param_13 = surf.P;
    float3 param_14 = -surf.plane_N;
    float3 _6081 = offset_ray(param_13, param_14);
    new_ray.o[0] = _6081.x;
    new_ray.o[1] = _6081.y;
    new_ray.o[2] = _6081.z;
    new_ray.d[0] = param_9.x;
    new_ray.d[1] = param_9.y;
    new_ray.d[2] = param_9.z;
}

lobe_weights_t get_lobe_weights(float base_color_lum, float spec_color_lum, float specular, float metallic, float transmission, float clearcoat)
{
    float _1715 = 1.0f - metallic;
    float _9393 = (base_color_lum * _1715) * (1.0f - transmission);
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
    float _9394 = _1726;
    float _1736 = 0.25f * clearcoat;
    float _9395 = _1736 * _1715;
    float _9396 = _1722 * base_color_lum;
    float _1745 = _9393;
    float _1754 = mad(_1722, base_color_lum, mad(_1736, _1715, _1745 + _1726));
    if (_1754 != 0.0f)
    {
        _9393 /= _1754;
        _9394 /= _1754;
        _9395 /= _1754;
        _9396 /= _1754;
    }
    lobe_weights_t _9401 = { _9393, _9394, _9395, _9396 };
    return _9401;
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
    float _9253;
    do
    {
        float _2470 = dot(N, L);
        if (_2470 <= 0.0f)
        {
            _9253 = 0.0f;
            break;
        }
        float param = _2470;
        float param_1 = dot(N, V);
        float _2491 = dot(L, H);
        float _2499 = mad((2.0f * _2491) * _2491, roughness, 0.5f);
        _9253 = lerp(1.0f, _2499, schlick_weight(param)) * lerp(1.0f, _2499, schlick_weight(param_1));
        break;
    } while(false);
    return _9253;
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
    float _9258;
    do
    {
        if (a >= 1.0f)
        {
            _9258 = 0.3183098733425140380859375f;
            break;
        }
        float _2218 = mad(a, a, -1.0f);
        _9258 = _2218 / ((3.1415927410125732421875f * log(a * a)) * mad(_2218 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _9258;
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
    float3 _9210;
    do
    {
        float3 _6104 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 lcol = 0.0f.xxx;
        float bsdf_pdf = 0.0f;
        bool _6109 = N_dot_L > 0.0f;
        [branch]
        if ((lobe_weights.diffuse > 1.0000000116860974230803549289703e-07f) && _6109)
        {
            float3 param = -_6104;
            float3 param_1 = surf.N;
            float3 param_2 = ls.L;
            float param_3 = diff.roughness;
            float3 param_4 = diff.base_color;
            float3 param_5 = diff.sheen_color;
            bool param_6 = false;
            float4 _6128 = Evaluate_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6);
            bsdf_pdf = mad(lobe_weights.diffuse, _6128.w, bsdf_pdf);
            lcol += (((ls.col * N_dot_L) * (_6128 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * ls.pdf).xxx);
        }
        float3 H;
        [flatten]
        if (_6109)
        {
            H = normalize(ls.L - _6104);
        }
        else
        {
            H = normalize(ls.L - (_6104 * trans.eta));
        }
        float _6167 = spec.roughness * spec.roughness;
        float _6172 = sqrt(mad(-0.89999997615814208984375f, spec.anisotropy, 1.0f));
        float _6176 = _6167 / _6172;
        float _6180 = _6167 * _6172;
        float3 param_7 = surf.T;
        float3 param_8 = surf.B;
        float3 param_9 = surf.N;
        float3 param_10 = -_6104;
        float3 _6191 = tangent_from_world(param_7, param_8, param_9, param_10);
        float3 param_11 = surf.T;
        float3 param_12 = surf.B;
        float3 param_13 = surf.N;
        float3 param_14 = ls.L;
        float3 _6201 = tangent_from_world(param_11, param_12, param_13, param_14);
        float3 param_15 = surf.T;
        float3 param_16 = surf.B;
        float3 param_17 = surf.N;
        float3 param_18 = H;
        float3 _6211 = tangent_from_world(param_15, param_16, param_17, param_18);
        bool _6213 = lobe_weights.specular > 0.0f;
        bool _6220;
        if (_6213)
        {
            _6220 = (_6176 * _6180) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _6220 = _6213;
        }
        [branch]
        if (_6220 && _6109)
        {
            float3 param_19 = _6191;
            float3 param_20 = _6211;
            float3 param_21 = _6201;
            float param_22 = _6176;
            float param_23 = _6180;
            float param_24 = spec.ior;
            float param_25 = spec.F0;
            float3 param_26 = spec.tmp_col;
            float4 _6242 = Evaluate_GGXSpecular_BSDF(param_19, param_20, param_21, param_22, param_23, param_24, param_25, param_26);
            bsdf_pdf = mad(lobe_weights.specular, _6242.w, bsdf_pdf);
            lcol += ((ls.col * _6242.xyz) / ls.pdf.xxx);
        }
        float _6261 = coat.roughness * coat.roughness;
        bool _6263 = lobe_weights.clearcoat > 0.0f;
        bool _6270;
        if (_6263)
        {
            _6270 = (_6261 * _6261) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _6270 = _6263;
        }
        [branch]
        if (_6270 && _6109)
        {
            float3 param_27 = _6191;
            float3 param_28 = _6211;
            float3 param_29 = _6201;
            float param_30 = _6261;
            float param_31 = coat.ior;
            float param_32 = coat.F0;
            float4 _6288 = Evaluate_PrincipledClearcoat_BSDF(param_27, param_28, param_29, param_30, param_31, param_32);
            bsdf_pdf = mad(lobe_weights.clearcoat, _6288.w, bsdf_pdf);
            lcol += (((ls.col * 0.25f) * _6288.xyz) / ls.pdf.xxx);
        }
        [branch]
        if (lobe_weights.refraction > 0.0f)
        {
            bool _6310 = trans.fresnel != 0.0f;
            bool _6317;
            if (_6310)
            {
                _6317 = (_6167 * _6167) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _6317 = _6310;
            }
            [branch]
            if (_6317 && _6109)
            {
                float3 param_33 = _6191;
                float3 param_34 = _6211;
                float3 param_35 = _6201;
                float param_36 = _6167;
                float param_37 = _6167;
                float param_38 = 1.0f;
                float param_39 = 0.0f;
                float3 param_40 = 1.0f.xxx;
                float4 _6336 = Evaluate_GGXSpecular_BSDF(param_33, param_34, param_35, param_36, param_37, param_38, param_39, param_40);
                bsdf_pdf = mad(lobe_weights.refraction * trans.fresnel, _6336.w, bsdf_pdf);
                lcol += ((ls.col * _6336.xyz) * (trans.fresnel / ls.pdf));
            }
            float _6358 = trans.roughness * trans.roughness;
            bool _6360 = trans.fresnel != 1.0f;
            bool _6367;
            if (_6360)
            {
                _6367 = (_6358 * _6358) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _6367 = _6360;
            }
            [branch]
            if (_6367 && (N_dot_L < 0.0f))
            {
                float3 param_41 = _6191;
                float3 param_42 = _6211;
                float3 param_43 = _6201;
                float param_44 = _6358;
                float param_45 = trans.eta;
                float3 param_46 = diff.base_color;
                float4 _6385 = Evaluate_GGXRefraction_BSDF(param_41, param_42, param_43, param_44, param_45, param_46);
                float _6388 = 1.0f - trans.fresnel;
                bsdf_pdf = mad(lobe_weights.refraction * _6388, _6385.w, bsdf_pdf);
                lcol += ((ls.col * _6385.xyz) * (_6388 / ls.pdf));
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
            _9210 = lcol;
            break;
        }
        float3 _6428;
        if (N_dot_L < 0.0f)
        {
            _6428 = -surf.plane_N;
        }
        else
        {
            _6428 = surf.plane_N;
        }
        float3 param_49 = surf.P;
        float3 param_50 = _6428;
        float3 _6439 = offset_ray(param_49, param_50);
        sh_r.o[0] = _6439.x;
        sh_r.o[1] = _6439.y;
        sh_r.o[2] = _6439.z;
        sh_r.c[0] = ray.c[0] * lcol.x;
        sh_r.c[1] = ray.c[1] * lcol.y;
        sh_r.c[2] = ray.c[2] * lcol.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9210 = 0.0f.xxx;
        break;
    } while(false);
    return _9210;
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
    float4 _9243;
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
            _9243 = float4(_3099, _3099, _3099, 1000000.0f);
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
        _9243 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _9243;
}

void Sample_PrincipledNode(ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float rand_u, float rand_v, inout float mix_rand, float mix_weight, inout ray_data_t new_ray)
{
    float3 _6474 = float3(ray.d[0], ray.d[1], ray.d[2]);
    int _6478 = ray.depth & 255;
    int _6482 = (ray.depth >> 8) & 255;
    int _6486 = (ray.depth >> 16) & 255;
    int _6497 = (_6478 + _6482) + _6486;
    [branch]
    if (mix_rand < lobe_weights.diffuse)
    {
        bool _6506 = _6478 < _3539_g_params.max_diff_depth;
        bool _6513;
        if (_6506)
        {
            _6513 = _6497 < _3539_g_params.max_total_depth;
        }
        else
        {
            _6513 = _6506;
        }
        if (_6513)
        {
            float3 param = surf.T;
            float3 param_1 = surf.B;
            float3 param_2 = surf.N;
            float3 param_3 = _6474;
            float param_4 = diff.roughness;
            float3 param_5 = diff.base_color;
            float3 param_6 = diff.sheen_color;
            bool param_7 = false;
            float param_8 = rand_u;
            float param_9 = rand_v;
            float3 param_10;
            float4 _6536 = Sample_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9, param_10);
            float3 _6541 = _6536.xyz * (1.0f - metallic);
            new_ray.depth = ray.depth + 1;
            float3 param_11 = surf.P;
            float3 param_12 = surf.plane_N;
            float3 _6556 = offset_ray(param_11, param_12);
            new_ray.o[0] = _6556.x;
            new_ray.o[1] = _6556.y;
            new_ray.o[2] = _6556.z;
            new_ray.d[0] = param_10.x;
            new_ray.d[1] = param_10.y;
            new_ray.d[2] = param_10.z;
            new_ray.c[0] = ((ray.c[0] * _6541.x) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[1] = ((ray.c[1] * _6541.y) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[2] = ((ray.c[2] * _6541.z) * mix_weight) / lobe_weights.diffuse;
            new_ray.pdf = _6536.w;
        }
    }
    else
    {
        float _6606 = lobe_weights.diffuse + lobe_weights.specular;
        [branch]
        if (mix_rand < _6606)
        {
            bool _6613 = _6482 < _3539_g_params.max_spec_depth;
            bool _6620;
            if (_6613)
            {
                _6620 = _6497 < _3539_g_params.max_total_depth;
            }
            else
            {
                _6620 = _6613;
            }
            if (_6620)
            {
                float3 param_13 = surf.T;
                float3 param_14 = surf.B;
                float3 param_15 = surf.N;
                float3 param_16 = _6474;
                float3 param_17;
                float4 _6639 = Sample_GGXSpecular_BSDF(param_13, param_14, param_15, param_16, spec.roughness, spec.anisotropy, spec.ior, spec.F0, spec.tmp_col, rand_u, rand_v, param_17);
                float _6644 = _6639.w * lobe_weights.specular;
                new_ray.depth = ray.depth + 256;
                new_ray.c[0] = ((ray.c[0] * _6639.x) * mix_weight) / _6644;
                new_ray.c[1] = ((ray.c[1] * _6639.y) * mix_weight) / _6644;
                new_ray.c[2] = ((ray.c[2] * _6639.z) * mix_weight) / _6644;
                new_ray.pdf = _6644;
                float3 param_18 = surf.P;
                float3 param_19 = surf.plane_N;
                float3 _6684 = offset_ray(param_18, param_19);
                new_ray.o[0] = _6684.x;
                new_ray.o[1] = _6684.y;
                new_ray.o[2] = _6684.z;
                new_ray.d[0] = param_17.x;
                new_ray.d[1] = param_17.y;
                new_ray.d[2] = param_17.z;
            }
        }
        else
        {
            float _6709 = _6606 + lobe_weights.clearcoat;
            [branch]
            if (mix_rand < _6709)
            {
                bool _6716 = _6482 < _3539_g_params.max_spec_depth;
                bool _6723;
                if (_6716)
                {
                    _6723 = _6497 < _3539_g_params.max_total_depth;
                }
                else
                {
                    _6723 = _6716;
                }
                if (_6723)
                {
                    float3 param_20 = surf.T;
                    float3 param_21 = surf.B;
                    float3 param_22 = surf.N;
                    float3 param_23 = _6474;
                    float param_24 = coat.roughness * coat.roughness;
                    float param_25 = coat.ior;
                    float param_26 = coat.F0;
                    float param_27 = rand_u;
                    float param_28 = rand_v;
                    float3 param_29;
                    float4 _6747 = Sample_PrincipledClearcoat_BSDF(param_20, param_21, param_22, param_23, param_24, param_25, param_26, param_27, param_28, param_29);
                    float _6752 = _6747.w * lobe_weights.clearcoat;
                    new_ray.depth = ray.depth + 256;
                    new_ray.c[0] = (((0.25f * ray.c[0]) * _6747.x) * mix_weight) / _6752;
                    new_ray.c[1] = (((0.25f * ray.c[1]) * _6747.y) * mix_weight) / _6752;
                    new_ray.c[2] = (((0.25f * ray.c[2]) * _6747.z) * mix_weight) / _6752;
                    new_ray.pdf = _6752;
                    float3 param_30 = surf.P;
                    float3 param_31 = surf.plane_N;
                    float3 _6795 = offset_ray(param_30, param_31);
                    new_ray.o[0] = _6795.x;
                    new_ray.o[1] = _6795.y;
                    new_ray.o[2] = _6795.z;
                    new_ray.d[0] = param_29.x;
                    new_ray.d[1] = param_29.y;
                    new_ray.d[2] = param_29.z;
                }
            }
            else
            {
                bool _6817 = mix_rand >= trans.fresnel;
                bool _6824;
                if (_6817)
                {
                    _6824 = _6486 < _3539_g_params.max_refr_depth;
                }
                else
                {
                    _6824 = _6817;
                }
                bool _6838;
                if (!_6824)
                {
                    bool _6830 = mix_rand < trans.fresnel;
                    bool _6837;
                    if (_6830)
                    {
                        _6837 = _6482 < _3539_g_params.max_spec_depth;
                    }
                    else
                    {
                        _6837 = _6830;
                    }
                    _6838 = _6837;
                }
                else
                {
                    _6838 = _6824;
                }
                bool _6845;
                if (_6838)
                {
                    _6845 = _6497 < _3539_g_params.max_total_depth;
                }
                else
                {
                    _6845 = _6838;
                }
                [branch]
                if (_6845)
                {
                    mix_rand -= _6709;
                    mix_rand /= lobe_weights.refraction;
                    float4 F;
                    float3 V;
                    [branch]
                    if (mix_rand < trans.fresnel)
                    {
                        float3 param_32 = surf.T;
                        float3 param_33 = surf.B;
                        float3 param_34 = surf.N;
                        float3 param_35 = _6474;
                        float3 param_36;
                        float4 _6875 = Sample_GGXSpecular_BSDF(param_32, param_33, param_34, param_35, spec.roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, rand_u, rand_v, param_36);
                        V = param_36;
                        F = _6875;
                        new_ray.depth = ray.depth + 256;
                        float3 param_37 = surf.P;
                        float3 param_38 = surf.plane_N;
                        float3 _6885 = offset_ray(param_37, param_38);
                        new_ray.o[0] = _6885.x;
                        new_ray.o[1] = _6885.y;
                        new_ray.o[2] = _6885.z;
                    }
                    else
                    {
                        float3 param_39 = surf.T;
                        float3 param_40 = surf.B;
                        float3 param_41 = surf.N;
                        float3 param_42 = _6474;
                        float param_43 = trans.roughness;
                        float param_44 = trans.eta;
                        float3 param_45 = diff.base_color;
                        float param_46 = rand_u;
                        float param_47 = rand_v;
                        float4 param_48;
                        float4 _6914 = Sample_GGXRefraction_BSDF(param_39, param_40, param_41, param_42, param_43, param_44, param_45, param_46, param_47, param_48);
                        F = _6914;
                        V = param_48.xyz;
                        new_ray.depth = ray.depth + 65536;
                        float3 param_49 = surf.P;
                        float3 param_50 = -surf.plane_N;
                        float3 _6927 = offset_ray(param_49, param_50);
                        new_ray.o[0] = _6927.x;
                        new_ray.o[1] = _6927.y;
                        new_ray.o[2] = _6927.z;
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
                            float _6953 = pop_ior_stack(param_52, param_53);
                            new_ray.ior = param_52;
                        }
                    }
                    float4 _10786 = F;
                    float _6959 = _10786.w * lobe_weights.refraction;
                    float4 _10788 = _10786;
                    _10788.w = _6959;
                    F = _10788;
                    new_ray.c[0] = ((ray.c[0] * _10786.x) * mix_weight) / _6959;
                    new_ray.c[1] = ((ray.c[1] * _10786.y) * mix_weight) / _6959;
                    new_ray.c[2] = ((ray.c[2] * _10786.z) * mix_weight) / _6959;
                    new_ray.pdf = _6959;
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
    float3 _9180;
    do
    {
        float3 _7015 = float3(ray.d[0], ray.d[1], ray.d[2]);
        [branch]
        if (inter.mask == 0)
        {
            ray_data_t param = ray;
            float3 _7024 = Evaluate_EnvColor(param);
            _9180 = float3(ray.c[0] * _7024.x, ray.c[1] * _7024.y, ray.c[2] * _7024.z);
            break;
        }
        float3 _7051 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_7015 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            ray_data_t param_1 = ray;
            hit_data_t param_2 = inter;
            float3 _7063 = Evaluate_LightColor(param_1, param_2);
            _9180 = float3(ray.c[0] * _7063.x, ray.c[1] * _7063.y, ray.c[2] * _7063.z);
            break;
        }
        bool _7084 = inter.prim_index < 0;
        int _7087;
        if (_7084)
        {
            _7087 = (-1) - inter.prim_index;
        }
        else
        {
            _7087 = inter.prim_index;
        }
        uint _7098 = uint(_7087);
        material_t _7106;
        [unroll]
        for (int _89ident = 0; _89ident < 5; _89ident++)
        {
            _7106.textures[_89ident] = _4759.Load(_89ident * 4 + ((_4763.Load(_7098 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
        }
        [unroll]
        for (int _90ident = 0; _90ident < 3; _90ident++)
        {
            _7106.base_color[_90ident] = asfloat(_4759.Load(_90ident * 4 + ((_4763.Load(_7098 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
        }
        _7106.flags = _4759.Load(((_4763.Load(_7098 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
        _7106.type = _4759.Load(((_4763.Load(_7098 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
        _7106.tangent_rotation_or_strength = asfloat(_4759.Load(((_4763.Load(_7098 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
        _7106.roughness_and_anisotropic = _4759.Load(((_4763.Load(_7098 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
        _7106.ior = asfloat(_4759.Load(((_4763.Load(_7098 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
        _7106.sheen_and_sheen_tint = _4759.Load(((_4763.Load(_7098 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
        _7106.tint_and_metallic = _4759.Load(((_4763.Load(_7098 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
        _7106.transmission_and_transmission_roughness = _4759.Load(((_4763.Load(_7098 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
        _7106.specular_and_specular_tint = _4759.Load(((_4763.Load(_7098 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
        _7106.clearcoat_and_clearcoat_roughness = _4759.Load(((_4763.Load(_7098 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
        _7106.normal_map_strength_unorm = _4759.Load(((_4763.Load(_7098 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
        uint _10235 = _7106.textures[0];
        uint _10236 = _7106.textures[1];
        uint _10237 = _7106.textures[2];
        uint _10238 = _7106.textures[3];
        uint _10239 = _7106.textures[4];
        float _10240 = _7106.base_color[0];
        float _10241 = _7106.base_color[1];
        float _10242 = _7106.base_color[2];
        uint _9845 = _7106.flags;
        uint _9846 = _7106.type;
        float _9847 = _7106.tangent_rotation_or_strength;
        uint _9848 = _7106.roughness_and_anisotropic;
        float _9849 = _7106.ior;
        uint _9850 = _7106.sheen_and_sheen_tint;
        uint _9851 = _7106.tint_and_metallic;
        uint _9852 = _7106.transmission_and_transmission_roughness;
        uint _9853 = _7106.specular_and_specular_tint;
        uint _9854 = _7106.clearcoat_and_clearcoat_roughness;
        uint _9855 = _7106.normal_map_strength_unorm;
        transform_t _7161;
        _7161.xform = asfloat(uint4x4(_4406.Load4(asuint(asfloat(_7154.Load(inter.obj_index * 32 + 12))) * 128 + 0), _4406.Load4(asuint(asfloat(_7154.Load(inter.obj_index * 32 + 12))) * 128 + 16), _4406.Load4(asuint(asfloat(_7154.Load(inter.obj_index * 32 + 12))) * 128 + 32), _4406.Load4(asuint(asfloat(_7154.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _7161.inv_xform = asfloat(uint4x4(_4406.Load4(asuint(asfloat(_7154.Load(inter.obj_index * 32 + 12))) * 128 + 64), _4406.Load4(asuint(asfloat(_7154.Load(inter.obj_index * 32 + 12))) * 128 + 80), _4406.Load4(asuint(asfloat(_7154.Load(inter.obj_index * 32 + 12))) * 128 + 96), _4406.Load4(asuint(asfloat(_7154.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _7168 = _7098 * 3u;
        vertex_t _7173;
        [unroll]
        for (int _91ident = 0; _91ident < 3; _91ident++)
        {
            _7173.p[_91ident] = asfloat(_4431.Load(_91ident * 4 + _4435.Load(_7168 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _92ident = 0; _92ident < 3; _92ident++)
        {
            _7173.n[_92ident] = asfloat(_4431.Load(_92ident * 4 + _4435.Load(_7168 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _93ident = 0; _93ident < 3; _93ident++)
        {
            _7173.b[_93ident] = asfloat(_4431.Load(_93ident * 4 + _4435.Load(_7168 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _94ident = 0; _94ident < 2; _94ident++)
        {
            [unroll]
            for (int _95ident = 0; _95ident < 2; _95ident++)
            {
                _7173.t[_94ident][_95ident] = asfloat(_4431.Load(_95ident * 4 + _94ident * 8 + _4435.Load(_7168 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _7219;
        [unroll]
        for (int _96ident = 0; _96ident < 3; _96ident++)
        {
            _7219.p[_96ident] = asfloat(_4431.Load(_96ident * 4 + _4435.Load((_7168 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _97ident = 0; _97ident < 3; _97ident++)
        {
            _7219.n[_97ident] = asfloat(_4431.Load(_97ident * 4 + _4435.Load((_7168 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _98ident = 0; _98ident < 3; _98ident++)
        {
            _7219.b[_98ident] = asfloat(_4431.Load(_98ident * 4 + _4435.Load((_7168 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _99ident = 0; _99ident < 2; _99ident++)
        {
            [unroll]
            for (int _100ident = 0; _100ident < 2; _100ident++)
            {
                _7219.t[_99ident][_100ident] = asfloat(_4431.Load(_100ident * 4 + _99ident * 8 + _4435.Load((_7168 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _7265;
        [unroll]
        for (int _101ident = 0; _101ident < 3; _101ident++)
        {
            _7265.p[_101ident] = asfloat(_4431.Load(_101ident * 4 + _4435.Load((_7168 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _102ident = 0; _102ident < 3; _102ident++)
        {
            _7265.n[_102ident] = asfloat(_4431.Load(_102ident * 4 + _4435.Load((_7168 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _103ident = 0; _103ident < 3; _103ident++)
        {
            _7265.b[_103ident] = asfloat(_4431.Load(_103ident * 4 + _4435.Load((_7168 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _104ident = 0; _104ident < 2; _104ident++)
        {
            [unroll]
            for (int _105ident = 0; _105ident < 2; _105ident++)
            {
                _7265.t[_104ident][_105ident] = asfloat(_4431.Load(_105ident * 4 + _104ident * 8 + _4435.Load((_7168 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _7311 = float3(_7173.p[0], _7173.p[1], _7173.p[2]);
        float3 _7319 = float3(_7219.p[0], _7219.p[1], _7219.p[2]);
        float3 _7327 = float3(_7265.p[0], _7265.p[1], _7265.p[2]);
        float _7334 = (1.0f - inter.u) - inter.v;
        float3 _7366 = normalize(((float3(_7173.n[0], _7173.n[1], _7173.n[2]) * _7334) + (float3(_7219.n[0], _7219.n[1], _7219.n[2]) * inter.u)) + (float3(_7265.n[0], _7265.n[1], _7265.n[2]) * inter.v));
        float3 _9784 = _7366;
        float2 _7392 = ((float2(_7173.t[0][0], _7173.t[0][1]) * _7334) + (float2(_7219.t[0][0], _7219.t[0][1]) * inter.u)) + (float2(_7265.t[0][0], _7265.t[0][1]) * inter.v);
        float3 _7408 = cross(_7319 - _7311, _7327 - _7311);
        float _7413 = length(_7408);
        float3 _9785 = _7408 / _7413.xxx;
        float3 _7450 = ((float3(_7173.b[0], _7173.b[1], _7173.b[2]) * _7334) + (float3(_7219.b[0], _7219.b[1], _7219.b[2]) * inter.u)) + (float3(_7265.b[0], _7265.b[1], _7265.b[2]) * inter.v);
        float3 _9783 = _7450;
        float3 _9782 = cross(_7450, _7366);
        if (_7084)
        {
            if ((_4763.Load(_7098 * 4 + 0) & 65535u) == 65535u)
            {
                _9180 = 0.0f.xxx;
                break;
            }
            material_t _7475;
            [unroll]
            for (int _106ident = 0; _106ident < 5; _106ident++)
            {
                _7475.textures[_106ident] = _4759.Load(_106ident * 4 + (_4763.Load(_7098 * 4 + 0) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _107ident = 0; _107ident < 3; _107ident++)
            {
                _7475.base_color[_107ident] = asfloat(_4759.Load(_107ident * 4 + (_4763.Load(_7098 * 4 + 0) & 16383u) * 76 + 20));
            }
            _7475.flags = _4759.Load((_4763.Load(_7098 * 4 + 0) & 16383u) * 76 + 32);
            _7475.type = _4759.Load((_4763.Load(_7098 * 4 + 0) & 16383u) * 76 + 36);
            _7475.tangent_rotation_or_strength = asfloat(_4759.Load((_4763.Load(_7098 * 4 + 0) & 16383u) * 76 + 40));
            _7475.roughness_and_anisotropic = _4759.Load((_4763.Load(_7098 * 4 + 0) & 16383u) * 76 + 44);
            _7475.ior = asfloat(_4759.Load((_4763.Load(_7098 * 4 + 0) & 16383u) * 76 + 48));
            _7475.sheen_and_sheen_tint = _4759.Load((_4763.Load(_7098 * 4 + 0) & 16383u) * 76 + 52);
            _7475.tint_and_metallic = _4759.Load((_4763.Load(_7098 * 4 + 0) & 16383u) * 76 + 56);
            _7475.transmission_and_transmission_roughness = _4759.Load((_4763.Load(_7098 * 4 + 0) & 16383u) * 76 + 60);
            _7475.specular_and_specular_tint = _4759.Load((_4763.Load(_7098 * 4 + 0) & 16383u) * 76 + 64);
            _7475.clearcoat_and_clearcoat_roughness = _4759.Load((_4763.Load(_7098 * 4 + 0) & 16383u) * 76 + 68);
            _7475.normal_map_strength_unorm = _4759.Load((_4763.Load(_7098 * 4 + 0) & 16383u) * 76 + 72);
            _10235 = _7475.textures[0];
            _10236 = _7475.textures[1];
            _10237 = _7475.textures[2];
            _10238 = _7475.textures[3];
            _10239 = _7475.textures[4];
            _10240 = _7475.base_color[0];
            _10241 = _7475.base_color[1];
            _10242 = _7475.base_color[2];
            _9845 = _7475.flags;
            _9846 = _7475.type;
            _9847 = _7475.tangent_rotation_or_strength;
            _9848 = _7475.roughness_and_anisotropic;
            _9849 = _7475.ior;
            _9850 = _7475.sheen_and_sheen_tint;
            _9851 = _7475.tint_and_metallic;
            _9852 = _7475.transmission_and_transmission_roughness;
            _9853 = _7475.specular_and_specular_tint;
            _9854 = _7475.clearcoat_and_clearcoat_roughness;
            _9855 = _7475.normal_map_strength_unorm;
            _9785 = -_9785;
            _9784 = -_9784;
            _9783 = -_9783;
            _9782 = -_9782;
        }
        float3 param_3 = _9785;
        float4x4 param_4 = _7161.inv_xform;
        _9785 = TransformNormal(param_3, param_4);
        float3 param_5 = _9784;
        float4x4 param_6 = _7161.inv_xform;
        _9784 = TransformNormal(param_5, param_6);
        float3 param_7 = _9783;
        float4x4 param_8 = _7161.inv_xform;
        _9783 = TransformNormal(param_7, param_8);
        float3 param_9 = _9782;
        float4x4 param_10 = _7161.inv_xform;
        _9785 = normalize(_9785);
        _9784 = normalize(_9784);
        _9783 = normalize(_9783);
        _9782 = normalize(TransformNormal(param_9, param_10));
        float _7615 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _7625 = mad(0.5f, log2(abs(mad(_7219.t[0][0] - _7173.t[0][0], _7265.t[0][1] - _7173.t[0][1], -((_7265.t[0][0] - _7173.t[0][0]) * (_7219.t[0][1] - _7173.t[0][1])))) / _7413), log2(_7615));
        uint param_11 = uint(hash(ray.xy));
        float _7632 = construct_float(param_11);
        uint param_12 = uint(hash(hash(ray.xy)));
        float _7639 = construct_float(param_12);
        float param_13[4] = ray.ior;
        bool param_14 = _7084;
        float param_15 = 1.0f;
        float _7648 = peek_ior_stack(param_13, param_14, param_15);
        float3 col = 0.0f.xxx;
        int _7653 = ray.depth & 255;
        int _7658 = (ray.depth >> 8) & 255;
        int _7663 = (ray.depth >> 16) & 255;
        int _7674 = (_7653 + _7658) + _7663;
        int _7682 = _3539_g_params.hi + ((_7674 + ((ray.depth >> 24) & 255)) * 7);
        float mix_rand = frac(asfloat(_3523.Load(_7682 * 4 + 0)) + _7632);
        float mix_weight = 1.0f;
        float _7719;
        float _7736;
        float _7762;
        float _7829;
        while (_9846 == 4u)
        {
            float mix_val = _9847;
            if (_10236 != 4294967295u)
            {
                mix_val *= SampleBilinear(_10236, _7392, 0).x;
            }
            if (_7084)
            {
                _7719 = _7648 / _9849;
            }
            else
            {
                _7719 = _9849 / _7648;
            }
            if (_9849 != 0.0f)
            {
                float param_16 = dot(_7015, _9784);
                float param_17 = _7719;
                _7736 = fresnel_dielectric_cos(param_16, param_17);
            }
            else
            {
                _7736 = 1.0f;
            }
            float _7751 = mix_val;
            float _7752 = _7751 * clamp(_7736, 0.0f, 1.0f);
            mix_val = _7752;
            if (mix_rand > _7752)
            {
                if ((_9845 & 2u) != 0u)
                {
                    _7762 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _7762 = 1.0f;
                }
                mix_weight *= _7762;
                material_t _7775;
                [unroll]
                for (int _108ident = 0; _108ident < 5; _108ident++)
                {
                    _7775.textures[_108ident] = _4759.Load(_108ident * 4 + _10238 * 76 + 0);
                }
                [unroll]
                for (int _109ident = 0; _109ident < 3; _109ident++)
                {
                    _7775.base_color[_109ident] = asfloat(_4759.Load(_109ident * 4 + _10238 * 76 + 20));
                }
                _7775.flags = _4759.Load(_10238 * 76 + 32);
                _7775.type = _4759.Load(_10238 * 76 + 36);
                _7775.tangent_rotation_or_strength = asfloat(_4759.Load(_10238 * 76 + 40));
                _7775.roughness_and_anisotropic = _4759.Load(_10238 * 76 + 44);
                _7775.ior = asfloat(_4759.Load(_10238 * 76 + 48));
                _7775.sheen_and_sheen_tint = _4759.Load(_10238 * 76 + 52);
                _7775.tint_and_metallic = _4759.Load(_10238 * 76 + 56);
                _7775.transmission_and_transmission_roughness = _4759.Load(_10238 * 76 + 60);
                _7775.specular_and_specular_tint = _4759.Load(_10238 * 76 + 64);
                _7775.clearcoat_and_clearcoat_roughness = _4759.Load(_10238 * 76 + 68);
                _7775.normal_map_strength_unorm = _4759.Load(_10238 * 76 + 72);
                _10235 = _7775.textures[0];
                _10236 = _7775.textures[1];
                _10237 = _7775.textures[2];
                _10238 = _7775.textures[3];
                _10239 = _7775.textures[4];
                _10240 = _7775.base_color[0];
                _10241 = _7775.base_color[1];
                _10242 = _7775.base_color[2];
                _9845 = _7775.flags;
                _9846 = _7775.type;
                _9847 = _7775.tangent_rotation_or_strength;
                _9848 = _7775.roughness_and_anisotropic;
                _9849 = _7775.ior;
                _9850 = _7775.sheen_and_sheen_tint;
                _9851 = _7775.tint_and_metallic;
                _9852 = _7775.transmission_and_transmission_roughness;
                _9853 = _7775.specular_and_specular_tint;
                _9854 = _7775.clearcoat_and_clearcoat_roughness;
                _9855 = _7775.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_9845 & 2u) != 0u)
                {
                    _7829 = 1.0f / mix_val;
                }
                else
                {
                    _7829 = 1.0f;
                }
                mix_weight *= _7829;
                material_t _7841;
                [unroll]
                for (int _110ident = 0; _110ident < 5; _110ident++)
                {
                    _7841.textures[_110ident] = _4759.Load(_110ident * 4 + _10239 * 76 + 0);
                }
                [unroll]
                for (int _111ident = 0; _111ident < 3; _111ident++)
                {
                    _7841.base_color[_111ident] = asfloat(_4759.Load(_111ident * 4 + _10239 * 76 + 20));
                }
                _7841.flags = _4759.Load(_10239 * 76 + 32);
                _7841.type = _4759.Load(_10239 * 76 + 36);
                _7841.tangent_rotation_or_strength = asfloat(_4759.Load(_10239 * 76 + 40));
                _7841.roughness_and_anisotropic = _4759.Load(_10239 * 76 + 44);
                _7841.ior = asfloat(_4759.Load(_10239 * 76 + 48));
                _7841.sheen_and_sheen_tint = _4759.Load(_10239 * 76 + 52);
                _7841.tint_and_metallic = _4759.Load(_10239 * 76 + 56);
                _7841.transmission_and_transmission_roughness = _4759.Load(_10239 * 76 + 60);
                _7841.specular_and_specular_tint = _4759.Load(_10239 * 76 + 64);
                _7841.clearcoat_and_clearcoat_roughness = _4759.Load(_10239 * 76 + 68);
                _7841.normal_map_strength_unorm = _4759.Load(_10239 * 76 + 72);
                _10235 = _7841.textures[0];
                _10236 = _7841.textures[1];
                _10237 = _7841.textures[2];
                _10238 = _7841.textures[3];
                _10239 = _7841.textures[4];
                _10240 = _7841.base_color[0];
                _10241 = _7841.base_color[1];
                _10242 = _7841.base_color[2];
                _9845 = _7841.flags;
                _9846 = _7841.type;
                _9847 = _7841.tangent_rotation_or_strength;
                _9848 = _7841.roughness_and_anisotropic;
                _9849 = _7841.ior;
                _9850 = _7841.sheen_and_sheen_tint;
                _9851 = _7841.tint_and_metallic;
                _9852 = _7841.transmission_and_transmission_roughness;
                _9853 = _7841.specular_and_specular_tint;
                _9854 = _7841.clearcoat_and_clearcoat_roughness;
                _9855 = _7841.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_10235 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_10235, _7392, 0).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_1003.Load(_10235 * 80 + 0) & 16384u) != 0u)
            {
                float3 _10807 = normals;
                _10807.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _10807;
            }
            float3 _7925 = _9784;
            _9784 = normalize(((_9782 * normals.x) + (_7925 * normals.z)) + (_9783 * normals.y));
            if ((_9855 & 65535u) != 65535u)
            {
                _9784 = normalize(_7925 + ((_9784 - _7925) * clamp(float(_9855 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_18 = _9785;
            float3 param_19 = -_7015;
            float3 param_20 = _9784;
            _9784 = ensure_valid_reflection(param_18, param_19, param_20);
        }
        float3 _7991 = ((_7311 * _7334) + (_7319 * inter.u)) + (_7327 * inter.v);
        float3 _7998 = float3(-_7991.z, 0.0f, _7991.x);
        float3 tangent = _7998;
        float3 param_21 = _7998;
        float4x4 param_22 = _7161.inv_xform;
        float3 _8004 = TransformNormal(param_21, param_22);
        tangent = _8004;
        float3 _8008 = cross(_8004, _9784);
        if (dot(_8008, _8008) == 0.0f)
        {
            float3 param_23 = _7991;
            float4x4 param_24 = _7161.inv_xform;
            tangent = TransformNormal(param_23, param_24);
        }
        if (_9847 != 0.0f)
        {
            float3 param_25 = tangent;
            float3 param_26 = _9784;
            float param_27 = _9847;
            tangent = rotate_around_axis(param_25, param_26, param_27);
        }
        float3 _8041 = normalize(cross(tangent, _9784));
        _9783 = _8041;
        _9782 = cross(_9784, _8041);
        float3 _9934 = 0.0f.xxx;
        float3 _9933 = 0.0f.xxx;
        float _9938 = 0.0f;
        float _9936 = 0.0f;
        float _9937 = 1.0f;
        bool _8057 = _3539_g_params.li_count != 0;
        bool _8063;
        if (_8057)
        {
            _8063 = _9846 != 3u;
        }
        else
        {
            _8063 = _8057;
        }
        float3 _9935;
        bool _9939;
        bool _9940;
        if (_8063)
        {
            float3 param_28 = _7051;
            float3 param_29 = _9782;
            float3 param_30 = _9783;
            float3 param_31 = _9784;
            int param_32 = _7682;
            float2 param_33 = float2(_7632, _7639);
            light_sample_t _9949 = { _9933, _9934, _9935, _9936, _9937, _9938, _9939, _9940 };
            light_sample_t param_34 = _9949;
            SampleLightSource(param_28, param_29, param_30, param_31, param_32, param_33, param_34);
            _9933 = param_34.col;
            _9934 = param_34.L;
            _9935 = param_34.lp;
            _9936 = param_34.area;
            _9937 = param_34.dist_mul;
            _9938 = param_34.pdf;
            _9939 = param_34.cast_shadow;
            _9940 = param_34.from_env;
        }
        float _8091 = dot(_9784, _9934);
        float3 base_color = float3(_10240, _10241, _10242);
        [branch]
        if (_10236 != 4294967295u)
        {
            base_color *= SampleBilinear(_10236, _7392, int(get_texture_lod(texSize(_10236), _7625)), true, true).xyz;
        }
        out_base_color = base_color;
        out_normals = _9784;
        float3 tint_color = 0.0f.xxx;
        float _8127 = lum(base_color);
        [flatten]
        if (_8127 > 0.0f)
        {
            tint_color = base_color / _8127.xxx;
        }
        float roughness = clamp(float(_9848 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_10237 != 4294967295u)
        {
            roughness *= SampleBilinear(_10237, _7392, int(get_texture_lod(texSize(_10237), _7625)), false, true).x;
        }
        float _8172 = frac(asfloat(_3523.Load((_7682 + 1) * 4 + 0)) + _7632);
        float _8181 = frac(asfloat(_3523.Load((_7682 + 2) * 4 + 0)) + _7639);
        float _10362 = 0.0f;
        float _10361 = 0.0f;
        float _10360 = 0.0f;
        float _9998[4];
        [unroll]
        for (int i = 0; i < 4; )
        {
            _9998[i] = ray.ior[i];
            i++;
            continue;
        }
        float _9999 = _7615;
        float _10000 = ray.cone_spread;
        int _10001 = ray.xy;
        float _9996 = 0.0f;
        float _10467 = 0.0f;
        float _10466 = 0.0f;
        float _10465 = 0.0f;
        int _10103 = ray.depth;
        int _10107 = ray.xy;
        int _10002;
        float _10105;
        float _10290;
        float _10291;
        float _10292;
        float _10325;
        float _10326;
        float _10327;
        float _10395;
        float _10396;
        float _10397;
        float _10430;
        float _10431;
        float _10432;
        [branch]
        if (_9846 == 0u)
        {
            [branch]
            if ((_9938 > 0.0f) && (_8091 > 0.0f))
            {
                light_sample_t _9966 = { _9933, _9934, _9935, _9936, _9937, _9938, _9939, _9940 };
                surface_t _9793 = { _7051, _9782, _9783, _9784, _9785, _7392 };
                float _10471[3] = { _10465, _10466, _10467 };
                float _10436[3] = { _10430, _10431, _10432 };
                float _10401[3] = { _10395, _10396, _10397 };
                shadow_ray_t _10117 = { _10401, _10103, _10436, _10105, _10471, _10107 };
                shadow_ray_t param_35 = _10117;
                float3 _8241 = Evaluate_DiffuseNode(_9966, ray, _9793, base_color, roughness, mix_weight, param_35);
                _10395 = param_35.o[0];
                _10396 = param_35.o[1];
                _10397 = param_35.o[2];
                _10103 = param_35.depth;
                _10430 = param_35.d[0];
                _10431 = param_35.d[1];
                _10432 = param_35.d[2];
                _10105 = param_35.dist;
                _10465 = param_35.c[0];
                _10466 = param_35.c[1];
                _10467 = param_35.c[2];
                _10107 = param_35.xy;
                col += _8241;
            }
            bool _8248 = _7653 < _3539_g_params.max_diff_depth;
            bool _8255;
            if (_8248)
            {
                _8255 = _7674 < _3539_g_params.max_total_depth;
            }
            else
            {
                _8255 = _8248;
            }
            [branch]
            if (_8255)
            {
                surface_t _9800 = { _7051, _9782, _9783, _9784, _9785, _7392 };
                float _10366[3] = { _10360, _10361, _10362 };
                float _10331[3] = { _10325, _10326, _10327 };
                float _10296[3] = { _10290, _10291, _10292 };
                ray_data_t _10016 = { _10296, _10331, _9996, _10366, _9998, _9999, _10000, _10001, _10002 };
                ray_data_t param_36 = _10016;
                Sample_DiffuseNode(ray, _9800, base_color, roughness, _8172, _8181, mix_weight, param_36);
                _10290 = param_36.o[0];
                _10291 = param_36.o[1];
                _10292 = param_36.o[2];
                _10325 = param_36.d[0];
                _10326 = param_36.d[1];
                _10327 = param_36.d[2];
                _9996 = param_36.pdf;
                _10360 = param_36.c[0];
                _10361 = param_36.c[1];
                _10362 = param_36.c[2];
                _9998 = param_36.ior;
                _9999 = param_36.cone_width;
                _10000 = param_36.cone_spread;
                _10001 = param_36.xy;
                _10002 = param_36.depth;
            }
        }
        else
        {
            [branch]
            if (_9846 == 1u)
            {
                float param_37 = 1.0f;
                float param_38 = 1.5f;
                float _8279 = fresnel_dielectric_cos(param_37, param_38);
                float _8283 = roughness * roughness;
                bool _8286 = _9938 > 0.0f;
                bool _8293;
                if (_8286)
                {
                    _8293 = (_8283 * _8283) >= 1.0000000116860974230803549289703e-07f;
                }
                else
                {
                    _8293 = _8286;
                }
                [branch]
                if (_8293 && (_8091 > 0.0f))
                {
                    light_sample_t _9975 = { _9933, _9934, _9935, _9936, _9937, _9938, _9939, _9940 };
                    surface_t _9807 = { _7051, _9782, _9783, _9784, _9785, _7392 };
                    float _10478[3] = { _10465, _10466, _10467 };
                    float _10443[3] = { _10430, _10431, _10432 };
                    float _10408[3] = { _10395, _10396, _10397 };
                    shadow_ray_t _10130 = { _10408, _10103, _10443, _10105, _10478, _10107 };
                    shadow_ray_t param_39 = _10130;
                    float3 _8308 = Evaluate_GlossyNode(_9975, ray, _9807, base_color, roughness, 1.5f, _8279, mix_weight, param_39);
                    _10395 = param_39.o[0];
                    _10396 = param_39.o[1];
                    _10397 = param_39.o[2];
                    _10103 = param_39.depth;
                    _10430 = param_39.d[0];
                    _10431 = param_39.d[1];
                    _10432 = param_39.d[2];
                    _10105 = param_39.dist;
                    _10465 = param_39.c[0];
                    _10466 = param_39.c[1];
                    _10467 = param_39.c[2];
                    _10107 = param_39.xy;
                    col += _8308;
                }
                bool _8315 = _7658 < _3539_g_params.max_spec_depth;
                bool _8322;
                if (_8315)
                {
                    _8322 = _7674 < _3539_g_params.max_total_depth;
                }
                else
                {
                    _8322 = _8315;
                }
                [branch]
                if (_8322)
                {
                    surface_t _9814 = { _7051, _9782, _9783, _9784, _9785, _7392 };
                    float _10373[3] = { _10360, _10361, _10362 };
                    float _10338[3] = { _10325, _10326, _10327 };
                    float _10303[3] = { _10290, _10291, _10292 };
                    ray_data_t _10035 = { _10303, _10338, _9996, _10373, _9998, _9999, _10000, _10001, _10002 };
                    ray_data_t param_40 = _10035;
                    Sample_GlossyNode(ray, _9814, base_color, roughness, 1.5f, _8279, _8172, _8181, mix_weight, param_40);
                    _10290 = param_40.o[0];
                    _10291 = param_40.o[1];
                    _10292 = param_40.o[2];
                    _10325 = param_40.d[0];
                    _10326 = param_40.d[1];
                    _10327 = param_40.d[2];
                    _9996 = param_40.pdf;
                    _10360 = param_40.c[0];
                    _10361 = param_40.c[1];
                    _10362 = param_40.c[2];
                    _9998 = param_40.ior;
                    _9999 = param_40.cone_width;
                    _10000 = param_40.cone_spread;
                    _10001 = param_40.xy;
                    _10002 = param_40.depth;
                }
            }
            else
            {
                [branch]
                if (_9846 == 2u)
                {
                    float _8346 = roughness * roughness;
                    bool _8349 = _9938 > 0.0f;
                    bool _8356;
                    if (_8349)
                    {
                        _8356 = (_8346 * _8346) >= 1.0000000116860974230803549289703e-07f;
                    }
                    else
                    {
                        _8356 = _8349;
                    }
                    [branch]
                    if (_8356 && (_8091 < 0.0f))
                    {
                        float _8364;
                        if (_7084)
                        {
                            _8364 = _9849 / _7648;
                        }
                        else
                        {
                            _8364 = _7648 / _9849;
                        }
                        light_sample_t _9984 = { _9933, _9934, _9935, _9936, _9937, _9938, _9939, _9940 };
                        surface_t _9821 = { _7051, _9782, _9783, _9784, _9785, _7392 };
                        float _10485[3] = { _10465, _10466, _10467 };
                        float _10450[3] = { _10430, _10431, _10432 };
                        float _10415[3] = { _10395, _10396, _10397 };
                        shadow_ray_t _10143 = { _10415, _10103, _10450, _10105, _10485, _10107 };
                        shadow_ray_t param_41 = _10143;
                        float3 _8386 = Evaluate_RefractiveNode(_9984, ray, _9821, base_color, _8346, _8364, mix_weight, param_41);
                        _10395 = param_41.o[0];
                        _10396 = param_41.o[1];
                        _10397 = param_41.o[2];
                        _10103 = param_41.depth;
                        _10430 = param_41.d[0];
                        _10431 = param_41.d[1];
                        _10432 = param_41.d[2];
                        _10105 = param_41.dist;
                        _10465 = param_41.c[0];
                        _10466 = param_41.c[1];
                        _10467 = param_41.c[2];
                        _10107 = param_41.xy;
                        col += _8386;
                    }
                    bool _8393 = _7663 < _3539_g_params.max_refr_depth;
                    bool _8400;
                    if (_8393)
                    {
                        _8400 = _7674 < _3539_g_params.max_total_depth;
                    }
                    else
                    {
                        _8400 = _8393;
                    }
                    [branch]
                    if (_8400)
                    {
                        surface_t _9828 = { _7051, _9782, _9783, _9784, _9785, _7392 };
                        float _10380[3] = { _10360, _10361, _10362 };
                        float _10345[3] = { _10325, _10326, _10327 };
                        float _10310[3] = { _10290, _10291, _10292 };
                        ray_data_t _10054 = { _10310, _10345, _9996, _10380, _9998, _9999, _10000, _10001, _10002 };
                        ray_data_t param_42 = _10054;
                        Sample_RefractiveNode(ray, _9828, base_color, roughness, _7084, _9849, _7648, _8172, _8181, mix_weight, param_42);
                        _10290 = param_42.o[0];
                        _10291 = param_42.o[1];
                        _10292 = param_42.o[2];
                        _10325 = param_42.d[0];
                        _10326 = param_42.d[1];
                        _10327 = param_42.d[2];
                        _9996 = param_42.pdf;
                        _10360 = param_42.c[0];
                        _10361 = param_42.c[1];
                        _10362 = param_42.c[2];
                        _9998 = param_42.ior;
                        _9999 = param_42.cone_width;
                        _10000 = param_42.cone_spread;
                        _10001 = param_42.xy;
                        _10002 = param_42.depth;
                    }
                }
                else
                {
                    [branch]
                    if (_9846 == 3u)
                    {
                        col += (base_color * (mix_weight * _9847));
                    }
                    else
                    {
                        [branch]
                        if (_9846 == 6u)
                        {
                            float metallic = clamp(float((_9851 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_10238 != 4294967295u)
                            {
                                metallic *= SampleBilinear(_10238, _7392, int(get_texture_lod(texSize(_10238), _7625))).x;
                            }
                            float specular = clamp(float(_9853 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_10239 != 4294967295u)
                            {
                                specular *= SampleBilinear(_10239, _7392, int(get_texture_lod(texSize(_10239), _7625))).x;
                            }
                            float _8519 = clamp(float(_9854 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8527 = clamp(float((_9854 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8535 = 2.0f * clamp(float(_9850 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float3 _8553 = lerp(1.0f.xxx, tint_color, clamp(float((_9850 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _8535;
                            float3 _8573 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_9853 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                            float _8582 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                            float param_43 = 1.0f;
                            float param_44 = _8582;
                            float _8588 = fresnel_dielectric_cos(param_43, param_44);
                            float _8596 = clamp(float((_9848 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8607 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _8519))) - 1.0f;
                            float param_45 = 1.0f;
                            float param_46 = _8607;
                            float _8613 = fresnel_dielectric_cos(param_45, param_46);
                            float _8628 = mad(roughness - 1.0f, 1.0f - clamp(float((_9852 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                            float _8634;
                            if (_7084)
                            {
                                _8634 = _9849 / _7648;
                            }
                            else
                            {
                                _8634 = _7648 / _9849;
                            }
                            float param_47 = dot(_7015, _9784);
                            float param_48 = 1.0f / _8634;
                            float _8657 = fresnel_dielectric_cos(param_47, param_48);
                            float param_49 = dot(_7015, _9784);
                            float param_50 = _8582;
                            lobe_weights_t _8696 = get_lobe_weights(lerp(_8127, 1.0f, _8535), lum(lerp(_8573, 1.0f.xxx, ((fresnel_dielectric_cos(param_49, param_50) - _8588) / (1.0f - _8588)).xxx)), specular, metallic, clamp(float(_9852 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _8519);
                            [branch]
                            if (_9938 > 0.0f)
                            {
                                light_sample_t _9993 = { _9933, _9934, _9935, _9936, _9937, _9938, _9939, _9940 };
                                surface_t _9835 = { _7051, _9782, _9783, _9784, _9785, _7392 };
                                diff_params_t _10185 = { base_color, _8553, roughness };
                                spec_params_t _10200 = { _8573, roughness, _8582, _8588, _8596 };
                                clearcoat_params_t _10213 = { _8527, _8607, _8613 };
                                transmission_params_t _10228 = { _8628, _9849, _8634, _8657, _7084 };
                                float _10492[3] = { _10465, _10466, _10467 };
                                float _10457[3] = { _10430, _10431, _10432 };
                                float _10422[3] = { _10395, _10396, _10397 };
                                shadow_ray_t _10156 = { _10422, _10103, _10457, _10105, _10492, _10107 };
                                shadow_ray_t param_51 = _10156;
                                float3 _8715 = Evaluate_PrincipledNode(_9993, ray, _9835, _8696, _10185, _10200, _10213, _10228, metallic, _8091, mix_weight, param_51);
                                _10395 = param_51.o[0];
                                _10396 = param_51.o[1];
                                _10397 = param_51.o[2];
                                _10103 = param_51.depth;
                                _10430 = param_51.d[0];
                                _10431 = param_51.d[1];
                                _10432 = param_51.d[2];
                                _10105 = param_51.dist;
                                _10465 = param_51.c[0];
                                _10466 = param_51.c[1];
                                _10467 = param_51.c[2];
                                _10107 = param_51.xy;
                                col += _8715;
                            }
                            surface_t _9842 = { _7051, _9782, _9783, _9784, _9785, _7392 };
                            diff_params_t _10189 = { base_color, _8553, roughness };
                            spec_params_t _10206 = { _8573, roughness, _8582, _8588, _8596 };
                            clearcoat_params_t _10217 = { _8527, _8607, _8613 };
                            transmission_params_t _10234 = { _8628, _9849, _8634, _8657, _7084 };
                            float param_52 = mix_rand;
                            float _10387[3] = { _10360, _10361, _10362 };
                            float _10352[3] = { _10325, _10326, _10327 };
                            float _10317[3] = { _10290, _10291, _10292 };
                            ray_data_t _10073 = { _10317, _10352, _9996, _10387, _9998, _9999, _10000, _10001, _10002 };
                            ray_data_t param_53 = _10073;
                            Sample_PrincipledNode(ray, _9842, _8696, _10189, _10206, _10217, _10234, metallic, _8172, _8181, param_52, mix_weight, param_53);
                            _10290 = param_53.o[0];
                            _10291 = param_53.o[1];
                            _10292 = param_53.o[2];
                            _10325 = param_53.d[0];
                            _10326 = param_53.d[1];
                            _10327 = param_53.d[2];
                            _9996 = param_53.pdf;
                            _10360 = param_53.c[0];
                            _10361 = param_53.c[1];
                            _10362 = param_53.c[2];
                            _9998 = param_53.ior;
                            _9999 = param_53.cone_width;
                            _10000 = param_53.cone_spread;
                            _10001 = param_53.xy;
                            _10002 = param_53.depth;
                        }
                    }
                }
            }
        }
        float _8749 = max(_10360, max(_10361, _10362));
        float _8761;
        if (_7674 > _3539_g_params.min_total_depth)
        {
            _8761 = max(0.0500000007450580596923828125f, 1.0f - _8749);
        }
        else
        {
            _8761 = 0.0f;
        }
        bool _8775 = (frac(asfloat(_3523.Load((_7682 + 6) * 4 + 0)) + _7632) >= _8761) && (_8749 > 0.0f);
        bool _8781;
        if (_8775)
        {
            _8781 = _9996 > 0.0f;
        }
        else
        {
            _8781 = _8775;
        }
        [branch]
        if (_8781)
        {
            float _8785 = _9996;
            float _8786 = min(_8785, 1000000.0f);
            _9996 = _8786;
            float _8789 = 1.0f - _8761;
            float _8791 = _10360;
            float _8792 = _8791 / _8789;
            _10360 = _8792;
            float _8797 = _10361;
            float _8798 = _8797 / _8789;
            _10361 = _8798;
            float _8803 = _10362;
            float _8804 = _8803 / _8789;
            _10362 = _8804;
            uint _8812;
            _8810.InterlockedAdd(0, 1u, _8812);
            _8821.Store(_8812 * 72 + 0, asuint(_10290));
            _8821.Store(_8812 * 72 + 4, asuint(_10291));
            _8821.Store(_8812 * 72 + 8, asuint(_10292));
            _8821.Store(_8812 * 72 + 12, asuint(_10325));
            _8821.Store(_8812 * 72 + 16, asuint(_10326));
            _8821.Store(_8812 * 72 + 20, asuint(_10327));
            _8821.Store(_8812 * 72 + 24, asuint(_8786));
            _8821.Store(_8812 * 72 + 28, asuint(_8792));
            _8821.Store(_8812 * 72 + 32, asuint(_8798));
            _8821.Store(_8812 * 72 + 36, asuint(_8804));
            _8821.Store(_8812 * 72 + 40, asuint(_9998[0]));
            _8821.Store(_8812 * 72 + 44, asuint(_9998[1]));
            _8821.Store(_8812 * 72 + 48, asuint(_9998[2]));
            _8821.Store(_8812 * 72 + 52, asuint(_9998[3]));
            _8821.Store(_8812 * 72 + 56, asuint(_9999));
            _8821.Store(_8812 * 72 + 60, asuint(_10000));
            _8821.Store(_8812 * 72 + 64, uint(_10001));
            _8821.Store(_8812 * 72 + 68, uint(_10002));
        }
        [branch]
        if (max(_10465, max(_10466, _10467)) > 0.0f)
        {
            float3 _8898 = _9935 - float3(_10395, _10396, _10397);
            float _8901 = length(_8898);
            float3 _8905 = _8898 / _8901.xxx;
            float sh_dist = _8901 * _9937;
            if (_9940)
            {
                sh_dist = -sh_dist;
            }
            float _8917 = _8905.x;
            _10430 = _8917;
            float _8920 = _8905.y;
            _10431 = _8920;
            float _8923 = _8905.z;
            _10432 = _8923;
            _10105 = sh_dist;
            uint _8929;
            _8810.InterlockedAdd(8, 1u, _8929);
            _8937.Store(_8929 * 48 + 0, asuint(_10395));
            _8937.Store(_8929 * 48 + 4, asuint(_10396));
            _8937.Store(_8929 * 48 + 8, asuint(_10397));
            _8937.Store(_8929 * 48 + 12, uint(_10103));
            _8937.Store(_8929 * 48 + 16, asuint(_8917));
            _8937.Store(_8929 * 48 + 20, asuint(_8920));
            _8937.Store(_8929 * 48 + 24, asuint(_8923));
            _8937.Store(_8929 * 48 + 28, asuint(sh_dist));
            _8937.Store(_8929 * 48 + 32, asuint(_10465));
            _8937.Store(_8929 * 48 + 36, asuint(_10466));
            _8937.Store(_8929 * 48 + 40, asuint(_10467));
            _8937.Store(_8929 * 48 + 44, uint(_10107));
        }
        _9180 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _9180;
}

void comp_main()
{
    do
    {
        bool _9001 = gl_GlobalInvocationID.x >= _3539_g_params.img_size.x;
        bool _9010;
        if (!_9001)
        {
            _9010 = gl_GlobalInvocationID.y >= _3539_g_params.img_size.y;
        }
        else
        {
            _9010 = _9001;
        }
        if (_9010)
        {
            break;
        }
        int _9017 = int(gl_GlobalInvocationID.x);
        int _9021 = int(gl_GlobalInvocationID.y);
        int _9029 = (_9021 * int(_3539_g_params.img_size.x)) + _9017;
        hit_data_t _9039;
        _9039.mask = int(_9035.Load(_9029 * 24 + 0));
        _9039.obj_index = int(_9035.Load(_9029 * 24 + 4));
        _9039.prim_index = int(_9035.Load(_9029 * 24 + 8));
        _9039.t = asfloat(_9035.Load(_9029 * 24 + 12));
        _9039.u = asfloat(_9035.Load(_9029 * 24 + 16));
        _9039.v = asfloat(_9035.Load(_9029 * 24 + 20));
        ray_data_t _9059;
        [unroll]
        for (int _112ident = 0; _112ident < 3; _112ident++)
        {
            _9059.o[_112ident] = asfloat(_9056.Load(_112ident * 4 + _9029 * 72 + 0));
        }
        [unroll]
        for (int _113ident = 0; _113ident < 3; _113ident++)
        {
            _9059.d[_113ident] = asfloat(_9056.Load(_113ident * 4 + _9029 * 72 + 12));
        }
        _9059.pdf = asfloat(_9056.Load(_9029 * 72 + 24));
        [unroll]
        for (int _114ident = 0; _114ident < 3; _114ident++)
        {
            _9059.c[_114ident] = asfloat(_9056.Load(_114ident * 4 + _9029 * 72 + 28));
        }
        [unroll]
        for (int _115ident = 0; _115ident < 4; _115ident++)
        {
            _9059.ior[_115ident] = asfloat(_9056.Load(_115ident * 4 + _9029 * 72 + 40));
        }
        _9059.cone_width = asfloat(_9056.Load(_9029 * 72 + 56));
        _9059.cone_spread = asfloat(_9056.Load(_9029 * 72 + 60));
        _9059.xy = int(_9056.Load(_9029 * 72 + 64));
        _9059.depth = int(_9056.Load(_9029 * 72 + 68));
        hit_data_t _9274 = { _9039.mask, _9039.obj_index, _9039.prim_index, _9039.t, _9039.u, _9039.v };
        hit_data_t param = _9274;
        float _9323[4] = { _9059.ior[0], _9059.ior[1], _9059.ior[2], _9059.ior[3] };
        float _9314[3] = { _9059.c[0], _9059.c[1], _9059.c[2] };
        float _9307[3] = { _9059.d[0], _9059.d[1], _9059.d[2] };
        float _9300[3] = { _9059.o[0], _9059.o[1], _9059.o[2] };
        ray_data_t _9293 = { _9300, _9307, _9059.pdf, _9314, _9323, _9059.cone_width, _9059.cone_spread, _9059.xy, _9059.depth };
        ray_data_t param_1 = _9293;
        float3 param_2 = 0.0f.xxx;
        float3 param_3 = 0.0f.xxx;
        float3 _9115 = ShadeSurface(param, param_1, param_2, param_3);
        int2 _9124 = int2(_9017, _9021);
        g_out_img[_9124] = float4(_9115, 1.0f);
        g_out_base_color_img[_9124] = float4(param_2, 0.0f);
        g_out_depth_normals_img[_9124] = float4(param_3, _9039.t);
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
