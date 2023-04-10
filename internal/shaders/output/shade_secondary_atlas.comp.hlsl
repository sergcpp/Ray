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
ByteAddressBuffer _7189 : register(t14, space0);
RWByteAddressBuffer _8928 : register(u3, space0);
RWByteAddressBuffer _8939 : register(u1, space0);
RWByteAddressBuffer _9055 : register(u2, space0);
ByteAddressBuffer _9134 : register(t7, space0);
ByteAddressBuffer _9151 : register(t6, space0);
ByteAddressBuffer _9276 : register(t10, space0);
cbuffer UniformParams
{
    Params _3539_g_params : packoffset(c0);
};

Texture2DArray<float4> g_atlases[7] : register(t21, space0);
SamplerState _g_atlases_sampler[7] : register(s21, space0);
Texture2D<float4> g_env_qtree : register(t18, space0);
SamplerState _g_env_qtree_sampler : register(s18, space0);
RWTexture2D<float4> g_out_img : register(u0, space0);

static uint3 gl_WorkGroupID;
static uint gl_LocalInvocationIndex;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint gl_LocalInvocationIndex : SV_GroupIndex;
};

float2 TransformUV(float2 _uv, atlas_texture_t t, int mip_level)
{
    uint _9451[14] = t.pos;
    uint _9454[14] = t.pos;
    uint _1096 = t.size & 16383u;
    uint _1099 = t.size >> uint(16);
    uint _1100 = _1099 & 16383u;
    float2 size = float2(float(_1096), float(_1100));
    if ((_1099 & 32768u) != 0u)
    {
        size = float2(float(_1096 >> uint(mip_level)), float(_1100 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_9451[mip_level] & 65535u), float((_9454[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
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
    float3 _5019;
    if ((ray.depth & 16777215) != 0)
    {
        _5019 = _3539_g_params.env_col.xyz;
    }
    else
    {
        _5019 = _3539_g_params.back_col.xyz;
    }
    float3 env_col = _5019;
    uint _5035;
    if ((ray.depth & 16777215) != 0)
    {
        _5035 = asuint(_3539_g_params.env_col.w);
    }
    else
    {
        _5035 = asuint(_3539_g_params.back_col.w);
    }
    float _5051;
    if ((ray.depth & 16777215) != 0)
    {
        _5051 = _3539_g_params.env_rotation;
    }
    else
    {
        _5051 = _3539_g_params.back_rotation;
    }
    if (_5035 != 4294967295u)
    {
        atlas_texture_t _5067;
        _5067.size = _1003.Load(_5035 * 80 + 0);
        _5067.atlas = _1003.Load(_5035 * 80 + 4);
        [unroll]
        for (int _58ident = 0; _58ident < 4; _58ident++)
        {
            _5067.page[_58ident] = _1003.Load(_58ident * 4 + _5035 * 80 + 8);
        }
        [unroll]
        for (int _59ident = 0; _59ident < 14; _59ident++)
        {
            _5067.pos[_59ident] = _1003.Load(_59ident * 4 + _5035 * 80 + 24);
        }
        uint _9821[14] = { _5067.pos[0], _5067.pos[1], _5067.pos[2], _5067.pos[3], _5067.pos[4], _5067.pos[5], _5067.pos[6], _5067.pos[7], _5067.pos[8], _5067.pos[9], _5067.pos[10], _5067.pos[11], _5067.pos[12], _5067.pos[13] };
        uint _9792[4] = { _5067.page[0], _5067.page[1], _5067.page[2], _5067.page[3] };
        atlas_texture_t _9783 = { _5067.size, _5067.atlas, _9792, _9821 };
        float param = _5051;
        env_col *= SampleLatlong_RGBE(_9783, _5012, param);
    }
    if (_3539_g_params.env_qtree_levels > 0)
    {
        float param_1 = ray.pdf;
        float param_2 = Evaluate_EnvQTree(_5051, g_env_qtree, _g_env_qtree_sampler, _3539_g_params.env_qtree_levels, _5012);
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
    float3 _5179 = float3(ray.d[0], ray.d[1], ray.d[2]);
    light_t _5193;
    _5193.type_and_param0 = _3558.Load4(((-1) - inter.obj_index) * 64 + 0);
    _5193.param1 = asfloat(_3558.Load4(((-1) - inter.obj_index) * 64 + 16));
    _5193.param2 = asfloat(_3558.Load4(((-1) - inter.obj_index) * 64 + 32));
    _5193.param3 = asfloat(_3558.Load4(((-1) - inter.obj_index) * 64 + 48));
    float3 lcol = asfloat(_5193.type_and_param0.yzw);
    [branch]
    if ((_5193.type_and_param0.x & 128u) != 0u)
    {
        float3 env_col = _3539_g_params.env_col.xyz;
        uint _5220 = asuint(_3539_g_params.env_col.w);
        if (_5220 != 4294967295u)
        {
            atlas_texture_t _5227;
            _5227.size = _1003.Load(_5220 * 80 + 0);
            _5227.atlas = _1003.Load(_5220 * 80 + 4);
            [unroll]
            for (int _60ident = 0; _60ident < 4; _60ident++)
            {
                _5227.page[_60ident] = _1003.Load(_60ident * 4 + _5220 * 80 + 8);
            }
            [unroll]
            for (int _61ident = 0; _61ident < 14; _61ident++)
            {
                _5227.pos[_61ident] = _1003.Load(_61ident * 4 + _5220 * 80 + 24);
            }
            uint _9883[14] = { _5227.pos[0], _5227.pos[1], _5227.pos[2], _5227.pos[3], _5227.pos[4], _5227.pos[5], _5227.pos[6], _5227.pos[7], _5227.pos[8], _5227.pos[9], _5227.pos[10], _5227.pos[11], _5227.pos[12], _5227.pos[13] };
            uint _9854[4] = { _5227.page[0], _5227.page[1], _5227.page[2], _5227.page[3] };
            atlas_texture_t _9845 = { _5227.size, _5227.atlas, _9854, _9883 };
            float param = _3539_g_params.env_rotation;
            env_col *= SampleLatlong_RGBE(_9845, _5179, param);
        }
        lcol *= env_col;
    }
    uint _5287 = _5193.type_and_param0.x & 31u;
    if (_5287 == 0u)
    {
        float param_1 = ray.pdf;
        float param_2 = (inter.t * inter.t) / ((0.5f * _5193.param1.w) * dot(_5179, normalize(_5193.param1.xyz - (float3(ray.o[0], ray.o[1], ray.o[2]) + (_5179 * inter.t)))));
        lcol *= power_heuristic(param_1, param_2);
        bool _5354 = _5193.param3.x > 0.0f;
        bool _5360;
        if (_5354)
        {
            _5360 = _5193.param3.y > 0.0f;
        }
        else
        {
            _5360 = _5354;
        }
        [branch]
        if (_5360)
        {
            [flatten]
            if (_5193.param3.y > 0.0f)
            {
                lcol *= clamp((_5193.param3.x - acos(clamp(-dot(_5179, _5193.param2.xyz), 0.0f, 1.0f))) / _5193.param3.y, 0.0f, 1.0f);
            }
        }
    }
    else
    {
        if (_5287 == 4u)
        {
            float param_3 = ray.pdf;
            float param_4 = (inter.t * inter.t) / (_5193.param1.w * dot(_5179, normalize(cross(_5193.param2.xyz, _5193.param3.xyz))));
            lcol *= power_heuristic(param_3, param_4);
        }
        else
        {
            if (_5287 == 5u)
            {
                float param_5 = ray.pdf;
                float param_6 = (inter.t * inter.t) / (_5193.param1.w * dot(_5179, normalize(cross(_5193.param2.xyz, _5193.param3.xyz))));
                lcol *= power_heuristic(param_5, param_6);
            }
            else
            {
                if (_5287 == 3u)
                {
                    float param_7 = ray.pdf;
                    float param_8 = (inter.t * inter.t) / (_5193.param1.w * (1.0f - abs(dot(_5179, _5193.param3.xyz))));
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
    float _9288;
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
            _9288 = stack[3];
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
            _9288 = stack[2];
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
            _9288 = stack[1];
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
            _9288 = stack[0];
            break;
        }
        _9288 = default_value;
        break;
    } while(false);
    return _9288;
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
    uint _9459[4];
    _9459[0] = _1133.page[0];
    _9459[1] = _1133.page[1];
    _9459[2] = _1133.page[2];
    _9459[3] = _1133.page[3];
    uint _9495[14] = { _1133.pos[0], _1133.pos[1], _1133.pos[2], _1133.pos[3], _1133.pos[4], _1133.pos[5], _1133.pos[6], _1133.pos[7], _1133.pos[8], _1133.pos[9], _1133.pos[10], _1133.pos[11], _1133.pos[12], _1133.pos[13] };
    atlas_texture_t _9465 = { _1133.size, _1133.atlas, _9459, _9495 };
    uint _1203 = _1133.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_1203)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_1203)], float3(TransformUV(uvs, _9465, lod) * 0.000118371215648949146270751953125f.xx, float((_9459[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
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
        float4 _10647 = res;
        _10647.x = _1243.x;
        float4 _10649 = _10647;
        _10649.y = _1243.y;
        float4 _10651 = _10649;
        _10651.z = _1243.z;
        res = _10651;
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
    float3 _9293;
    do
    {
        float _1512 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1512)
        {
            _9293 = N;
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
            float _10950 = (-0.5f) / _1552;
            float param_1 = mad(_10950, _1576, 1.0f);
            float _1608 = safe_sqrtf(param_1);
            float param_2 = _1577;
            float _1611 = safe_sqrtf(param_2);
            float2 _1612 = float2(_1608, _1611);
            float param_3 = mad(_10950, _1583, 1.0f);
            float _1617 = safe_sqrtf(param_3);
            float param_4 = _1584;
            float _1620 = safe_sqrtf(param_4);
            float2 _1621 = float2(_1617, _1620);
            float _10952 = -_1540;
            float _1637 = mad(2.0f * mad(_1608, _1536, _1611 * _1540), _1611, _10952);
            float _1653 = mad(2.0f * mad(_1617, _1536, _1620 * _1540), _1620, _10952);
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
                _9293 = Ng;
                break;
            }
            float _1690 = valid1 ? _1577 : _1584;
            float param_5 = 1.0f - _1690;
            float param_6 = _1690;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _9293 = (_1532 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _9293;
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
    float3 _9318;
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
            _9318 = N;
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
        _9318 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _9318;
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
            float2 _10634 = origin;
            _10634.x = origin.x + _step;
            origin = _10634;
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
            float2 _10637 = origin;
            _10637.y = origin.y + _step;
            origin = _10637;
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
        float3 _10714 = sampled_dir;
        float3 _3681 = ((param * _10714.x) + (param_1 * _10714.y)) + (_3638 * _10714.z);
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
                        uint _9639[14] = { _3969.pos[0], _3969.pos[1], _3969.pos[2], _3969.pos[3], _3969.pos[4], _3969.pos[5], _3969.pos[6], _3969.pos[7], _3969.pos[8], _3969.pos[9], _3969.pos[10], _3969.pos[11], _3969.pos[12], _3969.pos[13] };
                        uint _9610[4] = { _3969.page[0], _3969.page[1], _3969.page[2], _3969.page[3] };
                        atlas_texture_t _9539 = { _3969.size, _3969.atlas, _9610, _9639 };
                        float param_10 = _3539_g_params.env_rotation;
                        env_col *= SampleLatlong_RGBE(_9539, ls.L, param_10);
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
                            uint _9677[14] = { _4211.pos[0], _4211.pos[1], _4211.pos[2], _4211.pos[3], _4211.pos[4], _4211.pos[5], _4211.pos[6], _4211.pos[7], _4211.pos[8], _4211.pos[9], _4211.pos[10], _4211.pos[11], _4211.pos[12], _4211.pos[13] };
                            uint _9648[4] = { _4211.page[0], _4211.page[1], _4211.page[2], _4211.page[3] };
                            atlas_texture_t _9548 = { _4211.size, _4211.atlas, _9648, _9677 };
                            float param_13 = _3539_g_params.env_rotation;
                            env_col_1 *= SampleLatlong_RGBE(_9548, ls.L, param_13);
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
                                    uint _9762[14] = { _4937.pos[0], _4937.pos[1], _4937.pos[2], _4937.pos[3], _4937.pos[4], _4937.pos[5], _4937.pos[6], _4937.pos[7], _4937.pos[8], _4937.pos[9], _4937.pos[10], _4937.pos[11], _4937.pos[12], _4937.pos[13] };
                                    uint _9733[4] = { _4937.page[0], _4937.page[1], _4937.page[2], _4937.page[3] };
                                    atlas_texture_t _9601 = { _4937.size, _4937.atlas, _9733, _9762 };
                                    float param_20 = _3539_g_params.env_rotation;
                                    ls.col *= SampleLatlong_RGBE(_9601, ls.L, param_20);
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
    float3 _9298;
    do
    {
        float3 param = -float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param_1 = surf.N;
        float3 param_2 = ls.L;
        float param_3 = roughness;
        float3 param_4 = base_color;
        float4 _5557 = Evaluate_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_5 = ls.pdf;
            float param_6 = _5557.w;
            mis_weight = power_heuristic(param_5, param_6);
        }
        float3 _5580 = (ls.col * _5557.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9298 = _5580;
            break;
        }
        float3 param_7 = surf.P;
        float3 param_8 = surf.plane_N;
        float3 _5592 = offset_ray(param_7, param_8);
        sh_r.o[0] = _5592.x;
        sh_r.o[1] = _5592.y;
        sh_r.o[2] = _5592.z;
        sh_r.c[0] = ray.c[0] * _5580.x;
        sh_r.c[1] = ray.c[1] * _5580.y;
        sh_r.c[2] = ray.c[2] * _5580.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9298 = 0.0f.xxx;
        break;
    } while(false);
    return _9298;
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
    float4 _5843 = Sample_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8);
    new_ray.depth = ray.depth + 1;
    float3 param_9 = surf.P;
    float3 param_10 = surf.plane_N;
    float3 _5853 = offset_ray(param_9, param_10);
    new_ray.o[0] = _5853.x;
    new_ray.o[1] = _5853.y;
    new_ray.o[2] = _5853.z;
    new_ray.d[0] = param_8.x;
    new_ray.d[1] = param_8.y;
    new_ray.d[2] = param_8.z;
    new_ray.c[0] = ((ray.c[0] * _5843.x) * mix_weight) / _5843.w;
    new_ray.c[1] = ((ray.c[1] * _5843.y) * mix_weight) / _5843.w;
    new_ray.c[2] = ((ray.c[2] * _5843.z) * mix_weight) / _5843.w;
    new_ray.pdf = _5843.w;
}

float3 tangent_from_world(float3 T, float3 B, float3 N, float3 V)
{
    return float3(dot(V, T), dot(V, B), dot(V, N));
}

float D_GGX(float3 H, float alpha_x, float alpha_y)
{
    float _9351;
    do
    {
        if (H.z == 0.0f)
        {
            _9351 = 0.0f;
            break;
        }
        float _2244 = (-H.x) / (H.z * alpha_x);
        float _2250 = (-H.y) / (H.z * alpha_y);
        float _2259 = mad(_2250, _2250, mad(_2244, _2244, 1.0f));
        _9351 = 1.0f / (((((_2259 * _2259) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _9351;
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
    float3 _9303;
    do
    {
        float3 _5628 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5628;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - _5628);
        float _5666 = roughness * roughness;
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = _5666;
        float param_16 = _5666;
        float param_17 = spec_ior;
        float param_18 = spec_F0;
        float3 param_19 = base_color;
        float4 _5681 = Evaluate_GGXSpecular_BSDF(param_12, param_13, param_14, param_15, param_16, param_17, param_18, param_19);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_20 = ls.pdf;
            float param_21 = _5681.w;
            mis_weight = power_heuristic(param_20, param_21);
        }
        float3 _5704 = (ls.col * _5681.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9303 = _5704;
            break;
        }
        float3 param_22 = surf.P;
        float3 param_23 = surf.plane_N;
        float3 _5716 = offset_ray(param_22, param_23);
        sh_r.o[0] = _5716.x;
        sh_r.o[1] = _5716.y;
        sh_r.o[2] = _5716.z;
        sh_r.c[0] = ray.c[0] * _5704.x;
        sh_r.c[1] = ray.c[1] * _5704.y;
        sh_r.c[2] = ray.c[2] * _5704.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9303 = 0.0f.xxx;
        break;
    } while(false);
    return _9303;
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
    float4 _9323;
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
            _9323 = float4(_2899.x * 1000000.0f, _2899.y * 1000000.0f, _2899.z * 1000000.0f, 1000000.0f);
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
        _9323 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _9323;
}

void Sample_GlossyNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 param_4;
    float4 _5763 = Sample_GGXSpecular_BSDF(param, param_1, param_2, param_3, roughness, 0.0f, spec_ior, spec_F0, base_color, rand_u, rand_v, param_4);
    new_ray.depth = ray.depth + 256;
    float3 param_5 = surf.P;
    float3 param_6 = surf.plane_N;
    float3 _5774 = offset_ray(param_5, param_6);
    new_ray.o[0] = _5774.x;
    new_ray.o[1] = _5774.y;
    new_ray.o[2] = _5774.z;
    new_ray.d[0] = param_4.x;
    new_ray.d[1] = param_4.y;
    new_ray.d[2] = param_4.z;
    new_ray.c[0] = ((ray.c[0] * _5763.x) * mix_weight) / _5763.w;
    new_ray.c[1] = ((ray.c[1] * _5763.y) * mix_weight) / _5763.w;
    new_ray.c[2] = ((ray.c[2] * _5763.z) * mix_weight) / _5763.w;
    new_ray.pdf = _5763.w;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _9328;
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
            _9328 = 0.0f.xxxx;
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
        _9328 = float4(refr_col * (((((_3182 * _3198) * _3190) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3218) / view_dir_ts.z), (((_3182 * _3190) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3218) / view_dir_ts.z);
        break;
    } while(false);
    return _9328;
}

float3 Evaluate_RefractiveNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness2, float eta, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _9308;
    do
    {
        float3 _5906 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5906;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - (_5906 * eta));
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = roughness2;
        float param_16 = eta;
        float3 param_17 = base_color;
        float4 _5954 = Evaluate_GGXRefraction_BSDF(param_12, param_13, param_14, param_15, param_16, param_17);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_18 = ls.pdf;
            float param_19 = _5954.w;
            mis_weight = power_heuristic(param_18, param_19);
        }
        float3 _5977 = (ls.col * _5954.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9308 = _5977;
            break;
        }
        float3 param_20 = surf.P;
        float3 param_21 = -surf.plane_N;
        float3 _5990 = offset_ray(param_20, param_21);
        sh_r.o[0] = _5990.x;
        sh_r.o[1] = _5990.y;
        sh_r.o[2] = _5990.z;
        sh_r.c[0] = ray.c[0] * _5977.x;
        sh_r.c[1] = ray.c[1] * _5977.y;
        sh_r.c[2] = ray.c[2] * _5977.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9308 = 0.0f.xxx;
        break;
    } while(false);
    return _9308;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _9333;
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
                _9333 = 0.0f.xxxx;
                break;
            }
            float _3295 = mad(eta, _3273, -sqrt(_3283));
            out_V = float4(normalize((I * eta) + (N * _3295)), _3295);
            _9333 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
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
            _9333 = 0.0f.xxxx;
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
        _9333 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _9333;
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
    float _9341;
    do
    {
        if (stack[3] > 0.0f)
        {
            float param = stack[3];
            float param_1 = -1.0f;
            float _2350 = exchange(param, param_1);
            stack[3] = param;
            _9341 = _2350;
            break;
        }
        if (stack[2] > 0.0f)
        {
            float param_2 = stack[2];
            float param_3 = -1.0f;
            float _2363 = exchange(param_2, param_3);
            stack[2] = param_2;
            _9341 = _2363;
            break;
        }
        if (stack[1] > 0.0f)
        {
            float param_4 = stack[1];
            float param_5 = -1.0f;
            float _2376 = exchange(param_4, param_5);
            stack[1] = param_4;
            _9341 = _2376;
            break;
        }
        if (stack[0] > 0.0f)
        {
            float param_6 = stack[0];
            float param_7 = -1.0f;
            float _2389 = exchange(param_6, param_7);
            stack[0] = param_6;
            _9341 = _2389;
            break;
        }
        _9341 = default_value;
        break;
    } while(false);
    return _9341;
}

void Sample_RefractiveNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, bool is_backfacing, float int_ior, float ext_ior, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float _6027;
    if (is_backfacing)
    {
        _6027 = int_ior / ext_ior;
    }
    else
    {
        _6027 = ext_ior / int_ior;
    }
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float param_4 = roughness;
    float param_5 = _6027;
    float3 param_6 = base_color;
    float param_7 = rand_u;
    float param_8 = rand_v;
    float4 param_9;
    float4 _6051 = Sample_GGXRefraction_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9);
    new_ray.depth = ray.depth + 65536;
    new_ray.c[0] = ((ray.c[0] * _6051.x) * mix_weight) / _6051.w;
    new_ray.c[1] = ((ray.c[1] * _6051.y) * mix_weight) / _6051.w;
    new_ray.c[2] = ((ray.c[2] * _6051.z) * mix_weight) / _6051.w;
    new_ray.pdf = _6051.w;
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
        float _6107 = pop_ior_stack(param_11, param_12);
        new_ray.ior = param_11;
    }
    float3 param_13 = surf.P;
    float3 param_14 = -surf.plane_N;
    float3 _6116 = offset_ray(param_13, param_14);
    new_ray.o[0] = _6116.x;
    new_ray.o[1] = _6116.y;
    new_ray.o[2] = _6116.z;
    new_ray.d[0] = param_9.x;
    new_ray.d[1] = param_9.y;
    new_ray.d[2] = param_9.z;
}

lobe_weights_t get_lobe_weights(float base_color_lum, float spec_color_lum, float specular, float metallic, float transmission, float clearcoat)
{
    float _1715 = 1.0f - metallic;
    float _9496 = (base_color_lum * _1715) * (1.0f - transmission);
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
    float _9497 = _1726;
    float _1736 = 0.25f * clearcoat;
    float _9498 = _1736 * _1715;
    float _9499 = _1722 * base_color_lum;
    float _1745 = _9496;
    float _1754 = mad(_1722, base_color_lum, mad(_1736, _1715, _1745 + _1726));
    if (_1754 != 0.0f)
    {
        _9496 /= _1754;
        _9497 /= _1754;
        _9498 /= _1754;
        _9499 /= _1754;
    }
    lobe_weights_t _9504 = { _9496, _9497, _9498, _9499 };
    return _9504;
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
    float _9356;
    do
    {
        float _2470 = dot(N, L);
        if (_2470 <= 0.0f)
        {
            _9356 = 0.0f;
            break;
        }
        float param = _2470;
        float param_1 = dot(N, V);
        float _2491 = dot(L, H);
        float _2499 = mad((2.0f * _2491) * _2491, roughness, 0.5f);
        _9356 = lerp(1.0f, _2499, schlick_weight(param)) * lerp(1.0f, _2499, schlick_weight(param_1));
        break;
    } while(false);
    return _9356;
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
    float _9361;
    do
    {
        if (a >= 1.0f)
        {
            _9361 = 0.3183098733425140380859375f;
            break;
        }
        float _2218 = mad(a, a, -1.0f);
        _9361 = _2218 / ((3.1415927410125732421875f * log(a * a)) * mad(_2218 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _9361;
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
    float3 _9313;
    do
    {
        float3 _6139 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 lcol = 0.0f.xxx;
        float bsdf_pdf = 0.0f;
        bool _6144 = N_dot_L > 0.0f;
        [branch]
        if ((lobe_weights.diffuse > 1.0000000116860974230803549289703e-07f) && _6144)
        {
            float3 param = -_6139;
            float3 param_1 = surf.N;
            float3 param_2 = ls.L;
            float param_3 = diff.roughness;
            float3 param_4 = diff.base_color;
            float3 param_5 = diff.sheen_color;
            bool param_6 = false;
            float4 _6163 = Evaluate_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6);
            bsdf_pdf = mad(lobe_weights.diffuse, _6163.w, bsdf_pdf);
            lcol += (((ls.col * N_dot_L) * (_6163 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * ls.pdf).xxx);
        }
        float3 H;
        [flatten]
        if (_6144)
        {
            H = normalize(ls.L - _6139);
        }
        else
        {
            H = normalize(ls.L - (_6139 * trans.eta));
        }
        float _6202 = spec.roughness * spec.roughness;
        float _6207 = sqrt(mad(-0.89999997615814208984375f, spec.anisotropy, 1.0f));
        float _6211 = _6202 / _6207;
        float _6215 = _6202 * _6207;
        float3 param_7 = surf.T;
        float3 param_8 = surf.B;
        float3 param_9 = surf.N;
        float3 param_10 = -_6139;
        float3 _6226 = tangent_from_world(param_7, param_8, param_9, param_10);
        float3 param_11 = surf.T;
        float3 param_12 = surf.B;
        float3 param_13 = surf.N;
        float3 param_14 = ls.L;
        float3 _6236 = tangent_from_world(param_11, param_12, param_13, param_14);
        float3 param_15 = surf.T;
        float3 param_16 = surf.B;
        float3 param_17 = surf.N;
        float3 param_18 = H;
        float3 _6246 = tangent_from_world(param_15, param_16, param_17, param_18);
        bool _6248 = lobe_weights.specular > 0.0f;
        bool _6255;
        if (_6248)
        {
            _6255 = (_6211 * _6215) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _6255 = _6248;
        }
        [branch]
        if (_6255 && _6144)
        {
            float3 param_19 = _6226;
            float3 param_20 = _6246;
            float3 param_21 = _6236;
            float param_22 = _6211;
            float param_23 = _6215;
            float param_24 = spec.ior;
            float param_25 = spec.F0;
            float3 param_26 = spec.tmp_col;
            float4 _6277 = Evaluate_GGXSpecular_BSDF(param_19, param_20, param_21, param_22, param_23, param_24, param_25, param_26);
            bsdf_pdf = mad(lobe_weights.specular, _6277.w, bsdf_pdf);
            lcol += ((ls.col * _6277.xyz) / ls.pdf.xxx);
        }
        float _6296 = coat.roughness * coat.roughness;
        bool _6298 = lobe_weights.clearcoat > 0.0f;
        bool _6305;
        if (_6298)
        {
            _6305 = (_6296 * _6296) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _6305 = _6298;
        }
        [branch]
        if (_6305 && _6144)
        {
            float3 param_27 = _6226;
            float3 param_28 = _6246;
            float3 param_29 = _6236;
            float param_30 = _6296;
            float param_31 = coat.ior;
            float param_32 = coat.F0;
            float4 _6323 = Evaluate_PrincipledClearcoat_BSDF(param_27, param_28, param_29, param_30, param_31, param_32);
            bsdf_pdf = mad(lobe_weights.clearcoat, _6323.w, bsdf_pdf);
            lcol += (((ls.col * 0.25f) * _6323.xyz) / ls.pdf.xxx);
        }
        [branch]
        if (lobe_weights.refraction > 0.0f)
        {
            bool _6345 = trans.fresnel != 0.0f;
            bool _6352;
            if (_6345)
            {
                _6352 = (_6202 * _6202) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _6352 = _6345;
            }
            [branch]
            if (_6352 && _6144)
            {
                float3 param_33 = _6226;
                float3 param_34 = _6246;
                float3 param_35 = _6236;
                float param_36 = _6202;
                float param_37 = _6202;
                float param_38 = 1.0f;
                float param_39 = 0.0f;
                float3 param_40 = 1.0f.xxx;
                float4 _6371 = Evaluate_GGXSpecular_BSDF(param_33, param_34, param_35, param_36, param_37, param_38, param_39, param_40);
                bsdf_pdf = mad(lobe_weights.refraction * trans.fresnel, _6371.w, bsdf_pdf);
                lcol += ((ls.col * _6371.xyz) * (trans.fresnel / ls.pdf));
            }
            float _6393 = trans.roughness * trans.roughness;
            bool _6395 = trans.fresnel != 1.0f;
            bool _6402;
            if (_6395)
            {
                _6402 = (_6393 * _6393) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _6402 = _6395;
            }
            [branch]
            if (_6402 && (N_dot_L < 0.0f))
            {
                float3 param_41 = _6226;
                float3 param_42 = _6246;
                float3 param_43 = _6236;
                float param_44 = _6393;
                float param_45 = trans.eta;
                float3 param_46 = diff.base_color;
                float4 _6420 = Evaluate_GGXRefraction_BSDF(param_41, param_42, param_43, param_44, param_45, param_46);
                float _6423 = 1.0f - trans.fresnel;
                bsdf_pdf = mad(lobe_weights.refraction * _6423, _6420.w, bsdf_pdf);
                lcol += ((ls.col * _6420.xyz) * (_6423 / ls.pdf));
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
            _9313 = lcol;
            break;
        }
        float3 _6463;
        if (N_dot_L < 0.0f)
        {
            _6463 = -surf.plane_N;
        }
        else
        {
            _6463 = surf.plane_N;
        }
        float3 param_49 = surf.P;
        float3 param_50 = _6463;
        float3 _6474 = offset_ray(param_49, param_50);
        sh_r.o[0] = _6474.x;
        sh_r.o[1] = _6474.y;
        sh_r.o[2] = _6474.z;
        sh_r.c[0] = ray.c[0] * lcol.x;
        sh_r.c[1] = ray.c[1] * lcol.y;
        sh_r.c[2] = ray.c[2] * lcol.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9313 = 0.0f.xxx;
        break;
    } while(false);
    return _9313;
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
    float4 _9346;
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
            _9346 = float4(_3099, _3099, _3099, 1000000.0f);
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
        _9346 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _9346;
}

void Sample_PrincipledNode(ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float rand_u, float rand_v, inout float mix_rand, float mix_weight, inout ray_data_t new_ray)
{
    float3 _6509 = float3(ray.d[0], ray.d[1], ray.d[2]);
    int _6513 = ray.depth & 255;
    int _6517 = (ray.depth >> 8) & 255;
    int _6521 = (ray.depth >> 16) & 255;
    int _6532 = (_6513 + _6517) + _6521;
    [branch]
    if (mix_rand < lobe_weights.diffuse)
    {
        bool _6541 = _6513 < _3539_g_params.max_diff_depth;
        bool _6548;
        if (_6541)
        {
            _6548 = _6532 < _3539_g_params.max_total_depth;
        }
        else
        {
            _6548 = _6541;
        }
        if (_6548)
        {
            float3 param = surf.T;
            float3 param_1 = surf.B;
            float3 param_2 = surf.N;
            float3 param_3 = _6509;
            float param_4 = diff.roughness;
            float3 param_5 = diff.base_color;
            float3 param_6 = diff.sheen_color;
            bool param_7 = false;
            float param_8 = rand_u;
            float param_9 = rand_v;
            float3 param_10;
            float4 _6571 = Sample_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9, param_10);
            float3 _6576 = _6571.xyz * (1.0f - metallic);
            new_ray.depth = ray.depth + 1;
            float3 param_11 = surf.P;
            float3 param_12 = surf.plane_N;
            float3 _6591 = offset_ray(param_11, param_12);
            new_ray.o[0] = _6591.x;
            new_ray.o[1] = _6591.y;
            new_ray.o[2] = _6591.z;
            new_ray.d[0] = param_10.x;
            new_ray.d[1] = param_10.y;
            new_ray.d[2] = param_10.z;
            new_ray.c[0] = ((ray.c[0] * _6576.x) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[1] = ((ray.c[1] * _6576.y) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[2] = ((ray.c[2] * _6576.z) * mix_weight) / lobe_weights.diffuse;
            new_ray.pdf = _6571.w;
        }
    }
    else
    {
        float _6641 = lobe_weights.diffuse + lobe_weights.specular;
        [branch]
        if (mix_rand < _6641)
        {
            bool _6648 = _6517 < _3539_g_params.max_spec_depth;
            bool _6655;
            if (_6648)
            {
                _6655 = _6532 < _3539_g_params.max_total_depth;
            }
            else
            {
                _6655 = _6648;
            }
            if (_6655)
            {
                float3 param_13 = surf.T;
                float3 param_14 = surf.B;
                float3 param_15 = surf.N;
                float3 param_16 = _6509;
                float3 param_17;
                float4 _6674 = Sample_GGXSpecular_BSDF(param_13, param_14, param_15, param_16, spec.roughness, spec.anisotropy, spec.ior, spec.F0, spec.tmp_col, rand_u, rand_v, param_17);
                float _6679 = _6674.w * lobe_weights.specular;
                new_ray.depth = ray.depth + 256;
                new_ray.c[0] = ((ray.c[0] * _6674.x) * mix_weight) / _6679;
                new_ray.c[1] = ((ray.c[1] * _6674.y) * mix_weight) / _6679;
                new_ray.c[2] = ((ray.c[2] * _6674.z) * mix_weight) / _6679;
                new_ray.pdf = _6679;
                float3 param_18 = surf.P;
                float3 param_19 = surf.plane_N;
                float3 _6719 = offset_ray(param_18, param_19);
                new_ray.o[0] = _6719.x;
                new_ray.o[1] = _6719.y;
                new_ray.o[2] = _6719.z;
                new_ray.d[0] = param_17.x;
                new_ray.d[1] = param_17.y;
                new_ray.d[2] = param_17.z;
            }
        }
        else
        {
            float _6744 = _6641 + lobe_weights.clearcoat;
            [branch]
            if (mix_rand < _6744)
            {
                bool _6751 = _6517 < _3539_g_params.max_spec_depth;
                bool _6758;
                if (_6751)
                {
                    _6758 = _6532 < _3539_g_params.max_total_depth;
                }
                else
                {
                    _6758 = _6751;
                }
                if (_6758)
                {
                    float3 param_20 = surf.T;
                    float3 param_21 = surf.B;
                    float3 param_22 = surf.N;
                    float3 param_23 = _6509;
                    float param_24 = coat.roughness * coat.roughness;
                    float param_25 = coat.ior;
                    float param_26 = coat.F0;
                    float param_27 = rand_u;
                    float param_28 = rand_v;
                    float3 param_29;
                    float4 _6782 = Sample_PrincipledClearcoat_BSDF(param_20, param_21, param_22, param_23, param_24, param_25, param_26, param_27, param_28, param_29);
                    float _6787 = _6782.w * lobe_weights.clearcoat;
                    new_ray.depth = ray.depth + 256;
                    new_ray.c[0] = (((0.25f * ray.c[0]) * _6782.x) * mix_weight) / _6787;
                    new_ray.c[1] = (((0.25f * ray.c[1]) * _6782.y) * mix_weight) / _6787;
                    new_ray.c[2] = (((0.25f * ray.c[2]) * _6782.z) * mix_weight) / _6787;
                    new_ray.pdf = _6787;
                    float3 param_30 = surf.P;
                    float3 param_31 = surf.plane_N;
                    float3 _6830 = offset_ray(param_30, param_31);
                    new_ray.o[0] = _6830.x;
                    new_ray.o[1] = _6830.y;
                    new_ray.o[2] = _6830.z;
                    new_ray.d[0] = param_29.x;
                    new_ray.d[1] = param_29.y;
                    new_ray.d[2] = param_29.z;
                }
            }
            else
            {
                bool _6852 = mix_rand >= trans.fresnel;
                bool _6859;
                if (_6852)
                {
                    _6859 = _6521 < _3539_g_params.max_refr_depth;
                }
                else
                {
                    _6859 = _6852;
                }
                bool _6873;
                if (!_6859)
                {
                    bool _6865 = mix_rand < trans.fresnel;
                    bool _6872;
                    if (_6865)
                    {
                        _6872 = _6517 < _3539_g_params.max_spec_depth;
                    }
                    else
                    {
                        _6872 = _6865;
                    }
                    _6873 = _6872;
                }
                else
                {
                    _6873 = _6859;
                }
                bool _6880;
                if (_6873)
                {
                    _6880 = _6532 < _3539_g_params.max_total_depth;
                }
                else
                {
                    _6880 = _6873;
                }
                [branch]
                if (_6880)
                {
                    mix_rand -= _6744;
                    mix_rand /= lobe_weights.refraction;
                    float4 F;
                    float3 V;
                    [branch]
                    if (mix_rand < trans.fresnel)
                    {
                        float3 param_32 = surf.T;
                        float3 param_33 = surf.B;
                        float3 param_34 = surf.N;
                        float3 param_35 = _6509;
                        float3 param_36;
                        float4 _6910 = Sample_GGXSpecular_BSDF(param_32, param_33, param_34, param_35, spec.roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, rand_u, rand_v, param_36);
                        V = param_36;
                        F = _6910;
                        new_ray.depth = ray.depth + 256;
                        float3 param_37 = surf.P;
                        float3 param_38 = surf.plane_N;
                        float3 _6920 = offset_ray(param_37, param_38);
                        new_ray.o[0] = _6920.x;
                        new_ray.o[1] = _6920.y;
                        new_ray.o[2] = _6920.z;
                    }
                    else
                    {
                        float3 param_39 = surf.T;
                        float3 param_40 = surf.B;
                        float3 param_41 = surf.N;
                        float3 param_42 = _6509;
                        float param_43 = trans.roughness;
                        float param_44 = trans.eta;
                        float3 param_45 = diff.base_color;
                        float param_46 = rand_u;
                        float param_47 = rand_v;
                        float4 param_48;
                        float4 _6949 = Sample_GGXRefraction_BSDF(param_39, param_40, param_41, param_42, param_43, param_44, param_45, param_46, param_47, param_48);
                        F = _6949;
                        V = param_48.xyz;
                        new_ray.depth = ray.depth + 65536;
                        float3 param_49 = surf.P;
                        float3 param_50 = -surf.plane_N;
                        float3 _6962 = offset_ray(param_49, param_50);
                        new_ray.o[0] = _6962.x;
                        new_ray.o[1] = _6962.y;
                        new_ray.o[2] = _6962.z;
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
                            float _6988 = pop_ior_stack(param_52, param_53);
                            new_ray.ior = param_52;
                        }
                    }
                    float4 _10898 = F;
                    float _6994 = _10898.w * lobe_weights.refraction;
                    float4 _10900 = _10898;
                    _10900.w = _6994;
                    F = _10900;
                    new_ray.c[0] = ((ray.c[0] * _10898.x) * mix_weight) / _6994;
                    new_ray.c[1] = ((ray.c[1] * _10898.y) * mix_weight) / _6994;
                    new_ray.c[2] = ((ray.c[2] * _10898.z) * mix_weight) / _6994;
                    new_ray.pdf = _6994;
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
    float3 _9283;
    do
    {
        float3 _7050 = float3(ray.d[0], ray.d[1], ray.d[2]);
        [branch]
        if (inter.mask == 0)
        {
            ray_data_t param = ray;
            float3 _7059 = Evaluate_EnvColor(param);
            _9283 = float3(ray.c[0] * _7059.x, ray.c[1] * _7059.y, ray.c[2] * _7059.z);
            break;
        }
        float3 _7086 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_7050 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            ray_data_t param_1 = ray;
            hit_data_t param_2 = inter;
            float3 _7098 = Evaluate_LightColor(param_1, param_2);
            _9283 = float3(ray.c[0] * _7098.x, ray.c[1] * _7098.y, ray.c[2] * _7098.z);
            break;
        }
        bool _7119 = inter.prim_index < 0;
        int _7122;
        if (_7119)
        {
            _7122 = (-1) - inter.prim_index;
        }
        else
        {
            _7122 = inter.prim_index;
        }
        uint _7133 = uint(_7122);
        material_t _7141;
        [unroll]
        for (int _89ident = 0; _89ident < 5; _89ident++)
        {
            _7141.textures[_89ident] = _4758.Load(_89ident * 4 + ((_4762.Load(_7133 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
        }
        [unroll]
        for (int _90ident = 0; _90ident < 3; _90ident++)
        {
            _7141.base_color[_90ident] = asfloat(_4758.Load(_90ident * 4 + ((_4762.Load(_7133 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
        }
        _7141.flags = _4758.Load(((_4762.Load(_7133 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
        _7141.type = _4758.Load(((_4762.Load(_7133 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
        _7141.tangent_rotation_or_strength = asfloat(_4758.Load(((_4762.Load(_7133 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
        _7141.roughness_and_anisotropic = _4758.Load(((_4762.Load(_7133 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
        _7141.ior = asfloat(_4758.Load(((_4762.Load(_7133 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
        _7141.sheen_and_sheen_tint = _4758.Load(((_4762.Load(_7133 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
        _7141.tint_and_metallic = _4758.Load(((_4762.Load(_7133 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
        _7141.transmission_and_transmission_roughness = _4758.Load(((_4762.Load(_7133 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
        _7141.specular_and_specular_tint = _4758.Load(((_4762.Load(_7133 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
        _7141.clearcoat_and_clearcoat_roughness = _4758.Load(((_4762.Load(_7133 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
        _7141.normal_map_strength_unorm = _4758.Load(((_4762.Load(_7133 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
        uint _10347 = _7141.textures[0];
        uint _10348 = _7141.textures[1];
        uint _10349 = _7141.textures[2];
        uint _10350 = _7141.textures[3];
        uint _10351 = _7141.textures[4];
        float _10352 = _7141.base_color[0];
        float _10353 = _7141.base_color[1];
        float _10354 = _7141.base_color[2];
        uint _9948 = _7141.flags;
        uint _9949 = _7141.type;
        float _9950 = _7141.tangent_rotation_or_strength;
        uint _9951 = _7141.roughness_and_anisotropic;
        float _9952 = _7141.ior;
        uint _9953 = _7141.sheen_and_sheen_tint;
        uint _9954 = _7141.tint_and_metallic;
        uint _9955 = _7141.transmission_and_transmission_roughness;
        uint _9956 = _7141.specular_and_specular_tint;
        uint _9957 = _7141.clearcoat_and_clearcoat_roughness;
        uint _9958 = _7141.normal_map_strength_unorm;
        transform_t _7196;
        _7196.xform = asfloat(uint4x4(_4405.Load4(asuint(asfloat(_7189.Load(inter.obj_index * 32 + 12))) * 128 + 0), _4405.Load4(asuint(asfloat(_7189.Load(inter.obj_index * 32 + 12))) * 128 + 16), _4405.Load4(asuint(asfloat(_7189.Load(inter.obj_index * 32 + 12))) * 128 + 32), _4405.Load4(asuint(asfloat(_7189.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _7196.inv_xform = asfloat(uint4x4(_4405.Load4(asuint(asfloat(_7189.Load(inter.obj_index * 32 + 12))) * 128 + 64), _4405.Load4(asuint(asfloat(_7189.Load(inter.obj_index * 32 + 12))) * 128 + 80), _4405.Load4(asuint(asfloat(_7189.Load(inter.obj_index * 32 + 12))) * 128 + 96), _4405.Load4(asuint(asfloat(_7189.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _7203 = _7133 * 3u;
        vertex_t _7208;
        [unroll]
        for (int _91ident = 0; _91ident < 3; _91ident++)
        {
            _7208.p[_91ident] = asfloat(_4430.Load(_91ident * 4 + _4434.Load(_7203 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _92ident = 0; _92ident < 3; _92ident++)
        {
            _7208.n[_92ident] = asfloat(_4430.Load(_92ident * 4 + _4434.Load(_7203 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _93ident = 0; _93ident < 3; _93ident++)
        {
            _7208.b[_93ident] = asfloat(_4430.Load(_93ident * 4 + _4434.Load(_7203 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _94ident = 0; _94ident < 2; _94ident++)
        {
            [unroll]
            for (int _95ident = 0; _95ident < 2; _95ident++)
            {
                _7208.t[_94ident][_95ident] = asfloat(_4430.Load(_95ident * 4 + _94ident * 8 + _4434.Load(_7203 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _7254;
        [unroll]
        for (int _96ident = 0; _96ident < 3; _96ident++)
        {
            _7254.p[_96ident] = asfloat(_4430.Load(_96ident * 4 + _4434.Load((_7203 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _97ident = 0; _97ident < 3; _97ident++)
        {
            _7254.n[_97ident] = asfloat(_4430.Load(_97ident * 4 + _4434.Load((_7203 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _98ident = 0; _98ident < 3; _98ident++)
        {
            _7254.b[_98ident] = asfloat(_4430.Load(_98ident * 4 + _4434.Load((_7203 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _99ident = 0; _99ident < 2; _99ident++)
        {
            [unroll]
            for (int _100ident = 0; _100ident < 2; _100ident++)
            {
                _7254.t[_99ident][_100ident] = asfloat(_4430.Load(_100ident * 4 + _99ident * 8 + _4434.Load((_7203 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _7300;
        [unroll]
        for (int _101ident = 0; _101ident < 3; _101ident++)
        {
            _7300.p[_101ident] = asfloat(_4430.Load(_101ident * 4 + _4434.Load((_7203 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _102ident = 0; _102ident < 3; _102ident++)
        {
            _7300.n[_102ident] = asfloat(_4430.Load(_102ident * 4 + _4434.Load((_7203 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _103ident = 0; _103ident < 3; _103ident++)
        {
            _7300.b[_103ident] = asfloat(_4430.Load(_103ident * 4 + _4434.Load((_7203 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _104ident = 0; _104ident < 2; _104ident++)
        {
            [unroll]
            for (int _105ident = 0; _105ident < 2; _105ident++)
            {
                _7300.t[_104ident][_105ident] = asfloat(_4430.Load(_105ident * 4 + _104ident * 8 + _4434.Load((_7203 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _7346 = float3(_7208.p[0], _7208.p[1], _7208.p[2]);
        float3 _7354 = float3(_7254.p[0], _7254.p[1], _7254.p[2]);
        float3 _7362 = float3(_7300.p[0], _7300.p[1], _7300.p[2]);
        float _7369 = (1.0f - inter.u) - inter.v;
        float3 _7401 = normalize(((float3(_7208.n[0], _7208.n[1], _7208.n[2]) * _7369) + (float3(_7254.n[0], _7254.n[1], _7254.n[2]) * inter.u)) + (float3(_7300.n[0], _7300.n[1], _7300.n[2]) * inter.v));
        float3 _9887 = _7401;
        float2 _7427 = ((float2(_7208.t[0][0], _7208.t[0][1]) * _7369) + (float2(_7254.t[0][0], _7254.t[0][1]) * inter.u)) + (float2(_7300.t[0][0], _7300.t[0][1]) * inter.v);
        float3 _7443 = cross(_7354 - _7346, _7362 - _7346);
        float _7448 = length(_7443);
        float3 _9888 = _7443 / _7448.xxx;
        float3 _7485 = ((float3(_7208.b[0], _7208.b[1], _7208.b[2]) * _7369) + (float3(_7254.b[0], _7254.b[1], _7254.b[2]) * inter.u)) + (float3(_7300.b[0], _7300.b[1], _7300.b[2]) * inter.v);
        float3 _9886 = _7485;
        float3 _9885 = cross(_7485, _7401);
        if (_7119)
        {
            if ((_4762.Load(_7133 * 4 + 0) & 65535u) == 65535u)
            {
                _9283 = 0.0f.xxx;
                break;
            }
            material_t _7510;
            [unroll]
            for (int _106ident = 0; _106ident < 5; _106ident++)
            {
                _7510.textures[_106ident] = _4758.Load(_106ident * 4 + (_4762.Load(_7133 * 4 + 0) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _107ident = 0; _107ident < 3; _107ident++)
            {
                _7510.base_color[_107ident] = asfloat(_4758.Load(_107ident * 4 + (_4762.Load(_7133 * 4 + 0) & 16383u) * 76 + 20));
            }
            _7510.flags = _4758.Load((_4762.Load(_7133 * 4 + 0) & 16383u) * 76 + 32);
            _7510.type = _4758.Load((_4762.Load(_7133 * 4 + 0) & 16383u) * 76 + 36);
            _7510.tangent_rotation_or_strength = asfloat(_4758.Load((_4762.Load(_7133 * 4 + 0) & 16383u) * 76 + 40));
            _7510.roughness_and_anisotropic = _4758.Load((_4762.Load(_7133 * 4 + 0) & 16383u) * 76 + 44);
            _7510.ior = asfloat(_4758.Load((_4762.Load(_7133 * 4 + 0) & 16383u) * 76 + 48));
            _7510.sheen_and_sheen_tint = _4758.Load((_4762.Load(_7133 * 4 + 0) & 16383u) * 76 + 52);
            _7510.tint_and_metallic = _4758.Load((_4762.Load(_7133 * 4 + 0) & 16383u) * 76 + 56);
            _7510.transmission_and_transmission_roughness = _4758.Load((_4762.Load(_7133 * 4 + 0) & 16383u) * 76 + 60);
            _7510.specular_and_specular_tint = _4758.Load((_4762.Load(_7133 * 4 + 0) & 16383u) * 76 + 64);
            _7510.clearcoat_and_clearcoat_roughness = _4758.Load((_4762.Load(_7133 * 4 + 0) & 16383u) * 76 + 68);
            _7510.normal_map_strength_unorm = _4758.Load((_4762.Load(_7133 * 4 + 0) & 16383u) * 76 + 72);
            _10347 = _7510.textures[0];
            _10348 = _7510.textures[1];
            _10349 = _7510.textures[2];
            _10350 = _7510.textures[3];
            _10351 = _7510.textures[4];
            _10352 = _7510.base_color[0];
            _10353 = _7510.base_color[1];
            _10354 = _7510.base_color[2];
            _9948 = _7510.flags;
            _9949 = _7510.type;
            _9950 = _7510.tangent_rotation_or_strength;
            _9951 = _7510.roughness_and_anisotropic;
            _9952 = _7510.ior;
            _9953 = _7510.sheen_and_sheen_tint;
            _9954 = _7510.tint_and_metallic;
            _9955 = _7510.transmission_and_transmission_roughness;
            _9956 = _7510.specular_and_specular_tint;
            _9957 = _7510.clearcoat_and_clearcoat_roughness;
            _9958 = _7510.normal_map_strength_unorm;
            _9888 = -_9888;
            _9887 = -_9887;
            _9886 = -_9886;
            _9885 = -_9885;
        }
        float3 param_3 = _9888;
        float4x4 param_4 = _7196.inv_xform;
        _9888 = TransformNormal(param_3, param_4);
        float3 param_5 = _9887;
        float4x4 param_6 = _7196.inv_xform;
        _9887 = TransformNormal(param_5, param_6);
        float3 param_7 = _9886;
        float4x4 param_8 = _7196.inv_xform;
        _9886 = TransformNormal(param_7, param_8);
        float3 param_9 = _9885;
        float4x4 param_10 = _7196.inv_xform;
        _9888 = normalize(_9888);
        _9887 = normalize(_9887);
        _9886 = normalize(_9886);
        _9885 = normalize(TransformNormal(param_9, param_10));
        float _7650 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _7660 = mad(0.5f, log2(abs(mad(_7254.t[0][0] - _7208.t[0][0], _7300.t[0][1] - _7208.t[0][1], -((_7300.t[0][0] - _7208.t[0][0]) * (_7254.t[0][1] - _7208.t[0][1])))) / _7448), log2(_7650));
        uint param_11 = uint(hash(ray.xy));
        float _7667 = construct_float(param_11);
        uint param_12 = uint(hash(hash(ray.xy)));
        float _7674 = construct_float(param_12);
        float param_13[4] = ray.ior;
        bool param_14 = _7119;
        float param_15 = 1.0f;
        float _7683 = peek_ior_stack(param_13, param_14, param_15);
        float3 col = 0.0f.xxx;
        int _7688 = ray.depth & 255;
        int _7693 = (ray.depth >> 8) & 255;
        int _7698 = (ray.depth >> 16) & 255;
        int _7709 = (_7688 + _7693) + _7698;
        int _7717 = _3539_g_params.hi + ((_7709 + ((ray.depth >> 24) & 255)) * 7);
        float mix_rand = frac(asfloat(_3523.Load(_7717 * 4 + 0)) + _7667);
        float mix_weight = 1.0f;
        float _7754;
        float _7771;
        float _7797;
        float _7864;
        while (_9949 == 4u)
        {
            float mix_val = _9950;
            if (_10348 != 4294967295u)
            {
                mix_val *= SampleBilinear(_10348, _7427, 0).x;
            }
            if (_7119)
            {
                _7754 = _7683 / _9952;
            }
            else
            {
                _7754 = _9952 / _7683;
            }
            if (_9952 != 0.0f)
            {
                float param_16 = dot(_7050, _9887);
                float param_17 = _7754;
                _7771 = fresnel_dielectric_cos(param_16, param_17);
            }
            else
            {
                _7771 = 1.0f;
            }
            float _7786 = mix_val;
            float _7787 = _7786 * clamp(_7771, 0.0f, 1.0f);
            mix_val = _7787;
            if (mix_rand > _7787)
            {
                if ((_9948 & 2u) != 0u)
                {
                    _7797 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _7797 = 1.0f;
                }
                mix_weight *= _7797;
                material_t _7810;
                [unroll]
                for (int _108ident = 0; _108ident < 5; _108ident++)
                {
                    _7810.textures[_108ident] = _4758.Load(_108ident * 4 + _10350 * 76 + 0);
                }
                [unroll]
                for (int _109ident = 0; _109ident < 3; _109ident++)
                {
                    _7810.base_color[_109ident] = asfloat(_4758.Load(_109ident * 4 + _10350 * 76 + 20));
                }
                _7810.flags = _4758.Load(_10350 * 76 + 32);
                _7810.type = _4758.Load(_10350 * 76 + 36);
                _7810.tangent_rotation_or_strength = asfloat(_4758.Load(_10350 * 76 + 40));
                _7810.roughness_and_anisotropic = _4758.Load(_10350 * 76 + 44);
                _7810.ior = asfloat(_4758.Load(_10350 * 76 + 48));
                _7810.sheen_and_sheen_tint = _4758.Load(_10350 * 76 + 52);
                _7810.tint_and_metallic = _4758.Load(_10350 * 76 + 56);
                _7810.transmission_and_transmission_roughness = _4758.Load(_10350 * 76 + 60);
                _7810.specular_and_specular_tint = _4758.Load(_10350 * 76 + 64);
                _7810.clearcoat_and_clearcoat_roughness = _4758.Load(_10350 * 76 + 68);
                _7810.normal_map_strength_unorm = _4758.Load(_10350 * 76 + 72);
                _10347 = _7810.textures[0];
                _10348 = _7810.textures[1];
                _10349 = _7810.textures[2];
                _10350 = _7810.textures[3];
                _10351 = _7810.textures[4];
                _10352 = _7810.base_color[0];
                _10353 = _7810.base_color[1];
                _10354 = _7810.base_color[2];
                _9948 = _7810.flags;
                _9949 = _7810.type;
                _9950 = _7810.tangent_rotation_or_strength;
                _9951 = _7810.roughness_and_anisotropic;
                _9952 = _7810.ior;
                _9953 = _7810.sheen_and_sheen_tint;
                _9954 = _7810.tint_and_metallic;
                _9955 = _7810.transmission_and_transmission_roughness;
                _9956 = _7810.specular_and_specular_tint;
                _9957 = _7810.clearcoat_and_clearcoat_roughness;
                _9958 = _7810.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_9948 & 2u) != 0u)
                {
                    _7864 = 1.0f / mix_val;
                }
                else
                {
                    _7864 = 1.0f;
                }
                mix_weight *= _7864;
                material_t _7876;
                [unroll]
                for (int _110ident = 0; _110ident < 5; _110ident++)
                {
                    _7876.textures[_110ident] = _4758.Load(_110ident * 4 + _10351 * 76 + 0);
                }
                [unroll]
                for (int _111ident = 0; _111ident < 3; _111ident++)
                {
                    _7876.base_color[_111ident] = asfloat(_4758.Load(_111ident * 4 + _10351 * 76 + 20));
                }
                _7876.flags = _4758.Load(_10351 * 76 + 32);
                _7876.type = _4758.Load(_10351 * 76 + 36);
                _7876.tangent_rotation_or_strength = asfloat(_4758.Load(_10351 * 76 + 40));
                _7876.roughness_and_anisotropic = _4758.Load(_10351 * 76 + 44);
                _7876.ior = asfloat(_4758.Load(_10351 * 76 + 48));
                _7876.sheen_and_sheen_tint = _4758.Load(_10351 * 76 + 52);
                _7876.tint_and_metallic = _4758.Load(_10351 * 76 + 56);
                _7876.transmission_and_transmission_roughness = _4758.Load(_10351 * 76 + 60);
                _7876.specular_and_specular_tint = _4758.Load(_10351 * 76 + 64);
                _7876.clearcoat_and_clearcoat_roughness = _4758.Load(_10351 * 76 + 68);
                _7876.normal_map_strength_unorm = _4758.Load(_10351 * 76 + 72);
                _10347 = _7876.textures[0];
                _10348 = _7876.textures[1];
                _10349 = _7876.textures[2];
                _10350 = _7876.textures[3];
                _10351 = _7876.textures[4];
                _10352 = _7876.base_color[0];
                _10353 = _7876.base_color[1];
                _10354 = _7876.base_color[2];
                _9948 = _7876.flags;
                _9949 = _7876.type;
                _9950 = _7876.tangent_rotation_or_strength;
                _9951 = _7876.roughness_and_anisotropic;
                _9952 = _7876.ior;
                _9953 = _7876.sheen_and_sheen_tint;
                _9954 = _7876.tint_and_metallic;
                _9955 = _7876.transmission_and_transmission_roughness;
                _9956 = _7876.specular_and_specular_tint;
                _9957 = _7876.clearcoat_and_clearcoat_roughness;
                _9958 = _7876.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_10347 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_10347, _7427, 0).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_1003.Load(_10347 * 80 + 0) & 16384u) != 0u)
            {
                float3 _10919 = normals;
                _10919.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _10919;
            }
            float3 _7960 = _9887;
            _9887 = normalize(((_9885 * normals.x) + (_7960 * normals.z)) + (_9886 * normals.y));
            if ((_9958 & 65535u) != 65535u)
            {
                _9887 = normalize(_7960 + ((_9887 - _7960) * clamp(float(_9958 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_18 = _9888;
            float3 param_19 = -_7050;
            float3 param_20 = _9887;
            _9887 = ensure_valid_reflection(param_18, param_19, param_20);
        }
        float3 _8026 = ((_7346 * _7369) + (_7354 * inter.u)) + (_7362 * inter.v);
        float3 _8033 = float3(-_8026.z, 0.0f, _8026.x);
        float3 tangent = _8033;
        float3 param_21 = _8033;
        float4x4 param_22 = _7196.inv_xform;
        float3 _8039 = TransformNormal(param_21, param_22);
        tangent = _8039;
        float3 _8043 = cross(_8039, _9887);
        if (dot(_8043, _8043) == 0.0f)
        {
            float3 param_23 = _8026;
            float4x4 param_24 = _7196.inv_xform;
            tangent = TransformNormal(param_23, param_24);
        }
        if (_9950 != 0.0f)
        {
            float3 param_25 = tangent;
            float3 param_26 = _9887;
            float param_27 = _9950;
            tangent = rotate_around_axis(param_25, param_26, param_27);
        }
        float3 _8076 = normalize(cross(tangent, _9887));
        _9886 = _8076;
        _9885 = cross(_9887, _8076);
        float3 _10046 = 0.0f.xxx;
        float3 _10045 = 0.0f.xxx;
        float _10050 = 0.0f;
        float _10048 = 0.0f;
        float _10049 = 1.0f;
        bool _8092 = _3539_g_params.li_count != 0;
        bool _8098;
        if (_8092)
        {
            _8098 = _9949 != 3u;
        }
        else
        {
            _8098 = _8092;
        }
        float3 _10047;
        bool _10051;
        bool _10052;
        if (_8098)
        {
            float3 param_28 = _7086;
            float3 param_29 = _9885;
            float3 param_30 = _9886;
            float3 param_31 = _9887;
            int param_32 = _7717;
            float2 param_33 = float2(_7667, _7674);
            light_sample_t _10061 = { _10045, _10046, _10047, _10048, _10049, _10050, _10051, _10052 };
            light_sample_t param_34 = _10061;
            SampleLightSource(param_28, param_29, param_30, param_31, param_32, param_33, param_34);
            _10045 = param_34.col;
            _10046 = param_34.L;
            _10047 = param_34.lp;
            _10048 = param_34.area;
            _10049 = param_34.dist_mul;
            _10050 = param_34.pdf;
            _10051 = param_34.cast_shadow;
            _10052 = param_34.from_env;
        }
        float _8126 = dot(_9887, _10046);
        float3 base_color = float3(_10352, _10353, _10354);
        [branch]
        if (_10348 != 4294967295u)
        {
            base_color *= SampleBilinear(_10348, _7427, int(get_texture_lod(texSize(_10348), _7660)), true, true).xyz;
        }
        out_base_color = base_color;
        out_normals = _9887;
        float3 tint_color = 0.0f.xxx;
        float _8162 = lum(base_color);
        [flatten]
        if (_8162 > 0.0f)
        {
            tint_color = base_color / _8162.xxx;
        }
        float roughness = clamp(float(_9951 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_10349 != 4294967295u)
        {
            roughness *= SampleBilinear(_10349, _7427, int(get_texture_lod(texSize(_10349), _7660)), false, true).x;
        }
        float _8207 = frac(asfloat(_3523.Load((_7717 + 1) * 4 + 0)) + _7667);
        float _8216 = frac(asfloat(_3523.Load((_7717 + 2) * 4 + 0)) + _7674);
        float _10474 = 0.0f;
        float _10473 = 0.0f;
        float _10472 = 0.0f;
        float _10110[4];
        [unroll]
        for (int i = 0; i < 4; )
        {
            _10110[i] = ray.ior[i];
            i++;
            continue;
        }
        float _10111 = _7650;
        float _10112 = ray.cone_spread;
        int _10113 = ray.xy;
        float _10108 = 0.0f;
        float _10579 = 0.0f;
        float _10578 = 0.0f;
        float _10577 = 0.0f;
        int _10215 = ray.depth;
        int _10219 = ray.xy;
        int _10114;
        float _10217;
        float _10402;
        float _10403;
        float _10404;
        float _10437;
        float _10438;
        float _10439;
        float _10507;
        float _10508;
        float _10509;
        float _10542;
        float _10543;
        float _10544;
        [branch]
        if (_9949 == 0u)
        {
            [branch]
            if ((_10050 > 0.0f) && (_8126 > 0.0f))
            {
                light_sample_t _10078 = { _10045, _10046, _10047, _10048, _10049, _10050, _10051, _10052 };
                surface_t _9896 = { _7086, _9885, _9886, _9887, _9888, _7427 };
                float _10583[3] = { _10577, _10578, _10579 };
                float _10548[3] = { _10542, _10543, _10544 };
                float _10513[3] = { _10507, _10508, _10509 };
                shadow_ray_t _10229 = { _10513, _10215, _10548, _10217, _10583, _10219 };
                shadow_ray_t param_35 = _10229;
                float3 _8276 = Evaluate_DiffuseNode(_10078, ray, _9896, base_color, roughness, mix_weight, param_35);
                _10507 = param_35.o[0];
                _10508 = param_35.o[1];
                _10509 = param_35.o[2];
                _10215 = param_35.depth;
                _10542 = param_35.d[0];
                _10543 = param_35.d[1];
                _10544 = param_35.d[2];
                _10217 = param_35.dist;
                _10577 = param_35.c[0];
                _10578 = param_35.c[1];
                _10579 = param_35.c[2];
                _10219 = param_35.xy;
                col += _8276;
            }
            bool _8283 = _7688 < _3539_g_params.max_diff_depth;
            bool _8290;
            if (_8283)
            {
                _8290 = _7709 < _3539_g_params.max_total_depth;
            }
            else
            {
                _8290 = _8283;
            }
            [branch]
            if (_8290)
            {
                surface_t _9903 = { _7086, _9885, _9886, _9887, _9888, _7427 };
                float _10478[3] = { _10472, _10473, _10474 };
                float _10443[3] = { _10437, _10438, _10439 };
                float _10408[3] = { _10402, _10403, _10404 };
                ray_data_t _10128 = { _10408, _10443, _10108, _10478, _10110, _10111, _10112, _10113, _10114 };
                ray_data_t param_36 = _10128;
                Sample_DiffuseNode(ray, _9903, base_color, roughness, _8207, _8216, mix_weight, param_36);
                _10402 = param_36.o[0];
                _10403 = param_36.o[1];
                _10404 = param_36.o[2];
                _10437 = param_36.d[0];
                _10438 = param_36.d[1];
                _10439 = param_36.d[2];
                _10108 = param_36.pdf;
                _10472 = param_36.c[0];
                _10473 = param_36.c[1];
                _10474 = param_36.c[2];
                _10110 = param_36.ior;
                _10111 = param_36.cone_width;
                _10112 = param_36.cone_spread;
                _10113 = param_36.xy;
                _10114 = param_36.depth;
            }
        }
        else
        {
            [branch]
            if (_9949 == 1u)
            {
                float param_37 = 1.0f;
                float param_38 = 1.5f;
                float _8314 = fresnel_dielectric_cos(param_37, param_38);
                float _8318 = roughness * roughness;
                bool _8321 = _10050 > 0.0f;
                bool _8328;
                if (_8321)
                {
                    _8328 = (_8318 * _8318) >= 1.0000000116860974230803549289703e-07f;
                }
                else
                {
                    _8328 = _8321;
                }
                [branch]
                if (_8328 && (_8126 > 0.0f))
                {
                    light_sample_t _10087 = { _10045, _10046, _10047, _10048, _10049, _10050, _10051, _10052 };
                    surface_t _9910 = { _7086, _9885, _9886, _9887, _9888, _7427 };
                    float _10590[3] = { _10577, _10578, _10579 };
                    float _10555[3] = { _10542, _10543, _10544 };
                    float _10520[3] = { _10507, _10508, _10509 };
                    shadow_ray_t _10242 = { _10520, _10215, _10555, _10217, _10590, _10219 };
                    shadow_ray_t param_39 = _10242;
                    float3 _8343 = Evaluate_GlossyNode(_10087, ray, _9910, base_color, roughness, 1.5f, _8314, mix_weight, param_39);
                    _10507 = param_39.o[0];
                    _10508 = param_39.o[1];
                    _10509 = param_39.o[2];
                    _10215 = param_39.depth;
                    _10542 = param_39.d[0];
                    _10543 = param_39.d[1];
                    _10544 = param_39.d[2];
                    _10217 = param_39.dist;
                    _10577 = param_39.c[0];
                    _10578 = param_39.c[1];
                    _10579 = param_39.c[2];
                    _10219 = param_39.xy;
                    col += _8343;
                }
                bool _8350 = _7693 < _3539_g_params.max_spec_depth;
                bool _8357;
                if (_8350)
                {
                    _8357 = _7709 < _3539_g_params.max_total_depth;
                }
                else
                {
                    _8357 = _8350;
                }
                [branch]
                if (_8357)
                {
                    surface_t _9917 = { _7086, _9885, _9886, _9887, _9888, _7427 };
                    float _10485[3] = { _10472, _10473, _10474 };
                    float _10450[3] = { _10437, _10438, _10439 };
                    float _10415[3] = { _10402, _10403, _10404 };
                    ray_data_t _10147 = { _10415, _10450, _10108, _10485, _10110, _10111, _10112, _10113, _10114 };
                    ray_data_t param_40 = _10147;
                    Sample_GlossyNode(ray, _9917, base_color, roughness, 1.5f, _8314, _8207, _8216, mix_weight, param_40);
                    _10402 = param_40.o[0];
                    _10403 = param_40.o[1];
                    _10404 = param_40.o[2];
                    _10437 = param_40.d[0];
                    _10438 = param_40.d[1];
                    _10439 = param_40.d[2];
                    _10108 = param_40.pdf;
                    _10472 = param_40.c[0];
                    _10473 = param_40.c[1];
                    _10474 = param_40.c[2];
                    _10110 = param_40.ior;
                    _10111 = param_40.cone_width;
                    _10112 = param_40.cone_spread;
                    _10113 = param_40.xy;
                    _10114 = param_40.depth;
                }
            }
            else
            {
                [branch]
                if (_9949 == 2u)
                {
                    float _8381 = roughness * roughness;
                    bool _8384 = _10050 > 0.0f;
                    bool _8391;
                    if (_8384)
                    {
                        _8391 = (_8381 * _8381) >= 1.0000000116860974230803549289703e-07f;
                    }
                    else
                    {
                        _8391 = _8384;
                    }
                    [branch]
                    if (_8391 && (_8126 < 0.0f))
                    {
                        float _8399;
                        if (_7119)
                        {
                            _8399 = _9952 / _7683;
                        }
                        else
                        {
                            _8399 = _7683 / _9952;
                        }
                        light_sample_t _10096 = { _10045, _10046, _10047, _10048, _10049, _10050, _10051, _10052 };
                        surface_t _9924 = { _7086, _9885, _9886, _9887, _9888, _7427 };
                        float _10597[3] = { _10577, _10578, _10579 };
                        float _10562[3] = { _10542, _10543, _10544 };
                        float _10527[3] = { _10507, _10508, _10509 };
                        shadow_ray_t _10255 = { _10527, _10215, _10562, _10217, _10597, _10219 };
                        shadow_ray_t param_41 = _10255;
                        float3 _8421 = Evaluate_RefractiveNode(_10096, ray, _9924, base_color, _8381, _8399, mix_weight, param_41);
                        _10507 = param_41.o[0];
                        _10508 = param_41.o[1];
                        _10509 = param_41.o[2];
                        _10215 = param_41.depth;
                        _10542 = param_41.d[0];
                        _10543 = param_41.d[1];
                        _10544 = param_41.d[2];
                        _10217 = param_41.dist;
                        _10577 = param_41.c[0];
                        _10578 = param_41.c[1];
                        _10579 = param_41.c[2];
                        _10219 = param_41.xy;
                        col += _8421;
                    }
                    bool _8428 = _7698 < _3539_g_params.max_refr_depth;
                    bool _8435;
                    if (_8428)
                    {
                        _8435 = _7709 < _3539_g_params.max_total_depth;
                    }
                    else
                    {
                        _8435 = _8428;
                    }
                    [branch]
                    if (_8435)
                    {
                        surface_t _9931 = { _7086, _9885, _9886, _9887, _9888, _7427 };
                        float _10492[3] = { _10472, _10473, _10474 };
                        float _10457[3] = { _10437, _10438, _10439 };
                        float _10422[3] = { _10402, _10403, _10404 };
                        ray_data_t _10166 = { _10422, _10457, _10108, _10492, _10110, _10111, _10112, _10113, _10114 };
                        ray_data_t param_42 = _10166;
                        Sample_RefractiveNode(ray, _9931, base_color, roughness, _7119, _9952, _7683, _8207, _8216, mix_weight, param_42);
                        _10402 = param_42.o[0];
                        _10403 = param_42.o[1];
                        _10404 = param_42.o[2];
                        _10437 = param_42.d[0];
                        _10438 = param_42.d[1];
                        _10439 = param_42.d[2];
                        _10108 = param_42.pdf;
                        _10472 = param_42.c[0];
                        _10473 = param_42.c[1];
                        _10474 = param_42.c[2];
                        _10110 = param_42.ior;
                        _10111 = param_42.cone_width;
                        _10112 = param_42.cone_spread;
                        _10113 = param_42.xy;
                        _10114 = param_42.depth;
                    }
                }
                else
                {
                    [branch]
                    if (_9949 == 3u)
                    {
                        float mis_weight = 1.0f;
                        [branch]
                        if ((_9948 & 1u) != 0u)
                        {
                            float3 _8505 = mul(float4(_7443, 0.0f), _7196.xform).xyz;
                            float _8508 = length(_8505);
                            float _8520 = abs(dot(_7050, _8505 / _8508.xxx));
                            if (_8520 > 0.0f)
                            {
                                float param_43 = ray.pdf;
                                float param_44 = (inter.t * inter.t) / ((0.5f * _8508) * _8520);
                                mis_weight = power_heuristic(param_43, param_44);
                            }
                        }
                        col += (base_color * ((mix_weight * mis_weight) * _9950));
                    }
                    else
                    {
                        [branch]
                        if (_9949 == 6u)
                        {
                            float metallic = clamp(float((_9954 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_10350 != 4294967295u)
                            {
                                metallic *= SampleBilinear(_10350, _7427, int(get_texture_lod(texSize(_10350), _7660))).x;
                            }
                            float specular = clamp(float(_9956 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_10351 != 4294967295u)
                            {
                                specular *= SampleBilinear(_10351, _7427, int(get_texture_lod(texSize(_10351), _7660))).x;
                            }
                            float _8637 = clamp(float(_9957 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8645 = clamp(float((_9957 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8653 = 2.0f * clamp(float(_9953 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float3 _8671 = lerp(1.0f.xxx, tint_color, clamp(float((_9953 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _8653;
                            float3 _8691 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_9956 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                            float _8700 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                            float param_45 = 1.0f;
                            float param_46 = _8700;
                            float _8706 = fresnel_dielectric_cos(param_45, param_46);
                            float _8714 = clamp(float((_9951 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8725 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _8637))) - 1.0f;
                            float param_47 = 1.0f;
                            float param_48 = _8725;
                            float _8731 = fresnel_dielectric_cos(param_47, param_48);
                            float _8746 = mad(roughness - 1.0f, 1.0f - clamp(float((_9955 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                            float _8752;
                            if (_7119)
                            {
                                _8752 = _9952 / _7683;
                            }
                            else
                            {
                                _8752 = _7683 / _9952;
                            }
                            float param_49 = dot(_7050, _9887);
                            float param_50 = 1.0f / _8752;
                            float _8775 = fresnel_dielectric_cos(param_49, param_50);
                            float param_51 = dot(_7050, _9887);
                            float param_52 = _8700;
                            lobe_weights_t _8814 = get_lobe_weights(lerp(_8162, 1.0f, _8653), lum(lerp(_8691, 1.0f.xxx, ((fresnel_dielectric_cos(param_51, param_52) - _8706) / (1.0f - _8706)).xxx)), specular, metallic, clamp(float(_9955 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _8637);
                            [branch]
                            if (_10050 > 0.0f)
                            {
                                light_sample_t _10105 = { _10045, _10046, _10047, _10048, _10049, _10050, _10051, _10052 };
                                surface_t _9938 = { _7086, _9885, _9886, _9887, _9888, _7427 };
                                diff_params_t _10297 = { base_color, _8671, roughness };
                                spec_params_t _10312 = { _8691, roughness, _8700, _8706, _8714 };
                                clearcoat_params_t _10325 = { _8645, _8725, _8731 };
                                transmission_params_t _10340 = { _8746, _9952, _8752, _8775, _7119 };
                                float _10604[3] = { _10577, _10578, _10579 };
                                float _10569[3] = { _10542, _10543, _10544 };
                                float _10534[3] = { _10507, _10508, _10509 };
                                shadow_ray_t _10268 = { _10534, _10215, _10569, _10217, _10604, _10219 };
                                shadow_ray_t param_53 = _10268;
                                float3 _8833 = Evaluate_PrincipledNode(_10105, ray, _9938, _8814, _10297, _10312, _10325, _10340, metallic, _8126, mix_weight, param_53);
                                _10507 = param_53.o[0];
                                _10508 = param_53.o[1];
                                _10509 = param_53.o[2];
                                _10215 = param_53.depth;
                                _10542 = param_53.d[0];
                                _10543 = param_53.d[1];
                                _10544 = param_53.d[2];
                                _10217 = param_53.dist;
                                _10577 = param_53.c[0];
                                _10578 = param_53.c[1];
                                _10579 = param_53.c[2];
                                _10219 = param_53.xy;
                                col += _8833;
                            }
                            surface_t _9945 = { _7086, _9885, _9886, _9887, _9888, _7427 };
                            diff_params_t _10301 = { base_color, _8671, roughness };
                            spec_params_t _10318 = { _8691, roughness, _8700, _8706, _8714 };
                            clearcoat_params_t _10329 = { _8645, _8725, _8731 };
                            transmission_params_t _10346 = { _8746, _9952, _8752, _8775, _7119 };
                            float param_54 = mix_rand;
                            float _10499[3] = { _10472, _10473, _10474 };
                            float _10464[3] = { _10437, _10438, _10439 };
                            float _10429[3] = { _10402, _10403, _10404 };
                            ray_data_t _10185 = { _10429, _10464, _10108, _10499, _10110, _10111, _10112, _10113, _10114 };
                            ray_data_t param_55 = _10185;
                            Sample_PrincipledNode(ray, _9945, _8814, _10301, _10318, _10329, _10346, metallic, _8207, _8216, param_54, mix_weight, param_55);
                            _10402 = param_55.o[0];
                            _10403 = param_55.o[1];
                            _10404 = param_55.o[2];
                            _10437 = param_55.d[0];
                            _10438 = param_55.d[1];
                            _10439 = param_55.d[2];
                            _10108 = param_55.pdf;
                            _10472 = param_55.c[0];
                            _10473 = param_55.c[1];
                            _10474 = param_55.c[2];
                            _10110 = param_55.ior;
                            _10111 = param_55.cone_width;
                            _10112 = param_55.cone_spread;
                            _10113 = param_55.xy;
                            _10114 = param_55.depth;
                        }
                    }
                }
            }
        }
        float _8867 = max(_10472, max(_10473, _10474));
        float _8879;
        if (_7709 > _3539_g_params.min_total_depth)
        {
            _8879 = max(0.0500000007450580596923828125f, 1.0f - _8867);
        }
        else
        {
            _8879 = 0.0f;
        }
        bool _8893 = (frac(asfloat(_3523.Load((_7717 + 6) * 4 + 0)) + _7667) >= _8879) && (_8867 > 0.0f);
        bool _8899;
        if (_8893)
        {
            _8899 = _10108 > 0.0f;
        }
        else
        {
            _8899 = _8893;
        }
        [branch]
        if (_8899)
        {
            float _8903 = _10108;
            float _8904 = min(_8903, 1000000.0f);
            _10108 = _8904;
            float _8907 = 1.0f - _8879;
            float _8909 = _10472;
            float _8910 = _8909 / _8907;
            _10472 = _8910;
            float _8915 = _10473;
            float _8916 = _8915 / _8907;
            _10473 = _8916;
            float _8921 = _10474;
            float _8922 = _8921 / _8907;
            _10474 = _8922;
            uint _8930;
            _8928.InterlockedAdd(0, 1u, _8930);
            _8939.Store(_8930 * 72 + 0, asuint(_10402));
            _8939.Store(_8930 * 72 + 4, asuint(_10403));
            _8939.Store(_8930 * 72 + 8, asuint(_10404));
            _8939.Store(_8930 * 72 + 12, asuint(_10437));
            _8939.Store(_8930 * 72 + 16, asuint(_10438));
            _8939.Store(_8930 * 72 + 20, asuint(_10439));
            _8939.Store(_8930 * 72 + 24, asuint(_8904));
            _8939.Store(_8930 * 72 + 28, asuint(_8910));
            _8939.Store(_8930 * 72 + 32, asuint(_8916));
            _8939.Store(_8930 * 72 + 36, asuint(_8922));
            _8939.Store(_8930 * 72 + 40, asuint(_10110[0]));
            _8939.Store(_8930 * 72 + 44, asuint(_10110[1]));
            _8939.Store(_8930 * 72 + 48, asuint(_10110[2]));
            _8939.Store(_8930 * 72 + 52, asuint(_10110[3]));
            _8939.Store(_8930 * 72 + 56, asuint(_10111));
            _8939.Store(_8930 * 72 + 60, asuint(_10112));
            _8939.Store(_8930 * 72 + 64, uint(_10113));
            _8939.Store(_8930 * 72 + 68, uint(_10114));
        }
        [branch]
        if (max(_10577, max(_10578, _10579)) > 0.0f)
        {
            float3 _9016 = _10047 - float3(_10507, _10508, _10509);
            float _9019 = length(_9016);
            float3 _9023 = _9016 / _9019.xxx;
            float sh_dist = _9019 * _10049;
            if (_10052)
            {
                sh_dist = -sh_dist;
            }
            float _9035 = _9023.x;
            _10542 = _9035;
            float _9038 = _9023.y;
            _10543 = _9038;
            float _9041 = _9023.z;
            _10544 = _9041;
            _10217 = sh_dist;
            uint _9047;
            _8928.InterlockedAdd(8, 1u, _9047);
            _9055.Store(_9047 * 48 + 0, asuint(_10507));
            _9055.Store(_9047 * 48 + 4, asuint(_10508));
            _9055.Store(_9047 * 48 + 8, asuint(_10509));
            _9055.Store(_9047 * 48 + 12, uint(_10215));
            _9055.Store(_9047 * 48 + 16, asuint(_9035));
            _9055.Store(_9047 * 48 + 20, asuint(_9038));
            _9055.Store(_9047 * 48 + 24, asuint(_9041));
            _9055.Store(_9047 * 48 + 28, asuint(sh_dist));
            _9055.Store(_9047 * 48 + 32, asuint(_10577));
            _9055.Store(_9047 * 48 + 36, asuint(_10578));
            _9055.Store(_9047 * 48 + 40, asuint(_10579));
            _9055.Store(_9047 * 48 + 44, uint(_10219));
        }
        _9283 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _9283;
}

void comp_main()
{
    do
    {
        int _9121 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_9121) >= _8928.Load(4))
        {
            break;
        }
        int _9137 = int(_9134.Load(_9121 * 72 + 64));
        int _9144 = int(_9134.Load(_9121 * 72 + 64));
        hit_data_t _9155;
        _9155.mask = int(_9151.Load(_9121 * 24 + 0));
        _9155.obj_index = int(_9151.Load(_9121 * 24 + 4));
        _9155.prim_index = int(_9151.Load(_9121 * 24 + 8));
        _9155.t = asfloat(_9151.Load(_9121 * 24 + 12));
        _9155.u = asfloat(_9151.Load(_9121 * 24 + 16));
        _9155.v = asfloat(_9151.Load(_9121 * 24 + 20));
        ray_data_t _9171;
        [unroll]
        for (int _112ident = 0; _112ident < 3; _112ident++)
        {
            _9171.o[_112ident] = asfloat(_9134.Load(_112ident * 4 + _9121 * 72 + 0));
        }
        [unroll]
        for (int _113ident = 0; _113ident < 3; _113ident++)
        {
            _9171.d[_113ident] = asfloat(_9134.Load(_113ident * 4 + _9121 * 72 + 12));
        }
        _9171.pdf = asfloat(_9134.Load(_9121 * 72 + 24));
        [unroll]
        for (int _114ident = 0; _114ident < 3; _114ident++)
        {
            _9171.c[_114ident] = asfloat(_9134.Load(_114ident * 4 + _9121 * 72 + 28));
        }
        [unroll]
        for (int _115ident = 0; _115ident < 4; _115ident++)
        {
            _9171.ior[_115ident] = asfloat(_9134.Load(_115ident * 4 + _9121 * 72 + 40));
        }
        _9171.cone_width = asfloat(_9134.Load(_9121 * 72 + 56));
        _9171.cone_spread = asfloat(_9134.Load(_9121 * 72 + 60));
        _9171.xy = int(_9134.Load(_9121 * 72 + 64));
        _9171.depth = int(_9134.Load(_9121 * 72 + 68));
        hit_data_t _9377 = { _9155.mask, _9155.obj_index, _9155.prim_index, _9155.t, _9155.u, _9155.v };
        hit_data_t param = _9377;
        float _9426[4] = { _9171.ior[0], _9171.ior[1], _9171.ior[2], _9171.ior[3] };
        float _9417[3] = { _9171.c[0], _9171.c[1], _9171.c[2] };
        float _9410[3] = { _9171.d[0], _9171.d[1], _9171.d[2] };
        float _9403[3] = { _9171.o[0], _9171.o[1], _9171.o[2] };
        ray_data_t _9396 = { _9403, _9410, _9171.pdf, _9417, _9426, _9171.cone_width, _9171.cone_spread, _9171.xy, _9171.depth };
        ray_data_t param_1 = _9396;
        float3 param_2 = 0.0f.xxx;
        float3 param_3 = 0.0f.xxx;
        float3 _9227 = ShadeSurface(param, param_1, param_2, param_3);
        int2 _9241 = int2((_9137 >> 16) & 65535, _9144 & 65535);
        g_out_img[_9241] = float4(min(_9227, _3539_g_params.clamp_val.xxx) + g_out_img[_9241].xyz, 1.0f);
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
