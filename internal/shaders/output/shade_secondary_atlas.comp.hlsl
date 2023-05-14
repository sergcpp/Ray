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

ByteAddressBuffer _1008 : register(t20, space0);
ByteAddressBuffer _3442 : register(t17, space0);
ByteAddressBuffer _3477 : register(t8, space0);
ByteAddressBuffer _3481 : register(t9, space0);
ByteAddressBuffer _4344 : register(t13, space0);
ByteAddressBuffer _4369 : register(t15, space0);
ByteAddressBuffer _4373 : register(t16, space0);
ByteAddressBuffer _4697 : register(t12, space0);
ByteAddressBuffer _4701 : register(t11, space0);
ByteAddressBuffer _7198 : register(t14, space0);
RWByteAddressBuffer _8895 : register(u3, space0);
RWByteAddressBuffer _8906 : register(u1, space0);
RWByteAddressBuffer _9022 : register(u2, space0);
ByteAddressBuffer _9101 : register(t7, space0);
ByteAddressBuffer _9118 : register(t6, space0);
ByteAddressBuffer _9243 : register(t10, space0);
cbuffer UniformParams
{
    Params _3458_g_params : packoffset(c0);
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

int hash(int x)
{
    uint _514 = uint(x);
    uint _521 = ((_514 >> uint(16)) ^ _514) * 73244475u;
    uint _526 = ((_521 >> uint(16)) ^ _521) * 73244475u;
    return int((_526 >> uint(16)) ^ _526);
}

float construct_float(inout uint m)
{
    m &= 8388607u;
    m |= 1065353216u;
    return asfloat(m) - 1.0f;
}

float2 TransformUV(float2 _uv, atlas_texture_t t, int mip_level)
{
    uint _9418[14] = t.pos;
    uint _9421[14] = t.pos;
    uint _1101 = t.size & 16383u;
    uint _1104 = t.size >> uint(16);
    uint _1105 = _1104 & 16383u;
    float2 size = float2(float(_1101), float(_1105));
    if ((_1104 & 32768u) != 0u)
    {
        size = float2(float(_1101 >> uint(mip_level)), float(_1105 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_9418[mip_level] & 65535u), float((_9421[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
}

float3 rgbe_to_rgb(float4 rgbe)
{
    return rgbe.xyz * exp2(mad(255.0f, rgbe.w, -128.0f));
}

float3 SampleLatlong_RGBE(atlas_texture_t t, float3 dir, float y_rotation, float2 rand)
{
    float _1276 = atan2(dir.z, dir.x) + y_rotation;
    float phi = _1276;
    if (_1276 < 0.0f)
    {
        phi += 6.283185482025146484375f;
    }
    if (phi > 6.283185482025146484375f)
    {
        phi -= 6.283185482025146484375f;
    }
    uint _1305 = t.atlas;
    float4 param = g_atlases[NonUniformResourceIndex(_1305)].Load(int4(int3(int2(TransformUV(float2(frac(phi * 0.15915493667125701904296875f), acos(clamp(dir.y, -1.0f, 1.0f)) * 0.3183098733425140380859375f), t, 0) + rand), int(t.page[0] & 255u)), 0));
    return rgbe_to_rgb(param);
}

float2 DirToCanonical(float3 d, float y_rotation)
{
    float _737 = (-atan2(d.z, d.x)) + y_rotation;
    float phi = _737;
    if (_737 < 0.0f)
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
    float2 _764 = DirToCanonical(L, -y_rotation);
    float factor = 1.0f;
    while (lod >= 0)
    {
        int2 _784 = clamp(int2(_764 * float(res)), int2(0, 0), (res - 1).xx);
        float4 quad = qtree_tex.Load(int3(_784 / int2(2, 2), lod));
        float _819 = ((quad.x + quad.y) + quad.z) + quad.w;
        if (_819 <= 0.0f)
        {
            break;
        }
        factor *= ((4.0f * quad[(0 | ((_784.x & 1) << 0)) | ((_784.y & 1) << 1)]) / _819);
        lod--;
        res *= 2;
    }
    return factor * 0.079577468335628509521484375f;
}

float power_heuristic(float a, float b)
{
    float _1326 = a * a;
    return _1326 / mad(b, b, _1326);
}

float3 Evaluate_EnvColor(ray_data_t ray, float2 tex_rand)
{
    float3 _4953 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 _4960;
    if ((ray.depth & 16777215) != 0)
    {
        _4960 = _3458_g_params.env_col.xyz;
    }
    else
    {
        _4960 = _3458_g_params.back_col.xyz;
    }
    float3 env_col = _4960;
    uint _4976;
    if ((ray.depth & 16777215) != 0)
    {
        _4976 = asuint(_3458_g_params.env_col.w);
    }
    else
    {
        _4976 = asuint(_3458_g_params.back_col.w);
    }
    float _4992;
    if ((ray.depth & 16777215) != 0)
    {
        _4992 = _3458_g_params.env_rotation;
    }
    else
    {
        _4992 = _3458_g_params.back_rotation;
    }
    if (_4976 != 4294967295u)
    {
        atlas_texture_t _5008;
        _5008.size = _1008.Load(_4976 * 80 + 0);
        _5008.atlas = _1008.Load(_4976 * 80 + 4);
        [unroll]
        for (int _58ident = 0; _58ident < 4; _58ident++)
        {
            _5008.page[_58ident] = _1008.Load(_58ident * 4 + _4976 * 80 + 8);
        }
        [unroll]
        for (int _59ident = 0; _59ident < 14; _59ident++)
        {
            _5008.pos[_59ident] = _1008.Load(_59ident * 4 + _4976 * 80 + 24);
        }
        uint _9788[14] = { _5008.pos[0], _5008.pos[1], _5008.pos[2], _5008.pos[3], _5008.pos[4], _5008.pos[5], _5008.pos[6], _5008.pos[7], _5008.pos[8], _5008.pos[9], _5008.pos[10], _5008.pos[11], _5008.pos[12], _5008.pos[13] };
        uint _9759[4] = { _5008.page[0], _5008.page[1], _5008.page[2], _5008.page[3] };
        atlas_texture_t _9750 = { _5008.size, _5008.atlas, _9759, _9788 };
        float param = _4992;
        env_col *= SampleLatlong_RGBE(_9750, _4953, param, tex_rand);
    }
    if (_3458_g_params.env_qtree_levels > 0)
    {
        float param_1 = ray.pdf;
        float param_2 = Evaluate_EnvQTree(_4992, g_env_qtree, _g_env_qtree_sampler, _3458_g_params.env_qtree_levels, _4953);
        env_col *= power_heuristic(param_1, param_2);
    }
    else
    {
        if (_3458_g_params.env_mult_importance != 0)
        {
            float param_3 = ray.pdf;
            float param_4 = 0.15915493667125701904296875f;
            env_col *= power_heuristic(param_3, param_4);
        }
    }
    return env_col;
}

float3 Evaluate_LightColor(ray_data_t ray, hit_data_t inter, float2 tex_rand)
{
    float3 _5120 = float3(ray.d[0], ray.d[1], ray.d[2]);
    light_t _5134;
    _5134.type_and_param0 = _3477.Load4(((-1) - inter.obj_index) * 64 + 0);
    _5134.param1 = asfloat(_3477.Load4(((-1) - inter.obj_index) * 64 + 16));
    _5134.param2 = asfloat(_3477.Load4(((-1) - inter.obj_index) * 64 + 32));
    _5134.param3 = asfloat(_3477.Load4(((-1) - inter.obj_index) * 64 + 48));
    float3 lcol = asfloat(_5134.type_and_param0.yzw);
    [branch]
    if ((_5134.type_and_param0.x & 128u) != 0u)
    {
        float3 env_col = _3458_g_params.env_col.xyz;
        uint _5161 = asuint(_3458_g_params.env_col.w);
        if (_5161 != 4294967295u)
        {
            atlas_texture_t _5168;
            _5168.size = _1008.Load(_5161 * 80 + 0);
            _5168.atlas = _1008.Load(_5161 * 80 + 4);
            [unroll]
            for (int _60ident = 0; _60ident < 4; _60ident++)
            {
                _5168.page[_60ident] = _1008.Load(_60ident * 4 + _5161 * 80 + 8);
            }
            [unroll]
            for (int _61ident = 0; _61ident < 14; _61ident++)
            {
                _5168.pos[_61ident] = _1008.Load(_61ident * 4 + _5161 * 80 + 24);
            }
            uint _9850[14] = { _5168.pos[0], _5168.pos[1], _5168.pos[2], _5168.pos[3], _5168.pos[4], _5168.pos[5], _5168.pos[6], _5168.pos[7], _5168.pos[8], _5168.pos[9], _5168.pos[10], _5168.pos[11], _5168.pos[12], _5168.pos[13] };
            uint _9821[4] = { _5168.page[0], _5168.page[1], _5168.page[2], _5168.page[3] };
            atlas_texture_t _9812 = { _5168.size, _5168.atlas, _9821, _9850 };
            float param = _3458_g_params.env_rotation;
            env_col *= SampleLatlong_RGBE(_9812, _5120, param, tex_rand);
        }
        lcol *= env_col;
    }
    uint _5228 = _5134.type_and_param0.x & 31u;
    if (_5228 == 0u)
    {
        float param_1 = ray.pdf;
        float param_2 = (inter.t * inter.t) / ((0.5f * _5134.param1.w) * dot(_5120, normalize(_5134.param1.xyz - (float3(ray.o[0], ray.o[1], ray.o[2]) + (_5120 * inter.t)))));
        lcol *= power_heuristic(param_1, param_2);
        bool _5295 = _5134.param3.x > 0.0f;
        bool _5301;
        if (_5295)
        {
            _5301 = _5134.param3.y > 0.0f;
        }
        else
        {
            _5301 = _5295;
        }
        [branch]
        if (_5301)
        {
            [flatten]
            if (_5134.param3.y > 0.0f)
            {
                lcol *= clamp((_5134.param3.x - acos(clamp(-dot(_5120, _5134.param2.xyz), 0.0f, 1.0f))) / _5134.param3.y, 0.0f, 1.0f);
            }
        }
    }
    else
    {
        if (_5228 == 4u)
        {
            float param_3 = ray.pdf;
            float param_4 = (inter.t * inter.t) / (_5134.param1.w * dot(_5120, normalize(cross(_5134.param2.xyz, _5134.param3.xyz))));
            lcol *= power_heuristic(param_3, param_4);
        }
        else
        {
            if (_5228 == 5u)
            {
                float param_5 = ray.pdf;
                float param_6 = (inter.t * inter.t) / (_5134.param1.w * dot(_5120, normalize(cross(_5134.param2.xyz, _5134.param3.xyz))));
                lcol *= power_heuristic(param_5, param_6);
            }
            else
            {
                if (_5228 == 3u)
                {
                    float param_7 = ray.pdf;
                    float param_8 = (inter.t * inter.t) / (_5134.param1.w * (1.0f - abs(dot(_5120, _5134.param3.xyz))));
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

bool exchange(inout bool old_value, bool new_value)
{
    bool _2233 = old_value;
    old_value = new_value;
    return _2233;
}

float peek_ior_stack(float stack[4], inout bool skip_first, float default_value)
{
    float _9255;
    do
    {
        bool _2317 = stack[3] > 0.0f;
        bool _2326;
        if (_2317)
        {
            bool param = skip_first;
            bool param_1 = false;
            bool _2323 = exchange(param, param_1);
            skip_first = param;
            _2326 = !_2323;
        }
        else
        {
            _2326 = _2317;
        }
        if (_2326)
        {
            _9255 = stack[3];
            break;
        }
        bool _2334 = stack[2] > 0.0f;
        bool _2343;
        if (_2334)
        {
            bool param_2 = skip_first;
            bool param_3 = false;
            bool _2340 = exchange(param_2, param_3);
            skip_first = param_2;
            _2343 = !_2340;
        }
        else
        {
            _2343 = _2334;
        }
        if (_2343)
        {
            _9255 = stack[2];
            break;
        }
        bool _2351 = stack[1] > 0.0f;
        bool _2360;
        if (_2351)
        {
            bool param_4 = skip_first;
            bool param_5 = false;
            bool _2357 = exchange(param_4, param_5);
            skip_first = param_4;
            _2360 = !_2357;
        }
        else
        {
            _2360 = _2351;
        }
        if (_2360)
        {
            _9255 = stack[1];
            break;
        }
        bool _2368 = stack[0] > 0.0f;
        bool _2377;
        if (_2368)
        {
            bool param_6 = skip_first;
            bool param_7 = false;
            bool _2374 = exchange(param_6, param_7);
            skip_first = param_6;
            _2377 = !_2374;
        }
        else
        {
            _2377 = _2368;
        }
        if (_2377)
        {
            _9255 = stack[0];
            break;
        }
        _9255 = default_value;
        break;
    } while(false);
    return _9255;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _613 = mad(col.z, 31.875f, 1.0f);
    float _623 = (col.x - 0.501960813999176025390625f) / _613;
    float _629 = (col.y - 0.501960813999176025390625f) / _613;
    return float3((col.w + _623) - _629, col.w + _629, (col.w - _623) - _629);
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
    atlas_texture_t _1138;
    _1138.size = _1008.Load(index * 80 + 0);
    _1138.atlas = _1008.Load(index * 80 + 4);
    [unroll]
    for (int _62ident = 0; _62ident < 4; _62ident++)
    {
        _1138.page[_62ident] = _1008.Load(_62ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _63ident = 0; _63ident < 14; _63ident++)
    {
        _1138.pos[_63ident] = _1008.Load(_63ident * 4 + index * 80 + 24);
    }
    uint _9426[4];
    _9426[0] = _1138.page[0];
    _9426[1] = _1138.page[1];
    _9426[2] = _1138.page[2];
    _9426[3] = _1138.page[3];
    uint _9462[14] = { _1138.pos[0], _1138.pos[1], _1138.pos[2], _1138.pos[3], _1138.pos[4], _1138.pos[5], _1138.pos[6], _1138.pos[7], _1138.pos[8], _1138.pos[9], _1138.pos[10], _1138.pos[11], _1138.pos[12], _1138.pos[13] };
    atlas_texture_t _9432 = { _1138.size, _1138.atlas, _9426, _9462 };
    uint _1211 = _1138.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_1211)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_1211)], float3(((TransformUV(uvs, _9432, lod) + rand) - 0.5f.xx) * 0.000118371215648949146270751953125f.xx, float((_9426[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
    bool _1226;
    if (maybe_YCoCg)
    {
        _1226 = _1138.atlas == 4u;
    }
    else
    {
        _1226 = maybe_YCoCg;
    }
    if (_1226)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _1245;
    if (maybe_SRGB)
    {
        _1245 = (_1138.size & 32768u) != 0u;
    }
    else
    {
        _1245 = maybe_SRGB;
    }
    if (_1245)
    {
        float3 param_1 = res.xyz;
        float3 _1251 = srgb_to_rgb(param_1);
        float4 _10614 = res;
        _10614.x = _1251.x;
        float4 _10616 = _10614;
        _10616.y = _1251.y;
        float4 _10618 = _10616;
        _10618.z = _1251.z;
        res = _10618;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod, float2 rand)
{
    return SampleBilinear(index, uvs, lod, rand, false, false);
}

float fresnel_dielectric_cos(float cosi, float eta)
{
    float _1358 = abs(cosi);
    float _1367 = mad(_1358, _1358, mad(eta, eta, -1.0f));
    float g = _1367;
    float result;
    if (_1367 > 0.0f)
    {
        float _1372 = g;
        float _1373 = sqrt(_1372);
        g = _1373;
        float _1377 = _1373 - _1358;
        float _1380 = _1373 + _1358;
        float _1381 = _1377 / _1380;
        float _1395 = mad(_1358, _1380, -1.0f) / mad(_1358, _1377, 1.0f);
        result = ((0.5f * _1381) * _1381) * mad(_1395, _1395, 1.0f);
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
    float3 _9260;
    do
    {
        float _1431 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1431)
        {
            _9260 = N;
            break;
        }
        float3 _1451 = normalize(N - (Ng * dot(N, Ng)));
        float _1455 = dot(I, _1451);
        float _1459 = dot(I, Ng);
        float _1471 = mad(_1455, _1455, _1459 * _1459);
        float param = (_1455 * _1455) * mad(-_1431, _1431, _1471);
        float _1481 = safe_sqrtf(param);
        float _1487 = mad(_1459, _1431, _1471);
        float _1490 = 0.5f / _1471;
        float _1495 = _1481 + _1487;
        float _1496 = _1490 * _1495;
        float _1502 = (-_1481) + _1487;
        float _1503 = _1490 * _1502;
        bool _1511 = (_1496 > 9.9999997473787516355514526367188e-06f) && (_1496 <= 1.000010013580322265625f);
        bool valid1 = _1511;
        bool _1517 = (_1503 > 9.9999997473787516355514526367188e-06f) && (_1503 <= 1.000010013580322265625f);
        bool valid2 = _1517;
        float2 N_new;
        if (_1511 && _1517)
        {
            float _10914 = (-0.5f) / _1471;
            float param_1 = mad(_10914, _1495, 1.0f);
            float _1527 = safe_sqrtf(param_1);
            float param_2 = _1496;
            float _1530 = safe_sqrtf(param_2);
            float2 _1531 = float2(_1527, _1530);
            float param_3 = mad(_10914, _1502, 1.0f);
            float _1536 = safe_sqrtf(param_3);
            float param_4 = _1503;
            float _1539 = safe_sqrtf(param_4);
            float2 _1540 = float2(_1536, _1539);
            float _10916 = -_1459;
            float _1556 = mad(2.0f * mad(_1527, _1455, _1530 * _1459), _1530, _10916);
            float _1572 = mad(2.0f * mad(_1536, _1455, _1539 * _1459), _1539, _10916);
            bool _1574 = _1556 >= 9.9999997473787516355514526367188e-06f;
            valid1 = _1574;
            bool _1576 = _1572 >= 9.9999997473787516355514526367188e-06f;
            valid2 = _1576;
            if (_1574 && _1576)
            {
                bool2 _1589 = (_1556 < _1572).xx;
                N_new = float2(_1589.x ? _1531.x : _1540.x, _1589.y ? _1531.y : _1540.y);
            }
            else
            {
                bool2 _1597 = (_1556 > _1572).xx;
                N_new = float2(_1597.x ? _1531.x : _1540.x, _1597.y ? _1531.y : _1540.y);
            }
        }
        else
        {
            if (!(valid1 || valid2))
            {
                _9260 = Ng;
                break;
            }
            float _1609 = valid1 ? _1496 : _1503;
            float param_5 = 1.0f - _1609;
            float param_6 = _1609;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _9260 = (_1451 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _9260;
}

float3 rotate_around_axis(float3 p, float3 axis, float angle)
{
    float _1703 = cos(angle);
    float _1706 = sin(angle);
    float _1710 = 1.0f - _1703;
    return float3(mad(mad(_1710 * axis.x, axis.z, axis.y * _1706), p.z, mad(mad(_1710 * axis.x, axis.x, _1703), p.x, mad(_1710 * axis.x, axis.y, -(axis.z * _1706)) * p.y)), mad(mad(_1710 * axis.y, axis.z, -(axis.x * _1706)), p.z, mad(mad(_1710 * axis.x, axis.y, axis.z * _1706), p.x, mad(_1710 * axis.y, axis.y, _1703) * p.y)), mad(mad(_1710 * axis.z, axis.z, _1703), p.z, mad(mad(_1710 * axis.x, axis.z, -(axis.y * _1706)), p.x, mad(_1710 * axis.y, axis.z, axis.x * _1706) * p.y)));
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
    int3 _1859 = int3(n * 128.0f);
    int _1867;
    if (p.x < 0.0f)
    {
        _1867 = -_1859.x;
    }
    else
    {
        _1867 = _1859.x;
    }
    int _1885;
    if (p.y < 0.0f)
    {
        _1885 = -_1859.y;
    }
    else
    {
        _1885 = _1859.y;
    }
    int _1903;
    if (p.z < 0.0f)
    {
        _1903 = -_1859.z;
    }
    else
    {
        _1903 = _1859.z;
    }
    float _1921;
    if (abs(p.x) < 0.03125f)
    {
        _1921 = mad(1.52587890625e-05f, n.x, p.x);
    }
    else
    {
        _1921 = asfloat(asint(p.x) + _1867);
    }
    float _1939;
    if (abs(p.y) < 0.03125f)
    {
        _1939 = mad(1.52587890625e-05f, n.y, p.y);
    }
    else
    {
        _1939 = asfloat(asint(p.y) + _1885);
    }
    float _1956;
    if (abs(p.z) < 0.03125f)
    {
        _1956 = mad(1.52587890625e-05f, n.z, p.z);
    }
    else
    {
        _1956 = asfloat(asint(p.z) + _1903);
    }
    return float3(_1921, _1939, _1956);
}

float3 MapToCone(float r1, float r2, float3 N, float radius)
{
    float3 _9285;
    do
    {
        float2 _3357 = (float2(r1, r2) * 2.0f) - 1.0f.xx;
        float _3359 = _3357.x;
        bool _3360 = _3359 == 0.0f;
        bool _3366;
        if (_3360)
        {
            _3366 = _3357.y == 0.0f;
        }
        else
        {
            _3366 = _3360;
        }
        if (_3366)
        {
            _9285 = N;
            break;
        }
        float _3375 = _3357.y;
        float r;
        float theta;
        if (abs(_3359) > abs(_3375))
        {
            r = _3359;
            theta = 0.785398185253143310546875f * (_3375 / _3359);
        }
        else
        {
            r = _3375;
            theta = 1.57079637050628662109375f * mad(-0.5f, _3359 / _3375, 1.0f);
        }
        float3 param;
        float3 param_1;
        create_tbn(N, param, param_1);
        _9285 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _9285;
}

float3 CanonicalToDir(float2 p, float y_rotation)
{
    float _687 = mad(2.0f, p.x, -1.0f);
    float _692 = mad(6.283185482025146484375f, p.y, y_rotation);
    float phi = _692;
    if (_692 < 0.0f)
    {
        phi += 6.283185482025146484375f;
    }
    if (phi > 6.283185482025146484375f)
    {
        phi -= 6.283185482025146484375f;
    }
    float _710 = sqrt(mad(-_687, _687, 1.0f));
    return float3(_710 * cos(phi), _687, (-_710) * sin(phi));
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
        float _885 = quad.x + quad.z;
        float partial = _885;
        float _892 = (_885 + quad.y) + quad.w;
        if (_892 <= 0.0f)
        {
            break;
        }
        float _901 = partial / _892;
        float boundary = _901;
        int index = 0;
        if (_sample < _901)
        {
            _sample /= boundary;
            boundary = quad.x / partial;
        }
        else
        {
            float _916 = partial;
            float _917 = _892 - _916;
            partial = _917;
            float2 _10601 = origin;
            _10601.x = origin.x + _step;
            origin = _10601;
            _sample = (_sample - boundary) / (1.0f - boundary);
            boundary = quad.y / _917;
            index |= 1;
        }
        if (_sample < boundary)
        {
            _sample /= boundary;
        }
        else
        {
            float2 _10604 = origin;
            _10604.y = origin.y + _step;
            origin = _10604;
            _sample = (_sample - boundary) / (1.0f - boundary);
            index |= 2;
        }
        factor *= ((4.0f * quad[index]) / _892);
        lod--;
        res *= 2;
        _step *= 0.5f;
    }
    float2 _974 = origin;
    float2 _975 = _974 + (float2(rx, ry) * (2.0f * _step));
    origin = _975;
    return float4(CanonicalToDir(_975, y_rotation), factor * 0.079577468335628509521484375f);
}

float3 world_from_tangent(float3 T, float3 B, float3 N, float3 V)
{
    return ((T * V.x) + (B * V.y)) + (N * V.z);
}

void SampleLightSource(float3 P, float3 T, float3 B, float3 N, int hi, float2 sample_off, inout light_sample_t ls)
{
    float _3451 = frac(asfloat(_3442.Load((hi + 3) * 4 + 0)) + sample_off.x);
    float _3462 = float(_3458_g_params.li_count);
    uint _3469 = min(uint(_3451 * _3462), uint(_3458_g_params.li_count - 1));
    light_t _3488;
    _3488.type_and_param0 = _3477.Load4(_3481.Load(_3469 * 4 + 0) * 64 + 0);
    _3488.param1 = asfloat(_3477.Load4(_3481.Load(_3469 * 4 + 0) * 64 + 16));
    _3488.param2 = asfloat(_3477.Load4(_3481.Load(_3469 * 4 + 0) * 64 + 32));
    _3488.param3 = asfloat(_3477.Load4(_3481.Load(_3469 * 4 + 0) * 64 + 48));
    ls.col = asfloat(_3488.type_and_param0.yzw);
    ls.col *= _3462;
    ls.cast_shadow = (_3488.type_and_param0.x & 32u) != 0u;
    ls.from_env = false;
    float2 _3535 = float2(frac(asfloat(_3442.Load((hi + 7) * 4 + 0)) + sample_off.x), frac(asfloat(_3442.Load((hi + 8) * 4 + 0)) + sample_off.y));
    uint _3540 = _3488.type_and_param0.x & 31u;
    [branch]
    if (_3540 == 0u)
    {
        float _3553 = frac(asfloat(_3442.Load((hi + 4) * 4 + 0)) + sample_off.x);
        float3 _3568 = P - _3488.param1.xyz;
        float3 _3575 = _3568 / length(_3568).xxx;
        float _3582 = sqrt(clamp(mad(-_3553, _3553, 1.0f), 0.0f, 1.0f));
        float _3585 = 6.283185482025146484375f * frac(asfloat(_3442.Load((hi + 5) * 4 + 0)) + sample_off.y);
        float3 sampled_dir = float3(_3582 * cos(_3585), _3582 * sin(_3585), _3553);
        float3 param;
        float3 param_1;
        create_tbn(_3575, param, param_1);
        float3 _10675 = sampled_dir;
        float3 _3618 = ((param * _10675.x) + (param_1 * _10675.y)) + (_3575 * _10675.z);
        sampled_dir = _3618;
        float3 _3627 = _3488.param1.xyz + (_3618 * _3488.param2.w);
        float3 _3634 = normalize(_3627 - _3488.param1.xyz);
        float3 param_2 = _3627;
        float3 param_3 = _3634;
        ls.lp = offset_ray(param_2, param_3);
        ls.L = _3627 - P;
        float3 _3647 = ls.L;
        float _3648 = length(_3647);
        ls.L /= _3648.xxx;
        ls.area = _3488.param1.w;
        float _3663 = abs(dot(ls.L, _3634));
        [flatten]
        if (_3663 > 0.0f)
        {
            ls.pdf = (_3648 * _3648) / ((0.5f * ls.area) * _3663);
        }
        [branch]
        if (_3488.param3.x > 0.0f)
        {
            float _3690 = -dot(ls.L, _3488.param2.xyz);
            if (_3690 > 0.0f)
            {
                ls.col *= clamp((_3488.param3.x - acos(clamp(_3690, 0.0f, 1.0f))) / _3488.param3.y, 0.0f, 1.0f);
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
        if (_3540 == 2u)
        {
            ls.L = _3488.param1.xyz;
            if (_3488.param1.w != 0.0f)
            {
                float param_4 = frac(asfloat(_3442.Load((hi + 4) * 4 + 0)) + sample_off.x);
                float param_5 = frac(asfloat(_3442.Load((hi + 5) * 4 + 0)) + sample_off.y);
                float3 param_6 = ls.L;
                float param_7 = tan(_3488.param1.w);
                ls.L = normalize(MapToCone(param_4, param_5, param_6, param_7));
            }
            ls.area = 0.0f;
            ls.lp = P + ls.L;
            ls.dist_mul = 3402823346297367662189621542912.0f;
            ls.pdf = 1.0f;
            if ((_3488.type_and_param0.x & 64u) == 0u)
            {
                ls.area = 0.0f;
            }
        }
        else
        {
            [branch]
            if (_3540 == 4u)
            {
                float3 _3827 = (_3488.param1.xyz + (_3488.param2.xyz * (frac(asfloat(_3442.Load((hi + 4) * 4 + 0)) + sample_off.x) - 0.5f))) + (_3488.param3.xyz * (frac(asfloat(_3442.Load((hi + 5) * 4 + 0)) + sample_off.y) - 0.5f));
                float3 _3832 = normalize(cross(_3488.param2.xyz, _3488.param3.xyz));
                float3 param_8 = _3827;
                float3 param_9 = _3832;
                ls.lp = offset_ray(param_8, param_9);
                ls.L = _3827 - P;
                float3 _3845 = ls.L;
                float _3846 = length(_3845);
                ls.L /= _3846.xxx;
                ls.area = _3488.param1.w;
                float _3861 = dot(-ls.L, _3832);
                if (_3861 > 0.0f)
                {
                    ls.pdf = (_3846 * _3846) / (ls.area * _3861);
                }
                if ((_3488.type_and_param0.x & 64u) == 0u)
                {
                    ls.area = 0.0f;
                }
                [branch]
                if ((_3488.type_and_param0.x & 128u) != 0u)
                {
                    float3 env_col = _3458_g_params.env_col.xyz;
                    uint _3898 = asuint(_3458_g_params.env_col.w);
                    if (_3898 != 4294967295u)
                    {
                        atlas_texture_t _3906;
                        _3906.size = _1008.Load(_3898 * 80 + 0);
                        _3906.atlas = _1008.Load(_3898 * 80 + 4);
                        [unroll]
                        for (int _64ident = 0; _64ident < 4; _64ident++)
                        {
                            _3906.page[_64ident] = _1008.Load(_64ident * 4 + _3898 * 80 + 8);
                        }
                        [unroll]
                        for (int _65ident = 0; _65ident < 14; _65ident++)
                        {
                            _3906.pos[_65ident] = _1008.Load(_65ident * 4 + _3898 * 80 + 24);
                        }
                        uint _9606[14] = { _3906.pos[0], _3906.pos[1], _3906.pos[2], _3906.pos[3], _3906.pos[4], _3906.pos[5], _3906.pos[6], _3906.pos[7], _3906.pos[8], _3906.pos[9], _3906.pos[10], _3906.pos[11], _3906.pos[12], _3906.pos[13] };
                        uint _9577[4] = { _3906.page[0], _3906.page[1], _3906.page[2], _3906.page[3] };
                        atlas_texture_t _9506 = { _3906.size, _3906.atlas, _9577, _9606 };
                        float param_10 = _3458_g_params.env_rotation;
                        env_col *= SampleLatlong_RGBE(_9506, ls.L, param_10, _3535);
                    }
                    ls.col *= env_col;
                    ls.from_env = true;
                }
            }
            else
            {
                [branch]
                if (_3540 == 5u)
                {
                    float2 _4010 = (float2(frac(asfloat(_3442.Load((hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_3442.Load((hi + 5) * 4 + 0)) + sample_off.y)) * 2.0f) - 1.0f.xx;
                    float2 offset = _4010;
                    bool _4013 = _4010.x != 0.0f;
                    bool _4019;
                    if (_4013)
                    {
                        _4019 = offset.y != 0.0f;
                    }
                    else
                    {
                        _4019 = _4013;
                    }
                    if (_4019)
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
                        float _4052 = 0.5f * r;
                        offset = float2(_4052 * cos(theta), _4052 * sin(theta));
                    }
                    float3 _4074 = (_3488.param1.xyz + (_3488.param2.xyz * offset.x)) + (_3488.param3.xyz * offset.y);
                    float3 _4079 = normalize(cross(_3488.param2.xyz, _3488.param3.xyz));
                    float3 param_11 = _4074;
                    float3 param_12 = _4079;
                    ls.lp = offset_ray(param_11, param_12);
                    ls.L = _4074 - P;
                    float3 _4092 = ls.L;
                    float _4093 = length(_4092);
                    ls.L /= _4093.xxx;
                    ls.area = _3488.param1.w;
                    float _4108 = dot(-ls.L, _4079);
                    [flatten]
                    if (_4108 > 0.0f)
                    {
                        ls.pdf = (_4093 * _4093) / (ls.area * _4108);
                    }
                    if ((_3488.type_and_param0.x & 64u) == 0u)
                    {
                        ls.area = 0.0f;
                    }
                    [branch]
                    if ((_3488.type_and_param0.x & 128u) != 0u)
                    {
                        float3 env_col_1 = _3458_g_params.env_col.xyz;
                        uint _4142 = asuint(_3458_g_params.env_col.w);
                        if (_4142 != 4294967295u)
                        {
                            atlas_texture_t _4149;
                            _4149.size = _1008.Load(_4142 * 80 + 0);
                            _4149.atlas = _1008.Load(_4142 * 80 + 4);
                            [unroll]
                            for (int _66ident = 0; _66ident < 4; _66ident++)
                            {
                                _4149.page[_66ident] = _1008.Load(_66ident * 4 + _4142 * 80 + 8);
                            }
                            [unroll]
                            for (int _67ident = 0; _67ident < 14; _67ident++)
                            {
                                _4149.pos[_67ident] = _1008.Load(_67ident * 4 + _4142 * 80 + 24);
                            }
                            uint _9644[14] = { _4149.pos[0], _4149.pos[1], _4149.pos[2], _4149.pos[3], _4149.pos[4], _4149.pos[5], _4149.pos[6], _4149.pos[7], _4149.pos[8], _4149.pos[9], _4149.pos[10], _4149.pos[11], _4149.pos[12], _4149.pos[13] };
                            uint _9615[4] = { _4149.page[0], _4149.page[1], _4149.page[2], _4149.page[3] };
                            atlas_texture_t _9515 = { _4149.size, _4149.atlas, _9615, _9644 };
                            float param_13 = _3458_g_params.env_rotation;
                            env_col_1 *= SampleLatlong_RGBE(_9515, ls.L, param_13, _3535);
                        }
                        ls.col *= env_col_1;
                        ls.from_env = true;
                    }
                }
                else
                {
                    [branch]
                    if (_3540 == 3u)
                    {
                        float3 _4250 = normalize(cross(P - _3488.param1.xyz, _3488.param3.xyz));
                        float _4257 = 3.1415927410125732421875f * frac(asfloat(_3442.Load((hi + 4) * 4 + 0)) + sample_off.x);
                        float3 _4282 = (_3488.param1.xyz + (((_4250 * cos(_4257)) + (cross(_4250, _3488.param3.xyz) * sin(_4257))) * _3488.param2.w)) + ((_3488.param3.xyz * (frac(asfloat(_3442.Load((hi + 5) * 4 + 0)) + sample_off.y) - 0.5f)) * _3488.param3.w);
                        ls.lp = _4282;
                        float3 _4288 = _4282 - P;
                        float _4291 = length(_4288);
                        ls.L = _4288 / _4291.xxx;
                        ls.area = _3488.param1.w;
                        float _4306 = 1.0f - abs(dot(ls.L, _3488.param3.xyz));
                        [flatten]
                        if (_4306 != 0.0f)
                        {
                            ls.pdf = (_4291 * _4291) / (ls.area * _4306);
                        }
                        if ((_3488.type_and_param0.x & 64u) == 0u)
                        {
                            ls.area = 0.0f;
                        }
                    }
                    else
                    {
                        [branch]
                        if (_3540 == 6u)
                        {
                            uint _4336 = asuint(_3488.param1.x);
                            transform_t _4350;
                            _4350.xform = asfloat(uint4x4(_4344.Load4(asuint(_3488.param1.y) * 128 + 0), _4344.Load4(asuint(_3488.param1.y) * 128 + 16), _4344.Load4(asuint(_3488.param1.y) * 128 + 32), _4344.Load4(asuint(_3488.param1.y) * 128 + 48)));
                            _4350.inv_xform = asfloat(uint4x4(_4344.Load4(asuint(_3488.param1.y) * 128 + 64), _4344.Load4(asuint(_3488.param1.y) * 128 + 80), _4344.Load4(asuint(_3488.param1.y) * 128 + 96), _4344.Load4(asuint(_3488.param1.y) * 128 + 112)));
                            uint _4375 = _4336 * 3u;
                            vertex_t _4381;
                            [unroll]
                            for (int _68ident = 0; _68ident < 3; _68ident++)
                            {
                                _4381.p[_68ident] = asfloat(_4369.Load(_68ident * 4 + _4373.Load(_4375 * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _69ident = 0; _69ident < 3; _69ident++)
                            {
                                _4381.n[_69ident] = asfloat(_4369.Load(_69ident * 4 + _4373.Load(_4375 * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _70ident = 0; _70ident < 3; _70ident++)
                            {
                                _4381.b[_70ident] = asfloat(_4369.Load(_70ident * 4 + _4373.Load(_4375 * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _71ident = 0; _71ident < 2; _71ident++)
                            {
                                [unroll]
                                for (int _72ident = 0; _72ident < 2; _72ident++)
                                {
                                    _4381.t[_71ident][_72ident] = asfloat(_4369.Load(_72ident * 4 + _71ident * 8 + _4373.Load(_4375 * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4430;
                            [unroll]
                            for (int _73ident = 0; _73ident < 3; _73ident++)
                            {
                                _4430.p[_73ident] = asfloat(_4369.Load(_73ident * 4 + _4373.Load((_4375 + 1u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _74ident = 0; _74ident < 3; _74ident++)
                            {
                                _4430.n[_74ident] = asfloat(_4369.Load(_74ident * 4 + _4373.Load((_4375 + 1u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _75ident = 0; _75ident < 3; _75ident++)
                            {
                                _4430.b[_75ident] = asfloat(_4369.Load(_75ident * 4 + _4373.Load((_4375 + 1u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _76ident = 0; _76ident < 2; _76ident++)
                            {
                                [unroll]
                                for (int _77ident = 0; _77ident < 2; _77ident++)
                                {
                                    _4430.t[_76ident][_77ident] = asfloat(_4369.Load(_77ident * 4 + _76ident * 8 + _4373.Load((_4375 + 1u) * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4476;
                            [unroll]
                            for (int _78ident = 0; _78ident < 3; _78ident++)
                            {
                                _4476.p[_78ident] = asfloat(_4369.Load(_78ident * 4 + _4373.Load((_4375 + 2u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _79ident = 0; _79ident < 3; _79ident++)
                            {
                                _4476.n[_79ident] = asfloat(_4369.Load(_79ident * 4 + _4373.Load((_4375 + 2u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _80ident = 0; _80ident < 3; _80ident++)
                            {
                                _4476.b[_80ident] = asfloat(_4369.Load(_80ident * 4 + _4373.Load((_4375 + 2u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _81ident = 0; _81ident < 2; _81ident++)
                            {
                                [unroll]
                                for (int _82ident = 0; _82ident < 2; _82ident++)
                                {
                                    _4476.t[_81ident][_82ident] = asfloat(_4369.Load(_82ident * 4 + _81ident * 8 + _4373.Load((_4375 + 2u) * 4 + 0) * 52 + 36));
                                }
                            }
                            float3 _4522 = float3(_4381.p[0], _4381.p[1], _4381.p[2]);
                            float3 _4530 = float3(_4430.p[0], _4430.p[1], _4430.p[2]);
                            float3 _4538 = float3(_4476.p[0], _4476.p[1], _4476.p[2]);
                            float _4566 = sqrt(frac(asfloat(_3442.Load((hi + 4) * 4 + 0)) + sample_off.x));
                            float _4575 = frac(asfloat(_3442.Load((hi + 5) * 4 + 0)) + sample_off.y);
                            float _4579 = 1.0f - _4566;
                            float _4584 = 1.0f - _4575;
                            float3 _4615 = mul(float4((_4522 * _4579) + (((_4530 * _4584) + (_4538 * _4575)) * _4566), 1.0f), _4350.xform).xyz;
                            float3 _4631 = mul(float4(cross(_4530 - _4522, _4538 - _4522), 0.0f), _4350.xform).xyz;
                            ls.area = 0.5f * length(_4631);
                            float3 _4637 = normalize(_4631);
                            ls.L = _4615 - P;
                            float3 _4644 = ls.L;
                            float _4645 = length(_4644);
                            ls.L /= _4645.xxx;
                            float _4656 = dot(ls.L, _4637);
                            float cos_theta = _4656;
                            float3 _4659;
                            if (_4656 >= 0.0f)
                            {
                                _4659 = -_4637;
                            }
                            else
                            {
                                _4659 = _4637;
                            }
                            float3 param_14 = _4615;
                            float3 param_15 = _4659;
                            ls.lp = offset_ray(param_14, param_15);
                            float _4672 = cos_theta;
                            float _4673 = abs(_4672);
                            cos_theta = _4673;
                            [flatten]
                            if (_4673 > 0.0f)
                            {
                                ls.pdf = (_4645 * _4645) / (ls.area * cos_theta);
                            }
                            material_t _4710;
                            [unroll]
                            for (int _83ident = 0; _83ident < 5; _83ident++)
                            {
                                _4710.textures[_83ident] = _4697.Load(_83ident * 4 + ((_4701.Load(_4336 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
                            }
                            [unroll]
                            for (int _84ident = 0; _84ident < 3; _84ident++)
                            {
                                _4710.base_color[_84ident] = asfloat(_4697.Load(_84ident * 4 + ((_4701.Load(_4336 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
                            }
                            _4710.flags = _4697.Load(((_4701.Load(_4336 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
                            _4710.type = _4697.Load(((_4701.Load(_4336 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
                            _4710.tangent_rotation_or_strength = asfloat(_4697.Load(((_4701.Load(_4336 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
                            _4710.roughness_and_anisotropic = _4697.Load(((_4701.Load(_4336 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
                            _4710.ior = asfloat(_4697.Load(((_4701.Load(_4336 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
                            _4710.sheen_and_sheen_tint = _4697.Load(((_4701.Load(_4336 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
                            _4710.tint_and_metallic = _4697.Load(((_4701.Load(_4336 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
                            _4710.transmission_and_transmission_roughness = _4697.Load(((_4701.Load(_4336 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
                            _4710.specular_and_specular_tint = _4697.Load(((_4701.Load(_4336 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
                            _4710.clearcoat_and_clearcoat_roughness = _4697.Load(((_4701.Load(_4336 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
                            _4710.normal_map_strength_unorm = _4697.Load(((_4701.Load(_4336 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
                            if (_4710.textures[1] != 4294967295u)
                            {
                                ls.col *= SampleBilinear(_4710.textures[1], (float2(_4381.t[0][0], _4381.t[0][1]) * _4579) + (((float2(_4430.t[0][0], _4430.t[0][1]) * _4584) + (float2(_4476.t[0][0], _4476.t[0][1]) * _4575)) * _4566), 0, _3535).xyz;
                            }
                        }
                        else
                        {
                            [branch]
                            if (_3540 == 7u)
                            {
                                float _4791 = frac(asfloat(_3442.Load((hi + 4) * 4 + 0)) + sample_off.x);
                                float _4800 = frac(asfloat(_3442.Load((hi + 5) * 4 + 0)) + sample_off.y);
                                float4 dir_and_pdf;
                                if (_3458_g_params.env_qtree_levels > 0)
                                {
                                    dir_and_pdf = Sample_EnvQTree(_3458_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3458_g_params.env_qtree_levels, mad(_3451, _3462, -float(_3469)), _4791, _4800);
                                }
                                else
                                {
                                    float _4819 = 6.283185482025146484375f * _4800;
                                    float _4831 = sqrt(mad(-_4791, _4791, 1.0f));
                                    float3 param_16 = T;
                                    float3 param_17 = B;
                                    float3 param_18 = N;
                                    float3 param_19 = float3(_4831 * cos(_4819), _4831 * sin(_4819), _4791);
                                    dir_and_pdf = float4(world_from_tangent(param_16, param_17, param_18, param_19), 0.15915493667125701904296875f);
                                }
                                ls.L = dir_and_pdf.xyz;
                                ls.col *= _3458_g_params.env_col.xyz;
                                uint _4870 = asuint(_3458_g_params.env_col.w);
                                if (_4870 != 4294967295u)
                                {
                                    atlas_texture_t _4877;
                                    _4877.size = _1008.Load(_4870 * 80 + 0);
                                    _4877.atlas = _1008.Load(_4870 * 80 + 4);
                                    [unroll]
                                    for (int _85ident = 0; _85ident < 4; _85ident++)
                                    {
                                        _4877.page[_85ident] = _1008.Load(_85ident * 4 + _4870 * 80 + 8);
                                    }
                                    [unroll]
                                    for (int _86ident = 0; _86ident < 14; _86ident++)
                                    {
                                        _4877.pos[_86ident] = _1008.Load(_86ident * 4 + _4870 * 80 + 24);
                                    }
                                    uint _9729[14] = { _4877.pos[0], _4877.pos[1], _4877.pos[2], _4877.pos[3], _4877.pos[4], _4877.pos[5], _4877.pos[6], _4877.pos[7], _4877.pos[8], _4877.pos[9], _4877.pos[10], _4877.pos[11], _4877.pos[12], _4877.pos[13] };
                                    uint _9700[4] = { _4877.page[0], _4877.page[1], _4877.page[2], _4877.page[3] };
                                    atlas_texture_t _9568 = { _4877.size, _4877.atlas, _9700, _9729 };
                                    float param_20 = _3458_g_params.env_rotation;
                                    ls.col *= SampleLatlong_RGBE(_9568, ls.L, param_20, _3535);
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
    atlas_texture_t _1011;
    _1011.size = _1008.Load(index * 80 + 0);
    _1011.atlas = _1008.Load(index * 80 + 4);
    [unroll]
    for (int _87ident = 0; _87ident < 4; _87ident++)
    {
        _1011.page[_87ident] = _1008.Load(_87ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _88ident = 0; _88ident < 14; _88ident++)
    {
        _1011.pos[_88ident] = _1008.Load(_88ident * 4 + index * 80 + 24);
    }
    return int2(int(_1011.size & 16383u), int((_1011.size >> uint(16)) & 16383u));
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
    float _2437 = 1.0f / mad(0.904129683971405029296875f, roughness, 3.1415927410125732421875f);
    float _2449 = max(dot(N, L), 0.0f);
    float _2454 = max(dot(N, V), 0.0f);
    float _2462 = mad(-_2449, _2454, dot(L, V));
    float t = _2462;
    if (_2462 > 0.0f)
    {
        t /= (max(_2449, _2454) + 1.1754943508222875079687365372222e-38f);
    }
    return float4(base_color * (_2449 * mad(roughness * _2437, t, _2437)), 0.15915493667125701904296875f);
}

float3 Evaluate_DiffuseNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _9265;
    do
    {
        float3 param = -float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param_1 = surf.N;
        float3 param_2 = ls.L;
        float param_3 = roughness;
        float3 param_4 = base_color;
        float4 _5498 = Evaluate_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_5 = ls.pdf;
            float param_6 = _5498.w;
            mis_weight = power_heuristic(param_5, param_6);
        }
        float3 _5521 = (ls.col * _5498.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9265 = _5521;
            break;
        }
        float3 param_7 = surf.P;
        float3 param_8 = surf.plane_N;
        float3 _5533 = offset_ray(param_7, param_8);
        sh_r.o[0] = _5533.x;
        sh_r.o[1] = _5533.y;
        sh_r.o[2] = _5533.z;
        sh_r.c[0] = ray.c[0] * _5521.x;
        sh_r.c[1] = ray.c[1] * _5521.y;
        sh_r.c[2] = ray.c[2] * _5521.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9265 = 0.0f.xxx;
        break;
    } while(false);
    return _9265;
}

float4 Sample_OrenDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float rand_u, float rand_v, inout float3 out_V)
{
    float _2496 = 6.283185482025146484375f * rand_v;
    float _2508 = sqrt(mad(-rand_u, rand_u, 1.0f));
    float3 param = T;
    float3 param_1 = B;
    float3 param_2 = N;
    float3 param_3 = float3(_2508 * cos(_2496), _2508 * sin(_2496), rand_u);
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
    float4 _5784 = Sample_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8);
    new_ray.depth = ray.depth + 1;
    float3 param_9 = surf.P;
    float3 param_10 = surf.plane_N;
    float3 _5794 = offset_ray(param_9, param_10);
    new_ray.o[0] = _5794.x;
    new_ray.o[1] = _5794.y;
    new_ray.o[2] = _5794.z;
    new_ray.d[0] = param_8.x;
    new_ray.d[1] = param_8.y;
    new_ray.d[2] = param_8.z;
    new_ray.c[0] = ((ray.c[0] * _5784.x) * mix_weight) / _5784.w;
    new_ray.c[1] = ((ray.c[1] * _5784.y) * mix_weight) / _5784.w;
    new_ray.c[2] = ((ray.c[2] * _5784.z) * mix_weight) / _5784.w;
    new_ray.pdf = _5784.w;
}

float3 tangent_from_world(float3 T, float3 B, float3 N, float3 V)
{
    return float3(dot(V, T), dot(V, B), dot(V, N));
}

float D_GGX(float3 H, float alpha_x, float alpha_y)
{
    float _9318;
    do
    {
        if (H.z == 0.0f)
        {
            _9318 = 0.0f;
            break;
        }
        float _2163 = (-H.x) / (H.z * alpha_x);
        float _2169 = (-H.y) / (H.z * alpha_y);
        float _2178 = mad(_2169, _2169, mad(_2163, _2163, 1.0f));
        _9318 = 1.0f / (((((_2178 * _2178) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _9318;
}

float G1(float3 Ve, inout float alpha_x, inout float alpha_y)
{
    alpha_x *= alpha_x;
    alpha_y *= alpha_y;
    return 1.0f / mad((-1.0f) + sqrt(1.0f + (mad(alpha_x * Ve.x, Ve.x, (alpha_y * Ve.y) * Ve.y) / (Ve.z * Ve.z))), 0.5f, 1.0f);
}

float4 Evaluate_GGXSpecular_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float alpha_x, float alpha_y, float spec_ior, float spec_F0, float3 spec_col)
{
    float _2678 = D_GGX(sampled_normal_ts, alpha_x, alpha_y);
    float3 param = view_dir_ts;
    float param_1 = alpha_x;
    float param_2 = alpha_y;
    float _2686 = G1(param, param_1, param_2);
    float3 param_3 = reflected_dir_ts;
    float param_4 = alpha_x;
    float param_5 = alpha_y;
    float _2693 = G1(param_3, param_4, param_5);
    float param_6 = dot(view_dir_ts, sampled_normal_ts);
    float param_7 = spec_ior;
    float3 F = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param_6, param_7) - spec_F0) / (1.0f - spec_F0)).xxx);
    float _2721 = 4.0f * abs(view_dir_ts.z * reflected_dir_ts.z);
    float _2724;
    if (_2721 != 0.0f)
    {
        _2724 = (_2678 * (_2686 * _2693)) / _2721;
    }
    else
    {
        _2724 = 0.0f;
    }
    F *= _2724;
    float3 param_8 = view_dir_ts;
    float param_9 = alpha_x;
    float param_10 = alpha_y;
    float _2744 = G1(param_8, param_9, param_10);
    float pdf = ((_2678 * _2744) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2759 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2759 != 0.0f)
    {
        pdf /= _2759;
    }
    float3 _2770 = F;
    float3 _2771 = _2770 * max(reflected_dir_ts.z, 0.0f);
    F = _2771;
    return float4(_2771, pdf);
}

float3 Evaluate_GlossyNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _9270;
    do
    {
        float3 _5569 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5569;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - _5569);
        float _5607 = roughness * roughness;
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = _5607;
        float param_16 = _5607;
        float param_17 = spec_ior;
        float param_18 = spec_F0;
        float3 param_19 = base_color;
        float4 _5622 = Evaluate_GGXSpecular_BSDF(param_12, param_13, param_14, param_15, param_16, param_17, param_18, param_19);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_20 = ls.pdf;
            float param_21 = _5622.w;
            mis_weight = power_heuristic(param_20, param_21);
        }
        float3 _5645 = (ls.col * _5622.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9270 = _5645;
            break;
        }
        float3 param_22 = surf.P;
        float3 param_23 = surf.plane_N;
        float3 _5657 = offset_ray(param_22, param_23);
        sh_r.o[0] = _5657.x;
        sh_r.o[1] = _5657.y;
        sh_r.o[2] = _5657.z;
        sh_r.c[0] = ray.c[0] * _5645.x;
        sh_r.c[1] = ray.c[1] * _5645.y;
        sh_r.c[2] = ray.c[2] * _5645.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9270 = 0.0f.xxx;
        break;
    } while(false);
    return _9270;
}

float3 SampleGGX_VNDF(float3 Ve, float alpha_x, float alpha_y, float U1, float U2)
{
    float3 _1981 = normalize(float3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
    float _1984 = _1981.x;
    float _1989 = _1981.y;
    float _1993 = mad(_1984, _1984, _1989 * _1989);
    float3 _1997;
    if (_1993 > 0.0f)
    {
        _1997 = float3(-_1989, _1984, 0.0f) / sqrt(_1993).xxx;
    }
    else
    {
        _1997 = float3(1.0f, 0.0f, 0.0f);
    }
    float _2019 = sqrt(U1);
    float _2022 = 6.283185482025146484375f * U2;
    float _2027 = _2019 * cos(_2022);
    float _2036 = 1.0f + _1981.z;
    float _2043 = mad(-_2027, _2027, 1.0f);
    float _2049 = mad(mad(-0.5f, _2036, 1.0f), sqrt(_2043), (0.5f * _2036) * (_2019 * sin(_2022)));
    float3 _2070 = ((_1997 * _2027) + (cross(_1981, _1997) * _2049)) + (_1981 * sqrt(max(0.0f, mad(-_2049, _2049, _2043))));
    return normalize(float3(alpha_x * _2070.x, alpha_y * _2070.y, max(0.0f, _2070.z)));
}

float4 Sample_GGXSpecular_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float anisotropic, float spec_ior, float spec_F0, float3 spec_col, float rand_u, float rand_v, inout float3 out_V)
{
    float4 _9290;
    do
    {
        float _2781 = roughness * roughness;
        float _2785 = sqrt(mad(-0.89999997615814208984375f, anisotropic, 1.0f));
        float _2789 = _2781 / _2785;
        float _2793 = _2781 * _2785;
        [branch]
        if ((_2789 * _2793) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2804 = reflect(I, N);
            float param = dot(_2804, N);
            float param_1 = spec_ior;
            float3 _2818 = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param, param_1) - spec_F0) / (1.0f - spec_F0)).xxx);
            out_V = _2804;
            _9290 = float4(_2818.x * 1000000.0f, _2818.y * 1000000.0f, _2818.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2843 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = _2789;
        float param_7 = _2793;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2852 = SampleGGX_VNDF(_2843, param_6, param_7, param_8, param_9);
        float3 _2863 = normalize(reflect(-_2843, _2852));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2863;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2843;
        float3 param_15 = _2852;
        float3 param_16 = _2863;
        float param_17 = _2789;
        float param_18 = _2793;
        float param_19 = spec_ior;
        float param_20 = spec_F0;
        float3 param_21 = spec_col;
        _9290 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _9290;
}

void Sample_GlossyNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 param_4;
    float4 _5704 = Sample_GGXSpecular_BSDF(param, param_1, param_2, param_3, roughness, 0.0f, spec_ior, spec_F0, base_color, rand_u, rand_v, param_4);
    new_ray.depth = ray.depth + 256;
    float3 param_5 = surf.P;
    float3 param_6 = surf.plane_N;
    float3 _5715 = offset_ray(param_5, param_6);
    new_ray.o[0] = _5715.x;
    new_ray.o[1] = _5715.y;
    new_ray.o[2] = _5715.z;
    new_ray.d[0] = param_4.x;
    new_ray.d[1] = param_4.y;
    new_ray.d[2] = param_4.z;
    new_ray.c[0] = ((ray.c[0] * _5704.x) * mix_weight) / _5704.w;
    new_ray.c[1] = ((ray.c[1] * _5704.y) * mix_weight) / _5704.w;
    new_ray.c[2] = ((ray.c[2] * _5704.z) * mix_weight) / _5704.w;
    new_ray.pdf = _5704.w;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _9295;
    do
    {
        bool _3085 = refr_dir_ts.z >= 0.0f;
        bool _3092;
        if (!_3085)
        {
            _3092 = view_dir_ts.z <= 0.0f;
        }
        else
        {
            _3092 = _3085;
        }
        if (_3092)
        {
            _9295 = 0.0f.xxxx;
            break;
        }
        float _3101 = D_GGX(sampled_normal_ts, roughness2, roughness2);
        float3 param = refr_dir_ts;
        float param_1 = roughness2;
        float param_2 = roughness2;
        float _3109 = G1(param, param_1, param_2);
        float3 param_3 = view_dir_ts;
        float param_4 = roughness2;
        float param_5 = roughness2;
        float _3117 = G1(param_3, param_4, param_5);
        float _3127 = mad(dot(view_dir_ts, sampled_normal_ts), eta, dot(refr_dir_ts, sampled_normal_ts));
        float _3137 = clamp(-dot(refr_dir_ts, sampled_normal_ts), 0.0f, 1.0f) / (_3127 * _3127);
        _9295 = float4(refr_col * (((((_3101 * _3117) * _3109) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3137) / view_dir_ts.z), (((_3101 * _3109) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3137) / view_dir_ts.z);
        break;
    } while(false);
    return _9295;
}

float3 Evaluate_RefractiveNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness2, float eta, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _9275;
    do
    {
        float3 _5847 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5847;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - (_5847 * eta));
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = roughness2;
        float param_16 = eta;
        float3 param_17 = base_color;
        float4 _5895 = Evaluate_GGXRefraction_BSDF(param_12, param_13, param_14, param_15, param_16, param_17);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_18 = ls.pdf;
            float param_19 = _5895.w;
            mis_weight = power_heuristic(param_18, param_19);
        }
        float3 _5918 = (ls.col * _5895.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9275 = _5918;
            break;
        }
        float3 param_20 = surf.P;
        float3 param_21 = -surf.plane_N;
        float3 _5931 = offset_ray(param_20, param_21);
        sh_r.o[0] = _5931.x;
        sh_r.o[1] = _5931.y;
        sh_r.o[2] = _5931.z;
        sh_r.c[0] = ray.c[0] * _5918.x;
        sh_r.c[1] = ray.c[1] * _5918.y;
        sh_r.c[2] = ray.c[2] * _5918.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9275 = 0.0f.xxx;
        break;
    } while(false);
    return _9275;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _9300;
    do
    {
        float _3181 = roughness * roughness;
        [branch]
        if ((_3181 * _3181) < 1.0000000116860974230803549289703e-07f)
        {
            float _3191 = dot(I, N);
            float _3192 = -_3191;
            float _3202 = mad(-(eta * eta), mad(_3191, _3192, 1.0f), 1.0f);
            if (_3202 < 0.0f)
            {
                _9300 = 0.0f.xxxx;
                break;
            }
            float _3214 = mad(eta, _3192, -sqrt(_3202));
            out_V = float4(normalize((I * eta) + (N * _3214)), _3214);
            _9300 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param = T;
        float3 param_1 = B;
        float3 param_2 = N;
        float3 param_3 = -I;
        float3 _3254 = normalize(tangent_from_world(param, param_1, param_2, param_3));
        float param_4 = _3181;
        float param_5 = _3181;
        float param_6 = rand_u;
        float param_7 = rand_v;
        float3 _3265 = SampleGGX_VNDF(_3254, param_4, param_5, param_6, param_7);
        float _3269 = dot(_3254, _3265);
        float _3279 = mad(-(eta * eta), mad(-_3269, _3269, 1.0f), 1.0f);
        if (_3279 < 0.0f)
        {
            _9300 = 0.0f.xxxx;
            break;
        }
        float _3291 = mad(eta, _3269, -sqrt(_3279));
        float3 _3301 = normalize((_3254 * (-eta)) + (_3265 * _3291));
        float3 param_8 = _3254;
        float3 param_9 = _3265;
        float3 param_10 = _3301;
        float param_11 = _3181;
        float param_12 = eta;
        float3 param_13 = refr_col;
        float3 param_14 = T;
        float3 param_15 = B;
        float3 param_16 = N;
        float3 param_17 = _3301;
        out_V = float4(world_from_tangent(param_14, param_15, param_16, param_17), _3291);
        _9300 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _9300;
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
    float _2227 = old_value;
    old_value = new_value;
    return _2227;
}

float pop_ior_stack(inout float stack[4], float default_value)
{
    float _9308;
    do
    {
        if (stack[3] > 0.0f)
        {
            float param = stack[3];
            float param_1 = -1.0f;
            float _2269 = exchange(param, param_1);
            stack[3] = param;
            _9308 = _2269;
            break;
        }
        if (stack[2] > 0.0f)
        {
            float param_2 = stack[2];
            float param_3 = -1.0f;
            float _2282 = exchange(param_2, param_3);
            stack[2] = param_2;
            _9308 = _2282;
            break;
        }
        if (stack[1] > 0.0f)
        {
            float param_4 = stack[1];
            float param_5 = -1.0f;
            float _2295 = exchange(param_4, param_5);
            stack[1] = param_4;
            _9308 = _2295;
            break;
        }
        if (stack[0] > 0.0f)
        {
            float param_6 = stack[0];
            float param_7 = -1.0f;
            float _2308 = exchange(param_6, param_7);
            stack[0] = param_6;
            _9308 = _2308;
            break;
        }
        _9308 = default_value;
        break;
    } while(false);
    return _9308;
}

void Sample_RefractiveNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, bool is_backfacing, float int_ior, float ext_ior, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float _5968;
    if (is_backfacing)
    {
        _5968 = int_ior / ext_ior;
    }
    else
    {
        _5968 = ext_ior / int_ior;
    }
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float param_4 = roughness;
    float param_5 = _5968;
    float3 param_6 = base_color;
    float param_7 = rand_u;
    float param_8 = rand_v;
    float4 param_9;
    float4 _5992 = Sample_GGXRefraction_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9);
    new_ray.depth = ray.depth + 65536;
    new_ray.c[0] = ((ray.c[0] * _5992.x) * mix_weight) / _5992.w;
    new_ray.c[1] = ((ray.c[1] * _5992.y) * mix_weight) / _5992.w;
    new_ray.c[2] = ((ray.c[2] * _5992.z) * mix_weight) / _5992.w;
    new_ray.pdf = _5992.w;
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
        float _6048 = pop_ior_stack(param_11, param_12);
        new_ray.ior = param_11;
    }
    float3 param_13 = surf.P;
    float3 param_14 = -surf.plane_N;
    float3 _6057 = offset_ray(param_13, param_14);
    new_ray.o[0] = _6057.x;
    new_ray.o[1] = _6057.y;
    new_ray.o[2] = _6057.z;
    new_ray.d[0] = param_9.x;
    new_ray.d[1] = param_9.y;
    new_ray.d[2] = param_9.z;
}

lobe_weights_t get_lobe_weights(float base_color_lum, float spec_color_lum, float specular, float metallic, float transmission, float clearcoat)
{
    float _1634 = 1.0f - metallic;
    float _9463 = (base_color_lum * _1634) * (1.0f - transmission);
    float _1641 = transmission * _1634;
    float _1645;
    if ((specular != 0.0f) || (metallic != 0.0f))
    {
        _1645 = spec_color_lum * mad(-transmission, _1634, 1.0f);
    }
    else
    {
        _1645 = 0.0f;
    }
    float _9464 = _1645;
    float _1655 = 0.25f * clearcoat;
    float _9465 = _1655 * _1634;
    float _9466 = _1641 * base_color_lum;
    float _1664 = _9463;
    float _1673 = mad(_1641, base_color_lum, mad(_1655, _1634, _1664 + _1645));
    if (_1673 != 0.0f)
    {
        _9463 /= _1673;
        _9464 /= _1673;
        _9465 /= _1673;
        _9466 /= _1673;
    }
    lobe_weights_t _9471 = { _9463, _9464, _9465, _9466 };
    return _9471;
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
    float _9323;
    do
    {
        float _2389 = dot(N, L);
        if (_2389 <= 0.0f)
        {
            _9323 = 0.0f;
            break;
        }
        float param = _2389;
        float param_1 = dot(N, V);
        float _2410 = dot(L, H);
        float _2418 = mad((2.0f * _2410) * _2410, roughness, 0.5f);
        _9323 = lerp(1.0f, _2418, schlick_weight(param)) * lerp(1.0f, _2418, schlick_weight(param_1));
        break;
    } while(false);
    return _9323;
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
    float3 _2559 = normalize(L + V);
    float3 H = _2559;
    if (dot(V, _2559) < 0.0f)
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
    float3 _2594 = diff_col;
    float3 _2595 = _2594 + (sheen_color * (3.1415927410125732421875f * schlick_weight(param_5)));
    diff_col = _2595;
    return float4(_2595, pdf);
}

float D_GTR1(float NDotH, float a)
{
    float _9328;
    do
    {
        if (a >= 1.0f)
        {
            _9328 = 0.3183098733425140380859375f;
            break;
        }
        float _2137 = mad(a, a, -1.0f);
        _9328 = _2137 / ((3.1415927410125732421875f * log(a * a)) * mad(_2137 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _9328;
}

float4 Evaluate_PrincipledClearcoat_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float clearcoat_roughness2, float clearcoat_ior, float clearcoat_F0)
{
    float param = sampled_normal_ts.z;
    float param_1 = clearcoat_roughness2;
    float _2895 = D_GTR1(param, param_1);
    float3 param_2 = view_dir_ts;
    float param_3 = 0.0625f;
    float param_4 = 0.0625f;
    float _2902 = G1(param_2, param_3, param_4);
    float3 param_5 = reflected_dir_ts;
    float param_6 = 0.0625f;
    float param_7 = 0.0625f;
    float _2907 = G1(param_5, param_6, param_7);
    float param_8 = dot(reflected_dir_ts, sampled_normal_ts);
    float param_9 = clearcoat_ior;
    float F = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param_8, param_9) - clearcoat_F0) / (1.0f - clearcoat_F0));
    float _2934 = (4.0f * abs(view_dir_ts.z)) * abs(reflected_dir_ts.z);
    float _2937;
    if (_2934 != 0.0f)
    {
        _2937 = (_2895 * (_2902 * _2907)) / _2934;
    }
    else
    {
        _2937 = 0.0f;
    }
    F *= _2937;
    float3 param_10 = view_dir_ts;
    float param_11 = 0.0625f;
    float param_12 = 0.0625f;
    float _2955 = G1(param_10, param_11, param_12);
    float pdf = ((_2895 * _2955) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2970 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2970 != 0.0f)
    {
        pdf /= _2970;
    }
    float _2981 = F;
    float _2982 = _2981 * clamp(reflected_dir_ts.z, 0.0f, 1.0f);
    F = _2982;
    return float4(_2982, _2982, _2982, pdf);
}

float3 Evaluate_PrincipledNode(light_sample_t ls, ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float N_dot_L, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _9280;
    do
    {
        float3 _6080 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 lcol = 0.0f.xxx;
        float bsdf_pdf = 0.0f;
        bool _6085 = N_dot_L > 0.0f;
        [branch]
        if ((lobe_weights.diffuse > 1.0000000116860974230803549289703e-07f) && _6085)
        {
            float3 param = -_6080;
            float3 param_1 = surf.N;
            float3 param_2 = ls.L;
            float param_3 = diff.roughness;
            float3 param_4 = diff.base_color;
            float3 param_5 = diff.sheen_color;
            bool param_6 = false;
            float4 _6104 = Evaluate_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6);
            bsdf_pdf = mad(lobe_weights.diffuse, _6104.w, bsdf_pdf);
            lcol += (((ls.col * N_dot_L) * (_6104 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * ls.pdf).xxx);
        }
        float3 H;
        [flatten]
        if (_6085)
        {
            H = normalize(ls.L - _6080);
        }
        else
        {
            H = normalize(ls.L - (_6080 * trans.eta));
        }
        float _6143 = spec.roughness * spec.roughness;
        float _6148 = sqrt(mad(-0.89999997615814208984375f, spec.anisotropy, 1.0f));
        float _6152 = _6143 / _6148;
        float _6156 = _6143 * _6148;
        float3 param_7 = surf.T;
        float3 param_8 = surf.B;
        float3 param_9 = surf.N;
        float3 param_10 = -_6080;
        float3 _6167 = tangent_from_world(param_7, param_8, param_9, param_10);
        float3 param_11 = surf.T;
        float3 param_12 = surf.B;
        float3 param_13 = surf.N;
        float3 param_14 = ls.L;
        float3 _6177 = tangent_from_world(param_11, param_12, param_13, param_14);
        float3 param_15 = surf.T;
        float3 param_16 = surf.B;
        float3 param_17 = surf.N;
        float3 param_18 = H;
        float3 _6187 = tangent_from_world(param_15, param_16, param_17, param_18);
        bool _6189 = lobe_weights.specular > 0.0f;
        bool _6196;
        if (_6189)
        {
            _6196 = (_6152 * _6156) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _6196 = _6189;
        }
        [branch]
        if (_6196 && _6085)
        {
            float3 param_19 = _6167;
            float3 param_20 = _6187;
            float3 param_21 = _6177;
            float param_22 = _6152;
            float param_23 = _6156;
            float param_24 = spec.ior;
            float param_25 = spec.F0;
            float3 param_26 = spec.tmp_col;
            float4 _6218 = Evaluate_GGXSpecular_BSDF(param_19, param_20, param_21, param_22, param_23, param_24, param_25, param_26);
            bsdf_pdf = mad(lobe_weights.specular, _6218.w, bsdf_pdf);
            lcol += ((ls.col * _6218.xyz) / ls.pdf.xxx);
        }
        float _6237 = coat.roughness * coat.roughness;
        bool _6239 = lobe_weights.clearcoat > 0.0f;
        bool _6246;
        if (_6239)
        {
            _6246 = (_6237 * _6237) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _6246 = _6239;
        }
        [branch]
        if (_6246 && _6085)
        {
            float3 param_27 = _6167;
            float3 param_28 = _6187;
            float3 param_29 = _6177;
            float param_30 = _6237;
            float param_31 = coat.ior;
            float param_32 = coat.F0;
            float4 _6264 = Evaluate_PrincipledClearcoat_BSDF(param_27, param_28, param_29, param_30, param_31, param_32);
            bsdf_pdf = mad(lobe_weights.clearcoat, _6264.w, bsdf_pdf);
            lcol += (((ls.col * 0.25f) * _6264.xyz) / ls.pdf.xxx);
        }
        [branch]
        if (lobe_weights.refraction > 0.0f)
        {
            bool _6286 = trans.fresnel != 0.0f;
            bool _6293;
            if (_6286)
            {
                _6293 = (_6143 * _6143) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _6293 = _6286;
            }
            [branch]
            if (_6293 && _6085)
            {
                float3 param_33 = _6167;
                float3 param_34 = _6187;
                float3 param_35 = _6177;
                float param_36 = _6143;
                float param_37 = _6143;
                float param_38 = 1.0f;
                float param_39 = 0.0f;
                float3 param_40 = 1.0f.xxx;
                float4 _6312 = Evaluate_GGXSpecular_BSDF(param_33, param_34, param_35, param_36, param_37, param_38, param_39, param_40);
                bsdf_pdf = mad(lobe_weights.refraction * trans.fresnel, _6312.w, bsdf_pdf);
                lcol += ((ls.col * _6312.xyz) * (trans.fresnel / ls.pdf));
            }
            float _6334 = trans.roughness * trans.roughness;
            bool _6336 = trans.fresnel != 1.0f;
            bool _6343;
            if (_6336)
            {
                _6343 = (_6334 * _6334) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _6343 = _6336;
            }
            [branch]
            if (_6343 && (N_dot_L < 0.0f))
            {
                float3 param_41 = _6167;
                float3 param_42 = _6187;
                float3 param_43 = _6177;
                float param_44 = _6334;
                float param_45 = trans.eta;
                float3 param_46 = diff.base_color;
                float4 _6361 = Evaluate_GGXRefraction_BSDF(param_41, param_42, param_43, param_44, param_45, param_46);
                float _6364 = 1.0f - trans.fresnel;
                bsdf_pdf = mad(lobe_weights.refraction * _6364, _6361.w, bsdf_pdf);
                lcol += ((ls.col * _6361.xyz) * (_6364 / ls.pdf));
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
            _9280 = lcol;
            break;
        }
        float3 _6404;
        if (N_dot_L < 0.0f)
        {
            _6404 = -surf.plane_N;
        }
        else
        {
            _6404 = surf.plane_N;
        }
        float3 param_49 = surf.P;
        float3 param_50 = _6404;
        float3 _6415 = offset_ray(param_49, param_50);
        sh_r.o[0] = _6415.x;
        sh_r.o[1] = _6415.y;
        sh_r.o[2] = _6415.z;
        sh_r.c[0] = ray.c[0] * lcol.x;
        sh_r.c[1] = ray.c[1] * lcol.y;
        sh_r.c[2] = ray.c[2] * lcol.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9280 = 0.0f.xxx;
        break;
    } while(false);
    return _9280;
}

float4 Sample_PrincipledDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float3 sheen_color, bool uniform_sampling, float rand_u, float rand_v, inout float3 out_V)
{
    float _2606 = 6.283185482025146484375f * rand_v;
    float _2609 = cos(_2606);
    float _2612 = sin(_2606);
    float3 V;
    if (uniform_sampling)
    {
        float _2621 = sqrt(mad(-rand_u, rand_u, 1.0f));
        V = float3(_2621 * _2609, _2621 * _2612, rand_u);
    }
    else
    {
        float _2634 = sqrt(rand_u);
        V = float3(_2634 * _2609, _2634 * _2612, sqrt(1.0f - rand_u));
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
    float4 _9313;
    do
    {
        [branch]
        if ((clearcoat_roughness2 * clearcoat_roughness2) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2999 = reflect(I, N);
            float param = dot(_2999, N);
            float param_1 = clearcoat_ior;
            out_V = _2999;
            float _3018 = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param, param_1) - clearcoat_F0) / (1.0f - clearcoat_F0)) * 1000000.0f;
            _9313 = float4(_3018, _3018, _3018, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _3036 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = clearcoat_roughness2;
        float param_7 = clearcoat_roughness2;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _3047 = SampleGGX_VNDF(_3036, param_6, param_7, param_8, param_9);
        float3 _3058 = normalize(reflect(-_3036, _3047));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _3058;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _3036;
        float3 param_15 = _3047;
        float3 param_16 = _3058;
        float param_17 = clearcoat_roughness2;
        float param_18 = clearcoat_ior;
        float param_19 = clearcoat_F0;
        _9313 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _9313;
}

void Sample_PrincipledNode(ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float rand_u, float rand_v, inout float mix_rand, float mix_weight, inout ray_data_t new_ray)
{
    float3 _6450 = float3(ray.d[0], ray.d[1], ray.d[2]);
    int _6454 = ray.depth & 255;
    int _6458 = (ray.depth >> 8) & 255;
    int _6462 = (ray.depth >> 16) & 255;
    int _6473 = (_6454 + _6458) + _6462;
    [branch]
    if (mix_rand < lobe_weights.diffuse)
    {
        bool _6482 = _6454 < _3458_g_params.max_diff_depth;
        bool _6489;
        if (_6482)
        {
            _6489 = _6473 < _3458_g_params.max_total_depth;
        }
        else
        {
            _6489 = _6482;
        }
        if (_6489)
        {
            float3 param = surf.T;
            float3 param_1 = surf.B;
            float3 param_2 = surf.N;
            float3 param_3 = _6450;
            float param_4 = diff.roughness;
            float3 param_5 = diff.base_color;
            float3 param_6 = diff.sheen_color;
            bool param_7 = false;
            float param_8 = rand_u;
            float param_9 = rand_v;
            float3 param_10;
            float4 _6512 = Sample_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9, param_10);
            float3 _6517 = _6512.xyz * (1.0f - metallic);
            new_ray.depth = ray.depth + 1;
            float3 param_11 = surf.P;
            float3 param_12 = surf.plane_N;
            float3 _6532 = offset_ray(param_11, param_12);
            new_ray.o[0] = _6532.x;
            new_ray.o[1] = _6532.y;
            new_ray.o[2] = _6532.z;
            new_ray.d[0] = param_10.x;
            new_ray.d[1] = param_10.y;
            new_ray.d[2] = param_10.z;
            new_ray.c[0] = ((ray.c[0] * _6517.x) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[1] = ((ray.c[1] * _6517.y) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[2] = ((ray.c[2] * _6517.z) * mix_weight) / lobe_weights.diffuse;
            new_ray.pdf = _6512.w;
        }
    }
    else
    {
        float _6582 = lobe_weights.diffuse + lobe_weights.specular;
        [branch]
        if (mix_rand < _6582)
        {
            bool _6589 = _6458 < _3458_g_params.max_spec_depth;
            bool _6596;
            if (_6589)
            {
                _6596 = _6473 < _3458_g_params.max_total_depth;
            }
            else
            {
                _6596 = _6589;
            }
            if (_6596)
            {
                float3 param_13 = surf.T;
                float3 param_14 = surf.B;
                float3 param_15 = surf.N;
                float3 param_16 = _6450;
                float3 param_17;
                float4 _6615 = Sample_GGXSpecular_BSDF(param_13, param_14, param_15, param_16, spec.roughness, spec.anisotropy, spec.ior, spec.F0, spec.tmp_col, rand_u, rand_v, param_17);
                float _6620 = _6615.w * lobe_weights.specular;
                new_ray.depth = ray.depth + 256;
                new_ray.c[0] = ((ray.c[0] * _6615.x) * mix_weight) / _6620;
                new_ray.c[1] = ((ray.c[1] * _6615.y) * mix_weight) / _6620;
                new_ray.c[2] = ((ray.c[2] * _6615.z) * mix_weight) / _6620;
                new_ray.pdf = _6620;
                float3 param_18 = surf.P;
                float3 param_19 = surf.plane_N;
                float3 _6660 = offset_ray(param_18, param_19);
                new_ray.o[0] = _6660.x;
                new_ray.o[1] = _6660.y;
                new_ray.o[2] = _6660.z;
                new_ray.d[0] = param_17.x;
                new_ray.d[1] = param_17.y;
                new_ray.d[2] = param_17.z;
            }
        }
        else
        {
            float _6685 = _6582 + lobe_weights.clearcoat;
            [branch]
            if (mix_rand < _6685)
            {
                bool _6692 = _6458 < _3458_g_params.max_spec_depth;
                bool _6699;
                if (_6692)
                {
                    _6699 = _6473 < _3458_g_params.max_total_depth;
                }
                else
                {
                    _6699 = _6692;
                }
                if (_6699)
                {
                    float3 param_20 = surf.T;
                    float3 param_21 = surf.B;
                    float3 param_22 = surf.N;
                    float3 param_23 = _6450;
                    float param_24 = coat.roughness * coat.roughness;
                    float param_25 = coat.ior;
                    float param_26 = coat.F0;
                    float param_27 = rand_u;
                    float param_28 = rand_v;
                    float3 param_29;
                    float4 _6723 = Sample_PrincipledClearcoat_BSDF(param_20, param_21, param_22, param_23, param_24, param_25, param_26, param_27, param_28, param_29);
                    float _6728 = _6723.w * lobe_weights.clearcoat;
                    new_ray.depth = ray.depth + 256;
                    new_ray.c[0] = (((0.25f * ray.c[0]) * _6723.x) * mix_weight) / _6728;
                    new_ray.c[1] = (((0.25f * ray.c[1]) * _6723.y) * mix_weight) / _6728;
                    new_ray.c[2] = (((0.25f * ray.c[2]) * _6723.z) * mix_weight) / _6728;
                    new_ray.pdf = _6728;
                    float3 param_30 = surf.P;
                    float3 param_31 = surf.plane_N;
                    float3 _6771 = offset_ray(param_30, param_31);
                    new_ray.o[0] = _6771.x;
                    new_ray.o[1] = _6771.y;
                    new_ray.o[2] = _6771.z;
                    new_ray.d[0] = param_29.x;
                    new_ray.d[1] = param_29.y;
                    new_ray.d[2] = param_29.z;
                }
            }
            else
            {
                bool _6793 = mix_rand >= trans.fresnel;
                bool _6800;
                if (_6793)
                {
                    _6800 = _6462 < _3458_g_params.max_refr_depth;
                }
                else
                {
                    _6800 = _6793;
                }
                bool _6814;
                if (!_6800)
                {
                    bool _6806 = mix_rand < trans.fresnel;
                    bool _6813;
                    if (_6806)
                    {
                        _6813 = _6458 < _3458_g_params.max_spec_depth;
                    }
                    else
                    {
                        _6813 = _6806;
                    }
                    _6814 = _6813;
                }
                else
                {
                    _6814 = _6800;
                }
                bool _6821;
                if (_6814)
                {
                    _6821 = _6473 < _3458_g_params.max_total_depth;
                }
                else
                {
                    _6821 = _6814;
                }
                [branch]
                if (_6821)
                {
                    mix_rand -= _6685;
                    mix_rand /= lobe_weights.refraction;
                    float4 F;
                    float3 V;
                    [branch]
                    if (mix_rand < trans.fresnel)
                    {
                        float3 param_32 = surf.T;
                        float3 param_33 = surf.B;
                        float3 param_34 = surf.N;
                        float3 param_35 = _6450;
                        float3 param_36;
                        float4 _6851 = Sample_GGXSpecular_BSDF(param_32, param_33, param_34, param_35, spec.roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, rand_u, rand_v, param_36);
                        V = param_36;
                        F = _6851;
                        new_ray.depth = ray.depth + 256;
                        float3 param_37 = surf.P;
                        float3 param_38 = surf.plane_N;
                        float3 _6861 = offset_ray(param_37, param_38);
                        new_ray.o[0] = _6861.x;
                        new_ray.o[1] = _6861.y;
                        new_ray.o[2] = _6861.z;
                    }
                    else
                    {
                        float3 param_39 = surf.T;
                        float3 param_40 = surf.B;
                        float3 param_41 = surf.N;
                        float3 param_42 = _6450;
                        float param_43 = trans.roughness;
                        float param_44 = trans.eta;
                        float3 param_45 = diff.base_color;
                        float param_46 = rand_u;
                        float param_47 = rand_v;
                        float4 param_48;
                        float4 _6890 = Sample_GGXRefraction_BSDF(param_39, param_40, param_41, param_42, param_43, param_44, param_45, param_46, param_47, param_48);
                        F = _6890;
                        V = param_48.xyz;
                        new_ray.depth = ray.depth + 65536;
                        float3 param_49 = surf.P;
                        float3 param_50 = -surf.plane_N;
                        float3 _6903 = offset_ray(param_49, param_50);
                        new_ray.o[0] = _6903.x;
                        new_ray.o[1] = _6903.y;
                        new_ray.o[2] = _6903.z;
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
                            float _6929 = pop_ior_stack(param_52, param_53);
                            new_ray.ior = param_52;
                        }
                    }
                    float4 _10859 = F;
                    float _6935 = _10859.w * lobe_weights.refraction;
                    float4 _10861 = _10859;
                    _10861.w = _6935;
                    F = _10861;
                    new_ray.c[0] = ((ray.c[0] * _10859.x) * mix_weight) / _6935;
                    new_ray.c[1] = ((ray.c[1] * _10859.y) * mix_weight) / _6935;
                    new_ray.c[2] = ((ray.c[2] * _10859.z) * mix_weight) / _6935;
                    new_ray.pdf = _6935;
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
    float3 _9250;
    do
    {
        float3 _6991 = float3(ray.d[0], ray.d[1], ray.d[2]);
        int _6995 = ray.depth & 255;
        int _7000 = (ray.depth >> 8) & 255;
        int _7005 = (ray.depth >> 16) & 255;
        int _7016 = (_6995 + _7000) + _7005;
        int _7024 = _3458_g_params.hi + ((_7016 + ((ray.depth >> 24) & 255)) * 9);
        uint param = uint(hash(ray.xy));
        float _7031 = construct_float(param);
        uint param_1 = uint(hash(hash(ray.xy)));
        float _7038 = construct_float(param_1);
        float2 _7057 = float2(frac(asfloat(_3442.Load((_7024 + 7) * 4 + 0)) + _7031), frac(asfloat(_3442.Load((_7024 + 8) * 4 + 0)) + _7038));
        [branch]
        if (inter.mask == 0)
        {
            ray_data_t param_2 = ray;
            float3 _7067 = Evaluate_EnvColor(param_2, _7057);
            _9250 = float3(ray.c[0] * _7067.x, ray.c[1] * _7067.y, ray.c[2] * _7067.z);
            break;
        }
        float3 _7094 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_6991 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            ray_data_t param_3 = ray;
            hit_data_t param_4 = inter;
            float3 _7107 = Evaluate_LightColor(param_3, param_4, _7057);
            _9250 = float3(ray.c[0] * _7107.x, ray.c[1] * _7107.y, ray.c[2] * _7107.z);
            break;
        }
        bool _7128 = inter.prim_index < 0;
        int _7131;
        if (_7128)
        {
            _7131 = (-1) - inter.prim_index;
        }
        else
        {
            _7131 = inter.prim_index;
        }
        uint _7142 = uint(_7131);
        material_t _7150;
        [unroll]
        for (int _89ident = 0; _89ident < 5; _89ident++)
        {
            _7150.textures[_89ident] = _4697.Load(_89ident * 4 + ((_4701.Load(_7142 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
        }
        [unroll]
        for (int _90ident = 0; _90ident < 3; _90ident++)
        {
            _7150.base_color[_90ident] = asfloat(_4697.Load(_90ident * 4 + ((_4701.Load(_7142 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
        }
        _7150.flags = _4697.Load(((_4701.Load(_7142 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
        _7150.type = _4697.Load(((_4701.Load(_7142 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
        _7150.tangent_rotation_or_strength = asfloat(_4697.Load(((_4701.Load(_7142 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
        _7150.roughness_and_anisotropic = _4697.Load(((_4701.Load(_7142 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
        _7150.ior = asfloat(_4697.Load(((_4701.Load(_7142 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
        _7150.sheen_and_sheen_tint = _4697.Load(((_4701.Load(_7142 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
        _7150.tint_and_metallic = _4697.Load(((_4701.Load(_7142 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
        _7150.transmission_and_transmission_roughness = _4697.Load(((_4701.Load(_7142 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
        _7150.specular_and_specular_tint = _4697.Load(((_4701.Load(_7142 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
        _7150.clearcoat_and_clearcoat_roughness = _4697.Load(((_4701.Load(_7142 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
        _7150.normal_map_strength_unorm = _4697.Load(((_4701.Load(_7142 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
        uint _10314 = _7150.textures[0];
        uint _10315 = _7150.textures[1];
        uint _10316 = _7150.textures[2];
        uint _10317 = _7150.textures[3];
        uint _10318 = _7150.textures[4];
        float _10319 = _7150.base_color[0];
        float _10320 = _7150.base_color[1];
        float _10321 = _7150.base_color[2];
        uint _9915 = _7150.flags;
        uint _9916 = _7150.type;
        float _9917 = _7150.tangent_rotation_or_strength;
        uint _9918 = _7150.roughness_and_anisotropic;
        float _9919 = _7150.ior;
        uint _9920 = _7150.sheen_and_sheen_tint;
        uint _9921 = _7150.tint_and_metallic;
        uint _9922 = _7150.transmission_and_transmission_roughness;
        uint _9923 = _7150.specular_and_specular_tint;
        uint _9924 = _7150.clearcoat_and_clearcoat_roughness;
        uint _9925 = _7150.normal_map_strength_unorm;
        transform_t _7205;
        _7205.xform = asfloat(uint4x4(_4344.Load4(asuint(asfloat(_7198.Load(inter.obj_index * 32 + 12))) * 128 + 0), _4344.Load4(asuint(asfloat(_7198.Load(inter.obj_index * 32 + 12))) * 128 + 16), _4344.Load4(asuint(asfloat(_7198.Load(inter.obj_index * 32 + 12))) * 128 + 32), _4344.Load4(asuint(asfloat(_7198.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _7205.inv_xform = asfloat(uint4x4(_4344.Load4(asuint(asfloat(_7198.Load(inter.obj_index * 32 + 12))) * 128 + 64), _4344.Load4(asuint(asfloat(_7198.Load(inter.obj_index * 32 + 12))) * 128 + 80), _4344.Load4(asuint(asfloat(_7198.Load(inter.obj_index * 32 + 12))) * 128 + 96), _4344.Load4(asuint(asfloat(_7198.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _7212 = _7142 * 3u;
        vertex_t _7217;
        [unroll]
        for (int _91ident = 0; _91ident < 3; _91ident++)
        {
            _7217.p[_91ident] = asfloat(_4369.Load(_91ident * 4 + _4373.Load(_7212 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _92ident = 0; _92ident < 3; _92ident++)
        {
            _7217.n[_92ident] = asfloat(_4369.Load(_92ident * 4 + _4373.Load(_7212 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _93ident = 0; _93ident < 3; _93ident++)
        {
            _7217.b[_93ident] = asfloat(_4369.Load(_93ident * 4 + _4373.Load(_7212 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _94ident = 0; _94ident < 2; _94ident++)
        {
            [unroll]
            for (int _95ident = 0; _95ident < 2; _95ident++)
            {
                _7217.t[_94ident][_95ident] = asfloat(_4369.Load(_95ident * 4 + _94ident * 8 + _4373.Load(_7212 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _7263;
        [unroll]
        for (int _96ident = 0; _96ident < 3; _96ident++)
        {
            _7263.p[_96ident] = asfloat(_4369.Load(_96ident * 4 + _4373.Load((_7212 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _97ident = 0; _97ident < 3; _97ident++)
        {
            _7263.n[_97ident] = asfloat(_4369.Load(_97ident * 4 + _4373.Load((_7212 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _98ident = 0; _98ident < 3; _98ident++)
        {
            _7263.b[_98ident] = asfloat(_4369.Load(_98ident * 4 + _4373.Load((_7212 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _99ident = 0; _99ident < 2; _99ident++)
        {
            [unroll]
            for (int _100ident = 0; _100ident < 2; _100ident++)
            {
                _7263.t[_99ident][_100ident] = asfloat(_4369.Load(_100ident * 4 + _99ident * 8 + _4373.Load((_7212 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _7309;
        [unroll]
        for (int _101ident = 0; _101ident < 3; _101ident++)
        {
            _7309.p[_101ident] = asfloat(_4369.Load(_101ident * 4 + _4373.Load((_7212 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _102ident = 0; _102ident < 3; _102ident++)
        {
            _7309.n[_102ident] = asfloat(_4369.Load(_102ident * 4 + _4373.Load((_7212 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _103ident = 0; _103ident < 3; _103ident++)
        {
            _7309.b[_103ident] = asfloat(_4369.Load(_103ident * 4 + _4373.Load((_7212 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _104ident = 0; _104ident < 2; _104ident++)
        {
            [unroll]
            for (int _105ident = 0; _105ident < 2; _105ident++)
            {
                _7309.t[_104ident][_105ident] = asfloat(_4369.Load(_105ident * 4 + _104ident * 8 + _4373.Load((_7212 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _7355 = float3(_7217.p[0], _7217.p[1], _7217.p[2]);
        float3 _7363 = float3(_7263.p[0], _7263.p[1], _7263.p[2]);
        float3 _7371 = float3(_7309.p[0], _7309.p[1], _7309.p[2]);
        float _7378 = (1.0f - inter.u) - inter.v;
        float3 _7410 = normalize(((float3(_7217.n[0], _7217.n[1], _7217.n[2]) * _7378) + (float3(_7263.n[0], _7263.n[1], _7263.n[2]) * inter.u)) + (float3(_7309.n[0], _7309.n[1], _7309.n[2]) * inter.v));
        float3 _9854 = _7410;
        float2 _7436 = ((float2(_7217.t[0][0], _7217.t[0][1]) * _7378) + (float2(_7263.t[0][0], _7263.t[0][1]) * inter.u)) + (float2(_7309.t[0][0], _7309.t[0][1]) * inter.v);
        float3 _7452 = cross(_7363 - _7355, _7371 - _7355);
        float _7457 = length(_7452);
        float3 _9855 = _7452 / _7457.xxx;
        float3 _7494 = ((float3(_7217.b[0], _7217.b[1], _7217.b[2]) * _7378) + (float3(_7263.b[0], _7263.b[1], _7263.b[2]) * inter.u)) + (float3(_7309.b[0], _7309.b[1], _7309.b[2]) * inter.v);
        float3 _9853 = _7494;
        float3 _9852 = cross(_7494, _7410);
        if (_7128)
        {
            if ((_4701.Load(_7142 * 4 + 0) & 65535u) == 65535u)
            {
                _9250 = 0.0f.xxx;
                break;
            }
            material_t _7519;
            [unroll]
            for (int _106ident = 0; _106ident < 5; _106ident++)
            {
                _7519.textures[_106ident] = _4697.Load(_106ident * 4 + (_4701.Load(_7142 * 4 + 0) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _107ident = 0; _107ident < 3; _107ident++)
            {
                _7519.base_color[_107ident] = asfloat(_4697.Load(_107ident * 4 + (_4701.Load(_7142 * 4 + 0) & 16383u) * 76 + 20));
            }
            _7519.flags = _4697.Load((_4701.Load(_7142 * 4 + 0) & 16383u) * 76 + 32);
            _7519.type = _4697.Load((_4701.Load(_7142 * 4 + 0) & 16383u) * 76 + 36);
            _7519.tangent_rotation_or_strength = asfloat(_4697.Load((_4701.Load(_7142 * 4 + 0) & 16383u) * 76 + 40));
            _7519.roughness_and_anisotropic = _4697.Load((_4701.Load(_7142 * 4 + 0) & 16383u) * 76 + 44);
            _7519.ior = asfloat(_4697.Load((_4701.Load(_7142 * 4 + 0) & 16383u) * 76 + 48));
            _7519.sheen_and_sheen_tint = _4697.Load((_4701.Load(_7142 * 4 + 0) & 16383u) * 76 + 52);
            _7519.tint_and_metallic = _4697.Load((_4701.Load(_7142 * 4 + 0) & 16383u) * 76 + 56);
            _7519.transmission_and_transmission_roughness = _4697.Load((_4701.Load(_7142 * 4 + 0) & 16383u) * 76 + 60);
            _7519.specular_and_specular_tint = _4697.Load((_4701.Load(_7142 * 4 + 0) & 16383u) * 76 + 64);
            _7519.clearcoat_and_clearcoat_roughness = _4697.Load((_4701.Load(_7142 * 4 + 0) & 16383u) * 76 + 68);
            _7519.normal_map_strength_unorm = _4697.Load((_4701.Load(_7142 * 4 + 0) & 16383u) * 76 + 72);
            _10314 = _7519.textures[0];
            _10315 = _7519.textures[1];
            _10316 = _7519.textures[2];
            _10317 = _7519.textures[3];
            _10318 = _7519.textures[4];
            _10319 = _7519.base_color[0];
            _10320 = _7519.base_color[1];
            _10321 = _7519.base_color[2];
            _9915 = _7519.flags;
            _9916 = _7519.type;
            _9917 = _7519.tangent_rotation_or_strength;
            _9918 = _7519.roughness_and_anisotropic;
            _9919 = _7519.ior;
            _9920 = _7519.sheen_and_sheen_tint;
            _9921 = _7519.tint_and_metallic;
            _9922 = _7519.transmission_and_transmission_roughness;
            _9923 = _7519.specular_and_specular_tint;
            _9924 = _7519.clearcoat_and_clearcoat_roughness;
            _9925 = _7519.normal_map_strength_unorm;
            _9855 = -_9855;
            _9854 = -_9854;
            _9853 = -_9853;
            _9852 = -_9852;
        }
        float3 param_5 = _9855;
        float4x4 param_6 = _7205.inv_xform;
        _9855 = TransformNormal(param_5, param_6);
        float3 param_7 = _9854;
        float4x4 param_8 = _7205.inv_xform;
        _9854 = TransformNormal(param_7, param_8);
        float3 param_9 = _9853;
        float4x4 param_10 = _7205.inv_xform;
        _9853 = TransformNormal(param_9, param_10);
        float3 param_11 = _9852;
        float4x4 param_12 = _7205.inv_xform;
        _9855 = normalize(_9855);
        _9854 = normalize(_9854);
        _9853 = normalize(_9853);
        _9852 = normalize(TransformNormal(param_11, param_12));
        float _7659 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _7669 = mad(0.5f, log2(abs(mad(_7263.t[0][0] - _7217.t[0][0], _7309.t[0][1] - _7217.t[0][1], -((_7309.t[0][0] - _7217.t[0][0]) * (_7263.t[0][1] - _7217.t[0][1])))) / _7457), log2(_7659));
        float param_13[4] = ray.ior;
        bool param_14 = _7128;
        float param_15 = 1.0f;
        float _7677 = peek_ior_stack(param_13, param_14, param_15);
        float3 col = 0.0f.xxx;
        float mix_rand = frac(asfloat(_3442.Load(_7024 * 4 + 0)) + _7031);
        float mix_weight = 1.0f;
        float _7716;
        float _7733;
        float _7759;
        float _7826;
        while (_9916 == 4u)
        {
            float mix_val = _9917;
            if (_10315 != 4294967295u)
            {
                mix_val *= SampleBilinear(_10315, _7436, 0, _7057).x;
            }
            if (_7128)
            {
                _7716 = _7677 / _9919;
            }
            else
            {
                _7716 = _9919 / _7677;
            }
            if (_9919 != 0.0f)
            {
                float param_16 = dot(_6991, _9854);
                float param_17 = _7716;
                _7733 = fresnel_dielectric_cos(param_16, param_17);
            }
            else
            {
                _7733 = 1.0f;
            }
            float _7748 = mix_val;
            float _7749 = _7748 * clamp(_7733, 0.0f, 1.0f);
            mix_val = _7749;
            if (mix_rand > _7749)
            {
                if ((_9915 & 2u) != 0u)
                {
                    _7759 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _7759 = 1.0f;
                }
                mix_weight *= _7759;
                material_t _7772;
                [unroll]
                for (int _108ident = 0; _108ident < 5; _108ident++)
                {
                    _7772.textures[_108ident] = _4697.Load(_108ident * 4 + _10317 * 76 + 0);
                }
                [unroll]
                for (int _109ident = 0; _109ident < 3; _109ident++)
                {
                    _7772.base_color[_109ident] = asfloat(_4697.Load(_109ident * 4 + _10317 * 76 + 20));
                }
                _7772.flags = _4697.Load(_10317 * 76 + 32);
                _7772.type = _4697.Load(_10317 * 76 + 36);
                _7772.tangent_rotation_or_strength = asfloat(_4697.Load(_10317 * 76 + 40));
                _7772.roughness_and_anisotropic = _4697.Load(_10317 * 76 + 44);
                _7772.ior = asfloat(_4697.Load(_10317 * 76 + 48));
                _7772.sheen_and_sheen_tint = _4697.Load(_10317 * 76 + 52);
                _7772.tint_and_metallic = _4697.Load(_10317 * 76 + 56);
                _7772.transmission_and_transmission_roughness = _4697.Load(_10317 * 76 + 60);
                _7772.specular_and_specular_tint = _4697.Load(_10317 * 76 + 64);
                _7772.clearcoat_and_clearcoat_roughness = _4697.Load(_10317 * 76 + 68);
                _7772.normal_map_strength_unorm = _4697.Load(_10317 * 76 + 72);
                _10314 = _7772.textures[0];
                _10315 = _7772.textures[1];
                _10316 = _7772.textures[2];
                _10317 = _7772.textures[3];
                _10318 = _7772.textures[4];
                _10319 = _7772.base_color[0];
                _10320 = _7772.base_color[1];
                _10321 = _7772.base_color[2];
                _9915 = _7772.flags;
                _9916 = _7772.type;
                _9917 = _7772.tangent_rotation_or_strength;
                _9918 = _7772.roughness_and_anisotropic;
                _9919 = _7772.ior;
                _9920 = _7772.sheen_and_sheen_tint;
                _9921 = _7772.tint_and_metallic;
                _9922 = _7772.transmission_and_transmission_roughness;
                _9923 = _7772.specular_and_specular_tint;
                _9924 = _7772.clearcoat_and_clearcoat_roughness;
                _9925 = _7772.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_9915 & 2u) != 0u)
                {
                    _7826 = 1.0f / mix_val;
                }
                else
                {
                    _7826 = 1.0f;
                }
                mix_weight *= _7826;
                material_t _7838;
                [unroll]
                for (int _110ident = 0; _110ident < 5; _110ident++)
                {
                    _7838.textures[_110ident] = _4697.Load(_110ident * 4 + _10318 * 76 + 0);
                }
                [unroll]
                for (int _111ident = 0; _111ident < 3; _111ident++)
                {
                    _7838.base_color[_111ident] = asfloat(_4697.Load(_111ident * 4 + _10318 * 76 + 20));
                }
                _7838.flags = _4697.Load(_10318 * 76 + 32);
                _7838.type = _4697.Load(_10318 * 76 + 36);
                _7838.tangent_rotation_or_strength = asfloat(_4697.Load(_10318 * 76 + 40));
                _7838.roughness_and_anisotropic = _4697.Load(_10318 * 76 + 44);
                _7838.ior = asfloat(_4697.Load(_10318 * 76 + 48));
                _7838.sheen_and_sheen_tint = _4697.Load(_10318 * 76 + 52);
                _7838.tint_and_metallic = _4697.Load(_10318 * 76 + 56);
                _7838.transmission_and_transmission_roughness = _4697.Load(_10318 * 76 + 60);
                _7838.specular_and_specular_tint = _4697.Load(_10318 * 76 + 64);
                _7838.clearcoat_and_clearcoat_roughness = _4697.Load(_10318 * 76 + 68);
                _7838.normal_map_strength_unorm = _4697.Load(_10318 * 76 + 72);
                _10314 = _7838.textures[0];
                _10315 = _7838.textures[1];
                _10316 = _7838.textures[2];
                _10317 = _7838.textures[3];
                _10318 = _7838.textures[4];
                _10319 = _7838.base_color[0];
                _10320 = _7838.base_color[1];
                _10321 = _7838.base_color[2];
                _9915 = _7838.flags;
                _9916 = _7838.type;
                _9917 = _7838.tangent_rotation_or_strength;
                _9918 = _7838.roughness_and_anisotropic;
                _9919 = _7838.ior;
                _9920 = _7838.sheen_and_sheen_tint;
                _9921 = _7838.tint_and_metallic;
                _9922 = _7838.transmission_and_transmission_roughness;
                _9923 = _7838.specular_and_specular_tint;
                _9924 = _7838.clearcoat_and_clearcoat_roughness;
                _9925 = _7838.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_10314 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_10314, _7436, 0, _7057).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_1008.Load(_10314 * 80 + 0) & 16384u) != 0u)
            {
                float3 _10882 = normals;
                _10882.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _10882;
            }
            float3 _7923 = _9854;
            _9854 = normalize(((_9852 * normals.x) + (_7923 * normals.z)) + (_9853 * normals.y));
            if ((_9925 & 65535u) != 65535u)
            {
                _9854 = normalize(_7923 + ((_9854 - _7923) * clamp(float(_9925 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_18 = _9855;
            float3 param_19 = -_6991;
            float3 param_20 = _9854;
            _9854 = ensure_valid_reflection(param_18, param_19, param_20);
        }
        float3 _7989 = ((_7355 * _7378) + (_7363 * inter.u)) + (_7371 * inter.v);
        float3 _7996 = float3(-_7989.z, 0.0f, _7989.x);
        float3 tangent = _7996;
        float3 param_21 = _7996;
        float4x4 param_22 = _7205.inv_xform;
        float3 _8002 = TransformNormal(param_21, param_22);
        tangent = _8002;
        float3 _8006 = cross(_8002, _9854);
        if (dot(_8006, _8006) == 0.0f)
        {
            float3 param_23 = _7989;
            float4x4 param_24 = _7205.inv_xform;
            tangent = TransformNormal(param_23, param_24);
        }
        if (_9917 != 0.0f)
        {
            float3 param_25 = tangent;
            float3 param_26 = _9854;
            float param_27 = _9917;
            tangent = rotate_around_axis(param_25, param_26, param_27);
        }
        float3 _8039 = normalize(cross(tangent, _9854));
        _9853 = _8039;
        _9852 = cross(_9854, _8039);
        float3 _10013 = 0.0f.xxx;
        float3 _10012 = 0.0f.xxx;
        float _10017 = 0.0f;
        float _10015 = 0.0f;
        float _10016 = 1.0f;
        bool _8055 = _3458_g_params.li_count != 0;
        bool _8061;
        if (_8055)
        {
            _8061 = _9916 != 3u;
        }
        else
        {
            _8061 = _8055;
        }
        float3 _10014;
        bool _10018;
        bool _10019;
        if (_8061)
        {
            float3 param_28 = _7094;
            float3 param_29 = _9852;
            float3 param_30 = _9853;
            float3 param_31 = _9854;
            int param_32 = _7024;
            float2 param_33 = float2(_7031, _7038);
            light_sample_t _10028 = { _10012, _10013, _10014, _10015, _10016, _10017, _10018, _10019 };
            light_sample_t param_34 = _10028;
            SampleLightSource(param_28, param_29, param_30, param_31, param_32, param_33, param_34);
            _10012 = param_34.col;
            _10013 = param_34.L;
            _10014 = param_34.lp;
            _10015 = param_34.area;
            _10016 = param_34.dist_mul;
            _10017 = param_34.pdf;
            _10018 = param_34.cast_shadow;
            _10019 = param_34.from_env;
        }
        float _8089 = dot(_9854, _10013);
        float3 base_color = float3(_10319, _10320, _10321);
        [branch]
        if (_10315 != 4294967295u)
        {
            base_color *= SampleBilinear(_10315, _7436, int(get_texture_lod(texSize(_10315), _7669)), _7057, true, true).xyz;
        }
        out_base_color = base_color;
        out_normals = _9854;
        float3 tint_color = 0.0f.xxx;
        float _8126 = lum(base_color);
        [flatten]
        if (_8126 > 0.0f)
        {
            tint_color = base_color / _8126.xxx;
        }
        float roughness = clamp(float(_9918 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_10316 != 4294967295u)
        {
            roughness *= SampleBilinear(_10316, _7436, int(get_texture_lod(texSize(_10316), _7669)), _7057, false, true).x;
        }
        float _8172 = frac(asfloat(_3442.Load((_7024 + 1) * 4 + 0)) + _7031);
        float _8181 = frac(asfloat(_3442.Load((_7024 + 2) * 4 + 0)) + _7038);
        float _10441 = 0.0f;
        float _10440 = 0.0f;
        float _10439 = 0.0f;
        float _10077[4];
        [unroll]
        for (int i = 0; i < 4; )
        {
            _10077[i] = ray.ior[i];
            i++;
            continue;
        }
        float _10078 = _7659;
        float _10079 = ray.cone_spread;
        int _10080 = ray.xy;
        float _10075 = 0.0f;
        float _10546 = 0.0f;
        float _10545 = 0.0f;
        float _10544 = 0.0f;
        int _10182 = ray.depth;
        int _10186 = ray.xy;
        int _10081;
        float _10184;
        float _10369;
        float _10370;
        float _10371;
        float _10404;
        float _10405;
        float _10406;
        float _10474;
        float _10475;
        float _10476;
        float _10509;
        float _10510;
        float _10511;
        [branch]
        if (_9916 == 0u)
        {
            [branch]
            if ((_10017 > 0.0f) && (_8089 > 0.0f))
            {
                light_sample_t _10045 = { _10012, _10013, _10014, _10015, _10016, _10017, _10018, _10019 };
                surface_t _9863 = { _7094, _9852, _9853, _9854, _9855, _7436 };
                float _10550[3] = { _10544, _10545, _10546 };
                float _10515[3] = { _10509, _10510, _10511 };
                float _10480[3] = { _10474, _10475, _10476 };
                shadow_ray_t _10196 = { _10480, _10182, _10515, _10184, _10550, _10186 };
                shadow_ray_t param_35 = _10196;
                float3 _8241 = Evaluate_DiffuseNode(_10045, ray, _9863, base_color, roughness, mix_weight, param_35);
                _10474 = param_35.o[0];
                _10475 = param_35.o[1];
                _10476 = param_35.o[2];
                _10182 = param_35.depth;
                _10509 = param_35.d[0];
                _10510 = param_35.d[1];
                _10511 = param_35.d[2];
                _10184 = param_35.dist;
                _10544 = param_35.c[0];
                _10545 = param_35.c[1];
                _10546 = param_35.c[2];
                _10186 = param_35.xy;
                col += _8241;
            }
            bool _8248 = _6995 < _3458_g_params.max_diff_depth;
            bool _8255;
            if (_8248)
            {
                _8255 = _7016 < _3458_g_params.max_total_depth;
            }
            else
            {
                _8255 = _8248;
            }
            [branch]
            if (_8255)
            {
                surface_t _9870 = { _7094, _9852, _9853, _9854, _9855, _7436 };
                float _10445[3] = { _10439, _10440, _10441 };
                float _10410[3] = { _10404, _10405, _10406 };
                float _10375[3] = { _10369, _10370, _10371 };
                ray_data_t _10095 = { _10375, _10410, _10075, _10445, _10077, _10078, _10079, _10080, _10081 };
                ray_data_t param_36 = _10095;
                Sample_DiffuseNode(ray, _9870, base_color, roughness, _8172, _8181, mix_weight, param_36);
                _10369 = param_36.o[0];
                _10370 = param_36.o[1];
                _10371 = param_36.o[2];
                _10404 = param_36.d[0];
                _10405 = param_36.d[1];
                _10406 = param_36.d[2];
                _10075 = param_36.pdf;
                _10439 = param_36.c[0];
                _10440 = param_36.c[1];
                _10441 = param_36.c[2];
                _10077 = param_36.ior;
                _10078 = param_36.cone_width;
                _10079 = param_36.cone_spread;
                _10080 = param_36.xy;
                _10081 = param_36.depth;
            }
        }
        else
        {
            [branch]
            if (_9916 == 1u)
            {
                float param_37 = 1.0f;
                float param_38 = 1.5f;
                float _8279 = fresnel_dielectric_cos(param_37, param_38);
                float _8283 = roughness * roughness;
                bool _8286 = _10017 > 0.0f;
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
                if (_8293 && (_8089 > 0.0f))
                {
                    light_sample_t _10054 = { _10012, _10013, _10014, _10015, _10016, _10017, _10018, _10019 };
                    surface_t _9877 = { _7094, _9852, _9853, _9854, _9855, _7436 };
                    float _10557[3] = { _10544, _10545, _10546 };
                    float _10522[3] = { _10509, _10510, _10511 };
                    float _10487[3] = { _10474, _10475, _10476 };
                    shadow_ray_t _10209 = { _10487, _10182, _10522, _10184, _10557, _10186 };
                    shadow_ray_t param_39 = _10209;
                    float3 _8308 = Evaluate_GlossyNode(_10054, ray, _9877, base_color, roughness, 1.5f, _8279, mix_weight, param_39);
                    _10474 = param_39.o[0];
                    _10475 = param_39.o[1];
                    _10476 = param_39.o[2];
                    _10182 = param_39.depth;
                    _10509 = param_39.d[0];
                    _10510 = param_39.d[1];
                    _10511 = param_39.d[2];
                    _10184 = param_39.dist;
                    _10544 = param_39.c[0];
                    _10545 = param_39.c[1];
                    _10546 = param_39.c[2];
                    _10186 = param_39.xy;
                    col += _8308;
                }
                bool _8315 = _7000 < _3458_g_params.max_spec_depth;
                bool _8322;
                if (_8315)
                {
                    _8322 = _7016 < _3458_g_params.max_total_depth;
                }
                else
                {
                    _8322 = _8315;
                }
                [branch]
                if (_8322)
                {
                    surface_t _9884 = { _7094, _9852, _9853, _9854, _9855, _7436 };
                    float _10452[3] = { _10439, _10440, _10441 };
                    float _10417[3] = { _10404, _10405, _10406 };
                    float _10382[3] = { _10369, _10370, _10371 };
                    ray_data_t _10114 = { _10382, _10417, _10075, _10452, _10077, _10078, _10079, _10080, _10081 };
                    ray_data_t param_40 = _10114;
                    Sample_GlossyNode(ray, _9884, base_color, roughness, 1.5f, _8279, _8172, _8181, mix_weight, param_40);
                    _10369 = param_40.o[0];
                    _10370 = param_40.o[1];
                    _10371 = param_40.o[2];
                    _10404 = param_40.d[0];
                    _10405 = param_40.d[1];
                    _10406 = param_40.d[2];
                    _10075 = param_40.pdf;
                    _10439 = param_40.c[0];
                    _10440 = param_40.c[1];
                    _10441 = param_40.c[2];
                    _10077 = param_40.ior;
                    _10078 = param_40.cone_width;
                    _10079 = param_40.cone_spread;
                    _10080 = param_40.xy;
                    _10081 = param_40.depth;
                }
            }
            else
            {
                [branch]
                if (_9916 == 2u)
                {
                    float _8346 = roughness * roughness;
                    bool _8349 = _10017 > 0.0f;
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
                    if (_8356 && (_8089 < 0.0f))
                    {
                        float _8364;
                        if (_7128)
                        {
                            _8364 = _9919 / _7677;
                        }
                        else
                        {
                            _8364 = _7677 / _9919;
                        }
                        light_sample_t _10063 = { _10012, _10013, _10014, _10015, _10016, _10017, _10018, _10019 };
                        surface_t _9891 = { _7094, _9852, _9853, _9854, _9855, _7436 };
                        float _10564[3] = { _10544, _10545, _10546 };
                        float _10529[3] = { _10509, _10510, _10511 };
                        float _10494[3] = { _10474, _10475, _10476 };
                        shadow_ray_t _10222 = { _10494, _10182, _10529, _10184, _10564, _10186 };
                        shadow_ray_t param_41 = _10222;
                        float3 _8386 = Evaluate_RefractiveNode(_10063, ray, _9891, base_color, _8346, _8364, mix_weight, param_41);
                        _10474 = param_41.o[0];
                        _10475 = param_41.o[1];
                        _10476 = param_41.o[2];
                        _10182 = param_41.depth;
                        _10509 = param_41.d[0];
                        _10510 = param_41.d[1];
                        _10511 = param_41.d[2];
                        _10184 = param_41.dist;
                        _10544 = param_41.c[0];
                        _10545 = param_41.c[1];
                        _10546 = param_41.c[2];
                        _10186 = param_41.xy;
                        col += _8386;
                    }
                    bool _8393 = _7005 < _3458_g_params.max_refr_depth;
                    bool _8400;
                    if (_8393)
                    {
                        _8400 = _7016 < _3458_g_params.max_total_depth;
                    }
                    else
                    {
                        _8400 = _8393;
                    }
                    [branch]
                    if (_8400)
                    {
                        surface_t _9898 = { _7094, _9852, _9853, _9854, _9855, _7436 };
                        float _10459[3] = { _10439, _10440, _10441 };
                        float _10424[3] = { _10404, _10405, _10406 };
                        float _10389[3] = { _10369, _10370, _10371 };
                        ray_data_t _10133 = { _10389, _10424, _10075, _10459, _10077, _10078, _10079, _10080, _10081 };
                        ray_data_t param_42 = _10133;
                        Sample_RefractiveNode(ray, _9898, base_color, roughness, _7128, _9919, _7677, _8172, _8181, mix_weight, param_42);
                        _10369 = param_42.o[0];
                        _10370 = param_42.o[1];
                        _10371 = param_42.o[2];
                        _10404 = param_42.d[0];
                        _10405 = param_42.d[1];
                        _10406 = param_42.d[2];
                        _10075 = param_42.pdf;
                        _10439 = param_42.c[0];
                        _10440 = param_42.c[1];
                        _10441 = param_42.c[2];
                        _10077 = param_42.ior;
                        _10078 = param_42.cone_width;
                        _10079 = param_42.cone_spread;
                        _10080 = param_42.xy;
                        _10081 = param_42.depth;
                    }
                }
                else
                {
                    [branch]
                    if (_9916 == 3u)
                    {
                        float mis_weight = 1.0f;
                        [branch]
                        if ((_9915 & 1u) != 0u)
                        {
                            float3 _8470 = mul(float4(_7452, 0.0f), _7205.xform).xyz;
                            float _8473 = length(_8470);
                            float _8485 = abs(dot(_6991, _8470 / _8473.xxx));
                            if (_8485 > 0.0f)
                            {
                                float param_43 = ray.pdf;
                                float param_44 = (inter.t * inter.t) / ((0.5f * _8473) * _8485);
                                mis_weight = power_heuristic(param_43, param_44);
                            }
                        }
                        col += (base_color * ((mix_weight * mis_weight) * _9917));
                    }
                    else
                    {
                        [branch]
                        if (_9916 == 6u)
                        {
                            float metallic = clamp(float((_9921 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_10317 != 4294967295u)
                            {
                                metallic *= SampleBilinear(_10317, _7436, int(get_texture_lod(texSize(_10317), _7669)), _7057).x;
                            }
                            float specular = clamp(float(_9923 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_10318 != 4294967295u)
                            {
                                specular *= SampleBilinear(_10318, _7436, int(get_texture_lod(texSize(_10318), _7669)), _7057).x;
                            }
                            float _8604 = clamp(float(_9924 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8612 = clamp(float((_9924 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8620 = 2.0f * clamp(float(_9920 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float3 _8638 = lerp(1.0f.xxx, tint_color, clamp(float((_9920 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _8620;
                            float3 _8658 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_9923 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                            float _8667 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                            float param_45 = 1.0f;
                            float param_46 = _8667;
                            float _8673 = fresnel_dielectric_cos(param_45, param_46);
                            float _8681 = clamp(float((_9918 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8692 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _8604))) - 1.0f;
                            float param_47 = 1.0f;
                            float param_48 = _8692;
                            float _8698 = fresnel_dielectric_cos(param_47, param_48);
                            float _8713 = mad(roughness - 1.0f, 1.0f - clamp(float((_9922 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                            float _8719;
                            if (_7128)
                            {
                                _8719 = _9919 / _7677;
                            }
                            else
                            {
                                _8719 = _7677 / _9919;
                            }
                            float param_49 = dot(_6991, _9854);
                            float param_50 = 1.0f / _8719;
                            float _8742 = fresnel_dielectric_cos(param_49, param_50);
                            float param_51 = dot(_6991, _9854);
                            float param_52 = _8667;
                            lobe_weights_t _8781 = get_lobe_weights(lerp(_8126, 1.0f, _8620), lum(lerp(_8658, 1.0f.xxx, ((fresnel_dielectric_cos(param_51, param_52) - _8673) / (1.0f - _8673)).xxx)), specular, metallic, clamp(float(_9922 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _8604);
                            [branch]
                            if (_10017 > 0.0f)
                            {
                                light_sample_t _10072 = { _10012, _10013, _10014, _10015, _10016, _10017, _10018, _10019 };
                                surface_t _9905 = { _7094, _9852, _9853, _9854, _9855, _7436 };
                                diff_params_t _10264 = { base_color, _8638, roughness };
                                spec_params_t _10279 = { _8658, roughness, _8667, _8673, _8681 };
                                clearcoat_params_t _10292 = { _8612, _8692, _8698 };
                                transmission_params_t _10307 = { _8713, _9919, _8719, _8742, _7128 };
                                float _10571[3] = { _10544, _10545, _10546 };
                                float _10536[3] = { _10509, _10510, _10511 };
                                float _10501[3] = { _10474, _10475, _10476 };
                                shadow_ray_t _10235 = { _10501, _10182, _10536, _10184, _10571, _10186 };
                                shadow_ray_t param_53 = _10235;
                                float3 _8800 = Evaluate_PrincipledNode(_10072, ray, _9905, _8781, _10264, _10279, _10292, _10307, metallic, _8089, mix_weight, param_53);
                                _10474 = param_53.o[0];
                                _10475 = param_53.o[1];
                                _10476 = param_53.o[2];
                                _10182 = param_53.depth;
                                _10509 = param_53.d[0];
                                _10510 = param_53.d[1];
                                _10511 = param_53.d[2];
                                _10184 = param_53.dist;
                                _10544 = param_53.c[0];
                                _10545 = param_53.c[1];
                                _10546 = param_53.c[2];
                                _10186 = param_53.xy;
                                col += _8800;
                            }
                            surface_t _9912 = { _7094, _9852, _9853, _9854, _9855, _7436 };
                            diff_params_t _10268 = { base_color, _8638, roughness };
                            spec_params_t _10285 = { _8658, roughness, _8667, _8673, _8681 };
                            clearcoat_params_t _10296 = { _8612, _8692, _8698 };
                            transmission_params_t _10313 = { _8713, _9919, _8719, _8742, _7128 };
                            float param_54 = mix_rand;
                            float _10466[3] = { _10439, _10440, _10441 };
                            float _10431[3] = { _10404, _10405, _10406 };
                            float _10396[3] = { _10369, _10370, _10371 };
                            ray_data_t _10152 = { _10396, _10431, _10075, _10466, _10077, _10078, _10079, _10080, _10081 };
                            ray_data_t param_55 = _10152;
                            Sample_PrincipledNode(ray, _9912, _8781, _10268, _10285, _10296, _10313, metallic, _8172, _8181, param_54, mix_weight, param_55);
                            _10369 = param_55.o[0];
                            _10370 = param_55.o[1];
                            _10371 = param_55.o[2];
                            _10404 = param_55.d[0];
                            _10405 = param_55.d[1];
                            _10406 = param_55.d[2];
                            _10075 = param_55.pdf;
                            _10439 = param_55.c[0];
                            _10440 = param_55.c[1];
                            _10441 = param_55.c[2];
                            _10077 = param_55.ior;
                            _10078 = param_55.cone_width;
                            _10079 = param_55.cone_spread;
                            _10080 = param_55.xy;
                            _10081 = param_55.depth;
                        }
                    }
                }
            }
        }
        float _8834 = max(_10439, max(_10440, _10441));
        float _8846;
        if (_7016 > _3458_g_params.min_total_depth)
        {
            _8846 = max(0.0500000007450580596923828125f, 1.0f - _8834);
        }
        else
        {
            _8846 = 0.0f;
        }
        bool _8860 = (frac(asfloat(_3442.Load((_7024 + 6) * 4 + 0)) + _7031) >= _8846) && (_8834 > 0.0f);
        bool _8866;
        if (_8860)
        {
            _8866 = _10075 > 0.0f;
        }
        else
        {
            _8866 = _8860;
        }
        [branch]
        if (_8866)
        {
            float _8870 = _10075;
            float _8871 = min(_8870, 1000000.0f);
            _10075 = _8871;
            float _8874 = 1.0f - _8846;
            float _8876 = _10439;
            float _8877 = _8876 / _8874;
            _10439 = _8877;
            float _8882 = _10440;
            float _8883 = _8882 / _8874;
            _10440 = _8883;
            float _8888 = _10441;
            float _8889 = _8888 / _8874;
            _10441 = _8889;
            uint _8897;
            _8895.InterlockedAdd(0, 1u, _8897);
            _8906.Store(_8897 * 72 + 0, asuint(_10369));
            _8906.Store(_8897 * 72 + 4, asuint(_10370));
            _8906.Store(_8897 * 72 + 8, asuint(_10371));
            _8906.Store(_8897 * 72 + 12, asuint(_10404));
            _8906.Store(_8897 * 72 + 16, asuint(_10405));
            _8906.Store(_8897 * 72 + 20, asuint(_10406));
            _8906.Store(_8897 * 72 + 24, asuint(_8871));
            _8906.Store(_8897 * 72 + 28, asuint(_8877));
            _8906.Store(_8897 * 72 + 32, asuint(_8883));
            _8906.Store(_8897 * 72 + 36, asuint(_8889));
            _8906.Store(_8897 * 72 + 40, asuint(_10077[0]));
            _8906.Store(_8897 * 72 + 44, asuint(_10077[1]));
            _8906.Store(_8897 * 72 + 48, asuint(_10077[2]));
            _8906.Store(_8897 * 72 + 52, asuint(_10077[3]));
            _8906.Store(_8897 * 72 + 56, asuint(_10078));
            _8906.Store(_8897 * 72 + 60, asuint(_10079));
            _8906.Store(_8897 * 72 + 64, uint(_10080));
            _8906.Store(_8897 * 72 + 68, uint(_10081));
        }
        [branch]
        if (max(_10544, max(_10545, _10546)) > 0.0f)
        {
            float3 _8983 = _10014 - float3(_10474, _10475, _10476);
            float _8986 = length(_8983);
            float3 _8990 = _8983 / _8986.xxx;
            float sh_dist = _8986 * _10016;
            if (_10019)
            {
                sh_dist = -sh_dist;
            }
            float _9002 = _8990.x;
            _10509 = _9002;
            float _9005 = _8990.y;
            _10510 = _9005;
            float _9008 = _8990.z;
            _10511 = _9008;
            _10184 = sh_dist;
            uint _9014;
            _8895.InterlockedAdd(8, 1u, _9014);
            _9022.Store(_9014 * 48 + 0, asuint(_10474));
            _9022.Store(_9014 * 48 + 4, asuint(_10475));
            _9022.Store(_9014 * 48 + 8, asuint(_10476));
            _9022.Store(_9014 * 48 + 12, uint(_10182));
            _9022.Store(_9014 * 48 + 16, asuint(_9002));
            _9022.Store(_9014 * 48 + 20, asuint(_9005));
            _9022.Store(_9014 * 48 + 24, asuint(_9008));
            _9022.Store(_9014 * 48 + 28, asuint(sh_dist));
            _9022.Store(_9014 * 48 + 32, asuint(_10544));
            _9022.Store(_9014 * 48 + 36, asuint(_10545));
            _9022.Store(_9014 * 48 + 40, asuint(_10546));
            _9022.Store(_9014 * 48 + 44, uint(_10186));
        }
        _9250 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _9250;
}

void comp_main()
{
    do
    {
        int _9088 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_9088) >= _8895.Load(4))
        {
            break;
        }
        int _9104 = int(_9101.Load(_9088 * 72 + 64));
        int _9111 = int(_9101.Load(_9088 * 72 + 64));
        hit_data_t _9122;
        _9122.mask = int(_9118.Load(_9088 * 24 + 0));
        _9122.obj_index = int(_9118.Load(_9088 * 24 + 4));
        _9122.prim_index = int(_9118.Load(_9088 * 24 + 8));
        _9122.t = asfloat(_9118.Load(_9088 * 24 + 12));
        _9122.u = asfloat(_9118.Load(_9088 * 24 + 16));
        _9122.v = asfloat(_9118.Load(_9088 * 24 + 20));
        ray_data_t _9138;
        [unroll]
        for (int _112ident = 0; _112ident < 3; _112ident++)
        {
            _9138.o[_112ident] = asfloat(_9101.Load(_112ident * 4 + _9088 * 72 + 0));
        }
        [unroll]
        for (int _113ident = 0; _113ident < 3; _113ident++)
        {
            _9138.d[_113ident] = asfloat(_9101.Load(_113ident * 4 + _9088 * 72 + 12));
        }
        _9138.pdf = asfloat(_9101.Load(_9088 * 72 + 24));
        [unroll]
        for (int _114ident = 0; _114ident < 3; _114ident++)
        {
            _9138.c[_114ident] = asfloat(_9101.Load(_114ident * 4 + _9088 * 72 + 28));
        }
        [unroll]
        for (int _115ident = 0; _115ident < 4; _115ident++)
        {
            _9138.ior[_115ident] = asfloat(_9101.Load(_115ident * 4 + _9088 * 72 + 40));
        }
        _9138.cone_width = asfloat(_9101.Load(_9088 * 72 + 56));
        _9138.cone_spread = asfloat(_9101.Load(_9088 * 72 + 60));
        _9138.xy = int(_9101.Load(_9088 * 72 + 64));
        _9138.depth = int(_9101.Load(_9088 * 72 + 68));
        hit_data_t _9344 = { _9122.mask, _9122.obj_index, _9122.prim_index, _9122.t, _9122.u, _9122.v };
        hit_data_t param = _9344;
        float _9393[4] = { _9138.ior[0], _9138.ior[1], _9138.ior[2], _9138.ior[3] };
        float _9384[3] = { _9138.c[0], _9138.c[1], _9138.c[2] };
        float _9377[3] = { _9138.d[0], _9138.d[1], _9138.d[2] };
        float _9370[3] = { _9138.o[0], _9138.o[1], _9138.o[2] };
        ray_data_t _9363 = { _9370, _9377, _9138.pdf, _9384, _9393, _9138.cone_width, _9138.cone_spread, _9138.xy, _9138.depth };
        ray_data_t param_1 = _9363;
        float3 param_2 = 0.0f.xxx;
        float3 param_3 = 0.0f.xxx;
        float3 _9194 = ShadeSurface(param, param_1, param_2, param_3);
        int2 _9208 = int2((_9104 >> 16) & 65535, _9111 & 65535);
        g_out_img[_9208] = float4(min(_9194, _3458_g_params.clamp_val.xxx) + g_out_img[_9208].xyz, 1.0f);
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
