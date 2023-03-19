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

ByteAddressBuffer _1001 : register(t20, space0);
ByteAddressBuffer _3521 : register(t15, space0);
ByteAddressBuffer _3557 : register(t6, space0);
ByteAddressBuffer _3561 : register(t7, space0);
ByteAddressBuffer _4404 : register(t11, space0);
ByteAddressBuffer _4429 : register(t13, space0);
ByteAddressBuffer _4433 : register(t14, space0);
ByteAddressBuffer _4757 : register(t10, space0);
ByteAddressBuffer _4761 : register(t9, space0);
ByteAddressBuffer _7152 : register(t12, space0);
RWByteAddressBuffer _8805 : register(u3, space0);
RWByteAddressBuffer _8816 : register(u1, space0);
RWByteAddressBuffer _8932 : register(u2, space0);
ByteAddressBuffer _9030 : register(t4, space0);
ByteAddressBuffer _9051 : register(t5, space0);
ByteAddressBuffer _9136 : register(t8, space0);
cbuffer UniformParams
{
    Params _3537_g_params : packoffset(c0);
};

Texture2DArray<float4> g_atlases[7] : register(t21, space0);
SamplerState _g_atlases_sampler[7] : register(s21, space0);
Texture2D<float4> g_env_qtree : register(t16, space0);
SamplerState _g_env_qtree_sampler : register(s16, space0);
RWTexture2D<float4> g_out_img : register(u0, space0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

float2 TransformUV(float2 _uv, atlas_texture_t t, int mip_level)
{
    uint _9311[14] = t.pos;
    uint _9314[14] = t.pos;
    uint _1094 = t.size & 16383u;
    uint _1097 = t.size >> uint(16);
    uint _1098 = _1097 & 16383u;
    float2 size = float2(float(_1094), float(_1098));
    if ((_1097 & 32768u) != 0u)
    {
        size = float2(float(_1094 >> uint(mip_level)), float(_1098 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_9311[mip_level] & 65535u), float((_9314[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
}

float3 rgbe_to_rgb(float4 rgbe)
{
    return rgbe.xyz * exp2(mad(255.0f, rgbe.w, -128.0f));
}

float3 SampleLatlong_RGBE(atlas_texture_t t, float3 dir, float y_rotation)
{
    float _1266 = atan2(dir.z, dir.x) + y_rotation;
    float phi = _1266;
    if (_1266 < 0.0f)
    {
        phi += 6.283185482025146484375f;
    }
    if (phi > 6.283185482025146484375f)
    {
        phi -= 6.283185482025146484375f;
    }
    float2 _1288 = TransformUV(float2(frac(phi * 0.15915493667125701904296875f), acos(clamp(dir.y, -1.0f, 1.0f)) * 0.3183098733425140380859375f), t, 0);
    uint _1295 = t.atlas;
    int3 _1304 = int3(int2(_1288), int(t.page[0] & 255u));
    float2 _1351 = frac(_1288);
    float4 param = g_atlases[NonUniformResourceIndex(_1295)].Load(int4(_1304, 0), int2(0, 0));
    float4 param_1 = g_atlases[NonUniformResourceIndex(_1295)].Load(int4(_1304, 0), int2(1, 0));
    float4 param_2 = g_atlases[NonUniformResourceIndex(_1295)].Load(int4(_1304, 0), int2(0, 1));
    float4 param_3 = g_atlases[NonUniformResourceIndex(_1295)].Load(int4(_1304, 0), int2(1, 1));
    float _1371 = _1351.x;
    float _1376 = 1.0f - _1371;
    float _1392 = _1351.y;
    return (((rgbe_to_rgb(param_3) * _1371) + (rgbe_to_rgb(param_2) * _1376)) * _1392) + (((rgbe_to_rgb(param_1) * _1371) + (rgbe_to_rgb(param) * _1376)) * (1.0f - _1392));
}

float2 DirToCanonical(float3 d, float y_rotation)
{
    float _730 = (-atan2(d.z, d.x)) + y_rotation;
    float phi = _730;
    if (_730 < 0.0f)
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
    float2 _757 = DirToCanonical(L, -y_rotation);
    float factor = 1.0f;
    while (lod >= 0)
    {
        int2 _777 = clamp(int2(_757 * float(res)), int2(0, 0), (res - 1).xx);
        float4 quad = qtree_tex.Load(int3(_777 / int2(2, 2), lod));
        float _812 = ((quad.x + quad.y) + quad.z) + quad.w;
        if (_812 <= 0.0f)
        {
            break;
        }
        factor *= ((4.0f * quad[(0 | ((_777.x & 1) << 0)) | ((_777.y & 1) << 1)]) / _812);
        lod--;
        res *= 2;
    }
    return factor * 0.079577468335628509521484375f;
}

float power_heuristic(float a, float b)
{
    float _1405 = a * a;
    return _1405 / mad(b, b, _1405);
}

float3 Evaluate_EnvColor(ray_data_t ray)
{
    float3 _5011 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 env_col = _3537_g_params.back_col.xyz;
    uint _5019 = asuint(_3537_g_params.back_col.w);
    if (_5019 != 4294967295u)
    {
        atlas_texture_t _5030;
        _5030.size = _1001.Load(_5019 * 80 + 0);
        _5030.atlas = _1001.Load(_5019 * 80 + 4);
        [unroll]
        for (int _58ident = 0; _58ident < 4; _58ident++)
        {
            _5030.page[_58ident] = _1001.Load(_58ident * 4 + _5019 * 80 + 8);
        }
        [unroll]
        for (int _59ident = 0; _59ident < 14; _59ident++)
        {
            _5030.pos[_59ident] = _1001.Load(_59ident * 4 + _5019 * 80 + 24);
        }
        uint _9681[14] = { _5030.pos[0], _5030.pos[1], _5030.pos[2], _5030.pos[3], _5030.pos[4], _5030.pos[5], _5030.pos[6], _5030.pos[7], _5030.pos[8], _5030.pos[9], _5030.pos[10], _5030.pos[11], _5030.pos[12], _5030.pos[13] };
        uint _9652[4] = { _5030.page[0], _5030.page[1], _5030.page[2], _5030.page[3] };
        atlas_texture_t _9643 = { _5030.size, _5030.atlas, _9652, _9681 };
        float param = _3537_g_params.back_rotation;
        env_col *= SampleLatlong_RGBE(_9643, _5011, param);
    }
    if (_3537_g_params.env_qtree_levels > 0)
    {
        float param_1 = ray.pdf;
        float param_2 = Evaluate_EnvQTree(_3537_g_params.back_rotation, g_env_qtree, _g_env_qtree_sampler, _3537_g_params.env_qtree_levels, _5011);
        env_col *= power_heuristic(param_1, param_2);
    }
    else
    {
        if (_3537_g_params.env_mult_importance != 0)
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
    float3 _5142 = float3(ray.d[0], ray.d[1], ray.d[2]);
    light_t _5156;
    _5156.type_and_param0 = _3557.Load4(((-1) - inter.obj_index) * 64 + 0);
    _5156.param1 = asfloat(_3557.Load4(((-1) - inter.obj_index) * 64 + 16));
    _5156.param2 = asfloat(_3557.Load4(((-1) - inter.obj_index) * 64 + 32));
    _5156.param3 = asfloat(_3557.Load4(((-1) - inter.obj_index) * 64 + 48));
    float3 lcol = asfloat(_5156.type_and_param0.yzw);
    [branch]
    if ((_5156.type_and_param0.x & 128u) != 0u)
    {
        float3 env_col = _3537_g_params.env_col.xyz;
        uint _5183 = asuint(_3537_g_params.env_col.w);
        if (_5183 != 4294967295u)
        {
            atlas_texture_t _5190;
            _5190.size = _1001.Load(_5183 * 80 + 0);
            _5190.atlas = _1001.Load(_5183 * 80 + 4);
            [unroll]
            for (int _60ident = 0; _60ident < 4; _60ident++)
            {
                _5190.page[_60ident] = _1001.Load(_60ident * 4 + _5183 * 80 + 8);
            }
            [unroll]
            for (int _61ident = 0; _61ident < 14; _61ident++)
            {
                _5190.pos[_61ident] = _1001.Load(_61ident * 4 + _5183 * 80 + 24);
            }
            uint _9743[14] = { _5190.pos[0], _5190.pos[1], _5190.pos[2], _5190.pos[3], _5190.pos[4], _5190.pos[5], _5190.pos[6], _5190.pos[7], _5190.pos[8], _5190.pos[9], _5190.pos[10], _5190.pos[11], _5190.pos[12], _5190.pos[13] };
            uint _9714[4] = { _5190.page[0], _5190.page[1], _5190.page[2], _5190.page[3] };
            atlas_texture_t _9705 = { _5190.size, _5190.atlas, _9714, _9743 };
            float param = _3537_g_params.env_rotation;
            env_col *= SampleLatlong_RGBE(_9705, _5142, param);
        }
        lcol *= env_col;
    }
    uint _5250 = _5156.type_and_param0.x & 31u;
    if (_5250 == 0u)
    {
        float param_1 = ray.pdf;
        float param_2 = (inter.t * inter.t) / ((0.5f * _5156.param1.w) * dot(_5142, normalize(_5156.param1.xyz - (float3(ray.o[0], ray.o[1], ray.o[2]) + (_5142 * inter.t)))));
        lcol *= power_heuristic(param_1, param_2);
        bool _5317 = _5156.param3.x > 0.0f;
        bool _5323;
        if (_5317)
        {
            _5323 = _5156.param3.y > 0.0f;
        }
        else
        {
            _5323 = _5317;
        }
        [branch]
        if (_5323)
        {
            [flatten]
            if (_5156.param3.y > 0.0f)
            {
                lcol *= clamp((_5156.param3.x - acos(clamp(-dot(_5142, _5156.param2.xyz), 0.0f, 1.0f))) / _5156.param3.y, 0.0f, 1.0f);
            }
        }
    }
    else
    {
        if (_5250 == 4u)
        {
            float param_3 = ray.pdf;
            float param_4 = (inter.t * inter.t) / (_5156.param1.w * dot(_5142, normalize(cross(_5156.param2.xyz, _5156.param3.xyz))));
            lcol *= power_heuristic(param_3, param_4);
        }
        else
        {
            if (_5250 == 5u)
            {
                float param_5 = ray.pdf;
                float param_6 = (inter.t * inter.t) / (_5156.param1.w * dot(_5142, normalize(cross(_5156.param2.xyz, _5156.param3.xyz))));
                lcol *= power_heuristic(param_5, param_6);
            }
            else
            {
                if (_5250 == 3u)
                {
                    float param_7 = ray.pdf;
                    float param_8 = (inter.t * inter.t) / (_5156.param1.w * (1.0f - abs(dot(_5142, _5156.param3.xyz))));
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
    uint _507 = uint(x);
    uint _514 = ((_507 >> uint(16)) ^ _507) * 73244475u;
    uint _519 = ((_514 >> uint(16)) ^ _514) * 73244475u;
    return int((_519 >> uint(16)) ^ _519);
}

float construct_float(inout uint m)
{
    m &= 8388607u;
    m |= 1065353216u;
    return asfloat(m) - 1.0f;
}

bool exchange(inout bool old_value, bool new_value)
{
    bool _2312 = old_value;
    old_value = new_value;
    return _2312;
}

float peek_ior_stack(float stack[4], inout bool skip_first, float default_value)
{
    float _9148;
    do
    {
        bool _2396 = stack[3] > 0.0f;
        bool _2405;
        if (_2396)
        {
            bool param = skip_first;
            bool param_1 = false;
            bool _2402 = exchange(param, param_1);
            skip_first = param;
            _2405 = !_2402;
        }
        else
        {
            _2405 = _2396;
        }
        if (_2405)
        {
            _9148 = stack[3];
            break;
        }
        bool _2413 = stack[2] > 0.0f;
        bool _2422;
        if (_2413)
        {
            bool param_2 = skip_first;
            bool param_3 = false;
            bool _2419 = exchange(param_2, param_3);
            skip_first = param_2;
            _2422 = !_2419;
        }
        else
        {
            _2422 = _2413;
        }
        if (_2422)
        {
            _9148 = stack[2];
            break;
        }
        bool _2430 = stack[1] > 0.0f;
        bool _2439;
        if (_2430)
        {
            bool param_4 = skip_first;
            bool param_5 = false;
            bool _2436 = exchange(param_4, param_5);
            skip_first = param_4;
            _2439 = !_2436;
        }
        else
        {
            _2439 = _2430;
        }
        if (_2439)
        {
            _9148 = stack[1];
            break;
        }
        bool _2447 = stack[0] > 0.0f;
        bool _2456;
        if (_2447)
        {
            bool param_6 = skip_first;
            bool param_7 = false;
            bool _2453 = exchange(param_6, param_7);
            skip_first = param_6;
            _2456 = !_2453;
        }
        else
        {
            _2456 = _2447;
        }
        if (_2456)
        {
            _9148 = stack[0];
            break;
        }
        _9148 = default_value;
        break;
    } while(false);
    return _9148;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _606 = mad(col.z, 31.875f, 1.0f);
    float _616 = (col.x - 0.501960813999176025390625f) / _606;
    float _622 = (col.y - 0.501960813999176025390625f) / _606;
    return float3((col.w + _616) - _622, col.w + _622, (col.w - _616) - _622);
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
    atlas_texture_t _1131;
    _1131.size = _1001.Load(index * 80 + 0);
    _1131.atlas = _1001.Load(index * 80 + 4);
    [unroll]
    for (int _62ident = 0; _62ident < 4; _62ident++)
    {
        _1131.page[_62ident] = _1001.Load(_62ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _63ident = 0; _63ident < 14; _63ident++)
    {
        _1131.pos[_63ident] = _1001.Load(_63ident * 4 + index * 80 + 24);
    }
    uint _9319[4];
    _9319[0] = _1131.page[0];
    _9319[1] = _1131.page[1];
    _9319[2] = _1131.page[2];
    _9319[3] = _1131.page[3];
    uint _9355[14] = { _1131.pos[0], _1131.pos[1], _1131.pos[2], _1131.pos[3], _1131.pos[4], _1131.pos[5], _1131.pos[6], _1131.pos[7], _1131.pos[8], _1131.pos[9], _1131.pos[10], _1131.pos[11], _1131.pos[12], _1131.pos[13] };
    atlas_texture_t _9325 = { _1131.size, _1131.atlas, _9319, _9355 };
    uint _1201 = _1131.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_1201)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_1201)], float3(TransformUV(uvs, _9325, lod) * 0.000118371215648949146270751953125f.xx, float((_9319[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
    bool _1216;
    if (maybe_YCoCg)
    {
        _1216 = _1131.atlas == 4u;
    }
    else
    {
        _1216 = maybe_YCoCg;
    }
    if (_1216)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _1235;
    if (maybe_SRGB)
    {
        _1235 = (_1131.size & 32768u) != 0u;
    }
    else
    {
        _1235 = maybe_SRGB;
    }
    if (_1235)
    {
        float3 param_1 = res.xyz;
        float3 _1241 = srgb_to_rgb(param_1);
        float4 _10498 = res;
        _10498.x = _1241.x;
        float4 _10500 = _10498;
        _10500.y = _1241.y;
        float4 _10502 = _10500;
        _10502.z = _1241.z;
        res = _10502;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod)
{
    return SampleBilinear(index, uvs, lod, false, false);
}

float fresnel_dielectric_cos(float cosi, float eta)
{
    float _1437 = abs(cosi);
    float _1446 = mad(_1437, _1437, mad(eta, eta, -1.0f));
    float g = _1446;
    float result;
    if (_1446 > 0.0f)
    {
        float _1451 = g;
        float _1452 = sqrt(_1451);
        g = _1452;
        float _1456 = _1452 - _1437;
        float _1459 = _1452 + _1437;
        float _1460 = _1456 / _1459;
        float _1474 = mad(_1437, _1459, -1.0f) / mad(_1437, _1456, 1.0f);
        result = ((0.5f * _1460) * _1460) * mad(_1474, _1474, 1.0f);
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
    float3 _9153;
    do
    {
        float _1510 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1510)
        {
            _9153 = N;
            break;
        }
        float3 _1530 = normalize(N - (Ng * dot(N, Ng)));
        float _1534 = dot(I, _1530);
        float _1538 = dot(I, Ng);
        float _1550 = mad(_1534, _1534, _1538 * _1538);
        float param = (_1534 * _1534) * mad(-_1510, _1510, _1550);
        float _1560 = safe_sqrtf(param);
        float _1566 = mad(_1538, _1510, _1550);
        float _1569 = 0.5f / _1550;
        float _1574 = _1560 + _1566;
        float _1575 = _1569 * _1574;
        float _1581 = (-_1560) + _1566;
        float _1582 = _1569 * _1581;
        bool _1590 = (_1575 > 9.9999997473787516355514526367188e-06f) && (_1575 <= 1.000010013580322265625f);
        bool valid1 = _1590;
        bool _1596 = (_1582 > 9.9999997473787516355514526367188e-06f) && (_1582 <= 1.000010013580322265625f);
        bool valid2 = _1596;
        float2 N_new;
        if (_1590 && _1596)
        {
            float _10801 = (-0.5f) / _1550;
            float param_1 = mad(_10801, _1574, 1.0f);
            float _1606 = safe_sqrtf(param_1);
            float param_2 = _1575;
            float _1609 = safe_sqrtf(param_2);
            float2 _1610 = float2(_1606, _1609);
            float param_3 = mad(_10801, _1581, 1.0f);
            float _1615 = safe_sqrtf(param_3);
            float param_4 = _1582;
            float _1618 = safe_sqrtf(param_4);
            float2 _1619 = float2(_1615, _1618);
            float _10803 = -_1538;
            float _1635 = mad(2.0f * mad(_1606, _1534, _1609 * _1538), _1609, _10803);
            float _1651 = mad(2.0f * mad(_1615, _1534, _1618 * _1538), _1618, _10803);
            bool _1653 = _1635 >= 9.9999997473787516355514526367188e-06f;
            valid1 = _1653;
            bool _1655 = _1651 >= 9.9999997473787516355514526367188e-06f;
            valid2 = _1655;
            if (_1653 && _1655)
            {
                bool2 _1668 = (_1635 < _1651).xx;
                N_new = float2(_1668.x ? _1610.x : _1619.x, _1668.y ? _1610.y : _1619.y);
            }
            else
            {
                bool2 _1676 = (_1635 > _1651).xx;
                N_new = float2(_1676.x ? _1610.x : _1619.x, _1676.y ? _1610.y : _1619.y);
            }
        }
        else
        {
            if (!(valid1 || valid2))
            {
                _9153 = Ng;
                break;
            }
            float _1688 = valid1 ? _1575 : _1582;
            float param_5 = 1.0f - _1688;
            float param_6 = _1688;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _9153 = (_1530 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _9153;
}

float3 rotate_around_axis(float3 p, float3 axis, float angle)
{
    float _1782 = cos(angle);
    float _1785 = sin(angle);
    float _1789 = 1.0f - _1782;
    return float3(mad(mad(_1789 * axis.x, axis.z, axis.y * _1785), p.z, mad(mad(_1789 * axis.x, axis.x, _1782), p.x, mad(_1789 * axis.x, axis.y, -(axis.z * _1785)) * p.y)), mad(mad(_1789 * axis.y, axis.z, -(axis.x * _1785)), p.z, mad(mad(_1789 * axis.x, axis.y, axis.z * _1785), p.x, mad(_1789 * axis.y, axis.y, _1782) * p.y)), mad(mad(_1789 * axis.z, axis.z, _1782), p.z, mad(mad(_1789 * axis.x, axis.z, -(axis.y * _1785)), p.x, mad(_1789 * axis.y, axis.z, axis.x * _1785) * p.y)));
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
    int3 _1938 = int3(n * 128.0f);
    int _1946;
    if (p.x < 0.0f)
    {
        _1946 = -_1938.x;
    }
    else
    {
        _1946 = _1938.x;
    }
    int _1964;
    if (p.y < 0.0f)
    {
        _1964 = -_1938.y;
    }
    else
    {
        _1964 = _1938.y;
    }
    int _1982;
    if (p.z < 0.0f)
    {
        _1982 = -_1938.z;
    }
    else
    {
        _1982 = _1938.z;
    }
    float _2000;
    if (abs(p.x) < 0.03125f)
    {
        _2000 = mad(1.52587890625e-05f, n.x, p.x);
    }
    else
    {
        _2000 = asfloat(asint(p.x) + _1946);
    }
    float _2018;
    if (abs(p.y) < 0.03125f)
    {
        _2018 = mad(1.52587890625e-05f, n.y, p.y);
    }
    else
    {
        _2018 = asfloat(asint(p.y) + _1964);
    }
    float _2035;
    if (abs(p.z) < 0.03125f)
    {
        _2035 = mad(1.52587890625e-05f, n.z, p.z);
    }
    else
    {
        _2035 = asfloat(asint(p.z) + _1982);
    }
    return float3(_2000, _2018, _2035);
}

float3 MapToCone(float r1, float r2, float3 N, float radius)
{
    float3 _9178;
    do
    {
        float2 _3436 = (float2(r1, r2) * 2.0f) - 1.0f.xx;
        float _3438 = _3436.x;
        bool _3439 = _3438 == 0.0f;
        bool _3445;
        if (_3439)
        {
            _3445 = _3436.y == 0.0f;
        }
        else
        {
            _3445 = _3439;
        }
        if (_3445)
        {
            _9178 = N;
            break;
        }
        float _3454 = _3436.y;
        float r;
        float theta;
        if (abs(_3438) > abs(_3454))
        {
            r = _3438;
            theta = 0.785398185253143310546875f * (_3454 / _3438);
        }
        else
        {
            r = _3454;
            theta = 1.57079637050628662109375f * mad(-0.5f, _3438 / _3454, 1.0f);
        }
        float3 param;
        float3 param_1;
        create_tbn(N, param, param_1);
        _9178 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _9178;
}

float3 CanonicalToDir(float2 p, float y_rotation)
{
    float _680 = mad(2.0f, p.x, -1.0f);
    float _685 = mad(6.283185482025146484375f, p.y, y_rotation);
    float phi = _685;
    if (_685 < 0.0f)
    {
        phi += 6.283185482025146484375f;
    }
    if (phi > 6.283185482025146484375f)
    {
        phi -= 6.283185482025146484375f;
    }
    float _703 = sqrt(mad(-_680, _680, 1.0f));
    return float3(_703 * cos(phi), _680, (-_703) * sin(phi));
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
        float _878 = quad.x + quad.z;
        float partial = _878;
        float _885 = (_878 + quad.y) + quad.w;
        if (_885 <= 0.0f)
        {
            break;
        }
        float _894 = partial / _885;
        float boundary = _894;
        int index = 0;
        if (_sample < _894)
        {
            _sample /= boundary;
            boundary = quad.x / partial;
        }
        else
        {
            float _909 = partial;
            float _910 = _885 - _909;
            partial = _910;
            float2 _10485 = origin;
            _10485.x = origin.x + _step;
            origin = _10485;
            _sample = (_sample - boundary) / (1.0f - boundary);
            boundary = quad.y / _910;
            index |= 1;
        }
        if (_sample < boundary)
        {
            _sample /= boundary;
        }
        else
        {
            float2 _10488 = origin;
            _10488.y = origin.y + _step;
            origin = _10488;
            _sample = (_sample - boundary) / (1.0f - boundary);
            index |= 2;
        }
        factor *= ((4.0f * quad[index]) / _885);
        lod--;
        res *= 2;
        _step *= 0.5f;
    }
    float2 _967 = origin;
    float2 _968 = _967 + (float2(rx, ry) * (2.0f * _step));
    origin = _968;
    return float4(CanonicalToDir(_968, y_rotation), factor * 0.079577468335628509521484375f);
}

float3 world_from_tangent(float3 T, float3 B, float3 N, float3 V)
{
    return ((T * V.x) + (B * V.y)) + (N * V.z);
}

void SampleLightSource(float3 P, float3 T, float3 B, float3 N, int hi, float2 sample_off, inout light_sample_t ls)
{
    float _3530 = frac(asfloat(_3521.Load((hi + 3) * 4 + 0)) + sample_off.x);
    float _3541 = float(_3537_g_params.li_count);
    uint _3548 = min(uint(_3530 * _3541), uint(_3537_g_params.li_count - 1));
    light_t _3568;
    _3568.type_and_param0 = _3557.Load4(_3561.Load(_3548 * 4 + 0) * 64 + 0);
    _3568.param1 = asfloat(_3557.Load4(_3561.Load(_3548 * 4 + 0) * 64 + 16));
    _3568.param2 = asfloat(_3557.Load4(_3561.Load(_3548 * 4 + 0) * 64 + 32));
    _3568.param3 = asfloat(_3557.Load4(_3561.Load(_3548 * 4 + 0) * 64 + 48));
    ls.col = asfloat(_3568.type_and_param0.yzw);
    ls.col *= _3541;
    ls.cast_shadow = (_3568.type_and_param0.x & 32u) != 0u;
    ls.from_env = false;
    uint _3602 = _3568.type_and_param0.x & 31u;
    [branch]
    if (_3602 == 0u)
    {
        float _3615 = frac(asfloat(_3521.Load((hi + 4) * 4 + 0)) + sample_off.x);
        float3 _3630 = P - _3568.param1.xyz;
        float3 _3637 = _3630 / length(_3630).xxx;
        float _3644 = sqrt(clamp(mad(-_3615, _3615, 1.0f), 0.0f, 1.0f));
        float _3647 = 6.283185482025146484375f * frac(asfloat(_3521.Load((hi + 5) * 4 + 0)) + sample_off.y);
        float3 sampled_dir = float3(_3644 * cos(_3647), _3644 * sin(_3647), _3615);
        float3 param;
        float3 param_1;
        create_tbn(_3637, param, param_1);
        float3 _10565 = sampled_dir;
        float3 _3680 = ((param * _10565.x) + (param_1 * _10565.y)) + (_3637 * _10565.z);
        sampled_dir = _3680;
        float3 _3689 = _3568.param1.xyz + (_3680 * _3568.param2.w);
        float3 _3696 = normalize(_3689 - _3568.param1.xyz);
        float3 param_2 = _3689;
        float3 param_3 = _3696;
        ls.lp = offset_ray(param_2, param_3);
        ls.L = _3689 - P;
        float3 _3709 = ls.L;
        float _3710 = length(_3709);
        ls.L /= _3710.xxx;
        ls.area = _3568.param1.w;
        float _3725 = abs(dot(ls.L, _3696));
        [flatten]
        if (_3725 > 0.0f)
        {
            ls.pdf = (_3710 * _3710) / ((0.5f * ls.area) * _3725);
        }
        [branch]
        if (_3568.param3.x > 0.0f)
        {
            float _3752 = -dot(ls.L, _3568.param2.xyz);
            if (_3752 > 0.0f)
            {
                ls.col *= clamp((_3568.param3.x - acos(clamp(_3752, 0.0f, 1.0f))) / _3568.param3.y, 0.0f, 1.0f);
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
        if (_3602 == 2u)
        {
            ls.L = _3568.param1.xyz;
            if (_3568.param1.w != 0.0f)
            {
                float param_4 = frac(asfloat(_3521.Load((hi + 4) * 4 + 0)) + sample_off.x);
                float param_5 = frac(asfloat(_3521.Load((hi + 5) * 4 + 0)) + sample_off.y);
                float3 param_6 = ls.L;
                float param_7 = tan(_3568.param1.w);
                ls.L = normalize(MapToCone(param_4, param_5, param_6, param_7));
            }
            ls.area = 0.0f;
            ls.lp = P + ls.L;
            ls.dist_mul = 3402823346297367662189621542912.0f;
            ls.pdf = 1.0f;
            if ((_3568.type_and_param0.x & 64u) == 0u)
            {
                ls.area = 0.0f;
            }
        }
        else
        {
            [branch]
            if (_3602 == 4u)
            {
                float3 _3889 = (_3568.param1.xyz + (_3568.param2.xyz * (frac(asfloat(_3521.Load((hi + 4) * 4 + 0)) + sample_off.x) - 0.5f))) + (_3568.param3.xyz * (frac(asfloat(_3521.Load((hi + 5) * 4 + 0)) + sample_off.y) - 0.5f));
                float3 _3894 = normalize(cross(_3568.param2.xyz, _3568.param3.xyz));
                float3 param_8 = _3889;
                float3 param_9 = _3894;
                ls.lp = offset_ray(param_8, param_9);
                ls.L = _3889 - P;
                float3 _3907 = ls.L;
                float _3908 = length(_3907);
                ls.L /= _3908.xxx;
                ls.area = _3568.param1.w;
                float _3923 = dot(-ls.L, _3894);
                if (_3923 > 0.0f)
                {
                    ls.pdf = (_3908 * _3908) / (ls.area * _3923);
                }
                if ((_3568.type_and_param0.x & 64u) == 0u)
                {
                    ls.area = 0.0f;
                }
                [branch]
                if ((_3568.type_and_param0.x & 128u) != 0u)
                {
                    float3 env_col = _3537_g_params.env_col.xyz;
                    uint _3960 = asuint(_3537_g_params.env_col.w);
                    if (_3960 != 4294967295u)
                    {
                        atlas_texture_t _3968;
                        _3968.size = _1001.Load(_3960 * 80 + 0);
                        _3968.atlas = _1001.Load(_3960 * 80 + 4);
                        [unroll]
                        for (int _64ident = 0; _64ident < 4; _64ident++)
                        {
                            _3968.page[_64ident] = _1001.Load(_64ident * 4 + _3960 * 80 + 8);
                        }
                        [unroll]
                        for (int _65ident = 0; _65ident < 14; _65ident++)
                        {
                            _3968.pos[_65ident] = _1001.Load(_65ident * 4 + _3960 * 80 + 24);
                        }
                        uint _9499[14] = { _3968.pos[0], _3968.pos[1], _3968.pos[2], _3968.pos[3], _3968.pos[4], _3968.pos[5], _3968.pos[6], _3968.pos[7], _3968.pos[8], _3968.pos[9], _3968.pos[10], _3968.pos[11], _3968.pos[12], _3968.pos[13] };
                        uint _9470[4] = { _3968.page[0], _3968.page[1], _3968.page[2], _3968.page[3] };
                        atlas_texture_t _9399 = { _3968.size, _3968.atlas, _9470, _9499 };
                        float param_10 = _3537_g_params.env_rotation;
                        env_col *= SampleLatlong_RGBE(_9399, ls.L, param_10);
                    }
                    ls.col *= env_col;
                    ls.from_env = true;
                }
            }
            else
            {
                [branch]
                if (_3602 == 5u)
                {
                    float2 _4071 = (float2(frac(asfloat(_3521.Load((hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_3521.Load((hi + 5) * 4 + 0)) + sample_off.y)) * 2.0f) - 1.0f.xx;
                    float2 offset = _4071;
                    bool _4074 = _4071.x != 0.0f;
                    bool _4080;
                    if (_4074)
                    {
                        _4080 = offset.y != 0.0f;
                    }
                    else
                    {
                        _4080 = _4074;
                    }
                    if (_4080)
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
                        float _4113 = 0.5f * r;
                        offset = float2(_4113 * cos(theta), _4113 * sin(theta));
                    }
                    float3 _4135 = (_3568.param1.xyz + (_3568.param2.xyz * offset.x)) + (_3568.param3.xyz * offset.y);
                    float3 _4140 = normalize(cross(_3568.param2.xyz, _3568.param3.xyz));
                    float3 param_11 = _4135;
                    float3 param_12 = _4140;
                    ls.lp = offset_ray(param_11, param_12);
                    ls.L = _4135 - P;
                    float3 _4153 = ls.L;
                    float _4154 = length(_4153);
                    ls.L /= _4154.xxx;
                    ls.area = _3568.param1.w;
                    float _4169 = dot(-ls.L, _4140);
                    [flatten]
                    if (_4169 > 0.0f)
                    {
                        ls.pdf = (_4154 * _4154) / (ls.area * _4169);
                    }
                    if ((_3568.type_and_param0.x & 64u) == 0u)
                    {
                        ls.area = 0.0f;
                    }
                    [branch]
                    if ((_3568.type_and_param0.x & 128u) != 0u)
                    {
                        float3 env_col_1 = _3537_g_params.env_col.xyz;
                        uint _4203 = asuint(_3537_g_params.env_col.w);
                        if (_4203 != 4294967295u)
                        {
                            atlas_texture_t _4210;
                            _4210.size = _1001.Load(_4203 * 80 + 0);
                            _4210.atlas = _1001.Load(_4203 * 80 + 4);
                            [unroll]
                            for (int _66ident = 0; _66ident < 4; _66ident++)
                            {
                                _4210.page[_66ident] = _1001.Load(_66ident * 4 + _4203 * 80 + 8);
                            }
                            [unroll]
                            for (int _67ident = 0; _67ident < 14; _67ident++)
                            {
                                _4210.pos[_67ident] = _1001.Load(_67ident * 4 + _4203 * 80 + 24);
                            }
                            uint _9537[14] = { _4210.pos[0], _4210.pos[1], _4210.pos[2], _4210.pos[3], _4210.pos[4], _4210.pos[5], _4210.pos[6], _4210.pos[7], _4210.pos[8], _4210.pos[9], _4210.pos[10], _4210.pos[11], _4210.pos[12], _4210.pos[13] };
                            uint _9508[4] = { _4210.page[0], _4210.page[1], _4210.page[2], _4210.page[3] };
                            atlas_texture_t _9408 = { _4210.size, _4210.atlas, _9508, _9537 };
                            float param_13 = _3537_g_params.env_rotation;
                            env_col_1 *= SampleLatlong_RGBE(_9408, ls.L, param_13);
                        }
                        ls.col *= env_col_1;
                        ls.from_env = true;
                    }
                }
                else
                {
                    [branch]
                    if (_3602 == 3u)
                    {
                        float3 _4310 = normalize(cross(P - _3568.param1.xyz, _3568.param3.xyz));
                        float _4317 = 3.1415927410125732421875f * frac(asfloat(_3521.Load((hi + 4) * 4 + 0)) + sample_off.x);
                        float3 _4342 = (_3568.param1.xyz + (((_4310 * cos(_4317)) + (cross(_4310, _3568.param3.xyz) * sin(_4317))) * _3568.param2.w)) + ((_3568.param3.xyz * (frac(asfloat(_3521.Load((hi + 5) * 4 + 0)) + sample_off.y) - 0.5f)) * _3568.param3.w);
                        ls.lp = _4342;
                        float3 _4348 = _4342 - P;
                        float _4351 = length(_4348);
                        ls.L = _4348 / _4351.xxx;
                        ls.area = _3568.param1.w;
                        float _4366 = 1.0f - abs(dot(ls.L, _3568.param3.xyz));
                        [flatten]
                        if (_4366 != 0.0f)
                        {
                            ls.pdf = (_4351 * _4351) / (ls.area * _4366);
                        }
                        if ((_3568.type_and_param0.x & 64u) == 0u)
                        {
                            ls.area = 0.0f;
                        }
                    }
                    else
                    {
                        [branch]
                        if (_3602 == 6u)
                        {
                            uint _4396 = asuint(_3568.param1.x);
                            transform_t _4410;
                            _4410.xform = asfloat(uint4x4(_4404.Load4(asuint(_3568.param1.y) * 128 + 0), _4404.Load4(asuint(_3568.param1.y) * 128 + 16), _4404.Load4(asuint(_3568.param1.y) * 128 + 32), _4404.Load4(asuint(_3568.param1.y) * 128 + 48)));
                            _4410.inv_xform = asfloat(uint4x4(_4404.Load4(asuint(_3568.param1.y) * 128 + 64), _4404.Load4(asuint(_3568.param1.y) * 128 + 80), _4404.Load4(asuint(_3568.param1.y) * 128 + 96), _4404.Load4(asuint(_3568.param1.y) * 128 + 112)));
                            uint _4435 = _4396 * 3u;
                            vertex_t _4441;
                            [unroll]
                            for (int _68ident = 0; _68ident < 3; _68ident++)
                            {
                                _4441.p[_68ident] = asfloat(_4429.Load(_68ident * 4 + _4433.Load(_4435 * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _69ident = 0; _69ident < 3; _69ident++)
                            {
                                _4441.n[_69ident] = asfloat(_4429.Load(_69ident * 4 + _4433.Load(_4435 * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _70ident = 0; _70ident < 3; _70ident++)
                            {
                                _4441.b[_70ident] = asfloat(_4429.Load(_70ident * 4 + _4433.Load(_4435 * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _71ident = 0; _71ident < 2; _71ident++)
                            {
                                [unroll]
                                for (int _72ident = 0; _72ident < 2; _72ident++)
                                {
                                    _4441.t[_71ident][_72ident] = asfloat(_4429.Load(_72ident * 4 + _71ident * 8 + _4433.Load(_4435 * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4490;
                            [unroll]
                            for (int _73ident = 0; _73ident < 3; _73ident++)
                            {
                                _4490.p[_73ident] = asfloat(_4429.Load(_73ident * 4 + _4433.Load((_4435 + 1u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _74ident = 0; _74ident < 3; _74ident++)
                            {
                                _4490.n[_74ident] = asfloat(_4429.Load(_74ident * 4 + _4433.Load((_4435 + 1u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _75ident = 0; _75ident < 3; _75ident++)
                            {
                                _4490.b[_75ident] = asfloat(_4429.Load(_75ident * 4 + _4433.Load((_4435 + 1u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _76ident = 0; _76ident < 2; _76ident++)
                            {
                                [unroll]
                                for (int _77ident = 0; _77ident < 2; _77ident++)
                                {
                                    _4490.t[_76ident][_77ident] = asfloat(_4429.Load(_77ident * 4 + _76ident * 8 + _4433.Load((_4435 + 1u) * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4536;
                            [unroll]
                            for (int _78ident = 0; _78ident < 3; _78ident++)
                            {
                                _4536.p[_78ident] = asfloat(_4429.Load(_78ident * 4 + _4433.Load((_4435 + 2u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _79ident = 0; _79ident < 3; _79ident++)
                            {
                                _4536.n[_79ident] = asfloat(_4429.Load(_79ident * 4 + _4433.Load((_4435 + 2u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _80ident = 0; _80ident < 3; _80ident++)
                            {
                                _4536.b[_80ident] = asfloat(_4429.Load(_80ident * 4 + _4433.Load((_4435 + 2u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _81ident = 0; _81ident < 2; _81ident++)
                            {
                                [unroll]
                                for (int _82ident = 0; _82ident < 2; _82ident++)
                                {
                                    _4536.t[_81ident][_82ident] = asfloat(_4429.Load(_82ident * 4 + _81ident * 8 + _4433.Load((_4435 + 2u) * 4 + 0) * 52 + 36));
                                }
                            }
                            float3 _4582 = float3(_4441.p[0], _4441.p[1], _4441.p[2]);
                            float3 _4590 = float3(_4490.p[0], _4490.p[1], _4490.p[2]);
                            float3 _4598 = float3(_4536.p[0], _4536.p[1], _4536.p[2]);
                            float _4626 = sqrt(frac(asfloat(_3521.Load((hi + 4) * 4 + 0)) + sample_off.x));
                            float _4635 = frac(asfloat(_3521.Load((hi + 5) * 4 + 0)) + sample_off.y);
                            float _4639 = 1.0f - _4626;
                            float _4644 = 1.0f - _4635;
                            float3 _4675 = mul(float4((_4582 * _4639) + (((_4590 * _4644) + (_4598 * _4635)) * _4626), 1.0f), _4410.xform).xyz;
                            float3 _4691 = mul(float4(cross(_4590 - _4582, _4598 - _4582), 0.0f), _4410.xform).xyz;
                            ls.area = 0.5f * length(_4691);
                            float3 _4697 = normalize(_4691);
                            ls.L = _4675 - P;
                            float3 _4704 = ls.L;
                            float _4705 = length(_4704);
                            ls.L /= _4705.xxx;
                            float _4716 = dot(ls.L, _4697);
                            float cos_theta = _4716;
                            float3 _4719;
                            if (_4716 >= 0.0f)
                            {
                                _4719 = -_4697;
                            }
                            else
                            {
                                _4719 = _4697;
                            }
                            float3 param_14 = _4675;
                            float3 param_15 = _4719;
                            ls.lp = offset_ray(param_14, param_15);
                            float _4732 = cos_theta;
                            float _4733 = abs(_4732);
                            cos_theta = _4733;
                            [flatten]
                            if (_4733 > 0.0f)
                            {
                                ls.pdf = (_4705 * _4705) / (ls.area * cos_theta);
                            }
                            material_t _4770;
                            [unroll]
                            for (int _83ident = 0; _83ident < 5; _83ident++)
                            {
                                _4770.textures[_83ident] = _4757.Load(_83ident * 4 + ((_4761.Load(_4396 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
                            }
                            [unroll]
                            for (int _84ident = 0; _84ident < 3; _84ident++)
                            {
                                _4770.base_color[_84ident] = asfloat(_4757.Load(_84ident * 4 + ((_4761.Load(_4396 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
                            }
                            _4770.flags = _4757.Load(((_4761.Load(_4396 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
                            _4770.type = _4757.Load(((_4761.Load(_4396 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
                            _4770.tangent_rotation_or_strength = asfloat(_4757.Load(((_4761.Load(_4396 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
                            _4770.roughness_and_anisotropic = _4757.Load(((_4761.Load(_4396 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
                            _4770.ior = asfloat(_4757.Load(((_4761.Load(_4396 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
                            _4770.sheen_and_sheen_tint = _4757.Load(((_4761.Load(_4396 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
                            _4770.tint_and_metallic = _4757.Load(((_4761.Load(_4396 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
                            _4770.transmission_and_transmission_roughness = _4757.Load(((_4761.Load(_4396 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
                            _4770.specular_and_specular_tint = _4757.Load(((_4761.Load(_4396 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
                            _4770.clearcoat_and_clearcoat_roughness = _4757.Load(((_4761.Load(_4396 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
                            _4770.normal_map_strength_unorm = _4757.Load(((_4761.Load(_4396 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
                            if (_4770.textures[1] != 4294967295u)
                            {
                                ls.col *= SampleBilinear(_4770.textures[1], (float2(_4441.t[0][0], _4441.t[0][1]) * _4639) + (((float2(_4490.t[0][0], _4490.t[0][1]) * _4644) + (float2(_4536.t[0][0], _4536.t[0][1]) * _4635)) * _4626), 0).xyz;
                            }
                        }
                        else
                        {
                            [branch]
                            if (_3602 == 7u)
                            {
                                float _4850 = frac(asfloat(_3521.Load((hi + 4) * 4 + 0)) + sample_off.x);
                                float _4859 = frac(asfloat(_3521.Load((hi + 5) * 4 + 0)) + sample_off.y);
                                float4 dir_and_pdf;
                                if (_3537_g_params.env_qtree_levels > 0)
                                {
                                    dir_and_pdf = Sample_EnvQTree(_3537_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3537_g_params.env_qtree_levels, mad(_3530, _3541, -float(_3548)), _4850, _4859);
                                }
                                else
                                {
                                    float _4878 = 6.283185482025146484375f * _4859;
                                    float _4890 = sqrt(mad(-_4850, _4850, 1.0f));
                                    float3 param_16 = T;
                                    float3 param_17 = B;
                                    float3 param_18 = N;
                                    float3 param_19 = float3(_4890 * cos(_4878), _4890 * sin(_4878), _4850);
                                    dir_and_pdf = float4(world_from_tangent(param_16, param_17, param_18, param_19), 0.15915493667125701904296875f);
                                }
                                ls.L = dir_and_pdf.xyz;
                                ls.col *= _3537_g_params.env_col.xyz;
                                uint _4929 = asuint(_3537_g_params.env_col.w);
                                if (_4929 != 4294967295u)
                                {
                                    atlas_texture_t _4936;
                                    _4936.size = _1001.Load(_4929 * 80 + 0);
                                    _4936.atlas = _1001.Load(_4929 * 80 + 4);
                                    [unroll]
                                    for (int _85ident = 0; _85ident < 4; _85ident++)
                                    {
                                        _4936.page[_85ident] = _1001.Load(_85ident * 4 + _4929 * 80 + 8);
                                    }
                                    [unroll]
                                    for (int _86ident = 0; _86ident < 14; _86ident++)
                                    {
                                        _4936.pos[_86ident] = _1001.Load(_86ident * 4 + _4929 * 80 + 24);
                                    }
                                    uint _9622[14] = { _4936.pos[0], _4936.pos[1], _4936.pos[2], _4936.pos[3], _4936.pos[4], _4936.pos[5], _4936.pos[6], _4936.pos[7], _4936.pos[8], _4936.pos[9], _4936.pos[10], _4936.pos[11], _4936.pos[12], _4936.pos[13] };
                                    uint _9593[4] = { _4936.page[0], _4936.page[1], _4936.page[2], _4936.page[3] };
                                    atlas_texture_t _9461 = { _4936.size, _4936.atlas, _9593, _9622 };
                                    float param_20 = _3537_g_params.env_rotation;
                                    ls.col *= SampleLatlong_RGBE(_9461, ls.L, param_20);
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
    atlas_texture_t _1004;
    _1004.size = _1001.Load(index * 80 + 0);
    _1004.atlas = _1001.Load(index * 80 + 4);
    [unroll]
    for (int _87ident = 0; _87ident < 4; _87ident++)
    {
        _1004.page[_87ident] = _1001.Load(_87ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _88ident = 0; _88ident < 14; _88ident++)
    {
        _1004.pos[_88ident] = _1001.Load(_88ident * 4 + index * 80 + 24);
    }
    return int2(int(_1004.size & 16383u), int((_1004.size >> uint(16)) & 16383u));
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
    float _2516 = 1.0f / mad(0.904129683971405029296875f, roughness, 3.1415927410125732421875f);
    float _2528 = max(dot(N, L), 0.0f);
    float _2533 = max(dot(N, V), 0.0f);
    float _2541 = mad(-_2528, _2533, dot(L, V));
    float t = _2541;
    if (_2541 > 0.0f)
    {
        t /= (max(_2528, _2533) + 1.1754943508222875079687365372222e-38f);
    }
    return float4(base_color * (_2528 * mad(roughness * _2516, t, _2516)), 0.15915493667125701904296875f);
}

float3 Evaluate_DiffuseNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _9158;
    do
    {
        float3 param = -float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param_1 = surf.N;
        float3 param_2 = ls.L;
        float param_3 = roughness;
        float3 param_4 = base_color;
        float4 _5520 = Evaluate_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_5 = ls.pdf;
            float param_6 = _5520.w;
            mis_weight = power_heuristic(param_5, param_6);
        }
        float3 _5543 = (ls.col * _5520.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9158 = _5543;
            break;
        }
        float3 param_7 = surf.P;
        float3 param_8 = surf.plane_N;
        float3 _5555 = offset_ray(param_7, param_8);
        sh_r.o[0] = _5555.x;
        sh_r.o[1] = _5555.y;
        sh_r.o[2] = _5555.z;
        sh_r.c[0] = ray.c[0] * _5543.x;
        sh_r.c[1] = ray.c[1] * _5543.y;
        sh_r.c[2] = ray.c[2] * _5543.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9158 = 0.0f.xxx;
        break;
    } while(false);
    return _9158;
}

float4 Sample_OrenDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float rand_u, float rand_v, inout float3 out_V)
{
    float _2575 = 6.283185482025146484375f * rand_v;
    float _2587 = sqrt(mad(-rand_u, rand_u, 1.0f));
    float3 param = T;
    float3 param_1 = B;
    float3 param_2 = N;
    float3 param_3 = float3(_2587 * cos(_2575), _2587 * sin(_2575), rand_u);
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
    float4 _5806 = Sample_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8);
    new_ray.depth = ray.depth + 1;
    float3 param_9 = surf.P;
    float3 param_10 = surf.plane_N;
    float3 _5816 = offset_ray(param_9, param_10);
    new_ray.o[0] = _5816.x;
    new_ray.o[1] = _5816.y;
    new_ray.o[2] = _5816.z;
    new_ray.d[0] = param_8.x;
    new_ray.d[1] = param_8.y;
    new_ray.d[2] = param_8.z;
    new_ray.c[0] = ((ray.c[0] * _5806.x) * mix_weight) / _5806.w;
    new_ray.c[1] = ((ray.c[1] * _5806.y) * mix_weight) / _5806.w;
    new_ray.c[2] = ((ray.c[2] * _5806.z) * mix_weight) / _5806.w;
    new_ray.pdf = _5806.w;
}

float3 tangent_from_world(float3 T, float3 B, float3 N, float3 V)
{
    return float3(dot(V, T), dot(V, B), dot(V, N));
}

float D_GGX(float3 H, float alpha_x, float alpha_y)
{
    float _9211;
    do
    {
        if (H.z == 0.0f)
        {
            _9211 = 0.0f;
            break;
        }
        float _2242 = (-H.x) / (H.z * alpha_x);
        float _2248 = (-H.y) / (H.z * alpha_y);
        float _2257 = mad(_2248, _2248, mad(_2242, _2242, 1.0f));
        _9211 = 1.0f / (((((_2257 * _2257) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _9211;
}

float G1(float3 Ve, inout float alpha_x, inout float alpha_y)
{
    alpha_x *= alpha_x;
    alpha_y *= alpha_y;
    return 1.0f / mad((-1.0f) + sqrt(1.0f + (mad(alpha_x * Ve.x, Ve.x, (alpha_y * Ve.y) * Ve.y) / (Ve.z * Ve.z))), 0.5f, 1.0f);
}

float4 Evaluate_GGXSpecular_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float alpha_x, float alpha_y, float spec_ior, float spec_F0, float3 spec_col)
{
    float _2757 = D_GGX(sampled_normal_ts, alpha_x, alpha_y);
    float3 param = view_dir_ts;
    float param_1 = alpha_x;
    float param_2 = alpha_y;
    float _2765 = G1(param, param_1, param_2);
    float3 param_3 = reflected_dir_ts;
    float param_4 = alpha_x;
    float param_5 = alpha_y;
    float _2772 = G1(param_3, param_4, param_5);
    float param_6 = dot(view_dir_ts, sampled_normal_ts);
    float param_7 = spec_ior;
    float3 F = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param_6, param_7) - spec_F0) / (1.0f - spec_F0)).xxx);
    float _2800 = 4.0f * abs(view_dir_ts.z * reflected_dir_ts.z);
    float _2803;
    if (_2800 != 0.0f)
    {
        _2803 = (_2757 * (_2765 * _2772)) / _2800;
    }
    else
    {
        _2803 = 0.0f;
    }
    F *= _2803;
    float3 param_8 = view_dir_ts;
    float param_9 = alpha_x;
    float param_10 = alpha_y;
    float _2823 = G1(param_8, param_9, param_10);
    float pdf = ((_2757 * _2823) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2838 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2838 != 0.0f)
    {
        pdf /= _2838;
    }
    float3 _2849 = F;
    float3 _2850 = _2849 * max(reflected_dir_ts.z, 0.0f);
    F = _2850;
    return float4(_2850, pdf);
}

float3 Evaluate_GlossyNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _9163;
    do
    {
        float3 _5591 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5591;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - _5591);
        float _5629 = roughness * roughness;
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = _5629;
        float param_16 = _5629;
        float param_17 = spec_ior;
        float param_18 = spec_F0;
        float3 param_19 = base_color;
        float4 _5644 = Evaluate_GGXSpecular_BSDF(param_12, param_13, param_14, param_15, param_16, param_17, param_18, param_19);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_20 = ls.pdf;
            float param_21 = _5644.w;
            mis_weight = power_heuristic(param_20, param_21);
        }
        float3 _5667 = (ls.col * _5644.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9163 = _5667;
            break;
        }
        float3 param_22 = surf.P;
        float3 param_23 = surf.plane_N;
        float3 _5679 = offset_ray(param_22, param_23);
        sh_r.o[0] = _5679.x;
        sh_r.o[1] = _5679.y;
        sh_r.o[2] = _5679.z;
        sh_r.c[0] = ray.c[0] * _5667.x;
        sh_r.c[1] = ray.c[1] * _5667.y;
        sh_r.c[2] = ray.c[2] * _5667.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9163 = 0.0f.xxx;
        break;
    } while(false);
    return _9163;
}

float3 SampleGGX_VNDF(float3 Ve, float alpha_x, float alpha_y, float U1, float U2)
{
    float3 _2060 = normalize(float3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
    float _2063 = _2060.x;
    float _2068 = _2060.y;
    float _2072 = mad(_2063, _2063, _2068 * _2068);
    float3 _2076;
    if (_2072 > 0.0f)
    {
        _2076 = float3(-_2068, _2063, 0.0f) / sqrt(_2072).xxx;
    }
    else
    {
        _2076 = float3(1.0f, 0.0f, 0.0f);
    }
    float _2098 = sqrt(U1);
    float _2101 = 6.283185482025146484375f * U2;
    float _2106 = _2098 * cos(_2101);
    float _2115 = 1.0f + _2060.z;
    float _2122 = mad(-_2106, _2106, 1.0f);
    float _2128 = mad(mad(-0.5f, _2115, 1.0f), sqrt(_2122), (0.5f * _2115) * (_2098 * sin(_2101)));
    float3 _2149 = ((_2076 * _2106) + (cross(_2060, _2076) * _2128)) + (_2060 * sqrt(max(0.0f, mad(-_2128, _2128, _2122))));
    return normalize(float3(alpha_x * _2149.x, alpha_y * _2149.y, max(0.0f, _2149.z)));
}

float4 Sample_GGXSpecular_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float anisotropic, float spec_ior, float spec_F0, float3 spec_col, float rand_u, float rand_v, inout float3 out_V)
{
    float4 _9183;
    do
    {
        float _2860 = roughness * roughness;
        float _2864 = sqrt(mad(-0.89999997615814208984375f, anisotropic, 1.0f));
        float _2868 = _2860 / _2864;
        float _2872 = _2860 * _2864;
        [branch]
        if ((_2868 * _2872) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2883 = reflect(I, N);
            float param = dot(_2883, N);
            float param_1 = spec_ior;
            float3 _2897 = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param, param_1) - spec_F0) / (1.0f - spec_F0)).xxx);
            out_V = _2883;
            _9183 = float4(_2897.x * 1000000.0f, _2897.y * 1000000.0f, _2897.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2922 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = _2868;
        float param_7 = _2872;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2931 = SampleGGX_VNDF(_2922, param_6, param_7, param_8, param_9);
        float3 _2942 = normalize(reflect(-_2922, _2931));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2942;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2922;
        float3 param_15 = _2931;
        float3 param_16 = _2942;
        float param_17 = _2868;
        float param_18 = _2872;
        float param_19 = spec_ior;
        float param_20 = spec_F0;
        float3 param_21 = spec_col;
        _9183 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _9183;
}

void Sample_GlossyNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 param_4;
    float4 _5726 = Sample_GGXSpecular_BSDF(param, param_1, param_2, param_3, roughness, 0.0f, spec_ior, spec_F0, base_color, rand_u, rand_v, param_4);
    new_ray.depth = ray.depth + 256;
    float3 param_5 = surf.P;
    float3 param_6 = surf.plane_N;
    float3 _5737 = offset_ray(param_5, param_6);
    new_ray.o[0] = _5737.x;
    new_ray.o[1] = _5737.y;
    new_ray.o[2] = _5737.z;
    new_ray.d[0] = param_4.x;
    new_ray.d[1] = param_4.y;
    new_ray.d[2] = param_4.z;
    new_ray.c[0] = ((ray.c[0] * _5726.x) * mix_weight) / _5726.w;
    new_ray.c[1] = ((ray.c[1] * _5726.y) * mix_weight) / _5726.w;
    new_ray.c[2] = ((ray.c[2] * _5726.z) * mix_weight) / _5726.w;
    new_ray.pdf = _5726.w;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _9188;
    do
    {
        bool _3164 = refr_dir_ts.z >= 0.0f;
        bool _3171;
        if (!_3164)
        {
            _3171 = view_dir_ts.z <= 0.0f;
        }
        else
        {
            _3171 = _3164;
        }
        if (_3171)
        {
            _9188 = 0.0f.xxxx;
            break;
        }
        float _3180 = D_GGX(sampled_normal_ts, roughness2, roughness2);
        float3 param = refr_dir_ts;
        float param_1 = roughness2;
        float param_2 = roughness2;
        float _3188 = G1(param, param_1, param_2);
        float3 param_3 = view_dir_ts;
        float param_4 = roughness2;
        float param_5 = roughness2;
        float _3196 = G1(param_3, param_4, param_5);
        float _3206 = mad(dot(view_dir_ts, sampled_normal_ts), eta, dot(refr_dir_ts, sampled_normal_ts));
        float _3216 = clamp(-dot(refr_dir_ts, sampled_normal_ts), 0.0f, 1.0f) / (_3206 * _3206);
        _9188 = float4(refr_col * (((((_3180 * _3196) * _3188) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3216) / view_dir_ts.z), (((_3180 * _3188) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3216) / view_dir_ts.z);
        break;
    } while(false);
    return _9188;
}

float3 Evaluate_RefractiveNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness2, float eta, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _9168;
    do
    {
        float3 _5869 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5869;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - (_5869 * eta));
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = roughness2;
        float param_16 = eta;
        float3 param_17 = base_color;
        float4 _5917 = Evaluate_GGXRefraction_BSDF(param_12, param_13, param_14, param_15, param_16, param_17);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_18 = ls.pdf;
            float param_19 = _5917.w;
            mis_weight = power_heuristic(param_18, param_19);
        }
        float3 _5940 = (ls.col * _5917.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9168 = _5940;
            break;
        }
        float3 param_20 = surf.P;
        float3 param_21 = -surf.plane_N;
        float3 _5953 = offset_ray(param_20, param_21);
        sh_r.o[0] = _5953.x;
        sh_r.o[1] = _5953.y;
        sh_r.o[2] = _5953.z;
        sh_r.c[0] = ray.c[0] * _5940.x;
        sh_r.c[1] = ray.c[1] * _5940.y;
        sh_r.c[2] = ray.c[2] * _5940.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9168 = 0.0f.xxx;
        break;
    } while(false);
    return _9168;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _9193;
    do
    {
        float _3260 = roughness * roughness;
        [branch]
        if ((_3260 * _3260) < 1.0000000116860974230803549289703e-07f)
        {
            float _3270 = dot(I, N);
            float _3271 = -_3270;
            float _3281 = mad(-(eta * eta), mad(_3270, _3271, 1.0f), 1.0f);
            if (_3281 < 0.0f)
            {
                _9193 = 0.0f.xxxx;
                break;
            }
            float _3293 = mad(eta, _3271, -sqrt(_3281));
            out_V = float4(normalize((I * eta) + (N * _3293)), _3293);
            _9193 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param = T;
        float3 param_1 = B;
        float3 param_2 = N;
        float3 param_3 = -I;
        float3 _3333 = normalize(tangent_from_world(param, param_1, param_2, param_3));
        float param_4 = _3260;
        float param_5 = _3260;
        float param_6 = rand_u;
        float param_7 = rand_v;
        float3 _3344 = SampleGGX_VNDF(_3333, param_4, param_5, param_6, param_7);
        float _3348 = dot(_3333, _3344);
        float _3358 = mad(-(eta * eta), mad(-_3348, _3348, 1.0f), 1.0f);
        if (_3358 < 0.0f)
        {
            _9193 = 0.0f.xxxx;
            break;
        }
        float _3370 = mad(eta, _3348, -sqrt(_3358));
        float3 _3380 = normalize((_3333 * (-eta)) + (_3344 * _3370));
        float3 param_8 = _3333;
        float3 param_9 = _3344;
        float3 param_10 = _3380;
        float param_11 = _3260;
        float param_12 = eta;
        float3 param_13 = refr_col;
        float3 param_14 = T;
        float3 param_15 = B;
        float3 param_16 = N;
        float3 param_17 = _3380;
        out_V = float4(world_from_tangent(param_14, param_15, param_16, param_17), _3370);
        _9193 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _9193;
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
    float _2306 = old_value;
    old_value = new_value;
    return _2306;
}

float pop_ior_stack(inout float stack[4], float default_value)
{
    float _9201;
    do
    {
        if (stack[3] > 0.0f)
        {
            float param = stack[3];
            float param_1 = -1.0f;
            float _2348 = exchange(param, param_1);
            stack[3] = param;
            _9201 = _2348;
            break;
        }
        if (stack[2] > 0.0f)
        {
            float param_2 = stack[2];
            float param_3 = -1.0f;
            float _2361 = exchange(param_2, param_3);
            stack[2] = param_2;
            _9201 = _2361;
            break;
        }
        if (stack[1] > 0.0f)
        {
            float param_4 = stack[1];
            float param_5 = -1.0f;
            float _2374 = exchange(param_4, param_5);
            stack[1] = param_4;
            _9201 = _2374;
            break;
        }
        if (stack[0] > 0.0f)
        {
            float param_6 = stack[0];
            float param_7 = -1.0f;
            float _2387 = exchange(param_6, param_7);
            stack[0] = param_6;
            _9201 = _2387;
            break;
        }
        _9201 = default_value;
        break;
    } while(false);
    return _9201;
}

void Sample_RefractiveNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, bool is_backfacing, float int_ior, float ext_ior, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float _5990;
    if (is_backfacing)
    {
        _5990 = int_ior / ext_ior;
    }
    else
    {
        _5990 = ext_ior / int_ior;
    }
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float param_4 = roughness;
    float param_5 = _5990;
    float3 param_6 = base_color;
    float param_7 = rand_u;
    float param_8 = rand_v;
    float4 param_9;
    float4 _6014 = Sample_GGXRefraction_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9);
    new_ray.depth = ray.depth + 65536;
    new_ray.c[0] = ((ray.c[0] * _6014.x) * mix_weight) / _6014.w;
    new_ray.c[1] = ((ray.c[1] * _6014.y) * mix_weight) / _6014.w;
    new_ray.c[2] = ((ray.c[2] * _6014.z) * mix_weight) / _6014.w;
    new_ray.pdf = _6014.w;
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
        float _6070 = pop_ior_stack(param_11, param_12);
        new_ray.ior = param_11;
    }
    float3 param_13 = surf.P;
    float3 param_14 = -surf.plane_N;
    float3 _6079 = offset_ray(param_13, param_14);
    new_ray.o[0] = _6079.x;
    new_ray.o[1] = _6079.y;
    new_ray.o[2] = _6079.z;
    new_ray.d[0] = param_9.x;
    new_ray.d[1] = param_9.y;
    new_ray.d[2] = param_9.z;
}

lobe_weights_t get_lobe_weights(float base_color_lum, float spec_color_lum, float specular, float metallic, float transmission, float clearcoat)
{
    float _1713 = 1.0f - metallic;
    float _9356 = (base_color_lum * _1713) * (1.0f - transmission);
    float _1720 = transmission * _1713;
    float _1724;
    if ((specular != 0.0f) || (metallic != 0.0f))
    {
        _1724 = spec_color_lum * mad(-transmission, _1713, 1.0f);
    }
    else
    {
        _1724 = 0.0f;
    }
    float _9357 = _1724;
    float _1734 = 0.25f * clearcoat;
    float _9358 = _1734 * _1713;
    float _9359 = _1720 * base_color_lum;
    float _1743 = _9356;
    float _1752 = mad(_1720, base_color_lum, mad(_1734, _1713, _1743 + _1724));
    if (_1752 != 0.0f)
    {
        _9356 /= _1752;
        _9357 /= _1752;
        _9358 /= _1752;
        _9359 /= _1752;
    }
    lobe_weights_t _9364 = { _9356, _9357, _9358, _9359 };
    return _9364;
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
    float _9216;
    do
    {
        float _2468 = dot(N, L);
        if (_2468 <= 0.0f)
        {
            _9216 = 0.0f;
            break;
        }
        float param = _2468;
        float param_1 = dot(N, V);
        float _2489 = dot(L, H);
        float _2497 = mad((2.0f * _2489) * _2489, roughness, 0.5f);
        _9216 = lerp(1.0f, _2497, schlick_weight(param)) * lerp(1.0f, _2497, schlick_weight(param_1));
        break;
    } while(false);
    return _9216;
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
    float3 _2638 = normalize(L + V);
    float3 H = _2638;
    if (dot(V, _2638) < 0.0f)
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
    float3 _2673 = diff_col;
    float3 _2674 = _2673 + (sheen_color * (3.1415927410125732421875f * schlick_weight(param_5)));
    diff_col = _2674;
    return float4(_2674, pdf);
}

float D_GTR1(float NDotH, float a)
{
    float _9221;
    do
    {
        if (a >= 1.0f)
        {
            _9221 = 0.3183098733425140380859375f;
            break;
        }
        float _2216 = mad(a, a, -1.0f);
        _9221 = _2216 / ((3.1415927410125732421875f * log(a * a)) * mad(_2216 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _9221;
}

float4 Evaluate_PrincipledClearcoat_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float clearcoat_roughness2, float clearcoat_ior, float clearcoat_F0)
{
    float param = sampled_normal_ts.z;
    float param_1 = clearcoat_roughness2;
    float _2974 = D_GTR1(param, param_1);
    float3 param_2 = view_dir_ts;
    float param_3 = 0.0625f;
    float param_4 = 0.0625f;
    float _2981 = G1(param_2, param_3, param_4);
    float3 param_5 = reflected_dir_ts;
    float param_6 = 0.0625f;
    float param_7 = 0.0625f;
    float _2986 = G1(param_5, param_6, param_7);
    float param_8 = dot(reflected_dir_ts, sampled_normal_ts);
    float param_9 = clearcoat_ior;
    float F = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param_8, param_9) - clearcoat_F0) / (1.0f - clearcoat_F0));
    float _3013 = (4.0f * abs(view_dir_ts.z)) * abs(reflected_dir_ts.z);
    float _3016;
    if (_3013 != 0.0f)
    {
        _3016 = (_2974 * (_2981 * _2986)) / _3013;
    }
    else
    {
        _3016 = 0.0f;
    }
    F *= _3016;
    float3 param_10 = view_dir_ts;
    float param_11 = 0.0625f;
    float param_12 = 0.0625f;
    float _3034 = G1(param_10, param_11, param_12);
    float pdf = ((_2974 * _3034) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _3049 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_3049 != 0.0f)
    {
        pdf /= _3049;
    }
    float _3060 = F;
    float _3061 = _3060 * clamp(reflected_dir_ts.z, 0.0f, 1.0f);
    F = _3061;
    return float4(_3061, _3061, _3061, pdf);
}

float3 Evaluate_PrincipledNode(light_sample_t ls, ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float N_dot_L, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _9173;
    do
    {
        float3 _6102 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 lcol = 0.0f.xxx;
        float bsdf_pdf = 0.0f;
        bool _6107 = N_dot_L > 0.0f;
        [branch]
        if ((lobe_weights.diffuse > 1.0000000116860974230803549289703e-07f) && _6107)
        {
            float3 param = -_6102;
            float3 param_1 = surf.N;
            float3 param_2 = ls.L;
            float param_3 = diff.roughness;
            float3 param_4 = diff.base_color;
            float3 param_5 = diff.sheen_color;
            bool param_6 = false;
            float4 _6126 = Evaluate_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6);
            bsdf_pdf = mad(lobe_weights.diffuse, _6126.w, bsdf_pdf);
            lcol += (((ls.col * N_dot_L) * (_6126 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * ls.pdf).xxx);
        }
        float3 H;
        [flatten]
        if (_6107)
        {
            H = normalize(ls.L - _6102);
        }
        else
        {
            H = normalize(ls.L - (_6102 * trans.eta));
        }
        float _6165 = spec.roughness * spec.roughness;
        float _6170 = sqrt(mad(-0.89999997615814208984375f, spec.anisotropy, 1.0f));
        float _6174 = _6165 / _6170;
        float _6178 = _6165 * _6170;
        float3 param_7 = surf.T;
        float3 param_8 = surf.B;
        float3 param_9 = surf.N;
        float3 param_10 = -_6102;
        float3 _6189 = tangent_from_world(param_7, param_8, param_9, param_10);
        float3 param_11 = surf.T;
        float3 param_12 = surf.B;
        float3 param_13 = surf.N;
        float3 param_14 = ls.L;
        float3 _6199 = tangent_from_world(param_11, param_12, param_13, param_14);
        float3 param_15 = surf.T;
        float3 param_16 = surf.B;
        float3 param_17 = surf.N;
        float3 param_18 = H;
        float3 _6209 = tangent_from_world(param_15, param_16, param_17, param_18);
        bool _6211 = lobe_weights.specular > 0.0f;
        bool _6218;
        if (_6211)
        {
            _6218 = (_6174 * _6178) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _6218 = _6211;
        }
        [branch]
        if (_6218 && _6107)
        {
            float3 param_19 = _6189;
            float3 param_20 = _6209;
            float3 param_21 = _6199;
            float param_22 = _6174;
            float param_23 = _6178;
            float param_24 = spec.ior;
            float param_25 = spec.F0;
            float3 param_26 = spec.tmp_col;
            float4 _6240 = Evaluate_GGXSpecular_BSDF(param_19, param_20, param_21, param_22, param_23, param_24, param_25, param_26);
            bsdf_pdf = mad(lobe_weights.specular, _6240.w, bsdf_pdf);
            lcol += ((ls.col * _6240.xyz) / ls.pdf.xxx);
        }
        float _6259 = coat.roughness * coat.roughness;
        bool _6261 = lobe_weights.clearcoat > 0.0f;
        bool _6268;
        if (_6261)
        {
            _6268 = (_6259 * _6259) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _6268 = _6261;
        }
        [branch]
        if (_6268 && _6107)
        {
            float3 param_27 = _6189;
            float3 param_28 = _6209;
            float3 param_29 = _6199;
            float param_30 = _6259;
            float param_31 = coat.ior;
            float param_32 = coat.F0;
            float4 _6286 = Evaluate_PrincipledClearcoat_BSDF(param_27, param_28, param_29, param_30, param_31, param_32);
            bsdf_pdf = mad(lobe_weights.clearcoat, _6286.w, bsdf_pdf);
            lcol += (((ls.col * 0.25f) * _6286.xyz) / ls.pdf.xxx);
        }
        [branch]
        if (lobe_weights.refraction > 0.0f)
        {
            bool _6308 = trans.fresnel != 0.0f;
            bool _6315;
            if (_6308)
            {
                _6315 = (_6165 * _6165) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _6315 = _6308;
            }
            [branch]
            if (_6315 && _6107)
            {
                float3 param_33 = _6189;
                float3 param_34 = _6209;
                float3 param_35 = _6199;
                float param_36 = _6165;
                float param_37 = _6165;
                float param_38 = 1.0f;
                float param_39 = 0.0f;
                float3 param_40 = 1.0f.xxx;
                float4 _6334 = Evaluate_GGXSpecular_BSDF(param_33, param_34, param_35, param_36, param_37, param_38, param_39, param_40);
                bsdf_pdf = mad(lobe_weights.refraction * trans.fresnel, _6334.w, bsdf_pdf);
                lcol += ((ls.col * _6334.xyz) * (trans.fresnel / ls.pdf));
            }
            float _6356 = trans.roughness * trans.roughness;
            bool _6358 = trans.fresnel != 1.0f;
            bool _6365;
            if (_6358)
            {
                _6365 = (_6356 * _6356) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _6365 = _6358;
            }
            [branch]
            if (_6365 && (N_dot_L < 0.0f))
            {
                float3 param_41 = _6189;
                float3 param_42 = _6209;
                float3 param_43 = _6199;
                float param_44 = _6356;
                float param_45 = trans.eta;
                float3 param_46 = diff.base_color;
                float4 _6383 = Evaluate_GGXRefraction_BSDF(param_41, param_42, param_43, param_44, param_45, param_46);
                float _6386 = 1.0f - trans.fresnel;
                bsdf_pdf = mad(lobe_weights.refraction * _6386, _6383.w, bsdf_pdf);
                lcol += ((ls.col * _6383.xyz) * (_6386 / ls.pdf));
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
            _9173 = lcol;
            break;
        }
        float3 _6426;
        if (N_dot_L < 0.0f)
        {
            _6426 = -surf.plane_N;
        }
        else
        {
            _6426 = surf.plane_N;
        }
        float3 param_49 = surf.P;
        float3 param_50 = _6426;
        float3 _6437 = offset_ray(param_49, param_50);
        sh_r.o[0] = _6437.x;
        sh_r.o[1] = _6437.y;
        sh_r.o[2] = _6437.z;
        sh_r.c[0] = ray.c[0] * lcol.x;
        sh_r.c[1] = ray.c[1] * lcol.y;
        sh_r.c[2] = ray.c[2] * lcol.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9173 = 0.0f.xxx;
        break;
    } while(false);
    return _9173;
}

float4 Sample_PrincipledDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float3 sheen_color, bool uniform_sampling, float rand_u, float rand_v, inout float3 out_V)
{
    float _2685 = 6.283185482025146484375f * rand_v;
    float _2688 = cos(_2685);
    float _2691 = sin(_2685);
    float3 V;
    if (uniform_sampling)
    {
        float _2700 = sqrt(mad(-rand_u, rand_u, 1.0f));
        V = float3(_2700 * _2688, _2700 * _2691, rand_u);
    }
    else
    {
        float _2713 = sqrt(rand_u);
        V = float3(_2713 * _2688, _2713 * _2691, sqrt(1.0f - rand_u));
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
    float4 _9206;
    do
    {
        [branch]
        if ((clearcoat_roughness2 * clearcoat_roughness2) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _3078 = reflect(I, N);
            float param = dot(_3078, N);
            float param_1 = clearcoat_ior;
            out_V = _3078;
            float _3097 = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param, param_1) - clearcoat_F0) / (1.0f - clearcoat_F0)) * 1000000.0f;
            _9206 = float4(_3097, _3097, _3097, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _3115 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = clearcoat_roughness2;
        float param_7 = clearcoat_roughness2;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _3126 = SampleGGX_VNDF(_3115, param_6, param_7, param_8, param_9);
        float3 _3137 = normalize(reflect(-_3115, _3126));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _3137;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _3115;
        float3 param_15 = _3126;
        float3 param_16 = _3137;
        float param_17 = clearcoat_roughness2;
        float param_18 = clearcoat_ior;
        float param_19 = clearcoat_F0;
        _9206 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _9206;
}

void Sample_PrincipledNode(ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float rand_u, float rand_v, inout float mix_rand, float mix_weight, inout ray_data_t new_ray)
{
    float3 _6472 = float3(ray.d[0], ray.d[1], ray.d[2]);
    int _6476 = ray.depth & 255;
    int _6480 = (ray.depth >> 8) & 255;
    int _6484 = (ray.depth >> 16) & 255;
    int _6495 = (_6476 + _6480) + _6484;
    [branch]
    if (mix_rand < lobe_weights.diffuse)
    {
        bool _6504 = _6476 < _3537_g_params.max_diff_depth;
        bool _6511;
        if (_6504)
        {
            _6511 = _6495 < _3537_g_params.max_total_depth;
        }
        else
        {
            _6511 = _6504;
        }
        if (_6511)
        {
            float3 param = surf.T;
            float3 param_1 = surf.B;
            float3 param_2 = surf.N;
            float3 param_3 = _6472;
            float param_4 = diff.roughness;
            float3 param_5 = diff.base_color;
            float3 param_6 = diff.sheen_color;
            bool param_7 = false;
            float param_8 = rand_u;
            float param_9 = rand_v;
            float3 param_10;
            float4 _6534 = Sample_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9, param_10);
            float3 _6539 = _6534.xyz * (1.0f - metallic);
            new_ray.depth = ray.depth + 1;
            float3 param_11 = surf.P;
            float3 param_12 = surf.plane_N;
            float3 _6554 = offset_ray(param_11, param_12);
            new_ray.o[0] = _6554.x;
            new_ray.o[1] = _6554.y;
            new_ray.o[2] = _6554.z;
            new_ray.d[0] = param_10.x;
            new_ray.d[1] = param_10.y;
            new_ray.d[2] = param_10.z;
            new_ray.c[0] = ((ray.c[0] * _6539.x) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[1] = ((ray.c[1] * _6539.y) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[2] = ((ray.c[2] * _6539.z) * mix_weight) / lobe_weights.diffuse;
            new_ray.pdf = _6534.w;
        }
    }
    else
    {
        float _6604 = lobe_weights.diffuse + lobe_weights.specular;
        [branch]
        if (mix_rand < _6604)
        {
            bool _6611 = _6480 < _3537_g_params.max_spec_depth;
            bool _6618;
            if (_6611)
            {
                _6618 = _6495 < _3537_g_params.max_total_depth;
            }
            else
            {
                _6618 = _6611;
            }
            if (_6618)
            {
                float3 param_13 = surf.T;
                float3 param_14 = surf.B;
                float3 param_15 = surf.N;
                float3 param_16 = _6472;
                float3 param_17;
                float4 _6637 = Sample_GGXSpecular_BSDF(param_13, param_14, param_15, param_16, spec.roughness, spec.anisotropy, spec.ior, spec.F0, spec.tmp_col, rand_u, rand_v, param_17);
                float _6642 = _6637.w * lobe_weights.specular;
                new_ray.depth = ray.depth + 256;
                new_ray.c[0] = ((ray.c[0] * _6637.x) * mix_weight) / _6642;
                new_ray.c[1] = ((ray.c[1] * _6637.y) * mix_weight) / _6642;
                new_ray.c[2] = ((ray.c[2] * _6637.z) * mix_weight) / _6642;
                new_ray.pdf = _6642;
                float3 param_18 = surf.P;
                float3 param_19 = surf.plane_N;
                float3 _6682 = offset_ray(param_18, param_19);
                new_ray.o[0] = _6682.x;
                new_ray.o[1] = _6682.y;
                new_ray.o[2] = _6682.z;
                new_ray.d[0] = param_17.x;
                new_ray.d[1] = param_17.y;
                new_ray.d[2] = param_17.z;
            }
        }
        else
        {
            float _6707 = _6604 + lobe_weights.clearcoat;
            [branch]
            if (mix_rand < _6707)
            {
                bool _6714 = _6480 < _3537_g_params.max_spec_depth;
                bool _6721;
                if (_6714)
                {
                    _6721 = _6495 < _3537_g_params.max_total_depth;
                }
                else
                {
                    _6721 = _6714;
                }
                if (_6721)
                {
                    float3 param_20 = surf.T;
                    float3 param_21 = surf.B;
                    float3 param_22 = surf.N;
                    float3 param_23 = _6472;
                    float param_24 = coat.roughness * coat.roughness;
                    float param_25 = coat.ior;
                    float param_26 = coat.F0;
                    float param_27 = rand_u;
                    float param_28 = rand_v;
                    float3 param_29;
                    float4 _6745 = Sample_PrincipledClearcoat_BSDF(param_20, param_21, param_22, param_23, param_24, param_25, param_26, param_27, param_28, param_29);
                    float _6750 = _6745.w * lobe_weights.clearcoat;
                    new_ray.depth = ray.depth + 256;
                    new_ray.c[0] = (((0.25f * ray.c[0]) * _6745.x) * mix_weight) / _6750;
                    new_ray.c[1] = (((0.25f * ray.c[1]) * _6745.y) * mix_weight) / _6750;
                    new_ray.c[2] = (((0.25f * ray.c[2]) * _6745.z) * mix_weight) / _6750;
                    new_ray.pdf = _6750;
                    float3 param_30 = surf.P;
                    float3 param_31 = surf.plane_N;
                    float3 _6793 = offset_ray(param_30, param_31);
                    new_ray.o[0] = _6793.x;
                    new_ray.o[1] = _6793.y;
                    new_ray.o[2] = _6793.z;
                    new_ray.d[0] = param_29.x;
                    new_ray.d[1] = param_29.y;
                    new_ray.d[2] = param_29.z;
                }
            }
            else
            {
                bool _6815 = mix_rand >= trans.fresnel;
                bool _6822;
                if (_6815)
                {
                    _6822 = _6484 < _3537_g_params.max_refr_depth;
                }
                else
                {
                    _6822 = _6815;
                }
                bool _6836;
                if (!_6822)
                {
                    bool _6828 = mix_rand < trans.fresnel;
                    bool _6835;
                    if (_6828)
                    {
                        _6835 = _6480 < _3537_g_params.max_spec_depth;
                    }
                    else
                    {
                        _6835 = _6828;
                    }
                    _6836 = _6835;
                }
                else
                {
                    _6836 = _6822;
                }
                bool _6843;
                if (_6836)
                {
                    _6843 = _6495 < _3537_g_params.max_total_depth;
                }
                else
                {
                    _6843 = _6836;
                }
                [branch]
                if (_6843)
                {
                    mix_rand -= _6707;
                    mix_rand /= lobe_weights.refraction;
                    float4 F;
                    float3 V;
                    [branch]
                    if (mix_rand < trans.fresnel)
                    {
                        float3 param_32 = surf.T;
                        float3 param_33 = surf.B;
                        float3 param_34 = surf.N;
                        float3 param_35 = _6472;
                        float3 param_36;
                        float4 _6873 = Sample_GGXSpecular_BSDF(param_32, param_33, param_34, param_35, spec.roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, rand_u, rand_v, param_36);
                        V = param_36;
                        F = _6873;
                        new_ray.depth = ray.depth + 256;
                        float3 param_37 = surf.P;
                        float3 param_38 = surf.plane_N;
                        float3 _6883 = offset_ray(param_37, param_38);
                        new_ray.o[0] = _6883.x;
                        new_ray.o[1] = _6883.y;
                        new_ray.o[2] = _6883.z;
                    }
                    else
                    {
                        float3 param_39 = surf.T;
                        float3 param_40 = surf.B;
                        float3 param_41 = surf.N;
                        float3 param_42 = _6472;
                        float param_43 = trans.roughness;
                        float param_44 = trans.eta;
                        float3 param_45 = diff.base_color;
                        float param_46 = rand_u;
                        float param_47 = rand_v;
                        float4 param_48;
                        float4 _6912 = Sample_GGXRefraction_BSDF(param_39, param_40, param_41, param_42, param_43, param_44, param_45, param_46, param_47, param_48);
                        F = _6912;
                        V = param_48.xyz;
                        new_ray.depth = ray.depth + 65536;
                        float3 param_49 = surf.P;
                        float3 param_50 = -surf.plane_N;
                        float3 _6925 = offset_ray(param_49, param_50);
                        new_ray.o[0] = _6925.x;
                        new_ray.o[1] = _6925.y;
                        new_ray.o[2] = _6925.z;
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
                            float _6951 = pop_ior_stack(param_52, param_53);
                            new_ray.ior = param_52;
                        }
                    }
                    float4 _10749 = F;
                    float _6957 = _10749.w * lobe_weights.refraction;
                    float4 _10751 = _10749;
                    _10751.w = _6957;
                    F = _10751;
                    new_ray.c[0] = ((ray.c[0] * _10749.x) * mix_weight) / _6957;
                    new_ray.c[1] = ((ray.c[1] * _10749.y) * mix_weight) / _6957;
                    new_ray.c[2] = ((ray.c[2] * _10749.z) * mix_weight) / _6957;
                    new_ray.pdf = _6957;
                    new_ray.d[0] = V.x;
                    new_ray.d[1] = V.y;
                    new_ray.d[2] = V.z;
                }
            }
        }
    }
}

float3 ShadeSurface(hit_data_t inter, ray_data_t ray)
{
    float3 _9143;
    do
    {
        float3 _7013 = float3(ray.d[0], ray.d[1], ray.d[2]);
        [branch]
        if (inter.mask == 0)
        {
            ray_data_t param = ray;
            float3 _7022 = Evaluate_EnvColor(param);
            _9143 = float3(ray.c[0] * _7022.x, ray.c[1] * _7022.y, ray.c[2] * _7022.z);
            break;
        }
        float3 _7049 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_7013 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            ray_data_t param_1 = ray;
            hit_data_t param_2 = inter;
            float3 _7061 = Evaluate_LightColor(param_1, param_2);
            _9143 = float3(ray.c[0] * _7061.x, ray.c[1] * _7061.y, ray.c[2] * _7061.z);
            break;
        }
        bool _7082 = inter.prim_index < 0;
        int _7085;
        if (_7082)
        {
            _7085 = (-1) - inter.prim_index;
        }
        else
        {
            _7085 = inter.prim_index;
        }
        uint _7096 = uint(_7085);
        material_t _7104;
        [unroll]
        for (int _89ident = 0; _89ident < 5; _89ident++)
        {
            _7104.textures[_89ident] = _4757.Load(_89ident * 4 + ((_4761.Load(_7096 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
        }
        [unroll]
        for (int _90ident = 0; _90ident < 3; _90ident++)
        {
            _7104.base_color[_90ident] = asfloat(_4757.Load(_90ident * 4 + ((_4761.Load(_7096 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
        }
        _7104.flags = _4757.Load(((_4761.Load(_7096 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
        _7104.type = _4757.Load(((_4761.Load(_7096 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
        _7104.tangent_rotation_or_strength = asfloat(_4757.Load(((_4761.Load(_7096 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
        _7104.roughness_and_anisotropic = _4757.Load(((_4761.Load(_7096 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
        _7104.ior = asfloat(_4757.Load(((_4761.Load(_7096 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
        _7104.sheen_and_sheen_tint = _4757.Load(((_4761.Load(_7096 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
        _7104.tint_and_metallic = _4757.Load(((_4761.Load(_7096 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
        _7104.transmission_and_transmission_roughness = _4757.Load(((_4761.Load(_7096 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
        _7104.specular_and_specular_tint = _4757.Load(((_4761.Load(_7096 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
        _7104.clearcoat_and_clearcoat_roughness = _4757.Load(((_4761.Load(_7096 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
        _7104.normal_map_strength_unorm = _4757.Load(((_4761.Load(_7096 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
        uint _10198 = _7104.textures[0];
        uint _10199 = _7104.textures[1];
        uint _10200 = _7104.textures[2];
        uint _10201 = _7104.textures[3];
        uint _10202 = _7104.textures[4];
        float _10203 = _7104.base_color[0];
        float _10204 = _7104.base_color[1];
        float _10205 = _7104.base_color[2];
        uint _9808 = _7104.flags;
        uint _9809 = _7104.type;
        float _9810 = _7104.tangent_rotation_or_strength;
        uint _9811 = _7104.roughness_and_anisotropic;
        float _9812 = _7104.ior;
        uint _9813 = _7104.sheen_and_sheen_tint;
        uint _9814 = _7104.tint_and_metallic;
        uint _9815 = _7104.transmission_and_transmission_roughness;
        uint _9816 = _7104.specular_and_specular_tint;
        uint _9817 = _7104.clearcoat_and_clearcoat_roughness;
        uint _9818 = _7104.normal_map_strength_unorm;
        transform_t _7159;
        _7159.xform = asfloat(uint4x4(_4404.Load4(asuint(asfloat(_7152.Load(inter.obj_index * 32 + 12))) * 128 + 0), _4404.Load4(asuint(asfloat(_7152.Load(inter.obj_index * 32 + 12))) * 128 + 16), _4404.Load4(asuint(asfloat(_7152.Load(inter.obj_index * 32 + 12))) * 128 + 32), _4404.Load4(asuint(asfloat(_7152.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _7159.inv_xform = asfloat(uint4x4(_4404.Load4(asuint(asfloat(_7152.Load(inter.obj_index * 32 + 12))) * 128 + 64), _4404.Load4(asuint(asfloat(_7152.Load(inter.obj_index * 32 + 12))) * 128 + 80), _4404.Load4(asuint(asfloat(_7152.Load(inter.obj_index * 32 + 12))) * 128 + 96), _4404.Load4(asuint(asfloat(_7152.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _7166 = _7096 * 3u;
        vertex_t _7171;
        [unroll]
        for (int _91ident = 0; _91ident < 3; _91ident++)
        {
            _7171.p[_91ident] = asfloat(_4429.Load(_91ident * 4 + _4433.Load(_7166 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _92ident = 0; _92ident < 3; _92ident++)
        {
            _7171.n[_92ident] = asfloat(_4429.Load(_92ident * 4 + _4433.Load(_7166 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _93ident = 0; _93ident < 3; _93ident++)
        {
            _7171.b[_93ident] = asfloat(_4429.Load(_93ident * 4 + _4433.Load(_7166 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _94ident = 0; _94ident < 2; _94ident++)
        {
            [unroll]
            for (int _95ident = 0; _95ident < 2; _95ident++)
            {
                _7171.t[_94ident][_95ident] = asfloat(_4429.Load(_95ident * 4 + _94ident * 8 + _4433.Load(_7166 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _7217;
        [unroll]
        for (int _96ident = 0; _96ident < 3; _96ident++)
        {
            _7217.p[_96ident] = asfloat(_4429.Load(_96ident * 4 + _4433.Load((_7166 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _97ident = 0; _97ident < 3; _97ident++)
        {
            _7217.n[_97ident] = asfloat(_4429.Load(_97ident * 4 + _4433.Load((_7166 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _98ident = 0; _98ident < 3; _98ident++)
        {
            _7217.b[_98ident] = asfloat(_4429.Load(_98ident * 4 + _4433.Load((_7166 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _99ident = 0; _99ident < 2; _99ident++)
        {
            [unroll]
            for (int _100ident = 0; _100ident < 2; _100ident++)
            {
                _7217.t[_99ident][_100ident] = asfloat(_4429.Load(_100ident * 4 + _99ident * 8 + _4433.Load((_7166 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _7263;
        [unroll]
        for (int _101ident = 0; _101ident < 3; _101ident++)
        {
            _7263.p[_101ident] = asfloat(_4429.Load(_101ident * 4 + _4433.Load((_7166 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _102ident = 0; _102ident < 3; _102ident++)
        {
            _7263.n[_102ident] = asfloat(_4429.Load(_102ident * 4 + _4433.Load((_7166 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _103ident = 0; _103ident < 3; _103ident++)
        {
            _7263.b[_103ident] = asfloat(_4429.Load(_103ident * 4 + _4433.Load((_7166 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _104ident = 0; _104ident < 2; _104ident++)
        {
            [unroll]
            for (int _105ident = 0; _105ident < 2; _105ident++)
            {
                _7263.t[_104ident][_105ident] = asfloat(_4429.Load(_105ident * 4 + _104ident * 8 + _4433.Load((_7166 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _7309 = float3(_7171.p[0], _7171.p[1], _7171.p[2]);
        float3 _7317 = float3(_7217.p[0], _7217.p[1], _7217.p[2]);
        float3 _7325 = float3(_7263.p[0], _7263.p[1], _7263.p[2]);
        float _7332 = (1.0f - inter.u) - inter.v;
        float3 _7364 = normalize(((float3(_7171.n[0], _7171.n[1], _7171.n[2]) * _7332) + (float3(_7217.n[0], _7217.n[1], _7217.n[2]) * inter.u)) + (float3(_7263.n[0], _7263.n[1], _7263.n[2]) * inter.v));
        float3 _9747 = _7364;
        float2 _7390 = ((float2(_7171.t[0][0], _7171.t[0][1]) * _7332) + (float2(_7217.t[0][0], _7217.t[0][1]) * inter.u)) + (float2(_7263.t[0][0], _7263.t[0][1]) * inter.v);
        float3 _7406 = cross(_7317 - _7309, _7325 - _7309);
        float _7411 = length(_7406);
        float3 _9748 = _7406 / _7411.xxx;
        float3 _7448 = ((float3(_7171.b[0], _7171.b[1], _7171.b[2]) * _7332) + (float3(_7217.b[0], _7217.b[1], _7217.b[2]) * inter.u)) + (float3(_7263.b[0], _7263.b[1], _7263.b[2]) * inter.v);
        float3 _9746 = _7448;
        float3 _9745 = cross(_7448, _7364);
        if (_7082)
        {
            if ((_4761.Load(_7096 * 4 + 0) & 65535u) == 65535u)
            {
                _9143 = 0.0f.xxx;
                break;
            }
            material_t _7473;
            [unroll]
            for (int _106ident = 0; _106ident < 5; _106ident++)
            {
                _7473.textures[_106ident] = _4757.Load(_106ident * 4 + (_4761.Load(_7096 * 4 + 0) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _107ident = 0; _107ident < 3; _107ident++)
            {
                _7473.base_color[_107ident] = asfloat(_4757.Load(_107ident * 4 + (_4761.Load(_7096 * 4 + 0) & 16383u) * 76 + 20));
            }
            _7473.flags = _4757.Load((_4761.Load(_7096 * 4 + 0) & 16383u) * 76 + 32);
            _7473.type = _4757.Load((_4761.Load(_7096 * 4 + 0) & 16383u) * 76 + 36);
            _7473.tangent_rotation_or_strength = asfloat(_4757.Load((_4761.Load(_7096 * 4 + 0) & 16383u) * 76 + 40));
            _7473.roughness_and_anisotropic = _4757.Load((_4761.Load(_7096 * 4 + 0) & 16383u) * 76 + 44);
            _7473.ior = asfloat(_4757.Load((_4761.Load(_7096 * 4 + 0) & 16383u) * 76 + 48));
            _7473.sheen_and_sheen_tint = _4757.Load((_4761.Load(_7096 * 4 + 0) & 16383u) * 76 + 52);
            _7473.tint_and_metallic = _4757.Load((_4761.Load(_7096 * 4 + 0) & 16383u) * 76 + 56);
            _7473.transmission_and_transmission_roughness = _4757.Load((_4761.Load(_7096 * 4 + 0) & 16383u) * 76 + 60);
            _7473.specular_and_specular_tint = _4757.Load((_4761.Load(_7096 * 4 + 0) & 16383u) * 76 + 64);
            _7473.clearcoat_and_clearcoat_roughness = _4757.Load((_4761.Load(_7096 * 4 + 0) & 16383u) * 76 + 68);
            _7473.normal_map_strength_unorm = _4757.Load((_4761.Load(_7096 * 4 + 0) & 16383u) * 76 + 72);
            _10198 = _7473.textures[0];
            _10199 = _7473.textures[1];
            _10200 = _7473.textures[2];
            _10201 = _7473.textures[3];
            _10202 = _7473.textures[4];
            _10203 = _7473.base_color[0];
            _10204 = _7473.base_color[1];
            _10205 = _7473.base_color[2];
            _9808 = _7473.flags;
            _9809 = _7473.type;
            _9810 = _7473.tangent_rotation_or_strength;
            _9811 = _7473.roughness_and_anisotropic;
            _9812 = _7473.ior;
            _9813 = _7473.sheen_and_sheen_tint;
            _9814 = _7473.tint_and_metallic;
            _9815 = _7473.transmission_and_transmission_roughness;
            _9816 = _7473.specular_and_specular_tint;
            _9817 = _7473.clearcoat_and_clearcoat_roughness;
            _9818 = _7473.normal_map_strength_unorm;
            _9748 = -_9748;
            _9747 = -_9747;
            _9746 = -_9746;
            _9745 = -_9745;
        }
        float3 param_3 = _9748;
        float4x4 param_4 = _7159.inv_xform;
        _9748 = TransformNormal(param_3, param_4);
        float3 param_5 = _9747;
        float4x4 param_6 = _7159.inv_xform;
        _9747 = TransformNormal(param_5, param_6);
        float3 param_7 = _9746;
        float4x4 param_8 = _7159.inv_xform;
        _9746 = TransformNormal(param_7, param_8);
        float3 param_9 = _9745;
        float4x4 param_10 = _7159.inv_xform;
        _9748 = normalize(_9748);
        _9747 = normalize(_9747);
        _9746 = normalize(_9746);
        _9745 = normalize(TransformNormal(param_9, param_10));
        float _7613 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _7623 = mad(0.5f, log2(abs(mad(_7217.t[0][0] - _7171.t[0][0], _7263.t[0][1] - _7171.t[0][1], -((_7263.t[0][0] - _7171.t[0][0]) * (_7217.t[0][1] - _7171.t[0][1])))) / _7411), log2(_7613));
        uint param_11 = uint(hash(ray.xy));
        float _7630 = construct_float(param_11);
        uint param_12 = uint(hash(hash(ray.xy)));
        float _7637 = construct_float(param_12);
        float param_13[4] = ray.ior;
        bool param_14 = _7082;
        float param_15 = 1.0f;
        float _7646 = peek_ior_stack(param_13, param_14, param_15);
        float3 col = 0.0f.xxx;
        int _7651 = ray.depth & 255;
        int _7656 = (ray.depth >> 8) & 255;
        int _7661 = (ray.depth >> 16) & 255;
        int _7672 = (_7651 + _7656) + _7661;
        int _7680 = _3537_g_params.hi + ((_7672 + ((ray.depth >> 24) & 255)) * 7);
        float mix_rand = frac(asfloat(_3521.Load(_7680 * 4 + 0)) + _7630);
        float mix_weight = 1.0f;
        float _7717;
        float _7734;
        float _7760;
        float _7827;
        while (_9809 == 4u)
        {
            float mix_val = _9810;
            if (_10199 != 4294967295u)
            {
                mix_val *= SampleBilinear(_10199, _7390, 0).x;
            }
            if (_7082)
            {
                _7717 = _7646 / _9812;
            }
            else
            {
                _7717 = _9812 / _7646;
            }
            if (_9812 != 0.0f)
            {
                float param_16 = dot(_7013, _9747);
                float param_17 = _7717;
                _7734 = fresnel_dielectric_cos(param_16, param_17);
            }
            else
            {
                _7734 = 1.0f;
            }
            float _7749 = mix_val;
            float _7750 = _7749 * clamp(_7734, 0.0f, 1.0f);
            mix_val = _7750;
            if (mix_rand > _7750)
            {
                if ((_9808 & 2u) != 0u)
                {
                    _7760 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _7760 = 1.0f;
                }
                mix_weight *= _7760;
                material_t _7773;
                [unroll]
                for (int _108ident = 0; _108ident < 5; _108ident++)
                {
                    _7773.textures[_108ident] = _4757.Load(_108ident * 4 + _10201 * 76 + 0);
                }
                [unroll]
                for (int _109ident = 0; _109ident < 3; _109ident++)
                {
                    _7773.base_color[_109ident] = asfloat(_4757.Load(_109ident * 4 + _10201 * 76 + 20));
                }
                _7773.flags = _4757.Load(_10201 * 76 + 32);
                _7773.type = _4757.Load(_10201 * 76 + 36);
                _7773.tangent_rotation_or_strength = asfloat(_4757.Load(_10201 * 76 + 40));
                _7773.roughness_and_anisotropic = _4757.Load(_10201 * 76 + 44);
                _7773.ior = asfloat(_4757.Load(_10201 * 76 + 48));
                _7773.sheen_and_sheen_tint = _4757.Load(_10201 * 76 + 52);
                _7773.tint_and_metallic = _4757.Load(_10201 * 76 + 56);
                _7773.transmission_and_transmission_roughness = _4757.Load(_10201 * 76 + 60);
                _7773.specular_and_specular_tint = _4757.Load(_10201 * 76 + 64);
                _7773.clearcoat_and_clearcoat_roughness = _4757.Load(_10201 * 76 + 68);
                _7773.normal_map_strength_unorm = _4757.Load(_10201 * 76 + 72);
                _10198 = _7773.textures[0];
                _10199 = _7773.textures[1];
                _10200 = _7773.textures[2];
                _10201 = _7773.textures[3];
                _10202 = _7773.textures[4];
                _10203 = _7773.base_color[0];
                _10204 = _7773.base_color[1];
                _10205 = _7773.base_color[2];
                _9808 = _7773.flags;
                _9809 = _7773.type;
                _9810 = _7773.tangent_rotation_or_strength;
                _9811 = _7773.roughness_and_anisotropic;
                _9812 = _7773.ior;
                _9813 = _7773.sheen_and_sheen_tint;
                _9814 = _7773.tint_and_metallic;
                _9815 = _7773.transmission_and_transmission_roughness;
                _9816 = _7773.specular_and_specular_tint;
                _9817 = _7773.clearcoat_and_clearcoat_roughness;
                _9818 = _7773.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_9808 & 2u) != 0u)
                {
                    _7827 = 1.0f / mix_val;
                }
                else
                {
                    _7827 = 1.0f;
                }
                mix_weight *= _7827;
                material_t _7839;
                [unroll]
                for (int _110ident = 0; _110ident < 5; _110ident++)
                {
                    _7839.textures[_110ident] = _4757.Load(_110ident * 4 + _10202 * 76 + 0);
                }
                [unroll]
                for (int _111ident = 0; _111ident < 3; _111ident++)
                {
                    _7839.base_color[_111ident] = asfloat(_4757.Load(_111ident * 4 + _10202 * 76 + 20));
                }
                _7839.flags = _4757.Load(_10202 * 76 + 32);
                _7839.type = _4757.Load(_10202 * 76 + 36);
                _7839.tangent_rotation_or_strength = asfloat(_4757.Load(_10202 * 76 + 40));
                _7839.roughness_and_anisotropic = _4757.Load(_10202 * 76 + 44);
                _7839.ior = asfloat(_4757.Load(_10202 * 76 + 48));
                _7839.sheen_and_sheen_tint = _4757.Load(_10202 * 76 + 52);
                _7839.tint_and_metallic = _4757.Load(_10202 * 76 + 56);
                _7839.transmission_and_transmission_roughness = _4757.Load(_10202 * 76 + 60);
                _7839.specular_and_specular_tint = _4757.Load(_10202 * 76 + 64);
                _7839.clearcoat_and_clearcoat_roughness = _4757.Load(_10202 * 76 + 68);
                _7839.normal_map_strength_unorm = _4757.Load(_10202 * 76 + 72);
                _10198 = _7839.textures[0];
                _10199 = _7839.textures[1];
                _10200 = _7839.textures[2];
                _10201 = _7839.textures[3];
                _10202 = _7839.textures[4];
                _10203 = _7839.base_color[0];
                _10204 = _7839.base_color[1];
                _10205 = _7839.base_color[2];
                _9808 = _7839.flags;
                _9809 = _7839.type;
                _9810 = _7839.tangent_rotation_or_strength;
                _9811 = _7839.roughness_and_anisotropic;
                _9812 = _7839.ior;
                _9813 = _7839.sheen_and_sheen_tint;
                _9814 = _7839.tint_and_metallic;
                _9815 = _7839.transmission_and_transmission_roughness;
                _9816 = _7839.specular_and_specular_tint;
                _9817 = _7839.clearcoat_and_clearcoat_roughness;
                _9818 = _7839.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_10198 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_10198, _7390, 0).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_1001.Load(_10198 * 80 + 0) & 16384u) != 0u)
            {
                float3 _10770 = normals;
                _10770.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _10770;
            }
            float3 _7923 = _9747;
            _9747 = normalize(((_9745 * normals.x) + (_7923 * normals.z)) + (_9746 * normals.y));
            if ((_9818 & 65535u) != 65535u)
            {
                _9747 = normalize(_7923 + ((_9747 - _7923) * clamp(float(_9818 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_18 = _9748;
            float3 param_19 = -_7013;
            float3 param_20 = _9747;
            _9747 = ensure_valid_reflection(param_18, param_19, param_20);
        }
        float3 _7989 = ((_7309 * _7332) + (_7317 * inter.u)) + (_7325 * inter.v);
        float3 _7996 = float3(-_7989.z, 0.0f, _7989.x);
        float3 tangent = _7996;
        float3 param_21 = _7996;
        float4x4 param_22 = _7159.inv_xform;
        float3 _8002 = TransformNormal(param_21, param_22);
        tangent = _8002;
        float3 _8006 = cross(_8002, _9747);
        if (dot(_8006, _8006) == 0.0f)
        {
            float3 param_23 = _7989;
            float4x4 param_24 = _7159.inv_xform;
            tangent = TransformNormal(param_23, param_24);
        }
        if (_9810 != 0.0f)
        {
            float3 param_25 = tangent;
            float3 param_26 = _9747;
            float param_27 = _9810;
            tangent = rotate_around_axis(param_25, param_26, param_27);
        }
        float3 _8039 = normalize(cross(tangent, _9747));
        _9746 = _8039;
        _9745 = cross(_9747, _8039);
        float3 _9897 = 0.0f.xxx;
        float3 _9896 = 0.0f.xxx;
        float _9901 = 0.0f;
        float _9899 = 0.0f;
        float _9900 = 1.0f;
        bool _8055 = _3537_g_params.li_count != 0;
        bool _8061;
        if (_8055)
        {
            _8061 = _9809 != 3u;
        }
        else
        {
            _8061 = _8055;
        }
        float3 _9898;
        bool _9902;
        bool _9903;
        if (_8061)
        {
            float3 param_28 = _7049;
            float3 param_29 = _9745;
            float3 param_30 = _9746;
            float3 param_31 = _9747;
            int param_32 = _7680;
            float2 param_33 = float2(_7630, _7637);
            light_sample_t _9912 = { _9896, _9897, _9898, _9899, _9900, _9901, _9902, _9903 };
            light_sample_t param_34 = _9912;
            SampleLightSource(param_28, param_29, param_30, param_31, param_32, param_33, param_34);
            _9896 = param_34.col;
            _9897 = param_34.L;
            _9898 = param_34.lp;
            _9899 = param_34.area;
            _9900 = param_34.dist_mul;
            _9901 = param_34.pdf;
            _9902 = param_34.cast_shadow;
            _9903 = param_34.from_env;
        }
        float _8089 = dot(_9747, _9897);
        float3 base_color = float3(_10203, _10204, _10205);
        [branch]
        if (_10199 != 4294967295u)
        {
            base_color *= SampleBilinear(_10199, _7390, int(get_texture_lod(texSize(_10199), _7623)), true, true).xyz;
        }
        float3 tint_color = 0.0f.xxx;
        float _8122 = lum(base_color);
        [flatten]
        if (_8122 > 0.0f)
        {
            tint_color = base_color / _8122.xxx;
        }
        float roughness = clamp(float(_9811 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_10200 != 4294967295u)
        {
            roughness *= SampleBilinear(_10200, _7390, int(get_texture_lod(texSize(_10200), _7623)), false, true).x;
        }
        float _8167 = frac(asfloat(_3521.Load((_7680 + 1) * 4 + 0)) + _7630);
        float _8176 = frac(asfloat(_3521.Load((_7680 + 2) * 4 + 0)) + _7637);
        float _10325 = 0.0f;
        float _10324 = 0.0f;
        float _10323 = 0.0f;
        float _9961[4];
        [unroll]
        for (int i = 0; i < 4; )
        {
            _9961[i] = ray.ior[i];
            i++;
            continue;
        }
        float _9962 = _7613;
        float _9963 = ray.cone_spread;
        int _9964 = ray.xy;
        float _9959 = 0.0f;
        float _10430 = 0.0f;
        float _10429 = 0.0f;
        float _10428 = 0.0f;
        int _10066 = ray.depth;
        int _10070 = ray.xy;
        int _9965;
        float _10068;
        float _10253;
        float _10254;
        float _10255;
        float _10288;
        float _10289;
        float _10290;
        float _10358;
        float _10359;
        float _10360;
        float _10393;
        float _10394;
        float _10395;
        [branch]
        if (_9809 == 0u)
        {
            [branch]
            if ((_9901 > 0.0f) && (_8089 > 0.0f))
            {
                light_sample_t _9929 = { _9896, _9897, _9898, _9899, _9900, _9901, _9902, _9903 };
                surface_t _9756 = { _7049, _9745, _9746, _9747, _9748, _7390 };
                float _10434[3] = { _10428, _10429, _10430 };
                float _10399[3] = { _10393, _10394, _10395 };
                float _10364[3] = { _10358, _10359, _10360 };
                shadow_ray_t _10080 = { _10364, _10066, _10399, _10068, _10434, _10070 };
                shadow_ray_t param_35 = _10080;
                float3 _8236 = Evaluate_DiffuseNode(_9929, ray, _9756, base_color, roughness, mix_weight, param_35);
                _10358 = param_35.o[0];
                _10359 = param_35.o[1];
                _10360 = param_35.o[2];
                _10066 = param_35.depth;
                _10393 = param_35.d[0];
                _10394 = param_35.d[1];
                _10395 = param_35.d[2];
                _10068 = param_35.dist;
                _10428 = param_35.c[0];
                _10429 = param_35.c[1];
                _10430 = param_35.c[2];
                _10070 = param_35.xy;
                col += _8236;
            }
            bool _8243 = _7651 < _3537_g_params.max_diff_depth;
            bool _8250;
            if (_8243)
            {
                _8250 = _7672 < _3537_g_params.max_total_depth;
            }
            else
            {
                _8250 = _8243;
            }
            [branch]
            if (_8250)
            {
                surface_t _9763 = { _7049, _9745, _9746, _9747, _9748, _7390 };
                float _10329[3] = { _10323, _10324, _10325 };
                float _10294[3] = { _10288, _10289, _10290 };
                float _10259[3] = { _10253, _10254, _10255 };
                ray_data_t _9979 = { _10259, _10294, _9959, _10329, _9961, _9962, _9963, _9964, _9965 };
                ray_data_t param_36 = _9979;
                Sample_DiffuseNode(ray, _9763, base_color, roughness, _8167, _8176, mix_weight, param_36);
                _10253 = param_36.o[0];
                _10254 = param_36.o[1];
                _10255 = param_36.o[2];
                _10288 = param_36.d[0];
                _10289 = param_36.d[1];
                _10290 = param_36.d[2];
                _9959 = param_36.pdf;
                _10323 = param_36.c[0];
                _10324 = param_36.c[1];
                _10325 = param_36.c[2];
                _9961 = param_36.ior;
                _9962 = param_36.cone_width;
                _9963 = param_36.cone_spread;
                _9964 = param_36.xy;
                _9965 = param_36.depth;
            }
        }
        else
        {
            [branch]
            if (_9809 == 1u)
            {
                float param_37 = 1.0f;
                float param_38 = 1.5f;
                float _8274 = fresnel_dielectric_cos(param_37, param_38);
                float _8278 = roughness * roughness;
                bool _8281 = _9901 > 0.0f;
                bool _8288;
                if (_8281)
                {
                    _8288 = (_8278 * _8278) >= 1.0000000116860974230803549289703e-07f;
                }
                else
                {
                    _8288 = _8281;
                }
                [branch]
                if (_8288 && (_8089 > 0.0f))
                {
                    light_sample_t _9938 = { _9896, _9897, _9898, _9899, _9900, _9901, _9902, _9903 };
                    surface_t _9770 = { _7049, _9745, _9746, _9747, _9748, _7390 };
                    float _10441[3] = { _10428, _10429, _10430 };
                    float _10406[3] = { _10393, _10394, _10395 };
                    float _10371[3] = { _10358, _10359, _10360 };
                    shadow_ray_t _10093 = { _10371, _10066, _10406, _10068, _10441, _10070 };
                    shadow_ray_t param_39 = _10093;
                    float3 _8303 = Evaluate_GlossyNode(_9938, ray, _9770, base_color, roughness, 1.5f, _8274, mix_weight, param_39);
                    _10358 = param_39.o[0];
                    _10359 = param_39.o[1];
                    _10360 = param_39.o[2];
                    _10066 = param_39.depth;
                    _10393 = param_39.d[0];
                    _10394 = param_39.d[1];
                    _10395 = param_39.d[2];
                    _10068 = param_39.dist;
                    _10428 = param_39.c[0];
                    _10429 = param_39.c[1];
                    _10430 = param_39.c[2];
                    _10070 = param_39.xy;
                    col += _8303;
                }
                bool _8310 = _7656 < _3537_g_params.max_spec_depth;
                bool _8317;
                if (_8310)
                {
                    _8317 = _7672 < _3537_g_params.max_total_depth;
                }
                else
                {
                    _8317 = _8310;
                }
                [branch]
                if (_8317)
                {
                    surface_t _9777 = { _7049, _9745, _9746, _9747, _9748, _7390 };
                    float _10336[3] = { _10323, _10324, _10325 };
                    float _10301[3] = { _10288, _10289, _10290 };
                    float _10266[3] = { _10253, _10254, _10255 };
                    ray_data_t _9998 = { _10266, _10301, _9959, _10336, _9961, _9962, _9963, _9964, _9965 };
                    ray_data_t param_40 = _9998;
                    Sample_GlossyNode(ray, _9777, base_color, roughness, 1.5f, _8274, _8167, _8176, mix_weight, param_40);
                    _10253 = param_40.o[0];
                    _10254 = param_40.o[1];
                    _10255 = param_40.o[2];
                    _10288 = param_40.d[0];
                    _10289 = param_40.d[1];
                    _10290 = param_40.d[2];
                    _9959 = param_40.pdf;
                    _10323 = param_40.c[0];
                    _10324 = param_40.c[1];
                    _10325 = param_40.c[2];
                    _9961 = param_40.ior;
                    _9962 = param_40.cone_width;
                    _9963 = param_40.cone_spread;
                    _9964 = param_40.xy;
                    _9965 = param_40.depth;
                }
            }
            else
            {
                [branch]
                if (_9809 == 2u)
                {
                    float _8341 = roughness * roughness;
                    bool _8344 = _9901 > 0.0f;
                    bool _8351;
                    if (_8344)
                    {
                        _8351 = (_8341 * _8341) >= 1.0000000116860974230803549289703e-07f;
                    }
                    else
                    {
                        _8351 = _8344;
                    }
                    [branch]
                    if (_8351 && (_8089 < 0.0f))
                    {
                        float _8359;
                        if (_7082)
                        {
                            _8359 = _9812 / _7646;
                        }
                        else
                        {
                            _8359 = _7646 / _9812;
                        }
                        light_sample_t _9947 = { _9896, _9897, _9898, _9899, _9900, _9901, _9902, _9903 };
                        surface_t _9784 = { _7049, _9745, _9746, _9747, _9748, _7390 };
                        float _10448[3] = { _10428, _10429, _10430 };
                        float _10413[3] = { _10393, _10394, _10395 };
                        float _10378[3] = { _10358, _10359, _10360 };
                        shadow_ray_t _10106 = { _10378, _10066, _10413, _10068, _10448, _10070 };
                        shadow_ray_t param_41 = _10106;
                        float3 _8381 = Evaluate_RefractiveNode(_9947, ray, _9784, base_color, _8341, _8359, mix_weight, param_41);
                        _10358 = param_41.o[0];
                        _10359 = param_41.o[1];
                        _10360 = param_41.o[2];
                        _10066 = param_41.depth;
                        _10393 = param_41.d[0];
                        _10394 = param_41.d[1];
                        _10395 = param_41.d[2];
                        _10068 = param_41.dist;
                        _10428 = param_41.c[0];
                        _10429 = param_41.c[1];
                        _10430 = param_41.c[2];
                        _10070 = param_41.xy;
                        col += _8381;
                    }
                    bool _8388 = _7661 < _3537_g_params.max_refr_depth;
                    bool _8395;
                    if (_8388)
                    {
                        _8395 = _7672 < _3537_g_params.max_total_depth;
                    }
                    else
                    {
                        _8395 = _8388;
                    }
                    [branch]
                    if (_8395)
                    {
                        surface_t _9791 = { _7049, _9745, _9746, _9747, _9748, _7390 };
                        float _10343[3] = { _10323, _10324, _10325 };
                        float _10308[3] = { _10288, _10289, _10290 };
                        float _10273[3] = { _10253, _10254, _10255 };
                        ray_data_t _10017 = { _10273, _10308, _9959, _10343, _9961, _9962, _9963, _9964, _9965 };
                        ray_data_t param_42 = _10017;
                        Sample_RefractiveNode(ray, _9791, base_color, roughness, _7082, _9812, _7646, _8167, _8176, mix_weight, param_42);
                        _10253 = param_42.o[0];
                        _10254 = param_42.o[1];
                        _10255 = param_42.o[2];
                        _10288 = param_42.d[0];
                        _10289 = param_42.d[1];
                        _10290 = param_42.d[2];
                        _9959 = param_42.pdf;
                        _10323 = param_42.c[0];
                        _10324 = param_42.c[1];
                        _10325 = param_42.c[2];
                        _9961 = param_42.ior;
                        _9962 = param_42.cone_width;
                        _9963 = param_42.cone_spread;
                        _9964 = param_42.xy;
                        _9965 = param_42.depth;
                    }
                }
                else
                {
                    [branch]
                    if (_9809 == 3u)
                    {
                        col += (base_color * (mix_weight * _9810));
                    }
                    else
                    {
                        [branch]
                        if (_9809 == 6u)
                        {
                            float metallic = clamp(float((_9814 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_10201 != 4294967295u)
                            {
                                metallic *= SampleBilinear(_10201, _7390, int(get_texture_lod(texSize(_10201), _7623))).x;
                            }
                            float specular = clamp(float(_9816 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_10202 != 4294967295u)
                            {
                                specular *= SampleBilinear(_10202, _7390, int(get_texture_lod(texSize(_10202), _7623))).x;
                            }
                            float _8514 = clamp(float(_9817 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8522 = clamp(float((_9817 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8530 = 2.0f * clamp(float(_9813 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float3 _8548 = lerp(1.0f.xxx, tint_color, clamp(float((_9813 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _8530;
                            float3 _8568 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_9816 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                            float _8577 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                            float param_43 = 1.0f;
                            float param_44 = _8577;
                            float _8583 = fresnel_dielectric_cos(param_43, param_44);
                            float _8591 = clamp(float((_9811 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8602 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _8514))) - 1.0f;
                            float param_45 = 1.0f;
                            float param_46 = _8602;
                            float _8608 = fresnel_dielectric_cos(param_45, param_46);
                            float _8623 = mad(roughness - 1.0f, 1.0f - clamp(float((_9815 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                            float _8629;
                            if (_7082)
                            {
                                _8629 = _9812 / _7646;
                            }
                            else
                            {
                                _8629 = _7646 / _9812;
                            }
                            float param_47 = dot(_7013, _9747);
                            float param_48 = 1.0f / _8629;
                            float _8652 = fresnel_dielectric_cos(param_47, param_48);
                            float param_49 = dot(_7013, _9747);
                            float param_50 = _8577;
                            lobe_weights_t _8691 = get_lobe_weights(lerp(_8122, 1.0f, _8530), lum(lerp(_8568, 1.0f.xxx, ((fresnel_dielectric_cos(param_49, param_50) - _8583) / (1.0f - _8583)).xxx)), specular, metallic, clamp(float(_9815 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _8514);
                            [branch]
                            if (_9901 > 0.0f)
                            {
                                light_sample_t _9956 = { _9896, _9897, _9898, _9899, _9900, _9901, _9902, _9903 };
                                surface_t _9798 = { _7049, _9745, _9746, _9747, _9748, _7390 };
                                diff_params_t _10148 = { base_color, _8548, roughness };
                                spec_params_t _10163 = { _8568, roughness, _8577, _8583, _8591 };
                                clearcoat_params_t _10176 = { _8522, _8602, _8608 };
                                transmission_params_t _10191 = { _8623, _9812, _8629, _8652, _7082 };
                                float _10455[3] = { _10428, _10429, _10430 };
                                float _10420[3] = { _10393, _10394, _10395 };
                                float _10385[3] = { _10358, _10359, _10360 };
                                shadow_ray_t _10119 = { _10385, _10066, _10420, _10068, _10455, _10070 };
                                shadow_ray_t param_51 = _10119;
                                float3 _8710 = Evaluate_PrincipledNode(_9956, ray, _9798, _8691, _10148, _10163, _10176, _10191, metallic, _8089, mix_weight, param_51);
                                _10358 = param_51.o[0];
                                _10359 = param_51.o[1];
                                _10360 = param_51.o[2];
                                _10066 = param_51.depth;
                                _10393 = param_51.d[0];
                                _10394 = param_51.d[1];
                                _10395 = param_51.d[2];
                                _10068 = param_51.dist;
                                _10428 = param_51.c[0];
                                _10429 = param_51.c[1];
                                _10430 = param_51.c[2];
                                _10070 = param_51.xy;
                                col += _8710;
                            }
                            surface_t _9805 = { _7049, _9745, _9746, _9747, _9748, _7390 };
                            diff_params_t _10152 = { base_color, _8548, roughness };
                            spec_params_t _10169 = { _8568, roughness, _8577, _8583, _8591 };
                            clearcoat_params_t _10180 = { _8522, _8602, _8608 };
                            transmission_params_t _10197 = { _8623, _9812, _8629, _8652, _7082 };
                            float param_52 = mix_rand;
                            float _10350[3] = { _10323, _10324, _10325 };
                            float _10315[3] = { _10288, _10289, _10290 };
                            float _10280[3] = { _10253, _10254, _10255 };
                            ray_data_t _10036 = { _10280, _10315, _9959, _10350, _9961, _9962, _9963, _9964, _9965 };
                            ray_data_t param_53 = _10036;
                            Sample_PrincipledNode(ray, _9805, _8691, _10152, _10169, _10180, _10197, metallic, _8167, _8176, param_52, mix_weight, param_53);
                            _10253 = param_53.o[0];
                            _10254 = param_53.o[1];
                            _10255 = param_53.o[2];
                            _10288 = param_53.d[0];
                            _10289 = param_53.d[1];
                            _10290 = param_53.d[2];
                            _9959 = param_53.pdf;
                            _10323 = param_53.c[0];
                            _10324 = param_53.c[1];
                            _10325 = param_53.c[2];
                            _9961 = param_53.ior;
                            _9962 = param_53.cone_width;
                            _9963 = param_53.cone_spread;
                            _9964 = param_53.xy;
                            _9965 = param_53.depth;
                        }
                    }
                }
            }
        }
        float _8744 = max(_10323, max(_10324, _10325));
        float _8756;
        if (_7672 > _3537_g_params.min_total_depth)
        {
            _8756 = max(0.0500000007450580596923828125f, 1.0f - _8744);
        }
        else
        {
            _8756 = 0.0f;
        }
        bool _8770 = (frac(asfloat(_3521.Load((_7680 + 6) * 4 + 0)) + _7630) >= _8756) && (_8744 > 0.0f);
        bool _8776;
        if (_8770)
        {
            _8776 = _9959 > 0.0f;
        }
        else
        {
            _8776 = _8770;
        }
        [branch]
        if (_8776)
        {
            float _8780 = _9959;
            float _8781 = min(_8780, 1000000.0f);
            _9959 = _8781;
            float _8784 = 1.0f - _8756;
            float _8786 = _10323;
            float _8787 = _8786 / _8784;
            _10323 = _8787;
            float _8792 = _10324;
            float _8793 = _8792 / _8784;
            _10324 = _8793;
            float _8798 = _10325;
            float _8799 = _8798 / _8784;
            _10325 = _8799;
            uint _8807;
            _8805.InterlockedAdd(0, 1u, _8807);
            _8816.Store(_8807 * 72 + 0, asuint(_10253));
            _8816.Store(_8807 * 72 + 4, asuint(_10254));
            _8816.Store(_8807 * 72 + 8, asuint(_10255));
            _8816.Store(_8807 * 72 + 12, asuint(_10288));
            _8816.Store(_8807 * 72 + 16, asuint(_10289));
            _8816.Store(_8807 * 72 + 20, asuint(_10290));
            _8816.Store(_8807 * 72 + 24, asuint(_8781));
            _8816.Store(_8807 * 72 + 28, asuint(_8787));
            _8816.Store(_8807 * 72 + 32, asuint(_8793));
            _8816.Store(_8807 * 72 + 36, asuint(_8799));
            _8816.Store(_8807 * 72 + 40, asuint(_9961[0]));
            _8816.Store(_8807 * 72 + 44, asuint(_9961[1]));
            _8816.Store(_8807 * 72 + 48, asuint(_9961[2]));
            _8816.Store(_8807 * 72 + 52, asuint(_9961[3]));
            _8816.Store(_8807 * 72 + 56, asuint(_9962));
            _8816.Store(_8807 * 72 + 60, asuint(_9963));
            _8816.Store(_8807 * 72 + 64, uint(_9964));
            _8816.Store(_8807 * 72 + 68, uint(_9965));
        }
        [branch]
        if (max(_10428, max(_10429, _10430)) > 0.0f)
        {
            float3 _8893 = _9898 - float3(_10358, _10359, _10360);
            float _8896 = length(_8893);
            float3 _8900 = _8893 / _8896.xxx;
            float sh_dist = _8896 * _9900;
            if (_9903)
            {
                sh_dist = -sh_dist;
            }
            float _8912 = _8900.x;
            _10393 = _8912;
            float _8915 = _8900.y;
            _10394 = _8915;
            float _8918 = _8900.z;
            _10395 = _8918;
            _10068 = sh_dist;
            uint _8924;
            _8805.InterlockedAdd(8, 1u, _8924);
            _8932.Store(_8924 * 48 + 0, asuint(_10358));
            _8932.Store(_8924 * 48 + 4, asuint(_10359));
            _8932.Store(_8924 * 48 + 8, asuint(_10360));
            _8932.Store(_8924 * 48 + 12, uint(_10066));
            _8932.Store(_8924 * 48 + 16, asuint(_8912));
            _8932.Store(_8924 * 48 + 20, asuint(_8915));
            _8932.Store(_8924 * 48 + 24, asuint(_8918));
            _8932.Store(_8924 * 48 + 28, asuint(sh_dist));
            _8932.Store(_8924 * 48 + 32, asuint(_10428));
            _8932.Store(_8924 * 48 + 36, asuint(_10429));
            _8932.Store(_8924 * 48 + 40, asuint(_10430));
            _8932.Store(_8924 * 48 + 44, uint(_10070));
        }
        _9143 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _9143;
}

void comp_main()
{
    do
    {
        bool _8996 = gl_GlobalInvocationID.x >= _3537_g_params.img_size.x;
        bool _9005;
        if (!_8996)
        {
            _9005 = gl_GlobalInvocationID.y >= _3537_g_params.img_size.y;
        }
        else
        {
            _9005 = _8996;
        }
        if (_9005)
        {
            break;
        }
        int _9012 = int(gl_GlobalInvocationID.x);
        int _9016 = int(gl_GlobalInvocationID.y);
        int _9024 = (_9016 * int(_3537_g_params.img_size.x)) + _9012;
        hit_data_t _9034;
        _9034.mask = int(_9030.Load(_9024 * 24 + 0));
        _9034.obj_index = int(_9030.Load(_9024 * 24 + 4));
        _9034.prim_index = int(_9030.Load(_9024 * 24 + 8));
        _9034.t = asfloat(_9030.Load(_9024 * 24 + 12));
        _9034.u = asfloat(_9030.Load(_9024 * 24 + 16));
        _9034.v = asfloat(_9030.Load(_9024 * 24 + 20));
        ray_data_t _9054;
        [unroll]
        for (int _112ident = 0; _112ident < 3; _112ident++)
        {
            _9054.o[_112ident] = asfloat(_9051.Load(_112ident * 4 + _9024 * 72 + 0));
        }
        [unroll]
        for (int _113ident = 0; _113ident < 3; _113ident++)
        {
            _9054.d[_113ident] = asfloat(_9051.Load(_113ident * 4 + _9024 * 72 + 12));
        }
        _9054.pdf = asfloat(_9051.Load(_9024 * 72 + 24));
        [unroll]
        for (int _114ident = 0; _114ident < 3; _114ident++)
        {
            _9054.c[_114ident] = asfloat(_9051.Load(_114ident * 4 + _9024 * 72 + 28));
        }
        [unroll]
        for (int _115ident = 0; _115ident < 4; _115ident++)
        {
            _9054.ior[_115ident] = asfloat(_9051.Load(_115ident * 4 + _9024 * 72 + 40));
        }
        _9054.cone_width = asfloat(_9051.Load(_9024 * 72 + 56));
        _9054.cone_spread = asfloat(_9051.Load(_9024 * 72 + 60));
        _9054.xy = int(_9051.Load(_9024 * 72 + 64));
        _9054.depth = int(_9051.Load(_9024 * 72 + 68));
        hit_data_t _9237 = { _9034.mask, _9034.obj_index, _9034.prim_index, _9034.t, _9034.u, _9034.v };
        hit_data_t param = _9237;
        float _9286[4] = { _9054.ior[0], _9054.ior[1], _9054.ior[2], _9054.ior[3] };
        float _9277[3] = { _9054.c[0], _9054.c[1], _9054.c[2] };
        float _9270[3] = { _9054.d[0], _9054.d[1], _9054.d[2] };
        float _9263[3] = { _9054.o[0], _9054.o[1], _9054.o[2] };
        ray_data_t _9256 = { _9263, _9270, _9054.pdf, _9277, _9286, _9054.cone_width, _9054.cone_spread, _9054.xy, _9054.depth };
        ray_data_t param_1 = _9256;
        float3 _9104 = ShadeSurface(param, param_1);
        g_out_img[int2(_9012, _9016)] = float4(_9104, 1.0f);
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
