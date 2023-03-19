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
ByteAddressBuffer _7188 : register(t12, space0);
RWByteAddressBuffer _8924 : register(u3, space0);
RWByteAddressBuffer _8935 : register(u1, space0);
RWByteAddressBuffer _9051 : register(u2, space0);
ByteAddressBuffer _9130 : register(t5, space0);
ByteAddressBuffer _9147 : register(t4, space0);
ByteAddressBuffer _9257 : register(t8, space0);
cbuffer UniformParams
{
    Params _3537_g_params : packoffset(c0);
};

Texture2DArray<float4> g_atlases[7] : register(t21, space0);
SamplerState _g_atlases_sampler[7] : register(s21, space0);
Texture2D<float4> g_env_qtree : register(t16, space0);
SamplerState _g_env_qtree_sampler : register(s16, space0);
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
    uint _9432[14] = t.pos;
    uint _9435[14] = t.pos;
    uint _1094 = t.size & 16383u;
    uint _1097 = t.size >> uint(16);
    uint _1098 = _1097 & 16383u;
    float2 size = float2(float(_1094), float(_1098));
    if ((_1097 & 32768u) != 0u)
    {
        size = float2(float(_1094 >> uint(mip_level)), float(_1098 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_9432[mip_level] & 65535u), float((_9435[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
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
    float3 _5018;
    if ((ray.depth & 16777215) != 0)
    {
        _5018 = _3537_g_params.env_col.xyz;
    }
    else
    {
        _5018 = _3537_g_params.back_col.xyz;
    }
    float3 env_col = _5018;
    uint _5034;
    if ((ray.depth & 16777215) != 0)
    {
        _5034 = asuint(_3537_g_params.env_col.w);
    }
    else
    {
        _5034 = asuint(_3537_g_params.back_col.w);
    }
    float _5050;
    if ((ray.depth & 16777215) != 0)
    {
        _5050 = _3537_g_params.env_rotation;
    }
    else
    {
        _5050 = _3537_g_params.back_rotation;
    }
    if (_5034 != 4294967295u)
    {
        atlas_texture_t _5066;
        _5066.size = _1001.Load(_5034 * 80 + 0);
        _5066.atlas = _1001.Load(_5034 * 80 + 4);
        [unroll]
        for (int _58ident = 0; _58ident < 4; _58ident++)
        {
            _5066.page[_58ident] = _1001.Load(_58ident * 4 + _5034 * 80 + 8);
        }
        [unroll]
        for (int _59ident = 0; _59ident < 14; _59ident++)
        {
            _5066.pos[_59ident] = _1001.Load(_59ident * 4 + _5034 * 80 + 24);
        }
        uint _9802[14] = { _5066.pos[0], _5066.pos[1], _5066.pos[2], _5066.pos[3], _5066.pos[4], _5066.pos[5], _5066.pos[6], _5066.pos[7], _5066.pos[8], _5066.pos[9], _5066.pos[10], _5066.pos[11], _5066.pos[12], _5066.pos[13] };
        uint _9773[4] = { _5066.page[0], _5066.page[1], _5066.page[2], _5066.page[3] };
        atlas_texture_t _9764 = { _5066.size, _5066.atlas, _9773, _9802 };
        float param = _5050;
        env_col *= SampleLatlong_RGBE(_9764, _5011, param);
    }
    if (_3537_g_params.env_qtree_levels > 0)
    {
        float param_1 = ray.pdf;
        float param_2 = Evaluate_EnvQTree(_5050, g_env_qtree, _g_env_qtree_sampler, _3537_g_params.env_qtree_levels, _5011);
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
    float3 _5178 = float3(ray.d[0], ray.d[1], ray.d[2]);
    light_t _5192;
    _5192.type_and_param0 = _3557.Load4(((-1) - inter.obj_index) * 64 + 0);
    _5192.param1 = asfloat(_3557.Load4(((-1) - inter.obj_index) * 64 + 16));
    _5192.param2 = asfloat(_3557.Load4(((-1) - inter.obj_index) * 64 + 32));
    _5192.param3 = asfloat(_3557.Load4(((-1) - inter.obj_index) * 64 + 48));
    float3 lcol = asfloat(_5192.type_and_param0.yzw);
    [branch]
    if ((_5192.type_and_param0.x & 128u) != 0u)
    {
        float3 env_col = _3537_g_params.env_col.xyz;
        uint _5219 = asuint(_3537_g_params.env_col.w);
        if (_5219 != 4294967295u)
        {
            atlas_texture_t _5226;
            _5226.size = _1001.Load(_5219 * 80 + 0);
            _5226.atlas = _1001.Load(_5219 * 80 + 4);
            [unroll]
            for (int _60ident = 0; _60ident < 4; _60ident++)
            {
                _5226.page[_60ident] = _1001.Load(_60ident * 4 + _5219 * 80 + 8);
            }
            [unroll]
            for (int _61ident = 0; _61ident < 14; _61ident++)
            {
                _5226.pos[_61ident] = _1001.Load(_61ident * 4 + _5219 * 80 + 24);
            }
            uint _9864[14] = { _5226.pos[0], _5226.pos[1], _5226.pos[2], _5226.pos[3], _5226.pos[4], _5226.pos[5], _5226.pos[6], _5226.pos[7], _5226.pos[8], _5226.pos[9], _5226.pos[10], _5226.pos[11], _5226.pos[12], _5226.pos[13] };
            uint _9835[4] = { _5226.page[0], _5226.page[1], _5226.page[2], _5226.page[3] };
            atlas_texture_t _9826 = { _5226.size, _5226.atlas, _9835, _9864 };
            float param = _3537_g_params.env_rotation;
            env_col *= SampleLatlong_RGBE(_9826, _5178, param);
        }
        lcol *= env_col;
    }
    uint _5286 = _5192.type_and_param0.x & 31u;
    if (_5286 == 0u)
    {
        float param_1 = ray.pdf;
        float param_2 = (inter.t * inter.t) / ((0.5f * _5192.param1.w) * dot(_5178, normalize(_5192.param1.xyz - (float3(ray.o[0], ray.o[1], ray.o[2]) + (_5178 * inter.t)))));
        lcol *= power_heuristic(param_1, param_2);
        bool _5353 = _5192.param3.x > 0.0f;
        bool _5359;
        if (_5353)
        {
            _5359 = _5192.param3.y > 0.0f;
        }
        else
        {
            _5359 = _5353;
        }
        [branch]
        if (_5359)
        {
            [flatten]
            if (_5192.param3.y > 0.0f)
            {
                lcol *= clamp((_5192.param3.x - acos(clamp(-dot(_5178, _5192.param2.xyz), 0.0f, 1.0f))) / _5192.param3.y, 0.0f, 1.0f);
            }
        }
    }
    else
    {
        if (_5286 == 4u)
        {
            float param_3 = ray.pdf;
            float param_4 = (inter.t * inter.t) / (_5192.param1.w * dot(_5178, normalize(cross(_5192.param2.xyz, _5192.param3.xyz))));
            lcol *= power_heuristic(param_3, param_4);
        }
        else
        {
            if (_5286 == 5u)
            {
                float param_5 = ray.pdf;
                float param_6 = (inter.t * inter.t) / (_5192.param1.w * dot(_5178, normalize(cross(_5192.param2.xyz, _5192.param3.xyz))));
                lcol *= power_heuristic(param_5, param_6);
            }
            else
            {
                if (_5286 == 3u)
                {
                    float param_7 = ray.pdf;
                    float param_8 = (inter.t * inter.t) / (_5192.param1.w * (1.0f - abs(dot(_5178, _5192.param3.xyz))));
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
    float _9269;
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
            _9269 = stack[3];
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
            _9269 = stack[2];
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
            _9269 = stack[1];
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
            _9269 = stack[0];
            break;
        }
        _9269 = default_value;
        break;
    } while(false);
    return _9269;
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
    uint _9440[4];
    _9440[0] = _1131.page[0];
    _9440[1] = _1131.page[1];
    _9440[2] = _1131.page[2];
    _9440[3] = _1131.page[3];
    uint _9476[14] = { _1131.pos[0], _1131.pos[1], _1131.pos[2], _1131.pos[3], _1131.pos[4], _1131.pos[5], _1131.pos[6], _1131.pos[7], _1131.pos[8], _1131.pos[9], _1131.pos[10], _1131.pos[11], _1131.pos[12], _1131.pos[13] };
    atlas_texture_t _9446 = { _1131.size, _1131.atlas, _9440, _9476 };
    uint _1201 = _1131.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_1201)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_1201)], float3(TransformUV(uvs, _9446, lod) * 0.000118371215648949146270751953125f.xx, float((_9440[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
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
        float4 _10628 = res;
        _10628.x = _1241.x;
        float4 _10630 = _10628;
        _10630.y = _1241.y;
        float4 _10632 = _10630;
        _10632.z = _1241.z;
        res = _10632;
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
    float3 _9274;
    do
    {
        float _1510 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1510)
        {
            _9274 = N;
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
            float _10931 = (-0.5f) / _1550;
            float param_1 = mad(_10931, _1574, 1.0f);
            float _1606 = safe_sqrtf(param_1);
            float param_2 = _1575;
            float _1609 = safe_sqrtf(param_2);
            float2 _1610 = float2(_1606, _1609);
            float param_3 = mad(_10931, _1581, 1.0f);
            float _1615 = safe_sqrtf(param_3);
            float param_4 = _1582;
            float _1618 = safe_sqrtf(param_4);
            float2 _1619 = float2(_1615, _1618);
            float _10933 = -_1538;
            float _1635 = mad(2.0f * mad(_1606, _1534, _1609 * _1538), _1609, _10933);
            float _1651 = mad(2.0f * mad(_1615, _1534, _1618 * _1538), _1618, _10933);
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
                _9274 = Ng;
                break;
            }
            float _1688 = valid1 ? _1575 : _1582;
            float param_5 = 1.0f - _1688;
            float param_6 = _1688;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _9274 = (_1530 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _9274;
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
    float3 _9299;
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
            _9299 = N;
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
        _9299 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _9299;
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
            float2 _10615 = origin;
            _10615.x = origin.x + _step;
            origin = _10615;
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
            float2 _10618 = origin;
            _10618.y = origin.y + _step;
            origin = _10618;
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
        float3 _10695 = sampled_dir;
        float3 _3680 = ((param * _10695.x) + (param_1 * _10695.y)) + (_3637 * _10695.z);
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
                        uint _9620[14] = { _3968.pos[0], _3968.pos[1], _3968.pos[2], _3968.pos[3], _3968.pos[4], _3968.pos[5], _3968.pos[6], _3968.pos[7], _3968.pos[8], _3968.pos[9], _3968.pos[10], _3968.pos[11], _3968.pos[12], _3968.pos[13] };
                        uint _9591[4] = { _3968.page[0], _3968.page[1], _3968.page[2], _3968.page[3] };
                        atlas_texture_t _9520 = { _3968.size, _3968.atlas, _9591, _9620 };
                        float param_10 = _3537_g_params.env_rotation;
                        env_col *= SampleLatlong_RGBE(_9520, ls.L, param_10);
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
                            uint _9658[14] = { _4210.pos[0], _4210.pos[1], _4210.pos[2], _4210.pos[3], _4210.pos[4], _4210.pos[5], _4210.pos[6], _4210.pos[7], _4210.pos[8], _4210.pos[9], _4210.pos[10], _4210.pos[11], _4210.pos[12], _4210.pos[13] };
                            uint _9629[4] = { _4210.page[0], _4210.page[1], _4210.page[2], _4210.page[3] };
                            atlas_texture_t _9529 = { _4210.size, _4210.atlas, _9629, _9658 };
                            float param_13 = _3537_g_params.env_rotation;
                            env_col_1 *= SampleLatlong_RGBE(_9529, ls.L, param_13);
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
                                    uint _9743[14] = { _4936.pos[0], _4936.pos[1], _4936.pos[2], _4936.pos[3], _4936.pos[4], _4936.pos[5], _4936.pos[6], _4936.pos[7], _4936.pos[8], _4936.pos[9], _4936.pos[10], _4936.pos[11], _4936.pos[12], _4936.pos[13] };
                                    uint _9714[4] = { _4936.page[0], _4936.page[1], _4936.page[2], _4936.page[3] };
                                    atlas_texture_t _9582 = { _4936.size, _4936.atlas, _9714, _9743 };
                                    float param_20 = _3537_g_params.env_rotation;
                                    ls.col *= SampleLatlong_RGBE(_9582, ls.L, param_20);
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
    float3 _9279;
    do
    {
        float3 param = -float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param_1 = surf.N;
        float3 param_2 = ls.L;
        float param_3 = roughness;
        float3 param_4 = base_color;
        float4 _5556 = Evaluate_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_5 = ls.pdf;
            float param_6 = _5556.w;
            mis_weight = power_heuristic(param_5, param_6);
        }
        float3 _5579 = (ls.col * _5556.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9279 = _5579;
            break;
        }
        float3 param_7 = surf.P;
        float3 param_8 = surf.plane_N;
        float3 _5591 = offset_ray(param_7, param_8);
        sh_r.o[0] = _5591.x;
        sh_r.o[1] = _5591.y;
        sh_r.o[2] = _5591.z;
        sh_r.c[0] = ray.c[0] * _5579.x;
        sh_r.c[1] = ray.c[1] * _5579.y;
        sh_r.c[2] = ray.c[2] * _5579.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9279 = 0.0f.xxx;
        break;
    } while(false);
    return _9279;
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
    float4 _5842 = Sample_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8);
    new_ray.depth = ray.depth + 1;
    float3 param_9 = surf.P;
    float3 param_10 = surf.plane_N;
    float3 _5852 = offset_ray(param_9, param_10);
    new_ray.o[0] = _5852.x;
    new_ray.o[1] = _5852.y;
    new_ray.o[2] = _5852.z;
    new_ray.d[0] = param_8.x;
    new_ray.d[1] = param_8.y;
    new_ray.d[2] = param_8.z;
    new_ray.c[0] = ((ray.c[0] * _5842.x) * mix_weight) / _5842.w;
    new_ray.c[1] = ((ray.c[1] * _5842.y) * mix_weight) / _5842.w;
    new_ray.c[2] = ((ray.c[2] * _5842.z) * mix_weight) / _5842.w;
    new_ray.pdf = _5842.w;
}

float3 tangent_from_world(float3 T, float3 B, float3 N, float3 V)
{
    return float3(dot(V, T), dot(V, B), dot(V, N));
}

float D_GGX(float3 H, float alpha_x, float alpha_y)
{
    float _9332;
    do
    {
        if (H.z == 0.0f)
        {
            _9332 = 0.0f;
            break;
        }
        float _2242 = (-H.x) / (H.z * alpha_x);
        float _2248 = (-H.y) / (H.z * alpha_y);
        float _2257 = mad(_2248, _2248, mad(_2242, _2242, 1.0f));
        _9332 = 1.0f / (((((_2257 * _2257) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _9332;
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
    float3 _9284;
    do
    {
        float3 _5627 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5627;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - _5627);
        float _5665 = roughness * roughness;
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = _5665;
        float param_16 = _5665;
        float param_17 = spec_ior;
        float param_18 = spec_F0;
        float3 param_19 = base_color;
        float4 _5680 = Evaluate_GGXSpecular_BSDF(param_12, param_13, param_14, param_15, param_16, param_17, param_18, param_19);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_20 = ls.pdf;
            float param_21 = _5680.w;
            mis_weight = power_heuristic(param_20, param_21);
        }
        float3 _5703 = (ls.col * _5680.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9284 = _5703;
            break;
        }
        float3 param_22 = surf.P;
        float3 param_23 = surf.plane_N;
        float3 _5715 = offset_ray(param_22, param_23);
        sh_r.o[0] = _5715.x;
        sh_r.o[1] = _5715.y;
        sh_r.o[2] = _5715.z;
        sh_r.c[0] = ray.c[0] * _5703.x;
        sh_r.c[1] = ray.c[1] * _5703.y;
        sh_r.c[2] = ray.c[2] * _5703.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9284 = 0.0f.xxx;
        break;
    } while(false);
    return _9284;
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
    float4 _9304;
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
            _9304 = float4(_2897.x * 1000000.0f, _2897.y * 1000000.0f, _2897.z * 1000000.0f, 1000000.0f);
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
        _9304 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _9304;
}

void Sample_GlossyNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 param_4;
    float4 _5762 = Sample_GGXSpecular_BSDF(param, param_1, param_2, param_3, roughness, 0.0f, spec_ior, spec_F0, base_color, rand_u, rand_v, param_4);
    new_ray.depth = ray.depth + 256;
    float3 param_5 = surf.P;
    float3 param_6 = surf.plane_N;
    float3 _5773 = offset_ray(param_5, param_6);
    new_ray.o[0] = _5773.x;
    new_ray.o[1] = _5773.y;
    new_ray.o[2] = _5773.z;
    new_ray.d[0] = param_4.x;
    new_ray.d[1] = param_4.y;
    new_ray.d[2] = param_4.z;
    new_ray.c[0] = ((ray.c[0] * _5762.x) * mix_weight) / _5762.w;
    new_ray.c[1] = ((ray.c[1] * _5762.y) * mix_weight) / _5762.w;
    new_ray.c[2] = ((ray.c[2] * _5762.z) * mix_weight) / _5762.w;
    new_ray.pdf = _5762.w;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _9309;
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
            _9309 = 0.0f.xxxx;
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
        _9309 = float4(refr_col * (((((_3180 * _3196) * _3188) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3216) / view_dir_ts.z), (((_3180 * _3188) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3216) / view_dir_ts.z);
        break;
    } while(false);
    return _9309;
}

float3 Evaluate_RefractiveNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness2, float eta, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _9289;
    do
    {
        float3 _5905 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5905;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - (_5905 * eta));
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = roughness2;
        float param_16 = eta;
        float3 param_17 = base_color;
        float4 _5953 = Evaluate_GGXRefraction_BSDF(param_12, param_13, param_14, param_15, param_16, param_17);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_18 = ls.pdf;
            float param_19 = _5953.w;
            mis_weight = power_heuristic(param_18, param_19);
        }
        float3 _5976 = (ls.col * _5953.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _9289 = _5976;
            break;
        }
        float3 param_20 = surf.P;
        float3 param_21 = -surf.plane_N;
        float3 _5989 = offset_ray(param_20, param_21);
        sh_r.o[0] = _5989.x;
        sh_r.o[1] = _5989.y;
        sh_r.o[2] = _5989.z;
        sh_r.c[0] = ray.c[0] * _5976.x;
        sh_r.c[1] = ray.c[1] * _5976.y;
        sh_r.c[2] = ray.c[2] * _5976.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9289 = 0.0f.xxx;
        break;
    } while(false);
    return _9289;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _9314;
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
                _9314 = 0.0f.xxxx;
                break;
            }
            float _3293 = mad(eta, _3271, -sqrt(_3281));
            out_V = float4(normalize((I * eta) + (N * _3293)), _3293);
            _9314 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
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
            _9314 = 0.0f.xxxx;
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
        _9314 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _9314;
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
    float _9322;
    do
    {
        if (stack[3] > 0.0f)
        {
            float param = stack[3];
            float param_1 = -1.0f;
            float _2348 = exchange(param, param_1);
            stack[3] = param;
            _9322 = _2348;
            break;
        }
        if (stack[2] > 0.0f)
        {
            float param_2 = stack[2];
            float param_3 = -1.0f;
            float _2361 = exchange(param_2, param_3);
            stack[2] = param_2;
            _9322 = _2361;
            break;
        }
        if (stack[1] > 0.0f)
        {
            float param_4 = stack[1];
            float param_5 = -1.0f;
            float _2374 = exchange(param_4, param_5);
            stack[1] = param_4;
            _9322 = _2374;
            break;
        }
        if (stack[0] > 0.0f)
        {
            float param_6 = stack[0];
            float param_7 = -1.0f;
            float _2387 = exchange(param_6, param_7);
            stack[0] = param_6;
            _9322 = _2387;
            break;
        }
        _9322 = default_value;
        break;
    } while(false);
    return _9322;
}

void Sample_RefractiveNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, bool is_backfacing, float int_ior, float ext_ior, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float _6026;
    if (is_backfacing)
    {
        _6026 = int_ior / ext_ior;
    }
    else
    {
        _6026 = ext_ior / int_ior;
    }
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float param_4 = roughness;
    float param_5 = _6026;
    float3 param_6 = base_color;
    float param_7 = rand_u;
    float param_8 = rand_v;
    float4 param_9;
    float4 _6050 = Sample_GGXRefraction_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9);
    new_ray.depth = ray.depth + 65536;
    new_ray.c[0] = ((ray.c[0] * _6050.x) * mix_weight) / _6050.w;
    new_ray.c[1] = ((ray.c[1] * _6050.y) * mix_weight) / _6050.w;
    new_ray.c[2] = ((ray.c[2] * _6050.z) * mix_weight) / _6050.w;
    new_ray.pdf = _6050.w;
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
        float _6106 = pop_ior_stack(param_11, param_12);
        new_ray.ior = param_11;
    }
    float3 param_13 = surf.P;
    float3 param_14 = -surf.plane_N;
    float3 _6115 = offset_ray(param_13, param_14);
    new_ray.o[0] = _6115.x;
    new_ray.o[1] = _6115.y;
    new_ray.o[2] = _6115.z;
    new_ray.d[0] = param_9.x;
    new_ray.d[1] = param_9.y;
    new_ray.d[2] = param_9.z;
}

lobe_weights_t get_lobe_weights(float base_color_lum, float spec_color_lum, float specular, float metallic, float transmission, float clearcoat)
{
    float _1713 = 1.0f - metallic;
    float _9477 = (base_color_lum * _1713) * (1.0f - transmission);
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
    float _9478 = _1724;
    float _1734 = 0.25f * clearcoat;
    float _9479 = _1734 * _1713;
    float _9480 = _1720 * base_color_lum;
    float _1743 = _9477;
    float _1752 = mad(_1720, base_color_lum, mad(_1734, _1713, _1743 + _1724));
    if (_1752 != 0.0f)
    {
        _9477 /= _1752;
        _9478 /= _1752;
        _9479 /= _1752;
        _9480 /= _1752;
    }
    lobe_weights_t _9485 = { _9477, _9478, _9479, _9480 };
    return _9485;
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
    float _9337;
    do
    {
        float _2468 = dot(N, L);
        if (_2468 <= 0.0f)
        {
            _9337 = 0.0f;
            break;
        }
        float param = _2468;
        float param_1 = dot(N, V);
        float _2489 = dot(L, H);
        float _2497 = mad((2.0f * _2489) * _2489, roughness, 0.5f);
        _9337 = lerp(1.0f, _2497, schlick_weight(param)) * lerp(1.0f, _2497, schlick_weight(param_1));
        break;
    } while(false);
    return _9337;
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
    float _9342;
    do
    {
        if (a >= 1.0f)
        {
            _9342 = 0.3183098733425140380859375f;
            break;
        }
        float _2216 = mad(a, a, -1.0f);
        _9342 = _2216 / ((3.1415927410125732421875f * log(a * a)) * mad(_2216 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _9342;
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
    float3 _9294;
    do
    {
        float3 _6138 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 lcol = 0.0f.xxx;
        float bsdf_pdf = 0.0f;
        bool _6143 = N_dot_L > 0.0f;
        [branch]
        if ((lobe_weights.diffuse > 1.0000000116860974230803549289703e-07f) && _6143)
        {
            float3 param = -_6138;
            float3 param_1 = surf.N;
            float3 param_2 = ls.L;
            float param_3 = diff.roughness;
            float3 param_4 = diff.base_color;
            float3 param_5 = diff.sheen_color;
            bool param_6 = false;
            float4 _6162 = Evaluate_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6);
            bsdf_pdf = mad(lobe_weights.diffuse, _6162.w, bsdf_pdf);
            lcol += (((ls.col * N_dot_L) * (_6162 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * ls.pdf).xxx);
        }
        float3 H;
        [flatten]
        if (_6143)
        {
            H = normalize(ls.L - _6138);
        }
        else
        {
            H = normalize(ls.L - (_6138 * trans.eta));
        }
        float _6201 = spec.roughness * spec.roughness;
        float _6206 = sqrt(mad(-0.89999997615814208984375f, spec.anisotropy, 1.0f));
        float _6210 = _6201 / _6206;
        float _6214 = _6201 * _6206;
        float3 param_7 = surf.T;
        float3 param_8 = surf.B;
        float3 param_9 = surf.N;
        float3 param_10 = -_6138;
        float3 _6225 = tangent_from_world(param_7, param_8, param_9, param_10);
        float3 param_11 = surf.T;
        float3 param_12 = surf.B;
        float3 param_13 = surf.N;
        float3 param_14 = ls.L;
        float3 _6235 = tangent_from_world(param_11, param_12, param_13, param_14);
        float3 param_15 = surf.T;
        float3 param_16 = surf.B;
        float3 param_17 = surf.N;
        float3 param_18 = H;
        float3 _6245 = tangent_from_world(param_15, param_16, param_17, param_18);
        bool _6247 = lobe_weights.specular > 0.0f;
        bool _6254;
        if (_6247)
        {
            _6254 = (_6210 * _6214) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _6254 = _6247;
        }
        [branch]
        if (_6254 && _6143)
        {
            float3 param_19 = _6225;
            float3 param_20 = _6245;
            float3 param_21 = _6235;
            float param_22 = _6210;
            float param_23 = _6214;
            float param_24 = spec.ior;
            float param_25 = spec.F0;
            float3 param_26 = spec.tmp_col;
            float4 _6276 = Evaluate_GGXSpecular_BSDF(param_19, param_20, param_21, param_22, param_23, param_24, param_25, param_26);
            bsdf_pdf = mad(lobe_weights.specular, _6276.w, bsdf_pdf);
            lcol += ((ls.col * _6276.xyz) / ls.pdf.xxx);
        }
        float _6295 = coat.roughness * coat.roughness;
        bool _6297 = lobe_weights.clearcoat > 0.0f;
        bool _6304;
        if (_6297)
        {
            _6304 = (_6295 * _6295) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _6304 = _6297;
        }
        [branch]
        if (_6304 && _6143)
        {
            float3 param_27 = _6225;
            float3 param_28 = _6245;
            float3 param_29 = _6235;
            float param_30 = _6295;
            float param_31 = coat.ior;
            float param_32 = coat.F0;
            float4 _6322 = Evaluate_PrincipledClearcoat_BSDF(param_27, param_28, param_29, param_30, param_31, param_32);
            bsdf_pdf = mad(lobe_weights.clearcoat, _6322.w, bsdf_pdf);
            lcol += (((ls.col * 0.25f) * _6322.xyz) / ls.pdf.xxx);
        }
        [branch]
        if (lobe_weights.refraction > 0.0f)
        {
            bool _6344 = trans.fresnel != 0.0f;
            bool _6351;
            if (_6344)
            {
                _6351 = (_6201 * _6201) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _6351 = _6344;
            }
            [branch]
            if (_6351 && _6143)
            {
                float3 param_33 = _6225;
                float3 param_34 = _6245;
                float3 param_35 = _6235;
                float param_36 = _6201;
                float param_37 = _6201;
                float param_38 = 1.0f;
                float param_39 = 0.0f;
                float3 param_40 = 1.0f.xxx;
                float4 _6370 = Evaluate_GGXSpecular_BSDF(param_33, param_34, param_35, param_36, param_37, param_38, param_39, param_40);
                bsdf_pdf = mad(lobe_weights.refraction * trans.fresnel, _6370.w, bsdf_pdf);
                lcol += ((ls.col * _6370.xyz) * (trans.fresnel / ls.pdf));
            }
            float _6392 = trans.roughness * trans.roughness;
            bool _6394 = trans.fresnel != 1.0f;
            bool _6401;
            if (_6394)
            {
                _6401 = (_6392 * _6392) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _6401 = _6394;
            }
            [branch]
            if (_6401 && (N_dot_L < 0.0f))
            {
                float3 param_41 = _6225;
                float3 param_42 = _6245;
                float3 param_43 = _6235;
                float param_44 = _6392;
                float param_45 = trans.eta;
                float3 param_46 = diff.base_color;
                float4 _6419 = Evaluate_GGXRefraction_BSDF(param_41, param_42, param_43, param_44, param_45, param_46);
                float _6422 = 1.0f - trans.fresnel;
                bsdf_pdf = mad(lobe_weights.refraction * _6422, _6419.w, bsdf_pdf);
                lcol += ((ls.col * _6419.xyz) * (_6422 / ls.pdf));
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
            _9294 = lcol;
            break;
        }
        float3 _6462;
        if (N_dot_L < 0.0f)
        {
            _6462 = -surf.plane_N;
        }
        else
        {
            _6462 = surf.plane_N;
        }
        float3 param_49 = surf.P;
        float3 param_50 = _6462;
        float3 _6473 = offset_ray(param_49, param_50);
        sh_r.o[0] = _6473.x;
        sh_r.o[1] = _6473.y;
        sh_r.o[2] = _6473.z;
        sh_r.c[0] = ray.c[0] * lcol.x;
        sh_r.c[1] = ray.c[1] * lcol.y;
        sh_r.c[2] = ray.c[2] * lcol.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _9294 = 0.0f.xxx;
        break;
    } while(false);
    return _9294;
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
    float4 _9327;
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
            _9327 = float4(_3097, _3097, _3097, 1000000.0f);
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
        _9327 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _9327;
}

void Sample_PrincipledNode(ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float rand_u, float rand_v, inout float mix_rand, float mix_weight, inout ray_data_t new_ray)
{
    float3 _6508 = float3(ray.d[0], ray.d[1], ray.d[2]);
    int _6512 = ray.depth & 255;
    int _6516 = (ray.depth >> 8) & 255;
    int _6520 = (ray.depth >> 16) & 255;
    int _6531 = (_6512 + _6516) + _6520;
    [branch]
    if (mix_rand < lobe_weights.diffuse)
    {
        bool _6540 = _6512 < _3537_g_params.max_diff_depth;
        bool _6547;
        if (_6540)
        {
            _6547 = _6531 < _3537_g_params.max_total_depth;
        }
        else
        {
            _6547 = _6540;
        }
        if (_6547)
        {
            float3 param = surf.T;
            float3 param_1 = surf.B;
            float3 param_2 = surf.N;
            float3 param_3 = _6508;
            float param_4 = diff.roughness;
            float3 param_5 = diff.base_color;
            float3 param_6 = diff.sheen_color;
            bool param_7 = false;
            float param_8 = rand_u;
            float param_9 = rand_v;
            float3 param_10;
            float4 _6570 = Sample_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9, param_10);
            float3 _6575 = _6570.xyz * (1.0f - metallic);
            new_ray.depth = ray.depth + 1;
            float3 param_11 = surf.P;
            float3 param_12 = surf.plane_N;
            float3 _6590 = offset_ray(param_11, param_12);
            new_ray.o[0] = _6590.x;
            new_ray.o[1] = _6590.y;
            new_ray.o[2] = _6590.z;
            new_ray.d[0] = param_10.x;
            new_ray.d[1] = param_10.y;
            new_ray.d[2] = param_10.z;
            new_ray.c[0] = ((ray.c[0] * _6575.x) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[1] = ((ray.c[1] * _6575.y) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[2] = ((ray.c[2] * _6575.z) * mix_weight) / lobe_weights.diffuse;
            new_ray.pdf = _6570.w;
        }
    }
    else
    {
        float _6640 = lobe_weights.diffuse + lobe_weights.specular;
        [branch]
        if (mix_rand < _6640)
        {
            bool _6647 = _6516 < _3537_g_params.max_spec_depth;
            bool _6654;
            if (_6647)
            {
                _6654 = _6531 < _3537_g_params.max_total_depth;
            }
            else
            {
                _6654 = _6647;
            }
            if (_6654)
            {
                float3 param_13 = surf.T;
                float3 param_14 = surf.B;
                float3 param_15 = surf.N;
                float3 param_16 = _6508;
                float3 param_17;
                float4 _6673 = Sample_GGXSpecular_BSDF(param_13, param_14, param_15, param_16, spec.roughness, spec.anisotropy, spec.ior, spec.F0, spec.tmp_col, rand_u, rand_v, param_17);
                float _6678 = _6673.w * lobe_weights.specular;
                new_ray.depth = ray.depth + 256;
                new_ray.c[0] = ((ray.c[0] * _6673.x) * mix_weight) / _6678;
                new_ray.c[1] = ((ray.c[1] * _6673.y) * mix_weight) / _6678;
                new_ray.c[2] = ((ray.c[2] * _6673.z) * mix_weight) / _6678;
                new_ray.pdf = _6678;
                float3 param_18 = surf.P;
                float3 param_19 = surf.plane_N;
                float3 _6718 = offset_ray(param_18, param_19);
                new_ray.o[0] = _6718.x;
                new_ray.o[1] = _6718.y;
                new_ray.o[2] = _6718.z;
                new_ray.d[0] = param_17.x;
                new_ray.d[1] = param_17.y;
                new_ray.d[2] = param_17.z;
            }
        }
        else
        {
            float _6743 = _6640 + lobe_weights.clearcoat;
            [branch]
            if (mix_rand < _6743)
            {
                bool _6750 = _6516 < _3537_g_params.max_spec_depth;
                bool _6757;
                if (_6750)
                {
                    _6757 = _6531 < _3537_g_params.max_total_depth;
                }
                else
                {
                    _6757 = _6750;
                }
                if (_6757)
                {
                    float3 param_20 = surf.T;
                    float3 param_21 = surf.B;
                    float3 param_22 = surf.N;
                    float3 param_23 = _6508;
                    float param_24 = coat.roughness * coat.roughness;
                    float param_25 = coat.ior;
                    float param_26 = coat.F0;
                    float param_27 = rand_u;
                    float param_28 = rand_v;
                    float3 param_29;
                    float4 _6781 = Sample_PrincipledClearcoat_BSDF(param_20, param_21, param_22, param_23, param_24, param_25, param_26, param_27, param_28, param_29);
                    float _6786 = _6781.w * lobe_weights.clearcoat;
                    new_ray.depth = ray.depth + 256;
                    new_ray.c[0] = (((0.25f * ray.c[0]) * _6781.x) * mix_weight) / _6786;
                    new_ray.c[1] = (((0.25f * ray.c[1]) * _6781.y) * mix_weight) / _6786;
                    new_ray.c[2] = (((0.25f * ray.c[2]) * _6781.z) * mix_weight) / _6786;
                    new_ray.pdf = _6786;
                    float3 param_30 = surf.P;
                    float3 param_31 = surf.plane_N;
                    float3 _6829 = offset_ray(param_30, param_31);
                    new_ray.o[0] = _6829.x;
                    new_ray.o[1] = _6829.y;
                    new_ray.o[2] = _6829.z;
                    new_ray.d[0] = param_29.x;
                    new_ray.d[1] = param_29.y;
                    new_ray.d[2] = param_29.z;
                }
            }
            else
            {
                bool _6851 = mix_rand >= trans.fresnel;
                bool _6858;
                if (_6851)
                {
                    _6858 = _6520 < _3537_g_params.max_refr_depth;
                }
                else
                {
                    _6858 = _6851;
                }
                bool _6872;
                if (!_6858)
                {
                    bool _6864 = mix_rand < trans.fresnel;
                    bool _6871;
                    if (_6864)
                    {
                        _6871 = _6516 < _3537_g_params.max_spec_depth;
                    }
                    else
                    {
                        _6871 = _6864;
                    }
                    _6872 = _6871;
                }
                else
                {
                    _6872 = _6858;
                }
                bool _6879;
                if (_6872)
                {
                    _6879 = _6531 < _3537_g_params.max_total_depth;
                }
                else
                {
                    _6879 = _6872;
                }
                [branch]
                if (_6879)
                {
                    mix_rand -= _6743;
                    mix_rand /= lobe_weights.refraction;
                    float4 F;
                    float3 V;
                    [branch]
                    if (mix_rand < trans.fresnel)
                    {
                        float3 param_32 = surf.T;
                        float3 param_33 = surf.B;
                        float3 param_34 = surf.N;
                        float3 param_35 = _6508;
                        float3 param_36;
                        float4 _6909 = Sample_GGXSpecular_BSDF(param_32, param_33, param_34, param_35, spec.roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, rand_u, rand_v, param_36);
                        V = param_36;
                        F = _6909;
                        new_ray.depth = ray.depth + 256;
                        float3 param_37 = surf.P;
                        float3 param_38 = surf.plane_N;
                        float3 _6919 = offset_ray(param_37, param_38);
                        new_ray.o[0] = _6919.x;
                        new_ray.o[1] = _6919.y;
                        new_ray.o[2] = _6919.z;
                    }
                    else
                    {
                        float3 param_39 = surf.T;
                        float3 param_40 = surf.B;
                        float3 param_41 = surf.N;
                        float3 param_42 = _6508;
                        float param_43 = trans.roughness;
                        float param_44 = trans.eta;
                        float3 param_45 = diff.base_color;
                        float param_46 = rand_u;
                        float param_47 = rand_v;
                        float4 param_48;
                        float4 _6948 = Sample_GGXRefraction_BSDF(param_39, param_40, param_41, param_42, param_43, param_44, param_45, param_46, param_47, param_48);
                        F = _6948;
                        V = param_48.xyz;
                        new_ray.depth = ray.depth + 65536;
                        float3 param_49 = surf.P;
                        float3 param_50 = -surf.plane_N;
                        float3 _6961 = offset_ray(param_49, param_50);
                        new_ray.o[0] = _6961.x;
                        new_ray.o[1] = _6961.y;
                        new_ray.o[2] = _6961.z;
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
                            float _6987 = pop_ior_stack(param_52, param_53);
                            new_ray.ior = param_52;
                        }
                    }
                    float4 _10879 = F;
                    float _6993 = _10879.w * lobe_weights.refraction;
                    float4 _10881 = _10879;
                    _10881.w = _6993;
                    F = _10881;
                    new_ray.c[0] = ((ray.c[0] * _10879.x) * mix_weight) / _6993;
                    new_ray.c[1] = ((ray.c[1] * _10879.y) * mix_weight) / _6993;
                    new_ray.c[2] = ((ray.c[2] * _10879.z) * mix_weight) / _6993;
                    new_ray.pdf = _6993;
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
    float3 _9264;
    do
    {
        float3 _7049 = float3(ray.d[0], ray.d[1], ray.d[2]);
        [branch]
        if (inter.mask == 0)
        {
            ray_data_t param = ray;
            float3 _7058 = Evaluate_EnvColor(param);
            _9264 = float3(ray.c[0] * _7058.x, ray.c[1] * _7058.y, ray.c[2] * _7058.z);
            break;
        }
        float3 _7085 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_7049 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            ray_data_t param_1 = ray;
            hit_data_t param_2 = inter;
            float3 _7097 = Evaluate_LightColor(param_1, param_2);
            _9264 = float3(ray.c[0] * _7097.x, ray.c[1] * _7097.y, ray.c[2] * _7097.z);
            break;
        }
        bool _7118 = inter.prim_index < 0;
        int _7121;
        if (_7118)
        {
            _7121 = (-1) - inter.prim_index;
        }
        else
        {
            _7121 = inter.prim_index;
        }
        uint _7132 = uint(_7121);
        material_t _7140;
        [unroll]
        for (int _89ident = 0; _89ident < 5; _89ident++)
        {
            _7140.textures[_89ident] = _4757.Load(_89ident * 4 + ((_4761.Load(_7132 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
        }
        [unroll]
        for (int _90ident = 0; _90ident < 3; _90ident++)
        {
            _7140.base_color[_90ident] = asfloat(_4757.Load(_90ident * 4 + ((_4761.Load(_7132 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
        }
        _7140.flags = _4757.Load(((_4761.Load(_7132 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
        _7140.type = _4757.Load(((_4761.Load(_7132 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
        _7140.tangent_rotation_or_strength = asfloat(_4757.Load(((_4761.Load(_7132 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
        _7140.roughness_and_anisotropic = _4757.Load(((_4761.Load(_7132 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
        _7140.ior = asfloat(_4757.Load(((_4761.Load(_7132 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
        _7140.sheen_and_sheen_tint = _4757.Load(((_4761.Load(_7132 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
        _7140.tint_and_metallic = _4757.Load(((_4761.Load(_7132 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
        _7140.transmission_and_transmission_roughness = _4757.Load(((_4761.Load(_7132 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
        _7140.specular_and_specular_tint = _4757.Load(((_4761.Load(_7132 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
        _7140.clearcoat_and_clearcoat_roughness = _4757.Load(((_4761.Load(_7132 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
        _7140.normal_map_strength_unorm = _4757.Load(((_4761.Load(_7132 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
        uint _10328 = _7140.textures[0];
        uint _10329 = _7140.textures[1];
        uint _10330 = _7140.textures[2];
        uint _10331 = _7140.textures[3];
        uint _10332 = _7140.textures[4];
        float _10333 = _7140.base_color[0];
        float _10334 = _7140.base_color[1];
        float _10335 = _7140.base_color[2];
        uint _9929 = _7140.flags;
        uint _9930 = _7140.type;
        float _9931 = _7140.tangent_rotation_or_strength;
        uint _9932 = _7140.roughness_and_anisotropic;
        float _9933 = _7140.ior;
        uint _9934 = _7140.sheen_and_sheen_tint;
        uint _9935 = _7140.tint_and_metallic;
        uint _9936 = _7140.transmission_and_transmission_roughness;
        uint _9937 = _7140.specular_and_specular_tint;
        uint _9938 = _7140.clearcoat_and_clearcoat_roughness;
        uint _9939 = _7140.normal_map_strength_unorm;
        transform_t _7195;
        _7195.xform = asfloat(uint4x4(_4404.Load4(asuint(asfloat(_7188.Load(inter.obj_index * 32 + 12))) * 128 + 0), _4404.Load4(asuint(asfloat(_7188.Load(inter.obj_index * 32 + 12))) * 128 + 16), _4404.Load4(asuint(asfloat(_7188.Load(inter.obj_index * 32 + 12))) * 128 + 32), _4404.Load4(asuint(asfloat(_7188.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _7195.inv_xform = asfloat(uint4x4(_4404.Load4(asuint(asfloat(_7188.Load(inter.obj_index * 32 + 12))) * 128 + 64), _4404.Load4(asuint(asfloat(_7188.Load(inter.obj_index * 32 + 12))) * 128 + 80), _4404.Load4(asuint(asfloat(_7188.Load(inter.obj_index * 32 + 12))) * 128 + 96), _4404.Load4(asuint(asfloat(_7188.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _7202 = _7132 * 3u;
        vertex_t _7207;
        [unroll]
        for (int _91ident = 0; _91ident < 3; _91ident++)
        {
            _7207.p[_91ident] = asfloat(_4429.Load(_91ident * 4 + _4433.Load(_7202 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _92ident = 0; _92ident < 3; _92ident++)
        {
            _7207.n[_92ident] = asfloat(_4429.Load(_92ident * 4 + _4433.Load(_7202 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _93ident = 0; _93ident < 3; _93ident++)
        {
            _7207.b[_93ident] = asfloat(_4429.Load(_93ident * 4 + _4433.Load(_7202 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _94ident = 0; _94ident < 2; _94ident++)
        {
            [unroll]
            for (int _95ident = 0; _95ident < 2; _95ident++)
            {
                _7207.t[_94ident][_95ident] = asfloat(_4429.Load(_95ident * 4 + _94ident * 8 + _4433.Load(_7202 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _7253;
        [unroll]
        for (int _96ident = 0; _96ident < 3; _96ident++)
        {
            _7253.p[_96ident] = asfloat(_4429.Load(_96ident * 4 + _4433.Load((_7202 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _97ident = 0; _97ident < 3; _97ident++)
        {
            _7253.n[_97ident] = asfloat(_4429.Load(_97ident * 4 + _4433.Load((_7202 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _98ident = 0; _98ident < 3; _98ident++)
        {
            _7253.b[_98ident] = asfloat(_4429.Load(_98ident * 4 + _4433.Load((_7202 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _99ident = 0; _99ident < 2; _99ident++)
        {
            [unroll]
            for (int _100ident = 0; _100ident < 2; _100ident++)
            {
                _7253.t[_99ident][_100ident] = asfloat(_4429.Load(_100ident * 4 + _99ident * 8 + _4433.Load((_7202 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _7299;
        [unroll]
        for (int _101ident = 0; _101ident < 3; _101ident++)
        {
            _7299.p[_101ident] = asfloat(_4429.Load(_101ident * 4 + _4433.Load((_7202 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _102ident = 0; _102ident < 3; _102ident++)
        {
            _7299.n[_102ident] = asfloat(_4429.Load(_102ident * 4 + _4433.Load((_7202 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _103ident = 0; _103ident < 3; _103ident++)
        {
            _7299.b[_103ident] = asfloat(_4429.Load(_103ident * 4 + _4433.Load((_7202 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _104ident = 0; _104ident < 2; _104ident++)
        {
            [unroll]
            for (int _105ident = 0; _105ident < 2; _105ident++)
            {
                _7299.t[_104ident][_105ident] = asfloat(_4429.Load(_105ident * 4 + _104ident * 8 + _4433.Load((_7202 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _7345 = float3(_7207.p[0], _7207.p[1], _7207.p[2]);
        float3 _7353 = float3(_7253.p[0], _7253.p[1], _7253.p[2]);
        float3 _7361 = float3(_7299.p[0], _7299.p[1], _7299.p[2]);
        float _7368 = (1.0f - inter.u) - inter.v;
        float3 _7400 = normalize(((float3(_7207.n[0], _7207.n[1], _7207.n[2]) * _7368) + (float3(_7253.n[0], _7253.n[1], _7253.n[2]) * inter.u)) + (float3(_7299.n[0], _7299.n[1], _7299.n[2]) * inter.v));
        float3 _9868 = _7400;
        float2 _7426 = ((float2(_7207.t[0][0], _7207.t[0][1]) * _7368) + (float2(_7253.t[0][0], _7253.t[0][1]) * inter.u)) + (float2(_7299.t[0][0], _7299.t[0][1]) * inter.v);
        float3 _7442 = cross(_7353 - _7345, _7361 - _7345);
        float _7447 = length(_7442);
        float3 _9869 = _7442 / _7447.xxx;
        float3 _7484 = ((float3(_7207.b[0], _7207.b[1], _7207.b[2]) * _7368) + (float3(_7253.b[0], _7253.b[1], _7253.b[2]) * inter.u)) + (float3(_7299.b[0], _7299.b[1], _7299.b[2]) * inter.v);
        float3 _9867 = _7484;
        float3 _9866 = cross(_7484, _7400);
        if (_7118)
        {
            if ((_4761.Load(_7132 * 4 + 0) & 65535u) == 65535u)
            {
                _9264 = 0.0f.xxx;
                break;
            }
            material_t _7509;
            [unroll]
            for (int _106ident = 0; _106ident < 5; _106ident++)
            {
                _7509.textures[_106ident] = _4757.Load(_106ident * 4 + (_4761.Load(_7132 * 4 + 0) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _107ident = 0; _107ident < 3; _107ident++)
            {
                _7509.base_color[_107ident] = asfloat(_4757.Load(_107ident * 4 + (_4761.Load(_7132 * 4 + 0) & 16383u) * 76 + 20));
            }
            _7509.flags = _4757.Load((_4761.Load(_7132 * 4 + 0) & 16383u) * 76 + 32);
            _7509.type = _4757.Load((_4761.Load(_7132 * 4 + 0) & 16383u) * 76 + 36);
            _7509.tangent_rotation_or_strength = asfloat(_4757.Load((_4761.Load(_7132 * 4 + 0) & 16383u) * 76 + 40));
            _7509.roughness_and_anisotropic = _4757.Load((_4761.Load(_7132 * 4 + 0) & 16383u) * 76 + 44);
            _7509.ior = asfloat(_4757.Load((_4761.Load(_7132 * 4 + 0) & 16383u) * 76 + 48));
            _7509.sheen_and_sheen_tint = _4757.Load((_4761.Load(_7132 * 4 + 0) & 16383u) * 76 + 52);
            _7509.tint_and_metallic = _4757.Load((_4761.Load(_7132 * 4 + 0) & 16383u) * 76 + 56);
            _7509.transmission_and_transmission_roughness = _4757.Load((_4761.Load(_7132 * 4 + 0) & 16383u) * 76 + 60);
            _7509.specular_and_specular_tint = _4757.Load((_4761.Load(_7132 * 4 + 0) & 16383u) * 76 + 64);
            _7509.clearcoat_and_clearcoat_roughness = _4757.Load((_4761.Load(_7132 * 4 + 0) & 16383u) * 76 + 68);
            _7509.normal_map_strength_unorm = _4757.Load((_4761.Load(_7132 * 4 + 0) & 16383u) * 76 + 72);
            _10328 = _7509.textures[0];
            _10329 = _7509.textures[1];
            _10330 = _7509.textures[2];
            _10331 = _7509.textures[3];
            _10332 = _7509.textures[4];
            _10333 = _7509.base_color[0];
            _10334 = _7509.base_color[1];
            _10335 = _7509.base_color[2];
            _9929 = _7509.flags;
            _9930 = _7509.type;
            _9931 = _7509.tangent_rotation_or_strength;
            _9932 = _7509.roughness_and_anisotropic;
            _9933 = _7509.ior;
            _9934 = _7509.sheen_and_sheen_tint;
            _9935 = _7509.tint_and_metallic;
            _9936 = _7509.transmission_and_transmission_roughness;
            _9937 = _7509.specular_and_specular_tint;
            _9938 = _7509.clearcoat_and_clearcoat_roughness;
            _9939 = _7509.normal_map_strength_unorm;
            _9869 = -_9869;
            _9868 = -_9868;
            _9867 = -_9867;
            _9866 = -_9866;
        }
        float3 param_3 = _9869;
        float4x4 param_4 = _7195.inv_xform;
        _9869 = TransformNormal(param_3, param_4);
        float3 param_5 = _9868;
        float4x4 param_6 = _7195.inv_xform;
        _9868 = TransformNormal(param_5, param_6);
        float3 param_7 = _9867;
        float4x4 param_8 = _7195.inv_xform;
        _9867 = TransformNormal(param_7, param_8);
        float3 param_9 = _9866;
        float4x4 param_10 = _7195.inv_xform;
        _9869 = normalize(_9869);
        _9868 = normalize(_9868);
        _9867 = normalize(_9867);
        _9866 = normalize(TransformNormal(param_9, param_10));
        float _7649 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _7659 = mad(0.5f, log2(abs(mad(_7253.t[0][0] - _7207.t[0][0], _7299.t[0][1] - _7207.t[0][1], -((_7299.t[0][0] - _7207.t[0][0]) * (_7253.t[0][1] - _7207.t[0][1])))) / _7447), log2(_7649));
        uint param_11 = uint(hash(ray.xy));
        float _7666 = construct_float(param_11);
        uint param_12 = uint(hash(hash(ray.xy)));
        float _7673 = construct_float(param_12);
        float param_13[4] = ray.ior;
        bool param_14 = _7118;
        float param_15 = 1.0f;
        float _7682 = peek_ior_stack(param_13, param_14, param_15);
        float3 col = 0.0f.xxx;
        int _7687 = ray.depth & 255;
        int _7692 = (ray.depth >> 8) & 255;
        int _7697 = (ray.depth >> 16) & 255;
        int _7708 = (_7687 + _7692) + _7697;
        int _7716 = _3537_g_params.hi + ((_7708 + ((ray.depth >> 24) & 255)) * 7);
        float mix_rand = frac(asfloat(_3521.Load(_7716 * 4 + 0)) + _7666);
        float mix_weight = 1.0f;
        float _7753;
        float _7770;
        float _7796;
        float _7863;
        while (_9930 == 4u)
        {
            float mix_val = _9931;
            if (_10329 != 4294967295u)
            {
                mix_val *= SampleBilinear(_10329, _7426, 0).x;
            }
            if (_7118)
            {
                _7753 = _7682 / _9933;
            }
            else
            {
                _7753 = _9933 / _7682;
            }
            if (_9933 != 0.0f)
            {
                float param_16 = dot(_7049, _9868);
                float param_17 = _7753;
                _7770 = fresnel_dielectric_cos(param_16, param_17);
            }
            else
            {
                _7770 = 1.0f;
            }
            float _7785 = mix_val;
            float _7786 = _7785 * clamp(_7770, 0.0f, 1.0f);
            mix_val = _7786;
            if (mix_rand > _7786)
            {
                if ((_9929 & 2u) != 0u)
                {
                    _7796 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _7796 = 1.0f;
                }
                mix_weight *= _7796;
                material_t _7809;
                [unroll]
                for (int _108ident = 0; _108ident < 5; _108ident++)
                {
                    _7809.textures[_108ident] = _4757.Load(_108ident * 4 + _10331 * 76 + 0);
                }
                [unroll]
                for (int _109ident = 0; _109ident < 3; _109ident++)
                {
                    _7809.base_color[_109ident] = asfloat(_4757.Load(_109ident * 4 + _10331 * 76 + 20));
                }
                _7809.flags = _4757.Load(_10331 * 76 + 32);
                _7809.type = _4757.Load(_10331 * 76 + 36);
                _7809.tangent_rotation_or_strength = asfloat(_4757.Load(_10331 * 76 + 40));
                _7809.roughness_and_anisotropic = _4757.Load(_10331 * 76 + 44);
                _7809.ior = asfloat(_4757.Load(_10331 * 76 + 48));
                _7809.sheen_and_sheen_tint = _4757.Load(_10331 * 76 + 52);
                _7809.tint_and_metallic = _4757.Load(_10331 * 76 + 56);
                _7809.transmission_and_transmission_roughness = _4757.Load(_10331 * 76 + 60);
                _7809.specular_and_specular_tint = _4757.Load(_10331 * 76 + 64);
                _7809.clearcoat_and_clearcoat_roughness = _4757.Load(_10331 * 76 + 68);
                _7809.normal_map_strength_unorm = _4757.Load(_10331 * 76 + 72);
                _10328 = _7809.textures[0];
                _10329 = _7809.textures[1];
                _10330 = _7809.textures[2];
                _10331 = _7809.textures[3];
                _10332 = _7809.textures[4];
                _10333 = _7809.base_color[0];
                _10334 = _7809.base_color[1];
                _10335 = _7809.base_color[2];
                _9929 = _7809.flags;
                _9930 = _7809.type;
                _9931 = _7809.tangent_rotation_or_strength;
                _9932 = _7809.roughness_and_anisotropic;
                _9933 = _7809.ior;
                _9934 = _7809.sheen_and_sheen_tint;
                _9935 = _7809.tint_and_metallic;
                _9936 = _7809.transmission_and_transmission_roughness;
                _9937 = _7809.specular_and_specular_tint;
                _9938 = _7809.clearcoat_and_clearcoat_roughness;
                _9939 = _7809.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_9929 & 2u) != 0u)
                {
                    _7863 = 1.0f / mix_val;
                }
                else
                {
                    _7863 = 1.0f;
                }
                mix_weight *= _7863;
                material_t _7875;
                [unroll]
                for (int _110ident = 0; _110ident < 5; _110ident++)
                {
                    _7875.textures[_110ident] = _4757.Load(_110ident * 4 + _10332 * 76 + 0);
                }
                [unroll]
                for (int _111ident = 0; _111ident < 3; _111ident++)
                {
                    _7875.base_color[_111ident] = asfloat(_4757.Load(_111ident * 4 + _10332 * 76 + 20));
                }
                _7875.flags = _4757.Load(_10332 * 76 + 32);
                _7875.type = _4757.Load(_10332 * 76 + 36);
                _7875.tangent_rotation_or_strength = asfloat(_4757.Load(_10332 * 76 + 40));
                _7875.roughness_and_anisotropic = _4757.Load(_10332 * 76 + 44);
                _7875.ior = asfloat(_4757.Load(_10332 * 76 + 48));
                _7875.sheen_and_sheen_tint = _4757.Load(_10332 * 76 + 52);
                _7875.tint_and_metallic = _4757.Load(_10332 * 76 + 56);
                _7875.transmission_and_transmission_roughness = _4757.Load(_10332 * 76 + 60);
                _7875.specular_and_specular_tint = _4757.Load(_10332 * 76 + 64);
                _7875.clearcoat_and_clearcoat_roughness = _4757.Load(_10332 * 76 + 68);
                _7875.normal_map_strength_unorm = _4757.Load(_10332 * 76 + 72);
                _10328 = _7875.textures[0];
                _10329 = _7875.textures[1];
                _10330 = _7875.textures[2];
                _10331 = _7875.textures[3];
                _10332 = _7875.textures[4];
                _10333 = _7875.base_color[0];
                _10334 = _7875.base_color[1];
                _10335 = _7875.base_color[2];
                _9929 = _7875.flags;
                _9930 = _7875.type;
                _9931 = _7875.tangent_rotation_or_strength;
                _9932 = _7875.roughness_and_anisotropic;
                _9933 = _7875.ior;
                _9934 = _7875.sheen_and_sheen_tint;
                _9935 = _7875.tint_and_metallic;
                _9936 = _7875.transmission_and_transmission_roughness;
                _9937 = _7875.specular_and_specular_tint;
                _9938 = _7875.clearcoat_and_clearcoat_roughness;
                _9939 = _7875.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_10328 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_10328, _7426, 0).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_1001.Load(_10328 * 80 + 0) & 16384u) != 0u)
            {
                float3 _10900 = normals;
                _10900.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _10900;
            }
            float3 _7959 = _9868;
            _9868 = normalize(((_9866 * normals.x) + (_7959 * normals.z)) + (_9867 * normals.y));
            if ((_9939 & 65535u) != 65535u)
            {
                _9868 = normalize(_7959 + ((_9868 - _7959) * clamp(float(_9939 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_18 = _9869;
            float3 param_19 = -_7049;
            float3 param_20 = _9868;
            _9868 = ensure_valid_reflection(param_18, param_19, param_20);
        }
        float3 _8025 = ((_7345 * _7368) + (_7353 * inter.u)) + (_7361 * inter.v);
        float3 _8032 = float3(-_8025.z, 0.0f, _8025.x);
        float3 tangent = _8032;
        float3 param_21 = _8032;
        float4x4 param_22 = _7195.inv_xform;
        float3 _8038 = TransformNormal(param_21, param_22);
        tangent = _8038;
        float3 _8042 = cross(_8038, _9868);
        if (dot(_8042, _8042) == 0.0f)
        {
            float3 param_23 = _8025;
            float4x4 param_24 = _7195.inv_xform;
            tangent = TransformNormal(param_23, param_24);
        }
        if (_9931 != 0.0f)
        {
            float3 param_25 = tangent;
            float3 param_26 = _9868;
            float param_27 = _9931;
            tangent = rotate_around_axis(param_25, param_26, param_27);
        }
        float3 _8075 = normalize(cross(tangent, _9868));
        _9867 = _8075;
        _9866 = cross(_9868, _8075);
        float3 _10027 = 0.0f.xxx;
        float3 _10026 = 0.0f.xxx;
        float _10031 = 0.0f;
        float _10029 = 0.0f;
        float _10030 = 1.0f;
        bool _8091 = _3537_g_params.li_count != 0;
        bool _8097;
        if (_8091)
        {
            _8097 = _9930 != 3u;
        }
        else
        {
            _8097 = _8091;
        }
        float3 _10028;
        bool _10032;
        bool _10033;
        if (_8097)
        {
            float3 param_28 = _7085;
            float3 param_29 = _9866;
            float3 param_30 = _9867;
            float3 param_31 = _9868;
            int param_32 = _7716;
            float2 param_33 = float2(_7666, _7673);
            light_sample_t _10042 = { _10026, _10027, _10028, _10029, _10030, _10031, _10032, _10033 };
            light_sample_t param_34 = _10042;
            SampleLightSource(param_28, param_29, param_30, param_31, param_32, param_33, param_34);
            _10026 = param_34.col;
            _10027 = param_34.L;
            _10028 = param_34.lp;
            _10029 = param_34.area;
            _10030 = param_34.dist_mul;
            _10031 = param_34.pdf;
            _10032 = param_34.cast_shadow;
            _10033 = param_34.from_env;
        }
        float _8125 = dot(_9868, _10027);
        float3 base_color = float3(_10333, _10334, _10335);
        [branch]
        if (_10329 != 4294967295u)
        {
            base_color *= SampleBilinear(_10329, _7426, int(get_texture_lod(texSize(_10329), _7659)), true, true).xyz;
        }
        float3 tint_color = 0.0f.xxx;
        float _8158 = lum(base_color);
        [flatten]
        if (_8158 > 0.0f)
        {
            tint_color = base_color / _8158.xxx;
        }
        float roughness = clamp(float(_9932 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_10330 != 4294967295u)
        {
            roughness *= SampleBilinear(_10330, _7426, int(get_texture_lod(texSize(_10330), _7659)), false, true).x;
        }
        float _8203 = frac(asfloat(_3521.Load((_7716 + 1) * 4 + 0)) + _7666);
        float _8212 = frac(asfloat(_3521.Load((_7716 + 2) * 4 + 0)) + _7673);
        float _10455 = 0.0f;
        float _10454 = 0.0f;
        float _10453 = 0.0f;
        float _10091[4];
        [unroll]
        for (int i = 0; i < 4; )
        {
            _10091[i] = ray.ior[i];
            i++;
            continue;
        }
        float _10092 = _7649;
        float _10093 = ray.cone_spread;
        int _10094 = ray.xy;
        float _10089 = 0.0f;
        float _10560 = 0.0f;
        float _10559 = 0.0f;
        float _10558 = 0.0f;
        int _10196 = ray.depth;
        int _10200 = ray.xy;
        int _10095;
        float _10198;
        float _10383;
        float _10384;
        float _10385;
        float _10418;
        float _10419;
        float _10420;
        float _10488;
        float _10489;
        float _10490;
        float _10523;
        float _10524;
        float _10525;
        [branch]
        if (_9930 == 0u)
        {
            [branch]
            if ((_10031 > 0.0f) && (_8125 > 0.0f))
            {
                light_sample_t _10059 = { _10026, _10027, _10028, _10029, _10030, _10031, _10032, _10033 };
                surface_t _9877 = { _7085, _9866, _9867, _9868, _9869, _7426 };
                float _10564[3] = { _10558, _10559, _10560 };
                float _10529[3] = { _10523, _10524, _10525 };
                float _10494[3] = { _10488, _10489, _10490 };
                shadow_ray_t _10210 = { _10494, _10196, _10529, _10198, _10564, _10200 };
                shadow_ray_t param_35 = _10210;
                float3 _8272 = Evaluate_DiffuseNode(_10059, ray, _9877, base_color, roughness, mix_weight, param_35);
                _10488 = param_35.o[0];
                _10489 = param_35.o[1];
                _10490 = param_35.o[2];
                _10196 = param_35.depth;
                _10523 = param_35.d[0];
                _10524 = param_35.d[1];
                _10525 = param_35.d[2];
                _10198 = param_35.dist;
                _10558 = param_35.c[0];
                _10559 = param_35.c[1];
                _10560 = param_35.c[2];
                _10200 = param_35.xy;
                col += _8272;
            }
            bool _8279 = _7687 < _3537_g_params.max_diff_depth;
            bool _8286;
            if (_8279)
            {
                _8286 = _7708 < _3537_g_params.max_total_depth;
            }
            else
            {
                _8286 = _8279;
            }
            [branch]
            if (_8286)
            {
                surface_t _9884 = { _7085, _9866, _9867, _9868, _9869, _7426 };
                float _10459[3] = { _10453, _10454, _10455 };
                float _10424[3] = { _10418, _10419, _10420 };
                float _10389[3] = { _10383, _10384, _10385 };
                ray_data_t _10109 = { _10389, _10424, _10089, _10459, _10091, _10092, _10093, _10094, _10095 };
                ray_data_t param_36 = _10109;
                Sample_DiffuseNode(ray, _9884, base_color, roughness, _8203, _8212, mix_weight, param_36);
                _10383 = param_36.o[0];
                _10384 = param_36.o[1];
                _10385 = param_36.o[2];
                _10418 = param_36.d[0];
                _10419 = param_36.d[1];
                _10420 = param_36.d[2];
                _10089 = param_36.pdf;
                _10453 = param_36.c[0];
                _10454 = param_36.c[1];
                _10455 = param_36.c[2];
                _10091 = param_36.ior;
                _10092 = param_36.cone_width;
                _10093 = param_36.cone_spread;
                _10094 = param_36.xy;
                _10095 = param_36.depth;
            }
        }
        else
        {
            [branch]
            if (_9930 == 1u)
            {
                float param_37 = 1.0f;
                float param_38 = 1.5f;
                float _8310 = fresnel_dielectric_cos(param_37, param_38);
                float _8314 = roughness * roughness;
                bool _8317 = _10031 > 0.0f;
                bool _8324;
                if (_8317)
                {
                    _8324 = (_8314 * _8314) >= 1.0000000116860974230803549289703e-07f;
                }
                else
                {
                    _8324 = _8317;
                }
                [branch]
                if (_8324 && (_8125 > 0.0f))
                {
                    light_sample_t _10068 = { _10026, _10027, _10028, _10029, _10030, _10031, _10032, _10033 };
                    surface_t _9891 = { _7085, _9866, _9867, _9868, _9869, _7426 };
                    float _10571[3] = { _10558, _10559, _10560 };
                    float _10536[3] = { _10523, _10524, _10525 };
                    float _10501[3] = { _10488, _10489, _10490 };
                    shadow_ray_t _10223 = { _10501, _10196, _10536, _10198, _10571, _10200 };
                    shadow_ray_t param_39 = _10223;
                    float3 _8339 = Evaluate_GlossyNode(_10068, ray, _9891, base_color, roughness, 1.5f, _8310, mix_weight, param_39);
                    _10488 = param_39.o[0];
                    _10489 = param_39.o[1];
                    _10490 = param_39.o[2];
                    _10196 = param_39.depth;
                    _10523 = param_39.d[0];
                    _10524 = param_39.d[1];
                    _10525 = param_39.d[2];
                    _10198 = param_39.dist;
                    _10558 = param_39.c[0];
                    _10559 = param_39.c[1];
                    _10560 = param_39.c[2];
                    _10200 = param_39.xy;
                    col += _8339;
                }
                bool _8346 = _7692 < _3537_g_params.max_spec_depth;
                bool _8353;
                if (_8346)
                {
                    _8353 = _7708 < _3537_g_params.max_total_depth;
                }
                else
                {
                    _8353 = _8346;
                }
                [branch]
                if (_8353)
                {
                    surface_t _9898 = { _7085, _9866, _9867, _9868, _9869, _7426 };
                    float _10466[3] = { _10453, _10454, _10455 };
                    float _10431[3] = { _10418, _10419, _10420 };
                    float _10396[3] = { _10383, _10384, _10385 };
                    ray_data_t _10128 = { _10396, _10431, _10089, _10466, _10091, _10092, _10093, _10094, _10095 };
                    ray_data_t param_40 = _10128;
                    Sample_GlossyNode(ray, _9898, base_color, roughness, 1.5f, _8310, _8203, _8212, mix_weight, param_40);
                    _10383 = param_40.o[0];
                    _10384 = param_40.o[1];
                    _10385 = param_40.o[2];
                    _10418 = param_40.d[0];
                    _10419 = param_40.d[1];
                    _10420 = param_40.d[2];
                    _10089 = param_40.pdf;
                    _10453 = param_40.c[0];
                    _10454 = param_40.c[1];
                    _10455 = param_40.c[2];
                    _10091 = param_40.ior;
                    _10092 = param_40.cone_width;
                    _10093 = param_40.cone_spread;
                    _10094 = param_40.xy;
                    _10095 = param_40.depth;
                }
            }
            else
            {
                [branch]
                if (_9930 == 2u)
                {
                    float _8377 = roughness * roughness;
                    bool _8380 = _10031 > 0.0f;
                    bool _8387;
                    if (_8380)
                    {
                        _8387 = (_8377 * _8377) >= 1.0000000116860974230803549289703e-07f;
                    }
                    else
                    {
                        _8387 = _8380;
                    }
                    [branch]
                    if (_8387 && (_8125 < 0.0f))
                    {
                        float _8395;
                        if (_7118)
                        {
                            _8395 = _9933 / _7682;
                        }
                        else
                        {
                            _8395 = _7682 / _9933;
                        }
                        light_sample_t _10077 = { _10026, _10027, _10028, _10029, _10030, _10031, _10032, _10033 };
                        surface_t _9905 = { _7085, _9866, _9867, _9868, _9869, _7426 };
                        float _10578[3] = { _10558, _10559, _10560 };
                        float _10543[3] = { _10523, _10524, _10525 };
                        float _10508[3] = { _10488, _10489, _10490 };
                        shadow_ray_t _10236 = { _10508, _10196, _10543, _10198, _10578, _10200 };
                        shadow_ray_t param_41 = _10236;
                        float3 _8417 = Evaluate_RefractiveNode(_10077, ray, _9905, base_color, _8377, _8395, mix_weight, param_41);
                        _10488 = param_41.o[0];
                        _10489 = param_41.o[1];
                        _10490 = param_41.o[2];
                        _10196 = param_41.depth;
                        _10523 = param_41.d[0];
                        _10524 = param_41.d[1];
                        _10525 = param_41.d[2];
                        _10198 = param_41.dist;
                        _10558 = param_41.c[0];
                        _10559 = param_41.c[1];
                        _10560 = param_41.c[2];
                        _10200 = param_41.xy;
                        col += _8417;
                    }
                    bool _8424 = _7697 < _3537_g_params.max_refr_depth;
                    bool _8431;
                    if (_8424)
                    {
                        _8431 = _7708 < _3537_g_params.max_total_depth;
                    }
                    else
                    {
                        _8431 = _8424;
                    }
                    [branch]
                    if (_8431)
                    {
                        surface_t _9912 = { _7085, _9866, _9867, _9868, _9869, _7426 };
                        float _10473[3] = { _10453, _10454, _10455 };
                        float _10438[3] = { _10418, _10419, _10420 };
                        float _10403[3] = { _10383, _10384, _10385 };
                        ray_data_t _10147 = { _10403, _10438, _10089, _10473, _10091, _10092, _10093, _10094, _10095 };
                        ray_data_t param_42 = _10147;
                        Sample_RefractiveNode(ray, _9912, base_color, roughness, _7118, _9933, _7682, _8203, _8212, mix_weight, param_42);
                        _10383 = param_42.o[0];
                        _10384 = param_42.o[1];
                        _10385 = param_42.o[2];
                        _10418 = param_42.d[0];
                        _10419 = param_42.d[1];
                        _10420 = param_42.d[2];
                        _10089 = param_42.pdf;
                        _10453 = param_42.c[0];
                        _10454 = param_42.c[1];
                        _10455 = param_42.c[2];
                        _10091 = param_42.ior;
                        _10092 = param_42.cone_width;
                        _10093 = param_42.cone_spread;
                        _10094 = param_42.xy;
                        _10095 = param_42.depth;
                    }
                }
                else
                {
                    [branch]
                    if (_9930 == 3u)
                    {
                        float mis_weight = 1.0f;
                        [branch]
                        if ((_9929 & 1u) != 0u)
                        {
                            float3 _8501 = mul(float4(_7442, 0.0f), _7195.xform).xyz;
                            float _8504 = length(_8501);
                            float _8516 = abs(dot(_7049, _8501 / _8504.xxx));
                            if (_8516 > 0.0f)
                            {
                                float param_43 = ray.pdf;
                                float param_44 = (inter.t * inter.t) / ((0.5f * _8504) * _8516);
                                mis_weight = power_heuristic(param_43, param_44);
                            }
                        }
                        col += (base_color * ((mix_weight * mis_weight) * _9931));
                    }
                    else
                    {
                        [branch]
                        if (_9930 == 6u)
                        {
                            float metallic = clamp(float((_9935 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_10331 != 4294967295u)
                            {
                                metallic *= SampleBilinear(_10331, _7426, int(get_texture_lod(texSize(_10331), _7659))).x;
                            }
                            float specular = clamp(float(_9937 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_10332 != 4294967295u)
                            {
                                specular *= SampleBilinear(_10332, _7426, int(get_texture_lod(texSize(_10332), _7659))).x;
                            }
                            float _8633 = clamp(float(_9938 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8641 = clamp(float((_9938 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8649 = 2.0f * clamp(float(_9934 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float3 _8667 = lerp(1.0f.xxx, tint_color, clamp(float((_9934 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _8649;
                            float3 _8687 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_9937 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                            float _8696 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                            float param_45 = 1.0f;
                            float param_46 = _8696;
                            float _8702 = fresnel_dielectric_cos(param_45, param_46);
                            float _8710 = clamp(float((_9932 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8721 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _8633))) - 1.0f;
                            float param_47 = 1.0f;
                            float param_48 = _8721;
                            float _8727 = fresnel_dielectric_cos(param_47, param_48);
                            float _8742 = mad(roughness - 1.0f, 1.0f - clamp(float((_9936 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                            float _8748;
                            if (_7118)
                            {
                                _8748 = _9933 / _7682;
                            }
                            else
                            {
                                _8748 = _7682 / _9933;
                            }
                            float param_49 = dot(_7049, _9868);
                            float param_50 = 1.0f / _8748;
                            float _8771 = fresnel_dielectric_cos(param_49, param_50);
                            float param_51 = dot(_7049, _9868);
                            float param_52 = _8696;
                            lobe_weights_t _8810 = get_lobe_weights(lerp(_8158, 1.0f, _8649), lum(lerp(_8687, 1.0f.xxx, ((fresnel_dielectric_cos(param_51, param_52) - _8702) / (1.0f - _8702)).xxx)), specular, metallic, clamp(float(_9936 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _8633);
                            [branch]
                            if (_10031 > 0.0f)
                            {
                                light_sample_t _10086 = { _10026, _10027, _10028, _10029, _10030, _10031, _10032, _10033 };
                                surface_t _9919 = { _7085, _9866, _9867, _9868, _9869, _7426 };
                                diff_params_t _10278 = { base_color, _8667, roughness };
                                spec_params_t _10293 = { _8687, roughness, _8696, _8702, _8710 };
                                clearcoat_params_t _10306 = { _8641, _8721, _8727 };
                                transmission_params_t _10321 = { _8742, _9933, _8748, _8771, _7118 };
                                float _10585[3] = { _10558, _10559, _10560 };
                                float _10550[3] = { _10523, _10524, _10525 };
                                float _10515[3] = { _10488, _10489, _10490 };
                                shadow_ray_t _10249 = { _10515, _10196, _10550, _10198, _10585, _10200 };
                                shadow_ray_t param_53 = _10249;
                                float3 _8829 = Evaluate_PrincipledNode(_10086, ray, _9919, _8810, _10278, _10293, _10306, _10321, metallic, _8125, mix_weight, param_53);
                                _10488 = param_53.o[0];
                                _10489 = param_53.o[1];
                                _10490 = param_53.o[2];
                                _10196 = param_53.depth;
                                _10523 = param_53.d[0];
                                _10524 = param_53.d[1];
                                _10525 = param_53.d[2];
                                _10198 = param_53.dist;
                                _10558 = param_53.c[0];
                                _10559 = param_53.c[1];
                                _10560 = param_53.c[2];
                                _10200 = param_53.xy;
                                col += _8829;
                            }
                            surface_t _9926 = { _7085, _9866, _9867, _9868, _9869, _7426 };
                            diff_params_t _10282 = { base_color, _8667, roughness };
                            spec_params_t _10299 = { _8687, roughness, _8696, _8702, _8710 };
                            clearcoat_params_t _10310 = { _8641, _8721, _8727 };
                            transmission_params_t _10327 = { _8742, _9933, _8748, _8771, _7118 };
                            float param_54 = mix_rand;
                            float _10480[3] = { _10453, _10454, _10455 };
                            float _10445[3] = { _10418, _10419, _10420 };
                            float _10410[3] = { _10383, _10384, _10385 };
                            ray_data_t _10166 = { _10410, _10445, _10089, _10480, _10091, _10092, _10093, _10094, _10095 };
                            ray_data_t param_55 = _10166;
                            Sample_PrincipledNode(ray, _9926, _8810, _10282, _10299, _10310, _10327, metallic, _8203, _8212, param_54, mix_weight, param_55);
                            _10383 = param_55.o[0];
                            _10384 = param_55.o[1];
                            _10385 = param_55.o[2];
                            _10418 = param_55.d[0];
                            _10419 = param_55.d[1];
                            _10420 = param_55.d[2];
                            _10089 = param_55.pdf;
                            _10453 = param_55.c[0];
                            _10454 = param_55.c[1];
                            _10455 = param_55.c[2];
                            _10091 = param_55.ior;
                            _10092 = param_55.cone_width;
                            _10093 = param_55.cone_spread;
                            _10094 = param_55.xy;
                            _10095 = param_55.depth;
                        }
                    }
                }
            }
        }
        float _8863 = max(_10453, max(_10454, _10455));
        float _8875;
        if (_7708 > _3537_g_params.min_total_depth)
        {
            _8875 = max(0.0500000007450580596923828125f, 1.0f - _8863);
        }
        else
        {
            _8875 = 0.0f;
        }
        bool _8889 = (frac(asfloat(_3521.Load((_7716 + 6) * 4 + 0)) + _7666) >= _8875) && (_8863 > 0.0f);
        bool _8895;
        if (_8889)
        {
            _8895 = _10089 > 0.0f;
        }
        else
        {
            _8895 = _8889;
        }
        [branch]
        if (_8895)
        {
            float _8899 = _10089;
            float _8900 = min(_8899, 1000000.0f);
            _10089 = _8900;
            float _8903 = 1.0f - _8875;
            float _8905 = _10453;
            float _8906 = _8905 / _8903;
            _10453 = _8906;
            float _8911 = _10454;
            float _8912 = _8911 / _8903;
            _10454 = _8912;
            float _8917 = _10455;
            float _8918 = _8917 / _8903;
            _10455 = _8918;
            uint _8926;
            _8924.InterlockedAdd(0, 1u, _8926);
            _8935.Store(_8926 * 72 + 0, asuint(_10383));
            _8935.Store(_8926 * 72 + 4, asuint(_10384));
            _8935.Store(_8926 * 72 + 8, asuint(_10385));
            _8935.Store(_8926 * 72 + 12, asuint(_10418));
            _8935.Store(_8926 * 72 + 16, asuint(_10419));
            _8935.Store(_8926 * 72 + 20, asuint(_10420));
            _8935.Store(_8926 * 72 + 24, asuint(_8900));
            _8935.Store(_8926 * 72 + 28, asuint(_8906));
            _8935.Store(_8926 * 72 + 32, asuint(_8912));
            _8935.Store(_8926 * 72 + 36, asuint(_8918));
            _8935.Store(_8926 * 72 + 40, asuint(_10091[0]));
            _8935.Store(_8926 * 72 + 44, asuint(_10091[1]));
            _8935.Store(_8926 * 72 + 48, asuint(_10091[2]));
            _8935.Store(_8926 * 72 + 52, asuint(_10091[3]));
            _8935.Store(_8926 * 72 + 56, asuint(_10092));
            _8935.Store(_8926 * 72 + 60, asuint(_10093));
            _8935.Store(_8926 * 72 + 64, uint(_10094));
            _8935.Store(_8926 * 72 + 68, uint(_10095));
        }
        [branch]
        if (max(_10558, max(_10559, _10560)) > 0.0f)
        {
            float3 _9012 = _10028 - float3(_10488, _10489, _10490);
            float _9015 = length(_9012);
            float3 _9019 = _9012 / _9015.xxx;
            float sh_dist = _9015 * _10030;
            if (_10033)
            {
                sh_dist = -sh_dist;
            }
            float _9031 = _9019.x;
            _10523 = _9031;
            float _9034 = _9019.y;
            _10524 = _9034;
            float _9037 = _9019.z;
            _10525 = _9037;
            _10198 = sh_dist;
            uint _9043;
            _8924.InterlockedAdd(8, 1u, _9043);
            _9051.Store(_9043 * 48 + 0, asuint(_10488));
            _9051.Store(_9043 * 48 + 4, asuint(_10489));
            _9051.Store(_9043 * 48 + 8, asuint(_10490));
            _9051.Store(_9043 * 48 + 12, uint(_10196));
            _9051.Store(_9043 * 48 + 16, asuint(_9031));
            _9051.Store(_9043 * 48 + 20, asuint(_9034));
            _9051.Store(_9043 * 48 + 24, asuint(_9037));
            _9051.Store(_9043 * 48 + 28, asuint(sh_dist));
            _9051.Store(_9043 * 48 + 32, asuint(_10558));
            _9051.Store(_9043 * 48 + 36, asuint(_10559));
            _9051.Store(_9043 * 48 + 40, asuint(_10560));
            _9051.Store(_9043 * 48 + 44, uint(_10200));
        }
        _9264 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _9264;
}

void comp_main()
{
    do
    {
        int _9117 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_9117) >= _8924.Load(4))
        {
            break;
        }
        int _9133 = int(_9130.Load(_9117 * 72 + 64));
        int _9140 = int(_9130.Load(_9117 * 72 + 64));
        hit_data_t _9151;
        _9151.mask = int(_9147.Load(_9117 * 24 + 0));
        _9151.obj_index = int(_9147.Load(_9117 * 24 + 4));
        _9151.prim_index = int(_9147.Load(_9117 * 24 + 8));
        _9151.t = asfloat(_9147.Load(_9117 * 24 + 12));
        _9151.u = asfloat(_9147.Load(_9117 * 24 + 16));
        _9151.v = asfloat(_9147.Load(_9117 * 24 + 20));
        ray_data_t _9167;
        [unroll]
        for (int _112ident = 0; _112ident < 3; _112ident++)
        {
            _9167.o[_112ident] = asfloat(_9130.Load(_112ident * 4 + _9117 * 72 + 0));
        }
        [unroll]
        for (int _113ident = 0; _113ident < 3; _113ident++)
        {
            _9167.d[_113ident] = asfloat(_9130.Load(_113ident * 4 + _9117 * 72 + 12));
        }
        _9167.pdf = asfloat(_9130.Load(_9117 * 72 + 24));
        [unroll]
        for (int _114ident = 0; _114ident < 3; _114ident++)
        {
            _9167.c[_114ident] = asfloat(_9130.Load(_114ident * 4 + _9117 * 72 + 28));
        }
        [unroll]
        for (int _115ident = 0; _115ident < 4; _115ident++)
        {
            _9167.ior[_115ident] = asfloat(_9130.Load(_115ident * 4 + _9117 * 72 + 40));
        }
        _9167.cone_width = asfloat(_9130.Load(_9117 * 72 + 56));
        _9167.cone_spread = asfloat(_9130.Load(_9117 * 72 + 60));
        _9167.xy = int(_9130.Load(_9117 * 72 + 64));
        _9167.depth = int(_9130.Load(_9117 * 72 + 68));
        hit_data_t _9358 = { _9151.mask, _9151.obj_index, _9151.prim_index, _9151.t, _9151.u, _9151.v };
        hit_data_t param = _9358;
        float _9407[4] = { _9167.ior[0], _9167.ior[1], _9167.ior[2], _9167.ior[3] };
        float _9398[3] = { _9167.c[0], _9167.c[1], _9167.c[2] };
        float _9391[3] = { _9167.d[0], _9167.d[1], _9167.d[2] };
        float _9384[3] = { _9167.o[0], _9167.o[1], _9167.o[2] };
        ray_data_t _9377 = { _9384, _9391, _9167.pdf, _9398, _9407, _9167.cone_width, _9167.cone_spread, _9167.xy, _9167.depth };
        ray_data_t param_1 = _9377;
        float3 _9217 = ShadeSurface(param, param_1);
        int2 _9224 = int2((_9133 >> 16) & 65535, _9140 & 65535);
        g_out_img[_9224] = float4(_9217 + g_out_img[_9224].xyz, 1.0f);
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
