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

ByteAddressBuffer _3311 : register(t17, space0);
ByteAddressBuffer _3347 : register(t8, space0);
ByteAddressBuffer _3351 : register(t9, space0);
ByteAddressBuffer _4102 : register(t13, space0);
ByteAddressBuffer _4127 : register(t15, space0);
ByteAddressBuffer _4131 : register(t16, space0);
ByteAddressBuffer _4455 : register(t12, space0);
ByteAddressBuffer _4459 : register(t11, space0);
ByteAddressBuffer _6709 : register(t14, space0);
RWByteAddressBuffer _8364 : register(u3, space0);
RWByteAddressBuffer _8375 : register(u1, space0);
RWByteAddressBuffer _8491 : register(u2, space0);
ByteAddressBuffer _8589 : register(t6, space0);
ByteAddressBuffer _8610 : register(t7, space0);
ByteAddressBuffer _8714 : register(t10, space0);
cbuffer UniformParams
{
    Params _3327_g_params : packoffset(c0);
};

Texture2D<float4> g_textures[] : register(t0, space1);
SamplerState _g_textures_sampler[] : register(s0, space1);
Texture2D<float4> g_env_qtree : register(t18, space0);
SamplerState _g_env_qtree_sampler : register(s18, space0);
RWTexture2D<float4> g_out_img : register(u0, space0);
RWTexture2D<float4> g_out_depth_normals_img : register(u5, space0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

uint2 spvTextureSize(Texture2D<float4> Tex, uint Level, out uint Param)
{
    uint2 ret;
    Tex.GetDimensions(Level, ret.x, ret.y, Param);
    return ret;
}

float3 rgbe_to_rgb(float4 rgbe)
{
    return rgbe.xyz * exp2(mad(255.0f, rgbe.w, -128.0f));
}

float3 SampleLatlong_RGBE(uint index, float3 dir, float y_rotation)
{
    float _1066 = atan2(dir.z, dir.x) + y_rotation;
    float phi = _1066;
    if (_1066 < 0.0f)
    {
        phi += 6.283185482025146484375f;
    }
    if (phi > 6.283185482025146484375f)
    {
        phi -= 6.283185482025146484375f;
    }
    uint _1085 = index & 16777215u;
    uint _1091_dummy_parameter;
    float2 _1098 = float2(frac(phi * 0.15915493667125701904296875f), acos(clamp(dir.y, -1.0f, 1.0f)) * 0.3183098733425140380859375f) * float2(int2(spvTextureSize(g_textures[_1085], uint(0), _1091_dummy_parameter)));
    uint _1101 = _1085;
    int2 _1105 = int2(_1098);
    float2 _1140 = frac(_1098);
    float4 param = g_textures[NonUniformResourceIndex(_1101)].Load(int3(_1105, 0), int2(0, 0));
    float4 param_1 = g_textures[NonUniformResourceIndex(_1101)].Load(int3(_1105, 0), int2(1, 0));
    float4 param_2 = g_textures[NonUniformResourceIndex(_1101)].Load(int3(_1105, 0), int2(0, 1));
    float4 param_3 = g_textures[NonUniformResourceIndex(_1101)].Load(int3(_1105, 0), int2(1, 1));
    float _1160 = _1140.x;
    float _1165 = 1.0f - _1160;
    float _1181 = _1140.y;
    return (((rgbe_to_rgb(param_3) * _1160) + (rgbe_to_rgb(param_2) * _1165)) * _1181) + (((rgbe_to_rgb(param_1) * _1160) + (rgbe_to_rgb(param) * _1165)) * (1.0f - _1181));
}

float2 DirToCanonical(float3 d, float y_rotation)
{
    float _722 = (-atan2(d.z, d.x)) + y_rotation;
    float phi = _722;
    if (_722 < 0.0f)
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
    float2 _749 = DirToCanonical(L, -y_rotation);
    float factor = 1.0f;
    while (lod >= 0)
    {
        int2 _769 = clamp(int2(_749 * float(res)), int2(0, 0), (res - 1).xx);
        float4 quad = qtree_tex.Load(int3(_769 / int2(2, 2), lod));
        float _804 = ((quad.x + quad.y) + quad.z) + quad.w;
        if (_804 <= 0.0f)
        {
            break;
        }
        factor *= ((4.0f * quad[(0 | ((_769.x & 1) << 0)) | ((_769.y & 1) << 1)]) / _804);
        lod--;
        res *= 2;
    }
    return factor * 0.079577468335628509521484375f;
}

float power_heuristic(float a, float b)
{
    float _1194 = a * a;
    return _1194 / mad(b, b, _1194);
}

float3 Evaluate_EnvColor(ray_data_t ray)
{
    float3 _4666 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 env_col = _3327_g_params.back_col.xyz;
    uint _4674 = asuint(_3327_g_params.back_col.w);
    if (_4674 != 4294967295u)
    {
        env_col *= SampleLatlong_RGBE(_4674, _4666, _3327_g_params.back_rotation);
    }
    if (_3327_g_params.env_qtree_levels > 0)
    {
        float param = ray.pdf;
        float param_1 = Evaluate_EnvQTree(_3327_g_params.back_rotation, g_env_qtree, _g_env_qtree_sampler, _3327_g_params.env_qtree_levels, _4666);
        env_col *= power_heuristic(param, param_1);
    }
    else
    {
        if (_3327_g_params.env_mult_importance != 0)
        {
            float param_2 = ray.pdf;
            float param_3 = 0.15915493667125701904296875f;
            env_col *= power_heuristic(param_2, param_3);
        }
    }
    return env_col;
}

float3 Evaluate_LightColor(ray_data_t ray, hit_data_t inter)
{
    float3 _4748 = float3(ray.d[0], ray.d[1], ray.d[2]);
    light_t _4762;
    _4762.type_and_param0 = _3347.Load4(((-1) - inter.obj_index) * 64 + 0);
    _4762.param1 = asfloat(_3347.Load4(((-1) - inter.obj_index) * 64 + 16));
    _4762.param2 = asfloat(_3347.Load4(((-1) - inter.obj_index) * 64 + 32));
    _4762.param3 = asfloat(_3347.Load4(((-1) - inter.obj_index) * 64 + 48));
    float3 lcol = asfloat(_4762.type_and_param0.yzw);
    [branch]
    if ((_4762.type_and_param0.x & 128u) != 0u)
    {
        float3 env_col = _3327_g_params.env_col.xyz;
        uint _4789 = asuint(_3327_g_params.env_col.w);
        if (_4789 != 4294967295u)
        {
            env_col *= SampleLatlong_RGBE(_4789, _4748, _3327_g_params.env_rotation);
        }
        lcol *= env_col;
    }
    uint _4807 = _4762.type_and_param0.x & 31u;
    if (_4807 == 0u)
    {
        float param = ray.pdf;
        float param_1 = (inter.t * inter.t) / ((0.5f * _4762.param1.w) * dot(_4748, normalize(_4762.param1.xyz - (float3(ray.o[0], ray.o[1], ray.o[2]) + (_4748 * inter.t)))));
        lcol *= power_heuristic(param, param_1);
        bool _4874 = _4762.param3.x > 0.0f;
        bool _4880;
        if (_4874)
        {
            _4880 = _4762.param3.y > 0.0f;
        }
        else
        {
            _4880 = _4874;
        }
        [branch]
        if (_4880)
        {
            [flatten]
            if (_4762.param3.y > 0.0f)
            {
                lcol *= clamp((_4762.param3.x - acos(clamp(-dot(_4748, _4762.param2.xyz), 0.0f, 1.0f))) / _4762.param3.y, 0.0f, 1.0f);
            }
        }
    }
    else
    {
        if (_4807 == 4u)
        {
            float param_2 = ray.pdf;
            float param_3 = (inter.t * inter.t) / (_4762.param1.w * dot(_4748, normalize(cross(_4762.param2.xyz, _4762.param3.xyz))));
            lcol *= power_heuristic(param_2, param_3);
        }
        else
        {
            if (_4807 == 5u)
            {
                float param_4 = ray.pdf;
                float param_5 = (inter.t * inter.t) / (_4762.param1.w * dot(_4748, normalize(cross(_4762.param2.xyz, _4762.param3.xyz))));
                lcol *= power_heuristic(param_4, param_5);
            }
            else
            {
                if (_4807 == 3u)
                {
                    float param_6 = ray.pdf;
                    float param_7 = (inter.t * inter.t) / (_4762.param1.w * (1.0f - abs(dot(_4748, _4762.param3.xyz))));
                    lcol *= power_heuristic(param_6, param_7);
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
    uint _499 = uint(x);
    uint _506 = ((_499 >> uint(16)) ^ _499) * 73244475u;
    uint _511 = ((_506 >> uint(16)) ^ _506) * 73244475u;
    return int((_511 >> uint(16)) ^ _511);
}

float construct_float(inout uint m)
{
    m &= 8388607u;
    m |= 1065353216u;
    return asfloat(m) - 1.0f;
}

bool exchange(inout bool old_value, bool new_value)
{
    bool _2102 = old_value;
    old_value = new_value;
    return _2102;
}

float peek_ior_stack(float stack[4], inout bool skip_first, float default_value)
{
    float _8726;
    do
    {
        bool _2186 = stack[3] > 0.0f;
        bool _2195;
        if (_2186)
        {
            bool param = skip_first;
            bool param_1 = false;
            bool _2192 = exchange(param, param_1);
            skip_first = param;
            _2195 = !_2192;
        }
        else
        {
            _2195 = _2186;
        }
        if (_2195)
        {
            _8726 = stack[3];
            break;
        }
        bool _2203 = stack[2] > 0.0f;
        bool _2212;
        if (_2203)
        {
            bool param_2 = skip_first;
            bool param_3 = false;
            bool _2209 = exchange(param_2, param_3);
            skip_first = param_2;
            _2212 = !_2209;
        }
        else
        {
            _2212 = _2203;
        }
        if (_2212)
        {
            _8726 = stack[2];
            break;
        }
        bool _2220 = stack[1] > 0.0f;
        bool _2229;
        if (_2220)
        {
            bool param_4 = skip_first;
            bool param_5 = false;
            bool _2226 = exchange(param_4, param_5);
            skip_first = param_4;
            _2229 = !_2226;
        }
        else
        {
            _2229 = _2220;
        }
        if (_2229)
        {
            _8726 = stack[1];
            break;
        }
        bool _2237 = stack[0] > 0.0f;
        bool _2246;
        if (_2237)
        {
            bool param_6 = skip_first;
            bool param_7 = false;
            bool _2243 = exchange(param_6, param_7);
            skip_first = param_6;
            _2246 = !_2243;
        }
        else
        {
            _2246 = _2237;
        }
        if (_2246)
        {
            _8726 = stack[0];
            break;
        }
        _8726 = default_value;
        break;
    } while(false);
    return _8726;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _598 = mad(col.z, 31.875f, 1.0f);
    float _608 = (col.x - 0.501960813999176025390625f) / _598;
    float _614 = (col.y - 0.501960813999176025390625f) / _598;
    return float3((col.w + _608) - _614, col.w + _614, (col.w - _608) - _614);
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
    uint _999 = index & 16777215u;
    float4 res = g_textures[NonUniformResourceIndex(_999)].SampleLevel(_g_textures_sampler[NonUniformResourceIndex(_999)], uvs, float(lod));
    bool _1009;
    if (maybe_YCoCg)
    {
        _1009 = (index & 67108864u) != 0u;
    }
    else
    {
        _1009 = maybe_YCoCg;
    }
    if (_1009)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _1027;
    if (maybe_SRGB)
    {
        _1027 = (index & 16777216u) != 0u;
    }
    else
    {
        _1027 = maybe_SRGB;
    }
    if (_1027)
    {
        float3 param_1 = res.xyz;
        float3 _1033 = srgb_to_rgb(param_1);
        float4 _9772 = res;
        _9772.x = _1033.x;
        float4 _9774 = _9772;
        _9774.y = _1033.y;
        float4 _9776 = _9774;
        _9776.z = _1033.z;
        res = _9776;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod)
{
    return SampleBilinear(index, uvs, lod, false, false);
}

float fresnel_dielectric_cos(float cosi, float eta)
{
    float _1226 = abs(cosi);
    float _1235 = mad(_1226, _1226, mad(eta, eta, -1.0f));
    float g = _1235;
    float result;
    if (_1235 > 0.0f)
    {
        float _1240 = g;
        float _1241 = sqrt(_1240);
        g = _1241;
        float _1245 = _1241 - _1226;
        float _1248 = _1241 + _1226;
        float _1249 = _1245 / _1248;
        float _1263 = mad(_1226, _1248, -1.0f) / mad(_1226, _1245, 1.0f);
        result = ((0.5f * _1249) * _1249) * mad(_1263, _1263, 1.0f);
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
    float3 _8731;
    do
    {
        float _1299 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1299)
        {
            _8731 = N;
            break;
        }
        float3 _1319 = normalize(N - (Ng * dot(N, Ng)));
        float _1323 = dot(I, _1319);
        float _1327 = dot(I, Ng);
        float _1339 = mad(_1323, _1323, _1327 * _1327);
        float param = (_1323 * _1323) * mad(-_1299, _1299, _1339);
        float _1349 = safe_sqrtf(param);
        float _1355 = mad(_1327, _1299, _1339);
        float _1358 = 0.5f / _1339;
        float _1363 = _1349 + _1355;
        float _1364 = _1358 * _1363;
        float _1370 = (-_1349) + _1355;
        float _1371 = _1358 * _1370;
        bool _1379 = (_1364 > 9.9999997473787516355514526367188e-06f) && (_1364 <= 1.000010013580322265625f);
        bool valid1 = _1379;
        bool _1385 = (_1371 > 9.9999997473787516355514526367188e-06f) && (_1371 <= 1.000010013580322265625f);
        bool valid2 = _1385;
        float2 N_new;
        if (_1379 && _1385)
        {
            float _10073 = (-0.5f) / _1339;
            float param_1 = mad(_10073, _1363, 1.0f);
            float _1395 = safe_sqrtf(param_1);
            float param_2 = _1364;
            float _1398 = safe_sqrtf(param_2);
            float2 _1399 = float2(_1395, _1398);
            float param_3 = mad(_10073, _1370, 1.0f);
            float _1404 = safe_sqrtf(param_3);
            float param_4 = _1371;
            float _1407 = safe_sqrtf(param_4);
            float2 _1408 = float2(_1404, _1407);
            float _10075 = -_1327;
            float _1424 = mad(2.0f * mad(_1395, _1323, _1398 * _1327), _1398, _10075);
            float _1440 = mad(2.0f * mad(_1404, _1323, _1407 * _1327), _1407, _10075);
            bool _1442 = _1424 >= 9.9999997473787516355514526367188e-06f;
            valid1 = _1442;
            bool _1444 = _1440 >= 9.9999997473787516355514526367188e-06f;
            valid2 = _1444;
            if (_1442 && _1444)
            {
                bool2 _1457 = (_1424 < _1440).xx;
                N_new = float2(_1457.x ? _1399.x : _1408.x, _1457.y ? _1399.y : _1408.y);
            }
            else
            {
                bool2 _1465 = (_1424 > _1440).xx;
                N_new = float2(_1465.x ? _1399.x : _1408.x, _1465.y ? _1399.y : _1408.y);
            }
        }
        else
        {
            if (!(valid1 || valid2))
            {
                _8731 = Ng;
                break;
            }
            float _1477 = valid1 ? _1364 : _1371;
            float param_5 = 1.0f - _1477;
            float param_6 = _1477;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _8731 = (_1319 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _8731;
}

float3 rotate_around_axis(float3 p, float3 axis, float angle)
{
    float _1571 = cos(angle);
    float _1574 = sin(angle);
    float _1578 = 1.0f - _1571;
    return float3(mad(mad(_1578 * axis.x, axis.z, axis.y * _1574), p.z, mad(mad(_1578 * axis.x, axis.x, _1571), p.x, mad(_1578 * axis.x, axis.y, -(axis.z * _1574)) * p.y)), mad(mad(_1578 * axis.y, axis.z, -(axis.x * _1574)), p.z, mad(mad(_1578 * axis.x, axis.y, axis.z * _1574), p.x, mad(_1578 * axis.y, axis.y, _1571) * p.y)), mad(mad(_1578 * axis.z, axis.z, _1571), p.z, mad(mad(_1578 * axis.x, axis.z, -(axis.y * _1574)), p.x, mad(_1578 * axis.y, axis.z, axis.x * _1574) * p.y)));
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
    int3 _1728 = int3(n * 128.0f);
    int _1736;
    if (p.x < 0.0f)
    {
        _1736 = -_1728.x;
    }
    else
    {
        _1736 = _1728.x;
    }
    int _1754;
    if (p.y < 0.0f)
    {
        _1754 = -_1728.y;
    }
    else
    {
        _1754 = _1728.y;
    }
    int _1772;
    if (p.z < 0.0f)
    {
        _1772 = -_1728.z;
    }
    else
    {
        _1772 = _1728.z;
    }
    float _1790;
    if (abs(p.x) < 0.03125f)
    {
        _1790 = mad(1.52587890625e-05f, n.x, p.x);
    }
    else
    {
        _1790 = asfloat(asint(p.x) + _1736);
    }
    float _1808;
    if (abs(p.y) < 0.03125f)
    {
        _1808 = mad(1.52587890625e-05f, n.y, p.y);
    }
    else
    {
        _1808 = asfloat(asint(p.y) + _1754);
    }
    float _1825;
    if (abs(p.z) < 0.03125f)
    {
        _1825 = mad(1.52587890625e-05f, n.z, p.z);
    }
    else
    {
        _1825 = asfloat(asint(p.z) + _1772);
    }
    return float3(_1790, _1808, _1825);
}

float3 MapToCone(float r1, float r2, float3 N, float radius)
{
    float3 _8756;
    do
    {
        float2 _3226 = (float2(r1, r2) * 2.0f) - 1.0f.xx;
        float _3228 = _3226.x;
        bool _3229 = _3228 == 0.0f;
        bool _3235;
        if (_3229)
        {
            _3235 = _3226.y == 0.0f;
        }
        else
        {
            _3235 = _3229;
        }
        if (_3235)
        {
            _8756 = N;
            break;
        }
        float _3244 = _3226.y;
        float r;
        float theta;
        if (abs(_3228) > abs(_3244))
        {
            r = _3228;
            theta = 0.785398185253143310546875f * (_3244 / _3228);
        }
        else
        {
            r = _3244;
            theta = 1.57079637050628662109375f * mad(-0.5f, _3228 / _3244, 1.0f);
        }
        float3 param;
        float3 param_1;
        create_tbn(N, param, param_1);
        _8756 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _8756;
}

float3 CanonicalToDir(float2 p, float y_rotation)
{
    float _672 = mad(2.0f, p.x, -1.0f);
    float _677 = mad(6.283185482025146484375f, p.y, y_rotation);
    float phi = _677;
    if (_677 < 0.0f)
    {
        phi += 6.283185482025146484375f;
    }
    if (phi > 6.283185482025146484375f)
    {
        phi -= 6.283185482025146484375f;
    }
    float _695 = sqrt(mad(-_672, _672, 1.0f));
    return float3(_695 * cos(phi), _672, (-_695) * sin(phi));
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
        float _870 = quad.x + quad.z;
        float partial = _870;
        float _877 = (_870 + quad.y) + quad.w;
        if (_877 <= 0.0f)
        {
            break;
        }
        float _886 = partial / _877;
        float boundary = _886;
        int index = 0;
        if (_sample < _886)
        {
            _sample /= boundary;
            boundary = quad.x / partial;
        }
        else
        {
            float _901 = partial;
            float _902 = _877 - _901;
            partial = _902;
            float2 _9759 = origin;
            _9759.x = origin.x + _step;
            origin = _9759;
            _sample = (_sample - boundary) / (1.0f - boundary);
            boundary = quad.y / _902;
            index |= 1;
        }
        if (_sample < boundary)
        {
            _sample /= boundary;
        }
        else
        {
            float2 _9762 = origin;
            _9762.y = origin.y + _step;
            origin = _9762;
            _sample = (_sample - boundary) / (1.0f - boundary);
            index |= 2;
        }
        factor *= ((4.0f * quad[index]) / _877);
        lod--;
        res *= 2;
        _step *= 0.5f;
    }
    float2 _959 = origin;
    float2 _960 = _959 + (float2(rx, ry) * (2.0f * _step));
    origin = _960;
    return float4(CanonicalToDir(_960, y_rotation), factor * 0.079577468335628509521484375f);
}

float3 world_from_tangent(float3 T, float3 B, float3 N, float3 V)
{
    return ((T * V.x) + (B * V.y)) + (N * V.z);
}

void SampleLightSource(float3 P, float3 T, float3 B, float3 N, int hi, float2 sample_off, inout light_sample_t ls)
{
    float _3320 = frac(asfloat(_3311.Load((hi + 3) * 4 + 0)) + sample_off.x);
    float _3331 = float(_3327_g_params.li_count);
    uint _3338 = min(uint(_3320 * _3331), uint(_3327_g_params.li_count - 1));
    light_t _3358;
    _3358.type_and_param0 = _3347.Load4(_3351.Load(_3338 * 4 + 0) * 64 + 0);
    _3358.param1 = asfloat(_3347.Load4(_3351.Load(_3338 * 4 + 0) * 64 + 16));
    _3358.param2 = asfloat(_3347.Load4(_3351.Load(_3338 * 4 + 0) * 64 + 32));
    _3358.param3 = asfloat(_3347.Load4(_3351.Load(_3338 * 4 + 0) * 64 + 48));
    ls.col = asfloat(_3358.type_and_param0.yzw);
    ls.col *= _3331;
    ls.cast_shadow = (_3358.type_and_param0.x & 32u) != 0u;
    ls.from_env = false;
    uint _3394 = _3358.type_and_param0.x & 31u;
    [branch]
    if (_3394 == 0u)
    {
        float _3408 = frac(asfloat(_3311.Load((hi + 4) * 4 + 0)) + sample_off.x);
        float3 _3424 = P - _3358.param1.xyz;
        float3 _3431 = _3424 / length(_3424).xxx;
        float _3438 = sqrt(clamp(mad(-_3408, _3408, 1.0f), 0.0f, 1.0f));
        float _3441 = 6.283185482025146484375f * frac(asfloat(_3311.Load((hi + 5) * 4 + 0)) + sample_off.y);
        float3 sampled_dir = float3(_3438 * cos(_3441), _3438 * sin(_3441), _3408);
        float3 param;
        float3 param_1;
        create_tbn(_3431, param, param_1);
        float3 _9839 = sampled_dir;
        float3 _3474 = ((param * _9839.x) + (param_1 * _9839.y)) + (_3431 * _9839.z);
        sampled_dir = _3474;
        float3 _3483 = _3358.param1.xyz + (_3474 * _3358.param2.w);
        float3 _3490 = normalize(_3483 - _3358.param1.xyz);
        float3 param_2 = _3483;
        float3 param_3 = _3490;
        ls.lp = offset_ray(param_2, param_3);
        ls.L = _3483 - P;
        float3 _3503 = ls.L;
        float _3504 = length(_3503);
        ls.L /= _3504.xxx;
        ls.area = _3358.param1.w;
        float _3519 = abs(dot(ls.L, _3490));
        [flatten]
        if (_3519 > 0.0f)
        {
            ls.pdf = (_3504 * _3504) / ((0.5f * ls.area) * _3519);
        }
        [branch]
        if (_3358.param3.x > 0.0f)
        {
            float _3546 = -dot(ls.L, _3358.param2.xyz);
            if (_3546 > 0.0f)
            {
                ls.col *= clamp((_3358.param3.x - acos(clamp(_3546, 0.0f, 1.0f))) / _3358.param3.y, 0.0f, 1.0f);
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
        if (_3394 == 2u)
        {
            ls.L = _3358.param1.xyz;
            if (_3358.param1.w != 0.0f)
            {
                float param_4 = frac(asfloat(_3311.Load((hi + 4) * 4 + 0)) + sample_off.x);
                float param_5 = frac(asfloat(_3311.Load((hi + 5) * 4 + 0)) + sample_off.y);
                float3 param_6 = ls.L;
                float param_7 = tan(_3358.param1.w);
                ls.L = normalize(MapToCone(param_4, param_5, param_6, param_7));
            }
            ls.area = 0.0f;
            ls.lp = P + ls.L;
            ls.dist_mul = 3402823346297367662189621542912.0f;
            ls.pdf = 1.0f;
            if ((_3358.type_and_param0.x & 64u) == 0u)
            {
                ls.area = 0.0f;
            }
        }
        else
        {
            [branch]
            if (_3394 == 4u)
            {
                float3 _3683 = (_3358.param1.xyz + (_3358.param2.xyz * (frac(asfloat(_3311.Load((hi + 4) * 4 + 0)) + sample_off.x) - 0.5f))) + (_3358.param3.xyz * (frac(asfloat(_3311.Load((hi + 5) * 4 + 0)) + sample_off.y) - 0.5f));
                float3 _3688 = normalize(cross(_3358.param2.xyz, _3358.param3.xyz));
                float3 param_8 = _3683;
                float3 param_9 = _3688;
                ls.lp = offset_ray(param_8, param_9);
                ls.L = _3683 - P;
                float3 _3701 = ls.L;
                float _3702 = length(_3701);
                ls.L /= _3702.xxx;
                ls.area = _3358.param1.w;
                float _3717 = dot(-ls.L, _3688);
                if (_3717 > 0.0f)
                {
                    ls.pdf = (_3702 * _3702) / (ls.area * _3717);
                }
                if ((_3358.type_and_param0.x & 64u) == 0u)
                {
                    ls.area = 0.0f;
                }
                [branch]
                if ((_3358.type_and_param0.x & 128u) != 0u)
                {
                    float3 env_col = _3327_g_params.env_col.xyz;
                    uint _3755 = asuint(_3327_g_params.env_col.w);
                    if (_3755 != 4294967295u)
                    {
                        env_col *= SampleLatlong_RGBE(_3755, ls.L, _3327_g_params.env_rotation);
                    }
                    ls.col *= env_col;
                    ls.from_env = true;
                }
            }
            else
            {
                [branch]
                if (_3394 == 5u)
                {
                    float2 _3818 = (float2(frac(asfloat(_3311.Load((hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_3311.Load((hi + 5) * 4 + 0)) + sample_off.y)) * 2.0f) - 1.0f.xx;
                    float2 offset = _3818;
                    bool _3821 = _3818.x != 0.0f;
                    bool _3827;
                    if (_3821)
                    {
                        _3827 = offset.y != 0.0f;
                    }
                    else
                    {
                        _3827 = _3821;
                    }
                    if (_3827)
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
                        float _3860 = 0.5f * r;
                        offset = float2(_3860 * cos(theta), _3860 * sin(theta));
                    }
                    float3 _3882 = (_3358.param1.xyz + (_3358.param2.xyz * offset.x)) + (_3358.param3.xyz * offset.y);
                    float3 _3887 = normalize(cross(_3358.param2.xyz, _3358.param3.xyz));
                    float3 param_10 = _3882;
                    float3 param_11 = _3887;
                    ls.lp = offset_ray(param_10, param_11);
                    ls.L = _3882 - P;
                    float3 _3900 = ls.L;
                    float _3901 = length(_3900);
                    ls.L /= _3901.xxx;
                    ls.area = _3358.param1.w;
                    float _3916 = dot(-ls.L, _3887);
                    [flatten]
                    if (_3916 > 0.0f)
                    {
                        ls.pdf = (_3901 * _3901) / (ls.area * _3916);
                    }
                    if ((_3358.type_and_param0.x & 64u) == 0u)
                    {
                        ls.area = 0.0f;
                    }
                    [branch]
                    if ((_3358.type_and_param0.x & 128u) != 0u)
                    {
                        float3 env_col_1 = _3327_g_params.env_col.xyz;
                        uint _3950 = asuint(_3327_g_params.env_col.w);
                        if (_3950 != 4294967295u)
                        {
                            env_col_1 *= SampleLatlong_RGBE(_3950, ls.L, _3327_g_params.env_rotation);
                        }
                        ls.col *= env_col_1;
                        ls.from_env = true;
                    }
                }
                else
                {
                    [branch]
                    if (_3394 == 3u)
                    {
                        float3 _4008 = normalize(cross(P - _3358.param1.xyz, _3358.param3.xyz));
                        float _4015 = 3.1415927410125732421875f * frac(asfloat(_3311.Load((hi + 4) * 4 + 0)) + sample_off.x);
                        float3 _4040 = (_3358.param1.xyz + (((_4008 * cos(_4015)) + (cross(_4008, _3358.param3.xyz) * sin(_4015))) * _3358.param2.w)) + ((_3358.param3.xyz * (frac(asfloat(_3311.Load((hi + 5) * 4 + 0)) + sample_off.y) - 0.5f)) * _3358.param3.w);
                        ls.lp = _4040;
                        float3 _4046 = _4040 - P;
                        float _4049 = length(_4046);
                        ls.L = _4046 / _4049.xxx;
                        ls.area = _3358.param1.w;
                        float _4064 = 1.0f - abs(dot(ls.L, _3358.param3.xyz));
                        [flatten]
                        if (_4064 != 0.0f)
                        {
                            ls.pdf = (_4049 * _4049) / (ls.area * _4064);
                        }
                        if ((_3358.type_and_param0.x & 64u) == 0u)
                        {
                            ls.area = 0.0f;
                        }
                    }
                    else
                    {
                        [branch]
                        if (_3394 == 6u)
                        {
                            uint _4094 = asuint(_3358.param1.x);
                            transform_t _4108;
                            _4108.xform = asfloat(uint4x4(_4102.Load4(asuint(_3358.param1.y) * 128 + 0), _4102.Load4(asuint(_3358.param1.y) * 128 + 16), _4102.Load4(asuint(_3358.param1.y) * 128 + 32), _4102.Load4(asuint(_3358.param1.y) * 128 + 48)));
                            _4108.inv_xform = asfloat(uint4x4(_4102.Load4(asuint(_3358.param1.y) * 128 + 64), _4102.Load4(asuint(_3358.param1.y) * 128 + 80), _4102.Load4(asuint(_3358.param1.y) * 128 + 96), _4102.Load4(asuint(_3358.param1.y) * 128 + 112)));
                            uint _4133 = _4094 * 3u;
                            vertex_t _4139;
                            [unroll]
                            for (int _44ident = 0; _44ident < 3; _44ident++)
                            {
                                _4139.p[_44ident] = asfloat(_4127.Load(_44ident * 4 + _4131.Load(_4133 * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _45ident = 0; _45ident < 3; _45ident++)
                            {
                                _4139.n[_45ident] = asfloat(_4127.Load(_45ident * 4 + _4131.Load(_4133 * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _46ident = 0; _46ident < 3; _46ident++)
                            {
                                _4139.b[_46ident] = asfloat(_4127.Load(_46ident * 4 + _4131.Load(_4133 * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _47ident = 0; _47ident < 2; _47ident++)
                            {
                                [unroll]
                                for (int _48ident = 0; _48ident < 2; _48ident++)
                                {
                                    _4139.t[_47ident][_48ident] = asfloat(_4127.Load(_48ident * 4 + _47ident * 8 + _4131.Load(_4133 * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4188;
                            [unroll]
                            for (int _49ident = 0; _49ident < 3; _49ident++)
                            {
                                _4188.p[_49ident] = asfloat(_4127.Load(_49ident * 4 + _4131.Load((_4133 + 1u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _50ident = 0; _50ident < 3; _50ident++)
                            {
                                _4188.n[_50ident] = asfloat(_4127.Load(_50ident * 4 + _4131.Load((_4133 + 1u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _51ident = 0; _51ident < 3; _51ident++)
                            {
                                _4188.b[_51ident] = asfloat(_4127.Load(_51ident * 4 + _4131.Load((_4133 + 1u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _52ident = 0; _52ident < 2; _52ident++)
                            {
                                [unroll]
                                for (int _53ident = 0; _53ident < 2; _53ident++)
                                {
                                    _4188.t[_52ident][_53ident] = asfloat(_4127.Load(_53ident * 4 + _52ident * 8 + _4131.Load((_4133 + 1u) * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4234;
                            [unroll]
                            for (int _54ident = 0; _54ident < 3; _54ident++)
                            {
                                _4234.p[_54ident] = asfloat(_4127.Load(_54ident * 4 + _4131.Load((_4133 + 2u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _55ident = 0; _55ident < 3; _55ident++)
                            {
                                _4234.n[_55ident] = asfloat(_4127.Load(_55ident * 4 + _4131.Load((_4133 + 2u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _56ident = 0; _56ident < 3; _56ident++)
                            {
                                _4234.b[_56ident] = asfloat(_4127.Load(_56ident * 4 + _4131.Load((_4133 + 2u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _57ident = 0; _57ident < 2; _57ident++)
                            {
                                [unroll]
                                for (int _58ident = 0; _58ident < 2; _58ident++)
                                {
                                    _4234.t[_57ident][_58ident] = asfloat(_4127.Load(_58ident * 4 + _57ident * 8 + _4131.Load((_4133 + 2u) * 4 + 0) * 52 + 36));
                                }
                            }
                            float3 _4280 = float3(_4139.p[0], _4139.p[1], _4139.p[2]);
                            float3 _4288 = float3(_4188.p[0], _4188.p[1], _4188.p[2]);
                            float3 _4296 = float3(_4234.p[0], _4234.p[1], _4234.p[2]);
                            float _4324 = sqrt(frac(asfloat(_3311.Load((hi + 4) * 4 + 0)) + sample_off.x));
                            float _4333 = frac(asfloat(_3311.Load((hi + 5) * 4 + 0)) + sample_off.y);
                            float _4337 = 1.0f - _4324;
                            float _4342 = 1.0f - _4333;
                            float3 _4373 = mul(float4((_4280 * _4337) + (((_4288 * _4342) + (_4296 * _4333)) * _4324), 1.0f), _4108.xform).xyz;
                            float3 _4389 = mul(float4(cross(_4288 - _4280, _4296 - _4280), 0.0f), _4108.xform).xyz;
                            ls.area = 0.5f * length(_4389);
                            float3 _4395 = normalize(_4389);
                            ls.L = _4373 - P;
                            float3 _4402 = ls.L;
                            float _4403 = length(_4402);
                            ls.L /= _4403.xxx;
                            float _4414 = dot(ls.L, _4395);
                            float cos_theta = _4414;
                            float3 _4417;
                            if (_4414 >= 0.0f)
                            {
                                _4417 = -_4395;
                            }
                            else
                            {
                                _4417 = _4395;
                            }
                            float3 param_12 = _4373;
                            float3 param_13 = _4417;
                            ls.lp = offset_ray(param_12, param_13);
                            float _4430 = cos_theta;
                            float _4431 = abs(_4430);
                            cos_theta = _4431;
                            [flatten]
                            if (_4431 > 0.0f)
                            {
                                ls.pdf = (_4403 * _4403) / (ls.area * cos_theta);
                            }
                            material_t _4469;
                            [unroll]
                            for (int _59ident = 0; _59ident < 5; _59ident++)
                            {
                                _4469.textures[_59ident] = _4455.Load(_59ident * 4 + ((_4459.Load(_4094 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
                            }
                            [unroll]
                            for (int _60ident = 0; _60ident < 3; _60ident++)
                            {
                                _4469.base_color[_60ident] = asfloat(_4455.Load(_60ident * 4 + ((_4459.Load(_4094 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
                            }
                            _4469.flags = _4455.Load(((_4459.Load(_4094 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
                            _4469.type = _4455.Load(((_4459.Load(_4094 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
                            _4469.tangent_rotation_or_strength = asfloat(_4455.Load(((_4459.Load(_4094 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
                            _4469.roughness_and_anisotropic = _4455.Load(((_4459.Load(_4094 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
                            _4469.ior = asfloat(_4455.Load(((_4459.Load(_4094 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
                            _4469.sheen_and_sheen_tint = _4455.Load(((_4459.Load(_4094 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
                            _4469.tint_and_metallic = _4455.Load(((_4459.Load(_4094 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
                            _4469.transmission_and_transmission_roughness = _4455.Load(((_4459.Load(_4094 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
                            _4469.specular_and_specular_tint = _4455.Load(((_4459.Load(_4094 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
                            _4469.clearcoat_and_clearcoat_roughness = _4455.Load(((_4459.Load(_4094 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
                            _4469.normal_map_strength_unorm = _4455.Load(((_4459.Load(_4094 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
                            if (_4469.textures[1] != 4294967295u)
                            {
                                ls.col *= SampleBilinear(_4469.textures[1], (float2(_4139.t[0][0], _4139.t[0][1]) * _4337) + (((float2(_4188.t[0][0], _4188.t[0][1]) * _4342) + (float2(_4234.t[0][0], _4234.t[0][1]) * _4333)) * _4324), 0).xyz;
                            }
                        }
                        else
                        {
                            [branch]
                            if (_3394 == 7u)
                            {
                                float _4554 = frac(asfloat(_3311.Load((hi + 4) * 4 + 0)) + sample_off.x);
                                float _4563 = frac(asfloat(_3311.Load((hi + 5) * 4 + 0)) + sample_off.y);
                                float4 dir_and_pdf;
                                if (_3327_g_params.env_qtree_levels > 0)
                                {
                                    dir_and_pdf = Sample_EnvQTree(_3327_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3327_g_params.env_qtree_levels, mad(_3320, _3331, -float(_3338)), _4554, _4563);
                                }
                                else
                                {
                                    float _4582 = 6.283185482025146484375f * _4563;
                                    float _4594 = sqrt(mad(-_4554, _4554, 1.0f));
                                    float3 param_14 = T;
                                    float3 param_15 = B;
                                    float3 param_16 = N;
                                    float3 param_17 = float3(_4594 * cos(_4582), _4594 * sin(_4582), _4554);
                                    dir_and_pdf = float4(world_from_tangent(param_14, param_15, param_16, param_17), 0.15915493667125701904296875f);
                                }
                                ls.L = dir_and_pdf.xyz;
                                ls.col *= _3327_g_params.env_col.xyz;
                                uint _4633 = asuint(_3327_g_params.env_col.w);
                                if (_4633 != 4294967295u)
                                {
                                    ls.col *= SampleLatlong_RGBE(_4633, ls.L, _3327_g_params.env_rotation);
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
    uint _990 = index & 16777215u;
    uint _994_dummy_parameter;
    return int2(spvTextureSize(g_textures[NonUniformResourceIndex(_990)], uint(0), _994_dummy_parameter));
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
    float _2306 = 1.0f / mad(0.904129683971405029296875f, roughness, 3.1415927410125732421875f);
    float _2318 = max(dot(N, L), 0.0f);
    float _2323 = max(dot(N, V), 0.0f);
    float _2331 = mad(-_2318, _2323, dot(L, V));
    float t = _2331;
    if (_2331 > 0.0f)
    {
        t /= (max(_2318, _2323) + 1.1754943508222875079687365372222e-38f);
    }
    return float4(base_color * (_2318 * mad(roughness * _2306, t, _2306)), 0.15915493667125701904296875f);
}

float3 Evaluate_DiffuseNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _8736;
    do
    {
        float3 param = -float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param_1 = surf.N;
        float3 param_2 = ls.L;
        float param_3 = roughness;
        float3 param_4 = base_color;
        float4 _5077 = Evaluate_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_5 = ls.pdf;
            float param_6 = _5077.w;
            mis_weight = power_heuristic(param_5, param_6);
        }
        float3 _5100 = (ls.col * _5077.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8736 = _5100;
            break;
        }
        float3 param_7 = surf.P;
        float3 param_8 = surf.plane_N;
        float3 _5112 = offset_ray(param_7, param_8);
        sh_r.o[0] = _5112.x;
        sh_r.o[1] = _5112.y;
        sh_r.o[2] = _5112.z;
        sh_r.c[0] = ray.c[0] * _5100.x;
        sh_r.c[1] = ray.c[1] * _5100.y;
        sh_r.c[2] = ray.c[2] * _5100.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8736 = 0.0f.xxx;
        break;
    } while(false);
    return _8736;
}

float4 Sample_OrenDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float rand_u, float rand_v, inout float3 out_V)
{
    float _2365 = 6.283185482025146484375f * rand_v;
    float _2377 = sqrt(mad(-rand_u, rand_u, 1.0f));
    float3 param = T;
    float3 param_1 = B;
    float3 param_2 = N;
    float3 param_3 = float3(_2377 * cos(_2365), _2377 * sin(_2365), rand_u);
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
    float4 _5363 = Sample_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8);
    new_ray.depth = ray.depth + 1;
    float3 param_9 = surf.P;
    float3 param_10 = surf.plane_N;
    float3 _5373 = offset_ray(param_9, param_10);
    new_ray.o[0] = _5373.x;
    new_ray.o[1] = _5373.y;
    new_ray.o[2] = _5373.z;
    new_ray.d[0] = param_8.x;
    new_ray.d[1] = param_8.y;
    new_ray.d[2] = param_8.z;
    new_ray.c[0] = ((ray.c[0] * _5363.x) * mix_weight) / _5363.w;
    new_ray.c[1] = ((ray.c[1] * _5363.y) * mix_weight) / _5363.w;
    new_ray.c[2] = ((ray.c[2] * _5363.z) * mix_weight) / _5363.w;
    new_ray.pdf = _5363.w;
}

float3 tangent_from_world(float3 T, float3 B, float3 N, float3 V)
{
    return float3(dot(V, T), dot(V, B), dot(V, N));
}

float D_GGX(float3 H, float alpha_x, float alpha_y)
{
    float _8789;
    do
    {
        if (H.z == 0.0f)
        {
            _8789 = 0.0f;
            break;
        }
        float _2032 = (-H.x) / (H.z * alpha_x);
        float _2038 = (-H.y) / (H.z * alpha_y);
        float _2047 = mad(_2038, _2038, mad(_2032, _2032, 1.0f));
        _8789 = 1.0f / (((((_2047 * _2047) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _8789;
}

float G1(float3 Ve, inout float alpha_x, inout float alpha_y)
{
    alpha_x *= alpha_x;
    alpha_y *= alpha_y;
    return 1.0f / mad((-1.0f) + sqrt(1.0f + (mad(alpha_x * Ve.x, Ve.x, (alpha_y * Ve.y) * Ve.y) / (Ve.z * Ve.z))), 0.5f, 1.0f);
}

float4 Evaluate_GGXSpecular_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float alpha_x, float alpha_y, float spec_ior, float spec_F0, float3 spec_col)
{
    float _2547 = D_GGX(sampled_normal_ts, alpha_x, alpha_y);
    float3 param = view_dir_ts;
    float param_1 = alpha_x;
    float param_2 = alpha_y;
    float _2555 = G1(param, param_1, param_2);
    float3 param_3 = reflected_dir_ts;
    float param_4 = alpha_x;
    float param_5 = alpha_y;
    float _2562 = G1(param_3, param_4, param_5);
    float param_6 = dot(view_dir_ts, sampled_normal_ts);
    float param_7 = spec_ior;
    float3 F = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param_6, param_7) - spec_F0) / (1.0f - spec_F0)).xxx);
    float _2590 = 4.0f * abs(view_dir_ts.z * reflected_dir_ts.z);
    float _2593;
    if (_2590 != 0.0f)
    {
        _2593 = (_2547 * (_2555 * _2562)) / _2590;
    }
    else
    {
        _2593 = 0.0f;
    }
    F *= _2593;
    float3 param_8 = view_dir_ts;
    float param_9 = alpha_x;
    float param_10 = alpha_y;
    float _2613 = G1(param_8, param_9, param_10);
    float pdf = ((_2547 * _2613) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2628 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2628 != 0.0f)
    {
        pdf /= _2628;
    }
    float3 _2639 = F;
    float3 _2640 = _2639 * max(reflected_dir_ts.z, 0.0f);
    F = _2640;
    return float4(_2640, pdf);
}

float3 Evaluate_GlossyNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _8741;
    do
    {
        float3 _5148 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5148;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - _5148);
        float _5186 = roughness * roughness;
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = _5186;
        float param_16 = _5186;
        float param_17 = spec_ior;
        float param_18 = spec_F0;
        float3 param_19 = base_color;
        float4 _5201 = Evaluate_GGXSpecular_BSDF(param_12, param_13, param_14, param_15, param_16, param_17, param_18, param_19);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_20 = ls.pdf;
            float param_21 = _5201.w;
            mis_weight = power_heuristic(param_20, param_21);
        }
        float3 _5224 = (ls.col * _5201.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8741 = _5224;
            break;
        }
        float3 param_22 = surf.P;
        float3 param_23 = surf.plane_N;
        float3 _5236 = offset_ray(param_22, param_23);
        sh_r.o[0] = _5236.x;
        sh_r.o[1] = _5236.y;
        sh_r.o[2] = _5236.z;
        sh_r.c[0] = ray.c[0] * _5224.x;
        sh_r.c[1] = ray.c[1] * _5224.y;
        sh_r.c[2] = ray.c[2] * _5224.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8741 = 0.0f.xxx;
        break;
    } while(false);
    return _8741;
}

float3 SampleGGX_VNDF(float3 Ve, float alpha_x, float alpha_y, float U1, float U2)
{
    float3 _1850 = normalize(float3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
    float _1853 = _1850.x;
    float _1858 = _1850.y;
    float _1862 = mad(_1853, _1853, _1858 * _1858);
    float3 _1866;
    if (_1862 > 0.0f)
    {
        _1866 = float3(-_1858, _1853, 0.0f) / sqrt(_1862).xxx;
    }
    else
    {
        _1866 = float3(1.0f, 0.0f, 0.0f);
    }
    float _1888 = sqrt(U1);
    float _1891 = 6.283185482025146484375f * U2;
    float _1896 = _1888 * cos(_1891);
    float _1905 = 1.0f + _1850.z;
    float _1912 = mad(-_1896, _1896, 1.0f);
    float _1918 = mad(mad(-0.5f, _1905, 1.0f), sqrt(_1912), (0.5f * _1905) * (_1888 * sin(_1891)));
    float3 _1939 = ((_1866 * _1896) + (cross(_1850, _1866) * _1918)) + (_1850 * sqrt(max(0.0f, mad(-_1918, _1918, _1912))));
    return normalize(float3(alpha_x * _1939.x, alpha_y * _1939.y, max(0.0f, _1939.z)));
}

float4 Sample_GGXSpecular_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float anisotropic, float spec_ior, float spec_F0, float3 spec_col, float rand_u, float rand_v, inout float3 out_V)
{
    float4 _8761;
    do
    {
        float _2650 = roughness * roughness;
        float _2654 = sqrt(mad(-0.89999997615814208984375f, anisotropic, 1.0f));
        float _2658 = _2650 / _2654;
        float _2662 = _2650 * _2654;
        [branch]
        if ((_2658 * _2662) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2673 = reflect(I, N);
            float param = dot(_2673, N);
            float param_1 = spec_ior;
            float3 _2687 = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param, param_1) - spec_F0) / (1.0f - spec_F0)).xxx);
            out_V = _2673;
            _8761 = float4(_2687.x * 1000000.0f, _2687.y * 1000000.0f, _2687.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2712 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = _2658;
        float param_7 = _2662;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2721 = SampleGGX_VNDF(_2712, param_6, param_7, param_8, param_9);
        float3 _2732 = normalize(reflect(-_2712, _2721));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2732;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2712;
        float3 param_15 = _2721;
        float3 param_16 = _2732;
        float param_17 = _2658;
        float param_18 = _2662;
        float param_19 = spec_ior;
        float param_20 = spec_F0;
        float3 param_21 = spec_col;
        _8761 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _8761;
}

void Sample_GlossyNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 param_4;
    float4 _5283 = Sample_GGXSpecular_BSDF(param, param_1, param_2, param_3, roughness, 0.0f, spec_ior, spec_F0, base_color, rand_u, rand_v, param_4);
    new_ray.depth = ray.depth + 256;
    float3 param_5 = surf.P;
    float3 param_6 = surf.plane_N;
    float3 _5294 = offset_ray(param_5, param_6);
    new_ray.o[0] = _5294.x;
    new_ray.o[1] = _5294.y;
    new_ray.o[2] = _5294.z;
    new_ray.d[0] = param_4.x;
    new_ray.d[1] = param_4.y;
    new_ray.d[2] = param_4.z;
    new_ray.c[0] = ((ray.c[0] * _5283.x) * mix_weight) / _5283.w;
    new_ray.c[1] = ((ray.c[1] * _5283.y) * mix_weight) / _5283.w;
    new_ray.c[2] = ((ray.c[2] * _5283.z) * mix_weight) / _5283.w;
    new_ray.pdf = _5283.w;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _8766;
    do
    {
        bool _2954 = refr_dir_ts.z >= 0.0f;
        bool _2961;
        if (!_2954)
        {
            _2961 = view_dir_ts.z <= 0.0f;
        }
        else
        {
            _2961 = _2954;
        }
        if (_2961)
        {
            _8766 = 0.0f.xxxx;
            break;
        }
        float _2970 = D_GGX(sampled_normal_ts, roughness2, roughness2);
        float3 param = refr_dir_ts;
        float param_1 = roughness2;
        float param_2 = roughness2;
        float _2978 = G1(param, param_1, param_2);
        float3 param_3 = view_dir_ts;
        float param_4 = roughness2;
        float param_5 = roughness2;
        float _2986 = G1(param_3, param_4, param_5);
        float _2996 = mad(dot(view_dir_ts, sampled_normal_ts), eta, dot(refr_dir_ts, sampled_normal_ts));
        float _3006 = clamp(-dot(refr_dir_ts, sampled_normal_ts), 0.0f, 1.0f) / (_2996 * _2996);
        _8766 = float4(refr_col * (((((_2970 * _2986) * _2978) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3006) / view_dir_ts.z), (((_2970 * _2978) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3006) / view_dir_ts.z);
        break;
    } while(false);
    return _8766;
}

float3 Evaluate_RefractiveNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness2, float eta, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _8746;
    do
    {
        float3 _5426 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5426;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - (_5426 * eta));
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = roughness2;
        float param_16 = eta;
        float3 param_17 = base_color;
        float4 _5474 = Evaluate_GGXRefraction_BSDF(param_12, param_13, param_14, param_15, param_16, param_17);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_18 = ls.pdf;
            float param_19 = _5474.w;
            mis_weight = power_heuristic(param_18, param_19);
        }
        float3 _5497 = (ls.col * _5474.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8746 = _5497;
            break;
        }
        float3 param_20 = surf.P;
        float3 param_21 = -surf.plane_N;
        float3 _5510 = offset_ray(param_20, param_21);
        sh_r.o[0] = _5510.x;
        sh_r.o[1] = _5510.y;
        sh_r.o[2] = _5510.z;
        sh_r.c[0] = ray.c[0] * _5497.x;
        sh_r.c[1] = ray.c[1] * _5497.y;
        sh_r.c[2] = ray.c[2] * _5497.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8746 = 0.0f.xxx;
        break;
    } while(false);
    return _8746;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _8771;
    do
    {
        float _3050 = roughness * roughness;
        [branch]
        if ((_3050 * _3050) < 1.0000000116860974230803549289703e-07f)
        {
            float _3060 = dot(I, N);
            float _3061 = -_3060;
            float _3071 = mad(-(eta * eta), mad(_3060, _3061, 1.0f), 1.0f);
            if (_3071 < 0.0f)
            {
                _8771 = 0.0f.xxxx;
                break;
            }
            float _3083 = mad(eta, _3061, -sqrt(_3071));
            out_V = float4(normalize((I * eta) + (N * _3083)), _3083);
            _8771 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param = T;
        float3 param_1 = B;
        float3 param_2 = N;
        float3 param_3 = -I;
        float3 _3123 = normalize(tangent_from_world(param, param_1, param_2, param_3));
        float param_4 = _3050;
        float param_5 = _3050;
        float param_6 = rand_u;
        float param_7 = rand_v;
        float3 _3134 = SampleGGX_VNDF(_3123, param_4, param_5, param_6, param_7);
        float _3138 = dot(_3123, _3134);
        float _3148 = mad(-(eta * eta), mad(-_3138, _3138, 1.0f), 1.0f);
        if (_3148 < 0.0f)
        {
            _8771 = 0.0f.xxxx;
            break;
        }
        float _3160 = mad(eta, _3138, -sqrt(_3148));
        float3 _3170 = normalize((_3123 * (-eta)) + (_3134 * _3160));
        float3 param_8 = _3123;
        float3 param_9 = _3134;
        float3 param_10 = _3170;
        float param_11 = _3050;
        float param_12 = eta;
        float3 param_13 = refr_col;
        float3 param_14 = T;
        float3 param_15 = B;
        float3 param_16 = N;
        float3 param_17 = _3170;
        out_V = float4(world_from_tangent(param_14, param_15, param_16, param_17), _3160);
        _8771 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _8771;
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
    float _2096 = old_value;
    old_value = new_value;
    return _2096;
}

float pop_ior_stack(inout float stack[4], float default_value)
{
    float _8779;
    do
    {
        if (stack[3] > 0.0f)
        {
            float param = stack[3];
            float param_1 = -1.0f;
            float _2138 = exchange(param, param_1);
            stack[3] = param;
            _8779 = _2138;
            break;
        }
        if (stack[2] > 0.0f)
        {
            float param_2 = stack[2];
            float param_3 = -1.0f;
            float _2151 = exchange(param_2, param_3);
            stack[2] = param_2;
            _8779 = _2151;
            break;
        }
        if (stack[1] > 0.0f)
        {
            float param_4 = stack[1];
            float param_5 = -1.0f;
            float _2164 = exchange(param_4, param_5);
            stack[1] = param_4;
            _8779 = _2164;
            break;
        }
        if (stack[0] > 0.0f)
        {
            float param_6 = stack[0];
            float param_7 = -1.0f;
            float _2177 = exchange(param_6, param_7);
            stack[0] = param_6;
            _8779 = _2177;
            break;
        }
        _8779 = default_value;
        break;
    } while(false);
    return _8779;
}

void Sample_RefractiveNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, bool is_backfacing, float int_ior, float ext_ior, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float _5547;
    if (is_backfacing)
    {
        _5547 = int_ior / ext_ior;
    }
    else
    {
        _5547 = ext_ior / int_ior;
    }
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float param_4 = roughness;
    float param_5 = _5547;
    float3 param_6 = base_color;
    float param_7 = rand_u;
    float param_8 = rand_v;
    float4 param_9;
    float4 _5571 = Sample_GGXRefraction_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9);
    new_ray.depth = ray.depth + 65536;
    new_ray.c[0] = ((ray.c[0] * _5571.x) * mix_weight) / _5571.w;
    new_ray.c[1] = ((ray.c[1] * _5571.y) * mix_weight) / _5571.w;
    new_ray.c[2] = ((ray.c[2] * _5571.z) * mix_weight) / _5571.w;
    new_ray.pdf = _5571.w;
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
        float _5627 = pop_ior_stack(param_11, param_12);
        new_ray.ior = param_11;
    }
    float3 param_13 = surf.P;
    float3 param_14 = -surf.plane_N;
    float3 _5636 = offset_ray(param_13, param_14);
    new_ray.o[0] = _5636.x;
    new_ray.o[1] = _5636.y;
    new_ray.o[2] = _5636.z;
    new_ray.d[0] = param_9.x;
    new_ray.d[1] = param_9.y;
    new_ray.d[2] = param_9.z;
}

lobe_weights_t get_lobe_weights(float base_color_lum, float spec_color_lum, float specular, float metallic, float transmission, float clearcoat)
{
    float _1502 = 1.0f - metallic;
    float _8865 = (base_color_lum * _1502) * (1.0f - transmission);
    float _1509 = transmission * _1502;
    float _1513;
    if ((specular != 0.0f) || (metallic != 0.0f))
    {
        _1513 = spec_color_lum * mad(-transmission, _1502, 1.0f);
    }
    else
    {
        _1513 = 0.0f;
    }
    float _8866 = _1513;
    float _1523 = 0.25f * clearcoat;
    float _8867 = _1523 * _1502;
    float _8868 = _1509 * base_color_lum;
    float _1532 = _8865;
    float _1541 = mad(_1509, base_color_lum, mad(_1523, _1502, _1532 + _1513));
    if (_1541 != 0.0f)
    {
        _8865 /= _1541;
        _8866 /= _1541;
        _8867 /= _1541;
        _8868 /= _1541;
    }
    lobe_weights_t _8873 = { _8865, _8866, _8867, _8868 };
    return _8873;
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
    float _8794;
    do
    {
        float _2258 = dot(N, L);
        if (_2258 <= 0.0f)
        {
            _8794 = 0.0f;
            break;
        }
        float param = _2258;
        float param_1 = dot(N, V);
        float _2279 = dot(L, H);
        float _2287 = mad((2.0f * _2279) * _2279, roughness, 0.5f);
        _8794 = lerp(1.0f, _2287, schlick_weight(param)) * lerp(1.0f, _2287, schlick_weight(param_1));
        break;
    } while(false);
    return _8794;
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
    float3 _2428 = normalize(L + V);
    float3 H = _2428;
    if (dot(V, _2428) < 0.0f)
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
    float3 _2463 = diff_col;
    float3 _2464 = _2463 + (sheen_color * (3.1415927410125732421875f * schlick_weight(param_5)));
    diff_col = _2464;
    return float4(_2464, pdf);
}

float D_GTR1(float NDotH, float a)
{
    float _8799;
    do
    {
        if (a >= 1.0f)
        {
            _8799 = 0.3183098733425140380859375f;
            break;
        }
        float _2006 = mad(a, a, -1.0f);
        _8799 = _2006 / ((3.1415927410125732421875f * log(a * a)) * mad(_2006 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _8799;
}

float4 Evaluate_PrincipledClearcoat_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float clearcoat_roughness2, float clearcoat_ior, float clearcoat_F0)
{
    float param = sampled_normal_ts.z;
    float param_1 = clearcoat_roughness2;
    float _2764 = D_GTR1(param, param_1);
    float3 param_2 = view_dir_ts;
    float param_3 = 0.0625f;
    float param_4 = 0.0625f;
    float _2771 = G1(param_2, param_3, param_4);
    float3 param_5 = reflected_dir_ts;
    float param_6 = 0.0625f;
    float param_7 = 0.0625f;
    float _2776 = G1(param_5, param_6, param_7);
    float param_8 = dot(reflected_dir_ts, sampled_normal_ts);
    float param_9 = clearcoat_ior;
    float F = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param_8, param_9) - clearcoat_F0) / (1.0f - clearcoat_F0));
    float _2803 = (4.0f * abs(view_dir_ts.z)) * abs(reflected_dir_ts.z);
    float _2806;
    if (_2803 != 0.0f)
    {
        _2806 = (_2764 * (_2771 * _2776)) / _2803;
    }
    else
    {
        _2806 = 0.0f;
    }
    F *= _2806;
    float3 param_10 = view_dir_ts;
    float param_11 = 0.0625f;
    float param_12 = 0.0625f;
    float _2824 = G1(param_10, param_11, param_12);
    float pdf = ((_2764 * _2824) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2839 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2839 != 0.0f)
    {
        pdf /= _2839;
    }
    float _2850 = F;
    float _2851 = _2850 * clamp(reflected_dir_ts.z, 0.0f, 1.0f);
    F = _2851;
    return float4(_2851, _2851, _2851, pdf);
}

float3 Evaluate_PrincipledNode(light_sample_t ls, ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float N_dot_L, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _8751;
    do
    {
        float3 _5659 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 lcol = 0.0f.xxx;
        float bsdf_pdf = 0.0f;
        bool _5664 = N_dot_L > 0.0f;
        [branch]
        if ((lobe_weights.diffuse > 1.0000000116860974230803549289703e-07f) && _5664)
        {
            float3 param = -_5659;
            float3 param_1 = surf.N;
            float3 param_2 = ls.L;
            float param_3 = diff.roughness;
            float3 param_4 = diff.base_color;
            float3 param_5 = diff.sheen_color;
            bool param_6 = false;
            float4 _5683 = Evaluate_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6);
            bsdf_pdf = mad(lobe_weights.diffuse, _5683.w, bsdf_pdf);
            lcol += (((ls.col * N_dot_L) * (_5683 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * ls.pdf).xxx);
        }
        float3 H;
        [flatten]
        if (_5664)
        {
            H = normalize(ls.L - _5659);
        }
        else
        {
            H = normalize(ls.L - (_5659 * trans.eta));
        }
        float _5722 = spec.roughness * spec.roughness;
        float _5727 = sqrt(mad(-0.89999997615814208984375f, spec.anisotropy, 1.0f));
        float _5731 = _5722 / _5727;
        float _5735 = _5722 * _5727;
        float3 param_7 = surf.T;
        float3 param_8 = surf.B;
        float3 param_9 = surf.N;
        float3 param_10 = -_5659;
        float3 _5746 = tangent_from_world(param_7, param_8, param_9, param_10);
        float3 param_11 = surf.T;
        float3 param_12 = surf.B;
        float3 param_13 = surf.N;
        float3 param_14 = ls.L;
        float3 _5756 = tangent_from_world(param_11, param_12, param_13, param_14);
        float3 param_15 = surf.T;
        float3 param_16 = surf.B;
        float3 param_17 = surf.N;
        float3 param_18 = H;
        float3 _5766 = tangent_from_world(param_15, param_16, param_17, param_18);
        bool _5768 = lobe_weights.specular > 0.0f;
        bool _5775;
        if (_5768)
        {
            _5775 = (_5731 * _5735) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _5775 = _5768;
        }
        [branch]
        if (_5775 && _5664)
        {
            float3 param_19 = _5746;
            float3 param_20 = _5766;
            float3 param_21 = _5756;
            float param_22 = _5731;
            float param_23 = _5735;
            float param_24 = spec.ior;
            float param_25 = spec.F0;
            float3 param_26 = spec.tmp_col;
            float4 _5797 = Evaluate_GGXSpecular_BSDF(param_19, param_20, param_21, param_22, param_23, param_24, param_25, param_26);
            bsdf_pdf = mad(lobe_weights.specular, _5797.w, bsdf_pdf);
            lcol += ((ls.col * _5797.xyz) / ls.pdf.xxx);
        }
        float _5816 = coat.roughness * coat.roughness;
        bool _5818 = lobe_weights.clearcoat > 0.0f;
        bool _5825;
        if (_5818)
        {
            _5825 = (_5816 * _5816) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _5825 = _5818;
        }
        [branch]
        if (_5825 && _5664)
        {
            float3 param_27 = _5746;
            float3 param_28 = _5766;
            float3 param_29 = _5756;
            float param_30 = _5816;
            float param_31 = coat.ior;
            float param_32 = coat.F0;
            float4 _5843 = Evaluate_PrincipledClearcoat_BSDF(param_27, param_28, param_29, param_30, param_31, param_32);
            bsdf_pdf = mad(lobe_weights.clearcoat, _5843.w, bsdf_pdf);
            lcol += (((ls.col * 0.25f) * _5843.xyz) / ls.pdf.xxx);
        }
        [branch]
        if (lobe_weights.refraction > 0.0f)
        {
            bool _5865 = trans.fresnel != 0.0f;
            bool _5872;
            if (_5865)
            {
                _5872 = (_5722 * _5722) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _5872 = _5865;
            }
            [branch]
            if (_5872 && _5664)
            {
                float3 param_33 = _5746;
                float3 param_34 = _5766;
                float3 param_35 = _5756;
                float param_36 = _5722;
                float param_37 = _5722;
                float param_38 = 1.0f;
                float param_39 = 0.0f;
                float3 param_40 = 1.0f.xxx;
                float4 _5891 = Evaluate_GGXSpecular_BSDF(param_33, param_34, param_35, param_36, param_37, param_38, param_39, param_40);
                bsdf_pdf = mad(lobe_weights.refraction * trans.fresnel, _5891.w, bsdf_pdf);
                lcol += ((ls.col * _5891.xyz) * (trans.fresnel / ls.pdf));
            }
            float _5913 = trans.roughness * trans.roughness;
            bool _5915 = trans.fresnel != 1.0f;
            bool _5922;
            if (_5915)
            {
                _5922 = (_5913 * _5913) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _5922 = _5915;
            }
            [branch]
            if (_5922 && (N_dot_L < 0.0f))
            {
                float3 param_41 = _5746;
                float3 param_42 = _5766;
                float3 param_43 = _5756;
                float param_44 = _5913;
                float param_45 = trans.eta;
                float3 param_46 = diff.base_color;
                float4 _5940 = Evaluate_GGXRefraction_BSDF(param_41, param_42, param_43, param_44, param_45, param_46);
                float _5943 = 1.0f - trans.fresnel;
                bsdf_pdf = mad(lobe_weights.refraction * _5943, _5940.w, bsdf_pdf);
                lcol += ((ls.col * _5940.xyz) * (_5943 / ls.pdf));
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
            _8751 = lcol;
            break;
        }
        float3 _5983;
        if (N_dot_L < 0.0f)
        {
            _5983 = -surf.plane_N;
        }
        else
        {
            _5983 = surf.plane_N;
        }
        float3 param_49 = surf.P;
        float3 param_50 = _5983;
        float3 _5994 = offset_ray(param_49, param_50);
        sh_r.o[0] = _5994.x;
        sh_r.o[1] = _5994.y;
        sh_r.o[2] = _5994.z;
        sh_r.c[0] = ray.c[0] * lcol.x;
        sh_r.c[1] = ray.c[1] * lcol.y;
        sh_r.c[2] = ray.c[2] * lcol.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8751 = 0.0f.xxx;
        break;
    } while(false);
    return _8751;
}

float4 Sample_PrincipledDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float3 sheen_color, bool uniform_sampling, float rand_u, float rand_v, inout float3 out_V)
{
    float _2475 = 6.283185482025146484375f * rand_v;
    float _2478 = cos(_2475);
    float _2481 = sin(_2475);
    float3 V;
    if (uniform_sampling)
    {
        float _2490 = sqrt(mad(-rand_u, rand_u, 1.0f));
        V = float3(_2490 * _2478, _2490 * _2481, rand_u);
    }
    else
    {
        float _2503 = sqrt(rand_u);
        V = float3(_2503 * _2478, _2503 * _2481, sqrt(1.0f - rand_u));
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
    float4 _8784;
    do
    {
        [branch]
        if ((clearcoat_roughness2 * clearcoat_roughness2) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2868 = reflect(I, N);
            float param = dot(_2868, N);
            float param_1 = clearcoat_ior;
            out_V = _2868;
            float _2887 = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param, param_1) - clearcoat_F0) / (1.0f - clearcoat_F0)) * 1000000.0f;
            _8784 = float4(_2887, _2887, _2887, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2905 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = clearcoat_roughness2;
        float param_7 = clearcoat_roughness2;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2916 = SampleGGX_VNDF(_2905, param_6, param_7, param_8, param_9);
        float3 _2927 = normalize(reflect(-_2905, _2916));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2927;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2905;
        float3 param_15 = _2916;
        float3 param_16 = _2927;
        float param_17 = clearcoat_roughness2;
        float param_18 = clearcoat_ior;
        float param_19 = clearcoat_F0;
        _8784 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _8784;
}

void Sample_PrincipledNode(ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float rand_u, float rand_v, inout float mix_rand, float mix_weight, inout ray_data_t new_ray)
{
    float3 _6029 = float3(ray.d[0], ray.d[1], ray.d[2]);
    int _6033 = ray.depth & 255;
    int _6037 = (ray.depth >> 8) & 255;
    int _6041 = (ray.depth >> 16) & 255;
    int _6052 = (_6033 + _6037) + _6041;
    [branch]
    if (mix_rand < lobe_weights.diffuse)
    {
        bool _6061 = _6033 < _3327_g_params.max_diff_depth;
        bool _6068;
        if (_6061)
        {
            _6068 = _6052 < _3327_g_params.max_total_depth;
        }
        else
        {
            _6068 = _6061;
        }
        if (_6068)
        {
            float3 param = surf.T;
            float3 param_1 = surf.B;
            float3 param_2 = surf.N;
            float3 param_3 = _6029;
            float param_4 = diff.roughness;
            float3 param_5 = diff.base_color;
            float3 param_6 = diff.sheen_color;
            bool param_7 = false;
            float param_8 = rand_u;
            float param_9 = rand_v;
            float3 param_10;
            float4 _6091 = Sample_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9, param_10);
            float3 _6096 = _6091.xyz * (1.0f - metallic);
            new_ray.depth = ray.depth + 1;
            float3 param_11 = surf.P;
            float3 param_12 = surf.plane_N;
            float3 _6111 = offset_ray(param_11, param_12);
            new_ray.o[0] = _6111.x;
            new_ray.o[1] = _6111.y;
            new_ray.o[2] = _6111.z;
            new_ray.d[0] = param_10.x;
            new_ray.d[1] = param_10.y;
            new_ray.d[2] = param_10.z;
            new_ray.c[0] = ((ray.c[0] * _6096.x) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[1] = ((ray.c[1] * _6096.y) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[2] = ((ray.c[2] * _6096.z) * mix_weight) / lobe_weights.diffuse;
            new_ray.pdf = _6091.w;
        }
    }
    else
    {
        float _6161 = lobe_weights.diffuse + lobe_weights.specular;
        [branch]
        if (mix_rand < _6161)
        {
            bool _6168 = _6037 < _3327_g_params.max_spec_depth;
            bool _6175;
            if (_6168)
            {
                _6175 = _6052 < _3327_g_params.max_total_depth;
            }
            else
            {
                _6175 = _6168;
            }
            if (_6175)
            {
                float3 param_13 = surf.T;
                float3 param_14 = surf.B;
                float3 param_15 = surf.N;
                float3 param_16 = _6029;
                float3 param_17;
                float4 _6194 = Sample_GGXSpecular_BSDF(param_13, param_14, param_15, param_16, spec.roughness, spec.anisotropy, spec.ior, spec.F0, spec.tmp_col, rand_u, rand_v, param_17);
                float _6199 = _6194.w * lobe_weights.specular;
                new_ray.depth = ray.depth + 256;
                new_ray.c[0] = ((ray.c[0] * _6194.x) * mix_weight) / _6199;
                new_ray.c[1] = ((ray.c[1] * _6194.y) * mix_weight) / _6199;
                new_ray.c[2] = ((ray.c[2] * _6194.z) * mix_weight) / _6199;
                new_ray.pdf = _6199;
                float3 param_18 = surf.P;
                float3 param_19 = surf.plane_N;
                float3 _6239 = offset_ray(param_18, param_19);
                new_ray.o[0] = _6239.x;
                new_ray.o[1] = _6239.y;
                new_ray.o[2] = _6239.z;
                new_ray.d[0] = param_17.x;
                new_ray.d[1] = param_17.y;
                new_ray.d[2] = param_17.z;
            }
        }
        else
        {
            float _6264 = _6161 + lobe_weights.clearcoat;
            [branch]
            if (mix_rand < _6264)
            {
                bool _6271 = _6037 < _3327_g_params.max_spec_depth;
                bool _6278;
                if (_6271)
                {
                    _6278 = _6052 < _3327_g_params.max_total_depth;
                }
                else
                {
                    _6278 = _6271;
                }
                if (_6278)
                {
                    float3 param_20 = surf.T;
                    float3 param_21 = surf.B;
                    float3 param_22 = surf.N;
                    float3 param_23 = _6029;
                    float param_24 = coat.roughness * coat.roughness;
                    float param_25 = coat.ior;
                    float param_26 = coat.F0;
                    float param_27 = rand_u;
                    float param_28 = rand_v;
                    float3 param_29;
                    float4 _6302 = Sample_PrincipledClearcoat_BSDF(param_20, param_21, param_22, param_23, param_24, param_25, param_26, param_27, param_28, param_29);
                    float _6307 = _6302.w * lobe_weights.clearcoat;
                    new_ray.depth = ray.depth + 256;
                    new_ray.c[0] = (((0.25f * ray.c[0]) * _6302.x) * mix_weight) / _6307;
                    new_ray.c[1] = (((0.25f * ray.c[1]) * _6302.y) * mix_weight) / _6307;
                    new_ray.c[2] = (((0.25f * ray.c[2]) * _6302.z) * mix_weight) / _6307;
                    new_ray.pdf = _6307;
                    float3 param_30 = surf.P;
                    float3 param_31 = surf.plane_N;
                    float3 _6350 = offset_ray(param_30, param_31);
                    new_ray.o[0] = _6350.x;
                    new_ray.o[1] = _6350.y;
                    new_ray.o[2] = _6350.z;
                    new_ray.d[0] = param_29.x;
                    new_ray.d[1] = param_29.y;
                    new_ray.d[2] = param_29.z;
                }
            }
            else
            {
                bool _6372 = mix_rand >= trans.fresnel;
                bool _6379;
                if (_6372)
                {
                    _6379 = _6041 < _3327_g_params.max_refr_depth;
                }
                else
                {
                    _6379 = _6372;
                }
                bool _6393;
                if (!_6379)
                {
                    bool _6385 = mix_rand < trans.fresnel;
                    bool _6392;
                    if (_6385)
                    {
                        _6392 = _6037 < _3327_g_params.max_spec_depth;
                    }
                    else
                    {
                        _6392 = _6385;
                    }
                    _6393 = _6392;
                }
                else
                {
                    _6393 = _6379;
                }
                bool _6400;
                if (_6393)
                {
                    _6400 = _6052 < _3327_g_params.max_total_depth;
                }
                else
                {
                    _6400 = _6393;
                }
                [branch]
                if (_6400)
                {
                    mix_rand -= _6264;
                    mix_rand /= lobe_weights.refraction;
                    float4 F;
                    float3 V;
                    [branch]
                    if (mix_rand < trans.fresnel)
                    {
                        float3 param_32 = surf.T;
                        float3 param_33 = surf.B;
                        float3 param_34 = surf.N;
                        float3 param_35 = _6029;
                        float3 param_36;
                        float4 _6430 = Sample_GGXSpecular_BSDF(param_32, param_33, param_34, param_35, spec.roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, rand_u, rand_v, param_36);
                        V = param_36;
                        F = _6430;
                        new_ray.depth = ray.depth + 256;
                        float3 param_37 = surf.P;
                        float3 param_38 = surf.plane_N;
                        float3 _6440 = offset_ray(param_37, param_38);
                        new_ray.o[0] = _6440.x;
                        new_ray.o[1] = _6440.y;
                        new_ray.o[2] = _6440.z;
                    }
                    else
                    {
                        float3 param_39 = surf.T;
                        float3 param_40 = surf.B;
                        float3 param_41 = surf.N;
                        float3 param_42 = _6029;
                        float param_43 = trans.roughness;
                        float param_44 = trans.eta;
                        float3 param_45 = diff.base_color;
                        float param_46 = rand_u;
                        float param_47 = rand_v;
                        float4 param_48;
                        float4 _6469 = Sample_GGXRefraction_BSDF(param_39, param_40, param_41, param_42, param_43, param_44, param_45, param_46, param_47, param_48);
                        F = _6469;
                        V = param_48.xyz;
                        new_ray.depth = ray.depth + 65536;
                        float3 param_49 = surf.P;
                        float3 param_50 = -surf.plane_N;
                        float3 _6482 = offset_ray(param_49, param_50);
                        new_ray.o[0] = _6482.x;
                        new_ray.o[1] = _6482.y;
                        new_ray.o[2] = _6482.z;
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
                            float _6508 = pop_ior_stack(param_52, param_53);
                            new_ray.ior = param_52;
                        }
                    }
                    float4 _10023 = F;
                    float _6514 = _10023.w * lobe_weights.refraction;
                    float4 _10025 = _10023;
                    _10025.w = _6514;
                    F = _10025;
                    new_ray.c[0] = ((ray.c[0] * _10023.x) * mix_weight) / _6514;
                    new_ray.c[1] = ((ray.c[1] * _10023.y) * mix_weight) / _6514;
                    new_ray.c[2] = ((ray.c[2] * _10023.z) * mix_weight) / _6514;
                    new_ray.pdf = _6514;
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
    float3 _8721;
    do
    {
        float3 _6570 = float3(ray.d[0], ray.d[1], ray.d[2]);
        [branch]
        if (inter.mask == 0)
        {
            ray_data_t param = ray;
            float3 _6579 = Evaluate_EnvColor(param);
            _8721 = float3(ray.c[0] * _6579.x, ray.c[1] * _6579.y, ray.c[2] * _6579.z);
            break;
        }
        float3 _6606 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_6570 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            ray_data_t param_1 = ray;
            hit_data_t param_2 = inter;
            float3 _6618 = Evaluate_LightColor(param_1, param_2);
            _8721 = float3(ray.c[0] * _6618.x, ray.c[1] * _6618.y, ray.c[2] * _6618.z);
            break;
        }
        bool _6639 = inter.prim_index < 0;
        int _6642;
        if (_6639)
        {
            _6642 = (-1) - inter.prim_index;
        }
        else
        {
            _6642 = inter.prim_index;
        }
        uint _6653 = uint(_6642);
        material_t _6661;
        [unroll]
        for (int _61ident = 0; _61ident < 5; _61ident++)
        {
            _6661.textures[_61ident] = _4455.Load(_61ident * 4 + ((_4459.Load(_6653 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
        }
        [unroll]
        for (int _62ident = 0; _62ident < 3; _62ident++)
        {
            _6661.base_color[_62ident] = asfloat(_4455.Load(_62ident * 4 + ((_4459.Load(_6653 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
        }
        _6661.flags = _4455.Load(((_4459.Load(_6653 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
        _6661.type = _4455.Load(((_4459.Load(_6653 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
        _6661.tangent_rotation_or_strength = asfloat(_4455.Load(((_4459.Load(_6653 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
        _6661.roughness_and_anisotropic = _4455.Load(((_4459.Load(_6653 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
        _6661.ior = asfloat(_4455.Load(((_4459.Load(_6653 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
        _6661.sheen_and_sheen_tint = _4455.Load(((_4459.Load(_6653 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
        _6661.tint_and_metallic = _4455.Load(((_4459.Load(_6653 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
        _6661.transmission_and_transmission_roughness = _4455.Load(((_4459.Load(_6653 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
        _6661.specular_and_specular_tint = _4455.Load(((_4459.Load(_6653 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
        _6661.clearcoat_and_clearcoat_roughness = _4455.Load(((_4459.Load(_6653 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
        _6661.normal_map_strength_unorm = _4455.Load(((_4459.Load(_6653 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
        uint _9472 = _6661.textures[0];
        uint _9473 = _6661.textures[1];
        uint _9474 = _6661.textures[2];
        uint _9475 = _6661.textures[3];
        uint _9476 = _6661.textures[4];
        float _9477 = _6661.base_color[0];
        float _9478 = _6661.base_color[1];
        float _9479 = _6661.base_color[2];
        uint _9082 = _6661.flags;
        uint _9083 = _6661.type;
        float _9084 = _6661.tangent_rotation_or_strength;
        uint _9085 = _6661.roughness_and_anisotropic;
        float _9086 = _6661.ior;
        uint _9087 = _6661.sheen_and_sheen_tint;
        uint _9088 = _6661.tint_and_metallic;
        uint _9089 = _6661.transmission_and_transmission_roughness;
        uint _9090 = _6661.specular_and_specular_tint;
        uint _9091 = _6661.clearcoat_and_clearcoat_roughness;
        uint _9092 = _6661.normal_map_strength_unorm;
        transform_t _6716;
        _6716.xform = asfloat(uint4x4(_4102.Load4(asuint(asfloat(_6709.Load(inter.obj_index * 32 + 12))) * 128 + 0), _4102.Load4(asuint(asfloat(_6709.Load(inter.obj_index * 32 + 12))) * 128 + 16), _4102.Load4(asuint(asfloat(_6709.Load(inter.obj_index * 32 + 12))) * 128 + 32), _4102.Load4(asuint(asfloat(_6709.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _6716.inv_xform = asfloat(uint4x4(_4102.Load4(asuint(asfloat(_6709.Load(inter.obj_index * 32 + 12))) * 128 + 64), _4102.Load4(asuint(asfloat(_6709.Load(inter.obj_index * 32 + 12))) * 128 + 80), _4102.Load4(asuint(asfloat(_6709.Load(inter.obj_index * 32 + 12))) * 128 + 96), _4102.Load4(asuint(asfloat(_6709.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _6723 = _6653 * 3u;
        vertex_t _6728;
        [unroll]
        for (int _63ident = 0; _63ident < 3; _63ident++)
        {
            _6728.p[_63ident] = asfloat(_4127.Load(_63ident * 4 + _4131.Load(_6723 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _64ident = 0; _64ident < 3; _64ident++)
        {
            _6728.n[_64ident] = asfloat(_4127.Load(_64ident * 4 + _4131.Load(_6723 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _65ident = 0; _65ident < 3; _65ident++)
        {
            _6728.b[_65ident] = asfloat(_4127.Load(_65ident * 4 + _4131.Load(_6723 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _66ident = 0; _66ident < 2; _66ident++)
        {
            [unroll]
            for (int _67ident = 0; _67ident < 2; _67ident++)
            {
                _6728.t[_66ident][_67ident] = asfloat(_4127.Load(_67ident * 4 + _66ident * 8 + _4131.Load(_6723 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _6774;
        [unroll]
        for (int _68ident = 0; _68ident < 3; _68ident++)
        {
            _6774.p[_68ident] = asfloat(_4127.Load(_68ident * 4 + _4131.Load((_6723 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _69ident = 0; _69ident < 3; _69ident++)
        {
            _6774.n[_69ident] = asfloat(_4127.Load(_69ident * 4 + _4131.Load((_6723 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _70ident = 0; _70ident < 3; _70ident++)
        {
            _6774.b[_70ident] = asfloat(_4127.Load(_70ident * 4 + _4131.Load((_6723 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _71ident = 0; _71ident < 2; _71ident++)
        {
            [unroll]
            for (int _72ident = 0; _72ident < 2; _72ident++)
            {
                _6774.t[_71ident][_72ident] = asfloat(_4127.Load(_72ident * 4 + _71ident * 8 + _4131.Load((_6723 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _6820;
        [unroll]
        for (int _73ident = 0; _73ident < 3; _73ident++)
        {
            _6820.p[_73ident] = asfloat(_4127.Load(_73ident * 4 + _4131.Load((_6723 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _74ident = 0; _74ident < 3; _74ident++)
        {
            _6820.n[_74ident] = asfloat(_4127.Load(_74ident * 4 + _4131.Load((_6723 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _75ident = 0; _75ident < 3; _75ident++)
        {
            _6820.b[_75ident] = asfloat(_4127.Load(_75ident * 4 + _4131.Load((_6723 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _76ident = 0; _76ident < 2; _76ident++)
        {
            [unroll]
            for (int _77ident = 0; _77ident < 2; _77ident++)
            {
                _6820.t[_76ident][_77ident] = asfloat(_4127.Load(_77ident * 4 + _76ident * 8 + _4131.Load((_6723 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _6866 = float3(_6728.p[0], _6728.p[1], _6728.p[2]);
        float3 _6874 = float3(_6774.p[0], _6774.p[1], _6774.p[2]);
        float3 _6882 = float3(_6820.p[0], _6820.p[1], _6820.p[2]);
        float _6889 = (1.0f - inter.u) - inter.v;
        float3 _6921 = normalize(((float3(_6728.n[0], _6728.n[1], _6728.n[2]) * _6889) + (float3(_6774.n[0], _6774.n[1], _6774.n[2]) * inter.u)) + (float3(_6820.n[0], _6820.n[1], _6820.n[2]) * inter.v));
        float3 _9021 = _6921;
        float2 _6947 = ((float2(_6728.t[0][0], _6728.t[0][1]) * _6889) + (float2(_6774.t[0][0], _6774.t[0][1]) * inter.u)) + (float2(_6820.t[0][0], _6820.t[0][1]) * inter.v);
        float3 _6963 = cross(_6874 - _6866, _6882 - _6866);
        float _6968 = length(_6963);
        float3 _9022 = _6963 / _6968.xxx;
        float3 _7005 = ((float3(_6728.b[0], _6728.b[1], _6728.b[2]) * _6889) + (float3(_6774.b[0], _6774.b[1], _6774.b[2]) * inter.u)) + (float3(_6820.b[0], _6820.b[1], _6820.b[2]) * inter.v);
        float3 _9020 = _7005;
        float3 _9019 = cross(_7005, _6921);
        if (_6639)
        {
            if ((_4459.Load(_6653 * 4 + 0) & 65535u) == 65535u)
            {
                _8721 = 0.0f.xxx;
                break;
            }
            material_t _7031;
            [unroll]
            for (int _78ident = 0; _78ident < 5; _78ident++)
            {
                _7031.textures[_78ident] = _4455.Load(_78ident * 4 + (_4459.Load(_6653 * 4 + 0) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _79ident = 0; _79ident < 3; _79ident++)
            {
                _7031.base_color[_79ident] = asfloat(_4455.Load(_79ident * 4 + (_4459.Load(_6653 * 4 + 0) & 16383u) * 76 + 20));
            }
            _7031.flags = _4455.Load((_4459.Load(_6653 * 4 + 0) & 16383u) * 76 + 32);
            _7031.type = _4455.Load((_4459.Load(_6653 * 4 + 0) & 16383u) * 76 + 36);
            _7031.tangent_rotation_or_strength = asfloat(_4455.Load((_4459.Load(_6653 * 4 + 0) & 16383u) * 76 + 40));
            _7031.roughness_and_anisotropic = _4455.Load((_4459.Load(_6653 * 4 + 0) & 16383u) * 76 + 44);
            _7031.ior = asfloat(_4455.Load((_4459.Load(_6653 * 4 + 0) & 16383u) * 76 + 48));
            _7031.sheen_and_sheen_tint = _4455.Load((_4459.Load(_6653 * 4 + 0) & 16383u) * 76 + 52);
            _7031.tint_and_metallic = _4455.Load((_4459.Load(_6653 * 4 + 0) & 16383u) * 76 + 56);
            _7031.transmission_and_transmission_roughness = _4455.Load((_4459.Load(_6653 * 4 + 0) & 16383u) * 76 + 60);
            _7031.specular_and_specular_tint = _4455.Load((_4459.Load(_6653 * 4 + 0) & 16383u) * 76 + 64);
            _7031.clearcoat_and_clearcoat_roughness = _4455.Load((_4459.Load(_6653 * 4 + 0) & 16383u) * 76 + 68);
            _7031.normal_map_strength_unorm = _4455.Load((_4459.Load(_6653 * 4 + 0) & 16383u) * 76 + 72);
            _9472 = _7031.textures[0];
            _9473 = _7031.textures[1];
            _9474 = _7031.textures[2];
            _9475 = _7031.textures[3];
            _9476 = _7031.textures[4];
            _9477 = _7031.base_color[0];
            _9478 = _7031.base_color[1];
            _9479 = _7031.base_color[2];
            _9082 = _7031.flags;
            _9083 = _7031.type;
            _9084 = _7031.tangent_rotation_or_strength;
            _9085 = _7031.roughness_and_anisotropic;
            _9086 = _7031.ior;
            _9087 = _7031.sheen_and_sheen_tint;
            _9088 = _7031.tint_and_metallic;
            _9089 = _7031.transmission_and_transmission_roughness;
            _9090 = _7031.specular_and_specular_tint;
            _9091 = _7031.clearcoat_and_clearcoat_roughness;
            _9092 = _7031.normal_map_strength_unorm;
            _9022 = -_9022;
            _9021 = -_9021;
            _9020 = -_9020;
            _9019 = -_9019;
        }
        float3 param_3 = _9022;
        float4x4 param_4 = _6716.inv_xform;
        _9022 = TransformNormal(param_3, param_4);
        float3 param_5 = _9021;
        float4x4 param_6 = _6716.inv_xform;
        _9021 = TransformNormal(param_5, param_6);
        float3 param_7 = _9020;
        float4x4 param_8 = _6716.inv_xform;
        _9020 = TransformNormal(param_7, param_8);
        float3 param_9 = _9019;
        float4x4 param_10 = _6716.inv_xform;
        _9022 = normalize(_9022);
        _9021 = normalize(_9021);
        _9020 = normalize(_9020);
        _9019 = normalize(TransformNormal(param_9, param_10));
        float _7171 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _7181 = mad(0.5f, log2(abs(mad(_6774.t[0][0] - _6728.t[0][0], _6820.t[0][1] - _6728.t[0][1], -((_6820.t[0][0] - _6728.t[0][0]) * (_6774.t[0][1] - _6728.t[0][1])))) / _6968), log2(_7171));
        uint param_11 = uint(hash(ray.xy));
        float _7188 = construct_float(param_11);
        uint param_12 = uint(hash(hash(ray.xy)));
        float _7195 = construct_float(param_12);
        float param_13[4] = ray.ior;
        bool param_14 = _6639;
        float param_15 = 1.0f;
        float _7204 = peek_ior_stack(param_13, param_14, param_15);
        float3 col = 0.0f.xxx;
        int _7209 = ray.depth & 255;
        int _7214 = (ray.depth >> 8) & 255;
        int _7219 = (ray.depth >> 16) & 255;
        int _7230 = (_7209 + _7214) + _7219;
        int _7238 = _3327_g_params.hi + ((_7230 + ((ray.depth >> 24) & 255)) * 7);
        float mix_rand = frac(asfloat(_3311.Load(_7238 * 4 + 0)) + _7188);
        float mix_weight = 1.0f;
        float _7275;
        float _7292;
        float _7318;
        float _7385;
        while (_9083 == 4u)
        {
            float mix_val = _9084;
            if (_9473 != 4294967295u)
            {
                mix_val *= SampleBilinear(_9473, _6947, 0).x;
            }
            if (_6639)
            {
                _7275 = _7204 / _9086;
            }
            else
            {
                _7275 = _9086 / _7204;
            }
            if (_9086 != 0.0f)
            {
                float param_16 = dot(_6570, _9021);
                float param_17 = _7275;
                _7292 = fresnel_dielectric_cos(param_16, param_17);
            }
            else
            {
                _7292 = 1.0f;
            }
            float _7307 = mix_val;
            float _7308 = _7307 * clamp(_7292, 0.0f, 1.0f);
            mix_val = _7308;
            if (mix_rand > _7308)
            {
                if ((_9082 & 2u) != 0u)
                {
                    _7318 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _7318 = 1.0f;
                }
                mix_weight *= _7318;
                material_t _7331;
                [unroll]
                for (int _80ident = 0; _80ident < 5; _80ident++)
                {
                    _7331.textures[_80ident] = _4455.Load(_80ident * 4 + _9475 * 76 + 0);
                }
                [unroll]
                for (int _81ident = 0; _81ident < 3; _81ident++)
                {
                    _7331.base_color[_81ident] = asfloat(_4455.Load(_81ident * 4 + _9475 * 76 + 20));
                }
                _7331.flags = _4455.Load(_9475 * 76 + 32);
                _7331.type = _4455.Load(_9475 * 76 + 36);
                _7331.tangent_rotation_or_strength = asfloat(_4455.Load(_9475 * 76 + 40));
                _7331.roughness_and_anisotropic = _4455.Load(_9475 * 76 + 44);
                _7331.ior = asfloat(_4455.Load(_9475 * 76 + 48));
                _7331.sheen_and_sheen_tint = _4455.Load(_9475 * 76 + 52);
                _7331.tint_and_metallic = _4455.Load(_9475 * 76 + 56);
                _7331.transmission_and_transmission_roughness = _4455.Load(_9475 * 76 + 60);
                _7331.specular_and_specular_tint = _4455.Load(_9475 * 76 + 64);
                _7331.clearcoat_and_clearcoat_roughness = _4455.Load(_9475 * 76 + 68);
                _7331.normal_map_strength_unorm = _4455.Load(_9475 * 76 + 72);
                _9472 = _7331.textures[0];
                _9473 = _7331.textures[1];
                _9474 = _7331.textures[2];
                _9475 = _7331.textures[3];
                _9476 = _7331.textures[4];
                _9477 = _7331.base_color[0];
                _9478 = _7331.base_color[1];
                _9479 = _7331.base_color[2];
                _9082 = _7331.flags;
                _9083 = _7331.type;
                _9084 = _7331.tangent_rotation_or_strength;
                _9085 = _7331.roughness_and_anisotropic;
                _9086 = _7331.ior;
                _9087 = _7331.sheen_and_sheen_tint;
                _9088 = _7331.tint_and_metallic;
                _9089 = _7331.transmission_and_transmission_roughness;
                _9090 = _7331.specular_and_specular_tint;
                _9091 = _7331.clearcoat_and_clearcoat_roughness;
                _9092 = _7331.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_9082 & 2u) != 0u)
                {
                    _7385 = 1.0f / mix_val;
                }
                else
                {
                    _7385 = 1.0f;
                }
                mix_weight *= _7385;
                material_t _7397;
                [unroll]
                for (int _82ident = 0; _82ident < 5; _82ident++)
                {
                    _7397.textures[_82ident] = _4455.Load(_82ident * 4 + _9476 * 76 + 0);
                }
                [unroll]
                for (int _83ident = 0; _83ident < 3; _83ident++)
                {
                    _7397.base_color[_83ident] = asfloat(_4455.Load(_83ident * 4 + _9476 * 76 + 20));
                }
                _7397.flags = _4455.Load(_9476 * 76 + 32);
                _7397.type = _4455.Load(_9476 * 76 + 36);
                _7397.tangent_rotation_or_strength = asfloat(_4455.Load(_9476 * 76 + 40));
                _7397.roughness_and_anisotropic = _4455.Load(_9476 * 76 + 44);
                _7397.ior = asfloat(_4455.Load(_9476 * 76 + 48));
                _7397.sheen_and_sheen_tint = _4455.Load(_9476 * 76 + 52);
                _7397.tint_and_metallic = _4455.Load(_9476 * 76 + 56);
                _7397.transmission_and_transmission_roughness = _4455.Load(_9476 * 76 + 60);
                _7397.specular_and_specular_tint = _4455.Load(_9476 * 76 + 64);
                _7397.clearcoat_and_clearcoat_roughness = _4455.Load(_9476 * 76 + 68);
                _7397.normal_map_strength_unorm = _4455.Load(_9476 * 76 + 72);
                _9472 = _7397.textures[0];
                _9473 = _7397.textures[1];
                _9474 = _7397.textures[2];
                _9475 = _7397.textures[3];
                _9476 = _7397.textures[4];
                _9477 = _7397.base_color[0];
                _9478 = _7397.base_color[1];
                _9479 = _7397.base_color[2];
                _9082 = _7397.flags;
                _9083 = _7397.type;
                _9084 = _7397.tangent_rotation_or_strength;
                _9085 = _7397.roughness_and_anisotropic;
                _9086 = _7397.ior;
                _9087 = _7397.sheen_and_sheen_tint;
                _9088 = _7397.tint_and_metallic;
                _9089 = _7397.transmission_and_transmission_roughness;
                _9090 = _7397.specular_and_specular_tint;
                _9091 = _7397.clearcoat_and_clearcoat_roughness;
                _9092 = _7397.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_9472 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_9472, _6947, 0).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_9472 & 33554432u) != 0u)
            {
                float3 _10044 = normals;
                _10044.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _10044;
            }
            float3 _7479 = _9021;
            _9021 = normalize(((_9019 * normals.x) + (_7479 * normals.z)) + (_9020 * normals.y));
            if ((_9092 & 65535u) != 65535u)
            {
                _9021 = normalize(_7479 + ((_9021 - _7479) * clamp(float(_9092 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_18 = _9022;
            float3 param_19 = -_6570;
            float3 param_20 = _9021;
            _9021 = ensure_valid_reflection(param_18, param_19, param_20);
        }
        float3 _7545 = ((_6866 * _6889) + (_6874 * inter.u)) + (_6882 * inter.v);
        float3 _7552 = float3(-_7545.z, 0.0f, _7545.x);
        float3 tangent = _7552;
        float3 param_21 = _7552;
        float4x4 param_22 = _6716.inv_xform;
        float3 _7558 = TransformNormal(param_21, param_22);
        tangent = _7558;
        float3 _7562 = cross(_7558, _9021);
        if (dot(_7562, _7562) == 0.0f)
        {
            float3 param_23 = _7545;
            float4x4 param_24 = _6716.inv_xform;
            tangent = TransformNormal(param_23, param_24);
        }
        if (_9084 != 0.0f)
        {
            float3 param_25 = tangent;
            float3 param_26 = _9021;
            float param_27 = _9084;
            tangent = rotate_around_axis(param_25, param_26, param_27);
        }
        float3 _7595 = normalize(cross(tangent, _9021));
        _9020 = _7595;
        _9019 = cross(_9021, _7595);
        float3 _9171 = 0.0f.xxx;
        float3 _9170 = 0.0f.xxx;
        float _9175 = 0.0f;
        float _9173 = 0.0f;
        float _9174 = 1.0f;
        bool _7611 = _3327_g_params.li_count != 0;
        bool _7617;
        if (_7611)
        {
            _7617 = _9083 != 3u;
        }
        else
        {
            _7617 = _7611;
        }
        float3 _9172;
        bool _9176;
        bool _9177;
        if (_7617)
        {
            float3 param_28 = _6606;
            float3 param_29 = _9019;
            float3 param_30 = _9020;
            float3 param_31 = _9021;
            int param_32 = _7238;
            float2 param_33 = float2(_7188, _7195);
            light_sample_t _9186 = { _9170, _9171, _9172, _9173, _9174, _9175, _9176, _9177 };
            light_sample_t param_34 = _9186;
            SampleLightSource(param_28, param_29, param_30, param_31, param_32, param_33, param_34);
            _9170 = param_34.col;
            _9171 = param_34.L;
            _9172 = param_34.lp;
            _9173 = param_34.area;
            _9174 = param_34.dist_mul;
            _9175 = param_34.pdf;
            _9176 = param_34.cast_shadow;
            _9177 = param_34.from_env;
        }
        float _7645 = dot(_9021, _9171);
        float3 base_color = float3(_9477, _9478, _9479);
        [branch]
        if (_9473 != 4294967295u)
        {
            base_color *= SampleBilinear(_9473, _6947, int(get_texture_lod(texSize(_9473), _7181)), true, true).xyz;
        }
        out_base_color = base_color;
        out_normals = _9021;
        float3 tint_color = 0.0f.xxx;
        float _7681 = lum(base_color);
        [flatten]
        if (_7681 > 0.0f)
        {
            tint_color = base_color / _7681.xxx;
        }
        float roughness = clamp(float(_9085 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_9474 != 4294967295u)
        {
            roughness *= SampleBilinear(_9474, _6947, int(get_texture_lod(texSize(_9474), _7181)), false, true).x;
        }
        float _7726 = frac(asfloat(_3311.Load((_7238 + 1) * 4 + 0)) + _7188);
        float _7735 = frac(asfloat(_3311.Load((_7238 + 2) * 4 + 0)) + _7195);
        float _9599 = 0.0f;
        float _9598 = 0.0f;
        float _9597 = 0.0f;
        float _9235[4];
        [unroll]
        for (int i = 0; i < 4; )
        {
            _9235[i] = ray.ior[i];
            i++;
            continue;
        }
        float _9236 = _7171;
        float _9237 = ray.cone_spread;
        int _9238 = ray.xy;
        float _9233 = 0.0f;
        float _9704 = 0.0f;
        float _9703 = 0.0f;
        float _9702 = 0.0f;
        int _9340 = ray.depth;
        int _9344 = ray.xy;
        int _9239;
        float _9342;
        float _9527;
        float _9528;
        float _9529;
        float _9562;
        float _9563;
        float _9564;
        float _9632;
        float _9633;
        float _9634;
        float _9667;
        float _9668;
        float _9669;
        [branch]
        if (_9083 == 0u)
        {
            [branch]
            if ((_9175 > 0.0f) && (_7645 > 0.0f))
            {
                light_sample_t _9203 = { _9170, _9171, _9172, _9173, _9174, _9175, _9176, _9177 };
                surface_t _9030 = { _6606, _9019, _9020, _9021, _9022, _6947 };
                float _9708[3] = { _9702, _9703, _9704 };
                float _9673[3] = { _9667, _9668, _9669 };
                float _9638[3] = { _9632, _9633, _9634 };
                shadow_ray_t _9354 = { _9638, _9340, _9673, _9342, _9708, _9344 };
                shadow_ray_t param_35 = _9354;
                float3 _7795 = Evaluate_DiffuseNode(_9203, ray, _9030, base_color, roughness, mix_weight, param_35);
                _9632 = param_35.o[0];
                _9633 = param_35.o[1];
                _9634 = param_35.o[2];
                _9340 = param_35.depth;
                _9667 = param_35.d[0];
                _9668 = param_35.d[1];
                _9669 = param_35.d[2];
                _9342 = param_35.dist;
                _9702 = param_35.c[0];
                _9703 = param_35.c[1];
                _9704 = param_35.c[2];
                _9344 = param_35.xy;
                col += _7795;
            }
            bool _7802 = _7209 < _3327_g_params.max_diff_depth;
            bool _7809;
            if (_7802)
            {
                _7809 = _7230 < _3327_g_params.max_total_depth;
            }
            else
            {
                _7809 = _7802;
            }
            [branch]
            if (_7809)
            {
                surface_t _9037 = { _6606, _9019, _9020, _9021, _9022, _6947 };
                float _9603[3] = { _9597, _9598, _9599 };
                float _9568[3] = { _9562, _9563, _9564 };
                float _9533[3] = { _9527, _9528, _9529 };
                ray_data_t _9253 = { _9533, _9568, _9233, _9603, _9235, _9236, _9237, _9238, _9239 };
                ray_data_t param_36 = _9253;
                Sample_DiffuseNode(ray, _9037, base_color, roughness, _7726, _7735, mix_weight, param_36);
                _9527 = param_36.o[0];
                _9528 = param_36.o[1];
                _9529 = param_36.o[2];
                _9562 = param_36.d[0];
                _9563 = param_36.d[1];
                _9564 = param_36.d[2];
                _9233 = param_36.pdf;
                _9597 = param_36.c[0];
                _9598 = param_36.c[1];
                _9599 = param_36.c[2];
                _9235 = param_36.ior;
                _9236 = param_36.cone_width;
                _9237 = param_36.cone_spread;
                _9238 = param_36.xy;
                _9239 = param_36.depth;
            }
        }
        else
        {
            [branch]
            if (_9083 == 1u)
            {
                float param_37 = 1.0f;
                float param_38 = 1.5f;
                float _7833 = fresnel_dielectric_cos(param_37, param_38);
                float _7837 = roughness * roughness;
                bool _7840 = _9175 > 0.0f;
                bool _7847;
                if (_7840)
                {
                    _7847 = (_7837 * _7837) >= 1.0000000116860974230803549289703e-07f;
                }
                else
                {
                    _7847 = _7840;
                }
                [branch]
                if (_7847 && (_7645 > 0.0f))
                {
                    light_sample_t _9212 = { _9170, _9171, _9172, _9173, _9174, _9175, _9176, _9177 };
                    surface_t _9044 = { _6606, _9019, _9020, _9021, _9022, _6947 };
                    float _9715[3] = { _9702, _9703, _9704 };
                    float _9680[3] = { _9667, _9668, _9669 };
                    float _9645[3] = { _9632, _9633, _9634 };
                    shadow_ray_t _9367 = { _9645, _9340, _9680, _9342, _9715, _9344 };
                    shadow_ray_t param_39 = _9367;
                    float3 _7862 = Evaluate_GlossyNode(_9212, ray, _9044, base_color, roughness, 1.5f, _7833, mix_weight, param_39);
                    _9632 = param_39.o[0];
                    _9633 = param_39.o[1];
                    _9634 = param_39.o[2];
                    _9340 = param_39.depth;
                    _9667 = param_39.d[0];
                    _9668 = param_39.d[1];
                    _9669 = param_39.d[2];
                    _9342 = param_39.dist;
                    _9702 = param_39.c[0];
                    _9703 = param_39.c[1];
                    _9704 = param_39.c[2];
                    _9344 = param_39.xy;
                    col += _7862;
                }
                bool _7869 = _7214 < _3327_g_params.max_spec_depth;
                bool _7876;
                if (_7869)
                {
                    _7876 = _7230 < _3327_g_params.max_total_depth;
                }
                else
                {
                    _7876 = _7869;
                }
                [branch]
                if (_7876)
                {
                    surface_t _9051 = { _6606, _9019, _9020, _9021, _9022, _6947 };
                    float _9610[3] = { _9597, _9598, _9599 };
                    float _9575[3] = { _9562, _9563, _9564 };
                    float _9540[3] = { _9527, _9528, _9529 };
                    ray_data_t _9272 = { _9540, _9575, _9233, _9610, _9235, _9236, _9237, _9238, _9239 };
                    ray_data_t param_40 = _9272;
                    Sample_GlossyNode(ray, _9051, base_color, roughness, 1.5f, _7833, _7726, _7735, mix_weight, param_40);
                    _9527 = param_40.o[0];
                    _9528 = param_40.o[1];
                    _9529 = param_40.o[2];
                    _9562 = param_40.d[0];
                    _9563 = param_40.d[1];
                    _9564 = param_40.d[2];
                    _9233 = param_40.pdf;
                    _9597 = param_40.c[0];
                    _9598 = param_40.c[1];
                    _9599 = param_40.c[2];
                    _9235 = param_40.ior;
                    _9236 = param_40.cone_width;
                    _9237 = param_40.cone_spread;
                    _9238 = param_40.xy;
                    _9239 = param_40.depth;
                }
            }
            else
            {
                [branch]
                if (_9083 == 2u)
                {
                    float _7900 = roughness * roughness;
                    bool _7903 = _9175 > 0.0f;
                    bool _7910;
                    if (_7903)
                    {
                        _7910 = (_7900 * _7900) >= 1.0000000116860974230803549289703e-07f;
                    }
                    else
                    {
                        _7910 = _7903;
                    }
                    [branch]
                    if (_7910 && (_7645 < 0.0f))
                    {
                        float _7918;
                        if (_6639)
                        {
                            _7918 = _9086 / _7204;
                        }
                        else
                        {
                            _7918 = _7204 / _9086;
                        }
                        light_sample_t _9221 = { _9170, _9171, _9172, _9173, _9174, _9175, _9176, _9177 };
                        surface_t _9058 = { _6606, _9019, _9020, _9021, _9022, _6947 };
                        float _9722[3] = { _9702, _9703, _9704 };
                        float _9687[3] = { _9667, _9668, _9669 };
                        float _9652[3] = { _9632, _9633, _9634 };
                        shadow_ray_t _9380 = { _9652, _9340, _9687, _9342, _9722, _9344 };
                        shadow_ray_t param_41 = _9380;
                        float3 _7940 = Evaluate_RefractiveNode(_9221, ray, _9058, base_color, _7900, _7918, mix_weight, param_41);
                        _9632 = param_41.o[0];
                        _9633 = param_41.o[1];
                        _9634 = param_41.o[2];
                        _9340 = param_41.depth;
                        _9667 = param_41.d[0];
                        _9668 = param_41.d[1];
                        _9669 = param_41.d[2];
                        _9342 = param_41.dist;
                        _9702 = param_41.c[0];
                        _9703 = param_41.c[1];
                        _9704 = param_41.c[2];
                        _9344 = param_41.xy;
                        col += _7940;
                    }
                    bool _7947 = _7219 < _3327_g_params.max_refr_depth;
                    bool _7954;
                    if (_7947)
                    {
                        _7954 = _7230 < _3327_g_params.max_total_depth;
                    }
                    else
                    {
                        _7954 = _7947;
                    }
                    [branch]
                    if (_7954)
                    {
                        surface_t _9065 = { _6606, _9019, _9020, _9021, _9022, _6947 };
                        float _9617[3] = { _9597, _9598, _9599 };
                        float _9582[3] = { _9562, _9563, _9564 };
                        float _9547[3] = { _9527, _9528, _9529 };
                        ray_data_t _9291 = { _9547, _9582, _9233, _9617, _9235, _9236, _9237, _9238, _9239 };
                        ray_data_t param_42 = _9291;
                        Sample_RefractiveNode(ray, _9065, base_color, roughness, _6639, _9086, _7204, _7726, _7735, mix_weight, param_42);
                        _9527 = param_42.o[0];
                        _9528 = param_42.o[1];
                        _9529 = param_42.o[2];
                        _9562 = param_42.d[0];
                        _9563 = param_42.d[1];
                        _9564 = param_42.d[2];
                        _9233 = param_42.pdf;
                        _9597 = param_42.c[0];
                        _9598 = param_42.c[1];
                        _9599 = param_42.c[2];
                        _9235 = param_42.ior;
                        _9236 = param_42.cone_width;
                        _9237 = param_42.cone_spread;
                        _9238 = param_42.xy;
                        _9239 = param_42.depth;
                    }
                }
                else
                {
                    [branch]
                    if (_9083 == 3u)
                    {
                        col += (base_color * (mix_weight * _9084));
                    }
                    else
                    {
                        [branch]
                        if (_9083 == 6u)
                        {
                            float metallic = clamp(float((_9088 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_9475 != 4294967295u)
                            {
                                metallic *= SampleBilinear(_9475, _6947, int(get_texture_lod(texSize(_9475), _7181))).x;
                            }
                            float specular = clamp(float(_9090 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_9476 != 4294967295u)
                            {
                                specular *= SampleBilinear(_9476, _6947, int(get_texture_lod(texSize(_9476), _7181))).x;
                            }
                            float _8073 = clamp(float(_9091 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8081 = clamp(float((_9091 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8089 = 2.0f * clamp(float(_9087 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float3 _8107 = lerp(1.0f.xxx, tint_color, clamp(float((_9087 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _8089;
                            float3 _8127 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_9090 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                            float _8136 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                            float param_43 = 1.0f;
                            float param_44 = _8136;
                            float _8142 = fresnel_dielectric_cos(param_43, param_44);
                            float _8150 = clamp(float((_9085 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8161 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _8073))) - 1.0f;
                            float param_45 = 1.0f;
                            float param_46 = _8161;
                            float _8167 = fresnel_dielectric_cos(param_45, param_46);
                            float _8182 = mad(roughness - 1.0f, 1.0f - clamp(float((_9089 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                            float _8188;
                            if (_6639)
                            {
                                _8188 = _9086 / _7204;
                            }
                            else
                            {
                                _8188 = _7204 / _9086;
                            }
                            float param_47 = dot(_6570, _9021);
                            float param_48 = 1.0f / _8188;
                            float _8211 = fresnel_dielectric_cos(param_47, param_48);
                            float param_49 = dot(_6570, _9021);
                            float param_50 = _8136;
                            lobe_weights_t _8250 = get_lobe_weights(lerp(_7681, 1.0f, _8089), lum(lerp(_8127, 1.0f.xxx, ((fresnel_dielectric_cos(param_49, param_50) - _8142) / (1.0f - _8142)).xxx)), specular, metallic, clamp(float(_9089 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _8073);
                            [branch]
                            if (_9175 > 0.0f)
                            {
                                light_sample_t _9230 = { _9170, _9171, _9172, _9173, _9174, _9175, _9176, _9177 };
                                surface_t _9072 = { _6606, _9019, _9020, _9021, _9022, _6947 };
                                diff_params_t _9422 = { base_color, _8107, roughness };
                                spec_params_t _9437 = { _8127, roughness, _8136, _8142, _8150 };
                                clearcoat_params_t _9450 = { _8081, _8161, _8167 };
                                transmission_params_t _9465 = { _8182, _9086, _8188, _8211, _6639 };
                                float _9729[3] = { _9702, _9703, _9704 };
                                float _9694[3] = { _9667, _9668, _9669 };
                                float _9659[3] = { _9632, _9633, _9634 };
                                shadow_ray_t _9393 = { _9659, _9340, _9694, _9342, _9729, _9344 };
                                shadow_ray_t param_51 = _9393;
                                float3 _8269 = Evaluate_PrincipledNode(_9230, ray, _9072, _8250, _9422, _9437, _9450, _9465, metallic, _7645, mix_weight, param_51);
                                _9632 = param_51.o[0];
                                _9633 = param_51.o[1];
                                _9634 = param_51.o[2];
                                _9340 = param_51.depth;
                                _9667 = param_51.d[0];
                                _9668 = param_51.d[1];
                                _9669 = param_51.d[2];
                                _9342 = param_51.dist;
                                _9702 = param_51.c[0];
                                _9703 = param_51.c[1];
                                _9704 = param_51.c[2];
                                _9344 = param_51.xy;
                                col += _8269;
                            }
                            surface_t _9079 = { _6606, _9019, _9020, _9021, _9022, _6947 };
                            diff_params_t _9426 = { base_color, _8107, roughness };
                            spec_params_t _9443 = { _8127, roughness, _8136, _8142, _8150 };
                            clearcoat_params_t _9454 = { _8081, _8161, _8167 };
                            transmission_params_t _9471 = { _8182, _9086, _8188, _8211, _6639 };
                            float param_52 = mix_rand;
                            float _9624[3] = { _9597, _9598, _9599 };
                            float _9589[3] = { _9562, _9563, _9564 };
                            float _9554[3] = { _9527, _9528, _9529 };
                            ray_data_t _9310 = { _9554, _9589, _9233, _9624, _9235, _9236, _9237, _9238, _9239 };
                            ray_data_t param_53 = _9310;
                            Sample_PrincipledNode(ray, _9079, _8250, _9426, _9443, _9454, _9471, metallic, _7726, _7735, param_52, mix_weight, param_53);
                            _9527 = param_53.o[0];
                            _9528 = param_53.o[1];
                            _9529 = param_53.o[2];
                            _9562 = param_53.d[0];
                            _9563 = param_53.d[1];
                            _9564 = param_53.d[2];
                            _9233 = param_53.pdf;
                            _9597 = param_53.c[0];
                            _9598 = param_53.c[1];
                            _9599 = param_53.c[2];
                            _9235 = param_53.ior;
                            _9236 = param_53.cone_width;
                            _9237 = param_53.cone_spread;
                            _9238 = param_53.xy;
                            _9239 = param_53.depth;
                        }
                    }
                }
            }
        }
        float _8303 = max(_9597, max(_9598, _9599));
        float _8315;
        if (_7230 > _3327_g_params.min_total_depth)
        {
            _8315 = max(0.0500000007450580596923828125f, 1.0f - _8303);
        }
        else
        {
            _8315 = 0.0f;
        }
        bool _8329 = (frac(asfloat(_3311.Load((_7238 + 6) * 4 + 0)) + _7188) >= _8315) && (_8303 > 0.0f);
        bool _8335;
        if (_8329)
        {
            _8335 = _9233 > 0.0f;
        }
        else
        {
            _8335 = _8329;
        }
        [branch]
        if (_8335)
        {
            float _8339 = _9233;
            float _8340 = min(_8339, 1000000.0f);
            _9233 = _8340;
            float _8343 = 1.0f - _8315;
            float _8345 = _9597;
            float _8346 = _8345 / _8343;
            _9597 = _8346;
            float _8351 = _9598;
            float _8352 = _8351 / _8343;
            _9598 = _8352;
            float _8357 = _9599;
            float _8358 = _8357 / _8343;
            _9599 = _8358;
            uint _8366;
            _8364.InterlockedAdd(0, 1u, _8366);
            _8375.Store(_8366 * 72 + 0, asuint(_9527));
            _8375.Store(_8366 * 72 + 4, asuint(_9528));
            _8375.Store(_8366 * 72 + 8, asuint(_9529));
            _8375.Store(_8366 * 72 + 12, asuint(_9562));
            _8375.Store(_8366 * 72 + 16, asuint(_9563));
            _8375.Store(_8366 * 72 + 20, asuint(_9564));
            _8375.Store(_8366 * 72 + 24, asuint(_8340));
            _8375.Store(_8366 * 72 + 28, asuint(_8346));
            _8375.Store(_8366 * 72 + 32, asuint(_8352));
            _8375.Store(_8366 * 72 + 36, asuint(_8358));
            _8375.Store(_8366 * 72 + 40, asuint(_9235[0]));
            _8375.Store(_8366 * 72 + 44, asuint(_9235[1]));
            _8375.Store(_8366 * 72 + 48, asuint(_9235[2]));
            _8375.Store(_8366 * 72 + 52, asuint(_9235[3]));
            _8375.Store(_8366 * 72 + 56, asuint(_9236));
            _8375.Store(_8366 * 72 + 60, asuint(_9237));
            _8375.Store(_8366 * 72 + 64, uint(_9238));
            _8375.Store(_8366 * 72 + 68, uint(_9239));
        }
        [branch]
        if (max(_9702, max(_9703, _9704)) > 0.0f)
        {
            float3 _8452 = _9172 - float3(_9632, _9633, _9634);
            float _8455 = length(_8452);
            float3 _8459 = _8452 / _8455.xxx;
            float sh_dist = _8455 * _9174;
            if (_9177)
            {
                sh_dist = -sh_dist;
            }
            float _8471 = _8459.x;
            _9667 = _8471;
            float _8474 = _8459.y;
            _9668 = _8474;
            float _8477 = _8459.z;
            _9669 = _8477;
            _9342 = sh_dist;
            uint _8483;
            _8364.InterlockedAdd(8, 1u, _8483);
            _8491.Store(_8483 * 48 + 0, asuint(_9632));
            _8491.Store(_8483 * 48 + 4, asuint(_9633));
            _8491.Store(_8483 * 48 + 8, asuint(_9634));
            _8491.Store(_8483 * 48 + 12, uint(_9340));
            _8491.Store(_8483 * 48 + 16, asuint(_8471));
            _8491.Store(_8483 * 48 + 20, asuint(_8474));
            _8491.Store(_8483 * 48 + 24, asuint(_8477));
            _8491.Store(_8483 * 48 + 28, asuint(sh_dist));
            _8491.Store(_8483 * 48 + 32, asuint(_9702));
            _8491.Store(_8483 * 48 + 36, asuint(_9703));
            _8491.Store(_8483 * 48 + 40, asuint(_9704));
            _8491.Store(_8483 * 48 + 44, uint(_9344));
        }
        _8721 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _8721;
}

void comp_main()
{
    do
    {
        bool _8555 = gl_GlobalInvocationID.x >= _3327_g_params.img_size.x;
        bool _8564;
        if (!_8555)
        {
            _8564 = gl_GlobalInvocationID.y >= _3327_g_params.img_size.y;
        }
        else
        {
            _8564 = _8555;
        }
        if (_8564)
        {
            break;
        }
        int _8571 = int(gl_GlobalInvocationID.x);
        int _8575 = int(gl_GlobalInvocationID.y);
        int _8583 = (_8575 * int(_3327_g_params.img_size.x)) + _8571;
        hit_data_t _8593;
        _8593.mask = int(_8589.Load(_8583 * 24 + 0));
        _8593.obj_index = int(_8589.Load(_8583 * 24 + 4));
        _8593.prim_index = int(_8589.Load(_8583 * 24 + 8));
        _8593.t = asfloat(_8589.Load(_8583 * 24 + 12));
        _8593.u = asfloat(_8589.Load(_8583 * 24 + 16));
        _8593.v = asfloat(_8589.Load(_8583 * 24 + 20));
        ray_data_t _8613;
        [unroll]
        for (int _84ident = 0; _84ident < 3; _84ident++)
        {
            _8613.o[_84ident] = asfloat(_8610.Load(_84ident * 4 + _8583 * 72 + 0));
        }
        [unroll]
        for (int _85ident = 0; _85ident < 3; _85ident++)
        {
            _8613.d[_85ident] = asfloat(_8610.Load(_85ident * 4 + _8583 * 72 + 12));
        }
        _8613.pdf = asfloat(_8610.Load(_8583 * 72 + 24));
        [unroll]
        for (int _86ident = 0; _86ident < 3; _86ident++)
        {
            _8613.c[_86ident] = asfloat(_8610.Load(_86ident * 4 + _8583 * 72 + 28));
        }
        [unroll]
        for (int _87ident = 0; _87ident < 4; _87ident++)
        {
            _8613.ior[_87ident] = asfloat(_8610.Load(_87ident * 4 + _8583 * 72 + 40));
        }
        _8613.cone_width = asfloat(_8610.Load(_8583 * 72 + 56));
        _8613.cone_spread = asfloat(_8610.Load(_8583 * 72 + 60));
        _8613.xy = int(_8610.Load(_8583 * 72 + 64));
        _8613.depth = int(_8610.Load(_8583 * 72 + 68));
        hit_data_t _8815 = { _8593.mask, _8593.obj_index, _8593.prim_index, _8593.t, _8593.u, _8593.v };
        hit_data_t param = _8815;
        float _8864[4] = { _8613.ior[0], _8613.ior[1], _8613.ior[2], _8613.ior[3] };
        float _8855[3] = { _8613.c[0], _8613.c[1], _8613.c[2] };
        float _8848[3] = { _8613.d[0], _8613.d[1], _8613.d[2] };
        float _8841[3] = { _8613.o[0], _8613.o[1], _8613.o[2] };
        ray_data_t _8834 = { _8841, _8848, _8613.pdf, _8855, _8864, _8613.cone_width, _8613.cone_spread, _8613.xy, _8613.depth };
        ray_data_t param_1 = _8834;
        float3 param_2 = 0.0f.xxx;
        float3 param_3 = 0.0f.xxx;
        float3 _8669 = ShadeSurface(param, param_1, param_2, param_3);
        int2 _8678 = int2(_8571, _8575);
        g_out_img[_8678] = float4(_8669, 1.0f);
        g_out_depth_normals_img[_8678] = float4(param_3, _8593.t);
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
