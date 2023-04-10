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

ByteAddressBuffer _3311 : register(t17, space0);
ByteAddressBuffer _3347 : register(t8, space0);
ByteAddressBuffer _3351 : register(t9, space0);
ByteAddressBuffer _4100 : register(t13, space0);
ByteAddressBuffer _4125 : register(t15, space0);
ByteAddressBuffer _4129 : register(t16, space0);
ByteAddressBuffer _4453 : register(t12, space0);
ByteAddressBuffer _4457 : register(t11, space0);
ByteAddressBuffer _6744 : register(t14, space0);
RWByteAddressBuffer _8482 : register(u3, space0);
RWByteAddressBuffer _8493 : register(u1, space0);
RWByteAddressBuffer _8609 : register(u2, space0);
ByteAddressBuffer _8688 : register(t7, space0);
ByteAddressBuffer _8705 : register(t6, space0);
ByteAddressBuffer _8827 : register(t10, space0);
cbuffer UniformParams
{
    Params _3327_g_params : packoffset(c0);
};

Texture2D<float4> g_textures[] : register(t0, space1);
SamplerState _g_textures_sampler[] : register(s0, space1);
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
    float3 _4665 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 _4672;
    if ((ray.depth & 16777215) != 0)
    {
        _4672 = _3327_g_params.env_col.xyz;
    }
    else
    {
        _4672 = _3327_g_params.back_col.xyz;
    }
    float3 env_col = _4672;
    uint _4688;
    if ((ray.depth & 16777215) != 0)
    {
        _4688 = asuint(_3327_g_params.env_col.w);
    }
    else
    {
        _4688 = asuint(_3327_g_params.back_col.w);
    }
    float _4704;
    if ((ray.depth & 16777215) != 0)
    {
        _4704 = _3327_g_params.env_rotation;
    }
    else
    {
        _4704 = _3327_g_params.back_rotation;
    }
    if (_4688 != 4294967295u)
    {
        env_col *= SampleLatlong_RGBE(_4688, _4665, _4704);
    }
    if (_3327_g_params.env_qtree_levels > 0)
    {
        float param = ray.pdf;
        float param_1 = Evaluate_EnvQTree(_4704, g_env_qtree, _g_env_qtree_sampler, _3327_g_params.env_qtree_levels, _4665);
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
    float3 _4783 = float3(ray.d[0], ray.d[1], ray.d[2]);
    light_t _4797;
    _4797.type_and_param0 = _3347.Load4(((-1) - inter.obj_index) * 64 + 0);
    _4797.param1 = asfloat(_3347.Load4(((-1) - inter.obj_index) * 64 + 16));
    _4797.param2 = asfloat(_3347.Load4(((-1) - inter.obj_index) * 64 + 32));
    _4797.param3 = asfloat(_3347.Load4(((-1) - inter.obj_index) * 64 + 48));
    float3 lcol = asfloat(_4797.type_and_param0.yzw);
    [branch]
    if ((_4797.type_and_param0.x & 128u) != 0u)
    {
        float3 env_col = _3327_g_params.env_col.xyz;
        uint _4824 = asuint(_3327_g_params.env_col.w);
        if (_4824 != 4294967295u)
        {
            env_col *= SampleLatlong_RGBE(_4824, _4783, _3327_g_params.env_rotation);
        }
        lcol *= env_col;
    }
    uint _4842 = _4797.type_and_param0.x & 31u;
    if (_4842 == 0u)
    {
        float param = ray.pdf;
        float param_1 = (inter.t * inter.t) / ((0.5f * _4797.param1.w) * dot(_4783, normalize(_4797.param1.xyz - (float3(ray.o[0], ray.o[1], ray.o[2]) + (_4783 * inter.t)))));
        lcol *= power_heuristic(param, param_1);
        bool _4909 = _4797.param3.x > 0.0f;
        bool _4915;
        if (_4909)
        {
            _4915 = _4797.param3.y > 0.0f;
        }
        else
        {
            _4915 = _4909;
        }
        [branch]
        if (_4915)
        {
            [flatten]
            if (_4797.param3.y > 0.0f)
            {
                lcol *= clamp((_4797.param3.x - acos(clamp(-dot(_4783, _4797.param2.xyz), 0.0f, 1.0f))) / _4797.param3.y, 0.0f, 1.0f);
            }
        }
    }
    else
    {
        if (_4842 == 4u)
        {
            float param_2 = ray.pdf;
            float param_3 = (inter.t * inter.t) / (_4797.param1.w * dot(_4783, normalize(cross(_4797.param2.xyz, _4797.param3.xyz))));
            lcol *= power_heuristic(param_2, param_3);
        }
        else
        {
            if (_4842 == 5u)
            {
                float param_4 = ray.pdf;
                float param_5 = (inter.t * inter.t) / (_4797.param1.w * dot(_4783, normalize(cross(_4797.param2.xyz, _4797.param3.xyz))));
                lcol *= power_heuristic(param_4, param_5);
            }
            else
            {
                if (_4842 == 3u)
                {
                    float param_6 = ray.pdf;
                    float param_7 = (inter.t * inter.t) / (_4797.param1.w * (1.0f - abs(dot(_4783, _4797.param3.xyz))));
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
    float _8839;
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
            _8839 = stack[3];
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
            _8839 = stack[2];
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
            _8839 = stack[1];
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
            _8839 = stack[0];
            break;
        }
        _8839 = default_value;
        break;
    } while(false);
    return _8839;
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
        float4 _9894 = res;
        _9894.x = _1033.x;
        float4 _9896 = _9894;
        _9896.y = _1033.y;
        float4 _9898 = _9896;
        _9898.z = _1033.z;
        res = _9898;
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
    float3 _8844;
    do
    {
        float _1299 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1299)
        {
            _8844 = N;
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
            float _10195 = (-0.5f) / _1339;
            float param_1 = mad(_10195, _1363, 1.0f);
            float _1395 = safe_sqrtf(param_1);
            float param_2 = _1364;
            float _1398 = safe_sqrtf(param_2);
            float2 _1399 = float2(_1395, _1398);
            float param_3 = mad(_10195, _1370, 1.0f);
            float _1404 = safe_sqrtf(param_3);
            float param_4 = _1371;
            float _1407 = safe_sqrtf(param_4);
            float2 _1408 = float2(_1404, _1407);
            float _10197 = -_1327;
            float _1424 = mad(2.0f * mad(_1395, _1323, _1398 * _1327), _1398, _10197);
            float _1440 = mad(2.0f * mad(_1404, _1323, _1407 * _1327), _1407, _10197);
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
                _8844 = Ng;
                break;
            }
            float _1477 = valid1 ? _1364 : _1371;
            float param_5 = 1.0f - _1477;
            float param_6 = _1477;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _8844 = (_1319 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _8844;
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
    float3 _8869;
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
            _8869 = N;
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
        _8869 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _8869;
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
            float2 _9881 = origin;
            _9881.x = origin.x + _step;
            origin = _9881;
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
            float2 _9884 = origin;
            _9884.y = origin.y + _step;
            origin = _9884;
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
    float _3332 = float(_3327_g_params.li_count);
    uint _3339 = min(uint(_3320 * _3332), uint(_3327_g_params.li_count - 1));
    light_t _3358;
    _3358.type_and_param0 = _3347.Load4(_3351.Load(_3339 * 4 + 0) * 64 + 0);
    _3358.param1 = asfloat(_3347.Load4(_3351.Load(_3339 * 4 + 0) * 64 + 16));
    _3358.param2 = asfloat(_3347.Load4(_3351.Load(_3339 * 4 + 0) * 64 + 32));
    _3358.param3 = asfloat(_3347.Load4(_3351.Load(_3339 * 4 + 0) * 64 + 48));
    ls.col = asfloat(_3358.type_and_param0.yzw);
    ls.col *= _3332;
    ls.cast_shadow = (_3358.type_and_param0.x & 32u) != 0u;
    ls.from_env = false;
    uint _3394 = _3358.type_and_param0.x & 31u;
    [branch]
    if (_3394 == 0u)
    {
        float _3407 = frac(asfloat(_3311.Load((hi + 4) * 4 + 0)) + sample_off.x);
        float3 _3423 = P - _3358.param1.xyz;
        float3 _3430 = _3423 / length(_3423).xxx;
        float _3437 = sqrt(clamp(mad(-_3407, _3407, 1.0f), 0.0f, 1.0f));
        float _3440 = 6.283185482025146484375f * frac(asfloat(_3311.Load((hi + 5) * 4 + 0)) + sample_off.y);
        float3 sampled_dir = float3(_3437 * cos(_3440), _3437 * sin(_3440), _3407);
        float3 param;
        float3 param_1;
        create_tbn(_3430, param, param_1);
        float3 _9961 = sampled_dir;
        float3 _3473 = ((param * _9961.x) + (param_1 * _9961.y)) + (_3430 * _9961.z);
        sampled_dir = _3473;
        float3 _3482 = _3358.param1.xyz + (_3473 * _3358.param2.w);
        float3 _3489 = normalize(_3482 - _3358.param1.xyz);
        float3 param_2 = _3482;
        float3 param_3 = _3489;
        ls.lp = offset_ray(param_2, param_3);
        ls.L = _3482 - P;
        float3 _3502 = ls.L;
        float _3503 = length(_3502);
        ls.L /= _3503.xxx;
        ls.area = _3358.param1.w;
        float _3518 = abs(dot(ls.L, _3489));
        [flatten]
        if (_3518 > 0.0f)
        {
            ls.pdf = (_3503 * _3503) / ((0.5f * ls.area) * _3518);
        }
        [branch]
        if (_3358.param3.x > 0.0f)
        {
            float _3545 = -dot(ls.L, _3358.param2.xyz);
            if (_3545 > 0.0f)
            {
                ls.col *= clamp((_3358.param3.x - acos(clamp(_3545, 0.0f, 1.0f))) / _3358.param3.y, 0.0f, 1.0f);
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
                float3 _3682 = (_3358.param1.xyz + (_3358.param2.xyz * (frac(asfloat(_3311.Load((hi + 4) * 4 + 0)) + sample_off.x) - 0.5f))) + (_3358.param3.xyz * (frac(asfloat(_3311.Load((hi + 5) * 4 + 0)) + sample_off.y) - 0.5f));
                float3 _3687 = normalize(cross(_3358.param2.xyz, _3358.param3.xyz));
                float3 param_8 = _3682;
                float3 param_9 = _3687;
                ls.lp = offset_ray(param_8, param_9);
                ls.L = _3682 - P;
                float3 _3700 = ls.L;
                float _3701 = length(_3700);
                ls.L /= _3701.xxx;
                ls.area = _3358.param1.w;
                float _3716 = dot(-ls.L, _3687);
                if (_3716 > 0.0f)
                {
                    ls.pdf = (_3701 * _3701) / (ls.area * _3716);
                }
                if ((_3358.type_and_param0.x & 64u) == 0u)
                {
                    ls.area = 0.0f;
                }
                [branch]
                if ((_3358.type_and_param0.x & 128u) != 0u)
                {
                    float3 env_col = _3327_g_params.env_col.xyz;
                    uint _3753 = asuint(_3327_g_params.env_col.w);
                    if (_3753 != 4294967295u)
                    {
                        env_col *= SampleLatlong_RGBE(_3753, ls.L, _3327_g_params.env_rotation);
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
                    float2 _3816 = (float2(frac(asfloat(_3311.Load((hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_3311.Load((hi + 5) * 4 + 0)) + sample_off.y)) * 2.0f) - 1.0f.xx;
                    float2 offset = _3816;
                    bool _3819 = _3816.x != 0.0f;
                    bool _3825;
                    if (_3819)
                    {
                        _3825 = offset.y != 0.0f;
                    }
                    else
                    {
                        _3825 = _3819;
                    }
                    if (_3825)
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
                        float _3858 = 0.5f * r;
                        offset = float2(_3858 * cos(theta), _3858 * sin(theta));
                    }
                    float3 _3880 = (_3358.param1.xyz + (_3358.param2.xyz * offset.x)) + (_3358.param3.xyz * offset.y);
                    float3 _3885 = normalize(cross(_3358.param2.xyz, _3358.param3.xyz));
                    float3 param_10 = _3880;
                    float3 param_11 = _3885;
                    ls.lp = offset_ray(param_10, param_11);
                    ls.L = _3880 - P;
                    float3 _3898 = ls.L;
                    float _3899 = length(_3898);
                    ls.L /= _3899.xxx;
                    ls.area = _3358.param1.w;
                    float _3914 = dot(-ls.L, _3885);
                    [flatten]
                    if (_3914 > 0.0f)
                    {
                        ls.pdf = (_3899 * _3899) / (ls.area * _3914);
                    }
                    if ((_3358.type_and_param0.x & 64u) == 0u)
                    {
                        ls.area = 0.0f;
                    }
                    [branch]
                    if ((_3358.type_and_param0.x & 128u) != 0u)
                    {
                        float3 env_col_1 = _3327_g_params.env_col.xyz;
                        uint _3948 = asuint(_3327_g_params.env_col.w);
                        if (_3948 != 4294967295u)
                        {
                            env_col_1 *= SampleLatlong_RGBE(_3948, ls.L, _3327_g_params.env_rotation);
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
                        float3 _4006 = normalize(cross(P - _3358.param1.xyz, _3358.param3.xyz));
                        float _4013 = 3.1415927410125732421875f * frac(asfloat(_3311.Load((hi + 4) * 4 + 0)) + sample_off.x);
                        float3 _4038 = (_3358.param1.xyz + (((_4006 * cos(_4013)) + (cross(_4006, _3358.param3.xyz) * sin(_4013))) * _3358.param2.w)) + ((_3358.param3.xyz * (frac(asfloat(_3311.Load((hi + 5) * 4 + 0)) + sample_off.y) - 0.5f)) * _3358.param3.w);
                        ls.lp = _4038;
                        float3 _4044 = _4038 - P;
                        float _4047 = length(_4044);
                        ls.L = _4044 / _4047.xxx;
                        ls.area = _3358.param1.w;
                        float _4062 = 1.0f - abs(dot(ls.L, _3358.param3.xyz));
                        [flatten]
                        if (_4062 != 0.0f)
                        {
                            ls.pdf = (_4047 * _4047) / (ls.area * _4062);
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
                            uint _4092 = asuint(_3358.param1.x);
                            transform_t _4106;
                            _4106.xform = asfloat(uint4x4(_4100.Load4(asuint(_3358.param1.y) * 128 + 0), _4100.Load4(asuint(_3358.param1.y) * 128 + 16), _4100.Load4(asuint(_3358.param1.y) * 128 + 32), _4100.Load4(asuint(_3358.param1.y) * 128 + 48)));
                            _4106.inv_xform = asfloat(uint4x4(_4100.Load4(asuint(_3358.param1.y) * 128 + 64), _4100.Load4(asuint(_3358.param1.y) * 128 + 80), _4100.Load4(asuint(_3358.param1.y) * 128 + 96), _4100.Load4(asuint(_3358.param1.y) * 128 + 112)));
                            uint _4131 = _4092 * 3u;
                            vertex_t _4137;
                            [unroll]
                            for (int _44ident = 0; _44ident < 3; _44ident++)
                            {
                                _4137.p[_44ident] = asfloat(_4125.Load(_44ident * 4 + _4129.Load(_4131 * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _45ident = 0; _45ident < 3; _45ident++)
                            {
                                _4137.n[_45ident] = asfloat(_4125.Load(_45ident * 4 + _4129.Load(_4131 * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _46ident = 0; _46ident < 3; _46ident++)
                            {
                                _4137.b[_46ident] = asfloat(_4125.Load(_46ident * 4 + _4129.Load(_4131 * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _47ident = 0; _47ident < 2; _47ident++)
                            {
                                [unroll]
                                for (int _48ident = 0; _48ident < 2; _48ident++)
                                {
                                    _4137.t[_47ident][_48ident] = asfloat(_4125.Load(_48ident * 4 + _47ident * 8 + _4129.Load(_4131 * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4186;
                            [unroll]
                            for (int _49ident = 0; _49ident < 3; _49ident++)
                            {
                                _4186.p[_49ident] = asfloat(_4125.Load(_49ident * 4 + _4129.Load((_4131 + 1u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _50ident = 0; _50ident < 3; _50ident++)
                            {
                                _4186.n[_50ident] = asfloat(_4125.Load(_50ident * 4 + _4129.Load((_4131 + 1u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _51ident = 0; _51ident < 3; _51ident++)
                            {
                                _4186.b[_51ident] = asfloat(_4125.Load(_51ident * 4 + _4129.Load((_4131 + 1u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _52ident = 0; _52ident < 2; _52ident++)
                            {
                                [unroll]
                                for (int _53ident = 0; _53ident < 2; _53ident++)
                                {
                                    _4186.t[_52ident][_53ident] = asfloat(_4125.Load(_53ident * 4 + _52ident * 8 + _4129.Load((_4131 + 1u) * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4232;
                            [unroll]
                            for (int _54ident = 0; _54ident < 3; _54ident++)
                            {
                                _4232.p[_54ident] = asfloat(_4125.Load(_54ident * 4 + _4129.Load((_4131 + 2u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _55ident = 0; _55ident < 3; _55ident++)
                            {
                                _4232.n[_55ident] = asfloat(_4125.Load(_55ident * 4 + _4129.Load((_4131 + 2u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _56ident = 0; _56ident < 3; _56ident++)
                            {
                                _4232.b[_56ident] = asfloat(_4125.Load(_56ident * 4 + _4129.Load((_4131 + 2u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _57ident = 0; _57ident < 2; _57ident++)
                            {
                                [unroll]
                                for (int _58ident = 0; _58ident < 2; _58ident++)
                                {
                                    _4232.t[_57ident][_58ident] = asfloat(_4125.Load(_58ident * 4 + _57ident * 8 + _4129.Load((_4131 + 2u) * 4 + 0) * 52 + 36));
                                }
                            }
                            float3 _4278 = float3(_4137.p[0], _4137.p[1], _4137.p[2]);
                            float3 _4286 = float3(_4186.p[0], _4186.p[1], _4186.p[2]);
                            float3 _4294 = float3(_4232.p[0], _4232.p[1], _4232.p[2]);
                            float _4322 = sqrt(frac(asfloat(_3311.Load((hi + 4) * 4 + 0)) + sample_off.x));
                            float _4331 = frac(asfloat(_3311.Load((hi + 5) * 4 + 0)) + sample_off.y);
                            float _4335 = 1.0f - _4322;
                            float _4340 = 1.0f - _4331;
                            float3 _4371 = mul(float4((_4278 * _4335) + (((_4286 * _4340) + (_4294 * _4331)) * _4322), 1.0f), _4106.xform).xyz;
                            float3 _4387 = mul(float4(cross(_4286 - _4278, _4294 - _4278), 0.0f), _4106.xform).xyz;
                            ls.area = 0.5f * length(_4387);
                            float3 _4393 = normalize(_4387);
                            ls.L = _4371 - P;
                            float3 _4400 = ls.L;
                            float _4401 = length(_4400);
                            ls.L /= _4401.xxx;
                            float _4412 = dot(ls.L, _4393);
                            float cos_theta = _4412;
                            float3 _4415;
                            if (_4412 >= 0.0f)
                            {
                                _4415 = -_4393;
                            }
                            else
                            {
                                _4415 = _4393;
                            }
                            float3 param_12 = _4371;
                            float3 param_13 = _4415;
                            ls.lp = offset_ray(param_12, param_13);
                            float _4428 = cos_theta;
                            float _4429 = abs(_4428);
                            cos_theta = _4429;
                            [flatten]
                            if (_4429 > 0.0f)
                            {
                                ls.pdf = (_4401 * _4401) / (ls.area * cos_theta);
                            }
                            material_t _4467;
                            [unroll]
                            for (int _59ident = 0; _59ident < 5; _59ident++)
                            {
                                _4467.textures[_59ident] = _4453.Load(_59ident * 4 + ((_4457.Load(_4092 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
                            }
                            [unroll]
                            for (int _60ident = 0; _60ident < 3; _60ident++)
                            {
                                _4467.base_color[_60ident] = asfloat(_4453.Load(_60ident * 4 + ((_4457.Load(_4092 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
                            }
                            _4467.flags = _4453.Load(((_4457.Load(_4092 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
                            _4467.type = _4453.Load(((_4457.Load(_4092 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
                            _4467.tangent_rotation_or_strength = asfloat(_4453.Load(((_4457.Load(_4092 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
                            _4467.roughness_and_anisotropic = _4453.Load(((_4457.Load(_4092 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
                            _4467.ior = asfloat(_4453.Load(((_4457.Load(_4092 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
                            _4467.sheen_and_sheen_tint = _4453.Load(((_4457.Load(_4092 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
                            _4467.tint_and_metallic = _4453.Load(((_4457.Load(_4092 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
                            _4467.transmission_and_transmission_roughness = _4453.Load(((_4457.Load(_4092 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
                            _4467.specular_and_specular_tint = _4453.Load(((_4457.Load(_4092 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
                            _4467.clearcoat_and_clearcoat_roughness = _4453.Load(((_4457.Load(_4092 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
                            _4467.normal_map_strength_unorm = _4453.Load(((_4457.Load(_4092 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
                            if (_4467.textures[1] != 4294967295u)
                            {
                                ls.col *= SampleBilinear(_4467.textures[1], (float2(_4137.t[0][0], _4137.t[0][1]) * _4335) + (((float2(_4186.t[0][0], _4186.t[0][1]) * _4340) + (float2(_4232.t[0][0], _4232.t[0][1]) * _4331)) * _4322), 0).xyz;
                            }
                        }
                        else
                        {
                            [branch]
                            if (_3394 == 7u)
                            {
                                float _4553 = frac(asfloat(_3311.Load((hi + 4) * 4 + 0)) + sample_off.x);
                                float _4562 = frac(asfloat(_3311.Load((hi + 5) * 4 + 0)) + sample_off.y);
                                float4 dir_and_pdf;
                                if (_3327_g_params.env_qtree_levels > 0)
                                {
                                    dir_and_pdf = Sample_EnvQTree(_3327_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3327_g_params.env_qtree_levels, mad(_3320, _3332, -float(_3339)), _4553, _4562);
                                }
                                else
                                {
                                    float _4581 = 6.283185482025146484375f * _4562;
                                    float _4593 = sqrt(mad(-_4553, _4553, 1.0f));
                                    float3 param_14 = T;
                                    float3 param_15 = B;
                                    float3 param_16 = N;
                                    float3 param_17 = float3(_4593 * cos(_4581), _4593 * sin(_4581), _4553);
                                    dir_and_pdf = float4(world_from_tangent(param_14, param_15, param_16, param_17), 0.15915493667125701904296875f);
                                }
                                ls.L = dir_and_pdf.xyz;
                                ls.col *= _3327_g_params.env_col.xyz;
                                uint _4632 = asuint(_3327_g_params.env_col.w);
                                if (_4632 != 4294967295u)
                                {
                                    ls.col *= SampleLatlong_RGBE(_4632, ls.L, _3327_g_params.env_rotation);
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
    float3 _8849;
    do
    {
        float3 param = -float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param_1 = surf.N;
        float3 param_2 = ls.L;
        float param_3 = roughness;
        float3 param_4 = base_color;
        float4 _5112 = Evaluate_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_5 = ls.pdf;
            float param_6 = _5112.w;
            mis_weight = power_heuristic(param_5, param_6);
        }
        float3 _5135 = (ls.col * _5112.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8849 = _5135;
            break;
        }
        float3 param_7 = surf.P;
        float3 param_8 = surf.plane_N;
        float3 _5147 = offset_ray(param_7, param_8);
        sh_r.o[0] = _5147.x;
        sh_r.o[1] = _5147.y;
        sh_r.o[2] = _5147.z;
        sh_r.c[0] = ray.c[0] * _5135.x;
        sh_r.c[1] = ray.c[1] * _5135.y;
        sh_r.c[2] = ray.c[2] * _5135.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8849 = 0.0f.xxx;
        break;
    } while(false);
    return _8849;
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
    float4 _5398 = Sample_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8);
    new_ray.depth = ray.depth + 1;
    float3 param_9 = surf.P;
    float3 param_10 = surf.plane_N;
    float3 _5408 = offset_ray(param_9, param_10);
    new_ray.o[0] = _5408.x;
    new_ray.o[1] = _5408.y;
    new_ray.o[2] = _5408.z;
    new_ray.d[0] = param_8.x;
    new_ray.d[1] = param_8.y;
    new_ray.d[2] = param_8.z;
    new_ray.c[0] = ((ray.c[0] * _5398.x) * mix_weight) / _5398.w;
    new_ray.c[1] = ((ray.c[1] * _5398.y) * mix_weight) / _5398.w;
    new_ray.c[2] = ((ray.c[2] * _5398.z) * mix_weight) / _5398.w;
    new_ray.pdf = _5398.w;
}

float3 tangent_from_world(float3 T, float3 B, float3 N, float3 V)
{
    return float3(dot(V, T), dot(V, B), dot(V, N));
}

float D_GGX(float3 H, float alpha_x, float alpha_y)
{
    float _8902;
    do
    {
        if (H.z == 0.0f)
        {
            _8902 = 0.0f;
            break;
        }
        float _2032 = (-H.x) / (H.z * alpha_x);
        float _2038 = (-H.y) / (H.z * alpha_y);
        float _2047 = mad(_2038, _2038, mad(_2032, _2032, 1.0f));
        _8902 = 1.0f / (((((_2047 * _2047) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _8902;
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
    float3 _8854;
    do
    {
        float3 _5183 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5183;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - _5183);
        float _5221 = roughness * roughness;
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = _5221;
        float param_16 = _5221;
        float param_17 = spec_ior;
        float param_18 = spec_F0;
        float3 param_19 = base_color;
        float4 _5236 = Evaluate_GGXSpecular_BSDF(param_12, param_13, param_14, param_15, param_16, param_17, param_18, param_19);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_20 = ls.pdf;
            float param_21 = _5236.w;
            mis_weight = power_heuristic(param_20, param_21);
        }
        float3 _5259 = (ls.col * _5236.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8854 = _5259;
            break;
        }
        float3 param_22 = surf.P;
        float3 param_23 = surf.plane_N;
        float3 _5271 = offset_ray(param_22, param_23);
        sh_r.o[0] = _5271.x;
        sh_r.o[1] = _5271.y;
        sh_r.o[2] = _5271.z;
        sh_r.c[0] = ray.c[0] * _5259.x;
        sh_r.c[1] = ray.c[1] * _5259.y;
        sh_r.c[2] = ray.c[2] * _5259.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8854 = 0.0f.xxx;
        break;
    } while(false);
    return _8854;
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
    float4 _8874;
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
            _8874 = float4(_2687.x * 1000000.0f, _2687.y * 1000000.0f, _2687.z * 1000000.0f, 1000000.0f);
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
        _8874 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _8874;
}

void Sample_GlossyNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 param_4;
    float4 _5318 = Sample_GGXSpecular_BSDF(param, param_1, param_2, param_3, roughness, 0.0f, spec_ior, spec_F0, base_color, rand_u, rand_v, param_4);
    new_ray.depth = ray.depth + 256;
    float3 param_5 = surf.P;
    float3 param_6 = surf.plane_N;
    float3 _5329 = offset_ray(param_5, param_6);
    new_ray.o[0] = _5329.x;
    new_ray.o[1] = _5329.y;
    new_ray.o[2] = _5329.z;
    new_ray.d[0] = param_4.x;
    new_ray.d[1] = param_4.y;
    new_ray.d[2] = param_4.z;
    new_ray.c[0] = ((ray.c[0] * _5318.x) * mix_weight) / _5318.w;
    new_ray.c[1] = ((ray.c[1] * _5318.y) * mix_weight) / _5318.w;
    new_ray.c[2] = ((ray.c[2] * _5318.z) * mix_weight) / _5318.w;
    new_ray.pdf = _5318.w;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _8879;
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
            _8879 = 0.0f.xxxx;
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
        _8879 = float4(refr_col * (((((_2970 * _2986) * _2978) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3006) / view_dir_ts.z), (((_2970 * _2978) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3006) / view_dir_ts.z);
        break;
    } while(false);
    return _8879;
}

float3 Evaluate_RefractiveNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness2, float eta, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _8859;
    do
    {
        float3 _5461 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5461;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - (_5461 * eta));
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = roughness2;
        float param_16 = eta;
        float3 param_17 = base_color;
        float4 _5509 = Evaluate_GGXRefraction_BSDF(param_12, param_13, param_14, param_15, param_16, param_17);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_18 = ls.pdf;
            float param_19 = _5509.w;
            mis_weight = power_heuristic(param_18, param_19);
        }
        float3 _5532 = (ls.col * _5509.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8859 = _5532;
            break;
        }
        float3 param_20 = surf.P;
        float3 param_21 = -surf.plane_N;
        float3 _5545 = offset_ray(param_20, param_21);
        sh_r.o[0] = _5545.x;
        sh_r.o[1] = _5545.y;
        sh_r.o[2] = _5545.z;
        sh_r.c[0] = ray.c[0] * _5532.x;
        sh_r.c[1] = ray.c[1] * _5532.y;
        sh_r.c[2] = ray.c[2] * _5532.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8859 = 0.0f.xxx;
        break;
    } while(false);
    return _8859;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _8884;
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
                _8884 = 0.0f.xxxx;
                break;
            }
            float _3083 = mad(eta, _3061, -sqrt(_3071));
            out_V = float4(normalize((I * eta) + (N * _3083)), _3083);
            _8884 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
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
            _8884 = 0.0f.xxxx;
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
        _8884 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _8884;
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
    float _8892;
    do
    {
        if (stack[3] > 0.0f)
        {
            float param = stack[3];
            float param_1 = -1.0f;
            float _2138 = exchange(param, param_1);
            stack[3] = param;
            _8892 = _2138;
            break;
        }
        if (stack[2] > 0.0f)
        {
            float param_2 = stack[2];
            float param_3 = -1.0f;
            float _2151 = exchange(param_2, param_3);
            stack[2] = param_2;
            _8892 = _2151;
            break;
        }
        if (stack[1] > 0.0f)
        {
            float param_4 = stack[1];
            float param_5 = -1.0f;
            float _2164 = exchange(param_4, param_5);
            stack[1] = param_4;
            _8892 = _2164;
            break;
        }
        if (stack[0] > 0.0f)
        {
            float param_6 = stack[0];
            float param_7 = -1.0f;
            float _2177 = exchange(param_6, param_7);
            stack[0] = param_6;
            _8892 = _2177;
            break;
        }
        _8892 = default_value;
        break;
    } while(false);
    return _8892;
}

void Sample_RefractiveNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, bool is_backfacing, float int_ior, float ext_ior, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float _5582;
    if (is_backfacing)
    {
        _5582 = int_ior / ext_ior;
    }
    else
    {
        _5582 = ext_ior / int_ior;
    }
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float param_4 = roughness;
    float param_5 = _5582;
    float3 param_6 = base_color;
    float param_7 = rand_u;
    float param_8 = rand_v;
    float4 param_9;
    float4 _5606 = Sample_GGXRefraction_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9);
    new_ray.depth = ray.depth + 65536;
    new_ray.c[0] = ((ray.c[0] * _5606.x) * mix_weight) / _5606.w;
    new_ray.c[1] = ((ray.c[1] * _5606.y) * mix_weight) / _5606.w;
    new_ray.c[2] = ((ray.c[2] * _5606.z) * mix_weight) / _5606.w;
    new_ray.pdf = _5606.w;
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
        float _5662 = pop_ior_stack(param_11, param_12);
        new_ray.ior = param_11;
    }
    float3 param_13 = surf.P;
    float3 param_14 = -surf.plane_N;
    float3 _5671 = offset_ray(param_13, param_14);
    new_ray.o[0] = _5671.x;
    new_ray.o[1] = _5671.y;
    new_ray.o[2] = _5671.z;
    new_ray.d[0] = param_9.x;
    new_ray.d[1] = param_9.y;
    new_ray.d[2] = param_9.z;
}

lobe_weights_t get_lobe_weights(float base_color_lum, float spec_color_lum, float specular, float metallic, float transmission, float clearcoat)
{
    float _1502 = 1.0f - metallic;
    float _8978 = (base_color_lum * _1502) * (1.0f - transmission);
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
    float _8979 = _1513;
    float _1523 = 0.25f * clearcoat;
    float _8980 = _1523 * _1502;
    float _8981 = _1509 * base_color_lum;
    float _1532 = _8978;
    float _1541 = mad(_1509, base_color_lum, mad(_1523, _1502, _1532 + _1513));
    if (_1541 != 0.0f)
    {
        _8978 /= _1541;
        _8979 /= _1541;
        _8980 /= _1541;
        _8981 /= _1541;
    }
    lobe_weights_t _8986 = { _8978, _8979, _8980, _8981 };
    return _8986;
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
    float _8907;
    do
    {
        float _2258 = dot(N, L);
        if (_2258 <= 0.0f)
        {
            _8907 = 0.0f;
            break;
        }
        float param = _2258;
        float param_1 = dot(N, V);
        float _2279 = dot(L, H);
        float _2287 = mad((2.0f * _2279) * _2279, roughness, 0.5f);
        _8907 = lerp(1.0f, _2287, schlick_weight(param)) * lerp(1.0f, _2287, schlick_weight(param_1));
        break;
    } while(false);
    return _8907;
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
    float _8912;
    do
    {
        if (a >= 1.0f)
        {
            _8912 = 0.3183098733425140380859375f;
            break;
        }
        float _2006 = mad(a, a, -1.0f);
        _8912 = _2006 / ((3.1415927410125732421875f * log(a * a)) * mad(_2006 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _8912;
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
    float3 _8864;
    do
    {
        float3 _5694 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 lcol = 0.0f.xxx;
        float bsdf_pdf = 0.0f;
        bool _5699 = N_dot_L > 0.0f;
        [branch]
        if ((lobe_weights.diffuse > 1.0000000116860974230803549289703e-07f) && _5699)
        {
            float3 param = -_5694;
            float3 param_1 = surf.N;
            float3 param_2 = ls.L;
            float param_3 = diff.roughness;
            float3 param_4 = diff.base_color;
            float3 param_5 = diff.sheen_color;
            bool param_6 = false;
            float4 _5718 = Evaluate_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6);
            bsdf_pdf = mad(lobe_weights.diffuse, _5718.w, bsdf_pdf);
            lcol += (((ls.col * N_dot_L) * (_5718 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * ls.pdf).xxx);
        }
        float3 H;
        [flatten]
        if (_5699)
        {
            H = normalize(ls.L - _5694);
        }
        else
        {
            H = normalize(ls.L - (_5694 * trans.eta));
        }
        float _5757 = spec.roughness * spec.roughness;
        float _5762 = sqrt(mad(-0.89999997615814208984375f, spec.anisotropy, 1.0f));
        float _5766 = _5757 / _5762;
        float _5770 = _5757 * _5762;
        float3 param_7 = surf.T;
        float3 param_8 = surf.B;
        float3 param_9 = surf.N;
        float3 param_10 = -_5694;
        float3 _5781 = tangent_from_world(param_7, param_8, param_9, param_10);
        float3 param_11 = surf.T;
        float3 param_12 = surf.B;
        float3 param_13 = surf.N;
        float3 param_14 = ls.L;
        float3 _5791 = tangent_from_world(param_11, param_12, param_13, param_14);
        float3 param_15 = surf.T;
        float3 param_16 = surf.B;
        float3 param_17 = surf.N;
        float3 param_18 = H;
        float3 _5801 = tangent_from_world(param_15, param_16, param_17, param_18);
        bool _5803 = lobe_weights.specular > 0.0f;
        bool _5810;
        if (_5803)
        {
            _5810 = (_5766 * _5770) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _5810 = _5803;
        }
        [branch]
        if (_5810 && _5699)
        {
            float3 param_19 = _5781;
            float3 param_20 = _5801;
            float3 param_21 = _5791;
            float param_22 = _5766;
            float param_23 = _5770;
            float param_24 = spec.ior;
            float param_25 = spec.F0;
            float3 param_26 = spec.tmp_col;
            float4 _5832 = Evaluate_GGXSpecular_BSDF(param_19, param_20, param_21, param_22, param_23, param_24, param_25, param_26);
            bsdf_pdf = mad(lobe_weights.specular, _5832.w, bsdf_pdf);
            lcol += ((ls.col * _5832.xyz) / ls.pdf.xxx);
        }
        float _5851 = coat.roughness * coat.roughness;
        bool _5853 = lobe_weights.clearcoat > 0.0f;
        bool _5860;
        if (_5853)
        {
            _5860 = (_5851 * _5851) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _5860 = _5853;
        }
        [branch]
        if (_5860 && _5699)
        {
            float3 param_27 = _5781;
            float3 param_28 = _5801;
            float3 param_29 = _5791;
            float param_30 = _5851;
            float param_31 = coat.ior;
            float param_32 = coat.F0;
            float4 _5878 = Evaluate_PrincipledClearcoat_BSDF(param_27, param_28, param_29, param_30, param_31, param_32);
            bsdf_pdf = mad(lobe_weights.clearcoat, _5878.w, bsdf_pdf);
            lcol += (((ls.col * 0.25f) * _5878.xyz) / ls.pdf.xxx);
        }
        [branch]
        if (lobe_weights.refraction > 0.0f)
        {
            bool _5900 = trans.fresnel != 0.0f;
            bool _5907;
            if (_5900)
            {
                _5907 = (_5757 * _5757) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _5907 = _5900;
            }
            [branch]
            if (_5907 && _5699)
            {
                float3 param_33 = _5781;
                float3 param_34 = _5801;
                float3 param_35 = _5791;
                float param_36 = _5757;
                float param_37 = _5757;
                float param_38 = 1.0f;
                float param_39 = 0.0f;
                float3 param_40 = 1.0f.xxx;
                float4 _5926 = Evaluate_GGXSpecular_BSDF(param_33, param_34, param_35, param_36, param_37, param_38, param_39, param_40);
                bsdf_pdf = mad(lobe_weights.refraction * trans.fresnel, _5926.w, bsdf_pdf);
                lcol += ((ls.col * _5926.xyz) * (trans.fresnel / ls.pdf));
            }
            float _5948 = trans.roughness * trans.roughness;
            bool _5950 = trans.fresnel != 1.0f;
            bool _5957;
            if (_5950)
            {
                _5957 = (_5948 * _5948) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _5957 = _5950;
            }
            [branch]
            if (_5957 && (N_dot_L < 0.0f))
            {
                float3 param_41 = _5781;
                float3 param_42 = _5801;
                float3 param_43 = _5791;
                float param_44 = _5948;
                float param_45 = trans.eta;
                float3 param_46 = diff.base_color;
                float4 _5975 = Evaluate_GGXRefraction_BSDF(param_41, param_42, param_43, param_44, param_45, param_46);
                float _5978 = 1.0f - trans.fresnel;
                bsdf_pdf = mad(lobe_weights.refraction * _5978, _5975.w, bsdf_pdf);
                lcol += ((ls.col * _5975.xyz) * (_5978 / ls.pdf));
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
            _8864 = lcol;
            break;
        }
        float3 _6018;
        if (N_dot_L < 0.0f)
        {
            _6018 = -surf.plane_N;
        }
        else
        {
            _6018 = surf.plane_N;
        }
        float3 param_49 = surf.P;
        float3 param_50 = _6018;
        float3 _6029 = offset_ray(param_49, param_50);
        sh_r.o[0] = _6029.x;
        sh_r.o[1] = _6029.y;
        sh_r.o[2] = _6029.z;
        sh_r.c[0] = ray.c[0] * lcol.x;
        sh_r.c[1] = ray.c[1] * lcol.y;
        sh_r.c[2] = ray.c[2] * lcol.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8864 = 0.0f.xxx;
        break;
    } while(false);
    return _8864;
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
    float4 _8897;
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
            _8897 = float4(_2887, _2887, _2887, 1000000.0f);
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
        _8897 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _8897;
}

void Sample_PrincipledNode(ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float rand_u, float rand_v, inout float mix_rand, float mix_weight, inout ray_data_t new_ray)
{
    float3 _6064 = float3(ray.d[0], ray.d[1], ray.d[2]);
    int _6068 = ray.depth & 255;
    int _6072 = (ray.depth >> 8) & 255;
    int _6076 = (ray.depth >> 16) & 255;
    int _6087 = (_6068 + _6072) + _6076;
    [branch]
    if (mix_rand < lobe_weights.diffuse)
    {
        bool _6096 = _6068 < _3327_g_params.max_diff_depth;
        bool _6103;
        if (_6096)
        {
            _6103 = _6087 < _3327_g_params.max_total_depth;
        }
        else
        {
            _6103 = _6096;
        }
        if (_6103)
        {
            float3 param = surf.T;
            float3 param_1 = surf.B;
            float3 param_2 = surf.N;
            float3 param_3 = _6064;
            float param_4 = diff.roughness;
            float3 param_5 = diff.base_color;
            float3 param_6 = diff.sheen_color;
            bool param_7 = false;
            float param_8 = rand_u;
            float param_9 = rand_v;
            float3 param_10;
            float4 _6126 = Sample_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9, param_10);
            float3 _6131 = _6126.xyz * (1.0f - metallic);
            new_ray.depth = ray.depth + 1;
            float3 param_11 = surf.P;
            float3 param_12 = surf.plane_N;
            float3 _6146 = offset_ray(param_11, param_12);
            new_ray.o[0] = _6146.x;
            new_ray.o[1] = _6146.y;
            new_ray.o[2] = _6146.z;
            new_ray.d[0] = param_10.x;
            new_ray.d[1] = param_10.y;
            new_ray.d[2] = param_10.z;
            new_ray.c[0] = ((ray.c[0] * _6131.x) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[1] = ((ray.c[1] * _6131.y) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[2] = ((ray.c[2] * _6131.z) * mix_weight) / lobe_weights.diffuse;
            new_ray.pdf = _6126.w;
        }
    }
    else
    {
        float _6196 = lobe_weights.diffuse + lobe_weights.specular;
        [branch]
        if (mix_rand < _6196)
        {
            bool _6203 = _6072 < _3327_g_params.max_spec_depth;
            bool _6210;
            if (_6203)
            {
                _6210 = _6087 < _3327_g_params.max_total_depth;
            }
            else
            {
                _6210 = _6203;
            }
            if (_6210)
            {
                float3 param_13 = surf.T;
                float3 param_14 = surf.B;
                float3 param_15 = surf.N;
                float3 param_16 = _6064;
                float3 param_17;
                float4 _6229 = Sample_GGXSpecular_BSDF(param_13, param_14, param_15, param_16, spec.roughness, spec.anisotropy, spec.ior, spec.F0, spec.tmp_col, rand_u, rand_v, param_17);
                float _6234 = _6229.w * lobe_weights.specular;
                new_ray.depth = ray.depth + 256;
                new_ray.c[0] = ((ray.c[0] * _6229.x) * mix_weight) / _6234;
                new_ray.c[1] = ((ray.c[1] * _6229.y) * mix_weight) / _6234;
                new_ray.c[2] = ((ray.c[2] * _6229.z) * mix_weight) / _6234;
                new_ray.pdf = _6234;
                float3 param_18 = surf.P;
                float3 param_19 = surf.plane_N;
                float3 _6274 = offset_ray(param_18, param_19);
                new_ray.o[0] = _6274.x;
                new_ray.o[1] = _6274.y;
                new_ray.o[2] = _6274.z;
                new_ray.d[0] = param_17.x;
                new_ray.d[1] = param_17.y;
                new_ray.d[2] = param_17.z;
            }
        }
        else
        {
            float _6299 = _6196 + lobe_weights.clearcoat;
            [branch]
            if (mix_rand < _6299)
            {
                bool _6306 = _6072 < _3327_g_params.max_spec_depth;
                bool _6313;
                if (_6306)
                {
                    _6313 = _6087 < _3327_g_params.max_total_depth;
                }
                else
                {
                    _6313 = _6306;
                }
                if (_6313)
                {
                    float3 param_20 = surf.T;
                    float3 param_21 = surf.B;
                    float3 param_22 = surf.N;
                    float3 param_23 = _6064;
                    float param_24 = coat.roughness * coat.roughness;
                    float param_25 = coat.ior;
                    float param_26 = coat.F0;
                    float param_27 = rand_u;
                    float param_28 = rand_v;
                    float3 param_29;
                    float4 _6337 = Sample_PrincipledClearcoat_BSDF(param_20, param_21, param_22, param_23, param_24, param_25, param_26, param_27, param_28, param_29);
                    float _6342 = _6337.w * lobe_weights.clearcoat;
                    new_ray.depth = ray.depth + 256;
                    new_ray.c[0] = (((0.25f * ray.c[0]) * _6337.x) * mix_weight) / _6342;
                    new_ray.c[1] = (((0.25f * ray.c[1]) * _6337.y) * mix_weight) / _6342;
                    new_ray.c[2] = (((0.25f * ray.c[2]) * _6337.z) * mix_weight) / _6342;
                    new_ray.pdf = _6342;
                    float3 param_30 = surf.P;
                    float3 param_31 = surf.plane_N;
                    float3 _6385 = offset_ray(param_30, param_31);
                    new_ray.o[0] = _6385.x;
                    new_ray.o[1] = _6385.y;
                    new_ray.o[2] = _6385.z;
                    new_ray.d[0] = param_29.x;
                    new_ray.d[1] = param_29.y;
                    new_ray.d[2] = param_29.z;
                }
            }
            else
            {
                bool _6407 = mix_rand >= trans.fresnel;
                bool _6414;
                if (_6407)
                {
                    _6414 = _6076 < _3327_g_params.max_refr_depth;
                }
                else
                {
                    _6414 = _6407;
                }
                bool _6428;
                if (!_6414)
                {
                    bool _6420 = mix_rand < trans.fresnel;
                    bool _6427;
                    if (_6420)
                    {
                        _6427 = _6072 < _3327_g_params.max_spec_depth;
                    }
                    else
                    {
                        _6427 = _6420;
                    }
                    _6428 = _6427;
                }
                else
                {
                    _6428 = _6414;
                }
                bool _6435;
                if (_6428)
                {
                    _6435 = _6087 < _3327_g_params.max_total_depth;
                }
                else
                {
                    _6435 = _6428;
                }
                [branch]
                if (_6435)
                {
                    mix_rand -= _6299;
                    mix_rand /= lobe_weights.refraction;
                    float4 F;
                    float3 V;
                    [branch]
                    if (mix_rand < trans.fresnel)
                    {
                        float3 param_32 = surf.T;
                        float3 param_33 = surf.B;
                        float3 param_34 = surf.N;
                        float3 param_35 = _6064;
                        float3 param_36;
                        float4 _6465 = Sample_GGXSpecular_BSDF(param_32, param_33, param_34, param_35, spec.roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, rand_u, rand_v, param_36);
                        V = param_36;
                        F = _6465;
                        new_ray.depth = ray.depth + 256;
                        float3 param_37 = surf.P;
                        float3 param_38 = surf.plane_N;
                        float3 _6475 = offset_ray(param_37, param_38);
                        new_ray.o[0] = _6475.x;
                        new_ray.o[1] = _6475.y;
                        new_ray.o[2] = _6475.z;
                    }
                    else
                    {
                        float3 param_39 = surf.T;
                        float3 param_40 = surf.B;
                        float3 param_41 = surf.N;
                        float3 param_42 = _6064;
                        float param_43 = trans.roughness;
                        float param_44 = trans.eta;
                        float3 param_45 = diff.base_color;
                        float param_46 = rand_u;
                        float param_47 = rand_v;
                        float4 param_48;
                        float4 _6504 = Sample_GGXRefraction_BSDF(param_39, param_40, param_41, param_42, param_43, param_44, param_45, param_46, param_47, param_48);
                        F = _6504;
                        V = param_48.xyz;
                        new_ray.depth = ray.depth + 65536;
                        float3 param_49 = surf.P;
                        float3 param_50 = -surf.plane_N;
                        float3 _6517 = offset_ray(param_49, param_50);
                        new_ray.o[0] = _6517.x;
                        new_ray.o[1] = _6517.y;
                        new_ray.o[2] = _6517.z;
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
                            float _6543 = pop_ior_stack(param_52, param_53);
                            new_ray.ior = param_52;
                        }
                    }
                    float4 _10145 = F;
                    float _6549 = _10145.w * lobe_weights.refraction;
                    float4 _10147 = _10145;
                    _10147.w = _6549;
                    F = _10147;
                    new_ray.c[0] = ((ray.c[0] * _10145.x) * mix_weight) / _6549;
                    new_ray.c[1] = ((ray.c[1] * _10145.y) * mix_weight) / _6549;
                    new_ray.c[2] = ((ray.c[2] * _10145.z) * mix_weight) / _6549;
                    new_ray.pdf = _6549;
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
    float3 _8834;
    do
    {
        float3 _6605 = float3(ray.d[0], ray.d[1], ray.d[2]);
        [branch]
        if (inter.mask == 0)
        {
            ray_data_t param = ray;
            float3 _6614 = Evaluate_EnvColor(param);
            _8834 = float3(ray.c[0] * _6614.x, ray.c[1] * _6614.y, ray.c[2] * _6614.z);
            break;
        }
        float3 _6641 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_6605 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            ray_data_t param_1 = ray;
            hit_data_t param_2 = inter;
            float3 _6653 = Evaluate_LightColor(param_1, param_2);
            _8834 = float3(ray.c[0] * _6653.x, ray.c[1] * _6653.y, ray.c[2] * _6653.z);
            break;
        }
        bool _6674 = inter.prim_index < 0;
        int _6677;
        if (_6674)
        {
            _6677 = (-1) - inter.prim_index;
        }
        else
        {
            _6677 = inter.prim_index;
        }
        uint _6688 = uint(_6677);
        material_t _6696;
        [unroll]
        for (int _61ident = 0; _61ident < 5; _61ident++)
        {
            _6696.textures[_61ident] = _4453.Load(_61ident * 4 + ((_4457.Load(_6688 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
        }
        [unroll]
        for (int _62ident = 0; _62ident < 3; _62ident++)
        {
            _6696.base_color[_62ident] = asfloat(_4453.Load(_62ident * 4 + ((_4457.Load(_6688 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
        }
        _6696.flags = _4453.Load(((_4457.Load(_6688 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
        _6696.type = _4453.Load(((_4457.Load(_6688 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
        _6696.tangent_rotation_or_strength = asfloat(_4453.Load(((_4457.Load(_6688 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
        _6696.roughness_and_anisotropic = _4453.Load(((_4457.Load(_6688 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
        _6696.ior = asfloat(_4453.Load(((_4457.Load(_6688 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
        _6696.sheen_and_sheen_tint = _4453.Load(((_4457.Load(_6688 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
        _6696.tint_and_metallic = _4453.Load(((_4457.Load(_6688 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
        _6696.transmission_and_transmission_roughness = _4453.Load(((_4457.Load(_6688 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
        _6696.specular_and_specular_tint = _4453.Load(((_4457.Load(_6688 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
        _6696.clearcoat_and_clearcoat_roughness = _4453.Load(((_4457.Load(_6688 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
        _6696.normal_map_strength_unorm = _4453.Load(((_4457.Load(_6688 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
        uint _9594 = _6696.textures[0];
        uint _9595 = _6696.textures[1];
        uint _9596 = _6696.textures[2];
        uint _9597 = _6696.textures[3];
        uint _9598 = _6696.textures[4];
        float _9599 = _6696.base_color[0];
        float _9600 = _6696.base_color[1];
        float _9601 = _6696.base_color[2];
        uint _9195 = _6696.flags;
        uint _9196 = _6696.type;
        float _9197 = _6696.tangent_rotation_or_strength;
        uint _9198 = _6696.roughness_and_anisotropic;
        float _9199 = _6696.ior;
        uint _9200 = _6696.sheen_and_sheen_tint;
        uint _9201 = _6696.tint_and_metallic;
        uint _9202 = _6696.transmission_and_transmission_roughness;
        uint _9203 = _6696.specular_and_specular_tint;
        uint _9204 = _6696.clearcoat_and_clearcoat_roughness;
        uint _9205 = _6696.normal_map_strength_unorm;
        transform_t _6751;
        _6751.xform = asfloat(uint4x4(_4100.Load4(asuint(asfloat(_6744.Load(inter.obj_index * 32 + 12))) * 128 + 0), _4100.Load4(asuint(asfloat(_6744.Load(inter.obj_index * 32 + 12))) * 128 + 16), _4100.Load4(asuint(asfloat(_6744.Load(inter.obj_index * 32 + 12))) * 128 + 32), _4100.Load4(asuint(asfloat(_6744.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _6751.inv_xform = asfloat(uint4x4(_4100.Load4(asuint(asfloat(_6744.Load(inter.obj_index * 32 + 12))) * 128 + 64), _4100.Load4(asuint(asfloat(_6744.Load(inter.obj_index * 32 + 12))) * 128 + 80), _4100.Load4(asuint(asfloat(_6744.Load(inter.obj_index * 32 + 12))) * 128 + 96), _4100.Load4(asuint(asfloat(_6744.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _6758 = _6688 * 3u;
        vertex_t _6763;
        [unroll]
        for (int _63ident = 0; _63ident < 3; _63ident++)
        {
            _6763.p[_63ident] = asfloat(_4125.Load(_63ident * 4 + _4129.Load(_6758 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _64ident = 0; _64ident < 3; _64ident++)
        {
            _6763.n[_64ident] = asfloat(_4125.Load(_64ident * 4 + _4129.Load(_6758 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _65ident = 0; _65ident < 3; _65ident++)
        {
            _6763.b[_65ident] = asfloat(_4125.Load(_65ident * 4 + _4129.Load(_6758 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _66ident = 0; _66ident < 2; _66ident++)
        {
            [unroll]
            for (int _67ident = 0; _67ident < 2; _67ident++)
            {
                _6763.t[_66ident][_67ident] = asfloat(_4125.Load(_67ident * 4 + _66ident * 8 + _4129.Load(_6758 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _6809;
        [unroll]
        for (int _68ident = 0; _68ident < 3; _68ident++)
        {
            _6809.p[_68ident] = asfloat(_4125.Load(_68ident * 4 + _4129.Load((_6758 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _69ident = 0; _69ident < 3; _69ident++)
        {
            _6809.n[_69ident] = asfloat(_4125.Load(_69ident * 4 + _4129.Load((_6758 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _70ident = 0; _70ident < 3; _70ident++)
        {
            _6809.b[_70ident] = asfloat(_4125.Load(_70ident * 4 + _4129.Load((_6758 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _71ident = 0; _71ident < 2; _71ident++)
        {
            [unroll]
            for (int _72ident = 0; _72ident < 2; _72ident++)
            {
                _6809.t[_71ident][_72ident] = asfloat(_4125.Load(_72ident * 4 + _71ident * 8 + _4129.Load((_6758 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _6855;
        [unroll]
        for (int _73ident = 0; _73ident < 3; _73ident++)
        {
            _6855.p[_73ident] = asfloat(_4125.Load(_73ident * 4 + _4129.Load((_6758 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _74ident = 0; _74ident < 3; _74ident++)
        {
            _6855.n[_74ident] = asfloat(_4125.Load(_74ident * 4 + _4129.Load((_6758 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _75ident = 0; _75ident < 3; _75ident++)
        {
            _6855.b[_75ident] = asfloat(_4125.Load(_75ident * 4 + _4129.Load((_6758 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _76ident = 0; _76ident < 2; _76ident++)
        {
            [unroll]
            for (int _77ident = 0; _77ident < 2; _77ident++)
            {
                _6855.t[_76ident][_77ident] = asfloat(_4125.Load(_77ident * 4 + _76ident * 8 + _4129.Load((_6758 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _6901 = float3(_6763.p[0], _6763.p[1], _6763.p[2]);
        float3 _6909 = float3(_6809.p[0], _6809.p[1], _6809.p[2]);
        float3 _6917 = float3(_6855.p[0], _6855.p[1], _6855.p[2]);
        float _6924 = (1.0f - inter.u) - inter.v;
        float3 _6956 = normalize(((float3(_6763.n[0], _6763.n[1], _6763.n[2]) * _6924) + (float3(_6809.n[0], _6809.n[1], _6809.n[2]) * inter.u)) + (float3(_6855.n[0], _6855.n[1], _6855.n[2]) * inter.v));
        float3 _9134 = _6956;
        float2 _6982 = ((float2(_6763.t[0][0], _6763.t[0][1]) * _6924) + (float2(_6809.t[0][0], _6809.t[0][1]) * inter.u)) + (float2(_6855.t[0][0], _6855.t[0][1]) * inter.v);
        float3 _6998 = cross(_6909 - _6901, _6917 - _6901);
        float _7003 = length(_6998);
        float3 _9135 = _6998 / _7003.xxx;
        float3 _7040 = ((float3(_6763.b[0], _6763.b[1], _6763.b[2]) * _6924) + (float3(_6809.b[0], _6809.b[1], _6809.b[2]) * inter.u)) + (float3(_6855.b[0], _6855.b[1], _6855.b[2]) * inter.v);
        float3 _9133 = _7040;
        float3 _9132 = cross(_7040, _6956);
        if (_6674)
        {
            if ((_4457.Load(_6688 * 4 + 0) & 65535u) == 65535u)
            {
                _8834 = 0.0f.xxx;
                break;
            }
            material_t _7066;
            [unroll]
            for (int _78ident = 0; _78ident < 5; _78ident++)
            {
                _7066.textures[_78ident] = _4453.Load(_78ident * 4 + (_4457.Load(_6688 * 4 + 0) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _79ident = 0; _79ident < 3; _79ident++)
            {
                _7066.base_color[_79ident] = asfloat(_4453.Load(_79ident * 4 + (_4457.Load(_6688 * 4 + 0) & 16383u) * 76 + 20));
            }
            _7066.flags = _4453.Load((_4457.Load(_6688 * 4 + 0) & 16383u) * 76 + 32);
            _7066.type = _4453.Load((_4457.Load(_6688 * 4 + 0) & 16383u) * 76 + 36);
            _7066.tangent_rotation_or_strength = asfloat(_4453.Load((_4457.Load(_6688 * 4 + 0) & 16383u) * 76 + 40));
            _7066.roughness_and_anisotropic = _4453.Load((_4457.Load(_6688 * 4 + 0) & 16383u) * 76 + 44);
            _7066.ior = asfloat(_4453.Load((_4457.Load(_6688 * 4 + 0) & 16383u) * 76 + 48));
            _7066.sheen_and_sheen_tint = _4453.Load((_4457.Load(_6688 * 4 + 0) & 16383u) * 76 + 52);
            _7066.tint_and_metallic = _4453.Load((_4457.Load(_6688 * 4 + 0) & 16383u) * 76 + 56);
            _7066.transmission_and_transmission_roughness = _4453.Load((_4457.Load(_6688 * 4 + 0) & 16383u) * 76 + 60);
            _7066.specular_and_specular_tint = _4453.Load((_4457.Load(_6688 * 4 + 0) & 16383u) * 76 + 64);
            _7066.clearcoat_and_clearcoat_roughness = _4453.Load((_4457.Load(_6688 * 4 + 0) & 16383u) * 76 + 68);
            _7066.normal_map_strength_unorm = _4453.Load((_4457.Load(_6688 * 4 + 0) & 16383u) * 76 + 72);
            _9594 = _7066.textures[0];
            _9595 = _7066.textures[1];
            _9596 = _7066.textures[2];
            _9597 = _7066.textures[3];
            _9598 = _7066.textures[4];
            _9599 = _7066.base_color[0];
            _9600 = _7066.base_color[1];
            _9601 = _7066.base_color[2];
            _9195 = _7066.flags;
            _9196 = _7066.type;
            _9197 = _7066.tangent_rotation_or_strength;
            _9198 = _7066.roughness_and_anisotropic;
            _9199 = _7066.ior;
            _9200 = _7066.sheen_and_sheen_tint;
            _9201 = _7066.tint_and_metallic;
            _9202 = _7066.transmission_and_transmission_roughness;
            _9203 = _7066.specular_and_specular_tint;
            _9204 = _7066.clearcoat_and_clearcoat_roughness;
            _9205 = _7066.normal_map_strength_unorm;
            _9135 = -_9135;
            _9134 = -_9134;
            _9133 = -_9133;
            _9132 = -_9132;
        }
        float3 param_3 = _9135;
        float4x4 param_4 = _6751.inv_xform;
        _9135 = TransformNormal(param_3, param_4);
        float3 param_5 = _9134;
        float4x4 param_6 = _6751.inv_xform;
        _9134 = TransformNormal(param_5, param_6);
        float3 param_7 = _9133;
        float4x4 param_8 = _6751.inv_xform;
        _9133 = TransformNormal(param_7, param_8);
        float3 param_9 = _9132;
        float4x4 param_10 = _6751.inv_xform;
        _9135 = normalize(_9135);
        _9134 = normalize(_9134);
        _9133 = normalize(_9133);
        _9132 = normalize(TransformNormal(param_9, param_10));
        float _7206 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _7216 = mad(0.5f, log2(abs(mad(_6809.t[0][0] - _6763.t[0][0], _6855.t[0][1] - _6763.t[0][1], -((_6855.t[0][0] - _6763.t[0][0]) * (_6809.t[0][1] - _6763.t[0][1])))) / _7003), log2(_7206));
        uint param_11 = uint(hash(ray.xy));
        float _7223 = construct_float(param_11);
        uint param_12 = uint(hash(hash(ray.xy)));
        float _7230 = construct_float(param_12);
        float param_13[4] = ray.ior;
        bool param_14 = _6674;
        float param_15 = 1.0f;
        float _7239 = peek_ior_stack(param_13, param_14, param_15);
        float3 col = 0.0f.xxx;
        int _7244 = ray.depth & 255;
        int _7249 = (ray.depth >> 8) & 255;
        int _7254 = (ray.depth >> 16) & 255;
        int _7265 = (_7244 + _7249) + _7254;
        int _7273 = _3327_g_params.hi + ((_7265 + ((ray.depth >> 24) & 255)) * 7);
        float mix_rand = frac(asfloat(_3311.Load(_7273 * 4 + 0)) + _7223);
        float mix_weight = 1.0f;
        float _7310;
        float _7327;
        float _7353;
        float _7420;
        while (_9196 == 4u)
        {
            float mix_val = _9197;
            if (_9595 != 4294967295u)
            {
                mix_val *= SampleBilinear(_9595, _6982, 0).x;
            }
            if (_6674)
            {
                _7310 = _7239 / _9199;
            }
            else
            {
                _7310 = _9199 / _7239;
            }
            if (_9199 != 0.0f)
            {
                float param_16 = dot(_6605, _9134);
                float param_17 = _7310;
                _7327 = fresnel_dielectric_cos(param_16, param_17);
            }
            else
            {
                _7327 = 1.0f;
            }
            float _7342 = mix_val;
            float _7343 = _7342 * clamp(_7327, 0.0f, 1.0f);
            mix_val = _7343;
            if (mix_rand > _7343)
            {
                if ((_9195 & 2u) != 0u)
                {
                    _7353 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _7353 = 1.0f;
                }
                mix_weight *= _7353;
                material_t _7366;
                [unroll]
                for (int _80ident = 0; _80ident < 5; _80ident++)
                {
                    _7366.textures[_80ident] = _4453.Load(_80ident * 4 + _9597 * 76 + 0);
                }
                [unroll]
                for (int _81ident = 0; _81ident < 3; _81ident++)
                {
                    _7366.base_color[_81ident] = asfloat(_4453.Load(_81ident * 4 + _9597 * 76 + 20));
                }
                _7366.flags = _4453.Load(_9597 * 76 + 32);
                _7366.type = _4453.Load(_9597 * 76 + 36);
                _7366.tangent_rotation_or_strength = asfloat(_4453.Load(_9597 * 76 + 40));
                _7366.roughness_and_anisotropic = _4453.Load(_9597 * 76 + 44);
                _7366.ior = asfloat(_4453.Load(_9597 * 76 + 48));
                _7366.sheen_and_sheen_tint = _4453.Load(_9597 * 76 + 52);
                _7366.tint_and_metallic = _4453.Load(_9597 * 76 + 56);
                _7366.transmission_and_transmission_roughness = _4453.Load(_9597 * 76 + 60);
                _7366.specular_and_specular_tint = _4453.Load(_9597 * 76 + 64);
                _7366.clearcoat_and_clearcoat_roughness = _4453.Load(_9597 * 76 + 68);
                _7366.normal_map_strength_unorm = _4453.Load(_9597 * 76 + 72);
                _9594 = _7366.textures[0];
                _9595 = _7366.textures[1];
                _9596 = _7366.textures[2];
                _9597 = _7366.textures[3];
                _9598 = _7366.textures[4];
                _9599 = _7366.base_color[0];
                _9600 = _7366.base_color[1];
                _9601 = _7366.base_color[2];
                _9195 = _7366.flags;
                _9196 = _7366.type;
                _9197 = _7366.tangent_rotation_or_strength;
                _9198 = _7366.roughness_and_anisotropic;
                _9199 = _7366.ior;
                _9200 = _7366.sheen_and_sheen_tint;
                _9201 = _7366.tint_and_metallic;
                _9202 = _7366.transmission_and_transmission_roughness;
                _9203 = _7366.specular_and_specular_tint;
                _9204 = _7366.clearcoat_and_clearcoat_roughness;
                _9205 = _7366.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_9195 & 2u) != 0u)
                {
                    _7420 = 1.0f / mix_val;
                }
                else
                {
                    _7420 = 1.0f;
                }
                mix_weight *= _7420;
                material_t _7432;
                [unroll]
                for (int _82ident = 0; _82ident < 5; _82ident++)
                {
                    _7432.textures[_82ident] = _4453.Load(_82ident * 4 + _9598 * 76 + 0);
                }
                [unroll]
                for (int _83ident = 0; _83ident < 3; _83ident++)
                {
                    _7432.base_color[_83ident] = asfloat(_4453.Load(_83ident * 4 + _9598 * 76 + 20));
                }
                _7432.flags = _4453.Load(_9598 * 76 + 32);
                _7432.type = _4453.Load(_9598 * 76 + 36);
                _7432.tangent_rotation_or_strength = asfloat(_4453.Load(_9598 * 76 + 40));
                _7432.roughness_and_anisotropic = _4453.Load(_9598 * 76 + 44);
                _7432.ior = asfloat(_4453.Load(_9598 * 76 + 48));
                _7432.sheen_and_sheen_tint = _4453.Load(_9598 * 76 + 52);
                _7432.tint_and_metallic = _4453.Load(_9598 * 76 + 56);
                _7432.transmission_and_transmission_roughness = _4453.Load(_9598 * 76 + 60);
                _7432.specular_and_specular_tint = _4453.Load(_9598 * 76 + 64);
                _7432.clearcoat_and_clearcoat_roughness = _4453.Load(_9598 * 76 + 68);
                _7432.normal_map_strength_unorm = _4453.Load(_9598 * 76 + 72);
                _9594 = _7432.textures[0];
                _9595 = _7432.textures[1];
                _9596 = _7432.textures[2];
                _9597 = _7432.textures[3];
                _9598 = _7432.textures[4];
                _9599 = _7432.base_color[0];
                _9600 = _7432.base_color[1];
                _9601 = _7432.base_color[2];
                _9195 = _7432.flags;
                _9196 = _7432.type;
                _9197 = _7432.tangent_rotation_or_strength;
                _9198 = _7432.roughness_and_anisotropic;
                _9199 = _7432.ior;
                _9200 = _7432.sheen_and_sheen_tint;
                _9201 = _7432.tint_and_metallic;
                _9202 = _7432.transmission_and_transmission_roughness;
                _9203 = _7432.specular_and_specular_tint;
                _9204 = _7432.clearcoat_and_clearcoat_roughness;
                _9205 = _7432.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_9594 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_9594, _6982, 0).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_9594 & 33554432u) != 0u)
            {
                float3 _10166 = normals;
                _10166.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _10166;
            }
            float3 _7514 = _9134;
            _9134 = normalize(((_9132 * normals.x) + (_7514 * normals.z)) + (_9133 * normals.y));
            if ((_9205 & 65535u) != 65535u)
            {
                _9134 = normalize(_7514 + ((_9134 - _7514) * clamp(float(_9205 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_18 = _9135;
            float3 param_19 = -_6605;
            float3 param_20 = _9134;
            _9134 = ensure_valid_reflection(param_18, param_19, param_20);
        }
        float3 _7580 = ((_6901 * _6924) + (_6909 * inter.u)) + (_6917 * inter.v);
        float3 _7587 = float3(-_7580.z, 0.0f, _7580.x);
        float3 tangent = _7587;
        float3 param_21 = _7587;
        float4x4 param_22 = _6751.inv_xform;
        float3 _7593 = TransformNormal(param_21, param_22);
        tangent = _7593;
        float3 _7597 = cross(_7593, _9134);
        if (dot(_7597, _7597) == 0.0f)
        {
            float3 param_23 = _7580;
            float4x4 param_24 = _6751.inv_xform;
            tangent = TransformNormal(param_23, param_24);
        }
        if (_9197 != 0.0f)
        {
            float3 param_25 = tangent;
            float3 param_26 = _9134;
            float param_27 = _9197;
            tangent = rotate_around_axis(param_25, param_26, param_27);
        }
        float3 _7630 = normalize(cross(tangent, _9134));
        _9133 = _7630;
        _9132 = cross(_9134, _7630);
        float3 _9293 = 0.0f.xxx;
        float3 _9292 = 0.0f.xxx;
        float _9297 = 0.0f;
        float _9295 = 0.0f;
        float _9296 = 1.0f;
        bool _7646 = _3327_g_params.li_count != 0;
        bool _7652;
        if (_7646)
        {
            _7652 = _9196 != 3u;
        }
        else
        {
            _7652 = _7646;
        }
        float3 _9294;
        bool _9298;
        bool _9299;
        if (_7652)
        {
            float3 param_28 = _6641;
            float3 param_29 = _9132;
            float3 param_30 = _9133;
            float3 param_31 = _9134;
            int param_32 = _7273;
            float2 param_33 = float2(_7223, _7230);
            light_sample_t _9308 = { _9292, _9293, _9294, _9295, _9296, _9297, _9298, _9299 };
            light_sample_t param_34 = _9308;
            SampleLightSource(param_28, param_29, param_30, param_31, param_32, param_33, param_34);
            _9292 = param_34.col;
            _9293 = param_34.L;
            _9294 = param_34.lp;
            _9295 = param_34.area;
            _9296 = param_34.dist_mul;
            _9297 = param_34.pdf;
            _9298 = param_34.cast_shadow;
            _9299 = param_34.from_env;
        }
        float _7680 = dot(_9134, _9293);
        float3 base_color = float3(_9599, _9600, _9601);
        [branch]
        if (_9595 != 4294967295u)
        {
            base_color *= SampleBilinear(_9595, _6982, int(get_texture_lod(texSize(_9595), _7216)), true, true).xyz;
        }
        out_base_color = base_color;
        out_normals = _9134;
        float3 tint_color = 0.0f.xxx;
        float _7716 = lum(base_color);
        [flatten]
        if (_7716 > 0.0f)
        {
            tint_color = base_color / _7716.xxx;
        }
        float roughness = clamp(float(_9198 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_9596 != 4294967295u)
        {
            roughness *= SampleBilinear(_9596, _6982, int(get_texture_lod(texSize(_9596), _7216)), false, true).x;
        }
        float _7761 = frac(asfloat(_3311.Load((_7273 + 1) * 4 + 0)) + _7223);
        float _7770 = frac(asfloat(_3311.Load((_7273 + 2) * 4 + 0)) + _7230);
        float _9721 = 0.0f;
        float _9720 = 0.0f;
        float _9719 = 0.0f;
        float _9357[4];
        [unroll]
        for (int i = 0; i < 4; )
        {
            _9357[i] = ray.ior[i];
            i++;
            continue;
        }
        float _9358 = _7206;
        float _9359 = ray.cone_spread;
        int _9360 = ray.xy;
        float _9355 = 0.0f;
        float _9826 = 0.0f;
        float _9825 = 0.0f;
        float _9824 = 0.0f;
        int _9462 = ray.depth;
        int _9466 = ray.xy;
        int _9361;
        float _9464;
        float _9649;
        float _9650;
        float _9651;
        float _9684;
        float _9685;
        float _9686;
        float _9754;
        float _9755;
        float _9756;
        float _9789;
        float _9790;
        float _9791;
        [branch]
        if (_9196 == 0u)
        {
            [branch]
            if ((_9297 > 0.0f) && (_7680 > 0.0f))
            {
                light_sample_t _9325 = { _9292, _9293, _9294, _9295, _9296, _9297, _9298, _9299 };
                surface_t _9143 = { _6641, _9132, _9133, _9134, _9135, _6982 };
                float _9830[3] = { _9824, _9825, _9826 };
                float _9795[3] = { _9789, _9790, _9791 };
                float _9760[3] = { _9754, _9755, _9756 };
                shadow_ray_t _9476 = { _9760, _9462, _9795, _9464, _9830, _9466 };
                shadow_ray_t param_35 = _9476;
                float3 _7830 = Evaluate_DiffuseNode(_9325, ray, _9143, base_color, roughness, mix_weight, param_35);
                _9754 = param_35.o[0];
                _9755 = param_35.o[1];
                _9756 = param_35.o[2];
                _9462 = param_35.depth;
                _9789 = param_35.d[0];
                _9790 = param_35.d[1];
                _9791 = param_35.d[2];
                _9464 = param_35.dist;
                _9824 = param_35.c[0];
                _9825 = param_35.c[1];
                _9826 = param_35.c[2];
                _9466 = param_35.xy;
                col += _7830;
            }
            bool _7837 = _7244 < _3327_g_params.max_diff_depth;
            bool _7844;
            if (_7837)
            {
                _7844 = _7265 < _3327_g_params.max_total_depth;
            }
            else
            {
                _7844 = _7837;
            }
            [branch]
            if (_7844)
            {
                surface_t _9150 = { _6641, _9132, _9133, _9134, _9135, _6982 };
                float _9725[3] = { _9719, _9720, _9721 };
                float _9690[3] = { _9684, _9685, _9686 };
                float _9655[3] = { _9649, _9650, _9651 };
                ray_data_t _9375 = { _9655, _9690, _9355, _9725, _9357, _9358, _9359, _9360, _9361 };
                ray_data_t param_36 = _9375;
                Sample_DiffuseNode(ray, _9150, base_color, roughness, _7761, _7770, mix_weight, param_36);
                _9649 = param_36.o[0];
                _9650 = param_36.o[1];
                _9651 = param_36.o[2];
                _9684 = param_36.d[0];
                _9685 = param_36.d[1];
                _9686 = param_36.d[2];
                _9355 = param_36.pdf;
                _9719 = param_36.c[0];
                _9720 = param_36.c[1];
                _9721 = param_36.c[2];
                _9357 = param_36.ior;
                _9358 = param_36.cone_width;
                _9359 = param_36.cone_spread;
                _9360 = param_36.xy;
                _9361 = param_36.depth;
            }
        }
        else
        {
            [branch]
            if (_9196 == 1u)
            {
                float param_37 = 1.0f;
                float param_38 = 1.5f;
                float _7868 = fresnel_dielectric_cos(param_37, param_38);
                float _7872 = roughness * roughness;
                bool _7875 = _9297 > 0.0f;
                bool _7882;
                if (_7875)
                {
                    _7882 = (_7872 * _7872) >= 1.0000000116860974230803549289703e-07f;
                }
                else
                {
                    _7882 = _7875;
                }
                [branch]
                if (_7882 && (_7680 > 0.0f))
                {
                    light_sample_t _9334 = { _9292, _9293, _9294, _9295, _9296, _9297, _9298, _9299 };
                    surface_t _9157 = { _6641, _9132, _9133, _9134, _9135, _6982 };
                    float _9837[3] = { _9824, _9825, _9826 };
                    float _9802[3] = { _9789, _9790, _9791 };
                    float _9767[3] = { _9754, _9755, _9756 };
                    shadow_ray_t _9489 = { _9767, _9462, _9802, _9464, _9837, _9466 };
                    shadow_ray_t param_39 = _9489;
                    float3 _7897 = Evaluate_GlossyNode(_9334, ray, _9157, base_color, roughness, 1.5f, _7868, mix_weight, param_39);
                    _9754 = param_39.o[0];
                    _9755 = param_39.o[1];
                    _9756 = param_39.o[2];
                    _9462 = param_39.depth;
                    _9789 = param_39.d[0];
                    _9790 = param_39.d[1];
                    _9791 = param_39.d[2];
                    _9464 = param_39.dist;
                    _9824 = param_39.c[0];
                    _9825 = param_39.c[1];
                    _9826 = param_39.c[2];
                    _9466 = param_39.xy;
                    col += _7897;
                }
                bool _7904 = _7249 < _3327_g_params.max_spec_depth;
                bool _7911;
                if (_7904)
                {
                    _7911 = _7265 < _3327_g_params.max_total_depth;
                }
                else
                {
                    _7911 = _7904;
                }
                [branch]
                if (_7911)
                {
                    surface_t _9164 = { _6641, _9132, _9133, _9134, _9135, _6982 };
                    float _9732[3] = { _9719, _9720, _9721 };
                    float _9697[3] = { _9684, _9685, _9686 };
                    float _9662[3] = { _9649, _9650, _9651 };
                    ray_data_t _9394 = { _9662, _9697, _9355, _9732, _9357, _9358, _9359, _9360, _9361 };
                    ray_data_t param_40 = _9394;
                    Sample_GlossyNode(ray, _9164, base_color, roughness, 1.5f, _7868, _7761, _7770, mix_weight, param_40);
                    _9649 = param_40.o[0];
                    _9650 = param_40.o[1];
                    _9651 = param_40.o[2];
                    _9684 = param_40.d[0];
                    _9685 = param_40.d[1];
                    _9686 = param_40.d[2];
                    _9355 = param_40.pdf;
                    _9719 = param_40.c[0];
                    _9720 = param_40.c[1];
                    _9721 = param_40.c[2];
                    _9357 = param_40.ior;
                    _9358 = param_40.cone_width;
                    _9359 = param_40.cone_spread;
                    _9360 = param_40.xy;
                    _9361 = param_40.depth;
                }
            }
            else
            {
                [branch]
                if (_9196 == 2u)
                {
                    float _7935 = roughness * roughness;
                    bool _7938 = _9297 > 0.0f;
                    bool _7945;
                    if (_7938)
                    {
                        _7945 = (_7935 * _7935) >= 1.0000000116860974230803549289703e-07f;
                    }
                    else
                    {
                        _7945 = _7938;
                    }
                    [branch]
                    if (_7945 && (_7680 < 0.0f))
                    {
                        float _7953;
                        if (_6674)
                        {
                            _7953 = _9199 / _7239;
                        }
                        else
                        {
                            _7953 = _7239 / _9199;
                        }
                        light_sample_t _9343 = { _9292, _9293, _9294, _9295, _9296, _9297, _9298, _9299 };
                        surface_t _9171 = { _6641, _9132, _9133, _9134, _9135, _6982 };
                        float _9844[3] = { _9824, _9825, _9826 };
                        float _9809[3] = { _9789, _9790, _9791 };
                        float _9774[3] = { _9754, _9755, _9756 };
                        shadow_ray_t _9502 = { _9774, _9462, _9809, _9464, _9844, _9466 };
                        shadow_ray_t param_41 = _9502;
                        float3 _7975 = Evaluate_RefractiveNode(_9343, ray, _9171, base_color, _7935, _7953, mix_weight, param_41);
                        _9754 = param_41.o[0];
                        _9755 = param_41.o[1];
                        _9756 = param_41.o[2];
                        _9462 = param_41.depth;
                        _9789 = param_41.d[0];
                        _9790 = param_41.d[1];
                        _9791 = param_41.d[2];
                        _9464 = param_41.dist;
                        _9824 = param_41.c[0];
                        _9825 = param_41.c[1];
                        _9826 = param_41.c[2];
                        _9466 = param_41.xy;
                        col += _7975;
                    }
                    bool _7982 = _7254 < _3327_g_params.max_refr_depth;
                    bool _7989;
                    if (_7982)
                    {
                        _7989 = _7265 < _3327_g_params.max_total_depth;
                    }
                    else
                    {
                        _7989 = _7982;
                    }
                    [branch]
                    if (_7989)
                    {
                        surface_t _9178 = { _6641, _9132, _9133, _9134, _9135, _6982 };
                        float _9739[3] = { _9719, _9720, _9721 };
                        float _9704[3] = { _9684, _9685, _9686 };
                        float _9669[3] = { _9649, _9650, _9651 };
                        ray_data_t _9413 = { _9669, _9704, _9355, _9739, _9357, _9358, _9359, _9360, _9361 };
                        ray_data_t param_42 = _9413;
                        Sample_RefractiveNode(ray, _9178, base_color, roughness, _6674, _9199, _7239, _7761, _7770, mix_weight, param_42);
                        _9649 = param_42.o[0];
                        _9650 = param_42.o[1];
                        _9651 = param_42.o[2];
                        _9684 = param_42.d[0];
                        _9685 = param_42.d[1];
                        _9686 = param_42.d[2];
                        _9355 = param_42.pdf;
                        _9719 = param_42.c[0];
                        _9720 = param_42.c[1];
                        _9721 = param_42.c[2];
                        _9357 = param_42.ior;
                        _9358 = param_42.cone_width;
                        _9359 = param_42.cone_spread;
                        _9360 = param_42.xy;
                        _9361 = param_42.depth;
                    }
                }
                else
                {
                    [branch]
                    if (_9196 == 3u)
                    {
                        float mis_weight = 1.0f;
                        [branch]
                        if ((_9195 & 1u) != 0u)
                        {
                            float3 _8059 = mul(float4(_6998, 0.0f), _6751.xform).xyz;
                            float _8062 = length(_8059);
                            float _8074 = abs(dot(_6605, _8059 / _8062.xxx));
                            if (_8074 > 0.0f)
                            {
                                float param_43 = ray.pdf;
                                float param_44 = (inter.t * inter.t) / ((0.5f * _8062) * _8074);
                                mis_weight = power_heuristic(param_43, param_44);
                            }
                        }
                        col += (base_color * ((mix_weight * mis_weight) * _9197));
                    }
                    else
                    {
                        [branch]
                        if (_9196 == 6u)
                        {
                            float metallic = clamp(float((_9201 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_9597 != 4294967295u)
                            {
                                metallic *= SampleBilinear(_9597, _6982, int(get_texture_lod(texSize(_9597), _7216))).x;
                            }
                            float specular = clamp(float(_9203 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_9598 != 4294967295u)
                            {
                                specular *= SampleBilinear(_9598, _6982, int(get_texture_lod(texSize(_9598), _7216))).x;
                            }
                            float _8191 = clamp(float(_9204 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8199 = clamp(float((_9204 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8207 = 2.0f * clamp(float(_9200 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float3 _8225 = lerp(1.0f.xxx, tint_color, clamp(float((_9200 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _8207;
                            float3 _8245 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_9203 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                            float _8254 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                            float param_45 = 1.0f;
                            float param_46 = _8254;
                            float _8260 = fresnel_dielectric_cos(param_45, param_46);
                            float _8268 = clamp(float((_9198 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8279 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _8191))) - 1.0f;
                            float param_47 = 1.0f;
                            float param_48 = _8279;
                            float _8285 = fresnel_dielectric_cos(param_47, param_48);
                            float _8300 = mad(roughness - 1.0f, 1.0f - clamp(float((_9202 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                            float _8306;
                            if (_6674)
                            {
                                _8306 = _9199 / _7239;
                            }
                            else
                            {
                                _8306 = _7239 / _9199;
                            }
                            float param_49 = dot(_6605, _9134);
                            float param_50 = 1.0f / _8306;
                            float _8329 = fresnel_dielectric_cos(param_49, param_50);
                            float param_51 = dot(_6605, _9134);
                            float param_52 = _8254;
                            lobe_weights_t _8368 = get_lobe_weights(lerp(_7716, 1.0f, _8207), lum(lerp(_8245, 1.0f.xxx, ((fresnel_dielectric_cos(param_51, param_52) - _8260) / (1.0f - _8260)).xxx)), specular, metallic, clamp(float(_9202 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _8191);
                            [branch]
                            if (_9297 > 0.0f)
                            {
                                light_sample_t _9352 = { _9292, _9293, _9294, _9295, _9296, _9297, _9298, _9299 };
                                surface_t _9185 = { _6641, _9132, _9133, _9134, _9135, _6982 };
                                diff_params_t _9544 = { base_color, _8225, roughness };
                                spec_params_t _9559 = { _8245, roughness, _8254, _8260, _8268 };
                                clearcoat_params_t _9572 = { _8199, _8279, _8285 };
                                transmission_params_t _9587 = { _8300, _9199, _8306, _8329, _6674 };
                                float _9851[3] = { _9824, _9825, _9826 };
                                float _9816[3] = { _9789, _9790, _9791 };
                                float _9781[3] = { _9754, _9755, _9756 };
                                shadow_ray_t _9515 = { _9781, _9462, _9816, _9464, _9851, _9466 };
                                shadow_ray_t param_53 = _9515;
                                float3 _8387 = Evaluate_PrincipledNode(_9352, ray, _9185, _8368, _9544, _9559, _9572, _9587, metallic, _7680, mix_weight, param_53);
                                _9754 = param_53.o[0];
                                _9755 = param_53.o[1];
                                _9756 = param_53.o[2];
                                _9462 = param_53.depth;
                                _9789 = param_53.d[0];
                                _9790 = param_53.d[1];
                                _9791 = param_53.d[2];
                                _9464 = param_53.dist;
                                _9824 = param_53.c[0];
                                _9825 = param_53.c[1];
                                _9826 = param_53.c[2];
                                _9466 = param_53.xy;
                                col += _8387;
                            }
                            surface_t _9192 = { _6641, _9132, _9133, _9134, _9135, _6982 };
                            diff_params_t _9548 = { base_color, _8225, roughness };
                            spec_params_t _9565 = { _8245, roughness, _8254, _8260, _8268 };
                            clearcoat_params_t _9576 = { _8199, _8279, _8285 };
                            transmission_params_t _9593 = { _8300, _9199, _8306, _8329, _6674 };
                            float param_54 = mix_rand;
                            float _9746[3] = { _9719, _9720, _9721 };
                            float _9711[3] = { _9684, _9685, _9686 };
                            float _9676[3] = { _9649, _9650, _9651 };
                            ray_data_t _9432 = { _9676, _9711, _9355, _9746, _9357, _9358, _9359, _9360, _9361 };
                            ray_data_t param_55 = _9432;
                            Sample_PrincipledNode(ray, _9192, _8368, _9548, _9565, _9576, _9593, metallic, _7761, _7770, param_54, mix_weight, param_55);
                            _9649 = param_55.o[0];
                            _9650 = param_55.o[1];
                            _9651 = param_55.o[2];
                            _9684 = param_55.d[0];
                            _9685 = param_55.d[1];
                            _9686 = param_55.d[2];
                            _9355 = param_55.pdf;
                            _9719 = param_55.c[0];
                            _9720 = param_55.c[1];
                            _9721 = param_55.c[2];
                            _9357 = param_55.ior;
                            _9358 = param_55.cone_width;
                            _9359 = param_55.cone_spread;
                            _9360 = param_55.xy;
                            _9361 = param_55.depth;
                        }
                    }
                }
            }
        }
        float _8421 = max(_9719, max(_9720, _9721));
        float _8433;
        if (_7265 > _3327_g_params.min_total_depth)
        {
            _8433 = max(0.0500000007450580596923828125f, 1.0f - _8421);
        }
        else
        {
            _8433 = 0.0f;
        }
        bool _8447 = (frac(asfloat(_3311.Load((_7273 + 6) * 4 + 0)) + _7223) >= _8433) && (_8421 > 0.0f);
        bool _8453;
        if (_8447)
        {
            _8453 = _9355 > 0.0f;
        }
        else
        {
            _8453 = _8447;
        }
        [branch]
        if (_8453)
        {
            float _8457 = _9355;
            float _8458 = min(_8457, 1000000.0f);
            _9355 = _8458;
            float _8461 = 1.0f - _8433;
            float _8463 = _9719;
            float _8464 = _8463 / _8461;
            _9719 = _8464;
            float _8469 = _9720;
            float _8470 = _8469 / _8461;
            _9720 = _8470;
            float _8475 = _9721;
            float _8476 = _8475 / _8461;
            _9721 = _8476;
            uint _8484;
            _8482.InterlockedAdd(0, 1u, _8484);
            _8493.Store(_8484 * 72 + 0, asuint(_9649));
            _8493.Store(_8484 * 72 + 4, asuint(_9650));
            _8493.Store(_8484 * 72 + 8, asuint(_9651));
            _8493.Store(_8484 * 72 + 12, asuint(_9684));
            _8493.Store(_8484 * 72 + 16, asuint(_9685));
            _8493.Store(_8484 * 72 + 20, asuint(_9686));
            _8493.Store(_8484 * 72 + 24, asuint(_8458));
            _8493.Store(_8484 * 72 + 28, asuint(_8464));
            _8493.Store(_8484 * 72 + 32, asuint(_8470));
            _8493.Store(_8484 * 72 + 36, asuint(_8476));
            _8493.Store(_8484 * 72 + 40, asuint(_9357[0]));
            _8493.Store(_8484 * 72 + 44, asuint(_9357[1]));
            _8493.Store(_8484 * 72 + 48, asuint(_9357[2]));
            _8493.Store(_8484 * 72 + 52, asuint(_9357[3]));
            _8493.Store(_8484 * 72 + 56, asuint(_9358));
            _8493.Store(_8484 * 72 + 60, asuint(_9359));
            _8493.Store(_8484 * 72 + 64, uint(_9360));
            _8493.Store(_8484 * 72 + 68, uint(_9361));
        }
        [branch]
        if (max(_9824, max(_9825, _9826)) > 0.0f)
        {
            float3 _8570 = _9294 - float3(_9754, _9755, _9756);
            float _8573 = length(_8570);
            float3 _8577 = _8570 / _8573.xxx;
            float sh_dist = _8573 * _9296;
            if (_9299)
            {
                sh_dist = -sh_dist;
            }
            float _8589 = _8577.x;
            _9789 = _8589;
            float _8592 = _8577.y;
            _9790 = _8592;
            float _8595 = _8577.z;
            _9791 = _8595;
            _9464 = sh_dist;
            uint _8601;
            _8482.InterlockedAdd(8, 1u, _8601);
            _8609.Store(_8601 * 48 + 0, asuint(_9754));
            _8609.Store(_8601 * 48 + 4, asuint(_9755));
            _8609.Store(_8601 * 48 + 8, asuint(_9756));
            _8609.Store(_8601 * 48 + 12, uint(_9462));
            _8609.Store(_8601 * 48 + 16, asuint(_8589));
            _8609.Store(_8601 * 48 + 20, asuint(_8592));
            _8609.Store(_8601 * 48 + 24, asuint(_8595));
            _8609.Store(_8601 * 48 + 28, asuint(sh_dist));
            _8609.Store(_8601 * 48 + 32, asuint(_9824));
            _8609.Store(_8601 * 48 + 36, asuint(_9825));
            _8609.Store(_8601 * 48 + 40, asuint(_9826));
            _8609.Store(_8601 * 48 + 44, uint(_9466));
        }
        _8834 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _8834;
}

void comp_main()
{
    do
    {
        int _8675 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_8675) >= _8482.Load(4))
        {
            break;
        }
        int _8691 = int(_8688.Load(_8675 * 72 + 64));
        int _8698 = int(_8688.Load(_8675 * 72 + 64));
        hit_data_t _8709;
        _8709.mask = int(_8705.Load(_8675 * 24 + 0));
        _8709.obj_index = int(_8705.Load(_8675 * 24 + 4));
        _8709.prim_index = int(_8705.Load(_8675 * 24 + 8));
        _8709.t = asfloat(_8705.Load(_8675 * 24 + 12));
        _8709.u = asfloat(_8705.Load(_8675 * 24 + 16));
        _8709.v = asfloat(_8705.Load(_8675 * 24 + 20));
        ray_data_t _8725;
        [unroll]
        for (int _84ident = 0; _84ident < 3; _84ident++)
        {
            _8725.o[_84ident] = asfloat(_8688.Load(_84ident * 4 + _8675 * 72 + 0));
        }
        [unroll]
        for (int _85ident = 0; _85ident < 3; _85ident++)
        {
            _8725.d[_85ident] = asfloat(_8688.Load(_85ident * 4 + _8675 * 72 + 12));
        }
        _8725.pdf = asfloat(_8688.Load(_8675 * 72 + 24));
        [unroll]
        for (int _86ident = 0; _86ident < 3; _86ident++)
        {
            _8725.c[_86ident] = asfloat(_8688.Load(_86ident * 4 + _8675 * 72 + 28));
        }
        [unroll]
        for (int _87ident = 0; _87ident < 4; _87ident++)
        {
            _8725.ior[_87ident] = asfloat(_8688.Load(_87ident * 4 + _8675 * 72 + 40));
        }
        _8725.cone_width = asfloat(_8688.Load(_8675 * 72 + 56));
        _8725.cone_spread = asfloat(_8688.Load(_8675 * 72 + 60));
        _8725.xy = int(_8688.Load(_8675 * 72 + 64));
        _8725.depth = int(_8688.Load(_8675 * 72 + 68));
        hit_data_t _8928 = { _8709.mask, _8709.obj_index, _8709.prim_index, _8709.t, _8709.u, _8709.v };
        hit_data_t param = _8928;
        float _8977[4] = { _8725.ior[0], _8725.ior[1], _8725.ior[2], _8725.ior[3] };
        float _8968[3] = { _8725.c[0], _8725.c[1], _8725.c[2] };
        float _8961[3] = { _8725.d[0], _8725.d[1], _8725.d[2] };
        float _8954[3] = { _8725.o[0], _8725.o[1], _8725.o[2] };
        ray_data_t _8947 = { _8954, _8961, _8725.pdf, _8968, _8977, _8725.cone_width, _8725.cone_spread, _8725.xy, _8725.depth };
        ray_data_t param_1 = _8947;
        float3 param_2 = 0.0f.xxx;
        float3 param_3 = 0.0f.xxx;
        float3 _8781 = ShadeSurface(param, param_1, param_2, param_3);
        int2 _8795 = int2((_8691 >> 16) & 65535, _8698 & 65535);
        g_out_img[_8795] = float4(min(_8781, _3327_g_params.clamp_val.xxx) + g_out_img[_8795].xyz, 1.0f);
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
