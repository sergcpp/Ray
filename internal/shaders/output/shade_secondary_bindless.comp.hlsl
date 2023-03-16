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
ByteAddressBuffer _6745 : register(t14, space0);
RWByteAddressBuffer _8483 : register(u3, space0);
RWByteAddressBuffer _8494 : register(u1, space0);
RWByteAddressBuffer _8610 : register(u2, space0);
ByteAddressBuffer _8689 : register(t7, space0);
ByteAddressBuffer _8706 : register(t6, space0);
ByteAddressBuffer _8823 : register(t10, space0);
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
    float3 _4666 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 _4673;
    if ((ray.depth & 16777215) != 0)
    {
        _4673 = _3327_g_params.env_col.xyz;
    }
    else
    {
        _4673 = _3327_g_params.back_col.xyz;
    }
    float3 env_col = _4673;
    uint _4689;
    if ((ray.depth & 16777215) != 0)
    {
        _4689 = asuint(_3327_g_params.env_col.w);
    }
    else
    {
        _4689 = asuint(_3327_g_params.back_col.w);
    }
    float _4705;
    if ((ray.depth & 16777215) != 0)
    {
        _4705 = _3327_g_params.env_rotation;
    }
    else
    {
        _4705 = _3327_g_params.back_rotation;
    }
    if (_4689 != 4294967295u)
    {
        env_col *= SampleLatlong_RGBE(_4689, _4666, _4705);
    }
    if (_3327_g_params.env_qtree_levels > 0)
    {
        float param = ray.pdf;
        float param_1 = Evaluate_EnvQTree(_4705, g_env_qtree, _g_env_qtree_sampler, _3327_g_params.env_qtree_levels, _4666);
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
    float3 _4784 = float3(ray.d[0], ray.d[1], ray.d[2]);
    light_t _4798;
    _4798.type_and_param0 = _3347.Load4(((-1) - inter.obj_index) * 64 + 0);
    _4798.param1 = asfloat(_3347.Load4(((-1) - inter.obj_index) * 64 + 16));
    _4798.param2 = asfloat(_3347.Load4(((-1) - inter.obj_index) * 64 + 32));
    _4798.param3 = asfloat(_3347.Load4(((-1) - inter.obj_index) * 64 + 48));
    float3 lcol = asfloat(_4798.type_and_param0.yzw);
    [branch]
    if ((_4798.type_and_param0.x & 128u) != 0u)
    {
        float3 env_col = _3327_g_params.env_col.xyz;
        uint _4825 = asuint(_3327_g_params.env_col.w);
        if (_4825 != 4294967295u)
        {
            env_col *= SampleLatlong_RGBE(_4825, _4784, _3327_g_params.env_rotation);
        }
        lcol *= env_col;
    }
    uint _4843 = _4798.type_and_param0.x & 31u;
    if (_4843 == 0u)
    {
        float param = ray.pdf;
        float param_1 = (inter.t * inter.t) / ((0.5f * _4798.param1.w) * dot(_4784, normalize(_4798.param1.xyz - (float3(ray.o[0], ray.o[1], ray.o[2]) + (_4784 * inter.t)))));
        lcol *= power_heuristic(param, param_1);
        bool _4910 = _4798.param3.x > 0.0f;
        bool _4916;
        if (_4910)
        {
            _4916 = _4798.param3.y > 0.0f;
        }
        else
        {
            _4916 = _4910;
        }
        [branch]
        if (_4916)
        {
            [flatten]
            if (_4798.param3.y > 0.0f)
            {
                lcol *= clamp((_4798.param3.x - acos(clamp(-dot(_4784, _4798.param2.xyz), 0.0f, 1.0f))) / _4798.param3.y, 0.0f, 1.0f);
            }
        }
    }
    else
    {
        if (_4843 == 4u)
        {
            float param_2 = ray.pdf;
            float param_3 = (inter.t * inter.t) / (_4798.param1.w * dot(_4784, normalize(cross(_4798.param2.xyz, _4798.param3.xyz))));
            lcol *= power_heuristic(param_2, param_3);
        }
        else
        {
            if (_4843 == 5u)
            {
                float param_4 = ray.pdf;
                float param_5 = (inter.t * inter.t) / (_4798.param1.w * dot(_4784, normalize(cross(_4798.param2.xyz, _4798.param3.xyz))));
                lcol *= power_heuristic(param_4, param_5);
            }
            else
            {
                if (_4843 == 3u)
                {
                    float param_6 = ray.pdf;
                    float param_7 = (inter.t * inter.t) / (_4798.param1.w * (1.0f - abs(dot(_4784, _4798.param3.xyz))));
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
    float _8835;
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
            _8835 = stack[3];
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
            _8835 = stack[2];
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
            _8835 = stack[1];
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
            _8835 = stack[0];
            break;
        }
        _8835 = default_value;
        break;
    } while(false);
    return _8835;
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
        float4 _9890 = res;
        _9890.x = _1033.x;
        float4 _9892 = _9890;
        _9892.y = _1033.y;
        float4 _9894 = _9892;
        _9894.z = _1033.z;
        res = _9894;
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
    float3 _8840;
    do
    {
        float _1299 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1299)
        {
            _8840 = N;
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
            float _10191 = (-0.5f) / _1339;
            float param_1 = mad(_10191, _1363, 1.0f);
            float _1395 = safe_sqrtf(param_1);
            float param_2 = _1364;
            float _1398 = safe_sqrtf(param_2);
            float2 _1399 = float2(_1395, _1398);
            float param_3 = mad(_10191, _1370, 1.0f);
            float _1404 = safe_sqrtf(param_3);
            float param_4 = _1371;
            float _1407 = safe_sqrtf(param_4);
            float2 _1408 = float2(_1404, _1407);
            float _10193 = -_1327;
            float _1424 = mad(2.0f * mad(_1395, _1323, _1398 * _1327), _1398, _10193);
            float _1440 = mad(2.0f * mad(_1404, _1323, _1407 * _1327), _1407, _10193);
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
                _8840 = Ng;
                break;
            }
            float _1477 = valid1 ? _1364 : _1371;
            float param_5 = 1.0f - _1477;
            float param_6 = _1477;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _8840 = (_1319 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _8840;
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
    float3 _8865;
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
            _8865 = N;
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
        _8865 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _8865;
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
            float2 _9877 = origin;
            _9877.x = origin.x + _step;
            origin = _9877;
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
            float2 _9880 = origin;
            _9880.y = origin.y + _step;
            origin = _9880;
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
        float3 _9957 = sampled_dir;
        float3 _3474 = ((param * _9957.x) + (param_1 * _9957.y)) + (_3431 * _9957.z);
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
    float3 _8845;
    do
    {
        float3 param = -float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param_1 = surf.N;
        float3 param_2 = ls.L;
        float param_3 = roughness;
        float3 param_4 = base_color;
        float4 _5113 = Evaluate_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_5 = ls.pdf;
            float param_6 = _5113.w;
            mis_weight = power_heuristic(param_5, param_6);
        }
        float3 _5136 = (ls.col * _5113.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8845 = _5136;
            break;
        }
        float3 param_7 = surf.P;
        float3 param_8 = surf.plane_N;
        float3 _5148 = offset_ray(param_7, param_8);
        sh_r.o[0] = _5148.x;
        sh_r.o[1] = _5148.y;
        sh_r.o[2] = _5148.z;
        sh_r.c[0] = ray.c[0] * _5136.x;
        sh_r.c[1] = ray.c[1] * _5136.y;
        sh_r.c[2] = ray.c[2] * _5136.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8845 = 0.0f.xxx;
        break;
    } while(false);
    return _8845;
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
    float4 _5399 = Sample_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8);
    new_ray.depth = ray.depth + 1;
    float3 param_9 = surf.P;
    float3 param_10 = surf.plane_N;
    float3 _5409 = offset_ray(param_9, param_10);
    new_ray.o[0] = _5409.x;
    new_ray.o[1] = _5409.y;
    new_ray.o[2] = _5409.z;
    new_ray.d[0] = param_8.x;
    new_ray.d[1] = param_8.y;
    new_ray.d[2] = param_8.z;
    new_ray.c[0] = ((ray.c[0] * _5399.x) * mix_weight) / _5399.w;
    new_ray.c[1] = ((ray.c[1] * _5399.y) * mix_weight) / _5399.w;
    new_ray.c[2] = ((ray.c[2] * _5399.z) * mix_weight) / _5399.w;
    new_ray.pdf = _5399.w;
}

float3 tangent_from_world(float3 T, float3 B, float3 N, float3 V)
{
    return float3(dot(V, T), dot(V, B), dot(V, N));
}

float D_GGX(float3 H, float alpha_x, float alpha_y)
{
    float _8898;
    do
    {
        if (H.z == 0.0f)
        {
            _8898 = 0.0f;
            break;
        }
        float _2032 = (-H.x) / (H.z * alpha_x);
        float _2038 = (-H.y) / (H.z * alpha_y);
        float _2047 = mad(_2038, _2038, mad(_2032, _2032, 1.0f));
        _8898 = 1.0f / (((((_2047 * _2047) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _8898;
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
    float3 _8850;
    do
    {
        float3 _5184 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5184;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - _5184);
        float _5222 = roughness * roughness;
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = _5222;
        float param_16 = _5222;
        float param_17 = spec_ior;
        float param_18 = spec_F0;
        float3 param_19 = base_color;
        float4 _5237 = Evaluate_GGXSpecular_BSDF(param_12, param_13, param_14, param_15, param_16, param_17, param_18, param_19);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_20 = ls.pdf;
            float param_21 = _5237.w;
            mis_weight = power_heuristic(param_20, param_21);
        }
        float3 _5260 = (ls.col * _5237.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8850 = _5260;
            break;
        }
        float3 param_22 = surf.P;
        float3 param_23 = surf.plane_N;
        float3 _5272 = offset_ray(param_22, param_23);
        sh_r.o[0] = _5272.x;
        sh_r.o[1] = _5272.y;
        sh_r.o[2] = _5272.z;
        sh_r.c[0] = ray.c[0] * _5260.x;
        sh_r.c[1] = ray.c[1] * _5260.y;
        sh_r.c[2] = ray.c[2] * _5260.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8850 = 0.0f.xxx;
        break;
    } while(false);
    return _8850;
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
    float4 _8870;
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
            _8870 = float4(_2687.x * 1000000.0f, _2687.y * 1000000.0f, _2687.z * 1000000.0f, 1000000.0f);
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
        _8870 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _8870;
}

void Sample_GlossyNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 param_4;
    float4 _5319 = Sample_GGXSpecular_BSDF(param, param_1, param_2, param_3, roughness, 0.0f, spec_ior, spec_F0, base_color, rand_u, rand_v, param_4);
    new_ray.depth = ray.depth + 256;
    float3 param_5 = surf.P;
    float3 param_6 = surf.plane_N;
    float3 _5330 = offset_ray(param_5, param_6);
    new_ray.o[0] = _5330.x;
    new_ray.o[1] = _5330.y;
    new_ray.o[2] = _5330.z;
    new_ray.d[0] = param_4.x;
    new_ray.d[1] = param_4.y;
    new_ray.d[2] = param_4.z;
    new_ray.c[0] = ((ray.c[0] * _5319.x) * mix_weight) / _5319.w;
    new_ray.c[1] = ((ray.c[1] * _5319.y) * mix_weight) / _5319.w;
    new_ray.c[2] = ((ray.c[2] * _5319.z) * mix_weight) / _5319.w;
    new_ray.pdf = _5319.w;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _8875;
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
            _8875 = 0.0f.xxxx;
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
        _8875 = float4(refr_col * (((((_2970 * _2986) * _2978) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3006) / view_dir_ts.z), (((_2970 * _2978) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3006) / view_dir_ts.z);
        break;
    } while(false);
    return _8875;
}

float3 Evaluate_RefractiveNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness2, float eta, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _8855;
    do
    {
        float3 _5462 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5462;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - (_5462 * eta));
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = roughness2;
        float param_16 = eta;
        float3 param_17 = base_color;
        float4 _5510 = Evaluate_GGXRefraction_BSDF(param_12, param_13, param_14, param_15, param_16, param_17);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_18 = ls.pdf;
            float param_19 = _5510.w;
            mis_weight = power_heuristic(param_18, param_19);
        }
        float3 _5533 = (ls.col * _5510.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8855 = _5533;
            break;
        }
        float3 param_20 = surf.P;
        float3 param_21 = -surf.plane_N;
        float3 _5546 = offset_ray(param_20, param_21);
        sh_r.o[0] = _5546.x;
        sh_r.o[1] = _5546.y;
        sh_r.o[2] = _5546.z;
        sh_r.c[0] = ray.c[0] * _5533.x;
        sh_r.c[1] = ray.c[1] * _5533.y;
        sh_r.c[2] = ray.c[2] * _5533.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8855 = 0.0f.xxx;
        break;
    } while(false);
    return _8855;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _8880;
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
                _8880 = 0.0f.xxxx;
                break;
            }
            float _3083 = mad(eta, _3061, -sqrt(_3071));
            out_V = float4(normalize((I * eta) + (N * _3083)), _3083);
            _8880 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
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
            _8880 = 0.0f.xxxx;
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
        _8880 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _8880;
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
    float _8888;
    do
    {
        if (stack[3] > 0.0f)
        {
            float param = stack[3];
            float param_1 = -1.0f;
            float _2138 = exchange(param, param_1);
            stack[3] = param;
            _8888 = _2138;
            break;
        }
        if (stack[2] > 0.0f)
        {
            float param_2 = stack[2];
            float param_3 = -1.0f;
            float _2151 = exchange(param_2, param_3);
            stack[2] = param_2;
            _8888 = _2151;
            break;
        }
        if (stack[1] > 0.0f)
        {
            float param_4 = stack[1];
            float param_5 = -1.0f;
            float _2164 = exchange(param_4, param_5);
            stack[1] = param_4;
            _8888 = _2164;
            break;
        }
        if (stack[0] > 0.0f)
        {
            float param_6 = stack[0];
            float param_7 = -1.0f;
            float _2177 = exchange(param_6, param_7);
            stack[0] = param_6;
            _8888 = _2177;
            break;
        }
        _8888 = default_value;
        break;
    } while(false);
    return _8888;
}

void Sample_RefractiveNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, bool is_backfacing, float int_ior, float ext_ior, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float _5583;
    if (is_backfacing)
    {
        _5583 = int_ior / ext_ior;
    }
    else
    {
        _5583 = ext_ior / int_ior;
    }
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float param_4 = roughness;
    float param_5 = _5583;
    float3 param_6 = base_color;
    float param_7 = rand_u;
    float param_8 = rand_v;
    float4 param_9;
    float4 _5607 = Sample_GGXRefraction_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9);
    new_ray.depth = ray.depth + 65536;
    new_ray.c[0] = ((ray.c[0] * _5607.x) * mix_weight) / _5607.w;
    new_ray.c[1] = ((ray.c[1] * _5607.y) * mix_weight) / _5607.w;
    new_ray.c[2] = ((ray.c[2] * _5607.z) * mix_weight) / _5607.w;
    new_ray.pdf = _5607.w;
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
        float _5663 = pop_ior_stack(param_11, param_12);
        new_ray.ior = param_11;
    }
    float3 param_13 = surf.P;
    float3 param_14 = -surf.plane_N;
    float3 _5672 = offset_ray(param_13, param_14);
    new_ray.o[0] = _5672.x;
    new_ray.o[1] = _5672.y;
    new_ray.o[2] = _5672.z;
    new_ray.d[0] = param_9.x;
    new_ray.d[1] = param_9.y;
    new_ray.d[2] = param_9.z;
}

lobe_weights_t get_lobe_weights(float base_color_lum, float spec_color_lum, float specular, float metallic, float transmission, float clearcoat)
{
    float _1502 = 1.0f - metallic;
    float _8974 = (base_color_lum * _1502) * (1.0f - transmission);
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
    float _8975 = _1513;
    float _1523 = 0.25f * clearcoat;
    float _8976 = _1523 * _1502;
    float _8977 = _1509 * base_color_lum;
    float _1532 = _8974;
    float _1541 = mad(_1509, base_color_lum, mad(_1523, _1502, _1532 + _1513));
    if (_1541 != 0.0f)
    {
        _8974 /= _1541;
        _8975 /= _1541;
        _8976 /= _1541;
        _8977 /= _1541;
    }
    lobe_weights_t _8982 = { _8974, _8975, _8976, _8977 };
    return _8982;
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
    float _8903;
    do
    {
        float _2258 = dot(N, L);
        if (_2258 <= 0.0f)
        {
            _8903 = 0.0f;
            break;
        }
        float param = _2258;
        float param_1 = dot(N, V);
        float _2279 = dot(L, H);
        float _2287 = mad((2.0f * _2279) * _2279, roughness, 0.5f);
        _8903 = lerp(1.0f, _2287, schlick_weight(param)) * lerp(1.0f, _2287, schlick_weight(param_1));
        break;
    } while(false);
    return _8903;
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
    float _8908;
    do
    {
        if (a >= 1.0f)
        {
            _8908 = 0.3183098733425140380859375f;
            break;
        }
        float _2006 = mad(a, a, -1.0f);
        _8908 = _2006 / ((3.1415927410125732421875f * log(a * a)) * mad(_2006 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _8908;
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
    float3 _8860;
    do
    {
        float3 _5695 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 lcol = 0.0f.xxx;
        float bsdf_pdf = 0.0f;
        bool _5700 = N_dot_L > 0.0f;
        [branch]
        if ((lobe_weights.diffuse > 1.0000000116860974230803549289703e-07f) && _5700)
        {
            float3 param = -_5695;
            float3 param_1 = surf.N;
            float3 param_2 = ls.L;
            float param_3 = diff.roughness;
            float3 param_4 = diff.base_color;
            float3 param_5 = diff.sheen_color;
            bool param_6 = false;
            float4 _5719 = Evaluate_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6);
            bsdf_pdf = mad(lobe_weights.diffuse, _5719.w, bsdf_pdf);
            lcol += (((ls.col * N_dot_L) * (_5719 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * ls.pdf).xxx);
        }
        float3 H;
        [flatten]
        if (_5700)
        {
            H = normalize(ls.L - _5695);
        }
        else
        {
            H = normalize(ls.L - (_5695 * trans.eta));
        }
        float _5758 = spec.roughness * spec.roughness;
        float _5763 = sqrt(mad(-0.89999997615814208984375f, spec.anisotropy, 1.0f));
        float _5767 = _5758 / _5763;
        float _5771 = _5758 * _5763;
        float3 param_7 = surf.T;
        float3 param_8 = surf.B;
        float3 param_9 = surf.N;
        float3 param_10 = -_5695;
        float3 _5782 = tangent_from_world(param_7, param_8, param_9, param_10);
        float3 param_11 = surf.T;
        float3 param_12 = surf.B;
        float3 param_13 = surf.N;
        float3 param_14 = ls.L;
        float3 _5792 = tangent_from_world(param_11, param_12, param_13, param_14);
        float3 param_15 = surf.T;
        float3 param_16 = surf.B;
        float3 param_17 = surf.N;
        float3 param_18 = H;
        float3 _5802 = tangent_from_world(param_15, param_16, param_17, param_18);
        bool _5804 = lobe_weights.specular > 0.0f;
        bool _5811;
        if (_5804)
        {
            _5811 = (_5767 * _5771) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _5811 = _5804;
        }
        [branch]
        if (_5811 && _5700)
        {
            float3 param_19 = _5782;
            float3 param_20 = _5802;
            float3 param_21 = _5792;
            float param_22 = _5767;
            float param_23 = _5771;
            float param_24 = spec.ior;
            float param_25 = spec.F0;
            float3 param_26 = spec.tmp_col;
            float4 _5833 = Evaluate_GGXSpecular_BSDF(param_19, param_20, param_21, param_22, param_23, param_24, param_25, param_26);
            bsdf_pdf = mad(lobe_weights.specular, _5833.w, bsdf_pdf);
            lcol += ((ls.col * _5833.xyz) / ls.pdf.xxx);
        }
        float _5852 = coat.roughness * coat.roughness;
        bool _5854 = lobe_weights.clearcoat > 0.0f;
        bool _5861;
        if (_5854)
        {
            _5861 = (_5852 * _5852) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _5861 = _5854;
        }
        [branch]
        if (_5861 && _5700)
        {
            float3 param_27 = _5782;
            float3 param_28 = _5802;
            float3 param_29 = _5792;
            float param_30 = _5852;
            float param_31 = coat.ior;
            float param_32 = coat.F0;
            float4 _5879 = Evaluate_PrincipledClearcoat_BSDF(param_27, param_28, param_29, param_30, param_31, param_32);
            bsdf_pdf = mad(lobe_weights.clearcoat, _5879.w, bsdf_pdf);
            lcol += (((ls.col * 0.25f) * _5879.xyz) / ls.pdf.xxx);
        }
        [branch]
        if (lobe_weights.refraction > 0.0f)
        {
            bool _5901 = trans.fresnel != 0.0f;
            bool _5908;
            if (_5901)
            {
                _5908 = (_5758 * _5758) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _5908 = _5901;
            }
            [branch]
            if (_5908 && _5700)
            {
                float3 param_33 = _5782;
                float3 param_34 = _5802;
                float3 param_35 = _5792;
                float param_36 = _5758;
                float param_37 = _5758;
                float param_38 = 1.0f;
                float param_39 = 0.0f;
                float3 param_40 = 1.0f.xxx;
                float4 _5927 = Evaluate_GGXSpecular_BSDF(param_33, param_34, param_35, param_36, param_37, param_38, param_39, param_40);
                bsdf_pdf = mad(lobe_weights.refraction * trans.fresnel, _5927.w, bsdf_pdf);
                lcol += ((ls.col * _5927.xyz) * (trans.fresnel / ls.pdf));
            }
            float _5949 = trans.roughness * trans.roughness;
            bool _5951 = trans.fresnel != 1.0f;
            bool _5958;
            if (_5951)
            {
                _5958 = (_5949 * _5949) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _5958 = _5951;
            }
            [branch]
            if (_5958 && (N_dot_L < 0.0f))
            {
                float3 param_41 = _5782;
                float3 param_42 = _5802;
                float3 param_43 = _5792;
                float param_44 = _5949;
                float param_45 = trans.eta;
                float3 param_46 = diff.base_color;
                float4 _5976 = Evaluate_GGXRefraction_BSDF(param_41, param_42, param_43, param_44, param_45, param_46);
                float _5979 = 1.0f - trans.fresnel;
                bsdf_pdf = mad(lobe_weights.refraction * _5979, _5976.w, bsdf_pdf);
                lcol += ((ls.col * _5976.xyz) * (_5979 / ls.pdf));
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
            _8860 = lcol;
            break;
        }
        float3 _6019;
        if (N_dot_L < 0.0f)
        {
            _6019 = -surf.plane_N;
        }
        else
        {
            _6019 = surf.plane_N;
        }
        float3 param_49 = surf.P;
        float3 param_50 = _6019;
        float3 _6030 = offset_ray(param_49, param_50);
        sh_r.o[0] = _6030.x;
        sh_r.o[1] = _6030.y;
        sh_r.o[2] = _6030.z;
        sh_r.c[0] = ray.c[0] * lcol.x;
        sh_r.c[1] = ray.c[1] * lcol.y;
        sh_r.c[2] = ray.c[2] * lcol.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8860 = 0.0f.xxx;
        break;
    } while(false);
    return _8860;
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
    float4 _8893;
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
            _8893 = float4(_2887, _2887, _2887, 1000000.0f);
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
        _8893 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _8893;
}

void Sample_PrincipledNode(ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float rand_u, float rand_v, inout float mix_rand, float mix_weight, inout ray_data_t new_ray)
{
    float3 _6065 = float3(ray.d[0], ray.d[1], ray.d[2]);
    int _6069 = ray.depth & 255;
    int _6073 = (ray.depth >> 8) & 255;
    int _6077 = (ray.depth >> 16) & 255;
    int _6088 = (_6069 + _6073) + _6077;
    [branch]
    if (mix_rand < lobe_weights.diffuse)
    {
        bool _6097 = _6069 < _3327_g_params.max_diff_depth;
        bool _6104;
        if (_6097)
        {
            _6104 = _6088 < _3327_g_params.max_total_depth;
        }
        else
        {
            _6104 = _6097;
        }
        if (_6104)
        {
            float3 param = surf.T;
            float3 param_1 = surf.B;
            float3 param_2 = surf.N;
            float3 param_3 = _6065;
            float param_4 = diff.roughness;
            float3 param_5 = diff.base_color;
            float3 param_6 = diff.sheen_color;
            bool param_7 = false;
            float param_8 = rand_u;
            float param_9 = rand_v;
            float3 param_10;
            float4 _6127 = Sample_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9, param_10);
            float3 _6132 = _6127.xyz * (1.0f - metallic);
            new_ray.depth = ray.depth + 1;
            float3 param_11 = surf.P;
            float3 param_12 = surf.plane_N;
            float3 _6147 = offset_ray(param_11, param_12);
            new_ray.o[0] = _6147.x;
            new_ray.o[1] = _6147.y;
            new_ray.o[2] = _6147.z;
            new_ray.d[0] = param_10.x;
            new_ray.d[1] = param_10.y;
            new_ray.d[2] = param_10.z;
            new_ray.c[0] = ((ray.c[0] * _6132.x) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[1] = ((ray.c[1] * _6132.y) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[2] = ((ray.c[2] * _6132.z) * mix_weight) / lobe_weights.diffuse;
            new_ray.pdf = _6127.w;
        }
    }
    else
    {
        float _6197 = lobe_weights.diffuse + lobe_weights.specular;
        [branch]
        if (mix_rand < _6197)
        {
            bool _6204 = _6073 < _3327_g_params.max_spec_depth;
            bool _6211;
            if (_6204)
            {
                _6211 = _6088 < _3327_g_params.max_total_depth;
            }
            else
            {
                _6211 = _6204;
            }
            if (_6211)
            {
                float3 param_13 = surf.T;
                float3 param_14 = surf.B;
                float3 param_15 = surf.N;
                float3 param_16 = _6065;
                float3 param_17;
                float4 _6230 = Sample_GGXSpecular_BSDF(param_13, param_14, param_15, param_16, spec.roughness, spec.anisotropy, spec.ior, spec.F0, spec.tmp_col, rand_u, rand_v, param_17);
                float _6235 = _6230.w * lobe_weights.specular;
                new_ray.depth = ray.depth + 256;
                new_ray.c[0] = ((ray.c[0] * _6230.x) * mix_weight) / _6235;
                new_ray.c[1] = ((ray.c[1] * _6230.y) * mix_weight) / _6235;
                new_ray.c[2] = ((ray.c[2] * _6230.z) * mix_weight) / _6235;
                new_ray.pdf = _6235;
                float3 param_18 = surf.P;
                float3 param_19 = surf.plane_N;
                float3 _6275 = offset_ray(param_18, param_19);
                new_ray.o[0] = _6275.x;
                new_ray.o[1] = _6275.y;
                new_ray.o[2] = _6275.z;
                new_ray.d[0] = param_17.x;
                new_ray.d[1] = param_17.y;
                new_ray.d[2] = param_17.z;
            }
        }
        else
        {
            float _6300 = _6197 + lobe_weights.clearcoat;
            [branch]
            if (mix_rand < _6300)
            {
                bool _6307 = _6073 < _3327_g_params.max_spec_depth;
                bool _6314;
                if (_6307)
                {
                    _6314 = _6088 < _3327_g_params.max_total_depth;
                }
                else
                {
                    _6314 = _6307;
                }
                if (_6314)
                {
                    float3 param_20 = surf.T;
                    float3 param_21 = surf.B;
                    float3 param_22 = surf.N;
                    float3 param_23 = _6065;
                    float param_24 = coat.roughness * coat.roughness;
                    float param_25 = coat.ior;
                    float param_26 = coat.F0;
                    float param_27 = rand_u;
                    float param_28 = rand_v;
                    float3 param_29;
                    float4 _6338 = Sample_PrincipledClearcoat_BSDF(param_20, param_21, param_22, param_23, param_24, param_25, param_26, param_27, param_28, param_29);
                    float _6343 = _6338.w * lobe_weights.clearcoat;
                    new_ray.depth = ray.depth + 256;
                    new_ray.c[0] = (((0.25f * ray.c[0]) * _6338.x) * mix_weight) / _6343;
                    new_ray.c[1] = (((0.25f * ray.c[1]) * _6338.y) * mix_weight) / _6343;
                    new_ray.c[2] = (((0.25f * ray.c[2]) * _6338.z) * mix_weight) / _6343;
                    new_ray.pdf = _6343;
                    float3 param_30 = surf.P;
                    float3 param_31 = surf.plane_N;
                    float3 _6386 = offset_ray(param_30, param_31);
                    new_ray.o[0] = _6386.x;
                    new_ray.o[1] = _6386.y;
                    new_ray.o[2] = _6386.z;
                    new_ray.d[0] = param_29.x;
                    new_ray.d[1] = param_29.y;
                    new_ray.d[2] = param_29.z;
                }
            }
            else
            {
                bool _6408 = mix_rand >= trans.fresnel;
                bool _6415;
                if (_6408)
                {
                    _6415 = _6077 < _3327_g_params.max_refr_depth;
                }
                else
                {
                    _6415 = _6408;
                }
                bool _6429;
                if (!_6415)
                {
                    bool _6421 = mix_rand < trans.fresnel;
                    bool _6428;
                    if (_6421)
                    {
                        _6428 = _6073 < _3327_g_params.max_spec_depth;
                    }
                    else
                    {
                        _6428 = _6421;
                    }
                    _6429 = _6428;
                }
                else
                {
                    _6429 = _6415;
                }
                bool _6436;
                if (_6429)
                {
                    _6436 = _6088 < _3327_g_params.max_total_depth;
                }
                else
                {
                    _6436 = _6429;
                }
                [branch]
                if (_6436)
                {
                    mix_rand -= _6300;
                    mix_rand /= lobe_weights.refraction;
                    float4 F;
                    float3 V;
                    [branch]
                    if (mix_rand < trans.fresnel)
                    {
                        float3 param_32 = surf.T;
                        float3 param_33 = surf.B;
                        float3 param_34 = surf.N;
                        float3 param_35 = _6065;
                        float3 param_36;
                        float4 _6466 = Sample_GGXSpecular_BSDF(param_32, param_33, param_34, param_35, spec.roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, rand_u, rand_v, param_36);
                        V = param_36;
                        F = _6466;
                        new_ray.depth = ray.depth + 256;
                        float3 param_37 = surf.P;
                        float3 param_38 = surf.plane_N;
                        float3 _6476 = offset_ray(param_37, param_38);
                        new_ray.o[0] = _6476.x;
                        new_ray.o[1] = _6476.y;
                        new_ray.o[2] = _6476.z;
                    }
                    else
                    {
                        float3 param_39 = surf.T;
                        float3 param_40 = surf.B;
                        float3 param_41 = surf.N;
                        float3 param_42 = _6065;
                        float param_43 = trans.roughness;
                        float param_44 = trans.eta;
                        float3 param_45 = diff.base_color;
                        float param_46 = rand_u;
                        float param_47 = rand_v;
                        float4 param_48;
                        float4 _6505 = Sample_GGXRefraction_BSDF(param_39, param_40, param_41, param_42, param_43, param_44, param_45, param_46, param_47, param_48);
                        F = _6505;
                        V = param_48.xyz;
                        new_ray.depth = ray.depth + 65536;
                        float3 param_49 = surf.P;
                        float3 param_50 = -surf.plane_N;
                        float3 _6518 = offset_ray(param_49, param_50);
                        new_ray.o[0] = _6518.x;
                        new_ray.o[1] = _6518.y;
                        new_ray.o[2] = _6518.z;
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
                            float _6544 = pop_ior_stack(param_52, param_53);
                            new_ray.ior = param_52;
                        }
                    }
                    float4 _10141 = F;
                    float _6550 = _10141.w * lobe_weights.refraction;
                    float4 _10143 = _10141;
                    _10143.w = _6550;
                    F = _10143;
                    new_ray.c[0] = ((ray.c[0] * _10141.x) * mix_weight) / _6550;
                    new_ray.c[1] = ((ray.c[1] * _10141.y) * mix_weight) / _6550;
                    new_ray.c[2] = ((ray.c[2] * _10141.z) * mix_weight) / _6550;
                    new_ray.pdf = _6550;
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
    float3 _8830;
    do
    {
        float3 _6606 = float3(ray.d[0], ray.d[1], ray.d[2]);
        [branch]
        if (inter.mask == 0)
        {
            ray_data_t param = ray;
            float3 _6615 = Evaluate_EnvColor(param);
            _8830 = float3(ray.c[0] * _6615.x, ray.c[1] * _6615.y, ray.c[2] * _6615.z);
            break;
        }
        float3 _6642 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_6606 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            ray_data_t param_1 = ray;
            hit_data_t param_2 = inter;
            float3 _6654 = Evaluate_LightColor(param_1, param_2);
            _8830 = float3(ray.c[0] * _6654.x, ray.c[1] * _6654.y, ray.c[2] * _6654.z);
            break;
        }
        bool _6675 = inter.prim_index < 0;
        int _6678;
        if (_6675)
        {
            _6678 = (-1) - inter.prim_index;
        }
        else
        {
            _6678 = inter.prim_index;
        }
        uint _6689 = uint(_6678);
        material_t _6697;
        [unroll]
        for (int _61ident = 0; _61ident < 5; _61ident++)
        {
            _6697.textures[_61ident] = _4455.Load(_61ident * 4 + ((_4459.Load(_6689 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
        }
        [unroll]
        for (int _62ident = 0; _62ident < 3; _62ident++)
        {
            _6697.base_color[_62ident] = asfloat(_4455.Load(_62ident * 4 + ((_4459.Load(_6689 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
        }
        _6697.flags = _4455.Load(((_4459.Load(_6689 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
        _6697.type = _4455.Load(((_4459.Load(_6689 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
        _6697.tangent_rotation_or_strength = asfloat(_4455.Load(((_4459.Load(_6689 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
        _6697.roughness_and_anisotropic = _4455.Load(((_4459.Load(_6689 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
        _6697.ior = asfloat(_4455.Load(((_4459.Load(_6689 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
        _6697.sheen_and_sheen_tint = _4455.Load(((_4459.Load(_6689 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
        _6697.tint_and_metallic = _4455.Load(((_4459.Load(_6689 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
        _6697.transmission_and_transmission_roughness = _4455.Load(((_4459.Load(_6689 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
        _6697.specular_and_specular_tint = _4455.Load(((_4459.Load(_6689 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
        _6697.clearcoat_and_clearcoat_roughness = _4455.Load(((_4459.Load(_6689 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
        _6697.normal_map_strength_unorm = _4455.Load(((_4459.Load(_6689 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
        uint _9590 = _6697.textures[0];
        uint _9591 = _6697.textures[1];
        uint _9592 = _6697.textures[2];
        uint _9593 = _6697.textures[3];
        uint _9594 = _6697.textures[4];
        float _9595 = _6697.base_color[0];
        float _9596 = _6697.base_color[1];
        float _9597 = _6697.base_color[2];
        uint _9191 = _6697.flags;
        uint _9192 = _6697.type;
        float _9193 = _6697.tangent_rotation_or_strength;
        uint _9194 = _6697.roughness_and_anisotropic;
        float _9195 = _6697.ior;
        uint _9196 = _6697.sheen_and_sheen_tint;
        uint _9197 = _6697.tint_and_metallic;
        uint _9198 = _6697.transmission_and_transmission_roughness;
        uint _9199 = _6697.specular_and_specular_tint;
        uint _9200 = _6697.clearcoat_and_clearcoat_roughness;
        uint _9201 = _6697.normal_map_strength_unorm;
        transform_t _6752;
        _6752.xform = asfloat(uint4x4(_4102.Load4(asuint(asfloat(_6745.Load(inter.obj_index * 32 + 12))) * 128 + 0), _4102.Load4(asuint(asfloat(_6745.Load(inter.obj_index * 32 + 12))) * 128 + 16), _4102.Load4(asuint(asfloat(_6745.Load(inter.obj_index * 32 + 12))) * 128 + 32), _4102.Load4(asuint(asfloat(_6745.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _6752.inv_xform = asfloat(uint4x4(_4102.Load4(asuint(asfloat(_6745.Load(inter.obj_index * 32 + 12))) * 128 + 64), _4102.Load4(asuint(asfloat(_6745.Load(inter.obj_index * 32 + 12))) * 128 + 80), _4102.Load4(asuint(asfloat(_6745.Load(inter.obj_index * 32 + 12))) * 128 + 96), _4102.Load4(asuint(asfloat(_6745.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _6759 = _6689 * 3u;
        vertex_t _6764;
        [unroll]
        for (int _63ident = 0; _63ident < 3; _63ident++)
        {
            _6764.p[_63ident] = asfloat(_4127.Load(_63ident * 4 + _4131.Load(_6759 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _64ident = 0; _64ident < 3; _64ident++)
        {
            _6764.n[_64ident] = asfloat(_4127.Load(_64ident * 4 + _4131.Load(_6759 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _65ident = 0; _65ident < 3; _65ident++)
        {
            _6764.b[_65ident] = asfloat(_4127.Load(_65ident * 4 + _4131.Load(_6759 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _66ident = 0; _66ident < 2; _66ident++)
        {
            [unroll]
            for (int _67ident = 0; _67ident < 2; _67ident++)
            {
                _6764.t[_66ident][_67ident] = asfloat(_4127.Load(_67ident * 4 + _66ident * 8 + _4131.Load(_6759 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _6810;
        [unroll]
        for (int _68ident = 0; _68ident < 3; _68ident++)
        {
            _6810.p[_68ident] = asfloat(_4127.Load(_68ident * 4 + _4131.Load((_6759 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _69ident = 0; _69ident < 3; _69ident++)
        {
            _6810.n[_69ident] = asfloat(_4127.Load(_69ident * 4 + _4131.Load((_6759 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _70ident = 0; _70ident < 3; _70ident++)
        {
            _6810.b[_70ident] = asfloat(_4127.Load(_70ident * 4 + _4131.Load((_6759 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _71ident = 0; _71ident < 2; _71ident++)
        {
            [unroll]
            for (int _72ident = 0; _72ident < 2; _72ident++)
            {
                _6810.t[_71ident][_72ident] = asfloat(_4127.Load(_72ident * 4 + _71ident * 8 + _4131.Load((_6759 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _6856;
        [unroll]
        for (int _73ident = 0; _73ident < 3; _73ident++)
        {
            _6856.p[_73ident] = asfloat(_4127.Load(_73ident * 4 + _4131.Load((_6759 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _74ident = 0; _74ident < 3; _74ident++)
        {
            _6856.n[_74ident] = asfloat(_4127.Load(_74ident * 4 + _4131.Load((_6759 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _75ident = 0; _75ident < 3; _75ident++)
        {
            _6856.b[_75ident] = asfloat(_4127.Load(_75ident * 4 + _4131.Load((_6759 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _76ident = 0; _76ident < 2; _76ident++)
        {
            [unroll]
            for (int _77ident = 0; _77ident < 2; _77ident++)
            {
                _6856.t[_76ident][_77ident] = asfloat(_4127.Load(_77ident * 4 + _76ident * 8 + _4131.Load((_6759 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _6902 = float3(_6764.p[0], _6764.p[1], _6764.p[2]);
        float3 _6910 = float3(_6810.p[0], _6810.p[1], _6810.p[2]);
        float3 _6918 = float3(_6856.p[0], _6856.p[1], _6856.p[2]);
        float _6925 = (1.0f - inter.u) - inter.v;
        float3 _6957 = normalize(((float3(_6764.n[0], _6764.n[1], _6764.n[2]) * _6925) + (float3(_6810.n[0], _6810.n[1], _6810.n[2]) * inter.u)) + (float3(_6856.n[0], _6856.n[1], _6856.n[2]) * inter.v));
        float3 _9130 = _6957;
        float2 _6983 = ((float2(_6764.t[0][0], _6764.t[0][1]) * _6925) + (float2(_6810.t[0][0], _6810.t[0][1]) * inter.u)) + (float2(_6856.t[0][0], _6856.t[0][1]) * inter.v);
        float3 _6999 = cross(_6910 - _6902, _6918 - _6902);
        float _7004 = length(_6999);
        float3 _9131 = _6999 / _7004.xxx;
        float3 _7041 = ((float3(_6764.b[0], _6764.b[1], _6764.b[2]) * _6925) + (float3(_6810.b[0], _6810.b[1], _6810.b[2]) * inter.u)) + (float3(_6856.b[0], _6856.b[1], _6856.b[2]) * inter.v);
        float3 _9129 = _7041;
        float3 _9128 = cross(_7041, _6957);
        if (_6675)
        {
            if ((_4459.Load(_6689 * 4 + 0) & 65535u) == 65535u)
            {
                _8830 = 0.0f.xxx;
                break;
            }
            material_t _7067;
            [unroll]
            for (int _78ident = 0; _78ident < 5; _78ident++)
            {
                _7067.textures[_78ident] = _4455.Load(_78ident * 4 + (_4459.Load(_6689 * 4 + 0) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _79ident = 0; _79ident < 3; _79ident++)
            {
                _7067.base_color[_79ident] = asfloat(_4455.Load(_79ident * 4 + (_4459.Load(_6689 * 4 + 0) & 16383u) * 76 + 20));
            }
            _7067.flags = _4455.Load((_4459.Load(_6689 * 4 + 0) & 16383u) * 76 + 32);
            _7067.type = _4455.Load((_4459.Load(_6689 * 4 + 0) & 16383u) * 76 + 36);
            _7067.tangent_rotation_or_strength = asfloat(_4455.Load((_4459.Load(_6689 * 4 + 0) & 16383u) * 76 + 40));
            _7067.roughness_and_anisotropic = _4455.Load((_4459.Load(_6689 * 4 + 0) & 16383u) * 76 + 44);
            _7067.ior = asfloat(_4455.Load((_4459.Load(_6689 * 4 + 0) & 16383u) * 76 + 48));
            _7067.sheen_and_sheen_tint = _4455.Load((_4459.Load(_6689 * 4 + 0) & 16383u) * 76 + 52);
            _7067.tint_and_metallic = _4455.Load((_4459.Load(_6689 * 4 + 0) & 16383u) * 76 + 56);
            _7067.transmission_and_transmission_roughness = _4455.Load((_4459.Load(_6689 * 4 + 0) & 16383u) * 76 + 60);
            _7067.specular_and_specular_tint = _4455.Load((_4459.Load(_6689 * 4 + 0) & 16383u) * 76 + 64);
            _7067.clearcoat_and_clearcoat_roughness = _4455.Load((_4459.Load(_6689 * 4 + 0) & 16383u) * 76 + 68);
            _7067.normal_map_strength_unorm = _4455.Load((_4459.Load(_6689 * 4 + 0) & 16383u) * 76 + 72);
            _9590 = _7067.textures[0];
            _9591 = _7067.textures[1];
            _9592 = _7067.textures[2];
            _9593 = _7067.textures[3];
            _9594 = _7067.textures[4];
            _9595 = _7067.base_color[0];
            _9596 = _7067.base_color[1];
            _9597 = _7067.base_color[2];
            _9191 = _7067.flags;
            _9192 = _7067.type;
            _9193 = _7067.tangent_rotation_or_strength;
            _9194 = _7067.roughness_and_anisotropic;
            _9195 = _7067.ior;
            _9196 = _7067.sheen_and_sheen_tint;
            _9197 = _7067.tint_and_metallic;
            _9198 = _7067.transmission_and_transmission_roughness;
            _9199 = _7067.specular_and_specular_tint;
            _9200 = _7067.clearcoat_and_clearcoat_roughness;
            _9201 = _7067.normal_map_strength_unorm;
            _9131 = -_9131;
            _9130 = -_9130;
            _9129 = -_9129;
            _9128 = -_9128;
        }
        float3 param_3 = _9131;
        float4x4 param_4 = _6752.inv_xform;
        _9131 = TransformNormal(param_3, param_4);
        float3 param_5 = _9130;
        float4x4 param_6 = _6752.inv_xform;
        _9130 = TransformNormal(param_5, param_6);
        float3 param_7 = _9129;
        float4x4 param_8 = _6752.inv_xform;
        _9129 = TransformNormal(param_7, param_8);
        float3 param_9 = _9128;
        float4x4 param_10 = _6752.inv_xform;
        _9131 = normalize(_9131);
        _9130 = normalize(_9130);
        _9129 = normalize(_9129);
        _9128 = normalize(TransformNormal(param_9, param_10));
        float _7207 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _7217 = mad(0.5f, log2(abs(mad(_6810.t[0][0] - _6764.t[0][0], _6856.t[0][1] - _6764.t[0][1], -((_6856.t[0][0] - _6764.t[0][0]) * (_6810.t[0][1] - _6764.t[0][1])))) / _7004), log2(_7207));
        uint param_11 = uint(hash(ray.xy));
        float _7224 = construct_float(param_11);
        uint param_12 = uint(hash(hash(ray.xy)));
        float _7231 = construct_float(param_12);
        float param_13[4] = ray.ior;
        bool param_14 = _6675;
        float param_15 = 1.0f;
        float _7240 = peek_ior_stack(param_13, param_14, param_15);
        float3 col = 0.0f.xxx;
        int _7245 = ray.depth & 255;
        int _7250 = (ray.depth >> 8) & 255;
        int _7255 = (ray.depth >> 16) & 255;
        int _7266 = (_7245 + _7250) + _7255;
        int _7274 = _3327_g_params.hi + ((_7266 + ((ray.depth >> 24) & 255)) * 7);
        float mix_rand = frac(asfloat(_3311.Load(_7274 * 4 + 0)) + _7224);
        float mix_weight = 1.0f;
        float _7311;
        float _7328;
        float _7354;
        float _7421;
        while (_9192 == 4u)
        {
            float mix_val = _9193;
            if (_9591 != 4294967295u)
            {
                mix_val *= SampleBilinear(_9591, _6983, 0).x;
            }
            if (_6675)
            {
                _7311 = _7240 / _9195;
            }
            else
            {
                _7311 = _9195 / _7240;
            }
            if (_9195 != 0.0f)
            {
                float param_16 = dot(_6606, _9130);
                float param_17 = _7311;
                _7328 = fresnel_dielectric_cos(param_16, param_17);
            }
            else
            {
                _7328 = 1.0f;
            }
            float _7343 = mix_val;
            float _7344 = _7343 * clamp(_7328, 0.0f, 1.0f);
            mix_val = _7344;
            if (mix_rand > _7344)
            {
                if ((_9191 & 2u) != 0u)
                {
                    _7354 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _7354 = 1.0f;
                }
                mix_weight *= _7354;
                material_t _7367;
                [unroll]
                for (int _80ident = 0; _80ident < 5; _80ident++)
                {
                    _7367.textures[_80ident] = _4455.Load(_80ident * 4 + _9593 * 76 + 0);
                }
                [unroll]
                for (int _81ident = 0; _81ident < 3; _81ident++)
                {
                    _7367.base_color[_81ident] = asfloat(_4455.Load(_81ident * 4 + _9593 * 76 + 20));
                }
                _7367.flags = _4455.Load(_9593 * 76 + 32);
                _7367.type = _4455.Load(_9593 * 76 + 36);
                _7367.tangent_rotation_or_strength = asfloat(_4455.Load(_9593 * 76 + 40));
                _7367.roughness_and_anisotropic = _4455.Load(_9593 * 76 + 44);
                _7367.ior = asfloat(_4455.Load(_9593 * 76 + 48));
                _7367.sheen_and_sheen_tint = _4455.Load(_9593 * 76 + 52);
                _7367.tint_and_metallic = _4455.Load(_9593 * 76 + 56);
                _7367.transmission_and_transmission_roughness = _4455.Load(_9593 * 76 + 60);
                _7367.specular_and_specular_tint = _4455.Load(_9593 * 76 + 64);
                _7367.clearcoat_and_clearcoat_roughness = _4455.Load(_9593 * 76 + 68);
                _7367.normal_map_strength_unorm = _4455.Load(_9593 * 76 + 72);
                _9590 = _7367.textures[0];
                _9591 = _7367.textures[1];
                _9592 = _7367.textures[2];
                _9593 = _7367.textures[3];
                _9594 = _7367.textures[4];
                _9595 = _7367.base_color[0];
                _9596 = _7367.base_color[1];
                _9597 = _7367.base_color[2];
                _9191 = _7367.flags;
                _9192 = _7367.type;
                _9193 = _7367.tangent_rotation_or_strength;
                _9194 = _7367.roughness_and_anisotropic;
                _9195 = _7367.ior;
                _9196 = _7367.sheen_and_sheen_tint;
                _9197 = _7367.tint_and_metallic;
                _9198 = _7367.transmission_and_transmission_roughness;
                _9199 = _7367.specular_and_specular_tint;
                _9200 = _7367.clearcoat_and_clearcoat_roughness;
                _9201 = _7367.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_9191 & 2u) != 0u)
                {
                    _7421 = 1.0f / mix_val;
                }
                else
                {
                    _7421 = 1.0f;
                }
                mix_weight *= _7421;
                material_t _7433;
                [unroll]
                for (int _82ident = 0; _82ident < 5; _82ident++)
                {
                    _7433.textures[_82ident] = _4455.Load(_82ident * 4 + _9594 * 76 + 0);
                }
                [unroll]
                for (int _83ident = 0; _83ident < 3; _83ident++)
                {
                    _7433.base_color[_83ident] = asfloat(_4455.Load(_83ident * 4 + _9594 * 76 + 20));
                }
                _7433.flags = _4455.Load(_9594 * 76 + 32);
                _7433.type = _4455.Load(_9594 * 76 + 36);
                _7433.tangent_rotation_or_strength = asfloat(_4455.Load(_9594 * 76 + 40));
                _7433.roughness_and_anisotropic = _4455.Load(_9594 * 76 + 44);
                _7433.ior = asfloat(_4455.Load(_9594 * 76 + 48));
                _7433.sheen_and_sheen_tint = _4455.Load(_9594 * 76 + 52);
                _7433.tint_and_metallic = _4455.Load(_9594 * 76 + 56);
                _7433.transmission_and_transmission_roughness = _4455.Load(_9594 * 76 + 60);
                _7433.specular_and_specular_tint = _4455.Load(_9594 * 76 + 64);
                _7433.clearcoat_and_clearcoat_roughness = _4455.Load(_9594 * 76 + 68);
                _7433.normal_map_strength_unorm = _4455.Load(_9594 * 76 + 72);
                _9590 = _7433.textures[0];
                _9591 = _7433.textures[1];
                _9592 = _7433.textures[2];
                _9593 = _7433.textures[3];
                _9594 = _7433.textures[4];
                _9595 = _7433.base_color[0];
                _9596 = _7433.base_color[1];
                _9597 = _7433.base_color[2];
                _9191 = _7433.flags;
                _9192 = _7433.type;
                _9193 = _7433.tangent_rotation_or_strength;
                _9194 = _7433.roughness_and_anisotropic;
                _9195 = _7433.ior;
                _9196 = _7433.sheen_and_sheen_tint;
                _9197 = _7433.tint_and_metallic;
                _9198 = _7433.transmission_and_transmission_roughness;
                _9199 = _7433.specular_and_specular_tint;
                _9200 = _7433.clearcoat_and_clearcoat_roughness;
                _9201 = _7433.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_9590 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_9590, _6983, 0).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_9590 & 33554432u) != 0u)
            {
                float3 _10162 = normals;
                _10162.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _10162;
            }
            float3 _7515 = _9130;
            _9130 = normalize(((_9128 * normals.x) + (_7515 * normals.z)) + (_9129 * normals.y));
            if ((_9201 & 65535u) != 65535u)
            {
                _9130 = normalize(_7515 + ((_9130 - _7515) * clamp(float(_9201 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_18 = _9131;
            float3 param_19 = -_6606;
            float3 param_20 = _9130;
            _9130 = ensure_valid_reflection(param_18, param_19, param_20);
        }
        float3 _7581 = ((_6902 * _6925) + (_6910 * inter.u)) + (_6918 * inter.v);
        float3 _7588 = float3(-_7581.z, 0.0f, _7581.x);
        float3 tangent = _7588;
        float3 param_21 = _7588;
        float4x4 param_22 = _6752.inv_xform;
        float3 _7594 = TransformNormal(param_21, param_22);
        tangent = _7594;
        float3 _7598 = cross(_7594, _9130);
        if (dot(_7598, _7598) == 0.0f)
        {
            float3 param_23 = _7581;
            float4x4 param_24 = _6752.inv_xform;
            tangent = TransformNormal(param_23, param_24);
        }
        if (_9193 != 0.0f)
        {
            float3 param_25 = tangent;
            float3 param_26 = _9130;
            float param_27 = _9193;
            tangent = rotate_around_axis(param_25, param_26, param_27);
        }
        float3 _7631 = normalize(cross(tangent, _9130));
        _9129 = _7631;
        _9128 = cross(_9130, _7631);
        float3 _9289 = 0.0f.xxx;
        float3 _9288 = 0.0f.xxx;
        float _9293 = 0.0f;
        float _9291 = 0.0f;
        float _9292 = 1.0f;
        bool _7647 = _3327_g_params.li_count != 0;
        bool _7653;
        if (_7647)
        {
            _7653 = _9192 != 3u;
        }
        else
        {
            _7653 = _7647;
        }
        float3 _9290;
        bool _9294;
        bool _9295;
        if (_7653)
        {
            float3 param_28 = _6642;
            float3 param_29 = _9128;
            float3 param_30 = _9129;
            float3 param_31 = _9130;
            int param_32 = _7274;
            float2 param_33 = float2(_7224, _7231);
            light_sample_t _9304 = { _9288, _9289, _9290, _9291, _9292, _9293, _9294, _9295 };
            light_sample_t param_34 = _9304;
            SampleLightSource(param_28, param_29, param_30, param_31, param_32, param_33, param_34);
            _9288 = param_34.col;
            _9289 = param_34.L;
            _9290 = param_34.lp;
            _9291 = param_34.area;
            _9292 = param_34.dist_mul;
            _9293 = param_34.pdf;
            _9294 = param_34.cast_shadow;
            _9295 = param_34.from_env;
        }
        float _7681 = dot(_9130, _9289);
        float3 base_color = float3(_9595, _9596, _9597);
        [branch]
        if (_9591 != 4294967295u)
        {
            base_color *= SampleBilinear(_9591, _6983, int(get_texture_lod(texSize(_9591), _7217)), true, true).xyz;
        }
        out_base_color = base_color;
        out_normals = _9130;
        float3 tint_color = 0.0f.xxx;
        float _7717 = lum(base_color);
        [flatten]
        if (_7717 > 0.0f)
        {
            tint_color = base_color / _7717.xxx;
        }
        float roughness = clamp(float(_9194 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_9592 != 4294967295u)
        {
            roughness *= SampleBilinear(_9592, _6983, int(get_texture_lod(texSize(_9592), _7217)), false, true).x;
        }
        float _7762 = frac(asfloat(_3311.Load((_7274 + 1) * 4 + 0)) + _7224);
        float _7771 = frac(asfloat(_3311.Load((_7274 + 2) * 4 + 0)) + _7231);
        float _9717 = 0.0f;
        float _9716 = 0.0f;
        float _9715 = 0.0f;
        float _9353[4];
        [unroll]
        for (int i = 0; i < 4; )
        {
            _9353[i] = ray.ior[i];
            i++;
            continue;
        }
        float _9354 = _7207;
        float _9355 = ray.cone_spread;
        int _9356 = ray.xy;
        float _9351 = 0.0f;
        float _9822 = 0.0f;
        float _9821 = 0.0f;
        float _9820 = 0.0f;
        int _9458 = ray.depth;
        int _9462 = ray.xy;
        int _9357;
        float _9460;
        float _9645;
        float _9646;
        float _9647;
        float _9680;
        float _9681;
        float _9682;
        float _9750;
        float _9751;
        float _9752;
        float _9785;
        float _9786;
        float _9787;
        [branch]
        if (_9192 == 0u)
        {
            [branch]
            if ((_9293 > 0.0f) && (_7681 > 0.0f))
            {
                light_sample_t _9321 = { _9288, _9289, _9290, _9291, _9292, _9293, _9294, _9295 };
                surface_t _9139 = { _6642, _9128, _9129, _9130, _9131, _6983 };
                float _9826[3] = { _9820, _9821, _9822 };
                float _9791[3] = { _9785, _9786, _9787 };
                float _9756[3] = { _9750, _9751, _9752 };
                shadow_ray_t _9472 = { _9756, _9458, _9791, _9460, _9826, _9462 };
                shadow_ray_t param_35 = _9472;
                float3 _7831 = Evaluate_DiffuseNode(_9321, ray, _9139, base_color, roughness, mix_weight, param_35);
                _9750 = param_35.o[0];
                _9751 = param_35.o[1];
                _9752 = param_35.o[2];
                _9458 = param_35.depth;
                _9785 = param_35.d[0];
                _9786 = param_35.d[1];
                _9787 = param_35.d[2];
                _9460 = param_35.dist;
                _9820 = param_35.c[0];
                _9821 = param_35.c[1];
                _9822 = param_35.c[2];
                _9462 = param_35.xy;
                col += _7831;
            }
            bool _7838 = _7245 < _3327_g_params.max_diff_depth;
            bool _7845;
            if (_7838)
            {
                _7845 = _7266 < _3327_g_params.max_total_depth;
            }
            else
            {
                _7845 = _7838;
            }
            [branch]
            if (_7845)
            {
                surface_t _9146 = { _6642, _9128, _9129, _9130, _9131, _6983 };
                float _9721[3] = { _9715, _9716, _9717 };
                float _9686[3] = { _9680, _9681, _9682 };
                float _9651[3] = { _9645, _9646, _9647 };
                ray_data_t _9371 = { _9651, _9686, _9351, _9721, _9353, _9354, _9355, _9356, _9357 };
                ray_data_t param_36 = _9371;
                Sample_DiffuseNode(ray, _9146, base_color, roughness, _7762, _7771, mix_weight, param_36);
                _9645 = param_36.o[0];
                _9646 = param_36.o[1];
                _9647 = param_36.o[2];
                _9680 = param_36.d[0];
                _9681 = param_36.d[1];
                _9682 = param_36.d[2];
                _9351 = param_36.pdf;
                _9715 = param_36.c[0];
                _9716 = param_36.c[1];
                _9717 = param_36.c[2];
                _9353 = param_36.ior;
                _9354 = param_36.cone_width;
                _9355 = param_36.cone_spread;
                _9356 = param_36.xy;
                _9357 = param_36.depth;
            }
        }
        else
        {
            [branch]
            if (_9192 == 1u)
            {
                float param_37 = 1.0f;
                float param_38 = 1.5f;
                float _7869 = fresnel_dielectric_cos(param_37, param_38);
                float _7873 = roughness * roughness;
                bool _7876 = _9293 > 0.0f;
                bool _7883;
                if (_7876)
                {
                    _7883 = (_7873 * _7873) >= 1.0000000116860974230803549289703e-07f;
                }
                else
                {
                    _7883 = _7876;
                }
                [branch]
                if (_7883 && (_7681 > 0.0f))
                {
                    light_sample_t _9330 = { _9288, _9289, _9290, _9291, _9292, _9293, _9294, _9295 };
                    surface_t _9153 = { _6642, _9128, _9129, _9130, _9131, _6983 };
                    float _9833[3] = { _9820, _9821, _9822 };
                    float _9798[3] = { _9785, _9786, _9787 };
                    float _9763[3] = { _9750, _9751, _9752 };
                    shadow_ray_t _9485 = { _9763, _9458, _9798, _9460, _9833, _9462 };
                    shadow_ray_t param_39 = _9485;
                    float3 _7898 = Evaluate_GlossyNode(_9330, ray, _9153, base_color, roughness, 1.5f, _7869, mix_weight, param_39);
                    _9750 = param_39.o[0];
                    _9751 = param_39.o[1];
                    _9752 = param_39.o[2];
                    _9458 = param_39.depth;
                    _9785 = param_39.d[0];
                    _9786 = param_39.d[1];
                    _9787 = param_39.d[2];
                    _9460 = param_39.dist;
                    _9820 = param_39.c[0];
                    _9821 = param_39.c[1];
                    _9822 = param_39.c[2];
                    _9462 = param_39.xy;
                    col += _7898;
                }
                bool _7905 = _7250 < _3327_g_params.max_spec_depth;
                bool _7912;
                if (_7905)
                {
                    _7912 = _7266 < _3327_g_params.max_total_depth;
                }
                else
                {
                    _7912 = _7905;
                }
                [branch]
                if (_7912)
                {
                    surface_t _9160 = { _6642, _9128, _9129, _9130, _9131, _6983 };
                    float _9728[3] = { _9715, _9716, _9717 };
                    float _9693[3] = { _9680, _9681, _9682 };
                    float _9658[3] = { _9645, _9646, _9647 };
                    ray_data_t _9390 = { _9658, _9693, _9351, _9728, _9353, _9354, _9355, _9356, _9357 };
                    ray_data_t param_40 = _9390;
                    Sample_GlossyNode(ray, _9160, base_color, roughness, 1.5f, _7869, _7762, _7771, mix_weight, param_40);
                    _9645 = param_40.o[0];
                    _9646 = param_40.o[1];
                    _9647 = param_40.o[2];
                    _9680 = param_40.d[0];
                    _9681 = param_40.d[1];
                    _9682 = param_40.d[2];
                    _9351 = param_40.pdf;
                    _9715 = param_40.c[0];
                    _9716 = param_40.c[1];
                    _9717 = param_40.c[2];
                    _9353 = param_40.ior;
                    _9354 = param_40.cone_width;
                    _9355 = param_40.cone_spread;
                    _9356 = param_40.xy;
                    _9357 = param_40.depth;
                }
            }
            else
            {
                [branch]
                if (_9192 == 2u)
                {
                    float _7936 = roughness * roughness;
                    bool _7939 = _9293 > 0.0f;
                    bool _7946;
                    if (_7939)
                    {
                        _7946 = (_7936 * _7936) >= 1.0000000116860974230803549289703e-07f;
                    }
                    else
                    {
                        _7946 = _7939;
                    }
                    [branch]
                    if (_7946 && (_7681 < 0.0f))
                    {
                        float _7954;
                        if (_6675)
                        {
                            _7954 = _9195 / _7240;
                        }
                        else
                        {
                            _7954 = _7240 / _9195;
                        }
                        light_sample_t _9339 = { _9288, _9289, _9290, _9291, _9292, _9293, _9294, _9295 };
                        surface_t _9167 = { _6642, _9128, _9129, _9130, _9131, _6983 };
                        float _9840[3] = { _9820, _9821, _9822 };
                        float _9805[3] = { _9785, _9786, _9787 };
                        float _9770[3] = { _9750, _9751, _9752 };
                        shadow_ray_t _9498 = { _9770, _9458, _9805, _9460, _9840, _9462 };
                        shadow_ray_t param_41 = _9498;
                        float3 _7976 = Evaluate_RefractiveNode(_9339, ray, _9167, base_color, _7936, _7954, mix_weight, param_41);
                        _9750 = param_41.o[0];
                        _9751 = param_41.o[1];
                        _9752 = param_41.o[2];
                        _9458 = param_41.depth;
                        _9785 = param_41.d[0];
                        _9786 = param_41.d[1];
                        _9787 = param_41.d[2];
                        _9460 = param_41.dist;
                        _9820 = param_41.c[0];
                        _9821 = param_41.c[1];
                        _9822 = param_41.c[2];
                        _9462 = param_41.xy;
                        col += _7976;
                    }
                    bool _7983 = _7255 < _3327_g_params.max_refr_depth;
                    bool _7990;
                    if (_7983)
                    {
                        _7990 = _7266 < _3327_g_params.max_total_depth;
                    }
                    else
                    {
                        _7990 = _7983;
                    }
                    [branch]
                    if (_7990)
                    {
                        surface_t _9174 = { _6642, _9128, _9129, _9130, _9131, _6983 };
                        float _9735[3] = { _9715, _9716, _9717 };
                        float _9700[3] = { _9680, _9681, _9682 };
                        float _9665[3] = { _9645, _9646, _9647 };
                        ray_data_t _9409 = { _9665, _9700, _9351, _9735, _9353, _9354, _9355, _9356, _9357 };
                        ray_data_t param_42 = _9409;
                        Sample_RefractiveNode(ray, _9174, base_color, roughness, _6675, _9195, _7240, _7762, _7771, mix_weight, param_42);
                        _9645 = param_42.o[0];
                        _9646 = param_42.o[1];
                        _9647 = param_42.o[2];
                        _9680 = param_42.d[0];
                        _9681 = param_42.d[1];
                        _9682 = param_42.d[2];
                        _9351 = param_42.pdf;
                        _9715 = param_42.c[0];
                        _9716 = param_42.c[1];
                        _9717 = param_42.c[2];
                        _9353 = param_42.ior;
                        _9354 = param_42.cone_width;
                        _9355 = param_42.cone_spread;
                        _9356 = param_42.xy;
                        _9357 = param_42.depth;
                    }
                }
                else
                {
                    [branch]
                    if (_9192 == 3u)
                    {
                        float mis_weight = 1.0f;
                        [branch]
                        if ((_9191 & 1u) != 0u)
                        {
                            float3 _8060 = mul(float4(_6999, 0.0f), _6752.xform).xyz;
                            float _8063 = length(_8060);
                            float _8075 = abs(dot(_6606, _8060 / _8063.xxx));
                            if (_8075 > 0.0f)
                            {
                                float param_43 = ray.pdf;
                                float param_44 = (inter.t * inter.t) / ((0.5f * _8063) * _8075);
                                mis_weight = power_heuristic(param_43, param_44);
                            }
                        }
                        col += (base_color * ((mix_weight * mis_weight) * _9193));
                    }
                    else
                    {
                        [branch]
                        if (_9192 == 6u)
                        {
                            float metallic = clamp(float((_9197 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_9593 != 4294967295u)
                            {
                                metallic *= SampleBilinear(_9593, _6983, int(get_texture_lod(texSize(_9593), _7217))).x;
                            }
                            float specular = clamp(float(_9199 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_9594 != 4294967295u)
                            {
                                specular *= SampleBilinear(_9594, _6983, int(get_texture_lod(texSize(_9594), _7217))).x;
                            }
                            float _8192 = clamp(float(_9200 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8200 = clamp(float((_9200 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8208 = 2.0f * clamp(float(_9196 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float3 _8226 = lerp(1.0f.xxx, tint_color, clamp(float((_9196 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _8208;
                            float3 _8246 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_9199 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                            float _8255 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                            float param_45 = 1.0f;
                            float param_46 = _8255;
                            float _8261 = fresnel_dielectric_cos(param_45, param_46);
                            float _8269 = clamp(float((_9194 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8280 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _8192))) - 1.0f;
                            float param_47 = 1.0f;
                            float param_48 = _8280;
                            float _8286 = fresnel_dielectric_cos(param_47, param_48);
                            float _8301 = mad(roughness - 1.0f, 1.0f - clamp(float((_9198 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                            float _8307;
                            if (_6675)
                            {
                                _8307 = _9195 / _7240;
                            }
                            else
                            {
                                _8307 = _7240 / _9195;
                            }
                            float param_49 = dot(_6606, _9130);
                            float param_50 = 1.0f / _8307;
                            float _8330 = fresnel_dielectric_cos(param_49, param_50);
                            float param_51 = dot(_6606, _9130);
                            float param_52 = _8255;
                            lobe_weights_t _8369 = get_lobe_weights(lerp(_7717, 1.0f, _8208), lum(lerp(_8246, 1.0f.xxx, ((fresnel_dielectric_cos(param_51, param_52) - _8261) / (1.0f - _8261)).xxx)), specular, metallic, clamp(float(_9198 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _8192);
                            [branch]
                            if (_9293 > 0.0f)
                            {
                                light_sample_t _9348 = { _9288, _9289, _9290, _9291, _9292, _9293, _9294, _9295 };
                                surface_t _9181 = { _6642, _9128, _9129, _9130, _9131, _6983 };
                                diff_params_t _9540 = { base_color, _8226, roughness };
                                spec_params_t _9555 = { _8246, roughness, _8255, _8261, _8269 };
                                clearcoat_params_t _9568 = { _8200, _8280, _8286 };
                                transmission_params_t _9583 = { _8301, _9195, _8307, _8330, _6675 };
                                float _9847[3] = { _9820, _9821, _9822 };
                                float _9812[3] = { _9785, _9786, _9787 };
                                float _9777[3] = { _9750, _9751, _9752 };
                                shadow_ray_t _9511 = { _9777, _9458, _9812, _9460, _9847, _9462 };
                                shadow_ray_t param_53 = _9511;
                                float3 _8388 = Evaluate_PrincipledNode(_9348, ray, _9181, _8369, _9540, _9555, _9568, _9583, metallic, _7681, mix_weight, param_53);
                                _9750 = param_53.o[0];
                                _9751 = param_53.o[1];
                                _9752 = param_53.o[2];
                                _9458 = param_53.depth;
                                _9785 = param_53.d[0];
                                _9786 = param_53.d[1];
                                _9787 = param_53.d[2];
                                _9460 = param_53.dist;
                                _9820 = param_53.c[0];
                                _9821 = param_53.c[1];
                                _9822 = param_53.c[2];
                                _9462 = param_53.xy;
                                col += _8388;
                            }
                            surface_t _9188 = { _6642, _9128, _9129, _9130, _9131, _6983 };
                            diff_params_t _9544 = { base_color, _8226, roughness };
                            spec_params_t _9561 = { _8246, roughness, _8255, _8261, _8269 };
                            clearcoat_params_t _9572 = { _8200, _8280, _8286 };
                            transmission_params_t _9589 = { _8301, _9195, _8307, _8330, _6675 };
                            float param_54 = mix_rand;
                            float _9742[3] = { _9715, _9716, _9717 };
                            float _9707[3] = { _9680, _9681, _9682 };
                            float _9672[3] = { _9645, _9646, _9647 };
                            ray_data_t _9428 = { _9672, _9707, _9351, _9742, _9353, _9354, _9355, _9356, _9357 };
                            ray_data_t param_55 = _9428;
                            Sample_PrincipledNode(ray, _9188, _8369, _9544, _9561, _9572, _9589, metallic, _7762, _7771, param_54, mix_weight, param_55);
                            _9645 = param_55.o[0];
                            _9646 = param_55.o[1];
                            _9647 = param_55.o[2];
                            _9680 = param_55.d[0];
                            _9681 = param_55.d[1];
                            _9682 = param_55.d[2];
                            _9351 = param_55.pdf;
                            _9715 = param_55.c[0];
                            _9716 = param_55.c[1];
                            _9717 = param_55.c[2];
                            _9353 = param_55.ior;
                            _9354 = param_55.cone_width;
                            _9355 = param_55.cone_spread;
                            _9356 = param_55.xy;
                            _9357 = param_55.depth;
                        }
                    }
                }
            }
        }
        float _8422 = max(_9715, max(_9716, _9717));
        float _8434;
        if (_7266 > _3327_g_params.min_total_depth)
        {
            _8434 = max(0.0500000007450580596923828125f, 1.0f - _8422);
        }
        else
        {
            _8434 = 0.0f;
        }
        bool _8448 = (frac(asfloat(_3311.Load((_7274 + 6) * 4 + 0)) + _7224) >= _8434) && (_8422 > 0.0f);
        bool _8454;
        if (_8448)
        {
            _8454 = _9351 > 0.0f;
        }
        else
        {
            _8454 = _8448;
        }
        [branch]
        if (_8454)
        {
            float _8458 = _9351;
            float _8459 = min(_8458, 1000000.0f);
            _9351 = _8459;
            float _8462 = 1.0f - _8434;
            float _8464 = _9715;
            float _8465 = _8464 / _8462;
            _9715 = _8465;
            float _8470 = _9716;
            float _8471 = _8470 / _8462;
            _9716 = _8471;
            float _8476 = _9717;
            float _8477 = _8476 / _8462;
            _9717 = _8477;
            uint _8485;
            _8483.InterlockedAdd(0, 1u, _8485);
            _8494.Store(_8485 * 72 + 0, asuint(_9645));
            _8494.Store(_8485 * 72 + 4, asuint(_9646));
            _8494.Store(_8485 * 72 + 8, asuint(_9647));
            _8494.Store(_8485 * 72 + 12, asuint(_9680));
            _8494.Store(_8485 * 72 + 16, asuint(_9681));
            _8494.Store(_8485 * 72 + 20, asuint(_9682));
            _8494.Store(_8485 * 72 + 24, asuint(_8459));
            _8494.Store(_8485 * 72 + 28, asuint(_8465));
            _8494.Store(_8485 * 72 + 32, asuint(_8471));
            _8494.Store(_8485 * 72 + 36, asuint(_8477));
            _8494.Store(_8485 * 72 + 40, asuint(_9353[0]));
            _8494.Store(_8485 * 72 + 44, asuint(_9353[1]));
            _8494.Store(_8485 * 72 + 48, asuint(_9353[2]));
            _8494.Store(_8485 * 72 + 52, asuint(_9353[3]));
            _8494.Store(_8485 * 72 + 56, asuint(_9354));
            _8494.Store(_8485 * 72 + 60, asuint(_9355));
            _8494.Store(_8485 * 72 + 64, uint(_9356));
            _8494.Store(_8485 * 72 + 68, uint(_9357));
        }
        [branch]
        if (max(_9820, max(_9821, _9822)) > 0.0f)
        {
            float3 _8571 = _9290 - float3(_9750, _9751, _9752);
            float _8574 = length(_8571);
            float3 _8578 = _8571 / _8574.xxx;
            float sh_dist = _8574 * _9292;
            if (_9295)
            {
                sh_dist = -sh_dist;
            }
            float _8590 = _8578.x;
            _9785 = _8590;
            float _8593 = _8578.y;
            _9786 = _8593;
            float _8596 = _8578.z;
            _9787 = _8596;
            _9460 = sh_dist;
            uint _8602;
            _8483.InterlockedAdd(8, 1u, _8602);
            _8610.Store(_8602 * 48 + 0, asuint(_9750));
            _8610.Store(_8602 * 48 + 4, asuint(_9751));
            _8610.Store(_8602 * 48 + 8, asuint(_9752));
            _8610.Store(_8602 * 48 + 12, uint(_9458));
            _8610.Store(_8602 * 48 + 16, asuint(_8590));
            _8610.Store(_8602 * 48 + 20, asuint(_8593));
            _8610.Store(_8602 * 48 + 24, asuint(_8596));
            _8610.Store(_8602 * 48 + 28, asuint(sh_dist));
            _8610.Store(_8602 * 48 + 32, asuint(_9820));
            _8610.Store(_8602 * 48 + 36, asuint(_9821));
            _8610.Store(_8602 * 48 + 40, asuint(_9822));
            _8610.Store(_8602 * 48 + 44, uint(_9462));
        }
        _8830 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _8830;
}

void comp_main()
{
    do
    {
        int _8676 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_8676) >= _8483.Load(4))
        {
            break;
        }
        int _8692 = int(_8689.Load(_8676 * 72 + 64));
        int _8699 = int(_8689.Load(_8676 * 72 + 64));
        hit_data_t _8710;
        _8710.mask = int(_8706.Load(_8676 * 24 + 0));
        _8710.obj_index = int(_8706.Load(_8676 * 24 + 4));
        _8710.prim_index = int(_8706.Load(_8676 * 24 + 8));
        _8710.t = asfloat(_8706.Load(_8676 * 24 + 12));
        _8710.u = asfloat(_8706.Load(_8676 * 24 + 16));
        _8710.v = asfloat(_8706.Load(_8676 * 24 + 20));
        ray_data_t _8726;
        [unroll]
        for (int _84ident = 0; _84ident < 3; _84ident++)
        {
            _8726.o[_84ident] = asfloat(_8689.Load(_84ident * 4 + _8676 * 72 + 0));
        }
        [unroll]
        for (int _85ident = 0; _85ident < 3; _85ident++)
        {
            _8726.d[_85ident] = asfloat(_8689.Load(_85ident * 4 + _8676 * 72 + 12));
        }
        _8726.pdf = asfloat(_8689.Load(_8676 * 72 + 24));
        [unroll]
        for (int _86ident = 0; _86ident < 3; _86ident++)
        {
            _8726.c[_86ident] = asfloat(_8689.Load(_86ident * 4 + _8676 * 72 + 28));
        }
        [unroll]
        for (int _87ident = 0; _87ident < 4; _87ident++)
        {
            _8726.ior[_87ident] = asfloat(_8689.Load(_87ident * 4 + _8676 * 72 + 40));
        }
        _8726.cone_width = asfloat(_8689.Load(_8676 * 72 + 56));
        _8726.cone_spread = asfloat(_8689.Load(_8676 * 72 + 60));
        _8726.xy = int(_8689.Load(_8676 * 72 + 64));
        _8726.depth = int(_8689.Load(_8676 * 72 + 68));
        hit_data_t _8924 = { _8710.mask, _8710.obj_index, _8710.prim_index, _8710.t, _8710.u, _8710.v };
        hit_data_t param = _8924;
        float _8973[4] = { _8726.ior[0], _8726.ior[1], _8726.ior[2], _8726.ior[3] };
        float _8964[3] = { _8726.c[0], _8726.c[1], _8726.c[2] };
        float _8957[3] = { _8726.d[0], _8726.d[1], _8726.d[2] };
        float _8950[3] = { _8726.o[0], _8726.o[1], _8726.o[2] };
        ray_data_t _8943 = { _8950, _8957, _8726.pdf, _8964, _8973, _8726.cone_width, _8726.cone_spread, _8726.xy, _8726.depth };
        ray_data_t param_1 = _8943;
        float3 param_2 = 0.0f.xxx;
        float3 param_3 = 0.0f.xxx;
        float3 _8782 = ShadeSurface(param, param_1, param_2, param_3);
        int2 _8791 = int2((_8692 >> 16) & 65535, _8699 & 65535);
        g_out_img[_8791] = float4(_8782 + g_out_img[_8791].xyz, 1.0f);
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
