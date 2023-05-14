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

ByteAddressBuffer _3252 : register(t17, space0);
ByteAddressBuffer _3288 : register(t8, space0);
ByteAddressBuffer _3292 : register(t9, space0);
ByteAddressBuffer _4062 : register(t13, space0);
ByteAddressBuffer _4087 : register(t15, space0);
ByteAddressBuffer _4091 : register(t16, space0);
ByteAddressBuffer _4415 : register(t12, space0);
ByteAddressBuffer _4419 : register(t11, space0);
ByteAddressBuffer _6739 : register(t14, space0);
RWByteAddressBuffer _8352 : register(u3, space0);
RWByteAddressBuffer _8363 : register(u1, space0);
RWByteAddressBuffer _8479 : register(u2, space0);
ByteAddressBuffer _8558 : register(t7, space0);
ByteAddressBuffer _8575 : register(t6, space0);
ByteAddressBuffer _8711 : register(t10, space0);
cbuffer UniformParams
{
    Params _3268_g_params : packoffset(c0);
};

Texture2D<float4> g_textures[] : register(t0, space1);
SamplerState _g_textures_sampler[] : register(s0, space1);
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

uint2 spvTextureSize(Texture2D<float4> Tex, uint Level, out uint Param)
{
    uint2 ret;
    Tex.GetDimensions(Level, ret.x, ret.y, Param);
    return ret;
}

int hash(int x)
{
    uint _504 = uint(x);
    uint _511 = ((_504 >> uint(16)) ^ _504) * 73244475u;
    uint _516 = ((_511 >> uint(16)) ^ _511) * 73244475u;
    return int((_516 >> uint(16)) ^ _516);
}

float construct_float(inout uint m)
{
    m &= 8388607u;
    m |= 1065353216u;
    return asfloat(m) - 1.0f;
}

float3 rgbe_to_rgb(float4 rgbe)
{
    return rgbe.xyz * exp2(mad(255.0f, rgbe.w, -128.0f));
}

float3 SampleLatlong_RGBE(uint index, float3 dir, float y_rotation, float2 rand)
{
    float _1084 = atan2(dir.z, dir.x) + y_rotation;
    float phi = _1084;
    if (_1084 < 0.0f)
    {
        phi += 6.283185482025146484375f;
    }
    if (phi > 6.283185482025146484375f)
    {
        phi -= 6.283185482025146484375f;
    }
    uint _1103 = index & 16777215u;
    uint _1109_dummy_parameter;
    uint _1119 = _1103;
    float4 param = g_textures[NonUniformResourceIndex(_1119)].Load(int3(int2(mad(float2(frac(phi * 0.15915493667125701904296875f), acos(clamp(dir.y, -1.0f, 1.0f)) * 0.3183098733425140380859375f), float2(int2(spvTextureSize(g_textures[_1103], uint(0), _1109_dummy_parameter))), rand)), 0));
    return rgbe_to_rgb(param);
}

float2 DirToCanonical(float3 d, float y_rotation)
{
    float _727 = (-atan2(d.z, d.x)) + y_rotation;
    float phi = _727;
    if (_727 < 0.0f)
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
    float2 _754 = DirToCanonical(L, -y_rotation);
    float factor = 1.0f;
    while (lod >= 0)
    {
        int2 _774 = clamp(int2(_754 * float(res)), int2(0, 0), (res - 1).xx);
        float4 quad = qtree_tex.Load(int3(_774 / int2(2, 2), lod));
        float _809 = ((quad.x + quad.y) + quad.z) + quad.w;
        if (_809 <= 0.0f)
        {
            break;
        }
        factor *= ((4.0f * quad[(0 | ((_774.x & 1) << 0)) | ((_774.y & 1) << 1)]) / _809);
        lod--;
        res *= 2;
    }
    return factor * 0.079577468335628509521484375f;
}

float power_heuristic(float a, float b)
{
    float _1135 = a * a;
    return _1135 / mad(b, b, _1135);
}

float3 Evaluate_EnvColor(ray_data_t ray, float2 tex_rand)
{
    float3 _4628 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 env_col = _3268_g_params.back_col.xyz;
    uint _4636 = asuint(_3268_g_params.back_col.w);
    if (_4636 != 4294967295u)
    {
        env_col *= SampleLatlong_RGBE(_4636, _4628, _3268_g_params.back_rotation, tex_rand);
    }
    if (_3268_g_params.env_qtree_levels > 0)
    {
        float param = ray.pdf;
        float param_1 = Evaluate_EnvQTree(_3268_g_params.back_rotation, g_env_qtree, _g_env_qtree_sampler, _3268_g_params.env_qtree_levels, _4628);
        env_col *= power_heuristic(param, param_1);
    }
    else
    {
        if (_3268_g_params.env_mult_importance != 0)
        {
            float param_2 = ray.pdf;
            float param_3 = 0.15915493667125701904296875f;
            env_col *= power_heuristic(param_2, param_3);
        }
    }
    return env_col;
}

float3 Evaluate_LightColor(ray_data_t ray, hit_data_t inter, float2 tex_rand)
{
    float3 _4710 = float3(ray.d[0], ray.d[1], ray.d[2]);
    light_t _4724;
    _4724.type_and_param0 = _3288.Load4(((-1) - inter.obj_index) * 64 + 0);
    _4724.param1 = asfloat(_3288.Load4(((-1) - inter.obj_index) * 64 + 16));
    _4724.param2 = asfloat(_3288.Load4(((-1) - inter.obj_index) * 64 + 32));
    _4724.param3 = asfloat(_3288.Load4(((-1) - inter.obj_index) * 64 + 48));
    float3 lcol = asfloat(_4724.type_and_param0.yzw);
    [branch]
    if ((_4724.type_and_param0.x & 128u) != 0u)
    {
        float3 env_col = _3268_g_params.env_col.xyz;
        uint _4751 = asuint(_3268_g_params.env_col.w);
        if (_4751 != 4294967295u)
        {
            env_col *= SampleLatlong_RGBE(_4751, _4710, _3268_g_params.env_rotation, tex_rand);
        }
        lcol *= env_col;
    }
    uint _4769 = _4724.type_and_param0.x & 31u;
    if (_4769 == 0u)
    {
        float param = ray.pdf;
        float param_1 = (inter.t * inter.t) / ((0.5f * _4724.param1.w) * dot(_4710, normalize(_4724.param1.xyz - (float3(ray.o[0], ray.o[1], ray.o[2]) + (_4710 * inter.t)))));
        lcol *= power_heuristic(param, param_1);
        bool _4836 = _4724.param3.x > 0.0f;
        bool _4842;
        if (_4836)
        {
            _4842 = _4724.param3.y > 0.0f;
        }
        else
        {
            _4842 = _4836;
        }
        [branch]
        if (_4842)
        {
            [flatten]
            if (_4724.param3.y > 0.0f)
            {
                lcol *= clamp((_4724.param3.x - acos(clamp(-dot(_4710, _4724.param2.xyz), 0.0f, 1.0f))) / _4724.param3.y, 0.0f, 1.0f);
            }
        }
    }
    else
    {
        if (_4769 == 4u)
        {
            float param_2 = ray.pdf;
            float param_3 = (inter.t * inter.t) / (_4724.param1.w * dot(_4710, normalize(cross(_4724.param2.xyz, _4724.param3.xyz))));
            lcol *= power_heuristic(param_2, param_3);
        }
        else
        {
            if (_4769 == 5u)
            {
                float param_4 = ray.pdf;
                float param_5 = (inter.t * inter.t) / (_4724.param1.w * dot(_4710, normalize(cross(_4724.param2.xyz, _4724.param3.xyz))));
                lcol *= power_heuristic(param_4, param_5);
            }
            else
            {
                if (_4769 == 3u)
                {
                    float param_6 = ray.pdf;
                    float param_7 = (inter.t * inter.t) / (_4724.param1.w * (1.0f - abs(dot(_4710, _4724.param3.xyz))));
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

bool exchange(inout bool old_value, bool new_value)
{
    bool _2043 = old_value;
    old_value = new_value;
    return _2043;
}

float peek_ior_stack(float stack[4], inout bool skip_first, float default_value)
{
    float _8723;
    do
    {
        bool _2127 = stack[3] > 0.0f;
        bool _2136;
        if (_2127)
        {
            bool param = skip_first;
            bool param_1 = false;
            bool _2133 = exchange(param, param_1);
            skip_first = param;
            _2136 = !_2133;
        }
        else
        {
            _2136 = _2127;
        }
        if (_2136)
        {
            _8723 = stack[3];
            break;
        }
        bool _2144 = stack[2] > 0.0f;
        bool _2153;
        if (_2144)
        {
            bool param_2 = skip_first;
            bool param_3 = false;
            bool _2150 = exchange(param_2, param_3);
            skip_first = param_2;
            _2153 = !_2150;
        }
        else
        {
            _2153 = _2144;
        }
        if (_2153)
        {
            _8723 = stack[2];
            break;
        }
        bool _2161 = stack[1] > 0.0f;
        bool _2170;
        if (_2161)
        {
            bool param_4 = skip_first;
            bool param_5 = false;
            bool _2167 = exchange(param_4, param_5);
            skip_first = param_4;
            _2170 = !_2167;
        }
        else
        {
            _2170 = _2161;
        }
        if (_2170)
        {
            _8723 = stack[1];
            break;
        }
        bool _2178 = stack[0] > 0.0f;
        bool _2187;
        if (_2178)
        {
            bool param_6 = skip_first;
            bool param_7 = false;
            bool _2184 = exchange(param_6, param_7);
            skip_first = param_6;
            _2187 = !_2184;
        }
        else
        {
            _2187 = _2178;
        }
        if (_2187)
        {
            _8723 = stack[0];
            break;
        }
        _8723 = default_value;
        break;
    } while(false);
    return _8723;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _603 = mad(col.z, 31.875f, 1.0f);
    float _613 = (col.x - 0.501960813999176025390625f) / _603;
    float _619 = (col.y - 0.501960813999176025390625f) / _603;
    return float3((col.w + _613) - _619, col.w + _619, (col.w - _613) - _619);
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
    uint _1004 = index & 16777215u;
    uint _1008_dummy_parameter;
    float4 res = g_textures[NonUniformResourceIndex(_1004)].SampleLevel(_g_textures_sampler[NonUniformResourceIndex(_1004)], uvs + ((rand - 0.5f.xx) / float2(int2(spvTextureSize(g_textures[NonUniformResourceIndex(_1004)], uint(lod), _1008_dummy_parameter)))), float(lod));
    bool _1027;
    if (maybe_YCoCg)
    {
        _1027 = (index & 67108864u) != 0u;
    }
    else
    {
        _1027 = maybe_YCoCg;
    }
    if (_1027)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _1045;
    if (maybe_SRGB)
    {
        _1045 = (index & 16777216u) != 0u;
    }
    else
    {
        _1045 = maybe_SRGB;
    }
    if (_1045)
    {
        float3 param_1 = res.xyz;
        float3 _1051 = srgb_to_rgb(param_1);
        float4 _9769 = res;
        _9769.x = _1051.x;
        float4 _9771 = _9769;
        _9771.y = _1051.y;
        float4 _9773 = _9771;
        _9773.z = _1051.z;
        res = _9773;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod, float2 rand)
{
    return SampleBilinear(index, uvs, lod, rand, false, false);
}

float fresnel_dielectric_cos(float cosi, float eta)
{
    float _1167 = abs(cosi);
    float _1176 = mad(_1167, _1167, mad(eta, eta, -1.0f));
    float g = _1176;
    float result;
    if (_1176 > 0.0f)
    {
        float _1181 = g;
        float _1182 = sqrt(_1181);
        g = _1182;
        float _1186 = _1182 - _1167;
        float _1189 = _1182 + _1167;
        float _1190 = _1186 / _1189;
        float _1204 = mad(_1167, _1189, -1.0f) / mad(_1167, _1186, 1.0f);
        result = ((0.5f * _1190) * _1190) * mad(_1204, _1204, 1.0f);
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
    float3 _8728;
    do
    {
        float _1240 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1240)
        {
            _8728 = N;
            break;
        }
        float3 _1260 = normalize(N - (Ng * dot(N, Ng)));
        float _1264 = dot(I, _1260);
        float _1268 = dot(I, Ng);
        float _1280 = mad(_1264, _1264, _1268 * _1268);
        float param = (_1264 * _1264) * mad(-_1240, _1240, _1280);
        float _1290 = safe_sqrtf(param);
        float _1296 = mad(_1268, _1240, _1280);
        float _1299 = 0.5f / _1280;
        float _1304 = _1290 + _1296;
        float _1305 = _1299 * _1304;
        float _1311 = (-_1290) + _1296;
        float _1312 = _1299 * _1311;
        bool _1320 = (_1305 > 9.9999997473787516355514526367188e-06f) && (_1305 <= 1.000010013580322265625f);
        bool valid1 = _1320;
        bool _1326 = (_1312 > 9.9999997473787516355514526367188e-06f) && (_1312 <= 1.000010013580322265625f);
        bool valid2 = _1326;
        float2 N_new;
        if (_1320 && _1326)
        {
            float _10067 = (-0.5f) / _1280;
            float param_1 = mad(_10067, _1304, 1.0f);
            float _1336 = safe_sqrtf(param_1);
            float param_2 = _1305;
            float _1339 = safe_sqrtf(param_2);
            float2 _1340 = float2(_1336, _1339);
            float param_3 = mad(_10067, _1311, 1.0f);
            float _1345 = safe_sqrtf(param_3);
            float param_4 = _1312;
            float _1348 = safe_sqrtf(param_4);
            float2 _1349 = float2(_1345, _1348);
            float _10069 = -_1268;
            float _1365 = mad(2.0f * mad(_1336, _1264, _1339 * _1268), _1339, _10069);
            float _1381 = mad(2.0f * mad(_1345, _1264, _1348 * _1268), _1348, _10069);
            bool _1383 = _1365 >= 9.9999997473787516355514526367188e-06f;
            valid1 = _1383;
            bool _1385 = _1381 >= 9.9999997473787516355514526367188e-06f;
            valid2 = _1385;
            if (_1383 && _1385)
            {
                bool2 _1398 = (_1365 < _1381).xx;
                N_new = float2(_1398.x ? _1340.x : _1349.x, _1398.y ? _1340.y : _1349.y);
            }
            else
            {
                bool2 _1406 = (_1365 > _1381).xx;
                N_new = float2(_1406.x ? _1340.x : _1349.x, _1406.y ? _1340.y : _1349.y);
            }
        }
        else
        {
            if (!(valid1 || valid2))
            {
                _8728 = Ng;
                break;
            }
            float _1418 = valid1 ? _1305 : _1312;
            float param_5 = 1.0f - _1418;
            float param_6 = _1418;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _8728 = (_1260 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _8728;
}

float3 rotate_around_axis(float3 p, float3 axis, float angle)
{
    float _1512 = cos(angle);
    float _1515 = sin(angle);
    float _1519 = 1.0f - _1512;
    return float3(mad(mad(_1519 * axis.x, axis.z, axis.y * _1515), p.z, mad(mad(_1519 * axis.x, axis.x, _1512), p.x, mad(_1519 * axis.x, axis.y, -(axis.z * _1515)) * p.y)), mad(mad(_1519 * axis.y, axis.z, -(axis.x * _1515)), p.z, mad(mad(_1519 * axis.x, axis.y, axis.z * _1515), p.x, mad(_1519 * axis.y, axis.y, _1512) * p.y)), mad(mad(_1519 * axis.z, axis.z, _1512), p.z, mad(mad(_1519 * axis.x, axis.z, -(axis.y * _1515)), p.x, mad(_1519 * axis.y, axis.z, axis.x * _1515) * p.y)));
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
    int3 _1669 = int3(n * 128.0f);
    int _1677;
    if (p.x < 0.0f)
    {
        _1677 = -_1669.x;
    }
    else
    {
        _1677 = _1669.x;
    }
    int _1695;
    if (p.y < 0.0f)
    {
        _1695 = -_1669.y;
    }
    else
    {
        _1695 = _1669.y;
    }
    int _1713;
    if (p.z < 0.0f)
    {
        _1713 = -_1669.z;
    }
    else
    {
        _1713 = _1669.z;
    }
    float _1731;
    if (abs(p.x) < 0.03125f)
    {
        _1731 = mad(1.52587890625e-05f, n.x, p.x);
    }
    else
    {
        _1731 = asfloat(asint(p.x) + _1677);
    }
    float _1749;
    if (abs(p.y) < 0.03125f)
    {
        _1749 = mad(1.52587890625e-05f, n.y, p.y);
    }
    else
    {
        _1749 = asfloat(asint(p.y) + _1695);
    }
    float _1766;
    if (abs(p.z) < 0.03125f)
    {
        _1766 = mad(1.52587890625e-05f, n.z, p.z);
    }
    else
    {
        _1766 = asfloat(asint(p.z) + _1713);
    }
    return float3(_1731, _1749, _1766);
}

float3 MapToCone(float r1, float r2, float3 N, float radius)
{
    float3 _8753;
    do
    {
        float2 _3167 = (float2(r1, r2) * 2.0f) - 1.0f.xx;
        float _3169 = _3167.x;
        bool _3170 = _3169 == 0.0f;
        bool _3176;
        if (_3170)
        {
            _3176 = _3167.y == 0.0f;
        }
        else
        {
            _3176 = _3170;
        }
        if (_3176)
        {
            _8753 = N;
            break;
        }
        float _3185 = _3167.y;
        float r;
        float theta;
        if (abs(_3169) > abs(_3185))
        {
            r = _3169;
            theta = 0.785398185253143310546875f * (_3185 / _3169);
        }
        else
        {
            r = _3185;
            theta = 1.57079637050628662109375f * mad(-0.5f, _3169 / _3185, 1.0f);
        }
        float3 param;
        float3 param_1;
        create_tbn(N, param, param_1);
        _8753 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _8753;
}

float3 CanonicalToDir(float2 p, float y_rotation)
{
    float _677 = mad(2.0f, p.x, -1.0f);
    float _682 = mad(6.283185482025146484375f, p.y, y_rotation);
    float phi = _682;
    if (_682 < 0.0f)
    {
        phi += 6.283185482025146484375f;
    }
    if (phi > 6.283185482025146484375f)
    {
        phi -= 6.283185482025146484375f;
    }
    float _700 = sqrt(mad(-_677, _677, 1.0f));
    return float3(_700 * cos(phi), _677, (-_700) * sin(phi));
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
        float _875 = quad.x + quad.z;
        float partial = _875;
        float _882 = (_875 + quad.y) + quad.w;
        if (_882 <= 0.0f)
        {
            break;
        }
        float _891 = partial / _882;
        float boundary = _891;
        int index = 0;
        if (_sample < _891)
        {
            _sample /= boundary;
            boundary = quad.x / partial;
        }
        else
        {
            float _906 = partial;
            float _907 = _882 - _906;
            partial = _907;
            float2 _9756 = origin;
            _9756.x = origin.x + _step;
            origin = _9756;
            _sample = (_sample - boundary) / (1.0f - boundary);
            boundary = quad.y / _907;
            index |= 1;
        }
        if (_sample < boundary)
        {
            _sample /= boundary;
        }
        else
        {
            float2 _9759 = origin;
            _9759.y = origin.y + _step;
            origin = _9759;
            _sample = (_sample - boundary) / (1.0f - boundary);
            index |= 2;
        }
        factor *= ((4.0f * quad[index]) / _882);
        lod--;
        res *= 2;
        _step *= 0.5f;
    }
    float2 _964 = origin;
    float2 _965 = _964 + (float2(rx, ry) * (2.0f * _step));
    origin = _965;
    return float4(CanonicalToDir(_965, y_rotation), factor * 0.079577468335628509521484375f);
}

float3 world_from_tangent(float3 T, float3 B, float3 N, float3 V)
{
    return ((T * V.x) + (B * V.y)) + (N * V.z);
}

void SampleLightSource(float3 P, float3 T, float3 B, float3 N, int hi, float2 sample_off, inout light_sample_t ls)
{
    float _3261 = frac(asfloat(_3252.Load((hi + 3) * 4 + 0)) + sample_off.x);
    float _3273 = float(_3268_g_params.li_count);
    uint _3280 = min(uint(_3261 * _3273), uint(_3268_g_params.li_count - 1));
    light_t _3299;
    _3299.type_and_param0 = _3288.Load4(_3292.Load(_3280 * 4 + 0) * 64 + 0);
    _3299.param1 = asfloat(_3288.Load4(_3292.Load(_3280 * 4 + 0) * 64 + 16));
    _3299.param2 = asfloat(_3288.Load4(_3292.Load(_3280 * 4 + 0) * 64 + 32));
    _3299.param3 = asfloat(_3288.Load4(_3292.Load(_3280 * 4 + 0) * 64 + 48));
    ls.col = asfloat(_3299.type_and_param0.yzw);
    ls.col *= _3273;
    ls.cast_shadow = (_3299.type_and_param0.x & 32u) != 0u;
    ls.from_env = false;
    float2 _3349 = float2(frac(asfloat(_3252.Load((hi + 7) * 4 + 0)) + sample_off.x), frac(asfloat(_3252.Load((hi + 8) * 4 + 0)) + sample_off.y));
    uint _3354 = _3299.type_and_param0.x & 31u;
    [branch]
    if (_3354 == 0u)
    {
        float _3367 = frac(asfloat(_3252.Load((hi + 4) * 4 + 0)) + sample_off.x);
        float3 _3383 = P - _3299.param1.xyz;
        float3 _3390 = _3383 / length(_3383).xxx;
        float _3397 = sqrt(clamp(mad(-_3367, _3367, 1.0f), 0.0f, 1.0f));
        float _3400 = 6.283185482025146484375f * frac(asfloat(_3252.Load((hi + 5) * 4 + 0)) + sample_off.y);
        float3 sampled_dir = float3(_3397 * cos(_3400), _3397 * sin(_3400), _3367);
        float3 param;
        float3 param_1;
        create_tbn(_3390, param, param_1);
        float3 _9830 = sampled_dir;
        float3 _3433 = ((param * _9830.x) + (param_1 * _9830.y)) + (_3390 * _9830.z);
        sampled_dir = _3433;
        float3 _3442 = _3299.param1.xyz + (_3433 * _3299.param2.w);
        float3 _3449 = normalize(_3442 - _3299.param1.xyz);
        float3 param_2 = _3442;
        float3 param_3 = _3449;
        ls.lp = offset_ray(param_2, param_3);
        ls.L = _3442 - P;
        float3 _3462 = ls.L;
        float _3463 = length(_3462);
        ls.L /= _3463.xxx;
        ls.area = _3299.param1.w;
        float _3478 = abs(dot(ls.L, _3449));
        [flatten]
        if (_3478 > 0.0f)
        {
            ls.pdf = (_3463 * _3463) / ((0.5f * ls.area) * _3478);
        }
        [branch]
        if (_3299.param3.x > 0.0f)
        {
            float _3505 = -dot(ls.L, _3299.param2.xyz);
            if (_3505 > 0.0f)
            {
                ls.col *= clamp((_3299.param3.x - acos(clamp(_3505, 0.0f, 1.0f))) / _3299.param3.y, 0.0f, 1.0f);
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
        if (_3354 == 2u)
        {
            ls.L = _3299.param1.xyz;
            if (_3299.param1.w != 0.0f)
            {
                float param_4 = frac(asfloat(_3252.Load((hi + 4) * 4 + 0)) + sample_off.x);
                float param_5 = frac(asfloat(_3252.Load((hi + 5) * 4 + 0)) + sample_off.y);
                float3 param_6 = ls.L;
                float param_7 = tan(_3299.param1.w);
                ls.L = normalize(MapToCone(param_4, param_5, param_6, param_7));
            }
            ls.area = 0.0f;
            ls.lp = P + ls.L;
            ls.dist_mul = 3402823346297367662189621542912.0f;
            ls.pdf = 1.0f;
            if ((_3299.type_and_param0.x & 64u) == 0u)
            {
                ls.area = 0.0f;
            }
        }
        else
        {
            [branch]
            if (_3354 == 4u)
            {
                float3 _3642 = (_3299.param1.xyz + (_3299.param2.xyz * (frac(asfloat(_3252.Load((hi + 4) * 4 + 0)) + sample_off.x) - 0.5f))) + (_3299.param3.xyz * (frac(asfloat(_3252.Load((hi + 5) * 4 + 0)) + sample_off.y) - 0.5f));
                float3 _3647 = normalize(cross(_3299.param2.xyz, _3299.param3.xyz));
                float3 param_8 = _3642;
                float3 param_9 = _3647;
                ls.lp = offset_ray(param_8, param_9);
                ls.L = _3642 - P;
                float3 _3660 = ls.L;
                float _3661 = length(_3660);
                ls.L /= _3661.xxx;
                ls.area = _3299.param1.w;
                float _3676 = dot(-ls.L, _3647);
                if (_3676 > 0.0f)
                {
                    ls.pdf = (_3661 * _3661) / (ls.area * _3676);
                }
                if ((_3299.type_and_param0.x & 64u) == 0u)
                {
                    ls.area = 0.0f;
                }
                [branch]
                if ((_3299.type_and_param0.x & 128u) != 0u)
                {
                    float3 env_col = _3268_g_params.env_col.xyz;
                    uint _3713 = asuint(_3268_g_params.env_col.w);
                    if (_3713 != 4294967295u)
                    {
                        env_col *= SampleLatlong_RGBE(_3713, ls.L, _3268_g_params.env_rotation, _3349);
                    }
                    ls.col *= env_col;
                    ls.from_env = true;
                }
            }
            else
            {
                [branch]
                if (_3354 == 5u)
                {
                    float2 _3777 = (float2(frac(asfloat(_3252.Load((hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_3252.Load((hi + 5) * 4 + 0)) + sample_off.y)) * 2.0f) - 1.0f.xx;
                    float2 offset = _3777;
                    bool _3780 = _3777.x != 0.0f;
                    bool _3786;
                    if (_3780)
                    {
                        _3786 = offset.y != 0.0f;
                    }
                    else
                    {
                        _3786 = _3780;
                    }
                    if (_3786)
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
                        float _3819 = 0.5f * r;
                        offset = float2(_3819 * cos(theta), _3819 * sin(theta));
                    }
                    float3 _3841 = (_3299.param1.xyz + (_3299.param2.xyz * offset.x)) + (_3299.param3.xyz * offset.y);
                    float3 _3846 = normalize(cross(_3299.param2.xyz, _3299.param3.xyz));
                    float3 param_10 = _3841;
                    float3 param_11 = _3846;
                    ls.lp = offset_ray(param_10, param_11);
                    ls.L = _3841 - P;
                    float3 _3859 = ls.L;
                    float _3860 = length(_3859);
                    ls.L /= _3860.xxx;
                    ls.area = _3299.param1.w;
                    float _3875 = dot(-ls.L, _3846);
                    [flatten]
                    if (_3875 > 0.0f)
                    {
                        ls.pdf = (_3860 * _3860) / (ls.area * _3875);
                    }
                    if ((_3299.type_and_param0.x & 64u) == 0u)
                    {
                        ls.area = 0.0f;
                    }
                    [branch]
                    if ((_3299.type_and_param0.x & 128u) != 0u)
                    {
                        float3 env_col_1 = _3268_g_params.env_col.xyz;
                        uint _3909 = asuint(_3268_g_params.env_col.w);
                        if (_3909 != 4294967295u)
                        {
                            env_col_1 *= SampleLatlong_RGBE(_3909, ls.L, _3268_g_params.env_rotation, _3349);
                        }
                        ls.col *= env_col_1;
                        ls.from_env = true;
                    }
                }
                else
                {
                    [branch]
                    if (_3354 == 3u)
                    {
                        float3 _3968 = normalize(cross(P - _3299.param1.xyz, _3299.param3.xyz));
                        float _3975 = 3.1415927410125732421875f * frac(asfloat(_3252.Load((hi + 4) * 4 + 0)) + sample_off.x);
                        float3 _4000 = (_3299.param1.xyz + (((_3968 * cos(_3975)) + (cross(_3968, _3299.param3.xyz) * sin(_3975))) * _3299.param2.w)) + ((_3299.param3.xyz * (frac(asfloat(_3252.Load((hi + 5) * 4 + 0)) + sample_off.y) - 0.5f)) * _3299.param3.w);
                        ls.lp = _4000;
                        float3 _4006 = _4000 - P;
                        float _4009 = length(_4006);
                        ls.L = _4006 / _4009.xxx;
                        ls.area = _3299.param1.w;
                        float _4024 = 1.0f - abs(dot(ls.L, _3299.param3.xyz));
                        [flatten]
                        if (_4024 != 0.0f)
                        {
                            ls.pdf = (_4009 * _4009) / (ls.area * _4024);
                        }
                        if ((_3299.type_and_param0.x & 64u) == 0u)
                        {
                            ls.area = 0.0f;
                        }
                    }
                    else
                    {
                        [branch]
                        if (_3354 == 6u)
                        {
                            uint _4054 = asuint(_3299.param1.x);
                            transform_t _4068;
                            _4068.xform = asfloat(uint4x4(_4062.Load4(asuint(_3299.param1.y) * 128 + 0), _4062.Load4(asuint(_3299.param1.y) * 128 + 16), _4062.Load4(asuint(_3299.param1.y) * 128 + 32), _4062.Load4(asuint(_3299.param1.y) * 128 + 48)));
                            _4068.inv_xform = asfloat(uint4x4(_4062.Load4(asuint(_3299.param1.y) * 128 + 64), _4062.Load4(asuint(_3299.param1.y) * 128 + 80), _4062.Load4(asuint(_3299.param1.y) * 128 + 96), _4062.Load4(asuint(_3299.param1.y) * 128 + 112)));
                            uint _4093 = _4054 * 3u;
                            vertex_t _4099;
                            [unroll]
                            for (int _44ident = 0; _44ident < 3; _44ident++)
                            {
                                _4099.p[_44ident] = asfloat(_4087.Load(_44ident * 4 + _4091.Load(_4093 * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _45ident = 0; _45ident < 3; _45ident++)
                            {
                                _4099.n[_45ident] = asfloat(_4087.Load(_45ident * 4 + _4091.Load(_4093 * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _46ident = 0; _46ident < 3; _46ident++)
                            {
                                _4099.b[_46ident] = asfloat(_4087.Load(_46ident * 4 + _4091.Load(_4093 * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _47ident = 0; _47ident < 2; _47ident++)
                            {
                                [unroll]
                                for (int _48ident = 0; _48ident < 2; _48ident++)
                                {
                                    _4099.t[_47ident][_48ident] = asfloat(_4087.Load(_48ident * 4 + _47ident * 8 + _4091.Load(_4093 * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4148;
                            [unroll]
                            for (int _49ident = 0; _49ident < 3; _49ident++)
                            {
                                _4148.p[_49ident] = asfloat(_4087.Load(_49ident * 4 + _4091.Load((_4093 + 1u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _50ident = 0; _50ident < 3; _50ident++)
                            {
                                _4148.n[_50ident] = asfloat(_4087.Load(_50ident * 4 + _4091.Load((_4093 + 1u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _51ident = 0; _51ident < 3; _51ident++)
                            {
                                _4148.b[_51ident] = asfloat(_4087.Load(_51ident * 4 + _4091.Load((_4093 + 1u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _52ident = 0; _52ident < 2; _52ident++)
                            {
                                [unroll]
                                for (int _53ident = 0; _53ident < 2; _53ident++)
                                {
                                    _4148.t[_52ident][_53ident] = asfloat(_4087.Load(_53ident * 4 + _52ident * 8 + _4091.Load((_4093 + 1u) * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4194;
                            [unroll]
                            for (int _54ident = 0; _54ident < 3; _54ident++)
                            {
                                _4194.p[_54ident] = asfloat(_4087.Load(_54ident * 4 + _4091.Load((_4093 + 2u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _55ident = 0; _55ident < 3; _55ident++)
                            {
                                _4194.n[_55ident] = asfloat(_4087.Load(_55ident * 4 + _4091.Load((_4093 + 2u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _56ident = 0; _56ident < 3; _56ident++)
                            {
                                _4194.b[_56ident] = asfloat(_4087.Load(_56ident * 4 + _4091.Load((_4093 + 2u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _57ident = 0; _57ident < 2; _57ident++)
                            {
                                [unroll]
                                for (int _58ident = 0; _58ident < 2; _58ident++)
                                {
                                    _4194.t[_57ident][_58ident] = asfloat(_4087.Load(_58ident * 4 + _57ident * 8 + _4091.Load((_4093 + 2u) * 4 + 0) * 52 + 36));
                                }
                            }
                            float3 _4240 = float3(_4099.p[0], _4099.p[1], _4099.p[2]);
                            float3 _4248 = float3(_4148.p[0], _4148.p[1], _4148.p[2]);
                            float3 _4256 = float3(_4194.p[0], _4194.p[1], _4194.p[2]);
                            float _4284 = sqrt(frac(asfloat(_3252.Load((hi + 4) * 4 + 0)) + sample_off.x));
                            float _4293 = frac(asfloat(_3252.Load((hi + 5) * 4 + 0)) + sample_off.y);
                            float _4297 = 1.0f - _4284;
                            float _4302 = 1.0f - _4293;
                            float3 _4333 = mul(float4((_4240 * _4297) + (((_4248 * _4302) + (_4256 * _4293)) * _4284), 1.0f), _4068.xform).xyz;
                            float3 _4349 = mul(float4(cross(_4248 - _4240, _4256 - _4240), 0.0f), _4068.xform).xyz;
                            ls.area = 0.5f * length(_4349);
                            float3 _4355 = normalize(_4349);
                            ls.L = _4333 - P;
                            float3 _4362 = ls.L;
                            float _4363 = length(_4362);
                            ls.L /= _4363.xxx;
                            float _4374 = dot(ls.L, _4355);
                            float cos_theta = _4374;
                            float3 _4377;
                            if (_4374 >= 0.0f)
                            {
                                _4377 = -_4355;
                            }
                            else
                            {
                                _4377 = _4355;
                            }
                            float3 param_12 = _4333;
                            float3 param_13 = _4377;
                            ls.lp = offset_ray(param_12, param_13);
                            float _4390 = cos_theta;
                            float _4391 = abs(_4390);
                            cos_theta = _4391;
                            [flatten]
                            if (_4391 > 0.0f)
                            {
                                ls.pdf = (_4363 * _4363) / (ls.area * cos_theta);
                            }
                            material_t _4429;
                            [unroll]
                            for (int _59ident = 0; _59ident < 5; _59ident++)
                            {
                                _4429.textures[_59ident] = _4415.Load(_59ident * 4 + ((_4419.Load(_4054 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
                            }
                            [unroll]
                            for (int _60ident = 0; _60ident < 3; _60ident++)
                            {
                                _4429.base_color[_60ident] = asfloat(_4415.Load(_60ident * 4 + ((_4419.Load(_4054 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
                            }
                            _4429.flags = _4415.Load(((_4419.Load(_4054 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
                            _4429.type = _4415.Load(((_4419.Load(_4054 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
                            _4429.tangent_rotation_or_strength = asfloat(_4415.Load(((_4419.Load(_4054 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
                            _4429.roughness_and_anisotropic = _4415.Load(((_4419.Load(_4054 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
                            _4429.ior = asfloat(_4415.Load(((_4419.Load(_4054 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
                            _4429.sheen_and_sheen_tint = _4415.Load(((_4419.Load(_4054 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
                            _4429.tint_and_metallic = _4415.Load(((_4419.Load(_4054 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
                            _4429.transmission_and_transmission_roughness = _4415.Load(((_4419.Load(_4054 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
                            _4429.specular_and_specular_tint = _4415.Load(((_4419.Load(_4054 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
                            _4429.clearcoat_and_clearcoat_roughness = _4415.Load(((_4419.Load(_4054 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
                            _4429.normal_map_strength_unorm = _4415.Load(((_4419.Load(_4054 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
                            if (_4429.textures[1] != 4294967295u)
                            {
                                ls.col *= SampleBilinear(_4429.textures[1], (float2(_4099.t[0][0], _4099.t[0][1]) * _4297) + (((float2(_4148.t[0][0], _4148.t[0][1]) * _4302) + (float2(_4194.t[0][0], _4194.t[0][1]) * _4293)) * _4284), 0, _3349).xyz;
                            }
                        }
                        else
                        {
                            [branch]
                            if (_3354 == 7u)
                            {
                                float _4515 = frac(asfloat(_3252.Load((hi + 4) * 4 + 0)) + sample_off.x);
                                float _4524 = frac(asfloat(_3252.Load((hi + 5) * 4 + 0)) + sample_off.y);
                                float4 dir_and_pdf;
                                if (_3268_g_params.env_qtree_levels > 0)
                                {
                                    dir_and_pdf = Sample_EnvQTree(_3268_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3268_g_params.env_qtree_levels, mad(_3261, _3273, -float(_3280)), _4515, _4524);
                                }
                                else
                                {
                                    float _4543 = 6.283185482025146484375f * _4524;
                                    float _4555 = sqrt(mad(-_4515, _4515, 1.0f));
                                    float3 param_14 = T;
                                    float3 param_15 = B;
                                    float3 param_16 = N;
                                    float3 param_17 = float3(_4555 * cos(_4543), _4555 * sin(_4543), _4515);
                                    dir_and_pdf = float4(world_from_tangent(param_14, param_15, param_16, param_17), 0.15915493667125701904296875f);
                                }
                                ls.L = dir_and_pdf.xyz;
                                ls.col *= _3268_g_params.env_col.xyz;
                                uint _4594 = asuint(_3268_g_params.env_col.w);
                                if (_4594 != 4294967295u)
                                {
                                    ls.col *= SampleLatlong_RGBE(_4594, ls.L, _3268_g_params.env_rotation, _3349);
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
    uint _995 = index & 16777215u;
    uint _999_dummy_parameter;
    return int2(spvTextureSize(g_textures[NonUniformResourceIndex(_995)], uint(0), _999_dummy_parameter));
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
    float _2247 = 1.0f / mad(0.904129683971405029296875f, roughness, 3.1415927410125732421875f);
    float _2259 = max(dot(N, L), 0.0f);
    float _2264 = max(dot(N, V), 0.0f);
    float _2272 = mad(-_2259, _2264, dot(L, V));
    float t = _2272;
    if (_2272 > 0.0f)
    {
        t /= (max(_2259, _2264) + 1.1754943508222875079687365372222e-38f);
    }
    return float4(base_color * (_2259 * mad(roughness * _2247, t, _2247)), 0.15915493667125701904296875f);
}

float3 Evaluate_DiffuseNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _8733;
    do
    {
        float3 param = -float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param_1 = surf.N;
        float3 param_2 = ls.L;
        float param_3 = roughness;
        float3 param_4 = base_color;
        float4 _5039 = Evaluate_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_5 = ls.pdf;
            float param_6 = _5039.w;
            mis_weight = power_heuristic(param_5, param_6);
        }
        float3 _5062 = (ls.col * _5039.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8733 = _5062;
            break;
        }
        float3 param_7 = surf.P;
        float3 param_8 = surf.plane_N;
        float3 _5074 = offset_ray(param_7, param_8);
        sh_r.o[0] = _5074.x;
        sh_r.o[1] = _5074.y;
        sh_r.o[2] = _5074.z;
        sh_r.c[0] = ray.c[0] * _5062.x;
        sh_r.c[1] = ray.c[1] * _5062.y;
        sh_r.c[2] = ray.c[2] * _5062.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8733 = 0.0f.xxx;
        break;
    } while(false);
    return _8733;
}

float4 Sample_OrenDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float rand_u, float rand_v, inout float3 out_V)
{
    float _2306 = 6.283185482025146484375f * rand_v;
    float _2318 = sqrt(mad(-rand_u, rand_u, 1.0f));
    float3 param = T;
    float3 param_1 = B;
    float3 param_2 = N;
    float3 param_3 = float3(_2318 * cos(_2306), _2318 * sin(_2306), rand_u);
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
    float4 _5325 = Sample_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8);
    new_ray.depth = ray.depth + 1;
    float3 param_9 = surf.P;
    float3 param_10 = surf.plane_N;
    float3 _5335 = offset_ray(param_9, param_10);
    new_ray.o[0] = _5335.x;
    new_ray.o[1] = _5335.y;
    new_ray.o[2] = _5335.z;
    new_ray.d[0] = param_8.x;
    new_ray.d[1] = param_8.y;
    new_ray.d[2] = param_8.z;
    new_ray.c[0] = ((ray.c[0] * _5325.x) * mix_weight) / _5325.w;
    new_ray.c[1] = ((ray.c[1] * _5325.y) * mix_weight) / _5325.w;
    new_ray.c[2] = ((ray.c[2] * _5325.z) * mix_weight) / _5325.w;
    new_ray.pdf = _5325.w;
}

float3 tangent_from_world(float3 T, float3 B, float3 N, float3 V)
{
    return float3(dot(V, T), dot(V, B), dot(V, N));
}

float D_GGX(float3 H, float alpha_x, float alpha_y)
{
    float _8786;
    do
    {
        if (H.z == 0.0f)
        {
            _8786 = 0.0f;
            break;
        }
        float _1973 = (-H.x) / (H.z * alpha_x);
        float _1979 = (-H.y) / (H.z * alpha_y);
        float _1988 = mad(_1979, _1979, mad(_1973, _1973, 1.0f));
        _8786 = 1.0f / (((((_1988 * _1988) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _8786;
}

float G1(float3 Ve, inout float alpha_x, inout float alpha_y)
{
    alpha_x *= alpha_x;
    alpha_y *= alpha_y;
    return 1.0f / mad((-1.0f) + sqrt(1.0f + (mad(alpha_x * Ve.x, Ve.x, (alpha_y * Ve.y) * Ve.y) / (Ve.z * Ve.z))), 0.5f, 1.0f);
}

float4 Evaluate_GGXSpecular_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float alpha_x, float alpha_y, float spec_ior, float spec_F0, float3 spec_col)
{
    float _2488 = D_GGX(sampled_normal_ts, alpha_x, alpha_y);
    float3 param = view_dir_ts;
    float param_1 = alpha_x;
    float param_2 = alpha_y;
    float _2496 = G1(param, param_1, param_2);
    float3 param_3 = reflected_dir_ts;
    float param_4 = alpha_x;
    float param_5 = alpha_y;
    float _2503 = G1(param_3, param_4, param_5);
    float param_6 = dot(view_dir_ts, sampled_normal_ts);
    float param_7 = spec_ior;
    float3 F = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param_6, param_7) - spec_F0) / (1.0f - spec_F0)).xxx);
    float _2531 = 4.0f * abs(view_dir_ts.z * reflected_dir_ts.z);
    float _2534;
    if (_2531 != 0.0f)
    {
        _2534 = (_2488 * (_2496 * _2503)) / _2531;
    }
    else
    {
        _2534 = 0.0f;
    }
    F *= _2534;
    float3 param_8 = view_dir_ts;
    float param_9 = alpha_x;
    float param_10 = alpha_y;
    float _2554 = G1(param_8, param_9, param_10);
    float pdf = ((_2488 * _2554) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2569 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2569 != 0.0f)
    {
        pdf /= _2569;
    }
    float3 _2580 = F;
    float3 _2581 = _2580 * max(reflected_dir_ts.z, 0.0f);
    F = _2581;
    return float4(_2581, pdf);
}

float3 Evaluate_GlossyNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _8738;
    do
    {
        float3 _5110 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5110;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - _5110);
        float _5148 = roughness * roughness;
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = _5148;
        float param_16 = _5148;
        float param_17 = spec_ior;
        float param_18 = spec_F0;
        float3 param_19 = base_color;
        float4 _5163 = Evaluate_GGXSpecular_BSDF(param_12, param_13, param_14, param_15, param_16, param_17, param_18, param_19);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_20 = ls.pdf;
            float param_21 = _5163.w;
            mis_weight = power_heuristic(param_20, param_21);
        }
        float3 _5186 = (ls.col * _5163.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8738 = _5186;
            break;
        }
        float3 param_22 = surf.P;
        float3 param_23 = surf.plane_N;
        float3 _5198 = offset_ray(param_22, param_23);
        sh_r.o[0] = _5198.x;
        sh_r.o[1] = _5198.y;
        sh_r.o[2] = _5198.z;
        sh_r.c[0] = ray.c[0] * _5186.x;
        sh_r.c[1] = ray.c[1] * _5186.y;
        sh_r.c[2] = ray.c[2] * _5186.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8738 = 0.0f.xxx;
        break;
    } while(false);
    return _8738;
}

float3 SampleGGX_VNDF(float3 Ve, float alpha_x, float alpha_y, float U1, float U2)
{
    float3 _1791 = normalize(float3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
    float _1794 = _1791.x;
    float _1799 = _1791.y;
    float _1803 = mad(_1794, _1794, _1799 * _1799);
    float3 _1807;
    if (_1803 > 0.0f)
    {
        _1807 = float3(-_1799, _1794, 0.0f) / sqrt(_1803).xxx;
    }
    else
    {
        _1807 = float3(1.0f, 0.0f, 0.0f);
    }
    float _1829 = sqrt(U1);
    float _1832 = 6.283185482025146484375f * U2;
    float _1837 = _1829 * cos(_1832);
    float _1846 = 1.0f + _1791.z;
    float _1853 = mad(-_1837, _1837, 1.0f);
    float _1859 = mad(mad(-0.5f, _1846, 1.0f), sqrt(_1853), (0.5f * _1846) * (_1829 * sin(_1832)));
    float3 _1880 = ((_1807 * _1837) + (cross(_1791, _1807) * _1859)) + (_1791 * sqrt(max(0.0f, mad(-_1859, _1859, _1853))));
    return normalize(float3(alpha_x * _1880.x, alpha_y * _1880.y, max(0.0f, _1880.z)));
}

float4 Sample_GGXSpecular_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float anisotropic, float spec_ior, float spec_F0, float3 spec_col, float rand_u, float rand_v, inout float3 out_V)
{
    float4 _8758;
    do
    {
        float _2591 = roughness * roughness;
        float _2595 = sqrt(mad(-0.89999997615814208984375f, anisotropic, 1.0f));
        float _2599 = _2591 / _2595;
        float _2603 = _2591 * _2595;
        [branch]
        if ((_2599 * _2603) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2614 = reflect(I, N);
            float param = dot(_2614, N);
            float param_1 = spec_ior;
            float3 _2628 = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param, param_1) - spec_F0) / (1.0f - spec_F0)).xxx);
            out_V = _2614;
            _8758 = float4(_2628.x * 1000000.0f, _2628.y * 1000000.0f, _2628.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2653 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = _2599;
        float param_7 = _2603;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2662 = SampleGGX_VNDF(_2653, param_6, param_7, param_8, param_9);
        float3 _2673 = normalize(reflect(-_2653, _2662));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2673;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2653;
        float3 param_15 = _2662;
        float3 param_16 = _2673;
        float param_17 = _2599;
        float param_18 = _2603;
        float param_19 = spec_ior;
        float param_20 = spec_F0;
        float3 param_21 = spec_col;
        _8758 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _8758;
}

void Sample_GlossyNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 param_4;
    float4 _5245 = Sample_GGXSpecular_BSDF(param, param_1, param_2, param_3, roughness, 0.0f, spec_ior, spec_F0, base_color, rand_u, rand_v, param_4);
    new_ray.depth = ray.depth + 256;
    float3 param_5 = surf.P;
    float3 param_6 = surf.plane_N;
    float3 _5256 = offset_ray(param_5, param_6);
    new_ray.o[0] = _5256.x;
    new_ray.o[1] = _5256.y;
    new_ray.o[2] = _5256.z;
    new_ray.d[0] = param_4.x;
    new_ray.d[1] = param_4.y;
    new_ray.d[2] = param_4.z;
    new_ray.c[0] = ((ray.c[0] * _5245.x) * mix_weight) / _5245.w;
    new_ray.c[1] = ((ray.c[1] * _5245.y) * mix_weight) / _5245.w;
    new_ray.c[2] = ((ray.c[2] * _5245.z) * mix_weight) / _5245.w;
    new_ray.pdf = _5245.w;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _8763;
    do
    {
        bool _2895 = refr_dir_ts.z >= 0.0f;
        bool _2902;
        if (!_2895)
        {
            _2902 = view_dir_ts.z <= 0.0f;
        }
        else
        {
            _2902 = _2895;
        }
        if (_2902)
        {
            _8763 = 0.0f.xxxx;
            break;
        }
        float _2911 = D_GGX(sampled_normal_ts, roughness2, roughness2);
        float3 param = refr_dir_ts;
        float param_1 = roughness2;
        float param_2 = roughness2;
        float _2919 = G1(param, param_1, param_2);
        float3 param_3 = view_dir_ts;
        float param_4 = roughness2;
        float param_5 = roughness2;
        float _2927 = G1(param_3, param_4, param_5);
        float _2937 = mad(dot(view_dir_ts, sampled_normal_ts), eta, dot(refr_dir_ts, sampled_normal_ts));
        float _2947 = clamp(-dot(refr_dir_ts, sampled_normal_ts), 0.0f, 1.0f) / (_2937 * _2937);
        _8763 = float4(refr_col * (((((_2911 * _2927) * _2919) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _2947) / view_dir_ts.z), (((_2911 * _2919) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _2947) / view_dir_ts.z);
        break;
    } while(false);
    return _8763;
}

float3 Evaluate_RefractiveNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness2, float eta, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _8743;
    do
    {
        float3 _5388 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5388;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - (_5388 * eta));
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = roughness2;
        float param_16 = eta;
        float3 param_17 = base_color;
        float4 _5436 = Evaluate_GGXRefraction_BSDF(param_12, param_13, param_14, param_15, param_16, param_17);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_18 = ls.pdf;
            float param_19 = _5436.w;
            mis_weight = power_heuristic(param_18, param_19);
        }
        float3 _5459 = (ls.col * _5436.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8743 = _5459;
            break;
        }
        float3 param_20 = surf.P;
        float3 param_21 = -surf.plane_N;
        float3 _5472 = offset_ray(param_20, param_21);
        sh_r.o[0] = _5472.x;
        sh_r.o[1] = _5472.y;
        sh_r.o[2] = _5472.z;
        sh_r.c[0] = ray.c[0] * _5459.x;
        sh_r.c[1] = ray.c[1] * _5459.y;
        sh_r.c[2] = ray.c[2] * _5459.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8743 = 0.0f.xxx;
        break;
    } while(false);
    return _8743;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _8768;
    do
    {
        float _2991 = roughness * roughness;
        [branch]
        if ((_2991 * _2991) < 1.0000000116860974230803549289703e-07f)
        {
            float _3001 = dot(I, N);
            float _3002 = -_3001;
            float _3012 = mad(-(eta * eta), mad(_3001, _3002, 1.0f), 1.0f);
            if (_3012 < 0.0f)
            {
                _8768 = 0.0f.xxxx;
                break;
            }
            float _3024 = mad(eta, _3002, -sqrt(_3012));
            out_V = float4(normalize((I * eta) + (N * _3024)), _3024);
            _8768 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param = T;
        float3 param_1 = B;
        float3 param_2 = N;
        float3 param_3 = -I;
        float3 _3064 = normalize(tangent_from_world(param, param_1, param_2, param_3));
        float param_4 = _2991;
        float param_5 = _2991;
        float param_6 = rand_u;
        float param_7 = rand_v;
        float3 _3075 = SampleGGX_VNDF(_3064, param_4, param_5, param_6, param_7);
        float _3079 = dot(_3064, _3075);
        float _3089 = mad(-(eta * eta), mad(-_3079, _3079, 1.0f), 1.0f);
        if (_3089 < 0.0f)
        {
            _8768 = 0.0f.xxxx;
            break;
        }
        float _3101 = mad(eta, _3079, -sqrt(_3089));
        float3 _3111 = normalize((_3064 * (-eta)) + (_3075 * _3101));
        float3 param_8 = _3064;
        float3 param_9 = _3075;
        float3 param_10 = _3111;
        float param_11 = _2991;
        float param_12 = eta;
        float3 param_13 = refr_col;
        float3 param_14 = T;
        float3 param_15 = B;
        float3 param_16 = N;
        float3 param_17 = _3111;
        out_V = float4(world_from_tangent(param_14, param_15, param_16, param_17), _3101);
        _8768 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _8768;
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
    float _2037 = old_value;
    old_value = new_value;
    return _2037;
}

float pop_ior_stack(inout float stack[4], float default_value)
{
    float _8776;
    do
    {
        if (stack[3] > 0.0f)
        {
            float param = stack[3];
            float param_1 = -1.0f;
            float _2079 = exchange(param, param_1);
            stack[3] = param;
            _8776 = _2079;
            break;
        }
        if (stack[2] > 0.0f)
        {
            float param_2 = stack[2];
            float param_3 = -1.0f;
            float _2092 = exchange(param_2, param_3);
            stack[2] = param_2;
            _8776 = _2092;
            break;
        }
        if (stack[1] > 0.0f)
        {
            float param_4 = stack[1];
            float param_5 = -1.0f;
            float _2105 = exchange(param_4, param_5);
            stack[1] = param_4;
            _8776 = _2105;
            break;
        }
        if (stack[0] > 0.0f)
        {
            float param_6 = stack[0];
            float param_7 = -1.0f;
            float _2118 = exchange(param_6, param_7);
            stack[0] = param_6;
            _8776 = _2118;
            break;
        }
        _8776 = default_value;
        break;
    } while(false);
    return _8776;
}

void Sample_RefractiveNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, bool is_backfacing, float int_ior, float ext_ior, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float _5509;
    if (is_backfacing)
    {
        _5509 = int_ior / ext_ior;
    }
    else
    {
        _5509 = ext_ior / int_ior;
    }
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float param_4 = roughness;
    float param_5 = _5509;
    float3 param_6 = base_color;
    float param_7 = rand_u;
    float param_8 = rand_v;
    float4 param_9;
    float4 _5533 = Sample_GGXRefraction_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9);
    new_ray.depth = ray.depth + 65536;
    new_ray.c[0] = ((ray.c[0] * _5533.x) * mix_weight) / _5533.w;
    new_ray.c[1] = ((ray.c[1] * _5533.y) * mix_weight) / _5533.w;
    new_ray.c[2] = ((ray.c[2] * _5533.z) * mix_weight) / _5533.w;
    new_ray.pdf = _5533.w;
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
        float _5589 = pop_ior_stack(param_11, param_12);
        new_ray.ior = param_11;
    }
    float3 param_13 = surf.P;
    float3 param_14 = -surf.plane_N;
    float3 _5598 = offset_ray(param_13, param_14);
    new_ray.o[0] = _5598.x;
    new_ray.o[1] = _5598.y;
    new_ray.o[2] = _5598.z;
    new_ray.d[0] = param_9.x;
    new_ray.d[1] = param_9.y;
    new_ray.d[2] = param_9.z;
}

lobe_weights_t get_lobe_weights(float base_color_lum, float spec_color_lum, float specular, float metallic, float transmission, float clearcoat)
{
    float _1443 = 1.0f - metallic;
    float _8862 = (base_color_lum * _1443) * (1.0f - transmission);
    float _1450 = transmission * _1443;
    float _1454;
    if ((specular != 0.0f) || (metallic != 0.0f))
    {
        _1454 = spec_color_lum * mad(-transmission, _1443, 1.0f);
    }
    else
    {
        _1454 = 0.0f;
    }
    float _8863 = _1454;
    float _1464 = 0.25f * clearcoat;
    float _8864 = _1464 * _1443;
    float _8865 = _1450 * base_color_lum;
    float _1473 = _8862;
    float _1482 = mad(_1450, base_color_lum, mad(_1464, _1443, _1473 + _1454));
    if (_1482 != 0.0f)
    {
        _8862 /= _1482;
        _8863 /= _1482;
        _8864 /= _1482;
        _8865 /= _1482;
    }
    lobe_weights_t _8870 = { _8862, _8863, _8864, _8865 };
    return _8870;
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
    float _8791;
    do
    {
        float _2199 = dot(N, L);
        if (_2199 <= 0.0f)
        {
            _8791 = 0.0f;
            break;
        }
        float param = _2199;
        float param_1 = dot(N, V);
        float _2220 = dot(L, H);
        float _2228 = mad((2.0f * _2220) * _2220, roughness, 0.5f);
        _8791 = lerp(1.0f, _2228, schlick_weight(param)) * lerp(1.0f, _2228, schlick_weight(param_1));
        break;
    } while(false);
    return _8791;
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
    float3 _2369 = normalize(L + V);
    float3 H = _2369;
    if (dot(V, _2369) < 0.0f)
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
    float3 _2404 = diff_col;
    float3 _2405 = _2404 + (sheen_color * (3.1415927410125732421875f * schlick_weight(param_5)));
    diff_col = _2405;
    return float4(_2405, pdf);
}

float D_GTR1(float NDotH, float a)
{
    float _8796;
    do
    {
        if (a >= 1.0f)
        {
            _8796 = 0.3183098733425140380859375f;
            break;
        }
        float _1947 = mad(a, a, -1.0f);
        _8796 = _1947 / ((3.1415927410125732421875f * log(a * a)) * mad(_1947 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _8796;
}

float4 Evaluate_PrincipledClearcoat_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float clearcoat_roughness2, float clearcoat_ior, float clearcoat_F0)
{
    float param = sampled_normal_ts.z;
    float param_1 = clearcoat_roughness2;
    float _2705 = D_GTR1(param, param_1);
    float3 param_2 = view_dir_ts;
    float param_3 = 0.0625f;
    float param_4 = 0.0625f;
    float _2712 = G1(param_2, param_3, param_4);
    float3 param_5 = reflected_dir_ts;
    float param_6 = 0.0625f;
    float param_7 = 0.0625f;
    float _2717 = G1(param_5, param_6, param_7);
    float param_8 = dot(reflected_dir_ts, sampled_normal_ts);
    float param_9 = clearcoat_ior;
    float F = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param_8, param_9) - clearcoat_F0) / (1.0f - clearcoat_F0));
    float _2744 = (4.0f * abs(view_dir_ts.z)) * abs(reflected_dir_ts.z);
    float _2747;
    if (_2744 != 0.0f)
    {
        _2747 = (_2705 * (_2712 * _2717)) / _2744;
    }
    else
    {
        _2747 = 0.0f;
    }
    F *= _2747;
    float3 param_10 = view_dir_ts;
    float param_11 = 0.0625f;
    float param_12 = 0.0625f;
    float _2765 = G1(param_10, param_11, param_12);
    float pdf = ((_2705 * _2765) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2780 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2780 != 0.0f)
    {
        pdf /= _2780;
    }
    float _2791 = F;
    float _2792 = _2791 * clamp(reflected_dir_ts.z, 0.0f, 1.0f);
    F = _2792;
    return float4(_2792, _2792, _2792, pdf);
}

float3 Evaluate_PrincipledNode(light_sample_t ls, ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float N_dot_L, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _8748;
    do
    {
        float3 _5621 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 lcol = 0.0f.xxx;
        float bsdf_pdf = 0.0f;
        bool _5626 = N_dot_L > 0.0f;
        [branch]
        if ((lobe_weights.diffuse > 1.0000000116860974230803549289703e-07f) && _5626)
        {
            float3 param = -_5621;
            float3 param_1 = surf.N;
            float3 param_2 = ls.L;
            float param_3 = diff.roughness;
            float3 param_4 = diff.base_color;
            float3 param_5 = diff.sheen_color;
            bool param_6 = false;
            float4 _5645 = Evaluate_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6);
            bsdf_pdf = mad(lobe_weights.diffuse, _5645.w, bsdf_pdf);
            lcol += (((ls.col * N_dot_L) * (_5645 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * ls.pdf).xxx);
        }
        float3 H;
        [flatten]
        if (_5626)
        {
            H = normalize(ls.L - _5621);
        }
        else
        {
            H = normalize(ls.L - (_5621 * trans.eta));
        }
        float _5684 = spec.roughness * spec.roughness;
        float _5689 = sqrt(mad(-0.89999997615814208984375f, spec.anisotropy, 1.0f));
        float _5693 = _5684 / _5689;
        float _5697 = _5684 * _5689;
        float3 param_7 = surf.T;
        float3 param_8 = surf.B;
        float3 param_9 = surf.N;
        float3 param_10 = -_5621;
        float3 _5708 = tangent_from_world(param_7, param_8, param_9, param_10);
        float3 param_11 = surf.T;
        float3 param_12 = surf.B;
        float3 param_13 = surf.N;
        float3 param_14 = ls.L;
        float3 _5718 = tangent_from_world(param_11, param_12, param_13, param_14);
        float3 param_15 = surf.T;
        float3 param_16 = surf.B;
        float3 param_17 = surf.N;
        float3 param_18 = H;
        float3 _5728 = tangent_from_world(param_15, param_16, param_17, param_18);
        bool _5730 = lobe_weights.specular > 0.0f;
        bool _5737;
        if (_5730)
        {
            _5737 = (_5693 * _5697) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _5737 = _5730;
        }
        [branch]
        if (_5737 && _5626)
        {
            float3 param_19 = _5708;
            float3 param_20 = _5728;
            float3 param_21 = _5718;
            float param_22 = _5693;
            float param_23 = _5697;
            float param_24 = spec.ior;
            float param_25 = spec.F0;
            float3 param_26 = spec.tmp_col;
            float4 _5759 = Evaluate_GGXSpecular_BSDF(param_19, param_20, param_21, param_22, param_23, param_24, param_25, param_26);
            bsdf_pdf = mad(lobe_weights.specular, _5759.w, bsdf_pdf);
            lcol += ((ls.col * _5759.xyz) / ls.pdf.xxx);
        }
        float _5778 = coat.roughness * coat.roughness;
        bool _5780 = lobe_weights.clearcoat > 0.0f;
        bool _5787;
        if (_5780)
        {
            _5787 = (_5778 * _5778) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _5787 = _5780;
        }
        [branch]
        if (_5787 && _5626)
        {
            float3 param_27 = _5708;
            float3 param_28 = _5728;
            float3 param_29 = _5718;
            float param_30 = _5778;
            float param_31 = coat.ior;
            float param_32 = coat.F0;
            float4 _5805 = Evaluate_PrincipledClearcoat_BSDF(param_27, param_28, param_29, param_30, param_31, param_32);
            bsdf_pdf = mad(lobe_weights.clearcoat, _5805.w, bsdf_pdf);
            lcol += (((ls.col * 0.25f) * _5805.xyz) / ls.pdf.xxx);
        }
        [branch]
        if (lobe_weights.refraction > 0.0f)
        {
            bool _5827 = trans.fresnel != 0.0f;
            bool _5834;
            if (_5827)
            {
                _5834 = (_5684 * _5684) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _5834 = _5827;
            }
            [branch]
            if (_5834 && _5626)
            {
                float3 param_33 = _5708;
                float3 param_34 = _5728;
                float3 param_35 = _5718;
                float param_36 = _5684;
                float param_37 = _5684;
                float param_38 = 1.0f;
                float param_39 = 0.0f;
                float3 param_40 = 1.0f.xxx;
                float4 _5853 = Evaluate_GGXSpecular_BSDF(param_33, param_34, param_35, param_36, param_37, param_38, param_39, param_40);
                bsdf_pdf = mad(lobe_weights.refraction * trans.fresnel, _5853.w, bsdf_pdf);
                lcol += ((ls.col * _5853.xyz) * (trans.fresnel / ls.pdf));
            }
            float _5875 = trans.roughness * trans.roughness;
            bool _5877 = trans.fresnel != 1.0f;
            bool _5884;
            if (_5877)
            {
                _5884 = (_5875 * _5875) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _5884 = _5877;
            }
            [branch]
            if (_5884 && (N_dot_L < 0.0f))
            {
                float3 param_41 = _5708;
                float3 param_42 = _5728;
                float3 param_43 = _5718;
                float param_44 = _5875;
                float param_45 = trans.eta;
                float3 param_46 = diff.base_color;
                float4 _5902 = Evaluate_GGXRefraction_BSDF(param_41, param_42, param_43, param_44, param_45, param_46);
                float _5905 = 1.0f - trans.fresnel;
                bsdf_pdf = mad(lobe_weights.refraction * _5905, _5902.w, bsdf_pdf);
                lcol += ((ls.col * _5902.xyz) * (_5905 / ls.pdf));
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
            _8748 = lcol;
            break;
        }
        float3 _5945;
        if (N_dot_L < 0.0f)
        {
            _5945 = -surf.plane_N;
        }
        else
        {
            _5945 = surf.plane_N;
        }
        float3 param_49 = surf.P;
        float3 param_50 = _5945;
        float3 _5956 = offset_ray(param_49, param_50);
        sh_r.o[0] = _5956.x;
        sh_r.o[1] = _5956.y;
        sh_r.o[2] = _5956.z;
        sh_r.c[0] = ray.c[0] * lcol.x;
        sh_r.c[1] = ray.c[1] * lcol.y;
        sh_r.c[2] = ray.c[2] * lcol.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8748 = 0.0f.xxx;
        break;
    } while(false);
    return _8748;
}

float4 Sample_PrincipledDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float3 sheen_color, bool uniform_sampling, float rand_u, float rand_v, inout float3 out_V)
{
    float _2416 = 6.283185482025146484375f * rand_v;
    float _2419 = cos(_2416);
    float _2422 = sin(_2416);
    float3 V;
    if (uniform_sampling)
    {
        float _2431 = sqrt(mad(-rand_u, rand_u, 1.0f));
        V = float3(_2431 * _2419, _2431 * _2422, rand_u);
    }
    else
    {
        float _2444 = sqrt(rand_u);
        V = float3(_2444 * _2419, _2444 * _2422, sqrt(1.0f - rand_u));
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
    float4 _8781;
    do
    {
        [branch]
        if ((clearcoat_roughness2 * clearcoat_roughness2) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2809 = reflect(I, N);
            float param = dot(_2809, N);
            float param_1 = clearcoat_ior;
            out_V = _2809;
            float _2828 = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param, param_1) - clearcoat_F0) / (1.0f - clearcoat_F0)) * 1000000.0f;
            _8781 = float4(_2828, _2828, _2828, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2846 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = clearcoat_roughness2;
        float param_7 = clearcoat_roughness2;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2857 = SampleGGX_VNDF(_2846, param_6, param_7, param_8, param_9);
        float3 _2868 = normalize(reflect(-_2846, _2857));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2868;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2846;
        float3 param_15 = _2857;
        float3 param_16 = _2868;
        float param_17 = clearcoat_roughness2;
        float param_18 = clearcoat_ior;
        float param_19 = clearcoat_F0;
        _8781 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _8781;
}

void Sample_PrincipledNode(ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float rand_u, float rand_v, inout float mix_rand, float mix_weight, inout ray_data_t new_ray)
{
    float3 _5991 = float3(ray.d[0], ray.d[1], ray.d[2]);
    int _5995 = ray.depth & 255;
    int _5999 = (ray.depth >> 8) & 255;
    int _6003 = (ray.depth >> 16) & 255;
    int _6014 = (_5995 + _5999) + _6003;
    [branch]
    if (mix_rand < lobe_weights.diffuse)
    {
        bool _6023 = _5995 < _3268_g_params.max_diff_depth;
        bool _6030;
        if (_6023)
        {
            _6030 = _6014 < _3268_g_params.max_total_depth;
        }
        else
        {
            _6030 = _6023;
        }
        if (_6030)
        {
            float3 param = surf.T;
            float3 param_1 = surf.B;
            float3 param_2 = surf.N;
            float3 param_3 = _5991;
            float param_4 = diff.roughness;
            float3 param_5 = diff.base_color;
            float3 param_6 = diff.sheen_color;
            bool param_7 = false;
            float param_8 = rand_u;
            float param_9 = rand_v;
            float3 param_10;
            float4 _6053 = Sample_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9, param_10);
            float3 _6058 = _6053.xyz * (1.0f - metallic);
            new_ray.depth = ray.depth + 1;
            float3 param_11 = surf.P;
            float3 param_12 = surf.plane_N;
            float3 _6073 = offset_ray(param_11, param_12);
            new_ray.o[0] = _6073.x;
            new_ray.o[1] = _6073.y;
            new_ray.o[2] = _6073.z;
            new_ray.d[0] = param_10.x;
            new_ray.d[1] = param_10.y;
            new_ray.d[2] = param_10.z;
            new_ray.c[0] = ((ray.c[0] * _6058.x) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[1] = ((ray.c[1] * _6058.y) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[2] = ((ray.c[2] * _6058.z) * mix_weight) / lobe_weights.diffuse;
            new_ray.pdf = _6053.w;
        }
    }
    else
    {
        float _6123 = lobe_weights.diffuse + lobe_weights.specular;
        [branch]
        if (mix_rand < _6123)
        {
            bool _6130 = _5999 < _3268_g_params.max_spec_depth;
            bool _6137;
            if (_6130)
            {
                _6137 = _6014 < _3268_g_params.max_total_depth;
            }
            else
            {
                _6137 = _6130;
            }
            if (_6137)
            {
                float3 param_13 = surf.T;
                float3 param_14 = surf.B;
                float3 param_15 = surf.N;
                float3 param_16 = _5991;
                float3 param_17;
                float4 _6156 = Sample_GGXSpecular_BSDF(param_13, param_14, param_15, param_16, spec.roughness, spec.anisotropy, spec.ior, spec.F0, spec.tmp_col, rand_u, rand_v, param_17);
                float _6161 = _6156.w * lobe_weights.specular;
                new_ray.depth = ray.depth + 256;
                new_ray.c[0] = ((ray.c[0] * _6156.x) * mix_weight) / _6161;
                new_ray.c[1] = ((ray.c[1] * _6156.y) * mix_weight) / _6161;
                new_ray.c[2] = ((ray.c[2] * _6156.z) * mix_weight) / _6161;
                new_ray.pdf = _6161;
                float3 param_18 = surf.P;
                float3 param_19 = surf.plane_N;
                float3 _6201 = offset_ray(param_18, param_19);
                new_ray.o[0] = _6201.x;
                new_ray.o[1] = _6201.y;
                new_ray.o[2] = _6201.z;
                new_ray.d[0] = param_17.x;
                new_ray.d[1] = param_17.y;
                new_ray.d[2] = param_17.z;
            }
        }
        else
        {
            float _6226 = _6123 + lobe_weights.clearcoat;
            [branch]
            if (mix_rand < _6226)
            {
                bool _6233 = _5999 < _3268_g_params.max_spec_depth;
                bool _6240;
                if (_6233)
                {
                    _6240 = _6014 < _3268_g_params.max_total_depth;
                }
                else
                {
                    _6240 = _6233;
                }
                if (_6240)
                {
                    float3 param_20 = surf.T;
                    float3 param_21 = surf.B;
                    float3 param_22 = surf.N;
                    float3 param_23 = _5991;
                    float param_24 = coat.roughness * coat.roughness;
                    float param_25 = coat.ior;
                    float param_26 = coat.F0;
                    float param_27 = rand_u;
                    float param_28 = rand_v;
                    float3 param_29;
                    float4 _6264 = Sample_PrincipledClearcoat_BSDF(param_20, param_21, param_22, param_23, param_24, param_25, param_26, param_27, param_28, param_29);
                    float _6269 = _6264.w * lobe_weights.clearcoat;
                    new_ray.depth = ray.depth + 256;
                    new_ray.c[0] = (((0.25f * ray.c[0]) * _6264.x) * mix_weight) / _6269;
                    new_ray.c[1] = (((0.25f * ray.c[1]) * _6264.y) * mix_weight) / _6269;
                    new_ray.c[2] = (((0.25f * ray.c[2]) * _6264.z) * mix_weight) / _6269;
                    new_ray.pdf = _6269;
                    float3 param_30 = surf.P;
                    float3 param_31 = surf.plane_N;
                    float3 _6312 = offset_ray(param_30, param_31);
                    new_ray.o[0] = _6312.x;
                    new_ray.o[1] = _6312.y;
                    new_ray.o[2] = _6312.z;
                    new_ray.d[0] = param_29.x;
                    new_ray.d[1] = param_29.y;
                    new_ray.d[2] = param_29.z;
                }
            }
            else
            {
                bool _6334 = mix_rand >= trans.fresnel;
                bool _6341;
                if (_6334)
                {
                    _6341 = _6003 < _3268_g_params.max_refr_depth;
                }
                else
                {
                    _6341 = _6334;
                }
                bool _6355;
                if (!_6341)
                {
                    bool _6347 = mix_rand < trans.fresnel;
                    bool _6354;
                    if (_6347)
                    {
                        _6354 = _5999 < _3268_g_params.max_spec_depth;
                    }
                    else
                    {
                        _6354 = _6347;
                    }
                    _6355 = _6354;
                }
                else
                {
                    _6355 = _6341;
                }
                bool _6362;
                if (_6355)
                {
                    _6362 = _6014 < _3268_g_params.max_total_depth;
                }
                else
                {
                    _6362 = _6355;
                }
                [branch]
                if (_6362)
                {
                    mix_rand -= _6226;
                    mix_rand /= lobe_weights.refraction;
                    float4 F;
                    float3 V;
                    [branch]
                    if (mix_rand < trans.fresnel)
                    {
                        float3 param_32 = surf.T;
                        float3 param_33 = surf.B;
                        float3 param_34 = surf.N;
                        float3 param_35 = _5991;
                        float3 param_36;
                        float4 _6392 = Sample_GGXSpecular_BSDF(param_32, param_33, param_34, param_35, spec.roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, rand_u, rand_v, param_36);
                        V = param_36;
                        F = _6392;
                        new_ray.depth = ray.depth + 256;
                        float3 param_37 = surf.P;
                        float3 param_38 = surf.plane_N;
                        float3 _6402 = offset_ray(param_37, param_38);
                        new_ray.o[0] = _6402.x;
                        new_ray.o[1] = _6402.y;
                        new_ray.o[2] = _6402.z;
                    }
                    else
                    {
                        float3 param_39 = surf.T;
                        float3 param_40 = surf.B;
                        float3 param_41 = surf.N;
                        float3 param_42 = _5991;
                        float param_43 = trans.roughness;
                        float param_44 = trans.eta;
                        float3 param_45 = diff.base_color;
                        float param_46 = rand_u;
                        float param_47 = rand_v;
                        float4 param_48;
                        float4 _6431 = Sample_GGXRefraction_BSDF(param_39, param_40, param_41, param_42, param_43, param_44, param_45, param_46, param_47, param_48);
                        F = _6431;
                        V = param_48.xyz;
                        new_ray.depth = ray.depth + 65536;
                        float3 param_49 = surf.P;
                        float3 param_50 = -surf.plane_N;
                        float3 _6444 = offset_ray(param_49, param_50);
                        new_ray.o[0] = _6444.x;
                        new_ray.o[1] = _6444.y;
                        new_ray.o[2] = _6444.z;
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
                            float _6470 = pop_ior_stack(param_52, param_53);
                            new_ray.ior = param_52;
                        }
                    }
                    float4 _10014 = F;
                    float _6476 = _10014.w * lobe_weights.refraction;
                    float4 _10016 = _10014;
                    _10016.w = _6476;
                    F = _10016;
                    new_ray.c[0] = ((ray.c[0] * _10014.x) * mix_weight) / _6476;
                    new_ray.c[1] = ((ray.c[1] * _10014.y) * mix_weight) / _6476;
                    new_ray.c[2] = ((ray.c[2] * _10014.z) * mix_weight) / _6476;
                    new_ray.pdf = _6476;
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
    float3 _8718;
    do
    {
        float3 _6532 = float3(ray.d[0], ray.d[1], ray.d[2]);
        int _6536 = ray.depth & 255;
        int _6541 = (ray.depth >> 8) & 255;
        int _6546 = (ray.depth >> 16) & 255;
        int _6557 = (_6536 + _6541) + _6546;
        int _6565 = _3268_g_params.hi + ((_6557 + ((ray.depth >> 24) & 255)) * 9);
        uint param = uint(hash(ray.xy));
        float _6572 = construct_float(param);
        uint param_1 = uint(hash(hash(ray.xy)));
        float _6579 = construct_float(param_1);
        float2 _6598 = float2(frac(asfloat(_3252.Load((_6565 + 7) * 4 + 0)) + _6572), frac(asfloat(_3252.Load((_6565 + 8) * 4 + 0)) + _6579));
        [branch]
        if (inter.mask == 0)
        {
            ray_data_t param_2 = ray;
            float3 _6608 = Evaluate_EnvColor(param_2, _6598);
            _8718 = float3(ray.c[0] * _6608.x, ray.c[1] * _6608.y, ray.c[2] * _6608.z);
            break;
        }
        float3 _6635 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_6532 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            ray_data_t param_3 = ray;
            hit_data_t param_4 = inter;
            float3 _6648 = Evaluate_LightColor(param_3, param_4, _6598);
            _8718 = float3(ray.c[0] * _6648.x, ray.c[1] * _6648.y, ray.c[2] * _6648.z);
            break;
        }
        bool _6669 = inter.prim_index < 0;
        int _6672;
        if (_6669)
        {
            _6672 = (-1) - inter.prim_index;
        }
        else
        {
            _6672 = inter.prim_index;
        }
        uint _6683 = uint(_6672);
        material_t _6691;
        [unroll]
        for (int _61ident = 0; _61ident < 5; _61ident++)
        {
            _6691.textures[_61ident] = _4415.Load(_61ident * 4 + ((_4419.Load(_6683 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
        }
        [unroll]
        for (int _62ident = 0; _62ident < 3; _62ident++)
        {
            _6691.base_color[_62ident] = asfloat(_4415.Load(_62ident * 4 + ((_4419.Load(_6683 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
        }
        _6691.flags = _4415.Load(((_4419.Load(_6683 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
        _6691.type = _4415.Load(((_4419.Load(_6683 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
        _6691.tangent_rotation_or_strength = asfloat(_4415.Load(((_4419.Load(_6683 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
        _6691.roughness_and_anisotropic = _4415.Load(((_4419.Load(_6683 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
        _6691.ior = asfloat(_4415.Load(((_4419.Load(_6683 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
        _6691.sheen_and_sheen_tint = _4415.Load(((_4419.Load(_6683 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
        _6691.tint_and_metallic = _4415.Load(((_4419.Load(_6683 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
        _6691.transmission_and_transmission_roughness = _4415.Load(((_4419.Load(_6683 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
        _6691.specular_and_specular_tint = _4415.Load(((_4419.Load(_6683 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
        _6691.clearcoat_and_clearcoat_roughness = _4415.Load(((_4419.Load(_6683 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
        _6691.normal_map_strength_unorm = _4415.Load(((_4419.Load(_6683 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
        uint _9469 = _6691.textures[0];
        uint _9470 = _6691.textures[1];
        uint _9471 = _6691.textures[2];
        uint _9472 = _6691.textures[3];
        uint _9473 = _6691.textures[4];
        float _9474 = _6691.base_color[0];
        float _9475 = _6691.base_color[1];
        float _9476 = _6691.base_color[2];
        uint _9079 = _6691.flags;
        uint _9080 = _6691.type;
        float _9081 = _6691.tangent_rotation_or_strength;
        uint _9082 = _6691.roughness_and_anisotropic;
        float _9083 = _6691.ior;
        uint _9084 = _6691.sheen_and_sheen_tint;
        uint _9085 = _6691.tint_and_metallic;
        uint _9086 = _6691.transmission_and_transmission_roughness;
        uint _9087 = _6691.specular_and_specular_tint;
        uint _9088 = _6691.clearcoat_and_clearcoat_roughness;
        uint _9089 = _6691.normal_map_strength_unorm;
        transform_t _6746;
        _6746.xform = asfloat(uint4x4(_4062.Load4(asuint(asfloat(_6739.Load(inter.obj_index * 32 + 12))) * 128 + 0), _4062.Load4(asuint(asfloat(_6739.Load(inter.obj_index * 32 + 12))) * 128 + 16), _4062.Load4(asuint(asfloat(_6739.Load(inter.obj_index * 32 + 12))) * 128 + 32), _4062.Load4(asuint(asfloat(_6739.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _6746.inv_xform = asfloat(uint4x4(_4062.Load4(asuint(asfloat(_6739.Load(inter.obj_index * 32 + 12))) * 128 + 64), _4062.Load4(asuint(asfloat(_6739.Load(inter.obj_index * 32 + 12))) * 128 + 80), _4062.Load4(asuint(asfloat(_6739.Load(inter.obj_index * 32 + 12))) * 128 + 96), _4062.Load4(asuint(asfloat(_6739.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _6753 = _6683 * 3u;
        vertex_t _6758;
        [unroll]
        for (int _63ident = 0; _63ident < 3; _63ident++)
        {
            _6758.p[_63ident] = asfloat(_4087.Load(_63ident * 4 + _4091.Load(_6753 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _64ident = 0; _64ident < 3; _64ident++)
        {
            _6758.n[_64ident] = asfloat(_4087.Load(_64ident * 4 + _4091.Load(_6753 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _65ident = 0; _65ident < 3; _65ident++)
        {
            _6758.b[_65ident] = asfloat(_4087.Load(_65ident * 4 + _4091.Load(_6753 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _66ident = 0; _66ident < 2; _66ident++)
        {
            [unroll]
            for (int _67ident = 0; _67ident < 2; _67ident++)
            {
                _6758.t[_66ident][_67ident] = asfloat(_4087.Load(_67ident * 4 + _66ident * 8 + _4091.Load(_6753 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _6804;
        [unroll]
        for (int _68ident = 0; _68ident < 3; _68ident++)
        {
            _6804.p[_68ident] = asfloat(_4087.Load(_68ident * 4 + _4091.Load((_6753 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _69ident = 0; _69ident < 3; _69ident++)
        {
            _6804.n[_69ident] = asfloat(_4087.Load(_69ident * 4 + _4091.Load((_6753 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _70ident = 0; _70ident < 3; _70ident++)
        {
            _6804.b[_70ident] = asfloat(_4087.Load(_70ident * 4 + _4091.Load((_6753 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _71ident = 0; _71ident < 2; _71ident++)
        {
            [unroll]
            for (int _72ident = 0; _72ident < 2; _72ident++)
            {
                _6804.t[_71ident][_72ident] = asfloat(_4087.Load(_72ident * 4 + _71ident * 8 + _4091.Load((_6753 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _6850;
        [unroll]
        for (int _73ident = 0; _73ident < 3; _73ident++)
        {
            _6850.p[_73ident] = asfloat(_4087.Load(_73ident * 4 + _4091.Load((_6753 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _74ident = 0; _74ident < 3; _74ident++)
        {
            _6850.n[_74ident] = asfloat(_4087.Load(_74ident * 4 + _4091.Load((_6753 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _75ident = 0; _75ident < 3; _75ident++)
        {
            _6850.b[_75ident] = asfloat(_4087.Load(_75ident * 4 + _4091.Load((_6753 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _76ident = 0; _76ident < 2; _76ident++)
        {
            [unroll]
            for (int _77ident = 0; _77ident < 2; _77ident++)
            {
                _6850.t[_76ident][_77ident] = asfloat(_4087.Load(_77ident * 4 + _76ident * 8 + _4091.Load((_6753 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _6896 = float3(_6758.p[0], _6758.p[1], _6758.p[2]);
        float3 _6904 = float3(_6804.p[0], _6804.p[1], _6804.p[2]);
        float3 _6912 = float3(_6850.p[0], _6850.p[1], _6850.p[2]);
        float _6919 = (1.0f - inter.u) - inter.v;
        float3 _6951 = normalize(((float3(_6758.n[0], _6758.n[1], _6758.n[2]) * _6919) + (float3(_6804.n[0], _6804.n[1], _6804.n[2]) * inter.u)) + (float3(_6850.n[0], _6850.n[1], _6850.n[2]) * inter.v));
        float3 _9018 = _6951;
        float2 _6977 = ((float2(_6758.t[0][0], _6758.t[0][1]) * _6919) + (float2(_6804.t[0][0], _6804.t[0][1]) * inter.u)) + (float2(_6850.t[0][0], _6850.t[0][1]) * inter.v);
        float3 _6993 = cross(_6904 - _6896, _6912 - _6896);
        float _6998 = length(_6993);
        float3 _9019 = _6993 / _6998.xxx;
        float3 _7035 = ((float3(_6758.b[0], _6758.b[1], _6758.b[2]) * _6919) + (float3(_6804.b[0], _6804.b[1], _6804.b[2]) * inter.u)) + (float3(_6850.b[0], _6850.b[1], _6850.b[2]) * inter.v);
        float3 _9017 = _7035;
        float3 _9016 = cross(_7035, _6951);
        if (_6669)
        {
            if ((_4419.Load(_6683 * 4 + 0) & 65535u) == 65535u)
            {
                _8718 = 0.0f.xxx;
                break;
            }
            material_t _7061;
            [unroll]
            for (int _78ident = 0; _78ident < 5; _78ident++)
            {
                _7061.textures[_78ident] = _4415.Load(_78ident * 4 + (_4419.Load(_6683 * 4 + 0) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _79ident = 0; _79ident < 3; _79ident++)
            {
                _7061.base_color[_79ident] = asfloat(_4415.Load(_79ident * 4 + (_4419.Load(_6683 * 4 + 0) & 16383u) * 76 + 20));
            }
            _7061.flags = _4415.Load((_4419.Load(_6683 * 4 + 0) & 16383u) * 76 + 32);
            _7061.type = _4415.Load((_4419.Load(_6683 * 4 + 0) & 16383u) * 76 + 36);
            _7061.tangent_rotation_or_strength = asfloat(_4415.Load((_4419.Load(_6683 * 4 + 0) & 16383u) * 76 + 40));
            _7061.roughness_and_anisotropic = _4415.Load((_4419.Load(_6683 * 4 + 0) & 16383u) * 76 + 44);
            _7061.ior = asfloat(_4415.Load((_4419.Load(_6683 * 4 + 0) & 16383u) * 76 + 48));
            _7061.sheen_and_sheen_tint = _4415.Load((_4419.Load(_6683 * 4 + 0) & 16383u) * 76 + 52);
            _7061.tint_and_metallic = _4415.Load((_4419.Load(_6683 * 4 + 0) & 16383u) * 76 + 56);
            _7061.transmission_and_transmission_roughness = _4415.Load((_4419.Load(_6683 * 4 + 0) & 16383u) * 76 + 60);
            _7061.specular_and_specular_tint = _4415.Load((_4419.Load(_6683 * 4 + 0) & 16383u) * 76 + 64);
            _7061.clearcoat_and_clearcoat_roughness = _4415.Load((_4419.Load(_6683 * 4 + 0) & 16383u) * 76 + 68);
            _7061.normal_map_strength_unorm = _4415.Load((_4419.Load(_6683 * 4 + 0) & 16383u) * 76 + 72);
            _9469 = _7061.textures[0];
            _9470 = _7061.textures[1];
            _9471 = _7061.textures[2];
            _9472 = _7061.textures[3];
            _9473 = _7061.textures[4];
            _9474 = _7061.base_color[0];
            _9475 = _7061.base_color[1];
            _9476 = _7061.base_color[2];
            _9079 = _7061.flags;
            _9080 = _7061.type;
            _9081 = _7061.tangent_rotation_or_strength;
            _9082 = _7061.roughness_and_anisotropic;
            _9083 = _7061.ior;
            _9084 = _7061.sheen_and_sheen_tint;
            _9085 = _7061.tint_and_metallic;
            _9086 = _7061.transmission_and_transmission_roughness;
            _9087 = _7061.specular_and_specular_tint;
            _9088 = _7061.clearcoat_and_clearcoat_roughness;
            _9089 = _7061.normal_map_strength_unorm;
            _9019 = -_9019;
            _9018 = -_9018;
            _9017 = -_9017;
            _9016 = -_9016;
        }
        float3 param_5 = _9019;
        float4x4 param_6 = _6746.inv_xform;
        _9019 = TransformNormal(param_5, param_6);
        float3 param_7 = _9018;
        float4x4 param_8 = _6746.inv_xform;
        _9018 = TransformNormal(param_7, param_8);
        float3 param_9 = _9017;
        float4x4 param_10 = _6746.inv_xform;
        _9017 = TransformNormal(param_9, param_10);
        float3 param_11 = _9016;
        float4x4 param_12 = _6746.inv_xform;
        _9019 = normalize(_9019);
        _9018 = normalize(_9018);
        _9017 = normalize(_9017);
        _9016 = normalize(TransformNormal(param_11, param_12));
        float _7201 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _7211 = mad(0.5f, log2(abs(mad(_6804.t[0][0] - _6758.t[0][0], _6850.t[0][1] - _6758.t[0][1], -((_6850.t[0][0] - _6758.t[0][0]) * (_6804.t[0][1] - _6758.t[0][1])))) / _6998), log2(_7201));
        float param_13[4] = ray.ior;
        bool param_14 = _6669;
        float param_15 = 1.0f;
        float _7219 = peek_ior_stack(param_13, param_14, param_15);
        float3 col = 0.0f.xxx;
        float mix_rand = frac(asfloat(_3252.Load(_6565 * 4 + 0)) + _6572);
        float mix_weight = 1.0f;
        float _7258;
        float _7275;
        float _7301;
        float _7368;
        while (_9080 == 4u)
        {
            float mix_val = _9081;
            if (_9470 != 4294967295u)
            {
                mix_val *= SampleBilinear(_9470, _6977, 0, _6598).x;
            }
            if (_6669)
            {
                _7258 = _7219 / _9083;
            }
            else
            {
                _7258 = _9083 / _7219;
            }
            if (_9083 != 0.0f)
            {
                float param_16 = dot(_6532, _9018);
                float param_17 = _7258;
                _7275 = fresnel_dielectric_cos(param_16, param_17);
            }
            else
            {
                _7275 = 1.0f;
            }
            float _7290 = mix_val;
            float _7291 = _7290 * clamp(_7275, 0.0f, 1.0f);
            mix_val = _7291;
            if (mix_rand > _7291)
            {
                if ((_9079 & 2u) != 0u)
                {
                    _7301 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _7301 = 1.0f;
                }
                mix_weight *= _7301;
                material_t _7314;
                [unroll]
                for (int _80ident = 0; _80ident < 5; _80ident++)
                {
                    _7314.textures[_80ident] = _4415.Load(_80ident * 4 + _9472 * 76 + 0);
                }
                [unroll]
                for (int _81ident = 0; _81ident < 3; _81ident++)
                {
                    _7314.base_color[_81ident] = asfloat(_4415.Load(_81ident * 4 + _9472 * 76 + 20));
                }
                _7314.flags = _4415.Load(_9472 * 76 + 32);
                _7314.type = _4415.Load(_9472 * 76 + 36);
                _7314.tangent_rotation_or_strength = asfloat(_4415.Load(_9472 * 76 + 40));
                _7314.roughness_and_anisotropic = _4415.Load(_9472 * 76 + 44);
                _7314.ior = asfloat(_4415.Load(_9472 * 76 + 48));
                _7314.sheen_and_sheen_tint = _4415.Load(_9472 * 76 + 52);
                _7314.tint_and_metallic = _4415.Load(_9472 * 76 + 56);
                _7314.transmission_and_transmission_roughness = _4415.Load(_9472 * 76 + 60);
                _7314.specular_and_specular_tint = _4415.Load(_9472 * 76 + 64);
                _7314.clearcoat_and_clearcoat_roughness = _4415.Load(_9472 * 76 + 68);
                _7314.normal_map_strength_unorm = _4415.Load(_9472 * 76 + 72);
                _9469 = _7314.textures[0];
                _9470 = _7314.textures[1];
                _9471 = _7314.textures[2];
                _9472 = _7314.textures[3];
                _9473 = _7314.textures[4];
                _9474 = _7314.base_color[0];
                _9475 = _7314.base_color[1];
                _9476 = _7314.base_color[2];
                _9079 = _7314.flags;
                _9080 = _7314.type;
                _9081 = _7314.tangent_rotation_or_strength;
                _9082 = _7314.roughness_and_anisotropic;
                _9083 = _7314.ior;
                _9084 = _7314.sheen_and_sheen_tint;
                _9085 = _7314.tint_and_metallic;
                _9086 = _7314.transmission_and_transmission_roughness;
                _9087 = _7314.specular_and_specular_tint;
                _9088 = _7314.clearcoat_and_clearcoat_roughness;
                _9089 = _7314.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_9079 & 2u) != 0u)
                {
                    _7368 = 1.0f / mix_val;
                }
                else
                {
                    _7368 = 1.0f;
                }
                mix_weight *= _7368;
                material_t _7380;
                [unroll]
                for (int _82ident = 0; _82ident < 5; _82ident++)
                {
                    _7380.textures[_82ident] = _4415.Load(_82ident * 4 + _9473 * 76 + 0);
                }
                [unroll]
                for (int _83ident = 0; _83ident < 3; _83ident++)
                {
                    _7380.base_color[_83ident] = asfloat(_4415.Load(_83ident * 4 + _9473 * 76 + 20));
                }
                _7380.flags = _4415.Load(_9473 * 76 + 32);
                _7380.type = _4415.Load(_9473 * 76 + 36);
                _7380.tangent_rotation_or_strength = asfloat(_4415.Load(_9473 * 76 + 40));
                _7380.roughness_and_anisotropic = _4415.Load(_9473 * 76 + 44);
                _7380.ior = asfloat(_4415.Load(_9473 * 76 + 48));
                _7380.sheen_and_sheen_tint = _4415.Load(_9473 * 76 + 52);
                _7380.tint_and_metallic = _4415.Load(_9473 * 76 + 56);
                _7380.transmission_and_transmission_roughness = _4415.Load(_9473 * 76 + 60);
                _7380.specular_and_specular_tint = _4415.Load(_9473 * 76 + 64);
                _7380.clearcoat_and_clearcoat_roughness = _4415.Load(_9473 * 76 + 68);
                _7380.normal_map_strength_unorm = _4415.Load(_9473 * 76 + 72);
                _9469 = _7380.textures[0];
                _9470 = _7380.textures[1];
                _9471 = _7380.textures[2];
                _9472 = _7380.textures[3];
                _9473 = _7380.textures[4];
                _9474 = _7380.base_color[0];
                _9475 = _7380.base_color[1];
                _9476 = _7380.base_color[2];
                _9079 = _7380.flags;
                _9080 = _7380.type;
                _9081 = _7380.tangent_rotation_or_strength;
                _9082 = _7380.roughness_and_anisotropic;
                _9083 = _7380.ior;
                _9084 = _7380.sheen_and_sheen_tint;
                _9085 = _7380.tint_and_metallic;
                _9086 = _7380.transmission_and_transmission_roughness;
                _9087 = _7380.specular_and_specular_tint;
                _9088 = _7380.clearcoat_and_clearcoat_roughness;
                _9089 = _7380.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_9469 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_9469, _6977, 0, _6598).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_9469 & 33554432u) != 0u)
            {
                float3 _10037 = normals;
                _10037.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _10037;
            }
            float3 _7463 = _9018;
            _9018 = normalize(((_9016 * normals.x) + (_7463 * normals.z)) + (_9017 * normals.y));
            if ((_9089 & 65535u) != 65535u)
            {
                _9018 = normalize(_7463 + ((_9018 - _7463) * clamp(float(_9089 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_18 = _9019;
            float3 param_19 = -_6532;
            float3 param_20 = _9018;
            _9018 = ensure_valid_reflection(param_18, param_19, param_20);
        }
        float3 _7529 = ((_6896 * _6919) + (_6904 * inter.u)) + (_6912 * inter.v);
        float3 _7536 = float3(-_7529.z, 0.0f, _7529.x);
        float3 tangent = _7536;
        float3 param_21 = _7536;
        float4x4 param_22 = _6746.inv_xform;
        float3 _7542 = TransformNormal(param_21, param_22);
        tangent = _7542;
        float3 _7546 = cross(_7542, _9018);
        if (dot(_7546, _7546) == 0.0f)
        {
            float3 param_23 = _7529;
            float4x4 param_24 = _6746.inv_xform;
            tangent = TransformNormal(param_23, param_24);
        }
        if (_9081 != 0.0f)
        {
            float3 param_25 = tangent;
            float3 param_26 = _9018;
            float param_27 = _9081;
            tangent = rotate_around_axis(param_25, param_26, param_27);
        }
        float3 _7579 = normalize(cross(tangent, _9018));
        _9017 = _7579;
        _9016 = cross(_9018, _7579);
        float3 _9168 = 0.0f.xxx;
        float3 _9167 = 0.0f.xxx;
        float _9172 = 0.0f;
        float _9170 = 0.0f;
        float _9171 = 1.0f;
        bool _7595 = _3268_g_params.li_count != 0;
        bool _7601;
        if (_7595)
        {
            _7601 = _9080 != 3u;
        }
        else
        {
            _7601 = _7595;
        }
        float3 _9169;
        bool _9173;
        bool _9174;
        if (_7601)
        {
            float3 param_28 = _6635;
            float3 param_29 = _9016;
            float3 param_30 = _9017;
            float3 param_31 = _9018;
            int param_32 = _6565;
            float2 param_33 = float2(_6572, _6579);
            light_sample_t _9183 = { _9167, _9168, _9169, _9170, _9171, _9172, _9173, _9174 };
            light_sample_t param_34 = _9183;
            SampleLightSource(param_28, param_29, param_30, param_31, param_32, param_33, param_34);
            _9167 = param_34.col;
            _9168 = param_34.L;
            _9169 = param_34.lp;
            _9170 = param_34.area;
            _9171 = param_34.dist_mul;
            _9172 = param_34.pdf;
            _9173 = param_34.cast_shadow;
            _9174 = param_34.from_env;
        }
        float _7629 = dot(_9018, _9168);
        float3 base_color = float3(_9474, _9475, _9476);
        [branch]
        if (_9470 != 4294967295u)
        {
            base_color *= SampleBilinear(_9470, _6977, int(get_texture_lod(texSize(_9470), _7211)), _6598, true, true).xyz;
        }
        out_base_color = base_color;
        out_normals = _9018;
        float3 tint_color = 0.0f.xxx;
        float _7666 = lum(base_color);
        [flatten]
        if (_7666 > 0.0f)
        {
            tint_color = base_color / _7666.xxx;
        }
        float roughness = clamp(float(_9082 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_9471 != 4294967295u)
        {
            roughness *= SampleBilinear(_9471, _6977, int(get_texture_lod(texSize(_9471), _7211)), _6598, false, true).x;
        }
        float _7712 = frac(asfloat(_3252.Load((_6565 + 1) * 4 + 0)) + _6572);
        float _7721 = frac(asfloat(_3252.Load((_6565 + 2) * 4 + 0)) + _6579);
        float _9596 = 0.0f;
        float _9595 = 0.0f;
        float _9594 = 0.0f;
        float _9232[4];
        [unroll]
        for (int i = 0; i < 4; )
        {
            _9232[i] = ray.ior[i];
            i++;
            continue;
        }
        float _9233 = _7201;
        float _9234 = ray.cone_spread;
        int _9235 = ray.xy;
        float _9230 = 0.0f;
        float _9701 = 0.0f;
        float _9700 = 0.0f;
        float _9699 = 0.0f;
        int _9337 = ray.depth;
        int _9341 = ray.xy;
        int _9236;
        float _9339;
        float _9524;
        float _9525;
        float _9526;
        float _9559;
        float _9560;
        float _9561;
        float _9629;
        float _9630;
        float _9631;
        float _9664;
        float _9665;
        float _9666;
        [branch]
        if (_9080 == 0u)
        {
            [branch]
            if ((_9172 > 0.0f) && (_7629 > 0.0f))
            {
                light_sample_t _9200 = { _9167, _9168, _9169, _9170, _9171, _9172, _9173, _9174 };
                surface_t _9027 = { _6635, _9016, _9017, _9018, _9019, _6977 };
                float _9705[3] = { _9699, _9700, _9701 };
                float _9670[3] = { _9664, _9665, _9666 };
                float _9635[3] = { _9629, _9630, _9631 };
                shadow_ray_t _9351 = { _9635, _9337, _9670, _9339, _9705, _9341 };
                shadow_ray_t param_35 = _9351;
                float3 _7781 = Evaluate_DiffuseNode(_9200, ray, _9027, base_color, roughness, mix_weight, param_35);
                _9629 = param_35.o[0];
                _9630 = param_35.o[1];
                _9631 = param_35.o[2];
                _9337 = param_35.depth;
                _9664 = param_35.d[0];
                _9665 = param_35.d[1];
                _9666 = param_35.d[2];
                _9339 = param_35.dist;
                _9699 = param_35.c[0];
                _9700 = param_35.c[1];
                _9701 = param_35.c[2];
                _9341 = param_35.xy;
                col += _7781;
            }
            bool _7788 = _6536 < _3268_g_params.max_diff_depth;
            bool _7795;
            if (_7788)
            {
                _7795 = _6557 < _3268_g_params.max_total_depth;
            }
            else
            {
                _7795 = _7788;
            }
            [branch]
            if (_7795)
            {
                surface_t _9034 = { _6635, _9016, _9017, _9018, _9019, _6977 };
                float _9600[3] = { _9594, _9595, _9596 };
                float _9565[3] = { _9559, _9560, _9561 };
                float _9530[3] = { _9524, _9525, _9526 };
                ray_data_t _9250 = { _9530, _9565, _9230, _9600, _9232, _9233, _9234, _9235, _9236 };
                ray_data_t param_36 = _9250;
                Sample_DiffuseNode(ray, _9034, base_color, roughness, _7712, _7721, mix_weight, param_36);
                _9524 = param_36.o[0];
                _9525 = param_36.o[1];
                _9526 = param_36.o[2];
                _9559 = param_36.d[0];
                _9560 = param_36.d[1];
                _9561 = param_36.d[2];
                _9230 = param_36.pdf;
                _9594 = param_36.c[0];
                _9595 = param_36.c[1];
                _9596 = param_36.c[2];
                _9232 = param_36.ior;
                _9233 = param_36.cone_width;
                _9234 = param_36.cone_spread;
                _9235 = param_36.xy;
                _9236 = param_36.depth;
            }
        }
        else
        {
            [branch]
            if (_9080 == 1u)
            {
                float param_37 = 1.0f;
                float param_38 = 1.5f;
                float _7819 = fresnel_dielectric_cos(param_37, param_38);
                float _7823 = roughness * roughness;
                bool _7826 = _9172 > 0.0f;
                bool _7833;
                if (_7826)
                {
                    _7833 = (_7823 * _7823) >= 1.0000000116860974230803549289703e-07f;
                }
                else
                {
                    _7833 = _7826;
                }
                [branch]
                if (_7833 && (_7629 > 0.0f))
                {
                    light_sample_t _9209 = { _9167, _9168, _9169, _9170, _9171, _9172, _9173, _9174 };
                    surface_t _9041 = { _6635, _9016, _9017, _9018, _9019, _6977 };
                    float _9712[3] = { _9699, _9700, _9701 };
                    float _9677[3] = { _9664, _9665, _9666 };
                    float _9642[3] = { _9629, _9630, _9631 };
                    shadow_ray_t _9364 = { _9642, _9337, _9677, _9339, _9712, _9341 };
                    shadow_ray_t param_39 = _9364;
                    float3 _7848 = Evaluate_GlossyNode(_9209, ray, _9041, base_color, roughness, 1.5f, _7819, mix_weight, param_39);
                    _9629 = param_39.o[0];
                    _9630 = param_39.o[1];
                    _9631 = param_39.o[2];
                    _9337 = param_39.depth;
                    _9664 = param_39.d[0];
                    _9665 = param_39.d[1];
                    _9666 = param_39.d[2];
                    _9339 = param_39.dist;
                    _9699 = param_39.c[0];
                    _9700 = param_39.c[1];
                    _9701 = param_39.c[2];
                    _9341 = param_39.xy;
                    col += _7848;
                }
                bool _7855 = _6541 < _3268_g_params.max_spec_depth;
                bool _7862;
                if (_7855)
                {
                    _7862 = _6557 < _3268_g_params.max_total_depth;
                }
                else
                {
                    _7862 = _7855;
                }
                [branch]
                if (_7862)
                {
                    surface_t _9048 = { _6635, _9016, _9017, _9018, _9019, _6977 };
                    float _9607[3] = { _9594, _9595, _9596 };
                    float _9572[3] = { _9559, _9560, _9561 };
                    float _9537[3] = { _9524, _9525, _9526 };
                    ray_data_t _9269 = { _9537, _9572, _9230, _9607, _9232, _9233, _9234, _9235, _9236 };
                    ray_data_t param_40 = _9269;
                    Sample_GlossyNode(ray, _9048, base_color, roughness, 1.5f, _7819, _7712, _7721, mix_weight, param_40);
                    _9524 = param_40.o[0];
                    _9525 = param_40.o[1];
                    _9526 = param_40.o[2];
                    _9559 = param_40.d[0];
                    _9560 = param_40.d[1];
                    _9561 = param_40.d[2];
                    _9230 = param_40.pdf;
                    _9594 = param_40.c[0];
                    _9595 = param_40.c[1];
                    _9596 = param_40.c[2];
                    _9232 = param_40.ior;
                    _9233 = param_40.cone_width;
                    _9234 = param_40.cone_spread;
                    _9235 = param_40.xy;
                    _9236 = param_40.depth;
                }
            }
            else
            {
                [branch]
                if (_9080 == 2u)
                {
                    float _7886 = roughness * roughness;
                    bool _7889 = _9172 > 0.0f;
                    bool _7896;
                    if (_7889)
                    {
                        _7896 = (_7886 * _7886) >= 1.0000000116860974230803549289703e-07f;
                    }
                    else
                    {
                        _7896 = _7889;
                    }
                    [branch]
                    if (_7896 && (_7629 < 0.0f))
                    {
                        float _7904;
                        if (_6669)
                        {
                            _7904 = _9083 / _7219;
                        }
                        else
                        {
                            _7904 = _7219 / _9083;
                        }
                        light_sample_t _9218 = { _9167, _9168, _9169, _9170, _9171, _9172, _9173, _9174 };
                        surface_t _9055 = { _6635, _9016, _9017, _9018, _9019, _6977 };
                        float _9719[3] = { _9699, _9700, _9701 };
                        float _9684[3] = { _9664, _9665, _9666 };
                        float _9649[3] = { _9629, _9630, _9631 };
                        shadow_ray_t _9377 = { _9649, _9337, _9684, _9339, _9719, _9341 };
                        shadow_ray_t param_41 = _9377;
                        float3 _7926 = Evaluate_RefractiveNode(_9218, ray, _9055, base_color, _7886, _7904, mix_weight, param_41);
                        _9629 = param_41.o[0];
                        _9630 = param_41.o[1];
                        _9631 = param_41.o[2];
                        _9337 = param_41.depth;
                        _9664 = param_41.d[0];
                        _9665 = param_41.d[1];
                        _9666 = param_41.d[2];
                        _9339 = param_41.dist;
                        _9699 = param_41.c[0];
                        _9700 = param_41.c[1];
                        _9701 = param_41.c[2];
                        _9341 = param_41.xy;
                        col += _7926;
                    }
                    bool _7933 = _6546 < _3268_g_params.max_refr_depth;
                    bool _7940;
                    if (_7933)
                    {
                        _7940 = _6557 < _3268_g_params.max_total_depth;
                    }
                    else
                    {
                        _7940 = _7933;
                    }
                    [branch]
                    if (_7940)
                    {
                        surface_t _9062 = { _6635, _9016, _9017, _9018, _9019, _6977 };
                        float _9614[3] = { _9594, _9595, _9596 };
                        float _9579[3] = { _9559, _9560, _9561 };
                        float _9544[3] = { _9524, _9525, _9526 };
                        ray_data_t _9288 = { _9544, _9579, _9230, _9614, _9232, _9233, _9234, _9235, _9236 };
                        ray_data_t param_42 = _9288;
                        Sample_RefractiveNode(ray, _9062, base_color, roughness, _6669, _9083, _7219, _7712, _7721, mix_weight, param_42);
                        _9524 = param_42.o[0];
                        _9525 = param_42.o[1];
                        _9526 = param_42.o[2];
                        _9559 = param_42.d[0];
                        _9560 = param_42.d[1];
                        _9561 = param_42.d[2];
                        _9230 = param_42.pdf;
                        _9594 = param_42.c[0];
                        _9595 = param_42.c[1];
                        _9596 = param_42.c[2];
                        _9232 = param_42.ior;
                        _9233 = param_42.cone_width;
                        _9234 = param_42.cone_spread;
                        _9235 = param_42.xy;
                        _9236 = param_42.depth;
                    }
                }
                else
                {
                    [branch]
                    if (_9080 == 3u)
                    {
                        col += (base_color * (mix_weight * _9081));
                    }
                    else
                    {
                        [branch]
                        if (_9080 == 6u)
                        {
                            float metallic = clamp(float((_9085 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_9472 != 4294967295u)
                            {
                                metallic *= SampleBilinear(_9472, _6977, int(get_texture_lod(texSize(_9472), _7211)), _6598).x;
                            }
                            float specular = clamp(float(_9087 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_9473 != 4294967295u)
                            {
                                specular *= SampleBilinear(_9473, _6977, int(get_texture_lod(texSize(_9473), _7211)), _6598).x;
                            }
                            float _8061 = clamp(float(_9088 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8069 = clamp(float((_9088 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8077 = 2.0f * clamp(float(_9084 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float3 _8095 = lerp(1.0f.xxx, tint_color, clamp(float((_9084 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _8077;
                            float3 _8115 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_9087 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                            float _8124 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                            float param_43 = 1.0f;
                            float param_44 = _8124;
                            float _8130 = fresnel_dielectric_cos(param_43, param_44);
                            float _8138 = clamp(float((_9082 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8149 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _8061))) - 1.0f;
                            float param_45 = 1.0f;
                            float param_46 = _8149;
                            float _8155 = fresnel_dielectric_cos(param_45, param_46);
                            float _8170 = mad(roughness - 1.0f, 1.0f - clamp(float((_9086 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                            float _8176;
                            if (_6669)
                            {
                                _8176 = _9083 / _7219;
                            }
                            else
                            {
                                _8176 = _7219 / _9083;
                            }
                            float param_47 = dot(_6532, _9018);
                            float param_48 = 1.0f / _8176;
                            float _8199 = fresnel_dielectric_cos(param_47, param_48);
                            float param_49 = dot(_6532, _9018);
                            float param_50 = _8124;
                            lobe_weights_t _8238 = get_lobe_weights(lerp(_7666, 1.0f, _8077), lum(lerp(_8115, 1.0f.xxx, ((fresnel_dielectric_cos(param_49, param_50) - _8130) / (1.0f - _8130)).xxx)), specular, metallic, clamp(float(_9086 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _8061);
                            [branch]
                            if (_9172 > 0.0f)
                            {
                                light_sample_t _9227 = { _9167, _9168, _9169, _9170, _9171, _9172, _9173, _9174 };
                                surface_t _9069 = { _6635, _9016, _9017, _9018, _9019, _6977 };
                                diff_params_t _9419 = { base_color, _8095, roughness };
                                spec_params_t _9434 = { _8115, roughness, _8124, _8130, _8138 };
                                clearcoat_params_t _9447 = { _8069, _8149, _8155 };
                                transmission_params_t _9462 = { _8170, _9083, _8176, _8199, _6669 };
                                float _9726[3] = { _9699, _9700, _9701 };
                                float _9691[3] = { _9664, _9665, _9666 };
                                float _9656[3] = { _9629, _9630, _9631 };
                                shadow_ray_t _9390 = { _9656, _9337, _9691, _9339, _9726, _9341 };
                                shadow_ray_t param_51 = _9390;
                                float3 _8257 = Evaluate_PrincipledNode(_9227, ray, _9069, _8238, _9419, _9434, _9447, _9462, metallic, _7629, mix_weight, param_51);
                                _9629 = param_51.o[0];
                                _9630 = param_51.o[1];
                                _9631 = param_51.o[2];
                                _9337 = param_51.depth;
                                _9664 = param_51.d[0];
                                _9665 = param_51.d[1];
                                _9666 = param_51.d[2];
                                _9339 = param_51.dist;
                                _9699 = param_51.c[0];
                                _9700 = param_51.c[1];
                                _9701 = param_51.c[2];
                                _9341 = param_51.xy;
                                col += _8257;
                            }
                            surface_t _9076 = { _6635, _9016, _9017, _9018, _9019, _6977 };
                            diff_params_t _9423 = { base_color, _8095, roughness };
                            spec_params_t _9440 = { _8115, roughness, _8124, _8130, _8138 };
                            clearcoat_params_t _9451 = { _8069, _8149, _8155 };
                            transmission_params_t _9468 = { _8170, _9083, _8176, _8199, _6669 };
                            float param_52 = mix_rand;
                            float _9621[3] = { _9594, _9595, _9596 };
                            float _9586[3] = { _9559, _9560, _9561 };
                            float _9551[3] = { _9524, _9525, _9526 };
                            ray_data_t _9307 = { _9551, _9586, _9230, _9621, _9232, _9233, _9234, _9235, _9236 };
                            ray_data_t param_53 = _9307;
                            Sample_PrincipledNode(ray, _9076, _8238, _9423, _9440, _9451, _9468, metallic, _7712, _7721, param_52, mix_weight, param_53);
                            _9524 = param_53.o[0];
                            _9525 = param_53.o[1];
                            _9526 = param_53.o[2];
                            _9559 = param_53.d[0];
                            _9560 = param_53.d[1];
                            _9561 = param_53.d[2];
                            _9230 = param_53.pdf;
                            _9594 = param_53.c[0];
                            _9595 = param_53.c[1];
                            _9596 = param_53.c[2];
                            _9232 = param_53.ior;
                            _9233 = param_53.cone_width;
                            _9234 = param_53.cone_spread;
                            _9235 = param_53.xy;
                            _9236 = param_53.depth;
                        }
                    }
                }
            }
        }
        float _8291 = max(_9594, max(_9595, _9596));
        float _8303;
        if (_6557 > _3268_g_params.min_total_depth)
        {
            _8303 = max(0.0500000007450580596923828125f, 1.0f - _8291);
        }
        else
        {
            _8303 = 0.0f;
        }
        bool _8317 = (frac(asfloat(_3252.Load((_6565 + 6) * 4 + 0)) + _6572) >= _8303) && (_8291 > 0.0f);
        bool _8323;
        if (_8317)
        {
            _8323 = _9230 > 0.0f;
        }
        else
        {
            _8323 = _8317;
        }
        [branch]
        if (_8323)
        {
            float _8327 = _9230;
            float _8328 = min(_8327, 1000000.0f);
            _9230 = _8328;
            float _8331 = 1.0f - _8303;
            float _8333 = _9594;
            float _8334 = _8333 / _8331;
            _9594 = _8334;
            float _8339 = _9595;
            float _8340 = _8339 / _8331;
            _9595 = _8340;
            float _8345 = _9596;
            float _8346 = _8345 / _8331;
            _9596 = _8346;
            uint _8354;
            _8352.InterlockedAdd(0, 1u, _8354);
            _8363.Store(_8354 * 72 + 0, asuint(_9524));
            _8363.Store(_8354 * 72 + 4, asuint(_9525));
            _8363.Store(_8354 * 72 + 8, asuint(_9526));
            _8363.Store(_8354 * 72 + 12, asuint(_9559));
            _8363.Store(_8354 * 72 + 16, asuint(_9560));
            _8363.Store(_8354 * 72 + 20, asuint(_9561));
            _8363.Store(_8354 * 72 + 24, asuint(_8328));
            _8363.Store(_8354 * 72 + 28, asuint(_8334));
            _8363.Store(_8354 * 72 + 32, asuint(_8340));
            _8363.Store(_8354 * 72 + 36, asuint(_8346));
            _8363.Store(_8354 * 72 + 40, asuint(_9232[0]));
            _8363.Store(_8354 * 72 + 44, asuint(_9232[1]));
            _8363.Store(_8354 * 72 + 48, asuint(_9232[2]));
            _8363.Store(_8354 * 72 + 52, asuint(_9232[3]));
            _8363.Store(_8354 * 72 + 56, asuint(_9233));
            _8363.Store(_8354 * 72 + 60, asuint(_9234));
            _8363.Store(_8354 * 72 + 64, uint(_9235));
            _8363.Store(_8354 * 72 + 68, uint(_9236));
        }
        [branch]
        if (max(_9699, max(_9700, _9701)) > 0.0f)
        {
            float3 _8440 = _9169 - float3(_9629, _9630, _9631);
            float _8443 = length(_8440);
            float3 _8447 = _8440 / _8443.xxx;
            float sh_dist = _8443 * _9171;
            if (_9174)
            {
                sh_dist = -sh_dist;
            }
            float _8459 = _8447.x;
            _9664 = _8459;
            float _8462 = _8447.y;
            _9665 = _8462;
            float _8465 = _8447.z;
            _9666 = _8465;
            _9339 = sh_dist;
            uint _8471;
            _8352.InterlockedAdd(8, 1u, _8471);
            _8479.Store(_8471 * 48 + 0, asuint(_9629));
            _8479.Store(_8471 * 48 + 4, asuint(_9630));
            _8479.Store(_8471 * 48 + 8, asuint(_9631));
            _8479.Store(_8471 * 48 + 12, uint(_9337));
            _8479.Store(_8471 * 48 + 16, asuint(_8459));
            _8479.Store(_8471 * 48 + 20, asuint(_8462));
            _8479.Store(_8471 * 48 + 24, asuint(_8465));
            _8479.Store(_8471 * 48 + 28, asuint(sh_dist));
            _8479.Store(_8471 * 48 + 32, asuint(_9699));
            _8479.Store(_8471 * 48 + 36, asuint(_9700));
            _8479.Store(_8471 * 48 + 40, asuint(_9701));
            _8479.Store(_8471 * 48 + 44, uint(_9341));
        }
        _8718 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _8718;
}

void comp_main()
{
    do
    {
        int _8545 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_8545) >= _8352.Load(4))
        {
            break;
        }
        int _8561 = int(_8558.Load(_8545 * 72 + 64));
        int _8568 = int(_8558.Load(_8545 * 72 + 64));
        hit_data_t _8579;
        _8579.mask = int(_8575.Load(_8545 * 24 + 0));
        _8579.obj_index = int(_8575.Load(_8545 * 24 + 4));
        _8579.prim_index = int(_8575.Load(_8545 * 24 + 8));
        _8579.t = asfloat(_8575.Load(_8545 * 24 + 12));
        _8579.u = asfloat(_8575.Load(_8545 * 24 + 16));
        _8579.v = asfloat(_8575.Load(_8545 * 24 + 20));
        ray_data_t _8595;
        [unroll]
        for (int _84ident = 0; _84ident < 3; _84ident++)
        {
            _8595.o[_84ident] = asfloat(_8558.Load(_84ident * 4 + _8545 * 72 + 0));
        }
        [unroll]
        for (int _85ident = 0; _85ident < 3; _85ident++)
        {
            _8595.d[_85ident] = asfloat(_8558.Load(_85ident * 4 + _8545 * 72 + 12));
        }
        _8595.pdf = asfloat(_8558.Load(_8545 * 72 + 24));
        [unroll]
        for (int _86ident = 0; _86ident < 3; _86ident++)
        {
            _8595.c[_86ident] = asfloat(_8558.Load(_86ident * 4 + _8545 * 72 + 28));
        }
        [unroll]
        for (int _87ident = 0; _87ident < 4; _87ident++)
        {
            _8595.ior[_87ident] = asfloat(_8558.Load(_87ident * 4 + _8545 * 72 + 40));
        }
        _8595.cone_width = asfloat(_8558.Load(_8545 * 72 + 56));
        _8595.cone_spread = asfloat(_8558.Load(_8545 * 72 + 60));
        _8595.xy = int(_8558.Load(_8545 * 72 + 64));
        _8595.depth = int(_8558.Load(_8545 * 72 + 68));
        hit_data_t _8812 = { _8579.mask, _8579.obj_index, _8579.prim_index, _8579.t, _8579.u, _8579.v };
        hit_data_t param = _8812;
        float _8861[4] = { _8595.ior[0], _8595.ior[1], _8595.ior[2], _8595.ior[3] };
        float _8852[3] = { _8595.c[0], _8595.c[1], _8595.c[2] };
        float _8845[3] = { _8595.d[0], _8595.d[1], _8595.d[2] };
        float _8838[3] = { _8595.o[0], _8595.o[1], _8595.o[2] };
        ray_data_t _8831 = { _8838, _8845, _8595.pdf, _8852, _8861, _8595.cone_width, _8595.cone_spread, _8595.xy, _8595.depth };
        ray_data_t param_1 = _8831;
        float3 param_2 = 0.0f.xxx;
        float3 param_3 = 0.0f.xxx;
        float3 _8651 = ShadeSurface(param, param_1, param_2, param_3);
        int2 _8665 = int2((_8561 >> 16) & 65535, _8568 & 65535);
        g_out_img[_8665] = float4(min(_8651, _3268_g_params.clamp_val.xxx), 1.0f);
        g_out_base_color_img[_8665] = float4(param_2, 0.0f);
        g_out_depth_normals_img[_8665] = float4(param_3, _8579.t);
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
