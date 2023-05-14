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
ByteAddressBuffer _6775 : register(t14, space0);
RWByteAddressBuffer _8471 : register(u3, space0);
RWByteAddressBuffer _8482 : register(u1, space0);
RWByteAddressBuffer _8598 : register(u2, space0);
ByteAddressBuffer _8677 : register(t7, space0);
ByteAddressBuffer _8694 : register(t6, space0);
ByteAddressBuffer _8816 : register(t10, space0);
cbuffer UniformParams
{
    Params _3268_g_params : packoffset(c0);
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
    float3 _4635;
    if ((ray.depth & 16777215) != 0)
    {
        _4635 = _3268_g_params.env_col.xyz;
    }
    else
    {
        _4635 = _3268_g_params.back_col.xyz;
    }
    float3 env_col = _4635;
    uint _4651;
    if ((ray.depth & 16777215) != 0)
    {
        _4651 = asuint(_3268_g_params.env_col.w);
    }
    else
    {
        _4651 = asuint(_3268_g_params.back_col.w);
    }
    float _4667;
    if ((ray.depth & 16777215) != 0)
    {
        _4667 = _3268_g_params.env_rotation;
    }
    else
    {
        _4667 = _3268_g_params.back_rotation;
    }
    if (_4651 != 4294967295u)
    {
        env_col *= SampleLatlong_RGBE(_4651, _4628, _4667, tex_rand);
    }
    if (_3268_g_params.env_qtree_levels > 0)
    {
        float param = ray.pdf;
        float param_1 = Evaluate_EnvQTree(_4667, g_env_qtree, _g_env_qtree_sampler, _3268_g_params.env_qtree_levels, _4628);
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
    float3 _4746 = float3(ray.d[0], ray.d[1], ray.d[2]);
    light_t _4760;
    _4760.type_and_param0 = _3288.Load4(((-1) - inter.obj_index) * 64 + 0);
    _4760.param1 = asfloat(_3288.Load4(((-1) - inter.obj_index) * 64 + 16));
    _4760.param2 = asfloat(_3288.Load4(((-1) - inter.obj_index) * 64 + 32));
    _4760.param3 = asfloat(_3288.Load4(((-1) - inter.obj_index) * 64 + 48));
    float3 lcol = asfloat(_4760.type_and_param0.yzw);
    [branch]
    if ((_4760.type_and_param0.x & 128u) != 0u)
    {
        float3 env_col = _3268_g_params.env_col.xyz;
        uint _4787 = asuint(_3268_g_params.env_col.w);
        if (_4787 != 4294967295u)
        {
            env_col *= SampleLatlong_RGBE(_4787, _4746, _3268_g_params.env_rotation, tex_rand);
        }
        lcol *= env_col;
    }
    uint _4805 = _4760.type_and_param0.x & 31u;
    if (_4805 == 0u)
    {
        float param = ray.pdf;
        float param_1 = (inter.t * inter.t) / ((0.5f * _4760.param1.w) * dot(_4746, normalize(_4760.param1.xyz - (float3(ray.o[0], ray.o[1], ray.o[2]) + (_4746 * inter.t)))));
        lcol *= power_heuristic(param, param_1);
        bool _4872 = _4760.param3.x > 0.0f;
        bool _4878;
        if (_4872)
        {
            _4878 = _4760.param3.y > 0.0f;
        }
        else
        {
            _4878 = _4872;
        }
        [branch]
        if (_4878)
        {
            [flatten]
            if (_4760.param3.y > 0.0f)
            {
                lcol *= clamp((_4760.param3.x - acos(clamp(-dot(_4746, _4760.param2.xyz), 0.0f, 1.0f))) / _4760.param3.y, 0.0f, 1.0f);
            }
        }
    }
    else
    {
        if (_4805 == 4u)
        {
            float param_2 = ray.pdf;
            float param_3 = (inter.t * inter.t) / (_4760.param1.w * dot(_4746, normalize(cross(_4760.param2.xyz, _4760.param3.xyz))));
            lcol *= power_heuristic(param_2, param_3);
        }
        else
        {
            if (_4805 == 5u)
            {
                float param_4 = ray.pdf;
                float param_5 = (inter.t * inter.t) / (_4760.param1.w * dot(_4746, normalize(cross(_4760.param2.xyz, _4760.param3.xyz))));
                lcol *= power_heuristic(param_4, param_5);
            }
            else
            {
                if (_4805 == 3u)
                {
                    float param_6 = ray.pdf;
                    float param_7 = (inter.t * inter.t) / (_4760.param1.w * (1.0f - abs(dot(_4746, _4760.param3.xyz))));
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
    float _8828;
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
            _8828 = stack[3];
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
            _8828 = stack[2];
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
            _8828 = stack[1];
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
            _8828 = stack[0];
            break;
        }
        _8828 = default_value;
        break;
    } while(false);
    return _8828;
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
        float4 _9883 = res;
        _9883.x = _1051.x;
        float4 _9885 = _9883;
        _9885.y = _1051.y;
        float4 _9887 = _9885;
        _9887.z = _1051.z;
        res = _9887;
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
    float3 _8833;
    do
    {
        float _1240 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1240)
        {
            _8833 = N;
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
            float _10181 = (-0.5f) / _1280;
            float param_1 = mad(_10181, _1304, 1.0f);
            float _1336 = safe_sqrtf(param_1);
            float param_2 = _1305;
            float _1339 = safe_sqrtf(param_2);
            float2 _1340 = float2(_1336, _1339);
            float param_3 = mad(_10181, _1311, 1.0f);
            float _1345 = safe_sqrtf(param_3);
            float param_4 = _1312;
            float _1348 = safe_sqrtf(param_4);
            float2 _1349 = float2(_1345, _1348);
            float _10183 = -_1268;
            float _1365 = mad(2.0f * mad(_1336, _1264, _1339 * _1268), _1339, _10183);
            float _1381 = mad(2.0f * mad(_1345, _1264, _1348 * _1268), _1348, _10183);
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
                _8833 = Ng;
                break;
            }
            float _1418 = valid1 ? _1305 : _1312;
            float param_5 = 1.0f - _1418;
            float param_6 = _1418;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _8833 = (_1260 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _8833;
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
    float3 _8858;
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
            _8858 = N;
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
        _8858 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _8858;
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
            float2 _9870 = origin;
            _9870.x = origin.x + _step;
            origin = _9870;
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
            float2 _9873 = origin;
            _9873.y = origin.y + _step;
            origin = _9873;
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
        float3 _9944 = sampled_dir;
        float3 _3433 = ((param * _9944.x) + (param_1 * _9944.y)) + (_3390 * _9944.z);
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
    float3 _8838;
    do
    {
        float3 param = -float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param_1 = surf.N;
        float3 param_2 = ls.L;
        float param_3 = roughness;
        float3 param_4 = base_color;
        float4 _5075 = Evaluate_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_5 = ls.pdf;
            float param_6 = _5075.w;
            mis_weight = power_heuristic(param_5, param_6);
        }
        float3 _5098 = (ls.col * _5075.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8838 = _5098;
            break;
        }
        float3 param_7 = surf.P;
        float3 param_8 = surf.plane_N;
        float3 _5110 = offset_ray(param_7, param_8);
        sh_r.o[0] = _5110.x;
        sh_r.o[1] = _5110.y;
        sh_r.o[2] = _5110.z;
        sh_r.c[0] = ray.c[0] * _5098.x;
        sh_r.c[1] = ray.c[1] * _5098.y;
        sh_r.c[2] = ray.c[2] * _5098.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8838 = 0.0f.xxx;
        break;
    } while(false);
    return _8838;
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
    float4 _5361 = Sample_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8);
    new_ray.depth = ray.depth + 1;
    float3 param_9 = surf.P;
    float3 param_10 = surf.plane_N;
    float3 _5371 = offset_ray(param_9, param_10);
    new_ray.o[0] = _5371.x;
    new_ray.o[1] = _5371.y;
    new_ray.o[2] = _5371.z;
    new_ray.d[0] = param_8.x;
    new_ray.d[1] = param_8.y;
    new_ray.d[2] = param_8.z;
    new_ray.c[0] = ((ray.c[0] * _5361.x) * mix_weight) / _5361.w;
    new_ray.c[1] = ((ray.c[1] * _5361.y) * mix_weight) / _5361.w;
    new_ray.c[2] = ((ray.c[2] * _5361.z) * mix_weight) / _5361.w;
    new_ray.pdf = _5361.w;
}

float3 tangent_from_world(float3 T, float3 B, float3 N, float3 V)
{
    return float3(dot(V, T), dot(V, B), dot(V, N));
}

float D_GGX(float3 H, float alpha_x, float alpha_y)
{
    float _8891;
    do
    {
        if (H.z == 0.0f)
        {
            _8891 = 0.0f;
            break;
        }
        float _1973 = (-H.x) / (H.z * alpha_x);
        float _1979 = (-H.y) / (H.z * alpha_y);
        float _1988 = mad(_1979, _1979, mad(_1973, _1973, 1.0f));
        _8891 = 1.0f / (((((_1988 * _1988) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _8891;
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
    float3 _8843;
    do
    {
        float3 _5146 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5146;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - _5146);
        float _5184 = roughness * roughness;
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = _5184;
        float param_16 = _5184;
        float param_17 = spec_ior;
        float param_18 = spec_F0;
        float3 param_19 = base_color;
        float4 _5199 = Evaluate_GGXSpecular_BSDF(param_12, param_13, param_14, param_15, param_16, param_17, param_18, param_19);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_20 = ls.pdf;
            float param_21 = _5199.w;
            mis_weight = power_heuristic(param_20, param_21);
        }
        float3 _5222 = (ls.col * _5199.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8843 = _5222;
            break;
        }
        float3 param_22 = surf.P;
        float3 param_23 = surf.plane_N;
        float3 _5234 = offset_ray(param_22, param_23);
        sh_r.o[0] = _5234.x;
        sh_r.o[1] = _5234.y;
        sh_r.o[2] = _5234.z;
        sh_r.c[0] = ray.c[0] * _5222.x;
        sh_r.c[1] = ray.c[1] * _5222.y;
        sh_r.c[2] = ray.c[2] * _5222.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8843 = 0.0f.xxx;
        break;
    } while(false);
    return _8843;
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
    float4 _8863;
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
            _8863 = float4(_2628.x * 1000000.0f, _2628.y * 1000000.0f, _2628.z * 1000000.0f, 1000000.0f);
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
        _8863 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _8863;
}

void Sample_GlossyNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 param_4;
    float4 _5281 = Sample_GGXSpecular_BSDF(param, param_1, param_2, param_3, roughness, 0.0f, spec_ior, spec_F0, base_color, rand_u, rand_v, param_4);
    new_ray.depth = ray.depth + 256;
    float3 param_5 = surf.P;
    float3 param_6 = surf.plane_N;
    float3 _5292 = offset_ray(param_5, param_6);
    new_ray.o[0] = _5292.x;
    new_ray.o[1] = _5292.y;
    new_ray.o[2] = _5292.z;
    new_ray.d[0] = param_4.x;
    new_ray.d[1] = param_4.y;
    new_ray.d[2] = param_4.z;
    new_ray.c[0] = ((ray.c[0] * _5281.x) * mix_weight) / _5281.w;
    new_ray.c[1] = ((ray.c[1] * _5281.y) * mix_weight) / _5281.w;
    new_ray.c[2] = ((ray.c[2] * _5281.z) * mix_weight) / _5281.w;
    new_ray.pdf = _5281.w;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _8868;
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
            _8868 = 0.0f.xxxx;
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
        _8868 = float4(refr_col * (((((_2911 * _2927) * _2919) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _2947) / view_dir_ts.z), (((_2911 * _2919) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _2947) / view_dir_ts.z);
        break;
    } while(false);
    return _8868;
}

float3 Evaluate_RefractiveNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness2, float eta, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _8848;
    do
    {
        float3 _5424 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5424;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - (_5424 * eta));
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = roughness2;
        float param_16 = eta;
        float3 param_17 = base_color;
        float4 _5472 = Evaluate_GGXRefraction_BSDF(param_12, param_13, param_14, param_15, param_16, param_17);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_18 = ls.pdf;
            float param_19 = _5472.w;
            mis_weight = power_heuristic(param_18, param_19);
        }
        float3 _5495 = (ls.col * _5472.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8848 = _5495;
            break;
        }
        float3 param_20 = surf.P;
        float3 param_21 = -surf.plane_N;
        float3 _5508 = offset_ray(param_20, param_21);
        sh_r.o[0] = _5508.x;
        sh_r.o[1] = _5508.y;
        sh_r.o[2] = _5508.z;
        sh_r.c[0] = ray.c[0] * _5495.x;
        sh_r.c[1] = ray.c[1] * _5495.y;
        sh_r.c[2] = ray.c[2] * _5495.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8848 = 0.0f.xxx;
        break;
    } while(false);
    return _8848;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _8873;
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
                _8873 = 0.0f.xxxx;
                break;
            }
            float _3024 = mad(eta, _3002, -sqrt(_3012));
            out_V = float4(normalize((I * eta) + (N * _3024)), _3024);
            _8873 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
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
            _8873 = 0.0f.xxxx;
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
        _8873 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _8873;
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
    float _8881;
    do
    {
        if (stack[3] > 0.0f)
        {
            float param = stack[3];
            float param_1 = -1.0f;
            float _2079 = exchange(param, param_1);
            stack[3] = param;
            _8881 = _2079;
            break;
        }
        if (stack[2] > 0.0f)
        {
            float param_2 = stack[2];
            float param_3 = -1.0f;
            float _2092 = exchange(param_2, param_3);
            stack[2] = param_2;
            _8881 = _2092;
            break;
        }
        if (stack[1] > 0.0f)
        {
            float param_4 = stack[1];
            float param_5 = -1.0f;
            float _2105 = exchange(param_4, param_5);
            stack[1] = param_4;
            _8881 = _2105;
            break;
        }
        if (stack[0] > 0.0f)
        {
            float param_6 = stack[0];
            float param_7 = -1.0f;
            float _2118 = exchange(param_6, param_7);
            stack[0] = param_6;
            _8881 = _2118;
            break;
        }
        _8881 = default_value;
        break;
    } while(false);
    return _8881;
}

void Sample_RefractiveNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, bool is_backfacing, float int_ior, float ext_ior, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float _5545;
    if (is_backfacing)
    {
        _5545 = int_ior / ext_ior;
    }
    else
    {
        _5545 = ext_ior / int_ior;
    }
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float param_4 = roughness;
    float param_5 = _5545;
    float3 param_6 = base_color;
    float param_7 = rand_u;
    float param_8 = rand_v;
    float4 param_9;
    float4 _5569 = Sample_GGXRefraction_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9);
    new_ray.depth = ray.depth + 65536;
    new_ray.c[0] = ((ray.c[0] * _5569.x) * mix_weight) / _5569.w;
    new_ray.c[1] = ((ray.c[1] * _5569.y) * mix_weight) / _5569.w;
    new_ray.c[2] = ((ray.c[2] * _5569.z) * mix_weight) / _5569.w;
    new_ray.pdf = _5569.w;
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
        float _5625 = pop_ior_stack(param_11, param_12);
        new_ray.ior = param_11;
    }
    float3 param_13 = surf.P;
    float3 param_14 = -surf.plane_N;
    float3 _5634 = offset_ray(param_13, param_14);
    new_ray.o[0] = _5634.x;
    new_ray.o[1] = _5634.y;
    new_ray.o[2] = _5634.z;
    new_ray.d[0] = param_9.x;
    new_ray.d[1] = param_9.y;
    new_ray.d[2] = param_9.z;
}

lobe_weights_t get_lobe_weights(float base_color_lum, float spec_color_lum, float specular, float metallic, float transmission, float clearcoat)
{
    float _1443 = 1.0f - metallic;
    float _8967 = (base_color_lum * _1443) * (1.0f - transmission);
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
    float _8968 = _1454;
    float _1464 = 0.25f * clearcoat;
    float _8969 = _1464 * _1443;
    float _8970 = _1450 * base_color_lum;
    float _1473 = _8967;
    float _1482 = mad(_1450, base_color_lum, mad(_1464, _1443, _1473 + _1454));
    if (_1482 != 0.0f)
    {
        _8967 /= _1482;
        _8968 /= _1482;
        _8969 /= _1482;
        _8970 /= _1482;
    }
    lobe_weights_t _8975 = { _8967, _8968, _8969, _8970 };
    return _8975;
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
    float _8896;
    do
    {
        float _2199 = dot(N, L);
        if (_2199 <= 0.0f)
        {
            _8896 = 0.0f;
            break;
        }
        float param = _2199;
        float param_1 = dot(N, V);
        float _2220 = dot(L, H);
        float _2228 = mad((2.0f * _2220) * _2220, roughness, 0.5f);
        _8896 = lerp(1.0f, _2228, schlick_weight(param)) * lerp(1.0f, _2228, schlick_weight(param_1));
        break;
    } while(false);
    return _8896;
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
    float _8901;
    do
    {
        if (a >= 1.0f)
        {
            _8901 = 0.3183098733425140380859375f;
            break;
        }
        float _1947 = mad(a, a, -1.0f);
        _8901 = _1947 / ((3.1415927410125732421875f * log(a * a)) * mad(_1947 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _8901;
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
    float3 _8853;
    do
    {
        float3 _5657 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 lcol = 0.0f.xxx;
        float bsdf_pdf = 0.0f;
        bool _5662 = N_dot_L > 0.0f;
        [branch]
        if ((lobe_weights.diffuse > 1.0000000116860974230803549289703e-07f) && _5662)
        {
            float3 param = -_5657;
            float3 param_1 = surf.N;
            float3 param_2 = ls.L;
            float param_3 = diff.roughness;
            float3 param_4 = diff.base_color;
            float3 param_5 = diff.sheen_color;
            bool param_6 = false;
            float4 _5681 = Evaluate_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6);
            bsdf_pdf = mad(lobe_weights.diffuse, _5681.w, bsdf_pdf);
            lcol += (((ls.col * N_dot_L) * (_5681 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * ls.pdf).xxx);
        }
        float3 H;
        [flatten]
        if (_5662)
        {
            H = normalize(ls.L - _5657);
        }
        else
        {
            H = normalize(ls.L - (_5657 * trans.eta));
        }
        float _5720 = spec.roughness * spec.roughness;
        float _5725 = sqrt(mad(-0.89999997615814208984375f, spec.anisotropy, 1.0f));
        float _5729 = _5720 / _5725;
        float _5733 = _5720 * _5725;
        float3 param_7 = surf.T;
        float3 param_8 = surf.B;
        float3 param_9 = surf.N;
        float3 param_10 = -_5657;
        float3 _5744 = tangent_from_world(param_7, param_8, param_9, param_10);
        float3 param_11 = surf.T;
        float3 param_12 = surf.B;
        float3 param_13 = surf.N;
        float3 param_14 = ls.L;
        float3 _5754 = tangent_from_world(param_11, param_12, param_13, param_14);
        float3 param_15 = surf.T;
        float3 param_16 = surf.B;
        float3 param_17 = surf.N;
        float3 param_18 = H;
        float3 _5764 = tangent_from_world(param_15, param_16, param_17, param_18);
        bool _5766 = lobe_weights.specular > 0.0f;
        bool _5773;
        if (_5766)
        {
            _5773 = (_5729 * _5733) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _5773 = _5766;
        }
        [branch]
        if (_5773 && _5662)
        {
            float3 param_19 = _5744;
            float3 param_20 = _5764;
            float3 param_21 = _5754;
            float param_22 = _5729;
            float param_23 = _5733;
            float param_24 = spec.ior;
            float param_25 = spec.F0;
            float3 param_26 = spec.tmp_col;
            float4 _5795 = Evaluate_GGXSpecular_BSDF(param_19, param_20, param_21, param_22, param_23, param_24, param_25, param_26);
            bsdf_pdf = mad(lobe_weights.specular, _5795.w, bsdf_pdf);
            lcol += ((ls.col * _5795.xyz) / ls.pdf.xxx);
        }
        float _5814 = coat.roughness * coat.roughness;
        bool _5816 = lobe_weights.clearcoat > 0.0f;
        bool _5823;
        if (_5816)
        {
            _5823 = (_5814 * _5814) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _5823 = _5816;
        }
        [branch]
        if (_5823 && _5662)
        {
            float3 param_27 = _5744;
            float3 param_28 = _5764;
            float3 param_29 = _5754;
            float param_30 = _5814;
            float param_31 = coat.ior;
            float param_32 = coat.F0;
            float4 _5841 = Evaluate_PrincipledClearcoat_BSDF(param_27, param_28, param_29, param_30, param_31, param_32);
            bsdf_pdf = mad(lobe_weights.clearcoat, _5841.w, bsdf_pdf);
            lcol += (((ls.col * 0.25f) * _5841.xyz) / ls.pdf.xxx);
        }
        [branch]
        if (lobe_weights.refraction > 0.0f)
        {
            bool _5863 = trans.fresnel != 0.0f;
            bool _5870;
            if (_5863)
            {
                _5870 = (_5720 * _5720) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _5870 = _5863;
            }
            [branch]
            if (_5870 && _5662)
            {
                float3 param_33 = _5744;
                float3 param_34 = _5764;
                float3 param_35 = _5754;
                float param_36 = _5720;
                float param_37 = _5720;
                float param_38 = 1.0f;
                float param_39 = 0.0f;
                float3 param_40 = 1.0f.xxx;
                float4 _5889 = Evaluate_GGXSpecular_BSDF(param_33, param_34, param_35, param_36, param_37, param_38, param_39, param_40);
                bsdf_pdf = mad(lobe_weights.refraction * trans.fresnel, _5889.w, bsdf_pdf);
                lcol += ((ls.col * _5889.xyz) * (trans.fresnel / ls.pdf));
            }
            float _5911 = trans.roughness * trans.roughness;
            bool _5913 = trans.fresnel != 1.0f;
            bool _5920;
            if (_5913)
            {
                _5920 = (_5911 * _5911) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _5920 = _5913;
            }
            [branch]
            if (_5920 && (N_dot_L < 0.0f))
            {
                float3 param_41 = _5744;
                float3 param_42 = _5764;
                float3 param_43 = _5754;
                float param_44 = _5911;
                float param_45 = trans.eta;
                float3 param_46 = diff.base_color;
                float4 _5938 = Evaluate_GGXRefraction_BSDF(param_41, param_42, param_43, param_44, param_45, param_46);
                float _5941 = 1.0f - trans.fresnel;
                bsdf_pdf = mad(lobe_weights.refraction * _5941, _5938.w, bsdf_pdf);
                lcol += ((ls.col * _5938.xyz) * (_5941 / ls.pdf));
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
            _8853 = lcol;
            break;
        }
        float3 _5981;
        if (N_dot_L < 0.0f)
        {
            _5981 = -surf.plane_N;
        }
        else
        {
            _5981 = surf.plane_N;
        }
        float3 param_49 = surf.P;
        float3 param_50 = _5981;
        float3 _5992 = offset_ray(param_49, param_50);
        sh_r.o[0] = _5992.x;
        sh_r.o[1] = _5992.y;
        sh_r.o[2] = _5992.z;
        sh_r.c[0] = ray.c[0] * lcol.x;
        sh_r.c[1] = ray.c[1] * lcol.y;
        sh_r.c[2] = ray.c[2] * lcol.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8853 = 0.0f.xxx;
        break;
    } while(false);
    return _8853;
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
    float4 _8886;
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
            _8886 = float4(_2828, _2828, _2828, 1000000.0f);
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
        _8886 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _8886;
}

void Sample_PrincipledNode(ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float rand_u, float rand_v, inout float mix_rand, float mix_weight, inout ray_data_t new_ray)
{
    float3 _6027 = float3(ray.d[0], ray.d[1], ray.d[2]);
    int _6031 = ray.depth & 255;
    int _6035 = (ray.depth >> 8) & 255;
    int _6039 = (ray.depth >> 16) & 255;
    int _6050 = (_6031 + _6035) + _6039;
    [branch]
    if (mix_rand < lobe_weights.diffuse)
    {
        bool _6059 = _6031 < _3268_g_params.max_diff_depth;
        bool _6066;
        if (_6059)
        {
            _6066 = _6050 < _3268_g_params.max_total_depth;
        }
        else
        {
            _6066 = _6059;
        }
        if (_6066)
        {
            float3 param = surf.T;
            float3 param_1 = surf.B;
            float3 param_2 = surf.N;
            float3 param_3 = _6027;
            float param_4 = diff.roughness;
            float3 param_5 = diff.base_color;
            float3 param_6 = diff.sheen_color;
            bool param_7 = false;
            float param_8 = rand_u;
            float param_9 = rand_v;
            float3 param_10;
            float4 _6089 = Sample_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9, param_10);
            float3 _6094 = _6089.xyz * (1.0f - metallic);
            new_ray.depth = ray.depth + 1;
            float3 param_11 = surf.P;
            float3 param_12 = surf.plane_N;
            float3 _6109 = offset_ray(param_11, param_12);
            new_ray.o[0] = _6109.x;
            new_ray.o[1] = _6109.y;
            new_ray.o[2] = _6109.z;
            new_ray.d[0] = param_10.x;
            new_ray.d[1] = param_10.y;
            new_ray.d[2] = param_10.z;
            new_ray.c[0] = ((ray.c[0] * _6094.x) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[1] = ((ray.c[1] * _6094.y) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[2] = ((ray.c[2] * _6094.z) * mix_weight) / lobe_weights.diffuse;
            new_ray.pdf = _6089.w;
        }
    }
    else
    {
        float _6159 = lobe_weights.diffuse + lobe_weights.specular;
        [branch]
        if (mix_rand < _6159)
        {
            bool _6166 = _6035 < _3268_g_params.max_spec_depth;
            bool _6173;
            if (_6166)
            {
                _6173 = _6050 < _3268_g_params.max_total_depth;
            }
            else
            {
                _6173 = _6166;
            }
            if (_6173)
            {
                float3 param_13 = surf.T;
                float3 param_14 = surf.B;
                float3 param_15 = surf.N;
                float3 param_16 = _6027;
                float3 param_17;
                float4 _6192 = Sample_GGXSpecular_BSDF(param_13, param_14, param_15, param_16, spec.roughness, spec.anisotropy, spec.ior, spec.F0, spec.tmp_col, rand_u, rand_v, param_17);
                float _6197 = _6192.w * lobe_weights.specular;
                new_ray.depth = ray.depth + 256;
                new_ray.c[0] = ((ray.c[0] * _6192.x) * mix_weight) / _6197;
                new_ray.c[1] = ((ray.c[1] * _6192.y) * mix_weight) / _6197;
                new_ray.c[2] = ((ray.c[2] * _6192.z) * mix_weight) / _6197;
                new_ray.pdf = _6197;
                float3 param_18 = surf.P;
                float3 param_19 = surf.plane_N;
                float3 _6237 = offset_ray(param_18, param_19);
                new_ray.o[0] = _6237.x;
                new_ray.o[1] = _6237.y;
                new_ray.o[2] = _6237.z;
                new_ray.d[0] = param_17.x;
                new_ray.d[1] = param_17.y;
                new_ray.d[2] = param_17.z;
            }
        }
        else
        {
            float _6262 = _6159 + lobe_weights.clearcoat;
            [branch]
            if (mix_rand < _6262)
            {
                bool _6269 = _6035 < _3268_g_params.max_spec_depth;
                bool _6276;
                if (_6269)
                {
                    _6276 = _6050 < _3268_g_params.max_total_depth;
                }
                else
                {
                    _6276 = _6269;
                }
                if (_6276)
                {
                    float3 param_20 = surf.T;
                    float3 param_21 = surf.B;
                    float3 param_22 = surf.N;
                    float3 param_23 = _6027;
                    float param_24 = coat.roughness * coat.roughness;
                    float param_25 = coat.ior;
                    float param_26 = coat.F0;
                    float param_27 = rand_u;
                    float param_28 = rand_v;
                    float3 param_29;
                    float4 _6300 = Sample_PrincipledClearcoat_BSDF(param_20, param_21, param_22, param_23, param_24, param_25, param_26, param_27, param_28, param_29);
                    float _6305 = _6300.w * lobe_weights.clearcoat;
                    new_ray.depth = ray.depth + 256;
                    new_ray.c[0] = (((0.25f * ray.c[0]) * _6300.x) * mix_weight) / _6305;
                    new_ray.c[1] = (((0.25f * ray.c[1]) * _6300.y) * mix_weight) / _6305;
                    new_ray.c[2] = (((0.25f * ray.c[2]) * _6300.z) * mix_weight) / _6305;
                    new_ray.pdf = _6305;
                    float3 param_30 = surf.P;
                    float3 param_31 = surf.plane_N;
                    float3 _6348 = offset_ray(param_30, param_31);
                    new_ray.o[0] = _6348.x;
                    new_ray.o[1] = _6348.y;
                    new_ray.o[2] = _6348.z;
                    new_ray.d[0] = param_29.x;
                    new_ray.d[1] = param_29.y;
                    new_ray.d[2] = param_29.z;
                }
            }
            else
            {
                bool _6370 = mix_rand >= trans.fresnel;
                bool _6377;
                if (_6370)
                {
                    _6377 = _6039 < _3268_g_params.max_refr_depth;
                }
                else
                {
                    _6377 = _6370;
                }
                bool _6391;
                if (!_6377)
                {
                    bool _6383 = mix_rand < trans.fresnel;
                    bool _6390;
                    if (_6383)
                    {
                        _6390 = _6035 < _3268_g_params.max_spec_depth;
                    }
                    else
                    {
                        _6390 = _6383;
                    }
                    _6391 = _6390;
                }
                else
                {
                    _6391 = _6377;
                }
                bool _6398;
                if (_6391)
                {
                    _6398 = _6050 < _3268_g_params.max_total_depth;
                }
                else
                {
                    _6398 = _6391;
                }
                [branch]
                if (_6398)
                {
                    mix_rand -= _6262;
                    mix_rand /= lobe_weights.refraction;
                    float4 F;
                    float3 V;
                    [branch]
                    if (mix_rand < trans.fresnel)
                    {
                        float3 param_32 = surf.T;
                        float3 param_33 = surf.B;
                        float3 param_34 = surf.N;
                        float3 param_35 = _6027;
                        float3 param_36;
                        float4 _6428 = Sample_GGXSpecular_BSDF(param_32, param_33, param_34, param_35, spec.roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, rand_u, rand_v, param_36);
                        V = param_36;
                        F = _6428;
                        new_ray.depth = ray.depth + 256;
                        float3 param_37 = surf.P;
                        float3 param_38 = surf.plane_N;
                        float3 _6438 = offset_ray(param_37, param_38);
                        new_ray.o[0] = _6438.x;
                        new_ray.o[1] = _6438.y;
                        new_ray.o[2] = _6438.z;
                    }
                    else
                    {
                        float3 param_39 = surf.T;
                        float3 param_40 = surf.B;
                        float3 param_41 = surf.N;
                        float3 param_42 = _6027;
                        float param_43 = trans.roughness;
                        float param_44 = trans.eta;
                        float3 param_45 = diff.base_color;
                        float param_46 = rand_u;
                        float param_47 = rand_v;
                        float4 param_48;
                        float4 _6467 = Sample_GGXRefraction_BSDF(param_39, param_40, param_41, param_42, param_43, param_44, param_45, param_46, param_47, param_48);
                        F = _6467;
                        V = param_48.xyz;
                        new_ray.depth = ray.depth + 65536;
                        float3 param_49 = surf.P;
                        float3 param_50 = -surf.plane_N;
                        float3 _6480 = offset_ray(param_49, param_50);
                        new_ray.o[0] = _6480.x;
                        new_ray.o[1] = _6480.y;
                        new_ray.o[2] = _6480.z;
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
                            float _6506 = pop_ior_stack(param_52, param_53);
                            new_ray.ior = param_52;
                        }
                    }
                    float4 _10128 = F;
                    float _6512 = _10128.w * lobe_weights.refraction;
                    float4 _10130 = _10128;
                    _10130.w = _6512;
                    F = _10130;
                    new_ray.c[0] = ((ray.c[0] * _10128.x) * mix_weight) / _6512;
                    new_ray.c[1] = ((ray.c[1] * _10128.y) * mix_weight) / _6512;
                    new_ray.c[2] = ((ray.c[2] * _10128.z) * mix_weight) / _6512;
                    new_ray.pdf = _6512;
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
    float3 _8823;
    do
    {
        float3 _6568 = float3(ray.d[0], ray.d[1], ray.d[2]);
        int _6572 = ray.depth & 255;
        int _6577 = (ray.depth >> 8) & 255;
        int _6582 = (ray.depth >> 16) & 255;
        int _6593 = (_6572 + _6577) + _6582;
        int _6601 = _3268_g_params.hi + ((_6593 + ((ray.depth >> 24) & 255)) * 9);
        uint param = uint(hash(ray.xy));
        float _6608 = construct_float(param);
        uint param_1 = uint(hash(hash(ray.xy)));
        float _6615 = construct_float(param_1);
        float2 _6634 = float2(frac(asfloat(_3252.Load((_6601 + 7) * 4 + 0)) + _6608), frac(asfloat(_3252.Load((_6601 + 8) * 4 + 0)) + _6615));
        [branch]
        if (inter.mask == 0)
        {
            ray_data_t param_2 = ray;
            float3 _6644 = Evaluate_EnvColor(param_2, _6634);
            _8823 = float3(ray.c[0] * _6644.x, ray.c[1] * _6644.y, ray.c[2] * _6644.z);
            break;
        }
        float3 _6671 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_6568 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            ray_data_t param_3 = ray;
            hit_data_t param_4 = inter;
            float3 _6684 = Evaluate_LightColor(param_3, param_4, _6634);
            _8823 = float3(ray.c[0] * _6684.x, ray.c[1] * _6684.y, ray.c[2] * _6684.z);
            break;
        }
        bool _6705 = inter.prim_index < 0;
        int _6708;
        if (_6705)
        {
            _6708 = (-1) - inter.prim_index;
        }
        else
        {
            _6708 = inter.prim_index;
        }
        uint _6719 = uint(_6708);
        material_t _6727;
        [unroll]
        for (int _61ident = 0; _61ident < 5; _61ident++)
        {
            _6727.textures[_61ident] = _4415.Load(_61ident * 4 + ((_4419.Load(_6719 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
        }
        [unroll]
        for (int _62ident = 0; _62ident < 3; _62ident++)
        {
            _6727.base_color[_62ident] = asfloat(_4415.Load(_62ident * 4 + ((_4419.Load(_6719 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
        }
        _6727.flags = _4415.Load(((_4419.Load(_6719 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
        _6727.type = _4415.Load(((_4419.Load(_6719 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
        _6727.tangent_rotation_or_strength = asfloat(_4415.Load(((_4419.Load(_6719 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
        _6727.roughness_and_anisotropic = _4415.Load(((_4419.Load(_6719 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
        _6727.ior = asfloat(_4415.Load(((_4419.Load(_6719 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
        _6727.sheen_and_sheen_tint = _4415.Load(((_4419.Load(_6719 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
        _6727.tint_and_metallic = _4415.Load(((_4419.Load(_6719 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
        _6727.transmission_and_transmission_roughness = _4415.Load(((_4419.Load(_6719 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
        _6727.specular_and_specular_tint = _4415.Load(((_4419.Load(_6719 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
        _6727.clearcoat_and_clearcoat_roughness = _4415.Load(((_4419.Load(_6719 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
        _6727.normal_map_strength_unorm = _4415.Load(((_4419.Load(_6719 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
        uint _9583 = _6727.textures[0];
        uint _9584 = _6727.textures[1];
        uint _9585 = _6727.textures[2];
        uint _9586 = _6727.textures[3];
        uint _9587 = _6727.textures[4];
        float _9588 = _6727.base_color[0];
        float _9589 = _6727.base_color[1];
        float _9590 = _6727.base_color[2];
        uint _9184 = _6727.flags;
        uint _9185 = _6727.type;
        float _9186 = _6727.tangent_rotation_or_strength;
        uint _9187 = _6727.roughness_and_anisotropic;
        float _9188 = _6727.ior;
        uint _9189 = _6727.sheen_and_sheen_tint;
        uint _9190 = _6727.tint_and_metallic;
        uint _9191 = _6727.transmission_and_transmission_roughness;
        uint _9192 = _6727.specular_and_specular_tint;
        uint _9193 = _6727.clearcoat_and_clearcoat_roughness;
        uint _9194 = _6727.normal_map_strength_unorm;
        transform_t _6782;
        _6782.xform = asfloat(uint4x4(_4062.Load4(asuint(asfloat(_6775.Load(inter.obj_index * 32 + 12))) * 128 + 0), _4062.Load4(asuint(asfloat(_6775.Load(inter.obj_index * 32 + 12))) * 128 + 16), _4062.Load4(asuint(asfloat(_6775.Load(inter.obj_index * 32 + 12))) * 128 + 32), _4062.Load4(asuint(asfloat(_6775.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _6782.inv_xform = asfloat(uint4x4(_4062.Load4(asuint(asfloat(_6775.Load(inter.obj_index * 32 + 12))) * 128 + 64), _4062.Load4(asuint(asfloat(_6775.Load(inter.obj_index * 32 + 12))) * 128 + 80), _4062.Load4(asuint(asfloat(_6775.Load(inter.obj_index * 32 + 12))) * 128 + 96), _4062.Load4(asuint(asfloat(_6775.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _6789 = _6719 * 3u;
        vertex_t _6794;
        [unroll]
        for (int _63ident = 0; _63ident < 3; _63ident++)
        {
            _6794.p[_63ident] = asfloat(_4087.Load(_63ident * 4 + _4091.Load(_6789 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _64ident = 0; _64ident < 3; _64ident++)
        {
            _6794.n[_64ident] = asfloat(_4087.Load(_64ident * 4 + _4091.Load(_6789 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _65ident = 0; _65ident < 3; _65ident++)
        {
            _6794.b[_65ident] = asfloat(_4087.Load(_65ident * 4 + _4091.Load(_6789 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _66ident = 0; _66ident < 2; _66ident++)
        {
            [unroll]
            for (int _67ident = 0; _67ident < 2; _67ident++)
            {
                _6794.t[_66ident][_67ident] = asfloat(_4087.Load(_67ident * 4 + _66ident * 8 + _4091.Load(_6789 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _6840;
        [unroll]
        for (int _68ident = 0; _68ident < 3; _68ident++)
        {
            _6840.p[_68ident] = asfloat(_4087.Load(_68ident * 4 + _4091.Load((_6789 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _69ident = 0; _69ident < 3; _69ident++)
        {
            _6840.n[_69ident] = asfloat(_4087.Load(_69ident * 4 + _4091.Load((_6789 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _70ident = 0; _70ident < 3; _70ident++)
        {
            _6840.b[_70ident] = asfloat(_4087.Load(_70ident * 4 + _4091.Load((_6789 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _71ident = 0; _71ident < 2; _71ident++)
        {
            [unroll]
            for (int _72ident = 0; _72ident < 2; _72ident++)
            {
                _6840.t[_71ident][_72ident] = asfloat(_4087.Load(_72ident * 4 + _71ident * 8 + _4091.Load((_6789 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _6886;
        [unroll]
        for (int _73ident = 0; _73ident < 3; _73ident++)
        {
            _6886.p[_73ident] = asfloat(_4087.Load(_73ident * 4 + _4091.Load((_6789 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _74ident = 0; _74ident < 3; _74ident++)
        {
            _6886.n[_74ident] = asfloat(_4087.Load(_74ident * 4 + _4091.Load((_6789 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _75ident = 0; _75ident < 3; _75ident++)
        {
            _6886.b[_75ident] = asfloat(_4087.Load(_75ident * 4 + _4091.Load((_6789 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _76ident = 0; _76ident < 2; _76ident++)
        {
            [unroll]
            for (int _77ident = 0; _77ident < 2; _77ident++)
            {
                _6886.t[_76ident][_77ident] = asfloat(_4087.Load(_77ident * 4 + _76ident * 8 + _4091.Load((_6789 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _6932 = float3(_6794.p[0], _6794.p[1], _6794.p[2]);
        float3 _6940 = float3(_6840.p[0], _6840.p[1], _6840.p[2]);
        float3 _6948 = float3(_6886.p[0], _6886.p[1], _6886.p[2]);
        float _6955 = (1.0f - inter.u) - inter.v;
        float3 _6987 = normalize(((float3(_6794.n[0], _6794.n[1], _6794.n[2]) * _6955) + (float3(_6840.n[0], _6840.n[1], _6840.n[2]) * inter.u)) + (float3(_6886.n[0], _6886.n[1], _6886.n[2]) * inter.v));
        float3 _9123 = _6987;
        float2 _7013 = ((float2(_6794.t[0][0], _6794.t[0][1]) * _6955) + (float2(_6840.t[0][0], _6840.t[0][1]) * inter.u)) + (float2(_6886.t[0][0], _6886.t[0][1]) * inter.v);
        float3 _7029 = cross(_6940 - _6932, _6948 - _6932);
        float _7034 = length(_7029);
        float3 _9124 = _7029 / _7034.xxx;
        float3 _7071 = ((float3(_6794.b[0], _6794.b[1], _6794.b[2]) * _6955) + (float3(_6840.b[0], _6840.b[1], _6840.b[2]) * inter.u)) + (float3(_6886.b[0], _6886.b[1], _6886.b[2]) * inter.v);
        float3 _9122 = _7071;
        float3 _9121 = cross(_7071, _6987);
        if (_6705)
        {
            if ((_4419.Load(_6719 * 4 + 0) & 65535u) == 65535u)
            {
                _8823 = 0.0f.xxx;
                break;
            }
            material_t _7097;
            [unroll]
            for (int _78ident = 0; _78ident < 5; _78ident++)
            {
                _7097.textures[_78ident] = _4415.Load(_78ident * 4 + (_4419.Load(_6719 * 4 + 0) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _79ident = 0; _79ident < 3; _79ident++)
            {
                _7097.base_color[_79ident] = asfloat(_4415.Load(_79ident * 4 + (_4419.Load(_6719 * 4 + 0) & 16383u) * 76 + 20));
            }
            _7097.flags = _4415.Load((_4419.Load(_6719 * 4 + 0) & 16383u) * 76 + 32);
            _7097.type = _4415.Load((_4419.Load(_6719 * 4 + 0) & 16383u) * 76 + 36);
            _7097.tangent_rotation_or_strength = asfloat(_4415.Load((_4419.Load(_6719 * 4 + 0) & 16383u) * 76 + 40));
            _7097.roughness_and_anisotropic = _4415.Load((_4419.Load(_6719 * 4 + 0) & 16383u) * 76 + 44);
            _7097.ior = asfloat(_4415.Load((_4419.Load(_6719 * 4 + 0) & 16383u) * 76 + 48));
            _7097.sheen_and_sheen_tint = _4415.Load((_4419.Load(_6719 * 4 + 0) & 16383u) * 76 + 52);
            _7097.tint_and_metallic = _4415.Load((_4419.Load(_6719 * 4 + 0) & 16383u) * 76 + 56);
            _7097.transmission_and_transmission_roughness = _4415.Load((_4419.Load(_6719 * 4 + 0) & 16383u) * 76 + 60);
            _7097.specular_and_specular_tint = _4415.Load((_4419.Load(_6719 * 4 + 0) & 16383u) * 76 + 64);
            _7097.clearcoat_and_clearcoat_roughness = _4415.Load((_4419.Load(_6719 * 4 + 0) & 16383u) * 76 + 68);
            _7097.normal_map_strength_unorm = _4415.Load((_4419.Load(_6719 * 4 + 0) & 16383u) * 76 + 72);
            _9583 = _7097.textures[0];
            _9584 = _7097.textures[1];
            _9585 = _7097.textures[2];
            _9586 = _7097.textures[3];
            _9587 = _7097.textures[4];
            _9588 = _7097.base_color[0];
            _9589 = _7097.base_color[1];
            _9590 = _7097.base_color[2];
            _9184 = _7097.flags;
            _9185 = _7097.type;
            _9186 = _7097.tangent_rotation_or_strength;
            _9187 = _7097.roughness_and_anisotropic;
            _9188 = _7097.ior;
            _9189 = _7097.sheen_and_sheen_tint;
            _9190 = _7097.tint_and_metallic;
            _9191 = _7097.transmission_and_transmission_roughness;
            _9192 = _7097.specular_and_specular_tint;
            _9193 = _7097.clearcoat_and_clearcoat_roughness;
            _9194 = _7097.normal_map_strength_unorm;
            _9124 = -_9124;
            _9123 = -_9123;
            _9122 = -_9122;
            _9121 = -_9121;
        }
        float3 param_5 = _9124;
        float4x4 param_6 = _6782.inv_xform;
        _9124 = TransformNormal(param_5, param_6);
        float3 param_7 = _9123;
        float4x4 param_8 = _6782.inv_xform;
        _9123 = TransformNormal(param_7, param_8);
        float3 param_9 = _9122;
        float4x4 param_10 = _6782.inv_xform;
        _9122 = TransformNormal(param_9, param_10);
        float3 param_11 = _9121;
        float4x4 param_12 = _6782.inv_xform;
        _9124 = normalize(_9124);
        _9123 = normalize(_9123);
        _9122 = normalize(_9122);
        _9121 = normalize(TransformNormal(param_11, param_12));
        float _7237 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _7247 = mad(0.5f, log2(abs(mad(_6840.t[0][0] - _6794.t[0][0], _6886.t[0][1] - _6794.t[0][1], -((_6886.t[0][0] - _6794.t[0][0]) * (_6840.t[0][1] - _6794.t[0][1])))) / _7034), log2(_7237));
        float param_13[4] = ray.ior;
        bool param_14 = _6705;
        float param_15 = 1.0f;
        float _7255 = peek_ior_stack(param_13, param_14, param_15);
        float3 col = 0.0f.xxx;
        float mix_rand = frac(asfloat(_3252.Load(_6601 * 4 + 0)) + _6608);
        float mix_weight = 1.0f;
        float _7294;
        float _7311;
        float _7337;
        float _7404;
        while (_9185 == 4u)
        {
            float mix_val = _9186;
            if (_9584 != 4294967295u)
            {
                mix_val *= SampleBilinear(_9584, _7013, 0, _6634).x;
            }
            if (_6705)
            {
                _7294 = _7255 / _9188;
            }
            else
            {
                _7294 = _9188 / _7255;
            }
            if (_9188 != 0.0f)
            {
                float param_16 = dot(_6568, _9123);
                float param_17 = _7294;
                _7311 = fresnel_dielectric_cos(param_16, param_17);
            }
            else
            {
                _7311 = 1.0f;
            }
            float _7326 = mix_val;
            float _7327 = _7326 * clamp(_7311, 0.0f, 1.0f);
            mix_val = _7327;
            if (mix_rand > _7327)
            {
                if ((_9184 & 2u) != 0u)
                {
                    _7337 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _7337 = 1.0f;
                }
                mix_weight *= _7337;
                material_t _7350;
                [unroll]
                for (int _80ident = 0; _80ident < 5; _80ident++)
                {
                    _7350.textures[_80ident] = _4415.Load(_80ident * 4 + _9586 * 76 + 0);
                }
                [unroll]
                for (int _81ident = 0; _81ident < 3; _81ident++)
                {
                    _7350.base_color[_81ident] = asfloat(_4415.Load(_81ident * 4 + _9586 * 76 + 20));
                }
                _7350.flags = _4415.Load(_9586 * 76 + 32);
                _7350.type = _4415.Load(_9586 * 76 + 36);
                _7350.tangent_rotation_or_strength = asfloat(_4415.Load(_9586 * 76 + 40));
                _7350.roughness_and_anisotropic = _4415.Load(_9586 * 76 + 44);
                _7350.ior = asfloat(_4415.Load(_9586 * 76 + 48));
                _7350.sheen_and_sheen_tint = _4415.Load(_9586 * 76 + 52);
                _7350.tint_and_metallic = _4415.Load(_9586 * 76 + 56);
                _7350.transmission_and_transmission_roughness = _4415.Load(_9586 * 76 + 60);
                _7350.specular_and_specular_tint = _4415.Load(_9586 * 76 + 64);
                _7350.clearcoat_and_clearcoat_roughness = _4415.Load(_9586 * 76 + 68);
                _7350.normal_map_strength_unorm = _4415.Load(_9586 * 76 + 72);
                _9583 = _7350.textures[0];
                _9584 = _7350.textures[1];
                _9585 = _7350.textures[2];
                _9586 = _7350.textures[3];
                _9587 = _7350.textures[4];
                _9588 = _7350.base_color[0];
                _9589 = _7350.base_color[1];
                _9590 = _7350.base_color[2];
                _9184 = _7350.flags;
                _9185 = _7350.type;
                _9186 = _7350.tangent_rotation_or_strength;
                _9187 = _7350.roughness_and_anisotropic;
                _9188 = _7350.ior;
                _9189 = _7350.sheen_and_sheen_tint;
                _9190 = _7350.tint_and_metallic;
                _9191 = _7350.transmission_and_transmission_roughness;
                _9192 = _7350.specular_and_specular_tint;
                _9193 = _7350.clearcoat_and_clearcoat_roughness;
                _9194 = _7350.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_9184 & 2u) != 0u)
                {
                    _7404 = 1.0f / mix_val;
                }
                else
                {
                    _7404 = 1.0f;
                }
                mix_weight *= _7404;
                material_t _7416;
                [unroll]
                for (int _82ident = 0; _82ident < 5; _82ident++)
                {
                    _7416.textures[_82ident] = _4415.Load(_82ident * 4 + _9587 * 76 + 0);
                }
                [unroll]
                for (int _83ident = 0; _83ident < 3; _83ident++)
                {
                    _7416.base_color[_83ident] = asfloat(_4415.Load(_83ident * 4 + _9587 * 76 + 20));
                }
                _7416.flags = _4415.Load(_9587 * 76 + 32);
                _7416.type = _4415.Load(_9587 * 76 + 36);
                _7416.tangent_rotation_or_strength = asfloat(_4415.Load(_9587 * 76 + 40));
                _7416.roughness_and_anisotropic = _4415.Load(_9587 * 76 + 44);
                _7416.ior = asfloat(_4415.Load(_9587 * 76 + 48));
                _7416.sheen_and_sheen_tint = _4415.Load(_9587 * 76 + 52);
                _7416.tint_and_metallic = _4415.Load(_9587 * 76 + 56);
                _7416.transmission_and_transmission_roughness = _4415.Load(_9587 * 76 + 60);
                _7416.specular_and_specular_tint = _4415.Load(_9587 * 76 + 64);
                _7416.clearcoat_and_clearcoat_roughness = _4415.Load(_9587 * 76 + 68);
                _7416.normal_map_strength_unorm = _4415.Load(_9587 * 76 + 72);
                _9583 = _7416.textures[0];
                _9584 = _7416.textures[1];
                _9585 = _7416.textures[2];
                _9586 = _7416.textures[3];
                _9587 = _7416.textures[4];
                _9588 = _7416.base_color[0];
                _9589 = _7416.base_color[1];
                _9590 = _7416.base_color[2];
                _9184 = _7416.flags;
                _9185 = _7416.type;
                _9186 = _7416.tangent_rotation_or_strength;
                _9187 = _7416.roughness_and_anisotropic;
                _9188 = _7416.ior;
                _9189 = _7416.sheen_and_sheen_tint;
                _9190 = _7416.tint_and_metallic;
                _9191 = _7416.transmission_and_transmission_roughness;
                _9192 = _7416.specular_and_specular_tint;
                _9193 = _7416.clearcoat_and_clearcoat_roughness;
                _9194 = _7416.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_9583 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_9583, _7013, 0, _6634).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_9583 & 33554432u) != 0u)
            {
                float3 _10151 = normals;
                _10151.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _10151;
            }
            float3 _7499 = _9123;
            _9123 = normalize(((_9121 * normals.x) + (_7499 * normals.z)) + (_9122 * normals.y));
            if ((_9194 & 65535u) != 65535u)
            {
                _9123 = normalize(_7499 + ((_9123 - _7499) * clamp(float(_9194 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_18 = _9124;
            float3 param_19 = -_6568;
            float3 param_20 = _9123;
            _9123 = ensure_valid_reflection(param_18, param_19, param_20);
        }
        float3 _7565 = ((_6932 * _6955) + (_6940 * inter.u)) + (_6948 * inter.v);
        float3 _7572 = float3(-_7565.z, 0.0f, _7565.x);
        float3 tangent = _7572;
        float3 param_21 = _7572;
        float4x4 param_22 = _6782.inv_xform;
        float3 _7578 = TransformNormal(param_21, param_22);
        tangent = _7578;
        float3 _7582 = cross(_7578, _9123);
        if (dot(_7582, _7582) == 0.0f)
        {
            float3 param_23 = _7565;
            float4x4 param_24 = _6782.inv_xform;
            tangent = TransformNormal(param_23, param_24);
        }
        if (_9186 != 0.0f)
        {
            float3 param_25 = tangent;
            float3 param_26 = _9123;
            float param_27 = _9186;
            tangent = rotate_around_axis(param_25, param_26, param_27);
        }
        float3 _7615 = normalize(cross(tangent, _9123));
        _9122 = _7615;
        _9121 = cross(_9123, _7615);
        float3 _9282 = 0.0f.xxx;
        float3 _9281 = 0.0f.xxx;
        float _9286 = 0.0f;
        float _9284 = 0.0f;
        float _9285 = 1.0f;
        bool _7631 = _3268_g_params.li_count != 0;
        bool _7637;
        if (_7631)
        {
            _7637 = _9185 != 3u;
        }
        else
        {
            _7637 = _7631;
        }
        float3 _9283;
        bool _9287;
        bool _9288;
        if (_7637)
        {
            float3 param_28 = _6671;
            float3 param_29 = _9121;
            float3 param_30 = _9122;
            float3 param_31 = _9123;
            int param_32 = _6601;
            float2 param_33 = float2(_6608, _6615);
            light_sample_t _9297 = { _9281, _9282, _9283, _9284, _9285, _9286, _9287, _9288 };
            light_sample_t param_34 = _9297;
            SampleLightSource(param_28, param_29, param_30, param_31, param_32, param_33, param_34);
            _9281 = param_34.col;
            _9282 = param_34.L;
            _9283 = param_34.lp;
            _9284 = param_34.area;
            _9285 = param_34.dist_mul;
            _9286 = param_34.pdf;
            _9287 = param_34.cast_shadow;
            _9288 = param_34.from_env;
        }
        float _7665 = dot(_9123, _9282);
        float3 base_color = float3(_9588, _9589, _9590);
        [branch]
        if (_9584 != 4294967295u)
        {
            base_color *= SampleBilinear(_9584, _7013, int(get_texture_lod(texSize(_9584), _7247)), _6634, true, true).xyz;
        }
        out_base_color = base_color;
        out_normals = _9123;
        float3 tint_color = 0.0f.xxx;
        float _7702 = lum(base_color);
        [flatten]
        if (_7702 > 0.0f)
        {
            tint_color = base_color / _7702.xxx;
        }
        float roughness = clamp(float(_9187 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_9585 != 4294967295u)
        {
            roughness *= SampleBilinear(_9585, _7013, int(get_texture_lod(texSize(_9585), _7247)), _6634, false, true).x;
        }
        float _7748 = frac(asfloat(_3252.Load((_6601 + 1) * 4 + 0)) + _6608);
        float _7757 = frac(asfloat(_3252.Load((_6601 + 2) * 4 + 0)) + _6615);
        float _9710 = 0.0f;
        float _9709 = 0.0f;
        float _9708 = 0.0f;
        float _9346[4];
        [unroll]
        for (int i = 0; i < 4; )
        {
            _9346[i] = ray.ior[i];
            i++;
            continue;
        }
        float _9347 = _7237;
        float _9348 = ray.cone_spread;
        int _9349 = ray.xy;
        float _9344 = 0.0f;
        float _9815 = 0.0f;
        float _9814 = 0.0f;
        float _9813 = 0.0f;
        int _9451 = ray.depth;
        int _9455 = ray.xy;
        int _9350;
        float _9453;
        float _9638;
        float _9639;
        float _9640;
        float _9673;
        float _9674;
        float _9675;
        float _9743;
        float _9744;
        float _9745;
        float _9778;
        float _9779;
        float _9780;
        [branch]
        if (_9185 == 0u)
        {
            [branch]
            if ((_9286 > 0.0f) && (_7665 > 0.0f))
            {
                light_sample_t _9314 = { _9281, _9282, _9283, _9284, _9285, _9286, _9287, _9288 };
                surface_t _9132 = { _6671, _9121, _9122, _9123, _9124, _7013 };
                float _9819[3] = { _9813, _9814, _9815 };
                float _9784[3] = { _9778, _9779, _9780 };
                float _9749[3] = { _9743, _9744, _9745 };
                shadow_ray_t _9465 = { _9749, _9451, _9784, _9453, _9819, _9455 };
                shadow_ray_t param_35 = _9465;
                float3 _7817 = Evaluate_DiffuseNode(_9314, ray, _9132, base_color, roughness, mix_weight, param_35);
                _9743 = param_35.o[0];
                _9744 = param_35.o[1];
                _9745 = param_35.o[2];
                _9451 = param_35.depth;
                _9778 = param_35.d[0];
                _9779 = param_35.d[1];
                _9780 = param_35.d[2];
                _9453 = param_35.dist;
                _9813 = param_35.c[0];
                _9814 = param_35.c[1];
                _9815 = param_35.c[2];
                _9455 = param_35.xy;
                col += _7817;
            }
            bool _7824 = _6572 < _3268_g_params.max_diff_depth;
            bool _7831;
            if (_7824)
            {
                _7831 = _6593 < _3268_g_params.max_total_depth;
            }
            else
            {
                _7831 = _7824;
            }
            [branch]
            if (_7831)
            {
                surface_t _9139 = { _6671, _9121, _9122, _9123, _9124, _7013 };
                float _9714[3] = { _9708, _9709, _9710 };
                float _9679[3] = { _9673, _9674, _9675 };
                float _9644[3] = { _9638, _9639, _9640 };
                ray_data_t _9364 = { _9644, _9679, _9344, _9714, _9346, _9347, _9348, _9349, _9350 };
                ray_data_t param_36 = _9364;
                Sample_DiffuseNode(ray, _9139, base_color, roughness, _7748, _7757, mix_weight, param_36);
                _9638 = param_36.o[0];
                _9639 = param_36.o[1];
                _9640 = param_36.o[2];
                _9673 = param_36.d[0];
                _9674 = param_36.d[1];
                _9675 = param_36.d[2];
                _9344 = param_36.pdf;
                _9708 = param_36.c[0];
                _9709 = param_36.c[1];
                _9710 = param_36.c[2];
                _9346 = param_36.ior;
                _9347 = param_36.cone_width;
                _9348 = param_36.cone_spread;
                _9349 = param_36.xy;
                _9350 = param_36.depth;
            }
        }
        else
        {
            [branch]
            if (_9185 == 1u)
            {
                float param_37 = 1.0f;
                float param_38 = 1.5f;
                float _7855 = fresnel_dielectric_cos(param_37, param_38);
                float _7859 = roughness * roughness;
                bool _7862 = _9286 > 0.0f;
                bool _7869;
                if (_7862)
                {
                    _7869 = (_7859 * _7859) >= 1.0000000116860974230803549289703e-07f;
                }
                else
                {
                    _7869 = _7862;
                }
                [branch]
                if (_7869 && (_7665 > 0.0f))
                {
                    light_sample_t _9323 = { _9281, _9282, _9283, _9284, _9285, _9286, _9287, _9288 };
                    surface_t _9146 = { _6671, _9121, _9122, _9123, _9124, _7013 };
                    float _9826[3] = { _9813, _9814, _9815 };
                    float _9791[3] = { _9778, _9779, _9780 };
                    float _9756[3] = { _9743, _9744, _9745 };
                    shadow_ray_t _9478 = { _9756, _9451, _9791, _9453, _9826, _9455 };
                    shadow_ray_t param_39 = _9478;
                    float3 _7884 = Evaluate_GlossyNode(_9323, ray, _9146, base_color, roughness, 1.5f, _7855, mix_weight, param_39);
                    _9743 = param_39.o[0];
                    _9744 = param_39.o[1];
                    _9745 = param_39.o[2];
                    _9451 = param_39.depth;
                    _9778 = param_39.d[0];
                    _9779 = param_39.d[1];
                    _9780 = param_39.d[2];
                    _9453 = param_39.dist;
                    _9813 = param_39.c[0];
                    _9814 = param_39.c[1];
                    _9815 = param_39.c[2];
                    _9455 = param_39.xy;
                    col += _7884;
                }
                bool _7891 = _6577 < _3268_g_params.max_spec_depth;
                bool _7898;
                if (_7891)
                {
                    _7898 = _6593 < _3268_g_params.max_total_depth;
                }
                else
                {
                    _7898 = _7891;
                }
                [branch]
                if (_7898)
                {
                    surface_t _9153 = { _6671, _9121, _9122, _9123, _9124, _7013 };
                    float _9721[3] = { _9708, _9709, _9710 };
                    float _9686[3] = { _9673, _9674, _9675 };
                    float _9651[3] = { _9638, _9639, _9640 };
                    ray_data_t _9383 = { _9651, _9686, _9344, _9721, _9346, _9347, _9348, _9349, _9350 };
                    ray_data_t param_40 = _9383;
                    Sample_GlossyNode(ray, _9153, base_color, roughness, 1.5f, _7855, _7748, _7757, mix_weight, param_40);
                    _9638 = param_40.o[0];
                    _9639 = param_40.o[1];
                    _9640 = param_40.o[2];
                    _9673 = param_40.d[0];
                    _9674 = param_40.d[1];
                    _9675 = param_40.d[2];
                    _9344 = param_40.pdf;
                    _9708 = param_40.c[0];
                    _9709 = param_40.c[1];
                    _9710 = param_40.c[2];
                    _9346 = param_40.ior;
                    _9347 = param_40.cone_width;
                    _9348 = param_40.cone_spread;
                    _9349 = param_40.xy;
                    _9350 = param_40.depth;
                }
            }
            else
            {
                [branch]
                if (_9185 == 2u)
                {
                    float _7922 = roughness * roughness;
                    bool _7925 = _9286 > 0.0f;
                    bool _7932;
                    if (_7925)
                    {
                        _7932 = (_7922 * _7922) >= 1.0000000116860974230803549289703e-07f;
                    }
                    else
                    {
                        _7932 = _7925;
                    }
                    [branch]
                    if (_7932 && (_7665 < 0.0f))
                    {
                        float _7940;
                        if (_6705)
                        {
                            _7940 = _9188 / _7255;
                        }
                        else
                        {
                            _7940 = _7255 / _9188;
                        }
                        light_sample_t _9332 = { _9281, _9282, _9283, _9284, _9285, _9286, _9287, _9288 };
                        surface_t _9160 = { _6671, _9121, _9122, _9123, _9124, _7013 };
                        float _9833[3] = { _9813, _9814, _9815 };
                        float _9798[3] = { _9778, _9779, _9780 };
                        float _9763[3] = { _9743, _9744, _9745 };
                        shadow_ray_t _9491 = { _9763, _9451, _9798, _9453, _9833, _9455 };
                        shadow_ray_t param_41 = _9491;
                        float3 _7962 = Evaluate_RefractiveNode(_9332, ray, _9160, base_color, _7922, _7940, mix_weight, param_41);
                        _9743 = param_41.o[0];
                        _9744 = param_41.o[1];
                        _9745 = param_41.o[2];
                        _9451 = param_41.depth;
                        _9778 = param_41.d[0];
                        _9779 = param_41.d[1];
                        _9780 = param_41.d[2];
                        _9453 = param_41.dist;
                        _9813 = param_41.c[0];
                        _9814 = param_41.c[1];
                        _9815 = param_41.c[2];
                        _9455 = param_41.xy;
                        col += _7962;
                    }
                    bool _7969 = _6582 < _3268_g_params.max_refr_depth;
                    bool _7976;
                    if (_7969)
                    {
                        _7976 = _6593 < _3268_g_params.max_total_depth;
                    }
                    else
                    {
                        _7976 = _7969;
                    }
                    [branch]
                    if (_7976)
                    {
                        surface_t _9167 = { _6671, _9121, _9122, _9123, _9124, _7013 };
                        float _9728[3] = { _9708, _9709, _9710 };
                        float _9693[3] = { _9673, _9674, _9675 };
                        float _9658[3] = { _9638, _9639, _9640 };
                        ray_data_t _9402 = { _9658, _9693, _9344, _9728, _9346, _9347, _9348, _9349, _9350 };
                        ray_data_t param_42 = _9402;
                        Sample_RefractiveNode(ray, _9167, base_color, roughness, _6705, _9188, _7255, _7748, _7757, mix_weight, param_42);
                        _9638 = param_42.o[0];
                        _9639 = param_42.o[1];
                        _9640 = param_42.o[2];
                        _9673 = param_42.d[0];
                        _9674 = param_42.d[1];
                        _9675 = param_42.d[2];
                        _9344 = param_42.pdf;
                        _9708 = param_42.c[0];
                        _9709 = param_42.c[1];
                        _9710 = param_42.c[2];
                        _9346 = param_42.ior;
                        _9347 = param_42.cone_width;
                        _9348 = param_42.cone_spread;
                        _9349 = param_42.xy;
                        _9350 = param_42.depth;
                    }
                }
                else
                {
                    [branch]
                    if (_9185 == 3u)
                    {
                        float mis_weight = 1.0f;
                        [branch]
                        if ((_9184 & 1u) != 0u)
                        {
                            float3 _8046 = mul(float4(_7029, 0.0f), _6782.xform).xyz;
                            float _8049 = length(_8046);
                            float _8061 = abs(dot(_6568, _8046 / _8049.xxx));
                            if (_8061 > 0.0f)
                            {
                                float param_43 = ray.pdf;
                                float param_44 = (inter.t * inter.t) / ((0.5f * _8049) * _8061);
                                mis_weight = power_heuristic(param_43, param_44);
                            }
                        }
                        col += (base_color * ((mix_weight * mis_weight) * _9186));
                    }
                    else
                    {
                        [branch]
                        if (_9185 == 6u)
                        {
                            float metallic = clamp(float((_9190 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_9586 != 4294967295u)
                            {
                                metallic *= SampleBilinear(_9586, _7013, int(get_texture_lod(texSize(_9586), _7247)), _6634).x;
                            }
                            float specular = clamp(float(_9192 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_9587 != 4294967295u)
                            {
                                specular *= SampleBilinear(_9587, _7013, int(get_texture_lod(texSize(_9587), _7247)), _6634).x;
                            }
                            float _8180 = clamp(float(_9193 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8188 = clamp(float((_9193 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8196 = 2.0f * clamp(float(_9189 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float3 _8214 = lerp(1.0f.xxx, tint_color, clamp(float((_9189 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _8196;
                            float3 _8234 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_9192 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                            float _8243 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                            float param_45 = 1.0f;
                            float param_46 = _8243;
                            float _8249 = fresnel_dielectric_cos(param_45, param_46);
                            float _8257 = clamp(float((_9187 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8268 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _8180))) - 1.0f;
                            float param_47 = 1.0f;
                            float param_48 = _8268;
                            float _8274 = fresnel_dielectric_cos(param_47, param_48);
                            float _8289 = mad(roughness - 1.0f, 1.0f - clamp(float((_9191 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                            float _8295;
                            if (_6705)
                            {
                                _8295 = _9188 / _7255;
                            }
                            else
                            {
                                _8295 = _7255 / _9188;
                            }
                            float param_49 = dot(_6568, _9123);
                            float param_50 = 1.0f / _8295;
                            float _8318 = fresnel_dielectric_cos(param_49, param_50);
                            float param_51 = dot(_6568, _9123);
                            float param_52 = _8243;
                            lobe_weights_t _8357 = get_lobe_weights(lerp(_7702, 1.0f, _8196), lum(lerp(_8234, 1.0f.xxx, ((fresnel_dielectric_cos(param_51, param_52) - _8249) / (1.0f - _8249)).xxx)), specular, metallic, clamp(float(_9191 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _8180);
                            [branch]
                            if (_9286 > 0.0f)
                            {
                                light_sample_t _9341 = { _9281, _9282, _9283, _9284, _9285, _9286, _9287, _9288 };
                                surface_t _9174 = { _6671, _9121, _9122, _9123, _9124, _7013 };
                                diff_params_t _9533 = { base_color, _8214, roughness };
                                spec_params_t _9548 = { _8234, roughness, _8243, _8249, _8257 };
                                clearcoat_params_t _9561 = { _8188, _8268, _8274 };
                                transmission_params_t _9576 = { _8289, _9188, _8295, _8318, _6705 };
                                float _9840[3] = { _9813, _9814, _9815 };
                                float _9805[3] = { _9778, _9779, _9780 };
                                float _9770[3] = { _9743, _9744, _9745 };
                                shadow_ray_t _9504 = { _9770, _9451, _9805, _9453, _9840, _9455 };
                                shadow_ray_t param_53 = _9504;
                                float3 _8376 = Evaluate_PrincipledNode(_9341, ray, _9174, _8357, _9533, _9548, _9561, _9576, metallic, _7665, mix_weight, param_53);
                                _9743 = param_53.o[0];
                                _9744 = param_53.o[1];
                                _9745 = param_53.o[2];
                                _9451 = param_53.depth;
                                _9778 = param_53.d[0];
                                _9779 = param_53.d[1];
                                _9780 = param_53.d[2];
                                _9453 = param_53.dist;
                                _9813 = param_53.c[0];
                                _9814 = param_53.c[1];
                                _9815 = param_53.c[2];
                                _9455 = param_53.xy;
                                col += _8376;
                            }
                            surface_t _9181 = { _6671, _9121, _9122, _9123, _9124, _7013 };
                            diff_params_t _9537 = { base_color, _8214, roughness };
                            spec_params_t _9554 = { _8234, roughness, _8243, _8249, _8257 };
                            clearcoat_params_t _9565 = { _8188, _8268, _8274 };
                            transmission_params_t _9582 = { _8289, _9188, _8295, _8318, _6705 };
                            float param_54 = mix_rand;
                            float _9735[3] = { _9708, _9709, _9710 };
                            float _9700[3] = { _9673, _9674, _9675 };
                            float _9665[3] = { _9638, _9639, _9640 };
                            ray_data_t _9421 = { _9665, _9700, _9344, _9735, _9346, _9347, _9348, _9349, _9350 };
                            ray_data_t param_55 = _9421;
                            Sample_PrincipledNode(ray, _9181, _8357, _9537, _9554, _9565, _9582, metallic, _7748, _7757, param_54, mix_weight, param_55);
                            _9638 = param_55.o[0];
                            _9639 = param_55.o[1];
                            _9640 = param_55.o[2];
                            _9673 = param_55.d[0];
                            _9674 = param_55.d[1];
                            _9675 = param_55.d[2];
                            _9344 = param_55.pdf;
                            _9708 = param_55.c[0];
                            _9709 = param_55.c[1];
                            _9710 = param_55.c[2];
                            _9346 = param_55.ior;
                            _9347 = param_55.cone_width;
                            _9348 = param_55.cone_spread;
                            _9349 = param_55.xy;
                            _9350 = param_55.depth;
                        }
                    }
                }
            }
        }
        float _8410 = max(_9708, max(_9709, _9710));
        float _8422;
        if (_6593 > _3268_g_params.min_total_depth)
        {
            _8422 = max(0.0500000007450580596923828125f, 1.0f - _8410);
        }
        else
        {
            _8422 = 0.0f;
        }
        bool _8436 = (frac(asfloat(_3252.Load((_6601 + 6) * 4 + 0)) + _6608) >= _8422) && (_8410 > 0.0f);
        bool _8442;
        if (_8436)
        {
            _8442 = _9344 > 0.0f;
        }
        else
        {
            _8442 = _8436;
        }
        [branch]
        if (_8442)
        {
            float _8446 = _9344;
            float _8447 = min(_8446, 1000000.0f);
            _9344 = _8447;
            float _8450 = 1.0f - _8422;
            float _8452 = _9708;
            float _8453 = _8452 / _8450;
            _9708 = _8453;
            float _8458 = _9709;
            float _8459 = _8458 / _8450;
            _9709 = _8459;
            float _8464 = _9710;
            float _8465 = _8464 / _8450;
            _9710 = _8465;
            uint _8473;
            _8471.InterlockedAdd(0, 1u, _8473);
            _8482.Store(_8473 * 72 + 0, asuint(_9638));
            _8482.Store(_8473 * 72 + 4, asuint(_9639));
            _8482.Store(_8473 * 72 + 8, asuint(_9640));
            _8482.Store(_8473 * 72 + 12, asuint(_9673));
            _8482.Store(_8473 * 72 + 16, asuint(_9674));
            _8482.Store(_8473 * 72 + 20, asuint(_9675));
            _8482.Store(_8473 * 72 + 24, asuint(_8447));
            _8482.Store(_8473 * 72 + 28, asuint(_8453));
            _8482.Store(_8473 * 72 + 32, asuint(_8459));
            _8482.Store(_8473 * 72 + 36, asuint(_8465));
            _8482.Store(_8473 * 72 + 40, asuint(_9346[0]));
            _8482.Store(_8473 * 72 + 44, asuint(_9346[1]));
            _8482.Store(_8473 * 72 + 48, asuint(_9346[2]));
            _8482.Store(_8473 * 72 + 52, asuint(_9346[3]));
            _8482.Store(_8473 * 72 + 56, asuint(_9347));
            _8482.Store(_8473 * 72 + 60, asuint(_9348));
            _8482.Store(_8473 * 72 + 64, uint(_9349));
            _8482.Store(_8473 * 72 + 68, uint(_9350));
        }
        [branch]
        if (max(_9813, max(_9814, _9815)) > 0.0f)
        {
            float3 _8559 = _9283 - float3(_9743, _9744, _9745);
            float _8562 = length(_8559);
            float3 _8566 = _8559 / _8562.xxx;
            float sh_dist = _8562 * _9285;
            if (_9288)
            {
                sh_dist = -sh_dist;
            }
            float _8578 = _8566.x;
            _9778 = _8578;
            float _8581 = _8566.y;
            _9779 = _8581;
            float _8584 = _8566.z;
            _9780 = _8584;
            _9453 = sh_dist;
            uint _8590;
            _8471.InterlockedAdd(8, 1u, _8590);
            _8598.Store(_8590 * 48 + 0, asuint(_9743));
            _8598.Store(_8590 * 48 + 4, asuint(_9744));
            _8598.Store(_8590 * 48 + 8, asuint(_9745));
            _8598.Store(_8590 * 48 + 12, uint(_9451));
            _8598.Store(_8590 * 48 + 16, asuint(_8578));
            _8598.Store(_8590 * 48 + 20, asuint(_8581));
            _8598.Store(_8590 * 48 + 24, asuint(_8584));
            _8598.Store(_8590 * 48 + 28, asuint(sh_dist));
            _8598.Store(_8590 * 48 + 32, asuint(_9813));
            _8598.Store(_8590 * 48 + 36, asuint(_9814));
            _8598.Store(_8590 * 48 + 40, asuint(_9815));
            _8598.Store(_8590 * 48 + 44, uint(_9455));
        }
        _8823 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _8823;
}

void comp_main()
{
    do
    {
        int _8664 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_8664) >= _8471.Load(4))
        {
            break;
        }
        int _8680 = int(_8677.Load(_8664 * 72 + 64));
        int _8687 = int(_8677.Load(_8664 * 72 + 64));
        hit_data_t _8698;
        _8698.mask = int(_8694.Load(_8664 * 24 + 0));
        _8698.obj_index = int(_8694.Load(_8664 * 24 + 4));
        _8698.prim_index = int(_8694.Load(_8664 * 24 + 8));
        _8698.t = asfloat(_8694.Load(_8664 * 24 + 12));
        _8698.u = asfloat(_8694.Load(_8664 * 24 + 16));
        _8698.v = asfloat(_8694.Load(_8664 * 24 + 20));
        ray_data_t _8714;
        [unroll]
        for (int _84ident = 0; _84ident < 3; _84ident++)
        {
            _8714.o[_84ident] = asfloat(_8677.Load(_84ident * 4 + _8664 * 72 + 0));
        }
        [unroll]
        for (int _85ident = 0; _85ident < 3; _85ident++)
        {
            _8714.d[_85ident] = asfloat(_8677.Load(_85ident * 4 + _8664 * 72 + 12));
        }
        _8714.pdf = asfloat(_8677.Load(_8664 * 72 + 24));
        [unroll]
        for (int _86ident = 0; _86ident < 3; _86ident++)
        {
            _8714.c[_86ident] = asfloat(_8677.Load(_86ident * 4 + _8664 * 72 + 28));
        }
        [unroll]
        for (int _87ident = 0; _87ident < 4; _87ident++)
        {
            _8714.ior[_87ident] = asfloat(_8677.Load(_87ident * 4 + _8664 * 72 + 40));
        }
        _8714.cone_width = asfloat(_8677.Load(_8664 * 72 + 56));
        _8714.cone_spread = asfloat(_8677.Load(_8664 * 72 + 60));
        _8714.xy = int(_8677.Load(_8664 * 72 + 64));
        _8714.depth = int(_8677.Load(_8664 * 72 + 68));
        hit_data_t _8917 = { _8698.mask, _8698.obj_index, _8698.prim_index, _8698.t, _8698.u, _8698.v };
        hit_data_t param = _8917;
        float _8966[4] = { _8714.ior[0], _8714.ior[1], _8714.ior[2], _8714.ior[3] };
        float _8957[3] = { _8714.c[0], _8714.c[1], _8714.c[2] };
        float _8950[3] = { _8714.d[0], _8714.d[1], _8714.d[2] };
        float _8943[3] = { _8714.o[0], _8714.o[1], _8714.o[2] };
        ray_data_t _8936 = { _8943, _8950, _8714.pdf, _8957, _8966, _8714.cone_width, _8714.cone_spread, _8714.xy, _8714.depth };
        ray_data_t param_1 = _8936;
        float3 param_2 = 0.0f.xxx;
        float3 param_3 = 0.0f.xxx;
        float3 _8770 = ShadeSurface(param, param_1, param_2, param_3);
        int2 _8784 = int2((_8680 >> 16) & 65535, _8687 & 65535);
        g_out_img[_8784] = float4(min(_8770, _3268_g_params.clamp_val.xxx) + g_out_img[_8784].xyz, 1.0f);
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
