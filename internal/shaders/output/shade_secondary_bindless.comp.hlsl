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

ByteAddressBuffer _3309 : register(t15, space0);
ByteAddressBuffer _3345 : register(t6, space0);
ByteAddressBuffer _3349 : register(t7, space0);
ByteAddressBuffer _4100 : register(t11, space0);
ByteAddressBuffer _4125 : register(t13, space0);
ByteAddressBuffer _4129 : register(t14, space0);
ByteAddressBuffer _4453 : register(t10, space0);
ByteAddressBuffer _4457 : register(t9, space0);
ByteAddressBuffer _6743 : register(t12, space0);
RWByteAddressBuffer _8478 : register(u3, space0);
RWByteAddressBuffer _8489 : register(u1, space0);
RWByteAddressBuffer _8605 : register(u2, space0);
ByteAddressBuffer _8684 : register(t5, space0);
ByteAddressBuffer _8701 : register(t4, space0);
ByteAddressBuffer _8808 : register(t8, space0);
cbuffer UniformParams
{
    Params _3325_g_params : packoffset(c0);
};

Texture2D<float4> g_textures[] : register(t0, space1);
SamplerState _g_textures_sampler[] : register(s0, space1);
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
    float _1064 = atan2(dir.z, dir.x) + y_rotation;
    float phi = _1064;
    if (_1064 < 0.0f)
    {
        phi += 6.283185482025146484375f;
    }
    if (phi > 6.283185482025146484375f)
    {
        phi -= 6.283185482025146484375f;
    }
    uint _1083 = index & 16777215u;
    uint _1089_dummy_parameter;
    float2 _1096 = float2(frac(phi * 0.15915493667125701904296875f), acos(clamp(dir.y, -1.0f, 1.0f)) * 0.3183098733425140380859375f) * float2(int2(spvTextureSize(g_textures[_1083], uint(0), _1089_dummy_parameter)));
    uint _1099 = _1083;
    int2 _1103 = int2(_1096);
    float2 _1138 = frac(_1096);
    float4 param = g_textures[NonUniformResourceIndex(_1099)].Load(int3(_1103, 0), int2(0, 0));
    float4 param_1 = g_textures[NonUniformResourceIndex(_1099)].Load(int3(_1103, 0), int2(1, 0));
    float4 param_2 = g_textures[NonUniformResourceIndex(_1099)].Load(int3(_1103, 0), int2(0, 1));
    float4 param_3 = g_textures[NonUniformResourceIndex(_1099)].Load(int3(_1103, 0), int2(1, 1));
    float _1158 = _1138.x;
    float _1163 = 1.0f - _1158;
    float _1179 = _1138.y;
    return (((rgbe_to_rgb(param_3) * _1158) + (rgbe_to_rgb(param_2) * _1163)) * _1179) + (((rgbe_to_rgb(param_1) * _1158) + (rgbe_to_rgb(param) * _1163)) * (1.0f - _1179));
}

float2 DirToCanonical(float3 d, float y_rotation)
{
    float _720 = (-atan2(d.z, d.x)) + y_rotation;
    float phi = _720;
    if (_720 < 0.0f)
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
    float2 _747 = DirToCanonical(L, -y_rotation);
    float factor = 1.0f;
    while (lod >= 0)
    {
        int2 _767 = clamp(int2(_747 * float(res)), int2(0, 0), (res - 1).xx);
        float4 quad = qtree_tex.Load(int3(_767 / int2(2, 2), lod));
        float _802 = ((quad.x + quad.y) + quad.z) + quad.w;
        if (_802 <= 0.0f)
        {
            break;
        }
        factor *= ((4.0f * quad[(0 | ((_767.x & 1) << 0)) | ((_767.y & 1) << 1)]) / _802);
        lod--;
        res *= 2;
    }
    return factor * 0.079577468335628509521484375f;
}

float power_heuristic(float a, float b)
{
    float _1192 = a * a;
    return _1192 / mad(b, b, _1192);
}

float3 Evaluate_EnvColor(ray_data_t ray)
{
    float3 _4664 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 _4671;
    if ((ray.depth & 16777215) != 0)
    {
        _4671 = _3325_g_params.env_col.xyz;
    }
    else
    {
        _4671 = _3325_g_params.back_col.xyz;
    }
    float3 env_col = _4671;
    uint _4687;
    if ((ray.depth & 16777215) != 0)
    {
        _4687 = asuint(_3325_g_params.env_col.w);
    }
    else
    {
        _4687 = asuint(_3325_g_params.back_col.w);
    }
    float _4703;
    if ((ray.depth & 16777215) != 0)
    {
        _4703 = _3325_g_params.env_rotation;
    }
    else
    {
        _4703 = _3325_g_params.back_rotation;
    }
    if (_4687 != 4294967295u)
    {
        env_col *= SampleLatlong_RGBE(_4687, _4664, _4703);
    }
    if (_3325_g_params.env_qtree_levels > 0)
    {
        float param = ray.pdf;
        float param_1 = Evaluate_EnvQTree(_4703, g_env_qtree, _g_env_qtree_sampler, _3325_g_params.env_qtree_levels, _4664);
        env_col *= power_heuristic(param, param_1);
    }
    else
    {
        if (_3325_g_params.env_mult_importance != 0)
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
    float3 _4782 = float3(ray.d[0], ray.d[1], ray.d[2]);
    light_t _4796;
    _4796.type_and_param0 = _3345.Load4(((-1) - inter.obj_index) * 64 + 0);
    _4796.param1 = asfloat(_3345.Load4(((-1) - inter.obj_index) * 64 + 16));
    _4796.param2 = asfloat(_3345.Load4(((-1) - inter.obj_index) * 64 + 32));
    _4796.param3 = asfloat(_3345.Load4(((-1) - inter.obj_index) * 64 + 48));
    float3 lcol = asfloat(_4796.type_and_param0.yzw);
    [branch]
    if ((_4796.type_and_param0.x & 128u) != 0u)
    {
        float3 env_col = _3325_g_params.env_col.xyz;
        uint _4823 = asuint(_3325_g_params.env_col.w);
        if (_4823 != 4294967295u)
        {
            env_col *= SampleLatlong_RGBE(_4823, _4782, _3325_g_params.env_rotation);
        }
        lcol *= env_col;
    }
    uint _4841 = _4796.type_and_param0.x & 31u;
    if (_4841 == 0u)
    {
        float param = ray.pdf;
        float param_1 = (inter.t * inter.t) / ((0.5f * _4796.param1.w) * dot(_4782, normalize(_4796.param1.xyz - (float3(ray.o[0], ray.o[1], ray.o[2]) + (_4782 * inter.t)))));
        lcol *= power_heuristic(param, param_1);
        bool _4908 = _4796.param3.x > 0.0f;
        bool _4914;
        if (_4908)
        {
            _4914 = _4796.param3.y > 0.0f;
        }
        else
        {
            _4914 = _4908;
        }
        [branch]
        if (_4914)
        {
            [flatten]
            if (_4796.param3.y > 0.0f)
            {
                lcol *= clamp((_4796.param3.x - acos(clamp(-dot(_4782, _4796.param2.xyz), 0.0f, 1.0f))) / _4796.param3.y, 0.0f, 1.0f);
            }
        }
    }
    else
    {
        if (_4841 == 4u)
        {
            float param_2 = ray.pdf;
            float param_3 = (inter.t * inter.t) / (_4796.param1.w * dot(_4782, normalize(cross(_4796.param2.xyz, _4796.param3.xyz))));
            lcol *= power_heuristic(param_2, param_3);
        }
        else
        {
            if (_4841 == 5u)
            {
                float param_4 = ray.pdf;
                float param_5 = (inter.t * inter.t) / (_4796.param1.w * dot(_4782, normalize(cross(_4796.param2.xyz, _4796.param3.xyz))));
                lcol *= power_heuristic(param_4, param_5);
            }
            else
            {
                if (_4841 == 3u)
                {
                    float param_6 = ray.pdf;
                    float param_7 = (inter.t * inter.t) / (_4796.param1.w * (1.0f - abs(dot(_4782, _4796.param3.xyz))));
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
    uint _497 = uint(x);
    uint _504 = ((_497 >> uint(16)) ^ _497) * 73244475u;
    uint _509 = ((_504 >> uint(16)) ^ _504) * 73244475u;
    return int((_509 >> uint(16)) ^ _509);
}

float construct_float(inout uint m)
{
    m &= 8388607u;
    m |= 1065353216u;
    return asfloat(m) - 1.0f;
}

bool exchange(inout bool old_value, bool new_value)
{
    bool _2100 = old_value;
    old_value = new_value;
    return _2100;
}

float peek_ior_stack(float stack[4], inout bool skip_first, float default_value)
{
    float _8820;
    do
    {
        bool _2184 = stack[3] > 0.0f;
        bool _2193;
        if (_2184)
        {
            bool param = skip_first;
            bool param_1 = false;
            bool _2190 = exchange(param, param_1);
            skip_first = param;
            _2193 = !_2190;
        }
        else
        {
            _2193 = _2184;
        }
        if (_2193)
        {
            _8820 = stack[3];
            break;
        }
        bool _2201 = stack[2] > 0.0f;
        bool _2210;
        if (_2201)
        {
            bool param_2 = skip_first;
            bool param_3 = false;
            bool _2207 = exchange(param_2, param_3);
            skip_first = param_2;
            _2210 = !_2207;
        }
        else
        {
            _2210 = _2201;
        }
        if (_2210)
        {
            _8820 = stack[2];
            break;
        }
        bool _2218 = stack[1] > 0.0f;
        bool _2227;
        if (_2218)
        {
            bool param_4 = skip_first;
            bool param_5 = false;
            bool _2224 = exchange(param_4, param_5);
            skip_first = param_4;
            _2227 = !_2224;
        }
        else
        {
            _2227 = _2218;
        }
        if (_2227)
        {
            _8820 = stack[1];
            break;
        }
        bool _2235 = stack[0] > 0.0f;
        bool _2244;
        if (_2235)
        {
            bool param_6 = skip_first;
            bool param_7 = false;
            bool _2241 = exchange(param_6, param_7);
            skip_first = param_6;
            _2244 = !_2241;
        }
        else
        {
            _2244 = _2235;
        }
        if (_2244)
        {
            _8820 = stack[0];
            break;
        }
        _8820 = default_value;
        break;
    } while(false);
    return _8820;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _596 = mad(col.z, 31.875f, 1.0f);
    float _606 = (col.x - 0.501960813999176025390625f) / _596;
    float _612 = (col.y - 0.501960813999176025390625f) / _596;
    return float3((col.w + _606) - _612, col.w + _612, (col.w - _606) - _612);
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
    uint _997 = index & 16777215u;
    float4 res = g_textures[NonUniformResourceIndex(_997)].SampleLevel(_g_textures_sampler[NonUniformResourceIndex(_997)], uvs, float(lod));
    bool _1007;
    if (maybe_YCoCg)
    {
        _1007 = (index & 67108864u) != 0u;
    }
    else
    {
        _1007 = maybe_YCoCg;
    }
    if (_1007)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _1025;
    if (maybe_SRGB)
    {
        _1025 = (index & 16777216u) != 0u;
    }
    else
    {
        _1025 = maybe_SRGB;
    }
    if (_1025)
    {
        float3 param_1 = res.xyz;
        float3 _1031 = srgb_to_rgb(param_1);
        float4 _9875 = res;
        _9875.x = _1031.x;
        float4 _9877 = _9875;
        _9877.y = _1031.y;
        float4 _9879 = _9877;
        _9879.z = _1031.z;
        res = _9879;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod)
{
    return SampleBilinear(index, uvs, lod, false, false);
}

float fresnel_dielectric_cos(float cosi, float eta)
{
    float _1224 = abs(cosi);
    float _1233 = mad(_1224, _1224, mad(eta, eta, -1.0f));
    float g = _1233;
    float result;
    if (_1233 > 0.0f)
    {
        float _1238 = g;
        float _1239 = sqrt(_1238);
        g = _1239;
        float _1243 = _1239 - _1224;
        float _1246 = _1239 + _1224;
        float _1247 = _1243 / _1246;
        float _1261 = mad(_1224, _1246, -1.0f) / mad(_1224, _1243, 1.0f);
        result = ((0.5f * _1247) * _1247) * mad(_1261, _1261, 1.0f);
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
    float3 _8825;
    do
    {
        float _1297 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1297)
        {
            _8825 = N;
            break;
        }
        float3 _1317 = normalize(N - (Ng * dot(N, Ng)));
        float _1321 = dot(I, _1317);
        float _1325 = dot(I, Ng);
        float _1337 = mad(_1321, _1321, _1325 * _1325);
        float param = (_1321 * _1321) * mad(-_1297, _1297, _1337);
        float _1347 = safe_sqrtf(param);
        float _1353 = mad(_1325, _1297, _1337);
        float _1356 = 0.5f / _1337;
        float _1361 = _1347 + _1353;
        float _1362 = _1356 * _1361;
        float _1368 = (-_1347) + _1353;
        float _1369 = _1356 * _1368;
        bool _1377 = (_1362 > 9.9999997473787516355514526367188e-06f) && (_1362 <= 1.000010013580322265625f);
        bool valid1 = _1377;
        bool _1383 = (_1369 > 9.9999997473787516355514526367188e-06f) && (_1369 <= 1.000010013580322265625f);
        bool valid2 = _1383;
        float2 N_new;
        if (_1377 && _1383)
        {
            float _10176 = (-0.5f) / _1337;
            float param_1 = mad(_10176, _1361, 1.0f);
            float _1393 = safe_sqrtf(param_1);
            float param_2 = _1362;
            float _1396 = safe_sqrtf(param_2);
            float2 _1397 = float2(_1393, _1396);
            float param_3 = mad(_10176, _1368, 1.0f);
            float _1402 = safe_sqrtf(param_3);
            float param_4 = _1369;
            float _1405 = safe_sqrtf(param_4);
            float2 _1406 = float2(_1402, _1405);
            float _10178 = -_1325;
            float _1422 = mad(2.0f * mad(_1393, _1321, _1396 * _1325), _1396, _10178);
            float _1438 = mad(2.0f * mad(_1402, _1321, _1405 * _1325), _1405, _10178);
            bool _1440 = _1422 >= 9.9999997473787516355514526367188e-06f;
            valid1 = _1440;
            bool _1442 = _1438 >= 9.9999997473787516355514526367188e-06f;
            valid2 = _1442;
            if (_1440 && _1442)
            {
                bool2 _1455 = (_1422 < _1438).xx;
                N_new = float2(_1455.x ? _1397.x : _1406.x, _1455.y ? _1397.y : _1406.y);
            }
            else
            {
                bool2 _1463 = (_1422 > _1438).xx;
                N_new = float2(_1463.x ? _1397.x : _1406.x, _1463.y ? _1397.y : _1406.y);
            }
        }
        else
        {
            if (!(valid1 || valid2))
            {
                _8825 = Ng;
                break;
            }
            float _1475 = valid1 ? _1362 : _1369;
            float param_5 = 1.0f - _1475;
            float param_6 = _1475;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _8825 = (_1317 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _8825;
}

float3 rotate_around_axis(float3 p, float3 axis, float angle)
{
    float _1569 = cos(angle);
    float _1572 = sin(angle);
    float _1576 = 1.0f - _1569;
    return float3(mad(mad(_1576 * axis.x, axis.z, axis.y * _1572), p.z, mad(mad(_1576 * axis.x, axis.x, _1569), p.x, mad(_1576 * axis.x, axis.y, -(axis.z * _1572)) * p.y)), mad(mad(_1576 * axis.y, axis.z, -(axis.x * _1572)), p.z, mad(mad(_1576 * axis.x, axis.y, axis.z * _1572), p.x, mad(_1576 * axis.y, axis.y, _1569) * p.y)), mad(mad(_1576 * axis.z, axis.z, _1569), p.z, mad(mad(_1576 * axis.x, axis.z, -(axis.y * _1572)), p.x, mad(_1576 * axis.y, axis.z, axis.x * _1572) * p.y)));
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
    int3 _1726 = int3(n * 128.0f);
    int _1734;
    if (p.x < 0.0f)
    {
        _1734 = -_1726.x;
    }
    else
    {
        _1734 = _1726.x;
    }
    int _1752;
    if (p.y < 0.0f)
    {
        _1752 = -_1726.y;
    }
    else
    {
        _1752 = _1726.y;
    }
    int _1770;
    if (p.z < 0.0f)
    {
        _1770 = -_1726.z;
    }
    else
    {
        _1770 = _1726.z;
    }
    float _1788;
    if (abs(p.x) < 0.03125f)
    {
        _1788 = mad(1.52587890625e-05f, n.x, p.x);
    }
    else
    {
        _1788 = asfloat(asint(p.x) + _1734);
    }
    float _1806;
    if (abs(p.y) < 0.03125f)
    {
        _1806 = mad(1.52587890625e-05f, n.y, p.y);
    }
    else
    {
        _1806 = asfloat(asint(p.y) + _1752);
    }
    float _1823;
    if (abs(p.z) < 0.03125f)
    {
        _1823 = mad(1.52587890625e-05f, n.z, p.z);
    }
    else
    {
        _1823 = asfloat(asint(p.z) + _1770);
    }
    return float3(_1788, _1806, _1823);
}

float3 MapToCone(float r1, float r2, float3 N, float radius)
{
    float3 _8850;
    do
    {
        float2 _3224 = (float2(r1, r2) * 2.0f) - 1.0f.xx;
        float _3226 = _3224.x;
        bool _3227 = _3226 == 0.0f;
        bool _3233;
        if (_3227)
        {
            _3233 = _3224.y == 0.0f;
        }
        else
        {
            _3233 = _3227;
        }
        if (_3233)
        {
            _8850 = N;
            break;
        }
        float _3242 = _3224.y;
        float r;
        float theta;
        if (abs(_3226) > abs(_3242))
        {
            r = _3226;
            theta = 0.785398185253143310546875f * (_3242 / _3226);
        }
        else
        {
            r = _3242;
            theta = 1.57079637050628662109375f * mad(-0.5f, _3226 / _3242, 1.0f);
        }
        float3 param;
        float3 param_1;
        create_tbn(N, param, param_1);
        _8850 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _8850;
}

float3 CanonicalToDir(float2 p, float y_rotation)
{
    float _670 = mad(2.0f, p.x, -1.0f);
    float _675 = mad(6.283185482025146484375f, p.y, y_rotation);
    float phi = _675;
    if (_675 < 0.0f)
    {
        phi += 6.283185482025146484375f;
    }
    if (phi > 6.283185482025146484375f)
    {
        phi -= 6.283185482025146484375f;
    }
    float _693 = sqrt(mad(-_670, _670, 1.0f));
    return float3(_693 * cos(phi), _670, (-_693) * sin(phi));
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
        float _868 = quad.x + quad.z;
        float partial = _868;
        float _875 = (_868 + quad.y) + quad.w;
        if (_875 <= 0.0f)
        {
            break;
        }
        float _884 = partial / _875;
        float boundary = _884;
        int index = 0;
        if (_sample < _884)
        {
            _sample /= boundary;
            boundary = quad.x / partial;
        }
        else
        {
            float _899 = partial;
            float _900 = _875 - _899;
            partial = _900;
            float2 _9862 = origin;
            _9862.x = origin.x + _step;
            origin = _9862;
            _sample = (_sample - boundary) / (1.0f - boundary);
            boundary = quad.y / _900;
            index |= 1;
        }
        if (_sample < boundary)
        {
            _sample /= boundary;
        }
        else
        {
            float2 _9865 = origin;
            _9865.y = origin.y + _step;
            origin = _9865;
            _sample = (_sample - boundary) / (1.0f - boundary);
            index |= 2;
        }
        factor *= ((4.0f * quad[index]) / _875);
        lod--;
        res *= 2;
        _step *= 0.5f;
    }
    float2 _957 = origin;
    float2 _958 = _957 + (float2(rx, ry) * (2.0f * _step));
    origin = _958;
    return float4(CanonicalToDir(_958, y_rotation), factor * 0.079577468335628509521484375f);
}

float3 world_from_tangent(float3 T, float3 B, float3 N, float3 V)
{
    return ((T * V.x) + (B * V.y)) + (N * V.z);
}

void SampleLightSource(float3 P, float3 T, float3 B, float3 N, int hi, float2 sample_off, inout light_sample_t ls)
{
    float _3318 = frac(asfloat(_3309.Load((hi + 3) * 4 + 0)) + sample_off.x);
    float _3329 = float(_3325_g_params.li_count);
    uint _3336 = min(uint(_3318 * _3329), uint(_3325_g_params.li_count - 1));
    light_t _3356;
    _3356.type_and_param0 = _3345.Load4(_3349.Load(_3336 * 4 + 0) * 64 + 0);
    _3356.param1 = asfloat(_3345.Load4(_3349.Load(_3336 * 4 + 0) * 64 + 16));
    _3356.param2 = asfloat(_3345.Load4(_3349.Load(_3336 * 4 + 0) * 64 + 32));
    _3356.param3 = asfloat(_3345.Load4(_3349.Load(_3336 * 4 + 0) * 64 + 48));
    ls.col = asfloat(_3356.type_and_param0.yzw);
    ls.col *= _3329;
    ls.cast_shadow = (_3356.type_and_param0.x & 32u) != 0u;
    ls.from_env = false;
    uint _3392 = _3356.type_and_param0.x & 31u;
    [branch]
    if (_3392 == 0u)
    {
        float _3406 = frac(asfloat(_3309.Load((hi + 4) * 4 + 0)) + sample_off.x);
        float3 _3422 = P - _3356.param1.xyz;
        float3 _3429 = _3422 / length(_3422).xxx;
        float _3436 = sqrt(clamp(mad(-_3406, _3406, 1.0f), 0.0f, 1.0f));
        float _3439 = 6.283185482025146484375f * frac(asfloat(_3309.Load((hi + 5) * 4 + 0)) + sample_off.y);
        float3 sampled_dir = float3(_3436 * cos(_3439), _3436 * sin(_3439), _3406);
        float3 param;
        float3 param_1;
        create_tbn(_3429, param, param_1);
        float3 _9942 = sampled_dir;
        float3 _3472 = ((param * _9942.x) + (param_1 * _9942.y)) + (_3429 * _9942.z);
        sampled_dir = _3472;
        float3 _3481 = _3356.param1.xyz + (_3472 * _3356.param2.w);
        float3 _3488 = normalize(_3481 - _3356.param1.xyz);
        float3 param_2 = _3481;
        float3 param_3 = _3488;
        ls.lp = offset_ray(param_2, param_3);
        ls.L = _3481 - P;
        float3 _3501 = ls.L;
        float _3502 = length(_3501);
        ls.L /= _3502.xxx;
        ls.area = _3356.param1.w;
        float _3517 = abs(dot(ls.L, _3488));
        [flatten]
        if (_3517 > 0.0f)
        {
            ls.pdf = (_3502 * _3502) / ((0.5f * ls.area) * _3517);
        }
        [branch]
        if (_3356.param3.x > 0.0f)
        {
            float _3544 = -dot(ls.L, _3356.param2.xyz);
            if (_3544 > 0.0f)
            {
                ls.col *= clamp((_3356.param3.x - acos(clamp(_3544, 0.0f, 1.0f))) / _3356.param3.y, 0.0f, 1.0f);
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
        if (_3392 == 2u)
        {
            ls.L = _3356.param1.xyz;
            if (_3356.param1.w != 0.0f)
            {
                float param_4 = frac(asfloat(_3309.Load((hi + 4) * 4 + 0)) + sample_off.x);
                float param_5 = frac(asfloat(_3309.Load((hi + 5) * 4 + 0)) + sample_off.y);
                float3 param_6 = ls.L;
                float param_7 = tan(_3356.param1.w);
                ls.L = normalize(MapToCone(param_4, param_5, param_6, param_7));
            }
            ls.area = 0.0f;
            ls.lp = P + ls.L;
            ls.dist_mul = 3402823346297367662189621542912.0f;
            ls.pdf = 1.0f;
            if ((_3356.type_and_param0.x & 64u) == 0u)
            {
                ls.area = 0.0f;
            }
        }
        else
        {
            [branch]
            if (_3392 == 4u)
            {
                float3 _3681 = (_3356.param1.xyz + (_3356.param2.xyz * (frac(asfloat(_3309.Load((hi + 4) * 4 + 0)) + sample_off.x) - 0.5f))) + (_3356.param3.xyz * (frac(asfloat(_3309.Load((hi + 5) * 4 + 0)) + sample_off.y) - 0.5f));
                float3 _3686 = normalize(cross(_3356.param2.xyz, _3356.param3.xyz));
                float3 param_8 = _3681;
                float3 param_9 = _3686;
                ls.lp = offset_ray(param_8, param_9);
                ls.L = _3681 - P;
                float3 _3699 = ls.L;
                float _3700 = length(_3699);
                ls.L /= _3700.xxx;
                ls.area = _3356.param1.w;
                float _3715 = dot(-ls.L, _3686);
                if (_3715 > 0.0f)
                {
                    ls.pdf = (_3700 * _3700) / (ls.area * _3715);
                }
                if ((_3356.type_and_param0.x & 64u) == 0u)
                {
                    ls.area = 0.0f;
                }
                [branch]
                if ((_3356.type_and_param0.x & 128u) != 0u)
                {
                    float3 env_col = _3325_g_params.env_col.xyz;
                    uint _3753 = asuint(_3325_g_params.env_col.w);
                    if (_3753 != 4294967295u)
                    {
                        env_col *= SampleLatlong_RGBE(_3753, ls.L, _3325_g_params.env_rotation);
                    }
                    ls.col *= env_col;
                    ls.from_env = true;
                }
            }
            else
            {
                [branch]
                if (_3392 == 5u)
                {
                    float2 _3816 = (float2(frac(asfloat(_3309.Load((hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_3309.Load((hi + 5) * 4 + 0)) + sample_off.y)) * 2.0f) - 1.0f.xx;
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
                    float3 _3880 = (_3356.param1.xyz + (_3356.param2.xyz * offset.x)) + (_3356.param3.xyz * offset.y);
                    float3 _3885 = normalize(cross(_3356.param2.xyz, _3356.param3.xyz));
                    float3 param_10 = _3880;
                    float3 param_11 = _3885;
                    ls.lp = offset_ray(param_10, param_11);
                    ls.L = _3880 - P;
                    float3 _3898 = ls.L;
                    float _3899 = length(_3898);
                    ls.L /= _3899.xxx;
                    ls.area = _3356.param1.w;
                    float _3914 = dot(-ls.L, _3885);
                    [flatten]
                    if (_3914 > 0.0f)
                    {
                        ls.pdf = (_3899 * _3899) / (ls.area * _3914);
                    }
                    if ((_3356.type_and_param0.x & 64u) == 0u)
                    {
                        ls.area = 0.0f;
                    }
                    [branch]
                    if ((_3356.type_and_param0.x & 128u) != 0u)
                    {
                        float3 env_col_1 = _3325_g_params.env_col.xyz;
                        uint _3948 = asuint(_3325_g_params.env_col.w);
                        if (_3948 != 4294967295u)
                        {
                            env_col_1 *= SampleLatlong_RGBE(_3948, ls.L, _3325_g_params.env_rotation);
                        }
                        ls.col *= env_col_1;
                        ls.from_env = true;
                    }
                }
                else
                {
                    [branch]
                    if (_3392 == 3u)
                    {
                        float3 _4006 = normalize(cross(P - _3356.param1.xyz, _3356.param3.xyz));
                        float _4013 = 3.1415927410125732421875f * frac(asfloat(_3309.Load((hi + 4) * 4 + 0)) + sample_off.x);
                        float3 _4038 = (_3356.param1.xyz + (((_4006 * cos(_4013)) + (cross(_4006, _3356.param3.xyz) * sin(_4013))) * _3356.param2.w)) + ((_3356.param3.xyz * (frac(asfloat(_3309.Load((hi + 5) * 4 + 0)) + sample_off.y) - 0.5f)) * _3356.param3.w);
                        ls.lp = _4038;
                        float3 _4044 = _4038 - P;
                        float _4047 = length(_4044);
                        ls.L = _4044 / _4047.xxx;
                        ls.area = _3356.param1.w;
                        float _4062 = 1.0f - abs(dot(ls.L, _3356.param3.xyz));
                        [flatten]
                        if (_4062 != 0.0f)
                        {
                            ls.pdf = (_4047 * _4047) / (ls.area * _4062);
                        }
                        if ((_3356.type_and_param0.x & 64u) == 0u)
                        {
                            ls.area = 0.0f;
                        }
                    }
                    else
                    {
                        [branch]
                        if (_3392 == 6u)
                        {
                            uint _4092 = asuint(_3356.param1.x);
                            transform_t _4106;
                            _4106.xform = asfloat(uint4x4(_4100.Load4(asuint(_3356.param1.y) * 128 + 0), _4100.Load4(asuint(_3356.param1.y) * 128 + 16), _4100.Load4(asuint(_3356.param1.y) * 128 + 32), _4100.Load4(asuint(_3356.param1.y) * 128 + 48)));
                            _4106.inv_xform = asfloat(uint4x4(_4100.Load4(asuint(_3356.param1.y) * 128 + 64), _4100.Load4(asuint(_3356.param1.y) * 128 + 80), _4100.Load4(asuint(_3356.param1.y) * 128 + 96), _4100.Load4(asuint(_3356.param1.y) * 128 + 112)));
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
                            float _4322 = sqrt(frac(asfloat(_3309.Load((hi + 4) * 4 + 0)) + sample_off.x));
                            float _4331 = frac(asfloat(_3309.Load((hi + 5) * 4 + 0)) + sample_off.y);
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
                            if (_3392 == 7u)
                            {
                                float _4552 = frac(asfloat(_3309.Load((hi + 4) * 4 + 0)) + sample_off.x);
                                float _4561 = frac(asfloat(_3309.Load((hi + 5) * 4 + 0)) + sample_off.y);
                                float4 dir_and_pdf;
                                if (_3325_g_params.env_qtree_levels > 0)
                                {
                                    dir_and_pdf = Sample_EnvQTree(_3325_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3325_g_params.env_qtree_levels, mad(_3318, _3329, -float(_3336)), _4552, _4561);
                                }
                                else
                                {
                                    float _4580 = 6.283185482025146484375f * _4561;
                                    float _4592 = sqrt(mad(-_4552, _4552, 1.0f));
                                    float3 param_14 = T;
                                    float3 param_15 = B;
                                    float3 param_16 = N;
                                    float3 param_17 = float3(_4592 * cos(_4580), _4592 * sin(_4580), _4552);
                                    dir_and_pdf = float4(world_from_tangent(param_14, param_15, param_16, param_17), 0.15915493667125701904296875f);
                                }
                                ls.L = dir_and_pdf.xyz;
                                ls.col *= _3325_g_params.env_col.xyz;
                                uint _4631 = asuint(_3325_g_params.env_col.w);
                                if (_4631 != 4294967295u)
                                {
                                    ls.col *= SampleLatlong_RGBE(_4631, ls.L, _3325_g_params.env_rotation);
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
    uint _988 = index & 16777215u;
    uint _992_dummy_parameter;
    return int2(spvTextureSize(g_textures[NonUniformResourceIndex(_988)], uint(0), _992_dummy_parameter));
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
    float _2304 = 1.0f / mad(0.904129683971405029296875f, roughness, 3.1415927410125732421875f);
    float _2316 = max(dot(N, L), 0.0f);
    float _2321 = max(dot(N, V), 0.0f);
    float _2329 = mad(-_2316, _2321, dot(L, V));
    float t = _2329;
    if (_2329 > 0.0f)
    {
        t /= (max(_2316, _2321) + 1.1754943508222875079687365372222e-38f);
    }
    return float4(base_color * (_2316 * mad(roughness * _2304, t, _2304)), 0.15915493667125701904296875f);
}

float3 Evaluate_DiffuseNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _8830;
    do
    {
        float3 param = -float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param_1 = surf.N;
        float3 param_2 = ls.L;
        float param_3 = roughness;
        float3 param_4 = base_color;
        float4 _5111 = Evaluate_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_5 = ls.pdf;
            float param_6 = _5111.w;
            mis_weight = power_heuristic(param_5, param_6);
        }
        float3 _5134 = (ls.col * _5111.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8830 = _5134;
            break;
        }
        float3 param_7 = surf.P;
        float3 param_8 = surf.plane_N;
        float3 _5146 = offset_ray(param_7, param_8);
        sh_r.o[0] = _5146.x;
        sh_r.o[1] = _5146.y;
        sh_r.o[2] = _5146.z;
        sh_r.c[0] = ray.c[0] * _5134.x;
        sh_r.c[1] = ray.c[1] * _5134.y;
        sh_r.c[2] = ray.c[2] * _5134.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8830 = 0.0f.xxx;
        break;
    } while(false);
    return _8830;
}

float4 Sample_OrenDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float rand_u, float rand_v, inout float3 out_V)
{
    float _2363 = 6.283185482025146484375f * rand_v;
    float _2375 = sqrt(mad(-rand_u, rand_u, 1.0f));
    float3 param = T;
    float3 param_1 = B;
    float3 param_2 = N;
    float3 param_3 = float3(_2375 * cos(_2363), _2375 * sin(_2363), rand_u);
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
    float4 _5397 = Sample_OrenDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8);
    new_ray.depth = ray.depth + 1;
    float3 param_9 = surf.P;
    float3 param_10 = surf.plane_N;
    float3 _5407 = offset_ray(param_9, param_10);
    new_ray.o[0] = _5407.x;
    new_ray.o[1] = _5407.y;
    new_ray.o[2] = _5407.z;
    new_ray.d[0] = param_8.x;
    new_ray.d[1] = param_8.y;
    new_ray.d[2] = param_8.z;
    new_ray.c[0] = ((ray.c[0] * _5397.x) * mix_weight) / _5397.w;
    new_ray.c[1] = ((ray.c[1] * _5397.y) * mix_weight) / _5397.w;
    new_ray.c[2] = ((ray.c[2] * _5397.z) * mix_weight) / _5397.w;
    new_ray.pdf = _5397.w;
}

float3 tangent_from_world(float3 T, float3 B, float3 N, float3 V)
{
    return float3(dot(V, T), dot(V, B), dot(V, N));
}

float D_GGX(float3 H, float alpha_x, float alpha_y)
{
    float _8883;
    do
    {
        if (H.z == 0.0f)
        {
            _8883 = 0.0f;
            break;
        }
        float _2030 = (-H.x) / (H.z * alpha_x);
        float _2036 = (-H.y) / (H.z * alpha_y);
        float _2045 = mad(_2036, _2036, mad(_2030, _2030, 1.0f));
        _8883 = 1.0f / (((((_2045 * _2045) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _8883;
}

float G1(float3 Ve, inout float alpha_x, inout float alpha_y)
{
    alpha_x *= alpha_x;
    alpha_y *= alpha_y;
    return 1.0f / mad((-1.0f) + sqrt(1.0f + (mad(alpha_x * Ve.x, Ve.x, (alpha_y * Ve.y) * Ve.y) / (Ve.z * Ve.z))), 0.5f, 1.0f);
}

float4 Evaluate_GGXSpecular_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float alpha_x, float alpha_y, float spec_ior, float spec_F0, float3 spec_col)
{
    float _2545 = D_GGX(sampled_normal_ts, alpha_x, alpha_y);
    float3 param = view_dir_ts;
    float param_1 = alpha_x;
    float param_2 = alpha_y;
    float _2553 = G1(param, param_1, param_2);
    float3 param_3 = reflected_dir_ts;
    float param_4 = alpha_x;
    float param_5 = alpha_y;
    float _2560 = G1(param_3, param_4, param_5);
    float param_6 = dot(view_dir_ts, sampled_normal_ts);
    float param_7 = spec_ior;
    float3 F = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param_6, param_7) - spec_F0) / (1.0f - spec_F0)).xxx);
    float _2588 = 4.0f * abs(view_dir_ts.z * reflected_dir_ts.z);
    float _2591;
    if (_2588 != 0.0f)
    {
        _2591 = (_2545 * (_2553 * _2560)) / _2588;
    }
    else
    {
        _2591 = 0.0f;
    }
    F *= _2591;
    float3 param_8 = view_dir_ts;
    float param_9 = alpha_x;
    float param_10 = alpha_y;
    float _2611 = G1(param_8, param_9, param_10);
    float pdf = ((_2545 * _2611) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2626 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2626 != 0.0f)
    {
        pdf /= _2626;
    }
    float3 _2637 = F;
    float3 _2638 = _2637 * max(reflected_dir_ts.z, 0.0f);
    F = _2638;
    return float4(_2638, pdf);
}

float3 Evaluate_GlossyNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _8835;
    do
    {
        float3 _5182 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5182;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - _5182);
        float _5220 = roughness * roughness;
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = _5220;
        float param_16 = _5220;
        float param_17 = spec_ior;
        float param_18 = spec_F0;
        float3 param_19 = base_color;
        float4 _5235 = Evaluate_GGXSpecular_BSDF(param_12, param_13, param_14, param_15, param_16, param_17, param_18, param_19);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_20 = ls.pdf;
            float param_21 = _5235.w;
            mis_weight = power_heuristic(param_20, param_21);
        }
        float3 _5258 = (ls.col * _5235.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8835 = _5258;
            break;
        }
        float3 param_22 = surf.P;
        float3 param_23 = surf.plane_N;
        float3 _5270 = offset_ray(param_22, param_23);
        sh_r.o[0] = _5270.x;
        sh_r.o[1] = _5270.y;
        sh_r.o[2] = _5270.z;
        sh_r.c[0] = ray.c[0] * _5258.x;
        sh_r.c[1] = ray.c[1] * _5258.y;
        sh_r.c[2] = ray.c[2] * _5258.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8835 = 0.0f.xxx;
        break;
    } while(false);
    return _8835;
}

float3 SampleGGX_VNDF(float3 Ve, float alpha_x, float alpha_y, float U1, float U2)
{
    float3 _1848 = normalize(float3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
    float _1851 = _1848.x;
    float _1856 = _1848.y;
    float _1860 = mad(_1851, _1851, _1856 * _1856);
    float3 _1864;
    if (_1860 > 0.0f)
    {
        _1864 = float3(-_1856, _1851, 0.0f) / sqrt(_1860).xxx;
    }
    else
    {
        _1864 = float3(1.0f, 0.0f, 0.0f);
    }
    float _1886 = sqrt(U1);
    float _1889 = 6.283185482025146484375f * U2;
    float _1894 = _1886 * cos(_1889);
    float _1903 = 1.0f + _1848.z;
    float _1910 = mad(-_1894, _1894, 1.0f);
    float _1916 = mad(mad(-0.5f, _1903, 1.0f), sqrt(_1910), (0.5f * _1903) * (_1886 * sin(_1889)));
    float3 _1937 = ((_1864 * _1894) + (cross(_1848, _1864) * _1916)) + (_1848 * sqrt(max(0.0f, mad(-_1916, _1916, _1910))));
    return normalize(float3(alpha_x * _1937.x, alpha_y * _1937.y, max(0.0f, _1937.z)));
}

float4 Sample_GGXSpecular_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float anisotropic, float spec_ior, float spec_F0, float3 spec_col, float rand_u, float rand_v, inout float3 out_V)
{
    float4 _8855;
    do
    {
        float _2648 = roughness * roughness;
        float _2652 = sqrt(mad(-0.89999997615814208984375f, anisotropic, 1.0f));
        float _2656 = _2648 / _2652;
        float _2660 = _2648 * _2652;
        [branch]
        if ((_2656 * _2660) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2671 = reflect(I, N);
            float param = dot(_2671, N);
            float param_1 = spec_ior;
            float3 _2685 = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param, param_1) - spec_F0) / (1.0f - spec_F0)).xxx);
            out_V = _2671;
            _8855 = float4(_2685.x * 1000000.0f, _2685.y * 1000000.0f, _2685.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2710 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = _2656;
        float param_7 = _2660;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2719 = SampleGGX_VNDF(_2710, param_6, param_7, param_8, param_9);
        float3 _2730 = normalize(reflect(-_2710, _2719));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2730;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2710;
        float3 param_15 = _2719;
        float3 param_16 = _2730;
        float param_17 = _2656;
        float param_18 = _2660;
        float param_19 = spec_ior;
        float param_20 = spec_F0;
        float3 param_21 = spec_col;
        _8855 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _8855;
}

void Sample_GlossyNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, float spec_ior, float spec_F0, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float3 param_4;
    float4 _5317 = Sample_GGXSpecular_BSDF(param, param_1, param_2, param_3, roughness, 0.0f, spec_ior, spec_F0, base_color, rand_u, rand_v, param_4);
    new_ray.depth = ray.depth + 256;
    float3 param_5 = surf.P;
    float3 param_6 = surf.plane_N;
    float3 _5328 = offset_ray(param_5, param_6);
    new_ray.o[0] = _5328.x;
    new_ray.o[1] = _5328.y;
    new_ray.o[2] = _5328.z;
    new_ray.d[0] = param_4.x;
    new_ray.d[1] = param_4.y;
    new_ray.d[2] = param_4.z;
    new_ray.c[0] = ((ray.c[0] * _5317.x) * mix_weight) / _5317.w;
    new_ray.c[1] = ((ray.c[1] * _5317.y) * mix_weight) / _5317.w;
    new_ray.c[2] = ((ray.c[2] * _5317.z) * mix_weight) / _5317.w;
    new_ray.pdf = _5317.w;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _8860;
    do
    {
        bool _2952 = refr_dir_ts.z >= 0.0f;
        bool _2959;
        if (!_2952)
        {
            _2959 = view_dir_ts.z <= 0.0f;
        }
        else
        {
            _2959 = _2952;
        }
        if (_2959)
        {
            _8860 = 0.0f.xxxx;
            break;
        }
        float _2968 = D_GGX(sampled_normal_ts, roughness2, roughness2);
        float3 param = refr_dir_ts;
        float param_1 = roughness2;
        float param_2 = roughness2;
        float _2976 = G1(param, param_1, param_2);
        float3 param_3 = view_dir_ts;
        float param_4 = roughness2;
        float param_5 = roughness2;
        float _2984 = G1(param_3, param_4, param_5);
        float _2994 = mad(dot(view_dir_ts, sampled_normal_ts), eta, dot(refr_dir_ts, sampled_normal_ts));
        float _3004 = clamp(-dot(refr_dir_ts, sampled_normal_ts), 0.0f, 1.0f) / (_2994 * _2994);
        _8860 = float4(refr_col * (((((_2968 * _2984) * _2976) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3004) / view_dir_ts.z), (((_2968 * _2976) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _3004) / view_dir_ts.z);
        break;
    } while(false);
    return _8860;
}

float3 Evaluate_RefractiveNode(light_sample_t ls, ray_data_t ray, surface_t surf, float3 base_color, float roughness2, float eta, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _8840;
    do
    {
        float3 _5460 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 param = surf.T;
        float3 param_1 = surf.B;
        float3 param_2 = surf.N;
        float3 param_3 = -_5460;
        float3 param_4 = surf.T;
        float3 param_5 = surf.B;
        float3 param_6 = surf.N;
        float3 param_7 = ls.L;
        float3 param_8 = surf.T;
        float3 param_9 = surf.B;
        float3 param_10 = surf.N;
        float3 param_11 = normalize(ls.L - (_5460 * eta));
        float3 param_12 = tangent_from_world(param, param_1, param_2, param_3);
        float3 param_13 = tangent_from_world(param_8, param_9, param_10, param_11);
        float3 param_14 = tangent_from_world(param_4, param_5, param_6, param_7);
        float param_15 = roughness2;
        float param_16 = eta;
        float3 param_17 = base_color;
        float4 _5508 = Evaluate_GGXRefraction_BSDF(param_12, param_13, param_14, param_15, param_16, param_17);
        float mis_weight = 1.0f;
        if (ls.area > 0.0f)
        {
            float param_18 = ls.pdf;
            float param_19 = _5508.w;
            mis_weight = power_heuristic(param_18, param_19);
        }
        float3 _5531 = (ls.col * _5508.xyz) * ((mix_weight * mis_weight) / ls.pdf);
        [branch]
        if (!ls.cast_shadow)
        {
            _8840 = _5531;
            break;
        }
        float3 param_20 = surf.P;
        float3 param_21 = -surf.plane_N;
        float3 _5544 = offset_ray(param_20, param_21);
        sh_r.o[0] = _5544.x;
        sh_r.o[1] = _5544.y;
        sh_r.o[2] = _5544.z;
        sh_r.c[0] = ray.c[0] * _5531.x;
        sh_r.c[1] = ray.c[1] * _5531.y;
        sh_r.c[2] = ray.c[2] * _5531.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8840 = 0.0f.xxx;
        break;
    } while(false);
    return _8840;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _8865;
    do
    {
        float _3048 = roughness * roughness;
        [branch]
        if ((_3048 * _3048) < 1.0000000116860974230803549289703e-07f)
        {
            float _3058 = dot(I, N);
            float _3059 = -_3058;
            float _3069 = mad(-(eta * eta), mad(_3058, _3059, 1.0f), 1.0f);
            if (_3069 < 0.0f)
            {
                _8865 = 0.0f.xxxx;
                break;
            }
            float _3081 = mad(eta, _3059, -sqrt(_3069));
            out_V = float4(normalize((I * eta) + (N * _3081)), _3081);
            _8865 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param = T;
        float3 param_1 = B;
        float3 param_2 = N;
        float3 param_3 = -I;
        float3 _3121 = normalize(tangent_from_world(param, param_1, param_2, param_3));
        float param_4 = _3048;
        float param_5 = _3048;
        float param_6 = rand_u;
        float param_7 = rand_v;
        float3 _3132 = SampleGGX_VNDF(_3121, param_4, param_5, param_6, param_7);
        float _3136 = dot(_3121, _3132);
        float _3146 = mad(-(eta * eta), mad(-_3136, _3136, 1.0f), 1.0f);
        if (_3146 < 0.0f)
        {
            _8865 = 0.0f.xxxx;
            break;
        }
        float _3158 = mad(eta, _3136, -sqrt(_3146));
        float3 _3168 = normalize((_3121 * (-eta)) + (_3132 * _3158));
        float3 param_8 = _3121;
        float3 param_9 = _3132;
        float3 param_10 = _3168;
        float param_11 = _3048;
        float param_12 = eta;
        float3 param_13 = refr_col;
        float3 param_14 = T;
        float3 param_15 = B;
        float3 param_16 = N;
        float3 param_17 = _3168;
        out_V = float4(world_from_tangent(param_14, param_15, param_16, param_17), _3158);
        _8865 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _8865;
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
    float _2094 = old_value;
    old_value = new_value;
    return _2094;
}

float pop_ior_stack(inout float stack[4], float default_value)
{
    float _8873;
    do
    {
        if (stack[3] > 0.0f)
        {
            float param = stack[3];
            float param_1 = -1.0f;
            float _2136 = exchange(param, param_1);
            stack[3] = param;
            _8873 = _2136;
            break;
        }
        if (stack[2] > 0.0f)
        {
            float param_2 = stack[2];
            float param_3 = -1.0f;
            float _2149 = exchange(param_2, param_3);
            stack[2] = param_2;
            _8873 = _2149;
            break;
        }
        if (stack[1] > 0.0f)
        {
            float param_4 = stack[1];
            float param_5 = -1.0f;
            float _2162 = exchange(param_4, param_5);
            stack[1] = param_4;
            _8873 = _2162;
            break;
        }
        if (stack[0] > 0.0f)
        {
            float param_6 = stack[0];
            float param_7 = -1.0f;
            float _2175 = exchange(param_6, param_7);
            stack[0] = param_6;
            _8873 = _2175;
            break;
        }
        _8873 = default_value;
        break;
    } while(false);
    return _8873;
}

void Sample_RefractiveNode(ray_data_t ray, surface_t surf, float3 base_color, float roughness, bool is_backfacing, float int_ior, float ext_ior, float rand_u, float rand_v, float mix_weight, inout ray_data_t new_ray)
{
    float _5581;
    if (is_backfacing)
    {
        _5581 = int_ior / ext_ior;
    }
    else
    {
        _5581 = ext_ior / int_ior;
    }
    float3 param = surf.T;
    float3 param_1 = surf.B;
    float3 param_2 = surf.N;
    float3 param_3 = float3(ray.d[0], ray.d[1], ray.d[2]);
    float param_4 = roughness;
    float param_5 = _5581;
    float3 param_6 = base_color;
    float param_7 = rand_u;
    float param_8 = rand_v;
    float4 param_9;
    float4 _5605 = Sample_GGXRefraction_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9);
    new_ray.depth = ray.depth + 65536;
    new_ray.c[0] = ((ray.c[0] * _5605.x) * mix_weight) / _5605.w;
    new_ray.c[1] = ((ray.c[1] * _5605.y) * mix_weight) / _5605.w;
    new_ray.c[2] = ((ray.c[2] * _5605.z) * mix_weight) / _5605.w;
    new_ray.pdf = _5605.w;
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
        float _5661 = pop_ior_stack(param_11, param_12);
        new_ray.ior = param_11;
    }
    float3 param_13 = surf.P;
    float3 param_14 = -surf.plane_N;
    float3 _5670 = offset_ray(param_13, param_14);
    new_ray.o[0] = _5670.x;
    new_ray.o[1] = _5670.y;
    new_ray.o[2] = _5670.z;
    new_ray.d[0] = param_9.x;
    new_ray.d[1] = param_9.y;
    new_ray.d[2] = param_9.z;
}

lobe_weights_t get_lobe_weights(float base_color_lum, float spec_color_lum, float specular, float metallic, float transmission, float clearcoat)
{
    float _1500 = 1.0f - metallic;
    float _8959 = (base_color_lum * _1500) * (1.0f - transmission);
    float _1507 = transmission * _1500;
    float _1511;
    if ((specular != 0.0f) || (metallic != 0.0f))
    {
        _1511 = spec_color_lum * mad(-transmission, _1500, 1.0f);
    }
    else
    {
        _1511 = 0.0f;
    }
    float _8960 = _1511;
    float _1521 = 0.25f * clearcoat;
    float _8961 = _1521 * _1500;
    float _8962 = _1507 * base_color_lum;
    float _1530 = _8959;
    float _1539 = mad(_1507, base_color_lum, mad(_1521, _1500, _1530 + _1511));
    if (_1539 != 0.0f)
    {
        _8959 /= _1539;
        _8960 /= _1539;
        _8961 /= _1539;
        _8962 /= _1539;
    }
    lobe_weights_t _8967 = { _8959, _8960, _8961, _8962 };
    return _8967;
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
    float _8888;
    do
    {
        float _2256 = dot(N, L);
        if (_2256 <= 0.0f)
        {
            _8888 = 0.0f;
            break;
        }
        float param = _2256;
        float param_1 = dot(N, V);
        float _2277 = dot(L, H);
        float _2285 = mad((2.0f * _2277) * _2277, roughness, 0.5f);
        _8888 = lerp(1.0f, _2285, schlick_weight(param)) * lerp(1.0f, _2285, schlick_weight(param_1));
        break;
    } while(false);
    return _8888;
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
    float3 _2426 = normalize(L + V);
    float3 H = _2426;
    if (dot(V, _2426) < 0.0f)
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
    float3 _2461 = diff_col;
    float3 _2462 = _2461 + (sheen_color * (3.1415927410125732421875f * schlick_weight(param_5)));
    diff_col = _2462;
    return float4(_2462, pdf);
}

float D_GTR1(float NDotH, float a)
{
    float _8893;
    do
    {
        if (a >= 1.0f)
        {
            _8893 = 0.3183098733425140380859375f;
            break;
        }
        float _2004 = mad(a, a, -1.0f);
        _8893 = _2004 / ((3.1415927410125732421875f * log(a * a)) * mad(_2004 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _8893;
}

float4 Evaluate_PrincipledClearcoat_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float clearcoat_roughness2, float clearcoat_ior, float clearcoat_F0)
{
    float param = sampled_normal_ts.z;
    float param_1 = clearcoat_roughness2;
    float _2762 = D_GTR1(param, param_1);
    float3 param_2 = view_dir_ts;
    float param_3 = 0.0625f;
    float param_4 = 0.0625f;
    float _2769 = G1(param_2, param_3, param_4);
    float3 param_5 = reflected_dir_ts;
    float param_6 = 0.0625f;
    float param_7 = 0.0625f;
    float _2774 = G1(param_5, param_6, param_7);
    float param_8 = dot(reflected_dir_ts, sampled_normal_ts);
    float param_9 = clearcoat_ior;
    float F = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param_8, param_9) - clearcoat_F0) / (1.0f - clearcoat_F0));
    float _2801 = (4.0f * abs(view_dir_ts.z)) * abs(reflected_dir_ts.z);
    float _2804;
    if (_2801 != 0.0f)
    {
        _2804 = (_2762 * (_2769 * _2774)) / _2801;
    }
    else
    {
        _2804 = 0.0f;
    }
    F *= _2804;
    float3 param_10 = view_dir_ts;
    float param_11 = 0.0625f;
    float param_12 = 0.0625f;
    float _2822 = G1(param_10, param_11, param_12);
    float pdf = ((_2762 * _2822) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2837 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2837 != 0.0f)
    {
        pdf /= _2837;
    }
    float _2848 = F;
    float _2849 = _2848 * clamp(reflected_dir_ts.z, 0.0f, 1.0f);
    F = _2849;
    return float4(_2849, _2849, _2849, pdf);
}

float3 Evaluate_PrincipledNode(light_sample_t ls, ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float N_dot_L, float mix_weight, inout shadow_ray_t sh_r)
{
    float3 _8845;
    do
    {
        float3 _5693 = float3(ray.d[0], ray.d[1], ray.d[2]);
        float3 lcol = 0.0f.xxx;
        float bsdf_pdf = 0.0f;
        bool _5698 = N_dot_L > 0.0f;
        [branch]
        if ((lobe_weights.diffuse > 1.0000000116860974230803549289703e-07f) && _5698)
        {
            float3 param = -_5693;
            float3 param_1 = surf.N;
            float3 param_2 = ls.L;
            float param_3 = diff.roughness;
            float3 param_4 = diff.base_color;
            float3 param_5 = diff.sheen_color;
            bool param_6 = false;
            float4 _5717 = Evaluate_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6);
            bsdf_pdf = mad(lobe_weights.diffuse, _5717.w, bsdf_pdf);
            lcol += (((ls.col * N_dot_L) * (_5717 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * ls.pdf).xxx);
        }
        float3 H;
        [flatten]
        if (_5698)
        {
            H = normalize(ls.L - _5693);
        }
        else
        {
            H = normalize(ls.L - (_5693 * trans.eta));
        }
        float _5756 = spec.roughness * spec.roughness;
        float _5761 = sqrt(mad(-0.89999997615814208984375f, spec.anisotropy, 1.0f));
        float _5765 = _5756 / _5761;
        float _5769 = _5756 * _5761;
        float3 param_7 = surf.T;
        float3 param_8 = surf.B;
        float3 param_9 = surf.N;
        float3 param_10 = -_5693;
        float3 _5780 = tangent_from_world(param_7, param_8, param_9, param_10);
        float3 param_11 = surf.T;
        float3 param_12 = surf.B;
        float3 param_13 = surf.N;
        float3 param_14 = ls.L;
        float3 _5790 = tangent_from_world(param_11, param_12, param_13, param_14);
        float3 param_15 = surf.T;
        float3 param_16 = surf.B;
        float3 param_17 = surf.N;
        float3 param_18 = H;
        float3 _5800 = tangent_from_world(param_15, param_16, param_17, param_18);
        bool _5802 = lobe_weights.specular > 0.0f;
        bool _5809;
        if (_5802)
        {
            _5809 = (_5765 * _5769) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _5809 = _5802;
        }
        [branch]
        if (_5809 && _5698)
        {
            float3 param_19 = _5780;
            float3 param_20 = _5800;
            float3 param_21 = _5790;
            float param_22 = _5765;
            float param_23 = _5769;
            float param_24 = spec.ior;
            float param_25 = spec.F0;
            float3 param_26 = spec.tmp_col;
            float4 _5831 = Evaluate_GGXSpecular_BSDF(param_19, param_20, param_21, param_22, param_23, param_24, param_25, param_26);
            bsdf_pdf = mad(lobe_weights.specular, _5831.w, bsdf_pdf);
            lcol += ((ls.col * _5831.xyz) / ls.pdf.xxx);
        }
        float _5850 = coat.roughness * coat.roughness;
        bool _5852 = lobe_weights.clearcoat > 0.0f;
        bool _5859;
        if (_5852)
        {
            _5859 = (_5850 * _5850) >= 1.0000000116860974230803549289703e-07f;
        }
        else
        {
            _5859 = _5852;
        }
        [branch]
        if (_5859 && _5698)
        {
            float3 param_27 = _5780;
            float3 param_28 = _5800;
            float3 param_29 = _5790;
            float param_30 = _5850;
            float param_31 = coat.ior;
            float param_32 = coat.F0;
            float4 _5877 = Evaluate_PrincipledClearcoat_BSDF(param_27, param_28, param_29, param_30, param_31, param_32);
            bsdf_pdf = mad(lobe_weights.clearcoat, _5877.w, bsdf_pdf);
            lcol += (((ls.col * 0.25f) * _5877.xyz) / ls.pdf.xxx);
        }
        [branch]
        if (lobe_weights.refraction > 0.0f)
        {
            bool _5899 = trans.fresnel != 0.0f;
            bool _5906;
            if (_5899)
            {
                _5906 = (_5756 * _5756) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _5906 = _5899;
            }
            [branch]
            if (_5906 && _5698)
            {
                float3 param_33 = _5780;
                float3 param_34 = _5800;
                float3 param_35 = _5790;
                float param_36 = _5756;
                float param_37 = _5756;
                float param_38 = 1.0f;
                float param_39 = 0.0f;
                float3 param_40 = 1.0f.xxx;
                float4 _5925 = Evaluate_GGXSpecular_BSDF(param_33, param_34, param_35, param_36, param_37, param_38, param_39, param_40);
                bsdf_pdf = mad(lobe_weights.refraction * trans.fresnel, _5925.w, bsdf_pdf);
                lcol += ((ls.col * _5925.xyz) * (trans.fresnel / ls.pdf));
            }
            float _5947 = trans.roughness * trans.roughness;
            bool _5949 = trans.fresnel != 1.0f;
            bool _5956;
            if (_5949)
            {
                _5956 = (_5947 * _5947) >= 1.0000000116860974230803549289703e-07f;
            }
            else
            {
                _5956 = _5949;
            }
            [branch]
            if (_5956 && (N_dot_L < 0.0f))
            {
                float3 param_41 = _5780;
                float3 param_42 = _5800;
                float3 param_43 = _5790;
                float param_44 = _5947;
                float param_45 = trans.eta;
                float3 param_46 = diff.base_color;
                float4 _5974 = Evaluate_GGXRefraction_BSDF(param_41, param_42, param_43, param_44, param_45, param_46);
                float _5977 = 1.0f - trans.fresnel;
                bsdf_pdf = mad(lobe_weights.refraction * _5977, _5974.w, bsdf_pdf);
                lcol += ((ls.col * _5974.xyz) * (_5977 / ls.pdf));
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
            _8845 = lcol;
            break;
        }
        float3 _6017;
        if (N_dot_L < 0.0f)
        {
            _6017 = -surf.plane_N;
        }
        else
        {
            _6017 = surf.plane_N;
        }
        float3 param_49 = surf.P;
        float3 param_50 = _6017;
        float3 _6028 = offset_ray(param_49, param_50);
        sh_r.o[0] = _6028.x;
        sh_r.o[1] = _6028.y;
        sh_r.o[2] = _6028.z;
        sh_r.c[0] = ray.c[0] * lcol.x;
        sh_r.c[1] = ray.c[1] * lcol.y;
        sh_r.c[2] = ray.c[2] * lcol.z;
        sh_r.xy = ray.xy;
        sh_r.depth = ray.depth;
        _8845 = 0.0f.xxx;
        break;
    } while(false);
    return _8845;
}

float4 Sample_PrincipledDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float3 sheen_color, bool uniform_sampling, float rand_u, float rand_v, inout float3 out_V)
{
    float _2473 = 6.283185482025146484375f * rand_v;
    float _2476 = cos(_2473);
    float _2479 = sin(_2473);
    float3 V;
    if (uniform_sampling)
    {
        float _2488 = sqrt(mad(-rand_u, rand_u, 1.0f));
        V = float3(_2488 * _2476, _2488 * _2479, rand_u);
    }
    else
    {
        float _2501 = sqrt(rand_u);
        V = float3(_2501 * _2476, _2501 * _2479, sqrt(1.0f - rand_u));
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
    float4 _8878;
    do
    {
        [branch]
        if ((clearcoat_roughness2 * clearcoat_roughness2) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2866 = reflect(I, N);
            float param = dot(_2866, N);
            float param_1 = clearcoat_ior;
            out_V = _2866;
            float _2885 = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param, param_1) - clearcoat_F0) / (1.0f - clearcoat_F0)) * 1000000.0f;
            _8878 = float4(_2885, _2885, _2885, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2903 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = clearcoat_roughness2;
        float param_7 = clearcoat_roughness2;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2914 = SampleGGX_VNDF(_2903, param_6, param_7, param_8, param_9);
        float3 _2925 = normalize(reflect(-_2903, _2914));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2925;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2903;
        float3 param_15 = _2914;
        float3 param_16 = _2925;
        float param_17 = clearcoat_roughness2;
        float param_18 = clearcoat_ior;
        float param_19 = clearcoat_F0;
        _8878 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _8878;
}

void Sample_PrincipledNode(ray_data_t ray, surface_t surf, lobe_weights_t lobe_weights, diff_params_t diff, spec_params_t spec, clearcoat_params_t coat, transmission_params_t trans, float metallic, float rand_u, float rand_v, inout float mix_rand, float mix_weight, inout ray_data_t new_ray)
{
    float3 _6063 = float3(ray.d[0], ray.d[1], ray.d[2]);
    int _6067 = ray.depth & 255;
    int _6071 = (ray.depth >> 8) & 255;
    int _6075 = (ray.depth >> 16) & 255;
    int _6086 = (_6067 + _6071) + _6075;
    [branch]
    if (mix_rand < lobe_weights.diffuse)
    {
        bool _6095 = _6067 < _3325_g_params.max_diff_depth;
        bool _6102;
        if (_6095)
        {
            _6102 = _6086 < _3325_g_params.max_total_depth;
        }
        else
        {
            _6102 = _6095;
        }
        if (_6102)
        {
            float3 param = surf.T;
            float3 param_1 = surf.B;
            float3 param_2 = surf.N;
            float3 param_3 = _6063;
            float param_4 = diff.roughness;
            float3 param_5 = diff.base_color;
            float3 param_6 = diff.sheen_color;
            bool param_7 = false;
            float param_8 = rand_u;
            float param_9 = rand_v;
            float3 param_10;
            float4 _6125 = Sample_PrincipledDiffuse_BSDF(param, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9, param_10);
            float3 _6130 = _6125.xyz * (1.0f - metallic);
            new_ray.depth = ray.depth + 1;
            float3 param_11 = surf.P;
            float3 param_12 = surf.plane_N;
            float3 _6145 = offset_ray(param_11, param_12);
            new_ray.o[0] = _6145.x;
            new_ray.o[1] = _6145.y;
            new_ray.o[2] = _6145.z;
            new_ray.d[0] = param_10.x;
            new_ray.d[1] = param_10.y;
            new_ray.d[2] = param_10.z;
            new_ray.c[0] = ((ray.c[0] * _6130.x) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[1] = ((ray.c[1] * _6130.y) * mix_weight) / lobe_weights.diffuse;
            new_ray.c[2] = ((ray.c[2] * _6130.z) * mix_weight) / lobe_weights.diffuse;
            new_ray.pdf = _6125.w;
        }
    }
    else
    {
        float _6195 = lobe_weights.diffuse + lobe_weights.specular;
        [branch]
        if (mix_rand < _6195)
        {
            bool _6202 = _6071 < _3325_g_params.max_spec_depth;
            bool _6209;
            if (_6202)
            {
                _6209 = _6086 < _3325_g_params.max_total_depth;
            }
            else
            {
                _6209 = _6202;
            }
            if (_6209)
            {
                float3 param_13 = surf.T;
                float3 param_14 = surf.B;
                float3 param_15 = surf.N;
                float3 param_16 = _6063;
                float3 param_17;
                float4 _6228 = Sample_GGXSpecular_BSDF(param_13, param_14, param_15, param_16, spec.roughness, spec.anisotropy, spec.ior, spec.F0, spec.tmp_col, rand_u, rand_v, param_17);
                float _6233 = _6228.w * lobe_weights.specular;
                new_ray.depth = ray.depth + 256;
                new_ray.c[0] = ((ray.c[0] * _6228.x) * mix_weight) / _6233;
                new_ray.c[1] = ((ray.c[1] * _6228.y) * mix_weight) / _6233;
                new_ray.c[2] = ((ray.c[2] * _6228.z) * mix_weight) / _6233;
                new_ray.pdf = _6233;
                float3 param_18 = surf.P;
                float3 param_19 = surf.plane_N;
                float3 _6273 = offset_ray(param_18, param_19);
                new_ray.o[0] = _6273.x;
                new_ray.o[1] = _6273.y;
                new_ray.o[2] = _6273.z;
                new_ray.d[0] = param_17.x;
                new_ray.d[1] = param_17.y;
                new_ray.d[2] = param_17.z;
            }
        }
        else
        {
            float _6298 = _6195 + lobe_weights.clearcoat;
            [branch]
            if (mix_rand < _6298)
            {
                bool _6305 = _6071 < _3325_g_params.max_spec_depth;
                bool _6312;
                if (_6305)
                {
                    _6312 = _6086 < _3325_g_params.max_total_depth;
                }
                else
                {
                    _6312 = _6305;
                }
                if (_6312)
                {
                    float3 param_20 = surf.T;
                    float3 param_21 = surf.B;
                    float3 param_22 = surf.N;
                    float3 param_23 = _6063;
                    float param_24 = coat.roughness * coat.roughness;
                    float param_25 = coat.ior;
                    float param_26 = coat.F0;
                    float param_27 = rand_u;
                    float param_28 = rand_v;
                    float3 param_29;
                    float4 _6336 = Sample_PrincipledClearcoat_BSDF(param_20, param_21, param_22, param_23, param_24, param_25, param_26, param_27, param_28, param_29);
                    float _6341 = _6336.w * lobe_weights.clearcoat;
                    new_ray.depth = ray.depth + 256;
                    new_ray.c[0] = (((0.25f * ray.c[0]) * _6336.x) * mix_weight) / _6341;
                    new_ray.c[1] = (((0.25f * ray.c[1]) * _6336.y) * mix_weight) / _6341;
                    new_ray.c[2] = (((0.25f * ray.c[2]) * _6336.z) * mix_weight) / _6341;
                    new_ray.pdf = _6341;
                    float3 param_30 = surf.P;
                    float3 param_31 = surf.plane_N;
                    float3 _6384 = offset_ray(param_30, param_31);
                    new_ray.o[0] = _6384.x;
                    new_ray.o[1] = _6384.y;
                    new_ray.o[2] = _6384.z;
                    new_ray.d[0] = param_29.x;
                    new_ray.d[1] = param_29.y;
                    new_ray.d[2] = param_29.z;
                }
            }
            else
            {
                bool _6406 = mix_rand >= trans.fresnel;
                bool _6413;
                if (_6406)
                {
                    _6413 = _6075 < _3325_g_params.max_refr_depth;
                }
                else
                {
                    _6413 = _6406;
                }
                bool _6427;
                if (!_6413)
                {
                    bool _6419 = mix_rand < trans.fresnel;
                    bool _6426;
                    if (_6419)
                    {
                        _6426 = _6071 < _3325_g_params.max_spec_depth;
                    }
                    else
                    {
                        _6426 = _6419;
                    }
                    _6427 = _6426;
                }
                else
                {
                    _6427 = _6413;
                }
                bool _6434;
                if (_6427)
                {
                    _6434 = _6086 < _3325_g_params.max_total_depth;
                }
                else
                {
                    _6434 = _6427;
                }
                [branch]
                if (_6434)
                {
                    mix_rand -= _6298;
                    mix_rand /= lobe_weights.refraction;
                    float4 F;
                    float3 V;
                    [branch]
                    if (mix_rand < trans.fresnel)
                    {
                        float3 param_32 = surf.T;
                        float3 param_33 = surf.B;
                        float3 param_34 = surf.N;
                        float3 param_35 = _6063;
                        float3 param_36;
                        float4 _6464 = Sample_GGXSpecular_BSDF(param_32, param_33, param_34, param_35, spec.roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, rand_u, rand_v, param_36);
                        V = param_36;
                        F = _6464;
                        new_ray.depth = ray.depth + 256;
                        float3 param_37 = surf.P;
                        float3 param_38 = surf.plane_N;
                        float3 _6474 = offset_ray(param_37, param_38);
                        new_ray.o[0] = _6474.x;
                        new_ray.o[1] = _6474.y;
                        new_ray.o[2] = _6474.z;
                    }
                    else
                    {
                        float3 param_39 = surf.T;
                        float3 param_40 = surf.B;
                        float3 param_41 = surf.N;
                        float3 param_42 = _6063;
                        float param_43 = trans.roughness;
                        float param_44 = trans.eta;
                        float3 param_45 = diff.base_color;
                        float param_46 = rand_u;
                        float param_47 = rand_v;
                        float4 param_48;
                        float4 _6503 = Sample_GGXRefraction_BSDF(param_39, param_40, param_41, param_42, param_43, param_44, param_45, param_46, param_47, param_48);
                        F = _6503;
                        V = param_48.xyz;
                        new_ray.depth = ray.depth + 65536;
                        float3 param_49 = surf.P;
                        float3 param_50 = -surf.plane_N;
                        float3 _6516 = offset_ray(param_49, param_50);
                        new_ray.o[0] = _6516.x;
                        new_ray.o[1] = _6516.y;
                        new_ray.o[2] = _6516.z;
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
                            float _6542 = pop_ior_stack(param_52, param_53);
                            new_ray.ior = param_52;
                        }
                    }
                    float4 _10126 = F;
                    float _6548 = _10126.w * lobe_weights.refraction;
                    float4 _10128 = _10126;
                    _10128.w = _6548;
                    F = _10128;
                    new_ray.c[0] = ((ray.c[0] * _10126.x) * mix_weight) / _6548;
                    new_ray.c[1] = ((ray.c[1] * _10126.y) * mix_weight) / _6548;
                    new_ray.c[2] = ((ray.c[2] * _10126.z) * mix_weight) / _6548;
                    new_ray.pdf = _6548;
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
    float3 _8815;
    do
    {
        float3 _6604 = float3(ray.d[0], ray.d[1], ray.d[2]);
        [branch]
        if (inter.mask == 0)
        {
            ray_data_t param = ray;
            float3 _6613 = Evaluate_EnvColor(param);
            _8815 = float3(ray.c[0] * _6613.x, ray.c[1] * _6613.y, ray.c[2] * _6613.z);
            break;
        }
        float3 _6640 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_6604 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            ray_data_t param_1 = ray;
            hit_data_t param_2 = inter;
            float3 _6652 = Evaluate_LightColor(param_1, param_2);
            _8815 = float3(ray.c[0] * _6652.x, ray.c[1] * _6652.y, ray.c[2] * _6652.z);
            break;
        }
        bool _6673 = inter.prim_index < 0;
        int _6676;
        if (_6673)
        {
            _6676 = (-1) - inter.prim_index;
        }
        else
        {
            _6676 = inter.prim_index;
        }
        uint _6687 = uint(_6676);
        material_t _6695;
        [unroll]
        for (int _61ident = 0; _61ident < 5; _61ident++)
        {
            _6695.textures[_61ident] = _4453.Load(_61ident * 4 + ((_4457.Load(_6687 * 4 + 0) >> 16u) & 16383u) * 76 + 0);
        }
        [unroll]
        for (int _62ident = 0; _62ident < 3; _62ident++)
        {
            _6695.base_color[_62ident] = asfloat(_4453.Load(_62ident * 4 + ((_4457.Load(_6687 * 4 + 0) >> 16u) & 16383u) * 76 + 20));
        }
        _6695.flags = _4453.Load(((_4457.Load(_6687 * 4 + 0) >> 16u) & 16383u) * 76 + 32);
        _6695.type = _4453.Load(((_4457.Load(_6687 * 4 + 0) >> 16u) & 16383u) * 76 + 36);
        _6695.tangent_rotation_or_strength = asfloat(_4453.Load(((_4457.Load(_6687 * 4 + 0) >> 16u) & 16383u) * 76 + 40));
        _6695.roughness_and_anisotropic = _4453.Load(((_4457.Load(_6687 * 4 + 0) >> 16u) & 16383u) * 76 + 44);
        _6695.ior = asfloat(_4453.Load(((_4457.Load(_6687 * 4 + 0) >> 16u) & 16383u) * 76 + 48));
        _6695.sheen_and_sheen_tint = _4453.Load(((_4457.Load(_6687 * 4 + 0) >> 16u) & 16383u) * 76 + 52);
        _6695.tint_and_metallic = _4453.Load(((_4457.Load(_6687 * 4 + 0) >> 16u) & 16383u) * 76 + 56);
        _6695.transmission_and_transmission_roughness = _4453.Load(((_4457.Load(_6687 * 4 + 0) >> 16u) & 16383u) * 76 + 60);
        _6695.specular_and_specular_tint = _4453.Load(((_4457.Load(_6687 * 4 + 0) >> 16u) & 16383u) * 76 + 64);
        _6695.clearcoat_and_clearcoat_roughness = _4453.Load(((_4457.Load(_6687 * 4 + 0) >> 16u) & 16383u) * 76 + 68);
        _6695.normal_map_strength_unorm = _4453.Load(((_4457.Load(_6687 * 4 + 0) >> 16u) & 16383u) * 76 + 72);
        uint _9575 = _6695.textures[0];
        uint _9576 = _6695.textures[1];
        uint _9577 = _6695.textures[2];
        uint _9578 = _6695.textures[3];
        uint _9579 = _6695.textures[4];
        float _9580 = _6695.base_color[0];
        float _9581 = _6695.base_color[1];
        float _9582 = _6695.base_color[2];
        uint _9176 = _6695.flags;
        uint _9177 = _6695.type;
        float _9178 = _6695.tangent_rotation_or_strength;
        uint _9179 = _6695.roughness_and_anisotropic;
        float _9180 = _6695.ior;
        uint _9181 = _6695.sheen_and_sheen_tint;
        uint _9182 = _6695.tint_and_metallic;
        uint _9183 = _6695.transmission_and_transmission_roughness;
        uint _9184 = _6695.specular_and_specular_tint;
        uint _9185 = _6695.clearcoat_and_clearcoat_roughness;
        uint _9186 = _6695.normal_map_strength_unorm;
        transform_t _6750;
        _6750.xform = asfloat(uint4x4(_4100.Load4(asuint(asfloat(_6743.Load(inter.obj_index * 32 + 12))) * 128 + 0), _4100.Load4(asuint(asfloat(_6743.Load(inter.obj_index * 32 + 12))) * 128 + 16), _4100.Load4(asuint(asfloat(_6743.Load(inter.obj_index * 32 + 12))) * 128 + 32), _4100.Load4(asuint(asfloat(_6743.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _6750.inv_xform = asfloat(uint4x4(_4100.Load4(asuint(asfloat(_6743.Load(inter.obj_index * 32 + 12))) * 128 + 64), _4100.Load4(asuint(asfloat(_6743.Load(inter.obj_index * 32 + 12))) * 128 + 80), _4100.Load4(asuint(asfloat(_6743.Load(inter.obj_index * 32 + 12))) * 128 + 96), _4100.Load4(asuint(asfloat(_6743.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _6757 = _6687 * 3u;
        vertex_t _6762;
        [unroll]
        for (int _63ident = 0; _63ident < 3; _63ident++)
        {
            _6762.p[_63ident] = asfloat(_4125.Load(_63ident * 4 + _4129.Load(_6757 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _64ident = 0; _64ident < 3; _64ident++)
        {
            _6762.n[_64ident] = asfloat(_4125.Load(_64ident * 4 + _4129.Load(_6757 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _65ident = 0; _65ident < 3; _65ident++)
        {
            _6762.b[_65ident] = asfloat(_4125.Load(_65ident * 4 + _4129.Load(_6757 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _66ident = 0; _66ident < 2; _66ident++)
        {
            [unroll]
            for (int _67ident = 0; _67ident < 2; _67ident++)
            {
                _6762.t[_66ident][_67ident] = asfloat(_4125.Load(_67ident * 4 + _66ident * 8 + _4129.Load(_6757 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _6808;
        [unroll]
        for (int _68ident = 0; _68ident < 3; _68ident++)
        {
            _6808.p[_68ident] = asfloat(_4125.Load(_68ident * 4 + _4129.Load((_6757 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _69ident = 0; _69ident < 3; _69ident++)
        {
            _6808.n[_69ident] = asfloat(_4125.Load(_69ident * 4 + _4129.Load((_6757 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _70ident = 0; _70ident < 3; _70ident++)
        {
            _6808.b[_70ident] = asfloat(_4125.Load(_70ident * 4 + _4129.Load((_6757 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _71ident = 0; _71ident < 2; _71ident++)
        {
            [unroll]
            for (int _72ident = 0; _72ident < 2; _72ident++)
            {
                _6808.t[_71ident][_72ident] = asfloat(_4125.Load(_72ident * 4 + _71ident * 8 + _4129.Load((_6757 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _6854;
        [unroll]
        for (int _73ident = 0; _73ident < 3; _73ident++)
        {
            _6854.p[_73ident] = asfloat(_4125.Load(_73ident * 4 + _4129.Load((_6757 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _74ident = 0; _74ident < 3; _74ident++)
        {
            _6854.n[_74ident] = asfloat(_4125.Load(_74ident * 4 + _4129.Load((_6757 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _75ident = 0; _75ident < 3; _75ident++)
        {
            _6854.b[_75ident] = asfloat(_4125.Load(_75ident * 4 + _4129.Load((_6757 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _76ident = 0; _76ident < 2; _76ident++)
        {
            [unroll]
            for (int _77ident = 0; _77ident < 2; _77ident++)
            {
                _6854.t[_76ident][_77ident] = asfloat(_4125.Load(_77ident * 4 + _76ident * 8 + _4129.Load((_6757 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _6900 = float3(_6762.p[0], _6762.p[1], _6762.p[2]);
        float3 _6908 = float3(_6808.p[0], _6808.p[1], _6808.p[2]);
        float3 _6916 = float3(_6854.p[0], _6854.p[1], _6854.p[2]);
        float _6923 = (1.0f - inter.u) - inter.v;
        float3 _6955 = normalize(((float3(_6762.n[0], _6762.n[1], _6762.n[2]) * _6923) + (float3(_6808.n[0], _6808.n[1], _6808.n[2]) * inter.u)) + (float3(_6854.n[0], _6854.n[1], _6854.n[2]) * inter.v));
        float3 _9115 = _6955;
        float2 _6981 = ((float2(_6762.t[0][0], _6762.t[0][1]) * _6923) + (float2(_6808.t[0][0], _6808.t[0][1]) * inter.u)) + (float2(_6854.t[0][0], _6854.t[0][1]) * inter.v);
        float3 _6997 = cross(_6908 - _6900, _6916 - _6900);
        float _7002 = length(_6997);
        float3 _9116 = _6997 / _7002.xxx;
        float3 _7039 = ((float3(_6762.b[0], _6762.b[1], _6762.b[2]) * _6923) + (float3(_6808.b[0], _6808.b[1], _6808.b[2]) * inter.u)) + (float3(_6854.b[0], _6854.b[1], _6854.b[2]) * inter.v);
        float3 _9114 = _7039;
        float3 _9113 = cross(_7039, _6955);
        if (_6673)
        {
            if ((_4457.Load(_6687 * 4 + 0) & 65535u) == 65535u)
            {
                _8815 = 0.0f.xxx;
                break;
            }
            material_t _7065;
            [unroll]
            for (int _78ident = 0; _78ident < 5; _78ident++)
            {
                _7065.textures[_78ident] = _4453.Load(_78ident * 4 + (_4457.Load(_6687 * 4 + 0) & 16383u) * 76 + 0);
            }
            [unroll]
            for (int _79ident = 0; _79ident < 3; _79ident++)
            {
                _7065.base_color[_79ident] = asfloat(_4453.Load(_79ident * 4 + (_4457.Load(_6687 * 4 + 0) & 16383u) * 76 + 20));
            }
            _7065.flags = _4453.Load((_4457.Load(_6687 * 4 + 0) & 16383u) * 76 + 32);
            _7065.type = _4453.Load((_4457.Load(_6687 * 4 + 0) & 16383u) * 76 + 36);
            _7065.tangent_rotation_or_strength = asfloat(_4453.Load((_4457.Load(_6687 * 4 + 0) & 16383u) * 76 + 40));
            _7065.roughness_and_anisotropic = _4453.Load((_4457.Load(_6687 * 4 + 0) & 16383u) * 76 + 44);
            _7065.ior = asfloat(_4453.Load((_4457.Load(_6687 * 4 + 0) & 16383u) * 76 + 48));
            _7065.sheen_and_sheen_tint = _4453.Load((_4457.Load(_6687 * 4 + 0) & 16383u) * 76 + 52);
            _7065.tint_and_metallic = _4453.Load((_4457.Load(_6687 * 4 + 0) & 16383u) * 76 + 56);
            _7065.transmission_and_transmission_roughness = _4453.Load((_4457.Load(_6687 * 4 + 0) & 16383u) * 76 + 60);
            _7065.specular_and_specular_tint = _4453.Load((_4457.Load(_6687 * 4 + 0) & 16383u) * 76 + 64);
            _7065.clearcoat_and_clearcoat_roughness = _4453.Load((_4457.Load(_6687 * 4 + 0) & 16383u) * 76 + 68);
            _7065.normal_map_strength_unorm = _4453.Load((_4457.Load(_6687 * 4 + 0) & 16383u) * 76 + 72);
            _9575 = _7065.textures[0];
            _9576 = _7065.textures[1];
            _9577 = _7065.textures[2];
            _9578 = _7065.textures[3];
            _9579 = _7065.textures[4];
            _9580 = _7065.base_color[0];
            _9581 = _7065.base_color[1];
            _9582 = _7065.base_color[2];
            _9176 = _7065.flags;
            _9177 = _7065.type;
            _9178 = _7065.tangent_rotation_or_strength;
            _9179 = _7065.roughness_and_anisotropic;
            _9180 = _7065.ior;
            _9181 = _7065.sheen_and_sheen_tint;
            _9182 = _7065.tint_and_metallic;
            _9183 = _7065.transmission_and_transmission_roughness;
            _9184 = _7065.specular_and_specular_tint;
            _9185 = _7065.clearcoat_and_clearcoat_roughness;
            _9186 = _7065.normal_map_strength_unorm;
            _9116 = -_9116;
            _9115 = -_9115;
            _9114 = -_9114;
            _9113 = -_9113;
        }
        float3 param_3 = _9116;
        float4x4 param_4 = _6750.inv_xform;
        _9116 = TransformNormal(param_3, param_4);
        float3 param_5 = _9115;
        float4x4 param_6 = _6750.inv_xform;
        _9115 = TransformNormal(param_5, param_6);
        float3 param_7 = _9114;
        float4x4 param_8 = _6750.inv_xform;
        _9114 = TransformNormal(param_7, param_8);
        float3 param_9 = _9113;
        float4x4 param_10 = _6750.inv_xform;
        _9116 = normalize(_9116);
        _9115 = normalize(_9115);
        _9114 = normalize(_9114);
        _9113 = normalize(TransformNormal(param_9, param_10));
        float _7205 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _7215 = mad(0.5f, log2(abs(mad(_6808.t[0][0] - _6762.t[0][0], _6854.t[0][1] - _6762.t[0][1], -((_6854.t[0][0] - _6762.t[0][0]) * (_6808.t[0][1] - _6762.t[0][1])))) / _7002), log2(_7205));
        uint param_11 = uint(hash(ray.xy));
        float _7222 = construct_float(param_11);
        uint param_12 = uint(hash(hash(ray.xy)));
        float _7229 = construct_float(param_12);
        float param_13[4] = ray.ior;
        bool param_14 = _6673;
        float param_15 = 1.0f;
        float _7238 = peek_ior_stack(param_13, param_14, param_15);
        float3 col = 0.0f.xxx;
        int _7243 = ray.depth & 255;
        int _7248 = (ray.depth >> 8) & 255;
        int _7253 = (ray.depth >> 16) & 255;
        int _7264 = (_7243 + _7248) + _7253;
        int _7272 = _3325_g_params.hi + ((_7264 + ((ray.depth >> 24) & 255)) * 7);
        float mix_rand = frac(asfloat(_3309.Load(_7272 * 4 + 0)) + _7222);
        float mix_weight = 1.0f;
        float _7309;
        float _7326;
        float _7352;
        float _7419;
        while (_9177 == 4u)
        {
            float mix_val = _9178;
            if (_9576 != 4294967295u)
            {
                mix_val *= SampleBilinear(_9576, _6981, 0).x;
            }
            if (_6673)
            {
                _7309 = _7238 / _9180;
            }
            else
            {
                _7309 = _9180 / _7238;
            }
            if (_9180 != 0.0f)
            {
                float param_16 = dot(_6604, _9115);
                float param_17 = _7309;
                _7326 = fresnel_dielectric_cos(param_16, param_17);
            }
            else
            {
                _7326 = 1.0f;
            }
            float _7341 = mix_val;
            float _7342 = _7341 * clamp(_7326, 0.0f, 1.0f);
            mix_val = _7342;
            if (mix_rand > _7342)
            {
                if ((_9176 & 2u) != 0u)
                {
                    _7352 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _7352 = 1.0f;
                }
                mix_weight *= _7352;
                material_t _7365;
                [unroll]
                for (int _80ident = 0; _80ident < 5; _80ident++)
                {
                    _7365.textures[_80ident] = _4453.Load(_80ident * 4 + _9578 * 76 + 0);
                }
                [unroll]
                for (int _81ident = 0; _81ident < 3; _81ident++)
                {
                    _7365.base_color[_81ident] = asfloat(_4453.Load(_81ident * 4 + _9578 * 76 + 20));
                }
                _7365.flags = _4453.Load(_9578 * 76 + 32);
                _7365.type = _4453.Load(_9578 * 76 + 36);
                _7365.tangent_rotation_or_strength = asfloat(_4453.Load(_9578 * 76 + 40));
                _7365.roughness_and_anisotropic = _4453.Load(_9578 * 76 + 44);
                _7365.ior = asfloat(_4453.Load(_9578 * 76 + 48));
                _7365.sheen_and_sheen_tint = _4453.Load(_9578 * 76 + 52);
                _7365.tint_and_metallic = _4453.Load(_9578 * 76 + 56);
                _7365.transmission_and_transmission_roughness = _4453.Load(_9578 * 76 + 60);
                _7365.specular_and_specular_tint = _4453.Load(_9578 * 76 + 64);
                _7365.clearcoat_and_clearcoat_roughness = _4453.Load(_9578 * 76 + 68);
                _7365.normal_map_strength_unorm = _4453.Load(_9578 * 76 + 72);
                _9575 = _7365.textures[0];
                _9576 = _7365.textures[1];
                _9577 = _7365.textures[2];
                _9578 = _7365.textures[3];
                _9579 = _7365.textures[4];
                _9580 = _7365.base_color[0];
                _9581 = _7365.base_color[1];
                _9582 = _7365.base_color[2];
                _9176 = _7365.flags;
                _9177 = _7365.type;
                _9178 = _7365.tangent_rotation_or_strength;
                _9179 = _7365.roughness_and_anisotropic;
                _9180 = _7365.ior;
                _9181 = _7365.sheen_and_sheen_tint;
                _9182 = _7365.tint_and_metallic;
                _9183 = _7365.transmission_and_transmission_roughness;
                _9184 = _7365.specular_and_specular_tint;
                _9185 = _7365.clearcoat_and_clearcoat_roughness;
                _9186 = _7365.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_9176 & 2u) != 0u)
                {
                    _7419 = 1.0f / mix_val;
                }
                else
                {
                    _7419 = 1.0f;
                }
                mix_weight *= _7419;
                material_t _7431;
                [unroll]
                for (int _82ident = 0; _82ident < 5; _82ident++)
                {
                    _7431.textures[_82ident] = _4453.Load(_82ident * 4 + _9579 * 76 + 0);
                }
                [unroll]
                for (int _83ident = 0; _83ident < 3; _83ident++)
                {
                    _7431.base_color[_83ident] = asfloat(_4453.Load(_83ident * 4 + _9579 * 76 + 20));
                }
                _7431.flags = _4453.Load(_9579 * 76 + 32);
                _7431.type = _4453.Load(_9579 * 76 + 36);
                _7431.tangent_rotation_or_strength = asfloat(_4453.Load(_9579 * 76 + 40));
                _7431.roughness_and_anisotropic = _4453.Load(_9579 * 76 + 44);
                _7431.ior = asfloat(_4453.Load(_9579 * 76 + 48));
                _7431.sheen_and_sheen_tint = _4453.Load(_9579 * 76 + 52);
                _7431.tint_and_metallic = _4453.Load(_9579 * 76 + 56);
                _7431.transmission_and_transmission_roughness = _4453.Load(_9579 * 76 + 60);
                _7431.specular_and_specular_tint = _4453.Load(_9579 * 76 + 64);
                _7431.clearcoat_and_clearcoat_roughness = _4453.Load(_9579 * 76 + 68);
                _7431.normal_map_strength_unorm = _4453.Load(_9579 * 76 + 72);
                _9575 = _7431.textures[0];
                _9576 = _7431.textures[1];
                _9577 = _7431.textures[2];
                _9578 = _7431.textures[3];
                _9579 = _7431.textures[4];
                _9580 = _7431.base_color[0];
                _9581 = _7431.base_color[1];
                _9582 = _7431.base_color[2];
                _9176 = _7431.flags;
                _9177 = _7431.type;
                _9178 = _7431.tangent_rotation_or_strength;
                _9179 = _7431.roughness_and_anisotropic;
                _9180 = _7431.ior;
                _9181 = _7431.sheen_and_sheen_tint;
                _9182 = _7431.tint_and_metallic;
                _9183 = _7431.transmission_and_transmission_roughness;
                _9184 = _7431.specular_and_specular_tint;
                _9185 = _7431.clearcoat_and_clearcoat_roughness;
                _9186 = _7431.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_9575 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_9575, _6981, 0).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_9575 & 33554432u) != 0u)
            {
                float3 _10147 = normals;
                _10147.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _10147;
            }
            float3 _7513 = _9115;
            _9115 = normalize(((_9113 * normals.x) + (_7513 * normals.z)) + (_9114 * normals.y));
            if ((_9186 & 65535u) != 65535u)
            {
                _9115 = normalize(_7513 + ((_9115 - _7513) * clamp(float(_9186 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_18 = _9116;
            float3 param_19 = -_6604;
            float3 param_20 = _9115;
            _9115 = ensure_valid_reflection(param_18, param_19, param_20);
        }
        float3 _7579 = ((_6900 * _6923) + (_6908 * inter.u)) + (_6916 * inter.v);
        float3 _7586 = float3(-_7579.z, 0.0f, _7579.x);
        float3 tangent = _7586;
        float3 param_21 = _7586;
        float4x4 param_22 = _6750.inv_xform;
        float3 _7592 = TransformNormal(param_21, param_22);
        tangent = _7592;
        float3 _7596 = cross(_7592, _9115);
        if (dot(_7596, _7596) == 0.0f)
        {
            float3 param_23 = _7579;
            float4x4 param_24 = _6750.inv_xform;
            tangent = TransformNormal(param_23, param_24);
        }
        if (_9178 != 0.0f)
        {
            float3 param_25 = tangent;
            float3 param_26 = _9115;
            float param_27 = _9178;
            tangent = rotate_around_axis(param_25, param_26, param_27);
        }
        float3 _7629 = normalize(cross(tangent, _9115));
        _9114 = _7629;
        _9113 = cross(_9115, _7629);
        float3 _9274 = 0.0f.xxx;
        float3 _9273 = 0.0f.xxx;
        float _9278 = 0.0f;
        float _9276 = 0.0f;
        float _9277 = 1.0f;
        bool _7645 = _3325_g_params.li_count != 0;
        bool _7651;
        if (_7645)
        {
            _7651 = _9177 != 3u;
        }
        else
        {
            _7651 = _7645;
        }
        float3 _9275;
        bool _9279;
        bool _9280;
        if (_7651)
        {
            float3 param_28 = _6640;
            float3 param_29 = _9113;
            float3 param_30 = _9114;
            float3 param_31 = _9115;
            int param_32 = _7272;
            float2 param_33 = float2(_7222, _7229);
            light_sample_t _9289 = { _9273, _9274, _9275, _9276, _9277, _9278, _9279, _9280 };
            light_sample_t param_34 = _9289;
            SampleLightSource(param_28, param_29, param_30, param_31, param_32, param_33, param_34);
            _9273 = param_34.col;
            _9274 = param_34.L;
            _9275 = param_34.lp;
            _9276 = param_34.area;
            _9277 = param_34.dist_mul;
            _9278 = param_34.pdf;
            _9279 = param_34.cast_shadow;
            _9280 = param_34.from_env;
        }
        float _7679 = dot(_9115, _9274);
        float3 base_color = float3(_9580, _9581, _9582);
        [branch]
        if (_9576 != 4294967295u)
        {
            base_color *= SampleBilinear(_9576, _6981, int(get_texture_lod(texSize(_9576), _7215)), true, true).xyz;
        }
        float3 tint_color = 0.0f.xxx;
        float _7712 = lum(base_color);
        [flatten]
        if (_7712 > 0.0f)
        {
            tint_color = base_color / _7712.xxx;
        }
        float roughness = clamp(float(_9179 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_9577 != 4294967295u)
        {
            roughness *= SampleBilinear(_9577, _6981, int(get_texture_lod(texSize(_9577), _7215)), false, true).x;
        }
        float _7757 = frac(asfloat(_3309.Load((_7272 + 1) * 4 + 0)) + _7222);
        float _7766 = frac(asfloat(_3309.Load((_7272 + 2) * 4 + 0)) + _7229);
        float _9702 = 0.0f;
        float _9701 = 0.0f;
        float _9700 = 0.0f;
        float _9338[4];
        [unroll]
        for (int i = 0; i < 4; )
        {
            _9338[i] = ray.ior[i];
            i++;
            continue;
        }
        float _9339 = _7205;
        float _9340 = ray.cone_spread;
        int _9341 = ray.xy;
        float _9336 = 0.0f;
        float _9807 = 0.0f;
        float _9806 = 0.0f;
        float _9805 = 0.0f;
        int _9443 = ray.depth;
        int _9447 = ray.xy;
        int _9342;
        float _9445;
        float _9630;
        float _9631;
        float _9632;
        float _9665;
        float _9666;
        float _9667;
        float _9735;
        float _9736;
        float _9737;
        float _9770;
        float _9771;
        float _9772;
        [branch]
        if (_9177 == 0u)
        {
            [branch]
            if ((_9278 > 0.0f) && (_7679 > 0.0f))
            {
                light_sample_t _9306 = { _9273, _9274, _9275, _9276, _9277, _9278, _9279, _9280 };
                surface_t _9124 = { _6640, _9113, _9114, _9115, _9116, _6981 };
                float _9811[3] = { _9805, _9806, _9807 };
                float _9776[3] = { _9770, _9771, _9772 };
                float _9741[3] = { _9735, _9736, _9737 };
                shadow_ray_t _9457 = { _9741, _9443, _9776, _9445, _9811, _9447 };
                shadow_ray_t param_35 = _9457;
                float3 _7826 = Evaluate_DiffuseNode(_9306, ray, _9124, base_color, roughness, mix_weight, param_35);
                _9735 = param_35.o[0];
                _9736 = param_35.o[1];
                _9737 = param_35.o[2];
                _9443 = param_35.depth;
                _9770 = param_35.d[0];
                _9771 = param_35.d[1];
                _9772 = param_35.d[2];
                _9445 = param_35.dist;
                _9805 = param_35.c[0];
                _9806 = param_35.c[1];
                _9807 = param_35.c[2];
                _9447 = param_35.xy;
                col += _7826;
            }
            bool _7833 = _7243 < _3325_g_params.max_diff_depth;
            bool _7840;
            if (_7833)
            {
                _7840 = _7264 < _3325_g_params.max_total_depth;
            }
            else
            {
                _7840 = _7833;
            }
            [branch]
            if (_7840)
            {
                surface_t _9131 = { _6640, _9113, _9114, _9115, _9116, _6981 };
                float _9706[3] = { _9700, _9701, _9702 };
                float _9671[3] = { _9665, _9666, _9667 };
                float _9636[3] = { _9630, _9631, _9632 };
                ray_data_t _9356 = { _9636, _9671, _9336, _9706, _9338, _9339, _9340, _9341, _9342 };
                ray_data_t param_36 = _9356;
                Sample_DiffuseNode(ray, _9131, base_color, roughness, _7757, _7766, mix_weight, param_36);
                _9630 = param_36.o[0];
                _9631 = param_36.o[1];
                _9632 = param_36.o[2];
                _9665 = param_36.d[0];
                _9666 = param_36.d[1];
                _9667 = param_36.d[2];
                _9336 = param_36.pdf;
                _9700 = param_36.c[0];
                _9701 = param_36.c[1];
                _9702 = param_36.c[2];
                _9338 = param_36.ior;
                _9339 = param_36.cone_width;
                _9340 = param_36.cone_spread;
                _9341 = param_36.xy;
                _9342 = param_36.depth;
            }
        }
        else
        {
            [branch]
            if (_9177 == 1u)
            {
                float param_37 = 1.0f;
                float param_38 = 1.5f;
                float _7864 = fresnel_dielectric_cos(param_37, param_38);
                float _7868 = roughness * roughness;
                bool _7871 = _9278 > 0.0f;
                bool _7878;
                if (_7871)
                {
                    _7878 = (_7868 * _7868) >= 1.0000000116860974230803549289703e-07f;
                }
                else
                {
                    _7878 = _7871;
                }
                [branch]
                if (_7878 && (_7679 > 0.0f))
                {
                    light_sample_t _9315 = { _9273, _9274, _9275, _9276, _9277, _9278, _9279, _9280 };
                    surface_t _9138 = { _6640, _9113, _9114, _9115, _9116, _6981 };
                    float _9818[3] = { _9805, _9806, _9807 };
                    float _9783[3] = { _9770, _9771, _9772 };
                    float _9748[3] = { _9735, _9736, _9737 };
                    shadow_ray_t _9470 = { _9748, _9443, _9783, _9445, _9818, _9447 };
                    shadow_ray_t param_39 = _9470;
                    float3 _7893 = Evaluate_GlossyNode(_9315, ray, _9138, base_color, roughness, 1.5f, _7864, mix_weight, param_39);
                    _9735 = param_39.o[0];
                    _9736 = param_39.o[1];
                    _9737 = param_39.o[2];
                    _9443 = param_39.depth;
                    _9770 = param_39.d[0];
                    _9771 = param_39.d[1];
                    _9772 = param_39.d[2];
                    _9445 = param_39.dist;
                    _9805 = param_39.c[0];
                    _9806 = param_39.c[1];
                    _9807 = param_39.c[2];
                    _9447 = param_39.xy;
                    col += _7893;
                }
                bool _7900 = _7248 < _3325_g_params.max_spec_depth;
                bool _7907;
                if (_7900)
                {
                    _7907 = _7264 < _3325_g_params.max_total_depth;
                }
                else
                {
                    _7907 = _7900;
                }
                [branch]
                if (_7907)
                {
                    surface_t _9145 = { _6640, _9113, _9114, _9115, _9116, _6981 };
                    float _9713[3] = { _9700, _9701, _9702 };
                    float _9678[3] = { _9665, _9666, _9667 };
                    float _9643[3] = { _9630, _9631, _9632 };
                    ray_data_t _9375 = { _9643, _9678, _9336, _9713, _9338, _9339, _9340, _9341, _9342 };
                    ray_data_t param_40 = _9375;
                    Sample_GlossyNode(ray, _9145, base_color, roughness, 1.5f, _7864, _7757, _7766, mix_weight, param_40);
                    _9630 = param_40.o[0];
                    _9631 = param_40.o[1];
                    _9632 = param_40.o[2];
                    _9665 = param_40.d[0];
                    _9666 = param_40.d[1];
                    _9667 = param_40.d[2];
                    _9336 = param_40.pdf;
                    _9700 = param_40.c[0];
                    _9701 = param_40.c[1];
                    _9702 = param_40.c[2];
                    _9338 = param_40.ior;
                    _9339 = param_40.cone_width;
                    _9340 = param_40.cone_spread;
                    _9341 = param_40.xy;
                    _9342 = param_40.depth;
                }
            }
            else
            {
                [branch]
                if (_9177 == 2u)
                {
                    float _7931 = roughness * roughness;
                    bool _7934 = _9278 > 0.0f;
                    bool _7941;
                    if (_7934)
                    {
                        _7941 = (_7931 * _7931) >= 1.0000000116860974230803549289703e-07f;
                    }
                    else
                    {
                        _7941 = _7934;
                    }
                    [branch]
                    if (_7941 && (_7679 < 0.0f))
                    {
                        float _7949;
                        if (_6673)
                        {
                            _7949 = _9180 / _7238;
                        }
                        else
                        {
                            _7949 = _7238 / _9180;
                        }
                        light_sample_t _9324 = { _9273, _9274, _9275, _9276, _9277, _9278, _9279, _9280 };
                        surface_t _9152 = { _6640, _9113, _9114, _9115, _9116, _6981 };
                        float _9825[3] = { _9805, _9806, _9807 };
                        float _9790[3] = { _9770, _9771, _9772 };
                        float _9755[3] = { _9735, _9736, _9737 };
                        shadow_ray_t _9483 = { _9755, _9443, _9790, _9445, _9825, _9447 };
                        shadow_ray_t param_41 = _9483;
                        float3 _7971 = Evaluate_RefractiveNode(_9324, ray, _9152, base_color, _7931, _7949, mix_weight, param_41);
                        _9735 = param_41.o[0];
                        _9736 = param_41.o[1];
                        _9737 = param_41.o[2];
                        _9443 = param_41.depth;
                        _9770 = param_41.d[0];
                        _9771 = param_41.d[1];
                        _9772 = param_41.d[2];
                        _9445 = param_41.dist;
                        _9805 = param_41.c[0];
                        _9806 = param_41.c[1];
                        _9807 = param_41.c[2];
                        _9447 = param_41.xy;
                        col += _7971;
                    }
                    bool _7978 = _7253 < _3325_g_params.max_refr_depth;
                    bool _7985;
                    if (_7978)
                    {
                        _7985 = _7264 < _3325_g_params.max_total_depth;
                    }
                    else
                    {
                        _7985 = _7978;
                    }
                    [branch]
                    if (_7985)
                    {
                        surface_t _9159 = { _6640, _9113, _9114, _9115, _9116, _6981 };
                        float _9720[3] = { _9700, _9701, _9702 };
                        float _9685[3] = { _9665, _9666, _9667 };
                        float _9650[3] = { _9630, _9631, _9632 };
                        ray_data_t _9394 = { _9650, _9685, _9336, _9720, _9338, _9339, _9340, _9341, _9342 };
                        ray_data_t param_42 = _9394;
                        Sample_RefractiveNode(ray, _9159, base_color, roughness, _6673, _9180, _7238, _7757, _7766, mix_weight, param_42);
                        _9630 = param_42.o[0];
                        _9631 = param_42.o[1];
                        _9632 = param_42.o[2];
                        _9665 = param_42.d[0];
                        _9666 = param_42.d[1];
                        _9667 = param_42.d[2];
                        _9336 = param_42.pdf;
                        _9700 = param_42.c[0];
                        _9701 = param_42.c[1];
                        _9702 = param_42.c[2];
                        _9338 = param_42.ior;
                        _9339 = param_42.cone_width;
                        _9340 = param_42.cone_spread;
                        _9341 = param_42.xy;
                        _9342 = param_42.depth;
                    }
                }
                else
                {
                    [branch]
                    if (_9177 == 3u)
                    {
                        float mis_weight = 1.0f;
                        [branch]
                        if ((_9176 & 1u) != 0u)
                        {
                            float3 _8055 = mul(float4(_6997, 0.0f), _6750.xform).xyz;
                            float _8058 = length(_8055);
                            float _8070 = abs(dot(_6604, _8055 / _8058.xxx));
                            if (_8070 > 0.0f)
                            {
                                float param_43 = ray.pdf;
                                float param_44 = (inter.t * inter.t) / ((0.5f * _8058) * _8070);
                                mis_weight = power_heuristic(param_43, param_44);
                            }
                        }
                        col += (base_color * ((mix_weight * mis_weight) * _9178));
                    }
                    else
                    {
                        [branch]
                        if (_9177 == 6u)
                        {
                            float metallic = clamp(float((_9182 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_9578 != 4294967295u)
                            {
                                metallic *= SampleBilinear(_9578, _6981, int(get_texture_lod(texSize(_9578), _7215))).x;
                            }
                            float specular = clamp(float(_9184 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            [branch]
                            if (_9579 != 4294967295u)
                            {
                                specular *= SampleBilinear(_9579, _6981, int(get_texture_lod(texSize(_9579), _7215))).x;
                            }
                            float _8187 = clamp(float(_9185 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8195 = clamp(float((_9185 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8203 = 2.0f * clamp(float(_9181 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float3 _8221 = lerp(1.0f.xxx, tint_color, clamp(float((_9181 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _8203;
                            float3 _8241 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_9184 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                            float _8250 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                            float param_45 = 1.0f;
                            float param_46 = _8250;
                            float _8256 = fresnel_dielectric_cos(param_45, param_46);
                            float _8264 = clamp(float((_9179 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                            float _8275 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _8187))) - 1.0f;
                            float param_47 = 1.0f;
                            float param_48 = _8275;
                            float _8281 = fresnel_dielectric_cos(param_47, param_48);
                            float _8296 = mad(roughness - 1.0f, 1.0f - clamp(float((_9183 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                            float _8302;
                            if (_6673)
                            {
                                _8302 = _9180 / _7238;
                            }
                            else
                            {
                                _8302 = _7238 / _9180;
                            }
                            float param_49 = dot(_6604, _9115);
                            float param_50 = 1.0f / _8302;
                            float _8325 = fresnel_dielectric_cos(param_49, param_50);
                            float param_51 = dot(_6604, _9115);
                            float param_52 = _8250;
                            lobe_weights_t _8364 = get_lobe_weights(lerp(_7712, 1.0f, _8203), lum(lerp(_8241, 1.0f.xxx, ((fresnel_dielectric_cos(param_51, param_52) - _8256) / (1.0f - _8256)).xxx)), specular, metallic, clamp(float(_9183 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _8187);
                            [branch]
                            if (_9278 > 0.0f)
                            {
                                light_sample_t _9333 = { _9273, _9274, _9275, _9276, _9277, _9278, _9279, _9280 };
                                surface_t _9166 = { _6640, _9113, _9114, _9115, _9116, _6981 };
                                diff_params_t _9525 = { base_color, _8221, roughness };
                                spec_params_t _9540 = { _8241, roughness, _8250, _8256, _8264 };
                                clearcoat_params_t _9553 = { _8195, _8275, _8281 };
                                transmission_params_t _9568 = { _8296, _9180, _8302, _8325, _6673 };
                                float _9832[3] = { _9805, _9806, _9807 };
                                float _9797[3] = { _9770, _9771, _9772 };
                                float _9762[3] = { _9735, _9736, _9737 };
                                shadow_ray_t _9496 = { _9762, _9443, _9797, _9445, _9832, _9447 };
                                shadow_ray_t param_53 = _9496;
                                float3 _8383 = Evaluate_PrincipledNode(_9333, ray, _9166, _8364, _9525, _9540, _9553, _9568, metallic, _7679, mix_weight, param_53);
                                _9735 = param_53.o[0];
                                _9736 = param_53.o[1];
                                _9737 = param_53.o[2];
                                _9443 = param_53.depth;
                                _9770 = param_53.d[0];
                                _9771 = param_53.d[1];
                                _9772 = param_53.d[2];
                                _9445 = param_53.dist;
                                _9805 = param_53.c[0];
                                _9806 = param_53.c[1];
                                _9807 = param_53.c[2];
                                _9447 = param_53.xy;
                                col += _8383;
                            }
                            surface_t _9173 = { _6640, _9113, _9114, _9115, _9116, _6981 };
                            diff_params_t _9529 = { base_color, _8221, roughness };
                            spec_params_t _9546 = { _8241, roughness, _8250, _8256, _8264 };
                            clearcoat_params_t _9557 = { _8195, _8275, _8281 };
                            transmission_params_t _9574 = { _8296, _9180, _8302, _8325, _6673 };
                            float param_54 = mix_rand;
                            float _9727[3] = { _9700, _9701, _9702 };
                            float _9692[3] = { _9665, _9666, _9667 };
                            float _9657[3] = { _9630, _9631, _9632 };
                            ray_data_t _9413 = { _9657, _9692, _9336, _9727, _9338, _9339, _9340, _9341, _9342 };
                            ray_data_t param_55 = _9413;
                            Sample_PrincipledNode(ray, _9173, _8364, _9529, _9546, _9557, _9574, metallic, _7757, _7766, param_54, mix_weight, param_55);
                            _9630 = param_55.o[0];
                            _9631 = param_55.o[1];
                            _9632 = param_55.o[2];
                            _9665 = param_55.d[0];
                            _9666 = param_55.d[1];
                            _9667 = param_55.d[2];
                            _9336 = param_55.pdf;
                            _9700 = param_55.c[0];
                            _9701 = param_55.c[1];
                            _9702 = param_55.c[2];
                            _9338 = param_55.ior;
                            _9339 = param_55.cone_width;
                            _9340 = param_55.cone_spread;
                            _9341 = param_55.xy;
                            _9342 = param_55.depth;
                        }
                    }
                }
            }
        }
        float _8417 = max(_9700, max(_9701, _9702));
        float _8429;
        if (_7264 > _3325_g_params.min_total_depth)
        {
            _8429 = max(0.0500000007450580596923828125f, 1.0f - _8417);
        }
        else
        {
            _8429 = 0.0f;
        }
        bool _8443 = (frac(asfloat(_3309.Load((_7272 + 6) * 4 + 0)) + _7222) >= _8429) && (_8417 > 0.0f);
        bool _8449;
        if (_8443)
        {
            _8449 = _9336 > 0.0f;
        }
        else
        {
            _8449 = _8443;
        }
        [branch]
        if (_8449)
        {
            float _8453 = _9336;
            float _8454 = min(_8453, 1000000.0f);
            _9336 = _8454;
            float _8457 = 1.0f - _8429;
            float _8459 = _9700;
            float _8460 = _8459 / _8457;
            _9700 = _8460;
            float _8465 = _9701;
            float _8466 = _8465 / _8457;
            _9701 = _8466;
            float _8471 = _9702;
            float _8472 = _8471 / _8457;
            _9702 = _8472;
            uint _8480;
            _8478.InterlockedAdd(0, 1u, _8480);
            _8489.Store(_8480 * 72 + 0, asuint(_9630));
            _8489.Store(_8480 * 72 + 4, asuint(_9631));
            _8489.Store(_8480 * 72 + 8, asuint(_9632));
            _8489.Store(_8480 * 72 + 12, asuint(_9665));
            _8489.Store(_8480 * 72 + 16, asuint(_9666));
            _8489.Store(_8480 * 72 + 20, asuint(_9667));
            _8489.Store(_8480 * 72 + 24, asuint(_8454));
            _8489.Store(_8480 * 72 + 28, asuint(_8460));
            _8489.Store(_8480 * 72 + 32, asuint(_8466));
            _8489.Store(_8480 * 72 + 36, asuint(_8472));
            _8489.Store(_8480 * 72 + 40, asuint(_9338[0]));
            _8489.Store(_8480 * 72 + 44, asuint(_9338[1]));
            _8489.Store(_8480 * 72 + 48, asuint(_9338[2]));
            _8489.Store(_8480 * 72 + 52, asuint(_9338[3]));
            _8489.Store(_8480 * 72 + 56, asuint(_9339));
            _8489.Store(_8480 * 72 + 60, asuint(_9340));
            _8489.Store(_8480 * 72 + 64, uint(_9341));
            _8489.Store(_8480 * 72 + 68, uint(_9342));
        }
        [branch]
        if (max(_9805, max(_9806, _9807)) > 0.0f)
        {
            float3 _8566 = _9275 - float3(_9735, _9736, _9737);
            float _8569 = length(_8566);
            float3 _8573 = _8566 / _8569.xxx;
            float sh_dist = _8569 * _9277;
            if (_9280)
            {
                sh_dist = -sh_dist;
            }
            float _8585 = _8573.x;
            _9770 = _8585;
            float _8588 = _8573.y;
            _9771 = _8588;
            float _8591 = _8573.z;
            _9772 = _8591;
            _9445 = sh_dist;
            uint _8597;
            _8478.InterlockedAdd(8, 1u, _8597);
            _8605.Store(_8597 * 48 + 0, asuint(_9735));
            _8605.Store(_8597 * 48 + 4, asuint(_9736));
            _8605.Store(_8597 * 48 + 8, asuint(_9737));
            _8605.Store(_8597 * 48 + 12, uint(_9443));
            _8605.Store(_8597 * 48 + 16, asuint(_8585));
            _8605.Store(_8597 * 48 + 20, asuint(_8588));
            _8605.Store(_8597 * 48 + 24, asuint(_8591));
            _8605.Store(_8597 * 48 + 28, asuint(sh_dist));
            _8605.Store(_8597 * 48 + 32, asuint(_9805));
            _8605.Store(_8597 * 48 + 36, asuint(_9806));
            _8605.Store(_8597 * 48 + 40, asuint(_9807));
            _8605.Store(_8597 * 48 + 44, uint(_9447));
        }
        _8815 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _8815;
}

void comp_main()
{
    do
    {
        int _8671 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_8671) >= _8478.Load(4))
        {
            break;
        }
        int _8687 = int(_8684.Load(_8671 * 72 + 64));
        int _8694 = int(_8684.Load(_8671 * 72 + 64));
        hit_data_t _8705;
        _8705.mask = int(_8701.Load(_8671 * 24 + 0));
        _8705.obj_index = int(_8701.Load(_8671 * 24 + 4));
        _8705.prim_index = int(_8701.Load(_8671 * 24 + 8));
        _8705.t = asfloat(_8701.Load(_8671 * 24 + 12));
        _8705.u = asfloat(_8701.Load(_8671 * 24 + 16));
        _8705.v = asfloat(_8701.Load(_8671 * 24 + 20));
        ray_data_t _8721;
        [unroll]
        for (int _84ident = 0; _84ident < 3; _84ident++)
        {
            _8721.o[_84ident] = asfloat(_8684.Load(_84ident * 4 + _8671 * 72 + 0));
        }
        [unroll]
        for (int _85ident = 0; _85ident < 3; _85ident++)
        {
            _8721.d[_85ident] = asfloat(_8684.Load(_85ident * 4 + _8671 * 72 + 12));
        }
        _8721.pdf = asfloat(_8684.Load(_8671 * 72 + 24));
        [unroll]
        for (int _86ident = 0; _86ident < 3; _86ident++)
        {
            _8721.c[_86ident] = asfloat(_8684.Load(_86ident * 4 + _8671 * 72 + 28));
        }
        [unroll]
        for (int _87ident = 0; _87ident < 4; _87ident++)
        {
            _8721.ior[_87ident] = asfloat(_8684.Load(_87ident * 4 + _8671 * 72 + 40));
        }
        _8721.cone_width = asfloat(_8684.Load(_8671 * 72 + 56));
        _8721.cone_spread = asfloat(_8684.Load(_8671 * 72 + 60));
        _8721.xy = int(_8684.Load(_8671 * 72 + 64));
        _8721.depth = int(_8684.Load(_8671 * 72 + 68));
        hit_data_t _8909 = { _8705.mask, _8705.obj_index, _8705.prim_index, _8705.t, _8705.u, _8705.v };
        hit_data_t param = _8909;
        float _8958[4] = { _8721.ior[0], _8721.ior[1], _8721.ior[2], _8721.ior[3] };
        float _8949[3] = { _8721.c[0], _8721.c[1], _8721.c[2] };
        float _8942[3] = { _8721.d[0], _8721.d[1], _8721.d[2] };
        float _8935[3] = { _8721.o[0], _8721.o[1], _8721.o[2] };
        ray_data_t _8928 = { _8935, _8942, _8721.pdf, _8949, _8958, _8721.cone_width, _8721.cone_spread, _8721.xy, _8721.depth };
        ray_data_t param_1 = _8928;
        float3 _8771 = ShadeSurface(param, param_1);
        int2 _8778 = int2((_8687 >> 16) & 65535, _8694 & 65535);
        g_out_img[_8778] = float4(_8771 + g_out_img[_8778].xyz, 1.0f);
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
