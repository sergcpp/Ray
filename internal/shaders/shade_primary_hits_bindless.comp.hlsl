struct light_sample_t
{
    float3 col;
    float3 L;
    float area;
    float dist;
    float pdf;
    float cast_shadow;
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

struct ray_data_t
{
    float o[3];
    float d[3];
    float pdf;
    float c[3];
    float cone_width;
    float cone_spread;
    int xy;
    int ray_depth;
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
    int termination_start_depth;
    float env_rotation;
    int env_qtree_levels;
    float4 env_col;
    float4 back_col;
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
    float int_ior;
    float ext_ior;
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

struct shadow_ray_t
{
    float o[3];
    float d[3];
    float dist;
    float c[3];
    int xy;
};

struct tri_accel_t
{
    float4 n_plane;
    float4 u_plane;
    float4 v_plane;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

ByteAddressBuffer _2996 : register(t15, space0);
ByteAddressBuffer _3033 : register(t6, space0);
ByteAddressBuffer _3037 : register(t7, space0);
ByteAddressBuffer _3808 : register(t11, space0);
ByteAddressBuffer _3833 : register(t13, space0);
ByteAddressBuffer _3837 : register(t14, space0);
ByteAddressBuffer _4148 : register(t10, space0);
ByteAddressBuffer _4152 : register(t9, space0);
ByteAddressBuffer _4796 : register(t12, space0);
RWByteAddressBuffer _5871 : register(u3, space0);
RWByteAddressBuffer _5881 : register(u2, space0);
RWByteAddressBuffer _8048 : register(u1, space0);
ByteAddressBuffer _8152 : register(t4, space0);
ByteAddressBuffer _8173 : register(t5, space0);
ByteAddressBuffer _8249 : register(t8, space0);
cbuffer UniformParams
{
    Params _3001_g_params : packoffset(c0);
};

Texture2D<float4> g_textures[] : register(t0, space1);
SamplerState _g_textures_sampler[] : register(s0, space1);
Texture2D<float4> g_env_qtree : register(t16, space0);
SamplerState _g_env_qtree_sampler : register(s16, space0);
RWTexture2D<float4> g_out_img : register(u0, space0);

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
    float _912 = sqrt(mad(dir.x, dir.x, dir.z * dir.z));
    float _917;
    if (_912 > 1.0000000116860974230803549289703e-07f)
    {
        _917 = clamp(dir.x / _912, -1.0f, 1.0f);
    }
    else
    {
        _917 = 0.0f;
    }
    float _927 = acos(_917) + y_rotation;
    float phi = _927;
    if (_927 < 0.0f)
    {
        phi += 6.283185482025146484375f;
    }
    if (phi > 6.283185482025146484375f)
    {
        phi -= 6.283185482025146484375f;
    }
    float u = frac(phi * 0.15915493667125701904296875f);
    [flatten]
    if (dir.z < 0.0f)
    {
        u = 1.0f - u;
    }
    uint _952 = index & 16777215u;
    uint _958_dummy_parameter;
    float2 _965 = float2(u, acos(clamp(dir.y, -1.0f, 1.0f)) * 0.3183098733425140380859375f) * float2(int2(spvTextureSize(g_textures[_952], uint(0), _958_dummy_parameter)));
    uint _968 = _952;
    int2 _972 = int2(_965);
    float2 _1007 = frac(_965);
    float4 param = g_textures[NonUniformResourceIndex(_968)].Load(int3(_972, 0), int2(0, 0));
    float4 param_1 = g_textures[NonUniformResourceIndex(_968)].Load(int3(_972, 0), int2(1, 0));
    float4 param_2 = g_textures[NonUniformResourceIndex(_968)].Load(int3(_972, 0), int2(0, 1));
    float4 param_3 = g_textures[NonUniformResourceIndex(_968)].Load(int3(_972, 0), int2(1, 1));
    float _1027 = _1007.x;
    float _1032 = 1.0f - _1027;
    float _1048 = _1007.y;
    return (((rgbe_to_rgb(param_3) * _1027) + (rgbe_to_rgb(param_2) * _1032)) * _1048) + (((rgbe_to_rgb(param_1) * _1027) + (rgbe_to_rgb(param) * _1032)) * (1.0f - _1048));
}

float2 DirToCanonical(float3 d, float y_rotation)
{
    float _574 = (-atan2(d.z, d.x)) + y_rotation;
    float phi = _574;
    if (_574 < 0.0f)
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
    float2 _600 = DirToCanonical(L, y_rotation);
    float factor = 1.0f;
    while (lod >= 0)
    {
        int2 _620 = clamp(int2(_600 * float(res)), int2(0, 0), (res - 1).xx);
        float4 quad = qtree_tex.Load(int3(_620 / int2(2, 2), lod));
        float _655 = ((quad.x + quad.y) + quad.z) + quad.w;
        if (_655 <= 0.0f)
        {
            break;
        }
        factor *= ((4.0f * quad[(0 | ((_620.x & 1) << 0)) | ((_620.y & 1) << 1)]) / _655);
        lod--;
        res *= 2;
    }
    return factor * 0.079577468335628509521484375f;
}

float power_heuristic(float a, float b)
{
    float _1061 = a * a;
    return _1061 / mad(b, b, _1061);
}

float3 TransformNormal(float3 n, float4x4 inv_xform)
{
    return mul(float4(n, 0.0f), transpose(inv_xform)).xyz;
}

int hash(int x)
{
    uint _351 = uint(x);
    uint _358 = ((_351 >> uint(16)) ^ _351) * 73244475u;
    uint _363 = ((_358 >> uint(16)) ^ _358) * 73244475u;
    return int((_363 >> uint(16)) ^ _363);
}

float construct_float(inout uint m)
{
    m &= 8388607u;
    m |= 1065353216u;
    return asfloat(m) - 1.0f;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _450 = mad(col.z, 31.875f, 1.0f);
    float _460 = (col.x - 0.501960813999176025390625f) / _450;
    float _466 = (col.y - 0.501960813999176025390625f) / _450;
    return float3((col.w + _460) - _466, col.w + _466, (col.w - _460) - _466);
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
    uint _850 = index & 16777215u;
    float4 res = g_textures[NonUniformResourceIndex(_850)].SampleLevel(_g_textures_sampler[NonUniformResourceIndex(_850)], uvs, float(lod));
    bool _860;
    if (maybe_YCoCg)
    {
        _860 = (index & 67108864u) != 0u;
    }
    else
    {
        _860 = maybe_YCoCg;
    }
    if (_860)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _878;
    if (maybe_SRGB)
    {
        _878 = (index & 16777216u) != 0u;
    }
    else
    {
        _878 = maybe_SRGB;
    }
    if (_878)
    {
        float3 param_1 = res.xyz;
        float3 _884 = srgb_to_rgb(param_1);
        float4 _9002 = res;
        _9002.x = _884.x;
        float4 _9004 = _9002;
        _9004.y = _884.y;
        float4 _9006 = _9004;
        _9006.z = _884.z;
        res = _9006;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod)
{
    return SampleBilinear(index, uvs, lod, false, false);
}

float fresnel_dielectric_cos(float cosi, float eta)
{
    float _1093 = abs(cosi);
    float _1102 = mad(_1093, _1093, mad(eta, eta, -1.0f));
    float g = _1102;
    float result;
    if (_1102 > 0.0f)
    {
        float _1107 = g;
        float _1108 = sqrt(_1107);
        g = _1108;
        float _1112 = _1108 - _1093;
        float _1115 = _1108 + _1093;
        float _1116 = _1112 / _1115;
        float _1130 = mad(_1093, _1115, -1.0f) / mad(_1093, _1112, 1.0f);
        result = ((0.5f * _1116) * _1116) * mad(_1130, _1130, 1.0f);
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
    float3 _8261;
    do
    {
        float _1166 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1166)
        {
            _8261 = N;
            break;
        }
        float3 _1186 = normalize(N - (Ng * dot(N, Ng)));
        float _1190 = dot(I, _1186);
        float _1194 = dot(I, Ng);
        float _1206 = mad(_1190, _1190, _1194 * _1194);
        float param = (_1190 * _1190) * mad(-_1166, _1166, _1206);
        float _1216 = safe_sqrtf(param);
        float _1222 = mad(_1194, _1166, _1206);
        float _1225 = 0.5f / _1206;
        float _1230 = _1216 + _1222;
        float _1231 = _1225 * _1230;
        float _1237 = (-_1216) + _1222;
        float _1238 = _1225 * _1237;
        bool _1246 = (_1231 > 9.9999997473787516355514526367188e-06f) && (_1231 <= 1.000010013580322265625f);
        bool valid1 = _1246;
        bool _1252 = (_1238 > 9.9999997473787516355514526367188e-06f) && (_1238 <= 1.000010013580322265625f);
        bool valid2 = _1252;
        float2 N_new;
        if (_1246 && _1252)
        {
            float _9308 = (-0.5f) / _1206;
            float param_1 = mad(_9308, _1230, 1.0f);
            float _1262 = safe_sqrtf(param_1);
            float param_2 = _1231;
            float _1265 = safe_sqrtf(param_2);
            float2 _1266 = float2(_1262, _1265);
            float param_3 = mad(_9308, _1237, 1.0f);
            float _1271 = safe_sqrtf(param_3);
            float param_4 = _1238;
            float _1274 = safe_sqrtf(param_4);
            float2 _1275 = float2(_1271, _1274);
            float _9310 = -_1194;
            float _1291 = mad(2.0f * mad(_1262, _1190, _1265 * _1194), _1265, _9310);
            float _1307 = mad(2.0f * mad(_1271, _1190, _1274 * _1194), _1274, _9310);
            bool _1309 = _1291 >= 9.9999997473787516355514526367188e-06f;
            valid1 = _1309;
            bool _1311 = _1307 >= 9.9999997473787516355514526367188e-06f;
            valid2 = _1311;
            if (_1309 && _1311)
            {
                bool2 _1324 = (_1291 < _1307).xx;
                N_new = float2(_1324.x ? _1266.x : _1275.x, _1324.y ? _1266.y : _1275.y);
            }
            else
            {
                bool2 _1332 = (_1291 > _1307).xx;
                N_new = float2(_1332.x ? _1266.x : _1275.x, _1332.y ? _1266.y : _1275.y);
            }
        }
        else
        {
            if (!(valid1 || valid2))
            {
                _8261 = Ng;
                break;
            }
            float _1344 = valid1 ? _1231 : _1238;
            float param_5 = 1.0f - _1344;
            float param_6 = _1344;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _8261 = (_1186 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _8261;
}

float3 rotate_around_axis(float3 p, float3 axis, float angle)
{
    float _1417 = cos(angle);
    float _1420 = sin(angle);
    float _1424 = 1.0f - _1417;
    return float3(mad(mad(_1424 * axis.x, axis.z, axis.y * _1420), p.z, mad(mad(_1424 * axis.x, axis.x, _1417), p.x, mad(_1424 * axis.x, axis.y, -(axis.z * _1420)) * p.y)), mad(mad(_1424 * axis.y, axis.z, -(axis.x * _1420)), p.z, mad(mad(_1424 * axis.x, axis.y, axis.z * _1420), p.x, mad(_1424 * axis.y, axis.y, _1417) * p.y)), mad(mad(_1424 * axis.z, axis.z, _1417), p.z, mad(mad(_1424 * axis.x, axis.z, -(axis.y * _1420)), p.x, mad(_1424 * axis.y, axis.z, axis.x * _1420) * p.y)));
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

float3 MapToCone(float r1, float r2, float3 N, float radius)
{
    float3 _8286;
    do
    {
        float2 _2911 = (float2(r1, r2) * 2.0f) - 1.0f.xx;
        float _2913 = _2911.x;
        bool _2914 = _2913 == 0.0f;
        bool _2920;
        if (_2914)
        {
            _2920 = _2911.y == 0.0f;
        }
        else
        {
            _2920 = _2914;
        }
        if (_2920)
        {
            _8286 = N;
            break;
        }
        float _2929 = _2911.y;
        float r;
        float theta;
        if (abs(_2913) > abs(_2929))
        {
            r = _2913;
            theta = 0.785398185253143310546875f * (_2929 / _2913);
        }
        else
        {
            r = _2929;
            theta = 1.57079637050628662109375f * mad(-0.5f, _2913 / _2929, 1.0f);
        }
        float3 param;
        float3 param_1;
        create_tbn(N, param, param_1);
        _8286 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _8286;
}

float3 CanonicalToDir(float2 p, float y_rotation)
{
    float _524 = mad(2.0f, p.x, -1.0f);
    float _529 = mad(6.283185482025146484375f, p.y, y_rotation);
    float phi = _529;
    if (_529 < 0.0f)
    {
        phi += 6.283185482025146484375f;
    }
    if (phi > 6.283185482025146484375f)
    {
        phi -= 6.283185482025146484375f;
    }
    float _547 = sqrt(mad(-_524, _524, 1.0f));
    return float3(_547 * cos(phi), _524, (-_547) * sin(phi));
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
        float _721 = quad.x + quad.z;
        float partial = _721;
        float _728 = (_721 + quad.y) + quad.w;
        if (_728 <= 0.0f)
        {
            break;
        }
        float _737 = partial / _728;
        float boundary = _737;
        int index = 0;
        if (_sample < _737)
        {
            _sample /= boundary;
            boundary = quad.x / partial;
        }
        else
        {
            float _752 = partial;
            float _753 = _728 - _752;
            partial = _753;
            float2 _8989 = origin;
            _8989.x = origin.x + _step;
            origin = _8989;
            _sample = (_sample - boundary) / (1.0f - boundary);
            boundary = quad.y / _753;
            index |= 1;
        }
        if (_sample < boundary)
        {
            _sample /= boundary;
        }
        else
        {
            float2 _8992 = origin;
            _8992.y = origin.y + _step;
            origin = _8992;
            _sample = (_sample - boundary) / (1.0f - boundary);
            index |= 2;
        }
        factor *= ((4.0f * quad[index]) / _728);
        lod--;
        res *= 2;
        _step *= 0.5f;
    }
    float2 _810 = origin;
    float2 _811 = _810 + (float2(rx, ry) * (2.0f * _step));
    origin = _811;
    return float4(CanonicalToDir(_811, y_rotation), factor * 0.079577468335628509521484375f);
}

void SampleLightSource(float3 P, float2 sample_off, inout light_sample_t ls)
{
    float _3012 = frac(asfloat(_2996.Load((_3001_g_params.hi + 3) * 4 + 0)) + sample_off.x);
    float _3017 = float(_3001_g_params.li_count);
    uint _3024 = min(uint(_3012 * _3017), uint(_3001_g_params.li_count - 1));
    light_t _3044;
    _3044.type_and_param0 = _3033.Load4(_3037.Load(_3024 * 4 + 0) * 64 + 0);
    _3044.param1 = asfloat(_3033.Load4(_3037.Load(_3024 * 4 + 0) * 64 + 16));
    _3044.param2 = asfloat(_3033.Load4(_3037.Load(_3024 * 4 + 0) * 64 + 32));
    _3044.param3 = asfloat(_3033.Load4(_3037.Load(_3024 * 4 + 0) * 64 + 48));
    ls.col = asfloat(_3044.type_and_param0.yzw);
    ls.col *= _3017;
    ls.cast_shadow = float((_3044.type_and_param0.x & 32u) != 0u);
    uint _3079 = _3044.type_and_param0.x & 31u;
    [branch]
    if (_3079 == 0u)
    {
        float _3094 = frac(asfloat(_2996.Load((_3001_g_params.hi + 4) * 4 + 0)) + sample_off.x);
        float3 _3110 = P - _3044.param1.xyz;
        float3 _3117 = _3110 / length(_3110).xxx;
        float _3124 = sqrt(clamp(mad(-_3094, _3094, 1.0f), 0.0f, 1.0f));
        float _3127 = 6.283185482025146484375f * frac(asfloat(_2996.Load((_3001_g_params.hi + 5) * 4 + 0)) + sample_off.y);
        float3 sampled_dir = float3(_3124 * cos(_3127), _3124 * sin(_3127), _3094);
        float3 param;
        float3 param_1;
        create_tbn(_3117, param, param_1);
        float3 _9069 = sampled_dir;
        float3 _3160 = ((param * _9069.x) + (param_1 * _9069.y)) + (_3117 * _9069.z);
        sampled_dir = _3160;
        float3 _3169 = _3044.param1.xyz + (_3160 * _3044.param2.w);
        ls.L = _3169 - P;
        ls.dist = length(ls.L);
        ls.L /= ls.dist.xxx;
        ls.area = _3044.param1.w;
        float _3200 = abs(dot(ls.L, normalize(_3169 - _3044.param1.xyz)));
        [flatten]
        if (_3200 > 0.0f)
        {
            ls.pdf = (ls.dist * ls.dist) / ((0.5f * ls.area) * _3200);
        }
        [branch]
        if (_3044.param3.x > 0.0f)
        {
            float _3229 = -dot(ls.L, _3044.param2.xyz);
            if (_3229 > 0.0f)
            {
                ls.col *= clamp((_3044.param3.x - acos(clamp(_3229, 0.0f, 1.0f))) / _3044.param3.y, 0.0f, 1.0f);
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
        if (_3079 == 2u)
        {
            ls.L = _3044.param1.xyz;
            if (_3044.param1.w != 0.0f)
            {
                float param_2 = frac(asfloat(_2996.Load((_3001_g_params.hi + 4) * 4 + 0)) + sample_off.x);
                float param_3 = frac(asfloat(_2996.Load((_3001_g_params.hi + 5) * 4 + 0)) + sample_off.y);
                float3 param_4 = ls.L;
                float param_5 = tan(_3044.param1.w);
                ls.L = normalize(MapToCone(param_2, param_3, param_4, param_5));
            }
            ls.area = 0.0f;
            ls.dist = 3402823346297367662189621542912.0f;
            ls.pdf = 1.0f;
            if ((_3044.type_and_param0.x & 64u) == 0u)
            {
                ls.area = 0.0f;
            }
        }
        else
        {
            [branch]
            if (_3079 == 4u)
            {
                float3 _3370 = ((_3044.param1.xyz + (_3044.param2.xyz * (frac(asfloat(_2996.Load((_3001_g_params.hi + 4) * 4 + 0)) + sample_off.x) - 0.5f))) + (_3044.param3.xyz * (frac(asfloat(_2996.Load((_3001_g_params.hi + 5) * 4 + 0)) + sample_off.y) - 0.5f))) - P;
                ls.dist = length(_3370);
                ls.L = _3370 / ls.dist.xxx;
                ls.area = _3044.param1.w;
                float _3393 = dot(-ls.L, normalize(cross(_3044.param2.xyz, _3044.param3.xyz)));
                if (_3393 > 0.0f)
                {
                    ls.pdf = (ls.dist * ls.dist) / (ls.area * _3393);
                }
                if ((_3044.type_and_param0.x & 64u) == 0u)
                {
                    ls.area = 0.0f;
                }
                [branch]
                if ((_3044.type_and_param0.w & 128u) != 0u)
                {
                    float3 env_col = _3001_g_params.env_col.xyz;
                    uint _3433 = asuint(_3001_g_params.env_col.w);
                    if (_3433 != 4294967295u)
                    {
                        env_col *= SampleLatlong_RGBE(_3433, ls.L, _3001_g_params.env_rotation);
                    }
                    ls.col *= env_col;
                }
            }
            else
            {
                [branch]
                if (_3079 == 5u)
                {
                    float2 _3496 = (float2(frac(asfloat(_2996.Load((_3001_g_params.hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_2996.Load((_3001_g_params.hi + 5) * 4 + 0)) + sample_off.y)) * 2.0f) - 1.0f.xx;
                    float2 offset = _3496;
                    bool _3499 = _3496.x != 0.0f;
                    bool _3505;
                    if (_3499)
                    {
                        _3505 = offset.y != 0.0f;
                    }
                    else
                    {
                        _3505 = _3499;
                    }
                    if (_3505)
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
                        float _3538 = 0.5f * r;
                        offset = float2(_3538 * cos(theta), _3538 * sin(theta));
                    }
                    float3 _3564 = ((_3044.param1.xyz + (_3044.param2.xyz * offset.x)) + (_3044.param3.xyz * offset.y)) - P;
                    ls.dist = length(_3564);
                    ls.L = _3564 / ls.dist.xxx;
                    ls.area = _3044.param1.w;
                    float _3587 = dot(-ls.L, normalize(cross(_3044.param2.xyz, _3044.param3.xyz)));
                    [flatten]
                    if (_3587 > 0.0f)
                    {
                        ls.pdf = (ls.dist * ls.dist) / (ls.area * _3587);
                    }
                    if ((_3044.type_and_param0.x & 64u) == 0u)
                    {
                        ls.area = 0.0f;
                    }
                    [branch]
                    if ((_3044.type_and_param0.w & 128u) != 0u)
                    {
                        float3 env_col_1 = _3001_g_params.env_col.xyz;
                        uint _3623 = asuint(_3001_g_params.env_col.w);
                        if (_3623 != 4294967295u)
                        {
                            env_col_1 *= SampleLatlong_RGBE(_3623, ls.L, _3001_g_params.env_rotation);
                        }
                        ls.col *= env_col_1;
                    }
                }
                else
                {
                    [branch]
                    if (_3079 == 3u)
                    {
                        float3 _3682 = normalize(cross(P - _3044.param1.xyz, _3044.param3.xyz));
                        float _3689 = 3.1415927410125732421875f * frac(asfloat(_2996.Load((_3001_g_params.hi + 4) * 4 + 0)) + sample_off.x);
                        float3 _3718 = ((_3044.param1.xyz + (((_3682 * cos(_3689)) + (cross(_3682, _3044.param3.xyz) * sin(_3689))) * _3044.param2.w)) + ((_3044.param3.xyz * (frac(asfloat(_2996.Load((_3001_g_params.hi + 5) * 4 + 0)) + sample_off.y) - 0.5f)) * _3044.param3.w)) - P;
                        ls.dist = length(_3718);
                        ls.L = _3718 / ls.dist.xxx;
                        ls.area = _3044.param1.w;
                        float _3737 = 1.0f - abs(dot(ls.L, _3044.param3.xyz));
                        [flatten]
                        if (_3737 != 0.0f)
                        {
                            ls.pdf = (ls.dist * ls.dist) / (ls.area * _3737);
                        }
                        if ((_3044.type_and_param0.x & 64u) == 0u)
                        {
                            ls.area = 0.0f;
                        }
                        [branch]
                        if ((_3044.type_and_param0.w & 128u) != 0u)
                        {
                            float3 env_col_2 = _3001_g_params.env_col.xyz;
                            uint _3773 = asuint(_3001_g_params.env_col.w);
                            if (_3773 != 4294967295u)
                            {
                                env_col_2 *= SampleLatlong_RGBE(_3773, ls.L, _3001_g_params.env_rotation);
                            }
                            ls.col *= env_col_2;
                        }
                    }
                    else
                    {
                        [branch]
                        if (_3079 == 6u)
                        {
                            uint _3800 = asuint(_3044.param1.x);
                            transform_t _3814;
                            _3814.xform = asfloat(uint4x4(_3808.Load4(asuint(_3044.param1.y) * 128 + 0), _3808.Load4(asuint(_3044.param1.y) * 128 + 16), _3808.Load4(asuint(_3044.param1.y) * 128 + 32), _3808.Load4(asuint(_3044.param1.y) * 128 + 48)));
                            _3814.inv_xform = asfloat(uint4x4(_3808.Load4(asuint(_3044.param1.y) * 128 + 64), _3808.Load4(asuint(_3044.param1.y) * 128 + 80), _3808.Load4(asuint(_3044.param1.y) * 128 + 96), _3808.Load4(asuint(_3044.param1.y) * 128 + 112)));
                            uint _3839 = _3800 * 3u;
                            vertex_t _3845;
                            [unroll]
                            for (int _43ident = 0; _43ident < 3; _43ident++)
                            {
                                _3845.p[_43ident] = asfloat(_3833.Load(_43ident * 4 + _3837.Load(_3839 * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _44ident = 0; _44ident < 3; _44ident++)
                            {
                                _3845.n[_44ident] = asfloat(_3833.Load(_44ident * 4 + _3837.Load(_3839 * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _45ident = 0; _45ident < 3; _45ident++)
                            {
                                _3845.b[_45ident] = asfloat(_3833.Load(_45ident * 4 + _3837.Load(_3839 * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _46ident = 0; _46ident < 2; _46ident++)
                            {
                                [unroll]
                                for (int _47ident = 0; _47ident < 2; _47ident++)
                                {
                                    _3845.t[_46ident][_47ident] = asfloat(_3833.Load(_47ident * 4 + _46ident * 8 + _3837.Load(_3839 * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _3894;
                            [unroll]
                            for (int _48ident = 0; _48ident < 3; _48ident++)
                            {
                                _3894.p[_48ident] = asfloat(_3833.Load(_48ident * 4 + _3837.Load((_3839 + 1u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _49ident = 0; _49ident < 3; _49ident++)
                            {
                                _3894.n[_49ident] = asfloat(_3833.Load(_49ident * 4 + _3837.Load((_3839 + 1u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _50ident = 0; _50ident < 3; _50ident++)
                            {
                                _3894.b[_50ident] = asfloat(_3833.Load(_50ident * 4 + _3837.Load((_3839 + 1u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _51ident = 0; _51ident < 2; _51ident++)
                            {
                                [unroll]
                                for (int _52ident = 0; _52ident < 2; _52ident++)
                                {
                                    _3894.t[_51ident][_52ident] = asfloat(_3833.Load(_52ident * 4 + _51ident * 8 + _3837.Load((_3839 + 1u) * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _3940;
                            [unroll]
                            for (int _53ident = 0; _53ident < 3; _53ident++)
                            {
                                _3940.p[_53ident] = asfloat(_3833.Load(_53ident * 4 + _3837.Load((_3839 + 2u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _54ident = 0; _54ident < 3; _54ident++)
                            {
                                _3940.n[_54ident] = asfloat(_3833.Load(_54ident * 4 + _3837.Load((_3839 + 2u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _55ident = 0; _55ident < 3; _55ident++)
                            {
                                _3940.b[_55ident] = asfloat(_3833.Load(_55ident * 4 + _3837.Load((_3839 + 2u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _56ident = 0; _56ident < 2; _56ident++)
                            {
                                [unroll]
                                for (int _57ident = 0; _57ident < 2; _57ident++)
                                {
                                    _3940.t[_56ident][_57ident] = asfloat(_3833.Load(_57ident * 4 + _56ident * 8 + _3837.Load((_3839 + 2u) * 4 + 0) * 52 + 36));
                                }
                            }
                            float3 _3986 = float3(_3845.p[0], _3845.p[1], _3845.p[2]);
                            float3 _3994 = float3(_3894.p[0], _3894.p[1], _3894.p[2]);
                            float3 _4002 = float3(_3940.p[0], _3940.p[1], _3940.p[2]);
                            float _4031 = sqrt(frac(asfloat(_2996.Load((_3001_g_params.hi + 4) * 4 + 0)) + sample_off.x));
                            float _4041 = frac(asfloat(_2996.Load((_3001_g_params.hi + 5) * 4 + 0)) + sample_off.y);
                            float _4045 = 1.0f - _4031;
                            float _4050 = 1.0f - _4041;
                            float3 _4097 = mul(float4(cross(_3994 - _3986, _4002 - _3986), 0.0f), _3814.xform).xyz;
                            ls.area = 0.5f * length(_4097);
                            float3 _4107 = mul(float4((_3986 * _4045) + (((_3994 * _4050) + (_4002 * _4041)) * _4031), 1.0f), _3814.xform).xyz - P;
                            ls.dist = length(_4107);
                            ls.L = _4107 / ls.dist.xxx;
                            float _4122 = abs(dot(ls.L, normalize(_4097)));
                            [flatten]
                            if (_4122 > 0.0f)
                            {
                                ls.pdf = (ls.dist * ls.dist) / (ls.area * _4122);
                            }
                            material_t _4162;
                            [unroll]
                            for (int _58ident = 0; _58ident < 5; _58ident++)
                            {
                                _4162.textures[_58ident] = _4148.Load(_58ident * 4 + ((_4152.Load(_3800 * 4 + 0) >> 16u) & 16383u) * 80 + 0);
                            }
                            [unroll]
                            for (int _59ident = 0; _59ident < 3; _59ident++)
                            {
                                _4162.base_color[_59ident] = asfloat(_4148.Load(_59ident * 4 + ((_4152.Load(_3800 * 4 + 0) >> 16u) & 16383u) * 80 + 20));
                            }
                            _4162.flags = _4148.Load(((_4152.Load(_3800 * 4 + 0) >> 16u) & 16383u) * 80 + 32);
                            _4162.type = _4148.Load(((_4152.Load(_3800 * 4 + 0) >> 16u) & 16383u) * 80 + 36);
                            _4162.tangent_rotation_or_strength = asfloat(_4148.Load(((_4152.Load(_3800 * 4 + 0) >> 16u) & 16383u) * 80 + 40));
                            _4162.roughness_and_anisotropic = _4148.Load(((_4152.Load(_3800 * 4 + 0) >> 16u) & 16383u) * 80 + 44);
                            _4162.int_ior = asfloat(_4148.Load(((_4152.Load(_3800 * 4 + 0) >> 16u) & 16383u) * 80 + 48));
                            _4162.ext_ior = asfloat(_4148.Load(((_4152.Load(_3800 * 4 + 0) >> 16u) & 16383u) * 80 + 52));
                            _4162.sheen_and_sheen_tint = _4148.Load(((_4152.Load(_3800 * 4 + 0) >> 16u) & 16383u) * 80 + 56);
                            _4162.tint_and_metallic = _4148.Load(((_4152.Load(_3800 * 4 + 0) >> 16u) & 16383u) * 80 + 60);
                            _4162.transmission_and_transmission_roughness = _4148.Load(((_4152.Load(_3800 * 4 + 0) >> 16u) & 16383u) * 80 + 64);
                            _4162.specular_and_specular_tint = _4148.Load(((_4152.Load(_3800 * 4 + 0) >> 16u) & 16383u) * 80 + 68);
                            _4162.clearcoat_and_clearcoat_roughness = _4148.Load(((_4152.Load(_3800 * 4 + 0) >> 16u) & 16383u) * 80 + 72);
                            _4162.normal_map_strength_unorm = _4148.Load(((_4152.Load(_3800 * 4 + 0) >> 16u) & 16383u) * 80 + 76);
                            if ((_4162.flags & 4u) == 0u)
                            {
                                if (_4162.textures[1] != 4294967295u)
                                {
                                    ls.col *= SampleBilinear(_4162.textures[1], (float2(_3845.t[0][0], _3845.t[0][1]) * _4045) + (((float2(_3894.t[0][0], _3894.t[0][1]) * _4050) + (float2(_3940.t[0][0], _3940.t[0][1]) * _4041)) * _4031), 0).xyz;
                                }
                            }
                            else
                            {
                                float3 env_col_3 = _3001_g_params.env_col.xyz;
                                uint _4242 = asuint(_3001_g_params.env_col.w);
                                if (_4242 != 4294967295u)
                                {
                                    env_col_3 *= SampleLatlong_RGBE(_4242, ls.L, _3001_g_params.env_rotation);
                                }
                                ls.col *= env_col_3;
                            }
                        }
                        else
                        {
                            [branch]
                            if (_3079 == 7u)
                            {
                                float4 _4304 = Sample_EnvQTree(_3001_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3001_g_params.env_qtree_levels, mad(_3012, _3017, -float(_3024)), frac(asfloat(_2996.Load((_3001_g_params.hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_2996.Load((_3001_g_params.hi + 5) * 4 + 0)) + sample_off.y));
                                ls.L = _4304.xyz;
                                ls.col *= _3001_g_params.env_col.xyz;
                                ls.col *= SampleLatlong_RGBE(asuint(_3001_g_params.env_col.w), ls.L, _3001_g_params.env_rotation);
                                ls.area = 1.0f;
                                ls.dist = 3402823346297367662189621542912.0f;
                                ls.pdf = _4304.w;
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
    uint _841 = index & 16777215u;
    uint _845_dummy_parameter;
    return int2(spvTextureSize(g_textures[NonUniformResourceIndex(_841)], uint(0), _845_dummy_parameter));
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
    float _1992 = 1.0f / mad(0.904129683971405029296875f, roughness, 3.1415927410125732421875f);
    float _2004 = max(dot(N, L), 0.0f);
    float _2009 = max(dot(N, V), 0.0f);
    float _2017 = mad(-_2004, _2009, dot(L, V));
    float t = _2017;
    if (_2017 > 0.0f)
    {
        t /= (max(_2004, _2009) + 1.1754943508222875079687365372222e-38f);
    }
    return float4(base_color * (_2004 * mad(roughness * _1992, t, _1992)), 0.15915493667125701904296875f);
}

float3 offset_ray(float3 p, float3 n)
{
    int3 _1574 = int3(n * 128.0f);
    int _1582;
    if (p.x < 0.0f)
    {
        _1582 = -_1574.x;
    }
    else
    {
        _1582 = _1574.x;
    }
    int _1600;
    if (p.y < 0.0f)
    {
        _1600 = -_1574.y;
    }
    else
    {
        _1600 = _1574.y;
    }
    int _1618;
    if (p.z < 0.0f)
    {
        _1618 = -_1574.z;
    }
    else
    {
        _1618 = _1574.z;
    }
    float _1636;
    if (abs(p.x) < 0.03125f)
    {
        _1636 = mad(1.52587890625e-05f, n.x, p.x);
    }
    else
    {
        _1636 = asfloat(asint(p.x) + _1582);
    }
    float _1654;
    if (abs(p.y) < 0.03125f)
    {
        _1654 = mad(1.52587890625e-05f, n.y, p.y);
    }
    else
    {
        _1654 = asfloat(asint(p.y) + _1600);
    }
    float _1671;
    if (abs(p.z) < 0.03125f)
    {
        _1671 = mad(1.52587890625e-05f, n.z, p.z);
    }
    else
    {
        _1671 = asfloat(asint(p.z) + _1618);
    }
    return float3(_1636, _1654, _1671);
}

float3 world_from_tangent(float3 T, float3 B, float3 N, float3 V)
{
    return ((T * V.x) + (B * V.y)) + (N * V.z);
}

float4 Sample_OrenDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float rand_u, float rand_v, inout float3 out_V)
{
    float _2051 = 6.283185482025146484375f * rand_v;
    float _2063 = sqrt(mad(-rand_u, rand_u, 1.0f));
    float3 param = T;
    float3 param_1 = B;
    float3 param_2 = N;
    float3 param_3 = float3(_2063 * cos(_2051), _2063 * sin(_2051), rand_u);
    out_V = world_from_tangent(param, param_1, param_2, param_3);
    float3 param_4 = -I;
    float3 param_5 = N;
    float3 param_6 = out_V;
    float param_7 = roughness;
    float3 param_8 = base_color;
    return Evaluate_OrenDiffuse_BSDF(param_4, param_5, param_6, param_7, param_8);
}

float3 tangent_from_world(float3 T, float3 B, float3 N, float3 V)
{
    return float3(dot(V, T), dot(V, B), dot(V, N));
}

float D_GGX(float3 H, float alpha_x, float alpha_y)
{
    float _8291;
    do
    {
        if (H.z == 0.0f)
        {
            _8291 = 0.0f;
            break;
        }
        float _1878 = (-H.x) / (H.z * alpha_x);
        float _1884 = (-H.y) / (H.z * alpha_y);
        float _1893 = mad(_1884, _1884, mad(_1878, _1878, 1.0f));
        _8291 = 1.0f / (((((_1893 * _1893) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _8291;
}

float G1(float3 Ve, inout float alpha_x, inout float alpha_y)
{
    alpha_x *= alpha_x;
    alpha_y *= alpha_y;
    return 1.0f / mad((-1.0f) + sqrt(1.0f + (mad(alpha_x * Ve.x, Ve.x, (alpha_y * Ve.y) * Ve.y) / (Ve.z * Ve.z))), 0.5f, 1.0f);
}

float4 Evaluate_GGXSpecular_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float alpha_x, float alpha_y, float spec_ior, float spec_F0, float3 spec_col)
{
    float _2233 = D_GGX(sampled_normal_ts, alpha_x, alpha_y);
    float3 param = view_dir_ts;
    float param_1 = alpha_x;
    float param_2 = alpha_y;
    float _2241 = G1(param, param_1, param_2);
    float3 param_3 = reflected_dir_ts;
    float param_4 = alpha_x;
    float param_5 = alpha_y;
    float _2248 = G1(param_3, param_4, param_5);
    float param_6 = dot(view_dir_ts, sampled_normal_ts);
    float param_7 = spec_ior;
    float3 F = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param_6, param_7) - spec_F0) / (1.0f - spec_F0)).xxx);
    float _2276 = 4.0f * abs(view_dir_ts.z * reflected_dir_ts.z);
    float _2279;
    if (_2276 != 0.0f)
    {
        _2279 = (_2233 * (_2241 * _2248)) / _2276;
    }
    else
    {
        _2279 = 0.0f;
    }
    F *= _2279;
    float3 param_8 = view_dir_ts;
    float param_9 = alpha_x;
    float param_10 = alpha_y;
    float _2299 = G1(param_8, param_9, param_10);
    float pdf = ((_2233 * _2299) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2314 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2314 != 0.0f)
    {
        pdf /= _2314;
    }
    float3 _2325 = F;
    float3 _2326 = _2325 * max(reflected_dir_ts.z, 0.0f);
    F = _2326;
    return float4(_2326, pdf);
}

float3 SampleGGX_VNDF(float3 Ve, float alpha_x, float alpha_y, float U1, float U2)
{
    float3 _1696 = normalize(float3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
    float _1699 = _1696.x;
    float _1704 = _1696.y;
    float _1708 = mad(_1699, _1699, _1704 * _1704);
    float3 _1712;
    if (_1708 > 0.0f)
    {
        _1712 = float3(-_1704, _1699, 0.0f) / sqrt(_1708).xxx;
    }
    else
    {
        _1712 = float3(1.0f, 0.0f, 0.0f);
    }
    float _1734 = sqrt(U1);
    float _1737 = 6.283185482025146484375f * U2;
    float _1742 = _1734 * cos(_1737);
    float _1751 = 1.0f + _1696.z;
    float _1758 = mad(-_1742, _1742, 1.0f);
    float _1764 = mad(mad(-0.5f, _1751, 1.0f), sqrt(_1758), (0.5f * _1751) * (_1734 * sin(_1737)));
    float3 _1785 = ((_1712 * _1742) + (cross(_1696, _1712) * _1764)) + (_1696 * sqrt(max(0.0f, mad(-_1764, _1764, _1758))));
    return normalize(float3(alpha_x * _1785.x, alpha_y * _1785.y, max(0.0f, _1785.z)));
}

float4 Sample_GGXSpecular_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float anisotropic, float spec_ior, float spec_F0, float3 spec_col, float rand_u, float rand_v, inout float3 out_V)
{
    float4 _8266;
    do
    {
        float _2336 = roughness * roughness;
        float _2340 = sqrt(mad(-0.89999997615814208984375f, anisotropic, 1.0f));
        float _2344 = _2336 / _2340;
        float _2348 = _2336 * _2340;
        [branch]
        if ((_2344 * _2348) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2358 = reflect(I, N);
            float param = dot(_2358, N);
            float param_1 = spec_ior;
            float3 _2372 = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param, param_1) - spec_F0) / (1.0f - spec_F0)).xxx);
            out_V = _2358;
            _8266 = float4(_2372.x * 1000000.0f, _2372.y * 1000000.0f, _2372.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2397 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = _2344;
        float param_7 = _2348;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2406 = SampleGGX_VNDF(_2397, param_6, param_7, param_8, param_9);
        float3 _2417 = normalize(reflect(-_2397, _2406));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2417;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2397;
        float3 param_15 = _2406;
        float3 param_16 = _2417;
        float param_17 = _2344;
        float param_18 = _2348;
        float param_19 = spec_ior;
        float param_20 = spec_F0;
        float3 param_21 = spec_col;
        _8266 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _8266;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _8271;
    do
    {
        bool _2639 = refr_dir_ts.z >= 0.0f;
        bool _2646;
        if (!_2639)
        {
            _2646 = view_dir_ts.z <= 0.0f;
        }
        else
        {
            _2646 = _2639;
        }
        if (_2646)
        {
            _8271 = 0.0f.xxxx;
            break;
        }
        float _2655 = D_GGX(sampled_normal_ts, roughness2, roughness2);
        float3 param = refr_dir_ts;
        float param_1 = roughness2;
        float param_2 = roughness2;
        float _2663 = G1(param, param_1, param_2);
        float3 param_3 = view_dir_ts;
        float param_4 = roughness2;
        float param_5 = roughness2;
        float _2671 = G1(param_3, param_4, param_5);
        float _2681 = mad(dot(view_dir_ts, sampled_normal_ts), eta, dot(refr_dir_ts, sampled_normal_ts));
        float _2691 = clamp(-dot(refr_dir_ts, sampled_normal_ts), 0.0f, 1.0f) / (_2681 * _2681);
        _8271 = float4(refr_col * (((((_2655 * _2671) * _2663) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _2691) / view_dir_ts.z), (((_2655 * _2663) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _2691) / view_dir_ts.z);
        break;
    } while(false);
    return _8271;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _8276;
    do
    {
        float _2735 = roughness * roughness;
        [branch]
        if ((_2735 * _2735) < 1.0000000116860974230803549289703e-07f)
        {
            float _2745 = dot(I, N);
            float _2746 = -_2745;
            float _2756 = mad(-(eta * eta), mad(_2745, _2746, 1.0f), 1.0f);
            if (_2756 < 0.0f)
            {
                _8276 = 0.0f.xxxx;
                break;
            }
            float _2768 = mad(eta, _2746, -sqrt(_2756));
            out_V = float4(normalize((I * eta) + (N * _2768)), _2768);
            _8276 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param = T;
        float3 param_1 = B;
        float3 param_2 = N;
        float3 param_3 = -I;
        float3 _2808 = normalize(tangent_from_world(param, param_1, param_2, param_3));
        float param_4 = _2735;
        float param_5 = _2735;
        float param_6 = rand_u;
        float param_7 = rand_v;
        float3 _2819 = SampleGGX_VNDF(_2808, param_4, param_5, param_6, param_7);
        float _2823 = dot(_2808, _2819);
        float _2833 = mad(-(eta * eta), mad(-_2823, _2823, 1.0f), 1.0f);
        if (_2833 < 0.0f)
        {
            _8276 = 0.0f.xxxx;
            break;
        }
        float _2845 = mad(eta, _2823, -sqrt(_2833));
        float3 _2855 = normalize((_2808 * (-eta)) + (_2819 * _2845));
        float3 param_8 = _2808;
        float3 param_9 = _2819;
        float3 param_10 = _2855;
        float param_11 = _2735;
        float param_12 = eta;
        float3 param_13 = refr_col;
        float3 param_14 = T;
        float3 param_15 = B;
        float3 param_16 = N;
        float3 param_17 = _2855;
        out_V = float4(world_from_tangent(param_14, param_15, param_16, param_17), _2845);
        _8276 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _8276;
}

void get_lobe_weights(float base_color_lum, float spec_color_lum, float specular, float metallic, float transmission, float clearcoat, inout float out_diffuse_weight, inout float out_specular_weight, inout float out_clearcoat_weight, inout float out_refraction_weight)
{
    float _1367 = 1.0f - metallic;
    out_diffuse_weight = (base_color_lum * _1367) * (1.0f - transmission);
    float _1377;
    if ((specular != 0.0f) || (metallic != 0.0f))
    {
        _1377 = spec_color_lum * mad(-transmission, _1367, 1.0f);
    }
    else
    {
        _1377 = 0.0f;
    }
    out_specular_weight = _1377;
    out_clearcoat_weight = (0.25f * clearcoat) * _1367;
    out_refraction_weight = (transmission * _1367) * base_color_lum;
    float _1392 = out_diffuse_weight;
    float _1393 = out_specular_weight;
    float _1395 = out_clearcoat_weight;
    float _1398 = ((_1392 + _1393) + _1395) + out_refraction_weight;
    if (_1398 != 0.0f)
    {
        out_diffuse_weight /= _1398;
        out_specular_weight /= _1398;
        out_clearcoat_weight /= _1398;
        out_refraction_weight /= _1398;
    }
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
    float _8296;
    do
    {
        float _1944 = dot(N, L);
        if (_1944 <= 0.0f)
        {
            _8296 = 0.0f;
            break;
        }
        float param = _1944;
        float param_1 = dot(N, V);
        float _1965 = dot(L, H);
        float _1973 = mad((2.0f * _1965) * _1965, roughness, 0.5f);
        _8296 = lerp(1.0f, _1973, schlick_weight(param)) * lerp(1.0f, _1973, schlick_weight(param_1));
        break;
    } while(false);
    return _8296;
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
    float3 _2114 = normalize(L + V);
    float3 H = _2114;
    if (dot(V, _2114) < 0.0f)
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
    float3 _2149 = diff_col;
    float3 _2150 = _2149 + (sheen_color * (3.1415927410125732421875f * schlick_weight(param_5)));
    diff_col = _2150;
    return float4(_2150, pdf);
}

float D_GTR1(float NDotH, float a)
{
    float _8301;
    do
    {
        if (a >= 1.0f)
        {
            _8301 = 0.3183098733425140380859375f;
            break;
        }
        float _1852 = mad(a, a, -1.0f);
        _8301 = _1852 / ((3.1415927410125732421875f * log(a * a)) * mad(_1852 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _8301;
}

float4 Evaluate_PrincipledClearcoat_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float clearcoat_roughness2, float clearcoat_ior, float clearcoat_F0)
{
    float param = sampled_normal_ts.z;
    float param_1 = clearcoat_roughness2;
    float _2449 = D_GTR1(param, param_1);
    float3 param_2 = view_dir_ts;
    float param_3 = 0.0625f;
    float param_4 = 0.0625f;
    float _2456 = G1(param_2, param_3, param_4);
    float3 param_5 = reflected_dir_ts;
    float param_6 = 0.0625f;
    float param_7 = 0.0625f;
    float _2461 = G1(param_5, param_6, param_7);
    float param_8 = dot(reflected_dir_ts, sampled_normal_ts);
    float param_9 = clearcoat_ior;
    float F = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param_8, param_9) - clearcoat_F0) / (1.0f - clearcoat_F0));
    float _2488 = (4.0f * abs(view_dir_ts.z)) * abs(reflected_dir_ts.z);
    float _2491;
    if (_2488 != 0.0f)
    {
        _2491 = (_2449 * (_2456 * _2461)) / _2488;
    }
    else
    {
        _2491 = 0.0f;
    }
    F *= _2491;
    float3 param_10 = view_dir_ts;
    float param_11 = 0.0625f;
    float param_12 = 0.0625f;
    float _2509 = G1(param_10, param_11, param_12);
    float pdf = ((_2449 * _2509) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2524 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2524 != 0.0f)
    {
        pdf /= _2524;
    }
    float _2535 = F;
    float _2536 = _2535 * clamp(reflected_dir_ts.z, 0.0f, 1.0f);
    F = _2536;
    return float4(_2536, _2536, _2536, pdf);
}

float4 Sample_PrincipledDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float3 sheen_color, bool uniform_sampling, float rand_u, float rand_v, inout float3 out_V)
{
    float _2161 = 6.283185482025146484375f * rand_v;
    float _2164 = cos(_2161);
    float _2167 = sin(_2161);
    float3 V;
    if (uniform_sampling)
    {
        float _2176 = sqrt(mad(-rand_u, rand_u, 1.0f));
        V = float3(_2176 * _2164, _2176 * _2167, rand_u);
    }
    else
    {
        float _2189 = sqrt(rand_u);
        V = float3(_2189 * _2164, _2189 * _2167, sqrt(1.0f - rand_u));
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
    float4 _8281;
    do
    {
        [branch]
        if ((clearcoat_roughness2 * clearcoat_roughness2) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2553 = reflect(I, N);
            float param = dot(_2553, N);
            float param_1 = clearcoat_ior;
            out_V = _2553;
            float _2572 = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param, param_1) - clearcoat_F0) / (1.0f - clearcoat_F0)) * 1000000.0f;
            _8281 = float4(_2572, _2572, _2572, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2590 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = clearcoat_roughness2;
        float param_7 = clearcoat_roughness2;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2601 = SampleGGX_VNDF(_2590, param_6, param_7, param_8, param_9);
        float3 _2612 = normalize(reflect(-_2590, _2601));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2612;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2590;
        float3 param_15 = _2601;
        float3 param_16 = _2612;
        float param_17 = clearcoat_roughness2;
        float param_18 = clearcoat_ior;
        float param_19 = clearcoat_F0;
        _8281 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _8281;
}

float3 ShadeSurface(int px_index, hit_data_t inter, ray_data_t ray)
{
    float3 _8256;
    do
    {
        float3 _4349 = float3(ray.d[0], ray.d[1], ray.d[2]);
        [branch]
        if (inter.mask == 0)
        {
            float3 env_col = _3001_g_params.back_col.xyz;
            uint _4362 = asuint(_3001_g_params.back_col.w);
            if (_4362 != 4294967295u)
            {
                env_col *= SampleLatlong_RGBE(_4362, _4349, _3001_g_params.env_rotation);
                if (_3001_g_params.env_qtree_levels > 0)
                {
                    float param = ray.pdf;
                    float param_1 = Evaluate_EnvQTree(_3001_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3001_g_params.env_qtree_levels, _4349);
                    env_col *= power_heuristic(param, param_1);
                }
            }
            _8256 = float3(ray.c[0] * env_col.x, ray.c[1] * env_col.y, ray.c[2] * env_col.z);
            break;
        }
        float3 _4423 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_4349 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            light_t _4435;
            _4435.type_and_param0 = _3033.Load4(((-1) - inter.obj_index) * 64 + 0);
            _4435.param1 = asfloat(_3033.Load4(((-1) - inter.obj_index) * 64 + 16));
            _4435.param2 = asfloat(_3033.Load4(((-1) - inter.obj_index) * 64 + 32));
            _4435.param3 = asfloat(_3033.Load4(((-1) - inter.obj_index) * 64 + 48));
            float3 lcol = asfloat(_4435.type_and_param0.yzw);
            uint _4452 = _4435.type_and_param0.x & 31u;
            if (_4452 == 0u)
            {
                float param_2 = ray.pdf;
                float param_3 = (inter.t * inter.t) / ((0.5f * _4435.param1.w) * dot(_4349, normalize(_4435.param1.xyz - _4423)));
                lcol *= power_heuristic(param_2, param_3);
                bool _4519 = _4435.param3.x > 0.0f;
                bool _4525;
                if (_4519)
                {
                    _4525 = _4435.param3.y > 0.0f;
                }
                else
                {
                    _4525 = _4519;
                }
                [branch]
                if (_4525)
                {
                    [flatten]
                    if (_4435.param3.y > 0.0f)
                    {
                        lcol *= clamp((_4435.param3.x - acos(clamp(-dot(_4349, _4435.param2.xyz), 0.0f, 1.0f))) / _4435.param3.y, 0.0f, 1.0f);
                    }
                }
            }
            else
            {
                if (_4452 == 4u)
                {
                    float param_4 = ray.pdf;
                    float param_5 = (inter.t * inter.t) / (_4435.param1.w * dot(_4349, normalize(cross(_4435.param2.xyz, _4435.param3.xyz))));
                    lcol *= power_heuristic(param_4, param_5);
                }
                else
                {
                    if (_4452 == 5u)
                    {
                        float param_6 = ray.pdf;
                        float param_7 = (inter.t * inter.t) / (_4435.param1.w * dot(_4349, normalize(cross(_4435.param2.xyz, _4435.param3.xyz))));
                        lcol *= power_heuristic(param_6, param_7);
                    }
                    else
                    {
                        if (_4452 == 3u)
                        {
                            float param_8 = ray.pdf;
                            float param_9 = (inter.t * inter.t) / (_4435.param1.w * (1.0f - abs(dot(_4349, _4435.param3.xyz))));
                            lcol *= power_heuristic(param_8, param_9);
                        }
                    }
                }
            }
            _8256 = float3(ray.c[0] * lcol.x, ray.c[1] * lcol.y, ray.c[2] * lcol.z);
            break;
        }
        bool _4724 = inter.prim_index < 0;
        int _4727;
        if (_4724)
        {
            _4727 = (-1) - inter.prim_index;
        }
        else
        {
            _4727 = inter.prim_index;
        }
        uint _4738 = uint(_4727);
        material_t _4746;
        [unroll]
        for (int _60ident = 0; _60ident < 5; _60ident++)
        {
            _4746.textures[_60ident] = _4148.Load(_60ident * 4 + ((_4152.Load(_4738 * 4 + 0) >> 16u) & 16383u) * 80 + 0);
        }
        [unroll]
        for (int _61ident = 0; _61ident < 3; _61ident++)
        {
            _4746.base_color[_61ident] = asfloat(_4148.Load(_61ident * 4 + ((_4152.Load(_4738 * 4 + 0) >> 16u) & 16383u) * 80 + 20));
        }
        _4746.flags = _4148.Load(((_4152.Load(_4738 * 4 + 0) >> 16u) & 16383u) * 80 + 32);
        _4746.type = _4148.Load(((_4152.Load(_4738 * 4 + 0) >> 16u) & 16383u) * 80 + 36);
        _4746.tangent_rotation_or_strength = asfloat(_4148.Load(((_4152.Load(_4738 * 4 + 0) >> 16u) & 16383u) * 80 + 40));
        _4746.roughness_and_anisotropic = _4148.Load(((_4152.Load(_4738 * 4 + 0) >> 16u) & 16383u) * 80 + 44);
        _4746.int_ior = asfloat(_4148.Load(((_4152.Load(_4738 * 4 + 0) >> 16u) & 16383u) * 80 + 48));
        _4746.ext_ior = asfloat(_4148.Load(((_4152.Load(_4738 * 4 + 0) >> 16u) & 16383u) * 80 + 52));
        _4746.sheen_and_sheen_tint = _4148.Load(((_4152.Load(_4738 * 4 + 0) >> 16u) & 16383u) * 80 + 56);
        _4746.tint_and_metallic = _4148.Load(((_4152.Load(_4738 * 4 + 0) >> 16u) & 16383u) * 80 + 60);
        _4746.transmission_and_transmission_roughness = _4148.Load(((_4152.Load(_4738 * 4 + 0) >> 16u) & 16383u) * 80 + 64);
        _4746.specular_and_specular_tint = _4148.Load(((_4152.Load(_4738 * 4 + 0) >> 16u) & 16383u) * 80 + 68);
        _4746.clearcoat_and_clearcoat_roughness = _4148.Load(((_4152.Load(_4738 * 4 + 0) >> 16u) & 16383u) * 80 + 72);
        _4746.normal_map_strength_unorm = _4148.Load(((_4152.Load(_4738 * 4 + 0) >> 16u) & 16383u) * 80 + 76);
        uint _8807 = _4746.textures[0];
        uint _8808 = _4746.textures[1];
        uint _8809 = _4746.textures[2];
        uint _8810 = _4746.textures[3];
        uint _8811 = _4746.textures[4];
        float _8812 = _4746.base_color[0];
        float _8813 = _4746.base_color[1];
        float _8814 = _4746.base_color[2];
        uint _8503 = _4746.flags;
        uint _8504 = _4746.type;
        float _8505 = _4746.tangent_rotation_or_strength;
        uint _8506 = _4746.roughness_and_anisotropic;
        float _8507 = _4746.int_ior;
        float _8508 = _4746.ext_ior;
        uint _8509 = _4746.sheen_and_sheen_tint;
        uint _8510 = _4746.tint_and_metallic;
        uint _8511 = _4746.transmission_and_transmission_roughness;
        uint _8512 = _4746.specular_and_specular_tint;
        uint _8513 = _4746.clearcoat_and_clearcoat_roughness;
        uint _8514 = _4746.normal_map_strength_unorm;
        transform_t _4803;
        _4803.xform = asfloat(uint4x4(_3808.Load4(asuint(asfloat(_4796.Load(inter.obj_index * 32 + 12))) * 128 + 0), _3808.Load4(asuint(asfloat(_4796.Load(inter.obj_index * 32 + 12))) * 128 + 16), _3808.Load4(asuint(asfloat(_4796.Load(inter.obj_index * 32 + 12))) * 128 + 32), _3808.Load4(asuint(asfloat(_4796.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _4803.inv_xform = asfloat(uint4x4(_3808.Load4(asuint(asfloat(_4796.Load(inter.obj_index * 32 + 12))) * 128 + 64), _3808.Load4(asuint(asfloat(_4796.Load(inter.obj_index * 32 + 12))) * 128 + 80), _3808.Load4(asuint(asfloat(_4796.Load(inter.obj_index * 32 + 12))) * 128 + 96), _3808.Load4(asuint(asfloat(_4796.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _4810 = _4738 * 3u;
        vertex_t _4815;
        [unroll]
        for (int _62ident = 0; _62ident < 3; _62ident++)
        {
            _4815.p[_62ident] = asfloat(_3833.Load(_62ident * 4 + _3837.Load(_4810 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _63ident = 0; _63ident < 3; _63ident++)
        {
            _4815.n[_63ident] = asfloat(_3833.Load(_63ident * 4 + _3837.Load(_4810 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _64ident = 0; _64ident < 3; _64ident++)
        {
            _4815.b[_64ident] = asfloat(_3833.Load(_64ident * 4 + _3837.Load(_4810 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _65ident = 0; _65ident < 2; _65ident++)
        {
            [unroll]
            for (int _66ident = 0; _66ident < 2; _66ident++)
            {
                _4815.t[_65ident][_66ident] = asfloat(_3833.Load(_66ident * 4 + _65ident * 8 + _3837.Load(_4810 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _4861;
        [unroll]
        for (int _67ident = 0; _67ident < 3; _67ident++)
        {
            _4861.p[_67ident] = asfloat(_3833.Load(_67ident * 4 + _3837.Load((_4810 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _68ident = 0; _68ident < 3; _68ident++)
        {
            _4861.n[_68ident] = asfloat(_3833.Load(_68ident * 4 + _3837.Load((_4810 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _69ident = 0; _69ident < 3; _69ident++)
        {
            _4861.b[_69ident] = asfloat(_3833.Load(_69ident * 4 + _3837.Load((_4810 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _70ident = 0; _70ident < 2; _70ident++)
        {
            [unroll]
            for (int _71ident = 0; _71ident < 2; _71ident++)
            {
                _4861.t[_70ident][_71ident] = asfloat(_3833.Load(_71ident * 4 + _70ident * 8 + _3837.Load((_4810 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _4907;
        [unroll]
        for (int _72ident = 0; _72ident < 3; _72ident++)
        {
            _4907.p[_72ident] = asfloat(_3833.Load(_72ident * 4 + _3837.Load((_4810 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _73ident = 0; _73ident < 3; _73ident++)
        {
            _4907.n[_73ident] = asfloat(_3833.Load(_73ident * 4 + _3837.Load((_4810 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _74ident = 0; _74ident < 3; _74ident++)
        {
            _4907.b[_74ident] = asfloat(_3833.Load(_74ident * 4 + _3837.Load((_4810 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _75ident = 0; _75ident < 2; _75ident++)
        {
            [unroll]
            for (int _76ident = 0; _76ident < 2; _76ident++)
            {
                _4907.t[_75ident][_76ident] = asfloat(_3833.Load(_76ident * 4 + _75ident * 8 + _3837.Load((_4810 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _4953 = float3(_4815.p[0], _4815.p[1], _4815.p[2]);
        float3 _4961 = float3(_4861.p[0], _4861.p[1], _4861.p[2]);
        float3 _4969 = float3(_4907.p[0], _4907.p[1], _4907.p[2]);
        float _4976 = (1.0f - inter.u) - inter.v;
        float3 _5009 = normalize(((float3(_4815.n[0], _4815.n[1], _4815.n[2]) * _4976) + (float3(_4861.n[0], _4861.n[1], _4861.n[2]) * inter.u)) + (float3(_4907.n[0], _4907.n[1], _4907.n[2]) * inter.v));
        float3 N = _5009;
        float2 _5035 = ((float2(_4815.t[0][0], _4815.t[0][1]) * _4976) + (float2(_4861.t[0][0], _4861.t[0][1]) * inter.u)) + (float2(_4907.t[0][0], _4907.t[0][1]) * inter.v);
        float3 _5051 = cross(_4961 - _4953, _4969 - _4953);
        float _5054 = length(_5051);
        float3 plane_N = _5051 / _5054.xxx;
        float3 _5090 = ((float3(_4815.b[0], _4815.b[1], _4815.b[2]) * _4976) + (float3(_4861.b[0], _4861.b[1], _4861.b[2]) * inter.u)) + (float3(_4907.b[0], _4907.b[1], _4907.b[2]) * inter.v);
        float3 B = _5090;
        float3 T = cross(_5090, _5009);
        if (_4724)
        {
            if ((_4152.Load(_4738 * 4 + 0) & 65535u) == 65535u)
            {
                _8256 = 0.0f.xxx;
                break;
            }
            material_t _5114;
            [unroll]
            for (int _77ident = 0; _77ident < 5; _77ident++)
            {
                _5114.textures[_77ident] = _4148.Load(_77ident * 4 + (_4152.Load(_4738 * 4 + 0) & 16383u) * 80 + 0);
            }
            [unroll]
            for (int _78ident = 0; _78ident < 3; _78ident++)
            {
                _5114.base_color[_78ident] = asfloat(_4148.Load(_78ident * 4 + (_4152.Load(_4738 * 4 + 0) & 16383u) * 80 + 20));
            }
            _5114.flags = _4148.Load((_4152.Load(_4738 * 4 + 0) & 16383u) * 80 + 32);
            _5114.type = _4148.Load((_4152.Load(_4738 * 4 + 0) & 16383u) * 80 + 36);
            _5114.tangent_rotation_or_strength = asfloat(_4148.Load((_4152.Load(_4738 * 4 + 0) & 16383u) * 80 + 40));
            _5114.roughness_and_anisotropic = _4148.Load((_4152.Load(_4738 * 4 + 0) & 16383u) * 80 + 44);
            _5114.int_ior = asfloat(_4148.Load((_4152.Load(_4738 * 4 + 0) & 16383u) * 80 + 48));
            _5114.ext_ior = asfloat(_4148.Load((_4152.Load(_4738 * 4 + 0) & 16383u) * 80 + 52));
            _5114.sheen_and_sheen_tint = _4148.Load((_4152.Load(_4738 * 4 + 0) & 16383u) * 80 + 56);
            _5114.tint_and_metallic = _4148.Load((_4152.Load(_4738 * 4 + 0) & 16383u) * 80 + 60);
            _5114.transmission_and_transmission_roughness = _4148.Load((_4152.Load(_4738 * 4 + 0) & 16383u) * 80 + 64);
            _5114.specular_and_specular_tint = _4148.Load((_4152.Load(_4738 * 4 + 0) & 16383u) * 80 + 68);
            _5114.clearcoat_and_clearcoat_roughness = _4148.Load((_4152.Load(_4738 * 4 + 0) & 16383u) * 80 + 72);
            _5114.normal_map_strength_unorm = _4148.Load((_4152.Load(_4738 * 4 + 0) & 16383u) * 80 + 76);
            _8807 = _5114.textures[0];
            _8808 = _5114.textures[1];
            _8809 = _5114.textures[2];
            _8810 = _5114.textures[3];
            _8811 = _5114.textures[4];
            _8812 = _5114.base_color[0];
            _8813 = _5114.base_color[1];
            _8814 = _5114.base_color[2];
            _8503 = _5114.flags;
            _8504 = _5114.type;
            _8505 = _5114.tangent_rotation_or_strength;
            _8506 = _5114.roughness_and_anisotropic;
            _8507 = _5114.int_ior;
            _8508 = _5114.ext_ior;
            _8509 = _5114.sheen_and_sheen_tint;
            _8510 = _5114.tint_and_metallic;
            _8511 = _5114.transmission_and_transmission_roughness;
            _8512 = _5114.specular_and_specular_tint;
            _8513 = _5114.clearcoat_and_clearcoat_roughness;
            _8514 = _5114.normal_map_strength_unorm;
            plane_N = -plane_N;
            N = -N;
            B = -B;
            T = -T;
        }
        float3 param_10 = plane_N;
        float4x4 param_11 = _4803.inv_xform;
        plane_N = TransformNormal(param_10, param_11);
        float3 param_12 = N;
        float4x4 param_13 = _4803.inv_xform;
        N = TransformNormal(param_12, param_13);
        float3 param_14 = B;
        float4x4 param_15 = _4803.inv_xform;
        B = TransformNormal(param_14, param_15);
        float3 param_16 = T;
        float4x4 param_17 = _4803.inv_xform;
        T = TransformNormal(param_16, param_17);
        float _5224 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _5234 = mad(0.5f, log2(abs(mad(_4861.t[0][0] - _4815.t[0][0], _4907.t[0][1] - _4815.t[0][1], -((_4907.t[0][0] - _4815.t[0][0]) * (_4861.t[0][1] - _4815.t[0][1])))) / _5054), log2(_5224));
        uint param_18 = uint(hash(px_index));
        float _5240 = construct_float(param_18);
        uint param_19 = uint(hash(hash(px_index)));
        float _5246 = construct_float(param_19);
        float3 col = 0.0f.xxx;
        int _5253 = ray.ray_depth & 255;
        int _5258 = (ray.ray_depth >> 8) & 255;
        int _5263 = (ray.ray_depth >> 16) & 255;
        int _5269 = (ray.ray_depth >> 24) & 255;
        int _5277 = ((_5253 + _5258) + _5263) + _5269;
        float mix_rand = frac(asfloat(_2996.Load(_3001_g_params.hi * 4 + 0)) + _5240);
        float mix_weight = 1.0f;
        float _5314;
        float _5333;
        float _5358;
        float _5427;
        while (_8504 == 4u)
        {
            float mix_val = _8505;
            if (_8808 != 4294967295u)
            {
                mix_val *= SampleBilinear(_8808, _5035, 0).x;
            }
            if (_4724)
            {
                _5314 = _8508 / _8507;
            }
            else
            {
                _5314 = _8507 / _8508;
            }
            if (_8507 != 0.0f)
            {
                float param_20 = dot(_4349, N);
                float param_21 = _5314;
                _5333 = fresnel_dielectric_cos(param_20, param_21);
            }
            else
            {
                _5333 = 1.0f;
            }
            float _5347 = mix_val;
            float _5348 = _5347 * clamp(_5333, 0.0f, 1.0f);
            mix_val = _5348;
            if (mix_rand > _5348)
            {
                if ((_8503 & 2u) != 0u)
                {
                    _5358 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _5358 = 1.0f;
                }
                mix_weight *= _5358;
                material_t _5371;
                [unroll]
                for (int _79ident = 0; _79ident < 5; _79ident++)
                {
                    _5371.textures[_79ident] = _4148.Load(_79ident * 4 + _8810 * 80 + 0);
                }
                [unroll]
                for (int _80ident = 0; _80ident < 3; _80ident++)
                {
                    _5371.base_color[_80ident] = asfloat(_4148.Load(_80ident * 4 + _8810 * 80 + 20));
                }
                _5371.flags = _4148.Load(_8810 * 80 + 32);
                _5371.type = _4148.Load(_8810 * 80 + 36);
                _5371.tangent_rotation_or_strength = asfloat(_4148.Load(_8810 * 80 + 40));
                _5371.roughness_and_anisotropic = _4148.Load(_8810 * 80 + 44);
                _5371.int_ior = asfloat(_4148.Load(_8810 * 80 + 48));
                _5371.ext_ior = asfloat(_4148.Load(_8810 * 80 + 52));
                _5371.sheen_and_sheen_tint = _4148.Load(_8810 * 80 + 56);
                _5371.tint_and_metallic = _4148.Load(_8810 * 80 + 60);
                _5371.transmission_and_transmission_roughness = _4148.Load(_8810 * 80 + 64);
                _5371.specular_and_specular_tint = _4148.Load(_8810 * 80 + 68);
                _5371.clearcoat_and_clearcoat_roughness = _4148.Load(_8810 * 80 + 72);
                _5371.normal_map_strength_unorm = _4148.Load(_8810 * 80 + 76);
                _8807 = _5371.textures[0];
                _8808 = _5371.textures[1];
                _8809 = _5371.textures[2];
                _8810 = _5371.textures[3];
                _8811 = _5371.textures[4];
                _8812 = _5371.base_color[0];
                _8813 = _5371.base_color[1];
                _8814 = _5371.base_color[2];
                _8503 = _5371.flags;
                _8504 = _5371.type;
                _8505 = _5371.tangent_rotation_or_strength;
                _8506 = _5371.roughness_and_anisotropic;
                _8507 = _5371.int_ior;
                _8508 = _5371.ext_ior;
                _8509 = _5371.sheen_and_sheen_tint;
                _8510 = _5371.tint_and_metallic;
                _8511 = _5371.transmission_and_transmission_roughness;
                _8512 = _5371.specular_and_specular_tint;
                _8513 = _5371.clearcoat_and_clearcoat_roughness;
                _8514 = _5371.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_8503 & 2u) != 0u)
                {
                    _5427 = 1.0f / mix_val;
                }
                else
                {
                    _5427 = 1.0f;
                }
                mix_weight *= _5427;
                material_t _5439;
                [unroll]
                for (int _81ident = 0; _81ident < 5; _81ident++)
                {
                    _5439.textures[_81ident] = _4148.Load(_81ident * 4 + _8811 * 80 + 0);
                }
                [unroll]
                for (int _82ident = 0; _82ident < 3; _82ident++)
                {
                    _5439.base_color[_82ident] = asfloat(_4148.Load(_82ident * 4 + _8811 * 80 + 20));
                }
                _5439.flags = _4148.Load(_8811 * 80 + 32);
                _5439.type = _4148.Load(_8811 * 80 + 36);
                _5439.tangent_rotation_or_strength = asfloat(_4148.Load(_8811 * 80 + 40));
                _5439.roughness_and_anisotropic = _4148.Load(_8811 * 80 + 44);
                _5439.int_ior = asfloat(_4148.Load(_8811 * 80 + 48));
                _5439.ext_ior = asfloat(_4148.Load(_8811 * 80 + 52));
                _5439.sheen_and_sheen_tint = _4148.Load(_8811 * 80 + 56);
                _5439.tint_and_metallic = _4148.Load(_8811 * 80 + 60);
                _5439.transmission_and_transmission_roughness = _4148.Load(_8811 * 80 + 64);
                _5439.specular_and_specular_tint = _4148.Load(_8811 * 80 + 68);
                _5439.clearcoat_and_clearcoat_roughness = _4148.Load(_8811 * 80 + 72);
                _5439.normal_map_strength_unorm = _4148.Load(_8811 * 80 + 76);
                _8807 = _5439.textures[0];
                _8808 = _5439.textures[1];
                _8809 = _5439.textures[2];
                _8810 = _5439.textures[3];
                _8811 = _5439.textures[4];
                _8812 = _5439.base_color[0];
                _8813 = _5439.base_color[1];
                _8814 = _5439.base_color[2];
                _8503 = _5439.flags;
                _8504 = _5439.type;
                _8505 = _5439.tangent_rotation_or_strength;
                _8506 = _5439.roughness_and_anisotropic;
                _8507 = _5439.int_ior;
                _8508 = _5439.ext_ior;
                _8509 = _5439.sheen_and_sheen_tint;
                _8510 = _5439.tint_and_metallic;
                _8511 = _5439.transmission_and_transmission_roughness;
                _8512 = _5439.specular_and_specular_tint;
                _8513 = _5439.clearcoat_and_clearcoat_roughness;
                _8514 = _5439.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_8807 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_8807, _5035, 0).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_8807 & 33554432u) != 0u)
            {
                float3 _9128 = normals;
                _9128.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _9128;
            }
            float3 _5521 = N;
            N = normalize(((T * normals.x) + (_5521 * normals.z)) + (B * normals.y));
            if ((_8514 & 65535u) != 65535u)
            {
                N = normalize(_5521 + ((N - _5521) * clamp(float(_8514 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_22 = plane_N;
            float3 param_23 = -_4349;
            float3 param_24 = N;
            N = ensure_valid_reflection(param_22, param_23, param_24);
        }
        float3 _5578 = ((_4953 * _4976) + (_4961 * inter.u)) + (_4969 * inter.v);
        float3 _5585 = float3(-_5578.z, 0.0f, _5578.x);
        float3 tangent = _5585;
        float3 param_25 = _5585;
        float4x4 param_26 = _4803.inv_xform;
        tangent = TransformNormal(param_25, param_26);
        if (_8505 != 0.0f)
        {
            float3 param_27 = tangent;
            float3 param_28 = N;
            float param_29 = _8505;
            tangent = rotate_around_axis(param_27, param_28, param_29);
        }
        float3 _5608 = normalize(cross(tangent, N));
        B = _5608;
        T = cross(N, _5608);
        float3 _8593 = 0.0f.xxx;
        float3 _8592 = 0.0f.xxx;
        float _8595 = 0.0f;
        float _8596 = 0.0f;
        float _8594 = 0.0f;
        bool _5620 = _3001_g_params.li_count != 0;
        bool _5626;
        if (_5620)
        {
            _5626 = _8504 != 3u;
        }
        else
        {
            _5626 = _5620;
        }
        float _8597;
        if (_5626)
        {
            float3 param_30 = _4423;
            float2 param_31 = float2(_5240, _5246);
            light_sample_t _8604 = { _8592, _8593, _8594, _8595, _8596, _8597 };
            light_sample_t param_32 = _8604;
            SampleLightSource(param_30, param_31, param_32);
            _8592 = param_32.col;
            _8593 = param_32.L;
            _8594 = param_32.area;
            _8595 = param_32.dist;
            _8596 = param_32.pdf;
            _8597 = param_32.cast_shadow;
        }
        float _5641 = dot(N, _8593);
        float3 base_color = float3(_8812, _8813, _8814);
        [branch]
        if (_8808 != 4294967295u)
        {
            base_color *= SampleBilinear(_8808, _5035, int(get_texture_lod(texSize(_8808), _5234)), true, true).xyz;
        }
        float3 tint_color = 0.0f.xxx;
        float _5685 = lum(base_color);
        [flatten]
        if (_5685 > 0.0f)
        {
            tint_color = base_color / _5685.xxx;
        }
        float roughness = clamp(float(_8506 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_8809 != 4294967295u)
        {
            roughness *= SampleBilinear(_8809, _5035, int(get_texture_lod(texSize(_8809), _5234)), false, true).x;
        }
        float _5726 = asfloat(_2996.Load((_3001_g_params.hi + 1) * 4 + 0));
        float _5730 = frac(_5726 + _5240);
        float _5736 = asfloat(_2996.Load((_3001_g_params.hi + 2) * 4 + 0));
        float _5740 = frac(_5736 + _5246);
        float _8878 = 0.0f;
        float _8877 = 0.0f;
        float _8876 = 0.0f;
        float _8625 = 0.0f;
        int _8630;
        float _8862;
        float _8863;
        float _8864;
        float _8869;
        float _8870;
        float _8871;
        [branch]
        if (_8504 == 0u)
        {
            [branch]
            if ((_8596 > 0.0f) && (_5641 > 0.0f))
            {
                float3 param_33 = -_4349;
                float3 param_34 = N;
                float3 param_35 = _8593;
                float param_36 = roughness;
                float3 param_37 = base_color;
                float4 _5780 = Evaluate_OrenDiffuse_BSDF(param_33, param_34, param_35, param_36, param_37);
                float mis_weight = 1.0f;
                if (_8594 > 0.0f)
                {
                    float param_38 = _8596;
                    float param_39 = _5780.w;
                    mis_weight = power_heuristic(param_38, param_39);
                }
                float3 _5808 = (_8592 * _5780.xyz) * ((mix_weight * mis_weight) / _8596);
                [branch]
                if (_8597 > 0.5f)
                {
                    float3 param_40 = _4423;
                    float3 param_41 = plane_N;
                    float3 _5819 = offset_ray(param_40, param_41);
                    uint _5873;
                    _5871.InterlockedAdd(8, 1u, _5873);
                    _5881.Store(_5873 * 44 + 0, asuint(_5819.x));
                    _5881.Store(_5873 * 44 + 4, asuint(_5819.y));
                    _5881.Store(_5873 * 44 + 8, asuint(_5819.z));
                    _5881.Store(_5873 * 44 + 12, asuint(_8593.x));
                    _5881.Store(_5873 * 44 + 16, asuint(_8593.y));
                    _5881.Store(_5873 * 44 + 20, asuint(_8593.z));
                    _5881.Store(_5873 * 44 + 24, asuint(_8595 - 9.9999997473787516355514526367188e-05f));
                    _5881.Store(_5873 * 44 + 28, asuint(ray.c[0] * _5808.x));
                    _5881.Store(_5873 * 44 + 32, asuint(ray.c[1] * _5808.y));
                    _5881.Store(_5873 * 44 + 36, asuint(ray.c[2] * _5808.z));
                    _5881.Store(_5873 * 44 + 40, uint(ray.xy));
                }
                else
                {
                    col += _5808;
                }
            }
            bool _5925 = _5253 < _3001_g_params.max_diff_depth;
            bool _5932;
            if (_5925)
            {
                _5932 = _5277 < _3001_g_params.max_total_depth;
            }
            else
            {
                _5932 = _5925;
            }
            [branch]
            if (_5932)
            {
                float3 param_42 = T;
                float3 param_43 = B;
                float3 param_44 = N;
                float3 param_45 = _4349;
                float param_46 = roughness;
                float3 param_47 = base_color;
                float param_48 = _5730;
                float param_49 = _5740;
                float3 param_50;
                float4 _5954 = Sample_OrenDiffuse_BSDF(param_42, param_43, param_44, param_45, param_46, param_47, param_48, param_49, param_50);
                _8630 = ray.ray_depth + 1;
                float3 param_51 = _4423;
                float3 param_52 = plane_N;
                float3 _5965 = offset_ray(param_51, param_52);
                _8862 = _5965.x;
                _8863 = _5965.y;
                _8864 = _5965.z;
                _8869 = param_50.x;
                _8870 = param_50.y;
                _8871 = param_50.z;
                _8876 = ((ray.c[0] * _5954.x) * mix_weight) / _5954.w;
                _8877 = ((ray.c[1] * _5954.y) * mix_weight) / _5954.w;
                _8878 = ((ray.c[2] * _5954.z) * mix_weight) / _5954.w;
                _8625 = _5954.w;
            }
        }
        else
        {
            [branch]
            if (_8504 == 1u)
            {
                float param_53 = 1.0f;
                float param_54 = 1.5f;
                float _6030 = fresnel_dielectric_cos(param_53, param_54);
                float _6034 = roughness * roughness;
                bool _6037 = _8596 > 0.0f;
                bool _6044;
                if (_6037)
                {
                    _6044 = (_6034 * _6034) >= 1.0000000116860974230803549289703e-07f;
                }
                else
                {
                    _6044 = _6037;
                }
                [branch]
                if (_6044 && (_5641 > 0.0f))
                {
                    float3 param_55 = T;
                    float3 param_56 = B;
                    float3 param_57 = N;
                    float3 param_58 = -_4349;
                    float3 param_59 = T;
                    float3 param_60 = B;
                    float3 param_61 = N;
                    float3 param_62 = _8593;
                    float3 param_63 = T;
                    float3 param_64 = B;
                    float3 param_65 = N;
                    float3 param_66 = normalize(_8593 - _4349);
                    float3 param_67 = tangent_from_world(param_55, param_56, param_57, param_58);
                    float3 param_68 = tangent_from_world(param_63, param_64, param_65, param_66);
                    float3 param_69 = tangent_from_world(param_59, param_60, param_61, param_62);
                    float param_70 = _6034;
                    float param_71 = _6034;
                    float param_72 = 1.5f;
                    float param_73 = _6030;
                    float3 param_74 = base_color;
                    float4 _6104 = Evaluate_GGXSpecular_BSDF(param_67, param_68, param_69, param_70, param_71, param_72, param_73, param_74);
                    float mis_weight_1 = 1.0f;
                    if (_8594 > 0.0f)
                    {
                        float param_75 = _8596;
                        float param_76 = _6104.w;
                        mis_weight_1 = power_heuristic(param_75, param_76);
                    }
                    float3 _6132 = (_8592 * _6104.xyz) * ((mix_weight * mis_weight_1) / _8596);
                    [branch]
                    if (_8597 > 0.5f)
                    {
                        float3 param_77 = _4423;
                        float3 param_78 = plane_N;
                        float3 _6143 = offset_ray(param_77, param_78);
                        uint _6190;
                        _5871.InterlockedAdd(8, 1u, _6190);
                        _5881.Store(_6190 * 44 + 0, asuint(_6143.x));
                        _5881.Store(_6190 * 44 + 4, asuint(_6143.y));
                        _5881.Store(_6190 * 44 + 8, asuint(_6143.z));
                        _5881.Store(_6190 * 44 + 12, asuint(_8593.x));
                        _5881.Store(_6190 * 44 + 16, asuint(_8593.y));
                        _5881.Store(_6190 * 44 + 20, asuint(_8593.z));
                        _5881.Store(_6190 * 44 + 24, asuint(_8595 - 9.9999997473787516355514526367188e-05f));
                        _5881.Store(_6190 * 44 + 28, asuint(ray.c[0] * _6132.x));
                        _5881.Store(_6190 * 44 + 32, asuint(ray.c[1] * _6132.y));
                        _5881.Store(_6190 * 44 + 36, asuint(ray.c[2] * _6132.z));
                        _5881.Store(_6190 * 44 + 40, uint(ray.xy));
                    }
                    else
                    {
                        col += _6132;
                    }
                }
                bool _6229 = _5258 < _3001_g_params.max_spec_depth;
                bool _6236;
                if (_6229)
                {
                    _6236 = _5277 < _3001_g_params.max_total_depth;
                }
                else
                {
                    _6236 = _6229;
                }
                [branch]
                if (_6236)
                {
                    float3 param_79 = T;
                    float3 param_80 = B;
                    float3 param_81 = N;
                    float3 param_82 = _4349;
                    float3 param_83;
                    float4 _6255 = Sample_GGXSpecular_BSDF(param_79, param_80, param_81, param_82, roughness, 0.0f, 1.5f, _6030, base_color, _5730, _5740, param_83);
                    _8630 = ray.ray_depth + 256;
                    float3 param_84 = _4423;
                    float3 param_85 = plane_N;
                    float3 _6267 = offset_ray(param_84, param_85);
                    _8862 = _6267.x;
                    _8863 = _6267.y;
                    _8864 = _6267.z;
                    _8869 = param_83.x;
                    _8870 = param_83.y;
                    _8871 = param_83.z;
                    _8876 = ((ray.c[0] * _6255.x) * mix_weight) / _6255.w;
                    _8877 = ((ray.c[1] * _6255.y) * mix_weight) / _6255.w;
                    _8878 = ((ray.c[2] * _6255.z) * mix_weight) / _6255.w;
                    _8625 = _6255.w;
                }
            }
            else
            {
                [branch]
                if (_8504 == 2u)
                {
                    float _6330;
                    if (_4724)
                    {
                        _6330 = _8507 / _8508;
                    }
                    else
                    {
                        _6330 = _8508 / _8507;
                    }
                    float _6348 = roughness * roughness;
                    bool _6351 = _8596 > 0.0f;
                    bool _6358;
                    if (_6351)
                    {
                        _6358 = (_6348 * _6348) >= 1.0000000116860974230803549289703e-07f;
                    }
                    else
                    {
                        _6358 = _6351;
                    }
                    [branch]
                    if (_6358 && (_5641 < 0.0f))
                    {
                        float3 param_86 = T;
                        float3 param_87 = B;
                        float3 param_88 = N;
                        float3 param_89 = -_4349;
                        float3 param_90 = T;
                        float3 param_91 = B;
                        float3 param_92 = N;
                        float3 param_93 = _8593;
                        float3 param_94 = T;
                        float3 param_95 = B;
                        float3 param_96 = N;
                        float3 param_97 = normalize(_8593 - (_4349 * _6330));
                        float3 param_98 = tangent_from_world(param_86, param_87, param_88, param_89);
                        float3 param_99 = tangent_from_world(param_94, param_95, param_96, param_97);
                        float3 param_100 = tangent_from_world(param_90, param_91, param_92, param_93);
                        float param_101 = _6348;
                        float param_102 = _6330;
                        float3 param_103 = base_color;
                        float4 _6417 = Evaluate_GGXRefraction_BSDF(param_98, param_99, param_100, param_101, param_102, param_103);
                        float mis_weight_2 = 1.0f;
                        if (_8594 > 0.0f)
                        {
                            float param_104 = _8596;
                            float param_105 = _6417.w;
                            mis_weight_2 = power_heuristic(param_104, param_105);
                        }
                        float3 _6445 = (_8592 * _6417.xyz) * ((mix_weight * mis_weight_2) / _8596);
                        [branch]
                        if (_8597 > 0.5f)
                        {
                            float3 param_106 = _4423;
                            float3 param_107 = -plane_N;
                            float3 _6457 = offset_ray(param_106, param_107);
                            uint _6504;
                            _5871.InterlockedAdd(8, 1u, _6504);
                            _5881.Store(_6504 * 44 + 0, asuint(_6457.x));
                            _5881.Store(_6504 * 44 + 4, asuint(_6457.y));
                            _5881.Store(_6504 * 44 + 8, asuint(_6457.z));
                            _5881.Store(_6504 * 44 + 12, asuint(_8593.x));
                            _5881.Store(_6504 * 44 + 16, asuint(_8593.y));
                            _5881.Store(_6504 * 44 + 20, asuint(_8593.z));
                            _5881.Store(_6504 * 44 + 24, asuint(_8595 - 9.9999997473787516355514526367188e-05f));
                            _5881.Store(_6504 * 44 + 28, asuint(ray.c[0] * _6445.x));
                            _5881.Store(_6504 * 44 + 32, asuint(ray.c[1] * _6445.y));
                            _5881.Store(_6504 * 44 + 36, asuint(ray.c[2] * _6445.z));
                            _5881.Store(_6504 * 44 + 40, uint(ray.xy));
                        }
                        else
                        {
                            col += _6445;
                        }
                    }
                    bool _6543 = _5263 < _3001_g_params.max_refr_depth;
                    bool _6550;
                    if (_6543)
                    {
                        _6550 = _5277 < _3001_g_params.max_total_depth;
                    }
                    else
                    {
                        _6550 = _6543;
                    }
                    [branch]
                    if (_6550)
                    {
                        float3 param_108 = T;
                        float3 param_109 = B;
                        float3 param_110 = N;
                        float3 param_111 = _4349;
                        float param_112 = roughness;
                        float param_113 = _6330;
                        float3 param_114 = base_color;
                        float param_115 = _5730;
                        float param_116 = _5740;
                        float4 param_117;
                        float4 _6574 = Sample_GGXRefraction_BSDF(param_108, param_109, param_110, param_111, param_112, param_113, param_114, param_115, param_116, param_117);
                        _8630 = ray.ray_depth + 65536;
                        _8876 = ((ray.c[0] * _6574.x) * mix_weight) / _6574.w;
                        _8877 = ((ray.c[1] * _6574.y) * mix_weight) / _6574.w;
                        _8878 = ((ray.c[2] * _6574.z) * mix_weight) / _6574.w;
                        _8625 = _6574.w;
                        float3 param_118 = _4423;
                        float3 param_119 = -plane_N;
                        float3 _6629 = offset_ray(param_118, param_119);
                        _8862 = _6629.x;
                        _8863 = _6629.y;
                        _8864 = _6629.z;
                        _8869 = param_117.x;
                        _8870 = param_117.y;
                        _8871 = param_117.z;
                    }
                }
                else
                {
                    [branch]
                    if (_8504 == 3u)
                    {
                        if ((_8503 & 4u) != 0u)
                        {
                            float3 env_col_1 = _3001_g_params.env_col.xyz;
                            uint _6668 = asuint(_3001_g_params.env_col.w);
                            if (_6668 != 4294967295u)
                            {
                                env_col_1 *= SampleLatlong_RGBE(_6668, _4349, _3001_g_params.env_rotation);
                            }
                            base_color *= env_col_1;
                        }
                        col += (base_color * (mix_weight * _8505));
                    }
                    else
                    {
                        [branch]
                        if (_8504 == 5u)
                        {
                            bool _6702 = _5269 < _3001_g_params.max_transp_depth;
                            bool _6709;
                            if (_6702)
                            {
                                _6709 = _5277 < _3001_g_params.max_total_depth;
                            }
                            else
                            {
                                _6709 = _6702;
                            }
                            [branch]
                            if (_6709)
                            {
                                _8630 = ray.ray_depth + 16777216;
                                _8625 = ray.pdf;
                                float3 param_120 = _4423;
                                float3 param_121 = -plane_N;
                                float3 _6726 = offset_ray(param_120, param_121);
                                _8862 = _6726.x;
                                _8863 = _6726.y;
                                _8864 = _6726.z;
                                _8869 = ray.d[0];
                                _8870 = ray.d[1];
                                _8871 = ray.d[2];
                                _8876 = ray.c[0];
                                _8877 = ray.c[1];
                                _8878 = ray.c[2];
                            }
                        }
                        else
                        {
                            if (_8504 == 6u)
                            {
                                float metallic = clamp(float((_8510 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                [branch]
                                if (_8810 != 4294967295u)
                                {
                                    metallic *= SampleBilinear(_8810, _5035, int(get_texture_lod(texSize(_8810), _5234))).x;
                                }
                                float specular = clamp(float(_8512 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                [branch]
                                if (_8811 != 4294967295u)
                                {
                                    specular *= SampleBilinear(_8811, _5035, int(get_texture_lod(texSize(_8811), _5234))).x;
                                }
                                float _6836 = clamp(float(_8513 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                float _6844 = clamp(float((_8513 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                float _6851 = clamp(float(_8509 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                float3 _6873 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_8512 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                                float _6880 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                                float param_122 = 1.0f;
                                float param_123 = _6880;
                                float _6885 = fresnel_dielectric_cos(param_122, param_123);
                                float param_124 = dot(_4349, N);
                                float param_125 = _6880;
                                float param_126;
                                float param_127;
                                float param_128;
                                float param_129;
                                get_lobe_weights(lerp(_5685, 1.0f, _6851), lum(lerp(_6873, 1.0f.xxx, ((fresnel_dielectric_cos(param_124, param_125) - _6885) / (1.0f - _6885)).xxx)), specular, metallic, clamp(float(_8511 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _6836, param_126, param_127, param_128, param_129);
                                float3 _6939 = lerp(1.0f.xxx, tint_color, clamp(float((_8509 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _6851;
                                float _6942;
                                if (_4724)
                                {
                                    _6942 = _8507 / _8508;
                                }
                                else
                                {
                                    _6942 = _8508 / _8507;
                                }
                                float param_130 = dot(_4349, N);
                                float param_131 = 1.0f / _6942;
                                float _6965 = fresnel_dielectric_cos(param_130, param_131);
                                float _6972 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _6836))) - 1.0f;
                                float param_132 = 1.0f;
                                float param_133 = _6972;
                                float _6977 = fresnel_dielectric_cos(param_132, param_133);
                                float _6981 = _6844 * _6844;
                                float _6994 = mad(roughness - 1.0f, 1.0f - clamp(float((_8511 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                                float _6998 = _6994 * _6994;
                                [branch]
                                if (_8596 > 0.0f)
                                {
                                    float3 lcol_1 = 0.0f.xxx;
                                    float bsdf_pdf = 0.0f;
                                    bool _7009 = _5641 > 0.0f;
                                    [branch]
                                    if ((param_126 > 1.0000000116860974230803549289703e-07f) && _7009)
                                    {
                                        float3 param_134 = -_4349;
                                        float3 param_135 = N;
                                        float3 param_136 = _8593;
                                        float param_137 = roughness;
                                        float3 param_138 = base_color.xyz;
                                        float3 param_139 = _6939;
                                        bool param_140 = false;
                                        float4 _7029 = Evaluate_PrincipledDiffuse_BSDF(param_134, param_135, param_136, param_137, param_138, param_139, param_140);
                                        bsdf_pdf = mad(param_126, _7029.w, bsdf_pdf);
                                        lcol_1 += (((_8592 * _5641) * (_7029 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * _8596).xxx);
                                    }
                                    float3 H;
                                    [flatten]
                                    if (_7009)
                                    {
                                        H = normalize(_8593 - _4349);
                                    }
                                    else
                                    {
                                        H = normalize(_8593 - (_4349 * _6942));
                                    }
                                    float _7075 = roughness * roughness;
                                    float _7086 = sqrt(mad(-0.89999997615814208984375f, clamp(float((_8506 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f));
                                    float _7090 = _7075 / _7086;
                                    float _7094 = _7075 * _7086;
                                    float3 param_141 = T;
                                    float3 param_142 = B;
                                    float3 param_143 = N;
                                    float3 param_144 = -_4349;
                                    float3 _7105 = tangent_from_world(param_141, param_142, param_143, param_144);
                                    float3 param_145 = T;
                                    float3 param_146 = B;
                                    float3 param_147 = N;
                                    float3 param_148 = _8593;
                                    float3 _7116 = tangent_from_world(param_145, param_146, param_147, param_148);
                                    float3 param_149 = T;
                                    float3 param_150 = B;
                                    float3 param_151 = N;
                                    float3 param_152 = H;
                                    float3 _7126 = tangent_from_world(param_149, param_150, param_151, param_152);
                                    bool _7128 = param_127 > 0.0f;
                                    bool _7135;
                                    if (_7128)
                                    {
                                        _7135 = (_7090 * _7094) >= 1.0000000116860974230803549289703e-07f;
                                    }
                                    else
                                    {
                                        _7135 = _7128;
                                    }
                                    [branch]
                                    if (_7135 && _7009)
                                    {
                                        float3 param_153 = _7105;
                                        float3 param_154 = _7126;
                                        float3 param_155 = _7116;
                                        float param_156 = _7090;
                                        float param_157 = _7094;
                                        float param_158 = _6880;
                                        float param_159 = _6885;
                                        float3 param_160 = _6873;
                                        float4 _7158 = Evaluate_GGXSpecular_BSDF(param_153, param_154, param_155, param_156, param_157, param_158, param_159, param_160);
                                        bsdf_pdf = mad(param_127, _7158.w, bsdf_pdf);
                                        lcol_1 += ((_8592 * _7158.xyz) / _8596.xxx);
                                    }
                                    bool _7177 = param_128 > 0.0f;
                                    bool _7184;
                                    if (_7177)
                                    {
                                        _7184 = (_6981 * _6981) >= 1.0000000116860974230803549289703e-07f;
                                    }
                                    else
                                    {
                                        _7184 = _7177;
                                    }
                                    [branch]
                                    if (_7184 && _7009)
                                    {
                                        float3 param_161 = _7105;
                                        float3 param_162 = _7126;
                                        float3 param_163 = _7116;
                                        float param_164 = _6981;
                                        float param_165 = _6972;
                                        float param_166 = _6977;
                                        float4 _7203 = Evaluate_PrincipledClearcoat_BSDF(param_161, param_162, param_163, param_164, param_165, param_166);
                                        bsdf_pdf = mad(param_128, _7203.w, bsdf_pdf);
                                        lcol_1 += (((_8592 * 0.25f) * _7203.xyz) / _8596.xxx);
                                    }
                                    [branch]
                                    if (param_129 > 0.0f)
                                    {
                                        bool _7227 = _6965 != 0.0f;
                                        bool _7234;
                                        if (_7227)
                                        {
                                            _7234 = (_7075 * _7075) >= 1.0000000116860974230803549289703e-07f;
                                        }
                                        else
                                        {
                                            _7234 = _7227;
                                        }
                                        [branch]
                                        if (_7234 && _7009)
                                        {
                                            float3 param_167 = _7105;
                                            float3 param_168 = _7126;
                                            float3 param_169 = _7116;
                                            float param_170 = _7075;
                                            float param_171 = _7075;
                                            float param_172 = 1.0f;
                                            float param_173 = 0.0f;
                                            float3 param_174 = 1.0f.xxx;
                                            float4 _7254 = Evaluate_GGXSpecular_BSDF(param_167, param_168, param_169, param_170, param_171, param_172, param_173, param_174);
                                            bsdf_pdf = mad(param_129 * _6965, _7254.w, bsdf_pdf);
                                            lcol_1 += ((_8592 * _7254.xyz) * (_6965 / _8596));
                                        }
                                        bool _7276 = _6965 != 1.0f;
                                        bool _7283;
                                        if (_7276)
                                        {
                                            _7283 = (_6998 * _6998) >= 1.0000000116860974230803549289703e-07f;
                                        }
                                        else
                                        {
                                            _7283 = _7276;
                                        }
                                        [branch]
                                        if (_7283 && (_5641 < 0.0f))
                                        {
                                            float3 param_175 = _7105;
                                            float3 param_176 = _7126;
                                            float3 param_177 = _7116;
                                            float param_178 = _6998;
                                            float param_179 = _6942;
                                            float3 param_180 = base_color;
                                            float4 _7302 = Evaluate_GGXRefraction_BSDF(param_175, param_176, param_177, param_178, param_179, param_180);
                                            float _7305 = 1.0f - _6965;
                                            bsdf_pdf = mad(param_129 * _7305, _7302.w, bsdf_pdf);
                                            lcol_1 += ((_8592 * _7302.xyz) * (_7305 / _8596));
                                        }
                                    }
                                    float mis_weight_3 = 1.0f;
                                    [flatten]
                                    if (_8594 > 0.0f)
                                    {
                                        float param_181 = _8596;
                                        float param_182 = bsdf_pdf;
                                        mis_weight_3 = power_heuristic(param_181, param_182);
                                    }
                                    lcol_1 *= (mix_weight * mis_weight_3);
                                    [branch]
                                    if (_8597 > 0.5f)
                                    {
                                        float3 _7350;
                                        if (_5641 < 0.0f)
                                        {
                                            _7350 = -plane_N;
                                        }
                                        else
                                        {
                                            _7350 = plane_N;
                                        }
                                        float3 param_183 = _4423;
                                        float3 param_184 = _7350;
                                        float3 _7361 = offset_ray(param_183, param_184);
                                        uint _7408;
                                        _5871.InterlockedAdd(8, 1u, _7408);
                                        _5881.Store(_7408 * 44 + 0, asuint(_7361.x));
                                        _5881.Store(_7408 * 44 + 4, asuint(_7361.y));
                                        _5881.Store(_7408 * 44 + 8, asuint(_7361.z));
                                        _5881.Store(_7408 * 44 + 12, asuint(_8593.x));
                                        _5881.Store(_7408 * 44 + 16, asuint(_8593.y));
                                        _5881.Store(_7408 * 44 + 20, asuint(_8593.z));
                                        _5881.Store(_7408 * 44 + 24, asuint(_8595 - 9.9999997473787516355514526367188e-05f));
                                        _5881.Store(_7408 * 44 + 28, asuint(ray.c[0] * lcol_1.x));
                                        _5881.Store(_7408 * 44 + 32, asuint(ray.c[1] * lcol_1.y));
                                        _5881.Store(_7408 * 44 + 36, asuint(ray.c[2] * lcol_1.z));
                                        _5881.Store(_7408 * 44 + 40, uint(ray.xy));
                                    }
                                    else
                                    {
                                        col += lcol_1;
                                    }
                                }
                                [branch]
                                if (mix_rand < param_126)
                                {
                                    bool _7452 = _5253 < _3001_g_params.max_diff_depth;
                                    bool _7459;
                                    if (_7452)
                                    {
                                        _7459 = _5277 < _3001_g_params.max_total_depth;
                                    }
                                    else
                                    {
                                        _7459 = _7452;
                                    }
                                    if (_7459)
                                    {
                                        float3 param_185 = T;
                                        float3 param_186 = B;
                                        float3 param_187 = N;
                                        float3 param_188 = _4349;
                                        float param_189 = roughness;
                                        float3 param_190 = base_color.xyz;
                                        float3 param_191 = _6939;
                                        bool param_192 = false;
                                        float param_193 = _5730;
                                        float param_194 = _5740;
                                        float3 param_195;
                                        float4 _7484 = Sample_PrincipledDiffuse_BSDF(param_185, param_186, param_187, param_188, param_189, param_190, param_191, param_192, param_193, param_194, param_195);
                                        float3 _7490 = _7484.xyz * (1.0f - metallic);
                                        _8630 = ray.ray_depth + 1;
                                        float3 param_196 = _4423;
                                        float3 param_197 = plane_N;
                                        float3 _7506 = offset_ray(param_196, param_197);
                                        _8862 = _7506.x;
                                        _8863 = _7506.y;
                                        _8864 = _7506.z;
                                        _8869 = param_195.x;
                                        _8870 = param_195.y;
                                        _8871 = param_195.z;
                                        _8876 = ((ray.c[0] * _7490.x) * mix_weight) / param_126;
                                        _8877 = ((ray.c[1] * _7490.y) * mix_weight) / param_126;
                                        _8878 = ((ray.c[2] * _7490.z) * mix_weight) / param_126;
                                        _8625 = _7484.w;
                                    }
                                }
                                else
                                {
                                    float _7562 = param_126 + param_127;
                                    [branch]
                                    if (mix_rand < _7562)
                                    {
                                        bool _7569 = _5258 < _3001_g_params.max_spec_depth;
                                        bool _7576;
                                        if (_7569)
                                        {
                                            _7576 = _5277 < _3001_g_params.max_total_depth;
                                        }
                                        else
                                        {
                                            _7576 = _7569;
                                        }
                                        if (_7576)
                                        {
                                            float3 param_198 = T;
                                            float3 param_199 = B;
                                            float3 param_200 = N;
                                            float3 param_201 = _4349;
                                            float3 param_202;
                                            float4 _7603 = Sample_GGXSpecular_BSDF(param_198, param_199, param_200, param_201, roughness, clamp(float((_8506 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _6880, _6885, _6873, _5730, _5740, param_202);
                                            float _7608 = _7603.w * param_127;
                                            _8630 = ray.ray_depth + 256;
                                            _8876 = ((ray.c[0] * _7603.x) * mix_weight) / _7608;
                                            _8877 = ((ray.c[1] * _7603.y) * mix_weight) / _7608;
                                            _8878 = ((ray.c[2] * _7603.z) * mix_weight) / _7608;
                                            _8625 = _7608;
                                            float3 param_203 = _4423;
                                            float3 param_204 = plane_N;
                                            float3 _7655 = offset_ray(param_203, param_204);
                                            _8862 = _7655.x;
                                            _8863 = _7655.y;
                                            _8864 = _7655.z;
                                            _8869 = param_202.x;
                                            _8870 = param_202.y;
                                            _8871 = param_202.z;
                                        }
                                    }
                                    else
                                    {
                                        float _7680 = _7562 + param_128;
                                        [branch]
                                        if (mix_rand < _7680)
                                        {
                                            bool _7687 = _5258 < _3001_g_params.max_spec_depth;
                                            bool _7694;
                                            if (_7687)
                                            {
                                                _7694 = _5277 < _3001_g_params.max_total_depth;
                                            }
                                            else
                                            {
                                                _7694 = _7687;
                                            }
                                            if (_7694)
                                            {
                                                float3 param_205 = T;
                                                float3 param_206 = B;
                                                float3 param_207 = N;
                                                float3 param_208 = _4349;
                                                float param_209 = _6981;
                                                float param_210 = _6972;
                                                float param_211 = _6977;
                                                float param_212 = _5730;
                                                float param_213 = _5740;
                                                float3 param_214;
                                                float4 _7718 = Sample_PrincipledClearcoat_BSDF(param_205, param_206, param_207, param_208, param_209, param_210, param_211, param_212, param_213, param_214);
                                                float _7723 = _7718.w * param_128;
                                                _8630 = ray.ray_depth + 256;
                                                _8876 = (((0.25f * ray.c[0]) * _7718.x) * mix_weight) / _7723;
                                                _8877 = (((0.25f * ray.c[1]) * _7718.y) * mix_weight) / _7723;
                                                _8878 = (((0.25f * ray.c[2]) * _7718.z) * mix_weight) / _7723;
                                                _8625 = _7723;
                                                float3 param_215 = _4423;
                                                float3 param_216 = plane_N;
                                                float3 _7773 = offset_ray(param_215, param_216);
                                                _8862 = _7773.x;
                                                _8863 = _7773.y;
                                                _8864 = _7773.z;
                                                _8869 = param_214.x;
                                                _8870 = param_214.y;
                                                _8871 = param_214.z;
                                            }
                                        }
                                        else
                                        {
                                            bool _7795 = mix_rand >= _6965;
                                            bool _7802;
                                            if (_7795)
                                            {
                                                _7802 = _5263 < _3001_g_params.max_refr_depth;
                                            }
                                            else
                                            {
                                                _7802 = _7795;
                                            }
                                            bool _7816;
                                            if (!_7802)
                                            {
                                                bool _7808 = mix_rand < _6965;
                                                bool _7815;
                                                if (_7808)
                                                {
                                                    _7815 = _5258 < _3001_g_params.max_spec_depth;
                                                }
                                                else
                                                {
                                                    _7815 = _7808;
                                                }
                                                _7816 = _7815;
                                            }
                                            else
                                            {
                                                _7816 = _7802;
                                            }
                                            bool _7823;
                                            if (_7816)
                                            {
                                                _7823 = _5277 < _3001_g_params.max_total_depth;
                                            }
                                            else
                                            {
                                                _7823 = _7816;
                                            }
                                            [branch]
                                            if (_7823)
                                            {
                                                float _7831 = mix_rand;
                                                float _7835 = (_7831 - _7680) / param_129;
                                                mix_rand = _7835;
                                                float4 F;
                                                float3 V;
                                                [branch]
                                                if (_7835 < _6965)
                                                {
                                                    float3 param_217 = T;
                                                    float3 param_218 = B;
                                                    float3 param_219 = N;
                                                    float3 param_220 = _4349;
                                                    float3 param_221;
                                                    float4 _7855 = Sample_GGXSpecular_BSDF(param_217, param_218, param_219, param_220, roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, _5730, _5740, param_221);
                                                    V = param_221;
                                                    F = _7855;
                                                    _8630 = ray.ray_depth + 256;
                                                    float3 param_222 = _4423;
                                                    float3 param_223 = plane_N;
                                                    float3 _7866 = offset_ray(param_222, param_223);
                                                    _8862 = _7866.x;
                                                    _8863 = _7866.y;
                                                    _8864 = _7866.z;
                                                }
                                                else
                                                {
                                                    float3 param_224 = T;
                                                    float3 param_225 = B;
                                                    float3 param_226 = N;
                                                    float3 param_227 = _4349;
                                                    float param_228 = _6994;
                                                    float param_229 = _6942;
                                                    float3 param_230 = base_color;
                                                    float param_231 = _5730;
                                                    float param_232 = _5740;
                                                    float4 param_233;
                                                    float4 _7897 = Sample_GGXRefraction_BSDF(param_224, param_225, param_226, param_227, param_228, param_229, param_230, param_231, param_232, param_233);
                                                    F = _7897;
                                                    V = param_233.xyz;
                                                    _8630 = ray.ray_depth + 65536;
                                                    float3 param_234 = _4423;
                                                    float3 param_235 = -plane_N;
                                                    float3 _7911 = offset_ray(param_234, param_235);
                                                    _8862 = _7911.x;
                                                    _8863 = _7911.y;
                                                    _8864 = _7911.z;
                                                }
                                                float4 _9276 = F;
                                                float _7924 = _9276.w * param_129;
                                                float4 _9278 = _9276;
                                                _9278.w = _7924;
                                                F = _9278;
                                                _8876 = ((ray.c[0] * _9276.x) * mix_weight) / _7924;
                                                _8877 = ((ray.c[1] * _9276.y) * mix_weight) / _7924;
                                                _8878 = ((ray.c[2] * _9276.z) * mix_weight) / _7924;
                                                _8625 = _7924;
                                                _8869 = V.x;
                                                _8870 = V.y;
                                                _8871 = V.z;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        float _7984 = max(_8876, max(_8877, _8878));
        float _7997;
        if (_5277 >= _3001_g_params.termination_start_depth)
        {
            _7997 = max(0.0500000007450580596923828125f, 1.0f - _7984);
        }
        else
        {
            _7997 = 0.0f;
        }
        bool _8011 = (frac(asfloat(_2996.Load((_3001_g_params.hi + 6) * 4 + 0)) + _5240) >= _7997) && (_7984 > 0.0f);
        bool _8017;
        if (_8011)
        {
            _8017 = _8625 > 0.0f;
        }
        else
        {
            _8017 = _8011;
        }
        [branch]
        if (_8017)
        {
            float _8021 = 1.0f - _7997;
            float _8023 = _8876;
            float _8024 = _8023 / _8021;
            _8876 = _8024;
            float _8029 = _8877;
            float _8030 = _8029 / _8021;
            _8877 = _8030;
            float _8035 = _8878;
            float _8036 = _8035 / _8021;
            _8878 = _8036;
            uint _8040;
            _5871.InterlockedAdd(0, 1u, _8040);
            _8048.Store(_8040 * 56 + 0, asuint(_8862));
            _8048.Store(_8040 * 56 + 4, asuint(_8863));
            _8048.Store(_8040 * 56 + 8, asuint(_8864));
            _8048.Store(_8040 * 56 + 12, asuint(_8869));
            _8048.Store(_8040 * 56 + 16, asuint(_8870));
            _8048.Store(_8040 * 56 + 20, asuint(_8871));
            _8048.Store(_8040 * 56 + 24, asuint(_8625));
            _8048.Store(_8040 * 56 + 28, asuint(_8024));
            _8048.Store(_8040 * 56 + 32, asuint(_8030));
            _8048.Store(_8040 * 56 + 36, asuint(_8036));
            _8048.Store(_8040 * 56 + 40, asuint(_5224));
            _8048.Store(_8040 * 56 + 44, asuint(ray.cone_spread));
            _8048.Store(_8040 * 56 + 48, uint(ray.xy));
            _8048.Store(_8040 * 56 + 52, uint(_8630));
        }
        _8256 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _8256;
}

void comp_main()
{
    do
    {
        bool _8116 = gl_GlobalInvocationID.x >= _3001_g_params.img_size.x;
        bool _8125;
        if (!_8116)
        {
            _8125 = gl_GlobalInvocationID.y >= _3001_g_params.img_size.y;
        }
        else
        {
            _8125 = _8116;
        }
        if (_8125)
        {
            break;
        }
        int _8132 = int(gl_GlobalInvocationID.x);
        int _8136 = int(gl_GlobalInvocationID.y);
        int _8144 = (_8136 * int(_3001_g_params.img_size.x)) + _8132;
        hit_data_t _8156;
        _8156.mask = int(_8152.Load(_8144 * 24 + 0));
        _8156.obj_index = int(_8152.Load(_8144 * 24 + 4));
        _8156.prim_index = int(_8152.Load(_8144 * 24 + 8));
        _8156.t = asfloat(_8152.Load(_8144 * 24 + 12));
        _8156.u = asfloat(_8152.Load(_8144 * 24 + 16));
        _8156.v = asfloat(_8152.Load(_8144 * 24 + 20));
        ray_data_t _8176;
        [unroll]
        for (int _83ident = 0; _83ident < 3; _83ident++)
        {
            _8176.o[_83ident] = asfloat(_8173.Load(_83ident * 4 + _8144 * 56 + 0));
        }
        [unroll]
        for (int _84ident = 0; _84ident < 3; _84ident++)
        {
            _8176.d[_84ident] = asfloat(_8173.Load(_84ident * 4 + _8144 * 56 + 12));
        }
        _8176.pdf = asfloat(_8173.Load(_8144 * 56 + 24));
        [unroll]
        for (int _85ident = 0; _85ident < 3; _85ident++)
        {
            _8176.c[_85ident] = asfloat(_8173.Load(_85ident * 4 + _8144 * 56 + 28));
        }
        _8176.cone_width = asfloat(_8173.Load(_8144 * 56 + 40));
        _8176.cone_spread = asfloat(_8173.Load(_8144 * 56 + 44));
        _8176.xy = int(_8173.Load(_8144 * 56 + 48));
        _8176.ray_depth = int(_8173.Load(_8144 * 56 + 52));
        int param = _8144;
        hit_data_t _8317 = { _8156.mask, _8156.obj_index, _8156.prim_index, _8156.t, _8156.u, _8156.v };
        hit_data_t param_1 = _8317;
        float _8355[3] = { _8176.c[0], _8176.c[1], _8176.c[2] };
        float _8348[3] = { _8176.d[0], _8176.d[1], _8176.d[2] };
        float _8341[3] = { _8176.o[0], _8176.o[1], _8176.o[2] };
        ray_data_t _8334 = { _8341, _8348, _8176.pdf, _8355, _8176.cone_width, _8176.cone_spread, _8176.xy, _8176.ray_depth };
        ray_data_t param_2 = _8334;
        float3 _8218 = ShadeSurface(param, param_1, param_2);
        g_out_img[int2(_8132, _8136)] = float4(_8218, 1.0f);
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
