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

ByteAddressBuffer _2995 : register(t15, space0);
ByteAddressBuffer _3032 : register(t6, space0);
ByteAddressBuffer _3036 : register(t7, space0);
ByteAddressBuffer _3769 : register(t11, space0);
ByteAddressBuffer _3794 : register(t13, space0);
ByteAddressBuffer _3798 : register(t14, space0);
ByteAddressBuffer _4109 : register(t10, space0);
ByteAddressBuffer _4113 : register(t9, space0);
ByteAddressBuffer _4720 : register(t12, space0);
RWByteAddressBuffer _5795 : register(u3, space0);
RWByteAddressBuffer _5805 : register(u2, space0);
RWByteAddressBuffer _7972 : register(u1, space0);
ByteAddressBuffer _8076 : register(t4, space0);
ByteAddressBuffer _8097 : register(t5, space0);
ByteAddressBuffer _8173 : register(t8, space0);
cbuffer UniformParams
{
    Params _3000_g_params : packoffset(c0);
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
    float u = phi * 0.15915493667125701904296875f;
    [flatten]
    if (dir.z < 0.0f)
    {
        u = 1.0f - u;
    }
    uint _951 = index & 16777215u;
    uint _957_dummy_parameter;
    float2 _964 = float2(u, acos(clamp(dir.y, -1.0f, 1.0f)) * 0.3183098733425140380859375f) * float2(int2(spvTextureSize(g_textures[_951], uint(0), _957_dummy_parameter)));
    uint _967 = _951;
    int2 _971 = int2(_964);
    float2 _1006 = frac(_964);
    float4 param = g_textures[NonUniformResourceIndex(_967)].Load(int3(_971, 0), int2(0, 0));
    float4 param_1 = g_textures[NonUniformResourceIndex(_967)].Load(int3(_971, 0), int2(1, 0));
    float4 param_2 = g_textures[NonUniformResourceIndex(_967)].Load(int3(_971, 0), int2(0, 1));
    float4 param_3 = g_textures[NonUniformResourceIndex(_967)].Load(int3(_971, 0), int2(1, 1));
    float _1026 = _1006.x;
    float _1031 = 1.0f - _1026;
    float _1047 = _1006.y;
    return (((rgbe_to_rgb(param_3) * _1026) + (rgbe_to_rgb(param_2) * _1031)) * _1047) + (((rgbe_to_rgb(param_1) * _1026) + (rgbe_to_rgb(param) * _1031)) * (1.0f - _1047));
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
    float _1060 = a * a;
    return _1060 / mad(b, b, _1060);
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
        float4 _8918 = res;
        _8918.x = _884.x;
        float4 _8920 = _8918;
        _8920.y = _884.y;
        float4 _8922 = _8920;
        _8922.z = _884.z;
        res = _8922;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod)
{
    return SampleBilinear(index, uvs, lod, false, false);
}

float fresnel_dielectric_cos(float cosi, float eta)
{
    float _1092 = abs(cosi);
    float _1101 = mad(_1092, _1092, mad(eta, eta, -1.0f));
    float g = _1101;
    float result;
    if (_1101 > 0.0f)
    {
        float _1106 = g;
        float _1107 = sqrt(_1106);
        g = _1107;
        float _1111 = _1107 - _1092;
        float _1114 = _1107 + _1092;
        float _1115 = _1111 / _1114;
        float _1129 = mad(_1092, _1114, -1.0f) / mad(_1092, _1111, 1.0f);
        result = ((0.5f * _1115) * _1115) * mad(_1129, _1129, 1.0f);
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
    float3 _8185;
    do
    {
        float _1165 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1165)
        {
            _8185 = N;
            break;
        }
        float3 _1185 = normalize(N - (Ng * dot(N, Ng)));
        float _1189 = dot(I, _1185);
        float _1193 = dot(I, Ng);
        float _1205 = mad(_1189, _1189, _1193 * _1193);
        float param = (_1189 * _1189) * mad(-_1165, _1165, _1205);
        float _1215 = safe_sqrtf(param);
        float _1221 = mad(_1193, _1165, _1205);
        float _1224 = 0.5f / _1205;
        float _1229 = _1215 + _1221;
        float _1230 = _1224 * _1229;
        float _1236 = (-_1215) + _1221;
        float _1237 = _1224 * _1236;
        bool _1245 = (_1230 > 9.9999997473787516355514526367188e-06f) && (_1230 <= 1.000010013580322265625f);
        bool valid1 = _1245;
        bool _1251 = (_1237 > 9.9999997473787516355514526367188e-06f) && (_1237 <= 1.000010013580322265625f);
        bool valid2 = _1251;
        float2 N_new;
        if (_1245 && _1251)
        {
            float _9215 = (-0.5f) / _1205;
            float param_1 = mad(_9215, _1229, 1.0f);
            float _1261 = safe_sqrtf(param_1);
            float param_2 = _1230;
            float _1264 = safe_sqrtf(param_2);
            float2 _1265 = float2(_1261, _1264);
            float param_3 = mad(_9215, _1236, 1.0f);
            float _1270 = safe_sqrtf(param_3);
            float param_4 = _1237;
            float _1273 = safe_sqrtf(param_4);
            float2 _1274 = float2(_1270, _1273);
            float _9217 = -_1193;
            float _1290 = mad(2.0f * mad(_1261, _1189, _1264 * _1193), _1264, _9217);
            float _1306 = mad(2.0f * mad(_1270, _1189, _1273 * _1193), _1273, _9217);
            bool _1308 = _1290 >= 9.9999997473787516355514526367188e-06f;
            valid1 = _1308;
            bool _1310 = _1306 >= 9.9999997473787516355514526367188e-06f;
            valid2 = _1310;
            if (_1308 && _1310)
            {
                bool2 _1323 = (_1290 < _1306).xx;
                N_new = float2(_1323.x ? _1265.x : _1274.x, _1323.y ? _1265.y : _1274.y);
            }
            else
            {
                bool2 _1331 = (_1290 > _1306).xx;
                N_new = float2(_1331.x ? _1265.x : _1274.x, _1331.y ? _1265.y : _1274.y);
            }
        }
        else
        {
            if (!(valid1 || valid2))
            {
                _8185 = Ng;
                break;
            }
            float _1343 = valid1 ? _1230 : _1237;
            float param_5 = 1.0f - _1343;
            float param_6 = _1343;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _8185 = (_1185 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _8185;
}

float3 rotate_around_axis(float3 p, float3 axis, float angle)
{
    float _1416 = cos(angle);
    float _1419 = sin(angle);
    float _1423 = 1.0f - _1416;
    return float3(mad(mad(_1423 * axis.x, axis.z, axis.y * _1419), p.z, mad(mad(_1423 * axis.x, axis.x, _1416), p.x, mad(_1423 * axis.x, axis.y, -(axis.z * _1419)) * p.y)), mad(mad(_1423 * axis.y, axis.z, -(axis.x * _1419)), p.z, mad(mad(_1423 * axis.x, axis.y, axis.z * _1419), p.x, mad(_1423 * axis.y, axis.y, _1416) * p.y)), mad(mad(_1423 * axis.z, axis.z, _1416), p.z, mad(mad(_1423 * axis.x, axis.z, -(axis.y * _1419)), p.x, mad(_1423 * axis.y, axis.z, axis.x * _1419) * p.y)));
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
    float3 _8210;
    do
    {
        float2 _2910 = (float2(r1, r2) * 2.0f) - 1.0f.xx;
        float _2912 = _2910.x;
        bool _2913 = _2912 == 0.0f;
        bool _2919;
        if (_2913)
        {
            _2919 = _2910.y == 0.0f;
        }
        else
        {
            _2919 = _2913;
        }
        if (_2919)
        {
            _8210 = N;
            break;
        }
        float _2928 = _2910.y;
        float r;
        float theta;
        if (abs(_2912) > abs(_2928))
        {
            r = _2912;
            theta = 0.785398185253143310546875f * (_2928 / _2912);
        }
        else
        {
            r = _2928;
            theta = 1.57079637050628662109375f * mad(-0.5f, _2912 / _2928, 1.0f);
        }
        float3 param;
        float3 param_1;
        create_tbn(N, param, param_1);
        _8210 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _8210;
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
            float2 _8905 = origin;
            _8905.x = origin.x + _step;
            origin = _8905;
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
            float2 _8908 = origin;
            _8908.y = origin.y + _step;
            origin = _8908;
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
    float _3011 = frac(asfloat(_2995.Load((_3000_g_params.hi + 3) * 4 + 0)) + sample_off.x);
    float _3016 = float(_3000_g_params.li_count);
    uint _3023 = min(uint(_3011 * _3016), uint(_3000_g_params.li_count - 1));
    light_t _3043;
    _3043.type_and_param0 = _3032.Load4(_3036.Load(_3023 * 4 + 0) * 64 + 0);
    _3043.param1 = asfloat(_3032.Load4(_3036.Load(_3023 * 4 + 0) * 64 + 16));
    _3043.param2 = asfloat(_3032.Load4(_3036.Load(_3023 * 4 + 0) * 64 + 32));
    _3043.param3 = asfloat(_3032.Load4(_3036.Load(_3023 * 4 + 0) * 64 + 48));
    ls.col = asfloat(_3043.type_and_param0.yzw);
    ls.col *= _3016;
    ls.cast_shadow = float((_3043.type_and_param0.x & 32u) != 0u);
    uint _3078 = _3043.type_and_param0.x & 31u;
    [branch]
    if (_3078 == 0u)
    {
        float _3093 = frac(asfloat(_2995.Load((_3000_g_params.hi + 4) * 4 + 0)) + sample_off.x);
        float3 _3109 = P - _3043.param1.xyz;
        float3 _3116 = _3109 / length(_3109).xxx;
        float _3123 = sqrt(clamp(mad(-_3093, _3093, 1.0f), 0.0f, 1.0f));
        float _3126 = 6.283185482025146484375f * frac(asfloat(_2995.Load((_3000_g_params.hi + 5) * 4 + 0)) + sample_off.y);
        float3 sampled_dir = float3(_3123 * cos(_3126), _3123 * sin(_3126), _3093);
        float3 param;
        float3 param_1;
        create_tbn(_3116, param, param_1);
        float3 _8985 = sampled_dir;
        float3 _3159 = ((param * _8985.x) + (param_1 * _8985.y)) + (_3116 * _8985.z);
        sampled_dir = _3159;
        float3 _3168 = _3043.param1.xyz + (_3159 * _3043.param2.x);
        ls.L = _3168 - P;
        ls.dist = length(ls.L);
        ls.L /= ls.dist.xxx;
        ls.area = _3043.param1.w;
        float _3199 = abs(dot(ls.L, normalize(_3168 - _3043.param1.xyz)));
        [flatten]
        if (_3199 > 0.0f)
        {
            ls.pdf = (ls.dist * ls.dist) / ((0.5f * ls.area) * _3199);
        }
    }
    else
    {
        [branch]
        if (_3078 == 2u)
        {
            ls.L = _3043.param1.xyz;
            if (_3043.param1.w != 0.0f)
            {
                float param_2 = frac(asfloat(_2995.Load((_3000_g_params.hi + 4) * 4 + 0)) + sample_off.x);
                float param_3 = frac(asfloat(_2995.Load((_3000_g_params.hi + 5) * 4 + 0)) + sample_off.y);
                float3 param_4 = ls.L;
                float param_5 = tan(_3043.param1.w);
                ls.L = normalize(MapToCone(param_2, param_3, param_4, param_5));
            }
            ls.area = 0.0f;
            ls.dist = 3402823346297367662189621542912.0f;
            ls.pdf = 1.0f;
            if ((_3043.type_and_param0.x & 64u) == 0u)
            {
                ls.area = 0.0f;
            }
        }
        else
        {
            [branch]
            if (_3078 == 4u)
            {
                float3 _3331 = ((_3043.param1.xyz + (_3043.param2.xyz * (frac(asfloat(_2995.Load((_3000_g_params.hi + 4) * 4 + 0)) + sample_off.x) - 0.5f))) + (_3043.param3.xyz * (frac(asfloat(_2995.Load((_3000_g_params.hi + 5) * 4 + 0)) + sample_off.y) - 0.5f))) - P;
                ls.dist = length(_3331);
                ls.L = _3331 / ls.dist.xxx;
                ls.area = _3043.param1.w;
                float _3354 = dot(-ls.L, normalize(cross(_3043.param2.xyz, _3043.param3.xyz)));
                if (_3354 > 0.0f)
                {
                    ls.pdf = (ls.dist * ls.dist) / (ls.area * _3354);
                }
                if ((_3043.type_and_param0.x & 64u) == 0u)
                {
                    ls.area = 0.0f;
                }
                [branch]
                if ((_3043.type_and_param0.w & 128u) != 0u)
                {
                    float3 env_col = _3000_g_params.env_col.xyz;
                    uint _3394 = asuint(_3000_g_params.env_col.w);
                    if (_3394 != 4294967295u)
                    {
                        env_col *= SampleLatlong_RGBE(_3394, ls.L, _3000_g_params.env_rotation);
                    }
                    ls.col *= env_col;
                }
            }
            else
            {
                [branch]
                if (_3078 == 5u)
                {
                    float2 _3457 = (float2(frac(asfloat(_2995.Load((_3000_g_params.hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_2995.Load((_3000_g_params.hi + 5) * 4 + 0)) + sample_off.y)) * 2.0f) - 1.0f.xx;
                    float2 offset = _3457;
                    bool _3460 = _3457.x != 0.0f;
                    bool _3466;
                    if (_3460)
                    {
                        _3466 = offset.y != 0.0f;
                    }
                    else
                    {
                        _3466 = _3460;
                    }
                    if (_3466)
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
                        float _3499 = 0.5f * r;
                        offset = float2(_3499 * cos(theta), _3499 * sin(theta));
                    }
                    float3 _3525 = ((_3043.param1.xyz + (_3043.param2.xyz * offset.x)) + (_3043.param3.xyz * offset.y)) - P;
                    ls.dist = length(_3525);
                    ls.L = _3525 / ls.dist.xxx;
                    ls.area = _3043.param1.w;
                    float _3548 = dot(-ls.L, normalize(cross(_3043.param2.xyz, _3043.param3.xyz)));
                    [flatten]
                    if (_3548 > 0.0f)
                    {
                        ls.pdf = (ls.dist * ls.dist) / (ls.area * _3548);
                    }
                    if ((_3043.type_and_param0.x & 64u) == 0u)
                    {
                        ls.area = 0.0f;
                    }
                    [branch]
                    if ((_3043.type_and_param0.w & 128u) != 0u)
                    {
                        float3 env_col_1 = _3000_g_params.env_col.xyz;
                        uint _3584 = asuint(_3000_g_params.env_col.w);
                        if (_3584 != 4294967295u)
                        {
                            env_col_1 *= SampleLatlong_RGBE(_3584, ls.L, _3000_g_params.env_rotation);
                        }
                        ls.col *= env_col_1;
                    }
                }
                else
                {
                    [branch]
                    if (_3078 == 3u)
                    {
                        float3 _3643 = normalize(cross(P - _3043.param1.xyz, _3043.param3.xyz));
                        float _3650 = 3.1415927410125732421875f * frac(asfloat(_2995.Load((_3000_g_params.hi + 4) * 4 + 0)) + sample_off.x);
                        float3 _3679 = ((_3043.param1.xyz + (((_3643 * cos(_3650)) + (cross(_3643, _3043.param3.xyz) * sin(_3650))) * _3043.param2.w)) + ((_3043.param3.xyz * (frac(asfloat(_2995.Load((_3000_g_params.hi + 5) * 4 + 0)) + sample_off.y) - 0.5f)) * _3043.param3.w)) - P;
                        ls.dist = length(_3679);
                        ls.L = _3679 / ls.dist.xxx;
                        ls.area = _3043.param1.w;
                        float _3698 = 1.0f - abs(dot(ls.L, _3043.param3.xyz));
                        [flatten]
                        if (_3698 != 0.0f)
                        {
                            ls.pdf = (ls.dist * ls.dist) / (ls.area * _3698);
                        }
                        if ((_3043.type_and_param0.x & 64u) == 0u)
                        {
                            ls.area = 0.0f;
                        }
                        [branch]
                        if ((_3043.type_and_param0.w & 128u) != 0u)
                        {
                            float3 env_col_2 = _3000_g_params.env_col.xyz;
                            uint _3734 = asuint(_3000_g_params.env_col.w);
                            if (_3734 != 4294967295u)
                            {
                                env_col_2 *= SampleLatlong_RGBE(_3734, ls.L, _3000_g_params.env_rotation);
                            }
                            ls.col *= env_col_2;
                        }
                    }
                    else
                    {
                        [branch]
                        if (_3078 == 6u)
                        {
                            uint _3761 = asuint(_3043.param1.x);
                            transform_t _3775;
                            _3775.xform = asfloat(uint4x4(_3769.Load4(asuint(_3043.param1.y) * 128 + 0), _3769.Load4(asuint(_3043.param1.y) * 128 + 16), _3769.Load4(asuint(_3043.param1.y) * 128 + 32), _3769.Load4(asuint(_3043.param1.y) * 128 + 48)));
                            _3775.inv_xform = asfloat(uint4x4(_3769.Load4(asuint(_3043.param1.y) * 128 + 64), _3769.Load4(asuint(_3043.param1.y) * 128 + 80), _3769.Load4(asuint(_3043.param1.y) * 128 + 96), _3769.Load4(asuint(_3043.param1.y) * 128 + 112)));
                            uint _3800 = _3761 * 3u;
                            vertex_t _3806;
                            [unroll]
                            for (int _43ident = 0; _43ident < 3; _43ident++)
                            {
                                _3806.p[_43ident] = asfloat(_3794.Load(_43ident * 4 + _3798.Load(_3800 * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _44ident = 0; _44ident < 3; _44ident++)
                            {
                                _3806.n[_44ident] = asfloat(_3794.Load(_44ident * 4 + _3798.Load(_3800 * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _45ident = 0; _45ident < 3; _45ident++)
                            {
                                _3806.b[_45ident] = asfloat(_3794.Load(_45ident * 4 + _3798.Load(_3800 * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _46ident = 0; _46ident < 2; _46ident++)
                            {
                                [unroll]
                                for (int _47ident = 0; _47ident < 2; _47ident++)
                                {
                                    _3806.t[_46ident][_47ident] = asfloat(_3794.Load(_47ident * 4 + _46ident * 8 + _3798.Load(_3800 * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _3855;
                            [unroll]
                            for (int _48ident = 0; _48ident < 3; _48ident++)
                            {
                                _3855.p[_48ident] = asfloat(_3794.Load(_48ident * 4 + _3798.Load((_3800 + 1u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _49ident = 0; _49ident < 3; _49ident++)
                            {
                                _3855.n[_49ident] = asfloat(_3794.Load(_49ident * 4 + _3798.Load((_3800 + 1u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _50ident = 0; _50ident < 3; _50ident++)
                            {
                                _3855.b[_50ident] = asfloat(_3794.Load(_50ident * 4 + _3798.Load((_3800 + 1u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _51ident = 0; _51ident < 2; _51ident++)
                            {
                                [unroll]
                                for (int _52ident = 0; _52ident < 2; _52ident++)
                                {
                                    _3855.t[_51ident][_52ident] = asfloat(_3794.Load(_52ident * 4 + _51ident * 8 + _3798.Load((_3800 + 1u) * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _3901;
                            [unroll]
                            for (int _53ident = 0; _53ident < 3; _53ident++)
                            {
                                _3901.p[_53ident] = asfloat(_3794.Load(_53ident * 4 + _3798.Load((_3800 + 2u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _54ident = 0; _54ident < 3; _54ident++)
                            {
                                _3901.n[_54ident] = asfloat(_3794.Load(_54ident * 4 + _3798.Load((_3800 + 2u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _55ident = 0; _55ident < 3; _55ident++)
                            {
                                _3901.b[_55ident] = asfloat(_3794.Load(_55ident * 4 + _3798.Load((_3800 + 2u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _56ident = 0; _56ident < 2; _56ident++)
                            {
                                [unroll]
                                for (int _57ident = 0; _57ident < 2; _57ident++)
                                {
                                    _3901.t[_56ident][_57ident] = asfloat(_3794.Load(_57ident * 4 + _56ident * 8 + _3798.Load((_3800 + 2u) * 4 + 0) * 52 + 36));
                                }
                            }
                            float3 _3947 = float3(_3806.p[0], _3806.p[1], _3806.p[2]);
                            float3 _3955 = float3(_3855.p[0], _3855.p[1], _3855.p[2]);
                            float3 _3963 = float3(_3901.p[0], _3901.p[1], _3901.p[2]);
                            float _3992 = sqrt(frac(asfloat(_2995.Load((_3000_g_params.hi + 4) * 4 + 0)) + sample_off.x));
                            float _4002 = frac(asfloat(_2995.Load((_3000_g_params.hi + 5) * 4 + 0)) + sample_off.y);
                            float _4006 = 1.0f - _3992;
                            float _4011 = 1.0f - _4002;
                            float3 _4058 = mul(float4(cross(_3955 - _3947, _3963 - _3947), 0.0f), _3775.xform).xyz;
                            ls.area = 0.5f * length(_4058);
                            float3 _4068 = mul(float4((_3947 * _4006) + (((_3955 * _4011) + (_3963 * _4002)) * _3992), 1.0f), _3775.xform).xyz - P;
                            ls.dist = length(_4068);
                            ls.L = _4068 / ls.dist.xxx;
                            float _4083 = abs(dot(ls.L, normalize(_4058)));
                            [flatten]
                            if (_4083 > 0.0f)
                            {
                                ls.pdf = (ls.dist * ls.dist) / (ls.area * _4083);
                            }
                            material_t _4123;
                            [unroll]
                            for (int _58ident = 0; _58ident < 5; _58ident++)
                            {
                                _4123.textures[_58ident] = _4109.Load(_58ident * 4 + ((_4113.Load(_3761 * 4 + 0) >> 16u) & 16383u) * 80 + 0);
                            }
                            [unroll]
                            for (int _59ident = 0; _59ident < 3; _59ident++)
                            {
                                _4123.base_color[_59ident] = asfloat(_4109.Load(_59ident * 4 + ((_4113.Load(_3761 * 4 + 0) >> 16u) & 16383u) * 80 + 20));
                            }
                            _4123.flags = _4109.Load(((_4113.Load(_3761 * 4 + 0) >> 16u) & 16383u) * 80 + 32);
                            _4123.type = _4109.Load(((_4113.Load(_3761 * 4 + 0) >> 16u) & 16383u) * 80 + 36);
                            _4123.tangent_rotation_or_strength = asfloat(_4109.Load(((_4113.Load(_3761 * 4 + 0) >> 16u) & 16383u) * 80 + 40));
                            _4123.roughness_and_anisotropic = _4109.Load(((_4113.Load(_3761 * 4 + 0) >> 16u) & 16383u) * 80 + 44);
                            _4123.int_ior = asfloat(_4109.Load(((_4113.Load(_3761 * 4 + 0) >> 16u) & 16383u) * 80 + 48));
                            _4123.ext_ior = asfloat(_4109.Load(((_4113.Load(_3761 * 4 + 0) >> 16u) & 16383u) * 80 + 52));
                            _4123.sheen_and_sheen_tint = _4109.Load(((_4113.Load(_3761 * 4 + 0) >> 16u) & 16383u) * 80 + 56);
                            _4123.tint_and_metallic = _4109.Load(((_4113.Load(_3761 * 4 + 0) >> 16u) & 16383u) * 80 + 60);
                            _4123.transmission_and_transmission_roughness = _4109.Load(((_4113.Load(_3761 * 4 + 0) >> 16u) & 16383u) * 80 + 64);
                            _4123.specular_and_specular_tint = _4109.Load(((_4113.Load(_3761 * 4 + 0) >> 16u) & 16383u) * 80 + 68);
                            _4123.clearcoat_and_clearcoat_roughness = _4109.Load(((_4113.Load(_3761 * 4 + 0) >> 16u) & 16383u) * 80 + 72);
                            _4123.normal_map_strength_unorm = _4109.Load(((_4113.Load(_3761 * 4 + 0) >> 16u) & 16383u) * 80 + 76);
                            if ((_4123.flags & 4u) == 0u)
                            {
                                if (_4123.textures[1] != 4294967295u)
                                {
                                    ls.col *= SampleBilinear(_4123.textures[1], (float2(_3806.t[0][0], _3806.t[0][1]) * _4006) + (((float2(_3855.t[0][0], _3855.t[0][1]) * _4011) + (float2(_3901.t[0][0], _3901.t[0][1]) * _4002)) * _3992), 0).xyz;
                                }
                            }
                            else
                            {
                                float3 env_col_3 = _3000_g_params.env_col.xyz;
                                uint _4203 = asuint(_3000_g_params.env_col.w);
                                if (_4203 != 4294967295u)
                                {
                                    env_col_3 *= SampleLatlong_RGBE(_4203, ls.L, _3000_g_params.env_rotation);
                                }
                                ls.col *= env_col_3;
                            }
                        }
                        else
                        {
                            [branch]
                            if (_3078 == 7u)
                            {
                                float4 _4265 = Sample_EnvQTree(_3000_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3000_g_params.env_qtree_levels, mad(_3011, _3016, -float(_3023)), frac(asfloat(_2995.Load((_3000_g_params.hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_2995.Load((_3000_g_params.hi + 5) * 4 + 0)) + sample_off.y));
                                ls.L = _4265.xyz;
                                ls.col *= _3000_g_params.env_col.xyz;
                                ls.col *= SampleLatlong_RGBE(asuint(_3000_g_params.env_col.w), ls.L, _3000_g_params.env_rotation);
                                ls.area = 1.0f;
                                ls.dist = 3402823346297367662189621542912.0f;
                                ls.pdf = _4265.w;
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
    float _1991 = 1.0f / mad(0.904129683971405029296875f, roughness, 3.1415927410125732421875f);
    float _2003 = max(dot(N, L), 0.0f);
    float _2008 = max(dot(N, V), 0.0f);
    float _2016 = mad(-_2003, _2008, dot(L, V));
    float t = _2016;
    if (_2016 > 0.0f)
    {
        t /= (max(_2003, _2008) + 1.1754943508222875079687365372222e-38f);
    }
    return float4(base_color * (_2003 * mad(roughness * _1991, t, _1991)), 0.15915493667125701904296875f);
}

float3 offset_ray(float3 p, float3 n)
{
    int3 _1573 = int3(n * 128.0f);
    int _1581;
    if (p.x < 0.0f)
    {
        _1581 = -_1573.x;
    }
    else
    {
        _1581 = _1573.x;
    }
    int _1599;
    if (p.y < 0.0f)
    {
        _1599 = -_1573.y;
    }
    else
    {
        _1599 = _1573.y;
    }
    int _1617;
    if (p.z < 0.0f)
    {
        _1617 = -_1573.z;
    }
    else
    {
        _1617 = _1573.z;
    }
    float _1635;
    if (abs(p.x) < 0.03125f)
    {
        _1635 = mad(1.52587890625e-05f, n.x, p.x);
    }
    else
    {
        _1635 = asfloat(asint(p.x) + _1581);
    }
    float _1653;
    if (abs(p.y) < 0.03125f)
    {
        _1653 = mad(1.52587890625e-05f, n.y, p.y);
    }
    else
    {
        _1653 = asfloat(asint(p.y) + _1599);
    }
    float _1670;
    if (abs(p.z) < 0.03125f)
    {
        _1670 = mad(1.52587890625e-05f, n.z, p.z);
    }
    else
    {
        _1670 = asfloat(asint(p.z) + _1617);
    }
    return float3(_1635, _1653, _1670);
}

float3 world_from_tangent(float3 T, float3 B, float3 N, float3 V)
{
    return ((T * V.x) + (B * V.y)) + (N * V.z);
}

float4 Sample_OrenDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float rand_u, float rand_v, inout float3 out_V)
{
    float _2050 = 6.283185482025146484375f * rand_v;
    float _2062 = sqrt(mad(-rand_u, rand_u, 1.0f));
    float3 param = T;
    float3 param_1 = B;
    float3 param_2 = N;
    float3 param_3 = float3(_2062 * cos(_2050), _2062 * sin(_2050), rand_u);
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
    float _8215;
    do
    {
        if (H.z == 0.0f)
        {
            _8215 = 0.0f;
            break;
        }
        float _1877 = (-H.x) / (H.z * alpha_x);
        float _1883 = (-H.y) / (H.z * alpha_y);
        float _1892 = mad(_1883, _1883, mad(_1877, _1877, 1.0f));
        _8215 = 1.0f / (((((_1892 * _1892) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _8215;
}

float G1(float3 Ve, inout float alpha_x, inout float alpha_y)
{
    alpha_x *= alpha_x;
    alpha_y *= alpha_y;
    return 1.0f / mad((-1.0f) + sqrt(1.0f + (mad(alpha_x * Ve.x, Ve.x, (alpha_y * Ve.y) * Ve.y) / (Ve.z * Ve.z))), 0.5f, 1.0f);
}

float4 Evaluate_GGXSpecular_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float alpha_x, float alpha_y, float spec_ior, float spec_F0, float3 spec_col)
{
    float _2232 = D_GGX(sampled_normal_ts, alpha_x, alpha_y);
    float3 param = view_dir_ts;
    float param_1 = alpha_x;
    float param_2 = alpha_y;
    float _2240 = G1(param, param_1, param_2);
    float3 param_3 = reflected_dir_ts;
    float param_4 = alpha_x;
    float param_5 = alpha_y;
    float _2247 = G1(param_3, param_4, param_5);
    float param_6 = dot(view_dir_ts, sampled_normal_ts);
    float param_7 = spec_ior;
    float3 F = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param_6, param_7) - spec_F0) / (1.0f - spec_F0)).xxx);
    float _2275 = 4.0f * abs(view_dir_ts.z * reflected_dir_ts.z);
    float _2278;
    if (_2275 != 0.0f)
    {
        _2278 = (_2232 * (_2240 * _2247)) / _2275;
    }
    else
    {
        _2278 = 0.0f;
    }
    F *= _2278;
    float3 param_8 = view_dir_ts;
    float param_9 = alpha_x;
    float param_10 = alpha_y;
    float _2298 = G1(param_8, param_9, param_10);
    float pdf = ((_2232 * _2298) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2313 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2313 != 0.0f)
    {
        pdf /= _2313;
    }
    float3 _2324 = F;
    float3 _2325 = _2324 * max(reflected_dir_ts.z, 0.0f);
    F = _2325;
    return float4(_2325, pdf);
}

float3 SampleGGX_VNDF(float3 Ve, float alpha_x, float alpha_y, float U1, float U2)
{
    float3 _1695 = normalize(float3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
    float _1698 = _1695.x;
    float _1703 = _1695.y;
    float _1707 = mad(_1698, _1698, _1703 * _1703);
    float3 _1711;
    if (_1707 > 0.0f)
    {
        _1711 = float3(-_1703, _1698, 0.0f) / sqrt(_1707).xxx;
    }
    else
    {
        _1711 = float3(1.0f, 0.0f, 0.0f);
    }
    float _1733 = sqrt(U1);
    float _1736 = 6.283185482025146484375f * U2;
    float _1741 = _1733 * cos(_1736);
    float _1750 = 1.0f + _1695.z;
    float _1757 = mad(-_1741, _1741, 1.0f);
    float _1763 = mad(mad(-0.5f, _1750, 1.0f), sqrt(_1757), (0.5f * _1750) * (_1733 * sin(_1736)));
    float3 _1784 = ((_1711 * _1741) + (cross(_1695, _1711) * _1763)) + (_1695 * sqrt(max(0.0f, mad(-_1763, _1763, _1757))));
    return normalize(float3(alpha_x * _1784.x, alpha_y * _1784.y, max(0.0f, _1784.z)));
}

float4 Sample_GGXSpecular_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float anisotropic, float spec_ior, float spec_F0, float3 spec_col, float rand_u, float rand_v, inout float3 out_V)
{
    float4 _8190;
    do
    {
        float _2335 = roughness * roughness;
        float _2339 = sqrt(mad(-0.89999997615814208984375f, anisotropic, 1.0f));
        float _2343 = _2335 / _2339;
        float _2347 = _2335 * _2339;
        [branch]
        if ((_2343 * _2347) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2357 = reflect(I, N);
            float param = dot(_2357, N);
            float param_1 = spec_ior;
            float3 _2371 = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param, param_1) - spec_F0) / (1.0f - spec_F0)).xxx);
            out_V = _2357;
            _8190 = float4(_2371.x * 1000000.0f, _2371.y * 1000000.0f, _2371.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2396 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = _2343;
        float param_7 = _2347;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2405 = SampleGGX_VNDF(_2396, param_6, param_7, param_8, param_9);
        float3 _2416 = normalize(reflect(-_2396, _2405));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2416;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2396;
        float3 param_15 = _2405;
        float3 param_16 = _2416;
        float param_17 = _2343;
        float param_18 = _2347;
        float param_19 = spec_ior;
        float param_20 = spec_F0;
        float3 param_21 = spec_col;
        _8190 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _8190;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _8195;
    do
    {
        bool _2638 = refr_dir_ts.z >= 0.0f;
        bool _2645;
        if (!_2638)
        {
            _2645 = view_dir_ts.z <= 0.0f;
        }
        else
        {
            _2645 = _2638;
        }
        if (_2645)
        {
            _8195 = 0.0f.xxxx;
            break;
        }
        float _2654 = D_GGX(sampled_normal_ts, roughness2, roughness2);
        float3 param = refr_dir_ts;
        float param_1 = roughness2;
        float param_2 = roughness2;
        float _2662 = G1(param, param_1, param_2);
        float3 param_3 = view_dir_ts;
        float param_4 = roughness2;
        float param_5 = roughness2;
        float _2670 = G1(param_3, param_4, param_5);
        float _2680 = mad(dot(view_dir_ts, sampled_normal_ts), eta, dot(refr_dir_ts, sampled_normal_ts));
        float _2690 = clamp(-dot(refr_dir_ts, sampled_normal_ts), 0.0f, 1.0f) / (_2680 * _2680);
        _8195 = float4(refr_col * (((((_2654 * _2670) * _2662) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _2690) / view_dir_ts.z), (((_2654 * _2662) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _2690) / view_dir_ts.z);
        break;
    } while(false);
    return _8195;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _8200;
    do
    {
        float _2734 = roughness * roughness;
        [branch]
        if ((_2734 * _2734) < 1.0000000116860974230803549289703e-07f)
        {
            float _2744 = dot(I, N);
            float _2745 = -_2744;
            float _2755 = mad(-(eta * eta), mad(_2744, _2745, 1.0f), 1.0f);
            if (_2755 < 0.0f)
            {
                _8200 = 0.0f.xxxx;
                break;
            }
            float _2767 = mad(eta, _2745, -sqrt(_2755));
            out_V = float4(normalize((I * eta) + (N * _2767)), _2767);
            _8200 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param = T;
        float3 param_1 = B;
        float3 param_2 = N;
        float3 param_3 = -I;
        float3 _2807 = normalize(tangent_from_world(param, param_1, param_2, param_3));
        float param_4 = _2734;
        float param_5 = _2734;
        float param_6 = rand_u;
        float param_7 = rand_v;
        float3 _2818 = SampleGGX_VNDF(_2807, param_4, param_5, param_6, param_7);
        float _2822 = dot(_2807, _2818);
        float _2832 = mad(-(eta * eta), mad(-_2822, _2822, 1.0f), 1.0f);
        if (_2832 < 0.0f)
        {
            _8200 = 0.0f.xxxx;
            break;
        }
        float _2844 = mad(eta, _2822, -sqrt(_2832));
        float3 _2854 = normalize((_2807 * (-eta)) + (_2818 * _2844));
        float3 param_8 = _2807;
        float3 param_9 = _2818;
        float3 param_10 = _2854;
        float param_11 = _2734;
        float param_12 = eta;
        float3 param_13 = refr_col;
        float3 param_14 = T;
        float3 param_15 = B;
        float3 param_16 = N;
        float3 param_17 = _2854;
        out_V = float4(world_from_tangent(param_14, param_15, param_16, param_17), _2844);
        _8200 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _8200;
}

void get_lobe_weights(float base_color_lum, float spec_color_lum, float specular, float metallic, float transmission, float clearcoat, inout float out_diffuse_weight, inout float out_specular_weight, inout float out_clearcoat_weight, inout float out_refraction_weight)
{
    float _1366 = 1.0f - metallic;
    out_diffuse_weight = (base_color_lum * _1366) * (1.0f - transmission);
    float _1376;
    if ((specular != 0.0f) || (metallic != 0.0f))
    {
        _1376 = spec_color_lum * mad(-transmission, _1366, 1.0f);
    }
    else
    {
        _1376 = 0.0f;
    }
    out_specular_weight = _1376;
    out_clearcoat_weight = (0.25f * clearcoat) * _1366;
    out_refraction_weight = (transmission * _1366) * base_color_lum;
    float _1391 = out_diffuse_weight;
    float _1392 = out_specular_weight;
    float _1394 = out_clearcoat_weight;
    float _1397 = ((_1391 + _1392) + _1394) + out_refraction_weight;
    if (_1397 != 0.0f)
    {
        out_diffuse_weight /= _1397;
        out_specular_weight /= _1397;
        out_clearcoat_weight /= _1397;
        out_refraction_weight /= _1397;
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
    float _8220;
    do
    {
        float _1943 = dot(N, L);
        if (_1943 <= 0.0f)
        {
            _8220 = 0.0f;
            break;
        }
        float param = _1943;
        float param_1 = dot(N, V);
        float _1964 = dot(L, H);
        float _1972 = mad((2.0f * _1964) * _1964, roughness, 0.5f);
        _8220 = lerp(1.0f, _1972, schlick_weight(param)) * lerp(1.0f, _1972, schlick_weight(param_1));
        break;
    } while(false);
    return _8220;
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
    float3 _2113 = normalize(L + V);
    float3 H = _2113;
    if (dot(V, _2113) < 0.0f)
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
    float3 _2148 = diff_col;
    float3 _2149 = _2148 + (sheen_color * (3.1415927410125732421875f * schlick_weight(param_5)));
    diff_col = _2149;
    return float4(_2149, pdf);
}

float D_GTR1(float NDotH, float a)
{
    float _8225;
    do
    {
        if (a >= 1.0f)
        {
            _8225 = 0.3183098733425140380859375f;
            break;
        }
        float _1851 = mad(a, a, -1.0f);
        _8225 = _1851 / ((3.1415927410125732421875f * log(a * a)) * mad(_1851 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _8225;
}

float4 Evaluate_PrincipledClearcoat_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float clearcoat_roughness2, float clearcoat_ior, float clearcoat_F0)
{
    float param = sampled_normal_ts.z;
    float param_1 = clearcoat_roughness2;
    float _2448 = D_GTR1(param, param_1);
    float3 param_2 = view_dir_ts;
    float param_3 = 0.0625f;
    float param_4 = 0.0625f;
    float _2455 = G1(param_2, param_3, param_4);
    float3 param_5 = reflected_dir_ts;
    float param_6 = 0.0625f;
    float param_7 = 0.0625f;
    float _2460 = G1(param_5, param_6, param_7);
    float param_8 = dot(reflected_dir_ts, sampled_normal_ts);
    float param_9 = clearcoat_ior;
    float F = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param_8, param_9) - clearcoat_F0) / (1.0f - clearcoat_F0));
    float _2487 = (4.0f * abs(view_dir_ts.z)) * abs(reflected_dir_ts.z);
    float _2490;
    if (_2487 != 0.0f)
    {
        _2490 = (_2448 * (_2455 * _2460)) / _2487;
    }
    else
    {
        _2490 = 0.0f;
    }
    F *= _2490;
    float3 param_10 = view_dir_ts;
    float param_11 = 0.0625f;
    float param_12 = 0.0625f;
    float _2508 = G1(param_10, param_11, param_12);
    float pdf = ((_2448 * _2508) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2523 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2523 != 0.0f)
    {
        pdf /= _2523;
    }
    float _2534 = F;
    float _2535 = _2534 * clamp(reflected_dir_ts.z, 0.0f, 1.0f);
    F = _2535;
    return float4(_2535, _2535, _2535, pdf);
}

float4 Sample_PrincipledDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float3 sheen_color, bool uniform_sampling, float rand_u, float rand_v, inout float3 out_V)
{
    float _2160 = 6.283185482025146484375f * rand_v;
    float _2163 = cos(_2160);
    float _2166 = sin(_2160);
    float3 V;
    if (uniform_sampling)
    {
        float _2175 = sqrt(mad(-rand_u, rand_u, 1.0f));
        V = float3(_2175 * _2163, _2175 * _2166, rand_u);
    }
    else
    {
        float _2188 = sqrt(rand_u);
        V = float3(_2188 * _2163, _2188 * _2166, sqrt(1.0f - rand_u));
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
    float4 _8205;
    do
    {
        [branch]
        if ((clearcoat_roughness2 * clearcoat_roughness2) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2552 = reflect(I, N);
            float param = dot(_2552, N);
            float param_1 = clearcoat_ior;
            out_V = _2552;
            float _2571 = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param, param_1) - clearcoat_F0) / (1.0f - clearcoat_F0)) * 1000000.0f;
            _8205 = float4(_2571, _2571, _2571, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2589 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = clearcoat_roughness2;
        float param_7 = clearcoat_roughness2;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2600 = SampleGGX_VNDF(_2589, param_6, param_7, param_8, param_9);
        float3 _2611 = normalize(reflect(-_2589, _2600));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2611;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2589;
        float3 param_15 = _2600;
        float3 param_16 = _2611;
        float param_17 = clearcoat_roughness2;
        float param_18 = clearcoat_ior;
        float param_19 = clearcoat_F0;
        _8205 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _8205;
}

float3 ShadeSurface(int px_index, hit_data_t inter, ray_data_t ray)
{
    float3 _8180;
    do
    {
        float3 _4310 = float3(ray.d[0], ray.d[1], ray.d[2]);
        [branch]
        if (inter.mask == 0)
        {
            float3 env_col = _3000_g_params.env_col.xyz;
            uint _4323 = asuint(_3000_g_params.env_col.w);
            if (_4323 != 4294967295u)
            {
                env_col *= SampleLatlong_RGBE(_4323, _4310, _3000_g_params.env_rotation);
                if (_3000_g_params.env_qtree_levels > 0)
                {
                    float param = ray.pdf;
                    float param_1 = Evaluate_EnvQTree(_3000_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3000_g_params.env_qtree_levels, _4310);
                    env_col *= power_heuristic(param, param_1);
                }
            }
            _8180 = float3(ray.c[0] * env_col.x, ray.c[1] * env_col.y, ray.c[2] * env_col.z);
            break;
        }
        float3 _4384 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_4310 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            light_t _4396;
            _4396.type_and_param0 = _3032.Load4(((-1) - inter.obj_index) * 64 + 0);
            _4396.param1 = asfloat(_3032.Load4(((-1) - inter.obj_index) * 64 + 16));
            _4396.param2 = asfloat(_3032.Load4(((-1) - inter.obj_index) * 64 + 32));
            _4396.param3 = asfloat(_3032.Load4(((-1) - inter.obj_index) * 64 + 48));
            float3 lcol = asfloat(_4396.type_and_param0.yzw);
            uint _4413 = _4396.type_and_param0.x & 31u;
            if (_4413 == 0u)
            {
                float param_2 = ray.pdf;
                float param_3 = (inter.t * inter.t) / ((0.5f * _4396.param1.w) * dot(_4310, normalize(_4396.param1.xyz - _4384)));
                lcol *= power_heuristic(param_2, param_3);
            }
            else
            {
                if (_4413 == 4u)
                {
                    float param_4 = ray.pdf;
                    float param_5 = (inter.t * inter.t) / (_4396.param1.w * dot(_4310, normalize(cross(_4396.param2.xyz, _4396.param3.xyz))));
                    lcol *= power_heuristic(param_4, param_5);
                }
                else
                {
                    if (_4413 == 5u)
                    {
                        float param_6 = ray.pdf;
                        float param_7 = (inter.t * inter.t) / (_4396.param1.w * dot(_4310, normalize(cross(_4396.param2.xyz, _4396.param3.xyz))));
                        lcol *= power_heuristic(param_6, param_7);
                    }
                    else
                    {
                        if (_4413 == 3u)
                        {
                            float param_8 = ray.pdf;
                            float param_9 = (inter.t * inter.t) / (_4396.param1.w * (1.0f - abs(dot(_4310, _4396.param3.xyz))));
                            lcol *= power_heuristic(param_8, param_9);
                        }
                    }
                }
            }
            _8180 = float3(ray.c[0] * lcol.x, ray.c[1] * lcol.y, ray.c[2] * lcol.z);
            break;
        }
        bool _4648 = inter.prim_index < 0;
        int _4651;
        if (_4648)
        {
            _4651 = (-1) - inter.prim_index;
        }
        else
        {
            _4651 = inter.prim_index;
        }
        uint _4662 = uint(_4651);
        material_t _4670;
        [unroll]
        for (int _60ident = 0; _60ident < 5; _60ident++)
        {
            _4670.textures[_60ident] = _4109.Load(_60ident * 4 + ((_4113.Load(_4662 * 4 + 0) >> 16u) & 16383u) * 80 + 0);
        }
        [unroll]
        for (int _61ident = 0; _61ident < 3; _61ident++)
        {
            _4670.base_color[_61ident] = asfloat(_4109.Load(_61ident * 4 + ((_4113.Load(_4662 * 4 + 0) >> 16u) & 16383u) * 80 + 20));
        }
        _4670.flags = _4109.Load(((_4113.Load(_4662 * 4 + 0) >> 16u) & 16383u) * 80 + 32);
        _4670.type = _4109.Load(((_4113.Load(_4662 * 4 + 0) >> 16u) & 16383u) * 80 + 36);
        _4670.tangent_rotation_or_strength = asfloat(_4109.Load(((_4113.Load(_4662 * 4 + 0) >> 16u) & 16383u) * 80 + 40));
        _4670.roughness_and_anisotropic = _4109.Load(((_4113.Load(_4662 * 4 + 0) >> 16u) & 16383u) * 80 + 44);
        _4670.int_ior = asfloat(_4109.Load(((_4113.Load(_4662 * 4 + 0) >> 16u) & 16383u) * 80 + 48));
        _4670.ext_ior = asfloat(_4109.Load(((_4113.Load(_4662 * 4 + 0) >> 16u) & 16383u) * 80 + 52));
        _4670.sheen_and_sheen_tint = _4109.Load(((_4113.Load(_4662 * 4 + 0) >> 16u) & 16383u) * 80 + 56);
        _4670.tint_and_metallic = _4109.Load(((_4113.Load(_4662 * 4 + 0) >> 16u) & 16383u) * 80 + 60);
        _4670.transmission_and_transmission_roughness = _4109.Load(((_4113.Load(_4662 * 4 + 0) >> 16u) & 16383u) * 80 + 64);
        _4670.specular_and_specular_tint = _4109.Load(((_4113.Load(_4662 * 4 + 0) >> 16u) & 16383u) * 80 + 68);
        _4670.clearcoat_and_clearcoat_roughness = _4109.Load(((_4113.Load(_4662 * 4 + 0) >> 16u) & 16383u) * 80 + 72);
        _4670.normal_map_strength_unorm = _4109.Load(((_4113.Load(_4662 * 4 + 0) >> 16u) & 16383u) * 80 + 76);
        uint _8723 = _4670.textures[0];
        uint _8724 = _4670.textures[1];
        uint _8725 = _4670.textures[2];
        uint _8726 = _4670.textures[3];
        uint _8727 = _4670.textures[4];
        float _8728 = _4670.base_color[0];
        float _8729 = _4670.base_color[1];
        float _8730 = _4670.base_color[2];
        uint _8419 = _4670.flags;
        uint _8420 = _4670.type;
        float _8421 = _4670.tangent_rotation_or_strength;
        uint _8422 = _4670.roughness_and_anisotropic;
        float _8423 = _4670.int_ior;
        float _8424 = _4670.ext_ior;
        uint _8425 = _4670.sheen_and_sheen_tint;
        uint _8426 = _4670.tint_and_metallic;
        uint _8427 = _4670.transmission_and_transmission_roughness;
        uint _8428 = _4670.specular_and_specular_tint;
        uint _8429 = _4670.clearcoat_and_clearcoat_roughness;
        uint _8430 = _4670.normal_map_strength_unorm;
        transform_t _4727;
        _4727.xform = asfloat(uint4x4(_3769.Load4(asuint(asfloat(_4720.Load(inter.obj_index * 32 + 12))) * 128 + 0), _3769.Load4(asuint(asfloat(_4720.Load(inter.obj_index * 32 + 12))) * 128 + 16), _3769.Load4(asuint(asfloat(_4720.Load(inter.obj_index * 32 + 12))) * 128 + 32), _3769.Load4(asuint(asfloat(_4720.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _4727.inv_xform = asfloat(uint4x4(_3769.Load4(asuint(asfloat(_4720.Load(inter.obj_index * 32 + 12))) * 128 + 64), _3769.Load4(asuint(asfloat(_4720.Load(inter.obj_index * 32 + 12))) * 128 + 80), _3769.Load4(asuint(asfloat(_4720.Load(inter.obj_index * 32 + 12))) * 128 + 96), _3769.Load4(asuint(asfloat(_4720.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _4734 = _4662 * 3u;
        vertex_t _4739;
        [unroll]
        for (int _62ident = 0; _62ident < 3; _62ident++)
        {
            _4739.p[_62ident] = asfloat(_3794.Load(_62ident * 4 + _3798.Load(_4734 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _63ident = 0; _63ident < 3; _63ident++)
        {
            _4739.n[_63ident] = asfloat(_3794.Load(_63ident * 4 + _3798.Load(_4734 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _64ident = 0; _64ident < 3; _64ident++)
        {
            _4739.b[_64ident] = asfloat(_3794.Load(_64ident * 4 + _3798.Load(_4734 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _65ident = 0; _65ident < 2; _65ident++)
        {
            [unroll]
            for (int _66ident = 0; _66ident < 2; _66ident++)
            {
                _4739.t[_65ident][_66ident] = asfloat(_3794.Load(_66ident * 4 + _65ident * 8 + _3798.Load(_4734 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _4785;
        [unroll]
        for (int _67ident = 0; _67ident < 3; _67ident++)
        {
            _4785.p[_67ident] = asfloat(_3794.Load(_67ident * 4 + _3798.Load((_4734 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _68ident = 0; _68ident < 3; _68ident++)
        {
            _4785.n[_68ident] = asfloat(_3794.Load(_68ident * 4 + _3798.Load((_4734 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _69ident = 0; _69ident < 3; _69ident++)
        {
            _4785.b[_69ident] = asfloat(_3794.Load(_69ident * 4 + _3798.Load((_4734 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _70ident = 0; _70ident < 2; _70ident++)
        {
            [unroll]
            for (int _71ident = 0; _71ident < 2; _71ident++)
            {
                _4785.t[_70ident][_71ident] = asfloat(_3794.Load(_71ident * 4 + _70ident * 8 + _3798.Load((_4734 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _4831;
        [unroll]
        for (int _72ident = 0; _72ident < 3; _72ident++)
        {
            _4831.p[_72ident] = asfloat(_3794.Load(_72ident * 4 + _3798.Load((_4734 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _73ident = 0; _73ident < 3; _73ident++)
        {
            _4831.n[_73ident] = asfloat(_3794.Load(_73ident * 4 + _3798.Load((_4734 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _74ident = 0; _74ident < 3; _74ident++)
        {
            _4831.b[_74ident] = asfloat(_3794.Load(_74ident * 4 + _3798.Load((_4734 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _75ident = 0; _75ident < 2; _75ident++)
        {
            [unroll]
            for (int _76ident = 0; _76ident < 2; _76ident++)
            {
                _4831.t[_75ident][_76ident] = asfloat(_3794.Load(_76ident * 4 + _75ident * 8 + _3798.Load((_4734 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _4877 = float3(_4739.p[0], _4739.p[1], _4739.p[2]);
        float3 _4885 = float3(_4785.p[0], _4785.p[1], _4785.p[2]);
        float3 _4893 = float3(_4831.p[0], _4831.p[1], _4831.p[2]);
        float _4900 = (1.0f - inter.u) - inter.v;
        float3 _4933 = normalize(((float3(_4739.n[0], _4739.n[1], _4739.n[2]) * _4900) + (float3(_4785.n[0], _4785.n[1], _4785.n[2]) * inter.u)) + (float3(_4831.n[0], _4831.n[1], _4831.n[2]) * inter.v));
        float3 N = _4933;
        float2 _4959 = ((float2(_4739.t[0][0], _4739.t[0][1]) * _4900) + (float2(_4785.t[0][0], _4785.t[0][1]) * inter.u)) + (float2(_4831.t[0][0], _4831.t[0][1]) * inter.v);
        float3 _4975 = cross(_4885 - _4877, _4893 - _4877);
        float _4978 = length(_4975);
        float3 plane_N = _4975 / _4978.xxx;
        float3 _5014 = ((float3(_4739.b[0], _4739.b[1], _4739.b[2]) * _4900) + (float3(_4785.b[0], _4785.b[1], _4785.b[2]) * inter.u)) + (float3(_4831.b[0], _4831.b[1], _4831.b[2]) * inter.v);
        float3 B = _5014;
        float3 T = cross(_5014, _4933);
        if (_4648)
        {
            if ((_4113.Load(_4662 * 4 + 0) & 65535u) == 65535u)
            {
                _8180 = 0.0f.xxx;
                break;
            }
            material_t _5038;
            [unroll]
            for (int _77ident = 0; _77ident < 5; _77ident++)
            {
                _5038.textures[_77ident] = _4109.Load(_77ident * 4 + (_4113.Load(_4662 * 4 + 0) & 16383u) * 80 + 0);
            }
            [unroll]
            for (int _78ident = 0; _78ident < 3; _78ident++)
            {
                _5038.base_color[_78ident] = asfloat(_4109.Load(_78ident * 4 + (_4113.Load(_4662 * 4 + 0) & 16383u) * 80 + 20));
            }
            _5038.flags = _4109.Load((_4113.Load(_4662 * 4 + 0) & 16383u) * 80 + 32);
            _5038.type = _4109.Load((_4113.Load(_4662 * 4 + 0) & 16383u) * 80 + 36);
            _5038.tangent_rotation_or_strength = asfloat(_4109.Load((_4113.Load(_4662 * 4 + 0) & 16383u) * 80 + 40));
            _5038.roughness_and_anisotropic = _4109.Load((_4113.Load(_4662 * 4 + 0) & 16383u) * 80 + 44);
            _5038.int_ior = asfloat(_4109.Load((_4113.Load(_4662 * 4 + 0) & 16383u) * 80 + 48));
            _5038.ext_ior = asfloat(_4109.Load((_4113.Load(_4662 * 4 + 0) & 16383u) * 80 + 52));
            _5038.sheen_and_sheen_tint = _4109.Load((_4113.Load(_4662 * 4 + 0) & 16383u) * 80 + 56);
            _5038.tint_and_metallic = _4109.Load((_4113.Load(_4662 * 4 + 0) & 16383u) * 80 + 60);
            _5038.transmission_and_transmission_roughness = _4109.Load((_4113.Load(_4662 * 4 + 0) & 16383u) * 80 + 64);
            _5038.specular_and_specular_tint = _4109.Load((_4113.Load(_4662 * 4 + 0) & 16383u) * 80 + 68);
            _5038.clearcoat_and_clearcoat_roughness = _4109.Load((_4113.Load(_4662 * 4 + 0) & 16383u) * 80 + 72);
            _5038.normal_map_strength_unorm = _4109.Load((_4113.Load(_4662 * 4 + 0) & 16383u) * 80 + 76);
            _8723 = _5038.textures[0];
            _8724 = _5038.textures[1];
            _8725 = _5038.textures[2];
            _8726 = _5038.textures[3];
            _8727 = _5038.textures[4];
            _8728 = _5038.base_color[0];
            _8729 = _5038.base_color[1];
            _8730 = _5038.base_color[2];
            _8419 = _5038.flags;
            _8420 = _5038.type;
            _8421 = _5038.tangent_rotation_or_strength;
            _8422 = _5038.roughness_and_anisotropic;
            _8423 = _5038.int_ior;
            _8424 = _5038.ext_ior;
            _8425 = _5038.sheen_and_sheen_tint;
            _8426 = _5038.tint_and_metallic;
            _8427 = _5038.transmission_and_transmission_roughness;
            _8428 = _5038.specular_and_specular_tint;
            _8429 = _5038.clearcoat_and_clearcoat_roughness;
            _8430 = _5038.normal_map_strength_unorm;
            plane_N = -plane_N;
            N = -N;
            B = -B;
            T = -T;
        }
        float3 param_10 = plane_N;
        float4x4 param_11 = _4727.inv_xform;
        plane_N = TransformNormal(param_10, param_11);
        float3 param_12 = N;
        float4x4 param_13 = _4727.inv_xform;
        N = TransformNormal(param_12, param_13);
        float3 param_14 = B;
        float4x4 param_15 = _4727.inv_xform;
        B = TransformNormal(param_14, param_15);
        float3 param_16 = T;
        float4x4 param_17 = _4727.inv_xform;
        T = TransformNormal(param_16, param_17);
        float _5148 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _5158 = mad(0.5f, log2(abs(mad(_4785.t[0][0] - _4739.t[0][0], _4831.t[0][1] - _4739.t[0][1], -((_4831.t[0][0] - _4739.t[0][0]) * (_4785.t[0][1] - _4739.t[0][1])))) / _4978), log2(_5148));
        uint param_18 = uint(hash(px_index));
        float _5164 = construct_float(param_18);
        uint param_19 = uint(hash(hash(px_index)));
        float _5170 = construct_float(param_19);
        float3 col = 0.0f.xxx;
        int _5177 = ray.ray_depth & 255;
        int _5182 = (ray.ray_depth >> 8) & 255;
        int _5187 = (ray.ray_depth >> 16) & 255;
        int _5193 = (ray.ray_depth >> 24) & 255;
        int _5201 = ((_5177 + _5182) + _5187) + _5193;
        float mix_rand = frac(asfloat(_2995.Load(_3000_g_params.hi * 4 + 0)) + _5164);
        float mix_weight = 1.0f;
        float _5238;
        float _5257;
        float _5282;
        float _5351;
        while (_8420 == 4u)
        {
            float mix_val = _8421;
            if (_8724 != 4294967295u)
            {
                mix_val *= SampleBilinear(_8724, _4959, 0).x;
            }
            if (_4648)
            {
                _5238 = _8424 / _8423;
            }
            else
            {
                _5238 = _8423 / _8424;
            }
            if (_8423 != 0.0f)
            {
                float param_20 = dot(_4310, N);
                float param_21 = _5238;
                _5257 = fresnel_dielectric_cos(param_20, param_21);
            }
            else
            {
                _5257 = 1.0f;
            }
            float _5271 = mix_val;
            float _5272 = _5271 * clamp(_5257, 0.0f, 1.0f);
            mix_val = _5272;
            if (mix_rand > _5272)
            {
                if ((_8419 & 2u) != 0u)
                {
                    _5282 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _5282 = 1.0f;
                }
                mix_weight *= _5282;
                material_t _5295;
                [unroll]
                for (int _79ident = 0; _79ident < 5; _79ident++)
                {
                    _5295.textures[_79ident] = _4109.Load(_79ident * 4 + _8726 * 80 + 0);
                }
                [unroll]
                for (int _80ident = 0; _80ident < 3; _80ident++)
                {
                    _5295.base_color[_80ident] = asfloat(_4109.Load(_80ident * 4 + _8726 * 80 + 20));
                }
                _5295.flags = _4109.Load(_8726 * 80 + 32);
                _5295.type = _4109.Load(_8726 * 80 + 36);
                _5295.tangent_rotation_or_strength = asfloat(_4109.Load(_8726 * 80 + 40));
                _5295.roughness_and_anisotropic = _4109.Load(_8726 * 80 + 44);
                _5295.int_ior = asfloat(_4109.Load(_8726 * 80 + 48));
                _5295.ext_ior = asfloat(_4109.Load(_8726 * 80 + 52));
                _5295.sheen_and_sheen_tint = _4109.Load(_8726 * 80 + 56);
                _5295.tint_and_metallic = _4109.Load(_8726 * 80 + 60);
                _5295.transmission_and_transmission_roughness = _4109.Load(_8726 * 80 + 64);
                _5295.specular_and_specular_tint = _4109.Load(_8726 * 80 + 68);
                _5295.clearcoat_and_clearcoat_roughness = _4109.Load(_8726 * 80 + 72);
                _5295.normal_map_strength_unorm = _4109.Load(_8726 * 80 + 76);
                _8723 = _5295.textures[0];
                _8724 = _5295.textures[1];
                _8725 = _5295.textures[2];
                _8726 = _5295.textures[3];
                _8727 = _5295.textures[4];
                _8728 = _5295.base_color[0];
                _8729 = _5295.base_color[1];
                _8730 = _5295.base_color[2];
                _8419 = _5295.flags;
                _8420 = _5295.type;
                _8421 = _5295.tangent_rotation_or_strength;
                _8422 = _5295.roughness_and_anisotropic;
                _8423 = _5295.int_ior;
                _8424 = _5295.ext_ior;
                _8425 = _5295.sheen_and_sheen_tint;
                _8426 = _5295.tint_and_metallic;
                _8427 = _5295.transmission_and_transmission_roughness;
                _8428 = _5295.specular_and_specular_tint;
                _8429 = _5295.clearcoat_and_clearcoat_roughness;
                _8430 = _5295.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_8419 & 2u) != 0u)
                {
                    _5351 = 1.0f / mix_val;
                }
                else
                {
                    _5351 = 1.0f;
                }
                mix_weight *= _5351;
                material_t _5363;
                [unroll]
                for (int _81ident = 0; _81ident < 5; _81ident++)
                {
                    _5363.textures[_81ident] = _4109.Load(_81ident * 4 + _8727 * 80 + 0);
                }
                [unroll]
                for (int _82ident = 0; _82ident < 3; _82ident++)
                {
                    _5363.base_color[_82ident] = asfloat(_4109.Load(_82ident * 4 + _8727 * 80 + 20));
                }
                _5363.flags = _4109.Load(_8727 * 80 + 32);
                _5363.type = _4109.Load(_8727 * 80 + 36);
                _5363.tangent_rotation_or_strength = asfloat(_4109.Load(_8727 * 80 + 40));
                _5363.roughness_and_anisotropic = _4109.Load(_8727 * 80 + 44);
                _5363.int_ior = asfloat(_4109.Load(_8727 * 80 + 48));
                _5363.ext_ior = asfloat(_4109.Load(_8727 * 80 + 52));
                _5363.sheen_and_sheen_tint = _4109.Load(_8727 * 80 + 56);
                _5363.tint_and_metallic = _4109.Load(_8727 * 80 + 60);
                _5363.transmission_and_transmission_roughness = _4109.Load(_8727 * 80 + 64);
                _5363.specular_and_specular_tint = _4109.Load(_8727 * 80 + 68);
                _5363.clearcoat_and_clearcoat_roughness = _4109.Load(_8727 * 80 + 72);
                _5363.normal_map_strength_unorm = _4109.Load(_8727 * 80 + 76);
                _8723 = _5363.textures[0];
                _8724 = _5363.textures[1];
                _8725 = _5363.textures[2];
                _8726 = _5363.textures[3];
                _8727 = _5363.textures[4];
                _8728 = _5363.base_color[0];
                _8729 = _5363.base_color[1];
                _8730 = _5363.base_color[2];
                _8419 = _5363.flags;
                _8420 = _5363.type;
                _8421 = _5363.tangent_rotation_or_strength;
                _8422 = _5363.roughness_and_anisotropic;
                _8423 = _5363.int_ior;
                _8424 = _5363.ext_ior;
                _8425 = _5363.sheen_and_sheen_tint;
                _8426 = _5363.tint_and_metallic;
                _8427 = _5363.transmission_and_transmission_roughness;
                _8428 = _5363.specular_and_specular_tint;
                _8429 = _5363.clearcoat_and_clearcoat_roughness;
                _8430 = _5363.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_8723 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_8723, _4959, 0).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_8723 & 33554432u) != 0u)
            {
                float3 _9036 = normals;
                _9036.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _9036;
            }
            float3 _5445 = N;
            N = normalize(((T * normals.x) + (_5445 * normals.z)) + (B * normals.y));
            if ((_8430 & 65535u) != 65535u)
            {
                N = normalize(_5445 + ((N - _5445) * clamp(float(_8430 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_22 = plane_N;
            float3 param_23 = -_4310;
            float3 param_24 = N;
            N = ensure_valid_reflection(param_22, param_23, param_24);
        }
        float3 _5502 = ((_4877 * _4900) + (_4885 * inter.u)) + (_4893 * inter.v);
        float3 _5509 = float3(-_5502.z, 0.0f, _5502.x);
        float3 tangent = _5509;
        float3 param_25 = _5509;
        float4x4 param_26 = _4727.inv_xform;
        tangent = TransformNormal(param_25, param_26);
        if (_8421 != 0.0f)
        {
            float3 param_27 = tangent;
            float3 param_28 = N;
            float param_29 = _8421;
            tangent = rotate_around_axis(param_27, param_28, param_29);
        }
        float3 _5532 = normalize(cross(tangent, N));
        B = _5532;
        T = cross(N, _5532);
        float3 _8509 = 0.0f.xxx;
        float3 _8508 = 0.0f.xxx;
        float _8511 = 0.0f;
        float _8512 = 0.0f;
        float _8510 = 0.0f;
        bool _5544 = _3000_g_params.li_count != 0;
        bool _5550;
        if (_5544)
        {
            _5550 = _8420 != 3u;
        }
        else
        {
            _5550 = _5544;
        }
        float _8513;
        if (_5550)
        {
            float3 param_30 = _4384;
            float2 param_31 = float2(_5164, _5170);
            light_sample_t _8520 = { _8508, _8509, _8510, _8511, _8512, _8513 };
            light_sample_t param_32 = _8520;
            SampleLightSource(param_30, param_31, param_32);
            _8508 = param_32.col;
            _8509 = param_32.L;
            _8510 = param_32.area;
            _8511 = param_32.dist;
            _8512 = param_32.pdf;
            _8513 = param_32.cast_shadow;
        }
        float _5565 = dot(N, _8509);
        float3 base_color = float3(_8728, _8729, _8730);
        [branch]
        if (_8724 != 4294967295u)
        {
            base_color *= SampleBilinear(_8724, _4959, int(get_texture_lod(texSize(_8724), _5158)), true, true).xyz;
        }
        float3 tint_color = 0.0f.xxx;
        float _5609 = lum(base_color);
        [flatten]
        if (_5609 > 0.0f)
        {
            tint_color = base_color / _5609.xxx;
        }
        float roughness = clamp(float(_8422 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_8725 != 4294967295u)
        {
            roughness *= SampleBilinear(_8725, _4959, int(get_texture_lod(texSize(_8725), _5158)), false, true).x;
        }
        float _5650 = asfloat(_2995.Load((_3000_g_params.hi + 1) * 4 + 0));
        float _5654 = frac(_5650 + _5164);
        float _5660 = asfloat(_2995.Load((_3000_g_params.hi + 2) * 4 + 0));
        float _5664 = frac(_5660 + _5170);
        float _8794 = 0.0f;
        float _8793 = 0.0f;
        float _8792 = 0.0f;
        float _8541 = 0.0f;
        int _8546;
        float _8778;
        float _8779;
        float _8780;
        float _8785;
        float _8786;
        float _8787;
        [branch]
        if (_8420 == 0u)
        {
            [branch]
            if ((_8512 > 0.0f) && (_5565 > 0.0f))
            {
                float3 param_33 = -_4310;
                float3 param_34 = N;
                float3 param_35 = _8509;
                float param_36 = roughness;
                float3 param_37 = base_color;
                float4 _5704 = Evaluate_OrenDiffuse_BSDF(param_33, param_34, param_35, param_36, param_37);
                float mis_weight = 1.0f;
                if (_8510 > 0.0f)
                {
                    float param_38 = _8512;
                    float param_39 = _5704.w;
                    mis_weight = power_heuristic(param_38, param_39);
                }
                float3 _5732 = (_8508 * _5704.xyz) * ((mix_weight * mis_weight) / _8512);
                [branch]
                if (_8513 > 0.5f)
                {
                    float3 param_40 = _4384;
                    float3 param_41 = plane_N;
                    float3 _5743 = offset_ray(param_40, param_41);
                    uint _5797;
                    _5795.InterlockedAdd(8, 1u, _5797);
                    _5805.Store(_5797 * 44 + 0, asuint(_5743.x));
                    _5805.Store(_5797 * 44 + 4, asuint(_5743.y));
                    _5805.Store(_5797 * 44 + 8, asuint(_5743.z));
                    _5805.Store(_5797 * 44 + 12, asuint(_8509.x));
                    _5805.Store(_5797 * 44 + 16, asuint(_8509.y));
                    _5805.Store(_5797 * 44 + 20, asuint(_8509.z));
                    _5805.Store(_5797 * 44 + 24, asuint(_8511 - 9.9999997473787516355514526367188e-05f));
                    _5805.Store(_5797 * 44 + 28, asuint(ray.c[0] * _5732.x));
                    _5805.Store(_5797 * 44 + 32, asuint(ray.c[1] * _5732.y));
                    _5805.Store(_5797 * 44 + 36, asuint(ray.c[2] * _5732.z));
                    _5805.Store(_5797 * 44 + 40, uint(ray.xy));
                }
                else
                {
                    col += _5732;
                }
            }
            bool _5849 = _5177 < _3000_g_params.max_diff_depth;
            bool _5856;
            if (_5849)
            {
                _5856 = _5201 < _3000_g_params.max_total_depth;
            }
            else
            {
                _5856 = _5849;
            }
            [branch]
            if (_5856)
            {
                float3 param_42 = T;
                float3 param_43 = B;
                float3 param_44 = N;
                float3 param_45 = _4310;
                float param_46 = roughness;
                float3 param_47 = base_color;
                float param_48 = _5654;
                float param_49 = _5664;
                float3 param_50;
                float4 _5878 = Sample_OrenDiffuse_BSDF(param_42, param_43, param_44, param_45, param_46, param_47, param_48, param_49, param_50);
                _8546 = ray.ray_depth + 1;
                float3 param_51 = _4384;
                float3 param_52 = plane_N;
                float3 _5889 = offset_ray(param_51, param_52);
                _8778 = _5889.x;
                _8779 = _5889.y;
                _8780 = _5889.z;
                _8785 = param_50.x;
                _8786 = param_50.y;
                _8787 = param_50.z;
                _8792 = ((ray.c[0] * _5878.x) * mix_weight) / _5878.w;
                _8793 = ((ray.c[1] * _5878.y) * mix_weight) / _5878.w;
                _8794 = ((ray.c[2] * _5878.z) * mix_weight) / _5878.w;
                _8541 = _5878.w;
            }
        }
        else
        {
            [branch]
            if (_8420 == 1u)
            {
                float param_53 = 1.0f;
                float param_54 = 1.5f;
                float _5954 = fresnel_dielectric_cos(param_53, param_54);
                float _5958 = roughness * roughness;
                bool _5961 = _8512 > 0.0f;
                bool _5968;
                if (_5961)
                {
                    _5968 = (_5958 * _5958) >= 1.0000000116860974230803549289703e-07f;
                }
                else
                {
                    _5968 = _5961;
                }
                [branch]
                if (_5968 && (_5565 > 0.0f))
                {
                    float3 param_55 = T;
                    float3 param_56 = B;
                    float3 param_57 = N;
                    float3 param_58 = -_4310;
                    float3 param_59 = T;
                    float3 param_60 = B;
                    float3 param_61 = N;
                    float3 param_62 = _8509;
                    float3 param_63 = T;
                    float3 param_64 = B;
                    float3 param_65 = N;
                    float3 param_66 = normalize(_8509 - _4310);
                    float3 param_67 = tangent_from_world(param_55, param_56, param_57, param_58);
                    float3 param_68 = tangent_from_world(param_63, param_64, param_65, param_66);
                    float3 param_69 = tangent_from_world(param_59, param_60, param_61, param_62);
                    float param_70 = _5958;
                    float param_71 = _5958;
                    float param_72 = 1.5f;
                    float param_73 = _5954;
                    float3 param_74 = base_color;
                    float4 _6028 = Evaluate_GGXSpecular_BSDF(param_67, param_68, param_69, param_70, param_71, param_72, param_73, param_74);
                    float mis_weight_1 = 1.0f;
                    if (_8510 > 0.0f)
                    {
                        float param_75 = _8512;
                        float param_76 = _6028.w;
                        mis_weight_1 = power_heuristic(param_75, param_76);
                    }
                    float3 _6056 = (_8508 * _6028.xyz) * ((mix_weight * mis_weight_1) / _8512);
                    [branch]
                    if (_8513 > 0.5f)
                    {
                        float3 param_77 = _4384;
                        float3 param_78 = plane_N;
                        float3 _6067 = offset_ray(param_77, param_78);
                        uint _6114;
                        _5795.InterlockedAdd(8, 1u, _6114);
                        _5805.Store(_6114 * 44 + 0, asuint(_6067.x));
                        _5805.Store(_6114 * 44 + 4, asuint(_6067.y));
                        _5805.Store(_6114 * 44 + 8, asuint(_6067.z));
                        _5805.Store(_6114 * 44 + 12, asuint(_8509.x));
                        _5805.Store(_6114 * 44 + 16, asuint(_8509.y));
                        _5805.Store(_6114 * 44 + 20, asuint(_8509.z));
                        _5805.Store(_6114 * 44 + 24, asuint(_8511 - 9.9999997473787516355514526367188e-05f));
                        _5805.Store(_6114 * 44 + 28, asuint(ray.c[0] * _6056.x));
                        _5805.Store(_6114 * 44 + 32, asuint(ray.c[1] * _6056.y));
                        _5805.Store(_6114 * 44 + 36, asuint(ray.c[2] * _6056.z));
                        _5805.Store(_6114 * 44 + 40, uint(ray.xy));
                    }
                    else
                    {
                        col += _6056;
                    }
                }
                bool _6153 = _5182 < _3000_g_params.max_spec_depth;
                bool _6160;
                if (_6153)
                {
                    _6160 = _5201 < _3000_g_params.max_total_depth;
                }
                else
                {
                    _6160 = _6153;
                }
                [branch]
                if (_6160)
                {
                    float3 param_79 = T;
                    float3 param_80 = B;
                    float3 param_81 = N;
                    float3 param_82 = _4310;
                    float3 param_83;
                    float4 _6179 = Sample_GGXSpecular_BSDF(param_79, param_80, param_81, param_82, roughness, 0.0f, 1.5f, _5954, base_color, _5654, _5664, param_83);
                    _8546 = ray.ray_depth + 256;
                    float3 param_84 = _4384;
                    float3 param_85 = plane_N;
                    float3 _6191 = offset_ray(param_84, param_85);
                    _8778 = _6191.x;
                    _8779 = _6191.y;
                    _8780 = _6191.z;
                    _8785 = param_83.x;
                    _8786 = param_83.y;
                    _8787 = param_83.z;
                    _8792 = ((ray.c[0] * _6179.x) * mix_weight) / _6179.w;
                    _8793 = ((ray.c[1] * _6179.y) * mix_weight) / _6179.w;
                    _8794 = ((ray.c[2] * _6179.z) * mix_weight) / _6179.w;
                    _8541 = _6179.w;
                }
            }
            else
            {
                [branch]
                if (_8420 == 2u)
                {
                    float _6254;
                    if (_4648)
                    {
                        _6254 = _8423 / _8424;
                    }
                    else
                    {
                        _6254 = _8424 / _8423;
                    }
                    float _6272 = roughness * roughness;
                    bool _6275 = _8512 > 0.0f;
                    bool _6282;
                    if (_6275)
                    {
                        _6282 = (_6272 * _6272) >= 1.0000000116860974230803549289703e-07f;
                    }
                    else
                    {
                        _6282 = _6275;
                    }
                    [branch]
                    if (_6282 && (_5565 < 0.0f))
                    {
                        float3 param_86 = T;
                        float3 param_87 = B;
                        float3 param_88 = N;
                        float3 param_89 = -_4310;
                        float3 param_90 = T;
                        float3 param_91 = B;
                        float3 param_92 = N;
                        float3 param_93 = _8509;
                        float3 param_94 = T;
                        float3 param_95 = B;
                        float3 param_96 = N;
                        float3 param_97 = normalize(_8509 - (_4310 * _6254));
                        float3 param_98 = tangent_from_world(param_86, param_87, param_88, param_89);
                        float3 param_99 = tangent_from_world(param_94, param_95, param_96, param_97);
                        float3 param_100 = tangent_from_world(param_90, param_91, param_92, param_93);
                        float param_101 = _6272;
                        float param_102 = _6254;
                        float3 param_103 = base_color;
                        float4 _6341 = Evaluate_GGXRefraction_BSDF(param_98, param_99, param_100, param_101, param_102, param_103);
                        float mis_weight_2 = 1.0f;
                        if (_8510 > 0.0f)
                        {
                            float param_104 = _8512;
                            float param_105 = _6341.w;
                            mis_weight_2 = power_heuristic(param_104, param_105);
                        }
                        float3 _6369 = (_8508 * _6341.xyz) * ((mix_weight * mis_weight_2) / _8512);
                        [branch]
                        if (_8513 > 0.5f)
                        {
                            float3 param_106 = _4384;
                            float3 param_107 = -plane_N;
                            float3 _6381 = offset_ray(param_106, param_107);
                            uint _6428;
                            _5795.InterlockedAdd(8, 1u, _6428);
                            _5805.Store(_6428 * 44 + 0, asuint(_6381.x));
                            _5805.Store(_6428 * 44 + 4, asuint(_6381.y));
                            _5805.Store(_6428 * 44 + 8, asuint(_6381.z));
                            _5805.Store(_6428 * 44 + 12, asuint(_8509.x));
                            _5805.Store(_6428 * 44 + 16, asuint(_8509.y));
                            _5805.Store(_6428 * 44 + 20, asuint(_8509.z));
                            _5805.Store(_6428 * 44 + 24, asuint(_8511 - 9.9999997473787516355514526367188e-05f));
                            _5805.Store(_6428 * 44 + 28, asuint(ray.c[0] * _6369.x));
                            _5805.Store(_6428 * 44 + 32, asuint(ray.c[1] * _6369.y));
                            _5805.Store(_6428 * 44 + 36, asuint(ray.c[2] * _6369.z));
                            _5805.Store(_6428 * 44 + 40, uint(ray.xy));
                        }
                        else
                        {
                            col += _6369;
                        }
                    }
                    bool _6467 = _5187 < _3000_g_params.max_refr_depth;
                    bool _6474;
                    if (_6467)
                    {
                        _6474 = _5201 < _3000_g_params.max_total_depth;
                    }
                    else
                    {
                        _6474 = _6467;
                    }
                    [branch]
                    if (_6474)
                    {
                        float3 param_108 = T;
                        float3 param_109 = B;
                        float3 param_110 = N;
                        float3 param_111 = _4310;
                        float param_112 = roughness;
                        float param_113 = _6254;
                        float3 param_114 = base_color;
                        float param_115 = _5654;
                        float param_116 = _5664;
                        float4 param_117;
                        float4 _6498 = Sample_GGXRefraction_BSDF(param_108, param_109, param_110, param_111, param_112, param_113, param_114, param_115, param_116, param_117);
                        _8546 = ray.ray_depth + 65536;
                        _8792 = ((ray.c[0] * _6498.x) * mix_weight) / _6498.w;
                        _8793 = ((ray.c[1] * _6498.y) * mix_weight) / _6498.w;
                        _8794 = ((ray.c[2] * _6498.z) * mix_weight) / _6498.w;
                        _8541 = _6498.w;
                        float3 param_118 = _4384;
                        float3 param_119 = -plane_N;
                        float3 _6553 = offset_ray(param_118, param_119);
                        _8778 = _6553.x;
                        _8779 = _6553.y;
                        _8780 = _6553.z;
                        _8785 = param_117.x;
                        _8786 = param_117.y;
                        _8787 = param_117.z;
                    }
                }
                else
                {
                    [branch]
                    if (_8420 == 3u)
                    {
                        if ((_8419 & 4u) != 0u)
                        {
                            float3 env_col_1 = _3000_g_params.env_col.xyz;
                            uint _6592 = asuint(_3000_g_params.env_col.w);
                            if (_6592 != 4294967295u)
                            {
                                env_col_1 *= SampleLatlong_RGBE(_6592, _4310, _3000_g_params.env_rotation);
                            }
                            base_color *= env_col_1;
                        }
                        col += (base_color * (mix_weight * _8421));
                    }
                    else
                    {
                        [branch]
                        if (_8420 == 5u)
                        {
                            bool _6626 = _5193 < _3000_g_params.max_transp_depth;
                            bool _6633;
                            if (_6626)
                            {
                                _6633 = _5201 < _3000_g_params.max_total_depth;
                            }
                            else
                            {
                                _6633 = _6626;
                            }
                            [branch]
                            if (_6633)
                            {
                                _8546 = ray.ray_depth + 16777216;
                                _8541 = ray.pdf;
                                float3 param_120 = _4384;
                                float3 param_121 = -plane_N;
                                float3 _6650 = offset_ray(param_120, param_121);
                                _8778 = _6650.x;
                                _8779 = _6650.y;
                                _8780 = _6650.z;
                                _8785 = ray.d[0];
                                _8786 = ray.d[1];
                                _8787 = ray.d[2];
                                _8792 = ray.c[0];
                                _8793 = ray.c[1];
                                _8794 = ray.c[2];
                            }
                        }
                        else
                        {
                            if (_8420 == 6u)
                            {
                                float metallic = clamp(float((_8426 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                [branch]
                                if (_8726 != 4294967295u)
                                {
                                    metallic *= SampleBilinear(_8726, _4959, int(get_texture_lod(texSize(_8726), _5158))).x;
                                }
                                float specular = clamp(float(_8428 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                [branch]
                                if (_8727 != 4294967295u)
                                {
                                    specular *= SampleBilinear(_8727, _4959, int(get_texture_lod(texSize(_8727), _5158))).x;
                                }
                                float _6760 = clamp(float(_8429 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                float _6768 = clamp(float((_8429 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                float _6775 = clamp(float(_8425 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                float3 _6797 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_8428 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                                float _6804 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                                float param_122 = 1.0f;
                                float param_123 = _6804;
                                float _6809 = fresnel_dielectric_cos(param_122, param_123);
                                float param_124 = dot(_4310, N);
                                float param_125 = _6804;
                                float param_126;
                                float param_127;
                                float param_128;
                                float param_129;
                                get_lobe_weights(lerp(_5609, 1.0f, _6775), lum(lerp(_6797, 1.0f.xxx, ((fresnel_dielectric_cos(param_124, param_125) - _6809) / (1.0f - _6809)).xxx)), specular, metallic, clamp(float(_8427 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _6760, param_126, param_127, param_128, param_129);
                                float3 _6863 = lerp(1.0f.xxx, tint_color, clamp(float((_8425 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _6775;
                                float _6866;
                                if (_4648)
                                {
                                    _6866 = _8423 / _8424;
                                }
                                else
                                {
                                    _6866 = _8424 / _8423;
                                }
                                float param_130 = dot(_4310, N);
                                float param_131 = 1.0f / _6866;
                                float _6889 = fresnel_dielectric_cos(param_130, param_131);
                                float _6896 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _6760))) - 1.0f;
                                float param_132 = 1.0f;
                                float param_133 = _6896;
                                float _6901 = fresnel_dielectric_cos(param_132, param_133);
                                float _6905 = _6768 * _6768;
                                float _6918 = mad(roughness - 1.0f, 1.0f - clamp(float((_8427 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                                float _6922 = _6918 * _6918;
                                [branch]
                                if (_8512 > 0.0f)
                                {
                                    float3 lcol_1 = 0.0f.xxx;
                                    float bsdf_pdf = 0.0f;
                                    bool _6933 = _5565 > 0.0f;
                                    [branch]
                                    if ((param_126 > 1.0000000116860974230803549289703e-07f) && _6933)
                                    {
                                        float3 param_134 = -_4310;
                                        float3 param_135 = N;
                                        float3 param_136 = _8509;
                                        float param_137 = roughness;
                                        float3 param_138 = base_color.xyz;
                                        float3 param_139 = _6863;
                                        bool param_140 = false;
                                        float4 _6953 = Evaluate_PrincipledDiffuse_BSDF(param_134, param_135, param_136, param_137, param_138, param_139, param_140);
                                        bsdf_pdf = mad(param_126, _6953.w, bsdf_pdf);
                                        lcol_1 += (((_8508 * _5565) * (_6953 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * _8512).xxx);
                                    }
                                    float3 H;
                                    [flatten]
                                    if (_6933)
                                    {
                                        H = normalize(_8509 - _4310);
                                    }
                                    else
                                    {
                                        H = normalize(_8509 - (_4310 * _6866));
                                    }
                                    float _6999 = roughness * roughness;
                                    float _7010 = sqrt(mad(-0.89999997615814208984375f, clamp(float((_8422 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f));
                                    float _7014 = _6999 / _7010;
                                    float _7018 = _6999 * _7010;
                                    float3 param_141 = T;
                                    float3 param_142 = B;
                                    float3 param_143 = N;
                                    float3 param_144 = -_4310;
                                    float3 _7029 = tangent_from_world(param_141, param_142, param_143, param_144);
                                    float3 param_145 = T;
                                    float3 param_146 = B;
                                    float3 param_147 = N;
                                    float3 param_148 = _8509;
                                    float3 _7040 = tangent_from_world(param_145, param_146, param_147, param_148);
                                    float3 param_149 = T;
                                    float3 param_150 = B;
                                    float3 param_151 = N;
                                    float3 param_152 = H;
                                    float3 _7050 = tangent_from_world(param_149, param_150, param_151, param_152);
                                    bool _7052 = param_127 > 0.0f;
                                    bool _7059;
                                    if (_7052)
                                    {
                                        _7059 = (_7014 * _7018) >= 1.0000000116860974230803549289703e-07f;
                                    }
                                    else
                                    {
                                        _7059 = _7052;
                                    }
                                    [branch]
                                    if (_7059 && _6933)
                                    {
                                        float3 param_153 = _7029;
                                        float3 param_154 = _7050;
                                        float3 param_155 = _7040;
                                        float param_156 = _7014;
                                        float param_157 = _7018;
                                        float param_158 = _6804;
                                        float param_159 = _6809;
                                        float3 param_160 = _6797;
                                        float4 _7082 = Evaluate_GGXSpecular_BSDF(param_153, param_154, param_155, param_156, param_157, param_158, param_159, param_160);
                                        bsdf_pdf = mad(param_127, _7082.w, bsdf_pdf);
                                        lcol_1 += ((_8508 * _7082.xyz) / _8512.xxx);
                                    }
                                    bool _7101 = param_128 > 0.0f;
                                    bool _7108;
                                    if (_7101)
                                    {
                                        _7108 = (_6905 * _6905) >= 1.0000000116860974230803549289703e-07f;
                                    }
                                    else
                                    {
                                        _7108 = _7101;
                                    }
                                    [branch]
                                    if (_7108 && _6933)
                                    {
                                        float3 param_161 = _7029;
                                        float3 param_162 = _7050;
                                        float3 param_163 = _7040;
                                        float param_164 = _6905;
                                        float param_165 = _6896;
                                        float param_166 = _6901;
                                        float4 _7127 = Evaluate_PrincipledClearcoat_BSDF(param_161, param_162, param_163, param_164, param_165, param_166);
                                        bsdf_pdf = mad(param_128, _7127.w, bsdf_pdf);
                                        lcol_1 += (((_8508 * 0.25f) * _7127.xyz) / _8512.xxx);
                                    }
                                    [branch]
                                    if (param_129 > 0.0f)
                                    {
                                        bool _7151 = _6889 != 0.0f;
                                        bool _7158;
                                        if (_7151)
                                        {
                                            _7158 = (_6999 * _6999) >= 1.0000000116860974230803549289703e-07f;
                                        }
                                        else
                                        {
                                            _7158 = _7151;
                                        }
                                        [branch]
                                        if (_7158 && _6933)
                                        {
                                            float3 param_167 = _7029;
                                            float3 param_168 = _7050;
                                            float3 param_169 = _7040;
                                            float param_170 = _6999;
                                            float param_171 = _6999;
                                            float param_172 = 1.0f;
                                            float param_173 = 0.0f;
                                            float3 param_174 = 1.0f.xxx;
                                            float4 _7178 = Evaluate_GGXSpecular_BSDF(param_167, param_168, param_169, param_170, param_171, param_172, param_173, param_174);
                                            bsdf_pdf = mad(param_129 * _6889, _7178.w, bsdf_pdf);
                                            lcol_1 += ((_8508 * _7178.xyz) * (_6889 / _8512));
                                        }
                                        bool _7200 = _6889 != 1.0f;
                                        bool _7207;
                                        if (_7200)
                                        {
                                            _7207 = (_6922 * _6922) >= 1.0000000116860974230803549289703e-07f;
                                        }
                                        else
                                        {
                                            _7207 = _7200;
                                        }
                                        [branch]
                                        if (_7207 && (_5565 < 0.0f))
                                        {
                                            float3 param_175 = _7029;
                                            float3 param_176 = _7050;
                                            float3 param_177 = _7040;
                                            float param_178 = _6922;
                                            float param_179 = _6866;
                                            float3 param_180 = base_color;
                                            float4 _7226 = Evaluate_GGXRefraction_BSDF(param_175, param_176, param_177, param_178, param_179, param_180);
                                            float _7229 = 1.0f - _6889;
                                            bsdf_pdf = mad(param_129 * _7229, _7226.w, bsdf_pdf);
                                            lcol_1 += ((_8508 * _7226.xyz) * (_7229 / _8512));
                                        }
                                    }
                                    float mis_weight_3 = 1.0f;
                                    [flatten]
                                    if (_8510 > 0.0f)
                                    {
                                        float param_181 = _8512;
                                        float param_182 = bsdf_pdf;
                                        mis_weight_3 = power_heuristic(param_181, param_182);
                                    }
                                    lcol_1 *= (mix_weight * mis_weight_3);
                                    [branch]
                                    if (_8513 > 0.5f)
                                    {
                                        float3 _7274;
                                        if (_5565 < 0.0f)
                                        {
                                            _7274 = -plane_N;
                                        }
                                        else
                                        {
                                            _7274 = plane_N;
                                        }
                                        float3 param_183 = _4384;
                                        float3 param_184 = _7274;
                                        float3 _7285 = offset_ray(param_183, param_184);
                                        uint _7332;
                                        _5795.InterlockedAdd(8, 1u, _7332);
                                        _5805.Store(_7332 * 44 + 0, asuint(_7285.x));
                                        _5805.Store(_7332 * 44 + 4, asuint(_7285.y));
                                        _5805.Store(_7332 * 44 + 8, asuint(_7285.z));
                                        _5805.Store(_7332 * 44 + 12, asuint(_8509.x));
                                        _5805.Store(_7332 * 44 + 16, asuint(_8509.y));
                                        _5805.Store(_7332 * 44 + 20, asuint(_8509.z));
                                        _5805.Store(_7332 * 44 + 24, asuint(_8511 - 9.9999997473787516355514526367188e-05f));
                                        _5805.Store(_7332 * 44 + 28, asuint(ray.c[0] * lcol_1.x));
                                        _5805.Store(_7332 * 44 + 32, asuint(ray.c[1] * lcol_1.y));
                                        _5805.Store(_7332 * 44 + 36, asuint(ray.c[2] * lcol_1.z));
                                        _5805.Store(_7332 * 44 + 40, uint(ray.xy));
                                    }
                                    else
                                    {
                                        col += lcol_1;
                                    }
                                }
                                [branch]
                                if (mix_rand < param_126)
                                {
                                    bool _7376 = _5177 < _3000_g_params.max_diff_depth;
                                    bool _7383;
                                    if (_7376)
                                    {
                                        _7383 = _5201 < _3000_g_params.max_total_depth;
                                    }
                                    else
                                    {
                                        _7383 = _7376;
                                    }
                                    if (_7383)
                                    {
                                        float3 param_185 = T;
                                        float3 param_186 = B;
                                        float3 param_187 = N;
                                        float3 param_188 = _4310;
                                        float param_189 = roughness;
                                        float3 param_190 = base_color.xyz;
                                        float3 param_191 = _6863;
                                        bool param_192 = false;
                                        float param_193 = _5654;
                                        float param_194 = _5664;
                                        float3 param_195;
                                        float4 _7408 = Sample_PrincipledDiffuse_BSDF(param_185, param_186, param_187, param_188, param_189, param_190, param_191, param_192, param_193, param_194, param_195);
                                        float3 _7414 = _7408.xyz * (1.0f - metallic);
                                        _8546 = ray.ray_depth + 1;
                                        float3 param_196 = _4384;
                                        float3 param_197 = plane_N;
                                        float3 _7430 = offset_ray(param_196, param_197);
                                        _8778 = _7430.x;
                                        _8779 = _7430.y;
                                        _8780 = _7430.z;
                                        _8785 = param_195.x;
                                        _8786 = param_195.y;
                                        _8787 = param_195.z;
                                        _8792 = ((ray.c[0] * _7414.x) * mix_weight) / param_126;
                                        _8793 = ((ray.c[1] * _7414.y) * mix_weight) / param_126;
                                        _8794 = ((ray.c[2] * _7414.z) * mix_weight) / param_126;
                                        _8541 = _7408.w;
                                    }
                                }
                                else
                                {
                                    float _7486 = param_126 + param_127;
                                    [branch]
                                    if (mix_rand < _7486)
                                    {
                                        bool _7493 = _5182 < _3000_g_params.max_spec_depth;
                                        bool _7500;
                                        if (_7493)
                                        {
                                            _7500 = _5201 < _3000_g_params.max_total_depth;
                                        }
                                        else
                                        {
                                            _7500 = _7493;
                                        }
                                        if (_7500)
                                        {
                                            float3 param_198 = T;
                                            float3 param_199 = B;
                                            float3 param_200 = N;
                                            float3 param_201 = _4310;
                                            float3 param_202;
                                            float4 _7527 = Sample_GGXSpecular_BSDF(param_198, param_199, param_200, param_201, roughness, clamp(float((_8422 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _6804, _6809, _6797, _5654, _5664, param_202);
                                            float _7532 = _7527.w * param_127;
                                            _8546 = ray.ray_depth + 256;
                                            _8792 = ((ray.c[0] * _7527.x) * mix_weight) / _7532;
                                            _8793 = ((ray.c[1] * _7527.y) * mix_weight) / _7532;
                                            _8794 = ((ray.c[2] * _7527.z) * mix_weight) / _7532;
                                            _8541 = _7532;
                                            float3 param_203 = _4384;
                                            float3 param_204 = plane_N;
                                            float3 _7579 = offset_ray(param_203, param_204);
                                            _8778 = _7579.x;
                                            _8779 = _7579.y;
                                            _8780 = _7579.z;
                                            _8785 = param_202.x;
                                            _8786 = param_202.y;
                                            _8787 = param_202.z;
                                        }
                                    }
                                    else
                                    {
                                        float _7604 = _7486 + param_128;
                                        [branch]
                                        if (mix_rand < _7604)
                                        {
                                            bool _7611 = _5182 < _3000_g_params.max_spec_depth;
                                            bool _7618;
                                            if (_7611)
                                            {
                                                _7618 = _5201 < _3000_g_params.max_total_depth;
                                            }
                                            else
                                            {
                                                _7618 = _7611;
                                            }
                                            if (_7618)
                                            {
                                                float3 param_205 = T;
                                                float3 param_206 = B;
                                                float3 param_207 = N;
                                                float3 param_208 = _4310;
                                                float param_209 = _6905;
                                                float param_210 = _6896;
                                                float param_211 = _6901;
                                                float param_212 = _5654;
                                                float param_213 = _5664;
                                                float3 param_214;
                                                float4 _7642 = Sample_PrincipledClearcoat_BSDF(param_205, param_206, param_207, param_208, param_209, param_210, param_211, param_212, param_213, param_214);
                                                float _7647 = _7642.w * param_128;
                                                _8546 = ray.ray_depth + 256;
                                                _8792 = (((0.25f * ray.c[0]) * _7642.x) * mix_weight) / _7647;
                                                _8793 = (((0.25f * ray.c[1]) * _7642.y) * mix_weight) / _7647;
                                                _8794 = (((0.25f * ray.c[2]) * _7642.z) * mix_weight) / _7647;
                                                _8541 = _7647;
                                                float3 param_215 = _4384;
                                                float3 param_216 = plane_N;
                                                float3 _7697 = offset_ray(param_215, param_216);
                                                _8778 = _7697.x;
                                                _8779 = _7697.y;
                                                _8780 = _7697.z;
                                                _8785 = param_214.x;
                                                _8786 = param_214.y;
                                                _8787 = param_214.z;
                                            }
                                        }
                                        else
                                        {
                                            bool _7719 = mix_rand >= _6889;
                                            bool _7726;
                                            if (_7719)
                                            {
                                                _7726 = _5187 < _3000_g_params.max_refr_depth;
                                            }
                                            else
                                            {
                                                _7726 = _7719;
                                            }
                                            bool _7740;
                                            if (!_7726)
                                            {
                                                bool _7732 = mix_rand < _6889;
                                                bool _7739;
                                                if (_7732)
                                                {
                                                    _7739 = _5182 < _3000_g_params.max_spec_depth;
                                                }
                                                else
                                                {
                                                    _7739 = _7732;
                                                }
                                                _7740 = _7739;
                                            }
                                            else
                                            {
                                                _7740 = _7726;
                                            }
                                            bool _7747;
                                            if (_7740)
                                            {
                                                _7747 = _5201 < _3000_g_params.max_total_depth;
                                            }
                                            else
                                            {
                                                _7747 = _7740;
                                            }
                                            [branch]
                                            if (_7747)
                                            {
                                                float _7755 = mix_rand;
                                                float _7759 = (_7755 - _7604) / param_129;
                                                mix_rand = _7759;
                                                float4 F;
                                                float3 V;
                                                [branch]
                                                if (_7759 < _6889)
                                                {
                                                    float3 param_217 = T;
                                                    float3 param_218 = B;
                                                    float3 param_219 = N;
                                                    float3 param_220 = _4310;
                                                    float3 param_221;
                                                    float4 _7779 = Sample_GGXSpecular_BSDF(param_217, param_218, param_219, param_220, roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, _5654, _5664, param_221);
                                                    V = param_221;
                                                    F = _7779;
                                                    _8546 = ray.ray_depth + 256;
                                                    float3 param_222 = _4384;
                                                    float3 param_223 = plane_N;
                                                    float3 _7790 = offset_ray(param_222, param_223);
                                                    _8778 = _7790.x;
                                                    _8779 = _7790.y;
                                                    _8780 = _7790.z;
                                                }
                                                else
                                                {
                                                    float3 param_224 = T;
                                                    float3 param_225 = B;
                                                    float3 param_226 = N;
                                                    float3 param_227 = _4310;
                                                    float param_228 = _6918;
                                                    float param_229 = _6866;
                                                    float3 param_230 = base_color;
                                                    float param_231 = _5654;
                                                    float param_232 = _5664;
                                                    float4 param_233;
                                                    float4 _7821 = Sample_GGXRefraction_BSDF(param_224, param_225, param_226, param_227, param_228, param_229, param_230, param_231, param_232, param_233);
                                                    F = _7821;
                                                    V = param_233.xyz;
                                                    _8546 = ray.ray_depth + 65536;
                                                    float3 param_234 = _4384;
                                                    float3 param_235 = -plane_N;
                                                    float3 _7835 = offset_ray(param_234, param_235);
                                                    _8778 = _7835.x;
                                                    _8779 = _7835.y;
                                                    _8780 = _7835.z;
                                                }
                                                float4 _9184 = F;
                                                float _7848 = _9184.w * param_129;
                                                float4 _9186 = _9184;
                                                _9186.w = _7848;
                                                F = _9186;
                                                _8792 = ((ray.c[0] * _9184.x) * mix_weight) / _7848;
                                                _8793 = ((ray.c[1] * _9184.y) * mix_weight) / _7848;
                                                _8794 = ((ray.c[2] * _9184.z) * mix_weight) / _7848;
                                                _8541 = _7848;
                                                _8785 = V.x;
                                                _8786 = V.y;
                                                _8787 = V.z;
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
        float _7908 = max(_8792, max(_8793, _8794));
        float _7921;
        if (_5201 >= _3000_g_params.termination_start_depth)
        {
            _7921 = max(0.0500000007450580596923828125f, 1.0f - _7908);
        }
        else
        {
            _7921 = 0.0f;
        }
        bool _7935 = (frac(asfloat(_2995.Load((_3000_g_params.hi + 6) * 4 + 0)) + _5164) >= _7921) && (_7908 > 0.0f);
        bool _7941;
        if (_7935)
        {
            _7941 = _8541 > 0.0f;
        }
        else
        {
            _7941 = _7935;
        }
        [branch]
        if (_7941)
        {
            float _7945 = 1.0f - _7921;
            float _7947 = _8792;
            float _7948 = _7947 / _7945;
            _8792 = _7948;
            float _7953 = _8793;
            float _7954 = _7953 / _7945;
            _8793 = _7954;
            float _7959 = _8794;
            float _7960 = _7959 / _7945;
            _8794 = _7960;
            uint _7964;
            _5795.InterlockedAdd(0, 1u, _7964);
            _7972.Store(_7964 * 56 + 0, asuint(_8778));
            _7972.Store(_7964 * 56 + 4, asuint(_8779));
            _7972.Store(_7964 * 56 + 8, asuint(_8780));
            _7972.Store(_7964 * 56 + 12, asuint(_8785));
            _7972.Store(_7964 * 56 + 16, asuint(_8786));
            _7972.Store(_7964 * 56 + 20, asuint(_8787));
            _7972.Store(_7964 * 56 + 24, asuint(_8541));
            _7972.Store(_7964 * 56 + 28, asuint(_7948));
            _7972.Store(_7964 * 56 + 32, asuint(_7954));
            _7972.Store(_7964 * 56 + 36, asuint(_7960));
            _7972.Store(_7964 * 56 + 40, asuint(_5148));
            _7972.Store(_7964 * 56 + 44, asuint(ray.cone_spread));
            _7972.Store(_7964 * 56 + 48, uint(ray.xy));
            _7972.Store(_7964 * 56 + 52, uint(_8546));
        }
        _8180 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _8180;
}

void comp_main()
{
    do
    {
        bool _8040 = gl_GlobalInvocationID.x >= _3000_g_params.img_size.x;
        bool _8049;
        if (!_8040)
        {
            _8049 = gl_GlobalInvocationID.y >= _3000_g_params.img_size.y;
        }
        else
        {
            _8049 = _8040;
        }
        if (_8049)
        {
            break;
        }
        int _8056 = int(gl_GlobalInvocationID.x);
        int _8060 = int(gl_GlobalInvocationID.y);
        int _8068 = (_8060 * int(_3000_g_params.img_size.x)) + _8056;
        hit_data_t _8080;
        _8080.mask = int(_8076.Load(_8068 * 24 + 0));
        _8080.obj_index = int(_8076.Load(_8068 * 24 + 4));
        _8080.prim_index = int(_8076.Load(_8068 * 24 + 8));
        _8080.t = asfloat(_8076.Load(_8068 * 24 + 12));
        _8080.u = asfloat(_8076.Load(_8068 * 24 + 16));
        _8080.v = asfloat(_8076.Load(_8068 * 24 + 20));
        ray_data_t _8100;
        [unroll]
        for (int _83ident = 0; _83ident < 3; _83ident++)
        {
            _8100.o[_83ident] = asfloat(_8097.Load(_83ident * 4 + _8068 * 56 + 0));
        }
        [unroll]
        for (int _84ident = 0; _84ident < 3; _84ident++)
        {
            _8100.d[_84ident] = asfloat(_8097.Load(_84ident * 4 + _8068 * 56 + 12));
        }
        _8100.pdf = asfloat(_8097.Load(_8068 * 56 + 24));
        [unroll]
        for (int _85ident = 0; _85ident < 3; _85ident++)
        {
            _8100.c[_85ident] = asfloat(_8097.Load(_85ident * 4 + _8068 * 56 + 28));
        }
        _8100.cone_width = asfloat(_8097.Load(_8068 * 56 + 40));
        _8100.cone_spread = asfloat(_8097.Load(_8068 * 56 + 44));
        _8100.xy = int(_8097.Load(_8068 * 56 + 48));
        _8100.ray_depth = int(_8097.Load(_8068 * 56 + 52));
        int param = _8068;
        hit_data_t _8241 = { _8080.mask, _8080.obj_index, _8080.prim_index, _8080.t, _8080.u, _8080.v };
        hit_data_t param_1 = _8241;
        float _8279[3] = { _8100.c[0], _8100.c[1], _8100.c[2] };
        float _8272[3] = { _8100.d[0], _8100.d[1], _8100.d[2] };
        float _8265[3] = { _8100.o[0], _8100.o[1], _8100.o[2] };
        ray_data_t _8258 = { _8265, _8272, _8100.pdf, _8279, _8100.cone_width, _8100.cone_spread, _8100.xy, _8100.ray_depth };
        ray_data_t param_2 = _8258;
        float3 _8142 = ShadeSurface(param, param_1, param_2);
        g_out_img[int2(_8056, _8060)] = float4(_8142, 1.0f);
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
