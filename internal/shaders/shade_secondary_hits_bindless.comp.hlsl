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
ByteAddressBuffer _3807 : register(t11, space0);
ByteAddressBuffer _3832 : register(t13, space0);
ByteAddressBuffer _3836 : register(t14, space0);
ByteAddressBuffer _4147 : register(t10, space0);
ByteAddressBuffer _4151 : register(t9, space0);
ByteAddressBuffer _4795 : register(t12, space0);
RWByteAddressBuffer _5870 : register(u3, space0);
RWByteAddressBuffer _5880 : register(u2, space0);
RWByteAddressBuffer _8130 : register(u1, space0);
ByteAddressBuffer _8213 : register(t5, space0);
ByteAddressBuffer _8239 : register(t4, space0);
ByteAddressBuffer _8340 : register(t8, space0);
cbuffer UniformParams
{
    Params _3000_g_params : packoffset(c0);
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
        float4 _9102 = res;
        _9102.x = _884.x;
        float4 _9104 = _9102;
        _9104.y = _884.y;
        float4 _9106 = _9104;
        _9106.z = _884.z;
        res = _9106;
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
    float3 _8352;
    do
    {
        float _1165 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1165)
        {
            _8352 = N;
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
            float _9408 = (-0.5f) / _1205;
            float param_1 = mad(_9408, _1229, 1.0f);
            float _1261 = safe_sqrtf(param_1);
            float param_2 = _1230;
            float _1264 = safe_sqrtf(param_2);
            float2 _1265 = float2(_1261, _1264);
            float param_3 = mad(_9408, _1236, 1.0f);
            float _1270 = safe_sqrtf(param_3);
            float param_4 = _1237;
            float _1273 = safe_sqrtf(param_4);
            float2 _1274 = float2(_1270, _1273);
            float _9410 = -_1193;
            float _1290 = mad(2.0f * mad(_1261, _1189, _1264 * _1193), _1264, _9410);
            float _1306 = mad(2.0f * mad(_1270, _1189, _1273 * _1193), _1273, _9410);
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
                _8352 = Ng;
                break;
            }
            float _1343 = valid1 ? _1230 : _1237;
            float param_5 = 1.0f - _1343;
            float param_6 = _1343;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _8352 = (_1185 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _8352;
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
    float3 _8377;
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
            _8377 = N;
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
        _8377 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _8377;
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
            float2 _9089 = origin;
            _9089.x = origin.x + _step;
            origin = _9089;
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
            float2 _9092 = origin;
            _9092.y = origin.y + _step;
            origin = _9092;
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
        float3 _9169 = sampled_dir;
        float3 _3159 = ((param * _9169.x) + (param_1 * _9169.y)) + (_3116 * _9169.z);
        sampled_dir = _3159;
        float3 _3168 = _3043.param1.xyz + (_3159 * _3043.param2.w);
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
        [branch]
        if (_3043.param3.x > 0.0f)
        {
            float _3228 = -dot(ls.L, _3043.param2.xyz);
            if (_3228 > 0.0f)
            {
                ls.col *= clamp((_3043.param3.x - acos(clamp(_3228, 0.0f, 1.0f))) / _3043.param3.y, 0.0f, 1.0f);
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
                float3 _3369 = ((_3043.param1.xyz + (_3043.param2.xyz * (frac(asfloat(_2995.Load((_3000_g_params.hi + 4) * 4 + 0)) + sample_off.x) - 0.5f))) + (_3043.param3.xyz * (frac(asfloat(_2995.Load((_3000_g_params.hi + 5) * 4 + 0)) + sample_off.y) - 0.5f))) - P;
                ls.dist = length(_3369);
                ls.L = _3369 / ls.dist.xxx;
                ls.area = _3043.param1.w;
                float _3392 = dot(-ls.L, normalize(cross(_3043.param2.xyz, _3043.param3.xyz)));
                if (_3392 > 0.0f)
                {
                    ls.pdf = (ls.dist * ls.dist) / (ls.area * _3392);
                }
                if ((_3043.type_and_param0.x & 64u) == 0u)
                {
                    ls.area = 0.0f;
                }
                [branch]
                if ((_3043.type_and_param0.w & 128u) != 0u)
                {
                    float3 env_col = _3000_g_params.env_col.xyz;
                    uint _3432 = asuint(_3000_g_params.env_col.w);
                    if (_3432 != 4294967295u)
                    {
                        env_col *= SampleLatlong_RGBE(_3432, ls.L, _3000_g_params.env_rotation);
                    }
                    ls.col *= env_col;
                }
            }
            else
            {
                [branch]
                if (_3078 == 5u)
                {
                    float2 _3495 = (float2(frac(asfloat(_2995.Load((_3000_g_params.hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_2995.Load((_3000_g_params.hi + 5) * 4 + 0)) + sample_off.y)) * 2.0f) - 1.0f.xx;
                    float2 offset = _3495;
                    bool _3498 = _3495.x != 0.0f;
                    bool _3504;
                    if (_3498)
                    {
                        _3504 = offset.y != 0.0f;
                    }
                    else
                    {
                        _3504 = _3498;
                    }
                    if (_3504)
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
                        float _3537 = 0.5f * r;
                        offset = float2(_3537 * cos(theta), _3537 * sin(theta));
                    }
                    float3 _3563 = ((_3043.param1.xyz + (_3043.param2.xyz * offset.x)) + (_3043.param3.xyz * offset.y)) - P;
                    ls.dist = length(_3563);
                    ls.L = _3563 / ls.dist.xxx;
                    ls.area = _3043.param1.w;
                    float _3586 = dot(-ls.L, normalize(cross(_3043.param2.xyz, _3043.param3.xyz)));
                    [flatten]
                    if (_3586 > 0.0f)
                    {
                        ls.pdf = (ls.dist * ls.dist) / (ls.area * _3586);
                    }
                    if ((_3043.type_and_param0.x & 64u) == 0u)
                    {
                        ls.area = 0.0f;
                    }
                    [branch]
                    if ((_3043.type_and_param0.w & 128u) != 0u)
                    {
                        float3 env_col_1 = _3000_g_params.env_col.xyz;
                        uint _3622 = asuint(_3000_g_params.env_col.w);
                        if (_3622 != 4294967295u)
                        {
                            env_col_1 *= SampleLatlong_RGBE(_3622, ls.L, _3000_g_params.env_rotation);
                        }
                        ls.col *= env_col_1;
                    }
                }
                else
                {
                    [branch]
                    if (_3078 == 3u)
                    {
                        float3 _3681 = normalize(cross(P - _3043.param1.xyz, _3043.param3.xyz));
                        float _3688 = 3.1415927410125732421875f * frac(asfloat(_2995.Load((_3000_g_params.hi + 4) * 4 + 0)) + sample_off.x);
                        float3 _3717 = ((_3043.param1.xyz + (((_3681 * cos(_3688)) + (cross(_3681, _3043.param3.xyz) * sin(_3688))) * _3043.param2.w)) + ((_3043.param3.xyz * (frac(asfloat(_2995.Load((_3000_g_params.hi + 5) * 4 + 0)) + sample_off.y) - 0.5f)) * _3043.param3.w)) - P;
                        ls.dist = length(_3717);
                        ls.L = _3717 / ls.dist.xxx;
                        ls.area = _3043.param1.w;
                        float _3736 = 1.0f - abs(dot(ls.L, _3043.param3.xyz));
                        [flatten]
                        if (_3736 != 0.0f)
                        {
                            ls.pdf = (ls.dist * ls.dist) / (ls.area * _3736);
                        }
                        if ((_3043.type_and_param0.x & 64u) == 0u)
                        {
                            ls.area = 0.0f;
                        }
                        [branch]
                        if ((_3043.type_and_param0.w & 128u) != 0u)
                        {
                            float3 env_col_2 = _3000_g_params.env_col.xyz;
                            uint _3772 = asuint(_3000_g_params.env_col.w);
                            if (_3772 != 4294967295u)
                            {
                                env_col_2 *= SampleLatlong_RGBE(_3772, ls.L, _3000_g_params.env_rotation);
                            }
                            ls.col *= env_col_2;
                        }
                    }
                    else
                    {
                        [branch]
                        if (_3078 == 6u)
                        {
                            uint _3799 = asuint(_3043.param1.x);
                            transform_t _3813;
                            _3813.xform = asfloat(uint4x4(_3807.Load4(asuint(_3043.param1.y) * 128 + 0), _3807.Load4(asuint(_3043.param1.y) * 128 + 16), _3807.Load4(asuint(_3043.param1.y) * 128 + 32), _3807.Load4(asuint(_3043.param1.y) * 128 + 48)));
                            _3813.inv_xform = asfloat(uint4x4(_3807.Load4(asuint(_3043.param1.y) * 128 + 64), _3807.Load4(asuint(_3043.param1.y) * 128 + 80), _3807.Load4(asuint(_3043.param1.y) * 128 + 96), _3807.Load4(asuint(_3043.param1.y) * 128 + 112)));
                            uint _3838 = _3799 * 3u;
                            vertex_t _3844;
                            [unroll]
                            for (int _43ident = 0; _43ident < 3; _43ident++)
                            {
                                _3844.p[_43ident] = asfloat(_3832.Load(_43ident * 4 + _3836.Load(_3838 * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _44ident = 0; _44ident < 3; _44ident++)
                            {
                                _3844.n[_44ident] = asfloat(_3832.Load(_44ident * 4 + _3836.Load(_3838 * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _45ident = 0; _45ident < 3; _45ident++)
                            {
                                _3844.b[_45ident] = asfloat(_3832.Load(_45ident * 4 + _3836.Load(_3838 * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _46ident = 0; _46ident < 2; _46ident++)
                            {
                                [unroll]
                                for (int _47ident = 0; _47ident < 2; _47ident++)
                                {
                                    _3844.t[_46ident][_47ident] = asfloat(_3832.Load(_47ident * 4 + _46ident * 8 + _3836.Load(_3838 * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _3893;
                            [unroll]
                            for (int _48ident = 0; _48ident < 3; _48ident++)
                            {
                                _3893.p[_48ident] = asfloat(_3832.Load(_48ident * 4 + _3836.Load((_3838 + 1u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _49ident = 0; _49ident < 3; _49ident++)
                            {
                                _3893.n[_49ident] = asfloat(_3832.Load(_49ident * 4 + _3836.Load((_3838 + 1u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _50ident = 0; _50ident < 3; _50ident++)
                            {
                                _3893.b[_50ident] = asfloat(_3832.Load(_50ident * 4 + _3836.Load((_3838 + 1u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _51ident = 0; _51ident < 2; _51ident++)
                            {
                                [unroll]
                                for (int _52ident = 0; _52ident < 2; _52ident++)
                                {
                                    _3893.t[_51ident][_52ident] = asfloat(_3832.Load(_52ident * 4 + _51ident * 8 + _3836.Load((_3838 + 1u) * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _3939;
                            [unroll]
                            for (int _53ident = 0; _53ident < 3; _53ident++)
                            {
                                _3939.p[_53ident] = asfloat(_3832.Load(_53ident * 4 + _3836.Load((_3838 + 2u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _54ident = 0; _54ident < 3; _54ident++)
                            {
                                _3939.n[_54ident] = asfloat(_3832.Load(_54ident * 4 + _3836.Load((_3838 + 2u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _55ident = 0; _55ident < 3; _55ident++)
                            {
                                _3939.b[_55ident] = asfloat(_3832.Load(_55ident * 4 + _3836.Load((_3838 + 2u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _56ident = 0; _56ident < 2; _56ident++)
                            {
                                [unroll]
                                for (int _57ident = 0; _57ident < 2; _57ident++)
                                {
                                    _3939.t[_56ident][_57ident] = asfloat(_3832.Load(_57ident * 4 + _56ident * 8 + _3836.Load((_3838 + 2u) * 4 + 0) * 52 + 36));
                                }
                            }
                            float3 _3985 = float3(_3844.p[0], _3844.p[1], _3844.p[2]);
                            float3 _3993 = float3(_3893.p[0], _3893.p[1], _3893.p[2]);
                            float3 _4001 = float3(_3939.p[0], _3939.p[1], _3939.p[2]);
                            float _4030 = sqrt(frac(asfloat(_2995.Load((_3000_g_params.hi + 4) * 4 + 0)) + sample_off.x));
                            float _4040 = frac(asfloat(_2995.Load((_3000_g_params.hi + 5) * 4 + 0)) + sample_off.y);
                            float _4044 = 1.0f - _4030;
                            float _4049 = 1.0f - _4040;
                            float3 _4096 = mul(float4(cross(_3993 - _3985, _4001 - _3985), 0.0f), _3813.xform).xyz;
                            ls.area = 0.5f * length(_4096);
                            float3 _4106 = mul(float4((_3985 * _4044) + (((_3993 * _4049) + (_4001 * _4040)) * _4030), 1.0f), _3813.xform).xyz - P;
                            ls.dist = length(_4106);
                            ls.L = _4106 / ls.dist.xxx;
                            float _4121 = abs(dot(ls.L, normalize(_4096)));
                            [flatten]
                            if (_4121 > 0.0f)
                            {
                                ls.pdf = (ls.dist * ls.dist) / (ls.area * _4121);
                            }
                            material_t _4161;
                            [unroll]
                            for (int _58ident = 0; _58ident < 5; _58ident++)
                            {
                                _4161.textures[_58ident] = _4147.Load(_58ident * 4 + ((_4151.Load(_3799 * 4 + 0) >> 16u) & 16383u) * 80 + 0);
                            }
                            [unroll]
                            for (int _59ident = 0; _59ident < 3; _59ident++)
                            {
                                _4161.base_color[_59ident] = asfloat(_4147.Load(_59ident * 4 + ((_4151.Load(_3799 * 4 + 0) >> 16u) & 16383u) * 80 + 20));
                            }
                            _4161.flags = _4147.Load(((_4151.Load(_3799 * 4 + 0) >> 16u) & 16383u) * 80 + 32);
                            _4161.type = _4147.Load(((_4151.Load(_3799 * 4 + 0) >> 16u) & 16383u) * 80 + 36);
                            _4161.tangent_rotation_or_strength = asfloat(_4147.Load(((_4151.Load(_3799 * 4 + 0) >> 16u) & 16383u) * 80 + 40));
                            _4161.roughness_and_anisotropic = _4147.Load(((_4151.Load(_3799 * 4 + 0) >> 16u) & 16383u) * 80 + 44);
                            _4161.int_ior = asfloat(_4147.Load(((_4151.Load(_3799 * 4 + 0) >> 16u) & 16383u) * 80 + 48));
                            _4161.ext_ior = asfloat(_4147.Load(((_4151.Load(_3799 * 4 + 0) >> 16u) & 16383u) * 80 + 52));
                            _4161.sheen_and_sheen_tint = _4147.Load(((_4151.Load(_3799 * 4 + 0) >> 16u) & 16383u) * 80 + 56);
                            _4161.tint_and_metallic = _4147.Load(((_4151.Load(_3799 * 4 + 0) >> 16u) & 16383u) * 80 + 60);
                            _4161.transmission_and_transmission_roughness = _4147.Load(((_4151.Load(_3799 * 4 + 0) >> 16u) & 16383u) * 80 + 64);
                            _4161.specular_and_specular_tint = _4147.Load(((_4151.Load(_3799 * 4 + 0) >> 16u) & 16383u) * 80 + 68);
                            _4161.clearcoat_and_clearcoat_roughness = _4147.Load(((_4151.Load(_3799 * 4 + 0) >> 16u) & 16383u) * 80 + 72);
                            _4161.normal_map_strength_unorm = _4147.Load(((_4151.Load(_3799 * 4 + 0) >> 16u) & 16383u) * 80 + 76);
                            if ((_4161.flags & 4u) == 0u)
                            {
                                if (_4161.textures[1] != 4294967295u)
                                {
                                    ls.col *= SampleBilinear(_4161.textures[1], (float2(_3844.t[0][0], _3844.t[0][1]) * _4044) + (((float2(_3893.t[0][0], _3893.t[0][1]) * _4049) + (float2(_3939.t[0][0], _3939.t[0][1]) * _4040)) * _4030), 0).xyz;
                                }
                            }
                            else
                            {
                                float3 env_col_3 = _3000_g_params.env_col.xyz;
                                uint _4241 = asuint(_3000_g_params.env_col.w);
                                if (_4241 != 4294967295u)
                                {
                                    env_col_3 *= SampleLatlong_RGBE(_4241, ls.L, _3000_g_params.env_rotation);
                                }
                                ls.col *= env_col_3;
                            }
                        }
                        else
                        {
                            [branch]
                            if (_3078 == 7u)
                            {
                                float4 _4303 = Sample_EnvQTree(_3000_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3000_g_params.env_qtree_levels, mad(_3011, _3016, -float(_3023)), frac(asfloat(_2995.Load((_3000_g_params.hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_2995.Load((_3000_g_params.hi + 5) * 4 + 0)) + sample_off.y));
                                ls.L = _4303.xyz;
                                ls.col *= _3000_g_params.env_col.xyz;
                                ls.col *= SampleLatlong_RGBE(asuint(_3000_g_params.env_col.w), ls.L, _3000_g_params.env_rotation);
                                ls.area = 1.0f;
                                ls.dist = 3402823346297367662189621542912.0f;
                                ls.pdf = _4303.w;
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
    float _8382;
    do
    {
        if (H.z == 0.0f)
        {
            _8382 = 0.0f;
            break;
        }
        float _1877 = (-H.x) / (H.z * alpha_x);
        float _1883 = (-H.y) / (H.z * alpha_y);
        float _1892 = mad(_1883, _1883, mad(_1877, _1877, 1.0f));
        _8382 = 1.0f / (((((_1892 * _1892) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _8382;
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
    float4 _8357;
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
            _8357 = float4(_2371.x * 1000000.0f, _2371.y * 1000000.0f, _2371.z * 1000000.0f, 1000000.0f);
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
        _8357 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _8357;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _8362;
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
            _8362 = 0.0f.xxxx;
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
        _8362 = float4(refr_col * (((((_2654 * _2670) * _2662) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _2690) / view_dir_ts.z), (((_2654 * _2662) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _2690) / view_dir_ts.z);
        break;
    } while(false);
    return _8362;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _8367;
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
                _8367 = 0.0f.xxxx;
                break;
            }
            float _2767 = mad(eta, _2745, -sqrt(_2755));
            out_V = float4(normalize((I * eta) + (N * _2767)), _2767);
            _8367 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
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
            _8367 = 0.0f.xxxx;
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
        _8367 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _8367;
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
    float _8387;
    do
    {
        float _1943 = dot(N, L);
        if (_1943 <= 0.0f)
        {
            _8387 = 0.0f;
            break;
        }
        float param = _1943;
        float param_1 = dot(N, V);
        float _1964 = dot(L, H);
        float _1972 = mad((2.0f * _1964) * _1964, roughness, 0.5f);
        _8387 = lerp(1.0f, _1972, schlick_weight(param)) * lerp(1.0f, _1972, schlick_weight(param_1));
        break;
    } while(false);
    return _8387;
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
    float _8392;
    do
    {
        if (a >= 1.0f)
        {
            _8392 = 0.3183098733425140380859375f;
            break;
        }
        float _1851 = mad(a, a, -1.0f);
        _8392 = _1851 / ((3.1415927410125732421875f * log(a * a)) * mad(_1851 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _8392;
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
    float4 _8372;
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
            _8372 = float4(_2571, _2571, _2571, 1000000.0f);
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
        _8372 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _8372;
}

float3 ShadeSurface(int px_index, hit_data_t inter, ray_data_t ray)
{
    float3 _8347;
    do
    {
        float3 _4348 = float3(ray.d[0], ray.d[1], ray.d[2]);
        [branch]
        if (inter.mask == 0)
        {
            float3 env_col = _3000_g_params.env_col.xyz;
            uint _4361 = asuint(_3000_g_params.env_col.w);
            if (_4361 != 4294967295u)
            {
                env_col *= SampleLatlong_RGBE(_4361, _4348, _3000_g_params.env_rotation);
                if (_3000_g_params.env_qtree_levels > 0)
                {
                    float param = ray.pdf;
                    float param_1 = Evaluate_EnvQTree(_3000_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3000_g_params.env_qtree_levels, _4348);
                    env_col *= power_heuristic(param, param_1);
                }
            }
            _8347 = float3(ray.c[0] * env_col.x, ray.c[1] * env_col.y, ray.c[2] * env_col.z);
            break;
        }
        float3 _4422 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_4348 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            light_t _4434;
            _4434.type_and_param0 = _3032.Load4(((-1) - inter.obj_index) * 64 + 0);
            _4434.param1 = asfloat(_3032.Load4(((-1) - inter.obj_index) * 64 + 16));
            _4434.param2 = asfloat(_3032.Load4(((-1) - inter.obj_index) * 64 + 32));
            _4434.param3 = asfloat(_3032.Load4(((-1) - inter.obj_index) * 64 + 48));
            float3 lcol = asfloat(_4434.type_and_param0.yzw);
            uint _4451 = _4434.type_and_param0.x & 31u;
            if (_4451 == 0u)
            {
                float param_2 = ray.pdf;
                float param_3 = (inter.t * inter.t) / ((0.5f * _4434.param1.w) * dot(_4348, normalize(_4434.param1.xyz - _4422)));
                lcol *= power_heuristic(param_2, param_3);
                bool _4518 = _4434.param3.x > 0.0f;
                bool _4524;
                if (_4518)
                {
                    _4524 = _4434.param3.y > 0.0f;
                }
                else
                {
                    _4524 = _4518;
                }
                [branch]
                if (_4524)
                {
                    [flatten]
                    if (_4434.param3.y > 0.0f)
                    {
                        lcol *= clamp((_4434.param3.x - acos(clamp(-dot(_4348, _4434.param2.xyz), 0.0f, 1.0f))) / _4434.param3.y, 0.0f, 1.0f);
                    }
                }
            }
            else
            {
                if (_4451 == 4u)
                {
                    float param_4 = ray.pdf;
                    float param_5 = (inter.t * inter.t) / (_4434.param1.w * dot(_4348, normalize(cross(_4434.param2.xyz, _4434.param3.xyz))));
                    lcol *= power_heuristic(param_4, param_5);
                }
                else
                {
                    if (_4451 == 5u)
                    {
                        float param_6 = ray.pdf;
                        float param_7 = (inter.t * inter.t) / (_4434.param1.w * dot(_4348, normalize(cross(_4434.param2.xyz, _4434.param3.xyz))));
                        lcol *= power_heuristic(param_6, param_7);
                    }
                    else
                    {
                        if (_4451 == 3u)
                        {
                            float param_8 = ray.pdf;
                            float param_9 = (inter.t * inter.t) / (_4434.param1.w * (1.0f - abs(dot(_4348, _4434.param3.xyz))));
                            lcol *= power_heuristic(param_8, param_9);
                        }
                    }
                }
            }
            _8347 = float3(ray.c[0] * lcol.x, ray.c[1] * lcol.y, ray.c[2] * lcol.z);
            break;
        }
        bool _4723 = inter.prim_index < 0;
        int _4726;
        if (_4723)
        {
            _4726 = (-1) - inter.prim_index;
        }
        else
        {
            _4726 = inter.prim_index;
        }
        uint _4737 = uint(_4726);
        material_t _4745;
        [unroll]
        for (int _60ident = 0; _60ident < 5; _60ident++)
        {
            _4745.textures[_60ident] = _4147.Load(_60ident * 4 + ((_4151.Load(_4737 * 4 + 0) >> 16u) & 16383u) * 80 + 0);
        }
        [unroll]
        for (int _61ident = 0; _61ident < 3; _61ident++)
        {
            _4745.base_color[_61ident] = asfloat(_4147.Load(_61ident * 4 + ((_4151.Load(_4737 * 4 + 0) >> 16u) & 16383u) * 80 + 20));
        }
        _4745.flags = _4147.Load(((_4151.Load(_4737 * 4 + 0) >> 16u) & 16383u) * 80 + 32);
        _4745.type = _4147.Load(((_4151.Load(_4737 * 4 + 0) >> 16u) & 16383u) * 80 + 36);
        _4745.tangent_rotation_or_strength = asfloat(_4147.Load(((_4151.Load(_4737 * 4 + 0) >> 16u) & 16383u) * 80 + 40));
        _4745.roughness_and_anisotropic = _4147.Load(((_4151.Load(_4737 * 4 + 0) >> 16u) & 16383u) * 80 + 44);
        _4745.int_ior = asfloat(_4147.Load(((_4151.Load(_4737 * 4 + 0) >> 16u) & 16383u) * 80 + 48));
        _4745.ext_ior = asfloat(_4147.Load(((_4151.Load(_4737 * 4 + 0) >> 16u) & 16383u) * 80 + 52));
        _4745.sheen_and_sheen_tint = _4147.Load(((_4151.Load(_4737 * 4 + 0) >> 16u) & 16383u) * 80 + 56);
        _4745.tint_and_metallic = _4147.Load(((_4151.Load(_4737 * 4 + 0) >> 16u) & 16383u) * 80 + 60);
        _4745.transmission_and_transmission_roughness = _4147.Load(((_4151.Load(_4737 * 4 + 0) >> 16u) & 16383u) * 80 + 64);
        _4745.specular_and_specular_tint = _4147.Load(((_4151.Load(_4737 * 4 + 0) >> 16u) & 16383u) * 80 + 68);
        _4745.clearcoat_and_clearcoat_roughness = _4147.Load(((_4151.Load(_4737 * 4 + 0) >> 16u) & 16383u) * 80 + 72);
        _4745.normal_map_strength_unorm = _4147.Load(((_4151.Load(_4737 * 4 + 0) >> 16u) & 16383u) * 80 + 76);
        uint _8907 = _4745.textures[0];
        uint _8908 = _4745.textures[1];
        uint _8909 = _4745.textures[2];
        uint _8910 = _4745.textures[3];
        uint _8911 = _4745.textures[4];
        float _8912 = _4745.base_color[0];
        float _8913 = _4745.base_color[1];
        float _8914 = _4745.base_color[2];
        uint _8594 = _4745.flags;
        uint _8595 = _4745.type;
        float _8596 = _4745.tangent_rotation_or_strength;
        uint _8597 = _4745.roughness_and_anisotropic;
        float _8598 = _4745.int_ior;
        float _8599 = _4745.ext_ior;
        uint _8600 = _4745.sheen_and_sheen_tint;
        uint _8601 = _4745.tint_and_metallic;
        uint _8602 = _4745.transmission_and_transmission_roughness;
        uint _8603 = _4745.specular_and_specular_tint;
        uint _8604 = _4745.clearcoat_and_clearcoat_roughness;
        uint _8605 = _4745.normal_map_strength_unorm;
        transform_t _4802;
        _4802.xform = asfloat(uint4x4(_3807.Load4(asuint(asfloat(_4795.Load(inter.obj_index * 32 + 12))) * 128 + 0), _3807.Load4(asuint(asfloat(_4795.Load(inter.obj_index * 32 + 12))) * 128 + 16), _3807.Load4(asuint(asfloat(_4795.Load(inter.obj_index * 32 + 12))) * 128 + 32), _3807.Load4(asuint(asfloat(_4795.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _4802.inv_xform = asfloat(uint4x4(_3807.Load4(asuint(asfloat(_4795.Load(inter.obj_index * 32 + 12))) * 128 + 64), _3807.Load4(asuint(asfloat(_4795.Load(inter.obj_index * 32 + 12))) * 128 + 80), _3807.Load4(asuint(asfloat(_4795.Load(inter.obj_index * 32 + 12))) * 128 + 96), _3807.Load4(asuint(asfloat(_4795.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _4809 = _4737 * 3u;
        vertex_t _4814;
        [unroll]
        for (int _62ident = 0; _62ident < 3; _62ident++)
        {
            _4814.p[_62ident] = asfloat(_3832.Load(_62ident * 4 + _3836.Load(_4809 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _63ident = 0; _63ident < 3; _63ident++)
        {
            _4814.n[_63ident] = asfloat(_3832.Load(_63ident * 4 + _3836.Load(_4809 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _64ident = 0; _64ident < 3; _64ident++)
        {
            _4814.b[_64ident] = asfloat(_3832.Load(_64ident * 4 + _3836.Load(_4809 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _65ident = 0; _65ident < 2; _65ident++)
        {
            [unroll]
            for (int _66ident = 0; _66ident < 2; _66ident++)
            {
                _4814.t[_65ident][_66ident] = asfloat(_3832.Load(_66ident * 4 + _65ident * 8 + _3836.Load(_4809 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _4860;
        [unroll]
        for (int _67ident = 0; _67ident < 3; _67ident++)
        {
            _4860.p[_67ident] = asfloat(_3832.Load(_67ident * 4 + _3836.Load((_4809 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _68ident = 0; _68ident < 3; _68ident++)
        {
            _4860.n[_68ident] = asfloat(_3832.Load(_68ident * 4 + _3836.Load((_4809 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _69ident = 0; _69ident < 3; _69ident++)
        {
            _4860.b[_69ident] = asfloat(_3832.Load(_69ident * 4 + _3836.Load((_4809 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _70ident = 0; _70ident < 2; _70ident++)
        {
            [unroll]
            for (int _71ident = 0; _71ident < 2; _71ident++)
            {
                _4860.t[_70ident][_71ident] = asfloat(_3832.Load(_71ident * 4 + _70ident * 8 + _3836.Load((_4809 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _4906;
        [unroll]
        for (int _72ident = 0; _72ident < 3; _72ident++)
        {
            _4906.p[_72ident] = asfloat(_3832.Load(_72ident * 4 + _3836.Load((_4809 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _73ident = 0; _73ident < 3; _73ident++)
        {
            _4906.n[_73ident] = asfloat(_3832.Load(_73ident * 4 + _3836.Load((_4809 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _74ident = 0; _74ident < 3; _74ident++)
        {
            _4906.b[_74ident] = asfloat(_3832.Load(_74ident * 4 + _3836.Load((_4809 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _75ident = 0; _75ident < 2; _75ident++)
        {
            [unroll]
            for (int _76ident = 0; _76ident < 2; _76ident++)
            {
                _4906.t[_75ident][_76ident] = asfloat(_3832.Load(_76ident * 4 + _75ident * 8 + _3836.Load((_4809 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _4952 = float3(_4814.p[0], _4814.p[1], _4814.p[2]);
        float3 _4960 = float3(_4860.p[0], _4860.p[1], _4860.p[2]);
        float3 _4968 = float3(_4906.p[0], _4906.p[1], _4906.p[2]);
        float _4975 = (1.0f - inter.u) - inter.v;
        float3 _5008 = normalize(((float3(_4814.n[0], _4814.n[1], _4814.n[2]) * _4975) + (float3(_4860.n[0], _4860.n[1], _4860.n[2]) * inter.u)) + (float3(_4906.n[0], _4906.n[1], _4906.n[2]) * inter.v));
        float3 N = _5008;
        float2 _5034 = ((float2(_4814.t[0][0], _4814.t[0][1]) * _4975) + (float2(_4860.t[0][0], _4860.t[0][1]) * inter.u)) + (float2(_4906.t[0][0], _4906.t[0][1]) * inter.v);
        float3 _5050 = cross(_4960 - _4952, _4968 - _4952);
        float _5053 = length(_5050);
        float3 plane_N = _5050 / _5053.xxx;
        float3 _5089 = ((float3(_4814.b[0], _4814.b[1], _4814.b[2]) * _4975) + (float3(_4860.b[0], _4860.b[1], _4860.b[2]) * inter.u)) + (float3(_4906.b[0], _4906.b[1], _4906.b[2]) * inter.v);
        float3 B = _5089;
        float3 T = cross(_5089, _5008);
        if (_4723)
        {
            if ((_4151.Load(_4737 * 4 + 0) & 65535u) == 65535u)
            {
                _8347 = 0.0f.xxx;
                break;
            }
            material_t _5113;
            [unroll]
            for (int _77ident = 0; _77ident < 5; _77ident++)
            {
                _5113.textures[_77ident] = _4147.Load(_77ident * 4 + (_4151.Load(_4737 * 4 + 0) & 16383u) * 80 + 0);
            }
            [unroll]
            for (int _78ident = 0; _78ident < 3; _78ident++)
            {
                _5113.base_color[_78ident] = asfloat(_4147.Load(_78ident * 4 + (_4151.Load(_4737 * 4 + 0) & 16383u) * 80 + 20));
            }
            _5113.flags = _4147.Load((_4151.Load(_4737 * 4 + 0) & 16383u) * 80 + 32);
            _5113.type = _4147.Load((_4151.Load(_4737 * 4 + 0) & 16383u) * 80 + 36);
            _5113.tangent_rotation_or_strength = asfloat(_4147.Load((_4151.Load(_4737 * 4 + 0) & 16383u) * 80 + 40));
            _5113.roughness_and_anisotropic = _4147.Load((_4151.Load(_4737 * 4 + 0) & 16383u) * 80 + 44);
            _5113.int_ior = asfloat(_4147.Load((_4151.Load(_4737 * 4 + 0) & 16383u) * 80 + 48));
            _5113.ext_ior = asfloat(_4147.Load((_4151.Load(_4737 * 4 + 0) & 16383u) * 80 + 52));
            _5113.sheen_and_sheen_tint = _4147.Load((_4151.Load(_4737 * 4 + 0) & 16383u) * 80 + 56);
            _5113.tint_and_metallic = _4147.Load((_4151.Load(_4737 * 4 + 0) & 16383u) * 80 + 60);
            _5113.transmission_and_transmission_roughness = _4147.Load((_4151.Load(_4737 * 4 + 0) & 16383u) * 80 + 64);
            _5113.specular_and_specular_tint = _4147.Load((_4151.Load(_4737 * 4 + 0) & 16383u) * 80 + 68);
            _5113.clearcoat_and_clearcoat_roughness = _4147.Load((_4151.Load(_4737 * 4 + 0) & 16383u) * 80 + 72);
            _5113.normal_map_strength_unorm = _4147.Load((_4151.Load(_4737 * 4 + 0) & 16383u) * 80 + 76);
            _8907 = _5113.textures[0];
            _8908 = _5113.textures[1];
            _8909 = _5113.textures[2];
            _8910 = _5113.textures[3];
            _8911 = _5113.textures[4];
            _8912 = _5113.base_color[0];
            _8913 = _5113.base_color[1];
            _8914 = _5113.base_color[2];
            _8594 = _5113.flags;
            _8595 = _5113.type;
            _8596 = _5113.tangent_rotation_or_strength;
            _8597 = _5113.roughness_and_anisotropic;
            _8598 = _5113.int_ior;
            _8599 = _5113.ext_ior;
            _8600 = _5113.sheen_and_sheen_tint;
            _8601 = _5113.tint_and_metallic;
            _8602 = _5113.transmission_and_transmission_roughness;
            _8603 = _5113.specular_and_specular_tint;
            _8604 = _5113.clearcoat_and_clearcoat_roughness;
            _8605 = _5113.normal_map_strength_unorm;
            plane_N = -plane_N;
            N = -N;
            B = -B;
            T = -T;
        }
        float3 param_10 = plane_N;
        float4x4 param_11 = _4802.inv_xform;
        plane_N = TransformNormal(param_10, param_11);
        float3 param_12 = N;
        float4x4 param_13 = _4802.inv_xform;
        N = TransformNormal(param_12, param_13);
        float3 param_14 = B;
        float4x4 param_15 = _4802.inv_xform;
        B = TransformNormal(param_14, param_15);
        float3 param_16 = T;
        float4x4 param_17 = _4802.inv_xform;
        T = TransformNormal(param_16, param_17);
        float _5223 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _5233 = mad(0.5f, log2(abs(mad(_4860.t[0][0] - _4814.t[0][0], _4906.t[0][1] - _4814.t[0][1], -((_4906.t[0][0] - _4814.t[0][0]) * (_4860.t[0][1] - _4814.t[0][1])))) / _5053), log2(_5223));
        uint param_18 = uint(hash(px_index));
        float _5239 = construct_float(param_18);
        uint param_19 = uint(hash(hash(px_index)));
        float _5245 = construct_float(param_19);
        float3 col = 0.0f.xxx;
        int _5252 = ray.ray_depth & 255;
        int _5257 = (ray.ray_depth >> 8) & 255;
        int _5262 = (ray.ray_depth >> 16) & 255;
        int _5268 = (ray.ray_depth >> 24) & 255;
        int _5276 = ((_5252 + _5257) + _5262) + _5268;
        float mix_rand = frac(asfloat(_2995.Load(_3000_g_params.hi * 4 + 0)) + _5239);
        float mix_weight = 1.0f;
        float _5313;
        float _5332;
        float _5357;
        float _5426;
        while (_8595 == 4u)
        {
            float mix_val = _8596;
            if (_8908 != 4294967295u)
            {
                mix_val *= SampleBilinear(_8908, _5034, 0).x;
            }
            if (_4723)
            {
                _5313 = _8599 / _8598;
            }
            else
            {
                _5313 = _8598 / _8599;
            }
            if (_8598 != 0.0f)
            {
                float param_20 = dot(_4348, N);
                float param_21 = _5313;
                _5332 = fresnel_dielectric_cos(param_20, param_21);
            }
            else
            {
                _5332 = 1.0f;
            }
            float _5346 = mix_val;
            float _5347 = _5346 * clamp(_5332, 0.0f, 1.0f);
            mix_val = _5347;
            if (mix_rand > _5347)
            {
                if ((_8594 & 2u) != 0u)
                {
                    _5357 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _5357 = 1.0f;
                }
                mix_weight *= _5357;
                material_t _5370;
                [unroll]
                for (int _79ident = 0; _79ident < 5; _79ident++)
                {
                    _5370.textures[_79ident] = _4147.Load(_79ident * 4 + _8910 * 80 + 0);
                }
                [unroll]
                for (int _80ident = 0; _80ident < 3; _80ident++)
                {
                    _5370.base_color[_80ident] = asfloat(_4147.Load(_80ident * 4 + _8910 * 80 + 20));
                }
                _5370.flags = _4147.Load(_8910 * 80 + 32);
                _5370.type = _4147.Load(_8910 * 80 + 36);
                _5370.tangent_rotation_or_strength = asfloat(_4147.Load(_8910 * 80 + 40));
                _5370.roughness_and_anisotropic = _4147.Load(_8910 * 80 + 44);
                _5370.int_ior = asfloat(_4147.Load(_8910 * 80 + 48));
                _5370.ext_ior = asfloat(_4147.Load(_8910 * 80 + 52));
                _5370.sheen_and_sheen_tint = _4147.Load(_8910 * 80 + 56);
                _5370.tint_and_metallic = _4147.Load(_8910 * 80 + 60);
                _5370.transmission_and_transmission_roughness = _4147.Load(_8910 * 80 + 64);
                _5370.specular_and_specular_tint = _4147.Load(_8910 * 80 + 68);
                _5370.clearcoat_and_clearcoat_roughness = _4147.Load(_8910 * 80 + 72);
                _5370.normal_map_strength_unorm = _4147.Load(_8910 * 80 + 76);
                _8907 = _5370.textures[0];
                _8908 = _5370.textures[1];
                _8909 = _5370.textures[2];
                _8910 = _5370.textures[3];
                _8911 = _5370.textures[4];
                _8912 = _5370.base_color[0];
                _8913 = _5370.base_color[1];
                _8914 = _5370.base_color[2];
                _8594 = _5370.flags;
                _8595 = _5370.type;
                _8596 = _5370.tangent_rotation_or_strength;
                _8597 = _5370.roughness_and_anisotropic;
                _8598 = _5370.int_ior;
                _8599 = _5370.ext_ior;
                _8600 = _5370.sheen_and_sheen_tint;
                _8601 = _5370.tint_and_metallic;
                _8602 = _5370.transmission_and_transmission_roughness;
                _8603 = _5370.specular_and_specular_tint;
                _8604 = _5370.clearcoat_and_clearcoat_roughness;
                _8605 = _5370.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_8594 & 2u) != 0u)
                {
                    _5426 = 1.0f / mix_val;
                }
                else
                {
                    _5426 = 1.0f;
                }
                mix_weight *= _5426;
                material_t _5438;
                [unroll]
                for (int _81ident = 0; _81ident < 5; _81ident++)
                {
                    _5438.textures[_81ident] = _4147.Load(_81ident * 4 + _8911 * 80 + 0);
                }
                [unroll]
                for (int _82ident = 0; _82ident < 3; _82ident++)
                {
                    _5438.base_color[_82ident] = asfloat(_4147.Load(_82ident * 4 + _8911 * 80 + 20));
                }
                _5438.flags = _4147.Load(_8911 * 80 + 32);
                _5438.type = _4147.Load(_8911 * 80 + 36);
                _5438.tangent_rotation_or_strength = asfloat(_4147.Load(_8911 * 80 + 40));
                _5438.roughness_and_anisotropic = _4147.Load(_8911 * 80 + 44);
                _5438.int_ior = asfloat(_4147.Load(_8911 * 80 + 48));
                _5438.ext_ior = asfloat(_4147.Load(_8911 * 80 + 52));
                _5438.sheen_and_sheen_tint = _4147.Load(_8911 * 80 + 56);
                _5438.tint_and_metallic = _4147.Load(_8911 * 80 + 60);
                _5438.transmission_and_transmission_roughness = _4147.Load(_8911 * 80 + 64);
                _5438.specular_and_specular_tint = _4147.Load(_8911 * 80 + 68);
                _5438.clearcoat_and_clearcoat_roughness = _4147.Load(_8911 * 80 + 72);
                _5438.normal_map_strength_unorm = _4147.Load(_8911 * 80 + 76);
                _8907 = _5438.textures[0];
                _8908 = _5438.textures[1];
                _8909 = _5438.textures[2];
                _8910 = _5438.textures[3];
                _8911 = _5438.textures[4];
                _8912 = _5438.base_color[0];
                _8913 = _5438.base_color[1];
                _8914 = _5438.base_color[2];
                _8594 = _5438.flags;
                _8595 = _5438.type;
                _8596 = _5438.tangent_rotation_or_strength;
                _8597 = _5438.roughness_and_anisotropic;
                _8598 = _5438.int_ior;
                _8599 = _5438.ext_ior;
                _8600 = _5438.sheen_and_sheen_tint;
                _8601 = _5438.tint_and_metallic;
                _8602 = _5438.transmission_and_transmission_roughness;
                _8603 = _5438.specular_and_specular_tint;
                _8604 = _5438.clearcoat_and_clearcoat_roughness;
                _8605 = _5438.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_8907 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_8907, _5034, 0).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_8907 & 33554432u) != 0u)
            {
                float3 _9228 = normals;
                _9228.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _9228;
            }
            float3 _5520 = N;
            N = normalize(((T * normals.x) + (_5520 * normals.z)) + (B * normals.y));
            if ((_8605 & 65535u) != 65535u)
            {
                N = normalize(_5520 + ((N - _5520) * clamp(float(_8605 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_22 = plane_N;
            float3 param_23 = -_4348;
            float3 param_24 = N;
            N = ensure_valid_reflection(param_22, param_23, param_24);
        }
        float3 _5577 = ((_4952 * _4975) + (_4960 * inter.u)) + (_4968 * inter.v);
        float3 _5584 = float3(-_5577.z, 0.0f, _5577.x);
        float3 tangent = _5584;
        float3 param_25 = _5584;
        float4x4 param_26 = _4802.inv_xform;
        tangent = TransformNormal(param_25, param_26);
        if (_8596 != 0.0f)
        {
            float3 param_27 = tangent;
            float3 param_28 = N;
            float param_29 = _8596;
            tangent = rotate_around_axis(param_27, param_28, param_29);
        }
        float3 _5607 = normalize(cross(tangent, N));
        B = _5607;
        T = cross(N, _5607);
        float3 _8693 = 0.0f.xxx;
        float3 _8692 = 0.0f.xxx;
        float _8695 = 0.0f;
        float _8696 = 0.0f;
        float _8694 = 0.0f;
        bool _5619 = _3000_g_params.li_count != 0;
        bool _5625;
        if (_5619)
        {
            _5625 = _8595 != 3u;
        }
        else
        {
            _5625 = _5619;
        }
        float _8697;
        if (_5625)
        {
            float3 param_30 = _4422;
            float2 param_31 = float2(_5239, _5245);
            light_sample_t _8704 = { _8692, _8693, _8694, _8695, _8696, _8697 };
            light_sample_t param_32 = _8704;
            SampleLightSource(param_30, param_31, param_32);
            _8692 = param_32.col;
            _8693 = param_32.L;
            _8694 = param_32.area;
            _8695 = param_32.dist;
            _8696 = param_32.pdf;
            _8697 = param_32.cast_shadow;
        }
        float _5640 = dot(N, _8693);
        float3 base_color = float3(_8912, _8913, _8914);
        [branch]
        if (_8908 != 4294967295u)
        {
            base_color *= SampleBilinear(_8908, _5034, int(get_texture_lod(texSize(_8908), _5233)), true, true).xyz;
        }
        float3 tint_color = 0.0f.xxx;
        float _5684 = lum(base_color);
        [flatten]
        if (_5684 > 0.0f)
        {
            tint_color = base_color / _5684.xxx;
        }
        float roughness = clamp(float(_8597 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_8909 != 4294967295u)
        {
            roughness *= SampleBilinear(_8909, _5034, int(get_texture_lod(texSize(_8909), _5233)), false, true).x;
        }
        float _5725 = asfloat(_2995.Load((_3000_g_params.hi + 1) * 4 + 0));
        float _5729 = frac(_5725 + _5239);
        float _5735 = asfloat(_2995.Load((_3000_g_params.hi + 2) * 4 + 0));
        float _5739 = frac(_5735 + _5245);
        float _8978 = 0.0f;
        float _8977 = 0.0f;
        float _8976 = 0.0f;
        float _8725 = 0.0f;
        int _8730;
        float _8962;
        float _8963;
        float _8964;
        float _8969;
        float _8970;
        float _8971;
        [branch]
        if (_8595 == 0u)
        {
            [branch]
            if ((_8696 > 0.0f) && (_5640 > 0.0f))
            {
                float3 param_33 = -_4348;
                float3 param_34 = N;
                float3 param_35 = _8693;
                float param_36 = roughness;
                float3 param_37 = base_color;
                float4 _5779 = Evaluate_OrenDiffuse_BSDF(param_33, param_34, param_35, param_36, param_37);
                float mis_weight = 1.0f;
                if (_8694 > 0.0f)
                {
                    float param_38 = _8696;
                    float param_39 = _5779.w;
                    mis_weight = power_heuristic(param_38, param_39);
                }
                float3 _5807 = (_8692 * _5779.xyz) * ((mix_weight * mis_weight) / _8696);
                [branch]
                if (_8697 > 0.5f)
                {
                    float3 param_40 = _4422;
                    float3 param_41 = plane_N;
                    float3 _5818 = offset_ray(param_40, param_41);
                    uint _5872;
                    _5870.InterlockedAdd(8, 1u, _5872);
                    _5880.Store(_5872 * 44 + 0, asuint(_5818.x));
                    _5880.Store(_5872 * 44 + 4, asuint(_5818.y));
                    _5880.Store(_5872 * 44 + 8, asuint(_5818.z));
                    _5880.Store(_5872 * 44 + 12, asuint(_8693.x));
                    _5880.Store(_5872 * 44 + 16, asuint(_8693.y));
                    _5880.Store(_5872 * 44 + 20, asuint(_8693.z));
                    _5880.Store(_5872 * 44 + 24, asuint(_8695 - 9.9999997473787516355514526367188e-05f));
                    _5880.Store(_5872 * 44 + 28, asuint(ray.c[0] * _5807.x));
                    _5880.Store(_5872 * 44 + 32, asuint(ray.c[1] * _5807.y));
                    _5880.Store(_5872 * 44 + 36, asuint(ray.c[2] * _5807.z));
                    _5880.Store(_5872 * 44 + 40, uint(ray.xy));
                }
                else
                {
                    col += _5807;
                }
            }
            bool _5924 = _5252 < _3000_g_params.max_diff_depth;
            bool _5931;
            if (_5924)
            {
                _5931 = _5276 < _3000_g_params.max_total_depth;
            }
            else
            {
                _5931 = _5924;
            }
            [branch]
            if (_5931)
            {
                float3 param_42 = T;
                float3 param_43 = B;
                float3 param_44 = N;
                float3 param_45 = _4348;
                float param_46 = roughness;
                float3 param_47 = base_color;
                float param_48 = _5729;
                float param_49 = _5739;
                float3 param_50;
                float4 _5953 = Sample_OrenDiffuse_BSDF(param_42, param_43, param_44, param_45, param_46, param_47, param_48, param_49, param_50);
                _8730 = ray.ray_depth + 1;
                float3 param_51 = _4422;
                float3 param_52 = plane_N;
                float3 _5964 = offset_ray(param_51, param_52);
                _8962 = _5964.x;
                _8963 = _5964.y;
                _8964 = _5964.z;
                _8969 = param_50.x;
                _8970 = param_50.y;
                _8971 = param_50.z;
                _8976 = ((ray.c[0] * _5953.x) * mix_weight) / _5953.w;
                _8977 = ((ray.c[1] * _5953.y) * mix_weight) / _5953.w;
                _8978 = ((ray.c[2] * _5953.z) * mix_weight) / _5953.w;
                _8725 = _5953.w;
            }
        }
        else
        {
            [branch]
            if (_8595 == 1u)
            {
                float param_53 = 1.0f;
                float param_54 = 1.5f;
                float _6029 = fresnel_dielectric_cos(param_53, param_54);
                float _6033 = roughness * roughness;
                bool _6036 = _8696 > 0.0f;
                bool _6043;
                if (_6036)
                {
                    _6043 = (_6033 * _6033) >= 1.0000000116860974230803549289703e-07f;
                }
                else
                {
                    _6043 = _6036;
                }
                [branch]
                if (_6043 && (_5640 > 0.0f))
                {
                    float3 param_55 = T;
                    float3 param_56 = B;
                    float3 param_57 = N;
                    float3 param_58 = -_4348;
                    float3 param_59 = T;
                    float3 param_60 = B;
                    float3 param_61 = N;
                    float3 param_62 = _8693;
                    float3 param_63 = T;
                    float3 param_64 = B;
                    float3 param_65 = N;
                    float3 param_66 = normalize(_8693 - _4348);
                    float3 param_67 = tangent_from_world(param_55, param_56, param_57, param_58);
                    float3 param_68 = tangent_from_world(param_63, param_64, param_65, param_66);
                    float3 param_69 = tangent_from_world(param_59, param_60, param_61, param_62);
                    float param_70 = _6033;
                    float param_71 = _6033;
                    float param_72 = 1.5f;
                    float param_73 = _6029;
                    float3 param_74 = base_color;
                    float4 _6103 = Evaluate_GGXSpecular_BSDF(param_67, param_68, param_69, param_70, param_71, param_72, param_73, param_74);
                    float mis_weight_1 = 1.0f;
                    if (_8694 > 0.0f)
                    {
                        float param_75 = _8696;
                        float param_76 = _6103.w;
                        mis_weight_1 = power_heuristic(param_75, param_76);
                    }
                    float3 _6131 = (_8692 * _6103.xyz) * ((mix_weight * mis_weight_1) / _8696);
                    [branch]
                    if (_8697 > 0.5f)
                    {
                        float3 param_77 = _4422;
                        float3 param_78 = plane_N;
                        float3 _6142 = offset_ray(param_77, param_78);
                        uint _6189;
                        _5870.InterlockedAdd(8, 1u, _6189);
                        _5880.Store(_6189 * 44 + 0, asuint(_6142.x));
                        _5880.Store(_6189 * 44 + 4, asuint(_6142.y));
                        _5880.Store(_6189 * 44 + 8, asuint(_6142.z));
                        _5880.Store(_6189 * 44 + 12, asuint(_8693.x));
                        _5880.Store(_6189 * 44 + 16, asuint(_8693.y));
                        _5880.Store(_6189 * 44 + 20, asuint(_8693.z));
                        _5880.Store(_6189 * 44 + 24, asuint(_8695 - 9.9999997473787516355514526367188e-05f));
                        _5880.Store(_6189 * 44 + 28, asuint(ray.c[0] * _6131.x));
                        _5880.Store(_6189 * 44 + 32, asuint(ray.c[1] * _6131.y));
                        _5880.Store(_6189 * 44 + 36, asuint(ray.c[2] * _6131.z));
                        _5880.Store(_6189 * 44 + 40, uint(ray.xy));
                    }
                    else
                    {
                        col += _6131;
                    }
                }
                bool _6228 = _5257 < _3000_g_params.max_spec_depth;
                bool _6235;
                if (_6228)
                {
                    _6235 = _5276 < _3000_g_params.max_total_depth;
                }
                else
                {
                    _6235 = _6228;
                }
                [branch]
                if (_6235)
                {
                    float3 param_79 = T;
                    float3 param_80 = B;
                    float3 param_81 = N;
                    float3 param_82 = _4348;
                    float3 param_83;
                    float4 _6254 = Sample_GGXSpecular_BSDF(param_79, param_80, param_81, param_82, roughness, 0.0f, 1.5f, _6029, base_color, _5729, _5739, param_83);
                    _8730 = ray.ray_depth + 256;
                    float3 param_84 = _4422;
                    float3 param_85 = plane_N;
                    float3 _6266 = offset_ray(param_84, param_85);
                    _8962 = _6266.x;
                    _8963 = _6266.y;
                    _8964 = _6266.z;
                    _8969 = param_83.x;
                    _8970 = param_83.y;
                    _8971 = param_83.z;
                    _8976 = ((ray.c[0] * _6254.x) * mix_weight) / _6254.w;
                    _8977 = ((ray.c[1] * _6254.y) * mix_weight) / _6254.w;
                    _8978 = ((ray.c[2] * _6254.z) * mix_weight) / _6254.w;
                    _8725 = _6254.w;
                }
            }
            else
            {
                [branch]
                if (_8595 == 2u)
                {
                    float _6329;
                    if (_4723)
                    {
                        _6329 = _8598 / _8599;
                    }
                    else
                    {
                        _6329 = _8599 / _8598;
                    }
                    float _6347 = roughness * roughness;
                    bool _6350 = _8696 > 0.0f;
                    bool _6357;
                    if (_6350)
                    {
                        _6357 = (_6347 * _6347) >= 1.0000000116860974230803549289703e-07f;
                    }
                    else
                    {
                        _6357 = _6350;
                    }
                    [branch]
                    if (_6357 && (_5640 < 0.0f))
                    {
                        float3 param_86 = T;
                        float3 param_87 = B;
                        float3 param_88 = N;
                        float3 param_89 = -_4348;
                        float3 param_90 = T;
                        float3 param_91 = B;
                        float3 param_92 = N;
                        float3 param_93 = _8693;
                        float3 param_94 = T;
                        float3 param_95 = B;
                        float3 param_96 = N;
                        float3 param_97 = normalize(_8693 - (_4348 * _6329));
                        float3 param_98 = tangent_from_world(param_86, param_87, param_88, param_89);
                        float3 param_99 = tangent_from_world(param_94, param_95, param_96, param_97);
                        float3 param_100 = tangent_from_world(param_90, param_91, param_92, param_93);
                        float param_101 = _6347;
                        float param_102 = _6329;
                        float3 param_103 = base_color;
                        float4 _6416 = Evaluate_GGXRefraction_BSDF(param_98, param_99, param_100, param_101, param_102, param_103);
                        float mis_weight_2 = 1.0f;
                        if (_8694 > 0.0f)
                        {
                            float param_104 = _8696;
                            float param_105 = _6416.w;
                            mis_weight_2 = power_heuristic(param_104, param_105);
                        }
                        float3 _6444 = (_8692 * _6416.xyz) * ((mix_weight * mis_weight_2) / _8696);
                        [branch]
                        if (_8697 > 0.5f)
                        {
                            float3 param_106 = _4422;
                            float3 param_107 = -plane_N;
                            float3 _6456 = offset_ray(param_106, param_107);
                            uint _6503;
                            _5870.InterlockedAdd(8, 1u, _6503);
                            _5880.Store(_6503 * 44 + 0, asuint(_6456.x));
                            _5880.Store(_6503 * 44 + 4, asuint(_6456.y));
                            _5880.Store(_6503 * 44 + 8, asuint(_6456.z));
                            _5880.Store(_6503 * 44 + 12, asuint(_8693.x));
                            _5880.Store(_6503 * 44 + 16, asuint(_8693.y));
                            _5880.Store(_6503 * 44 + 20, asuint(_8693.z));
                            _5880.Store(_6503 * 44 + 24, asuint(_8695 - 9.9999997473787516355514526367188e-05f));
                            _5880.Store(_6503 * 44 + 28, asuint(ray.c[0] * _6444.x));
                            _5880.Store(_6503 * 44 + 32, asuint(ray.c[1] * _6444.y));
                            _5880.Store(_6503 * 44 + 36, asuint(ray.c[2] * _6444.z));
                            _5880.Store(_6503 * 44 + 40, uint(ray.xy));
                        }
                        else
                        {
                            col += _6444;
                        }
                    }
                    bool _6542 = _5262 < _3000_g_params.max_refr_depth;
                    bool _6549;
                    if (_6542)
                    {
                        _6549 = _5276 < _3000_g_params.max_total_depth;
                    }
                    else
                    {
                        _6549 = _6542;
                    }
                    [branch]
                    if (_6549)
                    {
                        float3 param_108 = T;
                        float3 param_109 = B;
                        float3 param_110 = N;
                        float3 param_111 = _4348;
                        float param_112 = roughness;
                        float param_113 = _6329;
                        float3 param_114 = base_color;
                        float param_115 = _5729;
                        float param_116 = _5739;
                        float4 param_117;
                        float4 _6573 = Sample_GGXRefraction_BSDF(param_108, param_109, param_110, param_111, param_112, param_113, param_114, param_115, param_116, param_117);
                        _8730 = ray.ray_depth + 65536;
                        _8976 = ((ray.c[0] * _6573.x) * mix_weight) / _6573.w;
                        _8977 = ((ray.c[1] * _6573.y) * mix_weight) / _6573.w;
                        _8978 = ((ray.c[2] * _6573.z) * mix_weight) / _6573.w;
                        _8725 = _6573.w;
                        float3 param_118 = _4422;
                        float3 param_119 = -plane_N;
                        float3 _6628 = offset_ray(param_118, param_119);
                        _8962 = _6628.x;
                        _8963 = _6628.y;
                        _8964 = _6628.z;
                        _8969 = param_117.x;
                        _8970 = param_117.y;
                        _8971 = param_117.z;
                    }
                }
                else
                {
                    [branch]
                    if (_8595 == 3u)
                    {
                        float mis_weight_3 = 1.0f;
                        if ((_8594 & 4u) != 0u)
                        {
                            float3 env_col_1 = _3000_g_params.env_col.xyz;
                            uint _6667 = asuint(_3000_g_params.env_col.w);
                            if (_6667 != 4294967295u)
                            {
                                env_col_1 *= SampleLatlong_RGBE(_6667, _4348, _3000_g_params.env_rotation);
                            }
                            base_color *= env_col_1;
                        }
                        [branch]
                        if ((_8594 & 5u) != 0u)
                        {
                            float3 _6727 = mul(float4(_5050, 0.0f), _4802.xform).xyz;
                            float _6730 = length(_6727);
                            float _6742 = abs(dot(_4348, _6727 / _6730.xxx));
                            if (_6742 > 0.0f)
                            {
                                float param_120 = ray.pdf;
                                float param_121 = (inter.t * inter.t) / ((0.5f * _6730) * _6742);
                                mis_weight_3 = power_heuristic(param_120, param_121);
                            }
                        }
                        col += (base_color * ((mix_weight * mis_weight_3) * _8596));
                    }
                    else
                    {
                        [branch]
                        if (_8595 == 5u)
                        {
                            bool _6784 = _5268 < _3000_g_params.max_transp_depth;
                            bool _6791;
                            if (_6784)
                            {
                                _6791 = _5276 < _3000_g_params.max_total_depth;
                            }
                            else
                            {
                                _6791 = _6784;
                            }
                            [branch]
                            if (_6791)
                            {
                                _8730 = ray.ray_depth + 16777216;
                                _8725 = ray.pdf;
                                float3 param_122 = _4422;
                                float3 param_123 = -plane_N;
                                float3 _6808 = offset_ray(param_122, param_123);
                                _8962 = _6808.x;
                                _8963 = _6808.y;
                                _8964 = _6808.z;
                                _8969 = ray.d[0];
                                _8970 = ray.d[1];
                                _8971 = ray.d[2];
                                _8976 = ray.c[0];
                                _8977 = ray.c[1];
                                _8978 = ray.c[2];
                            }
                        }
                        else
                        {
                            if (_8595 == 6u)
                            {
                                float metallic = clamp(float((_8601 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                [branch]
                                if (_8910 != 4294967295u)
                                {
                                    metallic *= SampleBilinear(_8910, _5034, int(get_texture_lod(texSize(_8910), _5233))).x;
                                }
                                float specular = clamp(float(_8603 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                [branch]
                                if (_8911 != 4294967295u)
                                {
                                    specular *= SampleBilinear(_8911, _5034, int(get_texture_lod(texSize(_8911), _5233))).x;
                                }
                                float _6918 = clamp(float(_8604 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                float _6926 = clamp(float((_8604 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                float _6933 = clamp(float(_8600 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                float3 _6955 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_8603 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                                float _6962 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                                float param_124 = 1.0f;
                                float param_125 = _6962;
                                float _6967 = fresnel_dielectric_cos(param_124, param_125);
                                float param_126 = dot(_4348, N);
                                float param_127 = _6962;
                                float param_128;
                                float param_129;
                                float param_130;
                                float param_131;
                                get_lobe_weights(lerp(_5684, 1.0f, _6933), lum(lerp(_6955, 1.0f.xxx, ((fresnel_dielectric_cos(param_126, param_127) - _6967) / (1.0f - _6967)).xxx)), specular, metallic, clamp(float(_8602 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _6918, param_128, param_129, param_130, param_131);
                                float3 _7021 = lerp(1.0f.xxx, tint_color, clamp(float((_8600 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _6933;
                                float _7024;
                                if (_4723)
                                {
                                    _7024 = _8598 / _8599;
                                }
                                else
                                {
                                    _7024 = _8599 / _8598;
                                }
                                float param_132 = dot(_4348, N);
                                float param_133 = 1.0f / _7024;
                                float _7047 = fresnel_dielectric_cos(param_132, param_133);
                                float _7054 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _6918))) - 1.0f;
                                float param_134 = 1.0f;
                                float param_135 = _7054;
                                float _7059 = fresnel_dielectric_cos(param_134, param_135);
                                float _7063 = _6926 * _6926;
                                float _7076 = mad(roughness - 1.0f, 1.0f - clamp(float((_8602 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                                float _7080 = _7076 * _7076;
                                [branch]
                                if (_8696 > 0.0f)
                                {
                                    float3 lcol_1 = 0.0f.xxx;
                                    float bsdf_pdf = 0.0f;
                                    bool _7091 = _5640 > 0.0f;
                                    [branch]
                                    if ((param_128 > 1.0000000116860974230803549289703e-07f) && _7091)
                                    {
                                        float3 param_136 = -_4348;
                                        float3 param_137 = N;
                                        float3 param_138 = _8693;
                                        float param_139 = roughness;
                                        float3 param_140 = base_color.xyz;
                                        float3 param_141 = _7021;
                                        bool param_142 = false;
                                        float4 _7111 = Evaluate_PrincipledDiffuse_BSDF(param_136, param_137, param_138, param_139, param_140, param_141, param_142);
                                        bsdf_pdf = mad(param_128, _7111.w, bsdf_pdf);
                                        lcol_1 += (((_8692 * _5640) * (_7111 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * _8696).xxx);
                                    }
                                    float3 H;
                                    [flatten]
                                    if (_7091)
                                    {
                                        H = normalize(_8693 - _4348);
                                    }
                                    else
                                    {
                                        H = normalize(_8693 - (_4348 * _7024));
                                    }
                                    float _7157 = roughness * roughness;
                                    float _7168 = sqrt(mad(-0.89999997615814208984375f, clamp(float((_8597 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f));
                                    float _7172 = _7157 / _7168;
                                    float _7176 = _7157 * _7168;
                                    float3 param_143 = T;
                                    float3 param_144 = B;
                                    float3 param_145 = N;
                                    float3 param_146 = -_4348;
                                    float3 _7187 = tangent_from_world(param_143, param_144, param_145, param_146);
                                    float3 param_147 = T;
                                    float3 param_148 = B;
                                    float3 param_149 = N;
                                    float3 param_150 = _8693;
                                    float3 _7198 = tangent_from_world(param_147, param_148, param_149, param_150);
                                    float3 param_151 = T;
                                    float3 param_152 = B;
                                    float3 param_153 = N;
                                    float3 param_154 = H;
                                    float3 _7208 = tangent_from_world(param_151, param_152, param_153, param_154);
                                    bool _7210 = param_129 > 0.0f;
                                    bool _7217;
                                    if (_7210)
                                    {
                                        _7217 = (_7172 * _7176) >= 1.0000000116860974230803549289703e-07f;
                                    }
                                    else
                                    {
                                        _7217 = _7210;
                                    }
                                    [branch]
                                    if (_7217 && _7091)
                                    {
                                        float3 param_155 = _7187;
                                        float3 param_156 = _7208;
                                        float3 param_157 = _7198;
                                        float param_158 = _7172;
                                        float param_159 = _7176;
                                        float param_160 = _6962;
                                        float param_161 = _6967;
                                        float3 param_162 = _6955;
                                        float4 _7240 = Evaluate_GGXSpecular_BSDF(param_155, param_156, param_157, param_158, param_159, param_160, param_161, param_162);
                                        bsdf_pdf = mad(param_129, _7240.w, bsdf_pdf);
                                        lcol_1 += ((_8692 * _7240.xyz) / _8696.xxx);
                                    }
                                    bool _7259 = param_130 > 0.0f;
                                    bool _7266;
                                    if (_7259)
                                    {
                                        _7266 = (_7063 * _7063) >= 1.0000000116860974230803549289703e-07f;
                                    }
                                    else
                                    {
                                        _7266 = _7259;
                                    }
                                    [branch]
                                    if (_7266 && _7091)
                                    {
                                        float3 param_163 = _7187;
                                        float3 param_164 = _7208;
                                        float3 param_165 = _7198;
                                        float param_166 = _7063;
                                        float param_167 = _7054;
                                        float param_168 = _7059;
                                        float4 _7285 = Evaluate_PrincipledClearcoat_BSDF(param_163, param_164, param_165, param_166, param_167, param_168);
                                        bsdf_pdf = mad(param_130, _7285.w, bsdf_pdf);
                                        lcol_1 += (((_8692 * 0.25f) * _7285.xyz) / _8696.xxx);
                                    }
                                    [branch]
                                    if (param_131 > 0.0f)
                                    {
                                        bool _7309 = _7047 != 0.0f;
                                        bool _7316;
                                        if (_7309)
                                        {
                                            _7316 = (_7157 * _7157) >= 1.0000000116860974230803549289703e-07f;
                                        }
                                        else
                                        {
                                            _7316 = _7309;
                                        }
                                        [branch]
                                        if (_7316 && _7091)
                                        {
                                            float3 param_169 = _7187;
                                            float3 param_170 = _7208;
                                            float3 param_171 = _7198;
                                            float param_172 = _7157;
                                            float param_173 = _7157;
                                            float param_174 = 1.0f;
                                            float param_175 = 0.0f;
                                            float3 param_176 = 1.0f.xxx;
                                            float4 _7336 = Evaluate_GGXSpecular_BSDF(param_169, param_170, param_171, param_172, param_173, param_174, param_175, param_176);
                                            bsdf_pdf = mad(param_131 * _7047, _7336.w, bsdf_pdf);
                                            lcol_1 += ((_8692 * _7336.xyz) * (_7047 / _8696));
                                        }
                                        bool _7358 = _7047 != 1.0f;
                                        bool _7365;
                                        if (_7358)
                                        {
                                            _7365 = (_7080 * _7080) >= 1.0000000116860974230803549289703e-07f;
                                        }
                                        else
                                        {
                                            _7365 = _7358;
                                        }
                                        [branch]
                                        if (_7365 && (_5640 < 0.0f))
                                        {
                                            float3 param_177 = _7187;
                                            float3 param_178 = _7208;
                                            float3 param_179 = _7198;
                                            float param_180 = _7080;
                                            float param_181 = _7024;
                                            float3 param_182 = base_color;
                                            float4 _7384 = Evaluate_GGXRefraction_BSDF(param_177, param_178, param_179, param_180, param_181, param_182);
                                            float _7387 = 1.0f - _7047;
                                            bsdf_pdf = mad(param_131 * _7387, _7384.w, bsdf_pdf);
                                            lcol_1 += ((_8692 * _7384.xyz) * (_7387 / _8696));
                                        }
                                    }
                                    float mis_weight_4 = 1.0f;
                                    [flatten]
                                    if (_8694 > 0.0f)
                                    {
                                        float param_183 = _8696;
                                        float param_184 = bsdf_pdf;
                                        mis_weight_4 = power_heuristic(param_183, param_184);
                                    }
                                    lcol_1 *= (mix_weight * mis_weight_4);
                                    [branch]
                                    if (_8697 > 0.5f)
                                    {
                                        float3 _7432;
                                        if (_5640 < 0.0f)
                                        {
                                            _7432 = -plane_N;
                                        }
                                        else
                                        {
                                            _7432 = plane_N;
                                        }
                                        float3 param_185 = _4422;
                                        float3 param_186 = _7432;
                                        float3 _7443 = offset_ray(param_185, param_186);
                                        uint _7490;
                                        _5870.InterlockedAdd(8, 1u, _7490);
                                        _5880.Store(_7490 * 44 + 0, asuint(_7443.x));
                                        _5880.Store(_7490 * 44 + 4, asuint(_7443.y));
                                        _5880.Store(_7490 * 44 + 8, asuint(_7443.z));
                                        _5880.Store(_7490 * 44 + 12, asuint(_8693.x));
                                        _5880.Store(_7490 * 44 + 16, asuint(_8693.y));
                                        _5880.Store(_7490 * 44 + 20, asuint(_8693.z));
                                        _5880.Store(_7490 * 44 + 24, asuint(_8695 - 9.9999997473787516355514526367188e-05f));
                                        _5880.Store(_7490 * 44 + 28, asuint(ray.c[0] * lcol_1.x));
                                        _5880.Store(_7490 * 44 + 32, asuint(ray.c[1] * lcol_1.y));
                                        _5880.Store(_7490 * 44 + 36, asuint(ray.c[2] * lcol_1.z));
                                        _5880.Store(_7490 * 44 + 40, uint(ray.xy));
                                    }
                                    else
                                    {
                                        col += lcol_1;
                                    }
                                }
                                [branch]
                                if (mix_rand < param_128)
                                {
                                    bool _7534 = _5252 < _3000_g_params.max_diff_depth;
                                    bool _7541;
                                    if (_7534)
                                    {
                                        _7541 = _5276 < _3000_g_params.max_total_depth;
                                    }
                                    else
                                    {
                                        _7541 = _7534;
                                    }
                                    if (_7541)
                                    {
                                        float3 param_187 = T;
                                        float3 param_188 = B;
                                        float3 param_189 = N;
                                        float3 param_190 = _4348;
                                        float param_191 = roughness;
                                        float3 param_192 = base_color.xyz;
                                        float3 param_193 = _7021;
                                        bool param_194 = false;
                                        float param_195 = _5729;
                                        float param_196 = _5739;
                                        float3 param_197;
                                        float4 _7566 = Sample_PrincipledDiffuse_BSDF(param_187, param_188, param_189, param_190, param_191, param_192, param_193, param_194, param_195, param_196, param_197);
                                        float3 _7572 = _7566.xyz * (1.0f - metallic);
                                        _8730 = ray.ray_depth + 1;
                                        float3 param_198 = _4422;
                                        float3 param_199 = plane_N;
                                        float3 _7588 = offset_ray(param_198, param_199);
                                        _8962 = _7588.x;
                                        _8963 = _7588.y;
                                        _8964 = _7588.z;
                                        _8969 = param_197.x;
                                        _8970 = param_197.y;
                                        _8971 = param_197.z;
                                        _8976 = ((ray.c[0] * _7572.x) * mix_weight) / param_128;
                                        _8977 = ((ray.c[1] * _7572.y) * mix_weight) / param_128;
                                        _8978 = ((ray.c[2] * _7572.z) * mix_weight) / param_128;
                                        _8725 = _7566.w;
                                    }
                                }
                                else
                                {
                                    float _7644 = param_128 + param_129;
                                    [branch]
                                    if (mix_rand < _7644)
                                    {
                                        bool _7651 = _5257 < _3000_g_params.max_spec_depth;
                                        bool _7658;
                                        if (_7651)
                                        {
                                            _7658 = _5276 < _3000_g_params.max_total_depth;
                                        }
                                        else
                                        {
                                            _7658 = _7651;
                                        }
                                        if (_7658)
                                        {
                                            float3 param_200 = T;
                                            float3 param_201 = B;
                                            float3 param_202 = N;
                                            float3 param_203 = _4348;
                                            float3 param_204;
                                            float4 _7685 = Sample_GGXSpecular_BSDF(param_200, param_201, param_202, param_203, roughness, clamp(float((_8597 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _6962, _6967, _6955, _5729, _5739, param_204);
                                            float _7690 = _7685.w * param_129;
                                            _8730 = ray.ray_depth + 256;
                                            _8976 = ((ray.c[0] * _7685.x) * mix_weight) / _7690;
                                            _8977 = ((ray.c[1] * _7685.y) * mix_weight) / _7690;
                                            _8978 = ((ray.c[2] * _7685.z) * mix_weight) / _7690;
                                            _8725 = _7690;
                                            float3 param_205 = _4422;
                                            float3 param_206 = plane_N;
                                            float3 _7737 = offset_ray(param_205, param_206);
                                            _8962 = _7737.x;
                                            _8963 = _7737.y;
                                            _8964 = _7737.z;
                                            _8969 = param_204.x;
                                            _8970 = param_204.y;
                                            _8971 = param_204.z;
                                        }
                                    }
                                    else
                                    {
                                        float _7762 = _7644 + param_130;
                                        [branch]
                                        if (mix_rand < _7762)
                                        {
                                            bool _7769 = _5257 < _3000_g_params.max_spec_depth;
                                            bool _7776;
                                            if (_7769)
                                            {
                                                _7776 = _5276 < _3000_g_params.max_total_depth;
                                            }
                                            else
                                            {
                                                _7776 = _7769;
                                            }
                                            if (_7776)
                                            {
                                                float3 param_207 = T;
                                                float3 param_208 = B;
                                                float3 param_209 = N;
                                                float3 param_210 = _4348;
                                                float param_211 = _7063;
                                                float param_212 = _7054;
                                                float param_213 = _7059;
                                                float param_214 = _5729;
                                                float param_215 = _5739;
                                                float3 param_216;
                                                float4 _7800 = Sample_PrincipledClearcoat_BSDF(param_207, param_208, param_209, param_210, param_211, param_212, param_213, param_214, param_215, param_216);
                                                float _7805 = _7800.w * param_130;
                                                _8730 = ray.ray_depth + 256;
                                                _8976 = (((0.25f * ray.c[0]) * _7800.x) * mix_weight) / _7805;
                                                _8977 = (((0.25f * ray.c[1]) * _7800.y) * mix_weight) / _7805;
                                                _8978 = (((0.25f * ray.c[2]) * _7800.z) * mix_weight) / _7805;
                                                _8725 = _7805;
                                                float3 param_217 = _4422;
                                                float3 param_218 = plane_N;
                                                float3 _7855 = offset_ray(param_217, param_218);
                                                _8962 = _7855.x;
                                                _8963 = _7855.y;
                                                _8964 = _7855.z;
                                                _8969 = param_216.x;
                                                _8970 = param_216.y;
                                                _8971 = param_216.z;
                                            }
                                        }
                                        else
                                        {
                                            bool _7877 = mix_rand >= _7047;
                                            bool _7884;
                                            if (_7877)
                                            {
                                                _7884 = _5262 < _3000_g_params.max_refr_depth;
                                            }
                                            else
                                            {
                                                _7884 = _7877;
                                            }
                                            bool _7898;
                                            if (!_7884)
                                            {
                                                bool _7890 = mix_rand < _7047;
                                                bool _7897;
                                                if (_7890)
                                                {
                                                    _7897 = _5257 < _3000_g_params.max_spec_depth;
                                                }
                                                else
                                                {
                                                    _7897 = _7890;
                                                }
                                                _7898 = _7897;
                                            }
                                            else
                                            {
                                                _7898 = _7884;
                                            }
                                            bool _7905;
                                            if (_7898)
                                            {
                                                _7905 = _5276 < _3000_g_params.max_total_depth;
                                            }
                                            else
                                            {
                                                _7905 = _7898;
                                            }
                                            [branch]
                                            if (_7905)
                                            {
                                                float _7913 = mix_rand;
                                                float _7917 = (_7913 - _7762) / param_131;
                                                mix_rand = _7917;
                                                float4 F;
                                                float3 V;
                                                [branch]
                                                if (_7917 < _7047)
                                                {
                                                    float3 param_219 = T;
                                                    float3 param_220 = B;
                                                    float3 param_221 = N;
                                                    float3 param_222 = _4348;
                                                    float3 param_223;
                                                    float4 _7937 = Sample_GGXSpecular_BSDF(param_219, param_220, param_221, param_222, roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, _5729, _5739, param_223);
                                                    V = param_223;
                                                    F = _7937;
                                                    _8730 = ray.ray_depth + 256;
                                                    float3 param_224 = _4422;
                                                    float3 param_225 = plane_N;
                                                    float3 _7948 = offset_ray(param_224, param_225);
                                                    _8962 = _7948.x;
                                                    _8963 = _7948.y;
                                                    _8964 = _7948.z;
                                                }
                                                else
                                                {
                                                    float3 param_226 = T;
                                                    float3 param_227 = B;
                                                    float3 param_228 = N;
                                                    float3 param_229 = _4348;
                                                    float param_230 = _7076;
                                                    float param_231 = _7024;
                                                    float3 param_232 = base_color;
                                                    float param_233 = _5729;
                                                    float param_234 = _5739;
                                                    float4 param_235;
                                                    float4 _7979 = Sample_GGXRefraction_BSDF(param_226, param_227, param_228, param_229, param_230, param_231, param_232, param_233, param_234, param_235);
                                                    F = _7979;
                                                    V = param_235.xyz;
                                                    _8730 = ray.ray_depth + 65536;
                                                    float3 param_236 = _4422;
                                                    float3 param_237 = -plane_N;
                                                    float3 _7993 = offset_ray(param_236, param_237);
                                                    _8962 = _7993.x;
                                                    _8963 = _7993.y;
                                                    _8964 = _7993.z;
                                                }
                                                float4 _9376 = F;
                                                float _8006 = _9376.w * param_131;
                                                float4 _9378 = _9376;
                                                _9378.w = _8006;
                                                F = _9378;
                                                _8976 = ((ray.c[0] * _9376.x) * mix_weight) / _8006;
                                                _8977 = ((ray.c[1] * _9376.y) * mix_weight) / _8006;
                                                _8978 = ((ray.c[2] * _9376.z) * mix_weight) / _8006;
                                                _8725 = _8006;
                                                _8969 = V.x;
                                                _8970 = V.y;
                                                _8971 = V.z;
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
        float _8066 = max(_8976, max(_8977, _8978));
        float _8079;
        if (_5276 >= _3000_g_params.termination_start_depth)
        {
            _8079 = max(0.0500000007450580596923828125f, 1.0f - _8066);
        }
        else
        {
            _8079 = 0.0f;
        }
        bool _8093 = (frac(asfloat(_2995.Load((_3000_g_params.hi + 6) * 4 + 0)) + _5239) >= _8079) && (_8066 > 0.0f);
        bool _8099;
        if (_8093)
        {
            _8099 = _8725 > 0.0f;
        }
        else
        {
            _8099 = _8093;
        }
        [branch]
        if (_8099)
        {
            float _8103 = 1.0f - _8079;
            float _8105 = _8976;
            float _8106 = _8105 / _8103;
            _8976 = _8106;
            float _8111 = _8977;
            float _8112 = _8111 / _8103;
            _8977 = _8112;
            float _8117 = _8978;
            float _8118 = _8117 / _8103;
            _8978 = _8118;
            uint _8122;
            _5870.InterlockedAdd(0, 1u, _8122);
            _8130.Store(_8122 * 56 + 0, asuint(_8962));
            _8130.Store(_8122 * 56 + 4, asuint(_8963));
            _8130.Store(_8122 * 56 + 8, asuint(_8964));
            _8130.Store(_8122 * 56 + 12, asuint(_8969));
            _8130.Store(_8122 * 56 + 16, asuint(_8970));
            _8130.Store(_8122 * 56 + 20, asuint(_8971));
            _8130.Store(_8122 * 56 + 24, asuint(_8725));
            _8130.Store(_8122 * 56 + 28, asuint(_8106));
            _8130.Store(_8122 * 56 + 32, asuint(_8112));
            _8130.Store(_8122 * 56 + 36, asuint(_8118));
            _8130.Store(_8122 * 56 + 40, asuint(_5223));
            _8130.Store(_8122 * 56 + 44, asuint(ray.cone_spread));
            _8130.Store(_8122 * 56 + 48, uint(ray.xy));
            _8130.Store(_8122 * 56 + 52, uint(_8730));
        }
        _8347 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _8347;
}

void comp_main()
{
    do
    {
        int _8200 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_8200) >= _5870.Load(4))
        {
            break;
        }
        int _8216 = int(_8213.Load(_8200 * 56 + 48));
        int _8219 = (_8216 >> 16) & 65535;
        int _8223 = int(_8213.Load(_8200 * 56 + 48));
        int _8224 = _8223 & 65535;
        hit_data_t _8243;
        _8243.mask = int(_8239.Load(_8200 * 24 + 0));
        _8243.obj_index = int(_8239.Load(_8200 * 24 + 4));
        _8243.prim_index = int(_8239.Load(_8200 * 24 + 8));
        _8243.t = asfloat(_8239.Load(_8200 * 24 + 12));
        _8243.u = asfloat(_8239.Load(_8200 * 24 + 16));
        _8243.v = asfloat(_8239.Load(_8200 * 24 + 20));
        ray_data_t _8259;
        [unroll]
        for (int _83ident = 0; _83ident < 3; _83ident++)
        {
            _8259.o[_83ident] = asfloat(_8213.Load(_83ident * 4 + _8200 * 56 + 0));
        }
        [unroll]
        for (int _84ident = 0; _84ident < 3; _84ident++)
        {
            _8259.d[_84ident] = asfloat(_8213.Load(_84ident * 4 + _8200 * 56 + 12));
        }
        _8259.pdf = asfloat(_8213.Load(_8200 * 56 + 24));
        [unroll]
        for (int _85ident = 0; _85ident < 3; _85ident++)
        {
            _8259.c[_85ident] = asfloat(_8213.Load(_85ident * 4 + _8200 * 56 + 28));
        }
        _8259.cone_width = asfloat(_8213.Load(_8200 * 56 + 40));
        _8259.cone_spread = asfloat(_8213.Load(_8200 * 56 + 44));
        _8259.xy = int(_8213.Load(_8200 * 56 + 48));
        _8259.ray_depth = int(_8213.Load(_8200 * 56 + 52));
        int param = (_8224 * int(_3000_g_params.img_size.x)) + _8219;
        hit_data_t _8408 = { _8243.mask, _8243.obj_index, _8243.prim_index, _8243.t, _8243.u, _8243.v };
        hit_data_t param_1 = _8408;
        float _8446[3] = { _8259.c[0], _8259.c[1], _8259.c[2] };
        float _8439[3] = { _8259.d[0], _8259.d[1], _8259.d[2] };
        float _8432[3] = { _8259.o[0], _8259.o[1], _8259.o[2] };
        ray_data_t _8425 = { _8432, _8439, _8259.pdf, _8446, _8259.cone_width, _8259.cone_spread, _8259.xy, _8259.ray_depth };
        ray_data_t param_2 = _8425;
        float3 _8301 = ShadeSurface(param, param_1, param_2);
        int2 _8308 = int2(_8219, _8224);
        g_out_img[_8308] = float4(_8301 + g_out_img[_8308].xyz, 1.0f);
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
