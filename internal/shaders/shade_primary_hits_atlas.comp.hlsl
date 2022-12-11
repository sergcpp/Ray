struct atlas_texture_t
{
    uint size;
    uint atlas;
    uint page[4];
    uint pos[14];
};

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

ByteAddressBuffer _855 : register(t20, space0);
ByteAddressBuffer _3219 : register(t15, space0);
ByteAddressBuffer _3256 : register(t6, space0);
ByteAddressBuffer _3260 : register(t7, space0);
ByteAddressBuffer _4135 : register(t11, space0);
ByteAddressBuffer _4160 : register(t13, space0);
ByteAddressBuffer _4164 : register(t14, space0);
ByteAddressBuffer _4475 : register(t10, space0);
ByteAddressBuffer _4479 : register(t9, space0);
ByteAddressBuffer _5225 : register(t12, space0);
RWByteAddressBuffer _6301 : register(u3, space0);
RWByteAddressBuffer _6311 : register(u2, space0);
RWByteAddressBuffer _8527 : register(u1, space0);
ByteAddressBuffer _8631 : register(t4, space0);
ByteAddressBuffer _8652 : register(t5, space0);
ByteAddressBuffer _8731 : register(t8, space0);
cbuffer UniformParams
{
    Params _3224_g_params : packoffset(c0);
};

Texture2DArray<float4> g_atlases[7] : register(t21, space0);
SamplerState _g_atlases_sampler[7] : register(s21, space0);
Texture2D<float4> g_env_qtree : register(t16, space0);
SamplerState _g_env_qtree_sampler : register(s16, space0);
RWTexture2D<float4> g_out_img : register(u0, space0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

float2 TransformUV(float2 _uv, atlas_texture_t t, int mip_level)
{
    uint _8862[14] = t.pos;
    uint _8865[14] = t.pos;
    uint _948 = t.size & 16383u;
    uint _951 = t.size >> uint(16);
    uint _952 = _951 & 16383u;
    float2 size = float2(float(_948), float(_952));
    if ((_951 & 32768u) != 0u)
    {
        size = float2(float(_948 >> uint(mip_level)), float(_952 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_8862[mip_level] & 65535u), float((_8865[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
}

float3 rgbe_to_rgb(float4 rgbe)
{
    return rgbe.xyz * exp2(mad(255.0f, rgbe.w, -128.0f));
}

float3 SampleLatlong_RGBE(atlas_texture_t t, float3 dir, float y_rotation)
{
    float _1123 = sqrt(mad(dir.x, dir.x, dir.z * dir.z));
    float _1128;
    if (_1123 > 1.0000000116860974230803549289703e-07f)
    {
        _1128 = clamp(dir.x / _1123, -1.0f, 1.0f);
    }
    else
    {
        _1128 = 0.0f;
    }
    float _1139 = acos(_1128) + y_rotation;
    float phi = _1139;
    if (_1139 < 0.0f)
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
    float2 _1168 = TransformUV(float2(u, acos(clamp(dir.y, -1.0f, 1.0f)) * 0.3183098733425140380859375f), t, 0) + 1.0f.xx;
    uint _1175 = t.atlas;
    int3 _1184 = int3(int2(_1168), int(t.page[0] & 255u));
    float2 _1231 = frac(_1168);
    float4 param = g_atlases[NonUniformResourceIndex(_1175)].Load(int4(_1184, 0), int2(0, 0));
    float4 param_1 = g_atlases[NonUniformResourceIndex(_1175)].Load(int4(_1184, 0), int2(1, 0));
    float4 param_2 = g_atlases[NonUniformResourceIndex(_1175)].Load(int4(_1184, 0), int2(0, 1));
    float4 param_3 = g_atlases[NonUniformResourceIndex(_1175)].Load(int4(_1184, 0), int2(1, 1));
    float _1251 = _1231.x;
    float _1256 = 1.0f - _1251;
    float _1272 = _1231.y;
    return (((rgbe_to_rgb(param_3) * _1251) + (rgbe_to_rgb(param_2) * _1256)) * _1272) + (((rgbe_to_rgb(param_1) * _1251) + (rgbe_to_rgb(param) * _1256)) * (1.0f - _1272));
}

float2 DirToCanonical(float3 d, float y_rotation)
{
    float _585 = (-atan2(d.z, d.x)) + y_rotation;
    float phi = _585;
    if (_585 < 0.0f)
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
    float2 _611 = DirToCanonical(L, y_rotation);
    float factor = 1.0f;
    while (lod >= 0)
    {
        int2 _631 = clamp(int2(_611 * float(res)), int2(0, 0), (res - 1).xx);
        float4 quad = qtree_tex.Load(int3(_631 / int2(2, 2), lod));
        float _666 = ((quad.x + quad.y) + quad.z) + quad.w;
        if (_666 <= 0.0f)
        {
            break;
        }
        factor *= ((4.0f * quad[(0 | ((_631.x & 1) << 0)) | ((_631.y & 1) << 1)]) / _666);
        lod--;
        res *= 2;
    }
    return factor * 0.079577468335628509521484375f;
}

float power_heuristic(float a, float b)
{
    float _1285 = a * a;
    return _1285 / mad(b, b, _1285);
}

float3 TransformNormal(float3 n, float4x4 inv_xform)
{
    return mul(float4(n, 0.0f), transpose(inv_xform)).xyz;
}

int hash(int x)
{
    uint _362 = uint(x);
    uint _369 = ((_362 >> uint(16)) ^ _362) * 73244475u;
    uint _374 = ((_369 >> uint(16)) ^ _369) * 73244475u;
    return int((_374 >> uint(16)) ^ _374);
}

float construct_float(inout uint m)
{
    m &= 8388607u;
    m |= 1065353216u;
    return asfloat(m) - 1.0f;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _461 = mad(col.z, 31.875f, 1.0f);
    float _471 = (col.x - 0.501960813999176025390625f) / _461;
    float _477 = (col.y - 0.501960813999176025390625f) / _461;
    return float3((col.w + _471) - _477, col.w + _477, (col.w - _471) - _477);
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
    atlas_texture_t _985;
    _985.size = _855.Load(index * 80 + 0);
    _985.atlas = _855.Load(index * 80 + 4);
    [unroll]
    for (int _61ident = 0; _61ident < 4; _61ident++)
    {
        _985.page[_61ident] = _855.Load(_61ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _62ident = 0; _62ident < 14; _62ident++)
    {
        _985.pos[_62ident] = _855.Load(_62ident * 4 + index * 80 + 24);
    }
    uint _8870[4];
    _8870[0] = _985.page[0];
    _8870[1] = _985.page[1];
    _8870[2] = _985.page[2];
    _8870[3] = _985.page[3];
    uint _8906[14] = { _985.pos[0], _985.pos[1], _985.pos[2], _985.pos[3], _985.pos[4], _985.pos[5], _985.pos[6], _985.pos[7], _985.pos[8], _985.pos[9], _985.pos[10], _985.pos[11], _985.pos[12], _985.pos[13] };
    atlas_texture_t _8876 = { _985.size, _985.atlas, _8870, _8906 };
    uint _1055 = _985.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_1055)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_1055)], float3(TransformUV(uvs, _8876, lod) * 0.000118371215648949146270751953125f.xx, float((_8870[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
    bool _1070;
    if (maybe_YCoCg)
    {
        _1070 = _985.atlas == 4u;
    }
    else
    {
        _1070 = maybe_YCoCg;
    }
    if (_1070)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _1089;
    if (maybe_SRGB)
    {
        _1089 = (_985.size & 32768u) != 0u;
    }
    else
    {
        _1089 = maybe_SRGB;
    }
    if (_1089)
    {
        float3 param_1 = res.xyz;
        float3 _1095 = srgb_to_rgb(param_1);
        float4 _9874 = res;
        _9874.x = _1095.x;
        float4 _9876 = _9874;
        _9876.y = _1095.y;
        float4 _9878 = _9876;
        _9878.z = _1095.z;
        res = _9878;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod)
{
    return SampleBilinear(index, uvs, lod, false, false);
}

float fresnel_dielectric_cos(float cosi, float eta)
{
    float _1317 = abs(cosi);
    float _1326 = mad(_1317, _1317, mad(eta, eta, -1.0f));
    float g = _1326;
    float result;
    if (_1326 > 0.0f)
    {
        float _1331 = g;
        float _1332 = sqrt(_1331);
        g = _1332;
        float _1336 = _1332 - _1317;
        float _1339 = _1332 + _1317;
        float _1340 = _1336 / _1339;
        float _1354 = mad(_1317, _1339, -1.0f) / mad(_1317, _1336, 1.0f);
        result = ((0.5f * _1340) * _1340) * mad(_1354, _1354, 1.0f);
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
    float3 _8743;
    do
    {
        float _1390 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1390)
        {
            _8743 = N;
            break;
        }
        float3 _1410 = normalize(N - (Ng * dot(N, Ng)));
        float _1414 = dot(I, _1410);
        float _1418 = dot(I, Ng);
        float _1430 = mad(_1414, _1414, _1418 * _1418);
        float param = (_1414 * _1414) * mad(-_1390, _1390, _1430);
        float _1440 = safe_sqrtf(param);
        float _1446 = mad(_1418, _1390, _1430);
        float _1449 = 0.5f / _1430;
        float _1454 = _1440 + _1446;
        float _1455 = _1449 * _1454;
        float _1461 = (-_1440) + _1446;
        float _1462 = _1449 * _1461;
        bool _1470 = (_1455 > 9.9999997473787516355514526367188e-06f) && (_1455 <= 1.000010013580322265625f);
        bool valid1 = _1470;
        bool _1476 = (_1462 > 9.9999997473787516355514526367188e-06f) && (_1462 <= 1.000010013580322265625f);
        bool valid2 = _1476;
        float2 N_new;
        if (_1470 && _1476)
        {
            float _10173 = (-0.5f) / _1430;
            float param_1 = mad(_10173, _1454, 1.0f);
            float _1486 = safe_sqrtf(param_1);
            float param_2 = _1455;
            float _1489 = safe_sqrtf(param_2);
            float2 _1490 = float2(_1486, _1489);
            float param_3 = mad(_10173, _1461, 1.0f);
            float _1495 = safe_sqrtf(param_3);
            float param_4 = _1462;
            float _1498 = safe_sqrtf(param_4);
            float2 _1499 = float2(_1495, _1498);
            float _10175 = -_1418;
            float _1515 = mad(2.0f * mad(_1486, _1414, _1489 * _1418), _1489, _10175);
            float _1531 = mad(2.0f * mad(_1495, _1414, _1498 * _1418), _1498, _10175);
            bool _1533 = _1515 >= 9.9999997473787516355514526367188e-06f;
            valid1 = _1533;
            bool _1535 = _1531 >= 9.9999997473787516355514526367188e-06f;
            valid2 = _1535;
            if (_1533 && _1535)
            {
                bool2 _1548 = (_1515 < _1531).xx;
                N_new = float2(_1548.x ? _1490.x : _1499.x, _1548.y ? _1490.y : _1499.y);
            }
            else
            {
                bool2 _1556 = (_1515 > _1531).xx;
                N_new = float2(_1556.x ? _1490.x : _1499.x, _1556.y ? _1490.y : _1499.y);
            }
        }
        else
        {
            if (!(valid1 || valid2))
            {
                _8743 = Ng;
                break;
            }
            float _1568 = valid1 ? _1455 : _1462;
            float param_5 = 1.0f - _1568;
            float param_6 = _1568;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _8743 = (_1410 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _8743;
}

float3 rotate_around_axis(float3 p, float3 axis, float angle)
{
    float _1641 = cos(angle);
    float _1644 = sin(angle);
    float _1648 = 1.0f - _1641;
    return float3(mad(mad(_1648 * axis.x, axis.z, axis.y * _1644), p.z, mad(mad(_1648 * axis.x, axis.x, _1641), p.x, mad(_1648 * axis.x, axis.y, -(axis.z * _1644)) * p.y)), mad(mad(_1648 * axis.y, axis.z, -(axis.x * _1644)), p.z, mad(mad(_1648 * axis.x, axis.y, axis.z * _1644), p.x, mad(_1648 * axis.y, axis.y, _1641) * p.y)), mad(mad(_1648 * axis.z, axis.z, _1641), p.z, mad(mad(_1648 * axis.x, axis.z, -(axis.y * _1644)), p.x, mad(_1648 * axis.y, axis.z, axis.x * _1644) * p.y)));
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
    float3 _8768;
    do
    {
        float2 _3134 = (float2(r1, r2) * 2.0f) - 1.0f.xx;
        float _3136 = _3134.x;
        bool _3137 = _3136 == 0.0f;
        bool _3143;
        if (_3137)
        {
            _3143 = _3134.y == 0.0f;
        }
        else
        {
            _3143 = _3137;
        }
        if (_3143)
        {
            _8768 = N;
            break;
        }
        float _3152 = _3134.y;
        float r;
        float theta;
        if (abs(_3136) > abs(_3152))
        {
            r = _3136;
            theta = 0.785398185253143310546875f * (_3152 / _3136);
        }
        else
        {
            r = _3152;
            theta = 1.57079637050628662109375f * mad(-0.5f, _3136 / _3152, 1.0f);
        }
        float3 param;
        float3 param_1;
        create_tbn(N, param, param_1);
        _8768 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _8768;
}

float3 CanonicalToDir(float2 p, float y_rotation)
{
    float _535 = mad(2.0f, p.x, -1.0f);
    float _540 = mad(6.283185482025146484375f, p.y, y_rotation);
    float phi = _540;
    if (_540 < 0.0f)
    {
        phi += 6.283185482025146484375f;
    }
    if (phi > 6.283185482025146484375f)
    {
        phi -= 6.283185482025146484375f;
    }
    float _558 = sqrt(mad(-_535, _535, 1.0f));
    return float3(_558 * cos(phi), _535, (-_558) * sin(phi));
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
        float _732 = quad.x + quad.z;
        float partial = _732;
        float _739 = (_732 + quad.y) + quad.w;
        if (_739 <= 0.0f)
        {
            break;
        }
        float _748 = partial / _739;
        float boundary = _748;
        int index = 0;
        if (_sample < _748)
        {
            _sample /= boundary;
            boundary = quad.x / partial;
        }
        else
        {
            float _763 = partial;
            float _764 = _739 - _763;
            partial = _764;
            float2 _9861 = origin;
            _9861.x = origin.x + _step;
            origin = _9861;
            _sample = (_sample - boundary) / (1.0f - boundary);
            boundary = quad.y / _764;
            index |= 1;
        }
        if (_sample < boundary)
        {
            _sample /= boundary;
        }
        else
        {
            float2 _9864 = origin;
            _9864.y = origin.y + _step;
            origin = _9864;
            _sample = (_sample - boundary) / (1.0f - boundary);
            index |= 2;
        }
        factor *= ((4.0f * quad[index]) / _739);
        lod--;
        res *= 2;
        _step *= 0.5f;
    }
    float2 _821 = origin;
    float2 _822 = _821 + (float2(rx, ry) * (2.0f * _step));
    origin = _822;
    return float4(CanonicalToDir(_822, y_rotation), factor * 0.079577468335628509521484375f);
}

void SampleLightSource(float3 P, float2 sample_off, inout light_sample_t ls)
{
    float _3235 = frac(asfloat(_3219.Load((_3224_g_params.hi + 3) * 4 + 0)) + sample_off.x);
    float _3240 = float(_3224_g_params.li_count);
    uint _3247 = min(uint(_3235 * _3240), uint(_3224_g_params.li_count - 1));
    light_t _3267;
    _3267.type_and_param0 = _3256.Load4(_3260.Load(_3247 * 4 + 0) * 64 + 0);
    _3267.param1 = asfloat(_3256.Load4(_3260.Load(_3247 * 4 + 0) * 64 + 16));
    _3267.param2 = asfloat(_3256.Load4(_3260.Load(_3247 * 4 + 0) * 64 + 32));
    _3267.param3 = asfloat(_3256.Load4(_3260.Load(_3247 * 4 + 0) * 64 + 48));
    ls.col = asfloat(_3267.type_and_param0.yzw);
    ls.col *= _3240;
    ls.cast_shadow = float((_3267.type_and_param0.x & 32u) != 0u);
    uint _3301 = _3267.type_and_param0.x & 31u;
    [branch]
    if (_3301 == 0u)
    {
        float _3315 = frac(asfloat(_3219.Load((_3224_g_params.hi + 4) * 4 + 0)) + sample_off.x);
        float3 _3331 = P - _3267.param1.xyz;
        float3 _3338 = _3331 / length(_3331).xxx;
        float _3345 = sqrt(clamp(mad(-_3315, _3315, 1.0f), 0.0f, 1.0f));
        float _3348 = 6.283185482025146484375f * frac(asfloat(_3219.Load((_3224_g_params.hi + 5) * 4 + 0)) + sample_off.y);
        float3 sampled_dir = float3(_3345 * cos(_3348), _3345 * sin(_3348), _3315);
        float3 param;
        float3 param_1;
        create_tbn(_3338, param, param_1);
        float3 _9941 = sampled_dir;
        float3 _3381 = ((param * _9941.x) + (param_1 * _9941.y)) + (_3338 * _9941.z);
        sampled_dir = _3381;
        float3 _3390 = _3267.param1.xyz + (_3381 * _3267.param2.x);
        ls.L = _3390 - P;
        ls.dist = length(ls.L);
        ls.L /= ls.dist.xxx;
        ls.area = _3267.param1.w;
        float _3421 = abs(dot(ls.L, normalize(_3390 - _3267.param1.xyz)));
        [flatten]
        if (_3421 > 0.0f)
        {
            ls.pdf = (ls.dist * ls.dist) / ((0.5f * ls.area) * _3421);
        }
    }
    else
    {
        [branch]
        if (_3301 == 2u)
        {
            ls.L = _3267.param1.xyz;
            if (_3267.param1.w != 0.0f)
            {
                float param_2 = frac(asfloat(_3219.Load((_3224_g_params.hi + 4) * 4 + 0)) + sample_off.x);
                float param_3 = frac(asfloat(_3219.Load((_3224_g_params.hi + 5) * 4 + 0)) + sample_off.y);
                float3 param_4 = ls.L;
                float param_5 = tan(_3267.param1.w);
                ls.L = normalize(MapToCone(param_2, param_3, param_4, param_5));
            }
            ls.area = 0.0f;
            ls.dist = 3402823346297367662189621542912.0f;
            ls.pdf = 1.0f;
            if ((_3267.type_and_param0.x & 64u) == 0u)
            {
                ls.area = 0.0f;
            }
        }
        else
        {
            [branch]
            if (_3301 == 4u)
            {
                float3 _3552 = ((_3267.param1.xyz + (_3267.param2.xyz * (frac(asfloat(_3219.Load((_3224_g_params.hi + 4) * 4 + 0)) + sample_off.x) - 0.5f))) + (_3267.param3.xyz * (frac(asfloat(_3219.Load((_3224_g_params.hi + 5) * 4 + 0)) + sample_off.y) - 0.5f))) - P;
                ls.dist = length(_3552);
                ls.L = _3552 / ls.dist.xxx;
                ls.area = _3267.param1.w;
                float _3575 = dot(-ls.L, normalize(cross(_3267.param2.xyz, _3267.param3.xyz)));
                if (_3575 > 0.0f)
                {
                    ls.pdf = (ls.dist * ls.dist) / (ls.area * _3575);
                }
                if ((_3267.type_and_param0.x & 64u) == 0u)
                {
                    ls.area = 0.0f;
                }
                [branch]
                if ((_3267.type_and_param0.w & 128u) != 0u)
                {
                    float3 env_col = _3224_g_params.env_col.xyz;
                    uint _3614 = asuint(_3224_g_params.env_col.w);
                    if (_3614 != 4294967295u)
                    {
                        atlas_texture_t _3622;
                        _3622.size = _855.Load(_3614 * 80 + 0);
                        _3622.atlas = _855.Load(_3614 * 80 + 4);
                        [unroll]
                        for (int _63ident = 0; _63ident < 4; _63ident++)
                        {
                            _3622.page[_63ident] = _855.Load(_63ident * 4 + _3614 * 80 + 8);
                        }
                        [unroll]
                        for (int _64ident = 0; _64ident < 14; _64ident++)
                        {
                            _3622.pos[_64ident] = _855.Load(_64ident * 4 + _3614 * 80 + 24);
                        }
                        uint _9058[14] = { _3622.pos[0], _3622.pos[1], _3622.pos[2], _3622.pos[3], _3622.pos[4], _3622.pos[5], _3622.pos[6], _3622.pos[7], _3622.pos[8], _3622.pos[9], _3622.pos[10], _3622.pos[11], _3622.pos[12], _3622.pos[13] };
                        uint _9029[4] = { _3622.page[0], _3622.page[1], _3622.page[2], _3622.page[3] };
                        atlas_texture_t _8939 = { _3622.size, _3622.atlas, _9029, _9058 };
                        float param_6 = _3224_g_params.env_rotation;
                        env_col *= SampleLatlong_RGBE(_8939, ls.L, param_6);
                    }
                    ls.col *= env_col;
                }
            }
            else
            {
                [branch]
                if (_3301 == 5u)
                {
                    float2 _3725 = (float2(frac(asfloat(_3219.Load((_3224_g_params.hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_3219.Load((_3224_g_params.hi + 5) * 4 + 0)) + sample_off.y)) * 2.0f) - 1.0f.xx;
                    float2 offset = _3725;
                    bool _3728 = _3725.x != 0.0f;
                    bool _3734;
                    if (_3728)
                    {
                        _3734 = offset.y != 0.0f;
                    }
                    else
                    {
                        _3734 = _3728;
                    }
                    if (_3734)
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
                        float _3767 = 0.5f * r;
                        offset = float2(_3767 * cos(theta), _3767 * sin(theta));
                    }
                    float3 _3793 = ((_3267.param1.xyz + (_3267.param2.xyz * offset.x)) + (_3267.param3.xyz * offset.y)) - P;
                    ls.dist = length(_3793);
                    ls.L = _3793 / ls.dist.xxx;
                    ls.area = _3267.param1.w;
                    float _3816 = dot(-ls.L, normalize(cross(_3267.param2.xyz, _3267.param3.xyz)));
                    [flatten]
                    if (_3816 > 0.0f)
                    {
                        ls.pdf = (ls.dist * ls.dist) / (ls.area * _3816);
                    }
                    if ((_3267.type_and_param0.x & 64u) == 0u)
                    {
                        ls.area = 0.0f;
                    }
                    [branch]
                    if ((_3267.type_and_param0.w & 128u) != 0u)
                    {
                        float3 env_col_1 = _3224_g_params.env_col.xyz;
                        uint _3852 = asuint(_3224_g_params.env_col.w);
                        if (_3852 != 4294967295u)
                        {
                            atlas_texture_t _3859;
                            _3859.size = _855.Load(_3852 * 80 + 0);
                            _3859.atlas = _855.Load(_3852 * 80 + 4);
                            [unroll]
                            for (int _65ident = 0; _65ident < 4; _65ident++)
                            {
                                _3859.page[_65ident] = _855.Load(_65ident * 4 + _3852 * 80 + 8);
                            }
                            [unroll]
                            for (int _66ident = 0; _66ident < 14; _66ident++)
                            {
                                _3859.pos[_66ident] = _855.Load(_66ident * 4 + _3852 * 80 + 24);
                            }
                            uint _9096[14] = { _3859.pos[0], _3859.pos[1], _3859.pos[2], _3859.pos[3], _3859.pos[4], _3859.pos[5], _3859.pos[6], _3859.pos[7], _3859.pos[8], _3859.pos[9], _3859.pos[10], _3859.pos[11], _3859.pos[12], _3859.pos[13] };
                            uint _9067[4] = { _3859.page[0], _3859.page[1], _3859.page[2], _3859.page[3] };
                            atlas_texture_t _8948 = { _3859.size, _3859.atlas, _9067, _9096 };
                            float param_7 = _3224_g_params.env_rotation;
                            env_col_1 *= SampleLatlong_RGBE(_8948, ls.L, param_7);
                        }
                        ls.col *= env_col_1;
                    }
                }
                else
                {
                    [branch]
                    if (_3301 == 3u)
                    {
                        float3 _3960 = normalize(cross(P - _3267.param1.xyz, _3267.param3.xyz));
                        float _3967 = 3.1415927410125732421875f * frac(asfloat(_3219.Load((_3224_g_params.hi + 4) * 4 + 0)) + sample_off.x);
                        float3 _3996 = ((_3267.param1.xyz + (((_3960 * cos(_3967)) + (cross(_3960, _3267.param3.xyz) * sin(_3967))) * _3267.param2.w)) + ((_3267.param3.xyz * (frac(asfloat(_3219.Load((_3224_g_params.hi + 5) * 4 + 0)) + sample_off.y) - 0.5f)) * _3267.param3.w)) - P;
                        ls.dist = length(_3996);
                        ls.L = _3996 / ls.dist.xxx;
                        ls.area = _3267.param1.w;
                        float _4015 = 1.0f - abs(dot(ls.L, _3267.param3.xyz));
                        [flatten]
                        if (_4015 != 0.0f)
                        {
                            ls.pdf = (ls.dist * ls.dist) / (ls.area * _4015);
                        }
                        if ((_3267.type_and_param0.x & 64u) == 0u)
                        {
                            ls.area = 0.0f;
                        }
                        [branch]
                        if ((_3267.type_and_param0.w & 128u) != 0u)
                        {
                            float3 env_col_2 = _3224_g_params.env_col.xyz;
                            uint _4051 = asuint(_3224_g_params.env_col.w);
                            if (_4051 != 4294967295u)
                            {
                                atlas_texture_t _4058;
                                _4058.size = _855.Load(_4051 * 80 + 0);
                                _4058.atlas = _855.Load(_4051 * 80 + 4);
                                [unroll]
                                for (int _67ident = 0; _67ident < 4; _67ident++)
                                {
                                    _4058.page[_67ident] = _855.Load(_67ident * 4 + _4051 * 80 + 8);
                                }
                                [unroll]
                                for (int _68ident = 0; _68ident < 14; _68ident++)
                                {
                                    _4058.pos[_68ident] = _855.Load(_68ident * 4 + _4051 * 80 + 24);
                                }
                                uint _9134[14] = { _4058.pos[0], _4058.pos[1], _4058.pos[2], _4058.pos[3], _4058.pos[4], _4058.pos[5], _4058.pos[6], _4058.pos[7], _4058.pos[8], _4058.pos[9], _4058.pos[10], _4058.pos[11], _4058.pos[12], _4058.pos[13] };
                                uint _9105[4] = { _4058.page[0], _4058.page[1], _4058.page[2], _4058.page[3] };
                                atlas_texture_t _8957 = { _4058.size, _4058.atlas, _9105, _9134 };
                                float param_8 = _3224_g_params.env_rotation;
                                env_col_2 *= SampleLatlong_RGBE(_8957, ls.L, param_8);
                            }
                            ls.col *= env_col_2;
                        }
                    }
                    else
                    {
                        [branch]
                        if (_3301 == 6u)
                        {
                            uint _4127 = asuint(_3267.param1.x);
                            transform_t _4141;
                            _4141.xform = asfloat(uint4x4(_4135.Load4(asuint(_3267.param1.y) * 128 + 0), _4135.Load4(asuint(_3267.param1.y) * 128 + 16), _4135.Load4(asuint(_3267.param1.y) * 128 + 32), _4135.Load4(asuint(_3267.param1.y) * 128 + 48)));
                            _4141.inv_xform = asfloat(uint4x4(_4135.Load4(asuint(_3267.param1.y) * 128 + 64), _4135.Load4(asuint(_3267.param1.y) * 128 + 80), _4135.Load4(asuint(_3267.param1.y) * 128 + 96), _4135.Load4(asuint(_3267.param1.y) * 128 + 112)));
                            uint _4166 = _4127 * 3u;
                            vertex_t _4172;
                            [unroll]
                            for (int _69ident = 0; _69ident < 3; _69ident++)
                            {
                                _4172.p[_69ident] = asfloat(_4160.Load(_69ident * 4 + _4164.Load(_4166 * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _70ident = 0; _70ident < 3; _70ident++)
                            {
                                _4172.n[_70ident] = asfloat(_4160.Load(_70ident * 4 + _4164.Load(_4166 * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _71ident = 0; _71ident < 3; _71ident++)
                            {
                                _4172.b[_71ident] = asfloat(_4160.Load(_71ident * 4 + _4164.Load(_4166 * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _72ident = 0; _72ident < 2; _72ident++)
                            {
                                [unroll]
                                for (int _73ident = 0; _73ident < 2; _73ident++)
                                {
                                    _4172.t[_72ident][_73ident] = asfloat(_4160.Load(_73ident * 4 + _72ident * 8 + _4164.Load(_4166 * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4221;
                            [unroll]
                            for (int _74ident = 0; _74ident < 3; _74ident++)
                            {
                                _4221.p[_74ident] = asfloat(_4160.Load(_74ident * 4 + _4164.Load((_4166 + 1u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _75ident = 0; _75ident < 3; _75ident++)
                            {
                                _4221.n[_75ident] = asfloat(_4160.Load(_75ident * 4 + _4164.Load((_4166 + 1u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _76ident = 0; _76ident < 3; _76ident++)
                            {
                                _4221.b[_76ident] = asfloat(_4160.Load(_76ident * 4 + _4164.Load((_4166 + 1u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _77ident = 0; _77ident < 2; _77ident++)
                            {
                                [unroll]
                                for (int _78ident = 0; _78ident < 2; _78ident++)
                                {
                                    _4221.t[_77ident][_78ident] = asfloat(_4160.Load(_78ident * 4 + _77ident * 8 + _4164.Load((_4166 + 1u) * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4267;
                            [unroll]
                            for (int _79ident = 0; _79ident < 3; _79ident++)
                            {
                                _4267.p[_79ident] = asfloat(_4160.Load(_79ident * 4 + _4164.Load((_4166 + 2u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _80ident = 0; _80ident < 3; _80ident++)
                            {
                                _4267.n[_80ident] = asfloat(_4160.Load(_80ident * 4 + _4164.Load((_4166 + 2u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _81ident = 0; _81ident < 3; _81ident++)
                            {
                                _4267.b[_81ident] = asfloat(_4160.Load(_81ident * 4 + _4164.Load((_4166 + 2u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _82ident = 0; _82ident < 2; _82ident++)
                            {
                                [unroll]
                                for (int _83ident = 0; _83ident < 2; _83ident++)
                                {
                                    _4267.t[_82ident][_83ident] = asfloat(_4160.Load(_83ident * 4 + _82ident * 8 + _4164.Load((_4166 + 2u) * 4 + 0) * 52 + 36));
                                }
                            }
                            float3 _4313 = float3(_4172.p[0], _4172.p[1], _4172.p[2]);
                            float3 _4321 = float3(_4221.p[0], _4221.p[1], _4221.p[2]);
                            float3 _4329 = float3(_4267.p[0], _4267.p[1], _4267.p[2]);
                            float _4358 = sqrt(frac(asfloat(_3219.Load((_3224_g_params.hi + 4) * 4 + 0)) + sample_off.x));
                            float _4368 = frac(asfloat(_3219.Load((_3224_g_params.hi + 5) * 4 + 0)) + sample_off.y);
                            float _4372 = 1.0f - _4358;
                            float _4377 = 1.0f - _4368;
                            float3 _4424 = mul(float4(cross(_4321 - _4313, _4329 - _4313), 0.0f), _4141.xform).xyz;
                            ls.area = 0.5f * length(_4424);
                            float3 _4434 = mul(float4((_4313 * _4372) + (((_4321 * _4377) + (_4329 * _4368)) * _4358), 1.0f), _4141.xform).xyz - P;
                            ls.dist = length(_4434);
                            ls.L = _4434 / ls.dist.xxx;
                            float _4449 = abs(dot(ls.L, normalize(_4424)));
                            [flatten]
                            if (_4449 > 0.0f)
                            {
                                ls.pdf = (ls.dist * ls.dist) / (ls.area * _4449);
                            }
                            material_t _4488;
                            [unroll]
                            for (int _84ident = 0; _84ident < 5; _84ident++)
                            {
                                _4488.textures[_84ident] = _4475.Load(_84ident * 4 + ((_4479.Load(_4127 * 4 + 0) >> 16u) & 16383u) * 80 + 0);
                            }
                            [unroll]
                            for (int _85ident = 0; _85ident < 3; _85ident++)
                            {
                                _4488.base_color[_85ident] = asfloat(_4475.Load(_85ident * 4 + ((_4479.Load(_4127 * 4 + 0) >> 16u) & 16383u) * 80 + 20));
                            }
                            _4488.flags = _4475.Load(((_4479.Load(_4127 * 4 + 0) >> 16u) & 16383u) * 80 + 32);
                            _4488.type = _4475.Load(((_4479.Load(_4127 * 4 + 0) >> 16u) & 16383u) * 80 + 36);
                            _4488.tangent_rotation_or_strength = asfloat(_4475.Load(((_4479.Load(_4127 * 4 + 0) >> 16u) & 16383u) * 80 + 40));
                            _4488.roughness_and_anisotropic = _4475.Load(((_4479.Load(_4127 * 4 + 0) >> 16u) & 16383u) * 80 + 44);
                            _4488.int_ior = asfloat(_4475.Load(((_4479.Load(_4127 * 4 + 0) >> 16u) & 16383u) * 80 + 48));
                            _4488.ext_ior = asfloat(_4475.Load(((_4479.Load(_4127 * 4 + 0) >> 16u) & 16383u) * 80 + 52));
                            _4488.sheen_and_sheen_tint = _4475.Load(((_4479.Load(_4127 * 4 + 0) >> 16u) & 16383u) * 80 + 56);
                            _4488.tint_and_metallic = _4475.Load(((_4479.Load(_4127 * 4 + 0) >> 16u) & 16383u) * 80 + 60);
                            _4488.transmission_and_transmission_roughness = _4475.Load(((_4479.Load(_4127 * 4 + 0) >> 16u) & 16383u) * 80 + 64);
                            _4488.specular_and_specular_tint = _4475.Load(((_4479.Load(_4127 * 4 + 0) >> 16u) & 16383u) * 80 + 68);
                            _4488.clearcoat_and_clearcoat_roughness = _4475.Load(((_4479.Load(_4127 * 4 + 0) >> 16u) & 16383u) * 80 + 72);
                            _4488.normal_map_strength_unorm = _4475.Load(((_4479.Load(_4127 * 4 + 0) >> 16u) & 16383u) * 80 + 76);
                            if ((_4488.flags & 4u) == 0u)
                            {
                                if (_4488.textures[1] != 4294967295u)
                                {
                                    ls.col *= SampleBilinear(_4488.textures[1], (float2(_4172.t[0][0], _4172.t[0][1]) * _4372) + (((float2(_4221.t[0][0], _4221.t[0][1]) * _4377) + (float2(_4267.t[0][0], _4267.t[0][1]) * _4368)) * _4358), 0).xyz;
                                }
                            }
                            else
                            {
                                float3 env_col_3 = _3224_g_params.env_col.xyz;
                                uint _4562 = asuint(_3224_g_params.env_col.w);
                                if (_4562 != 4294967295u)
                                {
                                    atlas_texture_t _4569;
                                    _4569.size = _855.Load(_4562 * 80 + 0);
                                    _4569.atlas = _855.Load(_4562 * 80 + 4);
                                    [unroll]
                                    for (int _86ident = 0; _86ident < 4; _86ident++)
                                    {
                                        _4569.page[_86ident] = _855.Load(_86ident * 4 + _4562 * 80 + 8);
                                    }
                                    [unroll]
                                    for (int _87ident = 0; _87ident < 14; _87ident++)
                                    {
                                        _4569.pos[_87ident] = _855.Load(_87ident * 4 + _4562 * 80 + 24);
                                    }
                                    uint _9219[14] = { _4569.pos[0], _4569.pos[1], _4569.pos[2], _4569.pos[3], _4569.pos[4], _4569.pos[5], _4569.pos[6], _4569.pos[7], _4569.pos[8], _4569.pos[9], _4569.pos[10], _4569.pos[11], _4569.pos[12], _4569.pos[13] };
                                    uint _9190[4] = { _4569.page[0], _4569.page[1], _4569.page[2], _4569.page[3] };
                                    atlas_texture_t _9011 = { _4569.size, _4569.atlas, _9190, _9219 };
                                    float param_9 = _3224_g_params.env_rotation;
                                    env_col_3 *= SampleLatlong_RGBE(_9011, ls.L, param_9);
                                }
                                ls.col *= env_col_3;
                            }
                        }
                        else
                        {
                            [branch]
                            if (_3301 == 7u)
                            {
                                float4 _4672 = Sample_EnvQTree(_3224_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3224_g_params.env_qtree_levels, mad(_3235, _3240, -float(_3247)), frac(asfloat(_3219.Load((_3224_g_params.hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_3219.Load((_3224_g_params.hi + 5) * 4 + 0)) + sample_off.y));
                                ls.L = _4672.xyz;
                                ls.col *= _3224_g_params.env_col.xyz;
                                atlas_texture_t _4689;
                                _4689.size = _855.Load(asuint(_3224_g_params.env_col.w) * 80 + 0);
                                _4689.atlas = _855.Load(asuint(_3224_g_params.env_col.w) * 80 + 4);
                                [unroll]
                                for (int _88ident = 0; _88ident < 4; _88ident++)
                                {
                                    _4689.page[_88ident] = _855.Load(_88ident * 4 + asuint(_3224_g_params.env_col.w) * 80 + 8);
                                }
                                [unroll]
                                for (int _89ident = 0; _89ident < 14; _89ident++)
                                {
                                    _4689.pos[_89ident] = _855.Load(_89ident * 4 + asuint(_3224_g_params.env_col.w) * 80 + 24);
                                }
                                uint _9257[14] = { _4689.pos[0], _4689.pos[1], _4689.pos[2], _4689.pos[3], _4689.pos[4], _4689.pos[5], _4689.pos[6], _4689.pos[7], _4689.pos[8], _4689.pos[9], _4689.pos[10], _4689.pos[11], _4689.pos[12], _4689.pos[13] };
                                uint _9228[4] = { _4689.page[0], _4689.page[1], _4689.page[2], _4689.page[3] };
                                atlas_texture_t _9020 = { _4689.size, _4689.atlas, _9228, _9257 };
                                float param_10 = _3224_g_params.env_rotation;
                                ls.col *= SampleLatlong_RGBE(_9020, ls.L, param_10);
                                ls.area = 1.0f;
                                ls.dist = 3402823346297367662189621542912.0f;
                                ls.pdf = _4672.w;
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
    atlas_texture_t _858;
    _858.size = _855.Load(index * 80 + 0);
    _858.atlas = _855.Load(index * 80 + 4);
    [unroll]
    for (int _90ident = 0; _90ident < 4; _90ident++)
    {
        _858.page[_90ident] = _855.Load(_90ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _91ident = 0; _91ident < 14; _91ident++)
    {
        _858.pos[_91ident] = _855.Load(_91ident * 4 + index * 80 + 24);
    }
    return int2(int(_858.size & 16383u), int((_858.size >> uint(16)) & 16383u));
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
    float _2215 = 1.0f / mad(0.904129683971405029296875f, roughness, 3.1415927410125732421875f);
    float _2227 = max(dot(N, L), 0.0f);
    float _2232 = max(dot(N, V), 0.0f);
    float _2240 = mad(-_2227, _2232, dot(L, V));
    float t = _2240;
    if (_2240 > 0.0f)
    {
        t /= (max(_2227, _2232) + 1.1754943508222875079687365372222e-38f);
    }
    return float4(base_color * (_2227 * mad(roughness * _2215, t, _2215)), 0.15915493667125701904296875f);
}

float3 offset_ray(float3 p, float3 n)
{
    int3 _1797 = int3(n * 128.0f);
    int _1805;
    if (p.x < 0.0f)
    {
        _1805 = -_1797.x;
    }
    else
    {
        _1805 = _1797.x;
    }
    int _1823;
    if (p.y < 0.0f)
    {
        _1823 = -_1797.y;
    }
    else
    {
        _1823 = _1797.y;
    }
    int _1841;
    if (p.z < 0.0f)
    {
        _1841 = -_1797.z;
    }
    else
    {
        _1841 = _1797.z;
    }
    float _1859;
    if (abs(p.x) < 0.03125f)
    {
        _1859 = mad(1.52587890625e-05f, n.x, p.x);
    }
    else
    {
        _1859 = asfloat(asint(p.x) + _1805);
    }
    float _1877;
    if (abs(p.y) < 0.03125f)
    {
        _1877 = mad(1.52587890625e-05f, n.y, p.y);
    }
    else
    {
        _1877 = asfloat(asint(p.y) + _1823);
    }
    float _1894;
    if (abs(p.z) < 0.03125f)
    {
        _1894 = mad(1.52587890625e-05f, n.z, p.z);
    }
    else
    {
        _1894 = asfloat(asint(p.z) + _1841);
    }
    return float3(_1859, _1877, _1894);
}

float3 world_from_tangent(float3 T, float3 B, float3 N, float3 V)
{
    return ((T * V.x) + (B * V.y)) + (N * V.z);
}

float4 Sample_OrenDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float rand_u, float rand_v, inout float3 out_V)
{
    float _2274 = 6.283185482025146484375f * rand_v;
    float _2286 = sqrt(mad(-rand_u, rand_u, 1.0f));
    float3 param = T;
    float3 param_1 = B;
    float3 param_2 = N;
    float3 param_3 = float3(_2286 * cos(_2274), _2286 * sin(_2274), rand_u);
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
    float _8773;
    do
    {
        if (H.z == 0.0f)
        {
            _8773 = 0.0f;
            break;
        }
        float _2101 = (-H.x) / (H.z * alpha_x);
        float _2107 = (-H.y) / (H.z * alpha_y);
        float _2116 = mad(_2107, _2107, mad(_2101, _2101, 1.0f));
        _8773 = 1.0f / (((((_2116 * _2116) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _8773;
}

float G1(float3 Ve, inout float alpha_x, inout float alpha_y)
{
    alpha_x *= alpha_x;
    alpha_y *= alpha_y;
    return 1.0f / mad((-1.0f) + sqrt(1.0f + (mad(alpha_x * Ve.x, Ve.x, (alpha_y * Ve.y) * Ve.y) / (Ve.z * Ve.z))), 0.5f, 1.0f);
}

float4 Evaluate_GGXSpecular_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float alpha_x, float alpha_y, float spec_ior, float spec_F0, float3 spec_col)
{
    float _2456 = D_GGX(sampled_normal_ts, alpha_x, alpha_y);
    float3 param = view_dir_ts;
    float param_1 = alpha_x;
    float param_2 = alpha_y;
    float _2464 = G1(param, param_1, param_2);
    float3 param_3 = reflected_dir_ts;
    float param_4 = alpha_x;
    float param_5 = alpha_y;
    float _2471 = G1(param_3, param_4, param_5);
    float param_6 = dot(view_dir_ts, sampled_normal_ts);
    float param_7 = spec_ior;
    float3 F = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param_6, param_7) - spec_F0) / (1.0f - spec_F0)).xxx);
    float _2499 = 4.0f * abs(view_dir_ts.z * reflected_dir_ts.z);
    float _2502;
    if (_2499 != 0.0f)
    {
        _2502 = (_2456 * (_2464 * _2471)) / _2499;
    }
    else
    {
        _2502 = 0.0f;
    }
    F *= _2502;
    float3 param_8 = view_dir_ts;
    float param_9 = alpha_x;
    float param_10 = alpha_y;
    float _2522 = G1(param_8, param_9, param_10);
    float pdf = ((_2456 * _2522) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2537 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2537 != 0.0f)
    {
        pdf /= _2537;
    }
    float3 _2548 = F;
    float3 _2549 = _2548 * max(reflected_dir_ts.z, 0.0f);
    F = _2549;
    return float4(_2549, pdf);
}

float3 SampleGGX_VNDF(float3 Ve, float alpha_x, float alpha_y, float U1, float U2)
{
    float3 _1919 = normalize(float3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
    float _1922 = _1919.x;
    float _1927 = _1919.y;
    float _1931 = mad(_1922, _1922, _1927 * _1927);
    float3 _1935;
    if (_1931 > 0.0f)
    {
        _1935 = float3(-_1927, _1922, 0.0f) / sqrt(_1931).xxx;
    }
    else
    {
        _1935 = float3(1.0f, 0.0f, 0.0f);
    }
    float _1957 = sqrt(U1);
    float _1960 = 6.283185482025146484375f * U2;
    float _1965 = _1957 * cos(_1960);
    float _1974 = 1.0f + _1919.z;
    float _1981 = mad(-_1965, _1965, 1.0f);
    float _1987 = mad(mad(-0.5f, _1974, 1.0f), sqrt(_1981), (0.5f * _1974) * (_1957 * sin(_1960)));
    float3 _2008 = ((_1935 * _1965) + (cross(_1919, _1935) * _1987)) + (_1919 * sqrt(max(0.0f, mad(-_1987, _1987, _1981))));
    return normalize(float3(alpha_x * _2008.x, alpha_y * _2008.y, max(0.0f, _2008.z)));
}

float4 Sample_GGXSpecular_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float anisotropic, float spec_ior, float spec_F0, float3 spec_col, float rand_u, float rand_v, inout float3 out_V)
{
    float4 _8748;
    do
    {
        float _2559 = roughness * roughness;
        float _2563 = sqrt(mad(-0.89999997615814208984375f, anisotropic, 1.0f));
        float _2567 = _2559 / _2563;
        float _2571 = _2559 * _2563;
        [branch]
        if ((_2567 * _2571) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2581 = reflect(I, N);
            float param = dot(_2581, N);
            float param_1 = spec_ior;
            float3 _2595 = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param, param_1) - spec_F0) / (1.0f - spec_F0)).xxx);
            out_V = _2581;
            _8748 = float4(_2595.x * 1000000.0f, _2595.y * 1000000.0f, _2595.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2620 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = _2567;
        float param_7 = _2571;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2629 = SampleGGX_VNDF(_2620, param_6, param_7, param_8, param_9);
        float3 _2640 = normalize(reflect(-_2620, _2629));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2640;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2620;
        float3 param_15 = _2629;
        float3 param_16 = _2640;
        float param_17 = _2567;
        float param_18 = _2571;
        float param_19 = spec_ior;
        float param_20 = spec_F0;
        float3 param_21 = spec_col;
        _8748 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _8748;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _8753;
    do
    {
        bool _2862 = refr_dir_ts.z >= 0.0f;
        bool _2869;
        if (!_2862)
        {
            _2869 = view_dir_ts.z <= 0.0f;
        }
        else
        {
            _2869 = _2862;
        }
        if (_2869)
        {
            _8753 = 0.0f.xxxx;
            break;
        }
        float _2878 = D_GGX(sampled_normal_ts, roughness2, roughness2);
        float3 param = refr_dir_ts;
        float param_1 = roughness2;
        float param_2 = roughness2;
        float _2886 = G1(param, param_1, param_2);
        float3 param_3 = view_dir_ts;
        float param_4 = roughness2;
        float param_5 = roughness2;
        float _2894 = G1(param_3, param_4, param_5);
        float _2904 = mad(dot(view_dir_ts, sampled_normal_ts), eta, dot(refr_dir_ts, sampled_normal_ts));
        float _2914 = clamp(-dot(refr_dir_ts, sampled_normal_ts), 0.0f, 1.0f) / (_2904 * _2904);
        _8753 = float4(refr_col * (((((_2878 * _2894) * _2886) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _2914) / view_dir_ts.z), (((_2878 * _2886) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _2914) / view_dir_ts.z);
        break;
    } while(false);
    return _8753;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _8758;
    do
    {
        float _2958 = roughness * roughness;
        [branch]
        if ((_2958 * _2958) < 1.0000000116860974230803549289703e-07f)
        {
            float _2968 = dot(I, N);
            float _2969 = -_2968;
            float _2979 = mad(-(eta * eta), mad(_2968, _2969, 1.0f), 1.0f);
            if (_2979 < 0.0f)
            {
                _8758 = 0.0f.xxxx;
                break;
            }
            float _2991 = mad(eta, _2969, -sqrt(_2979));
            out_V = float4(normalize((I * eta) + (N * _2991)), _2991);
            _8758 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param = T;
        float3 param_1 = B;
        float3 param_2 = N;
        float3 param_3 = -I;
        float3 _3031 = normalize(tangent_from_world(param, param_1, param_2, param_3));
        float param_4 = _2958;
        float param_5 = _2958;
        float param_6 = rand_u;
        float param_7 = rand_v;
        float3 _3042 = SampleGGX_VNDF(_3031, param_4, param_5, param_6, param_7);
        float _3046 = dot(_3031, _3042);
        float _3056 = mad(-(eta * eta), mad(-_3046, _3046, 1.0f), 1.0f);
        if (_3056 < 0.0f)
        {
            _8758 = 0.0f.xxxx;
            break;
        }
        float _3068 = mad(eta, _3046, -sqrt(_3056));
        float3 _3078 = normalize((_3031 * (-eta)) + (_3042 * _3068));
        float3 param_8 = _3031;
        float3 param_9 = _3042;
        float3 param_10 = _3078;
        float param_11 = _2958;
        float param_12 = eta;
        float3 param_13 = refr_col;
        float3 param_14 = T;
        float3 param_15 = B;
        float3 param_16 = N;
        float3 param_17 = _3078;
        out_V = float4(world_from_tangent(param_14, param_15, param_16, param_17), _3068);
        _8758 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _8758;
}

void get_lobe_weights(float base_color_lum, float spec_color_lum, float specular, float metallic, float transmission, float clearcoat, inout float out_diffuse_weight, inout float out_specular_weight, inout float out_clearcoat_weight, inout float out_refraction_weight)
{
    float _1591 = 1.0f - metallic;
    out_diffuse_weight = (base_color_lum * _1591) * (1.0f - transmission);
    float _1601;
    if ((specular != 0.0f) || (metallic != 0.0f))
    {
        _1601 = spec_color_lum * mad(-transmission, _1591, 1.0f);
    }
    else
    {
        _1601 = 0.0f;
    }
    out_specular_weight = _1601;
    out_clearcoat_weight = (0.25f * clearcoat) * _1591;
    out_refraction_weight = (transmission * _1591) * base_color_lum;
    float _1616 = out_diffuse_weight;
    float _1617 = out_specular_weight;
    float _1619 = out_clearcoat_weight;
    float _1622 = ((_1616 + _1617) + _1619) + out_refraction_weight;
    if (_1622 != 0.0f)
    {
        out_diffuse_weight /= _1622;
        out_specular_weight /= _1622;
        out_clearcoat_weight /= _1622;
        out_refraction_weight /= _1622;
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
    float _8778;
    do
    {
        float _2167 = dot(N, L);
        if (_2167 <= 0.0f)
        {
            _8778 = 0.0f;
            break;
        }
        float param = _2167;
        float param_1 = dot(N, V);
        float _2188 = dot(L, H);
        float _2196 = mad((2.0f * _2188) * _2188, roughness, 0.5f);
        _8778 = lerp(1.0f, _2196, schlick_weight(param)) * lerp(1.0f, _2196, schlick_weight(param_1));
        break;
    } while(false);
    return _8778;
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
    float3 _2337 = normalize(L + V);
    float3 H = _2337;
    if (dot(V, _2337) < 0.0f)
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
    float3 _2372 = diff_col;
    float3 _2373 = _2372 + (sheen_color * (3.1415927410125732421875f * schlick_weight(param_5)));
    diff_col = _2373;
    return float4(_2373, pdf);
}

float D_GTR1(float NDotH, float a)
{
    float _8783;
    do
    {
        if (a >= 1.0f)
        {
            _8783 = 0.3183098733425140380859375f;
            break;
        }
        float _2075 = mad(a, a, -1.0f);
        _8783 = _2075 / ((3.1415927410125732421875f * log(a * a)) * mad(_2075 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _8783;
}

float4 Evaluate_PrincipledClearcoat_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float clearcoat_roughness2, float clearcoat_ior, float clearcoat_F0)
{
    float param = sampled_normal_ts.z;
    float param_1 = clearcoat_roughness2;
    float _2672 = D_GTR1(param, param_1);
    float3 param_2 = view_dir_ts;
    float param_3 = 0.0625f;
    float param_4 = 0.0625f;
    float _2679 = G1(param_2, param_3, param_4);
    float3 param_5 = reflected_dir_ts;
    float param_6 = 0.0625f;
    float param_7 = 0.0625f;
    float _2684 = G1(param_5, param_6, param_7);
    float param_8 = dot(reflected_dir_ts, sampled_normal_ts);
    float param_9 = clearcoat_ior;
    float F = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param_8, param_9) - clearcoat_F0) / (1.0f - clearcoat_F0));
    float _2711 = (4.0f * abs(view_dir_ts.z)) * abs(reflected_dir_ts.z);
    float _2714;
    if (_2711 != 0.0f)
    {
        _2714 = (_2672 * (_2679 * _2684)) / _2711;
    }
    else
    {
        _2714 = 0.0f;
    }
    F *= _2714;
    float3 param_10 = view_dir_ts;
    float param_11 = 0.0625f;
    float param_12 = 0.0625f;
    float _2732 = G1(param_10, param_11, param_12);
    float pdf = ((_2672 * _2732) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2747 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2747 != 0.0f)
    {
        pdf /= _2747;
    }
    float _2758 = F;
    float _2759 = _2758 * clamp(reflected_dir_ts.z, 0.0f, 1.0f);
    F = _2759;
    return float4(_2759, _2759, _2759, pdf);
}

float4 Sample_PrincipledDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float3 sheen_color, bool uniform_sampling, float rand_u, float rand_v, inout float3 out_V)
{
    float _2384 = 6.283185482025146484375f * rand_v;
    float _2387 = cos(_2384);
    float _2390 = sin(_2384);
    float3 V;
    if (uniform_sampling)
    {
        float _2399 = sqrt(mad(-rand_u, rand_u, 1.0f));
        V = float3(_2399 * _2387, _2399 * _2390, rand_u);
    }
    else
    {
        float _2412 = sqrt(rand_u);
        V = float3(_2412 * _2387, _2412 * _2390, sqrt(1.0f - rand_u));
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
    float4 _8763;
    do
    {
        [branch]
        if ((clearcoat_roughness2 * clearcoat_roughness2) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2776 = reflect(I, N);
            float param = dot(_2776, N);
            float param_1 = clearcoat_ior;
            out_V = _2776;
            float _2795 = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param, param_1) - clearcoat_F0) / (1.0f - clearcoat_F0)) * 1000000.0f;
            _8763 = float4(_2795, _2795, _2795, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2813 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = clearcoat_roughness2;
        float param_7 = clearcoat_roughness2;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2824 = SampleGGX_VNDF(_2813, param_6, param_7, param_8, param_9);
        float3 _2835 = normalize(reflect(-_2813, _2824));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2835;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2813;
        float3 param_15 = _2824;
        float3 param_16 = _2835;
        float param_17 = clearcoat_roughness2;
        float param_18 = clearcoat_ior;
        float param_19 = clearcoat_F0;
        _8763 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _8763;
}

float3 ShadeSurface(int px_index, hit_data_t inter, ray_data_t ray)
{
    float3 _8738;
    do
    {
        float3 _4766 = float3(ray.d[0], ray.d[1], ray.d[2]);
        [branch]
        if (inter.mask == 0)
        {
            float3 env_col = _3224_g_params.env_col.xyz;
            uint _4779 = asuint(_3224_g_params.env_col.w);
            if (_4779 != 4294967295u)
            {
                atlas_texture_t _4786;
                _4786.size = _855.Load(_4779 * 80 + 0);
                _4786.atlas = _855.Load(_4779 * 80 + 4);
                [unroll]
                for (int _92ident = 0; _92ident < 4; _92ident++)
                {
                    _4786.page[_92ident] = _855.Load(_92ident * 4 + _4779 * 80 + 8);
                }
                [unroll]
                for (int _93ident = 0; _93ident < 14; _93ident++)
                {
                    _4786.pos[_93ident] = _855.Load(_93ident * 4 + _4779 * 80 + 24);
                }
                uint _9640[14] = { _4786.pos[0], _4786.pos[1], _4786.pos[2], _4786.pos[3], _4786.pos[4], _4786.pos[5], _4786.pos[6], _4786.pos[7], _4786.pos[8], _4786.pos[9], _4786.pos[10], _4786.pos[11], _4786.pos[12], _4786.pos[13] };
                uint _9611[4] = { _4786.page[0], _4786.page[1], _4786.page[2], _4786.page[3] };
                atlas_texture_t _9278 = { _4786.size, _4786.atlas, _9611, _9640 };
                float param = _3224_g_params.env_rotation;
                env_col *= SampleLatlong_RGBE(_9278, _4766, param);
                if (_3224_g_params.env_qtree_levels > 0)
                {
                    float param_1 = ray.pdf;
                    float param_2 = Evaluate_EnvQTree(_3224_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3224_g_params.env_qtree_levels, _4766);
                    env_col *= power_heuristic(param_1, param_2);
                }
            }
            _8738 = float3(ray.c[0] * env_col.x, ray.c[1] * env_col.y, ray.c[2] * env_col.z);
            break;
        }
        float3 _4889 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_4766 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            light_t _4901;
            _4901.type_and_param0 = _3256.Load4(((-1) - inter.obj_index) * 64 + 0);
            _4901.param1 = asfloat(_3256.Load4(((-1) - inter.obj_index) * 64 + 16));
            _4901.param2 = asfloat(_3256.Load4(((-1) - inter.obj_index) * 64 + 32));
            _4901.param3 = asfloat(_3256.Load4(((-1) - inter.obj_index) * 64 + 48));
            float3 lcol = asfloat(_4901.type_and_param0.yzw);
            uint _4918 = _4901.type_and_param0.x & 31u;
            if (_4918 == 0u)
            {
                float param_3 = ray.pdf;
                float param_4 = (inter.t * inter.t) / ((0.5f * _4901.param1.w) * dot(_4766, normalize(_4901.param1.xyz - _4889)));
                lcol *= power_heuristic(param_3, param_4);
            }
            else
            {
                if (_4918 == 4u)
                {
                    float param_5 = ray.pdf;
                    float param_6 = (inter.t * inter.t) / (_4901.param1.w * dot(_4766, normalize(cross(_4901.param2.xyz, _4901.param3.xyz))));
                    lcol *= power_heuristic(param_5, param_6);
                }
                else
                {
                    if (_4918 == 5u)
                    {
                        float param_7 = ray.pdf;
                        float param_8 = (inter.t * inter.t) / (_4901.param1.w * dot(_4766, normalize(cross(_4901.param2.xyz, _4901.param3.xyz))));
                        lcol *= power_heuristic(param_7, param_8);
                    }
                    else
                    {
                        if (_4918 == 3u)
                        {
                            float param_9 = ray.pdf;
                            float param_10 = (inter.t * inter.t) / (_4901.param1.w * (1.0f - abs(dot(_4766, _4901.param3.xyz))));
                            lcol *= power_heuristic(param_9, param_10);
                        }
                    }
                }
            }
            _8738 = float3(ray.c[0] * lcol.x, ray.c[1] * lcol.y, ray.c[2] * lcol.z);
            break;
        }
        bool _5153 = inter.prim_index < 0;
        int _5156;
        if (_5153)
        {
            _5156 = (-1) - inter.prim_index;
        }
        else
        {
            _5156 = inter.prim_index;
        }
        uint _5167 = uint(_5156);
        material_t _5175;
        [unroll]
        for (int _94ident = 0; _94ident < 5; _94ident++)
        {
            _5175.textures[_94ident] = _4475.Load(_94ident * 4 + ((_4479.Load(_5167 * 4 + 0) >> 16u) & 16383u) * 80 + 0);
        }
        [unroll]
        for (int _95ident = 0; _95ident < 3; _95ident++)
        {
            _5175.base_color[_95ident] = asfloat(_4475.Load(_95ident * 4 + ((_4479.Load(_5167 * 4 + 0) >> 16u) & 16383u) * 80 + 20));
        }
        _5175.flags = _4475.Load(((_4479.Load(_5167 * 4 + 0) >> 16u) & 16383u) * 80 + 32);
        _5175.type = _4475.Load(((_4479.Load(_5167 * 4 + 0) >> 16u) & 16383u) * 80 + 36);
        _5175.tangent_rotation_or_strength = asfloat(_4475.Load(((_4479.Load(_5167 * 4 + 0) >> 16u) & 16383u) * 80 + 40));
        _5175.roughness_and_anisotropic = _4475.Load(((_4479.Load(_5167 * 4 + 0) >> 16u) & 16383u) * 80 + 44);
        _5175.int_ior = asfloat(_4475.Load(((_4479.Load(_5167 * 4 + 0) >> 16u) & 16383u) * 80 + 48));
        _5175.ext_ior = asfloat(_4475.Load(((_4479.Load(_5167 * 4 + 0) >> 16u) & 16383u) * 80 + 52));
        _5175.sheen_and_sheen_tint = _4475.Load(((_4479.Load(_5167 * 4 + 0) >> 16u) & 16383u) * 80 + 56);
        _5175.tint_and_metallic = _4475.Load(((_4479.Load(_5167 * 4 + 0) >> 16u) & 16383u) * 80 + 60);
        _5175.transmission_and_transmission_roughness = _4475.Load(((_4479.Load(_5167 * 4 + 0) >> 16u) & 16383u) * 80 + 64);
        _5175.specular_and_specular_tint = _4475.Load(((_4479.Load(_5167 * 4 + 0) >> 16u) & 16383u) * 80 + 68);
        _5175.clearcoat_and_clearcoat_roughness = _4475.Load(((_4479.Load(_5167 * 4 + 0) >> 16u) & 16383u) * 80 + 72);
        _5175.normal_map_strength_unorm = _4475.Load(((_4479.Load(_5167 * 4 + 0) >> 16u) & 16383u) * 80 + 76);
        uint _9641 = _5175.textures[0];
        uint _9642 = _5175.textures[1];
        uint _9643 = _5175.textures[2];
        uint _9644 = _5175.textures[3];
        uint _9645 = _5175.textures[4];
        float _9646 = _5175.base_color[0];
        float _9647 = _5175.base_color[1];
        float _9648 = _5175.base_color[2];
        uint _9290 = _5175.flags;
        uint _9291 = _5175.type;
        float _9292 = _5175.tangent_rotation_or_strength;
        uint _9293 = _5175.roughness_and_anisotropic;
        float _9294 = _5175.int_ior;
        float _9295 = _5175.ext_ior;
        uint _9296 = _5175.sheen_and_sheen_tint;
        uint _9297 = _5175.tint_and_metallic;
        uint _9298 = _5175.transmission_and_transmission_roughness;
        uint _9299 = _5175.specular_and_specular_tint;
        uint _9300 = _5175.clearcoat_and_clearcoat_roughness;
        uint _9301 = _5175.normal_map_strength_unorm;
        transform_t _5232;
        _5232.xform = asfloat(uint4x4(_4135.Load4(asuint(asfloat(_5225.Load(inter.obj_index * 32 + 12))) * 128 + 0), _4135.Load4(asuint(asfloat(_5225.Load(inter.obj_index * 32 + 12))) * 128 + 16), _4135.Load4(asuint(asfloat(_5225.Load(inter.obj_index * 32 + 12))) * 128 + 32), _4135.Load4(asuint(asfloat(_5225.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _5232.inv_xform = asfloat(uint4x4(_4135.Load4(asuint(asfloat(_5225.Load(inter.obj_index * 32 + 12))) * 128 + 64), _4135.Load4(asuint(asfloat(_5225.Load(inter.obj_index * 32 + 12))) * 128 + 80), _4135.Load4(asuint(asfloat(_5225.Load(inter.obj_index * 32 + 12))) * 128 + 96), _4135.Load4(asuint(asfloat(_5225.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _5239 = _5167 * 3u;
        vertex_t _5244;
        [unroll]
        for (int _96ident = 0; _96ident < 3; _96ident++)
        {
            _5244.p[_96ident] = asfloat(_4160.Load(_96ident * 4 + _4164.Load(_5239 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _97ident = 0; _97ident < 3; _97ident++)
        {
            _5244.n[_97ident] = asfloat(_4160.Load(_97ident * 4 + _4164.Load(_5239 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _98ident = 0; _98ident < 3; _98ident++)
        {
            _5244.b[_98ident] = asfloat(_4160.Load(_98ident * 4 + _4164.Load(_5239 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _99ident = 0; _99ident < 2; _99ident++)
        {
            [unroll]
            for (int _100ident = 0; _100ident < 2; _100ident++)
            {
                _5244.t[_99ident][_100ident] = asfloat(_4160.Load(_100ident * 4 + _99ident * 8 + _4164.Load(_5239 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _5290;
        [unroll]
        for (int _101ident = 0; _101ident < 3; _101ident++)
        {
            _5290.p[_101ident] = asfloat(_4160.Load(_101ident * 4 + _4164.Load((_5239 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _102ident = 0; _102ident < 3; _102ident++)
        {
            _5290.n[_102ident] = asfloat(_4160.Load(_102ident * 4 + _4164.Load((_5239 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _103ident = 0; _103ident < 3; _103ident++)
        {
            _5290.b[_103ident] = asfloat(_4160.Load(_103ident * 4 + _4164.Load((_5239 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _104ident = 0; _104ident < 2; _104ident++)
        {
            [unroll]
            for (int _105ident = 0; _105ident < 2; _105ident++)
            {
                _5290.t[_104ident][_105ident] = asfloat(_4160.Load(_105ident * 4 + _104ident * 8 + _4164.Load((_5239 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _5336;
        [unroll]
        for (int _106ident = 0; _106ident < 3; _106ident++)
        {
            _5336.p[_106ident] = asfloat(_4160.Load(_106ident * 4 + _4164.Load((_5239 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _107ident = 0; _107ident < 3; _107ident++)
        {
            _5336.n[_107ident] = asfloat(_4160.Load(_107ident * 4 + _4164.Load((_5239 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _108ident = 0; _108ident < 3; _108ident++)
        {
            _5336.b[_108ident] = asfloat(_4160.Load(_108ident * 4 + _4164.Load((_5239 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _109ident = 0; _109ident < 2; _109ident++)
        {
            [unroll]
            for (int _110ident = 0; _110ident < 2; _110ident++)
            {
                _5336.t[_109ident][_110ident] = asfloat(_4160.Load(_110ident * 4 + _109ident * 8 + _4164.Load((_5239 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _5382 = float3(_5244.p[0], _5244.p[1], _5244.p[2]);
        float3 _5390 = float3(_5290.p[0], _5290.p[1], _5290.p[2]);
        float3 _5398 = float3(_5336.p[0], _5336.p[1], _5336.p[2]);
        float _5405 = (1.0f - inter.u) - inter.v;
        float3 _5438 = normalize(((float3(_5244.n[0], _5244.n[1], _5244.n[2]) * _5405) + (float3(_5290.n[0], _5290.n[1], _5290.n[2]) * inter.u)) + (float3(_5336.n[0], _5336.n[1], _5336.n[2]) * inter.v));
        float3 N = _5438;
        float2 _5464 = ((float2(_5244.t[0][0], _5244.t[0][1]) * _5405) + (float2(_5290.t[0][0], _5290.t[0][1]) * inter.u)) + (float2(_5336.t[0][0], _5336.t[0][1]) * inter.v);
        float3 _5480 = cross(_5390 - _5382, _5398 - _5382);
        float _5483 = length(_5480);
        float3 plane_N = _5480 / _5483.xxx;
        float3 _5519 = ((float3(_5244.b[0], _5244.b[1], _5244.b[2]) * _5405) + (float3(_5290.b[0], _5290.b[1], _5290.b[2]) * inter.u)) + (float3(_5336.b[0], _5336.b[1], _5336.b[2]) * inter.v);
        float3 B = _5519;
        float3 T = cross(_5519, _5438);
        if (_5153)
        {
            if ((_4479.Load(_5167 * 4 + 0) & 65535u) == 65535u)
            {
                _8738 = 0.0f.xxx;
                break;
            }
            material_t _5542;
            [unroll]
            for (int _111ident = 0; _111ident < 5; _111ident++)
            {
                _5542.textures[_111ident] = _4475.Load(_111ident * 4 + (_4479.Load(_5167 * 4 + 0) & 16383u) * 80 + 0);
            }
            [unroll]
            for (int _112ident = 0; _112ident < 3; _112ident++)
            {
                _5542.base_color[_112ident] = asfloat(_4475.Load(_112ident * 4 + (_4479.Load(_5167 * 4 + 0) & 16383u) * 80 + 20));
            }
            _5542.flags = _4475.Load((_4479.Load(_5167 * 4 + 0) & 16383u) * 80 + 32);
            _5542.type = _4475.Load((_4479.Load(_5167 * 4 + 0) & 16383u) * 80 + 36);
            _5542.tangent_rotation_or_strength = asfloat(_4475.Load((_4479.Load(_5167 * 4 + 0) & 16383u) * 80 + 40));
            _5542.roughness_and_anisotropic = _4475.Load((_4479.Load(_5167 * 4 + 0) & 16383u) * 80 + 44);
            _5542.int_ior = asfloat(_4475.Load((_4479.Load(_5167 * 4 + 0) & 16383u) * 80 + 48));
            _5542.ext_ior = asfloat(_4475.Load((_4479.Load(_5167 * 4 + 0) & 16383u) * 80 + 52));
            _5542.sheen_and_sheen_tint = _4475.Load((_4479.Load(_5167 * 4 + 0) & 16383u) * 80 + 56);
            _5542.tint_and_metallic = _4475.Load((_4479.Load(_5167 * 4 + 0) & 16383u) * 80 + 60);
            _5542.transmission_and_transmission_roughness = _4475.Load((_4479.Load(_5167 * 4 + 0) & 16383u) * 80 + 64);
            _5542.specular_and_specular_tint = _4475.Load((_4479.Load(_5167 * 4 + 0) & 16383u) * 80 + 68);
            _5542.clearcoat_and_clearcoat_roughness = _4475.Load((_4479.Load(_5167 * 4 + 0) & 16383u) * 80 + 72);
            _5542.normal_map_strength_unorm = _4475.Load((_4479.Load(_5167 * 4 + 0) & 16383u) * 80 + 76);
            _9641 = _5542.textures[0];
            _9642 = _5542.textures[1];
            _9643 = _5542.textures[2];
            _9644 = _5542.textures[3];
            _9645 = _5542.textures[4];
            _9646 = _5542.base_color[0];
            _9647 = _5542.base_color[1];
            _9648 = _5542.base_color[2];
            _9290 = _5542.flags;
            _9291 = _5542.type;
            _9292 = _5542.tangent_rotation_or_strength;
            _9293 = _5542.roughness_and_anisotropic;
            _9294 = _5542.int_ior;
            _9295 = _5542.ext_ior;
            _9296 = _5542.sheen_and_sheen_tint;
            _9297 = _5542.tint_and_metallic;
            _9298 = _5542.transmission_and_transmission_roughness;
            _9299 = _5542.specular_and_specular_tint;
            _9300 = _5542.clearcoat_and_clearcoat_roughness;
            _9301 = _5542.normal_map_strength_unorm;
            plane_N = -plane_N;
            N = -N;
            B = -B;
            T = -T;
        }
        float3 param_11 = plane_N;
        float4x4 param_12 = _5232.inv_xform;
        plane_N = TransformNormal(param_11, param_12);
        float3 param_13 = N;
        float4x4 param_14 = _5232.inv_xform;
        N = TransformNormal(param_13, param_14);
        float3 param_15 = B;
        float4x4 param_16 = _5232.inv_xform;
        B = TransformNormal(param_15, param_16);
        float3 param_17 = T;
        float4x4 param_18 = _5232.inv_xform;
        T = TransformNormal(param_17, param_18);
        float _5652 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _5662 = mad(0.5f, log2(abs(mad(_5290.t[0][0] - _5244.t[0][0], _5336.t[0][1] - _5244.t[0][1], -((_5336.t[0][0] - _5244.t[0][0]) * (_5290.t[0][1] - _5244.t[0][1])))) / _5483), log2(_5652));
        uint param_19 = uint(hash(px_index));
        float _5668 = construct_float(param_19);
        uint param_20 = uint(hash(hash(px_index)));
        float _5674 = construct_float(param_20);
        float3 col = 0.0f.xxx;
        int _5681 = ray.ray_depth & 255;
        int _5686 = (ray.ray_depth >> 8) & 255;
        int _5691 = (ray.ray_depth >> 16) & 255;
        int _5697 = (ray.ray_depth >> 24) & 255;
        int _5705 = ((_5681 + _5686) + _5691) + _5697;
        float mix_rand = frac(asfloat(_3219.Load(_3224_g_params.hi * 4 + 0)) + _5668);
        float mix_weight = 1.0f;
        float _5742;
        float _5761;
        float _5786;
        float _5855;
        while (_9291 == 4u)
        {
            float mix_val = _9292;
            if (_9642 != 4294967295u)
            {
                mix_val *= SampleBilinear(_9642, _5464, 0).x;
            }
            if (_5153)
            {
                _5742 = _9295 / _9294;
            }
            else
            {
                _5742 = _9294 / _9295;
            }
            if (_9294 != 0.0f)
            {
                float param_21 = dot(_4766, N);
                float param_22 = _5742;
                _5761 = fresnel_dielectric_cos(param_21, param_22);
            }
            else
            {
                _5761 = 1.0f;
            }
            float _5775 = mix_val;
            float _5776 = _5775 * clamp(_5761, 0.0f, 1.0f);
            mix_val = _5776;
            if (mix_rand > _5776)
            {
                if ((_9290 & 2u) != 0u)
                {
                    _5786 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _5786 = 1.0f;
                }
                mix_weight *= _5786;
                material_t _5799;
                [unroll]
                for (int _113ident = 0; _113ident < 5; _113ident++)
                {
                    _5799.textures[_113ident] = _4475.Load(_113ident * 4 + _9644 * 80 + 0);
                }
                [unroll]
                for (int _114ident = 0; _114ident < 3; _114ident++)
                {
                    _5799.base_color[_114ident] = asfloat(_4475.Load(_114ident * 4 + _9644 * 80 + 20));
                }
                _5799.flags = _4475.Load(_9644 * 80 + 32);
                _5799.type = _4475.Load(_9644 * 80 + 36);
                _5799.tangent_rotation_or_strength = asfloat(_4475.Load(_9644 * 80 + 40));
                _5799.roughness_and_anisotropic = _4475.Load(_9644 * 80 + 44);
                _5799.int_ior = asfloat(_4475.Load(_9644 * 80 + 48));
                _5799.ext_ior = asfloat(_4475.Load(_9644 * 80 + 52));
                _5799.sheen_and_sheen_tint = _4475.Load(_9644 * 80 + 56);
                _5799.tint_and_metallic = _4475.Load(_9644 * 80 + 60);
                _5799.transmission_and_transmission_roughness = _4475.Load(_9644 * 80 + 64);
                _5799.specular_and_specular_tint = _4475.Load(_9644 * 80 + 68);
                _5799.clearcoat_and_clearcoat_roughness = _4475.Load(_9644 * 80 + 72);
                _5799.normal_map_strength_unorm = _4475.Load(_9644 * 80 + 76);
                _9641 = _5799.textures[0];
                _9642 = _5799.textures[1];
                _9643 = _5799.textures[2];
                _9644 = _5799.textures[3];
                _9645 = _5799.textures[4];
                _9646 = _5799.base_color[0];
                _9647 = _5799.base_color[1];
                _9648 = _5799.base_color[2];
                _9290 = _5799.flags;
                _9291 = _5799.type;
                _9292 = _5799.tangent_rotation_or_strength;
                _9293 = _5799.roughness_and_anisotropic;
                _9294 = _5799.int_ior;
                _9295 = _5799.ext_ior;
                _9296 = _5799.sheen_and_sheen_tint;
                _9297 = _5799.tint_and_metallic;
                _9298 = _5799.transmission_and_transmission_roughness;
                _9299 = _5799.specular_and_specular_tint;
                _9300 = _5799.clearcoat_and_clearcoat_roughness;
                _9301 = _5799.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_9290 & 2u) != 0u)
                {
                    _5855 = 1.0f / mix_val;
                }
                else
                {
                    _5855 = 1.0f;
                }
                mix_weight *= _5855;
                material_t _5867;
                [unroll]
                for (int _115ident = 0; _115ident < 5; _115ident++)
                {
                    _5867.textures[_115ident] = _4475.Load(_115ident * 4 + _9645 * 80 + 0);
                }
                [unroll]
                for (int _116ident = 0; _116ident < 3; _116ident++)
                {
                    _5867.base_color[_116ident] = asfloat(_4475.Load(_116ident * 4 + _9645 * 80 + 20));
                }
                _5867.flags = _4475.Load(_9645 * 80 + 32);
                _5867.type = _4475.Load(_9645 * 80 + 36);
                _5867.tangent_rotation_or_strength = asfloat(_4475.Load(_9645 * 80 + 40));
                _5867.roughness_and_anisotropic = _4475.Load(_9645 * 80 + 44);
                _5867.int_ior = asfloat(_4475.Load(_9645 * 80 + 48));
                _5867.ext_ior = asfloat(_4475.Load(_9645 * 80 + 52));
                _5867.sheen_and_sheen_tint = _4475.Load(_9645 * 80 + 56);
                _5867.tint_and_metallic = _4475.Load(_9645 * 80 + 60);
                _5867.transmission_and_transmission_roughness = _4475.Load(_9645 * 80 + 64);
                _5867.specular_and_specular_tint = _4475.Load(_9645 * 80 + 68);
                _5867.clearcoat_and_clearcoat_roughness = _4475.Load(_9645 * 80 + 72);
                _5867.normal_map_strength_unorm = _4475.Load(_9645 * 80 + 76);
                _9641 = _5867.textures[0];
                _9642 = _5867.textures[1];
                _9643 = _5867.textures[2];
                _9644 = _5867.textures[3];
                _9645 = _5867.textures[4];
                _9646 = _5867.base_color[0];
                _9647 = _5867.base_color[1];
                _9648 = _5867.base_color[2];
                _9290 = _5867.flags;
                _9291 = _5867.type;
                _9292 = _5867.tangent_rotation_or_strength;
                _9293 = _5867.roughness_and_anisotropic;
                _9294 = _5867.int_ior;
                _9295 = _5867.ext_ior;
                _9296 = _5867.sheen_and_sheen_tint;
                _9297 = _5867.tint_and_metallic;
                _9298 = _5867.transmission_and_transmission_roughness;
                _9299 = _5867.specular_and_specular_tint;
                _9300 = _5867.clearcoat_and_clearcoat_roughness;
                _9301 = _5867.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_9641 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_9641, _5464, 0).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_855.Load(_9641 * 80 + 0) & 16384u) != 0u)
            {
                float3 _9992 = normals;
                _9992.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _9992;
            }
            float3 _5951 = N;
            N = normalize(((T * normals.x) + (_5951 * normals.z)) + (B * normals.y));
            if ((_9301 & 65535u) != 65535u)
            {
                N = normalize(_5951 + ((N - _5951) * clamp(float(_9301 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_23 = plane_N;
            float3 param_24 = -_4766;
            float3 param_25 = N;
            N = ensure_valid_reflection(param_23, param_24, param_25);
        }
        float3 _6008 = ((_5382 * _5405) + (_5390 * inter.u)) + (_5398 * inter.v);
        float3 _6015 = float3(-_6008.z, 0.0f, _6008.x);
        float3 tangent = _6015;
        float3 param_26 = _6015;
        float4x4 param_27 = _5232.inv_xform;
        tangent = TransformNormal(param_26, param_27);
        if (_9292 != 0.0f)
        {
            float3 param_28 = tangent;
            float3 param_29 = N;
            float param_30 = _9292;
            tangent = rotate_around_axis(param_28, param_29, param_30);
        }
        float3 _6038 = normalize(cross(tangent, N));
        B = _6038;
        T = cross(N, _6038);
        float3 _9380 = 0.0f.xxx;
        float3 _9379 = 0.0f.xxx;
        float _9382 = 0.0f;
        float _9383 = 0.0f;
        float _9381 = 0.0f;
        bool _6050 = _3224_g_params.li_count != 0;
        bool _6056;
        if (_6050)
        {
            _6056 = _9291 != 3u;
        }
        else
        {
            _6056 = _6050;
        }
        float _9384;
        if (_6056)
        {
            float3 param_31 = _4889;
            float2 param_32 = float2(_5668, _5674);
            light_sample_t _9391 = { _9379, _9380, _9381, _9382, _9383, _9384 };
            light_sample_t param_33 = _9391;
            SampleLightSource(param_31, param_32, param_33);
            _9379 = param_33.col;
            _9380 = param_33.L;
            _9381 = param_33.area;
            _9382 = param_33.dist;
            _9383 = param_33.pdf;
            _9384 = param_33.cast_shadow;
        }
        float _6071 = dot(N, _9380);
        float3 base_color = float3(_9646, _9647, _9648);
        [branch]
        if (_9642 != 4294967295u)
        {
            base_color *= SampleBilinear(_9642, _5464, int(get_texture_lod(texSize(_9642), _5662)), true, true).xyz;
        }
        float3 tint_color = 0.0f.xxx;
        float _6115 = lum(base_color);
        [flatten]
        if (_6115 > 0.0f)
        {
            tint_color = base_color / _6115.xxx;
        }
        float roughness = clamp(float(_9293 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_9643 != 4294967295u)
        {
            roughness *= SampleBilinear(_9643, _5464, int(get_texture_lod(texSize(_9643), _5662)), false, true).x;
        }
        float _6156 = asfloat(_3219.Load((_3224_g_params.hi + 1) * 4 + 0));
        float _6160 = frac(_6156 + _5668);
        float _6166 = asfloat(_3219.Load((_3224_g_params.hi + 2) * 4 + 0));
        float _6170 = frac(_6166 + _5674);
        float _9712 = 0.0f;
        float _9711 = 0.0f;
        float _9710 = 0.0f;
        float _9412 = 0.0f;
        int _9417;
        float _9696;
        float _9697;
        float _9698;
        float _9703;
        float _9704;
        float _9705;
        [branch]
        if (_9291 == 0u)
        {
            [branch]
            if ((_9383 > 0.0f) && (_6071 > 0.0f))
            {
                float3 param_34 = -_4766;
                float3 param_35 = N;
                float3 param_36 = _9380;
                float param_37 = roughness;
                float3 param_38 = base_color;
                float4 _6210 = Evaluate_OrenDiffuse_BSDF(param_34, param_35, param_36, param_37, param_38);
                float mis_weight = 1.0f;
                if (_9381 > 0.0f)
                {
                    float param_39 = _9383;
                    float param_40 = _6210.w;
                    mis_weight = power_heuristic(param_39, param_40);
                }
                float3 _6238 = (_9379 * _6210.xyz) * ((mix_weight * mis_weight) / _9383);
                [branch]
                if (_9384 > 0.5f)
                {
                    float3 param_41 = _4889;
                    float3 param_42 = plane_N;
                    float3 _6249 = offset_ray(param_41, param_42);
                    uint _6303;
                    _6301.InterlockedAdd(8, 1u, _6303);
                    _6311.Store(_6303 * 44 + 0, asuint(_6249.x));
                    _6311.Store(_6303 * 44 + 4, asuint(_6249.y));
                    _6311.Store(_6303 * 44 + 8, asuint(_6249.z));
                    _6311.Store(_6303 * 44 + 12, asuint(_9380.x));
                    _6311.Store(_6303 * 44 + 16, asuint(_9380.y));
                    _6311.Store(_6303 * 44 + 20, asuint(_9380.z));
                    _6311.Store(_6303 * 44 + 24, asuint(_9382 - 9.9999997473787516355514526367188e-05f));
                    _6311.Store(_6303 * 44 + 28, asuint(ray.c[0] * _6238.x));
                    _6311.Store(_6303 * 44 + 32, asuint(ray.c[1] * _6238.y));
                    _6311.Store(_6303 * 44 + 36, asuint(ray.c[2] * _6238.z));
                    _6311.Store(_6303 * 44 + 40, uint(ray.xy));
                }
                else
                {
                    col += _6238;
                }
            }
            bool _6355 = _5681 < _3224_g_params.max_diff_depth;
            bool _6362;
            if (_6355)
            {
                _6362 = _5705 < _3224_g_params.max_total_depth;
            }
            else
            {
                _6362 = _6355;
            }
            [branch]
            if (_6362)
            {
                float3 param_43 = T;
                float3 param_44 = B;
                float3 param_45 = N;
                float3 param_46 = _4766;
                float param_47 = roughness;
                float3 param_48 = base_color;
                float param_49 = _6160;
                float param_50 = _6170;
                float3 param_51;
                float4 _6384 = Sample_OrenDiffuse_BSDF(param_43, param_44, param_45, param_46, param_47, param_48, param_49, param_50, param_51);
                _9417 = ray.ray_depth + 1;
                float3 param_52 = _4889;
                float3 param_53 = plane_N;
                float3 _6395 = offset_ray(param_52, param_53);
                _9696 = _6395.x;
                _9697 = _6395.y;
                _9698 = _6395.z;
                _9703 = param_51.x;
                _9704 = param_51.y;
                _9705 = param_51.z;
                _9710 = ((ray.c[0] * _6384.x) * mix_weight) / _6384.w;
                _9711 = ((ray.c[1] * _6384.y) * mix_weight) / _6384.w;
                _9712 = ((ray.c[2] * _6384.z) * mix_weight) / _6384.w;
                _9412 = _6384.w;
            }
        }
        else
        {
            [branch]
            if (_9291 == 1u)
            {
                float param_54 = 1.0f;
                float param_55 = 1.5f;
                float _6460 = fresnel_dielectric_cos(param_54, param_55);
                float _6464 = roughness * roughness;
                bool _6467 = _9383 > 0.0f;
                bool _6474;
                if (_6467)
                {
                    _6474 = (_6464 * _6464) >= 1.0000000116860974230803549289703e-07f;
                }
                else
                {
                    _6474 = _6467;
                }
                [branch]
                if (_6474 && (_6071 > 0.0f))
                {
                    float3 param_56 = T;
                    float3 param_57 = B;
                    float3 param_58 = N;
                    float3 param_59 = -_4766;
                    float3 param_60 = T;
                    float3 param_61 = B;
                    float3 param_62 = N;
                    float3 param_63 = _9380;
                    float3 param_64 = T;
                    float3 param_65 = B;
                    float3 param_66 = N;
                    float3 param_67 = normalize(_9380 - _4766);
                    float3 param_68 = tangent_from_world(param_56, param_57, param_58, param_59);
                    float3 param_69 = tangent_from_world(param_64, param_65, param_66, param_67);
                    float3 param_70 = tangent_from_world(param_60, param_61, param_62, param_63);
                    float param_71 = _6464;
                    float param_72 = _6464;
                    float param_73 = 1.5f;
                    float param_74 = _6460;
                    float3 param_75 = base_color;
                    float4 _6534 = Evaluate_GGXSpecular_BSDF(param_68, param_69, param_70, param_71, param_72, param_73, param_74, param_75);
                    float mis_weight_1 = 1.0f;
                    if (_9381 > 0.0f)
                    {
                        float param_76 = _9383;
                        float param_77 = _6534.w;
                        mis_weight_1 = power_heuristic(param_76, param_77);
                    }
                    float3 _6562 = (_9379 * _6534.xyz) * ((mix_weight * mis_weight_1) / _9383);
                    [branch]
                    if (_9384 > 0.5f)
                    {
                        float3 param_78 = _4889;
                        float3 param_79 = plane_N;
                        float3 _6573 = offset_ray(param_78, param_79);
                        uint _6620;
                        _6301.InterlockedAdd(8, 1u, _6620);
                        _6311.Store(_6620 * 44 + 0, asuint(_6573.x));
                        _6311.Store(_6620 * 44 + 4, asuint(_6573.y));
                        _6311.Store(_6620 * 44 + 8, asuint(_6573.z));
                        _6311.Store(_6620 * 44 + 12, asuint(_9380.x));
                        _6311.Store(_6620 * 44 + 16, asuint(_9380.y));
                        _6311.Store(_6620 * 44 + 20, asuint(_9380.z));
                        _6311.Store(_6620 * 44 + 24, asuint(_9382 - 9.9999997473787516355514526367188e-05f));
                        _6311.Store(_6620 * 44 + 28, asuint(ray.c[0] * _6562.x));
                        _6311.Store(_6620 * 44 + 32, asuint(ray.c[1] * _6562.y));
                        _6311.Store(_6620 * 44 + 36, asuint(ray.c[2] * _6562.z));
                        _6311.Store(_6620 * 44 + 40, uint(ray.xy));
                    }
                    else
                    {
                        col += _6562;
                    }
                }
                bool _6659 = _5686 < _3224_g_params.max_spec_depth;
                bool _6666;
                if (_6659)
                {
                    _6666 = _5705 < _3224_g_params.max_total_depth;
                }
                else
                {
                    _6666 = _6659;
                }
                [branch]
                if (_6666)
                {
                    float3 param_80 = T;
                    float3 param_81 = B;
                    float3 param_82 = N;
                    float3 param_83 = _4766;
                    float3 param_84;
                    float4 _6685 = Sample_GGXSpecular_BSDF(param_80, param_81, param_82, param_83, roughness, 0.0f, 1.5f, _6460, base_color, _6160, _6170, param_84);
                    _9417 = ray.ray_depth + 256;
                    float3 param_85 = _4889;
                    float3 param_86 = plane_N;
                    float3 _6697 = offset_ray(param_85, param_86);
                    _9696 = _6697.x;
                    _9697 = _6697.y;
                    _9698 = _6697.z;
                    _9703 = param_84.x;
                    _9704 = param_84.y;
                    _9705 = param_84.z;
                    _9710 = ((ray.c[0] * _6685.x) * mix_weight) / _6685.w;
                    _9711 = ((ray.c[1] * _6685.y) * mix_weight) / _6685.w;
                    _9712 = ((ray.c[2] * _6685.z) * mix_weight) / _6685.w;
                    _9412 = _6685.w;
                }
            }
            else
            {
                [branch]
                if (_9291 == 2u)
                {
                    float _6760;
                    if (_5153)
                    {
                        _6760 = _9294 / _9295;
                    }
                    else
                    {
                        _6760 = _9295 / _9294;
                    }
                    float _6778 = roughness * roughness;
                    bool _6781 = _9383 > 0.0f;
                    bool _6788;
                    if (_6781)
                    {
                        _6788 = (_6778 * _6778) >= 1.0000000116860974230803549289703e-07f;
                    }
                    else
                    {
                        _6788 = _6781;
                    }
                    [branch]
                    if (_6788 && (_6071 < 0.0f))
                    {
                        float3 param_87 = T;
                        float3 param_88 = B;
                        float3 param_89 = N;
                        float3 param_90 = -_4766;
                        float3 param_91 = T;
                        float3 param_92 = B;
                        float3 param_93 = N;
                        float3 param_94 = _9380;
                        float3 param_95 = T;
                        float3 param_96 = B;
                        float3 param_97 = N;
                        float3 param_98 = normalize(_9380 - (_4766 * _6760));
                        float3 param_99 = tangent_from_world(param_87, param_88, param_89, param_90);
                        float3 param_100 = tangent_from_world(param_95, param_96, param_97, param_98);
                        float3 param_101 = tangent_from_world(param_91, param_92, param_93, param_94);
                        float param_102 = _6778;
                        float param_103 = _6760;
                        float3 param_104 = base_color;
                        float4 _6847 = Evaluate_GGXRefraction_BSDF(param_99, param_100, param_101, param_102, param_103, param_104);
                        float mis_weight_2 = 1.0f;
                        if (_9381 > 0.0f)
                        {
                            float param_105 = _9383;
                            float param_106 = _6847.w;
                            mis_weight_2 = power_heuristic(param_105, param_106);
                        }
                        float3 _6875 = (_9379 * _6847.xyz) * ((mix_weight * mis_weight_2) / _9383);
                        [branch]
                        if (_9384 > 0.5f)
                        {
                            float3 param_107 = _4889;
                            float3 param_108 = -plane_N;
                            float3 _6887 = offset_ray(param_107, param_108);
                            uint _6934;
                            _6301.InterlockedAdd(8, 1u, _6934);
                            _6311.Store(_6934 * 44 + 0, asuint(_6887.x));
                            _6311.Store(_6934 * 44 + 4, asuint(_6887.y));
                            _6311.Store(_6934 * 44 + 8, asuint(_6887.z));
                            _6311.Store(_6934 * 44 + 12, asuint(_9380.x));
                            _6311.Store(_6934 * 44 + 16, asuint(_9380.y));
                            _6311.Store(_6934 * 44 + 20, asuint(_9380.z));
                            _6311.Store(_6934 * 44 + 24, asuint(_9382 - 9.9999997473787516355514526367188e-05f));
                            _6311.Store(_6934 * 44 + 28, asuint(ray.c[0] * _6875.x));
                            _6311.Store(_6934 * 44 + 32, asuint(ray.c[1] * _6875.y));
                            _6311.Store(_6934 * 44 + 36, asuint(ray.c[2] * _6875.z));
                            _6311.Store(_6934 * 44 + 40, uint(ray.xy));
                        }
                        else
                        {
                            col += _6875;
                        }
                    }
                    bool _6973 = _5691 < _3224_g_params.max_refr_depth;
                    bool _6980;
                    if (_6973)
                    {
                        _6980 = _5705 < _3224_g_params.max_total_depth;
                    }
                    else
                    {
                        _6980 = _6973;
                    }
                    [branch]
                    if (_6980)
                    {
                        float3 param_109 = T;
                        float3 param_110 = B;
                        float3 param_111 = N;
                        float3 param_112 = _4766;
                        float param_113 = roughness;
                        float param_114 = _6760;
                        float3 param_115 = base_color;
                        float param_116 = _6160;
                        float param_117 = _6170;
                        float4 param_118;
                        float4 _7004 = Sample_GGXRefraction_BSDF(param_109, param_110, param_111, param_112, param_113, param_114, param_115, param_116, param_117, param_118);
                        _9417 = ray.ray_depth + 65536;
                        _9710 = ((ray.c[0] * _7004.x) * mix_weight) / _7004.w;
                        _9711 = ((ray.c[1] * _7004.y) * mix_weight) / _7004.w;
                        _9712 = ((ray.c[2] * _7004.z) * mix_weight) / _7004.w;
                        _9412 = _7004.w;
                        float3 param_119 = _4889;
                        float3 param_120 = -plane_N;
                        float3 _7059 = offset_ray(param_119, param_120);
                        _9696 = _7059.x;
                        _9697 = _7059.y;
                        _9698 = _7059.z;
                        _9703 = param_118.x;
                        _9704 = param_118.y;
                        _9705 = param_118.z;
                    }
                }
                else
                {
                    [branch]
                    if (_9291 == 3u)
                    {
                        if ((_9290 & 4u) != 0u)
                        {
                            float3 env_col_1 = _3224_g_params.env_col.xyz;
                            uint _7098 = asuint(_3224_g_params.env_col.w);
                            if (_7098 != 4294967295u)
                            {
                                atlas_texture_t _7105;
                                _7105.size = _855.Load(_7098 * 80 + 0);
                                _7105.atlas = _855.Load(_7098 * 80 + 4);
                                [unroll]
                                for (int _117ident = 0; _117ident < 4; _117ident++)
                                {
                                    _7105.page[_117ident] = _855.Load(_117ident * 4 + _7098 * 80 + 8);
                                }
                                [unroll]
                                for (int _118ident = 0; _118ident < 14; _118ident++)
                                {
                                    _7105.pos[_118ident] = _855.Load(_118ident * 4 + _7098 * 80 + 24);
                                }
                                uint _9817[14] = { _7105.pos[0], _7105.pos[1], _7105.pos[2], _7105.pos[3], _7105.pos[4], _7105.pos[5], _7105.pos[6], _7105.pos[7], _7105.pos[8], _7105.pos[9], _7105.pos[10], _7105.pos[11], _7105.pos[12], _7105.pos[13] };
                                uint _9788[4] = { _7105.page[0], _7105.page[1], _7105.page[2], _7105.page[3] };
                                atlas_texture_t _9582 = { _7105.size, _7105.atlas, _9788, _9817 };
                                float param_121 = _3224_g_params.env_rotation;
                                env_col_1 *= SampleLatlong_RGBE(_9582, _4766, param_121);
                            }
                            base_color *= env_col_1;
                        }
                        col += (base_color * (mix_weight * _9292));
                    }
                    else
                    {
                        [branch]
                        if (_9291 == 5u)
                        {
                            bool _7181 = _5697 < _3224_g_params.max_transp_depth;
                            bool _7188;
                            if (_7181)
                            {
                                _7188 = _5705 < _3224_g_params.max_total_depth;
                            }
                            else
                            {
                                _7188 = _7181;
                            }
                            [branch]
                            if (_7188)
                            {
                                _9417 = ray.ray_depth + 16777216;
                                _9412 = ray.pdf;
                                float3 param_122 = _4889;
                                float3 param_123 = -plane_N;
                                float3 _7205 = offset_ray(param_122, param_123);
                                _9696 = _7205.x;
                                _9697 = _7205.y;
                                _9698 = _7205.z;
                                _9703 = ray.d[0];
                                _9704 = ray.d[1];
                                _9705 = ray.d[2];
                                _9710 = ray.c[0];
                                _9711 = ray.c[1];
                                _9712 = ray.c[2];
                            }
                        }
                        else
                        {
                            if (_9291 == 6u)
                            {
                                float metallic = clamp(float((_9297 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                [branch]
                                if (_9644 != 4294967295u)
                                {
                                    metallic *= SampleBilinear(_9644, _5464, int(get_texture_lod(texSize(_9644), _5662))).x;
                                }
                                float specular = clamp(float(_9299 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                [branch]
                                if (_9645 != 4294967295u)
                                {
                                    specular *= SampleBilinear(_9645, _5464, int(get_texture_lod(texSize(_9645), _5662))).x;
                                }
                                float _7315 = clamp(float(_9300 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                float _7323 = clamp(float((_9300 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                float _7330 = clamp(float(_9296 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                float3 _7352 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_9299 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                                float _7359 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                                float param_124 = 1.0f;
                                float param_125 = _7359;
                                float _7364 = fresnel_dielectric_cos(param_124, param_125);
                                float param_126 = dot(_4766, N);
                                float param_127 = _7359;
                                float param_128;
                                float param_129;
                                float param_130;
                                float param_131;
                                get_lobe_weights(lerp(_6115, 1.0f, _7330), lum(lerp(_7352, 1.0f.xxx, ((fresnel_dielectric_cos(param_126, param_127) - _7364) / (1.0f - _7364)).xxx)), specular, metallic, clamp(float(_9298 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _7315, param_128, param_129, param_130, param_131);
                                float3 _7418 = lerp(1.0f.xxx, tint_color, clamp(float((_9296 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _7330;
                                float _7421;
                                if (_5153)
                                {
                                    _7421 = _9294 / _9295;
                                }
                                else
                                {
                                    _7421 = _9295 / _9294;
                                }
                                float param_132 = dot(_4766, N);
                                float param_133 = 1.0f / _7421;
                                float _7444 = fresnel_dielectric_cos(param_132, param_133);
                                float _7451 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _7315))) - 1.0f;
                                float param_134 = 1.0f;
                                float param_135 = _7451;
                                float _7456 = fresnel_dielectric_cos(param_134, param_135);
                                float _7460 = _7323 * _7323;
                                float _7473 = mad(roughness - 1.0f, 1.0f - clamp(float((_9298 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                                float _7477 = _7473 * _7473;
                                [branch]
                                if (_9383 > 0.0f)
                                {
                                    float3 lcol_1 = 0.0f.xxx;
                                    float bsdf_pdf = 0.0f;
                                    bool _7488 = _6071 > 0.0f;
                                    [branch]
                                    if ((param_128 > 1.0000000116860974230803549289703e-07f) && _7488)
                                    {
                                        float3 param_136 = -_4766;
                                        float3 param_137 = N;
                                        float3 param_138 = _9380;
                                        float param_139 = roughness;
                                        float3 param_140 = base_color.xyz;
                                        float3 param_141 = _7418;
                                        bool param_142 = false;
                                        float4 _7508 = Evaluate_PrincipledDiffuse_BSDF(param_136, param_137, param_138, param_139, param_140, param_141, param_142);
                                        bsdf_pdf = mad(param_128, _7508.w, bsdf_pdf);
                                        lcol_1 += (((_9379 * _6071) * (_7508 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * _9383).xxx);
                                    }
                                    float3 H;
                                    [flatten]
                                    if (_7488)
                                    {
                                        H = normalize(_9380 - _4766);
                                    }
                                    else
                                    {
                                        H = normalize(_9380 - (_4766 * _7421));
                                    }
                                    float _7554 = roughness * roughness;
                                    float _7565 = sqrt(mad(-0.89999997615814208984375f, clamp(float((_9293 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f));
                                    float _7569 = _7554 / _7565;
                                    float _7573 = _7554 * _7565;
                                    float3 param_143 = T;
                                    float3 param_144 = B;
                                    float3 param_145 = N;
                                    float3 param_146 = -_4766;
                                    float3 _7584 = tangent_from_world(param_143, param_144, param_145, param_146);
                                    float3 param_147 = T;
                                    float3 param_148 = B;
                                    float3 param_149 = N;
                                    float3 param_150 = _9380;
                                    float3 _7595 = tangent_from_world(param_147, param_148, param_149, param_150);
                                    float3 param_151 = T;
                                    float3 param_152 = B;
                                    float3 param_153 = N;
                                    float3 param_154 = H;
                                    float3 _7605 = tangent_from_world(param_151, param_152, param_153, param_154);
                                    bool _7607 = param_129 > 0.0f;
                                    bool _7614;
                                    if (_7607)
                                    {
                                        _7614 = (_7569 * _7573) >= 1.0000000116860974230803549289703e-07f;
                                    }
                                    else
                                    {
                                        _7614 = _7607;
                                    }
                                    [branch]
                                    if (_7614 && _7488)
                                    {
                                        float3 param_155 = _7584;
                                        float3 param_156 = _7605;
                                        float3 param_157 = _7595;
                                        float param_158 = _7569;
                                        float param_159 = _7573;
                                        float param_160 = _7359;
                                        float param_161 = _7364;
                                        float3 param_162 = _7352;
                                        float4 _7637 = Evaluate_GGXSpecular_BSDF(param_155, param_156, param_157, param_158, param_159, param_160, param_161, param_162);
                                        bsdf_pdf = mad(param_129, _7637.w, bsdf_pdf);
                                        lcol_1 += ((_9379 * _7637.xyz) / _9383.xxx);
                                    }
                                    bool _7656 = param_130 > 0.0f;
                                    bool _7663;
                                    if (_7656)
                                    {
                                        _7663 = (_7460 * _7460) >= 1.0000000116860974230803549289703e-07f;
                                    }
                                    else
                                    {
                                        _7663 = _7656;
                                    }
                                    [branch]
                                    if (_7663 && _7488)
                                    {
                                        float3 param_163 = _7584;
                                        float3 param_164 = _7605;
                                        float3 param_165 = _7595;
                                        float param_166 = _7460;
                                        float param_167 = _7451;
                                        float param_168 = _7456;
                                        float4 _7682 = Evaluate_PrincipledClearcoat_BSDF(param_163, param_164, param_165, param_166, param_167, param_168);
                                        bsdf_pdf = mad(param_130, _7682.w, bsdf_pdf);
                                        lcol_1 += (((_9379 * 0.25f) * _7682.xyz) / _9383.xxx);
                                    }
                                    [branch]
                                    if (param_131 > 0.0f)
                                    {
                                        bool _7706 = _7444 != 0.0f;
                                        bool _7713;
                                        if (_7706)
                                        {
                                            _7713 = (_7554 * _7554) >= 1.0000000116860974230803549289703e-07f;
                                        }
                                        else
                                        {
                                            _7713 = _7706;
                                        }
                                        [branch]
                                        if (_7713 && _7488)
                                        {
                                            float3 param_169 = _7584;
                                            float3 param_170 = _7605;
                                            float3 param_171 = _7595;
                                            float param_172 = _7554;
                                            float param_173 = _7554;
                                            float param_174 = 1.0f;
                                            float param_175 = 0.0f;
                                            float3 param_176 = 1.0f.xxx;
                                            float4 _7733 = Evaluate_GGXSpecular_BSDF(param_169, param_170, param_171, param_172, param_173, param_174, param_175, param_176);
                                            bsdf_pdf = mad(param_131 * _7444, _7733.w, bsdf_pdf);
                                            lcol_1 += ((_9379 * _7733.xyz) * (_7444 / _9383));
                                        }
                                        bool _7755 = _7444 != 1.0f;
                                        bool _7762;
                                        if (_7755)
                                        {
                                            _7762 = (_7477 * _7477) >= 1.0000000116860974230803549289703e-07f;
                                        }
                                        else
                                        {
                                            _7762 = _7755;
                                        }
                                        [branch]
                                        if (_7762 && (_6071 < 0.0f))
                                        {
                                            float3 param_177 = _7584;
                                            float3 param_178 = _7605;
                                            float3 param_179 = _7595;
                                            float param_180 = _7477;
                                            float param_181 = _7421;
                                            float3 param_182 = base_color;
                                            float4 _7781 = Evaluate_GGXRefraction_BSDF(param_177, param_178, param_179, param_180, param_181, param_182);
                                            float _7784 = 1.0f - _7444;
                                            bsdf_pdf = mad(param_131 * _7784, _7781.w, bsdf_pdf);
                                            lcol_1 += ((_9379 * _7781.xyz) * (_7784 / _9383));
                                        }
                                    }
                                    float mis_weight_3 = 1.0f;
                                    [flatten]
                                    if (_9381 > 0.0f)
                                    {
                                        float param_183 = _9383;
                                        float param_184 = bsdf_pdf;
                                        mis_weight_3 = power_heuristic(param_183, param_184);
                                    }
                                    lcol_1 *= (mix_weight * mis_weight_3);
                                    [branch]
                                    if (_9384 > 0.5f)
                                    {
                                        float3 _7829;
                                        if (_6071 < 0.0f)
                                        {
                                            _7829 = -plane_N;
                                        }
                                        else
                                        {
                                            _7829 = plane_N;
                                        }
                                        float3 param_185 = _4889;
                                        float3 param_186 = _7829;
                                        float3 _7840 = offset_ray(param_185, param_186);
                                        uint _7887;
                                        _6301.InterlockedAdd(8, 1u, _7887);
                                        _6311.Store(_7887 * 44 + 0, asuint(_7840.x));
                                        _6311.Store(_7887 * 44 + 4, asuint(_7840.y));
                                        _6311.Store(_7887 * 44 + 8, asuint(_7840.z));
                                        _6311.Store(_7887 * 44 + 12, asuint(_9380.x));
                                        _6311.Store(_7887 * 44 + 16, asuint(_9380.y));
                                        _6311.Store(_7887 * 44 + 20, asuint(_9380.z));
                                        _6311.Store(_7887 * 44 + 24, asuint(_9382 - 9.9999997473787516355514526367188e-05f));
                                        _6311.Store(_7887 * 44 + 28, asuint(ray.c[0] * lcol_1.x));
                                        _6311.Store(_7887 * 44 + 32, asuint(ray.c[1] * lcol_1.y));
                                        _6311.Store(_7887 * 44 + 36, asuint(ray.c[2] * lcol_1.z));
                                        _6311.Store(_7887 * 44 + 40, uint(ray.xy));
                                    }
                                    else
                                    {
                                        col += lcol_1;
                                    }
                                }
                                [branch]
                                if (mix_rand < param_128)
                                {
                                    bool _7931 = _5681 < _3224_g_params.max_diff_depth;
                                    bool _7938;
                                    if (_7931)
                                    {
                                        _7938 = _5705 < _3224_g_params.max_total_depth;
                                    }
                                    else
                                    {
                                        _7938 = _7931;
                                    }
                                    if (_7938)
                                    {
                                        float3 param_187 = T;
                                        float3 param_188 = B;
                                        float3 param_189 = N;
                                        float3 param_190 = _4766;
                                        float param_191 = roughness;
                                        float3 param_192 = base_color.xyz;
                                        float3 param_193 = _7418;
                                        bool param_194 = false;
                                        float param_195 = _6160;
                                        float param_196 = _6170;
                                        float3 param_197;
                                        float4 _7963 = Sample_PrincipledDiffuse_BSDF(param_187, param_188, param_189, param_190, param_191, param_192, param_193, param_194, param_195, param_196, param_197);
                                        float3 _7969 = _7963.xyz * (1.0f - metallic);
                                        _9417 = ray.ray_depth + 1;
                                        float3 param_198 = _4889;
                                        float3 param_199 = plane_N;
                                        float3 _7985 = offset_ray(param_198, param_199);
                                        _9696 = _7985.x;
                                        _9697 = _7985.y;
                                        _9698 = _7985.z;
                                        _9703 = param_197.x;
                                        _9704 = param_197.y;
                                        _9705 = param_197.z;
                                        _9710 = ((ray.c[0] * _7969.x) * mix_weight) / param_128;
                                        _9711 = ((ray.c[1] * _7969.y) * mix_weight) / param_128;
                                        _9712 = ((ray.c[2] * _7969.z) * mix_weight) / param_128;
                                        _9412 = _7963.w;
                                    }
                                }
                                else
                                {
                                    float _8041 = param_128 + param_129;
                                    [branch]
                                    if (mix_rand < _8041)
                                    {
                                        bool _8048 = _5686 < _3224_g_params.max_spec_depth;
                                        bool _8055;
                                        if (_8048)
                                        {
                                            _8055 = _5705 < _3224_g_params.max_total_depth;
                                        }
                                        else
                                        {
                                            _8055 = _8048;
                                        }
                                        if (_8055)
                                        {
                                            float3 param_200 = T;
                                            float3 param_201 = B;
                                            float3 param_202 = N;
                                            float3 param_203 = _4766;
                                            float3 param_204;
                                            float4 _8082 = Sample_GGXSpecular_BSDF(param_200, param_201, param_202, param_203, roughness, clamp(float((_9293 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _7359, _7364, _7352, _6160, _6170, param_204);
                                            float _8087 = _8082.w * param_129;
                                            _9417 = ray.ray_depth + 256;
                                            _9710 = ((ray.c[0] * _8082.x) * mix_weight) / _8087;
                                            _9711 = ((ray.c[1] * _8082.y) * mix_weight) / _8087;
                                            _9712 = ((ray.c[2] * _8082.z) * mix_weight) / _8087;
                                            _9412 = _8087;
                                            float3 param_205 = _4889;
                                            float3 param_206 = plane_N;
                                            float3 _8134 = offset_ray(param_205, param_206);
                                            _9696 = _8134.x;
                                            _9697 = _8134.y;
                                            _9698 = _8134.z;
                                            _9703 = param_204.x;
                                            _9704 = param_204.y;
                                            _9705 = param_204.z;
                                        }
                                    }
                                    else
                                    {
                                        float _8159 = _8041 + param_130;
                                        [branch]
                                        if (mix_rand < _8159)
                                        {
                                            bool _8166 = _5686 < _3224_g_params.max_spec_depth;
                                            bool _8173;
                                            if (_8166)
                                            {
                                                _8173 = _5705 < _3224_g_params.max_total_depth;
                                            }
                                            else
                                            {
                                                _8173 = _8166;
                                            }
                                            if (_8173)
                                            {
                                                float3 param_207 = T;
                                                float3 param_208 = B;
                                                float3 param_209 = N;
                                                float3 param_210 = _4766;
                                                float param_211 = _7460;
                                                float param_212 = _7451;
                                                float param_213 = _7456;
                                                float param_214 = _6160;
                                                float param_215 = _6170;
                                                float3 param_216;
                                                float4 _8197 = Sample_PrincipledClearcoat_BSDF(param_207, param_208, param_209, param_210, param_211, param_212, param_213, param_214, param_215, param_216);
                                                float _8202 = _8197.w * param_130;
                                                _9417 = ray.ray_depth + 256;
                                                _9710 = (((0.25f * ray.c[0]) * _8197.x) * mix_weight) / _8202;
                                                _9711 = (((0.25f * ray.c[1]) * _8197.y) * mix_weight) / _8202;
                                                _9712 = (((0.25f * ray.c[2]) * _8197.z) * mix_weight) / _8202;
                                                _9412 = _8202;
                                                float3 param_217 = _4889;
                                                float3 param_218 = plane_N;
                                                float3 _8252 = offset_ray(param_217, param_218);
                                                _9696 = _8252.x;
                                                _9697 = _8252.y;
                                                _9698 = _8252.z;
                                                _9703 = param_216.x;
                                                _9704 = param_216.y;
                                                _9705 = param_216.z;
                                            }
                                        }
                                        else
                                        {
                                            bool _8274 = mix_rand >= _7444;
                                            bool _8281;
                                            if (_8274)
                                            {
                                                _8281 = _5691 < _3224_g_params.max_refr_depth;
                                            }
                                            else
                                            {
                                                _8281 = _8274;
                                            }
                                            bool _8295;
                                            if (!_8281)
                                            {
                                                bool _8287 = mix_rand < _7444;
                                                bool _8294;
                                                if (_8287)
                                                {
                                                    _8294 = _5686 < _3224_g_params.max_spec_depth;
                                                }
                                                else
                                                {
                                                    _8294 = _8287;
                                                }
                                                _8295 = _8294;
                                            }
                                            else
                                            {
                                                _8295 = _8281;
                                            }
                                            bool _8302;
                                            if (_8295)
                                            {
                                                _8302 = _5705 < _3224_g_params.max_total_depth;
                                            }
                                            else
                                            {
                                                _8302 = _8295;
                                            }
                                            [branch]
                                            if (_8302)
                                            {
                                                float _8310 = mix_rand;
                                                float _8314 = (_8310 - _8159) / param_131;
                                                mix_rand = _8314;
                                                float4 F;
                                                float3 V;
                                                [branch]
                                                if (_8314 < _7444)
                                                {
                                                    float3 param_219 = T;
                                                    float3 param_220 = B;
                                                    float3 param_221 = N;
                                                    float3 param_222 = _4766;
                                                    float3 param_223;
                                                    float4 _8334 = Sample_GGXSpecular_BSDF(param_219, param_220, param_221, param_222, roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, _6160, _6170, param_223);
                                                    V = param_223;
                                                    F = _8334;
                                                    _9417 = ray.ray_depth + 256;
                                                    float3 param_224 = _4889;
                                                    float3 param_225 = plane_N;
                                                    float3 _8345 = offset_ray(param_224, param_225);
                                                    _9696 = _8345.x;
                                                    _9697 = _8345.y;
                                                    _9698 = _8345.z;
                                                }
                                                else
                                                {
                                                    float3 param_226 = T;
                                                    float3 param_227 = B;
                                                    float3 param_228 = N;
                                                    float3 param_229 = _4766;
                                                    float param_230 = _7473;
                                                    float param_231 = _7421;
                                                    float3 param_232 = base_color;
                                                    float param_233 = _6160;
                                                    float param_234 = _6170;
                                                    float4 param_235;
                                                    float4 _8376 = Sample_GGXRefraction_BSDF(param_226, param_227, param_228, param_229, param_230, param_231, param_232, param_233, param_234, param_235);
                                                    F = _8376;
                                                    V = param_235.xyz;
                                                    _9417 = ray.ray_depth + 65536;
                                                    float3 param_236 = _4889;
                                                    float3 param_237 = -plane_N;
                                                    float3 _8390 = offset_ray(param_236, param_237);
                                                    _9696 = _8390.x;
                                                    _9697 = _8390.y;
                                                    _9698 = _8390.z;
                                                }
                                                float4 _10140 = F;
                                                float _8403 = _10140.w * param_131;
                                                float4 _10142 = _10140;
                                                _10142.w = _8403;
                                                F = _10142;
                                                _9710 = ((ray.c[0] * _10140.x) * mix_weight) / _8403;
                                                _9711 = ((ray.c[1] * _10140.y) * mix_weight) / _8403;
                                                _9712 = ((ray.c[2] * _10140.z) * mix_weight) / _8403;
                                                _9412 = _8403;
                                                _9703 = V.x;
                                                _9704 = V.y;
                                                _9705 = V.z;
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
        float _8463 = max(_9710, max(_9711, _9712));
        float _8476;
        if (_5705 >= _3224_g_params.termination_start_depth)
        {
            _8476 = max(0.0500000007450580596923828125f, 1.0f - _8463);
        }
        else
        {
            _8476 = 0.0f;
        }
        bool _8490 = (frac(asfloat(_3219.Load((_3224_g_params.hi + 6) * 4 + 0)) + _5668) >= _8476) && (_8463 > 0.0f);
        bool _8496;
        if (_8490)
        {
            _8496 = _9412 > 0.0f;
        }
        else
        {
            _8496 = _8490;
        }
        [branch]
        if (_8496)
        {
            float _8500 = 1.0f - _8476;
            float _8502 = _9710;
            float _8503 = _8502 / _8500;
            _9710 = _8503;
            float _8508 = _9711;
            float _8509 = _8508 / _8500;
            _9711 = _8509;
            float _8514 = _9712;
            float _8515 = _8514 / _8500;
            _9712 = _8515;
            uint _8519;
            _6301.InterlockedAdd(0, 1u, _8519);
            _8527.Store(_8519 * 56 + 0, asuint(_9696));
            _8527.Store(_8519 * 56 + 4, asuint(_9697));
            _8527.Store(_8519 * 56 + 8, asuint(_9698));
            _8527.Store(_8519 * 56 + 12, asuint(_9703));
            _8527.Store(_8519 * 56 + 16, asuint(_9704));
            _8527.Store(_8519 * 56 + 20, asuint(_9705));
            _8527.Store(_8519 * 56 + 24, asuint(_9412));
            _8527.Store(_8519 * 56 + 28, asuint(_8503));
            _8527.Store(_8519 * 56 + 32, asuint(_8509));
            _8527.Store(_8519 * 56 + 36, asuint(_8515));
            _8527.Store(_8519 * 56 + 40, asuint(_5652));
            _8527.Store(_8519 * 56 + 44, asuint(ray.cone_spread));
            _8527.Store(_8519 * 56 + 48, uint(ray.xy));
            _8527.Store(_8519 * 56 + 52, uint(_9417));
        }
        _8738 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _8738;
}

void comp_main()
{
    do
    {
        bool _8595 = gl_GlobalInvocationID.x >= _3224_g_params.img_size.x;
        bool _8604;
        if (!_8595)
        {
            _8604 = gl_GlobalInvocationID.y >= _3224_g_params.img_size.y;
        }
        else
        {
            _8604 = _8595;
        }
        if (_8604)
        {
            break;
        }
        int _8611 = int(gl_GlobalInvocationID.x);
        int _8615 = int(gl_GlobalInvocationID.y);
        int _8623 = (_8615 * int(_3224_g_params.img_size.x)) + _8611;
        hit_data_t _8635;
        _8635.mask = int(_8631.Load(_8623 * 24 + 0));
        _8635.obj_index = int(_8631.Load(_8623 * 24 + 4));
        _8635.prim_index = int(_8631.Load(_8623 * 24 + 8));
        _8635.t = asfloat(_8631.Load(_8623 * 24 + 12));
        _8635.u = asfloat(_8631.Load(_8623 * 24 + 16));
        _8635.v = asfloat(_8631.Load(_8623 * 24 + 20));
        ray_data_t _8655;
        [unroll]
        for (int _119ident = 0; _119ident < 3; _119ident++)
        {
            _8655.o[_119ident] = asfloat(_8652.Load(_119ident * 4 + _8623 * 56 + 0));
        }
        [unroll]
        for (int _120ident = 0; _120ident < 3; _120ident++)
        {
            _8655.d[_120ident] = asfloat(_8652.Load(_120ident * 4 + _8623 * 56 + 12));
        }
        _8655.pdf = asfloat(_8652.Load(_8623 * 56 + 24));
        [unroll]
        for (int _121ident = 0; _121ident < 3; _121ident++)
        {
            _8655.c[_121ident] = asfloat(_8652.Load(_121ident * 4 + _8623 * 56 + 28));
        }
        _8655.cone_width = asfloat(_8652.Load(_8623 * 56 + 40));
        _8655.cone_spread = asfloat(_8652.Load(_8623 * 56 + 44));
        _8655.xy = int(_8652.Load(_8623 * 56 + 48));
        _8655.ray_depth = int(_8652.Load(_8623 * 56 + 52));
        int param = _8623;
        hit_data_t _8799 = { _8635.mask, _8635.obj_index, _8635.prim_index, _8635.t, _8635.u, _8635.v };
        hit_data_t param_1 = _8799;
        float _8837[3] = { _8655.c[0], _8655.c[1], _8655.c[2] };
        float _8830[3] = { _8655.d[0], _8655.d[1], _8655.d[2] };
        float _8823[3] = { _8655.o[0], _8655.o[1], _8655.o[2] };
        ray_data_t _8816 = { _8823, _8830, _8655.pdf, _8837, _8655.cone_width, _8655.cone_spread, _8655.xy, _8655.ray_depth };
        ray_data_t param_2 = _8816;
        float3 _8697 = ShadeSurface(param, param_1, param_2);
        g_out_img[int2(_8611, _8615)] = float4(_8697, 1.0f);
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
