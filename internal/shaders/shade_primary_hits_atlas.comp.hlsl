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

ByteAddressBuffer _855 : register(t20, space0);
ByteAddressBuffer _3218 : register(t15, space0);
ByteAddressBuffer _3255 : register(t6, space0);
ByteAddressBuffer _3259 : register(t7, space0);
ByteAddressBuffer _4172 : register(t11, space0);
ByteAddressBuffer _4197 : register(t13, space0);
ByteAddressBuffer _4201 : register(t14, space0);
ByteAddressBuffer _4512 : register(t10, space0);
ByteAddressBuffer _4516 : register(t9, space0);
ByteAddressBuffer _5299 : register(t12, space0);
RWByteAddressBuffer _6375 : register(u3, space0);
RWByteAddressBuffer _6385 : register(u2, space0);
RWByteAddressBuffer _8601 : register(u1, space0);
ByteAddressBuffer _8705 : register(t4, space0);
ByteAddressBuffer _8726 : register(t5, space0);
ByteAddressBuffer _8805 : register(t8, space0);
cbuffer UniformParams
{
    Params _3223_g_params : packoffset(c0);
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
    uint _8936[14] = t.pos;
    uint _8939[14] = t.pos;
    uint _948 = t.size & 16383u;
    uint _951 = t.size >> uint(16);
    uint _952 = _951 & 16383u;
    float2 size = float2(float(_948), float(_952));
    if ((_951 & 32768u) != 0u)
    {
        size = float2(float(_948 >> uint(mip_level)), float(_952 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_8936[mip_level] & 65535u), float((_8939[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
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
    float u = frac(phi * 0.15915493667125701904296875f);
    [flatten]
    if (dir.z < 0.0f)
    {
        u = 1.0f - u;
    }
    float2 _1167 = TransformUV(float2(u, acos(clamp(dir.y, -1.0f, 1.0f)) * 0.3183098733425140380859375f), t, 0);
    uint _1174 = t.atlas;
    int3 _1183 = int3(int2(_1167), int(t.page[0] & 255u));
    float2 _1230 = frac(_1167);
    float4 param = g_atlases[NonUniformResourceIndex(_1174)].Load(int4(_1183, 0), int2(0, 0));
    float4 param_1 = g_atlases[NonUniformResourceIndex(_1174)].Load(int4(_1183, 0), int2(1, 0));
    float4 param_2 = g_atlases[NonUniformResourceIndex(_1174)].Load(int4(_1183, 0), int2(0, 1));
    float4 param_3 = g_atlases[NonUniformResourceIndex(_1174)].Load(int4(_1183, 0), int2(1, 1));
    float _1250 = _1230.x;
    float _1255 = 1.0f - _1250;
    float _1271 = _1230.y;
    return (((rgbe_to_rgb(param_3) * _1250) + (rgbe_to_rgb(param_2) * _1255)) * _1271) + (((rgbe_to_rgb(param_1) * _1250) + (rgbe_to_rgb(param) * _1255)) * (1.0f - _1271));
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
    float _1284 = a * a;
    return _1284 / mad(b, b, _1284);
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
    uint _8944[4];
    _8944[0] = _985.page[0];
    _8944[1] = _985.page[1];
    _8944[2] = _985.page[2];
    _8944[3] = _985.page[3];
    uint _8980[14] = { _985.pos[0], _985.pos[1], _985.pos[2], _985.pos[3], _985.pos[4], _985.pos[5], _985.pos[6], _985.pos[7], _985.pos[8], _985.pos[9], _985.pos[10], _985.pos[11], _985.pos[12], _985.pos[13] };
    atlas_texture_t _8950 = { _985.size, _985.atlas, _8944, _8980 };
    uint _1055 = _985.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_1055)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_1055)], float3(TransformUV(uvs, _8950, lod) * 0.000118371215648949146270751953125f.xx, float((_8944[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
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
        float4 _9956 = res;
        _9956.x = _1095.x;
        float4 _9958 = _9956;
        _9958.y = _1095.y;
        float4 _9960 = _9958;
        _9960.z = _1095.z;
        res = _9960;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod)
{
    return SampleBilinear(index, uvs, lod, false, false);
}

float fresnel_dielectric_cos(float cosi, float eta)
{
    float _1316 = abs(cosi);
    float _1325 = mad(_1316, _1316, mad(eta, eta, -1.0f));
    float g = _1325;
    float result;
    if (_1325 > 0.0f)
    {
        float _1330 = g;
        float _1331 = sqrt(_1330);
        g = _1331;
        float _1335 = _1331 - _1316;
        float _1338 = _1331 + _1316;
        float _1339 = _1335 / _1338;
        float _1353 = mad(_1316, _1338, -1.0f) / mad(_1316, _1335, 1.0f);
        result = ((0.5f * _1339) * _1339) * mad(_1353, _1353, 1.0f);
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
    float3 _8817;
    do
    {
        float _1389 = min(0.89999997615814208984375f * dot(Ng, I), 0.00999999977648258209228515625f);
        if (dot(Ng, (N * (2.0f * dot(N, I))) - I) >= _1389)
        {
            _8817 = N;
            break;
        }
        float3 _1409 = normalize(N - (Ng * dot(N, Ng)));
        float _1413 = dot(I, _1409);
        float _1417 = dot(I, Ng);
        float _1429 = mad(_1413, _1413, _1417 * _1417);
        float param = (_1413 * _1413) * mad(-_1389, _1389, _1429);
        float _1439 = safe_sqrtf(param);
        float _1445 = mad(_1417, _1389, _1429);
        float _1448 = 0.5f / _1429;
        float _1453 = _1439 + _1445;
        float _1454 = _1448 * _1453;
        float _1460 = (-_1439) + _1445;
        float _1461 = _1448 * _1460;
        bool _1469 = (_1454 > 9.9999997473787516355514526367188e-06f) && (_1454 <= 1.000010013580322265625f);
        bool valid1 = _1469;
        bool _1475 = (_1461 > 9.9999997473787516355514526367188e-06f) && (_1461 <= 1.000010013580322265625f);
        bool valid2 = _1475;
        float2 N_new;
        if (_1469 && _1475)
        {
            float _10264 = (-0.5f) / _1429;
            float param_1 = mad(_10264, _1453, 1.0f);
            float _1485 = safe_sqrtf(param_1);
            float param_2 = _1454;
            float _1488 = safe_sqrtf(param_2);
            float2 _1489 = float2(_1485, _1488);
            float param_3 = mad(_10264, _1460, 1.0f);
            float _1494 = safe_sqrtf(param_3);
            float param_4 = _1461;
            float _1497 = safe_sqrtf(param_4);
            float2 _1498 = float2(_1494, _1497);
            float _10266 = -_1417;
            float _1514 = mad(2.0f * mad(_1485, _1413, _1488 * _1417), _1488, _10266);
            float _1530 = mad(2.0f * mad(_1494, _1413, _1497 * _1417), _1497, _10266);
            bool _1532 = _1514 >= 9.9999997473787516355514526367188e-06f;
            valid1 = _1532;
            bool _1534 = _1530 >= 9.9999997473787516355514526367188e-06f;
            valid2 = _1534;
            if (_1532 && _1534)
            {
                bool2 _1547 = (_1514 < _1530).xx;
                N_new = float2(_1547.x ? _1489.x : _1498.x, _1547.y ? _1489.y : _1498.y);
            }
            else
            {
                bool2 _1555 = (_1514 > _1530).xx;
                N_new = float2(_1555.x ? _1489.x : _1498.x, _1555.y ? _1489.y : _1498.y);
            }
        }
        else
        {
            if (!(valid1 || valid2))
            {
                _8817 = Ng;
                break;
            }
            float _1567 = valid1 ? _1454 : _1461;
            float param_5 = 1.0f - _1567;
            float param_6 = _1567;
            N_new = float2(safe_sqrtf(param_5), safe_sqrtf(param_6));
        }
        _8817 = (_1409 * N_new.x) + (Ng * N_new.y);
        break;
    } while(false);
    return _8817;
}

float3 rotate_around_axis(float3 p, float3 axis, float angle)
{
    float _1640 = cos(angle);
    float _1643 = sin(angle);
    float _1647 = 1.0f - _1640;
    return float3(mad(mad(_1647 * axis.x, axis.z, axis.y * _1643), p.z, mad(mad(_1647 * axis.x, axis.x, _1640), p.x, mad(_1647 * axis.x, axis.y, -(axis.z * _1643)) * p.y)), mad(mad(_1647 * axis.y, axis.z, -(axis.x * _1643)), p.z, mad(mad(_1647 * axis.x, axis.y, axis.z * _1643), p.x, mad(_1647 * axis.y, axis.y, _1640) * p.y)), mad(mad(_1647 * axis.z, axis.z, _1640), p.z, mad(mad(_1647 * axis.x, axis.z, -(axis.y * _1643)), p.x, mad(_1647 * axis.y, axis.z, axis.x * _1643) * p.y)));
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
    float3 _8842;
    do
    {
        float2 _3133 = (float2(r1, r2) * 2.0f) - 1.0f.xx;
        float _3135 = _3133.x;
        bool _3136 = _3135 == 0.0f;
        bool _3142;
        if (_3136)
        {
            _3142 = _3133.y == 0.0f;
        }
        else
        {
            _3142 = _3136;
        }
        if (_3142)
        {
            _8842 = N;
            break;
        }
        float _3151 = _3133.y;
        float r;
        float theta;
        if (abs(_3135) > abs(_3151))
        {
            r = _3135;
            theta = 0.785398185253143310546875f * (_3151 / _3135);
        }
        else
        {
            r = _3151;
            theta = 1.57079637050628662109375f * mad(-0.5f, _3135 / _3151, 1.0f);
        }
        float3 param;
        float3 param_1;
        create_tbn(N, param, param_1);
        _8842 = (N + (param * ((radius * r) * cos(theta)))) + (param_1 * ((radius * r) * sin(theta)));
        break;
    } while(false);
    return _8842;
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
            float2 _9943 = origin;
            _9943.x = origin.x + _step;
            origin = _9943;
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
            float2 _9946 = origin;
            _9946.y = origin.y + _step;
            origin = _9946;
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
    float _3234 = frac(asfloat(_3218.Load((_3223_g_params.hi + 3) * 4 + 0)) + sample_off.x);
    float _3239 = float(_3223_g_params.li_count);
    uint _3246 = min(uint(_3234 * _3239), uint(_3223_g_params.li_count - 1));
    light_t _3266;
    _3266.type_and_param0 = _3255.Load4(_3259.Load(_3246 * 4 + 0) * 64 + 0);
    _3266.param1 = asfloat(_3255.Load4(_3259.Load(_3246 * 4 + 0) * 64 + 16));
    _3266.param2 = asfloat(_3255.Load4(_3259.Load(_3246 * 4 + 0) * 64 + 32));
    _3266.param3 = asfloat(_3255.Load4(_3259.Load(_3246 * 4 + 0) * 64 + 48));
    ls.col = asfloat(_3266.type_and_param0.yzw);
    ls.col *= _3239;
    ls.cast_shadow = float((_3266.type_and_param0.x & 32u) != 0u);
    uint _3300 = _3266.type_and_param0.x & 31u;
    [branch]
    if (_3300 == 0u)
    {
        float _3314 = frac(asfloat(_3218.Load((_3223_g_params.hi + 4) * 4 + 0)) + sample_off.x);
        float3 _3330 = P - _3266.param1.xyz;
        float3 _3337 = _3330 / length(_3330).xxx;
        float _3344 = sqrt(clamp(mad(-_3314, _3314, 1.0f), 0.0f, 1.0f));
        float _3347 = 6.283185482025146484375f * frac(asfloat(_3218.Load((_3223_g_params.hi + 5) * 4 + 0)) + sample_off.y);
        float3 sampled_dir = float3(_3344 * cos(_3347), _3344 * sin(_3347), _3314);
        float3 param;
        float3 param_1;
        create_tbn(_3337, param, param_1);
        float3 _10023 = sampled_dir;
        float3 _3380 = ((param * _10023.x) + (param_1 * _10023.y)) + (_3337 * _10023.z);
        sampled_dir = _3380;
        float3 _3389 = _3266.param1.xyz + (_3380 * _3266.param2.w);
        ls.L = _3389 - P;
        ls.dist = length(ls.L);
        ls.L /= ls.dist.xxx;
        ls.area = _3266.param1.w;
        float _3420 = abs(dot(ls.L, normalize(_3389 - _3266.param1.xyz)));
        [flatten]
        if (_3420 > 0.0f)
        {
            ls.pdf = (ls.dist * ls.dist) / ((0.5f * ls.area) * _3420);
        }
        [branch]
        if (_3266.param3.x > 0.0f)
        {
            float _3449 = -dot(ls.L, _3266.param2.xyz);
            if (_3449 > 0.0f)
            {
                ls.col *= clamp((_3266.param3.x - acos(clamp(_3449, 0.0f, 1.0f))) / _3266.param3.y, 0.0f, 1.0f);
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
        if (_3300 == 2u)
        {
            ls.L = _3266.param1.xyz;
            if (_3266.param1.w != 0.0f)
            {
                float param_2 = frac(asfloat(_3218.Load((_3223_g_params.hi + 4) * 4 + 0)) + sample_off.x);
                float param_3 = frac(asfloat(_3218.Load((_3223_g_params.hi + 5) * 4 + 0)) + sample_off.y);
                float3 param_4 = ls.L;
                float param_5 = tan(_3266.param1.w);
                ls.L = normalize(MapToCone(param_2, param_3, param_4, param_5));
            }
            ls.area = 0.0f;
            ls.dist = 3402823346297367662189621542912.0f;
            ls.pdf = 1.0f;
            if ((_3266.type_and_param0.x & 64u) == 0u)
            {
                ls.area = 0.0f;
            }
        }
        else
        {
            [branch]
            if (_3300 == 4u)
            {
                float3 _3589 = ((_3266.param1.xyz + (_3266.param2.xyz * (frac(asfloat(_3218.Load((_3223_g_params.hi + 4) * 4 + 0)) + sample_off.x) - 0.5f))) + (_3266.param3.xyz * (frac(asfloat(_3218.Load((_3223_g_params.hi + 5) * 4 + 0)) + sample_off.y) - 0.5f))) - P;
                ls.dist = length(_3589);
                ls.L = _3589 / ls.dist.xxx;
                ls.area = _3266.param1.w;
                float _3612 = dot(-ls.L, normalize(cross(_3266.param2.xyz, _3266.param3.xyz)));
                if (_3612 > 0.0f)
                {
                    ls.pdf = (ls.dist * ls.dist) / (ls.area * _3612);
                }
                if ((_3266.type_and_param0.x & 64u) == 0u)
                {
                    ls.area = 0.0f;
                }
                [branch]
                if ((_3266.type_and_param0.w & 128u) != 0u)
                {
                    float3 env_col = _3223_g_params.env_col.xyz;
                    uint _3651 = asuint(_3223_g_params.env_col.w);
                    if (_3651 != 4294967295u)
                    {
                        atlas_texture_t _3659;
                        _3659.size = _855.Load(_3651 * 80 + 0);
                        _3659.atlas = _855.Load(_3651 * 80 + 4);
                        [unroll]
                        for (int _63ident = 0; _63ident < 4; _63ident++)
                        {
                            _3659.page[_63ident] = _855.Load(_63ident * 4 + _3651 * 80 + 8);
                        }
                        [unroll]
                        for (int _64ident = 0; _64ident < 14; _64ident++)
                        {
                            _3659.pos[_64ident] = _855.Load(_64ident * 4 + _3651 * 80 + 24);
                        }
                        uint _9135[14] = { _3659.pos[0], _3659.pos[1], _3659.pos[2], _3659.pos[3], _3659.pos[4], _3659.pos[5], _3659.pos[6], _3659.pos[7], _3659.pos[8], _3659.pos[9], _3659.pos[10], _3659.pos[11], _3659.pos[12], _3659.pos[13] };
                        uint _9106[4] = { _3659.page[0], _3659.page[1], _3659.page[2], _3659.page[3] };
                        atlas_texture_t _9016 = { _3659.size, _3659.atlas, _9106, _9135 };
                        float param_6 = _3223_g_params.env_rotation;
                        env_col *= SampleLatlong_RGBE(_9016, ls.L, param_6);
                    }
                    ls.col *= env_col;
                }
            }
            else
            {
                [branch]
                if (_3300 == 5u)
                {
                    float2 _3762 = (float2(frac(asfloat(_3218.Load((_3223_g_params.hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_3218.Load((_3223_g_params.hi + 5) * 4 + 0)) + sample_off.y)) * 2.0f) - 1.0f.xx;
                    float2 offset = _3762;
                    bool _3765 = _3762.x != 0.0f;
                    bool _3771;
                    if (_3765)
                    {
                        _3771 = offset.y != 0.0f;
                    }
                    else
                    {
                        _3771 = _3765;
                    }
                    if (_3771)
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
                        float _3804 = 0.5f * r;
                        offset = float2(_3804 * cos(theta), _3804 * sin(theta));
                    }
                    float3 _3830 = ((_3266.param1.xyz + (_3266.param2.xyz * offset.x)) + (_3266.param3.xyz * offset.y)) - P;
                    ls.dist = length(_3830);
                    ls.L = _3830 / ls.dist.xxx;
                    ls.area = _3266.param1.w;
                    float _3853 = dot(-ls.L, normalize(cross(_3266.param2.xyz, _3266.param3.xyz)));
                    [flatten]
                    if (_3853 > 0.0f)
                    {
                        ls.pdf = (ls.dist * ls.dist) / (ls.area * _3853);
                    }
                    if ((_3266.type_and_param0.x & 64u) == 0u)
                    {
                        ls.area = 0.0f;
                    }
                    [branch]
                    if ((_3266.type_and_param0.w & 128u) != 0u)
                    {
                        float3 env_col_1 = _3223_g_params.env_col.xyz;
                        uint _3889 = asuint(_3223_g_params.env_col.w);
                        if (_3889 != 4294967295u)
                        {
                            atlas_texture_t _3896;
                            _3896.size = _855.Load(_3889 * 80 + 0);
                            _3896.atlas = _855.Load(_3889 * 80 + 4);
                            [unroll]
                            for (int _65ident = 0; _65ident < 4; _65ident++)
                            {
                                _3896.page[_65ident] = _855.Load(_65ident * 4 + _3889 * 80 + 8);
                            }
                            [unroll]
                            for (int _66ident = 0; _66ident < 14; _66ident++)
                            {
                                _3896.pos[_66ident] = _855.Load(_66ident * 4 + _3889 * 80 + 24);
                            }
                            uint _9173[14] = { _3896.pos[0], _3896.pos[1], _3896.pos[2], _3896.pos[3], _3896.pos[4], _3896.pos[5], _3896.pos[6], _3896.pos[7], _3896.pos[8], _3896.pos[9], _3896.pos[10], _3896.pos[11], _3896.pos[12], _3896.pos[13] };
                            uint _9144[4] = { _3896.page[0], _3896.page[1], _3896.page[2], _3896.page[3] };
                            atlas_texture_t _9025 = { _3896.size, _3896.atlas, _9144, _9173 };
                            float param_7 = _3223_g_params.env_rotation;
                            env_col_1 *= SampleLatlong_RGBE(_9025, ls.L, param_7);
                        }
                        ls.col *= env_col_1;
                    }
                }
                else
                {
                    [branch]
                    if (_3300 == 3u)
                    {
                        float3 _3997 = normalize(cross(P - _3266.param1.xyz, _3266.param3.xyz));
                        float _4004 = 3.1415927410125732421875f * frac(asfloat(_3218.Load((_3223_g_params.hi + 4) * 4 + 0)) + sample_off.x);
                        float3 _4033 = ((_3266.param1.xyz + (((_3997 * cos(_4004)) + (cross(_3997, _3266.param3.xyz) * sin(_4004))) * _3266.param2.w)) + ((_3266.param3.xyz * (frac(asfloat(_3218.Load((_3223_g_params.hi + 5) * 4 + 0)) + sample_off.y) - 0.5f)) * _3266.param3.w)) - P;
                        ls.dist = length(_4033);
                        ls.L = _4033 / ls.dist.xxx;
                        ls.area = _3266.param1.w;
                        float _4052 = 1.0f - abs(dot(ls.L, _3266.param3.xyz));
                        [flatten]
                        if (_4052 != 0.0f)
                        {
                            ls.pdf = (ls.dist * ls.dist) / (ls.area * _4052);
                        }
                        if ((_3266.type_and_param0.x & 64u) == 0u)
                        {
                            ls.area = 0.0f;
                        }
                        [branch]
                        if ((_3266.type_and_param0.w & 128u) != 0u)
                        {
                            float3 env_col_2 = _3223_g_params.env_col.xyz;
                            uint _4088 = asuint(_3223_g_params.env_col.w);
                            if (_4088 != 4294967295u)
                            {
                                atlas_texture_t _4095;
                                _4095.size = _855.Load(_4088 * 80 + 0);
                                _4095.atlas = _855.Load(_4088 * 80 + 4);
                                [unroll]
                                for (int _67ident = 0; _67ident < 4; _67ident++)
                                {
                                    _4095.page[_67ident] = _855.Load(_67ident * 4 + _4088 * 80 + 8);
                                }
                                [unroll]
                                for (int _68ident = 0; _68ident < 14; _68ident++)
                                {
                                    _4095.pos[_68ident] = _855.Load(_68ident * 4 + _4088 * 80 + 24);
                                }
                                uint _9211[14] = { _4095.pos[0], _4095.pos[1], _4095.pos[2], _4095.pos[3], _4095.pos[4], _4095.pos[5], _4095.pos[6], _4095.pos[7], _4095.pos[8], _4095.pos[9], _4095.pos[10], _4095.pos[11], _4095.pos[12], _4095.pos[13] };
                                uint _9182[4] = { _4095.page[0], _4095.page[1], _4095.page[2], _4095.page[3] };
                                atlas_texture_t _9034 = { _4095.size, _4095.atlas, _9182, _9211 };
                                float param_8 = _3223_g_params.env_rotation;
                                env_col_2 *= SampleLatlong_RGBE(_9034, ls.L, param_8);
                            }
                            ls.col *= env_col_2;
                        }
                    }
                    else
                    {
                        [branch]
                        if (_3300 == 6u)
                        {
                            uint _4164 = asuint(_3266.param1.x);
                            transform_t _4178;
                            _4178.xform = asfloat(uint4x4(_4172.Load4(asuint(_3266.param1.y) * 128 + 0), _4172.Load4(asuint(_3266.param1.y) * 128 + 16), _4172.Load4(asuint(_3266.param1.y) * 128 + 32), _4172.Load4(asuint(_3266.param1.y) * 128 + 48)));
                            _4178.inv_xform = asfloat(uint4x4(_4172.Load4(asuint(_3266.param1.y) * 128 + 64), _4172.Load4(asuint(_3266.param1.y) * 128 + 80), _4172.Load4(asuint(_3266.param1.y) * 128 + 96), _4172.Load4(asuint(_3266.param1.y) * 128 + 112)));
                            uint _4203 = _4164 * 3u;
                            vertex_t _4209;
                            [unroll]
                            for (int _69ident = 0; _69ident < 3; _69ident++)
                            {
                                _4209.p[_69ident] = asfloat(_4197.Load(_69ident * 4 + _4201.Load(_4203 * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _70ident = 0; _70ident < 3; _70ident++)
                            {
                                _4209.n[_70ident] = asfloat(_4197.Load(_70ident * 4 + _4201.Load(_4203 * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _71ident = 0; _71ident < 3; _71ident++)
                            {
                                _4209.b[_71ident] = asfloat(_4197.Load(_71ident * 4 + _4201.Load(_4203 * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _72ident = 0; _72ident < 2; _72ident++)
                            {
                                [unroll]
                                for (int _73ident = 0; _73ident < 2; _73ident++)
                                {
                                    _4209.t[_72ident][_73ident] = asfloat(_4197.Load(_73ident * 4 + _72ident * 8 + _4201.Load(_4203 * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4258;
                            [unroll]
                            for (int _74ident = 0; _74ident < 3; _74ident++)
                            {
                                _4258.p[_74ident] = asfloat(_4197.Load(_74ident * 4 + _4201.Load((_4203 + 1u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _75ident = 0; _75ident < 3; _75ident++)
                            {
                                _4258.n[_75ident] = asfloat(_4197.Load(_75ident * 4 + _4201.Load((_4203 + 1u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _76ident = 0; _76ident < 3; _76ident++)
                            {
                                _4258.b[_76ident] = asfloat(_4197.Load(_76ident * 4 + _4201.Load((_4203 + 1u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _77ident = 0; _77ident < 2; _77ident++)
                            {
                                [unroll]
                                for (int _78ident = 0; _78ident < 2; _78ident++)
                                {
                                    _4258.t[_77ident][_78ident] = asfloat(_4197.Load(_78ident * 4 + _77ident * 8 + _4201.Load((_4203 + 1u) * 4 + 0) * 52 + 36));
                                }
                            }
                            vertex_t _4304;
                            [unroll]
                            for (int _79ident = 0; _79ident < 3; _79ident++)
                            {
                                _4304.p[_79ident] = asfloat(_4197.Load(_79ident * 4 + _4201.Load((_4203 + 2u) * 4 + 0) * 52 + 0));
                            }
                            [unroll]
                            for (int _80ident = 0; _80ident < 3; _80ident++)
                            {
                                _4304.n[_80ident] = asfloat(_4197.Load(_80ident * 4 + _4201.Load((_4203 + 2u) * 4 + 0) * 52 + 12));
                            }
                            [unroll]
                            for (int _81ident = 0; _81ident < 3; _81ident++)
                            {
                                _4304.b[_81ident] = asfloat(_4197.Load(_81ident * 4 + _4201.Load((_4203 + 2u) * 4 + 0) * 52 + 24));
                            }
                            [unroll]
                            for (int _82ident = 0; _82ident < 2; _82ident++)
                            {
                                [unroll]
                                for (int _83ident = 0; _83ident < 2; _83ident++)
                                {
                                    _4304.t[_82ident][_83ident] = asfloat(_4197.Load(_83ident * 4 + _82ident * 8 + _4201.Load((_4203 + 2u) * 4 + 0) * 52 + 36));
                                }
                            }
                            float3 _4350 = float3(_4209.p[0], _4209.p[1], _4209.p[2]);
                            float3 _4358 = float3(_4258.p[0], _4258.p[1], _4258.p[2]);
                            float3 _4366 = float3(_4304.p[0], _4304.p[1], _4304.p[2]);
                            float _4395 = sqrt(frac(asfloat(_3218.Load((_3223_g_params.hi + 4) * 4 + 0)) + sample_off.x));
                            float _4405 = frac(asfloat(_3218.Load((_3223_g_params.hi + 5) * 4 + 0)) + sample_off.y);
                            float _4409 = 1.0f - _4395;
                            float _4414 = 1.0f - _4405;
                            float3 _4461 = mul(float4(cross(_4358 - _4350, _4366 - _4350), 0.0f), _4178.xform).xyz;
                            ls.area = 0.5f * length(_4461);
                            float3 _4471 = mul(float4((_4350 * _4409) + (((_4358 * _4414) + (_4366 * _4405)) * _4395), 1.0f), _4178.xform).xyz - P;
                            ls.dist = length(_4471);
                            ls.L = _4471 / ls.dist.xxx;
                            float _4486 = abs(dot(ls.L, normalize(_4461)));
                            [flatten]
                            if (_4486 > 0.0f)
                            {
                                ls.pdf = (ls.dist * ls.dist) / (ls.area * _4486);
                            }
                            material_t _4525;
                            [unroll]
                            for (int _84ident = 0; _84ident < 5; _84ident++)
                            {
                                _4525.textures[_84ident] = _4512.Load(_84ident * 4 + ((_4516.Load(_4164 * 4 + 0) >> 16u) & 16383u) * 80 + 0);
                            }
                            [unroll]
                            for (int _85ident = 0; _85ident < 3; _85ident++)
                            {
                                _4525.base_color[_85ident] = asfloat(_4512.Load(_85ident * 4 + ((_4516.Load(_4164 * 4 + 0) >> 16u) & 16383u) * 80 + 20));
                            }
                            _4525.flags = _4512.Load(((_4516.Load(_4164 * 4 + 0) >> 16u) & 16383u) * 80 + 32);
                            _4525.type = _4512.Load(((_4516.Load(_4164 * 4 + 0) >> 16u) & 16383u) * 80 + 36);
                            _4525.tangent_rotation_or_strength = asfloat(_4512.Load(((_4516.Load(_4164 * 4 + 0) >> 16u) & 16383u) * 80 + 40));
                            _4525.roughness_and_anisotropic = _4512.Load(((_4516.Load(_4164 * 4 + 0) >> 16u) & 16383u) * 80 + 44);
                            _4525.int_ior = asfloat(_4512.Load(((_4516.Load(_4164 * 4 + 0) >> 16u) & 16383u) * 80 + 48));
                            _4525.ext_ior = asfloat(_4512.Load(((_4516.Load(_4164 * 4 + 0) >> 16u) & 16383u) * 80 + 52));
                            _4525.sheen_and_sheen_tint = _4512.Load(((_4516.Load(_4164 * 4 + 0) >> 16u) & 16383u) * 80 + 56);
                            _4525.tint_and_metallic = _4512.Load(((_4516.Load(_4164 * 4 + 0) >> 16u) & 16383u) * 80 + 60);
                            _4525.transmission_and_transmission_roughness = _4512.Load(((_4516.Load(_4164 * 4 + 0) >> 16u) & 16383u) * 80 + 64);
                            _4525.specular_and_specular_tint = _4512.Load(((_4516.Load(_4164 * 4 + 0) >> 16u) & 16383u) * 80 + 68);
                            _4525.clearcoat_and_clearcoat_roughness = _4512.Load(((_4516.Load(_4164 * 4 + 0) >> 16u) & 16383u) * 80 + 72);
                            _4525.normal_map_strength_unorm = _4512.Load(((_4516.Load(_4164 * 4 + 0) >> 16u) & 16383u) * 80 + 76);
                            if ((_4525.flags & 4u) == 0u)
                            {
                                if (_4525.textures[1] != 4294967295u)
                                {
                                    ls.col *= SampleBilinear(_4525.textures[1], (float2(_4209.t[0][0], _4209.t[0][1]) * _4409) + (((float2(_4258.t[0][0], _4258.t[0][1]) * _4414) + (float2(_4304.t[0][0], _4304.t[0][1]) * _4405)) * _4395), 0).xyz;
                                }
                            }
                            else
                            {
                                float3 env_col_3 = _3223_g_params.env_col.xyz;
                                uint _4599 = asuint(_3223_g_params.env_col.w);
                                if (_4599 != 4294967295u)
                                {
                                    atlas_texture_t _4606;
                                    _4606.size = _855.Load(_4599 * 80 + 0);
                                    _4606.atlas = _855.Load(_4599 * 80 + 4);
                                    [unroll]
                                    for (int _86ident = 0; _86ident < 4; _86ident++)
                                    {
                                        _4606.page[_86ident] = _855.Load(_86ident * 4 + _4599 * 80 + 8);
                                    }
                                    [unroll]
                                    for (int _87ident = 0; _87ident < 14; _87ident++)
                                    {
                                        _4606.pos[_87ident] = _855.Load(_87ident * 4 + _4599 * 80 + 24);
                                    }
                                    uint _9296[14] = { _4606.pos[0], _4606.pos[1], _4606.pos[2], _4606.pos[3], _4606.pos[4], _4606.pos[5], _4606.pos[6], _4606.pos[7], _4606.pos[8], _4606.pos[9], _4606.pos[10], _4606.pos[11], _4606.pos[12], _4606.pos[13] };
                                    uint _9267[4] = { _4606.page[0], _4606.page[1], _4606.page[2], _4606.page[3] };
                                    atlas_texture_t _9088 = { _4606.size, _4606.atlas, _9267, _9296 };
                                    float param_9 = _3223_g_params.env_rotation;
                                    env_col_3 *= SampleLatlong_RGBE(_9088, ls.L, param_9);
                                }
                                ls.col *= env_col_3;
                            }
                        }
                        else
                        {
                            [branch]
                            if (_3300 == 7u)
                            {
                                float4 _4709 = Sample_EnvQTree(_3223_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3223_g_params.env_qtree_levels, mad(_3234, _3239, -float(_3246)), frac(asfloat(_3218.Load((_3223_g_params.hi + 4) * 4 + 0)) + sample_off.x), frac(asfloat(_3218.Load((_3223_g_params.hi + 5) * 4 + 0)) + sample_off.y));
                                ls.L = _4709.xyz;
                                ls.col *= _3223_g_params.env_col.xyz;
                                atlas_texture_t _4726;
                                _4726.size = _855.Load(asuint(_3223_g_params.env_col.w) * 80 + 0);
                                _4726.atlas = _855.Load(asuint(_3223_g_params.env_col.w) * 80 + 4);
                                [unroll]
                                for (int _88ident = 0; _88ident < 4; _88ident++)
                                {
                                    _4726.page[_88ident] = _855.Load(_88ident * 4 + asuint(_3223_g_params.env_col.w) * 80 + 8);
                                }
                                [unroll]
                                for (int _89ident = 0; _89ident < 14; _89ident++)
                                {
                                    _4726.pos[_89ident] = _855.Load(_89ident * 4 + asuint(_3223_g_params.env_col.w) * 80 + 24);
                                }
                                uint _9334[14] = { _4726.pos[0], _4726.pos[1], _4726.pos[2], _4726.pos[3], _4726.pos[4], _4726.pos[5], _4726.pos[6], _4726.pos[7], _4726.pos[8], _4726.pos[9], _4726.pos[10], _4726.pos[11], _4726.pos[12], _4726.pos[13] };
                                uint _9305[4] = { _4726.page[0], _4726.page[1], _4726.page[2], _4726.page[3] };
                                atlas_texture_t _9097 = { _4726.size, _4726.atlas, _9305, _9334 };
                                float param_10 = _3223_g_params.env_rotation;
                                ls.col *= SampleLatlong_RGBE(_9097, ls.L, param_10);
                                ls.area = 1.0f;
                                ls.dist = 3402823346297367662189621542912.0f;
                                ls.pdf = _4709.w;
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
    float _2214 = 1.0f / mad(0.904129683971405029296875f, roughness, 3.1415927410125732421875f);
    float _2226 = max(dot(N, L), 0.0f);
    float _2231 = max(dot(N, V), 0.0f);
    float _2239 = mad(-_2226, _2231, dot(L, V));
    float t = _2239;
    if (_2239 > 0.0f)
    {
        t /= (max(_2226, _2231) + 1.1754943508222875079687365372222e-38f);
    }
    return float4(base_color * (_2226 * mad(roughness * _2214, t, _2214)), 0.15915493667125701904296875f);
}

float3 offset_ray(float3 p, float3 n)
{
    int3 _1796 = int3(n * 128.0f);
    int _1804;
    if (p.x < 0.0f)
    {
        _1804 = -_1796.x;
    }
    else
    {
        _1804 = _1796.x;
    }
    int _1822;
    if (p.y < 0.0f)
    {
        _1822 = -_1796.y;
    }
    else
    {
        _1822 = _1796.y;
    }
    int _1840;
    if (p.z < 0.0f)
    {
        _1840 = -_1796.z;
    }
    else
    {
        _1840 = _1796.z;
    }
    float _1858;
    if (abs(p.x) < 0.03125f)
    {
        _1858 = mad(1.52587890625e-05f, n.x, p.x);
    }
    else
    {
        _1858 = asfloat(asint(p.x) + _1804);
    }
    float _1876;
    if (abs(p.y) < 0.03125f)
    {
        _1876 = mad(1.52587890625e-05f, n.y, p.y);
    }
    else
    {
        _1876 = asfloat(asint(p.y) + _1822);
    }
    float _1893;
    if (abs(p.z) < 0.03125f)
    {
        _1893 = mad(1.52587890625e-05f, n.z, p.z);
    }
    else
    {
        _1893 = asfloat(asint(p.z) + _1840);
    }
    return float3(_1858, _1876, _1893);
}

float3 world_from_tangent(float3 T, float3 B, float3 N, float3 V)
{
    return ((T * V.x) + (B * V.y)) + (N * V.z);
}

float4 Sample_OrenDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float rand_u, float rand_v, inout float3 out_V)
{
    float _2273 = 6.283185482025146484375f * rand_v;
    float _2285 = sqrt(mad(-rand_u, rand_u, 1.0f));
    float3 param = T;
    float3 param_1 = B;
    float3 param_2 = N;
    float3 param_3 = float3(_2285 * cos(_2273), _2285 * sin(_2273), rand_u);
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
    float _8847;
    do
    {
        if (H.z == 0.0f)
        {
            _8847 = 0.0f;
            break;
        }
        float _2100 = (-H.x) / (H.z * alpha_x);
        float _2106 = (-H.y) / (H.z * alpha_y);
        float _2115 = mad(_2106, _2106, mad(_2100, _2100, 1.0f));
        _8847 = 1.0f / (((((_2115 * _2115) * 3.1415927410125732421875f) * alpha_x) * alpha_y) * (((H.z * H.z) * H.z) * H.z));
        break;
    } while(false);
    return _8847;
}

float G1(float3 Ve, inout float alpha_x, inout float alpha_y)
{
    alpha_x *= alpha_x;
    alpha_y *= alpha_y;
    return 1.0f / mad((-1.0f) + sqrt(1.0f + (mad(alpha_x * Ve.x, Ve.x, (alpha_y * Ve.y) * Ve.y) / (Ve.z * Ve.z))), 0.5f, 1.0f);
}

float4 Evaluate_GGXSpecular_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float alpha_x, float alpha_y, float spec_ior, float spec_F0, float3 spec_col)
{
    float _2455 = D_GGX(sampled_normal_ts, alpha_x, alpha_y);
    float3 param = view_dir_ts;
    float param_1 = alpha_x;
    float param_2 = alpha_y;
    float _2463 = G1(param, param_1, param_2);
    float3 param_3 = reflected_dir_ts;
    float param_4 = alpha_x;
    float param_5 = alpha_y;
    float _2470 = G1(param_3, param_4, param_5);
    float param_6 = dot(view_dir_ts, sampled_normal_ts);
    float param_7 = spec_ior;
    float3 F = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param_6, param_7) - spec_F0) / (1.0f - spec_F0)).xxx);
    float _2498 = 4.0f * abs(view_dir_ts.z * reflected_dir_ts.z);
    float _2501;
    if (_2498 != 0.0f)
    {
        _2501 = (_2455 * (_2463 * _2470)) / _2498;
    }
    else
    {
        _2501 = 0.0f;
    }
    F *= _2501;
    float3 param_8 = view_dir_ts;
    float param_9 = alpha_x;
    float param_10 = alpha_y;
    float _2521 = G1(param_8, param_9, param_10);
    float pdf = ((_2455 * _2521) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2536 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2536 != 0.0f)
    {
        pdf /= _2536;
    }
    float3 _2547 = F;
    float3 _2548 = _2547 * max(reflected_dir_ts.z, 0.0f);
    F = _2548;
    return float4(_2548, pdf);
}

float3 SampleGGX_VNDF(float3 Ve, float alpha_x, float alpha_y, float U1, float U2)
{
    float3 _1918 = normalize(float3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
    float _1921 = _1918.x;
    float _1926 = _1918.y;
    float _1930 = mad(_1921, _1921, _1926 * _1926);
    float3 _1934;
    if (_1930 > 0.0f)
    {
        _1934 = float3(-_1926, _1921, 0.0f) / sqrt(_1930).xxx;
    }
    else
    {
        _1934 = float3(1.0f, 0.0f, 0.0f);
    }
    float _1956 = sqrt(U1);
    float _1959 = 6.283185482025146484375f * U2;
    float _1964 = _1956 * cos(_1959);
    float _1973 = 1.0f + _1918.z;
    float _1980 = mad(-_1964, _1964, 1.0f);
    float _1986 = mad(mad(-0.5f, _1973, 1.0f), sqrt(_1980), (0.5f * _1973) * (_1956 * sin(_1959)));
    float3 _2007 = ((_1934 * _1964) + (cross(_1918, _1934) * _1986)) + (_1918 * sqrt(max(0.0f, mad(-_1986, _1986, _1980))));
    return normalize(float3(alpha_x * _2007.x, alpha_y * _2007.y, max(0.0f, _2007.z)));
}

float4 Sample_GGXSpecular_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float anisotropic, float spec_ior, float spec_F0, float3 spec_col, float rand_u, float rand_v, inout float3 out_V)
{
    float4 _8822;
    do
    {
        float _2558 = roughness * roughness;
        float _2562 = sqrt(mad(-0.89999997615814208984375f, anisotropic, 1.0f));
        float _2566 = _2558 / _2562;
        float _2570 = _2558 * _2562;
        [branch]
        if ((_2566 * _2570) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2580 = reflect(I, N);
            float param = dot(_2580, N);
            float param_1 = spec_ior;
            float3 _2594 = lerp(spec_col, 1.0f.xxx, ((fresnel_dielectric_cos(param, param_1) - spec_F0) / (1.0f - spec_F0)).xxx);
            out_V = _2580;
            _8822 = float4(_2594.x * 1000000.0f, _2594.y * 1000000.0f, _2594.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2619 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = _2566;
        float param_7 = _2570;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2628 = SampleGGX_VNDF(_2619, param_6, param_7, param_8, param_9);
        float3 _2639 = normalize(reflect(-_2619, _2628));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2639;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2619;
        float3 param_15 = _2628;
        float3 param_16 = _2639;
        float param_17 = _2566;
        float param_18 = _2570;
        float param_19 = spec_ior;
        float param_20 = spec_F0;
        float3 param_21 = spec_col;
        _8822 = Evaluate_GGXSpecular_BSDF(param_14, param_15, param_16, param_17, param_18, param_19, param_20, param_21);
        break;
    } while(false);
    return _8822;
}

float4 Evaluate_GGXRefraction_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 refr_dir_ts, float roughness2, float eta, float3 refr_col)
{
    float4 _8827;
    do
    {
        bool _2861 = refr_dir_ts.z >= 0.0f;
        bool _2868;
        if (!_2861)
        {
            _2868 = view_dir_ts.z <= 0.0f;
        }
        else
        {
            _2868 = _2861;
        }
        if (_2868)
        {
            _8827 = 0.0f.xxxx;
            break;
        }
        float _2877 = D_GGX(sampled_normal_ts, roughness2, roughness2);
        float3 param = refr_dir_ts;
        float param_1 = roughness2;
        float param_2 = roughness2;
        float _2885 = G1(param, param_1, param_2);
        float3 param_3 = view_dir_ts;
        float param_4 = roughness2;
        float param_5 = roughness2;
        float _2893 = G1(param_3, param_4, param_5);
        float _2903 = mad(dot(view_dir_ts, sampled_normal_ts), eta, dot(refr_dir_ts, sampled_normal_ts));
        float _2913 = clamp(-dot(refr_dir_ts, sampled_normal_ts), 0.0f, 1.0f) / (_2903 * _2903);
        _8827 = float4(refr_col * (((((_2877 * _2893) * _2885) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _2913) / view_dir_ts.z), (((_2877 * _2885) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) * _2913) / view_dir_ts.z);
        break;
    } while(false);
    return _8827;
}

float4 Sample_GGXRefraction_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float eta, float3 refr_col, float rand_u, float rand_v, inout float4 out_V)
{
    float4 _8832;
    do
    {
        float _2957 = roughness * roughness;
        [branch]
        if ((_2957 * _2957) < 1.0000000116860974230803549289703e-07f)
        {
            float _2967 = dot(I, N);
            float _2968 = -_2967;
            float _2978 = mad(-(eta * eta), mad(_2967, _2968, 1.0f), 1.0f);
            if (_2978 < 0.0f)
            {
                _8832 = 0.0f.xxxx;
                break;
            }
            float _2990 = mad(eta, _2968, -sqrt(_2978));
            out_V = float4(normalize((I * eta) + (N * _2990)), _2990);
            _8832 = float4(refr_col.x * 1000000.0f, refr_col.y * 1000000.0f, refr_col.z * 1000000.0f, 1000000.0f);
            break;
        }
        float3 param = T;
        float3 param_1 = B;
        float3 param_2 = N;
        float3 param_3 = -I;
        float3 _3030 = normalize(tangent_from_world(param, param_1, param_2, param_3));
        float param_4 = _2957;
        float param_5 = _2957;
        float param_6 = rand_u;
        float param_7 = rand_v;
        float3 _3041 = SampleGGX_VNDF(_3030, param_4, param_5, param_6, param_7);
        float _3045 = dot(_3030, _3041);
        float _3055 = mad(-(eta * eta), mad(-_3045, _3045, 1.0f), 1.0f);
        if (_3055 < 0.0f)
        {
            _8832 = 0.0f.xxxx;
            break;
        }
        float _3067 = mad(eta, _3045, -sqrt(_3055));
        float3 _3077 = normalize((_3030 * (-eta)) + (_3041 * _3067));
        float3 param_8 = _3030;
        float3 param_9 = _3041;
        float3 param_10 = _3077;
        float param_11 = _2957;
        float param_12 = eta;
        float3 param_13 = refr_col;
        float3 param_14 = T;
        float3 param_15 = B;
        float3 param_16 = N;
        float3 param_17 = _3077;
        out_V = float4(world_from_tangent(param_14, param_15, param_16, param_17), _3067);
        _8832 = Evaluate_GGXRefraction_BSDF(param_8, param_9, param_10, param_11, param_12, param_13);
        break;
    } while(false);
    return _8832;
}

void get_lobe_weights(float base_color_lum, float spec_color_lum, float specular, float metallic, float transmission, float clearcoat, inout float out_diffuse_weight, inout float out_specular_weight, inout float out_clearcoat_weight, inout float out_refraction_weight)
{
    float _1590 = 1.0f - metallic;
    out_diffuse_weight = (base_color_lum * _1590) * (1.0f - transmission);
    float _1600;
    if ((specular != 0.0f) || (metallic != 0.0f))
    {
        _1600 = spec_color_lum * mad(-transmission, _1590, 1.0f);
    }
    else
    {
        _1600 = 0.0f;
    }
    out_specular_weight = _1600;
    out_clearcoat_weight = (0.25f * clearcoat) * _1590;
    out_refraction_weight = (transmission * _1590) * base_color_lum;
    float _1615 = out_diffuse_weight;
    float _1616 = out_specular_weight;
    float _1618 = out_clearcoat_weight;
    float _1621 = ((_1615 + _1616) + _1618) + out_refraction_weight;
    if (_1621 != 0.0f)
    {
        out_diffuse_weight /= _1621;
        out_specular_weight /= _1621;
        out_clearcoat_weight /= _1621;
        out_refraction_weight /= _1621;
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
    float _8852;
    do
    {
        float _2166 = dot(N, L);
        if (_2166 <= 0.0f)
        {
            _8852 = 0.0f;
            break;
        }
        float param = _2166;
        float param_1 = dot(N, V);
        float _2187 = dot(L, H);
        float _2195 = mad((2.0f * _2187) * _2187, roughness, 0.5f);
        _8852 = lerp(1.0f, _2195, schlick_weight(param)) * lerp(1.0f, _2195, schlick_weight(param_1));
        break;
    } while(false);
    return _8852;
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
    float3 _2336 = normalize(L + V);
    float3 H = _2336;
    if (dot(V, _2336) < 0.0f)
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
    float3 _2371 = diff_col;
    float3 _2372 = _2371 + (sheen_color * (3.1415927410125732421875f * schlick_weight(param_5)));
    diff_col = _2372;
    return float4(_2372, pdf);
}

float D_GTR1(float NDotH, float a)
{
    float _8857;
    do
    {
        if (a >= 1.0f)
        {
            _8857 = 0.3183098733425140380859375f;
            break;
        }
        float _2074 = mad(a, a, -1.0f);
        _8857 = _2074 / ((3.1415927410125732421875f * log(a * a)) * mad(_2074 * NDotH, NDotH, 1.0f));
        break;
    } while(false);
    return _8857;
}

float4 Evaluate_PrincipledClearcoat_BSDF(float3 view_dir_ts, float3 sampled_normal_ts, float3 reflected_dir_ts, float clearcoat_roughness2, float clearcoat_ior, float clearcoat_F0)
{
    float param = sampled_normal_ts.z;
    float param_1 = clearcoat_roughness2;
    float _2671 = D_GTR1(param, param_1);
    float3 param_2 = view_dir_ts;
    float param_3 = 0.0625f;
    float param_4 = 0.0625f;
    float _2678 = G1(param_2, param_3, param_4);
    float3 param_5 = reflected_dir_ts;
    float param_6 = 0.0625f;
    float param_7 = 0.0625f;
    float _2683 = G1(param_5, param_6, param_7);
    float param_8 = dot(reflected_dir_ts, sampled_normal_ts);
    float param_9 = clearcoat_ior;
    float F = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param_8, param_9) - clearcoat_F0) / (1.0f - clearcoat_F0));
    float _2710 = (4.0f * abs(view_dir_ts.z)) * abs(reflected_dir_ts.z);
    float _2713;
    if (_2710 != 0.0f)
    {
        _2713 = (_2671 * (_2678 * _2683)) / _2710;
    }
    else
    {
        _2713 = 0.0f;
    }
    F *= _2713;
    float3 param_10 = view_dir_ts;
    float param_11 = 0.0625f;
    float param_12 = 0.0625f;
    float _2731 = G1(param_10, param_11, param_12);
    float pdf = ((_2671 * _2731) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0f, 1.0f)) / abs(view_dir_ts.z);
    float _2746 = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    if (_2746 != 0.0f)
    {
        pdf /= _2746;
    }
    float _2757 = F;
    float _2758 = _2757 * clamp(reflected_dir_ts.z, 0.0f, 1.0f);
    F = _2758;
    return float4(_2758, _2758, _2758, pdf);
}

float4 Sample_PrincipledDiffuse_BSDF(float3 T, float3 B, float3 N, float3 I, float roughness, float3 base_color, float3 sheen_color, bool uniform_sampling, float rand_u, float rand_v, inout float3 out_V)
{
    float _2383 = 6.283185482025146484375f * rand_v;
    float _2386 = cos(_2383);
    float _2389 = sin(_2383);
    float3 V;
    if (uniform_sampling)
    {
        float _2398 = sqrt(mad(-rand_u, rand_u, 1.0f));
        V = float3(_2398 * _2386, _2398 * _2389, rand_u);
    }
    else
    {
        float _2411 = sqrt(rand_u);
        V = float3(_2411 * _2386, _2411 * _2389, sqrt(1.0f - rand_u));
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
    float4 _8837;
    do
    {
        [branch]
        if ((clearcoat_roughness2 * clearcoat_roughness2) < 1.0000000116860974230803549289703e-07f)
        {
            float3 _2775 = reflect(I, N);
            float param = dot(_2775, N);
            float param_1 = clearcoat_ior;
            out_V = _2775;
            float _2794 = lerp(0.039999999105930328369140625f, 1.0f, (fresnel_dielectric_cos(param, param_1) - clearcoat_F0) / (1.0f - clearcoat_F0)) * 1000000.0f;
            _8837 = float4(_2794, _2794, _2794, 1000000.0f);
            break;
        }
        float3 param_2 = T;
        float3 param_3 = B;
        float3 param_4 = N;
        float3 param_5 = -I;
        float3 _2812 = normalize(tangent_from_world(param_2, param_3, param_4, param_5));
        float param_6 = clearcoat_roughness2;
        float param_7 = clearcoat_roughness2;
        float param_8 = rand_u;
        float param_9 = rand_v;
        float3 _2823 = SampleGGX_VNDF(_2812, param_6, param_7, param_8, param_9);
        float3 _2834 = normalize(reflect(-_2812, _2823));
        float3 param_10 = T;
        float3 param_11 = B;
        float3 param_12 = N;
        float3 param_13 = _2834;
        out_V = world_from_tangent(param_10, param_11, param_12, param_13);
        float3 param_14 = _2812;
        float3 param_15 = _2823;
        float3 param_16 = _2834;
        float param_17 = clearcoat_roughness2;
        float param_18 = clearcoat_ior;
        float param_19 = clearcoat_F0;
        _8837 = Evaluate_PrincipledClearcoat_BSDF(param_14, param_15, param_16, param_17, param_18, param_19);
        break;
    } while(false);
    return _8837;
}

float3 ShadeSurface(int px_index, hit_data_t inter, ray_data_t ray)
{
    float3 _8812;
    do
    {
        float3 _4803 = float3(ray.d[0], ray.d[1], ray.d[2]);
        [branch]
        if (inter.mask == 0)
        {
            float3 env_col = _3223_g_params.back_col.xyz;
            uint _4816 = asuint(_3223_g_params.back_col.w);
            if (_4816 != 4294967295u)
            {
                atlas_texture_t _4823;
                _4823.size = _855.Load(_4816 * 80 + 0);
                _4823.atlas = _855.Load(_4816 * 80 + 4);
                [unroll]
                for (int _92ident = 0; _92ident < 4; _92ident++)
                {
                    _4823.page[_92ident] = _855.Load(_92ident * 4 + _4816 * 80 + 8);
                }
                [unroll]
                for (int _93ident = 0; _93ident < 14; _93ident++)
                {
                    _4823.pos[_93ident] = _855.Load(_93ident * 4 + _4816 * 80 + 24);
                }
                uint _9722[14] = { _4823.pos[0], _4823.pos[1], _4823.pos[2], _4823.pos[3], _4823.pos[4], _4823.pos[5], _4823.pos[6], _4823.pos[7], _4823.pos[8], _4823.pos[9], _4823.pos[10], _4823.pos[11], _4823.pos[12], _4823.pos[13] };
                uint _9693[4] = { _4823.page[0], _4823.page[1], _4823.page[2], _4823.page[3] };
                atlas_texture_t _9355 = { _4823.size, _4823.atlas, _9693, _9722 };
                float param = _3223_g_params.env_rotation;
                env_col *= SampleLatlong_RGBE(_9355, _4803, param);
                if (_3223_g_params.env_qtree_levels > 0)
                {
                    float param_1 = ray.pdf;
                    float param_2 = Evaluate_EnvQTree(_3223_g_params.env_rotation, g_env_qtree, _g_env_qtree_sampler, _3223_g_params.env_qtree_levels, _4803);
                    env_col *= power_heuristic(param_1, param_2);
                }
            }
            _8812 = float3(ray.c[0] * env_col.x, ray.c[1] * env_col.y, ray.c[2] * env_col.z);
            break;
        }
        float3 _4926 = float3(ray.o[0], ray.o[1], ray.o[2]) + (_4803 * inter.t);
        [branch]
        if (inter.obj_index < 0)
        {
            light_t _4938;
            _4938.type_and_param0 = _3255.Load4(((-1) - inter.obj_index) * 64 + 0);
            _4938.param1 = asfloat(_3255.Load4(((-1) - inter.obj_index) * 64 + 16));
            _4938.param2 = asfloat(_3255.Load4(((-1) - inter.obj_index) * 64 + 32));
            _4938.param3 = asfloat(_3255.Load4(((-1) - inter.obj_index) * 64 + 48));
            float3 lcol = asfloat(_4938.type_and_param0.yzw);
            uint _4955 = _4938.type_and_param0.x & 31u;
            if (_4955 == 0u)
            {
                float param_3 = ray.pdf;
                float param_4 = (inter.t * inter.t) / ((0.5f * _4938.param1.w) * dot(_4803, normalize(_4938.param1.xyz - _4926)));
                lcol *= power_heuristic(param_3, param_4);
                bool _5022 = _4938.param3.x > 0.0f;
                bool _5028;
                if (_5022)
                {
                    _5028 = _4938.param3.y > 0.0f;
                }
                else
                {
                    _5028 = _5022;
                }
                [branch]
                if (_5028)
                {
                    [flatten]
                    if (_4938.param3.y > 0.0f)
                    {
                        lcol *= clamp((_4938.param3.x - acos(clamp(-dot(_4803, _4938.param2.xyz), 0.0f, 1.0f))) / _4938.param3.y, 0.0f, 1.0f);
                    }
                }
            }
            else
            {
                if (_4955 == 4u)
                {
                    float param_5 = ray.pdf;
                    float param_6 = (inter.t * inter.t) / (_4938.param1.w * dot(_4803, normalize(cross(_4938.param2.xyz, _4938.param3.xyz))));
                    lcol *= power_heuristic(param_5, param_6);
                }
                else
                {
                    if (_4955 == 5u)
                    {
                        float param_7 = ray.pdf;
                        float param_8 = (inter.t * inter.t) / (_4938.param1.w * dot(_4803, normalize(cross(_4938.param2.xyz, _4938.param3.xyz))));
                        lcol *= power_heuristic(param_7, param_8);
                    }
                    else
                    {
                        if (_4955 == 3u)
                        {
                            float param_9 = ray.pdf;
                            float param_10 = (inter.t * inter.t) / (_4938.param1.w * (1.0f - abs(dot(_4803, _4938.param3.xyz))));
                            lcol *= power_heuristic(param_9, param_10);
                        }
                    }
                }
            }
            _8812 = float3(ray.c[0] * lcol.x, ray.c[1] * lcol.y, ray.c[2] * lcol.z);
            break;
        }
        bool _5227 = inter.prim_index < 0;
        int _5230;
        if (_5227)
        {
            _5230 = (-1) - inter.prim_index;
        }
        else
        {
            _5230 = inter.prim_index;
        }
        uint _5241 = uint(_5230);
        material_t _5249;
        [unroll]
        for (int _94ident = 0; _94ident < 5; _94ident++)
        {
            _5249.textures[_94ident] = _4512.Load(_94ident * 4 + ((_4516.Load(_5241 * 4 + 0) >> 16u) & 16383u) * 80 + 0);
        }
        [unroll]
        for (int _95ident = 0; _95ident < 3; _95ident++)
        {
            _5249.base_color[_95ident] = asfloat(_4512.Load(_95ident * 4 + ((_4516.Load(_5241 * 4 + 0) >> 16u) & 16383u) * 80 + 20));
        }
        _5249.flags = _4512.Load(((_4516.Load(_5241 * 4 + 0) >> 16u) & 16383u) * 80 + 32);
        _5249.type = _4512.Load(((_4516.Load(_5241 * 4 + 0) >> 16u) & 16383u) * 80 + 36);
        _5249.tangent_rotation_or_strength = asfloat(_4512.Load(((_4516.Load(_5241 * 4 + 0) >> 16u) & 16383u) * 80 + 40));
        _5249.roughness_and_anisotropic = _4512.Load(((_4516.Load(_5241 * 4 + 0) >> 16u) & 16383u) * 80 + 44);
        _5249.int_ior = asfloat(_4512.Load(((_4516.Load(_5241 * 4 + 0) >> 16u) & 16383u) * 80 + 48));
        _5249.ext_ior = asfloat(_4512.Load(((_4516.Load(_5241 * 4 + 0) >> 16u) & 16383u) * 80 + 52));
        _5249.sheen_and_sheen_tint = _4512.Load(((_4516.Load(_5241 * 4 + 0) >> 16u) & 16383u) * 80 + 56);
        _5249.tint_and_metallic = _4512.Load(((_4516.Load(_5241 * 4 + 0) >> 16u) & 16383u) * 80 + 60);
        _5249.transmission_and_transmission_roughness = _4512.Load(((_4516.Load(_5241 * 4 + 0) >> 16u) & 16383u) * 80 + 64);
        _5249.specular_and_specular_tint = _4512.Load(((_4516.Load(_5241 * 4 + 0) >> 16u) & 16383u) * 80 + 68);
        _5249.clearcoat_and_clearcoat_roughness = _4512.Load(((_4516.Load(_5241 * 4 + 0) >> 16u) & 16383u) * 80 + 72);
        _5249.normal_map_strength_unorm = _4512.Load(((_4516.Load(_5241 * 4 + 0) >> 16u) & 16383u) * 80 + 76);
        uint _9723 = _5249.textures[0];
        uint _9724 = _5249.textures[1];
        uint _9725 = _5249.textures[2];
        uint _9726 = _5249.textures[3];
        uint _9727 = _5249.textures[4];
        float _9728 = _5249.base_color[0];
        float _9729 = _5249.base_color[1];
        float _9730 = _5249.base_color[2];
        uint _9372 = _5249.flags;
        uint _9373 = _5249.type;
        float _9374 = _5249.tangent_rotation_or_strength;
        uint _9375 = _5249.roughness_and_anisotropic;
        float _9376 = _5249.int_ior;
        float _9377 = _5249.ext_ior;
        uint _9378 = _5249.sheen_and_sheen_tint;
        uint _9379 = _5249.tint_and_metallic;
        uint _9380 = _5249.transmission_and_transmission_roughness;
        uint _9381 = _5249.specular_and_specular_tint;
        uint _9382 = _5249.clearcoat_and_clearcoat_roughness;
        uint _9383 = _5249.normal_map_strength_unorm;
        transform_t _5306;
        _5306.xform = asfloat(uint4x4(_4172.Load4(asuint(asfloat(_5299.Load(inter.obj_index * 32 + 12))) * 128 + 0), _4172.Load4(asuint(asfloat(_5299.Load(inter.obj_index * 32 + 12))) * 128 + 16), _4172.Load4(asuint(asfloat(_5299.Load(inter.obj_index * 32 + 12))) * 128 + 32), _4172.Load4(asuint(asfloat(_5299.Load(inter.obj_index * 32 + 12))) * 128 + 48)));
        _5306.inv_xform = asfloat(uint4x4(_4172.Load4(asuint(asfloat(_5299.Load(inter.obj_index * 32 + 12))) * 128 + 64), _4172.Load4(asuint(asfloat(_5299.Load(inter.obj_index * 32 + 12))) * 128 + 80), _4172.Load4(asuint(asfloat(_5299.Load(inter.obj_index * 32 + 12))) * 128 + 96), _4172.Load4(asuint(asfloat(_5299.Load(inter.obj_index * 32 + 12))) * 128 + 112)));
        uint _5313 = _5241 * 3u;
        vertex_t _5318;
        [unroll]
        for (int _96ident = 0; _96ident < 3; _96ident++)
        {
            _5318.p[_96ident] = asfloat(_4197.Load(_96ident * 4 + _4201.Load(_5313 * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _97ident = 0; _97ident < 3; _97ident++)
        {
            _5318.n[_97ident] = asfloat(_4197.Load(_97ident * 4 + _4201.Load(_5313 * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _98ident = 0; _98ident < 3; _98ident++)
        {
            _5318.b[_98ident] = asfloat(_4197.Load(_98ident * 4 + _4201.Load(_5313 * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _99ident = 0; _99ident < 2; _99ident++)
        {
            [unroll]
            for (int _100ident = 0; _100ident < 2; _100ident++)
            {
                _5318.t[_99ident][_100ident] = asfloat(_4197.Load(_100ident * 4 + _99ident * 8 + _4201.Load(_5313 * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _5364;
        [unroll]
        for (int _101ident = 0; _101ident < 3; _101ident++)
        {
            _5364.p[_101ident] = asfloat(_4197.Load(_101ident * 4 + _4201.Load((_5313 + 1u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _102ident = 0; _102ident < 3; _102ident++)
        {
            _5364.n[_102ident] = asfloat(_4197.Load(_102ident * 4 + _4201.Load((_5313 + 1u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _103ident = 0; _103ident < 3; _103ident++)
        {
            _5364.b[_103ident] = asfloat(_4197.Load(_103ident * 4 + _4201.Load((_5313 + 1u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _104ident = 0; _104ident < 2; _104ident++)
        {
            [unroll]
            for (int _105ident = 0; _105ident < 2; _105ident++)
            {
                _5364.t[_104ident][_105ident] = asfloat(_4197.Load(_105ident * 4 + _104ident * 8 + _4201.Load((_5313 + 1u) * 4 + 0) * 52 + 36));
            }
        }
        vertex_t _5410;
        [unroll]
        for (int _106ident = 0; _106ident < 3; _106ident++)
        {
            _5410.p[_106ident] = asfloat(_4197.Load(_106ident * 4 + _4201.Load((_5313 + 2u) * 4 + 0) * 52 + 0));
        }
        [unroll]
        for (int _107ident = 0; _107ident < 3; _107ident++)
        {
            _5410.n[_107ident] = asfloat(_4197.Load(_107ident * 4 + _4201.Load((_5313 + 2u) * 4 + 0) * 52 + 12));
        }
        [unroll]
        for (int _108ident = 0; _108ident < 3; _108ident++)
        {
            _5410.b[_108ident] = asfloat(_4197.Load(_108ident * 4 + _4201.Load((_5313 + 2u) * 4 + 0) * 52 + 24));
        }
        [unroll]
        for (int _109ident = 0; _109ident < 2; _109ident++)
        {
            [unroll]
            for (int _110ident = 0; _110ident < 2; _110ident++)
            {
                _5410.t[_109ident][_110ident] = asfloat(_4197.Load(_110ident * 4 + _109ident * 8 + _4201.Load((_5313 + 2u) * 4 + 0) * 52 + 36));
            }
        }
        float3 _5456 = float3(_5318.p[0], _5318.p[1], _5318.p[2]);
        float3 _5464 = float3(_5364.p[0], _5364.p[1], _5364.p[2]);
        float3 _5472 = float3(_5410.p[0], _5410.p[1], _5410.p[2]);
        float _5479 = (1.0f - inter.u) - inter.v;
        float3 _5512 = normalize(((float3(_5318.n[0], _5318.n[1], _5318.n[2]) * _5479) + (float3(_5364.n[0], _5364.n[1], _5364.n[2]) * inter.u)) + (float3(_5410.n[0], _5410.n[1], _5410.n[2]) * inter.v));
        float3 N = _5512;
        float2 _5538 = ((float2(_5318.t[0][0], _5318.t[0][1]) * _5479) + (float2(_5364.t[0][0], _5364.t[0][1]) * inter.u)) + (float2(_5410.t[0][0], _5410.t[0][1]) * inter.v);
        float3 _5554 = cross(_5464 - _5456, _5472 - _5456);
        float _5557 = length(_5554);
        float3 plane_N = _5554 / _5557.xxx;
        float3 _5593 = ((float3(_5318.b[0], _5318.b[1], _5318.b[2]) * _5479) + (float3(_5364.b[0], _5364.b[1], _5364.b[2]) * inter.u)) + (float3(_5410.b[0], _5410.b[1], _5410.b[2]) * inter.v);
        float3 B = _5593;
        float3 T = cross(_5593, _5512);
        if (_5227)
        {
            if ((_4516.Load(_5241 * 4 + 0) & 65535u) == 65535u)
            {
                _8812 = 0.0f.xxx;
                break;
            }
            material_t _5616;
            [unroll]
            for (int _111ident = 0; _111ident < 5; _111ident++)
            {
                _5616.textures[_111ident] = _4512.Load(_111ident * 4 + (_4516.Load(_5241 * 4 + 0) & 16383u) * 80 + 0);
            }
            [unroll]
            for (int _112ident = 0; _112ident < 3; _112ident++)
            {
                _5616.base_color[_112ident] = asfloat(_4512.Load(_112ident * 4 + (_4516.Load(_5241 * 4 + 0) & 16383u) * 80 + 20));
            }
            _5616.flags = _4512.Load((_4516.Load(_5241 * 4 + 0) & 16383u) * 80 + 32);
            _5616.type = _4512.Load((_4516.Load(_5241 * 4 + 0) & 16383u) * 80 + 36);
            _5616.tangent_rotation_or_strength = asfloat(_4512.Load((_4516.Load(_5241 * 4 + 0) & 16383u) * 80 + 40));
            _5616.roughness_and_anisotropic = _4512.Load((_4516.Load(_5241 * 4 + 0) & 16383u) * 80 + 44);
            _5616.int_ior = asfloat(_4512.Load((_4516.Load(_5241 * 4 + 0) & 16383u) * 80 + 48));
            _5616.ext_ior = asfloat(_4512.Load((_4516.Load(_5241 * 4 + 0) & 16383u) * 80 + 52));
            _5616.sheen_and_sheen_tint = _4512.Load((_4516.Load(_5241 * 4 + 0) & 16383u) * 80 + 56);
            _5616.tint_and_metallic = _4512.Load((_4516.Load(_5241 * 4 + 0) & 16383u) * 80 + 60);
            _5616.transmission_and_transmission_roughness = _4512.Load((_4516.Load(_5241 * 4 + 0) & 16383u) * 80 + 64);
            _5616.specular_and_specular_tint = _4512.Load((_4516.Load(_5241 * 4 + 0) & 16383u) * 80 + 68);
            _5616.clearcoat_and_clearcoat_roughness = _4512.Load((_4516.Load(_5241 * 4 + 0) & 16383u) * 80 + 72);
            _5616.normal_map_strength_unorm = _4512.Load((_4516.Load(_5241 * 4 + 0) & 16383u) * 80 + 76);
            _9723 = _5616.textures[0];
            _9724 = _5616.textures[1];
            _9725 = _5616.textures[2];
            _9726 = _5616.textures[3];
            _9727 = _5616.textures[4];
            _9728 = _5616.base_color[0];
            _9729 = _5616.base_color[1];
            _9730 = _5616.base_color[2];
            _9372 = _5616.flags;
            _9373 = _5616.type;
            _9374 = _5616.tangent_rotation_or_strength;
            _9375 = _5616.roughness_and_anisotropic;
            _9376 = _5616.int_ior;
            _9377 = _5616.ext_ior;
            _9378 = _5616.sheen_and_sheen_tint;
            _9379 = _5616.tint_and_metallic;
            _9380 = _5616.transmission_and_transmission_roughness;
            _9381 = _5616.specular_and_specular_tint;
            _9382 = _5616.clearcoat_and_clearcoat_roughness;
            _9383 = _5616.normal_map_strength_unorm;
            plane_N = -plane_N;
            N = -N;
            B = -B;
            T = -T;
        }
        float3 param_11 = plane_N;
        float4x4 param_12 = _5306.inv_xform;
        plane_N = TransformNormal(param_11, param_12);
        float3 param_13 = N;
        float4x4 param_14 = _5306.inv_xform;
        N = TransformNormal(param_13, param_14);
        float3 param_15 = B;
        float4x4 param_16 = _5306.inv_xform;
        B = TransformNormal(param_15, param_16);
        float3 param_17 = T;
        float4x4 param_18 = _5306.inv_xform;
        T = TransformNormal(param_17, param_18);
        float _5726 = mad(ray.cone_spread, inter.t, ray.cone_width);
        float _5736 = mad(0.5f, log2(abs(mad(_5364.t[0][0] - _5318.t[0][0], _5410.t[0][1] - _5318.t[0][1], -((_5410.t[0][0] - _5318.t[0][0]) * (_5364.t[0][1] - _5318.t[0][1])))) / _5557), log2(_5726));
        uint param_19 = uint(hash(px_index));
        float _5742 = construct_float(param_19);
        uint param_20 = uint(hash(hash(px_index)));
        float _5748 = construct_float(param_20);
        float3 col = 0.0f.xxx;
        int _5755 = ray.ray_depth & 255;
        int _5760 = (ray.ray_depth >> 8) & 255;
        int _5765 = (ray.ray_depth >> 16) & 255;
        int _5771 = (ray.ray_depth >> 24) & 255;
        int _5779 = ((_5755 + _5760) + _5765) + _5771;
        float mix_rand = frac(asfloat(_3218.Load(_3223_g_params.hi * 4 + 0)) + _5742);
        float mix_weight = 1.0f;
        float _5816;
        float _5835;
        float _5860;
        float _5929;
        while (_9373 == 4u)
        {
            float mix_val = _9374;
            if (_9724 != 4294967295u)
            {
                mix_val *= SampleBilinear(_9724, _5538, 0).x;
            }
            if (_5227)
            {
                _5816 = _9377 / _9376;
            }
            else
            {
                _5816 = _9376 / _9377;
            }
            if (_9376 != 0.0f)
            {
                float param_21 = dot(_4803, N);
                float param_22 = _5816;
                _5835 = fresnel_dielectric_cos(param_21, param_22);
            }
            else
            {
                _5835 = 1.0f;
            }
            float _5849 = mix_val;
            float _5850 = _5849 * clamp(_5835, 0.0f, 1.0f);
            mix_val = _5850;
            if (mix_rand > _5850)
            {
                if ((_9372 & 2u) != 0u)
                {
                    _5860 = 1.0f / (1.0f - mix_val);
                }
                else
                {
                    _5860 = 1.0f;
                }
                mix_weight *= _5860;
                material_t _5873;
                [unroll]
                for (int _113ident = 0; _113ident < 5; _113ident++)
                {
                    _5873.textures[_113ident] = _4512.Load(_113ident * 4 + _9726 * 80 + 0);
                }
                [unroll]
                for (int _114ident = 0; _114ident < 3; _114ident++)
                {
                    _5873.base_color[_114ident] = asfloat(_4512.Load(_114ident * 4 + _9726 * 80 + 20));
                }
                _5873.flags = _4512.Load(_9726 * 80 + 32);
                _5873.type = _4512.Load(_9726 * 80 + 36);
                _5873.tangent_rotation_or_strength = asfloat(_4512.Load(_9726 * 80 + 40));
                _5873.roughness_and_anisotropic = _4512.Load(_9726 * 80 + 44);
                _5873.int_ior = asfloat(_4512.Load(_9726 * 80 + 48));
                _5873.ext_ior = asfloat(_4512.Load(_9726 * 80 + 52));
                _5873.sheen_and_sheen_tint = _4512.Load(_9726 * 80 + 56);
                _5873.tint_and_metallic = _4512.Load(_9726 * 80 + 60);
                _5873.transmission_and_transmission_roughness = _4512.Load(_9726 * 80 + 64);
                _5873.specular_and_specular_tint = _4512.Load(_9726 * 80 + 68);
                _5873.clearcoat_and_clearcoat_roughness = _4512.Load(_9726 * 80 + 72);
                _5873.normal_map_strength_unorm = _4512.Load(_9726 * 80 + 76);
                _9723 = _5873.textures[0];
                _9724 = _5873.textures[1];
                _9725 = _5873.textures[2];
                _9726 = _5873.textures[3];
                _9727 = _5873.textures[4];
                _9728 = _5873.base_color[0];
                _9729 = _5873.base_color[1];
                _9730 = _5873.base_color[2];
                _9372 = _5873.flags;
                _9373 = _5873.type;
                _9374 = _5873.tangent_rotation_or_strength;
                _9375 = _5873.roughness_and_anisotropic;
                _9376 = _5873.int_ior;
                _9377 = _5873.ext_ior;
                _9378 = _5873.sheen_and_sheen_tint;
                _9379 = _5873.tint_and_metallic;
                _9380 = _5873.transmission_and_transmission_roughness;
                _9381 = _5873.specular_and_specular_tint;
                _9382 = _5873.clearcoat_and_clearcoat_roughness;
                _9383 = _5873.normal_map_strength_unorm;
                mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
            }
            else
            {
                if ((_9372 & 2u) != 0u)
                {
                    _5929 = 1.0f / mix_val;
                }
                else
                {
                    _5929 = 1.0f;
                }
                mix_weight *= _5929;
                material_t _5941;
                [unroll]
                for (int _115ident = 0; _115ident < 5; _115ident++)
                {
                    _5941.textures[_115ident] = _4512.Load(_115ident * 4 + _9727 * 80 + 0);
                }
                [unroll]
                for (int _116ident = 0; _116ident < 3; _116ident++)
                {
                    _5941.base_color[_116ident] = asfloat(_4512.Load(_116ident * 4 + _9727 * 80 + 20));
                }
                _5941.flags = _4512.Load(_9727 * 80 + 32);
                _5941.type = _4512.Load(_9727 * 80 + 36);
                _5941.tangent_rotation_or_strength = asfloat(_4512.Load(_9727 * 80 + 40));
                _5941.roughness_and_anisotropic = _4512.Load(_9727 * 80 + 44);
                _5941.int_ior = asfloat(_4512.Load(_9727 * 80 + 48));
                _5941.ext_ior = asfloat(_4512.Load(_9727 * 80 + 52));
                _5941.sheen_and_sheen_tint = _4512.Load(_9727 * 80 + 56);
                _5941.tint_and_metallic = _4512.Load(_9727 * 80 + 60);
                _5941.transmission_and_transmission_roughness = _4512.Load(_9727 * 80 + 64);
                _5941.specular_and_specular_tint = _4512.Load(_9727 * 80 + 68);
                _5941.clearcoat_and_clearcoat_roughness = _4512.Load(_9727 * 80 + 72);
                _5941.normal_map_strength_unorm = _4512.Load(_9727 * 80 + 76);
                _9723 = _5941.textures[0];
                _9724 = _5941.textures[1];
                _9725 = _5941.textures[2];
                _9726 = _5941.textures[3];
                _9727 = _5941.textures[4];
                _9728 = _5941.base_color[0];
                _9729 = _5941.base_color[1];
                _9730 = _5941.base_color[2];
                _9372 = _5941.flags;
                _9373 = _5941.type;
                _9374 = _5941.tangent_rotation_or_strength;
                _9375 = _5941.roughness_and_anisotropic;
                _9376 = _5941.int_ior;
                _9377 = _5941.ext_ior;
                _9378 = _5941.sheen_and_sheen_tint;
                _9379 = _5941.tint_and_metallic;
                _9380 = _5941.transmission_and_transmission_roughness;
                _9381 = _5941.specular_and_specular_tint;
                _9382 = _5941.clearcoat_and_clearcoat_roughness;
                _9383 = _5941.normal_map_strength_unorm;
                mix_rand /= mix_val;
            }
        }
        [branch]
        if (_9723 != 4294967295u)
        {
            float3 normals = (float3(SampleBilinear(_9723, _5538, 0).xy, 1.0f) * 2.0f) - 1.0f.xxx;
            if ((_855.Load(_9723 * 80 + 0) & 16384u) != 0u)
            {
                float3 _10082 = normals;
                _10082.z = sqrt(1.0f - dot(normals.xy, normals.xy));
                normals = _10082;
            }
            float3 _6025 = N;
            N = normalize(((T * normals.x) + (_6025 * normals.z)) + (B * normals.y));
            if ((_9383 & 65535u) != 65535u)
            {
                N = normalize(_6025 + ((N - _6025) * clamp(float(_9383 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f)));
            }
            float3 param_23 = plane_N;
            float3 param_24 = -_4803;
            float3 param_25 = N;
            N = ensure_valid_reflection(param_23, param_24, param_25);
        }
        float3 _6082 = ((_5456 * _5479) + (_5464 * inter.u)) + (_5472 * inter.v);
        float3 _6089 = float3(-_6082.z, 0.0f, _6082.x);
        float3 tangent = _6089;
        float3 param_26 = _6089;
        float4x4 param_27 = _5306.inv_xform;
        tangent = TransformNormal(param_26, param_27);
        if (_9374 != 0.0f)
        {
            float3 param_28 = tangent;
            float3 param_29 = N;
            float param_30 = _9374;
            tangent = rotate_around_axis(param_28, param_29, param_30);
        }
        float3 _6112 = normalize(cross(tangent, N));
        B = _6112;
        T = cross(N, _6112);
        float3 _9462 = 0.0f.xxx;
        float3 _9461 = 0.0f.xxx;
        float _9464 = 0.0f;
        float _9465 = 0.0f;
        float _9463 = 0.0f;
        bool _6124 = _3223_g_params.li_count != 0;
        bool _6130;
        if (_6124)
        {
            _6130 = _9373 != 3u;
        }
        else
        {
            _6130 = _6124;
        }
        float _9466;
        if (_6130)
        {
            float3 param_31 = _4926;
            float2 param_32 = float2(_5742, _5748);
            light_sample_t _9473 = { _9461, _9462, _9463, _9464, _9465, _9466 };
            light_sample_t param_33 = _9473;
            SampleLightSource(param_31, param_32, param_33);
            _9461 = param_33.col;
            _9462 = param_33.L;
            _9463 = param_33.area;
            _9464 = param_33.dist;
            _9465 = param_33.pdf;
            _9466 = param_33.cast_shadow;
        }
        float _6145 = dot(N, _9462);
        float3 base_color = float3(_9728, _9729, _9730);
        [branch]
        if (_9724 != 4294967295u)
        {
            base_color *= SampleBilinear(_9724, _5538, int(get_texture_lod(texSize(_9724), _5736)), true, true).xyz;
        }
        float3 tint_color = 0.0f.xxx;
        float _6189 = lum(base_color);
        [flatten]
        if (_6189 > 0.0f)
        {
            tint_color = base_color / _6189.xxx;
        }
        float roughness = clamp(float(_9375 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
        [branch]
        if (_9725 != 4294967295u)
        {
            roughness *= SampleBilinear(_9725, _5538, int(get_texture_lod(texSize(_9725), _5736)), false, true).x;
        }
        float _6230 = asfloat(_3218.Load((_3223_g_params.hi + 1) * 4 + 0));
        float _6234 = frac(_6230 + _5742);
        float _6240 = asfloat(_3218.Load((_3223_g_params.hi + 2) * 4 + 0));
        float _6244 = frac(_6240 + _5748);
        float _9794 = 0.0f;
        float _9793 = 0.0f;
        float _9792 = 0.0f;
        float _9494 = 0.0f;
        int _9499;
        float _9778;
        float _9779;
        float _9780;
        float _9785;
        float _9786;
        float _9787;
        [branch]
        if (_9373 == 0u)
        {
            [branch]
            if ((_9465 > 0.0f) && (_6145 > 0.0f))
            {
                float3 param_34 = -_4803;
                float3 param_35 = N;
                float3 param_36 = _9462;
                float param_37 = roughness;
                float3 param_38 = base_color;
                float4 _6284 = Evaluate_OrenDiffuse_BSDF(param_34, param_35, param_36, param_37, param_38);
                float mis_weight = 1.0f;
                if (_9463 > 0.0f)
                {
                    float param_39 = _9465;
                    float param_40 = _6284.w;
                    mis_weight = power_heuristic(param_39, param_40);
                }
                float3 _6312 = (_9461 * _6284.xyz) * ((mix_weight * mis_weight) / _9465);
                [branch]
                if (_9466 > 0.5f)
                {
                    float3 param_41 = _4926;
                    float3 param_42 = plane_N;
                    float3 _6323 = offset_ray(param_41, param_42);
                    uint _6377;
                    _6375.InterlockedAdd(8, 1u, _6377);
                    _6385.Store(_6377 * 44 + 0, asuint(_6323.x));
                    _6385.Store(_6377 * 44 + 4, asuint(_6323.y));
                    _6385.Store(_6377 * 44 + 8, asuint(_6323.z));
                    _6385.Store(_6377 * 44 + 12, asuint(_9462.x));
                    _6385.Store(_6377 * 44 + 16, asuint(_9462.y));
                    _6385.Store(_6377 * 44 + 20, asuint(_9462.z));
                    _6385.Store(_6377 * 44 + 24, asuint(_9464 - 9.9999997473787516355514526367188e-05f));
                    _6385.Store(_6377 * 44 + 28, asuint(ray.c[0] * _6312.x));
                    _6385.Store(_6377 * 44 + 32, asuint(ray.c[1] * _6312.y));
                    _6385.Store(_6377 * 44 + 36, asuint(ray.c[2] * _6312.z));
                    _6385.Store(_6377 * 44 + 40, uint(ray.xy));
                }
                else
                {
                    col += _6312;
                }
            }
            bool _6429 = _5755 < _3223_g_params.max_diff_depth;
            bool _6436;
            if (_6429)
            {
                _6436 = _5779 < _3223_g_params.max_total_depth;
            }
            else
            {
                _6436 = _6429;
            }
            [branch]
            if (_6436)
            {
                float3 param_43 = T;
                float3 param_44 = B;
                float3 param_45 = N;
                float3 param_46 = _4803;
                float param_47 = roughness;
                float3 param_48 = base_color;
                float param_49 = _6234;
                float param_50 = _6244;
                float3 param_51;
                float4 _6458 = Sample_OrenDiffuse_BSDF(param_43, param_44, param_45, param_46, param_47, param_48, param_49, param_50, param_51);
                _9499 = ray.ray_depth + 1;
                float3 param_52 = _4926;
                float3 param_53 = plane_N;
                float3 _6469 = offset_ray(param_52, param_53);
                _9778 = _6469.x;
                _9779 = _6469.y;
                _9780 = _6469.z;
                _9785 = param_51.x;
                _9786 = param_51.y;
                _9787 = param_51.z;
                _9792 = ((ray.c[0] * _6458.x) * mix_weight) / _6458.w;
                _9793 = ((ray.c[1] * _6458.y) * mix_weight) / _6458.w;
                _9794 = ((ray.c[2] * _6458.z) * mix_weight) / _6458.w;
                _9494 = _6458.w;
            }
        }
        else
        {
            [branch]
            if (_9373 == 1u)
            {
                float param_54 = 1.0f;
                float param_55 = 1.5f;
                float _6534 = fresnel_dielectric_cos(param_54, param_55);
                float _6538 = roughness * roughness;
                bool _6541 = _9465 > 0.0f;
                bool _6548;
                if (_6541)
                {
                    _6548 = (_6538 * _6538) >= 1.0000000116860974230803549289703e-07f;
                }
                else
                {
                    _6548 = _6541;
                }
                [branch]
                if (_6548 && (_6145 > 0.0f))
                {
                    float3 param_56 = T;
                    float3 param_57 = B;
                    float3 param_58 = N;
                    float3 param_59 = -_4803;
                    float3 param_60 = T;
                    float3 param_61 = B;
                    float3 param_62 = N;
                    float3 param_63 = _9462;
                    float3 param_64 = T;
                    float3 param_65 = B;
                    float3 param_66 = N;
                    float3 param_67 = normalize(_9462 - _4803);
                    float3 param_68 = tangent_from_world(param_56, param_57, param_58, param_59);
                    float3 param_69 = tangent_from_world(param_64, param_65, param_66, param_67);
                    float3 param_70 = tangent_from_world(param_60, param_61, param_62, param_63);
                    float param_71 = _6538;
                    float param_72 = _6538;
                    float param_73 = 1.5f;
                    float param_74 = _6534;
                    float3 param_75 = base_color;
                    float4 _6608 = Evaluate_GGXSpecular_BSDF(param_68, param_69, param_70, param_71, param_72, param_73, param_74, param_75);
                    float mis_weight_1 = 1.0f;
                    if (_9463 > 0.0f)
                    {
                        float param_76 = _9465;
                        float param_77 = _6608.w;
                        mis_weight_1 = power_heuristic(param_76, param_77);
                    }
                    float3 _6636 = (_9461 * _6608.xyz) * ((mix_weight * mis_weight_1) / _9465);
                    [branch]
                    if (_9466 > 0.5f)
                    {
                        float3 param_78 = _4926;
                        float3 param_79 = plane_N;
                        float3 _6647 = offset_ray(param_78, param_79);
                        uint _6694;
                        _6375.InterlockedAdd(8, 1u, _6694);
                        _6385.Store(_6694 * 44 + 0, asuint(_6647.x));
                        _6385.Store(_6694 * 44 + 4, asuint(_6647.y));
                        _6385.Store(_6694 * 44 + 8, asuint(_6647.z));
                        _6385.Store(_6694 * 44 + 12, asuint(_9462.x));
                        _6385.Store(_6694 * 44 + 16, asuint(_9462.y));
                        _6385.Store(_6694 * 44 + 20, asuint(_9462.z));
                        _6385.Store(_6694 * 44 + 24, asuint(_9464 - 9.9999997473787516355514526367188e-05f));
                        _6385.Store(_6694 * 44 + 28, asuint(ray.c[0] * _6636.x));
                        _6385.Store(_6694 * 44 + 32, asuint(ray.c[1] * _6636.y));
                        _6385.Store(_6694 * 44 + 36, asuint(ray.c[2] * _6636.z));
                        _6385.Store(_6694 * 44 + 40, uint(ray.xy));
                    }
                    else
                    {
                        col += _6636;
                    }
                }
                bool _6733 = _5760 < _3223_g_params.max_spec_depth;
                bool _6740;
                if (_6733)
                {
                    _6740 = _5779 < _3223_g_params.max_total_depth;
                }
                else
                {
                    _6740 = _6733;
                }
                [branch]
                if (_6740)
                {
                    float3 param_80 = T;
                    float3 param_81 = B;
                    float3 param_82 = N;
                    float3 param_83 = _4803;
                    float3 param_84;
                    float4 _6759 = Sample_GGXSpecular_BSDF(param_80, param_81, param_82, param_83, roughness, 0.0f, 1.5f, _6534, base_color, _6234, _6244, param_84);
                    _9499 = ray.ray_depth + 256;
                    float3 param_85 = _4926;
                    float3 param_86 = plane_N;
                    float3 _6771 = offset_ray(param_85, param_86);
                    _9778 = _6771.x;
                    _9779 = _6771.y;
                    _9780 = _6771.z;
                    _9785 = param_84.x;
                    _9786 = param_84.y;
                    _9787 = param_84.z;
                    _9792 = ((ray.c[0] * _6759.x) * mix_weight) / _6759.w;
                    _9793 = ((ray.c[1] * _6759.y) * mix_weight) / _6759.w;
                    _9794 = ((ray.c[2] * _6759.z) * mix_weight) / _6759.w;
                    _9494 = _6759.w;
                }
            }
            else
            {
                [branch]
                if (_9373 == 2u)
                {
                    float _6834;
                    if (_5227)
                    {
                        _6834 = _9376 / _9377;
                    }
                    else
                    {
                        _6834 = _9377 / _9376;
                    }
                    float _6852 = roughness * roughness;
                    bool _6855 = _9465 > 0.0f;
                    bool _6862;
                    if (_6855)
                    {
                        _6862 = (_6852 * _6852) >= 1.0000000116860974230803549289703e-07f;
                    }
                    else
                    {
                        _6862 = _6855;
                    }
                    [branch]
                    if (_6862 && (_6145 < 0.0f))
                    {
                        float3 param_87 = T;
                        float3 param_88 = B;
                        float3 param_89 = N;
                        float3 param_90 = -_4803;
                        float3 param_91 = T;
                        float3 param_92 = B;
                        float3 param_93 = N;
                        float3 param_94 = _9462;
                        float3 param_95 = T;
                        float3 param_96 = B;
                        float3 param_97 = N;
                        float3 param_98 = normalize(_9462 - (_4803 * _6834));
                        float3 param_99 = tangent_from_world(param_87, param_88, param_89, param_90);
                        float3 param_100 = tangent_from_world(param_95, param_96, param_97, param_98);
                        float3 param_101 = tangent_from_world(param_91, param_92, param_93, param_94);
                        float param_102 = _6852;
                        float param_103 = _6834;
                        float3 param_104 = base_color;
                        float4 _6921 = Evaluate_GGXRefraction_BSDF(param_99, param_100, param_101, param_102, param_103, param_104);
                        float mis_weight_2 = 1.0f;
                        if (_9463 > 0.0f)
                        {
                            float param_105 = _9465;
                            float param_106 = _6921.w;
                            mis_weight_2 = power_heuristic(param_105, param_106);
                        }
                        float3 _6949 = (_9461 * _6921.xyz) * ((mix_weight * mis_weight_2) / _9465);
                        [branch]
                        if (_9466 > 0.5f)
                        {
                            float3 param_107 = _4926;
                            float3 param_108 = -plane_N;
                            float3 _6961 = offset_ray(param_107, param_108);
                            uint _7008;
                            _6375.InterlockedAdd(8, 1u, _7008);
                            _6385.Store(_7008 * 44 + 0, asuint(_6961.x));
                            _6385.Store(_7008 * 44 + 4, asuint(_6961.y));
                            _6385.Store(_7008 * 44 + 8, asuint(_6961.z));
                            _6385.Store(_7008 * 44 + 12, asuint(_9462.x));
                            _6385.Store(_7008 * 44 + 16, asuint(_9462.y));
                            _6385.Store(_7008 * 44 + 20, asuint(_9462.z));
                            _6385.Store(_7008 * 44 + 24, asuint(_9464 - 9.9999997473787516355514526367188e-05f));
                            _6385.Store(_7008 * 44 + 28, asuint(ray.c[0] * _6949.x));
                            _6385.Store(_7008 * 44 + 32, asuint(ray.c[1] * _6949.y));
                            _6385.Store(_7008 * 44 + 36, asuint(ray.c[2] * _6949.z));
                            _6385.Store(_7008 * 44 + 40, uint(ray.xy));
                        }
                        else
                        {
                            col += _6949;
                        }
                    }
                    bool _7047 = _5765 < _3223_g_params.max_refr_depth;
                    bool _7054;
                    if (_7047)
                    {
                        _7054 = _5779 < _3223_g_params.max_total_depth;
                    }
                    else
                    {
                        _7054 = _7047;
                    }
                    [branch]
                    if (_7054)
                    {
                        float3 param_109 = T;
                        float3 param_110 = B;
                        float3 param_111 = N;
                        float3 param_112 = _4803;
                        float param_113 = roughness;
                        float param_114 = _6834;
                        float3 param_115 = base_color;
                        float param_116 = _6234;
                        float param_117 = _6244;
                        float4 param_118;
                        float4 _7078 = Sample_GGXRefraction_BSDF(param_109, param_110, param_111, param_112, param_113, param_114, param_115, param_116, param_117, param_118);
                        _9499 = ray.ray_depth + 65536;
                        _9792 = ((ray.c[0] * _7078.x) * mix_weight) / _7078.w;
                        _9793 = ((ray.c[1] * _7078.y) * mix_weight) / _7078.w;
                        _9794 = ((ray.c[2] * _7078.z) * mix_weight) / _7078.w;
                        _9494 = _7078.w;
                        float3 param_119 = _4926;
                        float3 param_120 = -plane_N;
                        float3 _7133 = offset_ray(param_119, param_120);
                        _9778 = _7133.x;
                        _9779 = _7133.y;
                        _9780 = _7133.z;
                        _9785 = param_118.x;
                        _9786 = param_118.y;
                        _9787 = param_118.z;
                    }
                }
                else
                {
                    [branch]
                    if (_9373 == 3u)
                    {
                        if ((_9372 & 4u) != 0u)
                        {
                            float3 env_col_1 = _3223_g_params.env_col.xyz;
                            uint _7172 = asuint(_3223_g_params.env_col.w);
                            if (_7172 != 4294967295u)
                            {
                                atlas_texture_t _7179;
                                _7179.size = _855.Load(_7172 * 80 + 0);
                                _7179.atlas = _855.Load(_7172 * 80 + 4);
                                [unroll]
                                for (int _117ident = 0; _117ident < 4; _117ident++)
                                {
                                    _7179.page[_117ident] = _855.Load(_117ident * 4 + _7172 * 80 + 8);
                                }
                                [unroll]
                                for (int _118ident = 0; _118ident < 14; _118ident++)
                                {
                                    _7179.pos[_118ident] = _855.Load(_118ident * 4 + _7172 * 80 + 24);
                                }
                                uint _9899[14] = { _7179.pos[0], _7179.pos[1], _7179.pos[2], _7179.pos[3], _7179.pos[4], _7179.pos[5], _7179.pos[6], _7179.pos[7], _7179.pos[8], _7179.pos[9], _7179.pos[10], _7179.pos[11], _7179.pos[12], _7179.pos[13] };
                                uint _9870[4] = { _7179.page[0], _7179.page[1], _7179.page[2], _7179.page[3] };
                                atlas_texture_t _9664 = { _7179.size, _7179.atlas, _9870, _9899 };
                                float param_121 = _3223_g_params.env_rotation;
                                env_col_1 *= SampleLatlong_RGBE(_9664, _4803, param_121);
                            }
                            base_color *= env_col_1;
                        }
                        col += (base_color * (mix_weight * _9374));
                    }
                    else
                    {
                        [branch]
                        if (_9373 == 5u)
                        {
                            bool _7255 = _5771 < _3223_g_params.max_transp_depth;
                            bool _7262;
                            if (_7255)
                            {
                                _7262 = _5779 < _3223_g_params.max_total_depth;
                            }
                            else
                            {
                                _7262 = _7255;
                            }
                            [branch]
                            if (_7262)
                            {
                                _9499 = ray.ray_depth + 16777216;
                                _9494 = ray.pdf;
                                float3 param_122 = _4926;
                                float3 param_123 = -plane_N;
                                float3 _7279 = offset_ray(param_122, param_123);
                                _9778 = _7279.x;
                                _9779 = _7279.y;
                                _9780 = _7279.z;
                                _9785 = ray.d[0];
                                _9786 = ray.d[1];
                                _9787 = ray.d[2];
                                _9792 = ray.c[0];
                                _9793 = ray.c[1];
                                _9794 = ray.c[2];
                            }
                        }
                        else
                        {
                            if (_9373 == 6u)
                            {
                                float metallic = clamp(float((_9379 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                [branch]
                                if (_9726 != 4294967295u)
                                {
                                    metallic *= SampleBilinear(_9726, _5538, int(get_texture_lod(texSize(_9726), _5736))).x;
                                }
                                float specular = clamp(float(_9381 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                [branch]
                                if (_9727 != 4294967295u)
                                {
                                    specular *= SampleBilinear(_9727, _5538, int(get_texture_lod(texSize(_9727), _5736))).x;
                                }
                                float _7389 = clamp(float(_9382 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                float _7397 = clamp(float((_9382 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                float _7404 = clamp(float(_9378 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f);
                                float3 _7426 = lerp(lerp(1.0f.xxx, tint_color, clamp(float((_9381 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * (specular * 0.07999999821186065673828125f), base_color, metallic.xxx);
                                float _7433 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * specular))) - 1.0f;
                                float param_124 = 1.0f;
                                float param_125 = _7433;
                                float _7438 = fresnel_dielectric_cos(param_124, param_125);
                                float param_126 = dot(_4803, N);
                                float param_127 = _7433;
                                float param_128;
                                float param_129;
                                float param_130;
                                float param_131;
                                get_lobe_weights(lerp(_6189, 1.0f, _7404), lum(lerp(_7426, 1.0f.xxx, ((fresnel_dielectric_cos(param_126, param_127) - _7438) / (1.0f - _7438)).xxx)), specular, metallic, clamp(float(_9380 & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _7389, param_128, param_129, param_130, param_131);
                                float3 _7492 = lerp(1.0f.xxx, tint_color, clamp(float((_9378 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f).xxx) * _7404;
                                float _7495;
                                if (_5227)
                                {
                                    _7495 = _9376 / _9377;
                                }
                                else
                                {
                                    _7495 = _9377 / _9376;
                                }
                                float param_132 = dot(_4803, N);
                                float param_133 = 1.0f / _7495;
                                float _7518 = fresnel_dielectric_cos(param_132, param_133);
                                float _7525 = (2.0f / (1.0f - sqrt(0.07999999821186065673828125f * _7389))) - 1.0f;
                                float param_134 = 1.0f;
                                float param_135 = _7525;
                                float _7530 = fresnel_dielectric_cos(param_134, param_135);
                                float _7534 = _7397 * _7397;
                                float _7547 = mad(roughness - 1.0f, 1.0f - clamp(float((_9380 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f);
                                float _7551 = _7547 * _7547;
                                [branch]
                                if (_9465 > 0.0f)
                                {
                                    float3 lcol_1 = 0.0f.xxx;
                                    float bsdf_pdf = 0.0f;
                                    bool _7562 = _6145 > 0.0f;
                                    [branch]
                                    if ((param_128 > 1.0000000116860974230803549289703e-07f) && _7562)
                                    {
                                        float3 param_136 = -_4803;
                                        float3 param_137 = N;
                                        float3 param_138 = _9462;
                                        float param_139 = roughness;
                                        float3 param_140 = base_color.xyz;
                                        float3 param_141 = _7492;
                                        bool param_142 = false;
                                        float4 _7582 = Evaluate_PrincipledDiffuse_BSDF(param_136, param_137, param_138, param_139, param_140, param_141, param_142);
                                        bsdf_pdf = mad(param_128, _7582.w, bsdf_pdf);
                                        lcol_1 += (((_9461 * _6145) * (_7582 * (1.0f - metallic)).xyz) / (3.1415927410125732421875f * _9465).xxx);
                                    }
                                    float3 H;
                                    [flatten]
                                    if (_7562)
                                    {
                                        H = normalize(_9462 - _4803);
                                    }
                                    else
                                    {
                                        H = normalize(_9462 - (_4803 * _7495));
                                    }
                                    float _7628 = roughness * roughness;
                                    float _7639 = sqrt(mad(-0.89999997615814208984375f, clamp(float((_9375 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), 1.0f));
                                    float _7643 = _7628 / _7639;
                                    float _7647 = _7628 * _7639;
                                    float3 param_143 = T;
                                    float3 param_144 = B;
                                    float3 param_145 = N;
                                    float3 param_146 = -_4803;
                                    float3 _7658 = tangent_from_world(param_143, param_144, param_145, param_146);
                                    float3 param_147 = T;
                                    float3 param_148 = B;
                                    float3 param_149 = N;
                                    float3 param_150 = _9462;
                                    float3 _7669 = tangent_from_world(param_147, param_148, param_149, param_150);
                                    float3 param_151 = T;
                                    float3 param_152 = B;
                                    float3 param_153 = N;
                                    float3 param_154 = H;
                                    float3 _7679 = tangent_from_world(param_151, param_152, param_153, param_154);
                                    bool _7681 = param_129 > 0.0f;
                                    bool _7688;
                                    if (_7681)
                                    {
                                        _7688 = (_7643 * _7647) >= 1.0000000116860974230803549289703e-07f;
                                    }
                                    else
                                    {
                                        _7688 = _7681;
                                    }
                                    [branch]
                                    if (_7688 && _7562)
                                    {
                                        float3 param_155 = _7658;
                                        float3 param_156 = _7679;
                                        float3 param_157 = _7669;
                                        float param_158 = _7643;
                                        float param_159 = _7647;
                                        float param_160 = _7433;
                                        float param_161 = _7438;
                                        float3 param_162 = _7426;
                                        float4 _7711 = Evaluate_GGXSpecular_BSDF(param_155, param_156, param_157, param_158, param_159, param_160, param_161, param_162);
                                        bsdf_pdf = mad(param_129, _7711.w, bsdf_pdf);
                                        lcol_1 += ((_9461 * _7711.xyz) / _9465.xxx);
                                    }
                                    bool _7730 = param_130 > 0.0f;
                                    bool _7737;
                                    if (_7730)
                                    {
                                        _7737 = (_7534 * _7534) >= 1.0000000116860974230803549289703e-07f;
                                    }
                                    else
                                    {
                                        _7737 = _7730;
                                    }
                                    [branch]
                                    if (_7737 && _7562)
                                    {
                                        float3 param_163 = _7658;
                                        float3 param_164 = _7679;
                                        float3 param_165 = _7669;
                                        float param_166 = _7534;
                                        float param_167 = _7525;
                                        float param_168 = _7530;
                                        float4 _7756 = Evaluate_PrincipledClearcoat_BSDF(param_163, param_164, param_165, param_166, param_167, param_168);
                                        bsdf_pdf = mad(param_130, _7756.w, bsdf_pdf);
                                        lcol_1 += (((_9461 * 0.25f) * _7756.xyz) / _9465.xxx);
                                    }
                                    [branch]
                                    if (param_131 > 0.0f)
                                    {
                                        bool _7780 = _7518 != 0.0f;
                                        bool _7787;
                                        if (_7780)
                                        {
                                            _7787 = (_7628 * _7628) >= 1.0000000116860974230803549289703e-07f;
                                        }
                                        else
                                        {
                                            _7787 = _7780;
                                        }
                                        [branch]
                                        if (_7787 && _7562)
                                        {
                                            float3 param_169 = _7658;
                                            float3 param_170 = _7679;
                                            float3 param_171 = _7669;
                                            float param_172 = _7628;
                                            float param_173 = _7628;
                                            float param_174 = 1.0f;
                                            float param_175 = 0.0f;
                                            float3 param_176 = 1.0f.xxx;
                                            float4 _7807 = Evaluate_GGXSpecular_BSDF(param_169, param_170, param_171, param_172, param_173, param_174, param_175, param_176);
                                            bsdf_pdf = mad(param_131 * _7518, _7807.w, bsdf_pdf);
                                            lcol_1 += ((_9461 * _7807.xyz) * (_7518 / _9465));
                                        }
                                        bool _7829 = _7518 != 1.0f;
                                        bool _7836;
                                        if (_7829)
                                        {
                                            _7836 = (_7551 * _7551) >= 1.0000000116860974230803549289703e-07f;
                                        }
                                        else
                                        {
                                            _7836 = _7829;
                                        }
                                        [branch]
                                        if (_7836 && (_6145 < 0.0f))
                                        {
                                            float3 param_177 = _7658;
                                            float3 param_178 = _7679;
                                            float3 param_179 = _7669;
                                            float param_180 = _7551;
                                            float param_181 = _7495;
                                            float3 param_182 = base_color;
                                            float4 _7855 = Evaluate_GGXRefraction_BSDF(param_177, param_178, param_179, param_180, param_181, param_182);
                                            float _7858 = 1.0f - _7518;
                                            bsdf_pdf = mad(param_131 * _7858, _7855.w, bsdf_pdf);
                                            lcol_1 += ((_9461 * _7855.xyz) * (_7858 / _9465));
                                        }
                                    }
                                    float mis_weight_3 = 1.0f;
                                    [flatten]
                                    if (_9463 > 0.0f)
                                    {
                                        float param_183 = _9465;
                                        float param_184 = bsdf_pdf;
                                        mis_weight_3 = power_heuristic(param_183, param_184);
                                    }
                                    lcol_1 *= (mix_weight * mis_weight_3);
                                    [branch]
                                    if (_9466 > 0.5f)
                                    {
                                        float3 _7903;
                                        if (_6145 < 0.0f)
                                        {
                                            _7903 = -plane_N;
                                        }
                                        else
                                        {
                                            _7903 = plane_N;
                                        }
                                        float3 param_185 = _4926;
                                        float3 param_186 = _7903;
                                        float3 _7914 = offset_ray(param_185, param_186);
                                        uint _7961;
                                        _6375.InterlockedAdd(8, 1u, _7961);
                                        _6385.Store(_7961 * 44 + 0, asuint(_7914.x));
                                        _6385.Store(_7961 * 44 + 4, asuint(_7914.y));
                                        _6385.Store(_7961 * 44 + 8, asuint(_7914.z));
                                        _6385.Store(_7961 * 44 + 12, asuint(_9462.x));
                                        _6385.Store(_7961 * 44 + 16, asuint(_9462.y));
                                        _6385.Store(_7961 * 44 + 20, asuint(_9462.z));
                                        _6385.Store(_7961 * 44 + 24, asuint(_9464 - 9.9999997473787516355514526367188e-05f));
                                        _6385.Store(_7961 * 44 + 28, asuint(ray.c[0] * lcol_1.x));
                                        _6385.Store(_7961 * 44 + 32, asuint(ray.c[1] * lcol_1.y));
                                        _6385.Store(_7961 * 44 + 36, asuint(ray.c[2] * lcol_1.z));
                                        _6385.Store(_7961 * 44 + 40, uint(ray.xy));
                                    }
                                    else
                                    {
                                        col += lcol_1;
                                    }
                                }
                                [branch]
                                if (mix_rand < param_128)
                                {
                                    bool _8005 = _5755 < _3223_g_params.max_diff_depth;
                                    bool _8012;
                                    if (_8005)
                                    {
                                        _8012 = _5779 < _3223_g_params.max_total_depth;
                                    }
                                    else
                                    {
                                        _8012 = _8005;
                                    }
                                    if (_8012)
                                    {
                                        float3 param_187 = T;
                                        float3 param_188 = B;
                                        float3 param_189 = N;
                                        float3 param_190 = _4803;
                                        float param_191 = roughness;
                                        float3 param_192 = base_color.xyz;
                                        float3 param_193 = _7492;
                                        bool param_194 = false;
                                        float param_195 = _6234;
                                        float param_196 = _6244;
                                        float3 param_197;
                                        float4 _8037 = Sample_PrincipledDiffuse_BSDF(param_187, param_188, param_189, param_190, param_191, param_192, param_193, param_194, param_195, param_196, param_197);
                                        float3 _8043 = _8037.xyz * (1.0f - metallic);
                                        _9499 = ray.ray_depth + 1;
                                        float3 param_198 = _4926;
                                        float3 param_199 = plane_N;
                                        float3 _8059 = offset_ray(param_198, param_199);
                                        _9778 = _8059.x;
                                        _9779 = _8059.y;
                                        _9780 = _8059.z;
                                        _9785 = param_197.x;
                                        _9786 = param_197.y;
                                        _9787 = param_197.z;
                                        _9792 = ((ray.c[0] * _8043.x) * mix_weight) / param_128;
                                        _9793 = ((ray.c[1] * _8043.y) * mix_weight) / param_128;
                                        _9794 = ((ray.c[2] * _8043.z) * mix_weight) / param_128;
                                        _9494 = _8037.w;
                                    }
                                }
                                else
                                {
                                    float _8115 = param_128 + param_129;
                                    [branch]
                                    if (mix_rand < _8115)
                                    {
                                        bool _8122 = _5760 < _3223_g_params.max_spec_depth;
                                        bool _8129;
                                        if (_8122)
                                        {
                                            _8129 = _5779 < _3223_g_params.max_total_depth;
                                        }
                                        else
                                        {
                                            _8129 = _8122;
                                        }
                                        if (_8129)
                                        {
                                            float3 param_200 = T;
                                            float3 param_201 = B;
                                            float3 param_202 = N;
                                            float3 param_203 = _4803;
                                            float3 param_204;
                                            float4 _8156 = Sample_GGXSpecular_BSDF(param_200, param_201, param_202, param_203, roughness, clamp(float((_9375 >> uint(16)) & 65535u) * 1.525902189314365386962890625e-05f, 0.0f, 1.0f), _7433, _7438, _7426, _6234, _6244, param_204);
                                            float _8161 = _8156.w * param_129;
                                            _9499 = ray.ray_depth + 256;
                                            _9792 = ((ray.c[0] * _8156.x) * mix_weight) / _8161;
                                            _9793 = ((ray.c[1] * _8156.y) * mix_weight) / _8161;
                                            _9794 = ((ray.c[2] * _8156.z) * mix_weight) / _8161;
                                            _9494 = _8161;
                                            float3 param_205 = _4926;
                                            float3 param_206 = plane_N;
                                            float3 _8208 = offset_ray(param_205, param_206);
                                            _9778 = _8208.x;
                                            _9779 = _8208.y;
                                            _9780 = _8208.z;
                                            _9785 = param_204.x;
                                            _9786 = param_204.y;
                                            _9787 = param_204.z;
                                        }
                                    }
                                    else
                                    {
                                        float _8233 = _8115 + param_130;
                                        [branch]
                                        if (mix_rand < _8233)
                                        {
                                            bool _8240 = _5760 < _3223_g_params.max_spec_depth;
                                            bool _8247;
                                            if (_8240)
                                            {
                                                _8247 = _5779 < _3223_g_params.max_total_depth;
                                            }
                                            else
                                            {
                                                _8247 = _8240;
                                            }
                                            if (_8247)
                                            {
                                                float3 param_207 = T;
                                                float3 param_208 = B;
                                                float3 param_209 = N;
                                                float3 param_210 = _4803;
                                                float param_211 = _7534;
                                                float param_212 = _7525;
                                                float param_213 = _7530;
                                                float param_214 = _6234;
                                                float param_215 = _6244;
                                                float3 param_216;
                                                float4 _8271 = Sample_PrincipledClearcoat_BSDF(param_207, param_208, param_209, param_210, param_211, param_212, param_213, param_214, param_215, param_216);
                                                float _8276 = _8271.w * param_130;
                                                _9499 = ray.ray_depth + 256;
                                                _9792 = (((0.25f * ray.c[0]) * _8271.x) * mix_weight) / _8276;
                                                _9793 = (((0.25f * ray.c[1]) * _8271.y) * mix_weight) / _8276;
                                                _9794 = (((0.25f * ray.c[2]) * _8271.z) * mix_weight) / _8276;
                                                _9494 = _8276;
                                                float3 param_217 = _4926;
                                                float3 param_218 = plane_N;
                                                float3 _8326 = offset_ray(param_217, param_218);
                                                _9778 = _8326.x;
                                                _9779 = _8326.y;
                                                _9780 = _8326.z;
                                                _9785 = param_216.x;
                                                _9786 = param_216.y;
                                                _9787 = param_216.z;
                                            }
                                        }
                                        else
                                        {
                                            bool _8348 = mix_rand >= _7518;
                                            bool _8355;
                                            if (_8348)
                                            {
                                                _8355 = _5765 < _3223_g_params.max_refr_depth;
                                            }
                                            else
                                            {
                                                _8355 = _8348;
                                            }
                                            bool _8369;
                                            if (!_8355)
                                            {
                                                bool _8361 = mix_rand < _7518;
                                                bool _8368;
                                                if (_8361)
                                                {
                                                    _8368 = _5760 < _3223_g_params.max_spec_depth;
                                                }
                                                else
                                                {
                                                    _8368 = _8361;
                                                }
                                                _8369 = _8368;
                                            }
                                            else
                                            {
                                                _8369 = _8355;
                                            }
                                            bool _8376;
                                            if (_8369)
                                            {
                                                _8376 = _5779 < _3223_g_params.max_total_depth;
                                            }
                                            else
                                            {
                                                _8376 = _8369;
                                            }
                                            [branch]
                                            if (_8376)
                                            {
                                                float _8384 = mix_rand;
                                                float _8388 = (_8384 - _8233) / param_131;
                                                mix_rand = _8388;
                                                float4 F;
                                                float3 V;
                                                [branch]
                                                if (_8388 < _7518)
                                                {
                                                    float3 param_219 = T;
                                                    float3 param_220 = B;
                                                    float3 param_221 = N;
                                                    float3 param_222 = _4803;
                                                    float3 param_223;
                                                    float4 _8408 = Sample_GGXSpecular_BSDF(param_219, param_220, param_221, param_222, roughness, 0.0f, 1.0f, 0.0f, 1.0f.xxx, _6234, _6244, param_223);
                                                    V = param_223;
                                                    F = _8408;
                                                    _9499 = ray.ray_depth + 256;
                                                    float3 param_224 = _4926;
                                                    float3 param_225 = plane_N;
                                                    float3 _8419 = offset_ray(param_224, param_225);
                                                    _9778 = _8419.x;
                                                    _9779 = _8419.y;
                                                    _9780 = _8419.z;
                                                }
                                                else
                                                {
                                                    float3 param_226 = T;
                                                    float3 param_227 = B;
                                                    float3 param_228 = N;
                                                    float3 param_229 = _4803;
                                                    float param_230 = _7547;
                                                    float param_231 = _7495;
                                                    float3 param_232 = base_color;
                                                    float param_233 = _6234;
                                                    float param_234 = _6244;
                                                    float4 param_235;
                                                    float4 _8450 = Sample_GGXRefraction_BSDF(param_226, param_227, param_228, param_229, param_230, param_231, param_232, param_233, param_234, param_235);
                                                    F = _8450;
                                                    V = param_235.xyz;
                                                    _9499 = ray.ray_depth + 65536;
                                                    float3 param_236 = _4926;
                                                    float3 param_237 = -plane_N;
                                                    float3 _8464 = offset_ray(param_236, param_237);
                                                    _9778 = _8464.x;
                                                    _9779 = _8464.y;
                                                    _9780 = _8464.z;
                                                }
                                                float4 _10230 = F;
                                                float _8477 = _10230.w * param_131;
                                                float4 _10232 = _10230;
                                                _10232.w = _8477;
                                                F = _10232;
                                                _9792 = ((ray.c[0] * _10230.x) * mix_weight) / _8477;
                                                _9793 = ((ray.c[1] * _10230.y) * mix_weight) / _8477;
                                                _9794 = ((ray.c[2] * _10230.z) * mix_weight) / _8477;
                                                _9494 = _8477;
                                                _9785 = V.x;
                                                _9786 = V.y;
                                                _9787 = V.z;
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
        float _8537 = max(_9792, max(_9793, _9794));
        float _8550;
        if (_5779 >= _3223_g_params.termination_start_depth)
        {
            _8550 = max(0.0500000007450580596923828125f, 1.0f - _8537);
        }
        else
        {
            _8550 = 0.0f;
        }
        bool _8564 = (frac(asfloat(_3218.Load((_3223_g_params.hi + 6) * 4 + 0)) + _5742) >= _8550) && (_8537 > 0.0f);
        bool _8570;
        if (_8564)
        {
            _8570 = _9494 > 0.0f;
        }
        else
        {
            _8570 = _8564;
        }
        [branch]
        if (_8570)
        {
            float _8574 = 1.0f - _8550;
            float _8576 = _9792;
            float _8577 = _8576 / _8574;
            _9792 = _8577;
            float _8582 = _9793;
            float _8583 = _8582 / _8574;
            _9793 = _8583;
            float _8588 = _9794;
            float _8589 = _8588 / _8574;
            _9794 = _8589;
            uint _8593;
            _6375.InterlockedAdd(0, 1u, _8593);
            _8601.Store(_8593 * 56 + 0, asuint(_9778));
            _8601.Store(_8593 * 56 + 4, asuint(_9779));
            _8601.Store(_8593 * 56 + 8, asuint(_9780));
            _8601.Store(_8593 * 56 + 12, asuint(_9785));
            _8601.Store(_8593 * 56 + 16, asuint(_9786));
            _8601.Store(_8593 * 56 + 20, asuint(_9787));
            _8601.Store(_8593 * 56 + 24, asuint(_9494));
            _8601.Store(_8593 * 56 + 28, asuint(_8577));
            _8601.Store(_8593 * 56 + 32, asuint(_8583));
            _8601.Store(_8593 * 56 + 36, asuint(_8589));
            _8601.Store(_8593 * 56 + 40, asuint(_5726));
            _8601.Store(_8593 * 56 + 44, asuint(ray.cone_spread));
            _8601.Store(_8593 * 56 + 48, uint(ray.xy));
            _8601.Store(_8593 * 56 + 52, uint(_9499));
        }
        _8812 = float3(ray.c[0] * col.x, ray.c[1] * col.y, ray.c[2] * col.z);
        break;
    } while(false);
    return _8812;
}

void comp_main()
{
    do
    {
        bool _8669 = gl_GlobalInvocationID.x >= _3223_g_params.img_size.x;
        bool _8678;
        if (!_8669)
        {
            _8678 = gl_GlobalInvocationID.y >= _3223_g_params.img_size.y;
        }
        else
        {
            _8678 = _8669;
        }
        if (_8678)
        {
            break;
        }
        int _8685 = int(gl_GlobalInvocationID.x);
        int _8689 = int(gl_GlobalInvocationID.y);
        int _8697 = (_8689 * int(_3223_g_params.img_size.x)) + _8685;
        hit_data_t _8709;
        _8709.mask = int(_8705.Load(_8697 * 24 + 0));
        _8709.obj_index = int(_8705.Load(_8697 * 24 + 4));
        _8709.prim_index = int(_8705.Load(_8697 * 24 + 8));
        _8709.t = asfloat(_8705.Load(_8697 * 24 + 12));
        _8709.u = asfloat(_8705.Load(_8697 * 24 + 16));
        _8709.v = asfloat(_8705.Load(_8697 * 24 + 20));
        ray_data_t _8729;
        [unroll]
        for (int _119ident = 0; _119ident < 3; _119ident++)
        {
            _8729.o[_119ident] = asfloat(_8726.Load(_119ident * 4 + _8697 * 56 + 0));
        }
        [unroll]
        for (int _120ident = 0; _120ident < 3; _120ident++)
        {
            _8729.d[_120ident] = asfloat(_8726.Load(_120ident * 4 + _8697 * 56 + 12));
        }
        _8729.pdf = asfloat(_8726.Load(_8697 * 56 + 24));
        [unroll]
        for (int _121ident = 0; _121ident < 3; _121ident++)
        {
            _8729.c[_121ident] = asfloat(_8726.Load(_121ident * 4 + _8697 * 56 + 28));
        }
        _8729.cone_width = asfloat(_8726.Load(_8697 * 56 + 40));
        _8729.cone_spread = asfloat(_8726.Load(_8697 * 56 + 44));
        _8729.xy = int(_8726.Load(_8697 * 56 + 48));
        _8729.ray_depth = int(_8726.Load(_8697 * 56 + 52));
        int param = _8697;
        hit_data_t _8873 = { _8709.mask, _8709.obj_index, _8709.prim_index, _8709.t, _8709.u, _8709.v };
        hit_data_t param_1 = _8873;
        float _8911[3] = { _8729.c[0], _8729.c[1], _8729.c[2] };
        float _8904[3] = { _8729.d[0], _8729.d[1], _8729.d[2] };
        float _8897[3] = { _8729.o[0], _8729.o[1], _8729.o[2] };
        ray_data_t _8890 = { _8897, _8904, _8729.pdf, _8911, _8729.cone_width, _8729.cone_spread, _8729.xy, _8729.ray_depth };
        ray_data_t param_2 = _8890;
        float3 _8771 = ShadeSurface(param, param_1, param_2);
        g_out_img[int2(_8685, _8689)] = float4(_8771, 1.0f);
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
