struct atlas_texture_t
{
    uint size;
    uint atlas;
    uint page[4];
    uint pos[14];
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

struct Params
{
    uint2 img_size;
    uint node_index;
    int max_transp_depth;
    int blocker_lights_count;
    int _pad0;
    int _pad1;
    int _pad2;
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

struct light_t
{
    uint4 type_and_param0;
    float4 param1;
    float4 param2;
    float4 param3;
};

struct tri_accel_t
{
    float4 n_plane;
    float4 u_plane;
    float4 v_plane;
};

struct bvh_node_t
{
    float4 bbox_min;
    float4 bbox_max;
};

struct mesh_t
{
    float bbox_min[3];
    float bbox_max[3];
    uint node_index;
    uint node_count;
    uint tris_index;
    uint tris_count;
    uint vert_index;
    uint vert_count;
};

struct mesh_instance_t
{
    float4 bbox_min;
    float4 bbox_max;
};

struct transform_t
{
    row_major float4x4 xform;
    row_major float4x4 inv_xform;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

ByteAddressBuffer _231 : register(t20, space0);
ByteAddressBuffer _405 : register(t3, space0);
ByteAddressBuffer _481 : register(t10, space0);
ByteAddressBuffer _485 : register(t11, space0);
ByteAddressBuffer _584 : register(t4, space0);
ByteAddressBuffer _739 : register(t15, space0);
ByteAddressBuffer _751 : register(t14, space0);
ByteAddressBuffer _982 : register(t13, space0);
ByteAddressBuffer _997 : register(t12, space0);
ByteAddressBuffer _1078 : register(t1, space0);
ByteAddressBuffer _1082 : register(t2, space0);
ByteAddressBuffer _1087 : register(t5, space0);
ByteAddressBuffer _1094 : register(t6, space0);
ByteAddressBuffer _1099 : register(t7, space0);
ByteAddressBuffer _1103 : register(t8, space0);
ByteAddressBuffer _1109 : register(t9, space0);
cbuffer UniformParams
{
    Params _451_g_params : packoffset(c0);
};

Texture2DArray<float4> g_atlases[7] : register(t21, space0);
SamplerState _g_atlases_sampler[7] : register(s21, space0);
uniform ??? g_tlas;
RWTexture2D<float4> g_inout_img : register(u0, space0);

static uint3 gl_WorkGroupID;
static uint gl_LocalInvocationIndex;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint gl_LocalInvocationIndex : SV_GroupIndex;
};

static ??? rq;
groupshared uint g_stack[64][48];

float2 TransformUV(float2 _uv, atlas_texture_t t, int mip_level)
{
    uint _1137[14] = t.pos;
    uint _1140[14] = t.pos;
    uint _189 = t.size & 16383u;
    uint _192 = t.size >> uint(16);
    uint _193 = _192 & 16383u;
    float2 size = float2(float(_189), float(_193));
    if ((_192 & 32768u) != 0u)
    {
        size = float2(float(_189 >> uint(mip_level)), float(_193 >> uint(mip_level)));
    }
    return mad(frac(_uv), size, float2(float(_1137[mip_level] & 65535u), float((_1140[mip_level] >> uint(16)) & 65535u))) + 1.0f.xx;
}

float3 YCoCg_to_RGB(float4 col)
{
    float _131 = mad(col.z, 31.875f, 1.0f);
    float _141 = (col.x - 0.501960813999176025390625f) / _131;
    float _147 = (col.y - 0.501960813999176025390625f) / _131;
    return float3((col.w + _141) - _147, col.w + _147, (col.w - _141) - _147);
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
    atlas_texture_t _234;
    _234.size = _231.Load(index * 80 + 0);
    _234.atlas = _231.Load(index * 80 + 4);
    [unroll]
    for (int _22ident = 0; _22ident < 4; _22ident++)
    {
        _234.page[_22ident] = _231.Load(_22ident * 4 + index * 80 + 8);
    }
    [unroll]
    for (int _23ident = 0; _23ident < 14; _23ident++)
    {
        _234.pos[_23ident] = _231.Load(_23ident * 4 + index * 80 + 24);
    }
    atlas_texture_t _235;
    _235.size = _234.size;
    _235.atlas = _234.atlas;
    _235.page[0] = _234.page[0];
    _235.page[1] = _234.page[1];
    _235.page[2] = _234.page[2];
    _235.page[3] = _234.page[3];
    _235.pos[0] = _234.pos[0];
    _235.pos[1] = _234.pos[1];
    _235.pos[2] = _234.pos[2];
    _235.pos[3] = _234.pos[3];
    _235.pos[4] = _234.pos[4];
    _235.pos[5] = _234.pos[5];
    _235.pos[6] = _234.pos[6];
    _235.pos[7] = _234.pos[7];
    _235.pos[8] = _234.pos[8];
    _235.pos[9] = _234.pos[9];
    _235.pos[10] = _234.pos[10];
    _235.pos[11] = _234.pos[11];
    _235.pos[12] = _234.pos[12];
    _235.pos[13] = _234.pos[13];
    uint _1146[4] = _235.page;
    uint _264 = _235.atlas;
    float4 res = g_atlases[NonUniformResourceIndex(_264)].SampleLevel(_g_atlases_sampler[NonUniformResourceIndex(_264)], float3(TransformUV(uvs, _235, lod) * 0.000118371215648949146270751953125f.xx, float((_1146[lod / 4] >> uint((lod % 4) * 8)) & 255u)), 0.0f);
    bool _280;
    if (maybe_YCoCg)
    {
        _280 = _235.atlas == 4u;
    }
    else
    {
        _280 = maybe_YCoCg;
    }
    if (_280)
    {
        float4 param = res;
        res = float4(YCoCg_to_RGB(param), 1.0f);
    }
    bool _299;
    if (maybe_SRGB)
    {
        _299 = (_235.size & 32768u) != 0u;
    }
    else
    {
        _299 = maybe_SRGB;
    }
    if (_299)
    {
        float3 param_1 = res.xyz;
        float3 _305 = srgb_to_rgb(param_1);
        float4 _1246 = res;
        _1246.x = _305.x;
        float4 _1248 = _1246;
        _1248.y = _305.y;
        float4 _1250 = _1248;
        _1250.z = _305.z;
        res = _1250;
    }
    return res;
}

float4 SampleBilinear(uint index, float2 uvs, int lod)
{
    return SampleBilinear(index, uvs, lod, false, false);
}

float lum(float3 color)
{
    return mad(0.072168998420238494873046875f, color.z, mad(0.21267099678516387939453125f, color.x, 0.71516001224517822265625f * color.y));
}

float3 IntersectSceneShadow(shadow_ray_t r)
{
    bool _1119 = false;
    float3 _1116;
    do
    {
        float3 ro = float3(r.o[0], r.o[1], r.o[2]);
        float3 _334 = float3(r.d[0], r.d[1], r.d[2]);
        float3 rc = float3(r.c[0], r.c[1], r.c[2]);
        int depth = r.depth >> 24;
        float _352;
        if (r.dist > 0.0f)
        {
            _352 = r.dist;
        }
        else
        {
            _352 = 3402823346297367662189621542912.0f;
        }
        float dist = _352;
        while (dist > 9.9999997473787516355514526367188e-06f)
        {
            rayQueryInitializeEXT(rq, g_tlas, 0u, 255u, ro, 0.0f, _334, dist);
            for (;;)
            {
                bool _383 = rayQueryProceedEXT(rq);
                if (_383)
                {
                    uint _384 = rayQueryGetIntersectionTypeEXT(rq, bool(0));
                    if (_384 == 0u)
                    {
                        rayQueryConfirmIntersectionEXT(rq);
                    }
                    continue;
                }
                else
                {
                    break;
                }
            }
            uint _389 = rayQueryGetIntersectionTypeEXT(rq, bool(1));
            if (_389 != 0u)
            {
                int _394 = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, bool(1));
                int _396 = rayQueryGetIntersectionInstanceIdEXT(rq, bool(1));
                int _399 = rayQueryGetIntersectionPrimitiveIndexEXT(rq, bool(1));
                int _400 = _394 + _399;
                uint _412 = (_405.Load(_400 * 4 + 0) >> 16u) & 65535u;
                uint _417 = _405.Load(_400 * 4 + 0) & 65535u;
                bool _420 = rayQueryGetIntersectionFrontFaceEXT(rq, bool(1));
                bool _421 = !_420;
                bool _424 = !_421;
                bool _430;
                if (_424)
                {
                    _430 = (_412 & 32768u) != 0u;
                }
                else
                {
                    _430 = _424;
                }
                bool _441;
                if (!_430)
                {
                    bool _440;
                    if (_421)
                    {
                        _440 = (_417 & 32768u) != 0u;
                    }
                    else
                    {
                        _440 = _421;
                    }
                    _441 = _440;
                }
                else
                {
                    _441 = _430;
                }
                bool _456;
                if (!_441)
                {
                    _456 = depth > _451_g_params.max_transp_depth;
                }
                else
                {
                    _456 = _441;
                }
                if (_456)
                {
                    _1119 = true;
                    _1116 = 0.0f.xxx;
                    break;
                }
                int _487 = _400 * 3;
                vertex_t _493;
                [unroll]
                for (int _24ident = 0; _24ident < 3; _24ident++)
                {
                    _493.p[_24ident] = asfloat(_481.Load(_24ident * 4 + _485.Load(_487 * 4 + 0) * 52 + 0));
                }
                [unroll]
                for (int _25ident = 0; _25ident < 3; _25ident++)
                {
                    _493.n[_25ident] = asfloat(_481.Load(_25ident * 4 + _485.Load(_487 * 4 + 0) * 52 + 12));
                }
                [unroll]
                for (int _26ident = 0; _26ident < 3; _26ident++)
                {
                    _493.b[_26ident] = asfloat(_481.Load(_26ident * 4 + _485.Load(_487 * 4 + 0) * 52 + 24));
                }
                [unroll]
                for (int _27ident = 0; _27ident < 2; _27ident++)
                {
                    [unroll]
                    for (int _28ident = 0; _28ident < 2; _28ident++)
                    {
                        _493.t[_27ident][_28ident] = asfloat(_481.Load(_28ident * 4 + _27ident * 8 + _485.Load(_487 * 4 + 0) * 52 + 36));
                    }
                }
                vertex_t _494;
                _494.p[0] = _493.p[0];
                _494.p[1] = _493.p[1];
                _494.p[2] = _493.p[2];
                _494.n[0] = _493.n[0];
                _494.n[1] = _493.n[1];
                _494.n[2] = _493.n[2];
                _494.b[0] = _493.b[0];
                _494.b[1] = _493.b[1];
                _494.b[2] = _493.b[2];
                _494.t[0][0] = _493.t[0][0];
                _494.t[0][1] = _493.t[0][1];
                _494.t[1][0] = _493.t[1][0];
                _494.t[1][1] = _493.t[1][1];
                vertex_t _502;
                [unroll]
                for (int _29ident = 0; _29ident < 3; _29ident++)
                {
                    _502.p[_29ident] = asfloat(_481.Load(_29ident * 4 + _485.Load((_487 + 1) * 4 + 0) * 52 + 0));
                }
                [unroll]
                for (int _30ident = 0; _30ident < 3; _30ident++)
                {
                    _502.n[_30ident] = asfloat(_481.Load(_30ident * 4 + _485.Load((_487 + 1) * 4 + 0) * 52 + 12));
                }
                [unroll]
                for (int _31ident = 0; _31ident < 3; _31ident++)
                {
                    _502.b[_31ident] = asfloat(_481.Load(_31ident * 4 + _485.Load((_487 + 1) * 4 + 0) * 52 + 24));
                }
                [unroll]
                for (int _32ident = 0; _32ident < 2; _32ident++)
                {
                    [unroll]
                    for (int _33ident = 0; _33ident < 2; _33ident++)
                    {
                        _502.t[_32ident][_33ident] = asfloat(_481.Load(_33ident * 4 + _32ident * 8 + _485.Load((_487 + 1) * 4 + 0) * 52 + 36));
                    }
                }
                vertex_t _503;
                _503.p[0] = _502.p[0];
                _503.p[1] = _502.p[1];
                _503.p[2] = _502.p[2];
                _503.n[0] = _502.n[0];
                _503.n[1] = _502.n[1];
                _503.n[2] = _502.n[2];
                _503.b[0] = _502.b[0];
                _503.b[1] = _502.b[1];
                _503.b[2] = _502.b[2];
                _503.t[0][0] = _502.t[0][0];
                _503.t[0][1] = _502.t[0][1];
                _503.t[1][0] = _502.t[1][0];
                _503.t[1][1] = _502.t[1][1];
                vertex_t _511;
                [unroll]
                for (int _34ident = 0; _34ident < 3; _34ident++)
                {
                    _511.p[_34ident] = asfloat(_481.Load(_34ident * 4 + _485.Load((_487 + 2) * 4 + 0) * 52 + 0));
                }
                [unroll]
                for (int _35ident = 0; _35ident < 3; _35ident++)
                {
                    _511.n[_35ident] = asfloat(_481.Load(_35ident * 4 + _485.Load((_487 + 2) * 4 + 0) * 52 + 12));
                }
                [unroll]
                for (int _36ident = 0; _36ident < 3; _36ident++)
                {
                    _511.b[_36ident] = asfloat(_481.Load(_36ident * 4 + _485.Load((_487 + 2) * 4 + 0) * 52 + 24));
                }
                [unroll]
                for (int _37ident = 0; _37ident < 2; _37ident++)
                {
                    [unroll]
                    for (int _38ident = 0; _38ident < 2; _38ident++)
                    {
                        _511.t[_37ident][_38ident] = asfloat(_481.Load(_38ident * 4 + _37ident * 8 + _485.Load((_487 + 2) * 4 + 0) * 52 + 36));
                    }
                }
                vertex_t _512;
                _512.p[0] = _511.p[0];
                _512.p[1] = _511.p[1];
                _512.p[2] = _511.p[2];
                _512.n[0] = _511.n[0];
                _512.n[1] = _511.n[1];
                _512.n[2] = _511.n[2];
                _512.b[0] = _511.b[0];
                _512.b[1] = _511.b[1];
                _512.b[2] = _511.b[2];
                _512.t[0][0] = _511.t[0][0];
                _512.t[0][1] = _511.t[0][1];
                _512.t[1][0] = _511.t[1][0];
                _512.t[1][1] = _511.t[1][1];
                float2 _514 = rayQueryGetIntersectionBarycentricsEXT(rq, bool(1));
                float2 _547 = ((float2(_494.t[0][0], _494.t[0][1]) * ((1.0f - _514.x) - _514.y)) + (float2(_503.t[0][0], _503.t[0][1]) * _514.x)) + (float2(_512.t[0][0], _512.t[0][1]) * _514.y);
                g_stack[gl_LocalInvocationIndex][0] = (_421 ? _417 : _412) & 16383u;
                g_stack[gl_LocalInvocationIndex][1] = 1065353216u;
                int stack_size = 1;
                float3 throughput = 0.0f.xxx;
                for (;;)
                {
                    int _570 = stack_size;
                    stack_size = _570 - 1;
                    if (_570 != 0)
                    {
                        int _586 = stack_size;
                        int _587 = 2 * _586;
                        material_t _593;
                        [unroll]
                        for (int _39ident = 0; _39ident < 5; _39ident++)
                        {
                            _593.textures[_39ident] = _584.Load(_39ident * 4 + g_stack[gl_LocalInvocationIndex][_587] * 76 + 0);
                        }
                        [unroll]
                        for (int _40ident = 0; _40ident < 3; _40ident++)
                        {
                            _593.base_color[_40ident] = asfloat(_584.Load(_40ident * 4 + g_stack[gl_LocalInvocationIndex][_587] * 76 + 20));
                        }
                        _593.flags = _584.Load(g_stack[gl_LocalInvocationIndex][_587] * 76 + 32);
                        _593.type = _584.Load(g_stack[gl_LocalInvocationIndex][_587] * 76 + 36);
                        _593.tangent_rotation_or_strength = asfloat(_584.Load(g_stack[gl_LocalInvocationIndex][_587] * 76 + 40));
                        _593.roughness_and_anisotropic = _584.Load(g_stack[gl_LocalInvocationIndex][_587] * 76 + 44);
                        _593.ior = asfloat(_584.Load(g_stack[gl_LocalInvocationIndex][_587] * 76 + 48));
                        _593.sheen_and_sheen_tint = _584.Load(g_stack[gl_LocalInvocationIndex][_587] * 76 + 52);
                        _593.tint_and_metallic = _584.Load(g_stack[gl_LocalInvocationIndex][_587] * 76 + 56);
                        _593.transmission_and_transmission_roughness = _584.Load(g_stack[gl_LocalInvocationIndex][_587] * 76 + 60);
                        _593.specular_and_specular_tint = _584.Load(g_stack[gl_LocalInvocationIndex][_587] * 76 + 64);
                        _593.clearcoat_and_clearcoat_roughness = _584.Load(g_stack[gl_LocalInvocationIndex][_587] * 76 + 68);
                        _593.normal_map_strength_unorm = _584.Load(g_stack[gl_LocalInvocationIndex][_587] * 76 + 72);
                        material_t _594;
                        _594.textures[0] = _593.textures[0];
                        _594.textures[1] = _593.textures[1];
                        _594.textures[2] = _593.textures[2];
                        _594.textures[3] = _593.textures[3];
                        _594.textures[4] = _593.textures[4];
                        _594.base_color[0] = _593.base_color[0];
                        _594.base_color[1] = _593.base_color[1];
                        _594.base_color[2] = _593.base_color[2];
                        _594.flags = _593.flags;
                        _594.type = _593.type;
                        _594.tangent_rotation_or_strength = _593.tangent_rotation_or_strength;
                        _594.roughness_and_anisotropic = _593.roughness_and_anisotropic;
                        _594.ior = _593.ior;
                        _594.sheen_and_sheen_tint = _593.sheen_and_sheen_tint;
                        _594.tint_and_metallic = _593.tint_and_metallic;
                        _594.transmission_and_transmission_roughness = _593.transmission_and_transmission_roughness;
                        _594.specular_and_specular_tint = _593.specular_and_specular_tint;
                        _594.clearcoat_and_clearcoat_roughness = _593.clearcoat_and_clearcoat_roughness;
                        _594.normal_map_strength_unorm = _593.normal_map_strength_unorm;
                        uint _601 = g_stack[gl_LocalInvocationIndex][_587 + 1];
                        float _602 = asfloat(_601);
                        if (_594.type == 4u)
                        {
                            float mix_val = _594.tangent_rotation_or_strength;
                            if (_594.textures[1] != 4294967295u)
                            {
                                mix_val *= SampleBilinear(_594.textures[1], _547, 0).x;
                            }
                            int _626 = 2 * stack_size;
                            g_stack[gl_LocalInvocationIndex][_626] = _594.textures[3];
                            g_stack[gl_LocalInvocationIndex][_626 + 1] = asuint(_602 * (1.0f - mix_val));
                            int _645 = 2 * (stack_size + 1);
                            g_stack[gl_LocalInvocationIndex][_645] = _594.textures[4];
                            g_stack[gl_LocalInvocationIndex][_645 + 1] = asuint(_602 * mix_val);
                            stack_size += 2;
                        }
                        else
                        {
                            if (_594.type == 5u)
                            {
                                throughput += (float3(_594.base_color[0], _594.base_color[1], _594.base_color[2]) * _602);
                            }
                        }
                        continue;
                    }
                    else
                    {
                        break;
                    }
                }
                float3 _679 = rc;
                float3 _680 = _679 * throughput;
                rc = _680;
                if (lum(_680) < 1.0000000116860974230803549289703e-07f)
                {
                    break;
                }
            }
            float _689 = rayQueryGetIntersectionTEXT(rq, bool(1));
            float _690 = _689 + 9.9999997473787516355514526367188e-06f;
            ro += (_334 * _690);
            dist -= _690;
            depth++;
        }
        if (_1119)
        {
            break;
        }
        _1119 = true;
        _1116 = rc;
        break;
    } while(false);
    return _1116;
}

float IntersectAreaLightsShadow(shadow_ray_t r)
{
    bool _1126 = false;
    float _1123;
    do
    {
        float3 _711 = float3(r.o[0], r.o[1], r.o[2]);
        float3 _719 = float3(r.d[0], r.d[1], r.d[2]);
        float _723 = abs(r.dist);
        for (uint li = 0u; li < uint(_451_g_params.blocker_lights_count); li++)
        {
            light_t _755;
            _755.type_and_param0 = _751.Load4(_739.Load(li * 4 + 0) * 64 + 0);
            _755.param1 = asfloat(_751.Load4(_739.Load(li * 4 + 0) * 64 + 16));
            _755.param2 = asfloat(_751.Load4(_739.Load(li * 4 + 0) * 64 + 32));
            _755.param3 = asfloat(_751.Load4(_739.Load(li * 4 + 0) * 64 + 48));
            light_t _756;
            _756.type_and_param0 = _755.type_and_param0;
            _756.param1 = _755.param1;
            _756.param2 = _755.param2;
            _756.param3 = _755.param3;
            bool _761 = (_756.type_and_param0.x & 128u) != 0u;
            bool _767;
            if (_761)
            {
                _767 = r.dist >= 0.0f;
            }
            else
            {
                _767 = _761;
            }
            [branch]
            if (_767)
            {
                continue;
            }
            uint _775 = _756.type_and_param0.x & 31u;
            if (_775 == 4u)
            {
                float3 light_u = _756.param2.xyz;
                float3 light_v = _756.param3.xyz;
                float3 _796 = normalize(cross(_756.param2.xyz, _756.param3.xyz));
                float _804 = dot(_719, _796);
                float _812 = (dot(_796, _756.param1.xyz) - dot(_796, _711)) / _804;
                if (((_804 < 0.0f) && (_812 > 9.9999999747524270787835121154785e-07f)) && (_812 < _723))
                {
                    float3 _825 = light_u;
                    float3 _830 = _825 / dot(_825, _825).xxx;
                    light_u = _830;
                    light_v /= dot(light_v, light_v).xxx;
                    float3 _846 = (_711 + (_719 * _812)) - _756.param1.xyz;
                    float _850 = dot(_830, _846);
                    if ((_850 >= (-0.5f)) && (_850 <= 0.5f))
                    {
                        float _863 = dot(light_v, _846);
                        if ((_863 >= (-0.5f)) && (_863 <= 0.5f))
                        {
                            _1126 = true;
                            _1123 = 0.0f;
                            break;
                        }
                    }
                }
            }
            else
            {
                if (_775 == 5u)
                {
                    float3 light_u_1 = _756.param2.xyz;
                    float3 light_v_1 = _756.param3.xyz;
                    float3 _893 = normalize(cross(_756.param2.xyz, _756.param3.xyz));
                    float _901 = dot(_719, _893);
                    float _909 = (dot(_893, _756.param1.xyz) - dot(_893, _711)) / _901;
                    if (((_901 < 0.0f) && (_909 > 9.9999999747524270787835121154785e-07f)) && (_909 < _723))
                    {
                        float3 _921 = light_u_1;
                        float3 _926 = _921 / dot(_921, _921).xxx;
                        light_u_1 = _926;
                        float3 _927 = light_v_1;
                        float3 _932 = _927 / dot(_927, _927).xxx;
                        light_v_1 = _932;
                        float3 _942 = (_711 + (_719 * _909)) - _756.param1.xyz;
                        float _946 = dot(_926, _942);
                        float _950 = dot(_932, _942);
                        if (sqrt(mad(_946, _946, _950 * _950)) <= 0.5f)
                        {
                            _1126 = true;
                            _1123 = 0.0f;
                            break;
                        }
                    }
                }
            }
        }
        if (_1126)
        {
            break;
        }
        _1126 = true;
        _1123 = 1.0f;
        break;
    } while(false);
    return _1123;
}

void comp_main()
{
    do
    {
        int _976 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_976) >= _982.Load(12))
        {
            break;
        }
        shadow_ray_t _1001;
        [unroll]
        for (int _41ident = 0; _41ident < 3; _41ident++)
        {
            _1001.o[_41ident] = asfloat(_997.Load(_41ident * 4 + _976 * 48 + 0));
        }
        _1001.depth = int(_997.Load(_976 * 48 + 12));
        [unroll]
        for (int _42ident = 0; _42ident < 3; _42ident++)
        {
            _1001.d[_42ident] = asfloat(_997.Load(_42ident * 4 + _976 * 48 + 16));
        }
        _1001.dist = asfloat(_997.Load(_976 * 48 + 28));
        [unroll]
        for (int _43ident = 0; _43ident < 3; _43ident++)
        {
            _1001.c[_43ident] = asfloat(_997.Load(_43ident * 4 + _976 * 48 + 32));
        }
        _1001.xy = int(_997.Load(_976 * 48 + 44));
        shadow_ray_t _1002;
        _1002.o[0] = _1001.o[0];
        _1002.o[1] = _1001.o[1];
        _1002.o[2] = _1001.o[2];
        _1002.depth = _1001.depth;
        _1002.d[0] = _1001.d[0];
        _1002.d[1] = _1001.d[1];
        _1002.d[2] = _1001.d[2];
        _1002.dist = _1001.dist;
        _1002.c[0] = _1001.c[0];
        _1002.c[1] = _1001.c[1];
        _1002.c[2] = _1001.c[2];
        _1002.xy = _1001.xy;
        shadow_ray_t param = _1002;
        float3 _1006 = IntersectSceneShadow(param);
        shadow_ray_t param_1 = _1002;
        float3 _1010 = _1006 * IntersectAreaLightsShadow(param_1);
        if (lum(_1010) > 0.0f)
        {
            int2 _1035 = int2((_1002.xy >> 16) & 65535, _1002.xy & 65535);
            g_inout_img[_1035] = float4(g_inout_img[_1035].xyz + _1010, 1.0f);
        }
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
