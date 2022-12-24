struct tri_accel_t
{
    float4 n_plane;
    float4 u_plane;
    float4 v_plane;
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

struct bvh_node_t
{
    float4 bbox_min;
    float4 bbox_max;
};

struct mesh_instance_t
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

struct transform_t
{
    row_major float4x4 xform;
    row_major float4x4 inv_xform;
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
    uint node_index;
    float cam_clip_end;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

static const hit_data_t _1202 = { 0, 0, 0, 3402823346297367662189621542912.0f, 0.0f, 0.0f };

ByteAddressBuffer _298 : register(t1, space0);
ByteAddressBuffer _513 : register(t3, space0);
ByteAddressBuffer _760 : register(t5, space0);
ByteAddressBuffer _764 : register(t6, space0);
ByteAddressBuffer _786 : register(t4, space0);
ByteAddressBuffer _832 : register(t7, space0);
ByteAddressBuffer _925 : register(t10, space0);
ByteAddressBuffer _940 : register(t8, space0);
ByteAddressBuffer _1016 : register(t2, space0);
RWByteAddressBuffer _1038 : register(u0, space0);
cbuffer UniformParams
{
    Params _993_g_params : packoffset(c0);
};


static uint3 gl_WorkGroupID;
static uint gl_LocalInvocationIndex;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint gl_LocalInvocationIndex : SV_GroupIndex;
};

groupshared uint g_stack[64][48];

float3 safe_invert(float3 v)
{
    float3 inv_v = 1.0f.xxx / v;
    bool _76 = v.x <= 1.0000000116860974230803549289703e-07f;
    bool _83;
    if (_76)
    {
        _83 = v.x >= 0.0f;
    }
    else
    {
        _83 = _76;
    }
    if (_83)
    {
        float3 _1171 = inv_v;
        _1171.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _1171;
    }
    else
    {
        bool _92 = v.x >= (-1.0000000116860974230803549289703e-07f);
        bool _98;
        if (_92)
        {
            _98 = v.x < 0.0f;
        }
        else
        {
            _98 = _92;
        }
        if (_98)
        {
            float3 _1173 = inv_v;
            _1173.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _1173;
        }
    }
    bool _106 = v.y <= 1.0000000116860974230803549289703e-07f;
    bool _112;
    if (_106)
    {
        _112 = v.y >= 0.0f;
    }
    else
    {
        _112 = _106;
    }
    if (_112)
    {
        float3 _1175 = inv_v;
        _1175.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _1175;
    }
    else
    {
        bool _119 = v.y >= (-1.0000000116860974230803549289703e-07f);
        bool _125;
        if (_119)
        {
            _125 = v.y < 0.0f;
        }
        else
        {
            _125 = _119;
        }
        if (_125)
        {
            float3 _1177 = inv_v;
            _1177.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _1177;
        }
    }
    bool _132 = v.z <= 1.0000000116860974230803549289703e-07f;
    bool _138;
    if (_132)
    {
        _138 = v.z >= 0.0f;
    }
    else
    {
        _138 = _132;
    }
    if (_138)
    {
        float3 _1179 = inv_v;
        _1179.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _1179;
    }
    else
    {
        bool _145 = v.z >= (-1.0000000116860974230803549289703e-07f);
        bool _151;
        if (_145)
        {
            _151 = v.z < 0.0f;
        }
        else
        {
            _151 = _145;
        }
        if (_151)
        {
            float3 _1181 = inv_v;
            _1181.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _1181;
        }
    }
    return inv_v;
}

bool _bbox_test_fma(float3 inv_d, float3 neg_inv_d_o, float t, float3 bbox_min, float3 bbox_max)
{
    float _391 = mad(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float _399 = mad(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float _414 = mad(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    float _421 = mad(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    float _438 = mad(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    float _445 = mad(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    float _450 = max(max(min(_391, _399), min(_414, _421)), min(_438, _445));
    float _458 = min(min(max(_391, _399), max(_414, _421)), max(_438, _445)) * 1.0000002384185791015625f;
    return ((_450 <= _458) && (_450 <= t)) && (_458 > 0.0f);
}

void IntersectTri(float3 ro, float3 rd, tri_accel_t tri, uint prim_index, inout hit_data_t inter)
{
    do
    {
        float _165 = dot(rd, tri.n_plane.xyz);
        float _175 = tri.n_plane.w - dot(ro, tri.n_plane.xyz);
        if (sign(_175) != sign(mad(_165, inter.t, -_175)))
        {
            break;
        }
        float3 _197 = (ro * _165) + (rd * _175);
        float _209 = mad(_165, tri.u_plane.w, dot(_197, tri.u_plane.xyz));
        float _214 = _165 - _209;
        if (sign(_209) != sign(_214))
        {
            break;
        }
        float _231 = mad(_165, tri.v_plane.w, dot(_197, tri.v_plane.xyz));
        if (sign(_231) != sign(_214 - _231))
        {
            break;
        }
        float _246 = 1.0f / _165;
        inter.mask = -1;
        int _251;
        if (_165 < 0.0f)
        {
            _251 = int(prim_index);
        }
        else
        {
            _251 = (-1) - int(prim_index);
        }
        inter.prim_index = _251;
        inter.t = _175 * _246;
        inter.u = _209 * _246;
        inter.v = _231 * _246;
        break;
    } while(false);
}

void IntersectTris_ClosestHit(float3 ro, float3 rd, int tri_start, int tri_end, int obj_index, inout hit_data_t out_inter)
{
    int _1109 = 0;
    int _1110 = obj_index;
    float _1112 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _1111;
    float _1113;
    float _1114;
    for (int i = tri_start; i < tri_end; )
    {
        param = ro;
        param_1 = rd;
        tri_accel_t _309;
        _309.n_plane = asfloat(_298.Load4(i * 48 + 0));
        _309.u_plane = asfloat(_298.Load4(i * 48 + 16));
        _309.v_plane = asfloat(_298.Load4(i * 48 + 32));
        param_2.n_plane = _309.n_plane;
        param_2.u_plane = _309.u_plane;
        param_2.v_plane = _309.v_plane;
        param_3 = uint(i);
        hit_data_t _1121 = { _1109, _1110, _1111, _1112, _1113, _1114 };
        param_4 = _1121;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _1109 = param_4.mask;
        _1110 = param_4.obj_index;
        _1111 = param_4.prim_index;
        _1112 = param_4.t;
        _1113 = param_4.u;
        _1114 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _1109;
    int _332;
    if (_1109 != 0)
    {
        _332 = _1110;
    }
    else
    {
        _332 = out_inter.obj_index;
    }
    out_inter.obj_index = _332;
    int _345;
    if (_1109 != 0)
    {
        _345 = _1111;
    }
    else
    {
        _345 = out_inter.prim_index;
    }
    out_inter.prim_index = _345;
    out_inter.t = _1112;
    float _361;
    if (_1109 != 0)
    {
        _361 = _1113;
    }
    else
    {
        _361 = out_inter.u;
    }
    out_inter.u = _361;
    float _374;
    if (_1109 != 0)
    {
        _374 = _1114;
    }
    else
    {
        _374 = out_inter.v;
    }
    out_inter.v = _374;
}

void Traverse_MicroTree_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    float3 _475 = (-inv_d) * ro;
    uint _477 = stack_size;
    uint _487 = stack_size;
    stack_size = _487 + uint(1);
    g_stack[gl_LocalInvocationIndex][_487] = node_index;
    uint _561;
    uint _585;
    while (stack_size != _477)
    {
        uint _502 = stack_size;
        uint _503 = _502 - uint(1);
        stack_size = _503;
        bvh_node_t _517;
        _517.bbox_min = asfloat(_513.Load4(g_stack[gl_LocalInvocationIndex][_503] * 32 + 0));
        _517.bbox_max = asfloat(_513.Load4(g_stack[gl_LocalInvocationIndex][_503] * 32 + 16));
        float3 param = inv_d;
        float3 param_1 = _475;
        float param_2 = inter.t;
        float3 param_3 = _517.bbox_min.xyz;
        float3 param_4 = _517.bbox_max.xyz;
        if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
        {
            continue;
        }
        uint _544 = asuint(_517.bbox_min.w);
        if ((_544 & 2147483648u) == 0u)
        {
            uint _551 = stack_size;
            stack_size = _551 + uint(1);
            uint _555 = asuint(_517.bbox_max.w);
            uint _557 = _555 >> uint(30);
            if (rd[_557] < 0.0f)
            {
                _561 = _544;
            }
            else
            {
                _561 = _555 & 1073741823u;
            }
            g_stack[gl_LocalInvocationIndex][_551] = _561;
            uint _576 = stack_size;
            stack_size = _576 + uint(1);
            if (rd[_557] < 0.0f)
            {
                _585 = _555 & 1073741823u;
            }
            else
            {
                _585 = _544;
            }
            g_stack[gl_LocalInvocationIndex][_576] = _585;
        }
        else
        {
            int _605 = int(_544 & 2147483647u);
            float3 param_5 = ro;
            float3 param_6 = rd;
            int param_7 = _605;
            int param_8 = _605 + asint(_517.bbox_max.w);
            int param_9 = obj_index;
            hit_data_t param_10 = inter;
            IntersectTris_ClosestHit(param_5, param_6, param_7, param_8, param_9, param_10);
            inter = param_10;
        }
    }
}

void Traverse_MacroTree_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    float3 _630 = (-orig_inv_rd) * orig_ro;
    uint stack_size = 1u;
    g_stack[gl_LocalInvocationIndex][0u] = node_index;
    uint _695;
    uint _718;
    while (stack_size != 0u)
    {
        uint _646 = stack_size;
        uint _647 = _646 - uint(1);
        stack_size = _647;
        bvh_node_t _653;
        _653.bbox_min = asfloat(_513.Load4(g_stack[gl_LocalInvocationIndex][_647] * 32 + 0));
        _653.bbox_max = asfloat(_513.Load4(g_stack[gl_LocalInvocationIndex][_647] * 32 + 16));
        float3 param = orig_inv_rd;
        float3 param_1 = _630;
        float param_2 = inter.t;
        float3 param_3 = _653.bbox_min.xyz;
        float3 param_4 = _653.bbox_max.xyz;
        if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
        {
            continue;
        }
        uint _680 = asuint(_653.bbox_min.w);
        if ((_680 & 2147483648u) == 0u)
        {
            uint _686 = stack_size;
            stack_size = _686 + uint(1);
            uint _690 = asuint(_653.bbox_max.w);
            uint _691 = _690 >> uint(30);
            if (orig_rd[_691] < 0.0f)
            {
                _695 = _680;
            }
            else
            {
                _695 = _690 & 1073741823u;
            }
            g_stack[gl_LocalInvocationIndex][_686] = _695;
            uint _709 = stack_size;
            stack_size = _709 + uint(1);
            if (orig_rd[_691] < 0.0f)
            {
                _718 = _690 & 1073741823u;
            }
            else
            {
                _718 = _680;
            }
            g_stack[gl_LocalInvocationIndex][_709] = _718;
        }
        else
        {
            uint _736 = _680 & 2147483647u;
            uint _740 = asuint(_653.bbox_max.w);
            for (uint i = _736; i < (_736 + _740); i++)
            {
                mesh_instance_t _771;
                _771.bbox_min = asfloat(_760.Load4(_764.Load(i * 4 + 0) * 32 + 0));
                _771.bbox_max = asfloat(_760.Load4(_764.Load(i * 4 + 0) * 32 + 16));
                mesh_t _792;
                [unroll]
                for (int _2ident = 0; _2ident < 3; _2ident++)
                {
                    _792.bbox_min[_2ident] = asfloat(_786.Load(_2ident * 4 + asuint(_771.bbox_max.w) * 48 + 0));
                }
                [unroll]
                for (int _3ident = 0; _3ident < 3; _3ident++)
                {
                    _792.bbox_max[_3ident] = asfloat(_786.Load(_3ident * 4 + asuint(_771.bbox_max.w) * 48 + 12));
                }
                _792.node_index = _786.Load(asuint(_771.bbox_max.w) * 48 + 24);
                _792.node_count = _786.Load(asuint(_771.bbox_max.w) * 48 + 28);
                _792.tris_index = _786.Load(asuint(_771.bbox_max.w) * 48 + 32);
                _792.tris_count = _786.Load(asuint(_771.bbox_max.w) * 48 + 36);
                _792.vert_index = _786.Load(asuint(_771.bbox_max.w) * 48 + 40);
                _792.vert_count = _786.Load(asuint(_771.bbox_max.w) * 48 + 44);
                transform_t _838;
                _838.xform = asfloat(uint4x4(_832.Load4(asuint(_771.bbox_min.w) * 128 + 0), _832.Load4(asuint(_771.bbox_min.w) * 128 + 16), _832.Load4(asuint(_771.bbox_min.w) * 128 + 32), _832.Load4(asuint(_771.bbox_min.w) * 128 + 48)));
                _838.inv_xform = asfloat(uint4x4(_832.Load4(asuint(_771.bbox_min.w) * 128 + 64), _832.Load4(asuint(_771.bbox_min.w) * 128 + 80), _832.Load4(asuint(_771.bbox_min.w) * 128 + 96), _832.Load4(asuint(_771.bbox_min.w) * 128 + 112)));
                float3 param_5 = orig_inv_rd;
                float3 param_6 = _630;
                float param_7 = inter.t;
                float3 param_8 = _771.bbox_min.xyz;
                float3 param_9 = _771.bbox_max.xyz;
                if (!_bbox_test_fma(param_5, param_6, param_7, param_8, param_9))
                {
                    continue;
                }
                float3 _883 = mul(float4(orig_rd, 0.0f), _838.inv_xform).xyz;
                float3 param_10 = _883;
                float3 param_11 = mul(float4(orig_ro, 1.0f), _838.inv_xform).xyz;
                float3 param_12 = _883;
                float3 param_13 = safe_invert(param_10);
                int param_14 = int(_764.Load(i * 4 + 0));
                uint param_15 = _792.node_index;
                uint param_16 = stack_size;
                hit_data_t param_17 = inter;
                Traverse_MicroTree_WithStack(param_11, param_12, param_13, param_14, param_15, param_16, param_17);
                inter = param_17;
            }
        }
    }
}

void comp_main()
{
    do
    {
        int _919 = int((gl_WorkGroupID.x * 64u) + gl_LocalInvocationIndex);
        if (uint(_919) >= _925.Load(4))
        {
            break;
        }
        float3 _976 = float3(asfloat(_940.Load(_919 * 56 + 12)), asfloat(_940.Load(_919 * 56 + 16)), asfloat(_940.Load(_919 * 56 + 20)));
        float3 param = _976;
        int _1083 = 0;
        int _1085 = 0;
        int _1084 = 0;
        float _1086 = 3402823346297367662189621542912.0f;
        float _1088 = 0.0f;
        float _1087 = 0.0f;
        float3 param_1 = float3(asfloat(_940.Load(_919 * 56 + 0)), asfloat(_940.Load(_919 * 56 + 4)), asfloat(_940.Load(_919 * 56 + 8)));
        float3 param_2 = _976;
        float3 param_3 = safe_invert(param);
        uint param_4 = _993_g_params.node_index;
        hit_data_t param_5 = _1202;
        Traverse_MacroTree_WithStack(param_1, param_2, param_3, param_4, param_5);
        _1083 = param_5.mask;
        _1084 = param_5.obj_index;
        _1085 = param_5.prim_index;
        _1086 = param_5.t;
        _1087 = param_5.u;
        _1088 = param_5.v;
        if (param_5.prim_index < 0)
        {
            _1085 = (-1) - int(_1016.Load(((-1) - _1085) * 4 + 0));
        }
        else
        {
            _1085 = int(_1016.Load(_1085 * 4 + 0));
        }
        _1038.Store(_919 * 24 + 0, uint(_1083));
        _1038.Store(_919 * 24 + 4, uint(_1084));
        _1038.Store(_919 * 24 + 8, uint(_1085));
        _1038.Store(_919 * 24 + 12, asuint(_1086));
        _1038.Store(_919 * 24 + 16, asuint(_1087));
        _1038.Store(_919 * 24 + 20, asuint(_1088));
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
