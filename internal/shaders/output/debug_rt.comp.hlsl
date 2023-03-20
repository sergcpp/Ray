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

struct Params
{
    uint2 img_size;
    uint node_index;
    float random_val;
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

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

ByteAddressBuffer _298 : register(t1, space0);
ByteAddressBuffer _508 : register(t3, space0);
ByteAddressBuffer _749 : register(t5, space0);
ByteAddressBuffer _753 : register(t6, space0);
ByteAddressBuffer _772 : register(t4, space0);
ByteAddressBuffer _788 : register(t7, space0);
ByteAddressBuffer _915 : register(t8, space0);
ByteAddressBuffer _976 : register(t2, space0);
cbuffer UniformParams
{
    Params _872_g_params : packoffset(c0);
};

uniform ??? g_tlas;
RWTexture2D<float4> g_out_img : register(u0, space0);

static uint3 gl_GlobalInvocationID;
static uint gl_LocalInvocationIndex;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
    uint gl_LocalInvocationIndex : SV_GroupIndex;
};

groupshared uint g_stack[64][48];
static ??? rq;

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
        float3 _1188 = inv_v;
        _1188.x = 3.4028234663852885981170418348452e+38f;
        inv_v = _1188;
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
            float3 _1190 = inv_v;
            _1190.x = -3.4028234663852885981170418348452e+38f;
            inv_v = _1190;
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
        float3 _1192 = inv_v;
        _1192.y = 3.4028234663852885981170418348452e+38f;
        inv_v = _1192;
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
            float3 _1194 = inv_v;
            _1194.y = -3.4028234663852885981170418348452e+38f;
            inv_v = _1194;
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
        float3 _1196 = inv_v;
        _1196.z = 3.4028234663852885981170418348452e+38f;
        inv_v = _1196;
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
            float3 _1198 = inv_v;
            _1198.z = -3.4028234663852885981170418348452e+38f;
            inv_v = _1198;
        }
    }
    return inv_v;
}

bool _bbox_test_fma(float3 inv_d, float3 neg_inv_d_o, float t, float3 bbox_min, float3 bbox_max)
{
    float _386 = mad(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float _394 = mad(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float _409 = mad(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    float _416 = mad(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    float _433 = mad(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    float _440 = mad(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    float _445 = max(max(min(_386, _394), min(_409, _416)), min(_433, _440));
    float _453 = min(min(max(_386, _394), max(_409, _416)), max(_433, _440)) * 1.0000002384185791015625f;
    return ((_445 <= _453) && (_445 <= t)) && (_453 > 0.0f);
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
    int _1121 = 0;
    int _1122 = obj_index;
    float _1124 = out_inter.t;
    float3 param;
    float3 param_1;
    tri_accel_t param_2;
    uint param_3;
    hit_data_t param_4;
    int _1123;
    float _1125;
    float _1126;
    for (int i = tri_start; i < tri_end; )
    {
        param = ro;
        param_1 = rd;
        tri_accel_t _309;
        _309.n_plane = asfloat(_298.Load4(i * 48 + 0));
        _309.u_plane = asfloat(_298.Load4(i * 48 + 16));
        _309.v_plane = asfloat(_298.Load4(i * 48 + 32));
        tri_accel_t _310;
        _310.n_plane = _309.n_plane;
        _310.u_plane = _309.u_plane;
        _310.v_plane = _309.v_plane;
        param_2 = _310;
        param_3 = uint(i);
        hit_data_t _1133 = { _1121, _1122, _1123, _1124, _1125, _1126 };
        param_4 = _1133;
        IntersectTri(param, param_1, param_2, param_3, param_4);
        _1121 = param_4.mask;
        _1122 = param_4.obj_index;
        _1123 = param_4.prim_index;
        _1124 = param_4.t;
        _1125 = param_4.u;
        _1126 = param_4.v;
        i++;
        continue;
    }
    out_inter.mask |= _1121;
    int _327;
    if (_1121 != 0)
    {
        _327 = _1122;
    }
    else
    {
        _327 = out_inter.obj_index;
    }
    out_inter.obj_index = _327;
    int _340;
    if (_1121 != 0)
    {
        _340 = _1123;
    }
    else
    {
        _340 = out_inter.prim_index;
    }
    out_inter.prim_index = _340;
    out_inter.t = _1124;
    float _356;
    if (_1121 != 0)
    {
        _356 = _1125;
    }
    else
    {
        _356 = out_inter.u;
    }
    out_inter.u = _356;
    float _369;
    if (_1121 != 0)
    {
        _369 = _1126;
    }
    else
    {
        _369 = out_inter.v;
    }
    out_inter.v = _369;
}

void Traverse_MicroTree_WithStack(float3 ro, float3 rd, float3 inv_d, int obj_index, uint node_index, inout uint stack_size, inout hit_data_t inter)
{
    float3 _470 = (-inv_d) * ro;
    uint _472 = stack_size;
    uint _482 = stack_size;
    stack_size = _482 + uint(1);
    g_stack[gl_LocalInvocationIndex][_482] = node_index;
    uint _553;
    uint _577;
    while (stack_size != _472)
    {
        uint _497 = stack_size;
        uint _498 = _497 - uint(1);
        stack_size = _498;
        bvh_node_t _512;
        _512.bbox_min = asfloat(_508.Load4(g_stack[gl_LocalInvocationIndex][_498] * 32 + 0));
        _512.bbox_max = asfloat(_508.Load4(g_stack[gl_LocalInvocationIndex][_498] * 32 + 16));
        bvh_node_t _513;
        _513.bbox_min = _512.bbox_min;
        _513.bbox_max = _512.bbox_max;
        float3 param = inv_d;
        float3 param_1 = _470;
        float param_2 = inter.t;
        float3 param_3 = _513.bbox_min.xyz;
        float3 param_4 = _513.bbox_max.xyz;
        if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
        {
            continue;
        }
        uint _536 = asuint(_513.bbox_min.w);
        if ((_536 & 2147483648u) == 0u)
        {
            uint _543 = stack_size;
            stack_size = _543 + uint(1);
            uint _547 = asuint(_513.bbox_max.w);
            uint _549 = _547 >> uint(30);
            if (rd[_549] < 0.0f)
            {
                _553 = _536;
            }
            else
            {
                _553 = _547 & 1073741823u;
            }
            g_stack[gl_LocalInvocationIndex][_543] = _553;
            uint _568 = stack_size;
            stack_size = _568 + uint(1);
            if (rd[_549] < 0.0f)
            {
                _577 = _547 & 1073741823u;
            }
            else
            {
                _577 = _536;
            }
            g_stack[gl_LocalInvocationIndex][_568] = _577;
        }
        else
        {
            int _597 = int(_536 & 2147483647u);
            float3 param_5 = ro;
            float3 param_6 = rd;
            int param_7 = _597;
            int param_8 = _597 + asint(_513.bbox_max.w);
            int param_9 = obj_index;
            hit_data_t param_10 = inter;
            IntersectTris_ClosestHit(param_5, param_6, param_7, param_8, param_9, param_10);
            inter = param_10;
        }
    }
}

void Traverse_MacroTree_WithStack(float3 orig_ro, float3 orig_rd, float3 orig_inv_rd, uint node_index, inout hit_data_t inter)
{
    float3 _622 = (-orig_inv_rd) * orig_ro;
    uint stack_size = 1u;
    g_stack[gl_LocalInvocationIndex][0u] = node_index;
    uint _684;
    uint _707;
    while (stack_size != 0u)
    {
        uint _638 = stack_size;
        uint _639 = _638 - uint(1);
        stack_size = _639;
        bvh_node_t _645;
        _645.bbox_min = asfloat(_508.Load4(g_stack[gl_LocalInvocationIndex][_639] * 32 + 0));
        _645.bbox_max = asfloat(_508.Load4(g_stack[gl_LocalInvocationIndex][_639] * 32 + 16));
        bvh_node_t _646;
        _646.bbox_min = _645.bbox_min;
        _646.bbox_max = _645.bbox_max;
        float3 param = orig_inv_rd;
        float3 param_1 = _622;
        float param_2 = inter.t;
        float3 param_3 = _646.bbox_min.xyz;
        float3 param_4 = _646.bbox_max.xyz;
        if (!_bbox_test_fma(param, param_1, param_2, param_3, param_4))
        {
            continue;
        }
        uint _669 = asuint(_646.bbox_min.w);
        if ((_669 & 2147483648u) == 0u)
        {
            uint _675 = stack_size;
            stack_size = _675 + uint(1);
            uint _679 = asuint(_646.bbox_max.w);
            uint _680 = _679 >> uint(30);
            if (orig_rd[_680] < 0.0f)
            {
                _684 = _669;
            }
            else
            {
                _684 = _679 & 1073741823u;
            }
            g_stack[gl_LocalInvocationIndex][_675] = _684;
            uint _698 = stack_size;
            stack_size = _698 + uint(1);
            if (orig_rd[_680] < 0.0f)
            {
                _707 = _679 & 1073741823u;
            }
            else
            {
                _707 = _669;
            }
            g_stack[gl_LocalInvocationIndex][_698] = _707;
        }
        else
        {
            uint _725 = _669 & 2147483647u;
            uint _729 = asuint(_646.bbox_max.w);
            for (uint i = _725; i < (_725 + _729); i++)
            {
                mesh_instance_t _760;
                _760.bbox_min = asfloat(_749.Load4(_753.Load(i * 4 + 0) * 32 + 0));
                _760.bbox_max = asfloat(_749.Load4(_753.Load(i * 4 + 0) * 32 + 16));
                mesh_instance_t _761;
                _761.bbox_min = _760.bbox_min;
                _761.bbox_max = _760.bbox_max;
                mesh_t _778;
                [unroll]
                for (int _2ident = 0; _2ident < 3; _2ident++)
                {
                    _778.bbox_min[_2ident] = asfloat(_772.Load(_2ident * 4 + asuint(_761.bbox_max.w) * 48 + 0));
                }
                [unroll]
                for (int _3ident = 0; _3ident < 3; _3ident++)
                {
                    _778.bbox_max[_3ident] = asfloat(_772.Load(_3ident * 4 + asuint(_761.bbox_max.w) * 48 + 12));
                }
                _778.node_index = _772.Load(asuint(_761.bbox_max.w) * 48 + 24);
                _778.node_count = _772.Load(asuint(_761.bbox_max.w) * 48 + 28);
                _778.tris_index = _772.Load(asuint(_761.bbox_max.w) * 48 + 32);
                _778.tris_count = _772.Load(asuint(_761.bbox_max.w) * 48 + 36);
                _778.vert_index = _772.Load(asuint(_761.bbox_max.w) * 48 + 40);
                _778.vert_count = _772.Load(asuint(_761.bbox_max.w) * 48 + 44);
                mesh_t _779;
                _779.bbox_min[0] = _778.bbox_min[0];
                _779.bbox_min[1] = _778.bbox_min[1];
                _779.bbox_min[2] = _778.bbox_min[2];
                _779.bbox_max[0] = _778.bbox_max[0];
                _779.bbox_max[1] = _778.bbox_max[1];
                _779.bbox_max[2] = _778.bbox_max[2];
                _779.node_index = _778.node_index;
                _779.node_count = _778.node_count;
                _779.tris_index = _778.tris_index;
                _779.tris_count = _778.tris_count;
                _779.vert_index = _778.vert_index;
                _779.vert_count = _778.vert_count;
                transform_t _794;
                _794.xform = asfloat(uint4x4(_788.Load4(asuint(_761.bbox_min.w) * 128 + 0), _788.Load4(asuint(_761.bbox_min.w) * 128 + 16), _788.Load4(asuint(_761.bbox_min.w) * 128 + 32), _788.Load4(asuint(_761.bbox_min.w) * 128 + 48)));
                _794.inv_xform = asfloat(uint4x4(_788.Load4(asuint(_761.bbox_min.w) * 128 + 64), _788.Load4(asuint(_761.bbox_min.w) * 128 + 80), _788.Load4(asuint(_761.bbox_min.w) * 128 + 96), _788.Load4(asuint(_761.bbox_min.w) * 128 + 112)));
                transform_t _795;
                _795.xform = _794.xform;
                _795.inv_xform = _794.inv_xform;
                float3 param_5 = orig_inv_rd;
                float3 param_6 = _622;
                float param_7 = inter.t;
                float3 param_8 = _761.bbox_min.xyz;
                float3 param_9 = _761.bbox_max.xyz;
                if (!_bbox_test_fma(param_5, param_6, param_7, param_8, param_9))
                {
                    continue;
                }
                float3 _836 = mul(float4(orig_rd, 0.0f), _795.inv_xform).xyz;
                float3 param_10 = _836;
                float3 param_11 = mul(float4(orig_ro, 1.0f), _795.inv_xform).xyz;
                float3 param_12 = _836;
                float3 param_13 = safe_invert(param_10);
                int param_14 = int(_753.Load(i * 4 + 0));
                uint param_15 = _779.node_index;
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
        bool _876 = gl_GlobalInvocationID.x >= _872_g_params.img_size.x;
        bool _885;
        if (!_876)
        {
            _885 = gl_GlobalInvocationID.y >= _872_g_params.img_size.y;
        }
        else
        {
            _885 = _876;
        }
        if (_885)
        {
            break;
        }
        int _892 = int(gl_GlobalInvocationID.x);
        int _896 = int(gl_GlobalInvocationID.y);
        int _904 = (_896 * int(_872_g_params.img_size.x)) + _892;
        float _919 = asfloat(_915.Load(_904 * 72 + 0));
        float _922 = asfloat(_915.Load(_904 * 72 + 4));
        float _925 = asfloat(_915.Load(_904 * 72 + 8));
        float3 _926 = float3(_919, _922, _925);
        float _930 = asfloat(_915.Load(_904 * 72 + 12));
        float _933 = asfloat(_915.Load(_904 * 72 + 16));
        float _936 = asfloat(_915.Load(_904 * 72 + 20));
        float3 _937 = float3(_930, _933, _936);
        float3 param = _937;
        int _1102 = 0;
        int _1104 = 0;
        int _1103 = 0;
        float _1105 = 3402823346297367662189621542912.0f;
        float _1107 = 0.0f;
        float _1106 = 0.0f;
        [branch]
        if (_892 < 256)
        {
            float3 param_1 = _926;
            float3 param_2 = _937;
            float3 param_3 = safe_invert(param);
            uint param_4 = _872_g_params.node_index;
            hit_data_t _1114 = { _1102, _1103, _1104, _1105, _1106, _1107 };
            hit_data_t param_5 = _1114;
            Traverse_MacroTree_WithStack(param_1, param_2, param_3, param_4, param_5);
            _1102 = param_5.mask;
            _1103 = param_5.obj_index;
            _1104 = param_5.prim_index;
            _1105 = param_5.t;
            _1106 = param_5.u;
            _1107 = param_5.v;
            if (param_5.prim_index < 0)
            {
                _1104 = (-1) - int(_976.Load(((-1) - _1104) * 4 + 0));
            }
            else
            {
                _1104 = int(_976.Load(_1104 * 4 + 0));
            }
        }
        else
        {
            rayQueryInitializeEXT(rq, g_tlas, 0u, 255u, _926, 0.0f, _937, 100.0f);
            for (;;)
            {
                bool _1011 = rayQueryProceedEXT(rq);
                if (_1011)
                {
                    uint _1013 = rayQueryGetIntersectionTypeEXT(rq, bool(0));
                    if (_1013 == 0u)
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
            uint _1018 = rayQueryGetIntersectionTypeEXT(rq, bool(1));
            if (_1018 != 0u)
            {
                int _1023 = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, bool(1));
                _1102 = -1;
                int _1025 = rayQueryGetIntersectionInstanceIdEXT(rq, bool(1));
                _1103 = _1025;
                int _1028 = rayQueryGetIntersectionPrimitiveIndexEXT(rq, bool(1));
                _1104 = _1023 + _1028;
                bool _1031 = rayQueryGetIntersectionFrontFaceEXT(rq, bool(1));
                [flatten]
                if (!_1031)
                {
                    _1104 = (-1) - _1104;
                }
                float2 _1043 = rayQueryGetIntersectionBarycentricsEXT(rq, bool(1));
                _1106 = _1043.x;
                _1107 = _1043.y;
                float _1050 = rayQueryGetIntersectionTEXT(rq, bool(1));
                _1105 = _1050;
            }
        }
        g_out_img[int2(_892, _896)] = float4(_1105, _1105, _1105, 1.0f);
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    gl_LocalInvocationIndex = stage_input.gl_LocalInvocationIndex;
    comp_main();
}
