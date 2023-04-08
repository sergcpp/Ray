struct Params
{
    uint4 rect;
    float4 cam_origin;
    float4 cam_fwd;
    float4 cam_side;
    float4 cam_up;
    uint2 img_size;
    int hi;
    float spread_angle;
    float cam_fstop;
    float cam_focal_length;
    float cam_lens_rotation;
    float cam_lens_ratio;
    int cam_lens_blades;
    float cam_clip_start;
    int cam_filter;
    float shift_x;
    float shift_y;
    float _pad0;
    float _pad1;
    float _pad2;
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

ByteAddressBuffer _267 : register(t1, space0);
RWByteAddressBuffer _568 : register(u0, space0);
cbuffer UniformParams
{
    Params _76_g_params : packoffset(c0);
};


static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

int hash(int x)
{
    uint _34 = uint(x);
    uint _41 = ((_34 >> uint(16)) ^ _34) * 73244475u;
    uint _46 = ((_41 >> uint(16)) ^ _41) * 73244475u;
    return int((_46 >> uint(16)) ^ _46);
}

float construct_float(inout uint m)
{
    m &= 8388607u;
    m |= 1065353216u;
    return asfloat(m) - 1.0f;
}

float ngon_rad(float theta, float n)
{
    return cos(3.1415927410125732421875f / n) / cos(mad((-6.283185482025146484375f) / n, floor(mad(n, theta, 3.1415927410125732421875f) * 0.15915493667125701904296875f), theta));
}

float3 get_pix_dir(float x, float y, float3 _origin, float prop)
{
    float _86 = _76_g_params.cam_origin.w * _76_g_params.cam_side.w;
    float _90 = 2.0f * _86;
    return normalize((((_76_g_params.cam_origin.xyz + (_76_g_params.cam_side.xyz * (prop * mad(_90, (x / float(_76_g_params.img_size.x)) + (_76_g_params.shift_x / prop), -_86)))) + (_76_g_params.cam_up.xyz * mad(_90, ((-y) / float(_76_g_params.img_size.y)) + _76_g_params.shift_y, _86))) + (_76_g_params.cam_fwd.xyz * _76_g_params.cam_side.w)) - _origin);
}

void comp_main()
{
    do
    {
        bool _186 = gl_GlobalInvocationID.x >= _76_g_params.rect.z;
        bool _195;
        if (!_186)
        {
            _195 = gl_GlobalInvocationID.y >= _76_g_params.rect.w;
        }
        else
        {
            _195 = _186;
        }
        if (_195)
        {
            break;
        }
        int _206 = int(_76_g_params.rect.x + gl_GlobalInvocationID.x);
        int _213 = int(_76_g_params.rect.y + gl_GlobalInvocationID.y);
        int _230 = int(gl_GlobalInvocationID.y * _76_g_params.rect.z) + _206;
        int _235 = (_206 << 16) | _213;
        int _236 = hash(_235);
        float _x = float(_206);
        float _y = float(_213);
        uint param = uint(_236);
        float _249 = construct_float(param);
        uint param_1 = uint(hash(_236));
        float _254 = construct_float(param_1);
        if (_76_g_params.cam_filter == 1)
        {
            float _278 = frac(asfloat(_267.Load(_76_g_params.hi * 4 + 0)) + _249);
            float rx = _278;
            [flatten]
            if (_278 < 0.5f)
            {
                rx = sqrt(2.0f * rx) - 1.0f;
            }
            else
            {
                rx = 1.0f - sqrt(mad(-2.0f, rx, 2.0f));
            }
            float _303 = frac(asfloat(_267.Load((_76_g_params.hi + 1) * 4 + 0)) + _254);
            float ry = _303;
            [flatten]
            if (_303 < 0.5f)
            {
                ry = sqrt(2.0f * ry) - 1.0f;
            }
            else
            {
                ry = 1.0f - sqrt(mad(-2.0f, ry, 2.0f));
            }
            _x += (0.5f + rx);
            _y += (0.5f + ry);
        }
        else
        {
            _x += frac(asfloat(_267.Load(_76_g_params.hi * 4 + 0)) + _249);
            _y += frac(asfloat(_267.Load((_76_g_params.hi + 1) * 4 + 0)) + _254);
        }
        float2 offset = 0.0f.xx;
        if (_76_g_params.cam_fstop > 0.0f)
        {
            float2 _383 = (float2(frac(asfloat(_267.Load((_76_g_params.hi + 2) * 4 + 0)) + _249), frac(asfloat(_267.Load((_76_g_params.hi + 3) * 4 + 0)) + _254)) * 2.0f) - 1.0f.xx;
            offset = _383;
            bool _386 = _383.x != 0.0f;
            bool _392;
            if (_386)
            {
                _392 = offset.y != 0.0f;
            }
            else
            {
                _392 = _386;
            }
            if (_392)
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
                if (_76_g_params.cam_lens_blades > 0)
                {
                    r *= ngon_rad(theta, float(_76_g_params.cam_lens_blades));
                }
                float _442 = theta;
                float _443 = _442 + _76_g_params.cam_lens_rotation;
                theta = _443;
                float _445 = 0.5f * r;
                offset = float2((_445 * cos(_443)) / _76_g_params.cam_lens_ratio, _445 * sin(_443));
            }
            offset *= ((0.5f * (_76_g_params.cam_focal_length / _76_g_params.cam_fstop)) * _76_g_params.cam_up.w);
        }
        float3 _491 = (_76_g_params.cam_origin.xyz + (_76_g_params.cam_side.xyz * offset.x)) + (_76_g_params.cam_up.xyz * offset.y);
        float3 _origin = _491;
        float param_2 = _x;
        float param_3 = _y;
        float3 param_4 = _491;
        float param_5 = float(_76_g_params.img_size.x) / float(_76_g_params.img_size.y);
        float3 _501 = get_pix_dir(param_2, param_3, param_4, param_5);
        float3 _513 = _origin;
        float3 _514 = _513 + (_501 * (_76_g_params.cam_clip_start / dot(_501, _76_g_params.cam_fwd.xyz)));
        _origin = _514;
        _568.Store(_230 * 72 + 0, asuint(_514.x));
        _568.Store(_230 * 72 + 4, asuint(_514.y));
        _568.Store(_230 * 72 + 8, asuint(_514.z));
        _568.Store(_230 * 72 + 12, asuint(_501.x));
        _568.Store(_230 * 72 + 16, asuint(_501.y));
        _568.Store(_230 * 72 + 20, asuint(_501.z));
        _568.Store(_230 * 72 + 24, asuint(1000000.0f));
        _568.Store(_230 * 72 + 28, asuint(1.0f));
        _568.Store(_230 * 72 + 32, asuint(1.0f));
        _568.Store(_230 * 72 + 36, asuint(1.0f));
        _568.Store(_230 * 72 + 40, asuint(-1.0f));
        _568.Store(_230 * 72 + 44, asuint(-1.0f));
        _568.Store(_230 * 72 + 48, asuint(-1.0f));
        _568.Store(_230 * 72 + 52, asuint(-1.0f));
        _568.Store(_230 * 72 + 56, asuint(0.0f));
        _568.Store(_230 * 72 + 60, asuint(_76_g_params.spread_angle));
        _568.Store(_230 * 72 + 64, uint(_235));
        _568.Store(_230 * 72 + 68, uint(0));
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
