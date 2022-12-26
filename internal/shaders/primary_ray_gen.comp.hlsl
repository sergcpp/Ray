struct Params
{
    uint2 img_size;
    int hi;
    float spread_angle;
    float4 cam_origin;
    float4 cam_fwd;
    float4 cam_side;
    float4 cam_up;
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
    float cone_width;
    float cone_spread;
    int xy;
    int ray_depth;
};

static const uint3 gl_WorkGroupSize = uint3(8u, 8u, 1u);

ByteAddressBuffer _257 : register(t1, space0);
RWByteAddressBuffer _549 : register(u0, space0);
cbuffer UniformParams
{
    Params _75_g_params : packoffset(c0);
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
    float _85 = _75_g_params.cam_origin.w * _75_g_params.cam_side.w;
    float _89 = 2.0f * _85;
    return normalize((((_75_g_params.cam_origin.xyz + (_75_g_params.cam_side.xyz * (prop * mad(_89, (x / float(_75_g_params.img_size.x)) + (_75_g_params.shift_x / prop), -_85)))) + (_75_g_params.cam_up.xyz * mad(_89, ((-y) / float(_75_g_params.img_size.y)) + _75_g_params.shift_y, _85))) + (_75_g_params.cam_fwd.xyz * _75_g_params.cam_side.w)) - _origin);
}

void comp_main()
{
    do
    {
        bool _185 = gl_GlobalInvocationID.x >= _75_g_params.img_size.x;
        bool _194;
        if (!_185)
        {
            _194 = gl_GlobalInvocationID.y >= _75_g_params.img_size.y;
        }
        else
        {
            _194 = _185;
        }
        if (_194)
        {
            break;
        }
        int _210 = int(gl_GlobalInvocationID.x);
        int _214 = int(gl_GlobalInvocationID.y);
        int _222 = (_214 * int(_75_g_params.img_size.x)) + _210;
        int _225 = hash(_222);
        float _x = float(_210);
        float _y = float(_214);
        uint param = uint(_225);
        float _238 = construct_float(param);
        uint param_1 = uint(hash(_225));
        float _243 = construct_float(param_1);
        if (_75_g_params.cam_filter == 1)
        {
            float _267 = frac(asfloat(_257.Load(_75_g_params.hi * 4 + 0)) + _238);
            float rx = _267;
            [flatten]
            if (_267 < 0.5f)
            {
                rx = sqrt(2.0f * rx) - 1.0f;
            }
            else
            {
                rx = 1.0f - sqrt(mad(-2.0f, rx, 2.0f));
            }
            float _292 = frac(asfloat(_257.Load((_75_g_params.hi + 1) * 4 + 0)) + _243);
            float ry = _292;
            [flatten]
            if (_292 < 0.5f)
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
            _x += frac(asfloat(_257.Load(_75_g_params.hi * 4 + 0)) + _238);
            _y += frac(asfloat(_257.Load((_75_g_params.hi + 1) * 4 + 0)) + _243);
        }
        float2 offset = 0.0f.xx;
        if (_75_g_params.cam_fstop > 0.0f)
        {
            float2 _373 = (float2(frac(asfloat(_257.Load((_75_g_params.hi + 2) * 4 + 0)) + _238), frac(asfloat(_257.Load((_75_g_params.hi + 3) * 4 + 0)) + _243)) * 2.0f) - 1.0f.xx;
            offset = _373;
            bool _376 = _373.x != 0.0f;
            bool _382;
            if (_376)
            {
                _382 = offset.y != 0.0f;
            }
            else
            {
                _382 = _376;
            }
            if (_382)
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
                if (_75_g_params.cam_lens_blades > 0)
                {
                    r *= ngon_rad(theta, float(_75_g_params.cam_lens_blades));
                }
                float _432 = theta;
                float _433 = _432 + _75_g_params.cam_lens_rotation;
                theta = _433;
                float _435 = 0.5f * r;
                offset = float2((_435 * cos(_433)) / _75_g_params.cam_lens_ratio, _435 * sin(_433));
            }
            offset *= ((0.5f * (_75_g_params.cam_focal_length / _75_g_params.cam_fstop)) * _75_g_params.cam_up.w);
        }
        float3 _481 = (_75_g_params.cam_origin.xyz + (_75_g_params.cam_side.xyz * offset.x)) + (_75_g_params.cam_up.xyz * offset.y);
        float3 _origin = _481;
        float param_2 = _x;
        float param_3 = _y;
        float3 param_4 = _481;
        float param_5 = float(_75_g_params.img_size.x) / float(_75_g_params.img_size.y);
        float3 _491 = get_pix_dir(param_2, param_3, param_4, param_5);
        float3 _503 = _origin;
        float3 _504 = _503 + (_491 * (_75_g_params.cam_clip_start / dot(_491, _75_g_params.cam_fwd.xyz)));
        _origin = _504;
        _549.Store(_222 * 56 + 0, asuint(_504.x));
        _549.Store(_222 * 56 + 4, asuint(_504.y));
        _549.Store(_222 * 56 + 8, asuint(_504.z));
        _549.Store(_222 * 56 + 12, asuint(_491.x));
        _549.Store(_222 * 56 + 16, asuint(_491.y));
        _549.Store(_222 * 56 + 20, asuint(_491.z));
        _549.Store(_222 * 56 + 24, asuint(1000000.0f));
        _549.Store(_222 * 56 + 28, asuint(1.0f));
        _549.Store(_222 * 56 + 32, asuint(1.0f));
        _549.Store(_222 * 56 + 36, asuint(1.0f));
        _549.Store(_222 * 56 + 40, asuint(0.0f));
        _549.Store(_222 * 56 + 44, asuint(_75_g_params.spread_angle));
        _549.Store(_222 * 56 + 48, uint((_210 << 16) | _214));
        _549.Store(_222 * 56 + 52, uint(0));
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
