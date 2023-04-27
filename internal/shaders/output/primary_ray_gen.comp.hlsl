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
    int cam_filter_and_lens_blades;
    float shift_x;
    float shift_y;
    int iteration;
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

ByteAddressBuffer _270 : register(t1, space0);
RWByteAddressBuffer _571 : register(u0, space0);
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
        bool _187 = gl_GlobalInvocationID.x >= _76_g_params.rect.z;
        bool _196;
        if (!_187)
        {
            _196 = gl_GlobalInvocationID.y >= _76_g_params.rect.w;
        }
        else
        {
            _196 = _187;
        }
        if (_196)
        {
            break;
        }
        int _207 = int(_76_g_params.rect.x + gl_GlobalInvocationID.x);
        int _214 = int(_76_g_params.rect.y + gl_GlobalInvocationID.y);
        int _231 = int(gl_GlobalInvocationID.y * _76_g_params.rect.z) + _207;
        int _236 = (_207 << 16) | _214;
        int _237 = hash(_236);
        float _x = float(_207);
        float _y = float(_214);
        uint param = uint(_237);
        float _250 = construct_float(param);
        uint param_1 = uint(hash(_237));
        float _255 = construct_float(param_1);
        if ((_76_g_params.cam_filter_and_lens_blades >> 8) == 1)
        {
            float _281 = frac(asfloat(_270.Load(_76_g_params.hi * 4 + 0)) + _250);
            float rx = _281;
            [flatten]
            if (_281 < 0.5f)
            {
                rx = sqrt(2.0f * rx) - 1.0f;
            }
            else
            {
                rx = 1.0f - sqrt(mad(-2.0f, rx, 2.0f));
            }
            float _306 = frac(asfloat(_270.Load((_76_g_params.hi + 1) * 4 + 0)) + _255);
            float ry = _306;
            [flatten]
            if (_306 < 0.5f)
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
            _x += frac(asfloat(_270.Load(_76_g_params.hi * 4 + 0)) + _250);
            _y += frac(asfloat(_270.Load((_76_g_params.hi + 1) * 4 + 0)) + _255);
        }
        float2 offset = 0.0f.xx;
        if (_76_g_params.cam_fstop > 0.0f)
        {
            float2 _385 = (float2(frac(asfloat(_270.Load((_76_g_params.hi + 2) * 4 + 0)) + _250), frac(asfloat(_270.Load((_76_g_params.hi + 3) * 4 + 0)) + _255)) * 2.0f) - 1.0f.xx;
            offset = _385;
            bool _388 = _385.x != 0.0f;
            bool _394;
            if (_388)
            {
                _394 = offset.y != 0.0f;
            }
            else
            {
                _394 = _388;
            }
            if (_394)
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
                int _431 = _76_g_params.cam_filter_and_lens_blades & 255;
                if (_431 > 0)
                {
                    r *= ngon_rad(theta, float(_431));
                }
                float _446 = theta;
                float _447 = _446 + _76_g_params.cam_lens_rotation;
                theta = _447;
                float _449 = 0.5f * r;
                offset = float2((_449 * cos(_447)) / _76_g_params.cam_lens_ratio, _449 * sin(_447));
            }
            offset *= ((0.5f * (_76_g_params.cam_focal_length / _76_g_params.cam_fstop)) * _76_g_params.cam_up.w);
        }
        float3 _495 = (_76_g_params.cam_origin.xyz + (_76_g_params.cam_side.xyz * offset.x)) + (_76_g_params.cam_up.xyz * offset.y);
        float3 _origin = _495;
        float param_2 = _x;
        float param_3 = _y;
        float3 param_4 = _495;
        float param_5 = float(_76_g_params.img_size.x) / float(_76_g_params.img_size.y);
        float3 _505 = get_pix_dir(param_2, param_3, param_4, param_5);
        float3 _516 = _origin;
        float3 _517 = _516 + (_505 * (_76_g_params.cam_fwd.w / dot(_505, _76_g_params.cam_fwd.xyz)));
        _origin = _517;
        _571.Store(_231 * 72 + 0, asuint(_517.x));
        _571.Store(_231 * 72 + 4, asuint(_517.y));
        _571.Store(_231 * 72 + 8, asuint(_517.z));
        _571.Store(_231 * 72 + 12, asuint(_505.x));
        _571.Store(_231 * 72 + 16, asuint(_505.y));
        _571.Store(_231 * 72 + 20, asuint(_505.z));
        _571.Store(_231 * 72 + 24, asuint(1000000.0f));
        _571.Store(_231 * 72 + 28, asuint(1.0f));
        _571.Store(_231 * 72 + 32, asuint(1.0f));
        _571.Store(_231 * 72 + 36, asuint(1.0f));
        _571.Store(_231 * 72 + 40, asuint(-1.0f));
        _571.Store(_231 * 72 + 44, asuint(-1.0f));
        _571.Store(_231 * 72 + 48, asuint(-1.0f));
        _571.Store(_231 * 72 + 52, asuint(-1.0f));
        _571.Store(_231 * 72 + 56, asuint(0.0f));
        _571.Store(_231 * 72 + 60, asuint(_76_g_params.spread_angle));
        _571.Store(_231 * 72 + 64, uint(_236));
        _571.Store(_231 * 72 + 68, uint(0));
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
