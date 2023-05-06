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

RWByteAddressBuffer _238 : register(u1, space0);
ByteAddressBuffer _287 : register(t2, space0);
RWByteAddressBuffer _588 : register(u0, space0);
cbuffer UniformParams
{
    Params _76_g_params : packoffset(c0);
};

RWTexture2D<uint> g_required_samples_img : register(u3, space0);

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
        if (g_required_samples_img[int2(_207, _214)].xxxx.x < uint(_76_g_params.iteration))
        {
            break;
        }
        uint _241;
        _238.InterlockedAdd(0, 1u, _241);
        int _254 = (_207 << 16) | _214;
        int _255 = hash(_254);
        float _x = float(_207);
        float _y = float(_214);
        uint param = uint(_255);
        float _268 = construct_float(param);
        uint param_1 = uint(hash(_255));
        float _273 = construct_float(param_1);
        if ((_76_g_params.cam_filter_and_lens_blades >> 8) == 1)
        {
            float _298 = frac(asfloat(_287.Load(_76_g_params.hi * 4 + 0)) + _268);
            float rx = _298;
            [flatten]
            if (_298 < 0.5f)
            {
                rx = sqrt(2.0f * rx) - 1.0f;
            }
            else
            {
                rx = 1.0f - sqrt(mad(-2.0f, rx, 2.0f));
            }
            float _323 = frac(asfloat(_287.Load((_76_g_params.hi + 1) * 4 + 0)) + _273);
            float ry = _323;
            [flatten]
            if (_323 < 0.5f)
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
            _x += frac(asfloat(_287.Load(_76_g_params.hi * 4 + 0)) + _268);
            _y += frac(asfloat(_287.Load((_76_g_params.hi + 1) * 4 + 0)) + _273);
        }
        float2 offset = 0.0f.xx;
        if (_76_g_params.cam_fstop > 0.0f)
        {
            float2 _402 = (float2(frac(asfloat(_287.Load((_76_g_params.hi + 2) * 4 + 0)) + _268), frac(asfloat(_287.Load((_76_g_params.hi + 3) * 4 + 0)) + _273)) * 2.0f) - 1.0f.xx;
            offset = _402;
            bool _405 = _402.x != 0.0f;
            bool _411;
            if (_405)
            {
                _411 = offset.y != 0.0f;
            }
            else
            {
                _411 = _405;
            }
            if (_411)
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
                int _448 = _76_g_params.cam_filter_and_lens_blades & 255;
                if (_448 > 0)
                {
                    r *= ngon_rad(theta, float(_448));
                }
                float _463 = theta;
                float _464 = _463 + _76_g_params.cam_lens_rotation;
                theta = _464;
                float _466 = 0.5f * r;
                offset = float2((_466 * cos(_464)) / _76_g_params.cam_lens_ratio, _466 * sin(_464));
            }
            offset *= ((0.5f * (_76_g_params.cam_focal_length / _76_g_params.cam_fstop)) * _76_g_params.cam_up.w);
        }
        float3 _512 = (_76_g_params.cam_origin.xyz + (_76_g_params.cam_side.xyz * offset.x)) + (_76_g_params.cam_up.xyz * offset.y);
        float3 _origin = _512;
        float param_2 = _x;
        float param_3 = _y;
        float3 param_4 = _512;
        float param_5 = float(_76_g_params.img_size.x) / float(_76_g_params.img_size.y);
        float3 _522 = get_pix_dir(param_2, param_3, param_4, param_5);
        float3 _533 = _origin;
        float3 _534 = _533 + (_522 * (_76_g_params.cam_fwd.w / dot(_522, _76_g_params.cam_fwd.xyz)));
        _origin = _534;
        _588.Store(_241 * 72 + 0, asuint(_534.x));
        _588.Store(_241 * 72 + 4, asuint(_534.y));
        _588.Store(_241 * 72 + 8, asuint(_534.z));
        _588.Store(_241 * 72 + 12, asuint(_522.x));
        _588.Store(_241 * 72 + 16, asuint(_522.y));
        _588.Store(_241 * 72 + 20, asuint(_522.z));
        _588.Store(_241 * 72 + 24, asuint(1000000.0f));
        _588.Store(_241 * 72 + 28, asuint(1.0f));
        _588.Store(_241 * 72 + 32, asuint(1.0f));
        _588.Store(_241 * 72 + 36, asuint(1.0f));
        _588.Store(_241 * 72 + 40, asuint(-1.0f));
        _588.Store(_241 * 72 + 44, asuint(-1.0f));
        _588.Store(_241 * 72 + 48, asuint(-1.0f));
        _588.Store(_241 * 72 + 52, asuint(-1.0f));
        _588.Store(_241 * 72 + 56, asuint(0.0f));
        _588.Store(_241 * 72 + 60, asuint(_76_g_params.spread_angle));
        _588.Store(_241 * 72 + 64, uint(_254));
        _588.Store(_241 * 72 + 68, uint(0));
        break;
    } while(false);
}

[numthreads(8, 8, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
