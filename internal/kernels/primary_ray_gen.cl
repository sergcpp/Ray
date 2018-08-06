R"(

int hash(int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

float construct_float(uint m) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = as_float(m);                // Range [1:2]
    return f - 1.0f;                       // Range [0:1]
}

float3 get_cam_dir(const float x, const float y, const camera_t *cam, int w, int h) {
    float k = native_tan(0.5f * cam->origin.w * PI / 180.0f);    
    float3 d = (float3)(2 * k * x / w - k, 2 * k * -y / h + k, 1);
    d = d.x * cam->side.xyz + d.y * cam->up.xyz + d.z * cam->fwd.xyz;
    d = normalize(d);
    return d;
}

__kernel
void GeneratePrimaryRays(const int iteration, camera_t cam, int w, int h, __global const float *halton, __global ray_packet_t *out_rays) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    const int index = j * w + i;
    const int hi = (iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT;

    float _unused;
    const float x = (float)i + fract(halton[hi + 0] + construct_float(hash(index)), &_unused);
    const float y = (float)j + fract(halton[hi + 1] + construct_float(hash(hash(index))), &_unused);

    float3 d = get_cam_dir(x, y, &cam, w, h);

    __global ray_packet_t *r = &out_rays[index];

    r->o = (float4)(cam.origin.xyz, (float)i);
    r->d = (float4)(d, (float)j);
    r->c = (float4)(1.0f, 1.0f, 1.0f, 1.0f);

    r->do_dx = r->do_dy = (float3)(0, 0, 0);

    float3 _dx = get_cam_dir(x + 1, y, &cam, w, h),
           _dy = get_cam_dir(x, y + 1, &cam, w, h);

    r->dd_dx = _dx - d;
    r->dd_dy = _dy - d;
}

)"