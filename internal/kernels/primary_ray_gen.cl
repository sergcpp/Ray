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

float3 get_cam_dir(const float x, const float y, float3 origin, const camera_t *cam, int w, int h, float prop) {
    float k = native_tan(0.5f * cam->origin.w * PI / 180.0f) * cam->side.w;
    float3 p = (float3)(2 * k * x / w - k, 2 * k * -y / h + k, cam->side.w);
    p = cam->origin.xyz + p.x * cam->side.xyz + prop * p.y * cam->up.xyz + p.z * cam->fwd.xyz;
    return normalize(p - origin);
}

__kernel
void GeneratePrimaryRays(const int iteration, camera_t cam, int w, int h, __global const float *halton, __global ray_packet_t *out_rays) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    const int index = j * w + i;
    const int hi = (iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT;

    float k = ((float)h) / w;

    float x = (float)i;
    float y = (float)j;

    float _unused;
    int hash_val = hash(index);
    if (cam.flags & CAM_USE_TENT_FILTER) {
        float rx = fract(halton[hi + 0] + construct_float(hash_val), &_unused);
        if (rx < 0.5f) {
            rx = native_sqrt(2 * rx) - 1.0f;
        } else {
            rx = 1.0f - native_sqrt(2.0f - 2 * rx);
        }

        float ry = fract(halton[hi + 1] + construct_float(hash(hash_val)), &_unused);
        if (ry < 0.5f) {
            ry = native_sqrt(2 * ry) - 1.0f;
        } else {
            ry = 1.0f - native_sqrt(2.0f - 2 * ry);
        }

        x += 0.5f + rx;
        y += 0.5f + ry;
    } else {
        x += fract(halton[hi + 0] + construct_float(hash_val), &_unused);
        y += fract(halton[hi + 1] + construct_float(hash(hash_val)), &_unused);
    }

    __global ray_packet_t *r = &out_rays[index];

    float ff1 = cam.up.w * (-0.5f + fract(halton[hi + 2 + 0] + construct_float(hash_val), &_unused));
    float ff2 = cam.up.w * (-0.5f + fract(halton[hi + 2 + 1] + construct_float(hash(hash_val)), &_unused));

    float3 origin = cam.origin.xyz + cam.side.xyz * ff1 + cam.up.xyz * ff2;

    float3 d = get_cam_dir(x, y, origin, &cam, w, h, k);

    r->o = (float4)(origin, (float)i);
    r->d = (float4)(d, (float)j);
   
    r->c = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
    r->do_dx = r->do_dy = (float3)(0, 0, 0);

    float3 _dx = get_cam_dir(x + 1, y, origin, &cam, w, h, k),
           _dy = get_cam_dir(x, y + 1, origin, &cam, w, h, k);

    r->dd_dx = _dx - d;
    r->dd_dy = _dy - d;
}

)"