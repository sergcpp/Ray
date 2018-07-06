R"(

int hash(int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

float3 get_cam_dir(const float x, const float y, const camera_t *cam, int w, int h) {
    float3 d = (float3)(x / w - 0.5f, -y / h + 0.5f, 1);
    d = d.x * cam->side + d.y * cam->up + d.z * cam->fwd;
    d = normalize(d);
    return d;
}

__kernel
void GeneratePrimaryRays(const int iteration, camera_t cam, int w, int h, __global const float *halton, __global ray_packet_t *out_rays) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    const int index = j * w + i;
    const int hi = (hash(index) + iteration) & (HaltonSeqLen - 1);

    const float x = (float)i + halton[hi * 2];
    const float y = (float)j + halton[hi * 2 + 1];

    float3 d = get_cam_dir(x, y, &cam, w, h);

    __global ray_packet_t *r = &out_rays[index];

    r->o = (float4)(cam.origin, x);
    r->d = (float4)(d, y);
    r->c = (float4)(1.0f, 1.0f, 1.0f, 1.0f);

    r->do_dx = r->do_dy = (float3)(0, 0, 0);

    float3 _dx = get_cam_dir(x + 1, y, &cam, w, h),
           _dy = get_cam_dir(x, y + 1, &cam, w, h);

    r->dd_dx = _dx - d;
    r->dd_dy = _dy - d;
}

)"