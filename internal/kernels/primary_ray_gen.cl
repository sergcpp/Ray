R"(

int hash(int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

__kernel
void GeneratePrimaryRays(const int iteration, camera_t cam, __global const float *halton, __global ray_packet_t *out_rays) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    const int w = get_global_size(0);
    const int h = get_global_size(1);

    const int index = j * w + i;
    const int hi = (hash(index) + iteration) & (HaltonSeqLen - 1);

    const float x = (float)i;
    const float y = (float)j;

    float3 d = (float3)((x + halton[hi * 2]) / w - 0.5f, (-y - halton[hi * 2 + 1]) / h + 0.5f, 1);
    d = d.x * cam.side + d.y * cam.up + d.z * cam.fwd;

    __global ray_packet_t *r = &out_rays[index];

    ////////////////////////////////////////////////////

    float _dd = dot(d, d);
    float _dd_srqt = sqrt(_dd);

    r->do_dx = r->do_dy = (float3)(0, 0, 0);
    r->dd_dx = (_dd * cam.side - dot(d, cam.side) * d) / (_dd * _dd_srqt);
    r->dd_dy = (_dd * cam.up - dot(d, cam.up) * d) / (_dd * _dd_srqt);

    ///////////////////////////////////////////////////

    d /= _dd_srqt;

    r->o = (float4)(cam.origin, x);
    r->d = (float4)(d, y);
    r->c = (float3)(1.0f, 1.0f, 1.0f);
}

)"