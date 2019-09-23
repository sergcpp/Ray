R"(

ray_packet_t TransformRay(const ray_packet_t *r, const __global float16 *xform) {
    ray_packet_t _r = *r;
    
    _r.o.x = (*xform).s0 * r->o.x + (*xform).s4 * r->o.y + (*xform).s8 * r->o.z + (*xform).sc;
    _r.o.y = (*xform).s1 * r->o.x + (*xform).s5 * r->o.y + (*xform).s9 * r->o.z + (*xform).sd;
    _r.o.z = (*xform).s2 * r->o.x + (*xform).s6 * r->o.y + (*xform).sa * r->o.z + (*xform).se;

    _r.d.x = (*xform).s0 * r->d.x + (*xform).s4 * r->d.y + (*xform).s8 * r->d.z;
    _r.d.y = (*xform).s1 * r->d.x + (*xform).s5 * r->d.y + (*xform).s9 * r->d.z;
    _r.d.z = (*xform).s2 * r->d.x + (*xform).s6 * r->d.y + (*xform).sa * r->d.z;

    return _r;
}

float3 TransformPoint(const float3 p, const __global float16 *xform) {
    float3 _p;
    
    _p.x = (*xform).s0 * p.x + (*xform).s4 * p.y + (*xform).s8 * p.z + (*xform).sc;
    _p.y = (*xform).s1 * p.x + (*xform).s5 * p.y + (*xform).s9 * p.z + (*xform).sd;
    _p.z = (*xform).s2 * p.x + (*xform).s6 * p.y + (*xform).sa * p.z + (*xform).se;

    return _p;
}

float3 TransformDirection(const float3 d, const __global float16 *xform) {
    float3 _d;
    
    _d.x = (*xform).s0 * d.x + (*xform).s4 * d.y + (*xform).s8 * d.z;
    _d.y = (*xform).s1 * d.x + (*xform).s5 * d.y + (*xform).s9 * d.z;
    _d.z = (*xform).s2 * d.x + (*xform).s6 * d.y + (*xform).sa * d.z;

    return _d;
}

float3 TransformNormal(const float3 n, const __global float16 *inv_xform) {
    float3 _n;

    _n.x = (*inv_xform).s0 * n.x + (*inv_xform).s1 * n.y + (*inv_xform).s2 * n.z;
    _n.y = (*inv_xform).s4 * n.x + (*inv_xform).s5 * n.y + (*inv_xform).s6 * n.z;
    _n.z = (*inv_xform).s8 * n.x + (*inv_xform).s9 * n.y + (*inv_xform).sa * n.z;

    return _n;
}

float2 TransformUVs(float2 uv, const float2 tex_atlas_size, __global const texture_t *t, int mip_level) {
    float2 pos = (float2)((float)t->pos[mip_level][0], (float)t->pos[mip_level][1]);
    float2 size = (float2)((float)((t->width & TEXTURE_WIDTH_BITS) >> mip_level), (float)(t->height >> mip_level));
    uv = uv - floor(uv);
    float2 res = pos + uv * size + (float2)(1.0f, 1.0f);
    res /= tex_atlas_size;
    return res;
}

)"