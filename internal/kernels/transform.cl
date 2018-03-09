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

float3 TransformNormal(const float3 *n, const __global float16 *inv_xform) {
    float3 _n;

    _n.x = (*inv_xform).s0 * (*n).x + (*inv_xform).s1 * (*n).y + (*inv_xform).s2 * (*n).z;
    _n.y = (*inv_xform).s4 * (*n).x + (*inv_xform).s5 * (*n).y + (*inv_xform).s6 * (*n).z;
    _n.z = (*inv_xform).s8 * (*n).x + (*inv_xform).s9 * (*n).y + (*inv_xform).sa * (*n).z;

    return _n;
}

float2 TransformUVs(float2 uv, const float2 tex_atlas_size, __global const texture_t *t, int mip_level) {
    float2 pos = (float2)((float)t->pos[mip_level][0], (float)t->pos[mip_level][1]);
    float2 size = (float2)((float)(t->size[0] >> mip_level), (float)(t->size[1] >> mip_level));
    uv = uv - floor(uv);
    float2 res = pos + uv * size + (float2)(1.0f, 1.0f);
    res /= tex_atlas_size;
    return res;
}

)"