R"(

__constant int _next_u[] = { 1, 0, 0 },
               _next_v[] = { 2, 2, 1 };

__constant float _hit_eps = 0.0001f;

void IntersectTris(const ray_packet_t *r, __global const tri_accel_t *tris,
                   __global const uint *tri_indices, uint tri_index, uint tri_count, 
                   int obj_index, hit_data_t *hit) {

    const float *rd = (const float *)&r->d;
    const float *ro = (const float *)&r->o;

    for (uint j = tri_index; j < tri_index + tri_count; j++) {
        const __global tri_accel_t *tri = &tris[tri_indices[j]];

        int w = tri->ci & W_BITS,
            u = _next_u[w], v = _next_v[w];

        float det = rd[u] * tri->nu + rd[v] * tri->nv + rd[w];
        float dett = tri->np - (ro[u] * tri->nu + ro[v] * tri->nv + ro[w]);
        float Du = rd[u] * dett - (tri->pu - ro[u]) * det;
        float Dv = rd[v] * dett - (tri->pv - ro[v]) * det;
        float detu = (tri->e1v * Du - tri->e1u * Dv);
        float detv = (tri->e0u * Dv - tri->e0v * Du);

        float tmpdet0 = det - detu - detv;

        if ((tmpdet0 > -_hit_eps & detu > -_hit_eps && detv > -_hit_eps) || 
            (tmpdet0 < _hit_eps && detu < _hit_eps && detv < _hit_eps)) {

            float rdet = 1.0f / det;
            float t = dett * rdet;
            float u = detu * rdet;
            float v = detv * rdet;
            
            if (t > 0 && t < hit->t) {
                hit->mask = 0xffffffff;
                hit->obj_index = obj_index;
                hit->prim_index = tri_indices[j];
                hit->t = t;
                hit->u = u;
                hit->v = v;
            }
        }
    }
}

float IntersectTris_Shadow(const ray_packet_t *r, __global const tri_accel_t *tris, 
                           __global const uint *tri_indices, int tri_index, int tri_count) {
    const float *rd = (const float *)&r->d;
    const float *ro = (const float *)&r->o;

    for (int j = tri_index; j < tri_index + tri_count; j++) {
        const __global tri_accel_t *tri = &tris[tri_indices[j]];

        int w = tri->ci & W_BITS,
            u = _next_u[w], v = _next_v[w];

        float det = rd[u] * tri->nu + rd[v] * tri->nv + rd[w];
        float dett = tri->np - (ro[u] * tri->nu + ro[v] * tri->nv + ro[w]);
        float Du = rd[u] * dett - (tri->pu - ro[u]) * det;
        float Dv = rd[v] * dett - (tri->pv - ro[v]) * det;
        float detu = (tri->e1v * Du - tri->e1u * Dv);
        float detv = (tri->e0u * Dv - tri->e0v * Du);

        float tmpdet0 = det - detu - detv;

        if (sign(dett) == sign(det) && 
            ((tmpdet0 > -_hit_eps && detu > -_hit_eps && detv > -_hit_eps) || 
            (tmpdet0 < _hit_eps && detu < _hit_eps && detv < _hit_eps))) {
            return 0;
        }
    }

    return 1;
}

)"