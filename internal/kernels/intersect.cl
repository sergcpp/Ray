R"(

__constant int _next_u[] = { 1, 0, 0 },
               _next_v[] = { 2, 2, 1 };

void IntersectTris(const float3 r_o, const float3 r_d, __global const tri_accel_t *tris,
                   __global const uint *tri_indices, uint tri_index, uint tri_count, 
                   int obj_index, hit_data_t *hit) {
    const float *ro = (const float *)&r_o;
    const float *rd = (const float *)&r_d;

    for (uint j = tri_index; j < tri_index + tri_count; j++) {
        const __global tri_accel_t *tri = &tris[tri_indices[j]];

        int iw = tri->ci & TRI_W_BITS,
            iu = _next_u[iw], iv = _next_v[iw];

        float det = rd[iu] * tri->nu + rd[iv] * tri->nv + rd[iw];
        float dett = tri->np - (ro[iu] * tri->nu + ro[iv] * tri->nv + ro[iw]);
        float Du = rd[iu] * dett - (tri->pu - ro[iu]) * det;
        float Dv = rd[iv] * dett - (tri->pv - ro[iv]) * det;
        float detu = (tri->e1v * Du - tri->e1u * Dv);
        float detv = (tri->e0u * Dv - tri->e0v * Du);

        float tmpdet0 = det - detu - detv;

        if ((tmpdet0 > -HIT_EPS && detu > -HIT_EPS && detv > -HIT_EPS) ||
            (tmpdet0 < HIT_EPS && detu < HIT_EPS && detv < HIT_EPS)) {

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

float IntersectTris_Occlusion(const float3 r_o, const float3 r_d, float max_dist, __global const tri_accel_t *tris, 
                              __global const uint *tri_indices, int tri_index, int tri_count) {
    const float *rd = (const float *)&r_d;
    const float *ro = (const float *)&r_o;

    for (int j = tri_index; j < tri_index + tri_count; j++) {
        const __global tri_accel_t *tri = &tris[tri_indices[j]];

        int iw = tri->ci & TRI_W_BITS,
            iu = _next_u[iw], iv = _next_v[iw];

        float det = rd[iu] * tri->nu + rd[iv] * tri->nv + rd[iw];
        float dett = tri->np - (ro[iu] * tri->nu + ro[iv] * tri->nv + ro[iw]);
        float Du = rd[iu] * dett - (tri->pu - ro[iu]) * det;
        float Dv = rd[iv] * dett - (tri->pv - ro[iv]) * det;
        float detu = (tri->e1v * Du - tri->e1u * Dv);
        float detv = (tri->e0u * Dv - tri->e0v * Du);

        float tmpdet0 = det - detu - detv;

        if ((tmpdet0 > -HIT_EPS && detu > -HIT_EPS && detv > -HIT_EPS) ||
            (tmpdet0 < HIT_EPS && detu < HIT_EPS && detv < HIT_EPS)) {

            float t = dett / det;
            
            if (t > 0 && t < max_dist) {
                return 0;
            }
        }
    }

    return 1;
}

)"