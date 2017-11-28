R"(

__kernel
void IntersectCones(__global const ray_packet_t *rays, __global const cone_accel_t *cones, 
                    int num_cones, __global hit_data_t *hits, __global int *num_hits) {
    const int i = get_global_id(0);

    __global const ray_packet_t *r = &rays[i];

    hit_data_t hit;
    hit.mask = 0;
    hit.t = FLT_MAX;
    hit.ray_id = (float2)(r->o.w, r->d.w);

    float3 o = r->o.xyz;
    float3 d = r->d.xyz;

    for (int j = 0; j < num_cones; j++) {
        __global const cone_accel_t *cone = &cones[j];

        float3 co = o - cone->o;

        float a = dot(d, cone->v);
        float c = dot(co, cone->v);
        float b = 2 * (a * c - dot(d, co) * cone->cos_phi_sqr);
        a = a * a - cone->cos_phi_sqr;
        c = c * c - dot(co, co) * cone->cos_phi_sqr;

        float D = b * b - 4 * a * c;
        if (D >= 0) {
            D = sqrt(D);
            float t1 = (-b - D) / (2 * a), t2 = (-b + D) / (2 * a);

            if ((t1 > 0 && t1 < hit.t) || (t2 > 0 && t2 < hit.t)) {
                float3 p1 = o + t1 * d, p2 = o + t2 * d;
                float3 p1c = cone->o - p1, p2c = cone->o - p2;

                float dot1 = dot(p1c, cone->v), dot2 = dot(p2c, cone->v);

                if ((dot1 >= cone->cone_start && dot1 <= cone->cone_end) || (dot2 >= cone->cone_start && dot2 <= cone->cone_end)) {
                    hit.mask = 0xffffffff;
                    hit.obj_index = j;
                    hit.t = t1 < t2 ? t1 : t2;
                }
            }
        }
    }

    if (hit.mask) {
        int index = atomic_inc(num_hits);
        hits[index] = hit;
    }
}

)"