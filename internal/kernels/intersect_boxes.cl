R"(

__kernel
void IntersectBoxes(__global const ray_packet_t *rays, __global const aabox_t *boxes, 
                    int num_boxes, __global hit_data_t *hits, __global int *num_hits) {
    const int i = get_global_id(0);

    __global const ray_packet_t *r = &rays[i];

    hit_data_t hit;
    hit.mask = 0;
    hit.t = FLT_MAX;
    hit.ray_id = (float2)(r->o.w, r->d.w);

    float3 inv_d = 1.0f / r->d.xyz;

    for (int j = 0; j < num_boxes; j++) {
        __global const aabox_t *box = &boxes[j];

        float low = inv_d.x * (box->min.x - r->o.x);
        float high = inv_d.x * (box->max.x - r->o.x);
        float tmin = min(low, high);
        float tmax = max(low, high);

        low = inv_d.y * (box->min.x - r->o.y);
        high = inv_d.y * (box->max.x - r->o.y);
        tmin = max(tmin, min(low, high));
        tmax = min(tmax, max(low, high));

        low = inv_d.z * (box->min.z - r->o.z);
        high = inv_d.z * (box->max.z - r->o.z);
        tmin = max(tmin, min(low, high));
        tmax = min(tmax, max(low, high));

        if (tmin <= tmax && tmax > 0 && tmin < hit.t) {
            hit.mask = 0xffffffff;
            hit.obj_index = j;
            hit.t = tmin;
        }
    }

    if (hit.mask) {
        int index = atomic_inc(num_hits);
        hits[index] = hit;
    }
}

)"