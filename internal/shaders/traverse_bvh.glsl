#ifndef TRAVERSE_BVH_GLSL
#define TRAVERSE_BVH_GLSL

#include "intersect.glsl"

bool _bbox_test(vec3 o, vec3 inv_d, const float t, vec3 bbox_min, vec3 bbox_max) {
    float low = inv_d.x * (bbox_min.x - o.x);
    float high = inv_d.x * (bbox_max.x - o.x);
    float tmin = min(low, high);
    float tmax = max(low, high);

    low = inv_d.y * (bbox_min.y - o.y);
    high = inv_d.y * (bbox_max.y - o.y);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));

    low = inv_d.z * (bbox_min.z - o.z);
    high = inv_d.z * (bbox_max.z - o.z);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));
    tmax *= 1.00000024;

    return tmin <= tmax && tmin <= t && tmax > 0.0;
}

bool _bbox_test_fma(vec3 inv_d, vec3 neg_inv_d_o, float t, vec3 bbox_min, vec3 bbox_max) {
    float low = fma(inv_d.x, bbox_min.x, neg_inv_d_o.x);
    float high = fma(inv_d.x, bbox_max.x, neg_inv_d_o.x);
    float tmin = min(low, high);
    float tmax = max(low, high);

    low = fma(inv_d.y, bbox_min.y, neg_inv_d_o.y);
    high = fma(inv_d.y, bbox_max.y, neg_inv_d_o.y);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));

    low = fma(inv_d.z, bbox_min.z, neg_inv_d_o.z);
    high = fma(inv_d.z, bbox_max.z, neg_inv_d_o.z);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));
    tmax *= 1.00000024;

    return tmin <= tmax && tmin <= t && tmax > 0.0;
}

#endif // TRAVERSE_BVH_GLSL