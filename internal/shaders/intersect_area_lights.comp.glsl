#version 450
#extension GL_GOOGLE_include_directive : require

#include "intersect_area_lights_interface.glsl"
#include "types.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(std430, binding = LIGHTS_BUF_SLOT) readonly buffer Lights {
    light_t g_lights[];
};

layout(std430, binding = VISIBLE_LIGHTS_BUF_SLOT) readonly buffer VisibleLights {
    uint g_visible_lights[];
};

layout(std430, binding = TRANSFORMS_BUF_SLOT) readonly buffer Transforms {
    transform_t g_transforms[];
};

layout(std430, binding = RAYS_BUF_SLOT) readonly buffer Rays {
    ray_data_t g_rays[];
};

#if !PRIMARY
layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};
#endif

layout(std430, binding = INOUT_HITS_BUF_SLOT) buffer Hits {
    hit_data_t g_inout_hits[];
};

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

void main() {
#if PRIMARY
    if (gl_GlobalInvocationID.x >= g_params.img_size.x || gl_GlobalInvocationID.y >= g_params.img_size.y) {
        return;
    }

    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);

    int index = y * int(g_params.img_size.x) + x;
#else
    const int index = int(gl_WorkGroupID.x * 64 + gl_LocalInvocationIndex);
    if (index >= g_counters[1]) {
        return;
    }

    const int x = (g_rays[index].xy >> 16) & 0xffff;
    const int y = (g_rays[index].xy & 0xffff);
#endif

    ray_data_t ray = g_rays[index];

    vec3 ro = vec3(ray.o[0], ray.o[1], ray.o[2]);
    vec3 rd = vec3(ray.d[0], ray.d[1], ray.d[2]);

    hit_data_t inter = g_inout_hits[index];

    for (uint li = 0; li < g_params.visible_lights_count; ++li) {
        uint light_index = g_visible_lights[li];
        light_t l = g_lights[light_index];

        uint light_type = (l.type_and_param0.x & 0x3f);
        if (light_type == LIGHT_TYPE_SPHERE) {
            vec3 light_pos = l.SPH_POS;
            vec3 op = light_pos - ro;
            float b = dot(op, rd);
            float det = b * b - dot(op, op) + l.SPH_RADIUS * l.SPH_RADIUS;
            if (det >= 0.0) {
                det = sqrt(det);
                float t1 = b - det, t2 = b + det;
                if (t1 > HIT_EPS && t1 < inter.t) {
                    inter.mask = -1;
                    inter.obj_index = -int(light_index) - 1;
                    inter.t = t1;
                } else if (t2 > HIT_EPS && t2 < inter.t) {
                    inter.mask = -1;
                    inter.obj_index = -int(light_index) - 1;
                    inter.t = t2;
                }
            }
        } else if (light_type == LIGHT_TYPE_RECT) {
            vec3 light_pos = l.RECT_POS;
            vec3 light_u = l.RECT_U;
            vec3 light_v = l.RECT_V;

            vec3 light_forward = normalize(cross(light_u, light_v));

            float plane_dist = dot(light_forward, light_pos);
            float cos_theta = dot(rd, light_forward);
            float t = (plane_dist - dot(light_forward, ro)) / cos_theta;

            if (cos_theta < 0.0 && t > HIT_EPS && t < inter.t) {
                light_u /= dot(light_u, light_u);
                light_v /= dot(light_v, light_v);

                vec3 p = ro + rd * t;
                vec3 vi = p - light_pos;
                float a1 = dot(light_u, vi);
                if (a1 >= -0.5 && a1 <= 0.5) {
                    float a2 = dot(light_v, vi);
                    if (a2 >= -0.5 && a2 <= 0.5) {
                        inter.mask = -1;
                        inter.obj_index = -int(light_index) - 1;
                        inter.t = t;
                    }
                }
            }
        } else if (light_type == LIGHT_TYPE_DISK) {
            vec3 light_pos = l.DISK_POS;
            vec3 light_u = l.DISK_U;
            vec3 light_v = l.DISK_V;

            vec3 light_forward = normalize(cross(light_u, light_v));

            float plane_dist = dot(light_forward, light_pos);
            float cos_theta = dot(rd, light_forward);
            float t = (plane_dist - dot(light_forward, ro)) / cos_theta;

            if (cos_theta < 0.0 && t > HIT_EPS && t < inter.t) {
                light_u /= dot(light_u, light_u);
                light_v /= dot(light_v, light_v);

                vec3 p = ro + rd * t;
                vec3 vi = p - light_pos;
                float a1 = dot(light_u, vi);
                float a2 = dot(light_v, vi);

                if (sqrt(a1 * a1 + a2 * a2) <= 0.5) {
                    inter.mask = -1;
                    inter.obj_index = -int(light_index) - 1;
                    inter.t = t;
                }
            }
        }
    }

    g_inout_hits[index] = inter;
}