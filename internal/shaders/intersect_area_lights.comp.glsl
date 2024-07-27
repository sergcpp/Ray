#version 450
#extension GL_GOOGLE_include_directive : require

#include "intersect_area_lights_interface.h"
#include "common.glsl"
#include "light_bvh.glsl"
#include "traverse_bvh.glsl"

layout(push_constant) uniform UniformParams {
    Params g_params;
};

layout(std430, binding = LIGHTS_BUF_SLOT) readonly buffer Lights {
    light_t g_lights[];
};

layout(std430, binding = WNODES_BUF_SLOT) readonly buffer WNodes {
    light_wbvh_node_t g_wnodes[];
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

bool quadratic(float a, float b, float c, out float t0, out float t1) {
    const float d = b * b - 4.0 * a * c;
    if (d < 0.0) {
        return false;
    }
    const float sqrt_d = sqrt(d);
    float q;
    if (b < 0.0) {
        q = -0.5 * (b - sqrt_d);
    } else {
        q = -0.5 * (b + sqrt_d);
    }
    t0 = q / a;
    t1 = c / q;
    return true;
}

shared uint g_stack[LOCAL_GROUP_SIZE_X * LOCAL_GROUP_SIZE_Y][MAX_STACK_SIZE];
shared float g_stack_factors[LOCAL_GROUP_SIZE_X * LOCAL_GROUP_SIZE_Y][MAX_STACK_SIZE];

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

void main() {
#if PRIMARY
    if (gl_GlobalInvocationID.x >= g_params.rect.z || gl_GlobalInvocationID.y >= g_params.rect.w) {
        return;
    }

    int x = int(g_params.rect.x + gl_GlobalInvocationID.x);
    int y = int(g_params.rect.y + gl_GlobalInvocationID.y);

    int index = int(gl_GlobalInvocationID.y * g_params.rect.z) + x;
#else
    const int index = int(gl_WorkGroupID.x * 64 + gl_LocalInvocationIndex);
    if (index >= g_counters[1]) {
        return;
    }

    const int x = int((g_rays[index].xy >> 16) & 0xffff);
    const int y = int(g_rays[index].xy & 0xffff);
#endif

    ray_data_t ray = g_rays[index];

    vec3 ro = vec3(ray.o[0], ray.o[1], ray.o[2]);
    vec3 rd = vec3(ray.d[0], ray.d[1], ray.d[2]);
    uint rt = (1u << get_ray_type(ray.depth));

    hit_data_t inter = g_inout_hits[index];

    vec3 inv_d = safe_invert(rd);
    vec3 neg_inv_do = -inv_d * ro;

    uint stack_size = 0;
    g_stack[gl_LocalInvocationIndex][stack_size] = g_params.node_index;
    g_stack_factors[gl_LocalInvocationIndex][stack_size++] = 1.0;

    while (stack_size != 0) {
        uint cur = g_stack[gl_LocalInvocationIndex][--stack_size];
        float cur_factor = g_stack_factors[gl_LocalInvocationIndex][stack_size];

        light_wbvh_node_t n = g_wnodes[cur];

        if ((n.child[0] & LEAF_NODE_BIT) == 0) {
            float importance[8];
            const float total_importance = calc_lnode_importance(n, ro, importance);

            // TODO: loop in morton order based on ray direction
            for (int j = 0; j < 8; ++j) {
                if (importance[j] > 0.0 && _bbox_test(inv_d, neg_inv_do, inter.t,
                                                      vec3(n.bbox_min[0][j], n.bbox_min[1][j], n.bbox_min[2][j]),
                                                      vec3(n.bbox_max[0][j], n.bbox_max[1][j], n.bbox_max[2][j]))) {
                    g_stack_factors[gl_LocalInvocationIndex][stack_size] = cur_factor * importance[j] / total_importance;
                    g_stack[gl_LocalInvocationIndex][stack_size++] = n.child[j];
                }
            }
        } else {
            const int light_index = int(n.child[0] & PRIM_INDEX_BITS);
            light_t l = g_lights[light_index];
            [[dont_flatten]] if (!LIGHT_VISIBLE(l) || (LIGHT_RAY_VISIBILITY(l) & rt) == 0) {
                // Skip invisible light
                continue;
            }
            [[dont_flatten]] if (inter.v >= 0.0 && LIGHT_SKY_PORTAL(l)) {
                // Portal lights affect only missed rays
                continue;
            }

            bool no_shadow = !LIGHT_CAST_SHADOW(l);

            const uint light_type = LIGHT_TYPE(l);
            if (light_type == LIGHT_TYPE_SPHERE) {
                vec3 light_pos = l.SPH_POS;
                vec3 op = light_pos - ro;
                float b = dot(op, rd);
                float det = b * b - dot(op, op) + l.SPH_RADIUS * l.SPH_RADIUS;
                if (det >= 0.0) {
                    det = sqrt(det);
                    float t1 = b - det, t2 = b + det;
                    if (t1 > HIT_EPS && (t1 < inter.t || no_shadow)) {
                        bool accept = true;
                        if (l.SPH_SPOT > 0.0) {
                            const float _dot = -dot(rd, l.SPH_DIR);
                            if (_dot > 0.0) {
                                const float _angle = acos(saturate(_dot));
                                accept = accept && (_angle <= l.SPH_SPOT);
                            } else {
                                accept = false;
                            }
                        }
                        if (accept) {
                            inter.v = 0.0;
                            inter.obj_index = -int(light_index) - 1;
                            inter.t = t1;
                            inter.u = cur_factor;
                        }
                    } else if (t2 > HIT_EPS && (t2 < inter.t || no_shadow)) {
                        inter.v = 0.0;
                        inter.obj_index = -int(light_index) - 1;
                        inter.t = t2;
                        inter.u = cur_factor;
                    }
                }
            } else if (light_type == LIGHT_TYPE_DIR) {
                const float cos_theta = dot(rd, l.DIR_DIR);
                if ((inter.v < 0.0 || no_shadow) && cos_theta > cos(l.DIR_ANGLE)) {
                    inter.v = 0.0;
                    inter.obj_index = -int(light_index) - 1;
                    inter.t = 1.0 / cos_theta;
                    inter.u = cur_factor;
                }
            } else if (light_type == LIGHT_TYPE_RECT) {
                vec3 light_pos = l.RECT_POS;
                vec3 light_u = l.RECT_U;
                vec3 light_v = l.RECT_V;

                vec3 light_forward = normalize(cross(light_u, light_v));

                float plane_dist = dot(light_forward, light_pos);
                float cos_theta = dot(rd, light_forward);
                float t = (plane_dist - dot(light_forward, ro)) / cos_theta;

                if (cos_theta < 0.0 && t > HIT_EPS && (t < inter.t || no_shadow)) {
                    light_u /= dot(light_u, light_u);
                    light_v /= dot(light_v, light_v);

                    vec3 p = ro + rd * t;
                    vec3 vi = p - light_pos;
                    float a1 = dot(light_u, vi);
                    if (a1 >= -0.5 && a1 <= 0.5) {
                        float a2 = dot(light_v, vi);
                        if (a2 >= -0.5 && a2 <= 0.5) {
                            inter.v = 0.0;
                            inter.obj_index = -int(light_index) - 1;
                            inter.t = t;
                            inter.u = cur_factor;
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

                if (cos_theta < 0.0 && t > HIT_EPS && (t < inter.t || no_shadow)) {
                    light_u /= dot(light_u, light_u);
                    light_v /= dot(light_v, light_v);

                    vec3 p = ro + rd * t;
                    vec3 vi = p - light_pos;
                    float a1 = dot(light_u, vi);
                    float a2 = dot(light_v, vi);

                    if (sqrt(a1 * a1 + a2 * a2) <= 0.5) {
                        inter.v = 0.0;
                        inter.obj_index = -int(light_index) - 1;
                        inter.t = t;
                        inter.u = cur_factor;
                    }
                }
            } else if (light_type == LIGHT_TYPE_LINE) {
                vec3 light_pos = l.LINE_POS;
                vec3 light_u = l.LINE_U;
                vec3 light_dir = l.LINE_V;
                vec3 light_v = cross(light_u, light_dir);

                vec3 _ro = ro - light_pos;
                _ro = vec3(dot(_ro, light_dir), dot(_ro, light_u), dot(_ro, light_v));

                vec3 _rd = vec3(dot(rd, light_dir), dot(rd, light_u), dot(rd, light_v));

                float A = _rd[2] * _rd[2] + _rd[1] * _rd[1];
                float B = 2.0 * (_rd[2] * _ro[2] + _rd[1] * _ro[1]);
                float C = _ro[2] * _ro[2] + _ro[1] * _ro[1] - l.LINE_RADIUS * l.LINE_RADIUS;

                float t0, t1;
                if (quadratic(A, B, C, t0, t1) && t0 > HIT_EPS && t1 > HIT_EPS) {
                    const float t = min(t0, t1);
                    const vec3 p = _ro + t * _rd;
                    if (abs(p[0]) < 0.5 * l.LINE_HEIGHT && (t < inter.t || no_shadow)) {
                        inter.v = 0.0;
                        inter.obj_index = -int(light_index) - 1;
                        inter.t = t;
                        inter.u = cur_factor;
                    }
                }
            } else if (light_type == LIGHT_TYPE_ENV && inter.v < 0.0) {
                // NOTE: mask remains empty
                inter.obj_index = -int(light_index) - 1;
                inter.u = cur_factor;
            }
        }
    }

    g_inout_hits[index] = inter;
}