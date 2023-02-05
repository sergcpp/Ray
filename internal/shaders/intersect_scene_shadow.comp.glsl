#version 450
#extension GL_GOOGLE_include_directive : require
#if HWRT
#extension GL_EXT_ray_query : require
#endif

#include "intersect_scene_shadow_interface.h"
#include "common.glsl"
#include "texture.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(std430, binding = TRIS_BUF_SLOT) readonly buffer Tris {
    tri_accel_t g_tris[];
};

layout(std430, binding = TRI_INDICES_BUF_SLOT) readonly buffer TriIndices {
    uint g_tri_indices[];
};

layout(std430, binding = TRI_MATERIALS_BUF_SLOT) readonly buffer TriMaterials {
    uint g_tri_materials[];
};

layout(std430, binding = MATERIALS_BUF_SLOT) readonly buffer Materials {
    material_t g_materials[];
};

layout(std430, binding = NODES_BUF_SLOT) readonly buffer Nodes {
    bvh_node_t g_nodes[];
};

layout(std430, binding = MESHES_BUF_SLOT) readonly buffer Meshes {
    mesh_t g_meshes[];
};

layout(std430, binding = MESH_INSTANCES_BUF_SLOT) readonly buffer MeshInstances {
    mesh_instance_t g_mesh_instances[];
};

layout(std430, binding = MI_INDICES_BUF_SLOT) readonly buffer MiIndices {
    uint g_mi_indices[];
};

layout(std430, binding = TRANSFORMS_BUF_SLOT) readonly buffer Transforms {
    transform_t g_transforms[];
};

layout(std430, binding = VERTICES_BUF_SLOT) readonly buffer Vertices {
    vertex_t g_vertices[];
};

layout(std430, binding = VTX_INDICES_BUF_SLOT) readonly buffer VtxIndices {
    uint g_vtx_indices[];
};

layout(std430, binding = SH_RAYS_BUF_SLOT) readonly buffer Rays {
    shadow_ray_t g_sh_rays[];
};

layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};

layout(std430, binding = LIGHTS_BUF_SLOT) readonly buffer Lights {
    light_t g_lights[];
};

layout(std430, binding = BLOCKER_LIGHTS_BUF_SLOT) readonly buffer BlockerLights {
    uint g_blocker_lights[];
};

#if HWRT
layout(binding = TLAS_SLOT) uniform accelerationStructureEXT g_tlas;
#endif

layout(binding = INOUT_IMG_SLOT, rgba32f) uniform image2D g_inout_img;

#define FETCH_TRI(j) g_tris[j]
#include "traverse_bvh.glsl"

#define near_child(rd, n)   \
    (rd)[floatBitsToUint(n.bbox_max.w) >> 30] < 0 ? (floatBitsToUint(n.bbox_max.w) & RIGHT_CHILD_BITS) : floatBitsToUint(n.bbox_min.w)

#define far_child(rd, n)    \
    (rd)[floatBitsToUint(n.bbox_max.w) >> 30] < 0 ? floatBitsToUint(n.bbox_min.w) : (floatBitsToUint(n.bbox_max.w) & RIGHT_CHILD_BITS)

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

shared uint g_stack[LOCAL_GROUP_SIZE_X * LOCAL_GROUP_SIZE_Y][MAX_STACK_SIZE];

bool Traverse_MicroTree_WithStack(vec3 ro, vec3 rd, vec3 inv_d, int obj_index, uint node_index, uint stack_size,
                                  inout hit_data_t inter) {
    vec3 neg_inv_do = -inv_d * ro;

    uint initial_stack_size = stack_size;
    g_stack[gl_LocalInvocationIndex][stack_size++] = node_index;

    while (stack_size != initial_stack_size) {
        uint cur = g_stack[gl_LocalInvocationIndex][--stack_size];

        bvh_node_t n = g_nodes[cur];

        if (!_bbox_test_fma(inv_d, neg_inv_do, inter.t, n.bbox_min.xyz, n.bbox_max.xyz)) {
            continue;
        }

        if ((floatBitsToUint(n.bbox_min.w) & LEAF_NODE_BIT) == 0) {
            g_stack[gl_LocalInvocationIndex][stack_size++] = far_child(rd, n);
            g_stack[gl_LocalInvocationIndex][stack_size++] = near_child(rd, n);
        } else {
            const int tri_start = int(floatBitsToUint(n.bbox_min.w) & PRIM_INDEX_BITS);
            const int tri_end = tri_start + floatBitsToInt(n.bbox_max.w);

            const bool hit_found = IntersectTris_AnyHit(ro, rd, tri_start, tri_end, obj_index, inter);
            if (hit_found) {
                const bool is_backfacing = (inter.prim_index < 0);
                const uint prim_index = is_backfacing ? -inter.prim_index - 1 : inter.prim_index;
                const uint tri_index = g_tri_indices[prim_index];

                const uint front_mi = (g_tri_materials[tri_index] >> 16u) & 0xffff;
                const uint back_mi = (g_tri_materials[tri_index] & 0xffff);

                if ((!is_backfacing && (front_mi & MATERIAL_SOLID_BIT) != 0) ||
                    (is_backfacing && (back_mi & MATERIAL_SOLID_BIT) != 0)) {
                    return true;
                }
            }
        }
    }

    return false;
}

bool Traverse_MacroTree_WithStack(vec3 orig_ro, vec3 orig_rd, vec3 orig_inv_rd, uint node_index, inout hit_data_t inter) {
    vec3 orig_neg_inv_do = -orig_inv_rd * orig_ro;

    uint stack_size = 0;
    g_stack[gl_LocalInvocationIndex][stack_size++] = node_index;

    while (stack_size != 0) {
        uint cur = g_stack[gl_LocalInvocationIndex][--stack_size];

        bvh_node_t n = g_nodes[cur];

        if (!_bbox_test_fma(orig_inv_rd, orig_neg_inv_do, inter.t, n.bbox_min.xyz, n.bbox_max.xyz)) {
            continue;
        }

        if ((floatBitsToUint(n.bbox_min.w) & LEAF_NODE_BIT) == 0) {
            g_stack[gl_LocalInvocationIndex][stack_size++] = far_child(orig_rd, n);
            g_stack[gl_LocalInvocationIndex][stack_size++] = near_child(orig_rd, n);
        } else {
            uint prim_index = (floatBitsToUint(n.bbox_min.w) & PRIM_INDEX_BITS);
            uint prim_count = floatBitsToUint(n.bbox_max.w);
            for (uint i = prim_index; i < prim_index + prim_count; ++i) {
                mesh_instance_t mi = g_mesh_instances[g_mi_indices[i]];
                mesh_t m = g_meshes[floatBitsToUint(mi.bbox_max.w)];
                transform_t tr = g_transforms[floatBitsToUint(mi.bbox_min.w)];

                if (!_bbox_test_fma(orig_inv_rd, orig_neg_inv_do, inter.t, mi.bbox_min.xyz, mi.bbox_max.xyz)) {
                    continue;
                }

                const vec3 ro = (tr.inv_xform * vec4(orig_ro, 1.0)).xyz;
                const vec3 rd = (tr.inv_xform * vec4(orig_rd, 0.0)).xyz;
                const vec3 inv_d = safe_invert(rd);

                const bool solid_hit_found = Traverse_MicroTree_WithStack(ro, rd, inv_d, int(g_mi_indices[i]), m.node_index, stack_size, inter);
                if (solid_hit_found) {
                    return true;
                }
            }
        }
    }

    return false;
}

vec3 IntersectSceneShadow(shadow_ray_t r) {
    vec3 ro = vec3(r.o[0], r.o[1], r.o[2]);
    vec3 rd = vec3(r.d[0], r.d[1], r.d[2]);
    vec3 rc = vec3(r.c[0], r.c[1], r.c[2]);
    int depth = (r.depth >> 24);

    float dist = r.dist > 0.0 ? r.dist : MAX_DIST;
#if !HWRT
    const vec3 inv_d = safe_invert(rd);

    while (dist > HIT_BIAS) {
        hit_data_t sh_inter;
        sh_inter.mask = 0;
        sh_inter.t = dist;

        const bool solid_hit = Traverse_MacroTree_WithStack(ro, rd, inv_d, g_params.node_index, sh_inter);
        if (solid_hit || depth > g_params.max_transp_depth) {
            return vec3(0.0);
        }

        if (sh_inter.mask == 0) {
            return rc;
        }

        const bool is_backfacing = (sh_inter.prim_index < 0);
        const uint prim_index = is_backfacing ? -sh_inter.prim_index - 1 : sh_inter.prim_index;

        const uint tri_index = g_tri_indices[prim_index];

        const uint front_mi = (g_tri_materials[tri_index] >> 16u) & 0xffff;
        const uint back_mi = (g_tri_materials[tri_index] & 0xffff);

        const uint mat_index = (is_backfacing ? back_mi : front_mi) & MATERIAL_INDEX_BITS;

        const vertex_t v1 = g_vertices[g_vtx_indices[tri_index * 3 + 0]];
        const vertex_t v2 = g_vertices[g_vtx_indices[tri_index * 3 + 1]];
        const vertex_t v3 = g_vertices[g_vtx_indices[tri_index * 3 + 2]];

        const float w = 1.0 - sh_inter.u - sh_inter.v;
        const vec2 sh_uvs =
            vec2(v1.t[0][0], v1.t[0][1]) * w + vec2(v2.t[0][0], v2.t[0][1]) * sh_inter.u + vec2(v3.t[0][0], v3.t[0][1]) * sh_inter.v;

        // reuse traversal stack
        g_stack[gl_LocalInvocationIndex][0] = mat_index;
        g_stack[gl_LocalInvocationIndex][1] = floatBitsToUint(1.0);
        int stack_size = 1;

        vec3 throughput = vec3(0.0);

        while (stack_size-- != 0) {
            material_t mat = g_materials[g_stack[gl_LocalInvocationIndex][2 * stack_size + 0]];
            float weight = uintBitsToFloat(g_stack[gl_LocalInvocationIndex][2 * stack_size + 1]);

            // resolve mix material
            if (mat.type == MixNode) {
                float mix_val = mat.tangent_rotation_or_strength;
                if (mat.textures[BASE_TEXTURE] != 0xffffffff) {
                    mix_val *= SampleBilinear(mat.textures[BASE_TEXTURE], sh_uvs, 0).r;
                }

                g_stack[gl_LocalInvocationIndex][2 * stack_size + 0] = mat.textures[MIX_MAT1];
                g_stack[gl_LocalInvocationIndex][2 * stack_size + 1] = floatBitsToUint(weight * (1.0 - mix_val));
                ++stack_size;
                g_stack[gl_LocalInvocationIndex][2 * stack_size + 0] = mat.textures[MIX_MAT2];
                g_stack[gl_LocalInvocationIndex][2 * stack_size + 1] = floatBitsToUint(weight * mix_val);
                ++stack_size;
            } else if (mat.type == TransparentNode) {
                throughput += weight * vec3(mat.base_color[0], mat.base_color[1], mat.base_color[2]);
            }
        }

        rc *= throughput;
        if (lum(rc) < FLT_EPS) {
            break;
        }

        const float t = sh_inter.t + HIT_BIAS;
        ro += rd * t;
        dist -= t;

        ++depth;
    }

    return rc;
#else
    while (dist > HIT_BIAS) {
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq,               // rayQuery
                              g_tlas,           // topLevel
                              0,                // rayFlags
                              0xff,             // cullMask
                              ro,               // origin
                              0.0,              // tMin
                              rd,               // direction
                              dist              // tMax
                              );

        while(rayQueryProceedEXT(rq)) {
            if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
                rayQueryConfirmIntersectionEXT(rq);
            }
        }

        if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
            // perform alpha test
            const int primitive_offset = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
            const int obj_index = rayQueryGetIntersectionInstanceIdEXT(rq, true);
            const int tri_index = primitive_offset + rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);

            const uint front_mi = (g_tri_materials[tri_index] >> 16u) & 0xffff;
            const uint back_mi = (g_tri_materials[tri_index] & 0xffff);

            const bool is_backfacing = !rayQueryGetIntersectionFrontFaceEXT(rq, true);
            const bool solid_hit = (!is_backfacing && (front_mi & MATERIAL_SOLID_BIT) != 0) ||
                                   (is_backfacing && (back_mi & MATERIAL_SOLID_BIT) != 0);
            if (solid_hit || depth > g_params.max_transp_depth) {
                return vec3(0.0);
            }

            const uint mat_index = (is_backfacing ? back_mi : front_mi) & MATERIAL_INDEX_BITS;

            const vertex_t v1 = g_vertices[g_vtx_indices[tri_index * 3 + 0]];
            const vertex_t v2 = g_vertices[g_vtx_indices[tri_index * 3 + 1]];
            const vertex_t v3 = g_vertices[g_vtx_indices[tri_index * 3 + 2]];

            const vec2 uv = rayQueryGetIntersectionBarycentricsEXT(rq, true);
            const float w = 1.0 - uv.x - uv.y;
            const vec2 sh_uvs = vec2(v1.t[0][0], v1.t[0][1]) * w + vec2(v2.t[0][0], v2.t[0][1]) * uv.x + vec2(v3.t[0][0], v3.t[0][1]) * uv.y;

            // reuse traversal stack
            g_stack[gl_LocalInvocationIndex][0] = mat_index;
            g_stack[gl_LocalInvocationIndex][1] = floatBitsToUint(1.0);
            int stack_size = 1;

            vec3 throughput = vec3(0.0);

            while (stack_size-- != 0) {
                material_t mat = g_materials[g_stack[gl_LocalInvocationIndex][2 * stack_size + 0]];
                float weight = uintBitsToFloat(g_stack[gl_LocalInvocationIndex][2 * stack_size + 1]);

                // resolve mix material
                if (mat.type == MixNode) {
                    float mix_val = mat.tangent_rotation_or_strength;
                    if (mat.textures[BASE_TEXTURE] != 0xffffffff) {
                        mix_val *= SampleBilinear(mat.textures[BASE_TEXTURE], sh_uvs, 0).r;
                    }

                    g_stack[gl_LocalInvocationIndex][2 * stack_size + 0] = mat.textures[MIX_MAT1];
                    g_stack[gl_LocalInvocationIndex][2 * stack_size + 1] = floatBitsToUint(weight * (1.0 - mix_val));
                    ++stack_size;
                    g_stack[gl_LocalInvocationIndex][2 * stack_size + 0] = mat.textures[MIX_MAT2];
                    g_stack[gl_LocalInvocationIndex][2 * stack_size + 1] = floatBitsToUint(weight * mix_val);
                    ++stack_size;
                } else if (mat.type == TransparentNode) {
                    throughput += weight * vec3(mat.base_color[0], mat.base_color[1], mat.base_color[2]);
                }
            }

            rc *= throughput;
            if (lum(rc) < FLT_EPS) {
                break;
            }
        }

        const float t = rayQueryGetIntersectionTEXT(rq, true) + HIT_BIAS;
        ro += rd * t;
        dist -= t;

        ++depth;
    }

    return rc;
#endif
}

float IntersectAreaLightsShadow(shadow_ray_t r) {
    vec3 ro = vec3(r.o[0], r.o[1], r.o[2]);
    vec3 rd = vec3(r.d[0], r.d[1], r.d[2]);

    float rdist = abs(r.dist);

    for (uint li = 0; li < g_params.blocker_lights_count; ++li) {
        uint light_index = g_blocker_lights[li];
        light_t l = g_lights[light_index];

        uint light_type = (l.type_and_param0.x & 0x1f);
        if (light_type == LIGHT_TYPE_RECT) {
            vec3 light_pos = l.RECT_POS;
            vec3 light_u = l.RECT_U;
            vec3 light_v = l.RECT_V;

            vec3 light_forward = normalize(cross(light_u, light_v));

            float plane_dist = dot(light_forward, light_pos);
            float cos_theta = dot(rd, light_forward);
            float t = (plane_dist - dot(light_forward, ro)) / cos_theta;

            if (cos_theta < 0.0 && t > HIT_EPS && t < rdist) {
                light_u /= dot(light_u, light_u);
                light_v /= dot(light_v, light_v);

                vec3 p = ro + rd * t;
                vec3 vi = p - light_pos;
                float a1 = dot(light_u, vi);
                if (a1 >= -0.5 && a1 <= 0.5) {
                    float a2 = dot(light_v, vi);
                    if (a2 >= -0.5 && a2 <= 0.5) {
                        return 0.0;
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

            if (cos_theta < 0.0 && t > HIT_EPS && t < rdist) {
                light_u /= dot(light_u, light_u);
                light_v /= dot(light_v, light_v);

                vec3 p = ro + rd * t;
                vec3 vi = p - light_pos;
                float a1 = dot(light_u, vi);
                float a2 = dot(light_v, vi);

                if (sqrt(a1 * a1 + a2 * a2) <= 0.5) {
                    return 0.0;
                }
            }
        }
    }

    return 1.0;
}

void main() {
    const int index = int(gl_WorkGroupID.x * 64 + gl_LocalInvocationIndex);
    if (index >= g_counters[3]) {
        return;
    }

    shadow_ray_t sh_ray = g_sh_rays[index];

    const vec3 rc = IntersectSceneShadow(sh_ray) * IntersectAreaLightsShadow(sh_ray);
    if (lum(rc) > 0.0) {
        const int x = (sh_ray.xy >> 16) & 0xffff;
        const int y = (sh_ray.xy & 0xffff);

        vec3 col = imageLoad(g_inout_img, ivec2(x, y)).rgb;
        col += rc;
        imageStore(g_inout_img, ivec2(x, y), vec4(col, 1.0));
    }
}