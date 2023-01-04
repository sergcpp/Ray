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

//layout(std430, binding = HALTON_SEQ_BUF_SLOT) readonly buffer Halton {
//    float g_halton[];
//};

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

#if HWRT
layout(binding = TLAS_SLOT) uniform accelerationStructureEXT g_tlas;
#endif

layout(binding = OUT_IMG_SLOT, rgba32f) uniform image2D g_out_img;

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

bool ComputeVisibility(vec3 p, vec3 d, float dist, float rand_val, int rand_hash2) {
#if !HWRT
    const vec3 inv_d = safe_invert(d);

    while (dist > HIT_BIAS) {
        hit_data_t sh_inter;
        sh_inter.mask = 0;
        sh_inter.t = dist;

        const bool solid_hit = Traverse_MacroTree_WithStack(p, d, inv_d, g_params.node_index, sh_inter);
        if (solid_hit) {
            return false;
        }

        if (sh_inter.mask == 0) {
            return true;
        }

        const bool is_backfacing = (sh_inter.prim_index < 0);
        const uint prim_index = is_backfacing ? -sh_inter.prim_index - 1 : sh_inter.prim_index;

        const uint tri_index = g_tri_indices[prim_index];

        const uint front_mi = (g_tri_materials[tri_index] >> 16u) & 0xffff;
        const uint back_mi = (g_tri_materials[tri_index] & 0xffff);

        const uint mat_index = (is_backfacing ? back_mi : front_mi) & MATERIAL_INDEX_BITS;
        material_t mat = g_materials[mat_index];

        const transform_t tr = g_transforms[floatBitsToUint(g_mesh_instances[sh_inter.obj_index].bbox_min.w)];

        const vertex_t v1 = g_vertices[g_vtx_indices[tri_index * 3 + 0]];
        const vertex_t v2 = g_vertices[g_vtx_indices[tri_index * 3 + 1]];
        const vertex_t v3 = g_vertices[g_vtx_indices[tri_index * 3 + 2]];

        const float w = 1.0 - sh_inter.u - sh_inter.v;
        const vec2 sh_uvs =
            vec2(v1.t[0][0], v1.t[0][1]) * w + vec2(v2.t[0][0], v2.t[0][1]) * sh_inter.u + vec2(v3.t[0][0], v3.t[0][1]) * sh_inter.v;

        {
            const int sh_rand_hash = hash(rand_hash2);
            const float sh_rand_offset = construct_float(sh_rand_hash);

            float sh_r = fract(rand_val + sh_rand_offset);

            // resolve mix material
            while (mat.type == MixNode) {
                float mix_val = mat.tangent_rotation_or_strength;
                if (mat.textures[BASE_TEXTURE] != 0xffffffff) {
                    mix_val *= SampleBilinear(mat.textures[BASE_TEXTURE], sh_uvs, 0).r;
                }

                if (sh_r > mix_val) {
                    mat = g_materials[mat.textures[MIX_MAT1]];
                    sh_r = (sh_r - mix_val) / (1.0 - mix_val);
                } else {
                    mat = g_materials[mat.textures[MIX_MAT2]];
                    sh_r = sh_r / mix_val;
                }
            }

            if (mat.type != TransparentNode) {
                return false;
            }
        }

        const float t = sh_inter.t + HIT_BIAS;
        p += d * t;
        dist -= t;
    }

    return true;
#else
    const uint ray_flags = 0;//gl_RayFlagsCullBackFacingTrianglesEXT;

    rayQueryEXT rq;
    rayQueryInitializeEXT(rq,               // rayQuery
                          g_tlas,           // topLevel
                          ray_flags,        // rayFlags
                          0xff,             // cullMask
                          p,                // origin
                          0.0,              // tMin
                          d,                // direction
                          dist              // tMax
                          );
    while(rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
            // perform alpha test
            const int primitive_offset = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false);
            const int obj_index = rayQueryGetIntersectionInstanceIdEXT(rq, false);
            const int tri_index = primitive_offset + rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);

            const uint front_mi = (g_tri_materials[tri_index] >> 16u) & 0xffff;
            const uint back_mi = (g_tri_materials[tri_index] & 0xffff);

            const bool is_backfacing = !rayQueryGetIntersectionFrontFaceEXT(rq, false);
            if ((!is_backfacing && (front_mi & MATERIAL_SOLID_BIT) != 0 ||
                (is_backfacing && (back_mi & MATERIAL_SOLID_BIT) != 0))) {
                rayQueryConfirmIntersectionEXT(rq);
                break;
            }

            const uint mat_index = (is_backfacing ? back_mi : front_mi) & MATERIAL_INDEX_BITS;
            material_t mat = g_materials[mat_index];

            const transform_t tr = g_transforms[floatBitsToUint(g_mesh_instances[obj_index].bbox_min.w)];

            const vertex_t v1 = g_vertices[g_vtx_indices[tri_index * 3 + 0]];
            const vertex_t v2 = g_vertices[g_vtx_indices[tri_index * 3 + 1]];
            const vertex_t v3 = g_vertices[g_vtx_indices[tri_index * 3 + 2]];

            const vec2 uv = rayQueryGetIntersectionBarycentricsEXT(rq, false);
            const float w = 1.0 - uv.x - uv.y;
            const vec2 sh_uvs = vec2(v1.t[0][0], v1.t[0][1]) * w + vec2(v2.t[0][0], v2.t[0][1]) * uv.x + vec2(v3.t[0][0], v3.t[0][1]) * uv.y;

            const int sh_rand_hash = hash(rand_hash2);
            const float sh_rand_offset = construct_float(sh_rand_hash);

            float sh_r = fract(rand_val + sh_rand_offset);

            // resolve mix material
            while (mat.type == MixNode) {
                float mix_val = mat.tangent_rotation_or_strength;
                if (mat.textures[BASE_TEXTURE] != 0xffffffff) {
                    mix_val *= SampleBilinear(mat.textures[BASE_TEXTURE], sh_uvs, 0).r;
                }

                if (sh_r > mix_val) {
                    mat = g_materials[mat.textures[MIX_MAT1]];
                    sh_r = (sh_r - mix_val) / (1.0 - mix_val);
                } else {
                    mat = g_materials[mat.textures[MIX_MAT2]];
                    sh_r = sh_r / mix_val;
                }
            }

            if (mat.type != TransparentNode) {
                rayQueryConfirmIntersectionEXT(rq);
                break;
            }
        }
    }
    return rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT;
#endif
}

void main() {
    const int index = int(gl_WorkGroupID.x * 64 + gl_LocalInvocationIndex);
    if (index >= g_counters[3]) {
        return;
    }

    shadow_ray_t sh_ray = g_sh_rays[index];

    const int x = (sh_ray.xy >> 16) & 0xffff;
    const int y = (sh_ray.xy & 0xffff);

    vec3 ro = vec3(g_sh_rays[index].o[0], g_sh_rays[index].o[1], g_sh_rays[index].o[2]);
    vec3 rd = vec3(g_sh_rays[index].d[0], g_sh_rays[index].d[1], g_sh_rays[index].d[2]);

    if (ComputeVisibility(ro, rd, sh_ray.dist, g_params.random_val, hash((x << 16) | y))) {
        vec3 col = imageLoad(g_out_img, ivec2(x, y)).rgb;
        col += vec3(sh_ray.c[0], sh_ray.c[1], sh_ray.c[2]);
        imageStore(g_out_img, ivec2(x, y), vec4(col, 1.0));
    }
}