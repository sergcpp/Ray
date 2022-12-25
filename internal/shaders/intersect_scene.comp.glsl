#version 450
#extension GL_GOOGLE_include_directive : require
#if HWRT
#extension GL_EXT_ray_query : require
#endif

#include "intersect_scene_interface.h"
#include "common.glsl"
#include "texture.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

#if !HWRT
layout(std430, binding = TRIS_BUF_SLOT) readonly buffer Tris {
    tri_accel_t g_tris[];
};

layout(std430, binding = TRI_INDICES_BUF_SLOT) readonly buffer TriIndices {
    uint g_tri_indices[];
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
#else
layout(binding = TLAS_SLOT) uniform accelerationStructureEXT g_tlas;
#endif

layout(std430, binding = TRI_MATERIALS_BUF_SLOT) readonly buffer TriMaterials {
    uint g_tri_materials[];
};

layout(std430, binding = MATERIALS_BUF_SLOT) readonly buffer Materials {
    material_t g_materials[];
};

layout(std430, binding = RAYS_BUF_SLOT) buffer Rays {
    ray_data_t g_rays[];
};

#if !PRIMARY
layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};
#endif

layout(std430, binding = VERTICES_BUF_SLOT) readonly buffer Vertices {
    vertex_t g_vertices[];
};

layout(std430, binding = VTX_INDICES_BUF_SLOT) readonly buffer VtxIndices {
    uint g_vtx_indices[];
};

layout(std430, binding = RANDOM_SEQ_BUF_SLOT) readonly buffer Random {
    float g_random_seq[];
};

layout(std430, binding = OUT_HITS_BUF_SLOT) writeonly buffer Hits {
    hit_data_t g_out_hits[];
};

#if !HWRT
#define FETCH_TRI(j) g_tris[j]
#include "traverse_bvh.glsl"

#define near_child(rd, n)   \
    (rd)[floatBitsToUint(n.bbox_max.w) >> 30] < 0 ? (floatBitsToUint(n.bbox_max.w) & RIGHT_CHILD_BITS) : floatBitsToUint(n.bbox_min.w)

#define far_child(rd, n)    \
    (rd)[floatBitsToUint(n.bbox_max.w) >> 30] < 0 ? floatBitsToUint(n.bbox_min.w) : (floatBitsToUint(n.bbox_max.w) & RIGHT_CHILD_BITS)

shared uint g_stack[LOCAL_GROUP_SIZE_X * LOCAL_GROUP_SIZE_Y][MAX_STACK_SIZE];

void Traverse_MicroTree_WithStack(vec3 ro, vec3 rd, vec3 inv_d, int obj_index, uint node_index,
                                  uint stack_size, inout hit_data_t inter) {
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
            int tri_start = int(floatBitsToUint(n.bbox_min.w) & PRIM_INDEX_BITS);
            int tri_end = tri_start + floatBitsToInt(n.bbox_max.w);

            IntersectTris_ClosestHit(ro, rd, tri_start, tri_end, obj_index, inter);
        }
    }
}

void Traverse_MacroTree_WithStack(vec3 orig_ro, vec3 orig_rd, vec3 orig_inv_rd, uint node_index,
                                  inout hit_data_t inter) {
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

                vec3 ro = (tr.inv_xform * vec4(orig_ro, 1.0)).xyz;
                vec3 rd = (tr.inv_xform * vec4(orig_rd, 0.0)).xyz;
                vec3 inv_d = safe_invert(rd);

                Traverse_MicroTree_WithStack(ro, rd, inv_d, int(g_mi_indices[i]), m.node_index,
                                             stack_size, inter);
            }
        }
    }
}
#endif

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

void main() {
#if PRIMARY
    if (gl_GlobalInvocationID.x >= g_params.img_size.x || gl_GlobalInvocationID.y >= g_params.img_size.y) {
        return;
    }

    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);

    const int index = y * int(g_params.img_size.x) + x;
#else
    const int index = int(gl_WorkGroupID.x * 64 + gl_LocalInvocationIndex);
    if (index >= g_counters[1]) {
        return;
    }

    const int x = (g_rays[index].xy >> 16) & 0xffff;
    const int y = (g_rays[index].xy & 0xffff);
#endif

    vec3 ro = vec3(g_rays[index].o[0], g_rays[index].o[1], g_rays[index].o[2]);
    vec3 rd = vec3(g_rays[index].d[0], g_rays[index].d[1], g_rays[index].d[2]);
    vec3 inv_d = safe_invert(rd);

    hit_data_t inter;
    inter.mask = 0;
    inter.obj_index = inter.prim_index = 0;
#if PRIMARY
    inter.t = g_params.cam_clip_end;
#else
    inter.t = MAX_DIST;
#endif
    inter.u = inter.v = 0.0;

    const float rand_offset = construct_float(hash(g_rays[index].xy));
    int rand_index = g_params.hi + total_depth(g_rays[index]) * RAND_DIM_BOUNCE_COUNT;

    while (true) {
        const float t_val = inter.t;
#if !HWRT
        Traverse_MacroTree_WithStack(ro, rd, inv_d, g_params.node_index, inter);
        if (inter.prim_index < 0) {
            inter.prim_index = -int(g_tri_indices[-inter.prim_index - 1]) - 1;
        } else {
            inter.prim_index = int(g_tri_indices[inter.prim_index]);
        }
#else
        const uint ray_flags = 0;//gl_RayFlagsCullBackFacingTrianglesEXT;

        rayQueryEXT rq;
        rayQueryInitializeEXT(rq,               // rayQuery
                              g_tlas,           // topLevel
                              ray_flags,        // rayFlags
                              0xff,             // cullMask
                              ro,               // origin
                              0.0,              // tMin
                              rd,               // direction
                              inter.t           // tMax
                             );
        while (rayQueryProceedEXT(rq)) {
            rayQueryConfirmIntersectionEXT(rq);
        }

        if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
            const int primitive_offset = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);

            inter.mask = -1;
            inter.obj_index = rayQueryGetIntersectionInstanceIdEXT(rq, true);
            inter.prim_index = primitive_offset + rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
            [[flatten]] if (rayQueryGetIntersectionFrontFaceEXT(rq, true) == false) {
                inter.prim_index = -inter.prim_index - 1;
            }
            vec2 uv = rayQueryGetIntersectionBarycentricsEXT(rq, true);
            inter.u = uv.x;
            inter.v = uv.y;
            inter.t = rayQueryGetIntersectionTEXT(rq, true);
        }
#endif
        if (inter.mask == 0) {
            break;
        }

        const bool is_backfacing = (inter.prim_index < 0);
        const uint tri_index = is_backfacing ? -inter.prim_index - 1 : inter.prim_index;

        if ((!is_backfacing && ((g_tri_materials[tri_index] >> 16u) & MATERIAL_SOLID_BIT) != 0) ||
            (is_backfacing && (g_tri_materials[tri_index] & MATERIAL_SOLID_BIT) != 0)) {
            // solid hit found
            break;
        }

        material_t mat = g_materials[(g_tri_materials[tri_index] >> 16u) & MATERIAL_INDEX_BITS];
        if (is_backfacing) {
            mat = g_materials[g_tri_materials[tri_index] & MATERIAL_INDEX_BITS];
        }

        const vertex_t v1 = g_vertices[g_vtx_indices[tri_index * 3 + 0]];
        const vertex_t v2 = g_vertices[g_vtx_indices[tri_index * 3 + 1]];
        const vertex_t v3 = g_vertices[g_vtx_indices[tri_index * 3 + 2]];

        const float w = 1.0 - inter.u - inter.v;
        const vec2 uvs = vec2(v1.t[0][0], v1.t[0][1]) * w + vec2(v2.t[0][0], v2.t[0][1]) * inter.u + vec2(v3.t[0][0], v3.t[0][1]) * inter.v;

        float trans_r = fract(g_random_seq[rand_index + RAND_DIM_BSDF_PICK] + rand_offset);

        // resolve mix material
        while (mat.type == MixNode) {
            float mix_val = mat.tangent_rotation_or_strength;
            if (mat.textures[BASE_TEXTURE] != 0xffffffff) {
                mix_val *= SampleBilinear(mat.textures[BASE_TEXTURE], uvs, 0).r;
            }

            if (trans_r > mix_val) {
                mat = g_materials[mat.textures[MIX_MAT1]];
                trans_r = (trans_r - mix_val) / (1.0 - mix_val);
            } else {
                mat = g_materials[mat.textures[MIX_MAT2]];
                trans_r = trans_r / mix_val;
            }
        }

        if (mat.type != TransparentNode) {
            break;
        }

#if USE_PATH_TERMINATION
        const bool can_terminate_path = (g_rays[index].depth >> 24) > g_params.min_transp_depth;
#else
        const bool can_terminate_path = false;
#endif

        const float lum = max(g_rays[index].c[0], max(g_rays[index].c[1], g_rays[index].c[2]));
        const float p = fract(g_random_seq[rand_index + RAND_DIM_TERMINATE] + rand_offset);
        const float q = can_terminate_path ? max(0.05, 1.0 - lum) : 0.0;
        if (p < q || lum == 0.0 || (g_rays[index].depth >> 24) + 1 >= g_params.max_transp_depth) {
            // terminate ray
            g_rays[index].c[0] = g_rays[index].c[1] = g_rays[index].c[2] = 0.0;
            break;
        }

        g_rays[index].c[0] *= mat.base_color[0] / (1.0 - q);
        g_rays[index].c[1] *= mat.base_color[1] / (1.0 - q);
        g_rays[index].c[2] *= mat.base_color[2] / (1.0 - q);

        const float t = inter.t + HIT_BIAS;
        ro += rd * t;

        // discard current intersection
        inter.mask = 0;
        inter.t = t_val - inter.t;

        g_rays[index].depth += 0x01000000;
        rand_index += RAND_DIM_BOUNCE_COUNT;
    }

    inter.t += length(vec3(g_rays[index].o[0], g_rays[index].o[1], g_rays[index].o[2]) - ro);

    g_out_hits[index] = inter;
}