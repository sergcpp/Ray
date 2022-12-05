#version 450
#extension GL_GOOGLE_include_directive : require
#if HWRT
#extension GL_EXT_ray_query : require
#endif

#include "trace_rays_interface.h"
#include "common.glsl"

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

layout(std430, binding = RAYS_BUF_SLOT) readonly buffer Rays {
    ray_data_t g_rays[];
};

#if !PRIMARY
layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};
#endif

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

    g_out_hits[index] = inter;
}