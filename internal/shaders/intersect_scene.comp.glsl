#version 450
#extension GL_GOOGLE_include_directive : require
#if HWRT
#extension GL_EXT_ray_query : require
#endif

#include "intersect_scene_interface.h"
#include "common.glsl"
#include "texture.glsl"

layout(push_constant) uniform UniformParams {
    Params g_params;
};

#if HWRT
layout(binding = TLAS_SLOT) uniform accelerationStructureEXT g_tlas;
#else // HWRT
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
#endif // HWRT

layout(std430, binding = TRI_MATERIALS_BUF_SLOT) readonly buffer TriMaterials {
    uint g_tri_materials[];
};

layout(std430, binding = MATERIALS_BUF_SLOT) readonly buffer Materials {
    material_t g_materials[];
};

layout(std430, binding = RAYS_BUF_SLOT) buffer Rays {
    ray_data_t g_rays[];
};

#if INDIRECT
layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};
#endif // INDIRECT

layout(std430, binding = VERTICES_BUF_SLOT) readonly buffer Vertices {
    vertex_t g_vertices[];
};

layout(std430, binding = VTX_INDICES_BUF_SLOT) readonly buffer VtxIndices {
    uint g_vtx_indices[];
};

layout(std430, binding = RANDOM_SEQ_BUF_SLOT) readonly buffer Random {
    uint g_random_seq[];
};

layout(std430, binding = OUT_HITS_BUF_SLOT) writeonly buffer Hits {
    hit_data_t g_out_hits[];
};

vec2 get_scrambled_2d_rand(const uint dim, const uint seed, const int _sample) {
    const uint i_seed = hash_combine(seed, dim),
               x_seed = hash_combine(seed, 2 * dim + 0),
               y_seed = hash_combine(seed, 2 * dim + 1);

    const uint shuffled_dim = uint(nested_uniform_scramble_base2(dim, seed) & (RAND_DIMS_COUNT - 1));
    const uint shuffled_i = uint(nested_uniform_scramble_base2(_sample, i_seed) & (RAND_SAMPLES_COUNT - 1));
    return vec2(scramble_unorm(x_seed, g_random_seq[shuffled_dim * 2 * RAND_SAMPLES_COUNT + 2 * shuffled_i + 0]),
                scramble_unorm(y_seed, g_random_seq[shuffled_dim * 2 * RAND_SAMPLES_COUNT + 2 * shuffled_i + 1]));
}

#if !HWRT
#define FETCH_TRI(j) g_tris[j]
#include "traverse_bvh.glsl"

shared uint g_stack[LOCAL_GROUP_SIZE_X * LOCAL_GROUP_SIZE_Y][MAX_STACK_SIZE];

void Traverse_BLAS_WithStack(vec3 ro, vec3 rd, vec3 inv_d, int obj_index, uint node_index,
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
            const int tri_start = int(floatBitsToUint(n.bbox_min.w) & PRIM_INDEX_BITS);
            const int tri_end = tri_start + floatBitsToInt(n.bbox_max.w);

            IntersectTris_ClosestHit(ro, rd, tri_start, tri_end, obj_index, inter);
        }
    }
}

void Traverse_TLAS_WithStack(vec3 orig_ro, vec3 orig_rd, uint ray_flags, vec3 orig_inv_rd, uint node_index,
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
                if ((mi.block_ndx.w & ray_flags) == 0 ||
                    !_bbox_test_fma(orig_inv_rd, orig_neg_inv_do, inter.t, mi.bbox_min.xyz, mi.bbox_max.xyz)) {
                    continue;
                }

                mesh_t m = g_meshes[floatBitsToUint(mi.bbox_max.w)];

                vec3 ro = (mi.inv_xform * vec4(orig_ro, 1.0)).xyz;
                vec3 rd = (mi.inv_xform * vec4(orig_rd, 0.0)).xyz;
                vec3 inv_d = safe_invert(rd);

                Traverse_BLAS_WithStack(ro, rd, inv_d, int(g_mi_indices[i]), m.node_index,
                                        stack_size, inter);
            }
        }
    }
}
#endif

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

void main() {
#if !INDIRECT
    if (gl_GlobalInvocationID.x >= g_params.rect.z || gl_GlobalInvocationID.y >= g_params.rect.w) {
        return;
    }

    const int x = int(g_params.rect.x + gl_GlobalInvocationID.x);
    const int y = int(g_params.rect.y + gl_GlobalInvocationID.y);

    const int index = int(gl_GlobalInvocationID.y * g_params.rect.z + gl_GlobalInvocationID.x);
#else // !INDIRECT
    const int index = int(gl_WorkGroupID.x * 64 + gl_LocalInvocationIndex);
    if (index >= g_counters[1]) {
        return;
    }

    const int x = int((g_rays[index].xy >> 16) & 0xffff);
    const int y = int(g_rays[index].xy & 0xffff);
#endif // !INDIRECT

    vec3 ro = vec3(g_rays[index].o[0], g_rays[index].o[1], g_rays[index].o[2]);
    vec3 rd = vec3(g_rays[index].d[0], g_rays[index].d[1], g_rays[index].d[2]);
    vec3 inv_d = safe_invert(rd);

    hit_data_t inter;
    inter.obj_index = inter.prim_index = 0;
    inter.u = 0.0;
    inter.v = -1.0; // negative v means 'no intersection'
    if (g_params.clip_dist >= 0.0) {
        inter.t = g_params.clip_dist / dot(rd, g_params.cam_fwd.xyz);
    } else {
        inter.t = MAX_DIST;
    }

    const uint ray_flags = (1u << get_ray_type(g_rays[index].depth));

    const uint px_hash = hash(g_rays[index].xy);
    const uint rand_hash = hash_combine(px_hash, g_params.rand_seed);

    uint rand_dim = RAND_DIM_BASE_COUNT + get_total_depth(g_rays[index].depth) * RAND_DIM_BOUNCE_COUNT;
    while (true) {
        const float t_val = inter.t;
#if !HWRT
        Traverse_TLAS_WithStack(ro, rd, ray_flags, inv_d, g_params.node_index, inter);
        if (inter.prim_index < 0) {
            inter.prim_index = -int(g_tri_indices[-inter.prim_index - 1]) - 1;
        } else {
            inter.prim_index = int(g_tri_indices[inter.prim_index]);
        }
#else
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq,               // rayQuery
                              g_tlas,           // topLevel
                              0,                // rayFlags
                              ray_flags,        // cullMask
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
        if (inter.v < 0.0) {
            break;
        }

        const bool is_backfacing = (inter.prim_index < 0);
        const uint tri_index = is_backfacing ? -inter.prim_index - 1 : inter.prim_index;

        if ((!is_backfacing && (g_tri_materials[tri_index] & MATERIAL_SOLID_BIT) != 0) ||
            (is_backfacing && ((g_tri_materials[tri_index] >> 16u) & MATERIAL_SOLID_BIT) != 0)) {
            // solid hit found
            break;
        }

        material_t mat = g_materials[g_tri_materials[tri_index] & MATERIAL_INDEX_BITS];
        if (is_backfacing) {
            mat = g_materials[(g_tri_materials[tri_index] >> 16u) & MATERIAL_INDEX_BITS];
        }

        const vertex_t v1 = g_vertices[g_vtx_indices[tri_index * 3 + 0]];
        const vertex_t v2 = g_vertices[g_vtx_indices[tri_index * 3 + 1]];
        const vertex_t v3 = g_vertices[g_vtx_indices[tri_index * 3 + 2]];

        const float w = 1.0 - inter.u - inter.v;
        const vec2 uvs = vec2(v1.t[0], v1.t[1]) * w + vec2(v2.t[0], v2.t[1]) * inter.u + vec2(v3.t[0], v3.t[1]) * inter.v;

        const vec2 trans_term_rand = get_scrambled_2d_rand(rand_dim + RAND_DIM_BSDF_PICK, rand_hash, g_params.iteration - 1);
        const vec2 tex_rand = get_scrambled_2d_rand(rand_dim + RAND_DIM_TEX, rand_hash, g_params.iteration - 1);

        float trans_r = trans_term_rand.x;

        // resolve mix material
        while (mat.type == MixNode) {
            float mix_val = mat.tangent_rotation_or_strength;
            if (mat.textures[BASE_TEXTURE] != 0xffffffff) {
                mix_val *= SampleBilinear(mat.textures[BASE_TEXTURE], uvs, 0, tex_rand, true /* YCoCg */, true /* SRGB */).r;
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
        const bool can_terminate_path = get_transp_depth(g_rays[index].depth) > g_params.min_transp_depth;
#else
        const bool can_terminate_path = false;
#endif

        const float lum = max(g_rays[index].c[0], max(g_rays[index].c[1], g_rays[index].c[2]));
        const float p = trans_term_rand.y;
        const float q = can_terminate_path ? max(0.05, 1.0 - lum) : 0.0;
        if (p < q || lum == 0.0 || get_transp_depth(g_rays[index].depth) + 1 >= g_params.max_transp_depth) {
            // terminate ray
            g_rays[index].c[0] = 0.0;
            g_rays[index].c[1] = 0.0;
            g_rays[index].c[2] = 0.0;
            break;
        }

        float c[3] = {g_rays[index].c[0] * mat.base_color[0] / (1.0 - q),
                      g_rays[index].c[1] * mat.base_color[1] / (1.0 - q),
                      g_rays[index].c[2] * mat.base_color[2] / (1.0 - q)};
        g_rays[index].c[0] = c[0];
        g_rays[index].c[1] = c[1];
        g_rays[index].c[2] = c[2];

        const float t = inter.t + HIT_BIAS;
        ro += rd * t;

        // discard current intersection
        inter.v = -1.0;
        inter.t = t_val - inter.t;

        uint depth = g_rays[index].depth + pack_ray_depth(0, 0, 0, 1);
        g_rays[index].depth = depth;
        rand_dim += RAND_DIM_BOUNCE_COUNT;
    }

    inter.t += distance(vec3(g_rays[index].o[0], g_rays[index].o[1], g_rays[index].o[2]), ro);

    g_out_hits[index] = inter;
}