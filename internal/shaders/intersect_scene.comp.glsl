#version 450
#extension GL_GOOGLE_include_directive : require
#if HWRT
#extension GL_EXT_ray_query : require
#endif
#if SUBGROUP
#extension GL_KHR_shader_subgroup_vote : require
#else
#define subgroupAll(x) (x)
#define subgroupAny(x) (x)
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
    cwbvh_node_t g_nodes[];
};

layout(std430, binding = LEAF_PRIMS_BUF_SLOT) readonly buffer LeafPrims {
    uint g_leaf_prims[];
};

layout(std430, binding = MESH_INSTANCES_BUF_SLOT) readonly buffer MeshInstances {
    mesh_instance_t g_mesh_instances[];
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

layout(std430, binding = OUT_HITS_BUF_SLOT) writeonly restrict buffer Hits {
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

const uint SM_STACK_SIZE = 12;
shared uint g_stack[LOCAL_GROUP_SIZE_X * LOCAL_GROUP_SIZE_Y][SM_STACK_SIZE];
uint g_stack_spill[MAX_STACK_SIZE - SM_STACK_SIZE]; // expected to be placed into scratch memory

void Traverse_BLAS_WithStack(vec3 ro, vec3 rd, vec3 inv_d, int obj_index, uint node_index, uint stack_size, inout hit_data_t inter) {
    const vec3 neg_inv_do = -inv_d * ro;
    const uint flip_bits = (rd.x < 0.0 ? 1u : 0u) | (rd.y < 0.0 ? 2u : 0u) | (rd.z < 0.0 ? 4u : 0u);

    const uint initial_stack_size = stack_size;
    STACK_PUSH(node_index);

    uint index = 0u;
    while (true) {
        while (true) {
            if ((index & LEAF_NODE_BIT) == 0 && stack_size != initial_stack_size) {
                STACK_POP(index);
                [[flatten]] if ((index & LEAF_NODE_BIT) != 0) {
                    continue;
                }

                const cwbvh_node_t n = g_nodes[index];

                uint mask = cwbvh_child_hit_mask(n, inv_d, neg_inv_do, inter.t, flip_bits);
                while (mask != 0u) {
                    const uint key = uint(findMSB(mask));
                    mask ^= 1u << key;

                    const uint i = key ^ flip_bits;
                    const uint meta = cwbvh_meta(n, i);
                    if ((meta & 0x80u) != 0u) {
                        STACK_PUSH(cwbvh_internal_child(n, i));
                    } else {
                        // leaf child: meta4 rank -> the parent's payload block (8-triangle block base)
                        const uint tri_start = (n.prim_base + (meta & 0x3fu)) | LEAF_NODE_BIT;
                        if ((index & LEAF_NODE_BIT) == 0) {
                            index = tri_start;
                        } else {
                            STACK_PUSH(tri_start);
                        }
                    }
                }
            }

            if (subgroupAll((index & LEAF_NODE_BIT) != 0 || stack_size == initial_stack_size)) {
                break;
            }
        }

        if ((index & LEAF_NODE_BIT) != 0) {
            const int tri_start = int(g_leaf_prims[index & PRIM_INDEX_BITS]);
            index = 0u;

            IntersectTris_ClosestHit(ro, rd, tri_start, tri_start + 8, obj_index, inter);
        }

        if (subgroupAll((index & LEAF_NODE_BIT) == 0u && stack_size == initial_stack_size)) {
            break;
        }
    }
}

void Traverse_TLAS_WithStack(vec3 orig_ro, vec3 orig_rd, uint ray_flags, vec3 orig_inv_rd, uint node_index, inout hit_data_t inter) {
    const vec3 orig_neg_inv_do = -orig_inv_rd * orig_ro;
    const uint flip_bits = (orig_rd.x < 0.0 ? 1u : 0u) | (orig_rd.y < 0.0 ? 2u : 0u) | (orig_rd.z < 0.0 ? 4u : 0u);

    uint stack_size = 0;
    STACK_PUSH(node_index);

    uint index = 0u;
    while (true) {
        while (subgroupAny((index & LEAF_NODE_BIT) == 0 && stack_size != 0u)) {
            if ((index & LEAF_NODE_BIT) == 0 && stack_size != 0u) {
                STACK_POP(index);
                [[flatten]] if ((index & LEAF_NODE_BIT) != 0) {
                    continue;
                }

                const cwbvh_node_t n = g_nodes[index];

                uint mask = cwbvh_child_hit_mask(n, orig_inv_rd, orig_neg_inv_do, inter.t, flip_bits);
                while (mask != 0u) {
                    const uint key = uint(findMSB(mask));
                    mask ^= 1u << key;

                    const uint i = key ^ flip_bits;
                    const uint meta = cwbvh_meta(n, i);
                    if ((meta & 0x80u) != 0u) {
                        STACK_PUSH(cwbvh_internal_child(n, i));
                    } else {
                        // leaf child: TLAS leaves are single mesh instances (meta4 rank -> payload)
                        STACK_PUSH((n.prim_base + (meta & 0x3fu)) | LEAF_NODE_BIT);
                    }
                }
            }
        }

        if ((index & LEAF_NODE_BIT) != 0) {
            const uint mi_index = g_leaf_prims[index & PRIM_INDEX_BITS];
            index = 0u;
            const mesh_instance_t mi = g_mesh_instances[mi_index];
            if ((mi.data.w & ray_flags) != 0) {
                const vec3 ro = (mi.inv_xform * vec4(orig_ro, 1.0)).xyz;
                const vec3 rd = (mi.inv_xform * vec4(orig_rd, 0.0)).xyz;
                const vec3 inv_d = safe_invert(rd);

                Traverse_BLAS_WithStack(ro, rd, inv_d, int(mi_index), mi.data.y, stack_size, inter);
            }
        }

        if (subgroupAll(stack_size == 0u)) {
            break;
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
            // NOTE: All instances are opaque, no need to confirm intersection
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