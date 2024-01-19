#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "intersect_scene_interface.h"
#include "common.glsl"
#include "texture.glsl"

layout(push_constant) uniform UniformParams {
    Params g_params;
};

layout(binding = TLAS_SLOT) uniform accelerationStructureEXT g_tlas;

layout(std430, binding = TRI_MATERIALS_BUF_SLOT) readonly buffer TriMaterials {
    uint g_tri_materials[];
};

layout(std430, binding = MATERIALS_BUF_SLOT) readonly buffer Materials {
    material_t g_materials[];
};

layout(std430, binding = RAYS_BUF_SLOT) buffer Rays {
    ray_data_t g_rays[];
};

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

layout(location = 0) rayPayloadEXT hit_data_t g_pld;

vec2 get_scrambled_2d_rand(const uint dim, const uint seed, const int _sample) {
    const uint i_seed = hash_combine(seed, dim),
               x_seed = hash_combine(seed, 2 * dim + 0),
               y_seed = hash_combine(seed, 2 * dim + 1);

    const uint shuffled_dim = uint(nested_uniform_scramble_base2(dim, seed) & (RAND_DIMS_COUNT - 1));
    const uint shuffled_i = uint(nested_uniform_scramble_base2(_sample, i_seed) & (RAND_SAMPLES_COUNT - 1));
    return vec2(scramble_unorm(x_seed, g_random_seq[shuffled_dim * 2 * RAND_SAMPLES_COUNT + 2 * shuffled_i + 0]),
                scramble_unorm(y_seed, g_random_seq[shuffled_dim * 2 * RAND_SAMPLES_COUNT + 2 * shuffled_i + 1]));
}

void main() {
#if !INDIRECT
    const int index = int(gl_LaunchIDEXT.y * g_params.rect.z + gl_LaunchIDEXT.x);

    const int x = int(g_params.rect.x + gl_LaunchIDEXT.x);
    const int y = int(g_params.rect.y + gl_LaunchIDEXT.y);
#else // !INDIRECT
    const int index = int(gl_LaunchIDEXT.x);

    const int x = int((g_rays[index].xy >> 16) & 0xffff);
    const int y = int(g_rays[index].xy & 0xffff);
#endif // !INDIRECT

    vec3 ro = vec3(g_rays[index].o[0], g_rays[index].o[1], g_rays[index].o[2]);
    vec3 rd = vec3(g_rays[index].d[0], g_rays[index].d[1], g_rays[index].d[2]);
    vec3 inv_d = safe_invert(rd);

    const uint px_hash = hash(g_rays[index].xy);
    const uint rand_hash = hash_combine(px_hash, g_params.rand_seed);

    uint rand_dim = RAND_DIM_BASE_COUNT + get_total_depth(g_rays[index].depth) * RAND_DIM_BOUNCE_COUNT;

    hit_data_t inter;
    inter.obj_index = inter.prim_index = 0;
    inter.u = 0.0;
    inter.v = -1.0; // negative v means 'no intersection'
    if (g_params.clip_dist >= 0.0) {
        inter.t = g_params.clip_dist / dot(rd, g_params.cam_fwd.xyz);
    } else {
        inter.t = MAX_DIST;
    }

    while (true) {
        const float t_val = inter.t;
        traceRayEXT(g_tlas,     // topLevel
                    0,          // rayFlags
                    0xff,       // cullMask
                    0,          // sbtRecordOffset
                    0,          // sbtRecordStride
                    0,          // missIndex
                    ro,         // origin
                    0.0,        // Tmin
                    rd,         // direction
                    t_val,      // Tmax
                    0           // payload
                    );

        inter = g_pld;
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

        const vec2 trans_term_rand = get_scrambled_2d_rand( rand_dim + RAND_DIM_BSDF_PICK, rand_hash, g_params.iteration - 1);
        const vec2 tex_rand = get_scrambled_2d_rand(rand_dim + RAND_DIM_TEX, rand_hash, g_params.iteration - 1);

        float trans_r = trans_term_rand.x;

        // resolve mix material
        while (mat.type == MixNode) {
            float mix_val = mat.tangent_rotation_or_strength;
            if (mat.textures[BASE_TEXTURE] != 0xffffffff) {
                mix_val *= SampleBilinear(mat.textures[BASE_TEXTURE], uvs, 0, tex_rand).r;
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
            g_rays[index].c[0] = g_rays[index].c[1] = g_rays[index].c[2] = 0.0;
            break;
        }

        g_rays[index].c[0] *= mat.base_color[0] / (1.0 - q);
        g_rays[index].c[1] *= mat.base_color[1] / (1.0 - q);
        g_rays[index].c[2] *= mat.base_color[2] / (1.0 - q);

        const float t = inter.t + HIT_BIAS;
        ro += rd * t;

        // discard current intersection
        inter.v = -1.0;
        inter.t = t_val - inter.t;

        g_rays[index].depth += pack_ray_depth(0, 0, 0, 1);
        rand_dim += RAND_DIM_BOUNCE_COUNT;
    }

    inter.t += distance(vec3(g_rays[index].o[0], g_rays[index].o[1], g_rays[index].o[2]), ro);

    g_out_hits[index] = inter;
}
