#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "intersect_scene_interface.h"
#include "common.glsl"
#include "texture.glsl"

LAYOUT_PARAMS uniform UniformParams {
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
    float g_random_seq[];
};

layout(std430, binding = OUT_HITS_BUF_SLOT) writeonly buffer Hits {
    hit_data_t g_out_hits[];
};

layout(location = 0) rayPayloadEXT hit_data_t g_pld;

void main() {
#if !INDIRECT
    const int index = int(gl_LaunchIDEXT.y * g_params.rect.z + gl_LaunchIDEXT.x);

    const int x = int(g_params.rect.x + gl_LaunchIDEXT.x);
    const int y = int(g_params.rect.y + gl_LaunchIDEXT.y);
#else // !INDIRECT
    const int index = int(gl_LaunchIDEXT.x);

    const int x = (g_rays[index].xy >> 16) & 0xffff;
    const int y = (g_rays[index].xy & 0xffff);
#endif // !INDIRECT

    vec3 ro = vec3(g_rays[index].o[0], g_rays[index].o[1], g_rays[index].o[2]);
    vec3 rd = vec3(g_rays[index].d[0], g_rays[index].d[1], g_rays[index].d[2]);
    vec3 inv_d = safe_invert(rd);

    const vec2 rand_offset = vec2(construct_float(hash(g_rays[index].xy)),
                                  construct_float(hash(hash(g_rays[index].xy))));
    int rand_index = g_params.hi + total_depth(g_rays[index]) * RAND_DIM_BOUNCE_COUNT;

    hit_data_t inter;
    inter.mask = 0;
    inter.obj_index = inter.prim_index = 0;
    inter.t = g_params.inter_t;
    inter.u = inter.v = 0.0;

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

        float trans_r = fract(g_random_seq[rand_index + RAND_DIM_BSDF_PICK] + rand_offset[0]);

        const vec2 tex_rand = vec2(fract(g_random_seq[rand_index + RAND_DIM_TEX_U] + rand_offset[0]),
                                   fract(g_random_seq[rand_index + RAND_DIM_TEX_V] + rand_offset[1]));

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
        const bool can_terminate_path = (g_rays[index].depth >> 24) > g_params.min_transp_depth;
#else
        const bool can_terminate_path = false;
#endif

        const float lum = max(g_rays[index].c[0], max(g_rays[index].c[1], g_rays[index].c[2]));
        const float p = fract(g_random_seq[rand_index + RAND_DIM_TERMINATE] + rand_offset[0]);
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

    inter.t += distance(vec3(g_rays[index].o[0], g_rays[index].o[1], g_rays[index].o[2]), ro);

    g_out_hits[index] = inter;
}
