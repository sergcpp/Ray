#pragma once

#include <vector>

#include "Core.h"

#pragma push_macro("NS")
#undef NS

#define NS Ref
#if defined(_M_AMD64) || defined(_M_X64) || (!defined(__ANDROID__) && defined(__x86_64__)) || (defined(_M_IX86_FP) && _M_IX86_FP == 2)
#define USE_SSE
#pragma message("Ray::Ref::simd_vec will use SSE2")
#elif defined(__ARM_NEON__) || defined(__aarch64__)
#define USE_NEON
#pragma message("Ray::Ref::simd_vec will use NEON")
#elif defined(__ANDROID__) && (defined(__i386__) || defined(__x86_64__))
#define USE_SSE
#pragma message("Ray::Ref::simd_vec will use SSE2")
#else
#pragma message("Ray::Ref::simd_vec will not use SIMD")
#endif

#include "simd/simd_vec.h"

#undef USE_SSE
#undef USE_NEON
#undef NS

#pragma pop_macro("NS")

namespace Ray {
namespace Ref {
struct ray_packet_t {
    rays_id_t id;
    // origin and direction
    float o[3], d[3];
    // color of Ray and ior of medium
    float c[3], ior;
    // derivatives
    float do_dx[3], dd_dx[3], do_dy[3], dd_dy[3];
};

const int RayPacketDimX = 1;
const int RayPacketDimY = 1;
const int RayPacketSize = 1;

struct hit_data_t {
    int mask_values[RayPacketSize];
    int obj_indices[RayPacketSize];
    int prim_indices[RayPacketSize];
    float t, u, v;
    rays_id_t id;

    explicit hit_data_t(eUninitialize) {}
    hit_data_t();
};

struct environment_t {
    float env_col[3];
    float env_clamp;
    uint32_t env_map;
};

struct pass_info_t {
    int index;
    int iteration,
        bounce;
    const float *halton;
    uint32_t flags = 0;

    force_inline bool should_add_direct_light() const {
        // skip if we want only indirect light contribution
        // skip if secondary bounce and we want only direct light contribution (only mesh lights should contribute)
        return !((flags & SkipDirectLight) && (bounce < 3 || (flags & SkipIndirectLight)));
    }

    force_inline bool should_add_environment() const {
        return !(flags & NoBackground) || bounce > 2;
    }

    force_inline bool should_consider_albedo() const {
        // do not use albedo in lightmap mode for primary bounce
        return !(flags & LightingOnly) || bounce > 2;
    }
};

class TextureAtlas;

// Generation of rays
void GeneratePrimaryRays(int iteration, const camera_t &cam, const rect_t &r, int w, int h, const float *halton, aligned_vector<ray_packet_t> &out_rays);
void SampleMeshInTextureSpace(int iteration, int obj_index, int uv_layer, const mesh_t &mesh, const transform_t &tr, const uint32_t *vtx_indices, const vertex_t *vertices,
                              const rect_t &r, int w, int h, const float *halton, aligned_vector<ray_packet_t> &out_rays, aligned_vector<hit_data_t> &out_inters);

// Sorting of rays
void SortRays(ray_packet_t *rays, size_t rays_count, const float root_min[3], const float cell_size[3],
              uint32_t *hash_values, int *head_flags, uint32_t *scan_values, ray_chunk_t *chunks, ray_chunk_t *chunks_temp, uint32_t *skeleton);

// Intersect primitives
bool IntersectTris(const ray_packet_t &r, const tri_accel_t *tris, int num_tris, int obj_index, hit_data_t &out_inter);
bool IntersectTris(const ray_packet_t &r, const tri_accel_t *tris, const uint32_t *indices, int num_indices, int obj_index, hit_data_t &out_inter);

// Traverse acceleration structure
// stack-less cpu-style traversal of outer nodes
bool Traverse_MacroTree_Stackless_CPU(const ray_packet_t &r, const bvh_node_t *nodes, uint32_t root_index,
                                      const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                      const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter);
// stack-less gpu-style traversal of outer nodes
bool Traverse_MacroTree_Stackless_GPU(const ray_packet_t &r, const bvh_node_t *nodes, uint32_t root_index,
                                      const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                      const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter);
// stack-less cpu-style traversal of inner nodes
bool Traverse_MicroTree_Stackless_CPU(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t root_index,
                            const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t &inter);
// stack-less gpu-style traversal of inner nodes
bool Traverse_MicroTree_Stackless_GPU(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t root_index,
                                      const tri_accel_t *tris, const uint32_t *indices, int obj_index, hit_data_t &inter);

// traditional bvh traversal with stack for outer nodes
bool Traverse_MacroTree_WithStack(const ray_packet_t &r, const bvh_node_t *nodes, uint32_t root_index,
                                  const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                  const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter);
// traditional bvh traversal with stack for inner nodes
bool Traverse_MicroTree_WithStack(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t root_index,
                                  const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, uint32_t *stack, hit_data_t &inter);

// Transform
ray_packet_t TransformRay(const ray_packet_t &r, const float *xform);
simd_fvec3 TransformPoint(const simd_fvec3 &p, const float *xform);
simd_fvec3 TransformNormal(const simd_fvec3 &n, const float *inv_xform);
simd_fvec2 TransformUV(const simd_fvec2 &uv, const simd_fvec2 &tex_atlas_size, const texture_t &t, int mip_level);

// Sample Texture
simd_fvec4 SampleNearest(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs, float lod);
simd_fvec4 SampleBilinear(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs, int lod);
simd_fvec4 SampleBilinear(const TextureAtlas &atlas, const simd_fvec2 &uvs, int page);
simd_fvec4 SampleTrilinear(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs, float lod);
simd_fvec4 SampleAnisotropic(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs, const simd_fvec2 &duv_dx, const simd_fvec2 &duv_dy);
simd_fvec4 SampleLatlong_RGBE(const TextureAtlas &atlas, const texture_t &t, const simd_fvec3 &dir);

// Compute punctual lights contribution
simd_fvec3 ComputeDirectLighting(const simd_fvec3 &P, const simd_fvec3 &N, const simd_fvec3 &B, const simd_fvec3 &plane_N,
                                 const float *halton, const int hi, float rand_offset, float rand_offset2,
                                 const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                 const mesh_t *meshes, const transform_t *transforms,
                                 const uint32_t *vtx_indices, const vertex_t *vertices,
                                 const bvh_node_t *nodes, uint32_t node_index, const tri_accel_t *tris,
                                 const uint32_t *tri_indices, const light_t *lights,
                                 const uint32_t *li_indices, uint32_t light_node_index);

// Shade
Ray::pixel_color_t ShadeSurface(const pass_info_t &pi, const hit_data_t &inter, const ray_packet_t &ray,
                                const environment_t &env, const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                const mesh_t *meshes, const transform_t *transforms, const uint32_t *vtx_indices, const vertex_t *vertices,
                                const bvh_node_t *nodes, uint32_t node_index, const tri_accel_t *tris, const uint32_t *tri_indices,
                                const material_t *materials, const texture_t *textures, const TextureAtlas &tex_atlas,
                                const light_t *lights, const uint32_t *li_indices, uint32_t light_node_index, ray_packet_t *out_secondary_rays, int *out_secondary_rays_count);
}
}